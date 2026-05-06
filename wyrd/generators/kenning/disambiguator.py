"""LLM-based disambiguator for ambiguous fuzzy text matches.

When `fuzzy_search_attestations` finds a body word X within edit distance
≤ 1 of multiple distinct etymons, the gloss anchor alone isn't reliable
(see wyrd-c3x: ON `herath` ↔ OE `heath`). This module asks an LLM "given
this passage, which of these morphemes is being described?" and records
the choice plus a one-sentence reason.

Cost gate: skip the LLM call when there's only one plausible candidate.

Out of scope here (filed as wyrd-uct): the agentic `expand_context` loop
where the model can request more text when the initial snippet isn't
enough. This module uses a fixed-window snippet and either commits or
declines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from wyrd.generators.kenning.gemini_extractor import GeminiClient
from wyrd.generators.kenning.lexicon import levenshtein, normalize_ocr_form

if TYPE_CHECKING:
    from wyrd.generators.kenning.lexicon import LexiconDB


@dataclass(frozen=True)
class Candidate:
    """One etymon proposed for disambiguation."""

    etymon_id: int
    canonical_form: str
    language: str
    glosses: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class AmbiguityCase:
    """A group of fuzzy `etymon_text_match` rows sharing
    ``(source_id, matched_form)`` that need a single disambiguator decision.

    Multiple rows in one group means fuzzy-search tagged the same body
    word against multiple distinct etymons; one LLM call serves them all
    and the verdict is applied to every row. ``text_match_ids`` and
    ``current_etymon_ids`` are parallel tuples — index ``i`` in the
    first refers to the row whose current attribution is in the same
    index of the second.
    """

    source_id: str
    matched_form: str
    snippet: str
    candidates: tuple[Candidate, ...]
    text_match_ids: tuple[int, ...]
    current_etymon_ids: tuple[int, ...]


@dataclass(frozen=True)
class DisambiguatorResult:
    """The LLM's verdict on one ambiguity case."""

    chosen_etymon_id: int | None  # None means "not any of these"
    confidence: str  # 'high' | 'medium' | 'low'
    reason: str


# Gemini's OpenAPI-flavored schema for the disambiguator response. Mirrors
# the wyrd-6z7 spec: choice (etymon_id or 'none') + confidence + reason.
DISAMBIGUATOR_SCHEMA: dict = {
    "type": "OBJECT",
    "properties": {
        "choice": {
            "type": "STRING",
            "description": (
                "The chosen etymon's ID as a string (e.g. '4127'), or 'none' if "
                "no candidate fits the passage."
            ),
        },
        "confidence": {
            "type": "STRING",
            "enum": ["high", "medium", "low"],
        },
        "reason": {
            "type": "STRING",
            "description": "One sentence explaining the choice.",
        },
    },
    "required": ["choice", "confidence", "reason"],
}

_SYSTEM_PROMPT = (
    "You are a historical-linguistics disambiguator. Given a passage and a "
    "list of candidate morphemes, you decide which morpheme (if any) is "
    "being used or described in the passage. Match on meaning, language, "
    "and context — not just spelling proximity. If the passage doesn't "
    "clearly support any candidate, answer 'none'. Always cite the specific "
    "phrase or context cue that informed your choice."
)


def _format_candidates(candidates: tuple[Candidate, ...]) -> str:
    lines = []
    for c in candidates:
        gloss_str = "; ".join(c.glosses) if c.glosses else "(no gloss)"
        tag_str = f" [tags: {', '.join(c.tags)}]" if c.tags else ""
        lines.append(
            f"  id={c.etymon_id}: '{c.canonical_form}' ({c.language}) — {gloss_str}{tag_str}"
        )
    return "\n".join(lines)


def disambiguate_one(client: GeminiClient, case: AmbiguityCase) -> DisambiguatorResult:
    """Ask the LLM which of `case.candidates` fits the passage in `case.snippet`.

    Returns a `DisambiguatorResult`. The chosen_etymon_id is:
      - one of the candidate ids, when the LLM picks a clear winner
      - None, when the LLM says "none of these match"
    """
    user = (
        f"PASSAGE:\n{case.snippet}\n\n"
        f"The body text contains the surface form '{case.matched_form}'. "
        f"Which of these candidate morphemes (if any) is being described or "
        f"used here?\n\nCANDIDATES:\n{_format_candidates(case.candidates)}\n"
    )
    response = client.chat_json(_SYSTEM_PROMPT, user, response_schema=DISAMBIGUATOR_SCHEMA)
    choice_str = (response.get("choice") or "").strip()
    confidence = response.get("confidence", "low")
    reason = (response.get("reason") or "").strip()

    if choice_str.lower() == "none" or not choice_str:
        return DisambiguatorResult(chosen_etymon_id=None, confidence=confidence, reason=reason)
    try:
        chosen_id = int(choice_str)
    except ValueError:
        # Model returned a free-form string instead of an id. Treat as
        # 'none' rather than mis-applying.
        return DisambiguatorResult(
            chosen_etymon_id=None,
            confidence="low",
            reason=f"model returned non-id choice: {choice_str!r}",
        )
    valid_ids = {c.etymon_id for c in case.candidates}
    if chosen_id not in valid_ids:
        return DisambiguatorResult(
            chosen_etymon_id=None,
            confidence="low",
            reason=f"model chose id {chosen_id} which isn't a candidate; reason was: {reason}",
        )
    return DisambiguatorResult(chosen_etymon_id=chosen_id, confidence=confidence, reason=reason)


def find_ambiguous_rows(
    db: LexiconDB, *, max_distance: int = 1, limit: int | None = None
) -> list[AmbiguityCase]:
    """Scan `etymon_text_match` for fuzzy rows with multiple plausible etymons.

    A row is "ambiguous" when at least one OTHER live etymon's canonical
    form is within `max_distance` of `matched_form`. Returns a list of
    cases with the candidate set assembled, ready for the disambiguator.

    Excludes rows that have already been processed (method starts with
    'llm-disambiguated-').
    """
    cur = db.conn.execute(
        """
        SELECT m.id, m.etymon_id, m.source_id, m.matched_form, m.snippet
        FROM etymon_text_match m
        WHERE m.method LIKE 'fuzzy%'
          AND m.edit_distance > 0
          AND m.method NOT LIKE 'llm-disambiguated%'
        """
    )
    fuzzy_rows = cur.fetchall()

    # Pull every live etymon once; we'll index by normalized form. Use the
    # ASCII Unit Separator (\x1f) as the concat delimiter — commas and
    # pipes both occur in scholarly gloss text. Uniqueness within each
    # set is guaranteed by the etymon_gloss / etymon_tag primary keys.
    sep = "\x1f"
    etymon_rows = db.conn.execute(
        f"""
        SELECT e.id, e.canonical_form, e.language,
               (SELECT GROUP_CONCAT(gloss, '{sep}')
                FROM etymon_gloss WHERE etymon_id = e.id) AS glosses,
               (SELECT GROUP_CONCAT(tag, '{sep}')
                FROM etymon_tag WHERE etymon_id = e.id) AS tags
        FROM etymon e
        WHERE e.merged_into_id IS NULL
        """
    ).fetchall()

    # Build (normalized_form, [Candidate, ...]) so the inner loop can
    # gather every etymon close to a given matched_form.
    by_norm: dict[str, list[Candidate]] = {}
    for r in etymon_rows:
        norm = normalize_ocr_form(r["canonical_form"])
        glosses = tuple((r["glosses"] or "").split(sep)) if r["glosses"] else ()
        tags = tuple((r["tags"] or "").split(sep)) if r["tags"] else ()
        by_norm.setdefault(norm, []).append(
            Candidate(
                etymon_id=r["id"],
                canonical_form=r["canonical_form"],
                language=r["language"],
                glosses=tuple(g for g in glosses if g),
                tags=tuple(t for t in tags if t),
            )
        )

    # Group fuzzy rows by (source_id, matched_form) so a single
    # ambiguity decision serves every row that observed the same body
    # word in the same source — even when fuzzy-search attached them
    # to different etymons. wyrd-9ae: one LLM call instead of N for
    # the same question.
    grouped: dict[tuple[str, str], list[Any]] = {}
    for row in fuzzy_rows:
        grouped.setdefault((row["source_id"], row["matched_form"]), []).append(row)

    cases: list[AmbiguityCase] = []
    # Iterate in the original row-id order so the case ordering stays
    # stable across runs (the dict insertion order tracks first-seen
    # row id since fuzzy_rows came back ORDER BY id implicit).
    for (source_id, matched_form), rows in grouped.items():
        target = normalize_ocr_form(matched_form)
        candidates: list[Candidate] = []
        seen_etymon_ids: set[int] = set()
        for norm, etymons in by_norm.items():
            if abs(len(norm) - len(target)) > max_distance:
                continue
            d = levenshtein(norm, target, max_distance=max_distance)
            if d > max_distance:
                continue
            for c in etymons:
                if c.etymon_id in seen_etymon_ids:
                    continue
                seen_etymon_ids.add(c.etymon_id)
                candidates.append(c)
        # Cost gate: only ambiguous when ≥ 2 distinct etymons within range.
        # (One candidate means the matched_form is unambiguous already.)
        if len(candidates) < 2:
            continue
        # Pick the first non-empty snippet for the LLM. Snippets across
        # rows in a group will differ only in surrounding context for
        # the same matched_form; one is enough for the disambiguator.
        snippet = next((r["snippet"] for r in rows if r["snippet"]), "")
        cases.append(
            AmbiguityCase(
                source_id=source_id,
                matched_form=matched_form,
                snippet=snippet,
                candidates=tuple(candidates),
                text_match_ids=tuple(r["id"] for r in rows),
                current_etymon_ids=tuple(r["etymon_id"] for r in rows),
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def apply_disambiguator_result(
    db: LexiconDB, case: AmbiguityCase, result: DisambiguatorResult
) -> dict[str, int]:
    """Apply the disambiguator's verdict to every row in ``case``'s group.

    Returns a counts dict with keys 'kept' / 'reassigned' / 'deleted'
    summing to len(case.text_match_ids). Per-row semantics:
      'kept'       — chosen etymon == row's current etymon; method bumped
      'reassigned' — chosen etymon != row's current; row's etymon_id replaced
      'deleted'    — LLM said 'none'; row deleted

    A group of N rows produces a single LLM call (the disambiguator runs
    once on the whole case) but N row-level outcomes — different rows in
    the group may resolve to different actions because their current
    etymon attributions differ.
    """
    counts = {"kept": 0, "reassigned": 0, "deleted": 0}
    if result.chosen_etymon_id is None:
        # All rows in the group are dropped — the matched_form doesn't
        # cleanly match any of the candidate etymons.
        for tm_id in case.text_match_ids:
            db.conn.execute("DELETE FROM etymon_text_match WHERE id = ?", (tm_id,))
        counts["deleted"] = len(case.text_match_ids)
        return counts

    for tm_id, current_etymon_id in zip(case.text_match_ids, case.current_etymon_ids, strict=True):
        if result.chosen_etymon_id == current_etymon_id:
            db.conn.execute(
                "UPDATE etymon_text_match "
                "SET method = 'llm-disambiguated-v1', disambiguator_reason = ? "
                "WHERE id = ?",
                (result.reason, tm_id),
            )
            counts["kept"] += 1
            continue

        # Reassign to a different etymon. The UNIQUE (etymon_id, source_id,
        # matched_form) constraint means a row may already exist at the
        # destination — typically an exact match (edit_distance=0) discovered
        # earlier by reverse-search, OR (new with grouping) another row in
        # this same group whose current_etymon_id == chosen_etymon_id and
        # has already been bumped to the chosen etymon by an earlier
        # iteration of this loop. Per D21 (mining-evidence preservation),
        # we don't overwrite that evidence: we record the verdict on the
        # destination row and drop the now-misattributed source row.
        existing = db.conn.execute(
            "SELECT id, edit_distance FROM etymon_text_match "
            "WHERE etymon_id = ? AND source_id = ? AND matched_form = ?",
            (result.chosen_etymon_id, case.source_id, case.matched_form),
        ).fetchone()
        if existing is not None:
            db.conn.execute(
                "UPDATE etymon_text_match "
                "SET method = 'llm-disambiguated-v1', disambiguator_reason = ? "
                "WHERE id = ?",
                (result.reason, existing["id"]),
            )
            db.conn.execute("DELETE FROM etymon_text_match WHERE id = ?", (tm_id,))
            counts["reassigned"] += 1
            continue

        db.conn.execute(
            "UPDATE etymon_text_match "
            "SET etymon_id = ?, method = 'llm-disambiguated-v1', "
            "    disambiguator_reason = ? "
            "WHERE id = ?",
            (result.chosen_etymon_id, result.reason, tm_id),
        )
        counts["reassigned"] += 1
    return counts


__all__ = [
    "AmbiguityCase",
    "Candidate",
    "DISAMBIGUATOR_SCHEMA",
    "DisambiguatorResult",
    "apply_disambiguator_result",
    "disambiguate_one",
    "find_ambiguous_rows",
]
