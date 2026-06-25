"""LLM-based disambiguator for ambiguous fuzzy text matches.

When `fuzzy_search_attestations` finds a body word X within edit distance
≤ 1 of multiple distinct etymons, the gloss anchor alone isn't reliable
(see wyrd-c3x: ON `herath` ↔ OE `heath`). This module asks an LLM "given
this passage, which of these morphemes is being described?" and records
the choice plus a one-sentence reason.

Cost gate: skip the LLM call when there's only one plausible candidate.

Two flavors:
- ``disambiguate_one`` — single-shot, fixed-window snippet (the original
  wyrd-6z7 path).
- ``disambiguate_one_agentic`` — wyrd-uct: lets the model request a wider
  snippet when 200 chars isn't enough. Action-discriminated schema:
  ``{action: 'answer'|'expand', ...}``. Caps at 3 expansions or
  2000 chars total context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wyrd.generators.kenning.extractors.gemini import GeminiClient
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


def _parse_answer(response: dict, candidates: tuple[Candidate, ...]) -> DisambiguatorResult:
    """Parse a verdict-shaped response (``choice``/``confidence``/``reason``)
    into a `DisambiguatorResult`. Shared between the single-shot
    ``disambiguate_one`` and the agentic loop's terminal branch so the
    validation rules (id-must-be-in-candidates, non-numeric → 'low none',
    'none'/'' → None) stay in one place."""
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
    valid_ids = {c.etymon_id for c in candidates}
    if chosen_id not in valid_ids:
        return DisambiguatorResult(
            chosen_etymon_id=None,
            confidence="low",
            reason=f"model chose id {chosen_id} which isn't a candidate; reason was: {reason}",
        )
    return DisambiguatorResult(chosen_etymon_id=chosen_id, confidence=confidence, reason=reason)


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
    return _parse_answer(response, case.candidates)


# --- Agentic disambiguator (wyrd-uct) -------------------------------------
#
# The fixed-window snippet (±100 chars) sometimes isn't enough — the cue
# that resolves the ambiguity is two paragraphs up, or in the next
# sentence. Rather than picking a wider default and paying for tokens
# every call, let the model decide when to widen via an
# action-discriminated schema:
#
#   {action: 'answer', choice, confidence, reason}      # terminal
#   {action: 'expand', direction, chars}                # ask for more
#
# The orchestrator widens the snippet and re-prompts. Caps: 3 expansion
# rounds OR 2000 chars total context. On the final round the system
# prompt explicitly says "you must answer now" so the model commits
# instead of asking for more.


# Initial snippet radius before/after the matched form, mirroring
# `_TEXT_MATCH_SNIPPET_RADIUS` in lexicon.py — that's how the stored
# `etymon_text_match.snippet` was built, so the orchestrator's
# bookkeeping starts from the same baseline.
_AGENTIC_INITIAL_RADIUS = 100

# Per-expansion limits. Kept tight so a runaway model can't drive cost
# unboundedly: minimum widens at all (50), maximum keeps the prompt
# under a single Gemini call's structured-output budget (500).
_AGENTIC_MIN_EXPANSION_CHARS = 50
_AGENTIC_MAX_EXPANSION_CHARS = 500


# Default per-case caps. The acceptance criterion (wyrd-uct) targets a
# 200-char-ambiguous / 500-char-resolves pattern, well within these
# bounds. Operators can override for harder cases.
_AGENTIC_DEFAULT_MAX_EXPANSIONS = 3
_AGENTIC_DEFAULT_TOTAL_CHAR_CAP = 2000

# Fallback expansion size when the model returns ``action='expand'``
# but omits ``chars`` (or returns a non-positive value). 200 chars per
# side roughly doubles the initial ±100-char window, which is the
# smallest expansion likely to surface a new resolving cue.
_AGENTIC_DEFAULT_REQUEST_CHARS = 200


@dataclass(frozen=True)
class ExpansionStats:
    """Per-case telemetry for the agentic loop.

    ``expansions`` is the number of widened-context rounds that
    actually grew the snippet (model said 'expand' AND the orchestrator
    honored it). ``final_chars`` is the size of the snippet at commit
    time. ``forced_commit`` is True when the orchestrator forced the
    model to commit on the final round (cap reached, total-char-cap
    truncation, or the expander couldn't produce more context) — a
    case that always forces a commit suggests the initial radius is
    too narrow for that source/pattern.
    """

    expansions: int
    final_chars: int
    forced_commit: bool


@dataclass
class SnippetExpander:
    """Lazily loads source body texts and produces wider snippets around
    the matched form's first occurrence.

    Two ways to construct:
    - ``SnippetExpander(sources_dir=Path)`` — load lazily from a
      directory of ``*.txt`` files (the same shape as
      ``_load_normalized_source_texts`` consumes).
    - ``SnippetExpander.from_in_memory({source_id: body_text})`` — for
      tests that don't want to write fixture files.

    The body text is normalized via ``normalize_ocr_form`` (lowercased
    + ASCII-folded) so the matcher sees the same surface as the rest of
    the pipeline.
    """

    sources_dir: Path | None = None
    _cache: dict[str, str] | None = field(default=None, repr=False)

    @classmethod
    def from_in_memory(cls, source_texts: dict[str, str]) -> SnippetExpander:
        """Construct from an already-normalized dict — bypasses disk
        loading. Used by tests."""
        e = cls(sources_dir=None)
        # Bypass dataclass field semantics — we want the cache pre-populated
        # so ``get_body`` returns hits without calling _load_normalized_source_texts.
        e._cache = dict(source_texts)
        return e

    def _ensure_loaded(self) -> None:
        if self._cache is not None:
            return
        # Local import: keep the disambiguator deps off the cold-start
        # path for unrelated CLI commands. Same pattern as the CLI's
        # local imports in ``lexicon_disambiguate_fuzzy``.
        from wyrd.generators.kenning.lexicon.enrichment.reverse_search import (
            _load_normalized_source_texts,
        )

        if self.sources_dir is None:
            self._cache = {}
            return
        self._cache = _load_normalized_source_texts(self.sources_dir)

    def get_body(self, source_id: str) -> str | None:
        """Return the normalized body text for a source, or None if not
        loaded. Loads the directory on first call."""
        self._ensure_loaded()
        assert self._cache is not None  # _ensure_loaded post-condition
        return self._cache.get(source_id)

    def make_snippet(
        self,
        source_id: str,
        matched_form: str,
        *,
        radius_before: int,
        radius_after: int,
    ) -> str | None:
        """Find the first ``\\b``-bounded occurrence of ``matched_form``
        in the body for ``source_id`` and return a snippet of the
        requested radius around it. Same marker convention as
        ``reverse_search_attestations`` (``«matched_form»``) so the LLM
        prompt highlights the form being disambiguated.

        Word-boundary anchored to mirror ``fuzzy_search_attestations`` /
        ``reverse_search_attestations`` — using a plain ``str.find``
        would silently anchor onto a substring of a longer word
        (e.g. ``heath`` inside ``heathen``) and expand around the wrong
        position.

        Returns None when the source isn't loadable or the form isn't
        found — callers force a commit on the widest snippet they have.
        """
        body = self.get_body(source_id)
        if body is None:
            return None
        norm = normalize_ocr_form(matched_form)
        match = re.search(r"\b" + re.escape(norm) + r"\b", body)
        if match is None:
            return None
        start = max(0, match.start() - radius_before)
        end = min(len(body), match.end() + radius_after)
        # Mark by the match's POSITION, not ``snippet.replace(norm, ...)``: a plain
        # substring replace wraps the first substring occurrence in the window,
        # which — for a form that is also a substring of a longer word (``heath`` in
        # ``heathen``) — lands on the WRONG word, throwing away the ``\b`` anchoring
        # this method documents. ``match.group(0)`` is the body's own matched span.
        marked = body[start : match.start()] + f"«{match.group(0)}»" + body[match.end() : end]
        return marked.strip()


# Gemini's OpenAPI-flavored schema for the agentic response. Both
# branches share the envelope; the orchestrator validates the
# branch-specific fields after parsing. Conditionally-required fields
# aren't expressible in OpenAPI 3.0, so we keep all the leaf properties
# optional and enforce shape post-hoc.
AGENTIC_RESPONSE_SCHEMA: dict = {
    "type": "OBJECT",
    "properties": {
        "action": {
            "type": "STRING",
            "enum": ["answer", "expand"],
            "description": (
                "'answer' to commit a final verdict, 'expand' to request "
                "more surrounding text before deciding."
            ),
        },
        # action='answer' fields:
        "choice": {
            "type": "STRING",
            "description": (
                "When action='answer': the chosen etymon's id as a string "
                "(e.g. '4127'), or 'none'. Ignored otherwise."
            ),
        },
        "confidence": {
            "type": "STRING",
            "enum": ["high", "medium", "low"],
        },
        "reason": {
            "type": "STRING",
            "description": "One sentence explaining the choice or expansion request.",
        },
        # action='expand' fields:
        "direction": {
            "type": "STRING",
            "enum": ["before", "after", "both"],
            "description": "When action='expand': which side(s) to widen.",
        },
        "chars": {
            "type": "INTEGER",
            "description": (
                "When action='expand': how many chars to add to each "
                f"chosen side ({_AGENTIC_MIN_EXPANSION_CHARS}-"
                f"{_AGENTIC_MAX_EXPANSION_CHARS}). For direction='both', "
                "chars are added to BOTH sides — total widening is 2×chars."
            ),
        },
    },
    "required": ["action"],
}


_AGENTIC_SYSTEM_PROMPT = (
    "You are a historical-linguistics disambiguator. Given a passage and "
    "a list of candidate morphemes, you decide which morpheme (if any) is "
    "being used or described in the passage. Match on meaning, language, "
    "and context — not just spelling proximity. If the passage doesn't "
    "clearly support any candidate, answer 'none'. Always cite the specific "
    "phrase or context cue that informed your choice.\n\n"
    "You may either commit a final verdict OR ask for more surrounding "
    "text:\n"
    "  - To commit: respond with action='answer', choice (the etymon id "
    "    as a string, or 'none'), confidence ('high'|'medium'|'low'), and "
    "    a one-sentence reason.\n"
    "  - To request more context: respond with action='expand', direction "
    "    ('before'|'after'|'both'), and chars "
    f"    ({_AGENTIC_MIN_EXPANSION_CHARS}-{_AGENTIC_MAX_EXPANSION_CHARS}). "
    "    chars are added to each chosen side, so direction='both' "
    "    widens by 2×chars total.\n\n"
    "Use 'expand' only when more text would resolve a real ambiguity. "
    "Prefer answering with choice='none' over chasing context indefinitely."
)


def _format_agentic_user(case: AmbiguityCase, snippet: str, *, force_commit: bool) -> str:
    """Build the user-side prompt for one agentic round.

    On the final round we explicitly tell the model it must commit —
    otherwise a model that keeps requesting expansions would force the
    orchestrator into the 'forced_commit' fallback path, which yields
    a less informative verdict.
    """
    final_note = ""
    if force_commit:
        final_note = (
            "\n\nFINAL ROUND: no more expansions are available. You MUST "
            "respond with action='answer'."
        )
    return (
        f"PASSAGE:\n{snippet}\n\n"
        f"The body text contains the surface form '{case.matched_form}'. "
        f"Which of these candidate morphemes (if any) is being described "
        f"or used here?\n\nCANDIDATES:\n{_format_candidates(case.candidates)}"
        f"{final_note}\n"
    )


def _parse_expansion_request(response: dict) -> tuple[str, int]:
    """Pull and validate ``direction`` + ``chars`` out of an
    action='expand' response. Defaults to ``direction='both'`` and
    ``chars=_AGENTIC_DEFAULT_REQUEST_CHARS`` when the model omits
    them or returns an unknown direction / non-numeric chars.
    ``chars`` is clamped into [MIN, MAX]."""
    direction = (response.get("direction") or "both").lower()
    if direction not in ("before", "after", "both"):
        direction = "both"
    try:
        chars = int(response.get("chars") or 0)
    except (TypeError, ValueError):
        chars = 0
    if chars <= 0:
        chars = _AGENTIC_DEFAULT_REQUEST_CHARS
    return direction, max(_AGENTIC_MIN_EXPANSION_CHARS, min(chars, _AGENTIC_MAX_EXPANSION_CHARS))


def _force_commit_failure(
    *,
    expansions: int,
    current_snippet: str,
    reason: str,
) -> tuple[DisambiguatorResult, ExpansionStats]:
    """Build the ``(result, stats)`` pair for a forced-but-not-answered
    outcome. Used by both the in-loop "model refused to commit" branch
    and the post-loop defensive fallback so the failure shape stays
    consistent."""
    return (
        DisambiguatorResult(chosen_etymon_id=None, confidence="low", reason=reason),
        ExpansionStats(
            expansions=expansions,
            final_chars=len(current_snippet),
            forced_commit=True,
        ),
    )


def _try_widen(
    expander: SnippetExpander,
    case: AmbiguityCase,
    *,
    radius_before: int,
    radius_after: int,
    direction: str,
    chars: int,
    current_snippet: str,
    total_char_cap: int,
) -> tuple[str, int, int] | None:
    """Honor one expansion request. Returns ``(new_snippet,
    new_radius_before, new_radius_after)`` on success, or None when
    the expander can't widen further OR the resulting snippet would
    exceed ``total_char_cap`` (we discard the widening rather than
    feed a marker-truncated slice that would garble the «...»
    highlight)."""
    new_radius_before = radius_before + (chars if direction in ("before", "both") else 0)
    new_radius_after = radius_after + (chars if direction in ("after", "both") else 0)
    new_snippet = expander.make_snippet(
        case.source_id,
        case.matched_form,
        radius_before=new_radius_before,
        radius_after=new_radius_after,
    )
    if new_snippet is None or new_snippet == current_snippet:
        return None
    if len(new_snippet) > total_char_cap:
        return None
    return new_snippet, new_radius_before, new_radius_after


def disambiguate_one_agentic(
    client: GeminiClient,
    case: AmbiguityCase,
    expander: SnippetExpander,
    *,
    max_expansions: int = _AGENTIC_DEFAULT_MAX_EXPANSIONS,
    total_char_cap: int = _AGENTIC_DEFAULT_TOTAL_CHAR_CAP,
) -> tuple[DisambiguatorResult, ExpansionStats]:
    """Run the agentic disambiguator loop on one ambiguity case.

    The model can either commit ('answer') or request a wider snippet
    ('expand') for up to ``max_expansions`` rounds. The total snippet
    length is capped at ``total_char_cap`` chars. On the final round
    the system prompt forces a commit.

    Returns ``(DisambiguatorResult, ExpansionStats)``. The result is
    drop-in compatible with ``apply_disambiguator_result``; the stats
    are for telemetry/tuning the default starting window.

    When the expander can't widen further (source not loadable, form
    not found, or cap reached) the loop short-circuits to the final
    round so the model is asked to commit on the widest snippet
    available — never blocks waiting for context that doesn't exist.
    """
    radius_before = _AGENTIC_INITIAL_RADIUS
    radius_after = _AGENTIC_INITIAL_RADIUS
    current_snippet = case.snippet
    expansions = 0
    # ``force_next_round`` is the dedicated short-circuit signal: set
    # when ``_try_widen`` returns None (source missing / form not
    # found / would exceed cap). Kept separate from ``expansions`` so
    # stats remain truthful — a cap-truncation at round 0 yields
    # expansions=0/forced_commit=True, which a caller can distinguish
    # from "model hit the natural cap after 3 honored widenings".
    force_next_round = False
    last_action = ""

    # Loop bound: ``max_expansions + 1`` rounds — up to N honored
    # expansion rounds + one final forced-commit round. The forced-
    # commit gate fires when EITHER the natural expansions cap is hit
    # OR a short-circuit set ``force_next_round``.
    for _ in range(max_expansions + 1):
        force_commit = force_next_round or expansions >= max_expansions
        prompt = _format_agentic_user(case, current_snippet, force_commit=force_commit)
        response = client.chat_json(
            _AGENTIC_SYSTEM_PROMPT, prompt, response_schema=AGENTIC_RESPONSE_SCHEMA
        )
        action = (response.get("action") or "").lower()
        last_action = action
        stats = ExpansionStats(
            expansions=expansions,
            final_chars=len(current_snippet),
            forced_commit=force_commit,
        )

        if action == "answer":
            return _parse_answer(response, case.candidates), stats

        if action != "expand" or force_commit:
            # Model returned an unknown action OR refused to commit
            # even on the final round. Force a 'none' verdict so the
            # row gets dropped rather than mis-applied.
            return _force_commit_failure(
                expansions=expansions,
                current_snippet=current_snippet,
                reason=(
                    f"model failed to commit after {expansions} honored "
                    f"expansion(s); final action was {action!r}"
                ),
            )

        direction, chars = _parse_expansion_request(response)
        widened = _try_widen(
            expander,
            case,
            radius_before=radius_before,
            radius_after=radius_after,
            direction=direction,
            chars=chars,
            current_snippet=current_snippet,
            total_char_cap=total_char_cap,
        )
        if widened is None:
            force_next_round = True
            continue
        current_snippet, radius_before, radius_after = widened
        expansions += 1

    # Loop exit without an answer is unreachable: the final iteration
    # always sets force_commit=True, which forces the action!='answer'
    # branch above to return. Defensive fallback satisfies the type
    # checker and surfaces a loud reason if the invariant ever breaks.
    return _force_commit_failure(
        expansions=expansions,
        current_snippet=current_snippet,
        reason=(
            f"agentic loop exited unexpectedly after {expansions} "
            f"honored expansion(s); last action {last_action!r}"
        ),
    )


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
    # ORDER BY id makes the per-group ordering load-bearing — the
    # grouping dict's insertion order then tracks first-seen-row id so
    # the case sequence stays deterministic across runs (matters for
    # operator-readable summaries and reproducible test fixtures).
    cur = db.conn.execute(
        """
        SELECT m.id, m.etymon_id, m.source_id, m.matched_form, m.snippet
        FROM etymon_text_match m
        WHERE m.method LIKE 'fuzzy%'
          AND m.edit_distance > 0
          AND m.method NOT LIKE 'llm-disambiguated%'
        ORDER BY m.id
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

    # Build (normalized_form, [Candidate, ...]) so the per-group loop can
    # gather every etymon close to a given matched_form.
    by_norm = _index_etymons_by_norm(etymon_rows, sep)

    # Group fuzzy rows by (source_id, matched_form) so a single
    # ambiguity decision serves every row that observed the same body
    # word in the same source — even when fuzzy-search attached them
    # to different etymons. wyrd-9ae: one LLM call instead of N for
    # the same question.
    grouped: dict[tuple[str, str], list[Any]] = {}
    for row in fuzzy_rows:
        grouped.setdefault((row["source_id"], row["matched_form"]), []).append(row)

    cases: list[AmbiguityCase] = []
    for (source_id, matched_form), rows in grouped.items():
        candidates = _candidates_for_form(normalize_ocr_form(matched_form), by_norm, max_distance)
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


def _index_etymons_by_norm(etymon_rows: list[Any], sep: str) -> dict[str, list[Candidate]]:
    """Index live etymons by OCR-normalized canonical form → Candidates, so the
    ambiguity scan can gather every etymon close to a matched form. Glosses/tags
    come from the ``sep``-joined GROUP_CONCAT columns, split back to tuples."""
    by_norm: dict[str, list[Candidate]] = {}
    for r in etymon_rows:
        norm = normalize_ocr_form(r["canonical_form"])
        glosses = tuple(r["glosses"].split(sep)) if r["glosses"] else ()
        tags = tuple(r["tags"].split(sep)) if r["tags"] else ()
        by_norm.setdefault(norm, []).append(
            Candidate(
                etymon_id=r["id"],
                canonical_form=r["canonical_form"],
                language=r["language"],
                glosses=tuple(g for g in glosses if g),
                tags=tuple(t for t in tags if t),
            )
        )
    return by_norm


def _candidates_for_form(
    target: str, by_norm: dict[str, list[Candidate]], max_distance: int
) -> list[Candidate]:
    """All distinct live etymons whose normalized form is within ``max_distance``
    (Levenshtein) of ``target``, de-duped by etymon_id in by_norm iteration
    order (which tracks first-seen for deterministic case sequencing)."""
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
    return candidates


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
    "AGENTIC_RESPONSE_SCHEMA",
    "AmbiguityCase",
    "Candidate",
    "DISAMBIGUATOR_SCHEMA",
    "DisambiguatorResult",
    "ExpansionStats",
    "SnippetExpander",
    "apply_disambiguator_result",
    "disambiguate_one",
    "disambiguate_one_agentic",
    "find_ambiguous_rows",
]
