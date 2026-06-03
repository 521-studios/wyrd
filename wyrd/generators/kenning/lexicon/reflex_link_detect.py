"""wyrd-rogd.15: cross-era reflex-LINK candidate detection.

Surfaces pairs of etymons that are plausibly the SAME morpheme across eras — a
later-era reflex (modern ``-ton``) and an earlier-era ancestor (OE ``tūn``) —
that are NOT yet connected by a descent edge, for the LLM to judge (link / not).

Signal: a shared gloss + form similarity + a cross-language era step + no
existing edge. The LINK op (``enrichment.apply_collapses`` ``inherits``) then
adds an inheritance edge, keeping both era forms distinct (vs the same-language
variant FOLD, which is the existing ``merge-collapses`` recall problem — a
separate follow-up, deliberately NOT emitted here).
"""

from __future__ import annotations

import difflib
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

from wyrd.generators.kenning.era.cells import CANONICAL_LANGUAGE_FOR_CELL
from wyrd.generators.kenning.lexicon.collapse_merge import CONFIDENCE_RANK

LLM_REFLEX_LINK_METHOD = "llm-reflex-link-v1"
LLM_REFLEX_REJECT_METHOD = "llm-reflex-rejected-v1"

# language → era rank: first-seen ordinal in the cell map, which is ordered
# older→newer within each family. Orients a candidate pair (earlier = parent /
# ancestor, later = child / reflex). Unknown languages sort last.
_LANGUAGE_ERA_RANK: dict[str, int] = {}
for _i, _lang in enumerate(CANONICAL_LANGUAGE_FOR_CELL.values()):
    _LANGUAGE_ERA_RANK.setdefault(_lang, _i)


@dataclass(frozen=True)
class ReflexLinkCandidate:
    """A cross-era pair for the LLM to judge: is ``child`` a later-era reflex of
    ``parent`` (→ inheritance link), or coincidental same-gloss lookalikes?"""

    child_ref: str  # later-era reflex, "<language>:<canonical_form>"
    parent_ref: str  # earlier-era ancestor
    child_form: str
    parent_form: str
    shared_glosses: tuple[str, ...]
    similarity: float


def _bare(form: str) -> str:
    """Position/case-insensitive comparison key — dashes are a link, never part
    of the morpheme (wyrd-rogd.10)."""
    return form.strip().strip("-").lower()


def detect_reflex_link_candidates(
    conn: sqlite3.Connection,
    *,
    min_similarity: float = 0.6,
    max_per_gloss: int = 25,
) -> list[ReflexLinkCandidate]:
    """Cross-era reflex-link candidates: cross-language etymon pairs sharing a
    gloss, form-similar (difflib ratio ≥ ``min_similarity`` on the dash-stripped
    forms), and not already connected by a descent edge. ``max_per_gloss`` caps
    huge generic-gloss groups (the O(n²) pairing only runs inside a group)."""
    rows = conn.execute(
        "SELECT e.id, e.language, e.canonical_form, g.gloss "
        "FROM etymon e JOIN etymon_gloss g ON g.etymon_id = e.id "
        "WHERE e.merged_into_id IS NULL"
    ).fetchall()
    by_id: dict[int, dict] = {}
    by_gloss: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        rec = by_id.setdefault(
            r["id"], {"lang": r["language"], "form": r["canonical_form"], "glosses": set()}
        )
        rec["glosses"].add(r["gloss"])
        by_gloss[r["gloss"].strip().lower()].add(r["id"])

    def has_edge(a: int, b: int) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM etymon_descent WHERE (parent_id=? AND child_id=?) "
                "OR (parent_id=? AND child_id=?) LIMIT 1",
                (a, b, b, a),
            ).fetchone()
            is not None
        )

    out: list[ReflexLinkCandidate] = []
    seen: set[tuple[int, int]] = set()
    for ids in by_gloss.values():
        if not (2 <= len(ids) <= max_per_gloss):
            continue
        ordered = sorted(ids)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                ea, eb = by_id[a], by_id[b]
                if ea["lang"] == eb["lang"]:
                    continue  # same-language variant = FOLD (existing detector), not a link
                fa, fb = _bare(ea["form"]), _bare(eb["form"])
                if not fa or not fb:
                    continue
                sim = difflib.SequenceMatcher(None, fa, fb).ratio()
                if sim < min_similarity or (a, b) in seen or has_edge(a, b):
                    continue
                seen.add((a, b))
                ra = _LANGUAGE_ERA_RANK.get(ea["lang"], 10**6)
                rb = _LANGUAGE_ERA_RANK.get(eb["lang"], 10**6)
                parent, child = (a, b) if ra <= rb else (b, a)
                out.append(
                    ReflexLinkCandidate(
                        child_ref=f"{by_id[child]['lang']}:{by_id[child]['form']}",
                        parent_ref=f"{by_id[parent]['lang']}:{by_id[parent]['form']}",
                        child_form=by_id[child]["form"],
                        parent_form=by_id[parent]["form"],
                        shared_glosses=tuple(sorted(by_id[a]["glosses"] & by_id[b]["glosses"])),
                        similarity=round(sim, 3),
                    )
                )
    out.sort(key=lambda c: (c.parent_ref, c.child_ref))
    return out


@dataclass(frozen=True)
class ReflexVerdict:
    is_reflex: bool  # child is a later-era reflex (descendant) of parent
    confidence: str  # "high" | "medium" | "low"
    reason: str


_REFLEX_JUDGE_SYSTEM = (
    "You are a historical-linguistics editor building a place-name etymology "
    "lexicon. You are given two head-words for the SAME meaning from different "
    "language stages: an EARLIER form and a LATER form. Decide whether the later "
    "form is a direct later-era REFLEX (descendant) of the earlier form — i.e. "
    "the same word continuing/evolving into a later period (Old English 'tūn' -> "
    "modern '-ton'; OE 'hyll' -> modern 'hill'). Answer NO if instead they are: "
    "a parallel COGNATE in a different branch (same proto-root, not one descended "
    "from the other), a separate BORROWING, or a coincidental same-meaning "
    "look-alike. Judge by sound/spelling continuity AND meaning. Reply ONLY with "
    "a JSON object: "
    '{"is_reflex": true|false, "confidence": "high"|"medium"|"low", '
    '"reason": "<one short clause>"}.'
)


def build_reflex_judge_prompt(c: ReflexLinkCandidate) -> tuple[str, str]:
    """Return (system, user) prompts for judging a reflex-link candidate."""
    parent_lang = c.parent_ref.split(":", 1)[0]
    child_lang = c.child_ref.split(":", 1)[0]
    user = (
        f'Earlier form "{c.parent_form}" ({parent_lang})\n'
        f'Later form   "{c.child_form}" ({child_lang})\n'
        f"Shared gloss(es): {'; '.join(c.shared_glosses) or '(none)'}\n\n"
        f'Is "{c.child_form}" a later-era reflex (descendant) of "{c.parent_form}"?'
    )
    return _REFLEX_JUDGE_SYSTEM, user


def parse_reflex_verdict(raw: dict) -> ReflexVerdict | None:
    """Coerce an LLM JSON response into a :class:`ReflexVerdict`, or None if
    unusable (so the caller skips rather than crashing the run)."""
    if not isinstance(raw, dict):
        return None
    val = raw.get("is_reflex")
    if isinstance(val, str):
        val = val.strip().lower() in {"true", "yes", "y", "1"}
    if not isinstance(val, bool):
        return None
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in CONFIDENCE_RANK:
        conf = "low"
    reason = str(raw.get("reason", "")).strip()[:200]
    return ReflexVerdict(is_reflex=val, confidence=conf, reason=reason)


def reflex_verdict_to_row(
    c: ReflexLinkCandidate, v: ReflexVerdict, min_confidence: str
) -> dict[str, str]:
    """Build the ``_collapses.jsonl`` row for a reflex verdict. A confident
    reflex verdict becomes an ``inherits`` LINK (child inherits from parent);
    anything else is a no-op ``inherits: ""`` row that records the judgment and
    lets re-runs skip the candidate. ``ref`` is the child so the row keys by the
    reflex being linked (mirrors the fold ``ref``=from convention)."""
    link = v.is_reflex and CONFIDENCE_RANK[v.confidence] >= CONFIDENCE_RANK[min_confidence]
    return {
        "_type": "collapse",
        "ref": c.child_ref,
        "inherits": c.parent_ref if link else "",
        "edge_type": "inheritance",
        "method": LLM_REFLEX_LINK_METHOD if link else LLM_REFLEX_REJECT_METHOD,
        "confidence": v.confidence,
        "reason": v.reason,
    }
