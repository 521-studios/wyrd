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

# language → (family, era rank), from the cell map. The rank is a GLOBAL
# first-seen enumeration index over the cell items — it only orders eras
# correctly WITHIN a family (the cell map lists each family's stages
# contiguously, older→newer), so the detector compares ranks only after
# confirming both languages are in the SAME family. We also only pair KNOWN era
# stages here — that excludes cross-family / proto- / non-stage labels (e.g.
# "celtic", "scottish-gaelic"), whose pairs are cognates or borrowings, not
# reflexes.
_LANGUAGE_ERA_RANK: dict[str, int] = {}
_LANGUAGE_FAMILY: dict[str, str] = {}
for _i, ((_fam, _cell), _lang) in enumerate(CANONICAL_LANGUAGE_FOR_CELL.items()):
    _LANGUAGE_ERA_RANK.setdefault(_lang, _i)
    _LANGUAGE_FAMILY.setdefault(_lang, _fam)


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
    norm_glosses: dict[int, set[str]] = defaultdict(set)
    by_gloss: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        rec = by_id.setdefault(
            r["id"], {"lang": r["language"], "form": r["canonical_form"], "glosses": set()}
        )
        rec["glosses"].add(r["gloss"])
        ng = r["gloss"].strip().lower()
        norm_glosses[r["id"]].add(ng)
        by_gloss[ng].add(r["id"])

    # Bulk-load descent connectivity once (a per-pair SELECT would be N+1).
    edges: set[tuple[int, int]] = set()
    for p, c in conn.execute("SELECT parent_id, child_id FROM etymon_descent"):
        edges.add((p, c))

    # (child_ref, parent_ref, child_form, parent_form, shared, sim, parent_rank)
    raw: list[tuple] = []
    seen: set[tuple[int, int]] = set()
    for ids in by_gloss.values():
        if not (2 <= len(ids) <= max_per_gloss):
            continue
        ordered = sorted(ids)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if (a, b) in seen:  # the same pair can sit in two gloss groups
                    continue
                ea, eb = by_id[a], by_id[b]
                if ea["lang"] == eb["lang"]:
                    continue  # same-language variant = FOLD (existing detector), not a link
                ra = _LANGUAGE_ERA_RANK.get(ea["lang"])
                rb = _LANGUAGE_ERA_RANK.get(eb["lang"])
                # Only KNOWN era stages in the SAME family, ranked apart — so
                # orientation is reliable (parent = earlier) and we never pair a
                # cross-family / proto cognate.
                if ra is None or rb is None or ra == rb:
                    continue
                if _LANGUAGE_FAMILY[ea["lang"]] != _LANGUAGE_FAMILY[eb["lang"]]:
                    continue
                fa, fb = _bare(ea["form"]), _bare(eb["form"])
                if not fa or not fb:
                    continue
                sim = difflib.SequenceMatcher(None, fa, fb).ratio()
                if sim < min_similarity or (a, b) in edges or (b, a) in edges:
                    continue
                seen.add((a, b))
                (parent, prank), child = ((a, ra), b) if ra < rb else ((b, rb), a)
                # The child's raw glosses the parent also carries (normalized) —
                # never empty (the pair was grouped on a shared normalized gloss).
                shared = tuple(
                    sorted(
                        g
                        for g in by_id[child]["glosses"]
                        if g.strip().lower() in norm_glosses[parent]
                    )
                )
                raw.append(
                    (
                        f"{by_id[child]['lang']}:{by_id[child]['form']}",
                        f"{by_id[parent]['lang']}:{by_id[parent]['form']}",
                        by_id[child]["form"],
                        by_id[parent]["form"],
                        shared,
                        round(sim, 3),
                        prank,
                    )
                )

    # One best ancestor per child: the most IMMEDIATE (highest parent era rank,
    # i.e. closest to the child), tie-broken by form similarity — so a child
    # never writes multiple same-``ref`` ledger rows where collect_collapses'
    # last-write-wins would pick an arbitrary ancestor.
    best: dict[str, tuple] = {}
    for t in raw:
        cur = best.get(t[0])
        if cur is None or (t[6], t[5]) > (cur[6], cur[5]):
            best[t[0]] = t
    out = [
        ReflexLinkCandidate(
            child_ref=t[0],
            parent_ref=t[1],
            child_form=t[2],
            parent_form=t[3],
            shared_glosses=t[4],
            similarity=t[5],
        )
        for t in best.values()
    ]
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
