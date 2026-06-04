"""Cross-cluster same-meaning MERGE candidate detection (v3ow P3).

Where ``variant_fold_detect`` (wyrd-v3ow.1) folds a BARREN duplicate (cluster
≤ 1) into its rich lemma, this handles the harder case the barren fold can't
reach: two SAME-language clusters that are the SAME morpheme but tracked
separately — a MINOR cluster (a parallel anglicized / dialectal spelling
lineage, e.g. OE ``field`` cluster 6) and the MAJOR canonical cluster it
duplicates (OE ``feld`` cluster 224). Because the minor isn't barren, the v3ow.1
detector skips it, so generation can resolve a surface to the minor lineage —
which dead-ends short of the modern era (``field`` descends only to toponyms
like *Hundersfield*, never the bare modern reflex) — and the era-grid stops at
Middle English even though the word is plainly modern.

Signal: SAME language + a shared gloss CONTENT TOKEN + (form similarity OR a
shared consonant skeleton) + a MINOR cluster (``minor_max`` ≥ size ≥ 2) and a
MAJOR cluster at least ``ratio`` × richer (and ≥ ``major_min``) + DIFFERENT
cognate ids + no existing descent edge. The fold op tombstones the minor into
the major, so the surface resolves to the major's full cross-era lineage.

The LLM judge is load-bearing here in a way it isn't for the barren fold: a
non-barren minor cluster is much more often a genuinely DISTINCT morpheme that
merely shares a generic gloss token (``worþ`` 'enclosure' vs ``word`` 'speech'),
and the minor may be a HOMOGRAPH (``field`` = open-land AND fold/crease) where
only the dominant toponym sense merges. The judge decides same-morpheme on the
shared sense; verdicts round-trip ``_collapses.jsonl`` and replay deterministically
at rebuild (the LLM is NEVER re-run at rebuild).
"""

from __future__ import annotations

import difflib
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

from wyrd.generators.kenning.lexicon.collapse_merge import CONFIDENCE_RANK
from wyrd.generators.kenning.lexicon.variant_fold_detect import (
    _bare,
    _consonant_skeleton,
    _gloss_tokens,
)

LLM_MEANING_MERGE_METHOD = "llm-meaning-merge-v1"
LLM_MEANING_REJECT_METHOD = "llm-meaning-merge-rejected-v1"


@dataclass(frozen=True)
class MeaningMergeCandidate:
    """A same-language minor↔major cluster pair for the LLM to judge: is the
    MINOR the same morpheme as the MAJOR (a variant lineage to merge), or a
    distinct morpheme sharing a gloss token?"""

    minor_ref: str  # the minor lineage to merge away, "<language>:<form>"
    major_ref: str  # the rich canonical lineage to merge INTO
    minor_form: str
    major_form: str
    minor_glosses: tuple[str, ...]
    major_glosses: tuple[str, ...]
    similarity: float
    minor_cluster_size: int
    major_cluster_size: int


def detect_meaning_merge_candidates(
    conn: sqlite3.Connection,
    *,
    min_similarity: float = 0.7,
    minor_max: int = 10,
    major_min: int = 40,
    ratio: int = 5,
    max_per_gloss: int = 600,
) -> list[MeaningMergeCandidate]:
    """Same-language minor↔major cross-cluster merge candidates. A MINOR cluster
    (``2 ≤ size ≤ minor_max``) merges into a MAJOR cluster (``size ≥ major_min``
    and ``≥ ratio`` × the minor) of the SAME language sharing a gloss CONTENT
    TOKEN, form-similar (difflib ≥ ``min_similarity``) OR sharing a ≥2-consonant
    skeleton, with DIFFERENT cognate ids and no existing descent edge. One best
    major per minor (richest cluster, then highest similarity, then lowest id)."""
    rows = conn.execute(
        "SELECT e.id, e.language, e.canonical_form, e.cognate_id, g.gloss "
        "FROM etymon e JOIN etymon_gloss g ON g.etymon_id = e.id "
        "WHERE e.merged_into_id IS NULL ORDER BY e.id"
    ).fetchall()
    by_id: dict[int, dict] = {}
    by_token: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        eid = r["id"]
        rec = by_id.get(eid)
        if rec is None:
            rec = by_id[eid] = {
                "lang": r["language"],
                "form": r["canonical_form"],
                "bare": _bare(r["canonical_form"]),
                "skel": _consonant_skeleton(r["canonical_form"]),
                "clu": r["cognate_id"],
                "gl": set(),
                "tok": set(),
            }
        rec["gl"].add(r["gloss"])
        toks = _gloss_tokens(r["gloss"])
        rec["tok"].update(toks)
        for t in toks:
            by_token[t].add(eid)

    cluster_size: dict[int, int] = {}
    for r in conn.execute(
        "SELECT cognate_id, count(*) n FROM etymon "
        "WHERE cognate_id IS NOT NULL AND merged_into_id IS NULL GROUP BY cognate_id"
    ):
        cluster_size[r["cognate_id"]] = r["n"]

    edges: set[tuple[int, int]] = set()
    for p, c in conn.execute("SELECT parent_id, child_id FROM etymon_descent"):
        edges.add((p, c))

    def richness(rec: dict) -> int:
        return cluster_size.get(rec["clu"], 0) if rec["clu"] is not None else 0

    # best major per minor id -> (major id, sim)
    best: dict[int, tuple[int, float]] = {}
    for ids in by_token.values():
        if len(ids) < 2 or len(ids) > max_per_gloss:
            continue
        minor = [(i, by_id[i]) for i in ids if 2 <= richness(by_id[i]) <= minor_max]
        major = [(i, by_id[i]) for i in ids if richness(by_id[i]) >= major_min]
        if not minor or not major:
            continue
        for mi, mrec in minor:
            fb, skb, rmi = mrec["bare"], mrec["skel"], richness(mrec)
            if not fb:
                continue
            matcher = difflib.SequenceMatcher(None, fb)
            for ai, arec in major:
                if mrec["lang"] != arec["lang"] or mrec["clu"] == arec["clu"]:
                    continue  # same language, DIFFERENT clusters
                rma = richness(arec)
                if rma < ratio * rmi:
                    continue  # major must clearly dominate
                fr = arec["bare"]
                if not fr or fb == fr:
                    continue
                matcher.set_seq2(fr)
                sim = matcher.ratio()
                skeleton_match = len(skb) >= 2 and skb == arec["skel"]
                if sim < min_similarity and not skeleton_match:
                    continue
                if (mi, ai) in edges or (ai, mi) in edges:
                    continue
                cur = best.get(mi)
                if cur is None or (rma, sim, -ai) > (
                    richness(by_id[cur[0]]),
                    cur[1],
                    -cur[0],
                ):
                    best[mi] = (ai, sim)

    out: list[MeaningMergeCandidate] = []
    for minor, (major, sim) in best.items():
        em, ea = by_id[minor], by_id[major]
        out.append(
            MeaningMergeCandidate(
                minor_ref=f"{em['lang']}:{em['form']}",
                major_ref=f"{ea['lang']}:{ea['form']}",
                minor_form=em["form"],
                major_form=ea["form"],
                minor_glosses=tuple(sorted(em["gl"]))[:4],
                major_glosses=tuple(sorted(ea["gl"]))[:4],
                similarity=round(sim, 3),
                minor_cluster_size=cluster_size.get(em["clu"], 0),
                major_cluster_size=cluster_size.get(ea["clu"], 0),
            )
        )
    out.sort(key=lambda c: (c.major_ref, c.minor_ref))
    return out


@dataclass(frozen=True)
class MeaningMergeVerdict:
    same_morpheme: bool  # minor is the SAME morpheme as major (a variant lineage)
    confidence: str  # "high" | "medium" | "low"
    reason: str


_MERGE_JUDGE_SYSTEM = (
    "You are a historical-linguistics editor de-duplicating a place-name "
    "etymology lexicon. You are given two head-words in the SAME language that "
    "are tracked as SEPARATE word-families but may be the SAME morpheme: a MINOR "
    "lineage (a sparsely-attested, often anglicized or dialectal spelling — e.g. "
    "'field') and the MAJOR canonical lineage it may duplicate (the "
    "well-attested form — e.g. 'feld'). Decide whether the MINOR is the same "
    "morpheme as the MAJOR — the same word, merely a spelling/dialect variant — "
    "so they should be MERGED into one. The minor may be a HOMOGRAPH carrying an "
    "extra unrelated sense (e.g. 'field' also glossed 'fold, crease'); judge YES "
    "if they share the DOMINANT/place-name sense, even then. Answer NO if they "
    "are genuinely DIFFERENT morphemes that merely share a generic gloss word "
    "('worþ' enclosure vs 'word' speech), or unrelated look-alikes. Judge by "
    "sound/spelling closeness AND the shared core meaning. Reply ONLY with a "
    "JSON object: "
    '{"same_morpheme": true|false, "confidence": "high"|"medium"|"low", '
    '"reason": "<one short clause>"}.'
)


def build_merge_judge_prompt(c: MeaningMergeCandidate) -> tuple[str, str]:
    """Return (system, user) prompts for judging a meaning-merge candidate."""
    lang = c.minor_ref.split(":", 1)[0]
    user = (
        f"Language: {lang}\n"
        f'Minor lineage "{c.minor_form}" ({c.minor_cluster_size} relatives) — '
        f"gloss(es): {'; '.join(c.minor_glosses) or '(none)'}\n"
        f'Major lineage "{c.major_form}" ({c.major_cluster_size} relatives) — '
        f"gloss(es): {'; '.join(c.major_glosses) or '(none)'}\n\n"
        f'Is "{c.minor_form}" the same morpheme as "{c.major_form}" (a variant '
        f"to merge), sharing their core place-name sense?"
    )
    return _MERGE_JUDGE_SYSTEM, user


def parse_merge_verdict(raw: dict) -> MeaningMergeVerdict | None:
    """Coerce an LLM JSON response into a :class:`MeaningMergeVerdict`, or None
    if unusable (the caller skips rather than crashing the run)."""
    if not isinstance(raw, dict):
        return None
    val = raw.get("same_morpheme")
    if isinstance(val, str):
        val = val.strip().lower() in {"true", "yes", "y", "1"}
    if not isinstance(val, bool):
        return None
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in CONFIDENCE_RANK:
        conf = "low"
    reason = str(raw.get("reason", "")).strip()[:200]
    return MeaningMergeVerdict(same_morpheme=val, confidence=conf, reason=reason)


def merge_verdict_to_row(
    c: MeaningMergeCandidate, v: MeaningMergeVerdict, min_confidence: str
) -> dict[str, str]:
    """Build the ``_collapses.jsonl`` row for a merge verdict. A confident
    same-morpheme verdict becomes a real FOLD (``into`` the major lineage,
    tombstoning the minor); anything else is a no-op ``into: ""`` row recording
    the judgment so re-runs skip the pair. ``ref`` is the minor (the lineage
    folded away). ``candidate_lemma`` carries the major for (minor, major)
    pair-level dedup, matching the variant-fold convention."""
    merge = v.same_morpheme and CONFIDENCE_RANK[v.confidence] >= CONFIDENCE_RANK[min_confidence]
    return {
        "_type": "collapse",
        "ref": c.minor_ref,
        "into": c.major_ref if merge else "",
        "candidate_lemma": c.major_ref,
        "variant_class": "meaning-merge",
        "method": LLM_MEANING_MERGE_METHOD if merge else LLM_MEANING_REJECT_METHOD,
        "confidence": v.confidence,
        "reason": v.reason,
    }
