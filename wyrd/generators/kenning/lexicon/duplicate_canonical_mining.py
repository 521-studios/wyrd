"""Duplicate-canonical finder (wyrd-u6fn.5) — find two SEPARATE canonical etymon
nodes that are actually the same morpheme, and author reversible same-morpheme
IDENTITY assertions to collapse them.

Concrete failure (the regression target): Old-English 'new' exists as TWO
never-merged canonical etymons — 671826 'niwe' (cluster 330032) and 369498 'ne'
(cluster 369500) — both glossed 'new'. They sit in different cognate clusters, so
the same toponym decomposes differently per source. ``merged_into_id`` (D22) never
unified them because the surfaces diverge; ``cognate_id`` is relational, not
identity, so it can't either.

Approach — same shape as the spurious-breakdown finder (wyrd-u6fn.6), but the edge
is Family-A IDENTITY, not Family-C, and the decision is two-pass + adversarial:

  1. PRE-SCREEN (deterministic, recall-biased): pair canonical etymons
     (``merged_into_id IS NULL``) within a language whose gloss token-sets overlap
     (Jaccard >= threshold). Surface is deliberately NOT required — same morpheme,
     different spelling is the whole point.
  2. PROPOSE (LLM): are these two the same morpheme, or homographs that merely
     share a gloss? Default to leave-separate.
  3. ADVERSARIAL VERIFY (LLM, independent): a skeptic tries to REFUTE the merge.
     The combined confidence is floored by the weaker of propose / refute-survival
     (a wrong identity-collapse is corruption; a missed merge is harmless — D46).
  4. AUTHOR at/above the high gate; QUEUE the rest to an audit log. Authoring is
     the apply decision: ``merge-canonical`` is structural (D22 union) and the L3
     projection does not confidence-gate it, so we never author below threshold.

Authoring a confirmed pair (winner = smaller etymon id, deterministic): mint each
etymon's own deterministic ``canonical_morpheme`` hub + ``bind`` it there, then
``merge-canonical`` the two hubs. Minting each etymon's *own* hub (rather than
binding both to one) means the per-etymon binds match the ids the legacy rollup
would use (same node id => same hub, not a conflict — projection §_legacy_identity);
the merge unions the two hubs so a multi-member family on either side comes along.
Reversible: a ``retract`` of the merge-canonical un-collapses them.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from wyrd.generators.kenning.canonicalization.assertions import (
    Assertion,
    NodeRef,
    mint_canonical_id,
)
from wyrd.generators.kenning.lexicon.collapse_merge import CONFIDENCE_RANK
from wyrd.generators.kenning.lexicon.variant_fold_detect import _gloss_tokens

METHOD = "duplicate-canonical-finder-v1"
DEFAULT_SOURCE = "duplicate-canonical-audit"
_NODE = "canonical_morpheme"

# Pre-screen: pair canonical etymons (same language) whose gloss token-sets reach
# at least this Jaccard overlap. Recall-biased — the LLM does precision.
_MIN_GLOSS_OVERLAP = 0.5
# A (language, gloss-token) bucket bigger than this is non-discriminative (e.g. a
# token shared by thousands of etymons) and would blow up O(n^2); skip + report it.
_MAX_BUCKET = 200


@dataclass(frozen=True)
class DuplicateCandidate:
    """A pair of distinct canonical etymons sharing enough gloss to be suspected
    the same morpheme — a suspect for the LLM to judge."""

    a_id: int
    a_form: str
    b_id: int
    b_form: str
    language: str
    a_glosses: tuple[str, ...]
    b_glosses: tuple[str, ...]
    gloss_overlap: float  # Jaccard of gloss content-tokens

    @property
    def winner_id(self) -> int:
        return min(self.a_id, self.b_id)

    @property
    def loser_id(self) -> int:
        return max(self.a_id, self.b_id)


@dataclass(frozen=True)
class MergeVerdict:
    same: bool  # True → the two etymons are the same morpheme (COLLAPSE)
    confidence: Literal["high", "medium", "low"]
    reason: str


@dataclass
class DetectResult:
    candidates: list[DuplicateCandidate] = field(default_factory=list)
    dropped_buckets: int = 0  # (language, token) buckets skipped for exceeding the cap
    bucket_cap: int = _MAX_BUCKET


def _maybe_pair(da: dict, db: dict, a: int, b: int, min_gloss_overlap: float):
    """A DuplicateCandidate for two distinct canonical etymons if they aren't
    already collapsed and their gloss token-sets reach the Jaccard floor; else None."""
    if da["cm_id"] is not None and da["cm_id"] == db["cm_id"]:
        return None  # already collapsed into the same hub
    ta, tb = da["tokens"], db["tokens"]
    if not ta or not tb:
        return None
    jac = len(ta & tb) / len(ta | tb)
    if jac < min_gloss_overlap:
        return None
    return DuplicateCandidate(
        a_id=a,
        a_form=da["form"],
        b_id=b,
        b_form=db["form"],
        language=da["lang"],
        a_glosses=tuple(sorted(da["glosses"])),
        b_glosses=tuple(sorted(db["glosses"])),
        gloss_overlap=round(jac, 3),
    )


def _emit_pairs(
    by_id: dict[int, dict],
    index: dict[tuple[str, str], set[int]],
    *,
    min_gloss_overlap: float,
    max_bucket: int,
    res: DetectResult,
) -> None:
    seen: set[tuple[int, int]] = set()
    for ids in index.values():
        if len(ids) > max_bucket:
            res.dropped_buckets += 1
            continue
        ordered = sorted(ids)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                cand = _maybe_pair(by_id[a], by_id[b], a, b, min_gloss_overlap)
                if cand is not None:
                    res.candidates.append(cand)


def detect_candidates(
    conn: sqlite3.Connection,
    *,
    min_gloss_overlap: float = _MIN_GLOSS_OVERLAP,
    max_bucket: int = _MAX_BUCKET,
    limit: int | None = None,
) -> DetectResult:
    """Pre-screen: distinct canonical etymons (merged_into_id IS NULL) in the same
    language whose gloss token-sets overlap by Jaccard >= ``min_gloss_overlap``.
    Pairs already sharing a canonical_morpheme_id (collapsed by a prior run) are
    skipped. Recall-biased — the two-pass LLM judge decides sameness."""
    prev_factory = conn.row_factory  # restore below — don't mutate a shared conn
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT e.id AS id, e.language AS lang, e.canonical_form AS form,
                   e.canonical_morpheme_id AS cm_id, g.gloss AS gloss
            FROM etymon e
            LEFT JOIN etymon_gloss g ON g.etymon_id = e.id
            WHERE e.merged_into_id IS NULL
            """
        ).fetchall()
    finally:
        conn.row_factory = prev_factory

    by_id: dict[int, dict] = {}
    for r in rows:
        d = by_id.setdefault(
            r["id"],
            {
                "lang": r["lang"] or "",
                "form": r["form"] or "",
                "cm_id": r["cm_id"],
                "glosses": set(),
                "tokens": set(),
            },
        )
        if r["gloss"]:
            d["glosses"].add(r["gloss"])
            d["tokens"] |= _gloss_tokens(r["gloss"])

    # Inverted index: (language, gloss-token) -> etymon ids that carry it.
    index: dict[tuple[str, str], set[int]] = defaultdict(set)
    for eid, d in by_id.items():
        for tok in d["tokens"]:
            index[(d["lang"], tok)].add(eid)

    res = DetectResult(bucket_cap=max_bucket)
    _emit_pairs(by_id, index, min_gloss_overlap=min_gloss_overlap, max_bucket=max_bucket, res=res)
    res.candidates.sort(key=lambda c: (-c.gloss_overlap, c.a_id, c.b_id))
    if limit is not None:
        res.candidates = res.candidates[:limit]
    return res


# --- LLM: propose (same morpheme?) then adversarially refute -----------------

_PROPOSE_SYSTEM = (
    "You judge a historical ENGLISH/British PLACE-NAME etymology lexicon. Two "
    "canonical etymon entries in the SAME language share a similar gloss. Decide "
    "whether they are the SAME morpheme recorded under two spellings (so they "
    "should be unified) or DISTINCT words that merely overlap in meaning "
    "(homographs/near-synonyms that must stay separate). Same morpheme means a "
    "single etymological lexeme (e.g. OE 'niwe' and 'ne' BOTH meaning 'new' are "
    "spelling/abstraction variants of one word; but 'ea' (river) and 'eg' (island) "
    "are DISTINCT despite both being watery). Default to NOT-same unless the "
    "identity is clear. Reply ONLY with a JSON object: "
    '{"same_morpheme": true|false, "confidence": "high"|"medium"|"low", "reason": "<one short clause>"}.'
)

_REFUTE_SYSTEM = (
    "You are a skeptical historical-linguistics referee. A proposal claims two "
    "place-name etymons in the same language are the SAME morpheme and should be "
    "merged into one canonical node. Your job is to REFUTE it if you can: are they "
    "actually DISTINCT words (different roots, different declension class, only a "
    "coincidental gloss overlap, one a derivative not the lemma)? A wrong merge "
    "corrupts the lexicon, a missed merge is harmless — so refute on reasonable "
    "doubt. Confirm only if the identity genuinely holds. Reply ONLY with a JSON "
    'object: {"refuted": true|false, "confidence": "high"|"medium"|"low", "reason": "<one short clause>"}.'
)


def _pair_lines(c: DuplicateCandidate) -> str:
    return (
        f"Language: {c.language}\n"
        f'A: "{c.a_form}" — glosses: {", ".join(c.a_glosses) or "(none)"}\n'
        f'B: "{c.b_form}" — glosses: {", ".join(c.b_glosses) or "(none)"}'
    )


def build_propose_prompt(c: DuplicateCandidate) -> tuple[str, str]:
    user = _pair_lines(c) + "\nAre A and B the same morpheme (one word, two spellings)?"
    return _PROPOSE_SYSTEM, user


def build_refute_prompt(c: DuplicateCandidate) -> tuple[str, str]:
    user = (
        _pair_lines(c) + "\nA proposal says A and B are the same morpheme and should be merged. "
        "Refute it if they are actually distinct words."
    )
    return _REFUTE_SYSTEM, user


def _norm_conf(raw: object) -> Literal["high", "medium", "low"]:
    conf = str(raw or "").strip().lower()
    return conf if conf in CONFIDENCE_RANK else "low"  # type: ignore[return-value]


def _as_bool(val: object) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes", "y", "1"}
    return None


@dataclass(frozen=True)
class _Propose:
    same: bool
    confidence: Literal["high", "medium", "low"]
    reason: str


@dataclass(frozen=True)
class _Refute:
    refuted: bool
    confidence: Literal["high", "medium", "low"]
    reason: str


def parse_propose(raw: dict) -> _Propose | None:
    if not isinstance(raw, dict):
        return None
    same = _as_bool(raw.get("same_morpheme"))
    if same is None:
        return None
    return _Propose(
        same, _norm_conf(raw.get("confidence")), str(raw.get("reason", "")).strip()[:200]
    )


def parse_refute(raw: dict) -> _Refute | None:
    if not isinstance(raw, dict):
        return None
    refuted = _as_bool(raw.get("refuted"))
    if refuted is None:
        return None
    return _Refute(
        refuted, _norm_conf(raw.get("confidence")), str(raw.get("reason", "")).strip()[:200]
    )


def _min_conf(a: str, b: str) -> Literal["high", "medium", "low"]:
    lo = min(CONFIDENCE_RANK[a], CONFIDENCE_RANK[b])
    return {0: "low", 1: "medium", 2: "high"}[lo]  # type: ignore[return-value]


def combine_verdict(prop: _Propose | None, refute: _Refute | None) -> MergeVerdict:
    """Two-pass → one verdict. Leave-separate unless the proposal says same AND the
    skeptic fails to refute; combined confidence floored by the weaker side."""
    if prop is None:
        return MergeVerdict(False, "low", "no proposal")
    if not prop.same:
        return MergeVerdict(False, prop.confidence, prop.reason or "not the same morpheme")
    if refute is None:
        return MergeVerdict(False, "low", "verify failed — leave separate")
    if refute.refuted:
        return MergeVerdict(False, refute.confidence, f"refuted: {refute.reason}")
    return MergeVerdict(
        True, _min_conf(prop.confidence, refute.confidence), refute.reason or prop.reason
    )


def same_morpheme_assertions(
    c: DuplicateCandidate, v: MergeVerdict, *, source: str = DEFAULT_SOURCE, actor: str = ""
) -> list[Assertion]:
    """Reversible Family-A authoring to collapse the pair: mint each etymon's own
    deterministic hub + bind it there (dedups with the legacy rollup for a root,
    gives a singleton a hub), then merge-canonical the two hubs (D22 union)."""
    win_id, lose_id = c.winner_id, c.loser_id
    forms = {c.a_id: c.a_form, c.b_id: c.b_form}
    h_win = mint_canonical_id(_NODE, c.language, forms[win_id])
    h_lose = mint_canonical_id(_NODE, c.language, forms[lose_id])
    rationale = (
        f"same morpheme: '{forms[win_id]}' / '{forms[lose_id]}' ({c.language}), "
        f"shared gloss; {v.reason}"
    )
    out = []
    for eid, hub in ((win_id, h_win), (lose_id, h_lose)):
        out.append(
            Assertion(
                predicate="mint-canonical",
                subject=NodeRef(_NODE, hub),
                confidence="high",
                method=METHOD,
                source=source,
                actor=actor,
                rationale=f"hub for etymon {eid} ('{forms[eid]}')",
            ).with_id()
        )
        out.append(
            Assertion(
                predicate="bind",
                subject=NodeRef("etymon", str(eid)),
                object=NodeRef(_NODE, hub),
                qualifiers={"kind": "same-morpheme"},
                confidence=v.confidence,
                method=METHOD,
                source=source,
                actor=actor,
                rationale=rationale,
            ).with_id()
        )
    out.append(
        Assertion(
            predicate="merge-canonical",
            subject=NodeRef(_NODE, h_win),
            object=NodeRef(_NODE, h_lose),
            confidence=v.confidence,
            method=METHOD,
            source=source,
            actor=actor,
            rationale=rationale,
        ).with_id()
    )
    return out
