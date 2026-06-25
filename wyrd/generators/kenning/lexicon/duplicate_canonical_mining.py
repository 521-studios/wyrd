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
from wyrd.generators.kenning.lexicon.etymon_refs import etymon_ref
from wyrd.generators.kenning.lexicon.genitive_priors import fold_surface
from wyrd.generators.kenning.lexicon.morpheme_surface import _BOUNDARY_DASHES
from wyrd.generators.kenning.lexicon.variant_fold_detect import _gloss_tokens

METHOD = "duplicate-canonical-finder-v1"
DEFAULT_SOURCE = "duplicate-canonical-audit"
_NODE = "canonical_morpheme"

# A morpheme-boundary dash can be drawn with ANY dash-like codepoint, not just the
# ASCII hyphen — mining from PDF/Wiktionary/etymonline routinely yields U+2010 /
# en-dash forms, and ``normalize_morpheme_surface`` preserves *interior* dashes
# (it trims only boundaries). So ``gos–wic`` (en-dash) must read as a compound
# exactly like ``gos-wic``. Reuse the canonical D45 dash set as the membership
# test (the same single-source set #778 reused for the cluster key).
_DASH_CHARS = frozenset(_BOUNDARY_DASHES)


def _has_dash(form: str) -> bool:
    """True if ``form`` contains a dash of any variant (compound marker, D45)."""
    return any(ch in _DASH_CHARS for ch in form)


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
    # High-precision compound exclusion: a dash (ANY variant — :func:`_has_dash`)
    # marks a morpheme boundary, so a dashed form is a compound. If the two forms
    # don't fold to the SAME string, one is a compound and the other its constituent
    # (or a different compound) — not one etymon (gōs vs gos-wic). A fold-equal dashed
    # pair is the same etymon under a punctuation/diacritic variant (wulfpytt vs
    # wulf-pytt, kaup-maðr vs kaup-madr) and is kept for the LLM to judge. da["fold"]
    # is precomputed per etymon by detect_candidates (required key) so this O(n^2)
    # candidate loop doesn't re-fold.
    if (_has_dash(da["form"]) or _has_dash(db["form"])) and da["fold"] != db["fold"]:
        return None
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


def _reflex_link_child_roots(conn: sqlite3.Connection) -> set[int]:
    """Etymon lemma-roots that roll up to an ANCESTOR via a curated reflex-link
    inheritance edge — the ``collapse``-sourced ``etymon_descent`` edges that
    :func:`bundle._export.build_family_rollup` follows (wyrd-rogd.15). Such a root
    passes the cheap ``lemma_id = id`` "is-a-root" screen yet is NOT its legacy
    rollup root: the rollup hops it onto its ancestor's hub. The lemma-root is
    ``COALESCE(merged_into_id, lemma_id, id)`` (build_family_rollup's ``_lemma_root``);
    a child-root that lands on a DIFFERENT parent-root genuinely hops, so exclude it.
    Lazy import — enrichment imports the lexicon package (cycle), same as the rollup."""
    from wyrd.generators.kenning.enrichment import COLLAPSE_VARIANT_SOURCE_ID

    edges = conn.execute(
        "SELECT parent_id, child_id FROM etymon_descent "
        "WHERE edge_type = 'inheritance' AND source_id = ?",
        (COLLAPSE_VARIANT_SOURCE_ID,),
    ).fetchall()
    endpoints = {e for row in edges for e in (row[0], row[1])}
    if not endpoints:
        return set()
    placeholders = ",".join("?" * len(endpoints))
    root = {
        r[0]: (r[1] or r[2] or r[0])
        for r in conn.execute(
            f"SELECT id, merged_into_id, lemma_id FROM etymon WHERE id IN ({placeholders})",
            tuple(endpoints),
        )
    }
    out: set[int] = set()
    for parent_id, child_id in edges:
        child_root = root.get(child_id, child_id)
        if child_root != root.get(parent_id, parent_id):
            out.add(child_root)
    return out


def detect_candidates(
    conn: sqlite3.Connection,
    *,
    min_gloss_overlap: float = _MIN_GLOSS_OVERLAP,
    max_bucket: int = _MAX_BUCKET,
    limit: int | None = None,
) -> DetectResult:
    """Pre-screen: distinct canonical etymons that are their own legacy family ROOT,
    in the same language, whose gloss token-sets overlap by Jaccard >=
    ``min_gloss_overlap``. Pairs already sharing a canonical_morpheme_id (collapsed by
    a prior run) are skipped. Recall-biased — the two-pass LLM judge decides sameness.

    Roots only (``merged_into_id IS NULL AND (lemma_id IS NULL OR lemma_id = id)``,
    plus the reflex-link exclusion below): the authoring binds each etymon to a hub
    minted from ITS OWN form, which matches the legacy rollup hub only when the etymon
    is its family root. Pairing a non-root survivor (an inflection child, OR a
    reflex-link child that hops onto its ancestor's hub — see
    :func:`_reflex_link_child_roots`) would mint a divergent hub, conflict with the
    legacy bind (D24) and leave the etymon unbound — so we exclude them (a missed
    merge is harmless, D46)."""
    # Tuple rows + index unpack — no need to mutate the shared conn's row_factory.
    rows = conn.execute(
        """
        SELECT e.id, e.language, e.canonical_form, e.canonical_morpheme_id, g.gloss
        FROM etymon e
        LEFT JOIN etymon_gloss g ON g.etymon_id = e.id
        WHERE e.merged_into_id IS NULL AND (e.lemma_id IS NULL OR e.lemma_id = e.id)
        """
    ).fetchall()

    by_id: dict[int, dict] = {}
    for eid, lang, form, cm_id, gloss in rows:
        d = by_id.setdefault(
            eid,
            {
                "lang": lang or "",
                "form": form or "",
                "fold": fold_surface(
                    form
                ),  # precomputed per etymon — the O(n^2) pair loop reuses it
                "cm_id": cm_id,
                "glosses": set(),
                "tokens": set(),
            },
        )
        if gloss:
            d["glosses"].add(gloss)
            d["tokens"] |= _gloss_tokens(gloss)

    # Drop reflex-link children: they pass the cheap lemma_id=id "root" screen but
    # roll up to an ANCESTOR's hub in the legacy rollup, so binding them from their
    # OWN form conflicts with that bind (D24) and severs their reflex family. The
    # SQL screen can't see the inheritance hop; exclude them here (D46-harmless).
    for eid in _reflex_link_child_roots(conn):
        by_id.pop(eid, None)

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

# SAME-MORPHEME means ONE etymological lexeme recorded twice — only spelling /
# phonological / diacritic variants of a single word, or inflections of one lexeme.
# It does NOT mean "etymologically related". The judge must reject three classes
# that share a root but are NOT one lexeme (they corrupt the canonical graph if
# merged) — calibrated against the wyrd-u6fn.5 apply-run false-positives.
_SAME_MORPHEME_RULES = (
    "SAME MORPHEME = ONE lexeme written or inflected two ways. ONLY these count as same:\n"
    "  - spelling / phonological / diacritic variants of one word "
    "(mad=mat, mylen=myln, scēad=scéad, sūþ=sud, wulfpytt=wulf-pytt);\n"
    "  - inflections / case forms of ONE lexeme (crux=crucis, pons=pontem, bōc=bēce);\n"
    "  - a genuine spelling variant of ONE personal name (Katharine=Katherine, "
    "Margerie=Margery, Agnes=Annis).\n"
    "These are NOT the same morpheme (they share a ROOT but are DISTINCT lexemes) "
    "so answer NOT-same:\n"
    "  - COMPOUND vs its constituent/head: B is A plus another element "
    "(treow != plūm-trēow, bȳ != kross-bý, vellir != thingvellir, "
    "gōs != gos-wic, gortna != gortmore, bille != da-bille). A dash, or one form "
    "built by adding material to the other, signals a compound.\n"
    "  - DERIVATION: noun<->verb (willa != willian), base<->adjective-derivative "
    "(west != westerne, horu != horig, wilig != wiligen), nominalization "
    "(iuo != iubhar), zero-derivation or a different derivational suffix "
    "(linden != lind, byrgen != byrgels), extra suffix like -red/-hood "
    "(hund != hundred).\n"
    "  - DISTINCT NAME or HYPOCORISTIC / pet-form: a nickname is a different "
    "name-form (Margerie != Meg, Joan != Jane, Gilot != Gillota, Malkin != Mallin).\n"
    "When unsure, answer NOT-same: a missed merge is harmless, a wrong merge corrupts."
)

_PROPOSE_SYSTEM = (
    "You judge a historical ENGLISH/British PLACE-NAME etymology lexicon. Two "
    "canonical etymon entries in the SAME language share a similar gloss. Decide "
    "whether they are the SAME morpheme (one lexeme, to be unified) or merely "
    "related/overlapping words that must stay separate.\n" + _SAME_MORPHEME_RULES + "\n"
    "Reply ONLY with a JSON object: "
    '{"same_morpheme": true|false, "confidence": "high"|"medium"|"low", "reason": "<one short clause>"}.'
)

_REFUTE_SYSTEM = (
    "You are a skeptical historical-linguistics referee. A proposal claims two "
    "place-name etymons in the same language are the SAME morpheme and should be "
    "merged into one canonical node. REFUTE it unless they are genuinely ONE lexeme "
    "(a spelling/diacritic variant or an inflection of the same word). In "
    "particular REFUTE when B is a COMPOUND containing A, a DERIVATION of A "
    "(noun<->verb, base<->adjective-derivative, nominalization, extra suffix), or a "
    "DISTINCT NAME / hypocoristic.\n" + _SAME_MORPHEME_RULES + "\n"
    "A wrong merge corrupts the lexicon, a missed merge is harmless. Reply ONLY "
    'with a JSON object: {"refuted": true|false, "confidence": "high"|"medium"|"low", "reason": "<one short clause>"}.'
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
        same, _norm_conf(raw.get("confidence")), str(raw.get("reason") or "").strip()[:200]
    )


def parse_refute(raw: dict) -> _Refute | None:
    if not isinstance(raw, dict):
        return None
    refuted = _as_bool(raw.get("refuted"))
    if refuted is None:
        return None
    return _Refute(
        refuted, _norm_conf(raw.get("confidence")), str(raw.get("reason") or "").strip()[:200]
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
                # wyrd-c6wu: stable natural-key ref, not the etymon row-id.
                subject=NodeRef("etymon", etymon_ref(c.language, forms[eid])),
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
