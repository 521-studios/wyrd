"""Cognate-descent uplift for unclustered admitted breakdown morphemes (wyrd-zrce.1, D50).

Uplift 1b: an admitted breakdown morpheme (wyrd-oth3) with NO cognate cluster
(``cognate_id`` NULL) that is a cognate of an already-clustered etymon should join
that cluster — so its cognate variants group together and become eligible for
era-reflex projection (wyrd-g96o). Realized as a RELATIONAL ``descends-from`` edge
(D50 Family B, never-collapse), NOT an identity ``bind``: these are cognates/
reflexes (``silly``/``selig``), not the same morpheme. Identity duplicates were
handled by 1a (``same_morpheme_mining``); this is the orthogonal descent axis.

The signal: a cohort etymon ``X`` (unclustered + breakdown) and a CLUSTERED etymon
``Y`` share a **folded surface**. ``Y``'s ``cognate_id`` is its cluster root ``R``
(the most-ancestral known etymon, D27). We author::

    descends-from(subject=etymon X, object=etymon R, edge_type=inheritance)

so the ``cluster-cognates`` root→down BFS (parent-but-never-child) reaches ``X`` and
assigns it ``R``'s ``cognate_id``. We point ``X`` at the cluster ROOT, not at ``Y``:
cross-language cognates are siblings under the proto-root, not parent-child.

Confidence is TIERED against the homograph risk (``ton``=town vs ``ton``=wave: same
fold, unrelated cluster) — precision/recall is the projection's confidence gate to
set (D50.4 leave-separate), not the miner's to hard-code:

* folded match + **gloss overlap** with the cluster → ``medium`` (real cognate
  evidence);
* same-cluster-unique, **surface only** → ``low`` (homograph risk);
* candidates span **>1 cluster** → prefer the unique gloss-overlapping cluster, else
  leave-separate (D46 — ambiguous is harmless duplication, a wrong edge is corruption).

DORMANT until the descends-from→etymon_descent projection (also wyrd-zrce.1) applies
the assertions. Deterministic ids (no timestamp) → re-mining the same corpus is
idempotent (D36.9). LLM-free; the residue (no clustered surface match) is wyrd-zrce.2.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from wyrd.generators.kenning.canonicalization import Assertion, NodeRef
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.genitive_priors import fold_surface

_logger = logging.getLogger(__name__)


def _is_degenerate_ref(ref: str) -> bool:
    """A natural-key ref ``"language:canonical_form"`` whose form is empty or only
    dashes — a degenerate upstream etymon (``*:-``) that should never anchor a
    cognate-descent edge: clustering a morpheme under a formless root is meaningless
    (wyrd-zrce PR review). The form is everything after the FIRST ``:`` (a form may
    itself contain ``:``)."""
    form = ref.split(":", 1)[1] if ":" in ref else ref
    return not form.strip("-")


METHOD = "cognate-descent-uplift-v1"

# The edge_type authored for an inferred cognate descent. ``inheritance`` (the
# default reflex relationship); the evidence is a surface+gloss match, not a cited
# chain, so confidence — not edge_type — carries the uncertainty.
EDGE_TYPE = "inheritance"

# medium = folded match corroborated by gloss overlap; low = surface-only.
Confidence = Literal["medium", "low"]


@dataclass(frozen=True)
class _Target:
    """A clustered etymon a cohort morpheme might descend into."""

    cluster_root: int  # cognate_id (most-ancestral etymon, D27)
    glosses: frozenset[str]


@dataclass(frozen=True)
class _Cohort:
    """An unclustered admitted breakdown morpheme (the ``X`` we try to place)."""

    etymon_id: int
    folded: str
    glosses: frozenset[str]


@dataclass(frozen=True)
class DescentEdge:
    """A mined cognate-descent: ``child_etymon descends-from cluster_root``."""

    child_etymon: int
    cluster_root: int
    confidence: Confidence
    rationale: str


def _glosses_for(db: LexiconDB, ids: set[int]) -> dict[int, frozenset[str]]:
    """Lowercased gloss set per etymon id, only for the ids we care about."""
    acc: dict[int, set[str]] = defaultdict(set)
    for r in db.conn.execute("SELECT etymon_id, gloss FROM etymon_gloss"):
        if r["etymon_id"] in ids and r["gloss"]:
            acc[r["etymon_id"]].add(r["gloss"].strip().lower())
    return {eid: frozenset(g) for eid, g in acc.items()}


def _load(db: LexiconDB) -> tuple[list[_Cohort], dict[str, list[_Target]]]:
    """The cohort (unclustered breakdown morphemes) and the clustered targets
    indexed by folded surface (only folds the cohort actually shares, to bound the
    cost over the ~600k clustered rows).

    The match key is the folded surface ALONE — deliberately cross-language: a
    cognate cluster spans languages (D27), so an Old-English cohort morpheme placing
    into a proto-Germanic/Old-Norse cluster is the intended case, not a leak. (Precision
    against cross-language homographs is the job of the gloss corroboration in
    :func:`_place` + the downstream confidence gate, not a language filter here.)"""
    breakdown = {
        r["etymon_id"]
        for r in db.conn.execute("SELECT DISTINCT etymon_id FROM toponym_etymology_element")
    }

    # Cohort rows first, so we know which folded surfaces a target could match.
    cohort_raw: dict[int, str] = {}  # eid -> folded
    cohort_folds: set[str] = set()
    for r in db.conn.execute(
        "SELECT id, canonical_form FROM etymon WHERE cognate_id IS NULL AND merged_into_id IS NULL"
    ):
        if r["id"] not in breakdown:
            continue
        folded = fold_surface(r["canonical_form"])
        if not folded:
            continue
        cohort_raw[r["id"]] = folded
        cohort_folds.add(folded)

    # Clustered targets, kept only when their fold matches a cohort fold.
    target_raw: dict[str, list[tuple[int, int]]] = defaultdict(list)  # fold -> (root, eid)
    target_ids: set[int] = set()
    for r in db.conn.execute(
        "SELECT id, cognate_id, canonical_form FROM etymon WHERE cognate_id IS NOT NULL"
    ):
        folded = fold_surface(r["canonical_form"])
        if folded not in cohort_folds:
            continue
        target_raw[folded].append((r["cognate_id"], r["id"]))
        target_ids.add(r["id"])

    glosses = _glosses_for(db, set(cohort_raw) | target_ids)
    empty: frozenset[str] = frozenset()

    cohort = [
        _Cohort(etymon_id=eid, folded=folded, glosses=glosses.get(eid, empty))
        for eid, folded in cohort_raw.items()
    ]
    targets_by_fold = {
        fold: [_Target(cluster_root=root, glosses=glosses.get(tid, empty)) for root, tid in rows]
        for fold, rows in target_raw.items()
    }
    return cohort, targets_by_fold


def _place(x: _Cohort, candidates: list[_Target]) -> tuple[int, Confidence] | None:
    """The cluster root ``x`` should descend into, plus confidence. Unique cluster →
    medium if gloss-corroborated else low. Multiple clusters → the unique
    gloss-overlapping one (medium), else leave-separate (D46). The cohort is
    ``cognate_id`` NULL, so a candidate's root is never ``x`` itself."""
    by_root: dict[int, list[_Target]] = defaultdict(list)
    for t in candidates:
        by_root[t.cluster_root].append(t)
    gloss_roots = {root for root, ts in by_root.items() if any(x.glosses & t.glosses for t in ts)}
    if len(by_root) == 1:
        (root,) = by_root
        return root, ("medium" if gloss_roots else "low")
    if len(gloss_roots) == 1:
        (root,) = gloss_roots
        return root, "medium"
    return None  # no candidates, or ambiguous across clusters with no unique gloss


def mine_cognate_descents(db: LexiconDB) -> list[DescentEdge]:
    """Find unclustered breakdown morphemes that fold-match a clustered etymon and
    place each into the matched cognate cluster via a ``descends-from`` edge to the
    cluster root. Deterministic; sorted by ``(child_etymon)``."""
    cohort, targets_by_fold = _load(db)
    edges: list[DescentEdge] = []
    for x in cohort:
        candidates = targets_by_fold.get(x.folded)
        if not candidates:
            continue
        placed = _place(x, candidates)
        if placed is None:
            continue
        root, confidence = placed
        corroborated = "gloss-corroborated" if confidence == "medium" else "surface-only"
        edges.append(
            DescentEdge(
                child_etymon=x.etymon_id,
                cluster_root=root,
                confidence=confidence,
                rationale=(
                    # wyrd-s964: NO row-id in the rationale — the matched cluster's
                    # identity is the assertion's object ref (the root's natural key);
                    # the folded surface names the match here.
                    f"cognate-descent uplift: unclustered breakdown morpheme folds to "
                    f"'{x.folded}', matching a clustered cognate ({corroborated})"
                ),
            )
        )
    edges.sort(key=lambda e: e.child_etymon)
    return edges


def descent_assertions(
    edges: list[DescentEdge], refs: dict[int, str], *, source: str, actor: str = ""
) -> list[Assertion]:
    """Author one ``descends-from`` assertion per mined edge (Family B relational —
    no canonical node, unlike the identity binds). Deterministic ids (D36.9).

    ``refs`` maps ``etymon_id -> "language:canonical_form"`` (build it with
    :func:`etymon_refs.etymon_refs_for` over the edges' endpoint ids). wyrd-c6wu /
    D50.1: subject/object are the STABLE natural key, never the ``etymon.id``
    row-id — so the committed JSONL survives a rebuild's id reassignment. An edge
    whose endpoint id is absent from ``refs`` is a programming error (the caller
    must cover every endpoint); it raises ``KeyError`` rather than silently
    emitting a broken ref.

    An edge whose child or root resolves to a degenerate ``*:-`` ref (empty/dash
    canonical_form) is dropped + logged: anchoring a cognate cluster on a formless
    etymon is meaningless (wyrd-zrce PR review; the dash-form etymons themselves are
    wyrd-aicu's cleanup)."""
    out: list[Assertion] = []
    skipped = 0
    for e in edges:
        child_ref, root_ref = refs[e.child_etymon], refs[e.cluster_root]
        if _is_degenerate_ref(child_ref) or _is_degenerate_ref(root_ref):
            skipped += 1
            continue
        out.append(
            Assertion(
                predicate="descends-from",
                subject=NodeRef("etymon", child_ref),
                object=NodeRef("etymon", root_ref),
                qualifiers={"edge_type": EDGE_TYPE},
                confidence=e.confidence,
                method=METHOD,
                source=source,
                actor=actor,
                rationale=e.rationale,
            ).with_id()
        )
    if skipped:
        _logger.info(
            "descent_assertions: dropped %d edge(s) anchored on a degenerate (empty/dash) "
            "canonical_form endpoint",
            skipped,
        )
    return out
