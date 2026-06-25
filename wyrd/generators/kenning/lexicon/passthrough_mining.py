"""Passthrough mining (wyrd-h5u1) — stream 1: cross-scholar coarse-vs-fine.

A **composite** toponymic surface (``ington``, ``beretūn``, ``hohtun``) is an
acceptable thing for the matcher to match, but it must be RECORDED as a
**passthrough** that expands to its constituent morphemes (``ington`` →
``ing`` + ``tūn``) so attribution/rendering see the finer morphemes (D51.3).
Otherwise the Occam (fewer-morphemes) tiebreaker prefers the coarse single
composite over the finer correct split, dropping cluster-recall against the
scholarly corpus (measured by the wyrd-aicu.9 grader on the wyrd-oth3 re-emit).

The passthrough is modeled as a D50 Family-B relational **``composed-of``**
assertion (etymon → ordered etymon). Mining it gloss-correctly requires
**scholarly evidence, not surface segmentation** (D51.2, the gloss-blind rule) —
segmentation would wrongly expand ``barton`` → ``bar`` + ``ton``, but the
scholar's breakdown shows ``bere`` + ``tūn`` (barley-farm).

**Stream 1** (cleanest, corpus-only) is the seed: when the SAME toponym is
decomposed by two scholars and one breakdown coarsens a 2+-element span of the
other into a single composite element (the rest aligning), that directly attests
the passthrough — intra-toponym, both gloss-bearing, zero inference. (The
"whole-vs-split" disagreements aicu.9 investigation-3 treated as *noise*
inflating cross-region disagreement — ``burhtun == burh+tūn`` — are reclaimed
here as *signal*.) Guarded against a coincidental element-count mismatch: the
composite's folded surface must **word-break into exactly one cognate-cluster
reflex per constituent** (``ington`` → ``ing`` | ``ton``), which also yields the
constituent **order** (the ``ordinal`` of each ``composed-of`` edge).

Streams 2 (matcher-vs-scholar miss) and 3 (pooling across toponyms sharing the
composite ending) extend coverage and cross-confirm; stream 1 is the seed.
LLM-free, deterministic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations

from wyrd.generators.kenning.canonicalization import Assertion, NodeRef, validate
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.decomposition_grader import ClusterIndex, load_cluster_index
from wyrd.generators.kenning.lexicon.genitive_priors import fold_surface

METHOD = "passthrough-cross-scholar-v1"

# A composite morpheme decomposing into more than this many constituents is
# implausible (toponymic compounds are 2, occasionally 3); the cap also bounds
# the ``permutations(fine_only)`` search in ``_segment_order`` (N! — would
# explode for a large breakdown), so a pathological many-element mismatch is
# skipped rather than hung on.
_MAX_CONSTITUENTS = 4

# Structural shapes shared across the coarse/fine passthrough scan, named so the
# accumulator signatures stay legible (wyrd-gzjo review):
#   _Key — a passthrough's identity (composite cluster + ordered constituents),
#   _Obs — an observation (composite etymon id + ordered constituent etymon ids),
#   _Forms — the matching surface forms,
#   _Candidate — one detected (key, observation, forms) triple.
_Key = tuple[str, tuple[str, ...]]
_Obs = tuple[int, tuple[int, ...]]
_Forms = tuple[str, tuple[str, ...]]
_Candidate = tuple[_Key, _Obs, _Forms]


def _cluster_surfaces(db: LexiconDB, index: ClusterIndex) -> dict[str, set[str]]:
    """cluster key → folded surfaces it can take (reflex surfaces from the index,
    plus etymon canonical forms so an OE compound's canonical form like
    ``beretūn`` is available for the word-break check even when its reflexes
    aren't recorded)."""
    out: dict[str, set[str]] = defaultdict(set)
    for surface, clusters in index.surface_clusters.items():
        for cluster in clusters:
            out[cluster].add(surface)
    for row in db.conn.execute("SELECT id, cognate_id, canonical_form FROM etymon"):
        cid = row["cognate_id"]
        cluster = f"c{cid}" if cid is not None else f"e{row['id']}"
        folded = fold_surface(row["canonical_form"])
        if folded:
            out[cluster].add(folded)
    return out


def _segment_order(
    surface: str, clusters: list[str], cluster_surfaces: dict[str, set[str]]
) -> tuple[str, ...] | None:
    """If ``surface`` segments into exactly one reflex surface per cluster under a
    UNIQUE cluster order, return that order (left to right) — both the confirmation
    that the composite IS the merge of these constituents AND the ``ordinal``
    ordering. ``None`` when no clean segmentation exists OR the segmentation is
    AMBIGUOUS.

    Deterministic. An ambiguous tiling — more than one cluster order tiles the same
    surface, which happens when constituent clusters share folded reflex surfaces
    (two clusters both carrying ``tun``/``ton`` tile ``tunton`` as either
    ``tun``|``ton`` or ``ton``|``tun``) — is left SEPARATE (D46), not resolved by an
    arbitrary cluster-key sort: a guessed order would commit a wrong ``ordinal`` to
    the ``composed-of`` assertions (a wrong split corrupts; a missed passthrough is a
    harmless recall loss). Mirrors the genitive_priors "touches BOTH / NEITHER →
    skip, don't guess" rule."""
    matches: list[tuple[str, ...]] = []
    for order in sorted(permutations(clusters)):

        def _consume(pos: int, i: int, order: tuple[str, ...] = order) -> bool:
            if i == len(order):
                return pos == len(surface)
            for sf in sorted(cluster_surfaces.get(order[i], ())):
                if sf and surface.startswith(sf, pos) and _consume(pos + len(sf), i + 1):
                    return True
            return False

        if _consume(0, 0):
            matches.append(order)
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class Passthrough:
    """A mined passthrough: a composite morpheme composed of ordered constituents.

    Identity is the cognate-cluster shape (``composite_cluster`` +
    ``constituent_clusters`` ordered by the surface tiling) so it pools across
    toponyms. ``*_etymon`` are the representative observed etymon ids the
    ``composed-of`` assertions reference — the deterministic lexicographically
    smallest observation, since relational edges point at observations (D50.3)
    and the L3 projection rolls them up to canonical morphemes via binds.
    ``support`` = distinct toponyms attesting it (the credibility weight)."""

    composite_cluster: str
    composite_etymon: int
    constituent_clusters: tuple[str, ...]
    constituent_etymons: tuple[int, ...]
    composite_form: str
    constituent_forms: tuple[str, ...]
    support: int
    sample_toponyms: tuple[str, ...]


def _load_breakdowns(
    db: LexiconDB, index: ClusterIndex
) -> tuple[dict[int, str], dict[int, list[list[tuple[str, str, int]]]]]:
    """``(toponym_id → modern_name, toponym_id → [per-breakdown [(cluster, folded_form, etymon_id)]])``."""
    name: dict[int, str] = {}
    by_breakdown: dict[int, list[tuple[str, str, int]]] = defaultdict(list)
    breakdown_topo: dict[int, int] = {}
    for row in db.conn.execute(
        """
        SELECT te.toponym_id AS tid, te.id AS teid, t.modern_name AS nm,
               tee.etymon_id AS eid, e.canonical_form AS cf
        FROM toponym_etymology te
        JOIN toponym t ON t.id = te.toponym_id
        JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id
        JOIN etymon e ON e.id = tee.etymon_id
        """
    ):
        name[row["tid"]] = row["nm"]
        breakdown_topo[row["teid"]] = row["tid"]
        by_breakdown[row["teid"]].append(
            (index.cluster_of[row["eid"]], fold_surface(row["cf"]), row["eid"])
        )

    topo_breakdowns: dict[int, list[list[tuple[str, str, int]]]] = defaultdict(list)
    for teid, elems in by_breakdown.items():
        topo_breakdowns[breakdown_topo[teid]].append(elems)
    return name, topo_breakdowns


def _detect_coarse_fine_pair(
    fine: list[tuple[str, str, int]],
    fine_set: set[str],
    coarse: list[tuple[str, str, int]],
    coarse_set: set[str],
    cluster_surfaces: dict[str, set[str]],
) -> _Candidate | None:
    """One ordered (fine, coarse) breakdown pair → a passthrough candidate
    ``(key, observation, forms)`` or ``None``. The coarse breakdown must have
    exactly one cluster the fine lacks (the composite) and the fine ≥2 the coarse
    lacks (the constituents), and the composite's folded surface must word-break
    into one reflex per constituent (which yields their order). The smallest
    etymon per constituent cluster is the representative observation."""
    fine_only = fine_set - coarse_set
    coarse_only = coarse_set - fine_set
    if len(coarse_only) != 1 or not (2 <= len(fine_only) <= _MAX_CONSTITUENTS):
        return None
    composite_cluster = next(iter(coarse_only))
    comp = sorted((f, eid) for cl, f, eid in coarse if cl == composite_cluster and f)
    if not comp:
        return None
    fine_by_cluster: dict[str, tuple[int, str]] = {}
    for cl, f, eid in fine:
        if cl in fine_only and (cl not in fine_by_cluster or eid < fine_by_cluster[cl][0]):
            fine_by_cluster[cl] = (eid, f)
    for comp_form, comp_etymon in comp:
        order = _segment_order(comp_form, list(fine_only), cluster_surfaces)
        if order is not None:
            key = (composite_cluster, order)
            obs = (comp_etymon, tuple(fine_by_cluster[cl][0] for cl in order))
            forms = (comp_form, tuple(fine_by_cluster[cl][1] for cl in order))
            return key, obs, forms
    return None


def _record_pair(
    tid: int,
    cand: _Candidate,
    toponyms: dict[_Key, set[int]],
    best_obs: dict[_Key, _Obs],
    forms: dict[_Key, _Forms],
) -> None:
    """Accumulate one detected coarse/fine pair into the per-key maps: add ``tid``
    to the key's support, and keep the lexicographically smallest observation
    (and its forms) as the representative. Hoisted out of ``_aggregate_pairs``'s
    inner double loop to keep that accumulation block off the depth-4 nesting."""
    key, obs, fms = cand
    toponyms[key].add(tid)
    if key not in best_obs or obs < best_obs[key]:
        best_obs[key] = obs
        forms[key] = fms


def _aggregate_pairs(
    topo_breakdowns: dict[int, list[list[tuple[str, str, int]]]],
    cluster_surfaces: dict[str, set[str]],
) -> tuple[dict[_Key, set[int]], dict[_Key, _Obs], dict[_Key, _Forms]]:
    """Scan every toponym's breakdown pairs, accumulating per passthrough key:
    the attesting toponym ids (support), the representative (smallest)
    observation, and its forms."""
    toponyms: dict[_Key, set[int]] = defaultdict(set)
    best_obs: dict[_Key, _Obs] = {}
    forms: dict[_Key, _Forms] = {}
    for tid, breakdowns in topo_breakdowns.items():
        if len(breakdowns) < 2:
            continue
        cluster_sets = [{cl for cl, _, _ in b} for b in breakdowns]
        for i, fine in enumerate(breakdowns):
            for j, coarse in enumerate(breakdowns):
                if i == j:
                    continue
                cand = _detect_coarse_fine_pair(
                    fine, cluster_sets[i], coarse, cluster_sets[j], cluster_surfaces
                )
                if cand is not None:
                    _record_pair(tid, cand, toponyms, best_obs, forms)
    return toponyms, best_obs, forms


def extract_cross_scholar_passthroughs(
    db: LexiconDB,
    *,
    index: ClusterIndex | None = None,
    min_support: int = 1,
) -> list[Passthrough]:
    """Stream 1: mine passthroughs from cross-scholar coarse-vs-fine pairs.

    For every toponym with ≥2 breakdowns, compare each ordered pair (fine A,
    coarse B): when B has exactly one cluster A lacks (the composite) and A has
    ≥2 clusters B lacks (the constituents) with the rest shared, and the
    composite's folded surface word-breaks into one reflex per constituent
    cluster, record ``composite → ordered constituents``. Aggregated by
    (composite_cluster, surface-ordered constituent_clusters); ``support`` counts
    the distinct toponyms attesting it; the representative etymons are the
    lexicographically smallest observation (deterministic). Returns records with
    support ≥ ``min_support``, sorted by support desc.
    """
    if index is None:
        index = load_cluster_index(db)
    cluster_surfaces = _cluster_surfaces(db, index)
    name, topo_breakdowns = _load_breakdowns(db, index)
    toponyms, best_obs, forms = _aggregate_pairs(topo_breakdowns, cluster_surfaces)

    out: list[Passthrough] = []
    for key, tids in toponyms.items():
        if len(tids) < min_support:
            continue
        composite_cluster, constituent_clusters = key
        comp_etymon, constituent_etymons = best_obs[key]
        comp_form, constituent_forms = forms[key]
        out.append(
            Passthrough(
                composite_cluster=composite_cluster,
                composite_etymon=comp_etymon,
                constituent_clusters=constituent_clusters,
                constituent_etymons=constituent_etymons,
                composite_form=comp_form,
                constituent_forms=constituent_forms,
                support=len(tids),
                sample_toponyms=tuple(sorted(name[t] for t in tids)[:5]),
            )
        )
    # Sort fully by content (cluster keys included) — folded forms can tie across
    # distinct clusters (homograph surfaces), and the fallback would be dict-
    # insertion = the ORDER-BY-less SQL row order, which is not rebuild-stable.
    # The cluster keys make the order content-determined (D36.9 / seed repro).
    out.sort(
        key=lambda p: (
            -p.support,
            p.composite_form,
            p.constituent_forms,
            p.composite_cluster,
            p.constituent_clusters,
        )
    )
    return out


def build_passthrough_map(passthroughs: list[Passthrough]) -> dict[str, tuple[str, ...]]:
    """The grader's ``passthrough_map`` (composite cognate-cluster → ordered
    constituent clusters) from mined passthroughs — the bridge that lets the grader
    credit a matched composite (``ington``) with its constituents (``ing``+``tūn``)
    for recall/precision/head (wyrd-h5u1, D51.3).

    A composite cluster can attest >1 constituent tiling across toponym pairs (the
    miner keys on ``(composite_cluster, constituent_clusters)``). The HIGHEST-support
    tiling wins — the most-corroborated attribution — ties broken by the miner's
    deterministic input order (so the result is rebuild-stable, D36.9)."""
    out: dict[str, tuple[str, ...]] = {}
    best_support: dict[str, int] = {}
    for p in passthroughs:
        if p.composite_cluster not in best_support or p.support > best_support[p.composite_cluster]:
            best_support[p.composite_cluster] = p.support
            out[p.composite_cluster] = p.constituent_clusters
    return out


def _confidence_for(support: int) -> str:
    """Cross-scholar coarse/fine is direct, gloss-bearing evidence, so a single
    attestation is ``medium``; ≥2 distinct toponyms agreeing is ``high``."""
    return "high" if support >= 2 else "medium"


def passthrough_assertions(
    passthroughs: list[Passthrough],
    *,
    source: str,
    actor: str = "",
) -> list[Assertion]:
    """Convert mined passthroughs into validated D50 ``composed-of`` assertions —
    one per (composite, constituent, ordinal). Self-referential edges (a
    representative composite etymon equal to a constituent etymon) are skipped
    (the D50 no-self-loop guard). Ids are deterministic (no timestamp), so
    re-mining the same corpus is idempotent (D36.9)."""
    out: list[Assertion] = []
    for p in passthroughs:
        confidence = _confidence_for(p.support)
        rationale = (
            f"cross-scholar coarse-vs-fine: {p.composite_form} = "
            f"{'+'.join(p.constituent_forms)} "
            f"(support {p.support}; e.g. {', '.join(p.sample_toponyms)})"
        )
        for ordinal, constituent_etymon in enumerate(p.constituent_etymons):
            if constituent_etymon == p.composite_etymon:
                continue
            assertion = Assertion(
                predicate="composed-of",
                subject=NodeRef("etymon", str(p.composite_etymon)),
                object=NodeRef("etymon", str(constituent_etymon)),
                qualifiers={"ordinal": ordinal},
                confidence=confidence,
                method=METHOD,
                source=source,
                actor=actor,
                rationale=rationale,
            ).with_id()
            validate(assertion)
            out.append(assertion)
    return out
