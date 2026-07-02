"""Decomposition grader — grade the matcher against the scholarly corpus (wyrd-aicu.9).

The scholarly toponym breakdowns (``toponym_etymology_element``, ~17K toponyms)
are a **labeled test corpus for the decomposer**: each row says which morphemes
a place-name scholar split a town into. This module runs the runtime matcher
over every labeled toponym and scores agreement with the scholar's breakdown,
then diffs two matcher configurations head-to-head — the canonical use is
**connective OFF (today's matcher) vs connective ON (the wyrd-aicu.9 genitive
fix)** — so a decomposer change can be accepted on evidence: *net agreement up
AND the regression list empty or individually defensible*. The regression list
(toponyms the new config gets wrong that the old got right) is the **overfitting
tripwire** that keeps the matcher to GENERAL defensible rules.

Why the metrics look the way they do:

* **Comparison is by COGNATE CLUSTER, not raw surface.** The scholar writes
  ``tūn``; the matcher emits the reflex ``ton``. They are the same morpheme via
  cognate cluster ``c346202`` — the same stān/ston bridge :mod:`genitive_priors`
  uses. A matcher morpheme "recovers" a scholar element when the scholar
  element's cluster is among the clusters the matcher's surface resolves to.

* **Order-robust (SET, not sequence).** ``toponym_etymology_element.ordinal`` is
  NOT reliably surface order (wyrd-z3me: some breakdowns are gloss/head-first),
  so head-by-ordinal would be noisy. The load-bearing metrics are therefore
  order-free: per-cluster **recall** (did the matcher recover every morpheme the
  scholar found?) and **precision** (did it avoid inventing morphemes the
  scholar didn't — the spurious ``ston``/stone of the ``-ston`` bug?), plus
  **coverage** (a fully-accounted parse, 0 unaccounted chars — the count the
  generation proportions train on). ``head_attested`` is a deliberately weak
  proxy: the surface-final matcher morpheme resolves to *some* cluster the
  scholar used — it catches the ``-ston`` bug (OFF: head ``ston``→stone ∉
  {bishop, tūn}; ON: head ``ton``→town ∈) without asserting WHICH scholar
  element is the head, which z3me makes unsafe.

Production faithfulness: the grader's trie is built from the runtime
``meaning_db`` (the bundle) — the SAME inventory the per-culture generation
proportions train on (``rebuild-proportions`` → ``load_meanings`` →
``_decompose_corpus``), so the grade measures the bug where it actually bites.
The matcher is called directly with ``culture_languages`` + the connective
params, mirroring that path (Phase 1's ``Name.find_meaning`` threading is a
later step; the grader is the baseline/validation tool that precedes it).

DB-aware (reads the authoring corpus + the cluster bridge) and LLM-free /
deterministic.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.genitive_priors import fold_surface
from wyrd.generators.kenning.runtime.connective import (
    DEFAULT_CONNECTIVE_INVENTORY,
    ConnectiveInventory,
)
from wyrd.generators.kenning.runtime.trie_matcher import (
    MorphemeTrie,
    all_decompositions,
    canonical_decomposition,
    count_unaccounted,
    iter_morphemes,
)

# ---------------------------------------------------------------------------
# Cluster bridge + scholar corpus.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterIndex:
    """Surface ⇄ cognate-cluster bridge built once from the lexicon.

    ``cluster_of``        : etymon_id → cluster key (``c<cognate_id>`` or the
                            singleton ``e<id>`` when unclustered).
    ``surface_clusters``  : folded reflex surface → the cluster keys it resolves
                            to (ALL reflex positions — specifiers as well as
                            suffixes, unlike the suffix-only index the genitive
                            prior builds).
    """

    cluster_of: dict[int, str]
    surface_clusters: dict[str, frozenset[str]]

    def clusters_for(self, surface: str) -> frozenset[str]:
        """The cognate clusters a (raw) morpheme surface resolves to — empty for
        a surface the lexicon has no reflex for (e.g. the matcher's unaccounted
        fall-through, or a morpheme not in the corpus)."""
        return self.surface_clusters.get(fold_surface(surface), frozenset())


def load_cluster_index(db: LexiconDB) -> ClusterIndex:
    """Build the surface ⇄ cluster bridge from the lexicon's etymon + reflex
    tables (all positions)."""
    # wyrd-qdau: exclude merged losers (merged_into_id set) — a merge repoints
    # neither reflex_etymon nor toponym_etymology_element rows, so readers must
    # filter merged_into_id IS NULL (as collapse_merge.py / era_reflex.py /
    # variant_fold_detect.py all do). Without it, a merged loser (cognate_id
    # almost always NULL) yields a stale singleton e<id> key — its dead identity
    # — fracturing the surviving cluster and mis-keying downstream miners.
    cluster_of: dict[int, str] = {}
    for row in db.conn.execute("SELECT id, cognate_id FROM etymon WHERE merged_into_id IS NULL"):
        cid = row["cognate_id"]
        cluster_of[row["id"]] = f"c{cid}" if cid is not None else f"e{row['id']}"

    surface_etymons: dict[str, set[int]] = defaultdict(set)
    for row in db.conn.execute(
        "SELECT re.etymon_id AS eid, r.surface_form AS sf "
        "FROM reflex r JOIN reflex_etymon re ON re.reflex_id = r.id "
        "JOIN etymon e ON e.id = re.etymon_id "
        "WHERE e.merged_into_id IS NULL"
    ):
        folded = fold_surface(row["sf"])
        if folded:
            surface_etymons[folded].add(row["eid"])

    surface_clusters = {
        surface: frozenset(cluster_of[e] for e in eids) for surface, eids in surface_etymons.items()
    }
    return ClusterIndex(cluster_of=cluster_of, surface_clusters=surface_clusters)


@dataclass(frozen=True)
class ScholarToponym:
    """One labeled toponym: its modern name + the cognate clusters its
    scholarly breakdown(s) touch, pooled across every breakdown and region of
    that name (pooling is sound — the toponymic head is stable across the places
    a name appears, wyrd-aicu.9 investigation 3)."""

    name: str
    clusters: frozenset[str]


def load_scholar_corpus(
    db: LexiconDB,
    index: ClusterIndex,
    *,
    suffix: str | None = None,
) -> list[ScholarToponym]:
    """Every toponym with a scholarly breakdown, as ``ScholarToponym`` records
    sorted by name (deterministic).

    Pools all breakdown elements of a name (across multiple breakdowns and
    multiple regions sharing a ``modern_name``) into one cluster set. ``suffix``
    filters to names whose folded form ends with it (e.g. ``ston``) for fast
    focused probes. Toponyms whose breakdown yields no clusters are dropped (not
    gradable)."""
    pooled: dict[str, set[str]] = defaultdict(set)
    for row in db.conn.execute(
        """
        SELECT t.modern_name AS name, tee.etymon_id AS eid
        FROM toponym_etymology te
        JOIN toponym t ON t.id = te.toponym_id
        JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id
        -- wyrd-qdau: skip breakdown elements anchored on a merged loser; its
        -- eid is absent from cluster_of (built merged_into_id IS NULL above),
        -- and its dead identity must not pool into the corpus.
        JOIN etymon e ON e.id = tee.etymon_id
        WHERE e.merged_into_id IS NULL
        """
    ):
        pooled[row["name"]].add(index.cluster_of[row["eid"]])

    corpus: list[ScholarToponym] = []
    for name, clusters in pooled.items():
        if not clusters:
            continue
        if suffix and not fold_surface(name).endswith(suffix):
            continue
        corpus.append(ScholarToponym(name=name, clusters=frozenset(clusters)))
    corpus.sort(key=lambda sc: sc.name)
    return corpus


# ---------------------------------------------------------------------------
# Grading a single toponym.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToponymGrade:
    """The matcher's grade for one toponym under one configuration."""

    name: str
    morphemes: tuple[str, ...]  # content-morpheme surfaces the matcher emitted
    unaccounted: int
    recall: float  # scholar clusters recovered / scholar clusters
    precision: float  # non-spurious matcher morphemes / matcher morphemes
    exact: bool  # recall == 1 AND no spurious morpheme (full set agreement)
    coverage: bool  # 0 unaccounted chars (a clean parse)
    head_attested: bool  # surface-final morpheme resolves to a scholar cluster


@dataclass(frozen=True)
class MatcherConfig:
    """The matcher knobs the grader holds together across a configuration: the
    culture's allowed languages, the connective inventory, and the genitive
    prior. Bundled (wyrd-buye) so the three travel as one value through the grade
    entry points instead of three kw-only params threaded each.

    ``passthrough_map`` is deliberately NOT here — it's the grading-time
    composite-attribution toggle that ``grade_passthrough_diff`` VARIES (off vs
    on) while holding this matcher config identical, so it stays a separate
    argument. All three default to None (the no-op baseline matcher)."""

    culture_languages: frozenset[str] | None = None
    connective_inventory: ConnectiveInventory | None = None
    genitive_prior: dict[tuple[str, str], float] | None = None


def _decompose_name(
    name: str,
    trie: MorphemeTrie,
    *,
    config: MatcherConfig,
) -> tuple[list[str], int, str | None]:
    """Decompose a (possibly multi-word) toponym token-by-token, mirroring how
    ``Name`` splits on whitespace and decomposes each ``Word``. Returns
    ``(content_surfaces, total_unaccounted, surface_final_morpheme)``."""
    surfaces: list[str] = []
    unaccounted = 0
    head: str | None = None
    for token in name.split():
        if not token:
            continue
        decomp = canonical_decomposition(
            token,
            trie,
            culture_languages=config.culture_languages,
            connective_inventory=config.connective_inventory,
            genitive_prior=config.genitive_prior,
        )
        token_surfaces = [m.usage for m in iter_morphemes(decomp)]
        surfaces.extend(token_surfaces)
        unaccounted += count_unaccounted(decomp)
        if token_surfaces:
            head = token_surfaces[-1]
    return surfaces, unaccounted, head


def _attributed_clusters(
    surfaces: list[str],
    index: ClusterIndex,
    passthrough_map: dict[str, tuple[str, ...]] | None,
) -> list[frozenset[str]]:
    """Per matched morpheme, the cognate-cluster set it is ATTRIBUTED to —
    expanding a composite (a passthrough composite cluster) into one virtual
    morpheme per constituent cluster (D51.3 passthrough: surface ≠ attribution).
    ``passthrough_map`` (composite_cluster → ordered constituent clusters) None or
    empty → identity (one cluster set per surface), so the grade is bit-stable
    with the no-passthrough baseline."""
    out: list[frozenset[str]] = []
    for surface in surfaces:
        clusters = index.clusters_for(surface)
        # A surface usually resolves to one passthrough composite; if it maps to
        # several (homograph composites), expand the lexicographically-smallest
        # deterministically — a rare edge whose exact pick doesn't bias the grade.
        hit = sorted(c for c in clusters if c in passthrough_map) if passthrough_map else []
        if hit:
            out.extend(frozenset({c}) for c in passthrough_map[hit[0]])
        else:
            out.append(clusters)
    return out


def grade_toponym(
    scholar: ScholarToponym,
    trie: MorphemeTrie,
    index: ClusterIndex,
    *,
    config: MatcherConfig,
    passthrough_map: dict[str, tuple[str, ...]] | None = None,
) -> ToponymGrade:
    """Run the matcher on one labeled toponym and score it against the scholar's
    pooled cluster set. See the module docstring for the metric rationale.
    ``passthrough_map`` expands matched composites to their constituents for
    attribution (D51.3) — coverage is unaffected (the parse is unchanged); recall
    / precision / head are scored over the expanded virtual morphemes."""
    surfaces, unaccounted, _head = _decompose_name(scholar.name, trie, config=config)
    scholar_clusters = scholar.clusters
    morph_clusters = _attributed_clusters(surfaces, index, passthrough_map)
    recovered = {c for c in scholar_clusters if any(c in mc for mc in morph_clusters)}
    recall = len(recovered) / len(scholar_clusters)
    spurious = sum(1 for mc in morph_clusters if mc.isdisjoint(scholar_clusters))
    precision = (len(morph_clusters) - spurious) / len(morph_clusters) if morph_clusters else 0.0
    exact = bool(morph_clusters) and recall == 1.0 and spurious == 0
    head_clusters = morph_clusters[-1] if morph_clusters else frozenset()
    head_attested = bool(morph_clusters) and not head_clusters.isdisjoint(scholar_clusters)
    return ToponymGrade(
        name=scholar.name,
        morphemes=tuple(surfaces),
        unaccounted=unaccounted,
        recall=recall,
        precision=precision,
        exact=exact,
        coverage=unaccounted == 0,
        head_attested=head_attested,
    )


# ---------------------------------------------------------------------------
# Corpus grading + config diff.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GradeSummary:
    """Aggregate metrics for one configuration across the graded corpus."""

    graded: int
    mean_recall: float
    mean_precision: float
    exact_rate: float
    coverage_rate: float
    head_attested_rate: float


def summarize(grades: list[ToponymGrade]) -> GradeSummary:
    n = len(grades)
    if n == 0:
        return GradeSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return GradeSummary(
        graded=n,
        mean_recall=sum(g.recall for g in grades) / n,
        mean_precision=sum(g.precision for g in grades) / n,
        exact_rate=sum(g.exact for g in grades) / n,
        coverage_rate=sum(g.coverage for g in grades) / n,
        head_attested_rate=sum(g.head_attested for g in grades) / n,
    )


# Metric dimensions a config change can move, each higher-is-better. Used to
# classify a per-toponym off→on change as a regression (any dimension dropped)
# and/or an improvement (any dimension rose) — a trade-off lands on both lists.
_DIMENSIONS: tuple[tuple[str, Callable[[ToponymGrade], float]], ...] = (
    ("recall", lambda g: g.recall),
    ("precision", lambda g: g.precision),
    ("exact", lambda g: float(g.exact)),
    ("coverage", lambda g: float(g.coverage)),
    ("head", lambda g: float(g.head_attested)),
)


@dataclass(frozen=True)
class ConfigChange:
    """A toponym whose grade changed between configs, with the dimensions that
    worsened (a regression) or improved. ``dimensions`` lists the moved metrics
    in the direction this record represents."""

    name: str
    off_parse: tuple[str, ...]
    on_parse: tuple[str, ...]
    dimensions: tuple[str, ...]


@dataclass(frozen=True)
class CorpusDiff:
    """Off-vs-on grade across the corpus: per-config summaries plus the
    regression / improvement lists. The acceptance rule is *every summary metric
    up (or flat) AND ``regressions`` empty or individually defensible*."""

    off: GradeSummary
    on: GradeSummary
    regressions: tuple[ConfigChange, ...]
    improvements: tuple[ConfigChange, ...]


def _emit_progress(label: str, done: int, total: int, started: float) -> None:
    rate = (time.monotonic() - started) / done if done else 0.0
    print(f"  [{done}/{total}] {label} ({rate:.4f}s/entry)", file=sys.stderr, flush=True)


def grade_configuration(
    corpus: list[ScholarToponym],
    trie: MorphemeTrie,
    index: ClusterIndex,
    *,
    config: MatcherConfig,
    passthrough_map: dict[str, tuple[str, ...]] | None = None,
    label: str = "grade",
    progress_every: int = 0,
) -> list[ToponymGrade]:
    """Grade every toponym in ``corpus`` under one matcher configuration."""
    grades: list[ToponymGrade] = []
    total = len(corpus)
    started = time.monotonic()
    for i, scholar in enumerate(corpus, 1):
        grades.append(
            grade_toponym(
                scholar,
                trie,
                index,
                config=config,
                passthrough_map=passthrough_map,
            )
        )
        if progress_every and i % progress_every == 0:
            _emit_progress(label, i, total, started)
    if progress_every and total % progress_every != 0:
        _emit_progress(label, total, total, started)
    return grades


def _classify_change(
    off: ToponymGrade, on: ToponymGrade
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(worsened, improved)`` metric-dimension names for one toponym."""
    worsened = tuple(label for label, f in _DIMENSIONS if f(on) < f(off))
    improved = tuple(label for label, f in _DIMENSIONS if f(on) > f(off))
    return worsened, improved


def diff_grades(
    off_grades: list[ToponymGrade],
    on_grades: list[ToponymGrade],
) -> CorpusDiff:
    """Build the off-vs-on ``CorpusDiff`` from two same-corpus grade lists."""
    off_by_name = {g.name: g for g in off_grades}
    regressions: list[ConfigChange] = []
    improvements: list[ConfigChange] = []
    for on in on_grades:
        off = off_by_name[on.name]
        worsened, improved = _classify_change(off, on)
        if worsened:
            regressions.append(ConfigChange(on.name, off.morphemes, on.morphemes, worsened))
        if improved:
            improvements.append(ConfigChange(on.name, off.morphemes, on.morphemes, improved))
    return CorpusDiff(
        off=summarize(off_grades),
        on=summarize(on_grades),
        regressions=tuple(regressions),
        improvements=tuple(improvements),
    )


def grade_passthrough_diff(
    db: LexiconDB,
    trie: MorphemeTrie,
    *,
    passthrough_map: dict[str, tuple[str, ...]],
    config: MatcherConfig,
    suffix: str | None = None,
    limit: int | None = None,
    progress_every: int = 0,
) -> CorpusDiff:
    """OFF (no composed-of attribution) vs ON (a matched composite credited to its
    constituents via ``passthrough_map``), holding the matcher ``config`` IDENTICAL
    on both sides — so the diff isolates the wyrd-h5u1 attribution effect (D51.3),
    not the connective/prior. This is the graded net-win check that gates
    wyrd-oth3: coverage is unaffected (same parse), and recall/precision/head
    should recover where the matcher picks a coarse composite the scholar split
    finely."""
    index = load_cluster_index(db)
    corpus = load_scholar_corpus(db, index, suffix=suffix)
    if limit is not None:
        corpus = corpus[:limit]
    off = grade_configuration(
        corpus,
        trie,
        index,
        config=config,
        passthrough_map=None,
        label="off",
        progress_every=progress_every,
    )
    on = grade_configuration(
        corpus,
        trie,
        index,
        config=config,
        passthrough_map=passthrough_map,
        label="on",
        progress_every=progress_every,
    )
    return diff_grades(off, on)


def grade_corpus_diff(
    db: LexiconDB,
    trie: MorphemeTrie,
    *,
    config: MatcherConfig,
    suffix: str | None = None,
    limit: int | None = None,
    progress_every: int = 0,
) -> CorpusDiff:
    """End-to-end: load the bridge + scholar corpus, grade connective-OFF
    (today's matcher: ``culture_languages`` on, no connective, no prior) vs
    connective-ON (the full ``config``), and diff.

    ``config`` is the connective-ON configuration; the OFF side reuses its
    ``culture_languages`` but zeroes the connective inventory + genitive prior
    (``replace``), since ``culture_languages`` is part of today's matcher
    (wyrd-pfoo), not the change under test — so the diff isolates the connective's
    effect. ``suffix`` / ``limit`` focus the run for fast probes.

    A connective diff needs SOMETHING to turn on: if ``config`` carries no
    connective inventory the ON side defaults to ``DEFAULT_CONNECTIVE_INVENTORY``
    (the genitive ``-s-``) — matching the pre-wyrd-buye signature default, so an
    inventory-less call still diffs the connective rather than silently producing
    a no-op OFF==ON.
    """
    index = load_cluster_index(db)
    corpus = load_scholar_corpus(db, index, suffix=suffix)
    if limit is not None:
        corpus = corpus[:limit]
    on_config = (
        config
        if config.connective_inventory is not None
        else replace(config, connective_inventory=DEFAULT_CONNECTIVE_INVENTORY)
    )
    off_config = replace(on_config, connective_inventory=None, genitive_prior=None)
    off = grade_configuration(
        corpus, trie, index, config=off_config, label="off", progress_every=progress_every
    )
    on = grade_configuration(
        corpus, trie, index, config=on_config, label="on", progress_every=progress_every
    )
    return diff_grades(off, on)


# ---------------------------------------------------------------------------
# CAN-IT vs DID-WE (wyrd-eni4.3): the two-stage decomposition measurement.
#
# CAN-IT asks whether the decomposer GENERATES a parse that recovers the
# scholar's breakdown AT ALL (over the full candidate list, before selection) —
# the inventory ceiling the enrichment loop pushes up. DID-WE asks whether the
# single SELECTED parse recovers it — the selection question the human tackles
# later. GAP = generated-but-not-selected = recall lost purely to scoring/
# tiebreak. Measured both ways: ``exact`` (scholar breakdown == a candidate's
# cluster set) and ``recall1`` (some candidate recovers ALL scholar clusters,
# matching the grade's recall semantics, the load-bearing number).
#
# Per toponym: decompose each whitespace token the EXISTING way
# (``all_decompositions`` = the un-pruned candidate list the production walk
# would score), reduce each candidate to its cluster signature, and combine
# tokens by product (capped). No second decomposer — just visibility into the
# one we ship.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanItSummary:
    """Corpus-level CAN-IT / DID-WE / GAP counts, under both match criteria."""

    graded: int
    product_capped: int
    can_it_exact: int
    did_we_exact: int
    gap_exact: int
    can_it_recall1: int
    did_we_recall1: int
    gap_recall1: int


def _decomp_signature(decomp: list[Any], index: ClusterIndex) -> frozenset[str]:
    """One decomposition's cluster signature: the clusters across its morphemes."""
    clusters: set[str] = set()
    for m in iter_morphemes(decomp):
        clusters |= index.clusters_for(m.usage)
    return frozenset(clusters)


def _name_candidate_signatures(
    name: str, trie: MorphemeTrie, index: ClusterIndex, *, config: MatcherConfig, product_cap: int
) -> tuple[set[frozenset[str]], frozenset[str]]:
    """Per toponym: (set of full-name candidate cluster-signatures, selected
    signature). Tokens combine by product (each combo's clusters unioned), capped
    so a pathological multi-token name can't blow memory."""
    full: set[frozenset[str]] = {frozenset()}
    selected: set[str] = set()
    for token in name.split():
        if not token:
            continue
        cand_sigs = {_decomp_signature(d, index) for d in all_decompositions(token, trie)}
        selected |= _decomp_signature(
            canonical_decomposition(
                token,
                trie,
                culture_languages=config.culture_languages,
                connective_inventory=config.connective_inventory,
                genitive_prior=config.genitive_prior,
            ),
            index,
        )
        combined: set[frozenset[str]] = set()
        for a in full:
            for b in cand_sigs:
                combined.add(a | b)
                if len(combined) >= product_cap:
                    break
            if len(combined) >= product_cap:
                break
        full = combined
    return full, frozenset(selected)


def grade_can_it(
    corpus: list[ScholarToponym],
    trie: MorphemeTrie,
    index: ClusterIndex,
    *,
    config: MatcherConfig,
    product_cap: int = 20000,
    progress_every: int = 1000,
) -> CanItSummary:
    """Score the corpus for CAN-IT (scholar breakdown is among the generated
    candidates) vs DID-WE (it is the selected parse), under exact-equality and
    recall=1 criteria. See the section docstring."""
    n = capped = 0
    c_ex = d_ex = g_ex = c_r1 = d_r1 = g_r1 = 0
    total = len(corpus)
    started = time.monotonic()
    for sc in corpus:
        scholar = sc.clusters
        if not scholar:
            continue
        n += 1
        cands, sel_sig = _name_candidate_signatures(
            sc.name, trie, index, config=config, product_cap=product_cap
        )
        if len(cands) >= product_cap:
            capped += 1
        in_ex = scholar in cands
        sel_ex = scholar == sel_sig
        in_r1 = any(scholar <= sig for sig in cands)
        sel_r1 = scholar <= sel_sig
        c_ex += in_ex
        d_ex += sel_ex
        g_ex += in_ex and not sel_ex
        c_r1 += in_r1
        d_r1 += sel_r1
        g_r1 += in_r1 and not sel_r1
        if progress_every and n % progress_every == 0:
            _emit_progress("can-it", n, total, started)
    if progress_every and n % progress_every != 0:
        _emit_progress("can-it", n, total, started)
    return CanItSummary(
        graded=n,
        product_capped=capped,
        can_it_exact=c_ex,
        did_we_exact=d_ex,
        gap_exact=g_ex,
        can_it_recall1=c_r1,
        did_we_recall1=d_r1,
        gap_recall1=g_r1,
    )
