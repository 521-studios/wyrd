"""Genitive-``s`` split priors extraction (wyrd-aicu.9).

The decomposition matcher mis-parses the genitive ``s`` of ``X's-tūn`` (a
*town*) as ``X + ston`` (a *stone*): the greedy ``(unaccounted, morphemes)``
score prefers the longer ``ston`` (0 unaccounted) over ``X + s + ton`` (the
genitive ``s`` left unaccounted), so Occam picks stone. ``Bishopston`` =
Bishop's-tūn (a TOWN) decomposes as ``bishop + ston`` (STONE), inflating the
``stone`` morpheme's weight and starving ``ton`` in generation.

The fix is a **homograph-aware, per-suffix probabilistic split** seeded by the
scholarly breakdowns — NOT a hardcoded ``if ston`` rule (which ``-ster``/minster
and the ~6% genuine ``-stone`` cases would break). This module produces the
artifact that seeds it: per ``(long_form, short_form)`` genitive-overlap pair
(``ston`` / ``ton``, ``sley`` / ``ley`` …), the raw count of scholarly toponym
breakdowns whose final element is the **genitive-split** reading (the short
suffix ``S``, e.g. tūn → town) vs the **literal** long form (``L``, e.g.
stān → stone).

Design (validated against the live corpus, wyrd-aicu.9 investigations 1-3):

* **Candidate pairs are auto-discovered** from the reflex table: every surface
  ``L = 's' + S`` where both ``L`` and ``S`` are real reflexes. Most are
  coincidental noise (``sage`` ⊃ ``age``); only pairs with real scholarly split
  evidence become "active". No hardcoded suffix list — the set grows as the
  morpheme inventory grows.

* **Classification is by COGNATE CLUSTER, not surface membership.** The
  scholarly etymon (``stān``) and the surface reflex (``ston`` →
  ``middle-english:ston``) are different rows; surface membership is lossy and
  drops the genuine-stone cases. Cluster 358859 unifies stān/ston/stone/stan, so
  classifying a breakdown's final element by the cognate cluster of ``L``'s vs
  ``S``'s reflex-etymon recovers both classes. Fragmented town clusters leave a
  ``tun``/``ton`` straggler gap that shrinks as ``cluster-cognates`` coverage
  grows ("improves as the data improves"). A toponym whose pooled breakdown
  touches BOTH classes, or NEITHER, is skipped (and counted), not guessed.

* **Pooling across regions is sound** — the toponymic HEAD is stable across the
  places a name appears (the specifier varies, but that is not what this prior
  models). So every place's breakdown counts.

Per-toponym attestation disambiguation (the ``-es-``/``-s-`` historical genitive
marker) is **deferred to wyrd-aicu.9.1**: the whole-name ``toponym_attestation``
forms carry the bare modern surface (``ston`` = ``s``+``ton`` regardless of
whether the ``s`` is genitive or stem-initial), so a string marker cannot call
the head for the ``-ston`` family — it mis-split the protected genuine-stone set
(rudston→town). That needs per-element historical-form decomposition, scoped to
its own ticket; the cognate-cluster classifier is the reliable signal here.

LLM-free, deterministic, idempotent (mirrors ``empirical_priors``). **Raw counts
are persisted; smoothing + hierarchical backoff happen at LOOKUP time**
(:func:`split_probability`), so the table stays a faithful data mirror and the
prior sharpens automatically as breakdowns accrue.
"""

from __future__ import annotations

import json
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from wyrd.generators.kenning.lexicon import LexiconDB

# Reflex positions that carry a toponymic suffix (a genitive-overlap long form
# only matters word-finally / internally; pre-position heads are specifiers).
_SUFFIX_POSITIONS = ("post", "inner")


# Ligatures NFKD leaves intact (so ASCII-encode would drop them) — expand
# before normalizing so 'Ælfrīc' folds to 'aelfric', not 'lfric'.
_LIGATURES = str.maketrans(
    {"æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ð": "d", "Ð": "d", "þ": "th", "Þ": "th"}
)


def _fold(surface: str) -> str:
    """Bare-surface fold: expand ligatures, deaccent (NFKD → ASCII), lowercase,
    drop non-letters.

    D45 fold (``replace``-style — dashes are never identity). Sanctioned to also
    strip other non-letter noise so surface forms normalize consistently.
    """
    expanded = surface.translate(_LIGATURES)
    ascii_form = unicodedata.normalize("NFKD", expanded).encode("ascii", "ignore").decode()
    return "".join(ch for ch in ascii_form.lower() if ch.isalpha())


# ---------------------------------------------------------------------------
# Classifier data — built once from the DB, reused per toponym.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassifierData:
    """Cluster + reflex indexes used to classify a breakdown's elements.

    ``cluster_of``      : etymon_id → cluster key (``c<cognate_id>`` or the
                          singleton ``e<id>`` when unclustered).
    ``surface_etymons`` : folded reflex surface → set of etymon_ids it resolves
                          to (suffix positions only).
    """

    cluster_of: dict[int, str]
    surface_etymons: dict[str, set[int]]

    def clusters_for_surface(self, surface: str) -> set[str]:
        return {self.cluster_of[e] for e in self.surface_etymons.get(surface, set())}


def load_classifier_data(db: LexiconDB) -> ClassifierData:
    """Build the cluster + reflex indexes from the lexicon DB."""
    cluster_of: dict[int, str] = {}
    for row in db.conn.execute("SELECT id, cognate_id FROM etymon"):
        cid = row["cognate_id"]
        cluster_of[row["id"]] = f"c{cid}" if cid is not None else f"e{row['id']}"

    surface_etymons: dict[str, set[int]] = defaultdict(set)
    placeholders = ",".join("?" for _ in _SUFFIX_POSITIONS)
    for row in db.conn.execute(
        "SELECT re.etymon_id AS eid, r.surface_form AS sf "
        "FROM reflex r JOIN reflex_etymon re ON re.reflex_id = r.id "
        f"WHERE r.position IN ({placeholders})",
        _SUFFIX_POSITIONS,
    ):
        folded = _fold(row["sf"])
        if not folded:
            continue
        surface_etymons[folded].add(row["eid"])

    return ClassifierData(cluster_of=cluster_of, surface_etymons=dict(surface_etymons))


def discover_candidate_pairs(surfaces: set[str]) -> dict[str, str]:
    """Auto-discover genitive-overlap pairs ``L = 's' + S`` where both ``L`` and
    ``S`` are real reflex surfaces. Returns ``{long_form: short_form}``.

    This is intentionally permissive — most pairs are coincidental noise
    (``sage`` ⊃ ``age``) and only acquire a non-zero prior when real scholarly
    split evidence exists. Filtering to "active" pairs is a lookup-time concern.
    """
    return {
        long: long[1:]
        for long in surfaces
        if long.startswith("s") and len(long) >= 3 and long[1:] in surfaces
    }


# ---------------------------------------------------------------------------
# Extraction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairCounts:
    """Raw per-pair tallies: scholarly toponym breakdowns that are the
    genitive-split reading (short suffix ``S``) vs the literal long form ``L``."""

    split: int = 0
    literal: int = 0


@dataclass(frozen=True)
class GenitiveExtractionResult:
    """Per-run summary for the CLI + tests."""

    toponyms_scanned: int
    candidate_pairs: int
    active_pairs: int
    classified_split: int
    classified_literal: int
    skipped_both: int
    skipped_unclassified: int

    def as_dict(self) -> dict[str, int]:
        return {
            "toponyms_scanned": self.toponyms_scanned,
            "candidate_pairs": self.candidate_pairs,
            "active_pairs": self.active_pairs,
            "classified_split": self.classified_split,
            "classified_literal": self.classified_literal,
            "skipped_both": self.skipped_both,
            "skipped_unclassified": self.skipped_unclassified,
        }


def _pair_clusters(
    data: ClassifierData, candidates: dict[str, str]
) -> dict[tuple[str, str], tuple[set[str], set[str]]]:
    """Per pair, the ``(split_clusters, literal_clusters)`` cognate-cluster sets.
    The literal clusters are ``L``'s reflex-etymon clusters; the split clusters
    are ``S``'s, minus any that overlap the literal side (disambiguate toward
    literal so an etymon serving both surfaces never double-counts)."""
    out: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
    for long, short in candidates.items():
        literal = data.clusters_for_surface(long)
        split = data.clusters_for_surface(short) - literal
        out[(long, short)] = (split, literal)
    return out


def _load_toponyms(
    db: LexiconDB, data: ClassifierData
) -> tuple[dict[int, str], dict[int, set[str]]]:
    """``(toponym_id → modern_name, toponym_id → pooled element clusters)`` for
    every toponym that has a scholarly breakdown."""
    topo_name: dict[int, str] = {}
    topo_clusters: dict[int, set[str]] = defaultdict(set)
    for row in db.conn.execute(
        """
        SELECT te.toponym_id AS tid, t.modern_name AS name, tee.etymon_id AS eid
        FROM toponym_etymology te
        JOIN toponym t ON t.id = te.toponym_id
        JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id
    """
    ):
        topo_name[row["tid"]] = row["name"]
        topo_clusters[row["tid"]].add(data.cluster_of[row["eid"]])
    return topo_name, topo_clusters


def _classify(eclusters: set[str], split_clusters: set[str], literal_clusters: set[str]) -> str:
    """Verdict for one (toponym, pair): ``split`` / ``literal`` when the pooled
    breakdown clusters touch exactly one class; ``both`` / ``unclassified`` (both
    or neither) are skipped, never guessed."""
    has_split = bool(eclusters & split_clusters)
    has_literal = bool(eclusters & literal_clusters)
    if has_split and has_literal:
        return "both"
    if has_split:
        return "split"
    if has_literal:
        return "literal"
    return "unclassified"


def _tally(
    topo_name: dict[int, str],
    topo_clusters: dict[int, set[str]],
    pair_clusters: dict[tuple[str, str], tuple[set[str], set[str]]],
    progress_every: int,
) -> tuple[dict[tuple[str, str], dict[str, int]], dict[str, int]]:
    """Scan toponyms × pairs, tally per-pair split/literal counts + run stats."""
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"split": 0, "literal": 0})
    stats = dict.fromkeys(("scanned", "split", "literal", "both", "unclassified"), 0)
    total = len(topo_name)
    started = time.monotonic()

    def _progress() -> None:
        n = stats["scanned"]
        rate = (time.monotonic() - started) / n if n else 0.0
        print(
            f"  [{n}/{total}]  split={stats['split']} literal={stats['literal']} "
            f"({rate:.4f}s/entry)",
            file=sys.stderr,
            flush=True,
        )

    for tid, name in topo_name.items():
        stats["scanned"] += 1
        if progress_every and stats["scanned"] % progress_every == 0:
            _progress()
        folded_name = _fold(name)
        eclusters = topo_clusters[tid]
        for pair, (split_clusters, literal_clusters) in pair_clusters.items():
            if not folded_name.endswith(pair[0]):
                continue
            verdict = _classify(eclusters, split_clusters, literal_clusters)
            if verdict in ("split", "literal"):
                counts[pair][verdict] += 1
            stats[verdict] += 1

    if progress_every and stats["scanned"] % progress_every != 0:
        _progress()
    return counts, stats


def extract_genitive_priors(
    db: LexiconDB,
    *,
    progress_every: int = 0,
) -> tuple[dict[tuple[str, str], PairCounts], GenitiveExtractionResult]:
    """Walk every toponym that has a scholarly breakdown and tally split vs
    literal per genitive-overlap pair.

    Returns ``(counts, summary)`` where ``counts`` is keyed by
    ``(long_form, short_form)`` and sparse (only pairs that classified at least
    one toponym appear). One verdict per toponym from its pooled breakdown
    elements' cognate clusters; every place a name appears counts (pooling-
    across-regions is sound — wyrd-aicu.9 investigation 3). A toponym whose pool
    touches both classes, or neither, is skipped (and counted), never guessed.
    """
    data = load_classifier_data(db)
    candidates = discover_candidate_pairs(set(data.surface_etymons))
    pair_clusters = _pair_clusters(data, candidates)
    topo_name, topo_clusters = _load_toponyms(db, data)
    counts, stats = _tally(topo_name, topo_clusters, pair_clusters, progress_every)

    result_counts = {
        pair: PairCounts(split=c["split"], literal=c["literal"]) for pair, c in counts.items()
    }
    summary = GenitiveExtractionResult(
        toponyms_scanned=stats["scanned"],
        candidate_pairs=len(candidates),
        active_pairs=len(result_counts),
        classified_split=stats["split"],
        classified_literal=stats["literal"],
        skipped_both=stats["both"],
        skipped_unclassified=stats["unclassified"],
    )
    return result_counts, summary


def _replace_priors(db: LexiconDB, counts: dict[tuple[str, str], PairCounts]) -> None:
    """Clear and repopulate ``genitive_split_prior`` atomically (replace-not-
    merge — a stale pair from a removed observation must not survive)."""
    with db.conn:
        db.conn.execute("DELETE FROM genitive_split_prior")
        db.conn.executemany(
            "INSERT INTO genitive_split_prior "
            "(long_form, short_form, split_count, literal_count) VALUES (?, ?, ?, ?)",
            ((long, short, c.split, c.literal) for (long, short), c in counts.items()),
        )


def mine_genitive_priors(
    db: LexiconDB,
    *,
    apply: bool = False,
    progress_every: int = 0,
) -> dict[str, int]:
    """Extract genitive-split priors into ``genitive_split_prior``.

    ``apply=False`` is a dry-run returning the would-write summary. ``apply=True``
    atomically replaces the table. Idempotent on unchanged DB state.
    """
    counts, summary = extract_genitive_priors(db, progress_every=progress_every)
    if apply:
        _replace_priors(db, counts)
    return summary.as_dict()


# ---------------------------------------------------------------------------
# JSON sidecar + lookup-time smoothing.
# ---------------------------------------------------------------------------


def _pairs_payload(db: LexiconDB, *, version: str) -> dict:
    rows = db.conn.execute(
        "SELECT long_form, short_form, split_count, literal_count "
        "FROM genitive_split_prior "
        "ORDER BY split_count + literal_count DESC, long_form, short_form"
    )
    pairs = [
        {
            "long_form": r["long_form"],
            "short_form": r["short_form"],
            "split_count": r["split_count"],
            "literal_count": r["literal_count"],
        }
        for r in rows
    ]
    return {"version": version, "pairs": pairs}


def dump_genitive_priors_to_json(
    db: LexiconDB,
    output_path: Path,
    *,
    version: str = "unversioned",
) -> dict[str, int]:
    """Emit a sorted, byte-stable JSON sidecar of the raw counts."""
    artifact = _pairs_payload(db, version=version)
    artifact["pairs"].sort(
        key=lambda p: (-(p["split_count"] + p["literal_count"]), p["long_form"], p["short_form"])
    )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
        f.write("\n")
    return {"pairs": len(artifact["pairs"])}


def load_genitive_priors_from_json(input_path: Path) -> dict[tuple[str, str], PairCounts]:
    with input_path.open(encoding="utf-8") as f:
        artifact = json.load(f)
    return load_genitive_priors_from_payload(artifact)


def load_genitive_priors_from_payload(
    artifact: dict,
) -> dict[tuple[str, str], PairCounts]:
    out: dict[tuple[str, str], PairCounts] = {}
    for p in artifact.get("pairs", []):
        out[(p["long_form"], p["short_form"])] = PairCounts(
            split=int(p.get("split_count", 0)),
            literal=int(p.get("literal_count", 0)),
        )
    return out


def global_split_rate(counts: dict[tuple[str, str], PairCounts]) -> float:
    """Corpus-wide genitive-split fraction across all pairs — the backoff prior
    a thin pair smooths toward. 0.5 when there is no evidence at all."""
    split = sum(c.split for c in counts.values())
    total = sum(c.split + c.literal for c in counts.values())
    return split / total if total else 0.5


def split_probability(
    counts: dict[tuple[str, str], PairCounts],
    long_form: str,
    short_form: str,
    *,
    pseudocount: float = 5.0,
    backoff: float | None = None,
) -> float:
    """P(genitive split → ``short_form``) for a name ending in ``long_form``.

    Hierarchical empirical-Bayes smoothing: the per-pair ratio is pulled toward
    the corpus-wide ``backoff`` rate by ``pseudocount`` pseudo-observations, so a
    thin pair leans on the global rate and a well-attested one (``ston``,
    N≈390) trusts its own ratio. As breakdowns accrue the estimate converges to
    the empirical ratio — the "improves as the data improves" curve. Smoothing
    lives HERE, at lookup, not in the persisted counts.
    """
    if backoff is None:
        backoff = global_split_rate(counts)
    c = counts.get((long_form, short_form), PairCounts())
    n = c.split + c.literal
    return (c.split + pseudocount * backoff) / (n + pseudocount)
