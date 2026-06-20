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

* **Attestation refinement (wyrd-aicu.9.1) is SUBORDINATE.** The ``-es-``/``-s-``
  historical genitive marker (Kingston ← *cyninge·s·tun*) only resolves toponyms
  the cluster left ``both`` / ``unclassified``; it NEVER overrides a decisive
  cluster verdict. And it fires only on GENUINELY historical forms (folded form
  ≠ folded modern name) — ``toponym_attestation`` is ~97% modern-name echo, and
  the bare modern ``ston`` = ``s``+``ton`` carries no genitive signal (running
  the marker on it is what mis-split the protected genuine-stone set in the first
  cut). On real historical spellings the marker is reliable and never reaches the
  decisive cluster cases, so the protected stone set is safe by construction.

LLM-free, deterministic, idempotent (mirrors ``empirical_priors``). **Raw counts
are persisted; smoothing + hierarchical backoff happen at LOOKUP time**
(:func:`split_probability`), so the table stays a faithful data mirror and the
prior sharpens automatically as breakdowns accrue.
"""

from __future__ import annotations

import json
import re
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

# Minimum stem length for an attestation genitive-marker anchor. Below this the
# ``es`` + stem pattern over-fires on short coincidental endings.
_MIN_STEM = 3


# Ligatures NFKD leaves intact (so ASCII-encode would drop them) — expand
# before normalizing so 'Ælfrīc' folds to 'aelfric', not 'lfric'.
_LIGATURES = str.maketrans(
    {"æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ð": "d", "Ð": "d", "þ": "th", "Þ": "th"}
)


def fold_surface(surface: str | None) -> str:
    """Bare-surface fold: expand ligatures, deaccent (NFKD → ASCII), lowercase,
    drop non-letters.

    D45 fold (``replace``-style — dashes are never identity). Sanctioned to also
    strip other non-letter noise so surface forms normalize consistently.

    Returns ``""`` for ``None`` / empty input so callers can fold a possibly-NULL
    DB column (e.g. ``canonical_form``) without a guard — and the result is always
    a ``str``, so downstream ``.endswith`` / ``"".join`` are safe.
    """
    if not surface:
        return ""
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
        folded = fold_surface(row["sf"])
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
    resolved_by_attestation: int
    skipped_both: int
    skipped_unclassified: int

    def as_dict(self) -> dict[str, int]:
        return {
            "toponyms_scanned": self.toponyms_scanned,
            "candidate_pairs": self.candidate_pairs,
            "active_pairs": self.active_pairs,
            "classified_split": self.classified_split,
            "classified_literal": self.classified_literal,
            "resolved_by_attestation": self.resolved_by_attestation,
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


# ---------------------------------------------------------------------------
# Attestation refinement (wyrd-aicu.9.1) — the historical genitive marker.
#
# SUBORDINATE to the cognate-cluster classifier: it ONLY resolves toponyms the
# cluster left ``both`` / ``unclassified``, never overrides a decisive verdict.
# And it fires ONLY on GENUINELY HISTORICAL forms (folded form != folded modern
# name): ``toponym_attestation`` is ~97% modern-name echo, and the bare modern
# surface ``ston`` = ``s``+``ton`` carries no genitive signal — running the
# marker on it is what mis-split the protected genuine-stone set in the first
# (reverted) cut. Restricted to real historical spellings (``cyninge·s·tun``,
# ``alvrice·s·tone``) the ``-es-`` marker is reliable.
# ---------------------------------------------------------------------------


def _cluster_reflexes(data: ClassifierData) -> dict[str, set[str]]:
    """cluster key → folded reflex surfaces of its etymons (the historical-stem
    anchors), inverted from ``surface_etymons`` so ``ClassifierData`` stays the
    same shape the cluster-only path uses."""
    out: dict[str, set[str]] = defaultdict(set)
    for surface, eids in data.surface_etymons.items():
        for eid in eids:
            out[data.cluster_of[eid]].add(surface)
    return out


def _genitive_ambiguous(stem: str, town_stems: set[str]) -> bool:
    """A literal stem like ``ston`` = ``s``+``ton`` collides with the genitive
    town reading, so it carries no stone evidence; ``stan`` / ``stane`` (bound
    stone stems) do not. Used to keep only unambiguous stone stems."""
    return any(
        len(stem) > len(t) and stem.endswith(t) and stem[-len(t) - 1] == "s" for t in town_stems
    )


def _compile_town_regex(town_stems: set[str]) -> re.Pattern[str] | None:
    """One genitive-marker regex for ALL town stems (``es`` / consonant-``s``
    before any town stem, anchored). Longer stems sort first so the alternation
    prefers the longest match. ``None`` when there are no town stems."""
    if not town_stems:
        return None
    alt = "|".join(re.escape(t) for t in sorted(town_stems, key=len, reverse=True))
    return re.compile(rf".+(?:es|[bcdfghjklmnpqrstvwxyz]s)(?:{alt})$")


def pair_genitive_stems(
    data: ClassifierData,
    pair_clusters: dict[tuple[str, str], tuple[set[str], set[str]]],
) -> dict[tuple[str, str], tuple[re.Pattern[str] | None, set[str]]]:
    """Per pair, ``(town_regex, stone_stems)`` for the historical-form marker —
    the town regex compiled ONCE here, not per toponym.

    Town stems = the short suffix ``S`` + its cognate-cluster reflexes (the
    historical spelling variants, e.g. ``ton`` / ``tone`` / ``tune``), folded
    into a single alternation regex. ``stone_stems`` = ``L``'s cluster reflexes
    restricted to the UNAMBIGUOUS bound forms (``stan`` / ``stane``) — the
    ``s``+town-stem collisions (``ston`` / ``stone``) are dropped so they don't
    fight the genitive reading.
    """
    crefs = _cluster_reflexes(data)
    out: dict[tuple[str, str], tuple[re.Pattern[str] | None, set[str]]] = {}
    for (long, short), (split_clusters, literal_clusters) in pair_clusters.items():
        town = {short} | {
            s for c in split_clusters for s in crefs.get(c, set()) if len(s) >= _MIN_STEM
        }
        stone = {
            s
            for c in literal_clusters
            for s in crefs.get(c, set())
            if len(s) >= _MIN_STEM and not _genitive_ambiguous(s, town)
        }
        out[(long, short)] = (_compile_town_regex(town), stone)
    return out


def detect_historical_genitive(
    forms: list[str],
    modern_name: str,
    town_regex: re.Pattern[str] | None,
    stone_stems: set[str],
) -> str | None:
    """``'split'`` / ``'literal'`` / ``None`` from a toponym's historical forms.

    Considers ONLY genuinely historical spellings (folded form differs from the
    folded modern name). Returns ``'split'`` when one shows the genitive
    connective (``es`` / consonant-``s``) before a town-family stem and none
    shows a bound stone stem; ``'literal'`` for the inverse; ``None`` when there
    is no historical form, no marker, or both appear (ambiguous → defer to the
    cluster classifier).

    Both arms are suffix-anchored, so a bound stone stem only counts when it ends
    the form (avoids matching a ``stan`` buried in a specifier like *Dunstan*).
    The residual ``-eston`` ambiguity (a genuine stone spelled ``X-eston`` reads
    as ``es``+``ton``) is why this stays SUBORDINATE: a decisive ``stān`` cluster
    is never overridden, so only an *unclassified* stone with no ``-stan`` form
    could be mis-split — a rare residue the cluster prior smooths over.
    """
    modern = fold_surface(modern_name)
    hist = {f for f in (fold_surface(x) for x in forms) if f and f != modern}
    if not hist:
        return None
    saw_town = town_regex is not None and any(town_regex.search(f) for f in hist)
    saw_stone = any(f.endswith(stem) for f in hist for stem in stone_stems)
    if saw_town and not saw_stone:
        return "split"
    if saw_stone and not saw_town:
        return "literal"
    return None


def _load_historical_forms(db: LexiconDB) -> dict[int, list[str]]:
    """toponym_id → its historical spellings, from ``toponym_attestation.form``
    + ``toponym_etymology.historical_form`` (the richer ``-es-`` source)."""
    forms: dict[int, list[str]] = defaultdict(list)
    for row in db.conn.execute("SELECT toponym_id, form FROM toponym_attestation"):
        forms[row["toponym_id"]].append(row["form"])
    for row in db.conn.execute(
        "SELECT toponym_id, historical_form FROM toponym_etymology "
        "WHERE historical_form IS NOT NULL"
    ):
        forms[row["toponym_id"]].append(row["historical_form"])
    return forms


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


def _resolve(
    eclusters: set[str],
    pair_clusters: tuple[set[str], set[str]],
    pair_stems: tuple[re.Pattern[str] | None, set[str]],
    forms: list[str],
    modern_name: str,
) -> tuple[str | None, bool, str | None]:
    """``(verdict, via_attestation, skip_reason)`` for one (toponym, pair).

    The cognate-cluster verdict is authoritative when decisive (``split`` /
    ``literal``). Only when the cluster is ``both`` / ``unclassified`` does the
    historical genitive marker get a say — and it stays silent unless a real
    historical form carries the signal. ``verdict is None`` ⇒ skipped."""
    split_clusters, literal_clusters = pair_clusters
    verdict = _classify(eclusters, split_clusters, literal_clusters)
    if verdict in ("split", "literal"):
        return verdict, False, None
    town_regex, stone_stems = pair_stems
    av = detect_historical_genitive(forms, modern_name, town_regex, stone_stems)
    if av is not None:
        return av, True, None
    return None, False, verdict  # "both" / "unclassified"


def _tally(
    topo_name: dict[int, str],
    topo_clusters: dict[int, set[str]],
    topo_forms: dict[int, list[str]],
    pair_clusters: dict[tuple[str, str], tuple[set[str], set[str]]],
    pair_stems: dict[tuple[str, str], tuple[re.Pattern[str] | None, set[str]]],
    progress_every: int,
) -> tuple[dict[tuple[str, str], dict[str, int]], dict[str, int]]:
    """Scan toponyms × pairs, tally per-pair split/literal counts + run stats."""
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"split": 0, "literal": 0})
    stats = dict.fromkeys(
        ("scanned", "split", "literal", "from_attestation", "both", "unclassified"), 0
    )
    total = len(topo_name)
    started = time.monotonic()

    def _progress() -> None:
        n = stats["scanned"]
        rate = (time.monotonic() - started) / n if n else 0.0
        print(
            f"  [{n}/{total}]  split={stats['split']} literal={stats['literal']} "
            f"attest={stats['from_attestation']} ({rate:.4f}s/entry)",
            file=sys.stderr,
            flush=True,
        )

    for tid, name in topo_name.items():
        stats["scanned"] += 1
        if progress_every and stats["scanned"] % progress_every == 0:
            _progress()
        folded_name = fold_surface(name)
        eclusters = topo_clusters[tid]
        for pair in pair_clusters:
            if not folded_name.endswith(pair[0]):
                continue
            verdict, via_attestation, skip = _resolve(
                eclusters, pair_clusters[pair], pair_stems[pair], topo_forms.get(tid, []), name
            )
            if verdict is None:
                stats[skip] += 1
                continue
            counts[pair][verdict] += 1
            stats[verdict] += 1
            if via_attestation:
                stats["from_attestation"] += 1

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
    pair_stems = pair_genitive_stems(data, pair_clusters)
    topo_name, topo_clusters = _load_toponyms(db, data)
    topo_forms = _load_historical_forms(db)
    counts, stats = _tally(
        topo_name, topo_clusters, topo_forms, pair_clusters, pair_stems, progress_every
    )

    result_counts = {
        pair: PairCounts(split=c["split"], literal=c["literal"]) for pair, c in counts.items()
    }
    summary = GenitiveExtractionResult(
        toponyms_scanned=stats["scanned"],
        candidate_pairs=len(candidates),
        active_pairs=len(result_counts),
        classified_split=stats["split"],
        classified_literal=stats["literal"],
        resolved_by_attestation=stats["from_attestation"],
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


def load_genitive_priors_from_json(  # noqa: V103 — matcher-facing loader (consumed when the prior is wired into the decomposition matcher); pinned by tests
    input_path: Path,
) -> dict[tuple[str, str], PairCounts]:
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


def split_probability(  # noqa: V103 — matcher-facing lookup (the smoothed prior the decomposition matcher will consume); pinned by tests
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


def load_genitive_prior_counts(db: LexiconDB) -> dict[tuple[str, str], PairCounts]:
    """Read the persisted ``genitive_split_prior`` table into the in-memory
    counts map :func:`split_probability` consumes.

    Empty when the table is missing or unpopulated — a DB predating migration
    0018, or one where ``mine-genitive-priors --apply`` hasn't run. Callers
    (the matcher prob-map build, the decomposition grader) treat an empty map
    as "no genitive prior": the connective still wins the non-homograph
    coverage cases on score, only the homograph tiebreak goes quiet.
    """
    table = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='genitive_split_prior'"
    ).fetchone()
    if not table:
        return {}
    out: dict[tuple[str, str], PairCounts] = {}
    for r in db.conn.execute(
        "SELECT long_form, short_form, split_count, literal_count FROM genitive_split_prior"
    ):
        out[(r["long_form"], r["short_form"])] = PairCounts(
            split=r["split_count"], literal=r["literal_count"]
        )
    return out


def build_split_probability_map(
    counts: dict[tuple[str, str], PairCounts],
    *,
    pseudocount: float = 5.0,
) -> dict[tuple[str, str], float]:
    """Precompute the runtime ``{(long_form, short_form): P_split}`` prob-map the
    decomposition matcher consumes — :func:`split_probability` applied once per
    active pair against the shared global backoff.

    This is the smoothing-at-emit boundary (wyrd-aicu.9): smoothing + backoff run
    HERE so the runtime matcher stays DB-free, looking up a plain float per
    genitive decision (``_prefer_genitive_credible``). Shared by the
    decomposition grader (Phase 3) and the L4 bundle build (Phase 4) so both
    consume an identically-smoothed prior. Empty in → empty out.
    """
    if not counts:
        return {}
    backoff = global_split_rate(counts)
    return {
        pair: split_probability(counts, *pair, pseudocount=pseudocount, backoff=backoff)
        for pair in counts
    }
