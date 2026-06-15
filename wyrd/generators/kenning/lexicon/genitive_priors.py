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
  grows ("improves as the data improves").

* **Attestation is the PRIMARY per-toponym signal** (even one wins). Historical
  forms often carry the genitive connective explicitly — Alfriston ←
  *Alvrice·s·tone* (1085) shows ``es`` + a town stem. When a dated form marks the
  genitive before a town-family stem we trust it over the breakdown machinery;
  the breakdown's cluster class is the fallback.

* **Pooling across regions is sound** — the toponymic HEAD is stable across the
  places a name appears (the specifier varies, but that is not what this prior
  models). So every place's breakdown counts.

LLM-free, deterministic, idempotent (mirrors ``empirical_priors``). **Raw counts
are persisted; smoothing + hierarchical backoff happen at LOOKUP time**
(:func:`split_probability`), so the table stays a faithful data mirror and the
prior sharpens automatically as breakdowns accrue.
"""

from __future__ import annotations

import json
import re
import sys
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


def _fold(surface: str) -> str:
    """Bare-surface fold: expand ligatures, deaccent (NFKD → ASCII), lowercase,
    drop non-letters.

    D45 fold (``replace``-style — dashes are never identity). Sanctioned to also
    strip other non-letter noise so historical attestation spellings normalize.
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

    ``cluster_of``       : etymon_id → cluster key (``c<cognate_id>`` or the
                           singleton ``e<id>`` when unclustered).
    ``surface_etymons``  : folded reflex surface → set of etymon_ids it resolves
                           to (suffix positions only).
    ``cluster_reflexes`` : cluster key → folded reflex surfaces of its etymons
                           (the historical-spelling anchors for the attestation
                           detector).
    """

    cluster_of: dict[int, str]
    surface_etymons: dict[str, set[int]]
    cluster_reflexes: dict[str, set[str]]

    def clusters_for_surface(self, surface: str) -> set[str]:
        return {self.cluster_of[e] for e in self.surface_etymons.get(surface, set())}

    def stems_for_clusters(self, clusters: set[str]) -> set[str]:
        stems: set[str] = set()
        for c in clusters:
            stems |= {s for s in self.cluster_reflexes.get(c, set()) if len(s) >= _MIN_STEM}
        return stems


def load_classifier_data(db: LexiconDB) -> ClassifierData:
    """Build the cluster + reflex indexes from the lexicon DB."""
    cluster_of: dict[int, str] = {}
    for row in db.conn.execute("SELECT id, cognate_id FROM etymon"):
        cid = row["cognate_id"]
        cluster_of[row["id"]] = f"c{cid}" if cid is not None else f"e{row['id']}"

    surface_etymons: dict[str, set[int]] = defaultdict(set)
    cluster_reflexes: dict[str, set[str]] = defaultdict(set)
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
        cluster_reflexes[cluster_of[row["eid"]]].add(folded)

    return ClassifierData(
        cluster_of=cluster_of,
        surface_etymons=dict(surface_etymons),
        cluster_reflexes=dict(cluster_reflexes),
    )


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
# Attestation genitive-marker detector (PRIMARY per-toponym signal).
# ---------------------------------------------------------------------------


def attestation_verdict(
    forms: list[str],
    *,
    split_stems: set[str],
    literal_stems: set[str],
) -> str | None:
    """Classify a toponym from its historical attestation forms — but ONLY when
    the evidence is unambiguous.

    ``'split'`` when a dated form shows the genitive connective (``es`` /
    consonant-``s``) before a town-family stem (Kingston ← *Chinge·s·tun* → ``es``
    + ``tun``) and NO form ends in a literal stone stem. ``'literal'`` when a form
    ends in a stone stem and NO form shows a genitive-marked town stem. ``None``
    when nothing matches OR both patterns appear — the genitive form
    ``alvrice·s·tone`` also ends in the stone stem ``stone``, and that string-level
    ambiguity (``es``+``tone`` town vs ``e``+``stone`` stone) cannot be resolved
    from the spelling alone, so we DEFER to the breakdown cluster classifier
    rather than guess. Attestation overrides the breakdown only when it is
    genuinely decisive ("even one attestation wins" — but only a clear one).

    ``literal_stems`` must NOT contain the ambiguous long form ``L`` itself
    (every ``X's-ton`` form trivially ends in ``ston``); only the unambiguous
    stone-family stems belong here. The caller enforces that.
    """
    saw_split = saw_literal = False
    split_res = [
        re.compile(rf".+(?:es|[bcdfghjklmnpqrstvwxyz]s){re.escape(stem)}$") for stem in split_stems
    ]
    for form in forms:
        folded = _fold(form)
        if not folded:
            continue
        if any(rx.search(folded) for rx in split_res):
            saw_split = True
        if any(folded.endswith(stem) for stem in literal_stems):
            saw_literal = True
    if saw_split and not saw_literal:
        return "split"
    if saw_literal and not saw_split:
        return "literal"
    return None


# ---------------------------------------------------------------------------
# Extraction.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairCounts:
    """Raw per-pair tallies. Split/literal are the aggregate evidence; the
    ``attested_*`` subset records how much came from the strong attestation
    signal vs the breakdown-cluster fallback."""

    split: int = 0
    literal: int = 0
    attested_split: int = 0
    attested_literal: int = 0


@dataclass(frozen=True)
class GenitiveExtractionResult:
    """Per-run summary for the CLI + tests."""

    toponyms_scanned: int
    candidate_pairs: int
    active_pairs: int
    classified_split: int
    classified_literal: int
    from_attestation: int
    from_breakdown: int
    skipped_both: int
    skipped_unclassified: int

    def as_dict(self) -> dict[str, int]:
        return {
            "toponyms_scanned": self.toponyms_scanned,
            "candidate_pairs": self.candidate_pairs,
            "active_pairs": self.active_pairs,
            "classified_split": self.classified_split,
            "classified_literal": self.classified_literal,
            "from_attestation": self.from_attestation,
            "from_breakdown": self.from_breakdown,
            "skipped_both": self.skipped_both,
            "skipped_unclassified": self.skipped_unclassified,
        }


def _build_pair_meta(
    data: ClassifierData, candidates: dict[str, str]
) -> dict[tuple[str, str], dict]:
    """Per-pair cluster + attestation-stem sets, shared across all toponyms."""
    pair_meta: dict[tuple[str, str], dict] = {}
    for long, short in candidates.items():
        literal_clusters = data.clusters_for_surface(long)
        split_clusters = data.clusters_for_surface(short) - literal_clusters
        pair_meta[(long, short)] = {
            "split_clusters": split_clusters,
            "literal_clusters": literal_clusters,
            "split_stems": data.stems_for_clusters(split_clusters) | {short},
            # NB: the ambiguous long form L is deliberately excluded — every
            # genitive 'X's-S' form ends in L, so it carries no stone evidence.
            "literal_stems": data.stems_for_clusters(literal_clusters) - {long},
        }
    return pair_meta


# A scanned toponym/pair observation: (pair, has_split_cluster, has_literal_cluster, attestation).
_ScanRecord = tuple[tuple[str, str], bool, bool, "str | None"]


def _scan_toponyms(
    topo_name: dict[int, str],
    topo_clusters: dict[int, set[str]],
    topo_forms: dict[int, list[str]],
    pair_meta: dict[tuple[str, str], dict],
    progress_every: int,
) -> tuple[list[_ScanRecord], dict[tuple[str, str], int], dict[tuple[str, str], int], int]:
    """Pass 1 — per-toponym cluster facts + attestation verdict, plus the
    per-pair cluster-only split/literal totals used to gate attestation."""
    records: list[_ScanRecord] = []
    cluster_split_total: dict[tuple[str, str], int] = defaultdict(int)
    cluster_literal_total: dict[tuple[str, str], int] = defaultdict(int)
    scanned = 0
    for tid, name in topo_name.items():
        scanned += 1
        if progress_every and scanned % progress_every == 0:
            print(
                f"  [{scanned}] toponyms scanned  records={len(records)}",
                file=sys.stderr,
                flush=True,
            )
        folded_name = _fold(name)
        eclusters = topo_clusters[tid]
        for pair, meta in pair_meta.items():
            if not folded_name.endswith(pair[0]):
                continue
            has_split = bool(eclusters & meta["split_clusters"])
            has_literal = bool(eclusters & meta["literal_clusters"])
            av = attestation_verdict(
                topo_forms.get(tid, []),
                split_stems=meta["split_stems"],
                literal_stems=meta["literal_stems"],
            )
            records.append((pair, has_split, has_literal, av))
            if has_split and not has_literal:
                cluster_split_total[pair] += 1
            elif has_literal and not has_split:
                cluster_literal_total[pair] += 1
    if progress_every and scanned % progress_every != 0:
        print(
            f"  [{scanned}] toponyms scanned  records={len(records)}", file=sys.stderr, flush=True
        )
    return records, cluster_split_total, cluster_literal_total, scanned


def _classify_record(
    rec: _ScanRecord,
    cluster_split_total: dict[tuple[str, str], int],
    cluster_literal_total: dict[tuple[str, str], int],
) -> tuple[str | None, bool, str | None]:
    """``(verdict, via_attestation, skip_reason)``. Attestation is authoritative
    ONLY for a class the pair genuinely exhibits in breakdowns; otherwise the
    cluster class decides. ``verdict is None`` ⇒ skipped with ``skip_reason``."""
    pair, has_split, has_literal, av = rec
    if av == "split" and cluster_split_total[pair] > 0:
        return "split", True, None
    if av == "literal" and cluster_literal_total[pair] > 0:
        return "literal", True, None
    if has_split and has_literal:
        return None, False, "both"
    if has_split:
        return "split", False, None
    if has_literal:
        return "literal", False, None
    return None, False, "unclassified"


def _tally_records(
    records: list[_ScanRecord],
    cluster_split_total: dict[tuple[str, str], int],
    cluster_literal_total: dict[tuple[str, str], int],
) -> tuple[dict[tuple[str, str], PairCounts], dict[str, int]]:
    """Pass 2 — resolve each record to a final verdict and tally per pair."""
    cells: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"split": 0, "literal": 0, "attested_split": 0, "attested_literal": 0}
    )
    stats = dict.fromkeys(
        ("split", "literal", "attestation", "breakdown", "both", "unclassified"), 0
    )
    for rec in records:
        verdict, via_attestation, skip = _classify_record(
            rec, cluster_split_total, cluster_literal_total
        )
        if verdict is None:
            stats[skip] += 1
            continue
        cell = cells[rec[0]]
        cell[verdict] += 1
        if via_attestation:
            cell[f"attested_{verdict}"] += 1
            stats["attestation"] += 1
        else:
            stats["breakdown"] += 1
        stats[verdict] += 1
    counts = {
        pair: PairCounts(
            split=c["split"],
            literal=c["literal"],
            attested_split=c["attested_split"],
            attested_literal=c["attested_literal"],
        )
        for pair, c in cells.items()
    }
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
    one toponym appear). Per toponym: attestation verdict first (authoritative),
    breakdown cluster class as fallback. One verdict per toponym (its breakdown
    elements pooled); every place a name appears counts (pooling-across-regions
    is sound — wyrd-aicu.9 investigation 3).
    """
    data = load_classifier_data(db)
    candidates = discover_candidate_pairs(set(data.surface_etymons))

    # toponym_id → (folded modern_name, pooled element clusters)
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

    # toponym_id → historical attestation forms
    topo_forms: dict[int, list[str]] = defaultdict(list)
    for row in db.conn.execute("SELECT toponym_id, form FROM toponym_attestation"):
        topo_forms[row["toponym_id"]].append(row["form"])

    pair_meta = _build_pair_meta(data, candidates)
    # Pass 1: scan + gate-totals. Pass 2: resolve + tally. The two-pass split
    # lets attestation gate on a class the breakdowns actually support — it must
    # not invent a split side no breakdown exhibits (the ``-ster`` false-positive:
    # modern "Manchester" reads as manche·s·ter but ceaster is never a genitive).
    records, cluster_split_total, cluster_literal_total, scanned = _scan_toponyms(
        topo_name, topo_clusters, topo_forms, pair_meta, progress_every
    )
    counts, stats = _tally_records(records, cluster_split_total, cluster_literal_total)

    summary = GenitiveExtractionResult(
        toponyms_scanned=scanned,
        candidate_pairs=len(candidates),
        active_pairs=len(counts),
        classified_split=stats["split"],
        classified_literal=stats["literal"],
        from_attestation=stats["attestation"],
        from_breakdown=stats["breakdown"],
        skipped_both=stats["both"],
        skipped_unclassified=stats["unclassified"],
    )
    return counts, summary


def _replace_priors(db: LexiconDB, counts: dict[tuple[str, str], PairCounts]) -> None:
    """Clear and repopulate ``genitive_split_prior`` atomically (replace-not-
    merge — a stale pair from a removed observation must not survive)."""
    with db.conn:
        db.conn.execute("DELETE FROM genitive_split_prior")
        db.conn.executemany(
            "INSERT INTO genitive_split_prior "
            "(long_form, short_form, split_count, literal_count, "
            " attested_split, attested_literal) VALUES (?, ?, ?, ?, ?, ?)",
            (
                (long, short, c.split, c.literal, c.attested_split, c.attested_literal)
                for (long, short), c in counts.items()
            ),
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
        "SELECT long_form, short_form, split_count, literal_count, "
        "attested_split, attested_literal FROM genitive_split_prior "
        "ORDER BY split_count + literal_count DESC, long_form, short_form"
    )
    pairs = [
        {
            "long_form": r["long_form"],
            "short_form": r["short_form"],
            "split_count": r["split_count"],
            "literal_count": r["literal_count"],
            "attested_split": r["attested_split"],
            "attested_literal": r["attested_literal"],
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
            attested_split=int(p.get("attested_split", 0)),
            attested_literal=int(p.get("attested_literal", 0)),
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
    N≈380) trusts its own ratio. As breakdowns accrue the estimate converges to
    the empirical ratio — the "improves as the data improves" curve. Smoothing
    lives HERE, at lookup, not in the persisted counts.
    """
    if backoff is None:
        backoff = global_split_rate(counts)
    c = counts.get((long_form, short_form), PairCounts())
    n = c.split + c.literal
    return (c.split + pseudocount * backoff) / (n + pseudocount)
