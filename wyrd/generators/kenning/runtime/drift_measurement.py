"""Realism-retention measurement (wyrd-jfaz).

ABSOLUTE corpus-realism measurement: generates N names from the vector
scoring path and measures them against a corpus reference derived from
bundle data. Per-metric:

* **Per-tag distribution shift** — KL divergence + total-variation
  distance between the sample bag-of-tags distribution and the corpus
  reference distribution.
* **Decomposition rate** — fraction of generated names that
  fully-decompose (zero unaccounted chars) under the matcher.
  Real-corpus-like distributions decompose at higher rates;
  noise-shifted distributions drift downward.
* **Position distribution** — fraction of slots filled by pre /
  inner / post meanings, per culture.
* **Per-morpheme frequency rank correlation** — Spearman ρ between
  the sample top-K morpheme ranking and the corpus's expected ranking.
  High ρ = same morphemes emphasized in the same order; low ρ =
  re-ranked distribution.

The output is a :class:`RealismReport` that downstream callers (the
absolute-realism test suite) consume directly.

Pure-Python: no scipy / numpy dependency to keep the Lambda
cold-start budget intact. The metrics here are small enough that
hand-rolled implementations are clean.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class NameSample:
    """One generated name's relevant features for drift measurement.

    Captures only what the metrics need so the measurement code
    doesn't depend on the full Kenning Name object's surface. Tests
    can build NameSample instances directly without standing up a
    full bundle.

    * ``surface`` — the rendered name string (used by top-100 overlap)
    * ``tags`` — tuple of tag strings carried by the picked morphemes
      (used by per-tag distribution shift)
    * ``positions`` — per-slot position labels (used by position
      distribution); empty tuple when not tracked
    * ``decomposes`` — True if the name fully-decomposes under the
      matcher (used by decomposition-rate metric)
    * ``morphemes`` — tuple of bare morpheme usage strings (used by
      per-morpheme frequency rank correlation)
    """

    surface: str
    tags: tuple[str, ...] = ()
    positions: tuple[str, ...] = ()
    decomposes: bool = True
    morphemes: tuple[str, ...] = ()


# ---- per-tag distribution shift -----------------------------------------


def _tag_distribution(samples: Iterable[NameSample]) -> dict[str, float]:
    """Normalized tag-frequency distribution. Each tag's frequency is
    its share of total tag-occurrences across the sample set.

    A name with tags=('water', 'place') contributes 1 each to water +
    place; the denominator is the total tag-occurrence count, not the
    sample count. This way a sample of 1000 names where each carries
    2 tags has the same scale as 1000 names where each carries 5.
    """
    counts: Counter[str] = Counter()
    for s in samples:
        counts.update(s.tags)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {t: c / total for t, c in counts.items()}


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q) — relative entropy of p with respect to q.

    Per the standard definition: ``sum_over_t( p[t] * log(p[t] /
    q[t]) )``. When q[t] is zero, p[t]/q[t] is undefined; we apply
    additive smoothing (add a small epsilon to q for keys p covers).
    The smoothing constant is fixed (1e-9), so re-runs produce
    byte-identical drift values for the same input pair (determinism,
    not symmetry — KL is asymmetric by definition).

    Returns 0 when p is empty (no information to compare); for
    non-empty p, the result is mathematically >= 0 but may surface
    a tiny negative artifact (~1e-9) when p == q exactly because
    the epsilon shifts q without shifting p. Callers comparing
    against zero should use a tolerance (e.g. ``abs(kl) < 1e-6``)
    rather than ``kl == 0``.
    """
    if not p:
        return 0.0
    epsilon = 1e-9
    out = 0.0
    for t, pt in p.items():
        if pt <= 0:
            continue
        qt = q.get(t, 0.0) + epsilon
        out += pt * math.log(pt / qt)
    return out


def total_variation_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Total-variation distance (half the L1 norm of the difference).

    Range [0, 1]. 0 = identical distributions; 1 = disjoint supports.
    Symmetric: TV(p, q) == TV(q, p). More forgiving than KL since
    it doesn't blow up on zero-probability events.
    """
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


# ---- decomposition rate -------------------------------------------------


def decomposition_rate(samples: Iterable[NameSample]) -> float:
    """Fraction of samples that fully-decompose under the matcher
    (decomposes=True). Range [0, 1]. Real-corpus-like distributions
    decompose at high rates (~0.95+); a vector path that drifts
    toward noise drops this rate."""
    samples_list = list(samples)
    if not samples_list:
        return 0.0
    return sum(1 for s in samples_list if s.decomposes) / len(samples_list)


# ---- position distribution ----------------------------------------------


def position_distribution(samples: Iterable[NameSample]) -> dict[str, float]:
    """Normalized per-slot position-label distribution.

    Counts per-slot positions across all samples; returns the share
    of slots filled by each label (pre / inner / post). The legacy
    proportion-table sampler's distribution is shaped by the
    structure-pick weights; the vector path's distribution depends
    on which meanings score highest per slot. Material drift here
    indicates the position axis is shifting which slot positions
    are filled more often.
    """
    counts: Counter[str] = Counter()
    for s in samples:
        counts.update(s.positions)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {pos: c / total for pos, c in counts.items()}


# ---- morpheme rank correlation ------------------------------------------


def _spearman_correlation(
    ranks_a: dict[str, int],
    ranks_b: dict[str, int],
) -> float:
    """Spearman rank correlation between two morpheme-rank dicts.

    For morphemes appearing in BOTH ranks, compute the Pearson
    correlation of their rank values. Morphemes in only one dict
    are skipped (correlation undefined for them).

    Range [-1, 1]. 1 = identical ranking; -1 = inverse; 0 = no
    linear relationship. Returns 0.0 when fewer than 2 morphemes
    appear in both (correlation undefined).
    """
    shared = sorted(set(ranks_a) & set(ranks_b))
    n = len(shared)
    if n < 2:
        return 0.0
    # Extract paired rank vectors over the shared morphemes
    xs = [ranks_a[m] for m in shared]
    ys = [ranks_b[m] for m in shared]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        # All x's identical or all y's identical → variance 0 → undefined.
        # Return 0 (no information about correlation direction).
        return 0.0
    return cov / denom


def morpheme_rank_correlation(
    samples_a: Iterable[NameSample],
    samples_b: Iterable[NameSample],
    top_k: int = 100,
) -> tuple[float, int]:
    """Spearman rank correlation of morpheme frequencies between two
    sample sets, computed over the top-K morphemes from each.

    Returns ``(correlation, n_shared)`` where n_shared is the count
    of morphemes that appear in BOTH top-K sets (the correlation's
    effective sample size). Sparse intersection (n_shared < 2)
    returns correlation=0.0.
    """
    counts_a: Counter[str] = Counter()
    counts_b: Counter[str] = Counter()
    for s in samples_a:
        counts_a.update(s.morphemes)
    for s in samples_b:
        counts_b.update(s.morphemes)

    # Build rank dicts over the top-K most-frequent morphemes
    top_a = counts_a.most_common(top_k)
    top_b = counts_b.most_common(top_k)
    # Rank 1 = most frequent; ties broken by insertion order (stable
    # given Counter.most_common's sort-then-iter semantics).
    ranks_a = {morpheme: rank for rank, (morpheme, _) in enumerate(top_a, start=1)}
    ranks_b = {morpheme: rank for rank, (morpheme, _) in enumerate(top_b, start=1)}
    n_shared = len(set(ranks_a) & set(ranks_b))
    return _spearman_correlation(ranks_a, ranks_b), n_shared


# ---- absolute corpus-realism report (wyrd-jfaz) -------------------------


@dataclass
class RealismReport:
    """ABSOLUTE realism report: one sample set (typically vector-mode)
    measured against a :class:`realism_reference.CorpusReference`. The
    reference is derived from bundle data, so the gate is independent of
    any scoring baseline and tracks corpus growth (epic wyrd-ej28).

    Metric semantics, with ``q`` = the reference:
    ``tag_kl_divergence``/``tag_total_variation`` (tag-occurrence shift
    vs corpus), ``position_total_variation`` (slot-position shift vs
    corpus), ``morpheme_rank_correlation`` (Spearman vs the corpus's
    expected morpheme ranks). ``decomposition_rate`` is an absolute
    quality FLOOR.
    """

    culture: str
    sample_size: int
    tag_kl_divergence: float = 0.0
    tag_total_variation: float = 0.0
    position_total_variation: float = 0.0
    morpheme_rank_correlation: float = 0.0
    morpheme_rank_top_k: int = 0
    decomposition_rate: float = 0.0


def compute_realism_report(  # noqa: V103 — feeds the absolute corpus-realism CI gate
    culture: str,
    samples: Iterable[NameSample],
    reference,
    *,
    top_k_correlation: int = 100,
) -> RealismReport:
    """Measure ``samples`` against an absolute ``CorpusReference`` for one
    culture (wyrd-jfaz).

    ``reference`` is a :class:`realism_reference.CorpusReference`. Each
    sample distribution is compared against the reference's analytical
    distribution:

    * tag-KL / tag-TV: KL/TV of the sample tag distribution vs the
      reference tag distribution.
    * position-TV: TV of the sample position distribution vs the
      reference position distribution.
    * morpheme-rank ρ: Spearman of the sample's top-K morpheme ranks
      against the reference's top-K morpheme ranks (the reference's
      ranks come from its expected per-morpheme pick weights).
    * decomposition-rate: absolute (the corpus reference has no
      decomposition rate of its own; the band treats this as a floor).
    """
    samples_list = list(samples)

    sample_tag = _tag_distribution(samples_list)
    sample_position = position_distribution(samples_list)

    sample_counts: Counter[str] = Counter()
    for s in samples_list:
        sample_counts.update(s.morphemes)
    sample_ranks = {
        morpheme: rank
        for rank, (morpheme, _) in enumerate(sample_counts.most_common(top_k_correlation), start=1)
    }
    reference_ranks = {
        morpheme: rank
        for rank, (morpheme, _) in enumerate(
            sorted(reference.morpheme_counts.items(), key=lambda kv: -kv[1])[:top_k_correlation],
            start=1,
        )
    }
    correlation = _spearman_correlation(sample_ranks, reference_ranks)
    n_shared = len(set(sample_ranks) & set(reference_ranks))

    return RealismReport(
        culture=culture,
        sample_size=len(samples_list),
        tag_kl_divergence=kl_divergence(sample_tag, reference.tag_distribution),
        tag_total_variation=total_variation_distance(sample_tag, reference.tag_distribution),
        position_total_variation=total_variation_distance(
            sample_position, reference.position_distribution
        ),
        morpheme_rank_correlation=correlation,
        morpheme_rank_top_k=n_shared,
        decomposition_rate=decomposition_rate(samples_list),
    )
