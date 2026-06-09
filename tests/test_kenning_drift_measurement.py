"""Tests for the realism-retention drift measurement primitives
(wyrd-ecjp.6 Phase 6a).

The drift-measurement machinery computes per-metric drift between
two name-sample distributions. These tests pin the per-metric
contracts against hand-crafted synthetic samples — no live bundle
or generator dependency.
"""

from __future__ import annotations

import math

from wyrd.generators.kenning.runtime.drift_measurement import (
    NameSample,
    _spearman_correlation,
    _tag_distribution,
    decomposition_rate,
    kl_divergence,
    position_distribution,
    total_variation_distance,
)

# ---- helpers --------------------------------------------------------------


def _ns(surface: str, *, tags=(), positions=(), decomposes=True, morphemes=()) -> NameSample:
    return NameSample(
        surface=surface,
        tags=tuple(tags),
        positions=tuple(positions),
        decomposes=decomposes,
        morphemes=tuple(morphemes),
    )


# ---- _tag_distribution + KL divergence -----------------------------------


def test_tag_distribution_empty():
    assert _tag_distribution([]) == {}


def test_tag_distribution_normalizes_by_total_occurrence_count():
    samples = [
        _ns("name1", tags=("a", "b")),  # 2 tag-occurrences
        _ns("name2", tags=("a",)),  # 1 tag-occurrence
    ]
    dist = _tag_distribution(samples)
    # Total occurrences = 3; a appears twice, b once
    assert math.isclose(dist["a"], 2 / 3)
    assert math.isclose(dist["b"], 1 / 3)


def test_kl_divergence_zero_for_identical_distributions():
    p = {"a": 0.5, "b": 0.5}
    assert kl_divergence(p, p) < 1e-6


def test_kl_divergence_handles_zero_q_via_smoothing():
    """When q has zero probability for a key p covers, the smoothed
    KL still computes a finite (large) value — important for the
    drift-measurement use case where the new path may produce tags
    the legacy path never did."""
    p = {"a": 1.0}
    q = {"b": 1.0}  # q gives 0 prob to "a"
    kl = kl_divergence(p, q)
    assert kl > 0  # large but finite
    assert math.isfinite(kl)


def test_kl_divergence_empty_p_returns_zero():
    assert kl_divergence({}, {"a": 1.0}) == 0.0


# ---- total_variation_distance --------------------------------------------


def test_total_variation_identical():
    p = {"a": 0.5, "b": 0.5}
    assert total_variation_distance(p, p) == 0.0


def test_total_variation_disjoint_supports():
    p = {"a": 1.0}
    q = {"b": 1.0}
    assert total_variation_distance(p, q) == 1.0


def test_total_variation_symmetric():
    p = {"a": 0.3, "b": 0.7}
    q = {"a": 0.8, "b": 0.2}
    assert total_variation_distance(p, q) == total_variation_distance(q, p)


def test_total_variation_empty_inputs_returns_zero():
    assert total_variation_distance({}, {}) == 0.0


# ---- decomposition_rate --------------------------------------------------


def test_decomposition_rate_all_decompose():
    samples = [_ns(f"name{i}", decomposes=True) for i in range(10)]
    assert decomposition_rate(samples) == 1.0


def test_decomposition_rate_none_decompose():
    samples = [_ns(f"name{i}", decomposes=False) for i in range(10)]
    assert decomposition_rate(samples) == 0.0


def test_decomposition_rate_partial():
    samples = [
        _ns("a", decomposes=True),
        _ns("b", decomposes=True),
        _ns("c", decomposes=False),
        _ns("d", decomposes=False),
    ]
    assert decomposition_rate(samples) == 0.5


def test_decomposition_rate_empty():
    assert decomposition_rate([]) == 0.0


# ---- position_distribution -----------------------------------------------


def test_position_distribution_balanced():
    samples = [
        _ns("name1", positions=("pre", "post")),
        _ns("name2", positions=("pre", "post")),
    ]
    dist = position_distribution(samples)
    assert dist["pre"] == 0.5
    assert dist["post"] == 0.5


def test_position_distribution_skewed():
    samples = [
        _ns("n1", positions=("pre", "pre", "post")),
        _ns("n2", positions=("pre", "post")),
    ]
    dist = position_distribution(samples)
    # 3 pre + 2 post = 5 total
    assert math.isclose(dist["pre"], 3 / 5)
    assert math.isclose(dist["post"], 2 / 5)


def test_position_distribution_empty():
    assert position_distribution([]) == {}


# ---- _spearman_correlation -----------------------------------------------


def test_spearman_perfect_positive():
    ranks_a = {"x": 1, "y": 2, "z": 3}
    ranks_b = {"x": 1, "y": 2, "z": 3}
    assert math.isclose(_spearman_correlation(ranks_a, ranks_b), 1.0)


def test_spearman_perfect_negative():
    ranks_a = {"x": 1, "y": 2, "z": 3}
    ranks_b = {"x": 3, "y": 2, "z": 1}
    assert math.isclose(_spearman_correlation(ranks_a, ranks_b), -1.0)


def test_spearman_no_overlap_returns_zero():
    """Fewer than 2 shared morphemes → 0 (undefined correlation)."""
    ranks_a = {"x": 1}
    ranks_b = {"y": 1}
    assert _spearman_correlation(ranks_a, ranks_b) == 0.0


def test_spearman_constant_returns_zero():
    """All identical ranks on one side → 0 variance → 0 correlation."""
    ranks_a = {"x": 1, "y": 1}
    ranks_b = {"x": 1, "y": 2}
    assert _spearman_correlation(ranks_a, ranks_b) == 0.0


def test_spearman_constant_other_side_returns_zero():
    """Symmetric coverage: zero variance on the OTHER side also → 0."""
    ranks_a = {"x": 1, "y": 2}
    ranks_b = {"x": 5, "y": 5}
    assert _spearman_correlation(ranks_a, ranks_b) == 0.0
