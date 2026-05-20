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
    DriftReport,
    NameSample,
    _spearman_correlation,
    _tag_distribution,
    compute_drift_report,
    decomposition_rate,
    format_drift_report_json,
    format_drift_report_markdown,
    kl_divergence,
    morpheme_rank_correlation,
    position_distribution,
    top_n_name_overlap,
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


# ---- top_n_name_overlap --------------------------------------------------


def test_top_n_overlap_identical_samples():
    """Same samples on both sides → 1.0 overlap."""
    samples = [_ns(f"name{i}") for i in range(5)]
    assert top_n_name_overlap(samples, samples, n=100) == 1.0


def test_top_n_overlap_disjoint():
    a = [_ns("a1"), _ns("a2")]
    b = [_ns("b1"), _ns("b2")]
    assert top_n_name_overlap(a, b, n=100) == 0.0


def test_top_n_overlap_partial():
    a = [_ns("shared1"), _ns("shared2"), _ns("a_only")]
    b = [_ns("shared1"), _ns("shared2"), _ns("b_only")]
    # All 3 from each side land in top-100; intersection={shared1, shared2},
    # union={shared1, shared2, a_only, b_only} → 2/4 = 0.5
    assert math.isclose(top_n_name_overlap(a, b, n=100), 0.5)


def test_top_n_overlap_top_n_constraint():
    """When n < distinct count, only the top-n compete in the
    overlap calculation."""
    # 'common' appears 5 times in both; 'rare_a' / 'rare_b' once each
    a = [_ns("common")] * 5 + [_ns("rare_a")]
    b = [_ns("common")] * 5 + [_ns("rare_b")]
    # n=1: only 'common' is in both top-1 → 1.0 overlap
    assert top_n_name_overlap(a, b, n=1) == 1.0
    # n=2: top-2 for a = {common, rare_a}; top-2 for b = {common, rare_b}
    # → intersection={common}, union={common, rare_a, rare_b} → 1/3
    assert math.isclose(top_n_name_overlap(a, b, n=2), 1 / 3)


def test_top_n_overlap_empty_input():
    assert top_n_name_overlap([], [_ns("x")], n=100) == 0.0
    assert top_n_name_overlap([_ns("x")], [], n=100) == 0.0


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


# ---- morpheme_rank_correlation -------------------------------------------


def test_morpheme_rank_correlation_identical_samples():
    samples = [
        _ns("n1", morphemes=("a", "b")),
        _ns("n2", morphemes=("a", "c")),
    ]
    corr, n_shared = morpheme_rank_correlation(samples, samples, top_k=100)
    assert corr == 1.0
    assert n_shared == 3  # a, b, c all in both


def test_morpheme_rank_correlation_disjoint_morphemes():
    a = [_ns("n1", morphemes=("a", "b"))]
    b = [_ns("n2", morphemes=("c", "d"))]
    corr, n_shared = morpheme_rank_correlation(a, b, top_k=100)
    assert corr == 0.0
    assert n_shared == 0


def test_morpheme_rank_correlation_partial_overlap():
    # Both share 'shared1' + 'shared2', but ranks differ
    a = [_ns("n1", morphemes=("shared1",))] * 3 + [_ns("n2", morphemes=("shared2",))]
    b = [_ns("n1", morphemes=("shared2",))] * 3 + [_ns("n2", morphemes=("shared1",))]
    corr, n_shared = morpheme_rank_correlation(a, b, top_k=100)
    assert n_shared == 2
    # Inverse ranking on the two shared morphemes
    assert corr == -1.0


# ---- compute_drift_report ------------------------------------------------


def test_compute_drift_report_identical_distributions():
    """Same samples on both sides → all drift metrics indicate
    no-drift (KL=0, TV=0, overlap=1, correlation=1)."""
    samples = [
        _ns(
            f"name{i}",
            tags=("a", "b"),
            positions=("pre", "post"),
            decomposes=True,
            morphemes=("x", "y"),
        )
        for i in range(10)
    ]
    report = compute_drift_report("english", samples, samples)
    assert report.sample_size_a == 10
    assert report.sample_size_b == 10
    assert math.isclose(report.tag_kl_divergence, 0.0, abs_tol=1e-6)
    assert report.tag_total_variation == 0.0
    assert report.top_n_name_overlap == 1.0
    assert report.decomposition_rate_a == 1.0
    assert report.decomposition_rate_b == 1.0
    assert report.decomposition_rate_delta == 0.0
    assert report.morpheme_rank_correlation == 1.0
    assert report.morpheme_rank_top_k == 100


def test_compute_drift_report_culture_in_output():
    report = compute_drift_report("welsh", [], [])
    assert report.culture == "welsh"


def test_compute_drift_report_empty_samples_both_sides():
    """Empty input on both sides → all metrics degrade to 0
    gracefully (no division-by-zero)."""
    report = compute_drift_report("english", [], [])
    assert report.tag_kl_divergence == 0.0
    assert report.tag_total_variation == 0.0
    assert report.top_n_name_overlap == 0.0
    assert report.decomposition_rate_a == 0.0
    assert report.morpheme_rank_correlation == 0.0


def test_compute_drift_report_materializes_iterables_once():
    """Each metric needs a full pass; if the implementation re-
    iterated the input generator, the second pass would be empty.
    Verify by passing generators (one-shot iterables) and checking
    the metrics all compute correctly."""

    def gen():
        for i in range(5):
            # Use 2 distinct morphemes so Spearman rank correlation
            # has n >= 2 shared morphemes (otherwise undefined → 0).
            yield _ns(f"n{i}", tags=("a",), morphemes=("x", "y"))

    report = compute_drift_report("english", gen(), gen())
    assert report.sample_size_a == 5
    assert report.sample_size_b == 5
    # Correlation should be 1.0 since both generators yield the same
    # morphemes
    assert report.morpheme_rank_correlation == 1.0


# ---- formatters ----------------------------------------------------------


def test_format_drift_report_markdown_contains_culture():
    report = DriftReport(culture="welsh", sample_size_a=100, sample_size_b=100)
    md = format_drift_report_markdown(report)
    assert "welsh" in md
    assert "Drift Report" in md


def test_format_drift_report_markdown_uses_correct_top_n_label():
    """Round-1 reviewer caught: an earlier version labeled the
    Top-N overlap line with morpheme_rank_top_k (the morpheme
    correlation parameter) instead of top_n_overlap_n. Verify the
    field is the right one + the label matches the n we passed to
    compute_drift_report."""
    samples = [NameSample(surface=f"n{i}") for i in range(5)]
    report = compute_drift_report(
        "english", samples, samples, top_n_overlap=50, top_k_correlation=200
    )
    md = format_drift_report_markdown(report)
    # Top-N overlap label says 'top-50' (not 'top-200'); morpheme
    # rank correlation label says 'top-200' (not 'top-50').
    assert "top-50" in md
    assert "top-200" in md
    assert report.top_n_overlap_n == 50
    assert report.morpheme_rank_top_k == 200


def test_format_drift_report_json_round_trips():
    import json

    report = DriftReport(
        culture="english",
        sample_size_a=100,
        sample_size_b=100,
        tag_kl_divergence=0.5,
        decomposition_rate_a=0.95,
    )
    j = format_drift_report_json(report)
    # JSON-serializable
    s = json.dumps(j)
    parsed = json.loads(s)
    assert parsed["culture"] == "english"
    assert parsed["tag_kl_divergence"] == 0.5
    assert parsed["decomposition_rate_a"] == 0.95


def test_format_drift_report_json_includes_top_n_overlap_n():
    """Round-1 reviewer fix: the JSON shape now carries top_n_overlap_n
    so consumers can correctly label the overlap value's parameter."""
    report = DriftReport(culture="english", sample_size_a=10, sample_size_b=10, top_n_overlap_n=50)
    j = format_drift_report_json(report)
    assert j["top_n_overlap_n"] == 50
