"""Tests for the realism-retention tolerance-band machinery
(wyrd-ecjp.7 Phase 6b).

Pins the tolerance-check primitives. The actual regression-test
suite that runs against the live generators lives separately;
this file just verifies the per-metric tolerance logic.
"""

from __future__ import annotations

from wyrd.generators.kenning.runtime.drift_measurement import DriftReport
from wyrd.generators.kenning.runtime.realism_tolerance import (
    DEFAULT_TOLERANCE,
    PER_CULTURE_TOLERANCES,
    ToleranceBand,
    ToleranceViolation,
    check_drift_against_tolerance,
    format_violations,
    tolerance_for,
)

# ---- tolerance_for + per-culture resolution ------------------------------


def test_tolerance_for_unknown_culture_returns_default():
    assert tolerance_for("nonexistent") is DEFAULT_TOLERANCE


def test_tolerance_for_known_culture_returns_override():
    """If an operator adds an override to PER_CULTURE_TOLERANCES at
    module-import time, tolerance_for returns it. The test stages
    its own override via dict-mutation in a try/finally so the
    global state stays clean for sibling tests."""
    custom = ToleranceBand(max_kl_divergence=0.5)
    PER_CULTURE_TOLERANCES["test-culture"] = custom
    try:
        assert tolerance_for("test-culture") is custom
    finally:
        PER_CULTURE_TOLERANCES.pop("test-culture")


# ---- check_drift_against_tolerance ---------------------------------------


def _report(
    culture: str = "english",
    *,
    kl=0.0,
    tv=0.0,
    overlap=1.0,
    decomp_delta=0.0,
    rank=1.0,
) -> DriftReport:
    return DriftReport(
        culture=culture,
        sample_size_a=100,
        sample_size_b=100,
        tag_kl_divergence=kl,
        tag_total_variation=tv,
        top_n_name_overlap=overlap,
        decomposition_rate_delta=decomp_delta,
        morpheme_rank_correlation=rank,
    )


def test_default_tolerance_passes_for_no_drift_report():
    """A report with no drift (all metrics at their zero-drift values)
    has zero violations under the default wide tolerance."""
    report = _report()
    assert check_drift_against_tolerance(report, DEFAULT_TOLERANCE) == []


def test_default_tolerance_passes_for_maximum_drift_report():
    """Default tolerances are wide-open — even maximum drift passes.
    This is the deliberate Phase 6b 'inactive as a gate today'
    behavior; operator tightens via PER_CULTURE_TOLERANCES."""
    report = _report(kl=5.0, tv=1.0, overlap=0.0, decomp_delta=-1.0, rank=-1.0)
    assert check_drift_against_tolerance(report, DEFAULT_TOLERANCE) == []


def test_tolerance_violation_kl_too_high():
    tolerance = ToleranceBand(max_kl_divergence=0.5)
    report = _report(kl=1.5)
    violations = check_drift_against_tolerance(report, tolerance)
    assert len(violations) == 1
    v = violations[0]
    assert v.metric == "tag_kl_divergence"
    assert v.direction == "above"
    assert v.value == 1.5
    assert v.threshold == 0.5


def test_tolerance_violation_top_n_overlap_too_low():
    tolerance = ToleranceBand(min_top_n_overlap=0.5)
    report = _report(overlap=0.2)
    violations = check_drift_against_tolerance(report, tolerance)
    assert len(violations) == 1
    assert violations[0].metric == "top_n_name_overlap"
    assert violations[0].direction == "below"


def test_tolerance_violation_decomp_delta_symmetric():
    """decomposition_rate_delta is checked via abs() — both directions
    of drift fail."""
    tolerance = ToleranceBand(max_decomposition_rate_delta_abs=0.1)
    # Positive drift
    violations = check_drift_against_tolerance(_report(decomp_delta=0.3), tolerance)
    assert len(violations) == 1
    # Negative drift
    violations = check_drift_against_tolerance(_report(decomp_delta=-0.3), tolerance)
    assert len(violations) == 1


def test_tolerance_violation_morpheme_rank_too_low():
    tolerance = ToleranceBand(min_morpheme_rank_correlation=0.5)
    report = _report(rank=0.2)
    violations = check_drift_against_tolerance(report, tolerance)
    assert len(violations) == 1
    assert violations[0].metric == "morpheme_rank_correlation"


def test_tolerance_violation_total_variation_too_high():
    tolerance = ToleranceBand(max_total_variation=0.2)
    report = _report(tv=0.5)
    violations = check_drift_against_tolerance(report, tolerance)
    assert len(violations) == 1
    assert violations[0].metric == "tag_total_variation"


def test_tolerance_none_disables_check():
    """When a tolerance field is None, the corresponding metric
    isn't checked even when it would otherwise fail."""
    tolerance = ToleranceBand(max_kl_divergence=None)
    # KL of 100.0 with None tolerance → no violation
    report = _report(kl=100.0)
    violations = check_drift_against_tolerance(report, tolerance)
    # No KL violation, but other metrics still default
    assert not any(v.metric == "tag_kl_divergence" for v in violations)


def test_tolerance_multiple_violations_reported():
    """A drift report with multiple violations surfaces each one."""
    tolerance = ToleranceBand(
        max_kl_divergence=0.1,
        min_top_n_overlap=0.9,
        min_morpheme_rank_correlation=0.9,
    )
    report = _report(kl=2.0, overlap=0.1, rank=0.1)
    violations = check_drift_against_tolerance(report, tolerance)
    assert len(violations) == 3
    metrics = {v.metric for v in violations}
    assert metrics == {
        "tag_kl_divergence",
        "top_n_name_overlap",
        "morpheme_rank_correlation",
    }


def test_check_uses_report_culture_when_tolerance_omitted():
    """When tolerance arg is None, the function resolves the band
    from the report's culture via tolerance_for."""
    PER_CULTURE_TOLERANCES["mytest"] = ToleranceBand(max_kl_divergence=0.1)
    try:
        report = _report(culture="mytest", kl=0.5)
        violations = check_drift_against_tolerance(report)  # no tolerance arg
        assert len(violations) == 1
        assert violations[0].metric == "tag_kl_divergence"
    finally:
        PER_CULTURE_TOLERANCES.pop("mytest")


# ---- ToleranceViolation / format_violations ------------------------------


def test_violation_str_format():
    v = ToleranceViolation(
        culture="english",
        metric="tag_kl_divergence",
        value=2.5,
        threshold=0.5,
        direction="above",
    )
    s = str(v)
    assert "english" in s
    assert "tag_kl_divergence" in s
    assert "2.5" in s
    assert "above" in s


def test_format_violations_empty():
    assert format_violations([]) == "all metrics within tolerance"


def test_format_violations_multiple():
    violations = [
        ToleranceViolation(
            culture="english",
            metric="tag_kl_divergence",
            value=2.0,
            threshold=0.5,
            direction="above",
        ),
        ToleranceViolation(
            culture="welsh",
            metric="top_n_name_overlap",
            value=0.1,
            threshold=0.5,
            direction="below",
        ),
    ]
    out = format_violations(violations)
    assert "Tolerance violations" in out
    assert "english" in out
    assert "welsh" in out
