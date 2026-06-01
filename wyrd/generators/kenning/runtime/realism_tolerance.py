"""Realism-retention tolerance bands (wyrd-ecjp.7 Phase 6b).

Codifies the operator-decided tolerance per metric × culture for the
drift-measurement infrastructure (wyrd-ecjp.6 Phase 6a). The
regression test suite in ``tests/test_kenning_realism_regression.py``
reads these tolerances + runs drift measurements against the live
generators; tolerance violations fail the test.

The tolerances here are SCAFFOLDING with operator-tunable defaults.
The actual per-metric × per-culture bands need operator review of
real drift output (post-ecjp.8 bundle changes when vector mode
becomes more meaningfully different from proportions mode). Until
then the defaults are deliberately wide:

* KL divergence ceiling: 10.0 (very loose — accepts even highly
  divergent distributions)
* Total-variation ceiling: 1.0 (the maximum value; effectively
  unchecked)
* Top-N overlap floor: 0.0 (accepts no overlap)
* Decomposition delta ceiling: 1.0 (accepts complete drift)
* Spearman correlation floor: -1.0 (accepts inverse correlation)

These wide defaults mean the regression test suite is INACTIVE as a
drift gate today. The infrastructure is in place; the operator
tightens the bands once empirical drift evidence stabilizes
(Phase 6b's deliberate review-then-codify cycle).

The DEFAULT_TOLERANCES dict is the single source of truth; per-
culture overrides extend / replace it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToleranceBand:
    """Per-metric tolerance band. A drift report passes when every
    metric's value falls within its tolerance band.

    All fields default to "no constraint" — the operator tightens by
    overriding specific fields per-culture in PER_CULTURE_TOLERANCES.

    * ``max_kl_divergence`` — drift report's tag_kl_divergence must
      be <= this. Lower = stricter.
    * ``max_total_variation`` — tag_total_variation must be <=. [0,1].
    * ``min_top_n_overlap`` — top_n_name_overlap must be >=. [0,1].
    * ``max_decomposition_rate_delta_abs`` — abs(decomposition_rate_delta)
      must be <=. [0,1]. Symmetric: both directions of drift fail.
    * ``min_morpheme_rank_correlation`` — morpheme_rank_correlation
      must be >=. [-1, +1]. -1 = inverse ranking OK; 0 = at least
      uncorrelated; +1 = identical ranking required.

    None values disable the check for that metric (operator
    intentionally not gating on it).
    """

    max_kl_divergence: float | None = 10.0
    max_total_variation: float | None = 1.0
    min_top_n_overlap: float | None = 0.0
    max_decomposition_rate_delta_abs: float | None = 1.0
    min_morpheme_rank_correlation: float | None = -1.0


# Default tolerance for every culture not overridden below. Wide-open
# defaults — the regression suite is inactive as a drift gate today.
# Operator tightens via PER_CULTURE_TOLERANCES once Phase 6a evidence
# is in hand.
DEFAULT_TOLERANCE = ToleranceBand()


# Per-culture tolerance overrides. Operator policy lives here.
#
# DRAFT bands (2026-06-01, kenning/scoring-mode-investigation). Sized
# against the proportions-vs-vector drift the regression suite ACTUALLY
# runs — SAMPLES_PER_MODE=100, REGRESSION_BASE_SEED=0, bare request
# (priors are bundled in L4, so the baseline axis is already live; this
# is full-fidelity drift). Bands gate that config, so they're set from
# its (noisy) numbers, NOT the calmer N=1000 report.
#
#   culture   KL@100  TV@100  ρ@100    | KL@1000  ρ@1000   decompΔ
#   english   0.0735  0.0789  0.3455   | 0.0040   0.742    0.0000
#   scottish  0.0175  0.0627  0.2859   | 0.0034   0.710    0.0000
#   irish     0.0429  0.0680  0.4774   | 0.0152   0.742    0.0000
#   breton    0.1600  0.1036  0.4068   | 0.0043   0.872    0.0000
#   welsh     —  blocked by wyrd-cj6f (proportions crash) —
#
# At N=100 KL/ρ are noise-dominated (KL up to 0.16, ρ down to 0.29);
# the N=1000 columns show the real signal is far tighter. RECOMMENDATION:
# bump SAMPLES_PER_MODE to 1000 before activating this as a hard gate,
# then re-tighten toward the N=1000 numbers (e.g. KL<=0.05, ρ>=0.6). The
# bands below carry ~1.5-2x headroom over the N=100 observations so the
# gate is green today without being wide-open.
#
# Banding rationale:
#  * max_kl_divergence / max_total_variation — SEMANTIC-character guard:
#    vector should reshuffle WHICH names surface without shifting the tag
#    mix. Ceilings ~2x the N=100 value.
#  * max_decomposition_rate_delta_abs — QUALITY floor: vector must not
#    erode decomposition coverage. Observed 0.0 (stable across N); tight
#    at 0.02 — the one band worth keeping strict regardless of N.
#  * min_morpheme_rank_correlation — catches a frequency-collapse /
#    near-inversion without flagging the intended reorder. Floor ~=
#    N=100 observed minus ~0.20.
#  * min_top_n_overlap — DELIBERATELY None. The top-N name reshuffle is
#    the *intended* effect of vector scoring; gating on it fights the
#    feature.
#
# Review-then-codify STARTING POINTS, not final policy. NOT yet active as
# a real gate (the suite still skips a culture that yields 0 vector
# samples). welsh is intentionally absent (falls back to wide-open
# DEFAULT) until wyrd-cj6f is fixed and a clean welsh drift run exists.
PER_CULTURE_TOLERANCES: dict[str, ToleranceBand] = {
    "english": ToleranceBand(
        max_kl_divergence=0.15,
        max_total_variation=0.16,
        min_top_n_overlap=None,
        max_decomposition_rate_delta_abs=0.02,
        min_morpheme_rank_correlation=0.15,
    ),
    "scottish": ToleranceBand(
        max_kl_divergence=0.06,
        max_total_variation=0.15,
        min_top_n_overlap=None,
        max_decomposition_rate_delta_abs=0.02,
        min_morpheme_rank_correlation=0.10,
    ),
    "irish": ToleranceBand(
        max_kl_divergence=0.12,
        max_total_variation=0.16,
        min_top_n_overlap=None,
        max_decomposition_rate_delta_abs=0.02,
        min_morpheme_rank_correlation=0.25,
    ),
    "breton": ToleranceBand(
        max_kl_divergence=0.30,
        max_total_variation=0.20,
        min_top_n_overlap=None,
        max_decomposition_rate_delta_abs=0.02,
        min_morpheme_rank_correlation=0.20,
    ),
}


def tolerance_for(culture: str) -> ToleranceBand:
    """Resolve the tolerance band for ``culture``. Returns the
    per-culture override if one exists, otherwise the default."""
    return PER_CULTURE_TOLERANCES.get(culture, DEFAULT_TOLERANCE)


@dataclass(frozen=True)
class ToleranceViolation:
    """One per-metric violation. Surfaces in the regression report so
    the operator can see which metric / which culture failed."""

    culture: str
    metric: str
    value: float
    threshold: float
    direction: str  # "above" (value > max) or "below" (value < min)

    def __str__(self) -> str:
        return (
            f"{self.culture}.{self.metric} = {self.value:.4f} "
            f"is {self.direction} threshold {self.threshold:.4f}"
        )


def check_drift_against_tolerance(
    report,  # DriftReport — type-hint avoided to dodge a load-time cycle
    tolerance: ToleranceBand | None = None,
) -> list[ToleranceViolation]:
    """Compare a drift report's metrics against a tolerance band.

    Returns a list of :class:`ToleranceViolation` for any metric that
    falls outside its band. Empty list = all metrics within tolerance
    (regression test passes).

    None tolerance → resolve from the report's culture via
    :func:`tolerance_for`.
    """
    if tolerance is None:
        tolerance = tolerance_for(report.culture)

    violations: list[ToleranceViolation] = []

    if tolerance.max_kl_divergence is not None and (
        report.tag_kl_divergence > tolerance.max_kl_divergence
    ):
        violations.append(
            ToleranceViolation(
                culture=report.culture,
                metric="tag_kl_divergence",
                value=report.tag_kl_divergence,
                threshold=tolerance.max_kl_divergence,
                direction="above",
            )
        )

    if tolerance.max_total_variation is not None and (
        report.tag_total_variation > tolerance.max_total_variation
    ):
        violations.append(
            ToleranceViolation(
                culture=report.culture,
                metric="tag_total_variation",
                value=report.tag_total_variation,
                threshold=tolerance.max_total_variation,
                direction="above",
            )
        )

    if tolerance.min_top_n_overlap is not None and (
        report.top_n_name_overlap < tolerance.min_top_n_overlap
    ):
        violations.append(
            ToleranceViolation(
                culture=report.culture,
                metric="top_n_name_overlap",
                value=report.top_n_name_overlap,
                threshold=tolerance.min_top_n_overlap,
                direction="below",
            )
        )

    if tolerance.max_decomposition_rate_delta_abs is not None and (
        abs(report.decomposition_rate_delta) > tolerance.max_decomposition_rate_delta_abs
    ):
        violations.append(
            ToleranceViolation(
                culture=report.culture,
                metric="decomposition_rate_delta_abs",
                value=abs(report.decomposition_rate_delta),
                threshold=tolerance.max_decomposition_rate_delta_abs,
                direction="above",
            )
        )

    if tolerance.min_morpheme_rank_correlation is not None and (
        report.morpheme_rank_correlation < tolerance.min_morpheme_rank_correlation
    ):
        violations.append(
            ToleranceViolation(
                culture=report.culture,
                metric="morpheme_rank_correlation",
                value=report.morpheme_rank_correlation,
                threshold=tolerance.min_morpheme_rank_correlation,
                direction="below",
            )
        )

    return violations


def format_violations(violations: list[ToleranceViolation]) -> str:
    """Render a list of violations as a multi-line operator message.
    Empty list → ``"all metrics within tolerance"``.
    """
    if not violations:
        return "all metrics within tolerance"
    lines = ["Tolerance violations:"]
    lines.extend(f"  - {v}" for v in violations)
    return "\n".join(lines)
