"""Absolute corpus-realism tolerance bands (wyrd-jfaz).

Codifies the operator-decided tolerance per metric × culture for the
absolute realism gate. The test suite in
``tests/test_kenning_realism_absolute.py`` reads these tolerances +
runs realism measurements (vector-mode samples vs the corpus
reference) against the live generators; tolerance violations fail the
test.

The bands here gate the ABSOLUTE realism of vector-mode output against
a corpus reference derived from bundle data (``realism_reference.
CorpusReference``), which survives independently of any scoring
baseline and tracks corpus growth.

Two layers of policy:

* ``ABSOLUTE_DEFAULT`` — the draft :class:`AbsoluteToleranceBand`
  applied to every culture without an override.
* ``ABSOLUTE_PER_CULTURE`` — per-culture overrides (none today; the
  draft default holds for all 5 cultures on both bundles).
"""

from __future__ import annotations

from dataclasses import dataclass


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


def format_violations(violations: list[ToleranceViolation]) -> str:  # noqa: V103 — violation renderer used by both realism gates
    """Render a list of violations as a multi-line operator message.
    Empty list → ``"all metrics within tolerance"``.
    """
    if not violations:
        return "all metrics within tolerance"
    lines = ["Tolerance violations:"]
    lines.extend(f"  - {v}" for v in violations)
    return "\n".join(lines)


# ===== ABSOLUTE corpus-realism bands (wyrd-jfaz) =========================
#
# The bands above gate the RELATIVE drift of vector vs the proportions
# scoring baseline. Once proportions scoring retires (epic wyrd-ej28) that
# baseline is gone, so the gate re-anchors to ABSOLUTE bands: vector-mode
# output measured against the corpus reference (realism_reference.
# CorpusReference, derived from bundle data — survives the deletion and
# tracks corpus growth). These bands live alongside the relative ones
# during the transition; the relative gate is removed in Phase 3.
#
# DRAFT bands (2026-06-01), review-then-tighten. Baselined from the
# vector-vs-reference metrics at N=1000, seed 0, across all 5 cultures on
# the committed --dev bundle (the CI gate); confirmed to also hold on the
# full production bundle. The bands sit ~2-4x over the worst observed value
# so the gate absorbs N-noise + bundle differences; tighten toward the
# observed envelope once the cutover is validated.
#
#   metric          worst (dev, N=1000)    band      headroom
#   tag_kl          0.0139                 0.05      ~3.6x
#   tag_tv          0.0640                 0.15      ~2.3x
#   position_tv     0.0514                 0.15      ~2.9x
#   morpheme_rho    0.5085                 0.30      floor, ~0.21 slack
#   decomposition   1.000  / 1.000        0.95      floor (vector must decompose)


@dataclass(frozen=True)
class AbsoluteToleranceBand:
    """Absolute realism band for vector-vs-corpus-reference metrics
    (wyrd-jfaz). ``None`` disables a check.

    * ``max_tag_kl_divergence`` / ``max_tag_total_variation`` — the
      sample tag distribution must not diverge from the corpus tag
      distribution beyond these.
    * ``max_position_total_variation`` — same for the slot-position
      distribution.
    * ``min_morpheme_rank_correlation`` — the sample's top-K morpheme
      ranks must correlate with the corpus's expected ranks at least
      this much (catches a frequency-collapse / over-concentration).
    * ``min_decomposition_rate`` — vector names must fully decompose at
      least this often (absolute quality floor).
    """

    max_tag_kl_divergence: float | None = 0.05
    max_tag_total_variation: float | None = 0.15
    max_position_total_variation: float | None = 0.15
    min_morpheme_rank_correlation: float | None = 0.30
    min_decomposition_rate: float | None = 0.95


ABSOLUTE_DEFAULT = AbsoluteToleranceBand()

# Per-culture overrides — none today (the draft default holds for all 5
# cultures on both bundles). Add entries here as the bands tighten and a
# culture needs more slack than the default.
ABSOLUTE_PER_CULTURE: dict[str, AbsoluteToleranceBand] = {}


def absolute_tolerance_for(culture: str) -> AbsoluteToleranceBand:
    """Resolve the absolute realism band for ``culture`` (override if
    present, else the draft default)."""
    return ABSOLUTE_PER_CULTURE.get(culture, ABSOLUTE_DEFAULT)


def check_realism_against_tolerance(  # noqa: V103 — absolute-band checker for the realism CI gate
    report,  # RealismReport — type-hint avoided to dodge a load-time cycle
    tolerance: AbsoluteToleranceBand | None = None,
) -> list[ToleranceViolation]:
    """Compare an absolute :class:`RealismReport`'s metrics against an
    :class:`AbsoluteToleranceBand`. Returns a list of
    :class:`ToleranceViolation` (empty = all within tolerance). None
    tolerance → resolve from the report's culture."""
    if tolerance is None:
        tolerance = absolute_tolerance_for(report.culture)

    violations: list[ToleranceViolation] = []

    def _max(metric: str, value: float, threshold: float | None) -> None:
        if threshold is not None and value > threshold:
            violations.append(
                ToleranceViolation(
                    culture=report.culture,
                    metric=metric,
                    value=value,
                    threshold=threshold,
                    direction="above",
                )
            )

    def _min(metric: str, value: float, threshold: float | None) -> None:
        if threshold is not None and value < threshold:
            violations.append(
                ToleranceViolation(
                    culture=report.culture,
                    metric=metric,
                    value=value,
                    threshold=threshold,
                    direction="below",
                )
            )

    _max("tag_kl_divergence", report.tag_kl_divergence, tolerance.max_tag_kl_divergence)
    _max("tag_total_variation", report.tag_total_variation, tolerance.max_tag_total_variation)
    _max(
        "position_total_variation",
        report.position_total_variation,
        tolerance.max_position_total_variation,
    )
    _min(
        "morpheme_rank_correlation",
        report.morpheme_rank_correlation,
        tolerance.min_morpheme_rank_correlation,
    )
    _min("decomposition_rate", report.decomposition_rate, tolerance.min_decomposition_rate)

    return violations
