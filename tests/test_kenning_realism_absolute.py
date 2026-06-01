"""Absolute corpus-realism gate (wyrd-jfaz).

Re-anchors the realism regression from a RELATIVE vector-vs-proportions
drift comparison to an ABSOLUTE vector-vs-corpus-reference one. The
reference (``realism_reference.CorpusReference``) is derived from the
bundle data, so this gate survives the proportions scoring-mode deletion
(epic wyrd-ej28) and tracks corpus growth — unlike the relative gate,
which needs proportions samples to compare against.

Runs at N=1000 (vs the relative gate's N=100): at N=100 the
vector-vs-reference signal is noise-dominated for the absolute metrics
(wyrd-jfaz). The absolute gate only generates the vector side, so it's
~half the work of the relative gate at the same N.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.runtime.drift_measurement import compute_realism_report
from wyrd.generators.kenning.runtime.drift_runner import run_realism_samples
from wyrd.generators.kenning.runtime.realism_tolerance import (
    check_realism_against_tolerance,
    format_violations,
)

REGRESSION_CULTURES = ["english", "scottish", "welsh", "irish", "breton"]
SAMPLES = 1000
BASE_SEED = 0


@pytest.mark.parametrize("culture", REGRESSION_CULTURES)
def test_vector_within_absolute_corpus_bands(culture: str):
    """Vector-mode output must stay within the absolute corpus-realism
    bands for every culture. The reference is computed from the same
    bundle the samples are generated from, so this measures vector's
    fidelity to the corpus's tag / position / morpheme-frequency shape
    — no proportions baseline involved.
    """
    samples, reference = run_realism_samples(culture=culture, count=SAMPLES, base_seed=BASE_SEED)
    assert samples, f"{culture}: vector produced 0 samples — generator regressed"

    report = compute_realism_report(culture, samples, reference)
    violations = check_realism_against_tolerance(report)
    assert not violations, format_violations(violations)


@pytest.mark.parametrize("culture", REGRESSION_CULTURES)
def test_corpus_reference_proportions_cross_check(culture: str):
    """Correctness proof for the analytical reference (TRANSITION-ONLY —
    depends on the proportions scoring path; remove in Phase 3 with the
    relative gate).

    The proportions sampler draws structs by proportion and usages by
    frequency, under the SAME ``include_unglossed=False`` gloss policy the
    reference applies — exactly the distribution ``compute_corpus_reference``
    derives analytically. So proportions samples land at ~0 divergence
    against the reference for EVERY culture (incl. breton, which has the
    largest gloss-ineligible share — the gloss filter is what closes that
    gap). If THIS fails, the reference derivation is wrong (not the
    generator), which would silently mis-calibrate the absolute bands.
    """
    from wyrd.generators.kenning import _load_culture
    from wyrd.generators.kenning.runtime.drift_runner import run_drift_samples
    from wyrd.generators.kenning.runtime.realism_reference import compute_corpus_reference

    proportions_samples, _vector = run_drift_samples(culture=culture, count=2000, base_seed=0)
    name_gen, _ = _load_culture(culture)
    reference = compute_corpus_reference(culture, name_gen, include_unglossed=False)
    report = compute_realism_report(culture, proportions_samples, reference)

    # The corpus sampler IS the reference's generative model (post gloss-fix),
    # so divergence is ~0. The small residuals are N=2000 sampling noise + the
    # adjacent-dup dedup, NOT reference error. Observed worst across all 5
    # cultures: tag_kl 0.0012, tag_tv 0.019, pos_tv 0.014, morph_rho 0.895 —
    # thresholds sit just outside that with non-flaky headroom while still
    # catching a genuinely wrong derivation (which shows TV >> 0.1).
    assert report.tag_kl_divergence < 0.01, (culture, report.tag_kl_divergence)
    assert report.tag_total_variation < 0.04, (culture, report.tag_total_variation)
    assert report.position_total_variation < 0.05, (culture, report.position_total_variation)
    assert report.morpheme_rank_correlation > 0.80, (culture, report.morpheme_rank_correlation)


def test_compute_corpus_reference_structure():
    """Unit-level sanity for the reference derivation: distributions are
    normalized, and a (struct, slot, usage) maps to the expected tag /
    position / morpheme. Permanent (no proportions dependency)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator, NameGenerator
    from wyrd.generators.kenning.runtime.realism_reference import compute_corpus_reference

    db = {
        "Bre-": [Meaning("Bre-", ["topographic"], ["hill"], {})],
        "-ton": [Meaning("-ton", ["settlement"], ["town"], {})],
    }
    # part pool seeds the pre/post buckets the (pre, post) struct references.
    proportions = dict.fromkeys(db, 1)
    mg = MeaningGenerator(db, {}, proportions)
    structs = {((("pre",), ("post",)),): 1}
    name_gen = NameGenerator(db, mg, structs)

    ref = compute_corpus_reference("test", name_gen)

    # Distributions normalize to 1.0 (within float tolerance).
    assert abs(sum(ref.tag_distribution.values()) - 1.0) < 1e-9
    assert abs(sum(ref.position_distribution.values()) - 1.0) < 1e-9
    # The two morphemes' tags + positions are present.
    assert set(ref.tag_distribution) == {"topographic", "settlement"}
    assert set(ref.position_distribution) == {"pre", "post"}
    # Morpheme weights are dash-stripped surfaces (case preserved).
    assert set(ref.morpheme_counts) == {"Bre", "ton"}


# ---- check_realism_against_tolerance: violation logic (wyrd-jfaz) ---------
# The gate test above only ever exercises the PASS path (assert no violations).
# These pin the violation branches directly so an inverted comparison or a
# dropped metric check can't ship green. No bundle needed — hand-built reports.


def _band(**kw):
    from wyrd.generators.kenning.runtime.realism_tolerance import AbsoluteToleranceBand

    return AbsoluteToleranceBand(**kw)


def _report(**kw):
    from wyrd.generators.kenning.runtime.drift_measurement import RealismReport

    # Default to PASSING metric values so only the metric a test overrides
    # can violate (a bare RealismReport has decomposition_rate=0.0 /
    # morpheme_rank_correlation=0.0, which would trip the floor checks too).
    base = {
        "tag_kl_divergence": 0.0,
        "tag_total_variation": 0.0,
        "position_total_variation": 0.0,
        "morpheme_rank_correlation": 1.0,
        "decomposition_rate": 1.0,
    }
    base.update(kw)
    return RealismReport(culture="x", sample_size=1, **base)


def test_check_realism_flags_kl_above_band():
    from wyrd.generators.kenning.runtime.realism_tolerance import check_realism_against_tolerance

    v = check_realism_against_tolerance(
        _report(tag_kl_divergence=0.9), _band(max_tag_kl_divergence=0.05)
    )
    assert [x.metric for x in v] == ["tag_kl_divergence"]
    assert v[0].direction == "above"


def test_check_realism_flags_morpheme_rank_below_floor():
    from wyrd.generators.kenning.runtime.realism_tolerance import check_realism_against_tolerance

    v = check_realism_against_tolerance(
        _report(morpheme_rank_correlation=0.1), _band(min_morpheme_rank_correlation=0.30)
    )
    assert [x.metric for x in v] == ["morpheme_rank_correlation"]
    assert v[0].direction == "below"


def test_check_realism_flags_decomposition_below_floor():
    from wyrd.generators.kenning.runtime.realism_tolerance import check_realism_against_tolerance

    v = check_realism_against_tolerance(
        _report(decomposition_rate=0.5), _band(min_decomposition_rate=0.95)
    )
    assert [x.metric for x in v] == ["decomposition_rate"]
    assert v[0].direction == "below"


def test_check_realism_none_threshold_disables_check():
    from wyrd.generators.kenning.runtime.realism_tolerance import check_realism_against_tolerance

    # A wildly out-of-band value with the check disabled → no violation.
    v = check_realism_against_tolerance(
        _report(tag_kl_divergence=99.0), _band(max_tag_kl_divergence=None)
    )
    assert v == []


def test_check_realism_resolves_default_band_from_culture():
    # tolerance=None → absolute_tolerance_for(culture); ABSOLUTE_DEFAULT floors
    # morpheme_rank_correlation at 0.30, so -0.5 violates.
    from wyrd.generators.kenning.runtime.drift_measurement import RealismReport
    from wyrd.generators.kenning.runtime.realism_tolerance import check_realism_against_tolerance

    v = check_realism_against_tolerance(
        RealismReport(culture="english", sample_size=1, morpheme_rank_correlation=-0.5)
    )
    assert any(x.metric == "morpheme_rank_correlation" for x in v)
