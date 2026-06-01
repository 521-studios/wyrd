"""Absolute corpus-realism gate (wyrd-jfaz).

Re-anchors the realism regression from a RELATIVE vector-vs-proportions
drift comparison to an ABSOLUTE vector-vs-corpus-reference one. The
reference (``realism_reference.CorpusReference``) is derived from the
bundle data, so this gate survives the proportions scoring-mode deletion
(epic wyrd-ej28) and tracks corpus growth — unlike the relative gate,
which needs proportions samples to compare against.

Runs at N=1000 (vs the relative gate's N=100) per the wyrd-jfaz note that
N=100 is noise-dominated; the absolute gate only generates the vector
side, so it's ~half the work of the relative gate at the same N.
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


def test_corpus_reference_proportions_cross_check():
    """Correctness proof for the analytical reference (TRANSITION-ONLY —
    depends on the proportions scoring path; remove in Phase 3 with the
    relative gate).

    The proportions sampler draws structs by proportion and usages by
    frequency — exactly the distribution ``compute_corpus_reference``
    derives analytically. So proportions samples must land at ~0
    divergence against the reference. If THIS fails, the reference
    derivation is wrong (not the generator), which would silently
    mis-calibrate the absolute bands.
    """
    from wyrd.generators.kenning import _load_culture
    from wyrd.generators.kenning.runtime.drift_runner import run_drift_samples
    from wyrd.generators.kenning.runtime.realism_reference import compute_corpus_reference

    proportions_samples, _vector = run_drift_samples(culture="english", count=2000, base_seed=0)
    name_gen, _ = _load_culture("english")
    reference = compute_corpus_reference("english", name_gen)
    report = compute_realism_report("english", proportions_samples, reference)

    # Tag distribution must match almost exactly (the corpus sampler IS
    # the reference's generative model). The small residuals on position /
    # morpheme-rank are N=2000 sampling noise + the adjacent-dup dedup, not
    # reference error — kept loose enough to be non-flaky but tight enough
    # to catch a genuinely wrong derivation.
    assert report.tag_kl_divergence < 0.02, report.tag_kl_divergence
    assert report.tag_total_variation < 0.06, report.tag_total_variation
    assert report.position_total_variation < 0.12, report.position_total_variation
    assert report.morpheme_rank_correlation > 0.60, report.morpheme_rank_correlation


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
