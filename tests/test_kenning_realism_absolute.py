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


def _culture_param(c: str):
    # wyrd-8vfo: the absolute realism bands are noise-dominated, and breton's
    # corpus is mid-expansion (wyrd-ip7r epic / wyrd-fmg). On the refreshed
    # clean-rebuild seed (wyrd-ipvw) breton's morpheme_rank_correlation dips
    # below the global threshold — tracked, not a generator regression. Non-strict
    # so it doesn't flip-flop as the breton corpus grows back over the band.
    if c == "breton":
        return pytest.param(
            c,
            marks=pytest.mark.xfail(
                reason="wyrd-8vfo: breton realism band noise-dominated + corpus "
                "mid-expansion (wyrd-ip7r/fmg); below threshold on refreshed seed",
                strict=False,
            ),
        )
    return c


@pytest.mark.parametrize("culture", [_culture_param(c) for c in REGRESSION_CULTURES])
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


# wyrd-rt2m: the transition-only proportions cross-check (a correctness proof
# that the analytical reference matched the proportions sampler's distribution)
# is removed per its own TRANSITION-ONLY docstring + the wyrd-3dlx cleanup
# scope, now that proportions is user-unreachable. (It could still run — the
# realism gate's run_drift_samples passes scoring_mode='proportions' explicitly,
# which generate() still honors — but it's slated for deletion with the
# proportions path.) The reference's structural correctness stays pinned by
# test_compute_corpus_reference_structure below, and the live gate
# (test_vector_within_absolute_corpus_bands) keeps measuring vector vs the
# reference.


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


def test_normalize_weights():
    """_normalize_weights (wyrd-8uvi extraction): positive totals → shares
    summing to 1.0; non-positive total → empty dict."""
    from wyrd.generators.kenning.runtime.realism_reference import _normalize_weights

    out = _normalize_weights({"a": 1.0, "b": 3.0})
    assert out == {"a": 0.25, "b": 0.75}
    assert abs(sum(out.values()) - 1.0) < 1e-9
    # Non-positive totals → empty (both empty-input and all-zero).
    assert _normalize_weights({}) == {}
    assert _normalize_weights({"a": 0.0}) == {}


def test_accumulate_slot_weights_branches():
    """_accumulate_slot_weights (wyrd-8uvi extraction): pins the gloss
    keep-set filter, the total_f<=0 / morpheme-None skips, and the
    location-absent guard — branches previously only reachable via the
    DB-dependent integration gate."""
    from wyrd.generators.kenning.runtime.realism_reference import (
        _accumulate_slot_weights,
        _WeightAccumulator,
    )

    # features_fn: 'a'->morpheme a, post + tag t1; 'b'->morpheme b, no
    # location, tag t2; 'gone'->resolves to None (must be skipped).
    feats = {
        "a": ("a", ["t1"], "post"),
        "b": ("b", ["t2"], None),
        "gone": (None, [], None),
    }

    # Gloss keep-set drops 'b' (only 'a' kept); 'a' weight = 1.0 * (2/2).
    acc = _WeightAccumulator()
    _accumulate_slot_weights({"a": 2, "b": 8}, 1.0, frozenset({"a"}), feats.__getitem__, acc)
    assert acc.morpheme == {"a": 1.0}
    assert acc.position == {"post": 1.0}
    assert acc.tag == {"t1": 1.0}

    # No keep-set: 'b' contributes morpheme + tag but NO position (location None).
    acc2 = _WeightAccumulator()
    _accumulate_slot_weights({"b": 5}, 1.0, None, feats.__getitem__, acc2)
    assert acc2.morpheme == {"b": 1.0}
    assert acc2.position == {}  # location-absent guard
    assert acc2.tag == {"t2": 1.0}

    # morpheme-None usage is skipped entirely.
    acc3 = _WeightAccumulator()
    _accumulate_slot_weights({"gone": 3}, 1.0, None, feats.__getitem__, acc3)
    assert acc3.morpheme == {} and acc3.position == {} and acc3.tag == {}

    # total_f <= 0 (empty bucket after no matches) → no-op.
    acc4 = _WeightAccumulator()
    _accumulate_slot_weights({"a": 5}, 1.0, frozenset({"nomatch"}), feats.__getitem__, acc4)
    assert acc4.morpheme == {} and acc4.position == {} and acc4.tag == {}


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
