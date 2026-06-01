"""Realism-retention regression test suite (wyrd-ecjp.7 Phase 6b).

CI-level regression: generates N names per scoring mode per culture
and asserts the drift report falls within the operator-decided
tolerance bands in :mod:`runtime.realism_tolerance`.

The suite is a LIVE gate for cultures with a populated band in
:mod:`runtime.realism_tolerance` (english / scottish / irish /
breton) that produce non-zero vector samples: a drift report
exceeding the band fails the test. A culture with no per-culture
band falls back to the wide-open ``DEFAULT_TOLERANCE`` and is
effectively un-gated. welsh is parametrized but currently hits the
0-sample skip (pending wyrd-cj6f).

Per-test cost: each regression test generates 100 names per scoring
mode (200 total per culture). Reduced from the operator-facing
default of 1000 to keep CI runtime reasonable; the tolerance bands
should remain meaningful at N=100 since metrics stabilize quickly
for distribution-level signals.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.runtime.drift_measurement import compute_drift_report
from wyrd.generators.kenning.runtime.drift_runner import run_drift_samples
from wyrd.generators.kenning.runtime.realism_tolerance import (
    check_drift_against_tolerance,
    format_violations,
)

# Cultures to regression-test — the full kenning set. english / scottish /
# irish / breton have populated bands in PER_CULTURE_TOLERANCES and are
# live-gated; welsh is parametrized too but hits the 0-sample skip until
# wyrd-cj6f (proportions crash) is fixed. The test parametrizes across
# these; each culture independently checks tolerance compliance.
REGRESSION_CULTURES = ("english", "scottish", "welsh", "irish", "breton")

# Per-test sample size. Smaller than the operator-facing default
# (1000) to keep CI fast. Distribution-level metrics stabilize at
# N=100 for the gate-check use case.
SAMPLES_PER_MODE = 100

# Seed base. Same base seed across runs → byte-identical drift
# measurements (bit-stable regression contract).
REGRESSION_BASE_SEED = 0


@pytest.mark.parametrize("culture", REGRESSION_CULTURES)
def test_realism_regression_per_culture(culture: str):
    """Per-culture drift between legacy proportions and vector
    scoring must fall within the tolerance band.

    Cultures with a populated band in :data:`PER_CULTURE_TOLERANCES`
    (english / scottish / irish / breton) are gated: a drift report
    exceeding the band fails this test. A culture with no band falls
    back to the wide-open :data:`DEFAULT_TOLERANCE` and effectively
    passes regardless of drift.

    Per-seed isolation: ``run_drift_samples`` skips seeds that raise
    in either mode (proportions shouldn't raise; vector raises
    ValueError on empty pick). A culture whose vector path produces
    zero samples on a side is ``pytest.skip``ped below (the drift
    comparison against an empty distribution is meaningless) — it is
    NOT silently passed through the bands. welsh is the live example,
    pending wyrd-cj6f.
    """
    samples_a, samples_b = run_drift_samples(
        culture=culture,
        count=SAMPLES_PER_MODE,
        base_seed=REGRESSION_BASE_SEED,
    )

    # When one side has no samples, the drift comparison is
    # meaningless — KL against an empty distribution diverges, but
    # that's a data-presence question, not a drift signal. Skip the
    # tolerance check in that case rather than fail. Today this fires
    # only for welsh, whose proportions path crashes (wyrd-cj6f); the
    # other parametrized cultures produce samples and are gated.
    if not samples_a or not samples_b:
        pytest.skip(
            f"{culture}: one side produced 0 samples "
            f"(a={len(samples_a)}, b={len(samples_b)}) — "
            "drift comparison meaningless (expected for welsh until "
            "wyrd-cj6f is fixed)."
        )

    report = compute_drift_report(culture, samples_a, samples_b)
    violations = check_drift_against_tolerance(report)

    assert not violations, format_violations(violations)


# ---- smoke tests for new vector-scoring behaviors ------------------------


def test_register_phonological_bias_scores_lemmas_consistently():
    """Smoke: a RequestVector with a harsh-leaning register produces
    consistent score-direction across multiple invocations. Verifies
    the score → register-effect wiring doesn't silently no-op.

    Uses the vector_name_select primitive directly (not Kenning.generate)
    so the test isolates the register-bias contract from the dispatch
    layer + bundle dependencies."""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.vector_name_select import (
        select_via_vector_scoring,
    )
    from wyrd.generators.kenning.vectors.schemas import (
        EligibilityGate,
        EmpiricalPriors,
        PhonologicalVector,
        RegisterEffect,
        RequestVector,
        ScoringWeights,
    )

    # 3 post-position meanings with varying cluster_density. A
    # harsh-leaning register weights cluster_density positively, so
    # the highest-cluster_density meaning should score highest.
    harsh = Meaning(
        usage="-strikt",
        tags=["x"],
        meanings=[],
        sources=[],
        phonological_vector=PhonologicalVector(cluster_density=0.9),
    )
    medium = Meaning(
        usage="-mid",
        tags=["x"],
        meanings=[],
        sources=[],
        phonological_vector=PhonologicalVector(cluster_density=0.5),
    )
    soft = Meaning(
        usage="-aloha",
        tags=["x"],
        meanings=[],
        sources=[],
        phonological_vector=PhonologicalVector(cluster_density=0.1),
    )
    db = {"-strikt": [harsh], "-mid": [medium], "-aloha": [soft]}

    request = RequestVector(
        gate=EligibilityGate(culture="english"),
        register=RegisterEffect(
            name="harsh-test",
            phonological={"cluster_density": 1.0},
            semantic_tags={"x": 0.1},
        ),
        weights=ScoringWeights(phon_w=10.0, sem_w=1.0, pos_w=0.0, base_w=0.0),
    )

    # Run many seeds; harsh should be picked more often than soft
    # under the cluster_density-biased register.
    harsh_wins = 0
    soft_wins = 0
    for seed in range(100):
        picks = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["post"],
            request=request,
            priors=EmpiricalPriors(),
        )
        if not picks:
            continue
        if picks[0] is harsh:
            harsh_wins += 1
        elif picks[0] is soft:
            soft_wins += 1
    # Strong bias direction — harsh should win materially more often
    assert harsh_wins > soft_wins, (
        f"register phonological bias not observable: harsh={harsh_wins}, soft={soft_wins}"
    )


def test_register_semantic_bias_scores_matching_tags_higher():
    """Smoke: a RequestVector with a tag-weighted register prefers
    meanings carrying matching tags. Independent of the phon axis."""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.vector_name_select import (
        select_via_vector_scoring,
    )
    from wyrd.generators.kenning.vectors.schemas import (
        EligibilityGate,
        EmpiricalPriors,
        PhonologicalVector,
        RegisterEffect,
        RequestVector,
        ScoringWeights,
    )

    matching = Meaning(
        usage="-keep",
        tags=["fortified"],
        meanings=[],
        sources=[],
        phonological_vector=PhonologicalVector(),
    )
    mismatching = Meaning(
        usage="-glen",
        tags=["pastoral"],
        meanings=[],
        sources=[],
        phonological_vector=PhonologicalVector(),
    )
    db = {"-keep": [matching], "-glen": [mismatching]}

    request = RequestVector(
        gate=EligibilityGate(culture="english"),
        register=RegisterEffect(
            name="war",
            semantic_tags={"fortified": 1.0, "pastoral": 0.0},
        ),
        weights=ScoringWeights(phon_w=0.0, sem_w=1.0, pos_w=0.0, base_w=0.0),
    )

    matching_wins = 0
    for seed in range(100):
        picks = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["post"],
            request=request,
            priors=EmpiricalPriors(),
        )
        if picks and picks[0] is matching:
            matching_wins += 1
    # With pastoral weight 0, mismatching should NEVER win.
    assert matching_wins == 100, f"sem axis miswired: matching_wins={matching_wins}/100"


def test_register_composition_combines_phonological_and_semantic():
    """Smoke: register-effect composition (phon + sem) produces the
    expected combined bias — the score function sums per-axis
    contributions per the D36.2 canonical formula."""
    from wyrd.generators.kenning.vectors.schemas import (
        PhonologicalVector,
        RegisterEffect,
    )
    from wyrd.generators.kenning.vectors.scoring import phon_score, sem_score

    register = RegisterEffect(
        name="composed",
        phonological={"cluster_density": 1.0},
        semantic_tags={"animal": 0.5, "water": 0.5},
    )
    vector = PhonologicalVector(cluster_density=0.7)
    # phon_score = dot(vector, register.phonological) = 0.7
    assert abs(phon_score(vector, register) - 0.7) < 1e-9
    # sem_score = sum of weights for lemma tags in request
    assert abs(sem_score(["animal", "water"], register) - 1.0) < 1e-9
    # Tags the request doesn't weight contribute 0
    assert abs(sem_score(["rock"], register) - 0.0) < 1e-9


# ---- pack overlay smoke (currently degenerate — packs not wired) ---------


def test_pack_overlay_smoke_currently_skipped():
    """Pack overlays (PackOverlay class) exist as schema today but
    aren't wired into vector_name_select's gate or scoring (deferred
    to wyrd-ecjp.8 / Phase 7 / Phase 7's bundle changes). This test
    is the placeholder so the regression suite has a row for pack
    behavior; it skip-passes today and turns into a real assertion
    once packs ship."""
    pytest.skip("packs not yet wired into vector_name_select (wyrd-ecjp.8)")
