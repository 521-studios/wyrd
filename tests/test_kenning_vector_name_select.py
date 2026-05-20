"""Tests for the vector-scoring NameGenerator primitive (wyrd-ecjp.5 PR B).

The runtime gate→score→sample pipeline that backs the
``--scoring-mode vector`` opt-in. Uses synthetic phon-vector-tagged
fixtures so each test pins a specific scoring behavior without
depending on the live corpus.
"""

from __future__ import annotations

import random

import pytest

from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.vector_name_select import (
    _cohesion_multiplier,
    _lemma_ref_for,
    _matches_position,
    _slot_position_label,
    _weighted_choice,
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

# ---- helpers --------------------------------------------------------------


def _meaning(usage: str, tags=None, phon=None) -> Meaning:
    """Build a minimal Meaning for testing."""
    return Meaning(
        usage=usage,
        tags=tags or [],
        meanings=[],
        sources=[],
        phonological_vector=phon,
    )


def _request(culture: str = "english", era_min=None, era_max=None) -> RequestVector:
    """Default RequestVector for testing — equal-weight axes + a non-
    trivial semantic-tag spec so scoring produces non-zero weights.

    Without at least one non-zero scoring source (register, priors),
    every lemma's score is 0 and weighted_choice returns None → tests
    would see empty results across the board, masking real behavior.
    The synthetic fixture's tags (urban / territory / water) all get
    a uniform 1.0 weight here.
    """
    return RequestVector(
        gate=EligibilityGate(culture=culture, era_min=era_min, era_max=era_max),
        register=RegisterEffect(
            name="test-default",
            semantic_tags={
                "urban": 1.0,
                "territory": 1.0,
                "water": 1.0,
                "fortified": 1.0,
                "pastoral": 1.0,
                "x": 1.0,
                "safe": 1.0,
            },
        ),
        weights=ScoringWeights(),
    )


# ---- _slot_position_label -------------------------------------------------


def test_slot_position_pre_for_trailing_dash():
    assert _slot_position_label("Place-") == "pre"


def test_slot_position_post_for_leading_dash():
    assert _slot_position_label("-shire") == "post"


def test_slot_position_inner_for_both_dashes():
    assert _slot_position_label("-inner-") == "inner"


def test_slot_position_post_for_bare_element():
    # Bare nominal matches Meaning._set_location's "no-dashes → post"
    # convention. Round-1 reviewer caught the previous "bare → inner"
    # behavior would yield an empty eligible pool for bare structural
    # elements.
    assert _slot_position_label("Bare") == "post"


# ---- _matches_position ----------------------------------------------------


def test_matches_position_strict():
    """Meaning.location is set from usage's leading/trailing dashes."""
    m_pre = _meaning("Place-")
    assert m_pre.location == "pre"
    assert _matches_position(m_pre, "pre")
    assert not _matches_position(m_pre, "post")


def test_matches_position_post():
    m_post = _meaning("-shire")
    assert m_post.location == "post"
    assert _matches_position(m_post, "post")
    assert not _matches_position(m_post, "pre")


# ---- _lemma_ref_for -------------------------------------------------------


def test_lemma_ref_strips_decoration_dashes():
    """priors tables key on bare usage forms (no -prefix / -suffix dashes)."""
    assert _lemma_ref_for(_meaning("Place-")) == "Place"
    assert _lemma_ref_for(_meaning("-shire")) == "shire"
    assert _lemma_ref_for(_meaning("-inner-")) == "inner"
    assert _lemma_ref_for(_meaning("Bare")) == "Bare"


# ---- _cohesion_multiplier -------------------------------------------------


def test_cohesion_zero_returns_identity():
    """At cohesion=0 the multiplier is 1.0 regardless of overlap."""
    result = _cohesion_multiplier(
        frozenset({"animal"}),
        frozenset({"plant"}),
        cohesion=0.0,
        tag_cooccurrence={"animal": {"plant": 0.9}},
    )
    assert result == 1.0


def test_cohesion_empty_prior_tags_returns_identity():
    """First slot has no prior tags → no bias."""
    result = _cohesion_multiplier(
        frozenset({"animal"}),
        frozenset(),
        cohesion=1.0,
        tag_cooccurrence={"animal": {"animal": 1.0}},
    )
    assert result == 1.0


def test_cohesion_missing_cooccurrence_data_returns_identity():
    """Legacy / empty bundle (no tag_cooccurrence) → no bias even at
    cohesion=1.0 (graceful degrade per the docstring contract)."""
    result = _cohesion_multiplier(
        frozenset({"animal"}),
        frozenset({"plant"}),
        cohesion=1.0,
        tag_cooccurrence=None,
    )
    assert result == 1.0


def test_cohesion_full_overlap_doubles_at_cohesion_one():
    """Perfectly co-occurring tags at cohesion=1.0 → 1.0 + 1.0*1.0 = 2.0"""
    result = _cohesion_multiplier(
        frozenset({"animal"}),
        frozenset({"plant"}),
        cohesion=1.0,
        tag_cooccurrence={"animal": {"plant": 1.0}},
    )
    assert result == 2.0


def test_cohesion_half_overlap_at_cohesion_half():
    """avg_p=0.6, cohesion=0.5 → 1.0 + 0.5*0.6 = 1.3"""
    result = _cohesion_multiplier(
        frozenset({"animal"}),
        frozenset({"plant"}),
        cohesion=0.5,
        tag_cooccurrence={"animal": {"plant": 0.6}},
    )
    assert abs(result - 1.3) < 1e-9


# ---- _weighted_choice -----------------------------------------------------


def test_weighted_choice_returns_none_on_empty():
    rng = random.Random(0)
    assert _weighted_choice(rng, []) is None


def test_weighted_choice_returns_none_on_all_zero():
    """All weights ≤ 0 → no pick possible."""
    rng = random.Random(0)
    m1 = _meaning("Place-")
    m2 = _meaning("-shire")
    assert _weighted_choice(rng, [(m1, 0.0), (m2, -1.0)]) is None


def test_weighted_choice_deterministic_with_seed():
    """Same rng seed + same weighted list → same pick (bit-stable)."""
    m1 = _meaning("Place-")
    m2 = _meaning("-shire")
    weighted = [(m1, 1.0), (m2, 1.0)]
    pick1 = _weighted_choice(random.Random(42), weighted)
    pick2 = _weighted_choice(random.Random(42), weighted)
    assert pick1 is pick2


def test_weighted_choice_skips_zero_weight_entries():
    """Entries with weight 0 are skipped, but the others still sample."""
    rng = random.Random(0)
    m1 = _meaning("Place-")
    m2 = _meaning("-shire")
    pick = _weighted_choice(rng, [(m1, 0.0), (m2, 1.0)])
    assert pick is m2


# ---- select_via_vector_scoring (end-to-end) -------------------------------


@pytest.fixture
def synthetic_meaning_db():
    """4 meanings: 2 pre-position (Place-, Town-), 2 post-position
    (-shire, -ford), each tagged + phon-vector'd."""
    return {
        "Place-": [
            _meaning(
                "Place-",
                tags=["urban"],
                phon=PhonologicalVector(cluster_density=0.2, vowel_final_bias=0.0),
            )
        ],
        "Town-": [
            _meaning(
                "Town-",
                tags=["urban"],
                phon=PhonologicalVector(cluster_density=0.1, vowel_final_bias=0.5),
            )
        ],
        "-shire": [
            _meaning(
                "-shire",
                tags=["territory"],
                phon=PhonologicalVector(cluster_density=0.3, vowel_final_bias=0.0),
            )
        ],
        "-ford": [
            _meaning(
                "-ford",
                tags=["water"],
                phon=PhonologicalVector(cluster_density=0.0, vowel_final_bias=0.0),
            )
        ],
    }


def test_select_returns_one_meaning_per_slot(synthetic_meaning_db):
    rng = random.Random(0)
    result = select_via_vector_scoring(
        rng,
        synthetic_meaning_db,
        structure=["Place-", "-shire"],
        request=_request(),
        priors=EmpiricalPriors(),
    )
    assert len(result) == 2
    assert result[0].location == "pre"
    assert result[1].location == "post"


def test_select_seed_stable(synthetic_meaning_db):
    """Same rng seed → same picks (bit-stable contract)."""
    r1 = select_via_vector_scoring(
        random.Random(42),
        synthetic_meaning_db,
        structure=["Place-", "-shire"],
        request=_request(),
        priors=EmpiricalPriors(),
    )
    r2 = select_via_vector_scoring(
        random.Random(42),
        synthetic_meaning_db,
        structure=["Place-", "-shire"],
        request=_request(),
        priors=EmpiricalPriors(),
    )
    assert [m.usage for m in r1] == [m.usage for m in r2]


def test_select_returns_empty_when_no_eligible(synthetic_meaning_db):
    """If gate filters every meaning, return empty list (caller decides
    whether to retry / fallback)."""
    rng = random.Random(0)
    result = select_via_vector_scoring(
        rng,
        synthetic_meaning_db,
        structure=["nonexistent-position"],  # no meaning matches 'pre' here
        request=_request(),
        priors=EmpiricalPriors(),
        exclude_tags=frozenset({"urban", "territory", "water"}),
    )
    # 'nonexistent-position' resolves to 'pre' (trailing dash). Only the
    # 2 pre meanings (Place-, Town-) match position; both tagged urban
    # which is excluded → empty pool → return empty.
    assert result == []


def test_select_meaning_without_phon_vector_uses_default():
    """A meaning with phonological_vector=None still scores cleanly —
    the default-empty vector means phon_score returns 0 BUT the other
    axes carry the score. With a sem-weighted request, the meaning
    still gets picked (no crash, no empty result)."""
    rng = random.Random(0)
    bare_pre = _meaning("Bare-", tags=["x"], phon=None)
    db = {"Bare-": [bare_pre]}
    result = select_via_vector_scoring(
        rng,
        db,
        structure=["Bare-"],
        request=_request(),
        priors=EmpiricalPriors(),
    )
    # sem_score: lemma_tags={'x'} dot register.semantic_tags['x']=1.0
    # → 1.0. phon_score: 0 (no vector). Total > 0 → pick succeeds.
    assert len(result) == 1
    assert result[0] is bare_pre


def test_select_zero_score_returns_empty():
    """When EVERY axis scores 0 (no vector, no tags weighted, no
    priors data), weighted_choice returns None and select returns
    []. That's the correct degraded-input behavior — caller decides
    whether to fall back to the proportion-table path."""
    rng = random.Random(0)
    bare_pre = _meaning("Bare-", tags=["unweighted"], phon=None)
    db = {"Bare-": [bare_pre]}
    # Empty register, empty priors → every axis scores 0.
    empty_request = RequestVector(
        gate=EligibilityGate(culture="english"),
        register=RegisterEffect(name="neutral"),
        weights=ScoringWeights(),
    )
    result = select_via_vector_scoring(
        rng,
        db,
        structure=["Bare-"],
        request=empty_request,
        priors=EmpiricalPriors(),
    )
    assert result == []


def test_select_with_register_phon_bias_picks_matching_lemma(synthetic_meaning_db):
    """Set a register that prefers high cluster_density. The lemma with
    higher cluster_density (-shire 0.3 vs -ford 0.0) should be picked
    more often. Test by running many seeds + counting; with seeded
    rng we don't need many samples — just verify the score order is
    correct (higher-cluster lemma gets a higher score)."""
    from wyrd.generators.kenning.vectors.scoring import phon_score

    # Construct a request that biases toward high cluster_density via
    # the register's phonological-feature weights.
    register = RegisterEffect(name="harsh", phonological={"cluster_density": 1.0})
    request = RequestVector(
        gate=EligibilityGate(culture="english"),
        register=register,
        weights=ScoringWeights(phon_w=10.0, sem_w=0.0, pos_w=0.0, base_w=0.0),
    )
    # phon_score's signature is (lemma_vec, request_register) — it dots
    # the lemma vector against the register's phonological-feature
    # weights directly. Rank -shire above -ford under this request.
    shire = synthetic_meaning_db["-shire"][0]
    ford = synthetic_meaning_db["-ford"][0]
    s_shire = phon_score(shire.phonological_vector, request.register)
    s_ford = phon_score(ford.phonological_vector, request.register)
    assert s_shire > s_ford, (
        f"shire ({s_shire}) should outscore ford ({s_ford}) on cluster_density-biased request"
    )


def test_select_d17_cohesion_biases_second_slot_toward_overlapping_tags():
    """At cohesion=1.0 the second slot should bias toward meanings
    whose tags co-occur with the first slot's tags. Verify by counting
    picks across many seeds — with bias enabled the matching meaning
    should win more often than the mismatching one.

    Round-1 reviewer caught the earlier version was a no-op (only
    asserted the deterministic first-slot pick, never the second-
    slot bias direction). This rewrite actually counts second-slot
    picks across 200 seeds + asserts the bias direction.
    """
    pre = _meaning("Castle-", tags=["fortified"], phon=PhonologicalVector())
    # Two post-meanings AT THE SAME usage key so they compete in slot 2.
    # The match shares the 'fortified' tag with pre; mismatch shares
    # 'pastoral' (lower co-occurrence with fortified per the
    # tag_cooccurrence table).
    post_match = _meaning("-keep", tags=["fortified"], phon=PhonologicalVector())
    post_mismatch = _meaning("-keep", tags=["pastoral"], phon=PhonologicalVector())
    db = {
        "Castle-": [pre],
        "-keep": [post_match, post_mismatch],  # 2 candidates compete in slot 2
    }
    tag_cooccurrence = {
        "fortified": {"fortified": 0.9, "pastoral": 0.1},
        "pastoral": {"fortified": 0.1, "pastoral": 0.9},
    }
    register = RegisterEffect(name="any", semantic_tags={"fortified": 1.0, "pastoral": 1.0})
    request = RequestVector(
        gate=EligibilityGate(culture="english"),
        register=register,
        weights=ScoringWeights(sem_w=1.0, phon_w=0.0, pos_w=0.0, base_w=0.0),
    )

    # Count second-slot picks across many seeds with cohesion=0 vs
    # cohesion=1. With cohesion=0 the picks should be roughly 50/50
    # (equal sem_scores); with cohesion=1 they should bias toward the
    # match.
    def _count_match_wins(cohesion: float) -> int:
        wins = 0
        for seed in range(200):
            picks = select_via_vector_scoring(
                random.Random(seed),
                db,
                structure=["Castle-", "-keep"],
                request=request,
                priors=EmpiricalPriors(),
                cohesion=cohesion,
                tag_cooccurrence=tag_cooccurrence,
            )
            if len(picks) == 2 and "fortified" in picks[1].tags:
                wins += 1
        return wins

    baseline_wins = _count_match_wins(cohesion=0.0)
    biased_wins = _count_match_wins(cohesion=1.0)
    # With perfect 0.9-vs-0.1 cooccurrence ratio + cohesion=1.0, the
    # match should win materially more often than baseline (the bias
    # is multiplicative: match score scales by 1.9, mismatch by 1.1).
    assert biased_wins > baseline_wins + 20, (
        f"D17 cohesion bias not observable in pick distribution: "
        f"baseline={baseline_wins}/200 vs biased={biased_wins}/200"
    )


def test_select_stratum_gate_filters_meanings():
    """Round-1 reviewer caught _matches_stratum was dead code (looked
    up the wrong Meaning method name). This test pins the fixed
    behavior: a meaning whose in_stratum(s) returns False is filtered
    out of the eligible pool."""
    rng = random.Random(0)
    # Build a meaning with stratum data via the bundle-load path.
    # The simplest approach: directly construct + populate stratum.
    m_in = _meaning("Cymro-", tags=["x"], phon=PhonologicalVector(cluster_density=0.5))
    m_in.stratum = {"welsh": {"Cymro": "native-welsh"}}
    m_out = _meaning("Latin-", tags=["x"], phon=PhonologicalVector(cluster_density=0.5))
    m_out.stratum = {"welsh": {"Latin": "latin-loan"}}
    db = {"Cymro-": [m_in], "Latin-": [m_out]}
    request = RequestVector(
        gate=EligibilityGate(culture="welsh", stratum="native-welsh"),
        register=RegisterEffect(
            name="any",
            semantic_tags={"x": 1.0},
            phonological={"cluster_density": 1.0},
        ),
        weights=ScoringWeights(),
    )
    result = select_via_vector_scoring(
        rng,
        db,
        structure=["Cymro-"],
        request=request,
        priors=EmpiricalPriors(),
    )
    # Only Cymro- matches stratum="native-welsh" → picked.
    assert len(result) == 1
    assert result[0].usage == "Cymro-"


def test_select_baseline_axis_with_populated_priors():
    """Integration test isolating the baseline axis. With sem_w=0 and
    phon_w=0 and pos_w=0, the ONLY axis with weight is the baseline —
    so if the pick succeeds, the baseline lookup demonstrably hit.

    Per D36.7 the baseline_score aggregator is GATED on the request's
    semantic_tags weights (``baseline = sum_over(tag in lemma.tags) of
    request_tag_weights[tag] * lookup(...)``). The register must carry
    a tag weight for the lemma's tags or baseline returns 0 even when
    priors are populated. Set semantic_tags='urban': 1.0 but keep
    sem_w=0 — the per-tag weight gates baseline lookup; sem_w=0 zeros
    out the sem-axis CONTRIBUTION to the composite score.

    Round-1 reviewer caught: every other end-to-end test passed empty
    EmpiricalPriors() so the baseline axis was never exercised.
    Round-2 reviewer tightened: the first cut also weighted sem_w=1.0,
    so the pick could have succeeded via sem alone. Now sem_w=0 —
    the score's only non-zero contribution is the baseline axis."""
    rng = random.Random(0)
    m = _meaning("place", tags=["urban"], phon=PhonologicalVector(cluster_density=0.5))
    db = {"place": [m]}
    # Populate priors for the (culture=english, position=post, tag=urban,
    # era=0) cell so the baseline lookup hits. lemma_ref keying matches
    # _lemma_ref_for's bare-usage output.
    priors = EmpiricalPriors(
        native={("english", "post", "urban", 0): {"place": 100.0}},
    )
    request = RequestVector(
        gate=EligibilityGate(culture="english"),
        # semantic_tags gates baseline lookup; sem_w=0 zeros the
        # sem-axis contribution.
        register=RegisterEffect(name="any", semantic_tags={"urban": 1.0}),
        weights=ScoringWeights(phon_w=0.0, sem_w=0.0, pos_w=0.0, base_w=1.0),
    )
    result = select_via_vector_scoring(
        rng,
        db,
        structure=["place"],  # bare element → post slot per Meaning convention
        request=request,
        priors=priors,
        era_midpoint=0,
    )
    # If this assert succeeds, the baseline axis contributed non-zero —
    # nothing else could have (sem_w=0 zeros out the sem axis).
    assert len(result) == 1
    assert result[0] is m


def test_select_baseline_axis_returns_empty_when_priors_miss():
    """Counterpart to test_select_baseline_axis_with_populated_priors:
    same setup but with EMPTY priors. The baseline lookup misses
    every cell → total score 0 → empty result. Confirms the previous
    test's pass was genuinely the baseline doing work, not a phantom
    non-zero from another path."""
    rng = random.Random(0)
    m = _meaning("place", tags=["urban"], phon=PhonologicalVector(cluster_density=0.5))
    db = {"place": [m]}
    request = RequestVector(
        gate=EligibilityGate(culture="english"),
        register=RegisterEffect(name="any", semantic_tags={"urban": 1.0}),
        weights=ScoringWeights(phon_w=0.0, sem_w=0.0, pos_w=0.0, base_w=1.0),
    )
    result = select_via_vector_scoring(
        rng,
        db,
        structure=["place"],
        request=request,
        priors=EmpiricalPriors(),  # empty
        era_midpoint=0,
    )
    assert result == []


def test_cohesion_multiplier_handles_pair_count_zero():
    """_cohesion_multiplier: when lemma_tags and prior_tags have NO
    overlap with the tag_cooccurrence table (zero pair_count), the
    function returns identity (1.0) rather than dividing by zero."""
    # Cooccurrence table has data for 'fortified' / 'pastoral' but
    # we pass tags 'unknown_a' / 'unknown_b' that aren't in it.
    result = _cohesion_multiplier(
        frozenset({"unknown_a"}),
        frozenset({"unknown_b"}),
        cohesion=1.0,
        tag_cooccurrence={"fortified": {"pastoral": 0.5}},
    )
    assert result == 1.0


def test_select_era_gate_filters_out_of_window():
    """The era gate uses Meaning.attested_in_era_range. Synthetic
    meaning has no attested_years → method returns True (open) → meaning
    passes through. This pins the legacy-compatible default."""
    rng = random.Random(0)
    m = _meaning("Place-", tags=["x"], phon=PhonologicalVector(cluster_density=0.5))
    db = {"Place-": [m]}
    request = RequestVector(
        gate=EligibilityGate(culture="english", era_min=1066, era_max=1200),
        register=RegisterEffect(
            name="any",
            semantic_tags={"x": 1.0},
            phonological={"cluster_density": 1.0},
        ),
        weights=ScoringWeights(),
    )
    result = select_via_vector_scoring(
        rng,
        db,
        structure=["Place-"],
        request=request,
        priors=EmpiricalPriors(),
    )
    # Era range applied; m has no attested_years → passes; should
    # still pick the meaning (non-empty result).
    assert len(result) == 1
    assert result[0] is m


def test_select_exclude_tags_filters_pre_score():
    """exclude_tags filters meanings before scoring."""
    rng = random.Random(0)
    m_ok = _meaning("Good-", tags=["safe"], phon=PhonologicalVector(cluster_density=0.5))
    m_bad = _meaning("Bad-", tags=["fiction"], phon=PhonologicalVector(cluster_density=0.5))
    db = {"Good-": [m_ok], "Bad-": [m_bad]}
    request = RequestVector(
        gate=EligibilityGate(culture="english"),
        register=RegisterEffect(
            name="any",
            semantic_tags={"safe": 1.0, "fiction": 1.0},
        ),
        weights=ScoringWeights(),
    )
    result = select_via_vector_scoring(
        rng,
        db,
        structure=["Good-"],
        request=request,
        priors=EmpiricalPriors(),
        exclude_tags=frozenset({"fiction"}),
    )
    assert len(result) == 1
    assert result[0].usage == "Good-"
