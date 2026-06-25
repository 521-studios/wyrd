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
    _blend_uniform_by_novelty,
    _cohesion_multipliers,
    _cohesion_raw,
    _lemma_ref_for,
    _slot_position_label,
    _slot_weighted_pool,
    _weighted_choice,
    build_non_position_eligible,
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


def _request(culture: str = "english") -> RequestVector:
    """Default RequestVector for testing — equal-weight axes + a non-
    trivial semantic-tag spec so scoring produces non-zero weights.

    Without at least one non-zero scoring source (register, priors),
    every lemma's score is 0 and weighted_choice returns None → tests
    would see empty results across the board, masking real behavior.
    The synthetic fixture's tags (urban / territory / water) all get
    a uniform 1.0 weight here.
    """
    return RequestVector(
        gate=EligibilityGate(culture=culture),
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


# ---- _blend_uniform_by_novelty (wyrd-fcub) --------------------------------


def test_blend_uniform_by_novelty_boundaries():
    """novelty=1 → pure uniform (1/n each); novelty=0 → the score-normalized
    distribution; intermediate interpolates between them."""
    weighted = [(_meaning("a"), 3.0), (_meaning("b"), 1.0)]  # total 4
    full = _blend_uniform_by_novelty(weighted, 1.0)
    assert [round(w, 6) for _, w in full] == [0.5, 0.5]
    none = _blend_uniform_by_novelty(weighted, 0.0)
    assert [round(w, 6) for _, w in none] == [0.75, 0.25]  # 3/4, 1/4
    half = _blend_uniform_by_novelty(weighted, 0.5)
    assert [round(w, 6) for _, w in half] == [0.625, 0.375]  # 0.5*frac + 0.5*0.5


def test_blend_uniform_by_novelty_single_meaning_always_weight_one():
    """n==1: a lone eligible meaning gets weight 1.0 at any novelty (the
    score-normalized and uniform marginals both collapse to it)."""
    single = [(_meaning("a"), 2.5)]
    for nov in (0.0, 0.5, 1.0):
        out = _blend_uniform_by_novelty(single, nov)
        assert [round(w, 6) for _, w in out] == [1.0], nov


def test_blend_uniform_by_novelty_zero_total_returns_uniform():
    """All-zero scores → no axis to blend against → pure uniform."""
    weighted = [(_meaning("a"), 0.0), (_meaning("b"), 0.0)]
    out = _blend_uniform_by_novelty(weighted, 0.5)
    assert [round(w, 6) for _, w in out] == [0.5, 0.5]


# ---- _slot_position_label -------------------------------------------------


def test_slot_position_pre_for_trailing_dash():
    assert _slot_position_label("Place-") == "pre"


def test_slot_position_post_for_leading_dash():
    assert _slot_position_label("-shire") == "post"


def test_slot_position_inner_for_both_dashes():
    assert _slot_position_label("-inner-") == "inner"


def test_slot_position_bare_for_bare_element():
    # wyrd-vpri: a no-dash structural element resolves to its own
    # 'bare' label (was 'post' pre-fix), mirroring the updated
    # Meaning._set_location. wyrd-eyjk/D40: the label feeds position SCORING +
    # bucket-key lookup, not a match gate (the `_matches_position` gate is gone).
    assert _slot_position_label("Bare") == "bare"


# ---- _lemma_ref_for -------------------------------------------------------


def test_lemma_ref_strips_decoration_dashes_on_empty_sources():
    """Fallback path: when ``meaning.sources`` has no source-language
    data (e.g. a synthesized / sidecar Meaning), ``_lemma_ref_for``
    falls back to the bare usage form so the legacy ecjp.5 v1
    contract is preserved."""
    assert _lemma_ref_for(_meaning("Place-")) == "Place"
    assert _lemma_ref_for(_meaning("-shire")) == "shire"
    assert _lemma_ref_for(_meaning("-inner-")) == "inner"
    assert _lemma_ref_for(_meaning("Bare")) == "Bare"


def test_lemma_ref_emits_language_canonical_form_shape():
    """When ``meaning.sources`` carries the per-language form arrays,
    ``_lemma_ref_for`` emits the ``language:canonical_form`` shape the
    priors tables key on (per ``empirical_priors.extract_priors``).
    Walks ``sources`` alphabetically; picks the first non-empty entry."""
    m = _meaning("-shire")
    m.sources = {"old_english": ["scīr", "scir"]}
    assert _lemma_ref_for(m) == "old-english:scīr"


def test_lemma_ref_walks_sources_in_alphabetical_order():
    """Multi-language Meaning: pick the alphabetically-first
    non-empty source field. Deterministic across re-runs (sorted
    keys, not dict-insertion order)."""
    m = _meaning("River")
    # old_english would lose to celtic_mix alphabetically; the test
    # pins that sorting wins.
    m.sources = {"old_english": ["ēa"], "celtic_mix": ["abona"]}
    assert _lemma_ref_for(m) == "celtic:abona"


def test_lemma_ref_skips_empty_source_lists():
    """A lang_field with an empty forms list is skipped — walk
    continues to the next field."""
    m = _meaning("River")
    m.sources = {"old_english": [], "celtic_mix": ["abona"]}
    assert _lemma_ref_for(m) == "celtic:abona"


def test_lemma_ref_scans_past_a_leading_empty_form_within_a_language():
    """A language whose FIRST form is empty but a later one is non-empty
    (``["", "abona"]``) still attests via that later form — ``_lemma_ref_for``
    must scan ALL entries, not just ``forms[0]``, to stay aligned with
    ``Meaning.primary_language`` (the pool-eligibility key). Pre-fix it inspected
    only ``forms[0]``, so the leading empty made it fall THROUGH celtic_mix to
    old_english here, keying the priors-baseline lookup on old_english while
    ``primary_language`` (scanning all forms) kept the Meaning on celtic_mix — a
    per-Meaning identity split in the scoring path (D36)."""
    m = _meaning("River")
    # celtic_mix sorts first; its later non-empty 'abona' attests it, so the
    # leading empty must NOT fall through to old_english's 'wīc'.
    m.sources = {"celtic_mix": ["", "abona"], "old_english": ["wīc"]}
    assert _lemma_ref_for(m) == "celtic:abona"
    assert m.primary_language() == "celtic_mix"  # the two per-Meaning keys agree

    # Same for dict-shaped entries with a leading form-less dict.
    m2 = _meaning("River")
    m2.sources = {"celtic_mix": [{"form": ""}, {"form": "abona"}], "old_english": ["wīc"]}
    assert _lemma_ref_for(m2) == "celtic:abona"


def test_lemma_ref_unknown_lang_field_kebabs_underscore():
    """A bundle field not in ``_BUNDLE_FIELD_TO_L3_LANG`` falls back
    to ``snake_case → kebab-case`` conversion. Forward-compat path
    for bundle fields the runtime hasn't been taught about yet."""
    m = _meaning("X")
    m.sources = {"some_new_lang": ["foo"]}
    assert _lemma_ref_for(m) == "some-new-lang:foo"


def test_lemma_ref_maps_wave_two_bundle_fields():
    """Wave-2 non-Latin buckets (hebrew / arabic / persian / sanskrit /
    akkadian / egyptian / aramaic / armenian) map to the wikt language
    codes the L3 ingest carries — so the baseline lookup hits priors
    data shipped from those languages."""
    for field, expected_lang in (
        ("hebrew", "he"),
        ("arabic", "ar"),
        ("persian", "fa"),
        ("sanskrit", "sa"),
        ("akkadian", "akk"),
        ("egyptian", "egy"),
        ("aramaic", "arc"),
        ("armenian", "axm"),
    ):
        m = _meaning("X")
        m.sources = {field: ["form-x"]}
        assert _lemma_ref_for(m) == f"{expected_lang}:form-x", field


# ---- _cohesion_raw + _cohesion_multipliers (D17 slot-relative) ------------


def test_cohesion_raw_averages_pair_probabilities():
    """raw = mean P(candidate_tag | prior_tag) over the pairs with data."""
    raw = _cohesion_raw(
        frozenset({"animal"}),
        frozenset({"plant"}),
        cohesion_table={"animal": {"plant": 0.6}},
    )
    assert abs(raw - 0.6) < 1e-9


def test_cohesion_raw_zero_on_empty_prior_or_missing_data():
    """No prior (first slot), no table, or no overlapping pair → raw 0.0."""
    table = {"animal": {"plant": 0.9}}
    assert _cohesion_raw(frozenset({"animal"}), frozenset(), table) == 0.0
    assert _cohesion_raw(frozenset({"animal"}), frozenset({"plant"}), None) == 0.0
    # tags present but absent from the table → no pair → 0.0 (not a crash).
    assert _cohesion_raw(frozenset({"unknown_a"}), frozenset({"unknown_b"}), table) == 0.0


def test_cohesion_multipliers_zero_cohesion_is_identity():
    """cohesion=0 → every multiplier is exactly 1.0 (bit-stable default)."""
    assert _cohesion_multipliers([0.9, 0.1, 0.0], 0.0) == [1.0, 1.0, 1.0]


def test_cohesion_multipliers_no_signal_is_identity():
    """A pool with no coherence signal (all raw 0) → all 1.0 even at
    cohesion=1, so the slot can't collapse on a no-data slot."""
    assert _cohesion_multipliers([0.0, 0.0], 1.0) == [1.0, 1.0]


def test_cohesion_multipliers_slot_relative_mean_normalized():
    """Two candidates raw 0.9 / 0.1 (mean 0.5) at cohesion=1 → above-average
    gets 1.8x, below-average 0.2x; the multipliers sum to len(pool)."""
    mults = _cohesion_multipliers([0.9, 0.1], 1.0)
    assert abs(mults[0] - 1.8) < 1e-9
    assert abs(mults[1] - 0.2) < 1e-9
    assert abs(sum(mults) - 2.0) < 1e-9  # mean-normalized → mass preserved


def test_cohesion_multipliers_partial_cohesion_blends():
    """At cohesion=0.5, multiplier = 0.5 + 0.5*(raw/mean): raw 0.9/mean 0.5
    → 1.4, raw 0.1 → 0.6. Still sums to len(pool)."""
    mults = _cohesion_multipliers([0.9, 0.1], 0.5)
    assert abs(mults[0] - 1.4) < 1e-9
    assert abs(mults[1] - 0.6) < 1e-9
    assert abs(sum(mults) - 2.0) < 1e-9


def test_cohesion_multipliers_full_suppression_at_cohesion_one():
    """At cohesion=1 a zero-raw candidate alongside a positive one is fully
    suppressed (multiplier 0), but the positive candidate survives."""
    mults = _cohesion_multipliers([0.0, 0.5], 1.0)
    assert mults[0] == 0.0
    assert mults[1] > 0.0


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
    # wyrd-eyjk/D40: position is the SLOT's (from the structure) and is applied
    # at render time; the meaning chosen to fill a slot is selected by
    # surface + composition score and need NOT carry that slot's stored
    # .location (the per-position bucket frequency, not a hard gate, is what
    # biases a morpheme toward the positions it was actually observed in).
    # So we assert each slot is filled, not that location == slot.
    assert all(isinstance(m, Meaning) for m in result)


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
    # wyrd-eyjk/D40: with the position gate removed, slot separation now comes
    # from the per-(position) bucket frequency, not a hard location filter.
    # Restrict slot 0 to Castle- (pre) and slot 1 to the -keep pair (post) so
    # the second-slot cohesion competition is measured cleanly — mirroring how
    # production's usage_frequency_by_bucket keeps a morpheme in the positions
    # it was actually observed in.
    usage_frequency_by_bucket = {("pre",): {"Castle-": 1.0}, ("post",): {"-keep": 1.0}}
    slot_bucket_keys = [("pre",), ("post",)]

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
                cohesion_table=tag_cooccurrence,
                slot_bucket_keys=slot_bucket_keys,
                usage_frequency_by_bucket=usage_frequency_by_bucket,
            )
            if len(picks) == 2 and "fortified" in picks[1].tags:
                wins += 1
        return wins

    baseline_wins = _count_match_wins(cohesion=0.0)
    biased_wins = _count_match_wins(cohesion=1.0)
    # With a 0.9-vs-0.1 cooccurrence ratio + cohesion=1.0, the match should
    # win materially more often than baseline. The bias is slot-relative
    # (mean-normalized): raws 0.9/0.1, pool mean 0.5, so the match multiplier
    # is 1.8x and the mismatch 0.2x.
    assert biased_wins > baseline_wins + 20, (
        f"D17 cohesion bias not observable in pick distribution: "
        f"baseline={baseline_wins}/200 vs biased={biased_wins}/200"
    )


def test_select_d17_cohesion_suppresses_orphan_without_collapsing_slot():
    """wyrd-e2b4: the load-bearing safety property of the slot-relative
    multiplier, exercised end-to-end through a REAL mixed pool (not just the
    multiplier unit). At cohesion=1 a candidate whose tags have no
    co-occurrence with the priors (raw=0) is fully suppressed — multiplier 0,
    dropped from the weighted pool — yet the slot does NOT collapse, because
    the co-occurring candidate (raw>0) survives. The whole rework rests on
    'suppress without collapse'; this proves both halves through the
    generator, not just the multiplier math."""
    pre = _meaning("Castle-", tags=["fortified"], phon=PhonologicalVector())
    # Both compete in slot 1 at the same usage key. post_match co-occurs with
    # the prior 'fortified'; post_orphan's tag is absent from the table → raw=0.
    post_match = _meaning("-keep", tags=["fortified"], phon=PhonologicalVector())
    post_orphan = _meaning("-keep", tags=["orphan_tag"], phon=PhonologicalVector())
    db = {"Castle-": [pre], "-keep": [post_match, post_orphan]}
    cohesion_table = {"fortified": {"fortified": 0.9}}  # orphan_tag absent → raw 0
    register = RegisterEffect(name="any", semantic_tags={"fortified": 1.0, "orphan_tag": 1.0})
    request = RequestVector(
        gate=EligibilityGate(culture="english"),
        register=register,
        weights=ScoringWeights(sem_w=1.0, phon_w=0.0, pos_w=0.0, base_w=0.0),
    )
    usage_frequency_by_bucket = {("pre",): {"Castle-": 1.0}, ("post",): {"-keep": 1.0}}
    slot_bucket_keys = [("pre",), ("post",)]

    completed = 0
    orphan_picks = 0
    for seed in range(100):
        picks = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["Castle-", "-keep"],
            request=request,
            priors=EmpiricalPriors(),
            cohesion=1.0,
            cohesion_table=cohesion_table,
            slot_bucket_keys=slot_bucket_keys,
            usage_frequency_by_bucket=usage_frequency_by_bucket,
        )
        if len(picks) == 2 and picks[1] is not None:
            completed += 1
            if "orphan_tag" in picks[1].tags:
                orphan_picks += 1
    # Pool survived every time (slot never collapsed despite full suppression).
    assert completed == 100, f"slot collapsed on {100 - completed}/100 seeds"
    # The raw=0 orphan was fully suppressed — never picked at cohesion=1.
    assert orphan_picks == 0, f"fully-suppressed orphan picked {orphan_picks}/100 times"


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


def test_cohesion_raw_handles_pair_count_zero():
    """_cohesion_raw: when candidate and prior tags have NO overlap with the
    tag_cooccurrence table (zero pair_count), raw is 0.0 — not a divide-by-zero
    — so the candidate just sorts to the bottom of the slot-relative ranking."""
    # Cooccurrence table has data for 'fortified' / 'pastoral' but we pass
    # tags 'unknown_a' / 'unknown_b' that aren't in it.
    result = _cohesion_raw(
        frozenset({"unknown_a"}),
        frozenset({"unknown_b"}),
        cohesion_table={"fortified": {"pastoral": 0.5}},
    )
    assert result == 0.0


def test_select_ignores_attested_years_entirely():
    """D44/D46: attestation WINDOWS never gate — a meaning attested ONLY
    in one historical period is exactly as eligible as an undated one
    (the D46 record gate uses dates solely to vouch a morpheme OLDER).
    The retired window fields stay gone (constructing one raises)."""
    with pytest.raises(TypeError):
        EligibilityGate(culture="english", era_min=1066)  # type: ignore[call-arg]

    # The behavioral half: a morpheme whose only attestation is medieval
    # (1086) is selected without complaint — attested_years is inert data
    # to the eligibility/scoring path.
    rng = random.Random(0)
    m = _meaning("Place-", tags=["x"], phon=PhonologicalVector(cluster_density=0.5))
    m.attested_years = {"old_english": [("place", 1086)]}
    db = {"Place-": [m]}
    result = select_via_vector_scoring(
        rng,
        db,
        structure=["Place-"],
        request=_request(),
        priors=EmpiricalPriors(),
    )
    assert result == [m]


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


# ---- build_non_position_eligible + culture filter -------------------------


def _eligible_meanings(culture_attested=None, exclude_tags=frozenset()):
    """Tiny meaning_db fixture: ``-ham`` (post, OE) + ``Cardiff`` (pre,
    Welsh) + ``Bally-`` (pre, anglicized Irish). Used to pin which
    morphemes the culture filter admits / excludes."""
    db: dict[str, list[Meaning]] = {
        "-ham": [_meaning("-ham", tags=["settlement"])],
        "Cardiff": [_meaning("Cardiff", tags=["name"])],
        "Bally-": [_meaning("Bally-", tags=["name"])],
    }
    gate = EligibilityGate(culture="english")
    return build_non_position_eligible(
        db,
        gate=gate,
        exclude_tags=exclude_tags,
        pack_meaning_dbs=None,
        packs=(),
        culture_attested_usages=culture_attested,
    )


def test_build_non_position_eligible_filter_admits_only_attested_usages():
    """``culture_attested_usages`` filter excludes Meanings whose
    ``usage`` key isn't in the attested set. Only ``-ham`` survives
    when the attested set names just ``{"-ham"}`` — Cardiff (Welsh)
    + Bally- (anglicized Irish) get filtered."""
    pool = _eligible_meanings(culture_attested=frozenset({"ham"}))
    assert [m.usage for m in pool] == ["-ham"]


def test_build_non_position_eligible_filter_none_disables():
    """``None`` for the culture filter disables it — all three
    fixture Meanings pass through. Pins the back-compat default for
    non-NameGenerator callers."""
    pool = _eligible_meanings(culture_attested=None)
    assert sorted(m.usage for m in pool) == ["-ham", "Bally-", "Cardiff"]


def test_build_non_position_eligible_empty_filter_empties_pool():
    """An EMPTY frozenset filter excludes every Meaning. Pins that
    the ``None`` vs ``frozenset()`` distinction is meaningful so a
    caller passing an unintentionally-empty filter sees the
    consequence (vs ``None`` which means "no filter")."""
    pool = _eligible_meanings(culture_attested=frozenset())
    assert pool == []


def test_build_non_position_eligible_filter_composes_with_exclude_tags():
    """The culture filter and exclude_tags filter both apply —
    ``Cardiff`` excluded by culture (not in attested set) AND
    ``-ham`` excluded by exclude_tags={'settlement'}. Nothing
    survives both."""
    pool = _eligible_meanings(
        culture_attested=frozenset({"ham"}),
        exclude_tags=frozenset({"settlement"}),
    )
    assert pool == []


# ---- wyrd-izcr: slot_qualifiers honors name/saint flags -------------------


def _qualifier_test_db():
    """Three pre-position morphemes covering the three slot states
    the qualifier filter distinguishes:

    - ``Port-`` — bare (no qualifier flag), admitted only when the
      slot has no qualifier filter
    - ``Smith-`` — ``is_name()`` True via the ``family name`` tag;
      admitted only when ``slot_qualifiers=["name"]``
    - ``Saint-`` — literal "Saint" usage; admitted only when
      ``slot_qualifiers=["saint"]`` per the legacy
      :meth:`Meaning.key` bucket-routing rule

    The saint fixture must use the literal ``Saint-`` usage — saint-
    TAGGED morphemes whose usage isn't literally "Saint" (e.g.
    Andrew-, Botolph-) are routed through the name bucket in the
    legacy proportions path because :meth:`Meaning.key` checks
    is_name() first via if/elif. The vector qualifier filter
    matches that semantics to stay consistent between scoring modes.
    """
    bare = Meaning(
        usage="Port-",
        tags=["urban"],
        meanings=[],
        sources=[],
        phonological_vector=None,
    )
    family = Meaning(
        usage="Smith-",
        tags=["family name", "urban"],
        meanings=[],
        sources=[],
        phonological_vector=None,
    )
    saintly = Meaning(
        usage="Saint-",
        tags=["saint", "urban"],
        meanings=[],
        sources=[],
        phonological_vector=None,
    )
    return (
        {
            "Port-": [bare],
            "Smith-": [family],
            "Saint-": [saintly],
        },
        bare,
        family,
        saintly,
    )


def test_select_slot_qualifier_name_filters_out_bare_morphemes():
    """wyrd-izcr: a slot tagged with the 'name' qualifier (from a
    structure like ``[[pre+name], ...]``) must pick ONLY from
    is_name()=True morphemes. Pre-fix the vector path lost the flag
    during flattening and would draw bare pre-morphemes into name-
    flagged slots."""
    db, bare, family, saintly = _qualifier_test_db()
    picks = []
    for seed in range(20):
        result = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["pre"],
            request=_request(),
            priors=EmpiricalPriors(),
            slot_qualifiers=["name"],
        )
        assert len(result) == 1
        picks.append(result[0])
    # Every pick should be the family-name-tagged morpheme — bare
    # and saint-only morphemes are filtered out by the 'name'
    # qualifier.
    assert all(m is family for m in picks)


def test_select_slot_qualifier_saint_filters_to_literal_saint_usage():
    """wyrd-izcr: a slot tagged with the 'saint' qualifier picks
    ONLY morphemes whose dash-stripped lowercased usage equals
    "saint" — matching :meth:`Meaning.key`'s legacy bucket-routing
    rule. Saint-tagged morphemes with non-"saint" usages (Andrew-,
    Botolph-, etc.) are routed through the name bucket in the
    legacy path, so the vector qualifier filter mirrors that
    semantics."""
    db, bare, family, saintly = _qualifier_test_db()
    picks = []
    for seed in range(20):
        result = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["pre"],
            request=_request(),
            priors=EmpiricalPriors(),
            slot_qualifiers=["saint"],
        )
        assert len(result) == 1
        picks.append(result[0])
    assert all(m is saintly for m in picks)


def test_select_slot_qualifier_none_admits_all_position_matches():
    """wyrd-izcr: when ``slot_qualifiers`` is None or the slot's entry
    is None, no qualifier filter applies — every position-matching
    morpheme is eligible. Preserves the legacy bare-position
    behavior for compound-internal slots."""
    db, bare, family, saintly = _qualifier_test_db()
    seen_usages: set[str] = set()
    for seed in range(60):
        result = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["pre"],
            request=_request(),
            priors=EmpiricalPriors(),
            slot_qualifiers=[None],
        )
        assert len(result) == 1
        seen_usages.add(result[0].usage)
    # Over 60 seeds we should see all 3 pre-morphemes since none is
    # filtered out.
    assert seen_usages == {"Port-", "Smith-", "Saint-"}


def test_select_slot_qualifier_returns_empty_when_pool_is_empty():
    """wyrd-izcr: when no morpheme satisfies the qualifier, return
    empty list. Caller decides whether to retry with a different
    struct (the NameGenerator.select_via_vector wrapper does)."""
    bare = Meaning(
        usage="Port-",
        tags=["urban"],
        meanings=[],
        sources=[],
        phonological_vector=None,
    )
    db = {"Port-": [bare]}
    result = select_via_vector_scoring(
        random.Random(0),
        db,
        structure=["pre"],
        request=_request(),
        priors=EmpiricalPriors(),
        slot_qualifiers=["name"],
    )
    assert result == []


def test_select_slot_qualifier_cache_key_distinguishes_qualifier_variants():
    """wyrd-izcr: the same position label with different qualifier
    flags must populate distinct cache entries — otherwise a
    ``(pre, "name")`` slot's filtered base_scored would be served
    to a ``(pre, None)`` lookup later in the same dispatch. wyrd-bol9
    extended the key to a 3-tuple
    ``(slot_position, slot_qualifier, slot_bucket_key)``; with no
    slot_bucket_keys supplied here, the bucket-key slot is None."""
    db, bare, family, saintly = _qualifier_test_db()
    slot_base_scores: dict = {}
    # First call with qualifier=name fills cache[(pre, "name", None)].
    select_via_vector_scoring(
        random.Random(0),
        db,
        structure=["pre"],
        request=_request(),
        priors=EmpiricalPriors(),
        slot_qualifiers=["name"],
        slot_base_scores=slot_base_scores,
    )
    # Second call with no qualifier fills cache[(pre, None, None)] —
    # MUST be a separate entry with a larger eligible pool.
    select_via_vector_scoring(
        random.Random(0),
        db,
        structure=["pre"],
        request=_request(),
        priors=EmpiricalPriors(),
        slot_qualifiers=[None],
        slot_base_scores=slot_base_scores,
    )
    # Two distinct cache keys, two distinct base-scored lists.
    assert ("pre", "name", None) in slot_base_scores
    assert ("pre", None, None) in slot_base_scores
    assert len(slot_base_scores[("pre", "name", None)]) == 1  # family only
    assert len(slot_base_scores[("pre", None, None)]) == 3  # all 3 pre-morphemes


# ---- wyrd-bol9: frequency-weighted pool ----------------------------------


def _freq_test_db():
    """Two equally-tagged pre-position morphemes. With identical tag
    sets + no phon vectors, their D36.2 vector scores are equal —
    weighted_choice over the two would be 50/50 under uniform-pool
    sampling. The frequency multiplier is the ONLY differentiator
    under the no-mood / no-priors default request, so a non-uniform
    frequency map flows directly through to pick share."""
    common = Meaning(
        usage="Common-",
        tags=["urban"],
        meanings=[],
        sources=[],
        phonological_vector=None,
    )
    rare = Meaning(
        usage="Rare-",
        tags=["urban"],
        meanings=[],
        sources=[],
        phonological_vector=None,
    )
    return ({"Common-": [common], "Rare-": [rare]}, common, rare)


def test_frequency_weighted_pool_biases_pick_toward_frequent_usage():
    """When two equally-scored Meanings share a slot, the one with
    higher per-bucket frequency in the proportions map wins
    proportionally — proving the wyrd-bol9 weighting reaches the
    weighted_choice step."""
    db, common, rare = _freq_test_db()
    usage_frequency_by_bucket = {
        ("pre",): {"Common-": 9.0, "Rare-": 1.0},
    }
    # Run many sub-seeds; ratio should track frequency.
    common_wins = 0
    rare_wins = 0
    for seed in range(200):
        picks = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["pre"],
            request=_request(),
            priors=EmpiricalPriors(),
            slot_bucket_keys=[("pre",)],
            usage_frequency_by_bucket=usage_frequency_by_bucket,
        )
        if not picks:
            continue
        if picks[0] is common:
            common_wins += 1
        elif picks[0] is rare:
            rare_wins += 1
    # 9:1 split with small N — assert direction + roughly the right
    # ratio (allow loose binomial slop).
    total = common_wins + rare_wins
    assert total > 150
    assert common_wins > rare_wins * 4, (
        f"freq-weighted pool failed to bias toward frequent usage: "
        f"common={common_wins} rare={rare_wins}"
    )


def test_frequency_weighted_pool_zero_frequency_filters_usage():
    """A usage with frequency 0 in the slot's bucket is dropped from
    the pool entirely — matches proportions mode, where a usage
    absent from the bucket is unreachable by sampling."""
    db, common, rare = _freq_test_db()
    usage_frequency_by_bucket = {
        ("pre",): {"Common-": 1.0},  # Rare- absent → effectively 0
    }
    rare_wins = 0
    for seed in range(50):
        picks = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["pre"],
            request=_request(),
            priors=EmpiricalPriors(),
            slot_bucket_keys=[("pre",)],
            usage_frequency_by_bucket=usage_frequency_by_bucket,
        )
        if picks and picks[0] is rare:
            rare_wins += 1
    assert rare_wins == 0


def test_frequency_weighted_pool_none_lookup_keeps_uniform_behavior():
    """``usage_frequency_by_bucket=None`` (the back-compat default
    for non-NameGenerator callers) preserves the pre-bol9 unweighted-
    pool behavior — both equally-scored Meanings stay sampleable,
    neither is filtered."""
    db, common, rare = _freq_test_db()
    common_wins = 0
    rare_wins = 0
    for seed in range(50):
        picks = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["pre"],
            request=_request(),
            priors=EmpiricalPriors(),
            usage_frequency_by_bucket=None,
        )
        if not picks:
            continue
        if picks[0] is common:
            common_wins += 1
        elif picks[0] is rare:
            rare_wins += 1
    # Both should be reachable; rough balance (no bias either way).
    assert common_wins > 0
    assert rare_wins > 0


def test_frequency_weighted_pool_missing_bucket_key_filters_all():
    """A slot whose bucket key isn't in the frequency map gets an
    empty per-slot lookup → every Meaning in the pool is filtered
    (treated as freq=0). This is the correct behavior: if the
    proportions map doesn't carry weight for this slot shape,
    neither should the vector pool."""
    db, common, rare = _freq_test_db()
    usage_frequency_by_bucket = {
        # Map present, but ("pre",) — what this slot's bucket-key
        # resolves to — is NOT in it. Caller's bug; we filter as if
        # the slot has no proportions data.
        ("post",): {"Common-": 1.0, "Rare-": 1.0},
    }
    any_picks = 0
    for seed in range(50):
        picks = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["pre"],
            request=_request(),
            priors=EmpiricalPriors(),
            slot_bucket_keys=[("pre",)],
            usage_frequency_by_bucket=usage_frequency_by_bucket,
        )
        if picks:
            any_picks += 1
    assert any_picks == 0


def test_frequency_weighted_pool_distinguishes_single_vs_multi_bucket():
    """wyrd-bol9: the bucket-key threading must keep
    ``("pre",)`` (multi-element word) and ``("pre", "single")``
    (single-element word) distinct so they consult their own
    per-bucket frequencies — proportions treats them as separate
    buckets and the per-usage frequency distributions diverge."""
    db, common, rare = _freq_test_db()
    usage_frequency_by_bucket = {
        ("pre",): {"Common-": 1.0, "Rare-": 0.0},  # multi: common only
        ("pre", "single"): {"Common-": 0.0, "Rare-": 1.0},  # single: rare only
    }
    # Multi-element-word slot → ("pre",) bucket → Common- only.
    multi_meanings = set()
    for seed in range(20):
        picks = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["pre"],
            request=_request(),
            priors=EmpiricalPriors(),
            slot_bucket_keys=[("pre",)],
            usage_frequency_by_bucket=usage_frequency_by_bucket,
        )
        if picks:
            multi_meanings.add(picks[0])
    assert multi_meanings == {common}, (
        f"multi-element bucket should admit Common- only: {multi_meanings}"
    )
    single_meanings = set()
    for seed in range(20):
        picks = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["pre"],
            request=_request(),
            priors=EmpiricalPriors(),
            slot_bucket_keys=[("pre", "single")],
            usage_frequency_by_bucket=usage_frequency_by_bucket,
        )
        if picks:
            single_meanings.add(picks[0])
    assert single_meanings == {rare}, (
        f"single-element bucket should admit Rare- only: {single_meanings}"
    )


def test_frequency_weighted_pool_admits_zero_score_usages():
    """wyrd-bol9: when every Meaning of a usage scores 0 (untagged-
    only usages, common in Celtic corpora where Celtic morphemes
    haven't been tag-enriched), the usage stays pickable rather than
    being filtered as an empty pool. Without this admit-by-frequency
    fallback, vector mode silently drops ~30pp of Celtic content vs
    proportions for Welsh / Irish / Breton (was empirically the
    dominant remaining gap in the wyrd-bol9 pre-merge validation).
    The sibling
    :func:`test_frequency_weighted_pool_zero_score_fallback_within_usage_is_uniform`
    covers the within-usage uniform-distribution property; this test
    covers the admit-at-all property in isolation."""
    untagged_pre = Meaning(
        usage="Quiet-",
        tags=[],  # no tags → sem_score 0 against any request
        meanings=[],
        sources=[],
        phonological_vector=None,
    )
    untagged_post = Meaning(
        usage="-quiet",
        tags=[],
        meanings=[],
        sources=[],
        phonological_vector=None,
    )
    db = {"Quiet-": [untagged_pre], "-quiet": [untagged_post]}
    usage_frequency_by_bucket = {
        ("pre",): {"Quiet-": 5.0},
        ("post",): {"-quiet": 5.0},
    }
    # Pre-fix the usage would have been filtered (all scores 0 →
    # score_sum 0 → continue), producing an empty pool and failing
    # the pick. Post-fix the usage is admitted via frequency.
    pre_picked = 0
    post_picked = 0
    for seed in range(20):
        picks = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["pre", "post"],
            request=_request(),
            priors=EmpiricalPriors(),
            slot_bucket_keys=[("pre",), ("post",)],
            usage_frequency_by_bucket=usage_frequency_by_bucket,
        )
        if picks:
            # wyrd-eyjk/D40: Quiet- and -quiet are the SAME morpheme (surface
            # "quiet"); the per-surface bucket frequency admits it at either
            # slot and both stored variants render identically, so assert by
            # surface rather than by which variant object was drawn. The
            # admit-at-all property (a zero-vector-score usage is still
            # pickable via its frequency) is what this test pins.
            assert picks[0].usage.replace("-", "").lower() == "quiet"
            assert picks[1].usage.replace("-", "").lower() == "quiet"
            pre_picked += 1
            post_picked += 1
    assert pre_picked == 20
    assert post_picked == 20


def test_frequency_weighted_pool_zero_score_fallback_within_usage_is_uniform():
    """When all Meanings of a usage tie at score 0, freq distributes
    uniformly across them (rather than concentrating on one
    arbitrarily). The picked Meaning identity within usage doesn't
    affect lang-mix output downstream — NewName resolves the full
    Meaning list per usage — but the within-usage distribution
    should be even so a downstream consumer relying on the picked
    Meaning identity doesn't see a deterministic bias."""
    untagged_a = Meaning(
        usage="Quiet-",
        tags=[],
        meanings=[],
        sources=[],
        phonological_vector=None,
    )
    untagged_b = Meaning(
        usage="Quiet-",
        tags=[],
        meanings=[],
        sources=[],
        phonological_vector=None,
    )
    db = {"Quiet-": [untagged_a, untagged_b]}
    usage_frequency_by_bucket = {("pre",): {"Quiet-": 5.0}}
    a_wins = 0
    b_wins = 0
    for seed in range(200):
        picks = select_via_vector_scoring(
            random.Random(seed),
            db,
            structure=["pre"],
            request=_request(),
            priors=EmpiricalPriors(),
            slot_bucket_keys=[("pre",)],
            usage_frequency_by_bucket=usage_frequency_by_bucket,
        )
        if not picks:
            continue
        if picks[0] is untagged_a:
            a_wins += 1
        elif picks[0] is untagged_b:
            b_wins += 1
    # Both reachable + roughly balanced (binomial slop OK).
    assert a_wins > 60
    assert b_wins > 60
    assert a_wins + b_wins == 200


# ---------------------------------------------------------------------------
# wyrd-wv85: --tag HARD eligibility gate
# ---------------------------------------------------------------------------


def _tag_pool():
    return {
        "-mere": [_meaning("-mere", tags=["water"])],
        "-burn": [_meaning("-burn", tags=["water", "nature"])],
        "-ton": [_meaning("-ton", tags=["settlement"])],
        "-grim": [_meaning("-grim", tags=["death"])],
    }


def _eligible(gate):
    return {
        m.usage
        for m in build_non_position_eligible(
            _tag_pool(), gate=gate, exclude_tags=frozenset(), pack_meaning_dbs=None, packs=()
        )
    }


def test_required_tags_does_not_gate_the_pool():
    """wyrd-c6o1.4: required_tags is NOT a pool-wide gate (it was, under wv85,
    which collapsed every slot to tagged lemmas). The non-position pool stays
    tag-agnostic — the whole pool is eligible regardless of --tag — so the other
    slots sample freely; the "one tagged morpheme per name" guarantee is enforced
    by the single reserved slot in select_via_vector_scoring, not here."""
    got = _eligible(EligibilityGate(culture="english", required_tags=frozenset({"water"})))
    assert got == {"-mere", "-burn", "-ton", "-grim"}  # nothing dropped


def test_required_tags_empty_is_noop():
    """No --tag → no filtering; the whole pool is eligible (back-compat)."""
    got = _eligible(EligibilityGate(culture="english"))
    assert got == {"-mere", "-burn", "-ton", "-grim"}


def test_required_tags_multiple_still_does_not_gate_the_pool():
    """Even with multiple --tag values the pool is unchanged — the OR-across-tags
    semantics now applies at the reserved tag slot (see the select-level tests),
    not by pre-filtering the pool."""
    got = _eligible(EligibilityGate(culture="english", required_tags=frozenset({"water", "death"})))
    assert got == {"-mere", "-burn", "-ton", "-grim"}


# ---------------------------------------------------------------------------
# wyrd-c6o1.4: --tag reserves ONE slot (hard), the rest sample freely
# ---------------------------------------------------------------------------


def _tag_select_db():
    """Two morphemes per position (one tagged + one not) so the reserved tag slot
    has a choice AND a free slot can pick an un-tagged morpheme."""
    return {
        "Place-": [_meaning("Place-", tags=["urban"])],
        "Brook-": [_meaning("Brook-", tags=["water"])],
        "-shire": [_meaning("-shire", tags=["territory"])],
        "-ford": [_meaning("-ford", tags=["water"])],
        "-grave": [_meaning("-grave", tags=["death"])],
    }


def _tag_request(tags):
    base = _request()
    return RequestVector(
        gate=EligibilityGate(culture="english", required_tags=frozenset(tags)),
        register=base.register,
        weights=base.weights,
    )


def _run_tag_names(db, tags, *, structure=("Place-", "-shire"), seeds=None):
    names = []
    for s in seeds if seeds is not None else range(60):
        picked = select_via_vector_scoring(
            random.Random(s),
            db,
            structure=list(structure),
            request=_tag_request(tags),
            priors=EmpiricalPriors(),
        )
        if picked:
            names.append([m for m in picked if m is not None])
    return names


def test_tag_guarantees_at_least_one_tagged_morpheme_per_name():
    """--tag water: every generated name carries >=1 water-tagged morpheme (the
    reserved tag slot), restoring the historical guarantee."""
    names = _run_tag_names(_tag_select_db(), ["water"])
    assert names  # generation succeeded
    for name in names:
        assert any("water" in m.tags for m in name), [m.usage for m in name]


def test_tag_leaves_the_other_slots_free():
    """The non-reserved slots are NOT restricted to water — un-tagged morphemes
    still surface across seeds. Under the old wv85 all-slots gate EVERY slot was
    forced water, so a non-water morpheme could never appear."""
    names = _run_tag_names(_tag_select_db(), ["water"])
    appeared = {m.usage for name in names for m in name}
    assert appeared & {"Place-", "-shire"}, appeared  # a non-water morpheme surfaced
    # ...and at least one name is NOT all-water (only the reserved slot is water)
    assert any(any("water" not in m.tags for m in name) for name in names)


def test_tag_or_semantics_across_multiple_tags():
    """Multiple --tag values union: the reserved slot satisfies water OR death,
    so every name carries >=1 morpheme tagged water or death."""
    names = _run_tag_names(_tag_select_db(), ["water", "death"])
    assert names
    for name in names:
        assert any({"water", "death"} & set(m.tags) for m in name), [m.usage for m in name]


def test_tag_unsatisfiable_returns_empty():
    """A --tag no morpheme carries anywhere is unsatisfiable → select returns []
    so the caller raises 'no eligible name' (preserves the wv85 typo signal,
    e.g. 'religion' for the real 'religious')."""
    picked = select_via_vector_scoring(
        random.Random(0),
        _tag_select_db(),
        structure=["Place-", "-shire"],
        request=_tag_request(["flibbertigibbet"]),
        priors=EmpiricalPriors(),
    )
    assert picked == []


def test_require_tags_survives_cohesion_drop():
    """wyrd-c6o1.4 (regression): the reserved --tag slot restricts at the
    MEMBERSHIP level (``require_tags`` in _slot_weighted_pool), so a cohesion table
    that would zero a tagged candidate's score cannot break the >=1 guarantee.

    Setup: a 'territory' prior + a cohesion table where 'territory' co-occurs with
    itself but 'water' has NO co-occurrence. At cohesion=1 the water candidate's
    multiplier → 0, so it is DROPPED from the un-restricted pool. The tag slot must
    still surface it (cohesion re-weights WITHIN the tagged subset)."""
    db = {
        "-ford": [_meaning("-ford", tags=["water"])],
        "-shire": [_meaning("-shire", tags=["territory"])],
    }
    pool = build_non_position_eligible(
        db,
        gate=EligibilityGate(culture="english"),
        exclude_tags=frozenset(),
        pack_meaning_dbs=None,
        packs=(),
    )
    ctable = {"territory": {"territory": 1.0}}  # water has no row → raw(water)=0
    common = {
        "slot_position": "post",
        "slot_qualifier": None,
        "slot_bucket_key": None,
        "request": _request(),
        "priors": EmpiricalPriors(),
        "cohesion": 1.0,
        "cohesion_table": ctable,
        "usage_frequency_by_bucket": None,
        "novelty": 0.0,
        "prior_tags": frozenset({"territory"}),
        "slot_base_scores": None,
    }
    # un-restricted: cohesion drops the water candidate entirely
    free = _slot_weighted_pool(pool, **common)
    assert all("water" not in m.tags for m, _ in free), [m.usage for m, _ in free]
    # reserved tag slot: restricting first keeps the water candidate
    restricted = _slot_weighted_pool(pool, require_tags=frozenset({"water"}), **common)
    assert restricted and all("water" in m.tags for m, _ in restricted)


def test_tag_guarantee_survives_a_concurrent_mood():
    """wyrd-c6o1.4: a HARD --tag and a SOFT mood each reserve a slot. Even when
    they would collide, the tag is enforced at the pool level, so every name still
    carries >=1 required-tag morpheme (the mood never steals the tag's slot)."""
    base = _request()
    rv = RequestVector(
        gate=EligibilityGate(culture="english", required_tags=frozenset({"water"})),
        register=base.register,
        weights=base.weights,
        mood_tags={"death": 1.0},  # mood also wants a slot
    )
    ok = 0
    for s in range(40):
        picked = select_via_vector_scoring(
            random.Random(s),
            _tag_select_db(),
            structure=["Place-", "-shire"],
            request=rv,
            priors=EmpiricalPriors(),
        )
        if not picked:
            continue
        ok += 1
        assert any("water" in m.tags for m in picked if m is not None), [
            m.usage for m in picked if m is not None
        ]
    assert ok  # generation succeeded


def test_build_request_vector_tag_is_hard_gate_mood_is_soft_overlay():
    """wyrd-4rp8 / wyrd-c6o1.4: explicit --tag populates required_tags (the carrier
    the single reserved tag slot reads); a thematic mood is a SOFT preference on
    RequestVector.mood_tags, NOT a gate. (The old wyrd-wv85 design unioned the
    mood's tags into required_tags too, AND applied required_tags pool-wide — too
    hard on both counts. Now --tag reserves one slot, the mood reserves one slot,
    and neither pre-filters the pool.)"""
    from wyrd.generators.kenning.runtime.vector_kenning_adapter import build_request_vector

    # Explicit --tag → hard gate (unchanged).
    rv = build_request_vector(culture="english", tags=["water", "death"])
    assert rv.gate.required_tags == frozenset({"water", "death"})
    assert rv.mood_tags == {}

    # A mood does NOT touch the gate; its tags + weights ride on mood_tags.
    rv_mood = build_request_vector(culture="english", mood=["grim"])
    assert rv_mood.gate.required_tags == frozenset()
    assert set(rv_mood.mood_tags) == {"death", "military", "monster", "undead", "magic"}
    assert all(w > 0 for w in rv_mood.mood_tags.values())  # per-tag weights kept

    # --tag + --mood: only the --tag gates; the mood stays soft.
    rv_both = build_request_vector(culture="english", tags=["water"], mood=["grim"])
    assert rv_both.gate.required_tags == frozenset({"water"})
    assert set(rv_both.mood_tags) == {"death", "military", "monster", "undead", "magic"}

    # A purely PHONOLOGICAL mood (harsh: harshness only, no semantic tags)
    # contributes NO required tags — it gates nothing, only re-weights phon.
    rv_harsh = build_request_vector(culture="english", mood=["harsh"])
    assert rv_harsh.gate.required_tags == frozenset()


# ---------------------------------------------------------------------------
# wyrd-glos: gloss policy (glossed-only default + include_unglossed opt-in)
# ---------------------------------------------------------------------------


def _gloss_pool(include_unglossed):
    db = {
        "-ton": [_meaning("-ton", tags=["settlement"])],  # glossed below
        "-xyz": [_meaning("-xyz")],  # multi-char UNGLOSSED
        "-q": [_meaning("-q")],  # 1-char UNGLOSSED
        "a": [_meaning("a")],  # 1-char UNGLOSSED (no dash)
    }
    # give -ton a gloss; _meaning() defaults meanings=[] (unglossed)
    db["-ton"][0].meanings = ["estate, farmstead"]
    # a glossed single-char survives (Norse á = river)
    db["á"] = [_meaning("á")]
    db["á"][0].meanings = ["river"]
    return {
        m.usage
        for m in build_non_position_eligible(
            db,
            gate=EligibilityGate(culture="english"),
            exclude_tags=frozenset(),
            pack_meaning_dbs=None,
            packs=(),
            include_unglossed=include_unglossed,
        )
    }


def test_gloss_policy_default_is_glossed_only():
    """Default (include_unglossed=False): only morphemes with a gloss
    survive — the rando-era behavior, so every generated component can be
    explained."""
    assert _gloss_pool(include_unglossed=False) == {"-ton", "á"}


def test_gloss_policy_opt_in_admits_multichar_unglossed_but_never_single_char():
    """include_unglossed=True admits MULTI-char unglossed morphemes, but
    single-char unglossed fragments (the rando 'Silver' letters, stray
    -m-/A-) are ALWAYS dropped. Glossed single-char (á = river) survives
    on the has-gloss branch regardless."""
    assert _gloss_pool(include_unglossed=True) == {"-ton", "-xyz", "á"}
    # -q and "a" (1-char unglossed) excluded in BOTH states:
    assert "-q" not in _gloss_pool(include_unglossed=True)
    assert "a" not in _gloss_pool(include_unglossed=True)


def test_select_via_vector_scoring_permissive_returns_none_for_empty_slot():
    """wyrd-tbke: ``permissive=True`` makes an unsatisfiable slot contribute
    ``None`` and continue (full-length result), instead of the non-permissive
    ``[]`` that aborts the whole struct. Pins the degrade contract at the
    primitive level (the select_via_vector fallback relies on it)."""
    # slot 0: no qualifier → fills with the urban morpheme; slot 1: 'saint'
    # qualifier with no saint-usage morpheme in db → empty pool.
    db = {"Port-": [Meaning(usage="Port-", tags=["urban"], meanings=[], sources=[])]}
    kwargs = {
        "structure": ["pre", "pre"],
        "request": _request(),
        "priors": EmpiricalPriors(),
        "slot_qualifiers": [None, "saint"],
    }
    permissive = select_via_vector_scoring(random.Random(0), db, permissive=True, **kwargs)
    assert len(permissive) == 2, "permissive result is always full-length"
    assert permissive[0] is not None and permissive[0].usage == "Port-"
    assert permissive[1] is None, "unsatisfiable saint slot → None, not aborted"
    # Non-permissive (default) aborts the whole struct on the empty slot.
    assert select_via_vector_scoring(random.Random(0), db, **kwargs) == []


def test_select_via_vector_scoring_permissive_handles_zeroed_weights(monkeypatch):
    """wyrd-2idg: the 'not weighted' permissive branch. base_scored is non-empty
    but every final score collapses to <= 0 after the cohesion multiplier, so
    nothing is weight-eligible. This is defensive — build_slot_base_scores only
    emits base scores > 0 — so force the collapse by zeroing the cohesion
    multiplier. permissive=True → None for the slot (full-length result);
    non-permissive → [] (abort)."""
    db = {"Port-": [_meaning("Port-", tags=["urban"])]}
    # cohesion>0 so _slot_weighted_pool takes the multiplier path (the cohesion=0
    # default short-circuits past _cohesion_multipliers, which the monkeypatch
    # below targets). cohesion_table stays None, so the un-patched baseline still
    # fills normally (all multipliers 1.0).
    kwargs = {
        "structure": ["pre"],
        "request": _request(),
        "priors": EmpiricalPriors(),
        "cohesion": 1.0,
    }
    # Sanity (no monkeypatch yet): the slot fills normally, so base_scored is
    # non-empty — the collapse below is specifically the 'not weighted' branch,
    # not the already-covered empty-pool ('not base_scored') branch.
    baseline = select_via_vector_scoring(random.Random(0), db, **kwargs)
    assert baseline and baseline[0] is not None
    monkeypatch.setattr(
        "wyrd.generators.kenning.runtime.vector_name_select._cohesion_multipliers",
        lambda raws, cohesion: [0.0] * len(raws),
    )
    permissive = select_via_vector_scoring(random.Random(0), db, permissive=True, **kwargs)
    assert permissive == [None], "zeroed weights under permissive → None slot, full-length"
    # Non-permissive aborts the whole struct when a slot has no weighted pool.
    assert select_via_vector_scoring(random.Random(0), db, **kwargs) == []


def test_select_via_vector_scoring_permissive_handles_weighted_choice_none(monkeypatch):
    """wyrd-2idg: the '_weighted_choice returns None' permissive branch.
    base_scored AND weighted are both non-empty, but the weighted draw returns
    None. This is defensive — a positive-weight list won't normally yield None —
    so force it. permissive=True → None for the slot; non-permissive → []."""
    db = {"Port-": [_meaning("Port-", tags=["urban"])]}
    kwargs = {"structure": ["pre"], "request": _request(), "priors": EmpiricalPriors()}
    # Sanity (no monkeypatch yet): the slot fills normally, so base_scored and
    # weighted are non-empty — the [None] below is specifically the
    # '_weighted_choice returns None' branch, reached only after both pass.
    baseline = select_via_vector_scoring(random.Random(0), db, **kwargs)
    assert baseline and baseline[0] is not None
    monkeypatch.setattr(
        "wyrd.generators.kenning.runtime.vector_name_select._weighted_choice",
        lambda *a, **k: None,
    )
    permissive = select_via_vector_scoring(random.Random(0), db, permissive=True, **kwargs)
    assert permissive == [None], "weighted_choice None under permissive → None slot"
    # Non-permissive aborts the whole struct when the weighted draw returns None.
    assert select_via_vector_scoring(random.Random(0), db, **kwargs) == []


# ---------------------------------------------------------------------------
# Register-bias smoke tests (relocated from test_kenning_realism_regression.py)
# ---------------------------------------------------------------------------


def test_register_phonological_bias_scores_lemmas_consistently():
    """Smoke: a RequestVector with a harsh-leaning register produces
    consistent score-direction across multiple invocations. Verifies
    the score → register-effect wiring doesn't silently no-op.

    Uses the vector_name_select primitive directly (not Kenning.generate)
    so the test isolates the register-bias contract from the dispatch
    layer + bundle dependencies."""
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
