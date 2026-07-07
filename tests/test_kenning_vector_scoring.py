"""Tests for wyrd-ecjp.4 Phase 4: vector-scoring runtime.

Exercises the per-axis sub-scorers + aggregate against synthetic
PhonologicalVector / RegisterEffect / EmpiricalPriors / PackOverlay
fixtures. Production data won't be available until kq7w.1 corpus
enrichment lands; these tests pin the math + shape independently.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.vectors.schemas import (
    EligibilityGate,
    EmpiricalPriors,
    PackOverlay,
    PhonologicalVector,
    RegisterEffect,
    RequestVector,
    ScoringWeights,
)
from wyrd.generators.kenning.vectors.scoring import (
    _loan_lookup_with_fallback,
    _native_lookup_exact,
    _native_lookup_tag_specific,
    _native_lookup_with_fallback,
    aggregate_score,
    baseline_score,
    baseline_score_native,
    baseline_score_pack,
    phon_score,
    pos_score,
    score,
    sem_score,
)

# ---------- phon_score --------------------------------------------------


def test_phon_score_empty_register_phon_returns_zero():
    """Register with no phonological weights → 0 regardless of lemma."""
    lemma = PhonologicalVector(cluster_density=0.8, vowel_final_bias=-0.4)
    register = RegisterEffect(name="empty")
    assert phon_score(lemma, register) == 0.0


def test_phon_score_aligned_lemma_register_positive():
    """Lemma + register that both push 'cluster-heavy' → positive."""
    lemma = PhonologicalVector(cluster_density=0.6, final_fortition=0.5)
    register = RegisterEffect(
        name="harsh",
        phonological={"cluster_density": 0.6, "final_fortition": 0.5},
    )
    # dot: 0.6 * 0.6 + 0.5 * 0.5 = 0.61
    assert phon_score(lemma, register) == pytest.approx(0.61)


def test_phon_score_opposed_lemma_register_negative():
    """Lemma vowel-final-biased + register that prefers vowel-final-
    biased NEGATIVE (e.g. harsh) → negative score."""
    lemma = PhonologicalVector(vowel_final_bias=0.5)
    register = RegisterEffect(name="harsh", phonological={"vowel_final_bias": -0.4})
    # 0.5 * -0.4 = -0.20
    assert phon_score(lemma, register) == pytest.approx(-0.20)


def test_phon_score_ignores_unknown_dimension_in_register():
    """A register weight on a dimension name that's NOT in the canonical
    14-name set contributes 0 (PhonologicalVector.dot whitelist)."""
    lemma = PhonologicalVector(cluster_density=0.5)
    register = RegisterEffect(
        name="future",
        phonological={"made_up_dim": 0.99, "cluster_density": 0.4},
    )
    # made_up_dim ignored; only cluster_density contributes
    assert phon_score(lemma, register) == pytest.approx(0.5 * 0.4)


# ---------- sem_score ---------------------------------------------------


def test_sem_score_empty_register_tags_returns_zero():
    register = RegisterEffect(name="empty")
    assert sem_score(["plant", "tree"], register) == 0.0


def test_sem_score_no_tags_on_lemma_returns_zero():
    register = RegisterEffect(name="grim", semantic_tags={"death": 0.7})
    assert sem_score([], register) == 0.0


def test_sem_score_single_matching_tag():
    register = RegisterEffect(name="grim", semantic_tags={"death": 0.7})
    assert sem_score(["death"], register) == pytest.approx(0.7)


def test_sem_score_multiple_matching_tags_sum():
    register = RegisterEffect(
        name="grim", semantic_tags={"death": 0.7, "monster": 0.5, "magic": 0.4}
    )
    assert sem_score(["death", "monster"], register) == pytest.approx(1.2)


def test_sem_score_ignores_tags_request_does_not_weight():
    """Lemma carries water tag; request only cares about death.
    Water contributes 0; the score reflects only the matched
    intersection."""
    register = RegisterEffect(name="grim", semantic_tags={"death": 0.7})
    assert sem_score(["death", "water"], register) == pytest.approx(0.7)


def test_sem_score_dedupes_lemma_tags():
    """Lemma tag set is deduplicated before lookup — a lemma listed
    with ['plant', 'plant'] shouldn't double-score the plant weight."""
    register = RegisterEffect(name="pastoral", semantic_tags={"plant": 0.5})
    assert sem_score(["plant", "plant"], register) == pytest.approx(0.5)


# ---------- pos_score ---------------------------------------------------


def test_pos_score_empty_register_returns_zero():
    register = RegisterEffect(name="empty")
    assert pos_score("pre", register) == 0.0


def test_pos_score_register_biases_for_position():
    register = RegisterEffect(name="prefer-pre", position_bias={"pre": 0.5})
    assert pos_score("pre", register) == pytest.approx(0.5)


def test_pos_score_register_biases_against_position():
    register = RegisterEffect(name="anti-post", position_bias={"post": -0.3})
    assert pos_score("post", register) == pytest.approx(-0.3)


def test_pos_score_slot_position_not_in_bias_dict_returns_zero():
    register = RegisterEffect(name="prefer-pre", position_bias={"pre": 0.5})
    # Asking about a slot the register doesn't have an opinion on
    assert pos_score("inner", register) == 0.0


# ---------- _native_lookup_exact ----------------------------------------


def test_native_lookup_exact_hit_returns_weight():
    priors = EmpiricalPriors(native={("english", "pre", "plant", 950): {"old-english:āc": 7.0}})
    assert _native_lookup_exact(priors, "old-english:āc", "english", "pre", "plant", 950) == 7.0


def test_native_lookup_exact_missing_cell_returns_none():
    priors = EmpiricalPriors()
    assert _native_lookup_exact(priors, "old-english:āc", "english", "pre", "plant", 950) is None


def test_native_lookup_exact_cell_missing_lemma_returns_none():
    priors = EmpiricalPriors(native={("english", "pre", "plant", 950): {"old-english:other": 5.0}})
    assert _native_lookup_exact(priors, "old-english:āc", "english", "pre", "plant", 950) is None


def test_native_lookup_exact_lemma_stored_at_zero_returns_zero_not_none():
    """Defensive: a lemma explicitly stored with weight 0.0 must be
    reported as PRESENT (returning 0.0), not as MISSING (returning
    None). Missing-vs-stored-zero matters because returning None
    triggers hierarchical fallback to a coarser cell — a stored
    zero is real evidence that this lemma should not score in this
    cell, and fallback would silently override it.

    The L3-table schema CHECK currently excludes count=0, but the
    in-memory EmpiricalPriors dataclass doesn't enforce that and
    future schema changes might. Pinning this contract here keeps
    the behavior stable."""
    priors = EmpiricalPriors(native={("english", "pre", "plant", 950): {"old-english:āc": 0.0}})
    assert _native_lookup_exact(priors, "old-english:āc", "english", "pre", "plant", 950) == 0.0


# ---------- _native_lookup_with_fallback --------------------------------


def test_native_fallback_exact_match_short_circuits():
    """When the exact cell has the lemma, fallback never iterates."""
    priors = EmpiricalPriors(
        native={
            ("english", "pre", "plant", 950): {"old-english:āc": 5.0},
            # noise cells that should NOT be consulted
            ("english", "pre", "plant", 1300): {"old-english:āc": 99.0},
            ("english", "pre", "tree", 950): {"old-english:āc": 99.0},
        }
    )
    assert (
        _native_lookup_with_fallback(priors, "old-english:āc", "english", "pre", "plant", 950)
        == 5.0
    )


def test_native_fallback_era_wildcard():
    """Exact cell missing; (c, p, t, *) has the lemma → max weight."""
    priors = EmpiricalPriors(
        native={
            ("english", "pre", "plant", 1300): {"old-english:āc": 3.0},
            ("english", "pre", "plant", 1600): {"old-english:āc": 8.0},
            # Different position/tag — shouldn't match
            ("english", "post", "plant", 950): {"old-english:āc": 99.0},
        }
    )
    # exact (950) missing; era-wildcard returns max(3, 8) = 8
    assert (
        _native_lookup_with_fallback(priors, "old-english:āc", "english", "pre", "plant", 950)
        == 8.0
    )


def test_native_fallback_tag_and_era_wildcard():
    """(c, p, t, *) misses; (c, p, *, *) has the lemma."""
    priors = EmpiricalPriors(
        native={
            ("english", "pre", "tree", 950): {"old-english:āc": 4.0},
            ("english", "pre", "tree", 1300): {"old-english:āc": 6.0},
        }
    )
    # exact (plant, 950) missing; era-wildcard for plant missing too;
    # (c, p, *, *) finds tree cells → max(4, 6) = 6
    assert (
        _native_lookup_with_fallback(priors, "old-english:āc", "english", "pre", "plant", 950)
        == 6.0
    )


def test_native_fallback_position_tag_era_wildcard():
    """All previous levels miss; (c, *, *, *) has the lemma."""
    priors = EmpiricalPriors(
        native={
            ("english", "post", "tree", 1300): {"old-english:āc": 2.0},
            ("english", "inner", "water", 1600): {"old-english:āc": 9.0},
        }
    )
    # Nothing matches (pre, plant, 950) at any level above (c, *, *, *)
    assert (
        _native_lookup_with_fallback(priors, "old-english:āc", "english", "pre", "plant", 950)
        == 9.0
    )


def test_native_fallback_no_lemma_anywhere_returns_zero():
    """Lemma absent from every cell at every specificity level → 0.0."""
    priors = EmpiricalPriors(native={("english", "pre", "plant", 950): {"old-english:other": 5.0}})
    assert (
        _native_lookup_with_fallback(priors, "old-english:āc", "english", "pre", "plant", 950)
        == 0.0
    )


def test_native_fallback_different_culture_does_not_leak():
    """A lemma in welsh cells doesn't satisfy an english lookup."""
    priors = EmpiricalPriors(
        native={
            ("welsh", "pre", "plant", 950): {"welsh:tref": 99.0},
            ("welsh", "post", "plant", 1300): {"welsh:tref": 99.0},
        }
    )
    assert _native_lookup_with_fallback(priors, "welsh:tref", "english", "pre", "plant", 950) == 0.0


def test_native_fallback_index_matches_scan_max_and_zero_edge():
    """The ``native_fallback_index`` reverse maps (the O(1) replacement for the
    per-call O(N_cells) fallback scans) hold the SAME max the scans computed,
    across every collapsed dimension, including a lemma stored at weight 0.0
    (present, not absent). Guards the perf fix's parity contract directly."""
    priors = EmpiricalPriors(
        native={
            # same (c, p, t) across eras → l2 max = 8.0
            ("english", "pre", "plant", 1300): {"oe:a": 3.0, "oe:zero": 0.0},
            ("english", "pre", "plant", 1600): {"oe:a": 8.0},
            # different tag, same (c, p) → l3 must see both plant + tree → max 6
            ("english", "pre", "tree", 950): {"oe:a": 6.0},
            # different position, same culture → only l4 (c, *, *, *) sees it → 9
            ("english", "post", "water", 1600): {"oe:a": 9.0},
        }
    )
    l2, l3, l4 = priors.native_fallback_index
    # Level 2: era wildcard max over (english, pre, plant).
    assert l2[("english", "pre", "plant")]["oe:a"] == 8.0
    # Level 3: (english, pre) max over all tags/eras = max(8, 6) = 8.
    assert l3[("english", "pre")]["oe:a"] == 8.0
    # Level 4: (english) max over everything incl. the post/water cell = 9.
    assert l4["english"]["oe:a"] == 9.0
    # 0.0 lands in the index (present), so the lookup returns 0.0 not None.
    assert l2[("english", "pre", "plant")]["oe:zero"] == 0.0
    assert _native_lookup_tag_specific(priors, "oe:zero", "english", "pre", "plant", 950) == 0.0


# ---------- _loan_lookup_with_fallback ----------------------------------


def test_loan_lookup_exact_match():
    priors = EmpiricalPriors(
        loan_relationship={("old-norse", "english", "pre", "social", 950): {"old-norse:karl": 4.0}}
    )
    assert (
        _loan_lookup_with_fallback(
            priors, "old-norse:karl", "old-norse", "english", "pre", "social", 950
        )
        == 4.0
    )


def test_loan_lookup_era_wildcard():
    priors = EmpiricalPriors(
        loan_relationship={("old-norse", "english", "pre", "social", 1300): {"old-norse:karl": 7.0}}
    )
    # exact (950) missing; era-wildcard finds 7.0
    assert (
        _loan_lookup_with_fallback(
            priors, "old-norse:karl", "old-norse", "english", "pre", "social", 950
        )
        == 7.0
    )


def test_loan_lookup_donor_recipient_isolation():
    """ON->OE lookup doesn't leak from a NF->ME cell."""
    priors = EmpiricalPriors(
        loan_relationship={
            ("norman-french", "middle-english", "pre", "social", 950): {"old-norse:karl": 99.0}
        }
    )
    assert (
        _loan_lookup_with_fallback(
            priors, "old-norse:karl", "old-norse", "english", "pre", "social", 950
        )
        == 0.0
    )


def test_loan_lookup_tag_and_era_wildcard():
    """Level 3: exact + era-wild both miss; (d, r, p, *, *) has lemma."""
    priors = EmpiricalPriors(
        loan_relationship={
            ("old-norse", "english", "pre", "kinship", 950): {"old-norse:karl": 3.0},
            ("old-norse", "english", "pre", "kinship", 1300): {"old-norse:karl": 6.0},
        }
    )
    # exact (pre, social, 950) misses; era-wild (pre, social, *) misses;
    # (pre, *, *) finds kinship cells → max(3, 6) = 6
    assert (
        _loan_lookup_with_fallback(
            priors, "old-norse:karl", "old-norse", "english", "pre", "social", 950
        )
        == 6.0
    )


def test_loan_lookup_position_tag_era_wildcard():
    """Level 4: (d, r, *, *, *) finds the lemma in any cell for the
    donor/recipient."""
    priors = EmpiricalPriors(
        loan_relationship={
            ("old-norse", "english", "post", "kinship", 1300): {"old-norse:karl": 9.0},
        }
    )
    # exact and all narrower wildcards miss; (donor, recipient, *, *, *)
    # finds the post-cell → 9.0
    assert (
        _loan_lookup_with_fallback(
            priors, "old-norse:karl", "old-norse", "english", "pre", "social", 950
        )
        == 9.0
    )


def test_loan_lookup_no_lemma_anywhere_returns_zero():
    priors = EmpiricalPriors(
        loan_relationship={("old-norse", "english", "pre", "social", 950): {"old-norse:other": 4.0}}
    )
    assert (
        _loan_lookup_with_fallback(
            priors, "old-norse:karl", "old-norse", "english", "pre", "social", 950
        )
        == 0.0
    )


def test_native_lookup_fallback_exact_short_circuits_past_era_wild():
    """Pin the level-priority contract: when the exact cell has the
    lemma, the era-wild scan MUST NOT consult lower-specificity cells.
    A regression that took max across ALL levels (instead of
    short-circuiting at the most specific match) would silently
    return the lower-level weight instead of the exact-cell weight.

    The exact cell stores weight 3.0; lower-specificity cells store
    99.0. Correct behavior returns 3.0 (most-specific evidence wins);
    a broken aggregator returning max-across-all-levels returns 99.0.
    """
    priors = EmpiricalPriors(
        native={
            ("english", "pre", "plant", 950): {"old-english:āc": 3.0},
            # Lower-specificity (era-wild from plant's perspective)
            ("english", "pre", "plant", 1300): {"old-english:āc": 99.0},
            # Tag-wild
            ("english", "pre", "tree", 950): {"old-english:āc": 99.0},
            # Position-wild
            ("english", "post", "plant", 950): {"old-english:āc": 99.0},
        }
    )
    result = _native_lookup_with_fallback(priors, "old-english:āc", "english", "pre", "plant", 950)
    assert result == 3.0


def test_loan_lookup_fallback_exact_short_circuits_past_era_wild():
    """Same level-priority pin for the loan-relationship fallback."""
    priors = EmpiricalPriors(
        loan_relationship={
            ("old-norse", "english", "pre", "social", 950): {"old-norse:karl": 3.0},
            ("old-norse", "english", "pre", "social", 1300): {"old-norse:karl": 99.0},
            ("old-norse", "english", "pre", "kinship", 950): {"old-norse:karl": 99.0},
        }
    )
    result = _loan_lookup_with_fallback(
        priors, "old-norse:karl", "old-norse", "english", "pre", "social", 950
    )
    assert result == 3.0


# ---------- baseline_score_native ---------------------------------------


def _request_tag_weights(**tags: float) -> dict[str, float]:
    return dict(tags)


def test_baseline_native_empty_request_tag_weights_returns_zero():
    priors = EmpiricalPriors(native={("english", "pre", "plant", 950): {"old-english:āc": 5.0}})
    assert (
        baseline_score_native(
            "old-english:āc",
            ["plant"],
            culture="english",
            slot_position="pre",
            era_midpoint=950,
            priors=priors,
            request_tag_weights={},
        )
        == 0.0
    )


def test_baseline_native_empty_lemma_tags_returns_zero():
    """Lemma with no tags has nothing to aggregate over."""
    priors = EmpiricalPriors(native={("english", "pre", "plant", 950): {"old-english:āc": 5.0}})
    assert (
        baseline_score_native(
            "old-english:āc",
            [],
            culture="english",
            slot_position="pre",
            era_midpoint=950,
            priors=priors,
            request_tag_weights=_request_tag_weights(plant=0.7),
        )
        == 0.0
    )


def test_baseline_native_single_tag_weighted_lookup():
    priors = EmpiricalPriors(native={("english", "pre", "plant", 950): {"old-english:āc": 5.0}})
    # request_tag_weights[plant] = 0.7; per-tag lookup = 5.0
    # baseline = 0.7 * 5.0 = 3.5
    result = baseline_score_native(
        "old-english:āc",
        ["plant"],
        culture="english",
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(plant=0.7),
    )
    assert result == pytest.approx(3.5)


def test_baseline_native_multi_tag_aggregated():
    """Lemma has tags [plant, tree]; both are weighted by request.
    baseline = 0.7 * priors[plant] + 0.3 * priors[tree]"""
    priors = EmpiricalPriors(
        native={
            ("english", "pre", "plant", 950): {"old-english:āc": 5.0},
            ("english", "pre", "tree", 950): {"old-english:āc": 8.0},
        }
    )
    result = baseline_score_native(
        "old-english:āc",
        ["plant", "tree"],
        culture="english",
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(plant=0.7, tree=0.3),
    )
    # 0.7 * 5.0 + 0.3 * 8.0 = 3.5 + 2.4 = 5.9
    assert result == pytest.approx(5.9)


def test_baseline_native_skips_zero_weight_tags():
    """Tags the request weights at 0 short-circuit out of the per-tag
    loop — saves the lookup cost."""
    priors = EmpiricalPriors(
        native={
            ("english", "pre", "plant", 950): {"old-english:āc": 5.0},
            ("english", "pre", "irrelevant", 950): {"old-english:āc": 99.0},
        }
    )
    result = baseline_score_native(
        "old-english:āc",
        ["plant", "irrelevant"],
        culture="english",
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(plant=0.7, irrelevant=0.0),
    )
    # irrelevant contributes 0 (weight is 0); only plant counts
    assert result == pytest.approx(0.7 * 5.0)


# ---------- baseline_score_pack -----------------------------------------


def _pack(name: str = "neo-khuzdul", weight: float = 1.0) -> PackOverlay:
    return PackOverlay(
        pack_name=name,
        template_donor="old-norse",
        template_recipient="english",
        weight=weight,
    )


def test_baseline_pack_lookup_through_template_donor_recipient():
    """Pack lookup uses pack.template_donor + template_recipient as the
    loan-relationship key (D36.4 Option B)."""
    priors = EmpiricalPriors(
        loan_relationship={("old-norse", "english", "pre", "social", 950): {"khuz:karl": 4.0}}
    )
    result = baseline_score_pack(
        "khuz:karl",
        ["social"],
        pack=_pack(),
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(social=0.5),
    )
    # 0.5 * 4.0 = 2.0 (pack.weight NOT applied here — caller composes)
    assert result == pytest.approx(2.0)


def test_baseline_pack_does_not_apply_pack_weight():
    """baseline_score_pack returns the unscaled per-tag aggregate;
    pack.weight is composed by the caller (baseline_score) so the
    multiplier is observable independently."""
    priors = EmpiricalPriors(
        loan_relationship={("old-norse", "english", "pre", "social", 950): {"khuz:karl": 4.0}}
    )
    pack_full = _pack(weight=1.0)
    pack_half = _pack(weight=0.5)
    result_full = baseline_score_pack(
        "khuz:karl",
        ["social"],
        pack=pack_full,
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(social=0.5),
    )
    result_half = baseline_score_pack(
        "khuz:karl",
        ["social"],
        pack=pack_half,
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(social=0.5),
    )
    # Both ignore pack.weight — caller's responsibility
    assert result_full == result_half


def test_baseline_pack_empty_request_tag_weights_returns_zero():
    """Same short-circuit behavior as baseline_score_native: no
    request semantic interest → no per-tag aggregation."""
    priors = EmpiricalPriors(
        loan_relationship={("old-norse", "english", "pre", "social", 950): {"khuz:karl": 4.0}}
    )
    result = baseline_score_pack(
        "khuz:karl",
        ["social"],
        pack=_pack(),
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights={},
    )
    assert result == 0.0


def test_baseline_pack_empty_lemma_tags_returns_zero():
    """Lemma with no tags has nothing to aggregate over."""
    priors = EmpiricalPriors(
        loan_relationship={("old-norse", "english", "pre", "social", 950): {"khuz:karl": 4.0}}
    )
    result = baseline_score_pack(
        "khuz:karl",
        [],
        pack=_pack(),
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(social=0.5),
    )
    assert result == 0.0


def test_baseline_pack_skips_zero_weight_tags():
    """Tags the request weights at 0 short-circuit out of the per-tag
    loop, matching baseline_score_native's behavior."""
    priors = EmpiricalPriors(
        loan_relationship={
            ("old-norse", "english", "pre", "social", 950): {"khuz:karl": 4.0},
            ("old-norse", "english", "pre", "magic", 950): {"khuz:karl": 99.0},
        }
    )
    result = baseline_score_pack(
        "khuz:karl",
        ["social", "magic"],
        pack=_pack(),
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(social=0.5, magic=0.0),
    )
    # magic short-circuits; only social contributes: 0.5 * 4.0 = 2.0
    assert result == pytest.approx(2.0)


# ---------- baseline_score (full: native + per-pack scaled) -------------


def test_baseline_full_no_packs_equals_native():
    priors = EmpiricalPriors(native={("english", "pre", "plant", 950): {"old-english:āc": 5.0}})
    result = baseline_score(
        "old-english:āc",
        ["plant"],
        culture="english",
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(plant=0.7),
    )
    assert result == pytest.approx(0.7 * 5.0)


def test_baseline_full_with_pack_adds_weighted_pack_contribution():
    priors = EmpiricalPriors(
        native={("english", "pre", "social", 950): {"old-english:eorl": 3.0}},
        loan_relationship={("old-norse", "english", "pre", "social", 950): {"khuz:karl": 4.0}},
    )
    # Lemma carries "social" tag; gets weighted lookup in BOTH native
    # (3.0 * 0.5) and the pack's loan_relationship (4.0 * 0.5 * pack.weight 0.5).
    result = baseline_score(
        "common-lemma-ref",
        ["social"],  # unrelated lemma_ref to test pack-side lookup
        culture="english",
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(social=0.5),
        packs=[_pack(weight=0.5)],
    )
    # Native lookup for "common-lemma-ref": not in any cell → 0
    # Pack lookup for "common-lemma-ref" against (ON,english) at social/pre/950:
    #   not in the cell either → 0
    # Total = 0 (the test uses an unrelated lemma to isolate the
    # composition arithmetic from data-presence).
    assert result == 0.0


def test_baseline_full_pack_weight_scales_pack_contribution():
    """A pack with weight=0.5 contributes half as much as weight=1.0."""
    priors = EmpiricalPriors(
        loan_relationship={("old-norse", "english", "pre", "social", 950): {"khuz:karl": 4.0}}
    )
    # Pack at full weight
    r_full = baseline_score(
        "khuz:karl",
        ["social"],
        culture="english",
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(social=0.5),
        packs=[_pack(weight=1.0)],
    )
    r_half = baseline_score(
        "khuz:karl",
        ["social"],
        culture="english",
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(social=0.5),
        packs=[_pack(weight=0.5)],
    )
    # Both have native=0 (no native cell). Pack contribution:
    #   r_full = 1.0 * 0.5 * 4.0 = 2.0
    #   r_half = 0.5 * 0.5 * 4.0 = 1.0
    assert r_full == pytest.approx(2.0)
    assert r_half == pytest.approx(1.0)


def test_baseline_full_multi_pack_sums():
    """Multiple admitted packs sum on the baseline axis."""
    priors = EmpiricalPriors(
        loan_relationship={
            ("old-norse", "english", "pre", "social", 950): {"old-norse:karl": 4.0},
            ("old-french", "english", "pre", "social", 950): {"old-norse:karl": 2.0},
        }
    )
    pack_a = PackOverlay(
        pack_name="a", template_donor="old-norse", template_recipient="english", weight=1.0
    )
    pack_b = PackOverlay(
        pack_name="b",
        template_donor="old-french",
        template_recipient="english",
        weight=1.0,
    )
    result = baseline_score(
        "old-norse:karl",
        ["social"],
        culture="english",
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights=_request_tag_weights(social=0.5),
        packs=[pack_a, pack_b],
    )
    # pack_a: 1.0 * 0.5 * 4.0 = 2.0
    # pack_b: 1.0 * 0.5 * 2.0 = 1.0
    # native: 0
    # total: 3.0
    assert result == pytest.approx(3.0)


# ---------- aggregate_score --------------------------------------------


def test_aggregate_zero_weights_zero_inputs():
    w = ScoringWeights(phon_w=0.0, sem_w=0.0, pos_w=0.0, base_w=0.0)
    assert aggregate_score(phon=1, sem=1, pos=1, baseline=1, weights=w) == 0.0


def test_aggregate_default_weights_sum_axes():
    """Default ScoringWeights is 1.0 per axis; the aggregator sums."""
    w = ScoringWeights()
    assert aggregate_score(phon=0.2, sem=0.3, pos=0.1, baseline=0.4, weights=w) == pytest.approx(
        1.0
    )


def test_aggregate_per_axis_weight_scales_independently():
    """Verifies the formula is component-wise weighted sum."""
    w = ScoringWeights(phon_w=2.0, sem_w=0.5, pos_w=0.0, base_w=1.0)
    # 2.0 * 0.2 + 0.5 * 0.4 + 0.0 * 0.6 + 1.0 * 0.3 = 0.4 + 0.2 + 0 + 0.3 = 0.9
    assert aggregate_score(phon=0.2, sem=0.4, pos=0.6, baseline=0.3, weights=w) == pytest.approx(
        0.9
    )


def test_aggregate_negative_axes_pull_score_down():
    """A lemma whose phonology OPPOSES the register contributes
    negatively to the composed score (NOT just zero). Pins the
    canonical signed-arithmetic semantics."""
    w = ScoringWeights()
    assert aggregate_score(phon=-0.5, sem=0.3, pos=0.0, baseline=0.0, weights=w) == pytest.approx(
        -0.2
    )


# ---------- score (top-level integration) -------------------------------


def _gate(culture: str = "english") -> EligibilityGate:
    return EligibilityGate(culture=culture)


def _request(register: RegisterEffect, weights: ScoringWeights | None = None) -> RequestVector:
    return RequestVector(
        gate=_gate(),
        register=register,
        weights=weights or ScoringWeights(),
    )


def test_score_pure_phon_register():
    """Register with phon-only weights, default ScoringWeights → score
    reflects phon axis only (sem=pos=baseline=0)."""
    lemma_phon = PhonologicalVector(cluster_density=0.6)
    register = RegisterEffect(name="harsh", phonological={"cluster_density": 0.5})
    request = _request(register)
    priors = EmpiricalPriors()
    result = score(
        "old-english:tūn",
        [],  # no tags → sem + baseline collapse to 0
        lemma_phon,
        request=request,
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
    )
    # phon: 0.6 * 0.5 = 0.30; default phon_w=1.0; everything else 0
    assert result == pytest.approx(0.30)


def test_score_combined_axes_compose():
    """Lemma scoring across phon, sem, pos, baseline simultaneously."""
    lemma_phon = PhonologicalVector(cluster_density=0.5)
    register = RegisterEffect(
        name="combined",
        phonological={"cluster_density": 0.4},
        semantic_tags={"plant": 0.6},
        position_bias={"pre": 0.2},
    )
    priors = EmpiricalPriors(native={("english", "pre", "plant", 950): {"old-english:āc": 2.0}})
    request = _request(register)
    result = score(
        "old-english:āc",
        ["plant"],
        lemma_phon,
        request=request,
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
    )
    # phon  = 0.5 * 0.4 = 0.20
    # sem   = 0.6 (single matching tag)
    # pos   = 0.2 (pre in register's bias)
    # base  = 0.6 (request tag weight) * 2.0 (priors) = 1.20
    # total = 1.0 * 0.20 + 1.0 * 0.6 + 1.0 * 0.2 + 1.0 * 1.20 = 2.20
    assert result == pytest.approx(2.20)


def test_score_is_seed_stable_across_calls():
    """Same inputs → identical floats. No hash randomization, no
    iteration order, no datetime."""
    lemma_phon = PhonologicalVector(cluster_density=0.5, vowel_final_bias=-0.2)
    register = RegisterEffect(
        name="r",
        phonological={"cluster_density": 0.3, "vowel_final_bias": -0.4},
        semantic_tags={"plant": 0.6, "tree": 0.3},
        position_bias={"pre": 0.2},
    )
    priors = EmpiricalPriors(
        native={
            ("english", "pre", "plant", 950): {"ref": 2.0},
            ("english", "pre", "tree", 950): {"ref": 3.0},
        }
    )
    request = _request(register)
    a = score(
        "ref",
        ["plant", "tree"],
        lemma_phon,
        request=request,
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
    )
    b = score(
        "ref",
        ["plant", "tree"],
        lemma_phon,
        request=request,
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
    )
    assert a == b


def test_score_realism_zero_collapses_to_register_driven():
    """ScoringWeights(base_w=0) means baseline doesn't contribute —
    score becomes purely register-driven (phon + sem + pos)."""
    lemma_phon = PhonologicalVector(cluster_density=0.5)
    register = RegisterEffect(
        name="combined",
        phonological={"cluster_density": 0.4},
        semantic_tags={"plant": 0.6},
    )
    priors = EmpiricalPriors(native={("english", "pre", "plant", 950): {"old-english:āc": 999.0}})
    request = RequestVector(
        gate=_gate(),
        register=register,
        weights=ScoringWeights(base_w=0.0),  # realism = 0
    )
    result = score(
        "old-english:āc",
        ["plant"],
        lemma_phon,
        request=request,
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
    )
    # phon: 0.2, sem: 0.6, pos: 0, baseline: 999 * 0.6 = 599.4 but
    # base_w=0 cancels it → total = 0.2 + 0.6 = 0.8
    assert result == pytest.approx(0.8)


def test_score_with_admitted_pack_includes_pack_baseline():
    """Top-level score() with a non-empty packs list exercises the
    baseline_score's pack-side branch. Previously every score() test
    used default packs=() — the integration test set never reached
    the per-pack baseline composition."""
    lemma_phon = PhonologicalVector()  # all zeros — phon_score contributes 0
    register = RegisterEffect(name="r", semantic_tags={"social": 0.5})
    pack = PackOverlay(
        pack_name="khuz",
        template_donor="old-norse",
        template_recipient="english",
        weight=1.0,
    )
    priors = EmpiricalPriors(
        loan_relationship={("old-norse", "english", "pre", "social", 950): {"khuz:karl": 4.0}}
    )
    request = RequestVector(gate=_gate(), register=register, packs=(pack,))
    result = score(
        "khuz:karl",
        ["social"],
        lemma_phon,
        request=request,
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
    )
    # phon: 0
    # sem: 0.5 (single matching tag at request weight 0.5)
    # pos: 0
    # baseline: native (0) + pack.weight (1.0) * 0.5 * 4.0 = 2.0
    # total at default ScoringWeights: 0.5 + 2.0 = 2.5
    assert result == pytest.approx(2.5)


def test_score_with_generator_lemma_tags_is_multi_pass_safe():
    """Pin the contract that lemma_tags can be a generator/iterator
    even though it's consumed by multiple sub-scorers. The fix:
    score() materializes a frozenset once at top before passing to
    sub-scorers. A regression that re-passed the raw iterator
    would silently zero sem_score + baseline_score after the first
    consumer drained it."""
    lemma_phon = PhonologicalVector()
    register = RegisterEffect(
        name="r",
        semantic_tags={"plant": 0.6},
    )
    priors = EmpiricalPriors(native={("english", "pre", "plant", 950): {"old-english:āc": 2.0}})
    request = RequestVector(gate=_gate(), register=register)

    # Tags as a single-pass generator
    def tag_gen():
        yield "plant"

    result = score(
        "old-english:āc",
        tag_gen(),
        lemma_phon,
        request=request,
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
    )
    # sem: 0.6 (plant @ 0.6); baseline: 0.6 * 2.0 = 1.2; total: 1.8
    assert result == pytest.approx(1.8)


def test_score_seed_stable_across_string_set_iteration():
    """The set(lemma_tags) → sum() pattern, if not order-pinned,
    can produce bit-level different floats across Python processes
    because string hashes are PYTHONHASHSEED-randomized and FP
    addition is non-associative. The fix uses sorted(set(...)) for
    deterministic iteration order. This test exercises a multi-tag
    summation against a request with multiple weights — the order
    of summation must be deterministic for the same inputs.

    Pinning here means a regression that drops sorted() would
    immediately show up via comparing the result against the
    expected value computed in sorted-key order."""
    lemma_phon = PhonologicalVector()
    register = RegisterEffect(
        name="r",
        semantic_tags={"alpha": 0.1, "bravo": 0.2, "charlie": 0.3, "delta": 0.4},
    )
    priors = EmpiricalPriors()
    request = RequestVector(gate=_gate(), register=register)
    # Sum in sorted-key order: 0.1 + 0.2 + 0.3 + 0.4 = 1.0
    expected = 0.1 + 0.2 + 0.3 + 0.4
    result = score(
        "x",
        ["delta", "charlie", "bravo", "alpha"],  # arbitrary input order
        lemma_phon,
        request=request,
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
    )
    assert result == pytest.approx(expected)


def test_baseline_native_negative_weight_pulls_score_down():
    """A request that EXCLUDES a tag direction (negative weight) is
    expected to pull baseline DOWN, not short-circuit. The current
    short-circuit guard is `rw == 0.0`, which correctly preserves
    negative weights. Pinning the signed-arithmetic semantics."""
    priors = EmpiricalPriors(native={("english", "pre", "plant", 950): {"old-english:āc": 5.0}})
    result = baseline_score_native(
        "old-english:āc",
        ["plant"],
        culture="english",
        slot_position="pre",
        era_midpoint=950,
        priors=priors,
        request_tag_weights={"plant": -0.5},
    )
    # -0.5 * 5.0 = -2.5
    assert result == pytest.approx(-2.5)
