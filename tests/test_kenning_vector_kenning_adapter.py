"""Tests for the Kenning vector-scoring knob translator (wyrd-ecjp.5 PR C).

The adapter translates Kenning's existing per-call knobs (culture,
tags, harshness, mood, era, stratum) into a fully-built RequestVector.
This file pins the translation contract.
"""

from __future__ import annotations

from wyrd.generators.kenning.runtime.vector_kenning_adapter import (
    _harshness_to_phonological,
    _mood_to_tags_and_harshness,
    build_request_vector,
    era_midpoint_from_range,
)
from wyrd.generators.kenning.vectors.schemas import (
    RequestVector,
    ScoringWeights,
)

# ---- _harshness_to_phonological ------------------------------------------


def test_harshness_zero_returns_empty_dict():
    """harshness <= 0 short-circuits to empty — no contribution to
    register.phonological, no dot-product cost in the scoring loop."""
    assert _harshness_to_phonological(0.0) == {}
    assert _harshness_to_phonological(-0.5) == {}


def test_harshness_one_returns_full_bias():
    """harshness=1.0 produces the canonical harsh-mood-style bias:
    positive cluster_density / final_fortition, negative
    vowel_final_bias / soft_consonants."""
    result = _harshness_to_phonological(1.0)
    assert result["cluster_density"] == 1.0
    assert result["final_fortition"] == 0.8
    assert result["vowel_final_bias"] == -0.6
    assert result["soft_consonants"] == -0.5


def test_harshness_half_scales_linearly():
    result = _harshness_to_phonological(0.5)
    assert result["cluster_density"] == 0.5
    assert result["final_fortition"] == 0.4
    assert result["vowel_final_bias"] == -0.3
    assert result["soft_consonants"] == -0.25


# ---- _mood_to_tags_and_harshness -----------------------------------------


def test_mood_expansion_adds_tag_mood_tags():
    """'grim' is a tag-bearing mood → its tags union into base_tags."""
    tags, harshness = _mood_to_tags_and_harshness(["grim"], [], 0.0)
    # grim's tags include 'death', 'military', 'monster', 'undead', 'magic'
    assert "death" in tags
    assert "military" in tags
    assert harshness == 0.0  # grim isn't a harshness mood


def test_mood_expansion_max_harshness():
    """'harsh' is a harshness-bearing mood with default 1.0; max'd
    against base_harshness per the legacy composition rule."""
    _, harshness = _mood_to_tags_and_harshness(["harsh"], [], 0.0)
    assert harshness == 1.0


def test_mood_expansion_graduated_form_overrides_default():
    """'harsh:0.5' overrides the mood's default 1.0 with the
    operator-supplied 0.5."""
    _, harshness = _mood_to_tags_and_harshness(["harsh:0.5"], [], 0.0)
    assert harshness == 0.5


def test_mood_expansion_preserves_existing_higher_harshness():
    """Already-higher base_harshness is preserved (max-composition)."""
    _, harshness = _mood_to_tags_and_harshness(["harsh:0.3"], [], 0.7)
    assert harshness == 0.7


def test_mood_expansion_unknown_mood_raises():
    """Unknown mood names raise ValueError — matches the legacy
    _apply_mood's loud-failure behavior. Round-1 reviewer caught that
    the earlier silent-skip drift would let typo'd moods sneak through
    the vector path while the legacy path raised — splitting behavior
    across the two scoring modes."""
    import pytest

    with pytest.raises(ValueError, match=r"unknown mood"):
        _mood_to_tags_and_harshness(["unknown_mood"], ["x"], 0.5)


def test_mood_expansion_dedups_tags():
    """If a mood adds a tag already in base_tags, it doesn't dupe."""
    tags, _ = _mood_to_tags_and_harshness(["grim"], ["death"], 0.0)
    assert tags.count("death") == 1


# ---- build_request_vector ------------------------------------------------


def test_build_request_vector_minimal():
    """The simplest valid call: just a culture. Returns a
    RequestVector with the culture in the gate + empty
    register / default weights."""
    rv = build_request_vector(culture="english")
    assert isinstance(rv, RequestVector)
    assert rv.gate.culture == "english"
    assert rv.gate.era_min is None
    assert rv.gate.era_max is None
    assert rv.gate.stratum is None
    assert rv.register.phonological == {}
    assert rv.register.semantic_tags == {}
    assert rv.weights == ScoringWeights()  # default 1.0 per axis


def test_build_request_vector_with_tags():
    rv = build_request_vector(culture="english", tags=["urban", "water"])
    assert rv.register.semantic_tags == {"urban": 1.0, "water": 1.0}


def test_build_request_vector_with_harshness():
    rv = build_request_vector(culture="english", harshness=0.5)
    assert rv.register.phonological["cluster_density"] == 0.5


def test_build_request_vector_with_mood_expansion():
    """mood expands tags and harshness into the register."""
    rv = build_request_vector(culture="english", mood=["grim", "harsh:0.5"])
    # grim adds tags
    assert "death" in rv.register.semantic_tags
    # harsh:0.5 sets harshness which maps to phonological
    assert rv.register.phonological["cluster_density"] == 0.5


def test_build_request_vector_with_era_and_stratum():
    rv = build_request_vector(
        culture="welsh",
        era_min=1066,
        era_max=1300,
        stratum="native-welsh",
    )
    assert rv.gate.culture == "welsh"
    assert rv.gate.era_min == 1066
    assert rv.gate.era_max == 1300
    assert rv.gate.stratum == "native-welsh"


def test_build_request_vector_custom_weights():
    weights = ScoringWeights(phon_w=2.0, sem_w=0.5, pos_w=0.0, base_w=1.0)
    rv = build_request_vector(culture="english", weights=weights)
    assert rv.weights is weights


# ---- era_midpoint_from_range ---------------------------------------------


def test_era_midpoint_both_bounds():
    assert era_midpoint_from_range(1000, 1200) == 1100


def test_era_midpoint_only_min():
    """Single bound → use it directly (no synthetic midpoint)."""
    assert era_midpoint_from_range(1100, None) == 1100


def test_era_midpoint_only_max():
    assert era_midpoint_from_range(None, 1300) == 1300


def test_era_midpoint_both_none():
    """Open request → 0 (matches the priors-table fallback convention)."""
    assert era_midpoint_from_range(None, None) == 0


def test_era_midpoint_truncates():
    """Integer division — 1001-1002 midpoint is 1001 not 1001.5."""
    assert era_midpoint_from_range(1001, 1002) == 1001
