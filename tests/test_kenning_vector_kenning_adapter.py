"""Tests for the Kenning vector-scoring knob translator (wyrd-ecjp.5 PR C).

The adapter translates Kenning's existing per-call knobs (culture,
tags, harshness, mood, era, stratum) into a fully-built RequestVector.
This file pins the translation contract.
"""

from __future__ import annotations

from wyrd.generators.kenning.runtime.vector_kenning_adapter import (
    _harshness_to_phonological,
    _mood_specs_to_register_effects,
    build_request_vector,
    era_midpoint_from_range,
)
from wyrd.generators.kenning.vectors.schemas import (
    RegisterEffect,
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


# ---- _mood_specs_to_register_effects (wyrd-kq7w.3) ------------------------


def test_mood_expansion_returns_catalog_effect_for_grim() -> None:
    """'grim' resolves to the catalog's grim RegisterEffect with the
    catalog's semantic-tag weights intact (no MOODS-style key-drop).
    Verifies the catalog → effect path replaces the legacy
    MOODS-dict tag-union extraction."""
    effects = _mood_specs_to_register_effects(["grim"])
    assert len(effects) == 1
    effect = effects[0]
    assert isinstance(effect, RegisterEffect)
    assert effect.name == "grim"
    # grim's catalog semantic_tags weights (not key-only) propagate
    # through end-to-end on the vector path.
    assert effect.semantic_tags["death"] > 0
    assert effect.semantic_tags["military"] > 0
    # grim has no phonological dims in the catalog (tag-driven only).
    assert effect.phonological == {}


def test_mood_expansion_returns_catalog_effect_for_harsh() -> None:
    """'harsh' resolves to the catalog's harsh phonological weights.
    Earlier MOODS-dict path collapsed the catalog's 9 phon dims into
    a single harshness scalar — verifying the rip-and-replace preserves
    the catalog's fidelity."""
    effects = _mood_specs_to_register_effects(["harsh"])
    assert len(effects) == 1
    effect = effects[0]
    assert effect.name == "harsh"
    # Catalog harsh carries cluster_density 0.6 + 8 other dims (not the
    # MOODS-dict harshness=1.0 scalar).
    assert effect.phonological["cluster_density"] == 0.6
    assert effect.phonological["final_fortition"] == 0.5
    assert "stop_vs_continuant" in effect.phonological


def test_mood_expansion_graduation_scales_catalog_weights() -> None:
    """'harsh:0.5' resolves to the catalog's harsh effect with every
    weight scaled by 0.5 — matches RegisterEffect.scaled semantics."""
    effects = _mood_specs_to_register_effects(["harsh:0.5"])
    effect = effects[0]
    assert effect.phonological["cluster_density"] == 0.3  # 0.6 * 0.5
    assert effect.phonological["final_fortition"] == 0.25  # 0.5 * 0.5


def test_mood_expansion_multiple_specs_return_list_in_order() -> None:
    """Multiple specs produce a list in input order — caller composes
    them via compose_register_effects which sums + clamps."""
    effects = _mood_specs_to_register_effects(["grim", "harsh:0.4"])
    assert [e.name for e in effects] == ["grim", "harsh"]


def test_mood_expansion_unknown_mood_raises_value_error_with_catalog_names() -> None:
    """Unknown mood names raise ValueError — preserves the legacy
    _apply_mood error shape (callers grepping for 'unknown mood' still
    match). Catalog's KeyError gets re-raised as ValueError so the
    vector path's failure mode stays bug-compatible with the legacy
    proportion-table path."""
    import pytest

    with pytest.raises(ValueError, match=r"unknown mood"):
        _mood_specs_to_register_effects(["unknown_mood"])


def test_mood_expansion_empty_input_returns_empty_list() -> None:
    """No mood specs → no effects → adapter's composed register is
    just the adapter-shape effect (explicit tags + harshness scalar).
    Pins the no-mood bit-stability gate (composed catalog vector is
    zero, no contribution to the request's register)."""
    assert _mood_specs_to_register_effects([]) == []


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
    """wyrd-kq7w.3: mood specs compose into the request register
    component-wise (sum + clamp). Catalog effects replace the legacy
    MOODS-dict harshness-scalar collapse, so the composed register
    reflects each effect's per-dimension weights at their graduated
    strength (NOT the scalar harshness mapping in
    _harshness_to_phonological)."""
    rv = build_request_vector(culture="english", mood=["grim", "harsh:0.5"])
    # grim contributes its catalog semantic_tags (with weights).
    assert "death" in rv.register.semantic_tags
    assert rv.register.semantic_tags["death"] > 0
    # harsh:0.5 is the catalog's harsh effect scaled by 0.5, so
    # cluster_density = 0.6 * 0.5 = 0.3 (NOT 0.5 from the legacy
    # _harshness_to_phonological(0.5) path). The rip-and-replace
    # preserves catalog fidelity over the legacy scalar drift.
    assert rv.register.phonological["cluster_density"] == 0.3
    assert rv.register.phonological["final_fortition"] == 0.25  # 0.5 * 0.5


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
