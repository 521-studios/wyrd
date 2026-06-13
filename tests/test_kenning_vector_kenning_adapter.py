"""Tests for the Kenning vector-scoring knob translator (wyrd-ecjp.5 PR C).

The adapter translates Kenning's existing per-call knobs (culture,
tags, harshness, mood, stratum) into a fully-built RequestVector.
(Era is deliberately absent — D44: it renders, never gates.)
This file pins the translation contract.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.runtime.vector_kenning_adapter import (
    _harshness_to_phonological,
    _mood_specs_to_register_effects,
    build_request_vector,
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


def test_mood_expansion_unparseable_graduation_suffix_bubbles_value_error() -> None:
    """parse_mood_spec raises ValueError on unparseable colon-suffix
    (``harsh:x``). The adapter only catches KeyError (unknown name)
    + re-raises as ValueError; the graduation-suffix ValueError flows
    through unchanged with the per-spec message intact, so operators
    see ``"graduation suffix"`` not the catch-all
    ``"unknown mood; expected one of ..."``. Pins the divergent error-
    path message at the adapter boundary."""
    import pytest

    with pytest.raises(ValueError, match="graduation suffix"):
        _mood_specs_to_register_effects(["harsh:x"])


# ---- build_request_vector ------------------------------------------------


def test_build_request_vector_minimal():
    """The simplest valid call: just a culture. Returns a
    RequestVector with the culture in the gate + the uniform-tag-
    weights "no opinion default" register (every tag in the bundle's
    tag universe at 1.0). Without this default, ``baseline_score_native``
    early-returns 0 on the empty-register branch and the vector path
    collapses to empty — so vector mode would only fire with a mood."""
    rv = build_request_vector(culture="english")
    assert isinstance(rv, RequestVector)
    assert rv.gate.culture == "english"
    assert rv.gate.stratum is None
    assert rv.register.phonological == {}
    # No explicit tags + no mood + no harshness → uniform-tag default.
    # Bundled tag universe is ~50 tags; all carry weight 1.0.
    assert len(rv.register.semantic_tags) > 0
    assert all(v == 1.0 for v in rv.register.semantic_tags.values())
    assert rv.weights == ScoringWeights()  # default 1.0 per axis


def test_build_request_vector_explicit_tags_skips_uniform_default():
    """Operator-supplied ``tags`` opts out of the uniform default —
    only the requested tags get weight 1.0, the rest are absent
    (priors-baseline contributes 0 for tags the operator didn't ask
    for, as the original D36.7 weighting rule requires)."""
    rv = build_request_vector(culture="english", tags=["water", "tree"])
    assert set(rv.register.semantic_tags) == {"water", "tree"}


def test_build_request_vector_mood_skips_uniform_default():
    """A mood expresses semantic interest (via the catalog-effect
    tag expansion). The adapter's own tag dict stays empty for the
    mood-only call; the request register's tags come from the
    composed catalog effects rather than the uniform fallback.

    Critical: a list-typed mood AND a generator-typed mood must
    both produce the same result — the adapter materializes
    ``Iterable[str]`` inputs up front so a generator-typed mood
    isn't exhausted by the first walk."""

    def _mood_generator():
        yield "grim"

    list_rv = build_request_vector(culture="english", mood=["grim"])
    gen_rv = build_request_vector(culture="english", mood=_mood_generator())

    # Both shapes produce identical semantic_tags. If the generator
    # case fell through to the uniform-tag-default branch (which would
    # happen if ``mood`` were walked twice), the list of tags would be
    # ~50; for a grim-mood request it should be the grim-mood-expanded
    # set (~6-8 tags), much smaller.
    assert list_rv.register.semantic_tags == gen_rv.register.semantic_tags
    # And the grim-mood set should be SMALLER than the uniform-default
    # universe — that's how we know the uniform-default branch DIDN'T
    # fire for either shape.
    uniform_default = build_request_vector(culture="english")
    assert len(list_rv.register.semantic_tags) < len(uniform_default.register.semantic_tags)


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


def test_build_request_vector_with_stratum():
    rv = build_request_vector(culture="welsh", stratum="native-welsh")
    assert rv.gate.culture == "welsh"
    assert rv.gate.stratum == "native-welsh"


def test_build_request_vector_era_surface_is_the_record_cutoff_only():
    """D44/D46: the adapter's only era input is the record-entry cutoff;
    the retired attested-inside-window params stay gone."""
    with pytest.raises(TypeError):
        build_request_vector(culture="english", era_min=800)  # type: ignore[call-arg]
    rv = build_request_vector(culture="english")
    assert rv.gate.era_record_cutoff is None
    assert not hasattr(rv.gate, "era_min")
    assert not hasattr(rv.gate, "era_max")
    gated = build_request_vector(culture="english", era_record_cutoff=1500)
    assert gated.gate.era_record_cutoff == 1500


def test_build_request_vector_custom_weights():
    weights = ScoringWeights(phon_w=2.0, sem_w=0.5, pos_w=0.0, base_w=1.0)
    rv = build_request_vector(culture="english", weights=weights)
    assert rv.weights is weights
