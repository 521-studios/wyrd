"""Smoke tests for the vector-driven generator schemas (wyrd-ecjp.1).

These tests exercise the construction + composition contracts of the
typed dataclasses in ``wyrd.generators.kenning.vector_schemas``. They
do not test runtime scoring (that's Phase 4) — they pin the schema
shapes the downstream phases consume.
"""

from __future__ import annotations

from wyrd.generators.kenning.vector_schemas import (
    CohesionContext,
    EligibilityGate,
    EmpiricalPriors,
    PackOverlay,
    PhonologicalVector,
    RegisterEffect,
    RequestVector,
    ScoringWeights,
    compose_register_effects,
)

# ---- PhonologicalVector -------------------------------------------------


def test_phonological_vector_defaults_are_zero():
    v = PhonologicalVector()
    assert v.cluster_density == 0.0
    assert v.final_fortition == 0.0
    assert v.vowel_height == 0.0
    assert v.aspirated_voiceless == 0.0
    assert v.extras == {}


def test_phonological_vector_dot_uses_named_fields():
    v = PhonologicalVector(cluster_density=0.8, vowel_final_bias=-0.5)
    s = v.dot({"cluster_density": 0.6, "vowel_final_bias": -0.4})
    # 0.6 * 0.8 + (-0.4) * (-0.5) = 0.48 + 0.20 = 0.68
    assert abs(s - 0.68) < 1e-9


def test_phonological_vector_dot_uses_extras():
    v = PhonologicalVector(extras={"new_dim": 0.5})
    s = v.dot({"new_dim": 0.4})
    assert abs(s - 0.20) < 1e-9


def test_phonological_vector_dot_ignores_unknown_keys():
    """Forward-compat: a weight referencing a dimension that isn't in
    the v1 named fields AND isn't in extras must be ignored, not
    raise — older lexicons that haven't been re-enriched with newer
    dimensions should still score against newer register effects."""
    v = PhonologicalVector(cluster_density=0.7)
    s = v.dot({"cluster_density": 0.5, "future_dim_we_dont_have_yet": 0.9})
    assert abs(s - 0.35) < 1e-9


def test_phonological_vector_dot_zero_weight_skips():
    """Zero weights short-circuit (perf optimization). Verify zero
    weight doesn't crash on missing dimensions either."""
    v = PhonologicalVector()
    s = v.dot({"cluster_density": 0.0, "made_up_dim": 0.0})
    assert s == 0.0


# ---- RegisterEffect + composition --------------------------------------


def test_register_effect_constructs_with_defaults():
    e = RegisterEffect(name="harsh")
    assert e.name == "harsh"
    assert e.phonological == {}
    assert e.semantic_tags == {}
    assert e.position_bias == {}


def test_register_effect_scaled_multiplies_every_component():
    e = RegisterEffect(
        name="harsh",
        phonological={"cluster_density": 0.6, "vowel_final_bias": -0.4},
        semantic_tags={"military": 0.3},
        position_bias={"first": 0.2},
    )
    scaled = e.scaled(0.5)
    assert scaled.name == "harsh"
    assert scaled.phonological == {"cluster_density": 0.3, "vowel_final_bias": -0.2}
    assert scaled.semantic_tags == {"military": 0.15}
    assert scaled.position_bias == {"first": 0.1}


def test_compose_register_effects_sums_component_wise():
    harsh = RegisterEffect(
        name="harsh", phonological={"cluster_density": 0.6, "vowel_final_bias": -0.4}
    )
    grim = RegisterEffect(name="grim", semantic_tags={"death": 0.7, "monster": 0.5})
    composed = compose_register_effects([harsh, grim])
    assert composed.name == "harsh+grim"
    assert composed.phonological == {"cluster_density": 0.6, "vowel_final_bias": -0.4}
    assert composed.semantic_tags == {"death": 0.7, "monster": 0.5}


def test_compose_register_effects_sums_same_key():
    """Two effects that touch the same dimension must accumulate, not
    overwrite. This is the D36.4 multi-effect composition rule."""
    a = RegisterEffect(name="a", phonological={"cluster_density": 0.5})
    b = RegisterEffect(name="b", phonological={"cluster_density": 0.3})
    composed = compose_register_effects([a, b])
    assert composed.phonological == {"cluster_density": 0.8}


def test_compose_register_effects_clamps_to_plus_minus_one():
    """The 'each effect is one unit of pull' invariant: composed
    component weights can't exceed [-1, +1] regardless of how many
    effects pile on. Otherwise a caller can break the per-dimension
    invariant of the catalog by stacking effects."""
    a = RegisterEffect(name="a", phonological={"cluster_density": 0.7})
    b = RegisterEffect(name="b", phonological={"cluster_density": 0.8})
    # Sum is 1.5 — well over the upper bound; must clamp to 1.0.
    composed = compose_register_effects([a, b])
    assert composed.phonological["cluster_density"] == 1.0


def test_compose_register_effects_clamps_negative_overflow():
    a = RegisterEffect(name="a", phonological={"vowel_final_bias": -0.7})
    b = RegisterEffect(name="b", phonological={"vowel_final_bias": -0.6})
    composed = compose_register_effects([a, b])
    # Sum is -1.3; clamps to -1.0.
    assert composed.phonological["vowel_final_bias"] == -1.0


def test_compose_register_effects_empty_list_handled():
    """Zero-effect composition is a valid request shape (the operator
    didn't pass any --register flag); must return a sentinel empty
    effect rather than crash."""
    composed = compose_register_effects([])
    assert composed.name == "<empty>"
    assert composed.phonological == {}
    assert composed.semantic_tags == {}


def test_compose_register_effects_graduation_then_compose():
    """End-to-end graduation: 'harsh:0.5,grim:0.3' is scale-then-compose.
    Pin that the call order produces the expected weight."""
    harsh_half = RegisterEffect(name="harsh", phonological={"cluster_density": 0.6}).scaled(0.5)
    grim_thirty = RegisterEffect(name="grim", semantic_tags={"death": 0.7}).scaled(0.3)
    composed = compose_register_effects([harsh_half, grim_thirty])
    assert abs(composed.phonological["cluster_density"] - 0.30) < 1e-9
    assert abs(composed.semantic_tags["death"] - 0.21) < 1e-9


# ---- EligibilityGate ----------------------------------------------------


def test_eligibility_gate_required_culture_only():
    g = EligibilityGate(culture="english")
    assert g.culture == "english"
    assert g.era_min is None
    assert g.era_max is None
    assert g.stratum is None
    assert g.allowed_pack_tags == frozenset()
    assert g.excluded_pack_tags == frozenset()


def test_eligibility_gate_carries_era_range():
    g = EligibilityGate(culture="english", era_min=1066, era_max=1300)
    assert (g.era_min, g.era_max) == (1066, 1300)


def test_eligibility_gate_carries_pack_tag_filters():
    g = EligibilityGate(
        culture="english",
        allowed_pack_tags=frozenset({"war", "death"}),
        excluded_pack_tags=frozenset({"comic"}),
    )
    assert "war" in g.allowed_pack_tags
    assert "comic" in g.excluded_pack_tags


# ---- PackOverlay --------------------------------------------------------


def test_pack_overlay_defaults_weight_to_one():
    p = PackOverlay(
        pack_name="neo-khuzdul",
        template_donor="old-norse",
        template_recipient="old-english",
    )
    assert p.weight == 1.0


def test_pack_overlay_explicit_weight():
    p = PackOverlay(
        pack_name="neo-khuzdul",
        template_donor="old-norse",
        template_recipient="old-english",
        weight=0.6,
    )
    assert p.weight == 0.6


# ---- ScoringWeights -----------------------------------------------------


def test_scoring_weights_default_all_one():
    w = ScoringWeights()
    assert w.phon_w == 1.0
    assert w.sem_w == 1.0
    assert w.pos_w == 1.0
    assert w.base_w == 1.0


def test_scoring_weights_partial_override():
    """Per D36.3: --realism 0.5 halves base_w. The other axes stay at
    default. This is the operational consequence of the
    'baseline-is-just-one-axis' decision."""
    w = ScoringWeights(base_w=0.5)
    assert w.phon_w == 1.0
    assert w.sem_w == 1.0
    assert w.pos_w == 1.0
    assert w.base_w == 0.5


# ---- RequestVector ------------------------------------------------------


def test_request_vector_minimal_construction():
    """The minimal request: a culture + an empty register effect = a
    no-knobs-pulled native generation request. Default weights, no
    packs."""
    req = RequestVector(
        gate=EligibilityGate(culture="english"),
        register=RegisterEffect(name="<empty>"),
    )
    assert req.gate.culture == "english"
    assert req.register.name == "<empty>"
    assert req.weights == ScoringWeights()
    assert req.packs == ()


def test_request_vector_with_pack_overlay():
    """Native + pack: --culture english --pack neo-khuzdul. Pack
    inherits its template's empirical baseline (D36.4 Option B)."""
    req = RequestVector(
        gate=EligibilityGate(culture="english"),
        register=RegisterEffect(name="<empty>"),
        packs=(
            PackOverlay(
                pack_name="neo-khuzdul",
                template_donor="old-norse",
                template_recipient="old-english",
            ),
        ),
    )
    assert len(req.packs) == 1
    assert req.packs[0].pack_name == "neo-khuzdul"


def test_request_vector_realism_dial_via_weights():
    """--realism 0.5 maps onto base_w=0.5; other axes untouched."""
    req = RequestVector(
        gate=EligibilityGate(culture="english"),
        register=RegisterEffect(name="<empty>"),
        weights=ScoringWeights(base_w=0.5),
    )
    assert req.weights.base_w == 0.5
    assert req.weights.phon_w == 1.0


# ---- EmpiricalPriors ----------------------------------------------------


def test_empirical_priors_default_empty():
    p = EmpiricalPriors()
    assert p.native == {}
    assert p.loan_relationship == {}
    assert p.version == "unversioned"


def test_empirical_priors_native_lookup_shape():
    """Per D36.7: native priors keyed by (culture, position, tag, era).
    Verify the lookup shape works as advertised."""
    p = EmpiricalPriors(
        native={
            ("english", "first", "settlement", 1086): {"Edwulf": 12.0, "Aelfric": 8.0},
        }
    )
    cell = p.native[("english", "first", "settlement", 1086)]
    assert cell["Edwulf"] == 12.0


def test_empirical_priors_loan_relationship_lookup_shape():
    """Per D36.4: pack-baseline lookup uses (donor, recipient, position, tag, era)."""
    p = EmpiricalPriors(
        loan_relationship={
            ("old-norse", "old-english", "first", "settlement", 1086): {"Þorgrim": 4.0},
        }
    )
    cell = p.loan_relationship[("old-norse", "old-english", "first", "settlement", 1086)]
    assert cell["Þorgrim"] == 4.0


def test_empirical_priors_version_carries():
    """The versioned-artifact contract from D36.9: priors regenerations
    carry a version identifier that downstream caches key on."""
    p = EmpiricalPriors(version="2026-05-18-corpus-snapshot-v1")
    assert p.version == "2026-05-18-corpus-snapshot-v1"


# ---- CohesionContext (D17 adapter) --------------------------------------


def test_cohesion_context_defaults():
    """Default context: no picked tags, no novelty. Used for the
    first slot of generation (no neighbors to cohere with yet)."""
    ctx = CohesionContext()
    assert ctx.picked_tags == frozenset()
    assert ctx.novelty == 0.0


def test_cohesion_context_threading():
    """Slots 2+ carry the accumulated tag set from earlier slots,
    plus the user's --novelty knob. This is the D17 integration
    point (D36.5)."""
    ctx = CohesionContext(picked_tags=frozenset({"military", "death"}), novelty=0.3)
    assert "military" in ctx.picked_tags
    assert ctx.novelty == 0.3
