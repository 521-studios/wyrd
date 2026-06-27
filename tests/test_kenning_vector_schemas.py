"""Smoke tests for the vector-driven generator schemas (wyrd-ecjp.1).

These tests exercise the construction + composition contracts of the
typed dataclasses in ``wyrd.generators.kenning.vectors.schemas``. They
do not test runtime scoring (that's Phase 4) — they pin the schema
shapes the downstream phases consume.
"""

from __future__ import annotations

import math
from typing import get_type_hints

import pytest

from wyrd.generators.kenning.vectors.schemas import (
    _DIMENSION_NAMES,
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
    assert s == pytest.approx(0.68)


def test_phonological_vector_dot_uses_extras():
    v = PhonologicalVector(extras={"new_dim": 0.5})
    s = v.dot({"new_dim": 0.4})
    assert s == pytest.approx(0.20)


def test_phonological_vector_dot_ignores_unknown_keys():
    """Forward-compat: a weight referencing a dimension that isn't in
    the v1 named fields AND isn't in extras must be ignored, not
    raise — older lexicons that haven't been re-enriched with newer
    dimensions should still score against newer register effects."""
    v = PhonologicalVector(cluster_density=0.7)
    s = v.dot({"cluster_density": 0.5, "future_dim_we_dont_have_yet": 0.9})
    assert s == pytest.approx(0.35)


def test_phonological_vector_dot_zero_weight_skips():
    """Zero weights short-circuit (perf optimization). Verify zero
    weight doesn't crash on missing dimensions either."""
    v = PhonologicalVector()
    s = v.dot({"cluster_density": 0.0, "made_up_dim": 0.0})
    assert s == 0.0


def test_dimension_names_match_dataclass_fields():
    """``_DIMENSION_NAMES`` (the runtime mirror of the ``PhonologicalFeatureName``
    Literal) is the whitelist ``dot()`` scores against. It MUST stay in lockstep
    with the ``PhonologicalVector`` dataclass's DIMENSION fields — the two are
    hand-maintained on separate sides. A dimension field present on the dataclass
    but absent from the Literal has its weight SILENTLY IGNORED by ``dot()`` (a
    dimension that never contributes to scoring); a Literal name with no field is a
    dead weight key. Dimensions get added over time (wyrd-119p liquid_l_m_n/rhotic_r,
    wyrd-mkry vowel_tenseness), so pin the parity here — a one-sided addition fails
    CI rather than silently producing wrong scores.

    Identify the dimensions by their ``float`` annotation rather than
    'every field except extras': the dataclass deliberately permits future
    NON-dimension metadata fields (a provenance string, a version stamp — see
    ``test_dot_ignores_non_dimension_dataclass_fields``), which ``dot()`` ignores
    and which must not be demanded into the Literal. Only the float dimensions are
    compared."""
    dimension_fields = {n for n, t in get_type_hints(PhonologicalVector).items() if t is float}
    assert dimension_fields == _DIMENSION_NAMES, (
        f"float dimension on the dataclass but NOT scored by dot() (add to the "
        f"Literal): {dimension_fields - _DIMENSION_NAMES}; "
        f"in the Literal but no dataclass field (dead weight key): "
        f"{_DIMENSION_NAMES - dimension_fields}"
    )


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
    assert composed.phonological["cluster_density"] == pytest.approx(0.30)
    assert composed.semantic_tags["death"] == pytest.approx(0.21)


# ---- EligibilityGate ----------------------------------------------------


def test_eligibility_gate_required_culture_only():
    g = EligibilityGate(culture="english")
    assert g.culture == "english"
    assert g.stratum is None
    assert g.allowed_pack_tags == frozenset()
    assert g.excluded_pack_tags == frozenset()


def test_eligibility_gate_carries_record_cutoff():
    """D46: the one era field the gate carries — the record-entry
    cutoff (the era's END year). Defaults to None (no gate)."""
    assert EligibilityGate(culture="english").era_record_cutoff is None
    g = EligibilityGate(culture="english", era_record_cutoff=1500)
    assert g.era_record_cutoff == 1500


def test_eligibility_gate_rejects_nonpositive_record_cutoff():
    """A zero/negative cutoff would silently exclude every dated morpheme
    ('0 names generated', no diagnostic) — raise at construction."""
    with pytest.raises(ValueError, match="era_record_cutoff"):
        EligibilityGate(culture="english", era_record_cutoff=0)
    with pytest.raises(ValueError, match="era_record_cutoff"):
        EligibilityGate(culture="english", era_record_cutoff=-100)


def test_eligibility_gate_has_no_era_fields():
    """D44/D46: the retired attested-inside-WINDOW fields stay gone —
    the only era surface is the D46 record cutoff. Pin so era_min /
    era_max can't quietly return."""
    with pytest.raises(TypeError):
        EligibilityGate(culture="english", era_min=1066, era_max=1300)  # type: ignore[call-arg]
    g = EligibilityGate(culture="english")
    assert not hasattr(g, "era_min")
    assert not hasattr(g, "era_max")


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


# ---- compose: position_bias + semantic_tags clamping --------------------


def test_compose_register_effects_sums_position_bias_component_wise():
    """position_bias is one of the four canonical-axis dicts that
    compose_register_effects sums. The pure-phonological tests above
    don't exercise this path. (pr-test-analyzer round 1 finding.)"""
    a = RegisterEffect(name="a", position_bias={"first": 0.5, "second": -0.3})
    b = RegisterEffect(name="b", position_bias={"first": 0.2, "manorial-affix": 0.4})
    composed = compose_register_effects([a, b])
    assert composed.position_bias == {
        "first": 0.7,
        "second": -0.3,
        "manorial-affix": 0.4,
    }


def test_compose_register_effects_clamps_position_bias():
    a = RegisterEffect(name="a", position_bias={"first": 0.8})
    b = RegisterEffect(name="b", position_bias={"first": 0.7})
    composed = compose_register_effects([a, b])
    # Sum is 1.5; must clamp to 1.0 — same per-dimension invariant
    # the phonological-clamp test pins.
    assert composed.position_bias["first"] == 1.0


def test_compose_register_effects_clamps_semantic_tags():
    """semantic_tags clamping was also untested pre-round-1.
    Pin both positive and negative overflow."""
    a = RegisterEffect(name="a", semantic_tags={"death": 0.8, "monster": -0.6})
    b = RegisterEffect(name="b", semantic_tags={"death": 0.7, "monster": -0.7})
    composed = compose_register_effects([a, b])
    assert composed.semantic_tags["death"] == 1.0
    assert composed.semantic_tags["monster"] == -1.0


# ---- compose: NaN/Inf rejection -----------------------------------------


def test_compose_register_effects_rejects_nan():
    """NaN would pass through `v > 1.0` / `v < -1.0` unchanged and
    propagate as NaN scores downstream, silently corrupting ranking.
    _clamp_in_place raises loudly instead. (silent-failure-hunter
    round 1 finding.)"""

    a = RegisterEffect(name="a", phonological={"cluster_density": math.nan})
    with pytest.raises(ValueError, match=r"NaN/Inf inputs"):
        compose_register_effects([a])


def test_compose_register_effects_rejects_inf():

    a = RegisterEffect(name="a", semantic_tags={"death": math.inf})
    with pytest.raises(ValueError, match=r"NaN/Inf inputs"):
        compose_register_effects([a])


# ---- EmpiricalPriors version equality (frozen dataclass __eq__) ---------


def test_empirical_priors_version_in_equality():
    """The version field is the cache-invalidation key for every
    downstream consumer (D36.9). Frozen dataclass __eq__ includes
    it by default — pin that contract here so a future custom
    __eq__ doesn't silently break cache invalidation."""
    a = EmpiricalPriors(version="v1")
    b = EmpiricalPriors(version="v2")
    assert a != b


def test_empirical_priors_equality_with_same_content():
    """Symmetric pin: same content → equal. Confirms __eq__
    semantics work as expected for both equal and unequal cases."""
    a = EmpiricalPriors(version="v1")
    b = EmpiricalPriors(version="v1")
    assert a == b


# ---- dot() uses explicit dimension set ----------------------------------


def test_dot_ignores_non_dimension_dataclass_fields():
    """The PhonologicalVector.dot method uses the explicit
    _DIMENSION_NAMES whitelist (not __dataclass_fields__). If a
    future maintainer adds a non-dimension metadata field (e.g.
    a provenance string) to PhonologicalVector, AND a register
    effect accidentally targets a key with the same name, dot()
    must not include that metadata in the score. Test by
    confirming 'extras' (a dataclass field that isn't a
    dimension) is never picked up via the named-field branch."""
    v = PhonologicalVector(cluster_density=0.5)
    # 'extras' is a dataclass field BUT not in _DIMENSION_NAMES.
    # A weight keyed on 'extras' would have hit the old
    # __dataclass_fields__-based branch with self.extras (a dict)
    # treated as a numeric — TypeError or worse. Now ignored.
    s = v.dot({"cluster_density": 0.5, "extras": 0.5})
    # Only cluster_density contributes; extras key is unknown.
    assert s == pytest.approx(0.25)
