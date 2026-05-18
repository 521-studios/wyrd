"""Tests for wyrd-kq7w.2: register-effect catalog loader.

Exercises the YAML loader against the bundled catalog + synthetic
fixtures covering shape / dimension / weight validation.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.register_effects import (
    RegisterEffectCatalogError,
    _load_bundled,
    get_register_effect,
    load_register_effects,
    load_register_effects_from_text,
)
from wyrd.generators.kenning.vector_schemas import (
    RegisterEffect,
    compose_register_effects,
)

# ---------- bundled catalog ---------------------------------------------


def test_bundled_catalog_loads():
    """The bundled register_effects.yaml parses without errors."""
    catalog = _load_bundled.__wrapped__()  # bypass lru_cache for isolation
    assert isinstance(catalog, dict)
    assert len(catalog) > 0


def test_bundled_catalog_has_migrated_moods():
    """All five MOODS entries migrated v1: grim, harsh, pastoral,
    devotional, mortuary. Acceptance criterion from wyrd-kq7w.2."""
    catalog = _load_bundled.__wrapped__()
    migrated = {"grim", "harsh", "pastoral", "devotional", "mortuary"}
    assert migrated.issubset(catalog.keys())


def test_bundled_catalog_has_new_2026_05_16_entries():
    """New design entries from the 2026-05-16 discussion: noble,
    mystical, melodic, sinister, ancient, exotic, martial."""
    catalog = _load_bundled.__wrapped__()
    new = {"noble", "mystical", "melodic", "sinister", "ancient", "exotic", "martial"}
    assert new.issubset(catalog.keys())


def test_bundled_catalog_grim_is_tag_driven_phonology_neutral():
    """Migration faithfulness: grim's legacy semantics were
    tag-only (death/military/monster/undead/magic). Phonology
    intentionally stays empty so --register grim,harsh composes
    without double-counting harshness."""
    catalog = _load_bundled.__wrapped__()
    grim = catalog["grim"]
    assert grim.phonological == {}
    assert "death" in grim.semantic_tags
    assert "monster" in grim.semantic_tags
    assert "undead" in grim.semantic_tags
    assert "magic" in grim.semantic_tags
    assert "military" in grim.semantic_tags


def test_bundled_catalog_harsh_is_phonology_only():
    """Migration faithfulness: harsh's legacy semantics were
    phonology-only (harshness=1.0)."""
    catalog = _load_bundled.__wrapped__()
    harsh = catalog["harsh"]
    assert harsh.semantic_tags == {}
    assert harsh.phonological.get("cluster_density", 0) > 0
    assert harsh.phonological.get("final_fortition", 0) > 0


def test_bundled_catalog_round_trips_through_compose():
    """Every catalog entry survives compose_register_effects without
    raising. Pins the contract that catalog values are compose-ready."""
    catalog = _load_bundled.__wrapped__()
    for name, effect in catalog.items():
        composed = compose_register_effects([effect])
        # Single-effect compose preserves the effect's contents
        # (modulo the [-1, +1] clamp, which the catalog entries
        # respect by construction).
        assert composed.phonological == effect.phonological, name
        assert composed.semantic_tags == effect.semantic_tags, name
        assert composed.position_bias == effect.position_bias, name


def test_bundled_catalog_all_weights_in_range():
    """Defense-in-depth: every weight in the bundled catalog is in
    [-1, +1]. The loader enforces this at parse time but pin it
    explicitly so a future catalog edit that violates the range
    can't be silently merged."""
    catalog = _load_bundled.__wrapped__()
    for name, effect in catalog.items():
        for key, weight in effect.phonological.items():
            assert -1.0 <= weight <= 1.0, f"{name}.phonological.{key}={weight}"
        for key, weight in effect.semantic_tags.items():
            assert -1.0 <= weight <= 1.0, f"{name}.semantic_tags.{key}={weight}"
        for key, weight in effect.position_bias.items():
            assert -1.0 <= weight <= 1.0, f"{name}.position_bias.{key}={weight}"


# ---------- get_register_effect -----------------------------------------


def test_get_register_effect_resolves_known_name():
    grim = get_register_effect("grim")
    assert isinstance(grim, RegisterEffect)
    assert grim.name == "grim"


def test_get_register_effect_raises_on_unknown_name():
    with pytest.raises(KeyError, match="unknown register effect"):
        get_register_effect("nonexistent")


# ---------- shape validation --------------------------------------------


def test_load_empty_text_returns_empty_dict():
    assert load_register_effects_from_text("") == {}


def test_load_explicit_null_returns_empty_dict():
    """`---\n...\n` is YAML for explicit document end with no body."""
    assert load_register_effects_from_text("---\n") == {}


def test_load_rejects_non_mapping_root():
    with pytest.raises(RegisterEffectCatalogError, match="catalog root must be a mapping"):
        load_register_effects_from_text("- not\n- a\n- mapping\n")


def test_load_rejects_entry_missing_required_fields():
    yaml_text = "harsh:\n  phonological:\n    cluster_density: 0.6\n"
    with pytest.raises(RegisterEffectCatalogError, match="missing fields"):
        load_register_effects_from_text(yaml_text)


def test_load_rejects_entry_with_extra_fields():
    yaml_text = (
        "harsh:\n  phonological: {}\n  semantic_tags: {}\n  position_bias: {}\n  bonus_axis: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="unexpected fields"):
        load_register_effects_from_text(yaml_text)


def test_load_rejects_entry_with_non_mapping_value():
    yaml_text = "harsh: not a mapping\n"
    with pytest.raises(RegisterEffectCatalogError, match="catalog entry must be a mapping"):
        load_register_effects_from_text(yaml_text)


# ---------- dimension whitelist (phonological only) ---------------------


def test_load_rejects_unknown_phonological_feature():
    yaml_text = (
        "bogus:\n"
        "  phonological:\n"
        "    not_a_real_feature: 0.5\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="unknown key 'not_a_real_feature'"):
        load_register_effects_from_text(yaml_text)


def test_load_accepts_arbitrary_semantic_tag_keys():
    """Semantic tags are open-set — the meaning DB may grow new tags
    over time and the catalog should be forward-compat with them."""
    yaml_text = (
        "speculative:\n"
        "  phonological: {}\n"
        "  semantic_tags:\n"
        "    royalty: 0.5\n"
        "    deep-future-tag: 0.3\n"
        "  position_bias: {}\n"
    )
    catalog = load_register_effects_from_text(yaml_text)
    assert catalog["speculative"].semantic_tags == {"royalty": 0.5, "deep-future-tag": 0.3}


def test_load_accepts_arbitrary_position_bias_keys():
    """Position keys are free-form strings supporting future
    positions (manorial-affix, locative-phrase, etc.)."""
    yaml_text = (
        "place-trick:\n"
        "  phonological: {}\n"
        "  semantic_tags: {}\n"
        "  position_bias:\n"
        "    manorial-affix: 0.4\n"
        "    locative-phrase: 0.2\n"
    )
    catalog = load_register_effects_from_text(yaml_text)
    assert catalog["place-trick"].position_bias == {
        "manorial-affix": 0.4,
        "locative-phrase": 0.2,
    }


# ---------- weight validation -------------------------------------------


def test_load_rejects_out_of_range_weight():
    yaml_text = (
        "out:\n"
        "  phonological:\n"
        "    cluster_density: 1.5\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="outside"):
        load_register_effects_from_text(yaml_text)


def test_load_rejects_below_range_weight():
    yaml_text = (
        "out:\n"
        "  phonological:\n"
        "    cluster_density: -1.5\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="outside"):
        load_register_effects_from_text(yaml_text)


def test_load_accepts_boundary_weights():
    """Boundary values +1.0 and -1.0 are valid (inclusive range)."""
    yaml_text = (
        "boundary:\n"
        "  phonological:\n"
        "    cluster_density: 1.0\n"
        "    final_fortition: -1.0\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
    )
    catalog = load_register_effects_from_text(yaml_text)
    assert catalog["boundary"].phonological["cluster_density"] == 1.0
    assert catalog["boundary"].phonological["final_fortition"] == -1.0


def test_load_rejects_nan_weight():
    yaml_text = (
        "nany:\n"
        "  phonological:\n"
        "    cluster_density: .nan\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="must be finite"):
        load_register_effects_from_text(yaml_text)


def test_load_rejects_inf_weight():
    yaml_text = (
        "infy:\n"
        "  phonological:\n"
        "    cluster_density: .inf\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="must be finite"):
        load_register_effects_from_text(yaml_text)


def test_load_rejects_non_numeric_weight():
    yaml_text = (
        "stringy:\n"
        "  phonological:\n"
        "    cluster_density: 'oops'\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="must be a number"):
        load_register_effects_from_text(yaml_text)


def test_load_rejects_bool_weight():
    """`bool` is technically a numeric type in Python but it's
    almost certainly a YAML-coercion mistake (True/False/yes/no
    parsed as bool). Reject explicitly so the operator sees the
    type confusion."""
    yaml_text = (
        "boolean:\n"
        "  phonological:\n"
        "    cluster_density: true\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="must be a number"):
        load_register_effects_from_text(yaml_text)


def test_load_rejects_empty_string_key():
    yaml_text = (
        "weirdkey:\n  phonological: {}\n  semantic_tags:\n    '': 0.5\n  position_bias: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="non-empty strings"):
        load_register_effects_from_text(yaml_text)


# ---------- path-based loading ------------------------------------------


def test_load_from_path(tmp_path):
    """File-system load round-trip."""
    yaml_text = (
        "fromfile:\n"
        "  phonological:\n"
        "    cluster_density: 0.5\n"
        "  semantic_tags:\n"
        "    plant: 0.3\n"
        "  position_bias: {}\n"
    )
    catalog_path = tmp_path / "test_catalog.yaml"
    catalog_path.write_text(yaml_text)

    catalog = load_register_effects(catalog_path)

    assert "fromfile" in catalog
    assert catalog["fromfile"].phonological == {"cluster_density": 0.5}
    assert catalog["fromfile"].semantic_tags == {"plant": 0.3}
