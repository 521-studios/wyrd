"""Tests for wyrd-kq7w.2: register-effect catalog loader.

Exercises the YAML loader against the bundled catalog + synthetic
fixtures covering shape / dimension / weight validation.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.registers.effects import (
    RegisterEffectCatalogError,
    _load_bundled_cached,
    available_register_effects,
    get_register_effect,
    load_register_effects,
    load_register_effects_from_text,
    mood_spec_to_legacy_form,
    parse_mood_spec,
)
from wyrd.generators.kenning.vectors.schemas import (
    RegisterEffect,
    compose_register_effects,
)

# ---------- bundled catalog ---------------------------------------------


def test_bundled_catalog_loads():
    """The bundled register_effects.yaml parses without errors."""
    catalog = _load_bundled_cached.__wrapped__()  # bypass lru_cache for isolation
    assert isinstance(catalog, dict)
    assert len(catalog) > 0


def test_bundled_catalog_has_migrated_moods():
    """All five MOODS entries migrated v1: grim, harsh, pastoral,
    devotional, mortuary. Acceptance criterion from wyrd-kq7w.2."""
    catalog = _load_bundled_cached.__wrapped__()
    migrated = {"grim", "harsh", "pastoral", "devotional", "mortuary"}
    assert migrated.issubset(catalog.keys())


def test_bundled_catalog_has_new_2026_05_16_entries():
    """New design entries from the 2026-05-16 discussion: noble,
    mystical, melodic, sinister, ancient, exotic, martial."""
    catalog = _load_bundled_cached.__wrapped__()
    new = {"noble", "mystical", "melodic", "sinister", "ancient", "exotic", "martial"}
    assert new.issubset(catalog.keys())


def test_bundled_catalog_grim_is_tag_driven_phonology_neutral():
    """Migration faithfulness: grim's legacy semantics were
    tag-only (death/military/monster/undead/magic). Phonology
    intentionally stays empty so --register grim,harsh composes
    without double-counting harshness."""
    catalog = _load_bundled_cached.__wrapped__()
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
    catalog = _load_bundled_cached.__wrapped__()
    harsh = catalog["harsh"]
    assert harsh.semantic_tags == {}
    assert harsh.phonological.get("cluster_density", 0) > 0
    assert harsh.phonological.get("final_fortition", 0) > 0


def test_bundled_catalog_round_trips_through_compose():
    """Every catalog entry survives compose_register_effects without
    raising. Pins the contract that catalog values are compose-ready."""
    catalog = _load_bundled_cached.__wrapped__()
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
    catalog = _load_bundled_cached.__wrapped__()
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
    yaml_text = "incomplete:\n  phonological:\n    cluster_density: 0.6\n"
    with pytest.raises(RegisterEffectCatalogError, match="missing fields"):
        load_register_effects_from_text(yaml_text)


def test_load_rejects_entry_with_extra_fields():
    yaml_text = (
        "extra-fields:\n"
        "  phonological: {}\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
        "  bonus_axis: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="unexpected fields"):
        load_register_effects_from_text(yaml_text)


def test_load_rejects_entry_with_non_mapping_value():
    yaml_text = "not-a-mapping: scalar value\n"
    with pytest.raises(RegisterEffectCatalogError, match="catalog entry must be a mapping"):
        load_register_effects_from_text(yaml_text)


def test_load_rejects_entry_with_simultaneous_missing_and_extra_fields():
    """Single-pass error reporting: both 'missing fields [...]' and
    'unexpected fields [...]' parts appear in the same error message
    when both conditions are true for one entry. Without this test
    a regression that short-circuits on first failure goes
    undetected."""
    yaml_text = (
        "scrambled:\n"
        "  phonological: {}\n"
        "  semantic_tags: {}\n"
        # missing position_bias, unexpected bonus_axis
        "  bonus_axis: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError) as exc_info:
        load_register_effects_from_text(yaml_text)
    msg = str(exc_info.value)
    assert "missing fields" in msg
    assert "unexpected fields" in msg
    assert "position_bias" in msg
    assert "bonus_axis" in msg


def test_load_rejects_non_dict_weight_field():
    """Per-entry weight-dict fields must be mappings, not scalars."""
    yaml_text = (
        "badphono:\n"
        "  phonological: 'oops, scalar not dict'\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="must be a mapping"):
        load_register_effects_from_text(yaml_text)


def test_load_accepts_null_weight_field_as_empty():
    """Bare ``semantic_tags:`` (YAML null) is operator shorthand for
    'no semantic contribution' — equivalent to ``semantic_tags: {}``.
    Pin the behavior so a future strictness change doesn't silently
    break existing catalogs."""
    yaml_text = (
        "halfwritten:\n"
        "  phonological: {}\n"
        "  semantic_tags:\n"  # bare key, no value -> null
        "  position_bias: {}\n"
    )
    catalog = load_register_effects_from_text(yaml_text)
    assert catalog["halfwritten"].semantic_tags == {}


def test_load_rejects_non_string_key_in_weight_dict():
    """Numeric YAML keys produce int dict keys; the loader rejects
    them with a 'non-empty strings' error so the typo surfaces."""
    yaml_text = (
        "numerickey:\n  phonological: {}\n  semantic_tags:\n    123: 0.5\n  position_bias: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="non-empty strings"):
        load_register_effects_from_text(yaml_text)


def test_load_rejects_duplicate_top_level_key():
    """Operator copy-paste mistake on a register-effect block: two
    entries with the same name. PyYAML's default is silent last-wins;
    this catalog uses a custom loader that raises loudly so the data
    loss surfaces."""
    yaml_text = (
        "twin:\n"
        "  phonological: {}\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
        "twin:\n"
        "  phonological:\n"
        "    cluster_density: 0.5\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="duplicate key 'twin'"):
        load_register_effects_from_text(yaml_text)


def test_load_rejects_duplicate_nested_key_in_weight_dict():
    """Same fail-loud behavior applies to nested mappings too —
    duplicate keys in a per-effect weight dict are an operator
    mistake."""
    yaml_text = (
        "dupnested:\n"
        "  phonological:\n"
        "    cluster_density: 0.5\n"
        "    cluster_density: 0.7\n"
        "  semantic_tags: {}\n"
        "  position_bias: {}\n"
    )
    with pytest.raises(RegisterEffectCatalogError, match="duplicate key 'cluster_density'"):
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


def test_load_from_nonexistent_path_raises_file_not_found(tmp_path):
    """Pin the exception surface for nonexistent paths so a future
    wrapper-change is visible."""
    bogus = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_register_effects(bogus)


def test_load_from_malformed_yaml_raises_yaml_error(tmp_path):
    """Pin the exception surface for malformed YAML."""
    import yaml

    catalog_path = tmp_path / "broken.yaml"
    catalog_path.write_text("not: valid: yaml: at: all:\n")
    with pytest.raises(yaml.YAMLError):
        load_register_effects(catalog_path)


def test_load_explicit_path_bypasses_cache(tmp_path):
    """``load_register_effects(path)`` always reads from disk — no
    caching layer that would serve stale content if the same path is
    rewritten between calls. Pins the docstring's cache-bypass claim."""
    catalog_path = tmp_path / "iter.yaml"
    catalog_path.write_text(
        "iter:\n  phonological:\n    cluster_density: 0.3\n"
        "  semantic_tags: {}\n  position_bias: {}\n"
    )
    first = load_register_effects(catalog_path)
    assert first["iter"].phonological == {"cluster_density": 0.3}

    catalog_path.write_text(
        "iter:\n  phonological:\n    cluster_density: 0.7\n"
        "  semantic_tags: {}\n  position_bias: {}\n"
    )
    second = load_register_effects(catalog_path)
    assert second["iter"].phonological == {"cluster_density": 0.7}


# ---------- no-arg / mutation-isolation contracts -----------------------


def test_load_register_effects_no_arg_returns_bundled():
    """The production call (no arg) returns the bundled catalog —
    must contain the migrated MOODS entries."""
    catalog = load_register_effects()
    assert "grim" in catalog
    assert "harsh" in catalog


def test_load_register_effects_returns_fresh_copy_per_call():
    """Defensive copy: callers can mutate one returned catalog
    without polluting the next call's result. Pins the mutation-
    isolation contract that protects the @lru_cache from caller-side
    mutation hazards."""
    first = load_register_effects()
    first["grim"].semantic_tags["bogus_inserted_tag"] = 0.9
    first["polluted_top_level"] = first["grim"]

    second = load_register_effects()
    assert "bogus_inserted_tag" not in second["grim"].semantic_tags
    assert "polluted_top_level" not in second


def test_get_register_effect_returns_fresh_copy_per_call():
    """Same mutation-isolation contract for the single-effect API."""
    first = get_register_effect("grim")
    first.semantic_tags["bogus_inserted_tag"] = 0.9

    second = get_register_effect("grim")
    assert "bogus_inserted_tag" not in second.semantic_tags


# ---------- multi-effect compose ----------------------------------------


def test_multi_effect_compose_preserves_distinct_contributions(tmp_path):
    """The headline `--register grim,harsh` use case: composing two
    bundled effects sums their non-overlapping contributions
    (grim is tag-only; harsh is phonology-only) without clamp."""
    catalog = load_register_effects()
    composed = compose_register_effects([catalog["grim"], catalog["harsh"]])
    # grim's semantic tags survive.
    assert composed.semantic_tags.get("death", 0) > 0
    # harsh's phonological weights survive.
    assert composed.phonological.get("cluster_density", 0) > 0
    assert composed.phonological.get("final_fortition", 0) > 0


def test_multi_effect_compose_clamps_overlapping_contributions(tmp_path):
    """Two synthetic effects with the SAME phonological dim sum + clamp
    to the [-1, +1] boundary. Pins the clamp contract end-to-end
    through the catalog → compose pipeline."""
    yaml_text = (
        "a:\n  phonological:\n    cluster_density: 0.8\n"
        "  semantic_tags: {}\n  position_bias: {}\n"
        "b:\n  phonological:\n    cluster_density: 0.9\n"
        "  semantic_tags: {}\n  position_bias: {}\n"
    )
    path = tmp_path / "clamp.yaml"
    path.write_text(yaml_text)
    catalog = load_register_effects(path)
    composed = compose_register_effects([catalog["a"], catalog["b"]])
    # 0.8 + 0.9 = 1.7, clamped to 1.0.
    assert composed.phonological["cluster_density"] == pytest.approx(1.0)


# ---------- catalog content audit (exotic sign convention) --------------


def test_bundled_catalog_exotic_uses_back_vowels():
    """exotic biases vowel_backness POSITIVE (+1=back, -1=front per
    vectors.schemas convention). The catalog comment claims 'back
    vowels'; the weight must match — a sign flip silently changes
    the register's intended phonological color."""
    catalog = _load_bundled_cached.__wrapped__()
    exotic = catalog["exotic"]
    assert exotic.phonological.get("vowel_backness", 0) > 0, (
        "exotic.vowel_backness must be positive to match the catalog "
        "comment claim that exotic biases toward back vowels"
    )


def test_bundled_catalog_noble_grand_authoritative_signs():
    """noble's v1.1 vowel additions ground the Ohala 1994 frequency-
    code size-dominance signal: low-back vowels = large vocalizer =
    authoritative. Pinning the signs (vowel_height NEGATIVE for low,
    vowel_backness POSITIVE for back) prevents a silent flip turning
    noble into "small/ethereal" — which is mystical's territory."""
    catalog = _load_bundled_cached.__wrapped__()
    noble = catalog["noble"]
    assert noble.phonological.get("vowel_height", 0) < 0, (
        "noble.vowel_height must be negative to ground the Ohala "
        "frequency-code 'large vocalizer' signal (low vowels). See "
        "REGISTERS.md §5 noble."
    )
    assert noble.phonological.get("vowel_backness", 0) > 0, (
        "noble.vowel_backness must be positive to ground the Ohala "
        "frequency-code 'large vocalizer' signal (back vowels)."
    )


def test_bundled_catalog_mystical_small_ethereal_signs():
    """mystical's v1.1 vowel additions ground the Ohala 1994
    frequency-code size signal in the opposite direction from noble:
    high-front vowels = small vocalizer = ethereal/light. Pinning
    the signs (vowel_height POSITIVE for high, vowel_backness NEGATIVE
    for front) prevents a silent flip turning mystical into the
    grand/authoritative noble register."""
    catalog = _load_bundled_cached.__wrapped__()
    mystical = catalog["mystical"]
    assert mystical.phonological.get("vowel_height", 0) > 0, (
        "mystical.vowel_height must be positive to ground the Ohala "
        "frequency-code 'small/ethereal' signal (high vowels). See "
        "REGISTERS.md §5 mystical."
    )
    assert mystical.phonological.get("vowel_backness", 0) < 0, (
        "mystical.vowel_backness must be negative to ground the "
        "Ohala frequency-code 'small/ethereal' signal (front vowels)."
    )


def test_bundled_catalog_sinister_menacing_signs():
    """sinister's v1.1 vowel additions ground the Ohala 1994
    frequency-code dominance signal — same direction as noble
    (low-back = large) but in service of "menacing dread" rather than
    "authoritative grandeur" (Grawunder & Winter 2021 meta-analysis:
    dominance/threat = LOW acoustic frequency). Pinning the signs
    prevents a silent flip turning sinister into the small/sharp
    direction, which would be mystical-eerie territory."""
    catalog = _load_bundled_cached.__wrapped__()
    sinister = catalog["sinister"]
    assert sinister.phonological.get("vowel_height", 0) < 0, (
        "sinister.vowel_height must be negative to ground the Ohala "
        "frequency-code 'large/dominant' signal (low vowels) for "
        "menacing dread. See REGISTERS.md §5 sinister."
    )
    assert sinister.phonological.get("vowel_backness", 0) > 0, (
        "sinister.vowel_backness must be positive to ground the "
        "Ohala frequency-code 'large/dominant' signal (back vowels)."
    )


# ---------- mood-spec resolution (wyrd-kq7w.3) -----------------------------


def test_available_register_effects_returns_sorted_catalog_names() -> None:
    """available_register_effects surfaces the operator-facing mood
    vocabulary without the legacy MOODS-dict dependency. Sorted so
    --help / JSON-schema descriptions read deterministically."""
    names = available_register_effects()
    assert "grim" in names
    assert "harsh" in names
    assert names == sorted(names)


def test_parse_mood_spec_bare_name_returns_catalog_effect() -> None:
    """``"grim"`` resolves to the catalog's grim effect at full
    strength (multiplier 1.0). Returns a fresh copy — caller can
    mutate without polluting the cached catalog."""
    effect = parse_mood_spec("grim")
    assert isinstance(effect, RegisterEffect)
    assert effect.name == "grim"
    assert effect.semantic_tags == get_register_effect("grim").semantic_tags


def test_parse_mood_spec_graduated_form_scales_weights() -> None:
    """``"harsh:0.5"`` scales every dim of the catalog's harsh effect
    by 0.5 (RegisterEffect.scaled semantics)."""
    effect = parse_mood_spec("harsh:0.5")
    full = get_register_effect("harsh")
    assert effect.phonological["cluster_density"] == full.phonological["cluster_density"] * 0.5


def test_parse_mood_spec_unknown_name_raises_key_error() -> None:
    """``KeyError`` matches the catalog's get_register_effect contract.
    Callers that want a ValueError use mood_spec_to_legacy_form or
    catch + re-raise themselves."""
    with pytest.raises(KeyError, match="unknown register effect"):
        parse_mood_spec("whimsical")


def test_parse_mood_spec_unparseable_graduation_raises_value_error() -> None:
    """``"harsh:x"`` — typo'd graduation suffix is loud-failure, not
    silent 1.0 fallback."""
    with pytest.raises(ValueError, match="graduation suffix"):
        parse_mood_spec("harsh:x")


def test_mood_spec_to_legacy_form_grim_returns_tags_zero_harshness() -> None:
    """grim is a tag-driven catalog effect (no phonological dims),
    so the legacy translation returns its semantic_tag keys + 0
    harshness."""
    tags, harshness = mood_spec_to_legacy_form("grim")
    assert "death" in tags
    assert "military" in tags
    assert harshness == 0.0


def test_mood_spec_to_legacy_form_harsh_returns_cluster_density_proxy() -> None:
    """harsh's harshness scalar derives from the catalog's
    cluster_density dim (0.6). Slightly softer than the legacy
    MOODS hardcoded 1.0 — the bit-stability gate accepts this
    'distribution match within tolerance' drift."""
    tags, harshness = mood_spec_to_legacy_form("harsh")
    # harsh's catalog semantic_tags is empty, so no tags surface.
    assert tags == []
    assert harshness == 0.6


def test_mood_spec_to_legacy_form_graduated_harsh_scales_harshness() -> None:
    """harsh:0.5 = catalog's harsh.cluster_density (0.6) * 0.5 = 0.3."""
    _, harshness = mood_spec_to_legacy_form("harsh:0.5")
    assert harshness == 0.3


def test_mood_spec_to_legacy_form_unknown_name_raises_value_error() -> None:
    """ValueError preserves the legacy _apply_mood error shape so
    callers grep-matching ``"unknown mood"`` stay bug-compatible
    after the rip-and-replace."""
    with pytest.raises(ValueError, match="unknown mood"):
        mood_spec_to_legacy_form("whimsical")


# ---------- wyrd-we1u: per-weight tier metadata --------------------------


def test_bare_float_weight_loads_with_empty_tier_dict() -> None:
    """Back-compat: bare-float weights (the pre-we1u shape) carry no
    tier metadata. The parallel tier dicts stay empty rather than
    being populated with every-key-mapped-to-untagged bookkeeping —
    consumers default missing keys to 'untagged' via tier_for."""
    catalog = load_register_effects_from_text(
        """
plain:
  phonological: {cluster_density: 0.5}
  semantic_tags: {death: 0.3}
  position_bias: {}
"""
    )
    effect = catalog["plain"]
    assert effect.phonological == {"cluster_density": 0.5}
    assert effect.phonological_tiers == {}
    assert effect.semantic_tag_tiers == {}


def test_tagged_weight_loads_value_and_tier_into_parallel_dicts() -> None:
    """wyrd-we1u shape: per-weight {value, tier} mappings populate
    both the weight dict and the tier dict. Both halves are
    independently queryable."""
    catalog = load_register_effects_from_text(
        """
tagged:
  phonological:
    cluster_density: {value: 0.6, tier: universal}
    polysyllabic_bias: {value: 0.3, tier: ie-conventional}
  semantic_tags:
    death: {value: 0.7, tier: untagged}
  position_bias: {}
"""
    )
    effect = catalog["tagged"]
    assert effect.phonological == {"cluster_density": 0.6, "polysyllabic_bias": 0.3}
    assert effect.phonological_tiers == {
        "cluster_density": "universal",
        "polysyllabic_bias": "ie-conventional",
    }
    assert effect.semantic_tags == {"death": 0.7}
    assert effect.semantic_tag_tiers == {"death": "untagged"}


def test_mixed_bare_and_tagged_weights_in_one_register() -> None:
    """A register can carry some tier-tagged weights and some bare-
    float weights simultaneously — supports the incremental data-
    migration pattern (tag the universal/grounded weights first;
    untagged weights stay bare while their grounding gets researched)."""
    catalog = load_register_effects_from_text(
        """
mixed:
  phonological:
    cluster_density: {value: 0.6, tier: universal}
    palatalization: 0.2
  semantic_tags: {}
  position_bias: {}
"""
    )
    effect = catalog["mixed"]
    assert effect.phonological == {"cluster_density": 0.6, "palatalization": 0.2}
    # Only the tagged one appears in the tier dict.
    assert effect.phonological_tiers == {"cluster_density": "universal"}
    # tier_for returns the explicit value for the tagged key and the
    # default 'untagged' for the bare-float key.
    assert effect.tier_for("phonological", "cluster_density") == "universal"
    assert effect.tier_for("phonological", "palatalization") == "untagged"


def test_tier_for_returns_untagged_for_missing_key() -> None:
    """tier_for defaults missing keys to 'untagged' across all three
    axes. Consumers querying a register that doesn't carry a given
    dim get the safe default instead of KeyError."""
    catalog = load_register_effects_from_text(
        """
sparse:
  phonological: {}
  semantic_tags: {}
  position_bias: {}
"""
    )
    effect = catalog["sparse"]
    assert effect.tier_for("phonological", "cluster_density") == "untagged"
    assert effect.tier_for("semantic_tags", "death") == "untagged"
    assert effect.tier_for("position_bias", "pre") == "untagged"


def test_tier_for_unknown_axis_raises_value_error() -> None:
    """tier_for rejects axis-name typos at the boundary so callers
    don't silently lose tier filtering on a misspelled axis."""
    catalog = load_register_effects_from_text(
        """
e:
  phonological: {}
  semantic_tags: {}
  position_bias: {}
"""
    )
    effect = catalog["e"]
    with pytest.raises(ValueError, match="unknown axis"):
        effect.tier_for("semantics", "death")  # typo: should be 'semantic_tags'


def test_unknown_tier_value_raises_at_parse() -> None:
    """Typo'd tier (universla → universal) is loud-failure. A silent
    fallback to 'untagged' would drop the grounding metadata an
    editor intended to assign, surfacing only as a tier-filter miss
    much later."""
    with pytest.raises(RegisterEffectCatalogError, match="unknown tier"):
        load_register_effects_from_text(
            """
typo:
  phonological:
    cluster_density: {value: 0.6, tier: universla}
  semantic_tags: {}
  position_bias: {}
"""
        )


def test_tier_tagged_weight_missing_value_key_raises() -> None:
    """A per-weight mapping without 'value' is an authoring error
    (almost certainly a copy-paste mistake) rather than 'weight is 0'."""
    with pytest.raises(RegisterEffectCatalogError, match="must include 'value'"):
        load_register_effects_from_text(
            """
bad:
  phonological:
    cluster_density: {tier: universal}
  semantic_tags: {}
  position_bias: {}
"""
        )


def test_tier_tagged_weight_unexpected_key_raises() -> None:
    """Per-weight mapping rejects unknown keys at parse time. Without
    this guard, a typo like 'val' or 'rating' would silently produce
    a missing-value error AFTER the catalog had loaded a partial
    state."""
    with pytest.raises(RegisterEffectCatalogError, match="unexpected per-weight keys"):
        load_register_effects_from_text(
            """
bad:
  phonological:
    cluster_density: {value: 0.6, tier: universal, source: "Whissell 1999"}
  semantic_tags: {}
  position_bias: {}
"""
        )


def test_scaled_preserves_tier_metadata() -> None:
    """Graduation (RegisterEffect.scaled) preserves tiers unchanged —
    scaling a weight by 0.5 doesn't change the scholarly grounding
    behind it. Pins the contract so consumers reading post-scale
    tiers (e.g. audit-trail exporters) see the same tier as the
    pre-scale catalog entry."""
    catalog = load_register_effects_from_text(
        """
e:
  phonological:
    cluster_density: {value: 0.6, tier: universal}
  semantic_tags: {}
  position_bias: {}
"""
    )
    scaled = catalog["e"].scaled(0.5)
    assert scaled.phonological["cluster_density"] == 0.3
    assert scaled.phonological_tiers["cluster_density"] == "universal"


def test_compose_register_effects_drops_tier_metadata() -> None:
    """A composed effect has no source-tier attribution: a dim that
    accumulates from a UNIVERSAL weight + an IE-CONVENTIONAL weight
    has no single tier to report. Consumers that filter by tier
    apply the filter on the per-effect inputs BEFORE composition."""
    catalog = load_register_effects_from_text(
        """
a:
  phonological:
    cluster_density: {value: 0.3, tier: universal}
  semantic_tags: {}
  position_bias: {}
b:
  phonological:
    cluster_density: {value: 0.4, tier: ie-conventional}
  semantic_tags: {}
  position_bias: {}
"""
    )
    composed = compose_register_effects([catalog["a"], catalog["b"]])
    assert composed.phonological["cluster_density"] == 0.7
    # Composed effect deliberately carries no tier metadata.
    assert composed.phonological_tiers == {}
    assert composed.tier_for("phonological", "cluster_density") == "untagged"


def test_bundled_catalog_has_no_tiered_weights_yet() -> None:
    """wyrd-we1u Phase 1: schema + loader land first; the per-weight
    tier migration of ~80 weights against REGISTERS.md prose is a
    separate PR (Phase 2). Pin Phase 1's end-state so the migration
    PR has a clear baseline to compare against."""
    catalog = _load_bundled_cached.__wrapped__()
    for effect in catalog.values():
        # Every weight is still bare-float in YAML; tier dicts are empty.
        assert effect.phonological_tiers == {}, (
            f"{effect.name} has unexpected phonological tiers: {effect.phonological_tiers}"
        )
        assert effect.semantic_tag_tiers == {}
        assert effect.position_bias_tiers == {}


def test_position_bias_axis_loads_tier_tagged_weights() -> None:
    """The third axis (position_bias) was an oversight gap in the
    initial tier-tagged tests — both phonological and semantic_tags
    had explicit tier-tagged YAML coverage but position_bias was only
    ever loaded as {}. Pin the third axis's parser branch so a
    regression in _validate_weight_dict for that axis would surface."""
    catalog = load_register_effects_from_text(
        """
posbias:
  phonological: {}
  semantic_tags: {}
  position_bias:
    pre: {value: 0.5, tier: ie-conventional}
    post: 0.2
"""
    )
    effect = catalog["posbias"]
    assert effect.position_bias == {"pre": 0.5, "post": 0.2}
    assert effect.position_bias_tiers == {"pre": "ie-conventional"}
    assert effect.tier_for("position_bias", "pre") == "ie-conventional"
    assert effect.tier_for("position_bias", "post") == "untagged"


def test_get_register_effect_preserves_nonempty_tier_dicts() -> None:
    """_copy_effect (invoked through get_register_effect) must carry
    the tier dicts through to the returned fresh instance. The
    bundled-catalog test only exercises the empty-tier-dict path, so
    a regression dropping one of the three tier-dict copy lines
    would slip past without this assertion."""
    catalog = load_register_effects_from_text(
        """
multi:
  phonological:
    cluster_density: {value: 0.4, tier: universal}
  semantic_tags:
    death: {value: 0.6, tier: ie-conventional}
  position_bias:
    pre: {value: 0.3, tier: identity-marking}
"""
    )
    # Round through _copy_effect via the catalog dict (the loader's
    # output is already pre-copied; emulate the get_register_effect
    # invariant by importing and calling _copy_effect directly).
    from wyrd.generators.kenning.registers.effects import _copy_effect

    copied = _copy_effect(catalog["multi"])
    assert copied.phonological_tiers == {"cluster_density": "universal"}
    assert copied.semantic_tag_tiers == {"death": "ie-conventional"}
    assert copied.position_bias_tiers == {"pre": "identity-marking"}
    # Mutation isolation: copying the source's tier dict and then
    # mutating it doesn't bleed into the original.
    copied.phonological_tiers["other"] = "untagged"
    assert "other" not in catalog["multi"].phonological_tiers
