"""wyrd-c6o1.5: the operator structure allowlist (data/structures.yaml) + the
dump-structures CLI.

Fixture-safe: exercises the committed seed-runtime.db bundle via the runtime
loaders; no live lexicon DB.
"""

from __future__ import annotations

import pytest
import yaml

from wyrd.generators.kenning.runtime import structure_allowlist as sa
from wyrd.generators.kenning.runtime.structure_allowlist import (
    StructureAllowlistError,
    is_structure_enabled,
    load_structure_allowlist_from_text,
    struct_key_to_label,
)

# ---- label encoding -------------------------------------------------------


@pytest.mark.parametrize(
    "struct_key, label",
    [
        (((("bare", "single"),),), "(bare)"),
        (((("bare", "name", "single"),),), "(bare[name])"),
        (((("bare", "saint", "single"),),), "(bare[saint])"),
        (((("pre",), ("post",)),), "(pre+post)"),
        (((("pre",), ("inner",), ("post",)),), "(pre+inner+post)"),
        # two words → space-separated
        (((("bare", "single"),), (("bare", "single"),)), "(bare) (bare)"),
        # qualifier on a multi-morpheme word's morpheme
        (((("pre", "name"), ("post",)),), "(pre[name]+post)"),
    ],
)
def test_struct_key_to_label(struct_key, label):
    assert struct_key_to_label(struct_key) == label


# ---- YAML parsing ---------------------------------------------------------


def test_load_allowlist_enabled_defaults_and_overrides():
    text = """
    "(bare)": {enabled: false}
    "(pre+post)": {enabled: true}
    "(bare) (bare)": {}      # bare/empty entry → enabled
    "(x)":                    # YAML null → enabled
    """
    al = load_structure_allowlist_from_text(text)
    assert al == {
        "(bare)": False,
        "(pre+post)": True,
        "(bare) (bare)": True,
        "(x)": True,
    }


def test_load_allowlist_rejects_bad_shapes():
    with pytest.raises(StructureAllowlistError, match="must be a mapping"):
        load_structure_allowlist_from_text("- not\n- a mapping\n")
    with pytest.raises(StructureAllowlistError, match="must be a bool"):
        load_structure_allowlist_from_text('"(bare)": {enabled: nope}\n')
    with pytest.raises(StructureAllowlistError, match="duplicate"):
        load_structure_allowlist_from_text(
            '"(bare)": {enabled: false}\n"(bare)": {enabled: true}\n'
        )


def test_empty_text_is_empty_allowlist():
    assert load_structure_allowlist_from_text("") == {}
    assert load_structure_allowlist_from_text("# only a comment\n") == {}


# ---- is_structure_enabled (default-when-absent + overrides) ----------------


def test_is_structure_enabled_defaults_true_when_absent(monkeypatch):
    monkeypatch.setattr(sa, "_load_bundled_cached", lambda: {"(pre+post)": False})
    assert is_structure_enabled(((("xyz",),),)) is True  # absent → enabled
    assert is_structure_enabled(((("pre",), ("post",)),)) is False  # disabled


# ---- the SHIPPED structures.yaml ------------------------------------------


def test_shipped_allowlist_disables_lone_dictionary_words():
    """The migrated wyrd-g1hj special-case: lone-word structures ship disabled."""
    assert is_structure_enabled(((("bare", "single"),),)) is False
    assert is_structure_enabled(((("bare", "name", "single"),),)) is False
    # a real compound stays enabled
    assert is_structure_enabled(((("pre",), ("post",)),)) is True


def test_shipped_allowlist_parses_and_is_nonempty():
    al = sa._load_bundled_cached()
    assert al  # the bundle ships the file
    assert al.get("(bare)") is False


# ---- end-to-end: a disabled structure is excluded from generation ---------


def test_disabled_structure_excluded_from_name_generator(monkeypatch):
    """wyrd-c6o1.5: NameGenerator filters its struct pool through the allowlist —
    a disabled (but grammatical) structure never reaches generation."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NameGenerator

    monkeypatch.setattr(sa, "_load_bundled_cached", lambda: {"(pre+post)": False})
    pre_post = ((("pre",), ("post",)),)
    pre_inner_post = ((("pre",), ("inner",), ("post",)),)
    ng = NameGenerator(
        meaning_db={"-a": [Meaning("-a", [], [], {})]},
        meaning_gen=None,
        structs={pre_post: 100, pre_inner_post: 50},
    )
    assert pre_post not in ng.structs  # disabled by the allowlist
    assert pre_inner_post in ng.structs  # absent from allowlist → enabled


# ---- the dump-structures CLI ----------------------------------------------


def test_dump_structures_cli_emits_valid_allowlist():
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.dump_structures import dump_structures

    result = CliRunner().invoke(dump_structures, [])
    assert result.exit_code == 0, result.output
    parsed = yaml.safe_load(result.output)
    assert isinstance(parsed, dict) and parsed
    # the dump seeds the lone-dictionary-word structures disabled
    assert parsed["(bare)"]["enabled"] is False
    assert parsed["(bare[name])"]["enabled"] is False
    # a compound is enabled by default
    assert parsed["(pre+post)"]["enabled"] is True
    # round-trips through the allowlist parser
    assert load_structure_allowlist_from_text(result.output)["(bare)"] is False
