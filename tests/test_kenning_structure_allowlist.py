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
    with pytest.raises(StructureAllowlistError, match="label must be a string"):
        load_structure_allowlist_from_text("123: {enabled: false}\n")
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


# ---- drift guards + remaining branches ------------------------------------


def test_shipped_allowlist_labels_are_all_producible():
    """Drift guard (the headline coverage gap): every label in the shipped
    structures.yaml must correspond to a REAL producible struct_key. A corrupted
    or renamed disable-label would silently become an ABSENT label → default
    enabled → a structure the operator meant to forbid silently re-enables, with
    no other failing test. This catches that (and a rebuild that retires a
    structure leaving a stale entry)."""
    from wyrd.generators.kenning import CULTURES
    from wyrd.generators.kenning.runtime.proportions import (
        is_structurally_grammatical,
        word_to_key,
    )
    from wyrd.generators.kenning.runtime.runtime_db import get_runtime_db
    from wyrd.generators.kenning.runtime.runtime_db_adapter import proportions_dict_for_culture

    conn = get_runtime_db()
    producible = set()
    for culture in CULTURES:
        for element in proportions_dict_for_culture(conn, culture)["structures"]:
            words = tuple(word_to_key(w) for w in element["words"])
            if is_structurally_grammatical(words):
                producible.add(struct_key_to_label(words))
    orphans = set(sa._load_bundled_cached()) - producible
    assert not orphans, f"structures.yaml labels with no producible struct_key: {sorted(orphans)}"


def test_load_allowlist_rejects_non_mapping_entry():
    with pytest.raises(StructureAllowlistError, match="entry must be a mapping"):
        load_structure_allowlist_from_text('"(bare)": 5\n')


def test_missing_file_degrades_to_empty(monkeypatch):
    """A bundle without structures.yaml → empty allowlist (no filtering), not a
    crash."""

    class _Missing:
        def joinpath(self, *_a):
            return self

        def read_text(self, **_k):
            raise FileNotFoundError

    monkeypatch.setattr(sa.resources, "files", lambda _pkg: _Missing())
    sa._load_bundled_cached.cache_clear()
    try:
        assert sa._load_bundled_cached() == {}
    finally:
        sa._load_bundled_cached.cache_clear()  # don't leak the stub to other tests


def test_name_generator_raises_when_allowlist_disables_everything(monkeypatch):
    """The empty-after-filter guard: if the allowlist disables every structure a
    culture has, NameGenerator raises an operator-attributable error naming
    structures.yaml — not a later weighted_choice([]) crash."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NameGenerator

    monkeypatch.setattr(sa, "_load_bundled_cached", lambda: {"(pre+post)": False})
    with pytest.raises(ValueError, match="structures.yaml"):
        NameGenerator(
            meaning_db={"-a": [Meaning("-a", [], [], {})]},
            meaning_gen=None,
            structs={((("pre",), ("post",)),): 1},
        )


def test_dump_structures_cli_writes_output_file(tmp_path):
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.dump_structures import dump_structures

    out = tmp_path / "structures.yaml"
    result = CliRunner().invoke(dump_structures, ["--output", str(out)])
    assert result.exit_code == 0, result.output
    written = out.read_text(encoding="utf-8")
    assert load_structure_allowlist_from_text(written)["(bare)"] is False
