"""wyrd-c6o1.5 / wyrd-hzqs: the PER-LANGUAGE operator structure allowlist
(data/structures.yaml) + the refresh-merge dump-structures CLI.

Fixture-safe: exercises the committed seed-runtime.db bundle via the runtime
loaders; no live lexicon DB.
"""

from __future__ import annotations

import pytest
import yaml
from click.testing import CliRunner

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
        (((("bare", "single"),), (("bare", "single"),)), "(bare) (bare)"),
        (((("pre", "name"), ("post",)),), "(pre[name]+post)"),
    ],
)
def test_struct_key_to_label(struct_key, label):
    assert struct_key_to_label(struct_key) == label


# ---- YAML parsing (sectioned: {culture: {label: enabled}}) ----------------


def test_load_allowlist_sections_defaults_and_overrides():
    text = """
    english:
      "(bare)": {enabled: false}
      "(pre+post)": {enabled: true}
      "(bare) (bare)": {}      # bare/empty entry → enabled
      "(x)":                    # YAML null → enabled
    scottish:
      "(pre+post)": {enabled: false}
    welsh:                       # empty / null section → {}
    """
    al = load_structure_allowlist_from_text(text)
    assert al == {
        "english": {
            "(bare)": False,
            "(pre+post)": True,
            "(bare) (bare)": True,
            "(x)": True,
        },
        "scottish": {"(pre+post)": False},
        "welsh": {},
    }


def test_load_allowlist_rejects_bad_shapes():
    with pytest.raises(StructureAllowlistError, match="root must be a mapping"):
        load_structure_allowlist_from_text("- not\n- a mapping\n")
    with pytest.raises(StructureAllowlistError, match="section .* must be a mapping"):
        load_structure_allowlist_from_text("english: 5\n")
    with pytest.raises(StructureAllowlistError, match="must be a bool"):
        load_structure_allowlist_from_text('english:\n  "(bare)": {enabled: nope}\n')
    with pytest.raises(StructureAllowlistError, match="entry must be a mapping"):
        load_structure_allowlist_from_text('english:\n  "(bare)": 5\n')
    with pytest.raises(StructureAllowlistError, match="label must be a string"):
        load_structure_allowlist_from_text("english:\n  123: {enabled: false}\n")
    with pytest.raises(StructureAllowlistError, match="culture section name must be a string"):
        load_structure_allowlist_from_text('123:\n  "(bare)": {enabled: false}\n')
    with pytest.raises(StructureAllowlistError, match="duplicate"):
        load_structure_allowlist_from_text(
            'english:\n  "(bare)": {enabled: false}\n  "(bare)": {enabled: true}\n'
        )


def test_old_flat_format_is_rejected():
    """wyrd-hzqs hard-cut: a pre-hzqs flat ``{label: {enabled}}`` file (no culture
    sections) must fail loudly, not parse as a culture named '(bare)'."""
    with pytest.raises(StructureAllowlistError, match="entry must be a mapping"):
        load_structure_allowlist_from_text('"(bare)": {enabled: false}\n')


def test_empty_text_is_empty_allowlist():
    assert load_structure_allowlist_from_text("") == {}
    assert load_structure_allowlist_from_text("# only a comment\n") == {}


# ---- is_structure_enabled (per-culture, default-when-absent) ---------------


def test_is_structure_enabled_keys_on_culture(monkeypatch):
    monkeypatch.setattr(sa, "_load_bundled_cached", lambda: {"english": {"(pre+post)": False}})
    pre_post = ((("pre",), ("post",)),)
    assert is_structure_enabled(((("xyz",),),), "english") is True  # absent label → on
    assert is_structure_enabled(pre_post, "english") is False  # disabled in english
    assert is_structure_enabled(pre_post, "irish") is True  # other culture untouched
    assert is_structure_enabled(pre_post, None) is True  # no culture → fail-open
    assert is_structure_enabled(pre_post, "klingon") is True  # absent section → on


# ---- the SHIPPED structures.yaml ------------------------------------------


def test_shipped_allowlist_disables_lone_dictionary_words():
    """The migrated wyrd-g1hj special-case: lone-word structures ship disabled —
    per culture now."""
    assert is_structure_enabled(((("bare", "single"),),), "english") is False
    # (bare[name]) only occurs in scottish/welsh (where it seeds disabled)
    assert is_structure_enabled(((("bare", "name", "single"),),), "scottish") is False
    assert is_structure_enabled(((("pre",), ("post",)),), "english") is True


def test_shipped_allowlist_parses_nonempty_and_is_nested_readonly():
    al = sa._load_bundled_cached()
    assert al and "english" in al
    assert al["english"].get("(bare)") is False
    # nested read-only (wyrd-i7uy): neither level is mutable
    with pytest.raises(TypeError):
        al["english"]["x"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        al["new"] = {}  # type: ignore[index]


# ---- end-to-end: NameGenerator filters per culture ------------------------


def test_disabled_structure_excluded_per_culture(monkeypatch):
    """A structure disabled for english is filtered from english generation but
    not from a culture that doesn't disable it."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NameGenerator

    monkeypatch.setattr(sa, "_load_bundled_cached", lambda: {"english": {"(pre+post)": False}})
    pre_post = ((("pre",), ("post",)),)
    pre_inner_post = ((("pre",), ("inner",), ("post",)),)
    mdb = {"-a": [Meaning("-a", [], [], {})]}

    eng = NameGenerator(
        meaning_db=mdb,
        meaning_gen=None,
        structs={pre_post: 100, pre_inner_post: 50},
        culture="english",
    )
    assert pre_post not in eng.structs  # disabled for english
    assert pre_inner_post in eng.structs  # absent → enabled

    iri = NameGenerator(
        meaning_db=mdb,
        meaning_gen=None,
        structs={pre_post: 100, pre_inner_post: 50},
        culture="irish",
    )
    assert pre_post in iri.structs  # irish has no such disable → kept


def test_no_culture_means_no_allowlist_filtering(monkeypatch):
    """A legacy NameGenerator built without a culture fails open — the allowlist
    doesn't filter (there's no section to key on)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NameGenerator

    monkeypatch.setattr(sa, "_load_bundled_cached", lambda: {"english": {"(pre+post)": False}})
    pre_post = ((("pre",), ("post",)),)
    ng = NameGenerator(
        meaning_db={"-a": [Meaning("-a", [], [], {})]}, meaning_gen=None, structs={pre_post: 1}
    )  # no culture=
    assert pre_post in ng.structs  # fail-open: not filtered


def test_name_generator_raises_when_allowlist_disables_everything(monkeypatch):
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NameGenerator

    monkeypatch.setattr(sa, "_load_bundled_cached", lambda: {"english": {"(pre+post)": False}})
    with pytest.raises(ValueError, match="structures.yaml"):
        NameGenerator(
            meaning_db={"-a": [Meaning("-a", [], [], {})]},
            meaning_gen=None,
            structs={((("pre",), ("post",)),): 1},
            culture="english",
        )


# ---- the dump-structures CLI (sectioned + refresh-merge + freq) ------------


def _dump_to_file(tmp_path, extra_args=None):
    from wyrd.generators.kenning.cli.dump_structures import dump_structures

    out = tmp_path / "structures.yaml"
    res = CliRunner().invoke(dump_structures, ["--output", str(out), *(extra_args or [])])
    assert res.exit_code == 0, res.output + (res.stderr or "")
    return out, res


def test_dump_emits_sectioned_allowlist_with_seeds(tmp_path):
    out, _ = _dump_to_file(tmp_path)
    parsed = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict) and "english" in parsed and "irish" in parsed
    assert parsed["english"]["(bare)"]["enabled"] is False  # lone-word seeded off
    assert parsed["scottish"]["(bare[name])"]["enabled"] is False  # bare[name] is scottish/welsh
    assert parsed["english"]["(pre+post)"]["enabled"] is True  # compound on
    # round-trips through the parser, per-culture
    al = load_structure_allowlist_from_text(out.read_text(encoding="utf-8"))
    assert al["english"]["(bare)"] is False


def test_dump_rows_carry_frequency_comment_and_are_alpha_sorted(tmp_path):
    out, _ = _dump_to_file(tmp_path)
    lines = out.read_text(encoding="utf-8").splitlines()
    # find the english section's entry lines (indented, quoted label)
    in_eng = False
    eng_labels = []
    for ln in lines:
        if ln == "english:":
            in_eng = True
            continue
        if in_eng:
            if ln and not ln.startswith("  "):
                break  # next section
            if ln.strip().startswith('"'):
                # unquoted label (the dump sorts unquoted; the surrounding quotes
                # would otherwise reorder ' ' vs the closing '"')
                eng_labels.append(ln.split(":")[0].strip().strip('"'))
                # frequency comment: '# <int> (<pct>%)'
                assert "#" in ln and "%" in ln, ln
    assert eng_labels == sorted(eng_labels)  # stable alpha sort
    assert len(eng_labels) > 50


def test_dump_refresh_merge_preserves_false_defaults_new_drops_gone(tmp_path):
    """Re-dump keeps operator falses, seeds new labels by default, drops labels no
    longer in the corpus, and reports new/dropped on stderr."""
    from wyrd.generators.kenning.cli.dump_structures import dump_structures

    # A curation base: one real english structure disabled + one bogus label.
    base = tmp_path / "base.yaml"
    base.write_text(
        "english:\n"
        '  "(pre+inner+post)": {enabled: false}\n'
        '  "(zzz-not-producible)": {enabled: false}\n',
        encoding="utf-8",
    )
    out = tmp_path / "out.yaml"
    res = CliRunner().invoke(dump_structures, ["--output", str(out), "--merge-from", str(base)])
    assert res.exit_code == 0, res.output + (res.stderr or "")
    al = load_structure_allowlist_from_text(out.read_text(encoding="utf-8"))
    eng = al["english"]
    assert eng["(pre+inner+post)"] is False  # operator false PRESERVED
    assert eng["(bare)"] is False  # new label, single-morpheme → seeded false
    assert eng["(pre+post)"] is True  # new label → default on
    assert "(zzz-not-producible)" not in eng  # dropped (not in corpus)
    # stderr summary surfaces the new + dropped signal
    err = res.stderr or ""
    assert "DROPPED: (zzz-not-producible)" in err
    assert "new" in err and "NEW:" in err


def test_dump_merge_from_missing_path_errors(tmp_path):
    """A typo'd --merge-from must error (exists=True), not silently fall back to an
    empty base and wipe all curation on the next write (Gemini review, PR #714)."""
    from wyrd.generators.kenning.cli.dump_structures import dump_structures

    res = CliRunner().invoke(dump_structures, ["--merge-from", str(tmp_path / "typo.yaml")])
    assert res.exit_code != 0
    assert "does not exist" in (res.output + (res.stderr or ""))


def test_dump_to_stdout_smoke():
    from wyrd.generators.kenning.cli.dump_structures import dump_structures

    res = CliRunner().invoke(dump_structures, [])
    assert res.exit_code == 0, res.output + (res.stderr or "")
    # stdout carries the YAML; stderr the per-culture summary
    assert "english:" in res.output
    assert "english:" in (res.stderr or "") or "structures" in (res.stderr or "")


# ---- drift guards ---------------------------------------------------------


def test_shipped_allowlist_labels_are_all_producible_per_culture():
    """Every label in each culture section must be a producible struct_key IN THAT
    culture. A renamed/corrupted disable-label would silently become absent →
    default-enabled (re-enabling a forbidden structure); a retired structure would
    leave a stale entry. This catches both."""
    from wyrd.generators.kenning import CULTURES
    from wyrd.generators.kenning.runtime.proportions import (
        is_structurally_grammatical,
        word_to_key,
    )
    from wyrd.generators.kenning.runtime.runtime_db import get_runtime_db
    from wyrd.generators.kenning.runtime.runtime_db_adapter import proportions_dict_for_culture

    conn = get_runtime_db()
    al = sa._load_bundled_cached()
    # A typo'd section key (e.g. `englis:`) would parse fine but silently fail-open
    # for the intended culture at runtime — and the per-culture loop below would
    # never look at it. Catch unknown section keys here (Gemini review, PR #714).
    unknown = set(al) - set(CULTURES)
    assert not unknown, f"structures.yaml has unknown culture sections: {sorted(unknown)}"
    for culture in CULTURES:
        producible = set()
        for element in proportions_dict_for_culture(conn, culture)["structures"]:
            words = tuple(word_to_key(w) for w in element["words"])
            if is_structurally_grammatical(words):
                producible.add(struct_key_to_label(words))
        orphans = set(al.get(culture, {})) - producible
        assert not orphans, (
            f"{culture}: structures.yaml labels with no struct_key: {sorted(orphans)}"
        )


def test_missing_file_degrades_to_empty(monkeypatch):
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
