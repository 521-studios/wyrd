"""Tests for pfsrd2-data → wyrd-ami JSONL extractor (wyrd-ld5x).

Synthetic monster JSON fixtures matching the pfsrd2 creature.schema.1.4
shape. No I/O against the real pfsrd2-data corpus — that's an
operator-driven smoke run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.pfsrd2_monster_extractor import (
    _collect_section_texts,
    extract_corpus,
    extract_description,
    extract_input_name,
    extract_monster,
    iter_monster_files,
)

# --- minimal monster JSON builders ------------------------------------


def _monster(
    *,
    name: str,
    family_name: str | None = None,
    family_text: str | None = None,
    family_sections: list[dict[str, Any]] | None = None,
    sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal pfsrd2-shaped monster dict for tests."""
    monster: dict[str, Any] = {
        "name": name,
        "type": "creature",
        "sections": sections or [],
    }
    creature_type: dict[str, Any] = {}
    if family_name is not None:
        family: dict[str, Any] = {"name": family_name}
        if family_text is not None:
            family["text"] = family_text
        if family_sections is not None:
            family["sections"] = family_sections
        creature_type["family"] = family
    monster["stat_block"] = {"creature_type": creature_type}
    return monster


def _section(
    name: str, text: str, *, children: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    s: dict[str, Any] = {"name": name, "type": "section", "text": text}
    if children is not None:
        s["sections"] = children
    return s


# --- extract_input_name ----------------------------------------------


def test_extract_input_name_uses_family_root_for_variant():
    """Bugbear Thug variant: family.name='Bugbear' is the morpheme."""
    monster = _monster(name="Bugbear Thug", family_name="Bugbear")
    assert extract_input_name(monster) == "Bugbear"


def test_extract_input_name_strips_compound_subtype_from_family():
    """Dragon, Red → Dragon (split on comma, take base)."""
    monster = _monster(name="Young Red Dragon", family_name="Dragon, Red")
    assert extract_input_name(monster) == "Dragon"


def test_extract_input_name_rejects_multi_word_family_name():
    """Hell Hound family.name is multi-word; per the single-word rule
    we skip it rather than emit a multi-token morpheme."""
    monster = _monster(name="Hell Hound Pack", family_name="Hell Hound")
    assert extract_input_name(monster) is None


def test_extract_input_name_falls_back_to_top_name_when_no_family():
    """Harpy has no family; the top-level single-word name IS the morpheme."""
    monster = _monster(name="Harpy", family_name=None)
    assert extract_input_name(monster) == "Harpy"


def test_extract_input_name_skips_multi_word_name_with_no_family():
    """'Ancient Black Dragon' with no family is the user's flagged
    problem case — bail rather than guess."""
    monster = _monster(name="Ancient Black Dragon", family_name=None)
    assert extract_input_name(monster) is None


def test_extract_input_name_handles_empty_or_missing_name():
    assert extract_input_name(_monster(name="")) is None
    assert extract_input_name({"sections": []}) is None


def test_extract_input_name_handles_family_dict_without_name_key():
    """Defensive — a malformed family that lacks .name falls through
    to the top-level name path."""
    monster = {
        "name": "Goblin",
        "sections": [],
        "stat_block": {"creature_type": {"family": {}}},
    }
    assert extract_input_name(monster) == "Goblin"


# --- _collect_section_texts (recursive) -------------------------------


def test_collect_section_texts_flat():
    sections = [_section("a", "alpha"), _section("b", "beta")]
    assert _collect_section_texts(sections) == ["alpha", "beta"]


def test_collect_section_texts_nested():
    """The Harpy fixture pattern: a top-level section with one child
    section. Both texts should be collected in document order."""
    sections = [
        _section(
            "Harpy",
            "Harpies are filthy amalgamations.",
            children=[_section("Harpy Exiles", "Most harpies are cruel.")],
        )
    ]
    assert _collect_section_texts(sections) == [
        "Harpies are filthy amalgamations.",
        "Most harpies are cruel.",
    ]


def test_collect_section_texts_skips_empty_text():
    sections = [
        _section("a", "alpha"),
        {"name": "b", "type": "section"},  # no text
        _section("c", "gamma"),
    ]
    assert _collect_section_texts(sections) == ["alpha", "gamma"]


def test_collect_section_texts_handles_none_and_empty():
    assert _collect_section_texts(None) == []
    assert _collect_section_texts([]) == []


# --- extract_description ----------------------------------------------


def test_extract_description_combines_family_and_monster_sections():
    """Real-world pattern: family.text is the rich lore, monster
    sections add per-variant flavor. Both are concatenated."""
    monster = _monster(
        name="Bugbear Thug",
        family_name="Bugbear",
        family_text="These stealthy goblinoid creatures delight in spreading fear.",
        family_sections=[_section("Bugbear Lairs", "Bugbears live in small gangs.")],
        sections=[_section("Bugbear Thug", "The more common bugbear thug specializes...")],
    )
    desc = extract_description(monster)
    assert "stealthy goblinoid creatures" in desc
    assert "Bugbears live in small gangs" in desc
    assert "more common bugbear thug" in desc


def test_extract_description_falls_back_to_monster_sections_when_no_family():
    """Harpy has no family; description comes from monster sections only."""
    monster = _monster(
        name="Harpy",
        sections=[_section("Harpy", "Harpies are filthy amalgamations.")],
    )
    assert extract_description(monster) == "Harpies are filthy amalgamations."


def test_extract_description_uses_family_text_when_monster_sections_empty():
    """The Troll fixture pattern: monster sections is [], family.text
    has the full lore."""
    monster = _monster(
        name="Troll",
        family_name="Troll",
        family_text="Slavering, cruel, practically invincible brutes.",
        sections=[],
    )
    assert extract_description(monster) == "Slavering, cruel, practically invincible brutes."


def test_extract_description_returns_empty_when_nothing_found():
    monster = _monster(name="Mystery", family_name=None, sections=[])
    assert extract_description(monster) == ""


# --- extract_monster --------------------------------------------------


def test_extract_monster_returns_record_for_valid_monster():
    monster = _monster(
        name="Bugbear Thug",
        family_name="Bugbear",
        family_text="These stealthy goblinoid creatures.",
    )
    rec = extract_monster(monster)
    assert rec == {"name": "Bugbear", "description": "These stealthy goblinoid creatures."}


def test_extract_monster_returns_none_for_unusable():
    """No family + multi-word name → None (skip)."""
    monster = _monster(name="Ancient Black Dragon", family_name=None)
    assert extract_monster(monster) is None


# --- iter_monster_files + extract_corpus ------------------------------


def _write_monster_file(dir_path: Path, filename: str, monster: dict[str, Any]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / filename).write_text(json.dumps(monster))


def test_iter_monster_files_yields_sorted(tmp_path: Path) -> None:
    monsters_dir = tmp_path / "monsters"
    _write_monster_file(monsters_dir / "core", "zzz.json", _monster(name="Z"))
    _write_monster_file(monsters_dir / "core", "aaa.json", _monster(name="A"))
    _write_monster_file(monsters_dir / "bestiary", "mmm.json", _monster(name="M"))
    paths = list(iter_monster_files(tmp_path))
    # Sorted globally by full path so the dedupe-first-wins behavior is
    # deterministic across runs.
    assert paths == sorted(paths)


def test_iter_monster_files_returns_empty_when_no_monsters_dir(tmp_path: Path) -> None:
    """If `corpus_root/monsters/` doesn't exist, iter yields nothing."""
    assert list(iter_monster_files(tmp_path)) == []


def test_extract_corpus_dedupes_by_lowercased_name(tmp_path: Path) -> None:
    monsters_dir = tmp_path / "monsters" / "bestiary"
    _write_monster_file(
        monsters_dir,
        "bugbear_thug.json",
        _monster(name="Bugbear Thug", family_name="Bugbear", family_text="thug version"),
    )
    _write_monster_file(
        monsters_dir,
        "bugbear_tormentor.json",
        _monster(
            name="Bugbear Tormentor",
            family_name="Bugbear",
            family_text="tormentor version",
        ),
    )
    records = list(extract_corpus(tmp_path))
    # Both monsters share family.name='Bugbear'; only one record emitted.
    assert len(records) == 1
    assert records[0]["name"] == "Bugbear"
    # First file (lexicographic) wins. bugbear_thug.json sorts before
    # bugbear_tormentor.json so the description is from the thug.
    assert "thug version" in records[0]["description"]


def test_extract_corpus_skips_unusable_monsters(tmp_path: Path) -> None:
    """Multi-word no-family monsters don't appear in the output."""
    monsters_dir = tmp_path / "monsters" / "bestiary"
    _write_monster_file(monsters_dir, "compound.json", _monster(name="Ancient Black Dragon"))
    _write_monster_file(monsters_dir, "harpy.json", _monster(name="Harpy"))
    records = list(extract_corpus(tmp_path))
    names = [r["name"] for r in records]
    assert names == ["Harpy"]


def test_extract_corpus_skips_malformed_json(tmp_path: Path) -> None:
    """A broken JSON file is skipped without crashing the walk."""
    monsters_dir = tmp_path / "monsters" / "bestiary"
    monsters_dir.mkdir(parents=True)
    (monsters_dir / "broken.json").write_text("{not valid json")
    _write_monster_file(monsters_dir, "harpy.json", _monster(name="Harpy"))
    records = list(extract_corpus(tmp_path))
    names = [r["name"] for r in records]
    assert names == ["Harpy"]


def test_extract_corpus_respects_limit(tmp_path: Path) -> None:
    monsters_dir = tmp_path / "monsters" / "bestiary"
    for n in ("Aaa", "Bbb", "Ccc", "Ddd"):
        _write_monster_file(monsters_dir, f"{n.lower()}.json", _monster(name=n))
    records = list(extract_corpus(tmp_path, limit=2))
    assert len(records) == 2
    assert [r["name"] for r in records] == ["Aaa", "Bbb"]


def test_extract_corpus_first_occurrence_wins_on_dedupe(tmp_path: Path) -> None:
    """Two files for the same morpheme produce one record from the
    first-sorted file."""
    monsters_dir = tmp_path / "monsters" / "bestiary"
    _write_monster_file(
        monsters_dir,
        "alpha.json",
        _monster(name="Goblin", sections=[_section("Goblin", "first version")]),
    )
    _write_monster_file(
        monsters_dir,
        "beta.json",
        _monster(name="Goblin", sections=[_section("Goblin", "second version")]),
    )
    records = list(extract_corpus(tmp_path))
    assert len(records) == 1
    assert "first version" in records[0]["description"]


def test_extract_corpus_case_insensitive_dedupe(tmp_path: Path) -> None:
    """'Goblin' and 'goblin' are the same morpheme; only one wins."""
    monsters_dir = tmp_path / "monsters" / "bestiary"
    _write_monster_file(monsters_dir, "alpha.json", _monster(name="Goblin"))
    _write_monster_file(monsters_dir, "beta.json", _monster(name="goblin"))
    records = list(extract_corpus(tmp_path))
    assert len(records) == 1


# --- CLI integration --------------------------------------------------


def test_cli_extract_pfsrd2_monsters_writes_jsonl(tmp_path: Path) -> None:
    """End-to-end CLI: extract from a synthetic corpus, write to file,
    verify JSONL content."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    monsters_dir = tmp_path / "monsters" / "bestiary"
    _write_monster_file(
        monsters_dir,
        "harpy.json",
        _monster(name="Harpy", sections=[_section("Harpy", "winged creature")]),
    )
    out_path = tmp_path / "out.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "extract-pfsrd2-monsters",
            str(tmp_path),
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {"name": "Harpy", "description": "winged creature"}


def test_cli_extract_pfsrd2_monsters_stdout_default(tmp_path: Path) -> None:
    """Without --output, JSONL goes to stdout. CliRunner captures it."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    monsters_dir = tmp_path / "monsters" / "bestiary"
    _write_monster_file(monsters_dir, "harpy.json", _monster(name="Harpy"))
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["lexicon", "extract-pfsrd2-monsters", str(tmp_path)],
    )
    assert result.exit_code == 0
    # First line of stdout is the record; the summary line is on stderr.
    first_line = result.output.strip().splitlines()[0]
    record = json.loads(first_line)
    assert record["name"] == "Harpy"


def test_cli_extract_pfsrd2_monsters_limit_caps_output(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    monsters_dir = tmp_path / "monsters" / "bestiary"
    for n in ("Aaa", "Bbb", "Ccc"):
        _write_monster_file(monsters_dir, f"{n.lower()}.json", _monster(name=n))
    out_path = tmp_path / "out.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "extract-pfsrd2-monsters",
            str(tmp_path),
            "--output",
            str(out_path),
            "--limit",
            "2",
        ],
    )
    assert result.exit_code == 0
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_cli_extract_pfsrd2_monsters_nonexistent_dir_errors() -> None:
    """Click's exists=True validator should reject a missing path."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["lexicon", "extract-pfsrd2-monsters", "/nonexistent/path"],
    )
    assert result.exit_code != 0


def test_cli_extract_pfsrd2_monsters_empty_corpus(tmp_path: Path) -> None:
    """A directory with no `monsters/` subfolder produces zero records
    cleanly."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["lexicon", "extract-pfsrd2-monsters", str(tmp_path)],
    )
    assert result.exit_code == 0
    # The only output is the summary line on stderr; no JSONL records
    # mean no JSON-parseable lines anywhere in result.output.
    output_lines = [line for line in result.output.splitlines() if line.strip()]
    json_lines = [line for line in output_lines if line.startswith("{") and line.endswith("}")]
    assert json_lines == []
    # The summary mentions zero records.
    assert "Extracted 0 record(s)" in result.output


# --- Real-world fixture-based regression test (lightweight) -----------


def test_real_world_bugbear_thug_pattern_resolves_to_bugbear(tmp_path: Path) -> None:
    """Pinned regression: a pfsrd2-shaped Bugbear Thug entry produces
    a single 'Bugbear' record. Catches schema drift if pfsrd2 changes
    where it stores family.name."""
    monsters_dir = tmp_path / "monsters" / "bestiary"
    bugbear_thug = {
        "name": "Bugbear Thug",
        "type": "creature",
        "sections": [
            {
                "name": "Bugbear Thug",
                "type": "section",
                "text": "The more common bugbear thug specializes in lurking.",
            }
        ],
        "stat_block": {
            "creature_type": {
                "family": {
                    "name": "Bugbear",
                    "text": "These stealthy goblinoid creatures delight in spreading fear.",
                    "sections": [
                        {
                            "name": "Bugbear Lairs",
                            "type": "section",
                            "text": "Bugbears live in small gangs.",
                        }
                    ],
                }
            }
        },
    }
    _write_monster_file(monsters_dir, "bugbear_thug.json", bugbear_thug)
    records = list(extract_corpus(tmp_path))
    assert len(records) == 1
    assert records[0]["name"] == "Bugbear"
    desc = records[0]["description"]
    # Family lore comes first
    assert desc.startswith("These stealthy goblinoid creatures")
    # Family-section text included
    assert "Bugbears live in small gangs" in desc
    # Variant section text appended
    assert "more common bugbear thug" in desc
