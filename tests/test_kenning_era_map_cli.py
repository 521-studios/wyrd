"""wyrd-hyeb: ``kenning era-map --from-json`` — era-render pre-generated
``generate --json`` objects in era-map's table shape (mirrors rewind --from-json).

The DB-backed era-reflex rewinder (``rewind_from_morphemes``) + the live
``LexiconDB`` are stubbed so the test pins the era-map CLI's NEW glue (JSON parse →
rewind → table/JSON render + the culture-optional contract) without a lexicon DB
(CI has none)."""

from __future__ import annotations

import json
import types

import pytest
from click.testing import CliRunner

import wyrd.generators.kenning as kn
import wyrd.generators.kenning.era.rewind as rw
import wyrd.generators.kenning.lexicon as lex
from wyrd.generators.kenning.cli.era_map import era_map
from wyrd.generators.kenning.era.rewind import EraStop, Rewind


class _FakeDB:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def dbfile(tmp_path):
    """An existing (dummy) DB path so the from-json existence check passes; the
    real open is stubbed via LexiconDB, so the file's contents don't matter."""
    p = tmp_path / "lex.db"
    p.touch()
    return p


@pytest.fixture
def stub_rewind(monkeypatch):
    """Stub the DB + meanings load + rewind so --from-json runs without a live DB.
    The fake rewind echoes the input name and emits two fixed era stops."""
    monkeypatch.setattr(lex, "LexiconDB", lambda *a, **k: _FakeDB())
    monkeypatch.setattr(kn, "_load_meanings", lambda *a, **k: ({}, None))

    def fake_rewind(name, words, db, meaning_db, *, eras):
        return Rewind(
            name=name,
            decomposition="x + y",
            eras=[
                EraStop(family="english", cell="oe-late", morphemes=[], rendered=f"{name}-OE"),
                EraStop(family="english", cell="me", morphemes=[], rendered=f"{name}-ME"),
            ],
        )

    monkeypatch.setattr(rw, "rewind_from_morphemes", fake_rewind)
    return fake_rewind


def _objs(*names: str) -> str:
    return "\n".join(json.dumps({"name": n, "words": [[{"usage": n.lower()}]]}) for n in names)


def test_from_json_renders_era_map_table(stub_rewind, dbfile):
    result = CliRunner().invoke(
        era_map, ["--from-json", "-", "--db", str(dbfile)], input=_objs("Whitton", "Aldton")
    )
    assert result.exit_code == 0, result.output
    # era-map table shape: name header + 'family/cell  rendered' rows, both names.
    assert "Whitton" in result.output and "Aldton" in result.output
    assert "english/oe-late" in result.output and "english/me" in result.output
    assert "Whitton-OE" in result.output and "Aldton-ME" in result.output


def test_from_json_as_json_shape(stub_rewind, dbfile):
    result = CliRunner().invoke(
        era_map, ["--from-json", "-", "--as-json", "--db", str(dbfile)], input=_objs("Whitton")
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data == [
        {
            "name": "Whitton",
            "era_cells": [
                {"family": "english", "era": "oe-late", "rendered": "Whitton-OE"},
                {"family": "english", "era": "me", "rendered": "Whitton-ME"},
            ],
        }
    ]


def test_from_json_table_marks_fallback(stub_rewind, dbfile, monkeypatch):
    """A stop with a fallback morpheme gets the ' *' marker in the table."""

    def fake(name, words, db, meaning_db, *, eras):
        return Rewind(
            name=name,
            decomposition="x",
            eras=[
                EraStop(
                    family="english",
                    cell="me",
                    morphemes=[types.SimpleNamespace(fallback=True)],
                    rendered=f"{name}-ME",
                )
            ],
        )

    monkeypatch.setattr(rw, "rewind_from_morphemes", fake)
    result = CliRunner().invoke(
        era_map, ["--from-json", "-", "--db", str(dbfile)], input=_objs("X")
    )
    assert result.exit_code == 0, result.output
    assert "X-ME *" in result.output


def test_from_json_custom_era_ladder_is_parsed(stub_rewind, dbfile, monkeypatch):
    """--era values are resolved (via era_cell_for_input) and passed to the
    rewinder; a bad era exits 1."""
    seen = {}
    real = stub_rewind

    def capture(name, words, db, meaning_db, *, eras):
        seen["eras"] = eras
        return real(name, words, db, meaning_db, eras=eras)

    monkeypatch.setattr(rw, "rewind_from_morphemes", capture)
    ok = CliRunner().invoke(
        era_map,
        ["--from-json", "-", "--db", str(dbfile), "--era", "oe-early", "--era", "me"],
        input=_objs("X"),
    )
    assert ok.exit_code == 0, ok.output
    assert seen["eras"] and len(seen["eras"]) == 2

    bad = CliRunner().invoke(
        era_map, ["--from-json", "-", "--db", str(dbfile), "--era", "not-a-cell"], input=_objs("X")
    )
    assert bad.exit_code == 1


def test_from_json_empty_input_exits_loud(stub_rewind, dbfile):
    result = CliRunner().invoke(era_map, ["--from-json", "-", "--db", str(dbfile)], input="\n  \n")
    assert result.exit_code == 1
    assert "no JSON objects" in result.output


def test_from_json_missing_db_exits_loud(stub_rewind, tmp_path):
    """A nonexistent --db fails loud (SQLite would otherwise create an empty DB
    and render everything as era-less fallback)."""
    missing = tmp_path / "nope.db"
    result = CliRunner().invoke(
        era_map, ["--from-json", "-", "--db", str(missing)], input=_objs("X")
    )
    assert result.exit_code == 1
    assert "lexicon DB not found" in result.output


def test_from_json_default_db_path_is_resolved(stub_rewind, tmp_path, monkeypatch):
    """With no --db, the default lexicon path is resolved by CALLING
    _DEFAULT_LEXICON_PATH (it's a callable, not a Path) and run through the same
    existence check — a missing default fails loud, not with a TypeError."""
    import wyrd.generators.kenning.cli.utils as utils

    missing = tmp_path / "default-nope.db"
    monkeypatch.setattr(utils, "_DEFAULT_LEXICON_PATH", lambda: missing)
    result = CliRunner().invoke(era_map, ["--from-json", "-"], input=_objs("X"))
    assert result.exit_code == 1
    assert "lexicon DB not found" in result.output


def test_culture_required_without_from_json():
    """The generate path still requires CULTURE; omitting it (and --from-json) is
    a loud error, not a traceback."""
    result = CliRunner().invoke(era_map, [])
    assert result.exit_code == 1
    assert "CULTURE argument is required" in result.output
