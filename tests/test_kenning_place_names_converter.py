"""wyrd-bo01.2: gazetteer JSON → L2 JSONL toponym source."""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli.lexicon.convert_place_names import lexicon_convert_place_names
from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in
from wyrd.generators.kenning.jsonl.dump import DEFAULT_BULK_EXCLUDED_SOURCES
from wyrd.generators.kenning.lexicon import init_schema
from wyrd.generators.kenning.place_names_converter import (
    CULTURES,
    convert_file,
    emit_place_names_jsonl,
    source_id_for,
)

# Minimal 3-level {country: {region: [names]}} gazetteer. "California" appears
# in two counties (distinct toponyms) and "Bedford" is duplicated within one
# county (must dedupe).
SAMPLE = {
    "England": {
        "Bedfordshire": ["Bedford", "Bedford", "California"],
        "Berkshire": ["California", "Reading"],
    },
    "The Isle of Man": {"Isle of Man": ["Douglas"]},
}


def _emit(data) -> list[dict]:
    sink = io.StringIO()
    emit_place_names_jsonl("english", data, sink)
    return [json.loads(line) for line in sink.getvalue().splitlines()]


def test_emit_writes_source_then_toponyms():
    rows = _emit(SAMPLE)
    assert rows[0]["_type"] == "source"
    assert rows[0]["ref"] == source_id_for("english") == "english_place_names"
    topos = [r for r in rows if r["_type"] == "toponym"]
    # 5 distinct (name, country, region): Bedford(dup→1), California×2 (diff
    # county), Reading, Douglas.
    assert len(topos) == 5
    assert all(set(t) >= {"modern_name", "country", "region", "ref"} for t in topos)


def test_emit_dedupes_within_natural_key_keeps_cross_region():
    topos = [r for r in _emit(SAMPLE) if r["_type"] == "toponym"]
    keys = sorted((t["modern_name"], t["country"], t["region"]) for t in topos)
    assert keys == [
        ("Bedford", "England", "Bedfordshire"),  # the duplicate collapsed to one
        ("California", "England", "Bedfordshire"),  # same name, two counties...
        ("California", "England", "Berkshire"),  # ...both kept
        ("Douglas", "The Isle of Man", "Isle of Man"),
        ("Reading", "England", "Berkshire"),
    ]
    # distinct refs encode the full key so cross-region same-names don't merge.
    assert len({t["ref"] for t in topos}) == 5


def test_build_roundtrip_lands_toponyms_with_country_and_region(tmp_path: Path):
    """The emitted JSONL replays through build_from_jsonl into toponym rows
    carrying country + region (the gazetteer becomes DB toponyms)."""
    out = tmp_path / "english_place_names.jsonl"
    with out.open("w", encoding="utf-8") as sink:
        emit_place_names_jsonl("english", SAMPLE, sink)

    db = tmp_path / "lexicon.db"
    init_schema(db)
    conn = sqlite3.connect(db)
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM toponym").fetchone()[0] == 5
    bedford = conn.execute(
        "SELECT country, region FROM toponym WHERE modern_name = 'Bedford'"
    ).fetchone()
    assert bedford == ("England", "Bedfordshire")
    # both Californias survive as distinct rows
    assert (
        conn.execute("SELECT COUNT(*) FROM toponym WHERE modern_name = 'California'").fetchone()[0]
        == 2
    )
    conn.close()


def test_convert_file_reads_json_and_writes_jsonl(tmp_path: Path):
    """The disk leg: read a gazetteer JSON, mkdir the parent, write the
    JSONL, return the toponym count."""
    src = tmp_path / "english_place_names.json"
    src.write_text(json.dumps(SAMPLE), encoding="utf-8")
    out = tmp_path / "sub" / "english_place_names.jsonl"  # forces parent mkdir
    n = convert_file("english", src, out)
    assert n == 5
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["_type"] == "source"
    assert sum(r["_type"] == "toponym" for r in rows) == 5


def test_cli_single_culture_and_default_all(tmp_path: Path):
    """The CLI: --culture writes one file; no --culture writes all five
    (exercises the filter branch + bundled-resource resolution)."""
    runner = CliRunner()
    r = runner.invoke(
        lexicon_convert_place_names, ["--culture", "english", "--out-dir", str(tmp_path)]
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "english_place_names.jsonl").exists()
    assert not (tmp_path / "irish_place_names.jsonl").exists()

    r2 = runner.invoke(lexicon_convert_place_names, ["--out-dir", str(tmp_path)])
    assert r2.exit_code == 0, r2.output
    for c in CULTURES:
        assert (tmp_path / f"{c}_place_names.jsonl").exists()


def test_emit_empty_gazetteer_yields_only_source():
    rows = _emit({})
    assert len(rows) == 1 and rows[0]["_type"] == "source"


def test_emit_coerces_non_string_name():
    rows = _emit({"England": {"Berkshire": [42]}})
    topo = next(r for r in rows if r["_type"] == "toponym")
    assert topo["modern_name"] == "42"


def test_gazetteer_sources_excluded_from_bulk_dump():
    """The 5 gazetteer sources are converter-owned committed artifacts —
    they MUST be in DEFAULT_BULK_EXCLUDED_SOURCES so a bulk dump can't emit
    a competing (toponym-less) file and clobber the committed JSONL."""
    for c in CULTURES:
        assert source_id_for(c) in DEFAULT_BULK_EXCLUDED_SOURCES
