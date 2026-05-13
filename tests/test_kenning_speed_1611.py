"""Tests for the Speed 1611 ingester — wyrd-myh1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.speed_1611_ingester import (
    SPEED_1611_SOURCE_ID,
    SPEED_1611_YEAR,
    CsvSchemaError,
    _normalize_row,
    build_events,
    ingest,
    iter_csv_rows,
)


def _write_csv(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "speed_1611.csv"
    path.write_text(content)
    return path


def test_normalize_row_full():
    row = {
        "place_name": "Stratford uppon Avon",
        "county": "Warwickshire",
        "parish": "Stratford",
        "country": "England",
        "modern_name": "Stratford-upon-Avon",
    }
    toponym, attestation = _normalize_row(row)
    assert toponym == {
        "_type": "toponym",
        "ref": "Stratford-upon-Avon@Warwickshire",
        "modern_name": "Stratford-upon-Avon",
        "country": "England",
        "region": "Warwickshire",
    }
    assert attestation == {
        "_type": "attestation",
        "toponym_ref": "Stratford-upon-Avon@Warwickshire",
        "form": "Stratford uppon Avon",
        "date_year": 1611,
        "source_doc": "Speed 1611; Parish of Stratford; Warwickshire",
    }


def test_normalize_row_modern_name_falls_back_to_place_name():
    """Most early-modern place names survive unchanged."""
    row = {"place_name": "York", "county": "Yorkshire"}
    toponym, attestation = _normalize_row(row)
    assert toponym["modern_name"] == "York"
    assert toponym["ref"] == "York@Yorkshire"
    assert attestation["form"] == "York"


def test_normalize_row_country_defaults_to_england():
    row = {"place_name": "York"}
    toponym, _ = _normalize_row(row)
    assert toponym["country"] == "England"


def test_normalize_row_skips_blank_place_name():
    assert _normalize_row({"place_name": ""}) is None
    assert _normalize_row({"place_name": "   "}) is None
    assert _normalize_row({}) is None


def test_normalize_row_source_doc_omits_empty_parts():
    row = {"place_name": "York"}
    _, attestation = _normalize_row(row)
    assert attestation["source_doc"] == "Speed 1611"


def test_iter_csv_rows_requires_place_name(tmp_path: Path):
    path = _write_csv(tmp_path, "name,county\nFoo,Bar\n")
    with pytest.raises(CsvSchemaError, match="required columns missing"):
        list(iter_csv_rows(path))


def test_iter_csv_rows_empty_file_raises(tmp_path: Path):
    path = _write_csv(tmp_path, "")
    with pytest.raises(CsvSchemaError, match="empty file"):
        list(iter_csv_rows(path))


def test_build_events_emits_source_row(tmp_path: Path):
    path = _write_csv(
        tmp_path,
        "place_name,county,modern_name\nYork,Yorkshire,York\nBrissol,Bristol,Bristol\n",
    )
    events = build_events(iter_csv_rows(path))
    assert events[0]["_type"] == "source"
    assert events[0]["ref"] == SPEED_1611_SOURCE_ID
    assert events[0]["year"] == SPEED_1611_YEAR
    assert sum(1 for e in events if e["_type"] == "toponym") == 2
    assert sum(1 for e in events if e["_type"] == "attestation") == 2


def test_build_events_dedupes_toponym_refs(tmp_path: Path):
    path = _write_csv(
        tmp_path,
        "place_name,county,modern_name\nYorke,Yorkshire,York\nYork,Yorkshire,York\n",
    )
    events = build_events(iter_csv_rows(path))
    toponyms = [e for e in events if e["_type"] == "toponym" and e["ref"] == "York@Yorkshire"]
    attestations = [e for e in events if e["_type"] == "attestation"]
    assert len(toponyms) == 1
    assert len(attestations) == 2


def test_ingest_dry_run_no_write(tmp_path: Path):
    csv_path = _write_csv(tmp_path, "place_name,county\nYork,Yorkshire\n")
    out_path = tmp_path / "out.jsonl"
    counts = ingest(csv_path, out_path, apply=False)
    assert counts["csv_rows_scanned"] == 1
    assert not out_path.exists()


def test_ingest_apply_writes_jsonl(tmp_path: Path):
    csv_path = _write_csv(tmp_path, "place_name,county\nYork,Yorkshire\n")
    out_path = tmp_path / "out.jsonl"
    counts = ingest(csv_path, out_path, apply=True)
    assert counts["written"] == 1
    lines = out_path.read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["_type"] == "source"


def test_cli_ingest_speed_1611_dry_run(tmp_path: Path):
    csv_path = _write_csv(tmp_path, "place_name,county\nYork,Yorkshire\n")
    out_path = tmp_path / "out.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "ingest-speed-1611", str(csv_path), "--out", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert not out_path.exists()


def test_cli_ingest_speed_1611_apply(tmp_path: Path):
    csv_path = _write_csv(tmp_path, "place_name,county\nYork,Yorkshire\n")
    out_path = tmp_path / "out.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "ingest-speed-1611", str(csv_path), "--out", str(out_path), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
