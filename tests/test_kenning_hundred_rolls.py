"""Tests for the Hundred Rolls ingester — wyrd-3atv / Phase 3c."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.ingesters.hundred_rolls import (
    HUNDRED_ROLLS_SOURCE_ID,
    HUNDRED_ROLLS_YEAR,
    CsvSchemaError,
    _normalize_row,
    build_events,
    ingest,
    iter_csv_rows,
)


def _write_csv(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "hundred_rolls.csv"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# _normalize_row
# ---------------------------------------------------------------------------


def test_normalize_row_full_modern_identification():
    """When the operator's CSV has modern_name, region, country, all
    fields propagate to both the toponym and the attestation."""
    row = {
        "vill": "Tunes",
        "hundred": "Doddingtree",
        "county": "Worcestershire",
        "country": "England",
        "modern_name": "Acton",
    }
    pair = _normalize_row(row)
    assert pair is not None
    toponym, attestation = pair
    assert toponym == {
        "_type": "toponym",
        "ref": "Acton@Worcestershire",
        "modern_name": "Acton",
        "country": "England",
        "region": "Worcestershire",
    }
    assert attestation == {
        "_type": "attestation",
        "toponym_ref": "Acton@Worcestershire",
        "form": "Tunes",
        "date_year": 1279,
        "source_doc": "Hundred Rolls 1279; Hundred of Doddingtree; Worcestershire",
    }


def test_normalize_row_modern_name_falls_back_to_vill():
    """Many ME-era place names survive to the present unchanged.
    Without an explicit modern_name, the vill IS the modern_name."""
    row = {"vill": "Acton", "county": "Cheshire"}
    pair = _normalize_row(row)
    assert pair is not None
    toponym, attestation = pair
    assert toponym["modern_name"] == "Acton"
    assert toponym["ref"] == "Acton@Cheshire"
    # Form on the attestation IS the vill verbatim — preserves the
    # original Roll spelling.
    assert attestation["form"] == "Acton"


def test_normalize_row_country_defaults_to_england():
    """The Rolls cover the kingdom of England; absent country = England."""
    row = {"vill": "Acton"}
    pair = _normalize_row(row)
    assert pair is not None
    toponym, _ = pair
    assert toponym["country"] == "England"


def test_normalize_row_skips_blank_vill():
    """A blank vill isn't a settlement entry; the row is filtered out
    rather than raising. Lets operators leave row gaps in their CSV
    without breaking the parse."""
    assert _normalize_row({"vill": ""}) is None
    assert _normalize_row({"vill": "   "}) is None
    assert _normalize_row({}) is None


def test_normalize_row_source_doc_omits_empty_parts():
    """When hundred / region are missing, source_doc still records
    the year + whatever IS present — no trailing semicolons."""
    row = {"vill": "Acton"}
    pair = _normalize_row(row)
    assert pair is not None
    _, attestation = pair
    assert attestation["source_doc"] == "Hundred Rolls 1279"


def test_normalize_row_null_region_ref_uses_dash():
    """Toponym ref uses '@-' for null region — matches the convention
    elsewhere in the project."""
    row = {"vill": "Somewhere"}
    pair = _normalize_row(row)
    assert pair is not None
    toponym, _ = pair
    assert toponym["ref"] == "Somewhere@-"


# ---------------------------------------------------------------------------
# iter_csv_rows — schema validation
# ---------------------------------------------------------------------------


def test_iter_csv_rows_requires_vill_column(tmp_path: Path):
    """Required column missing → CsvSchemaError. Surfaces at the
    boundary instead of silently producing zero rows."""
    path = _write_csv(tmp_path, "name,county\nFoo,Bar\n")
    with pytest.raises(CsvSchemaError, match="required columns missing"):
        list(iter_csv_rows(path))


def test_iter_csv_rows_empty_file_raises(tmp_path: Path):
    path = _write_csv(tmp_path, "")
    with pytest.raises(CsvSchemaError, match="empty file"):
        list(iter_csv_rows(path))


def test_iter_csv_rows_passes_minimal_csv(tmp_path: Path):
    """Just the required column is enough; optional columns absent."""
    path = _write_csv(tmp_path, "vill\nActon\nTunes\n")
    rows = list(iter_csv_rows(path))
    assert len(rows) == 2
    assert [r["vill"] for r in rows] == ["Acton", "Tunes"]


# ---------------------------------------------------------------------------
# build_events
# ---------------------------------------------------------------------------


def test_build_events_emits_source_row(tmp_path: Path):
    path = _write_csv(
        tmp_path,
        "vill,hundred,county,modern_name\n"
        "Tunes,Doddingtree,Worcestershire,Acton\n"
        "Caisteus,Hodnet,Shropshire,Castor\n",
    )
    events = build_events(iter_csv_rows(path))
    assert events[0]["_type"] == "source"
    assert events[0]["ref"] == HUNDRED_ROLLS_SOURCE_ID
    assert events[0]["year"] == HUNDRED_ROLLS_YEAR
    toponyms = [e for e in events if e["_type"] == "toponym"]
    attestations = [e for e in events if e["_type"] == "attestation"]
    assert len(toponyms) == 2
    assert len(attestations) == 2


def test_build_events_dedupes_toponym_refs(tmp_path: Path):
    """Two Roll entries for the same modern_name → one toponym
    event, two attestations (one per Roll entry)."""
    path = _write_csv(
        tmp_path,
        "vill,county,modern_name\n"
        "Tunes,Cheshire,Acton\n"
        "Tunes,Cheshire,Acton\n",  # same ref, different roll entry
    )
    events = build_events(iter_csv_rows(path))
    toponyms = [e for e in events if e["_type"] == "toponym" and e["ref"] == "Acton@Cheshire"]
    attestations = [
        e for e in events if e["_type"] == "attestation" and e["toponym_ref"] == "Acton@Cheshire"
    ]
    assert len(toponyms) == 1
    assert len(attestations) == 2


def test_build_events_without_source_row(tmp_path: Path):
    path = _write_csv(tmp_path, "vill\nActon\n")
    events = build_events(iter_csv_rows(path), include_source_row=False)
    assert all(e["_type"] != "source" for e in events)


# ---------------------------------------------------------------------------
# ingest()
# ---------------------------------------------------------------------------


def test_ingest_dry_run_no_write(tmp_path: Path):
    csv_path = _write_csv(tmp_path, "vill,county\nActon,Cheshire\nTunes,Worcs\n")
    out_path = tmp_path / "out.jsonl"
    counts = ingest(csv_path, out_path, apply=False)
    assert counts["csv_rows_scanned"] == 2
    assert counts["toponym_events"] == 2
    assert counts["attestation_events"] == 2
    assert "written" not in counts
    assert not out_path.exists()


def test_ingest_apply_writes_jsonl(tmp_path: Path):
    csv_path = _write_csv(tmp_path, "vill,county\nActon,Cheshire\n")
    out_path = tmp_path / "out.jsonl"
    counts = ingest(csv_path, out_path, apply=True)
    assert counts["written"] == 1
    lines = out_path.read_text().splitlines()
    # source + toponym + attestation = 3.
    assert len(lines) == 3
    assert json.loads(lines[0])["_type"] == "source"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_ingest_hundred_rolls_dry_run(tmp_path: Path):
    csv_path = _write_csv(tmp_path, "vill,county\nActon,Cheshire\n")
    out_path = tmp_path / "out.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "ingest-hundred-rolls", str(csv_path), "--out", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert not out_path.exists()


def test_cli_ingest_hundred_rolls_apply(tmp_path: Path):
    csv_path = _write_csv(tmp_path, "vill,county\nActon,Cheshire\n")
    out_path = tmp_path / "out.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "ingest-hundred-rolls", str(csv_path), "--out", str(out_path), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert "Toponym events" in result.output
