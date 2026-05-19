"""Tests for the Hearth Tax ingester — wyrd-myh1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.ingesters.hearth_tax import (
    DEFAULT_HEARTH_TAX_YEAR,
    HEARTH_TAX_SOURCE_ID,
    CsvSchemaError,
    OutOfRangeYearWarning,
    _normalize_row,
    _parse_year,
    build_events,
    ingest,
    iter_csv_rows,
)


def _write_csv(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "hearth_tax.csv"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# _parse_year — Hearth Tax year_specific has the most parsing nuance
# ---------------------------------------------------------------------------


def test_parse_year_valid_in_range():
    assert _parse_year("1662") == 1662
    assert _parse_year("1670") == 1670
    assert _parse_year("1674") == 1674


def test_parse_year_blank_uses_default():
    assert _parse_year("") == DEFAULT_HEARTH_TAX_YEAR
    assert _parse_year("   ") == DEFAULT_HEARTH_TAX_YEAR


def test_parse_year_unparseable_uses_default():
    """Operator transcriptions sometimes have 'c.1665', '1662?' —
    we accept-with-default rather than abort the ingest."""
    assert _parse_year("c.1665") == DEFAULT_HEARTH_TAX_YEAR
    assert _parse_year("1662?") == DEFAULT_HEARTH_TAX_YEAR


def test_parse_year_out_of_range_uses_default_and_warns():
    """Hearth Tax was collected 1662-1674; anything outside is
    transcription error, not a meaningful per-row year. Emit a
    warning so the operator notices — but don't raise."""
    with pytest.warns(OutOfRangeYearWarning, match="outside Hearth Tax collection window"):
        assert _parse_year("1600") == DEFAULT_HEARTH_TAX_YEAR
    with pytest.warns(OutOfRangeYearWarning):
        assert _parse_year("1700") == DEFAULT_HEARTH_TAX_YEAR


def test_parse_year_unparseable_does_not_warn():
    """Unparseable cells (c.1665, 1662?) are common transcription
    artifacts — silent fallback only. Reserve the warning for cases
    where the operator likely has a bad year integer."""
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning would raise
        assert _parse_year("c.1665") == DEFAULT_HEARTH_TAX_YEAR
        assert _parse_year("") == DEFAULT_HEARTH_TAX_YEAR


# ---------------------------------------------------------------------------
# _normalize_row
# ---------------------------------------------------------------------------


def test_normalize_row_full():
    row = {
        "place_name": "Hagley",
        "parish": "Hagley",
        "county": "Worcestershire",
        "country": "England",
        "year_specific": "1664",
        "modern_name": "Hagley",
    }
    toponym, attestation = _normalize_row(row)
    assert toponym == {
        "_type": "toponym",
        "ref": "Hagley@Worcestershire",
        "modern_name": "Hagley",
        "country": "England",
        "region": "Worcestershire",
    }
    assert attestation == {
        "_type": "attestation",
        "toponym_ref": "Hagley@Worcestershire",
        "form": "Hagley",
        "date_year": 1664,
        "source_doc": "Hearth Tax 1664; Parish of Hagley; Worcestershire",
    }


def test_normalize_row_township_with_parish_container():
    """A township carrying a different parish: source_doc records
    both. The toponym ref is still keyed by the township's own
    modern_name + region, not the parish — townships are distinct
    settlement entities."""
    row = {
        "place_name": "Pedmore",
        "parish": "Hagley",
        "county": "Worcestershire",
        "year_specific": "1664",
    }
    toponym, attestation = _normalize_row(row)
    assert toponym["ref"] == "Pedmore@Worcestershire"
    assert attestation["source_doc"] == "Hearth Tax 1664; Parish of Hagley; Worcestershire"


def test_normalize_row_parish_omitted_falls_back():
    """When parish is blank, source_doc skips that segment."""
    row = {
        "place_name": "Hagley",
        "county": "Worcestershire",
        "year_specific": "1664",
    }
    _, attestation = _normalize_row(row)
    assert attestation["source_doc"] == "Hearth Tax 1664; Worcestershire"


def test_normalize_row_year_specific_blank_uses_default():
    row = {"place_name": "Hagley", "county": "Worcestershire"}
    _, attestation = _normalize_row(row)
    assert attestation["date_year"] == DEFAULT_HEARTH_TAX_YEAR
    assert attestation["source_doc"] == f"Hearth Tax {DEFAULT_HEARTH_TAX_YEAR}; Worcestershire"


def test_normalize_row_modern_name_falls_back_to_place_name():
    row = {"place_name": "Hagley", "county": "Worcestershire"}
    toponym, _ = _normalize_row(row)
    assert toponym["modern_name"] == "Hagley"


def test_normalize_row_country_defaults_to_england():
    row = {"place_name": "Hagley"}
    toponym, _ = _normalize_row(row)
    assert toponym["country"] == "England"


def test_normalize_row_skips_blank_place_name():
    assert _normalize_row({"place_name": ""}) is None
    assert _normalize_row({"place_name": "   "}) is None
    assert _normalize_row({}) is None


# ---------------------------------------------------------------------------
# iter_csv_rows
# ---------------------------------------------------------------------------


def test_iter_csv_rows_requires_place_name(tmp_path: Path):
    path = _write_csv(tmp_path, "parish,county\nFoo,Bar\n")
    with pytest.raises(CsvSchemaError, match="required columns missing"):
        list(iter_csv_rows(path))


def test_iter_csv_rows_empty_file_raises(tmp_path: Path):
    path = _write_csv(tmp_path, "")
    with pytest.raises(CsvSchemaError, match="empty file"):
        list(iter_csv_rows(path))


# ---------------------------------------------------------------------------
# build_events + ingest + CLI
# ---------------------------------------------------------------------------


def test_build_events_emits_source_row(tmp_path: Path):
    path = _write_csv(
        tmp_path,
        "place_name,county,year_specific\n"
        "Hagley,Worcestershire,1664\n"
        "Stourbridge,Worcestershire,1664\n",
    )
    events = build_events(iter_csv_rows(path))
    assert events[0]["_type"] == "source"
    assert events[0]["ref"] == HEARTH_TAX_SOURCE_ID
    assert sum(1 for e in events if e["_type"] == "toponym") == 2
    assert sum(1 for e in events if e["_type"] == "attestation") == 2


def test_build_events_dedupes_toponym_refs(tmp_path: Path):
    """Two attestations for the same parish (e.g. across separate
    Hearth Tax collection rounds) → one toponym, two attestations."""
    path = _write_csv(
        tmp_path,
        "place_name,county,year_specific\nHagley,Worcestershire,1662\nHagley,Worcestershire,1664\n",
    )
    events = build_events(iter_csv_rows(path))
    toponyms = [e for e in events if e["_type"] == "toponym"]
    attestations = [e for e in events if e["_type"] == "attestation"]
    assert len(toponyms) == 1
    assert len(attestations) == 2
    assert attestations[0]["date_year"] == 1662
    assert attestations[1]["date_year"] == 1664


def test_ingest_dry_run_no_write(tmp_path: Path):
    csv_path = _write_csv(tmp_path, "place_name,county\nHagley,Worcestershire\n")
    out_path = tmp_path / "out.jsonl"
    counts = ingest(csv_path, out_path, apply=False)
    assert counts["csv_rows_scanned"] == 1
    assert not out_path.exists()


def test_ingest_apply_writes_jsonl(tmp_path: Path):
    csv_path = _write_csv(tmp_path, "place_name,county\nHagley,Worcestershire\n")
    out_path = tmp_path / "out.jsonl"
    counts = ingest(csv_path, out_path, apply=True)
    assert counts["written"] == 1
    lines = out_path.read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["_type"] == "source"


def test_cli_ingest_hearth_tax_dry_run(tmp_path: Path):
    csv_path = _write_csv(tmp_path, "place_name,county\nHagley,Worcs\n")
    out_path = tmp_path / "out.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "ingest-hearth-tax", str(csv_path), "--out", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output


def test_cli_ingest_hearth_tax_apply(tmp_path: Path):
    csv_path = _write_csv(tmp_path, "place_name,county\nHagley,Worcs\n")
    out_path = tmp_path / "out.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "ingest-hearth-tax", str(csv_path), "--out", str(out_path), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
