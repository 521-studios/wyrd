"""Tests for the OS Open Names ingester — wyrd-3ypp / Phase 3b.

Two layers covered:
1. ``build_events`` — pure function that turns CSV rows into JSONL
   events. Tests target the filter (populated places only) + the
   toponym/attestation event shape + the per-ref dedup.
2. CLI ``lexicon ingest-os-open-names`` — verifies the dry-run /
   apply distinction and the per-row counts.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.ingesters.os_open_names import (
    _COLUMNS,
    OS_OPEN_NAMES_SOURCE_ID,
    _normalize_row,
    build_events,
    ingest,
    iter_csv_rows,
)

# Sample CSV rows in the OS Open Names schema. NAME1, COUNTY_UNITARY,
# COUNTRY, TYPE, LOCAL_TYPE are the columns we read; the rest just need
# to be present so the column-name → value mapping doesn't lose data.
_HEADERLESS_CSV_TEMPLATE = (
    # NAME1=Acton, county=Cheshire, country=England, populated village
    '"osgb1","https://example.com/1","Acton","en","","","populatedPlace","Village",'
    '"380000","375000","0","0","0","0","0","0","CW5","","Acton","","",'
    '"","","","Cheshire","","","","","England","",""'
    ',"",""\n'
    # NAME1=Acton, county=Suffolk — distinct ref by region
    '"osgb2","https://example.com/2","Acton","en","","","populatedPlace","Village",'
    '"395000","348000","0","0","0","0","0","0","CO10","","Acton","","",'
    '"","","","Suffolk","","","","","England","",""'
    ',"",""\n'
    # NAME1=Manchester, type=Town
    '"osgb3","https://example.com/3","Manchester","en","","","populatedPlace","City",'
    '"380000","395000","0","0","0","0","0","0","M1","","Manchester","","",'
    '"","","","Greater Manchester","","","","","England","",""'
    ',"",""\n'
    # Filtered out: TYPE=transportNetwork
    '"osgb4","https://example.com/4","A1","en","","","transportNetwork","A Road",'
    '"380000","395000","0","0","0","0","0","0","","","","","",'
    '"","","","","","","","","","",""'
    ',"",""\n'
    # Filtered out: TYPE=populatedPlace but LOCAL_TYPE=Postcode (which isn't in _POPULATED_LOCAL_TYPES)
    '"osgb5","https://example.com/5","CW5 6AB","en","","","populatedPlace","Postcode",'
    '"380000","375000","0","0","0","0","0","0","CW5 6AB","","","","",'
    '"","","","Cheshire","","","","","England","",""'
    ',"",""\n'
    # Duplicate ref of osgb1 (Acton@Cheshire) — toponym should dedupe,
    # attestation should still emit (per-source spec)
    '"osgb6","https://example.com/6","Acton","en","","","populatedPlace","Village",'
    '"380000","375000","0","0","0","0","0","0","CW5","","Acton","","",'
    '"","","","Cheshire","","","","","England","",""'
    ',"",""\n'
)


def _write_fixture_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "os_open_names.csv"
    csv_path.write_text(_HEADERLESS_CSV_TEMPLATE)
    return csv_path


# ---------------------------------------------------------------------------
# _normalize_row
# ---------------------------------------------------------------------------


def test_normalize_row_emits_toponym_and_attestation_pair():
    row = {
        "NAMES_URI": "https://example.com/acton",
        "NAME1": "Acton",
        "TYPE": "populatedPlace",
        "LOCAL_TYPE": "Village",
        "COUNTY_UNITARY": "Cheshire",
        "COUNTRY": "England",
    }
    pair = _normalize_row(row)
    assert pair is not None
    toponym, attestation = pair
    assert toponym == {
        "_type": "toponym",
        "ref": "Acton@Cheshire",
        "modern_name": "Acton",
        "country": "England",
        "region": "Cheshire",
    }
    assert attestation == {
        "_type": "attestation",
        "toponym_ref": "Acton@Cheshire",
        "form": "Acton",
        "source_doc": "OS Open Names (Village) https://example.com/acton",
    }


def test_normalize_row_skips_non_populated_type():
    """TYPE=transportNetwork (motorways, rail) is filtered out."""
    row = {
        "NAME1": "A1",
        "TYPE": "transportNetwork",
        "LOCAL_TYPE": "A Road",
    }
    assert _normalize_row(row) is None


def test_normalize_row_skips_non_settlement_local_type():
    """TYPE=populatedPlace can include postcodes etc. We filter to the
    settlement classes (City/Town/Village/Hamlet/Suburban Area/Other)."""
    row = {
        "NAME1": "CW5 6AB",
        "TYPE": "populatedPlace",
        "LOCAL_TYPE": "Postcode",
    }
    assert _normalize_row(row) is None


def test_normalize_row_truncated_row_present_none_trailing_cols_does_not_crash():
    """A SHORT CSV row: csv.DictReader fills missing trailing columns with
    restval=None (the columns are PRESENT with a None value, not absent), and the
    drift guard only catches EXTRA columns. COUNTY_UNITARY@24 / COUNTRY@29 sit
    AFTER the populated-place gate (TYPE@6/LOCAL_TYPE@7), so a gate-passing-but-
    truncated row presents them as None — and ``row.get(col, "")`` returns that
    None (its default applies to ABSENT keys only), so ``None.strip()`` would
    crash the whole ingest. This mirrors the real DictReader row shape (every
    fieldname present), unlike the sparse-dict tests above that omit keys."""
    row = dict.fromkeys(_COLUMNS, None)  # DictReader shape: all fieldnames present
    row["NAMES_URI"] = "http://example/1"  # before the gate — present in a real short row
    row["NAME1"] = "Edlingham"
    row["TYPE"] = "populatedPlace"
    row["LOCAL_TYPE"] = "Village"
    # COUNTRY / COUNTY_UNITARY left None (row truncated after the gate columns)
    pair = _normalize_row(row)  # must NOT raise (was None.strip() AttributeError)
    assert pair is not None
    toponym, _ = pair
    assert toponym["modern_name"] == "Edlingham"
    assert toponym["ref"] == "Edlingham@-"  # None region → @-
    assert "country" not in toponym  # present-None → "" → falsy → omitted
    assert "region" not in toponym


def test_normalize_row_handles_null_region():
    """Place with no COUNTY_UNITARY still produces a ref using @-."""
    row = {
        "NAME1": "Somewhere",
        "TYPE": "populatedPlace",
        "LOCAL_TYPE": "Hamlet",
        "COUNTRY": "England",
    }
    pair = _normalize_row(row)
    assert pair is not None
    toponym, _ = pair
    assert toponym["ref"] == "Somewhere@-"
    assert "region" not in toponym


def test_iter_csv_rows_detects_column_drift(tmp_path: Path):
    """OS Open Names schema has evolved across releases (e.g. CENSUS_CODE
    added). A CSV with surplus columns triggers ColumnDriftError instead
    of silently misaligning every column after the drift point."""
    from wyrd.generators.kenning.ingesters.os_open_names import ColumnDriftError

    # 36 fields where _COLUMNS expects 34. csv.DictReader will put the
    # 2 extra values under the None key, which the guard checks for.
    csv_path = tmp_path / "drifted.csv"
    csv_path.write_text(",".join(f'"v{i}"' for i in range(36)) + "\n")
    import pytest

    with pytest.raises(ColumnDriftError, match="more columns than expected"):
        list(iter_csv_rows(csv_path))


def test_normalize_row_skips_empty_name():
    """A blank NAME1 isn't a place, even if all other fields are valid."""
    row = {
        "NAME1": "",
        "TYPE": "populatedPlace",
        "LOCAL_TYPE": "Village",
    }
    assert _normalize_row(row) is None


# ---------------------------------------------------------------------------
# build_events
# ---------------------------------------------------------------------------


def test_build_events_filters_and_emits_source_row(tmp_path: Path):
    csv_path = _write_fixture_csv(tmp_path)
    rows = list(iter_csv_rows(csv_path))
    events = build_events(rows)
    # First event is the synthetic source row.
    assert events[0]["_type"] == "source"
    assert events[0]["ref"] == OS_OPEN_NAMES_SOURCE_ID
    # 3 distinct toponyms (Acton@Cheshire, Acton@Suffolk, Manchester@Greater Manchester),
    # 4 attestations (Acton@Cheshire appears twice in the fixture so emits 2 attestations).
    toponyms = [e for e in events if e["_type"] == "toponym"]
    attestations = [e for e in events if e["_type"] == "attestation"]
    assert len(toponyms) == 3
    assert len(attestations) == 4


def test_build_events_dedupes_toponym_refs(tmp_path: Path):
    """Two CSV rows for the same (name, region) produce ONE toponym
    event (the kernel rejects duplicate adds) but TWO attestations."""
    csv_path = _write_fixture_csv(tmp_path)
    rows = list(iter_csv_rows(csv_path))
    events = build_events(rows)
    acton_cheshire_toponyms = [
        e for e in events if e["_type"] == "toponym" and e["ref"] == "Acton@Cheshire"
    ]
    acton_cheshire_attestations = [
        e for e in events if e["_type"] == "attestation" and e["toponym_ref"] == "Acton@Cheshire"
    ]
    assert len(acton_cheshire_toponyms) == 1
    assert len(acton_cheshire_attestations) == 2


def test_build_events_without_source_row(tmp_path: Path):
    """include_source_row=False suppresses the leading source event —
    useful when merging into an existing file that already has one."""
    csv_path = _write_fixture_csv(tmp_path)
    rows = list(iter_csv_rows(csv_path))
    events = build_events(rows, include_source_row=False)
    assert all(e["_type"] != "source" for e in events)


# ---------------------------------------------------------------------------
# ingest()
# ---------------------------------------------------------------------------


def test_ingest_dry_run_does_not_write_file(tmp_path: Path):
    csv_path = _write_fixture_csv(tmp_path)
    out_path = tmp_path / "os_open_names.jsonl"
    counts = ingest(csv_path, out_path, apply=False)
    assert counts["csv_rows_scanned"] == 6
    assert counts["populated_places_kept"] == 3
    assert counts["toponym_events"] == 3
    assert counts["attestation_events"] == 4
    assert "written" not in counts
    assert not out_path.exists()


def test_ingest_apply_writes_jsonl(tmp_path: Path):
    csv_path = _write_fixture_csv(tmp_path)
    out_path = tmp_path / "os_open_names.jsonl"
    counts = ingest(csv_path, out_path, apply=True)
    assert counts["written"] == 1
    assert out_path.exists()
    lines = out_path.read_text().splitlines()
    # 1 source + 3 toponyms + 4 attestations = 8.
    assert len(lines) == 8
    # Source row first.
    assert json.loads(lines[0])["_type"] == "source"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_ingest_os_open_names_dry_run(tmp_path: Path):
    csv_path = _write_fixture_csv(tmp_path)
    out_path = tmp_path / "os_open_names.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "ingest-os-open-names",
            str(csv_path),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert not out_path.exists()


def test_cli_ingest_os_open_names_apply(tmp_path: Path):
    csv_path = _write_fixture_csv(tmp_path)
    out_path = tmp_path / "os_open_names.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "ingest-os-open-names",
            str(csv_path),
            "--out",
            str(out_path),
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    # Counts visible in stderr-redirected output.
    assert "Populated places kept" in result.output
