"""Hundred Rolls (1279-80) ingester — wyrd-3atv / Phase 3c.

Reads a transcribed Rotuli Hundredorum (or modern county-by-county
index) CSV and emits ``toponym`` + ``attestation`` events to
``data/mining/rotuli_hundredorum.jsonl``. Date year is 1279 (the
Edward I survey).

This is the Middle English bridge between Domesday (1086) and
Speed/Hearth Tax (1611+) — fills the gap in the Dashboard F (Tier 3
period-form) coverage.

Why the operator provides a CSV
==============================

The Rotuli Hundredorum (Record Commission, 1812-18) is a Latin
manuscript transcription with non-uniform formatting; no clean
machine-readable dump exists. The expected ingest workflow is:

1. Operator transcribes / OCRs / extracts entries from one of the
   PD sources (Record Commission edition on Internet Archive,
   Adolphus Ballard's 1916 county work, etc.) into a CSV.
2. CSV columns the ingester reads:

   - ``vill`` (required) — settlement name as it appears in the
     1279 Roll. Often Latinized: ``Tunes``, ``Caisteus``, etc.
   - ``hundred`` — administrative hundred grouping (optional).
   - ``county`` — modern county equivalent (goes to toponym.region).
   - ``country`` — England / Wales (defaults to England).
   - ``modern_name`` — the scholar's modern-name identification
     when known. When absent, the vill IS the modern_name (i.e.
     the place still bears its 1279 form).

3. ``lexicon ingest-hundred-rolls <csv-path> --apply`` writes events
   to the JSONL.

Cross-era cross-linking is automatic: if a Hundred Rolls entry's
``modern_name`` matches a Domesday or OS Open Names toponym's
``modern_name@region``, all three attestation rows attach to the
same toponym in the rebuilt DB — exactly the Dashboard F use case.

The JSONL file is in git (small relative to OS Open Names — ~50K
entries × ~150 bytes ≈ 7-8MB).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.ingesters import _csv_events
from wyrd.generators.kenning.jsonl.dump import toponym_ref

HUNDRED_ROLLS_SOURCE_ID = "rotuli_hundredorum"
HUNDRED_ROLLS_YEAR = 1279

# Required + optional CSV columns. Required must be present + non-
# empty; optional may be absent or blank.
_REQUIRED_COLUMNS = ("vill",)
_OPTIONAL_COLUMNS = ("hundred", "county", "country", "modern_name")
_ALL_COLUMNS = _REQUIRED_COLUMNS + _OPTIONAL_COLUMNS


class CsvSchemaError(ValueError):
    """Raised when the input CSV doesn't carry the required columns.
    System-boundary validation — the format is operator-supplied so
    surface mistakes loudly rather than silently producing zero rows."""


_SOURCE_ROW = {
    "_type": "source",
    "ref": HUNDRED_ROLLS_SOURCE_ID,
    "author": "Edward I survey (transcribed; see notes)",
    "title": "Rotuli Hundredorum",
    "year": HUNDRED_ROLLS_YEAR,
    "region": "England",
    "language_focus": "middle-english",
    "notes": (
        "Hundred Rolls of 1279-80 — Edward I's survey of England by "
        "hundreds. Public-domain transcriptions: Record Commission "
        "(1812-18) on Internet Archive; Adolphus Ballard's modern "
        "indices. Operator transcribes/OCRs entries into a CSV "
        "(columns: vill, hundred, county, country, modern_name) and "
        "feeds via `lexicon ingest-hundred-rolls`. Fills the ME-era "
        "attestation layer between Domesday (1086) and Speed (1611)."
    ),
}


def _normalize_row(row: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Convert one CSV row into a (toponym, attestation) event pair.

    Returns ``None`` if the row is unusable (blank vill, all-whitespace
    after strip). Modern-name resolution falls back to the vill itself
    when ``modern_name`` is absent or blank — many small 13th-century
    settlements still bear their original name.
    """
    vill = (row.get("vill") or "").strip()
    if not vill:
        return None

    modern_name = (row.get("modern_name") or "").strip() or vill
    region = (row.get("county") or "").strip() or None
    country = (row.get("country") or "").strip() or "England"
    hundred = (row.get("hundred") or "").strip() or None

    ref = toponym_ref(modern_name, region)
    toponym_event: dict[str, Any] = {
        "_type": "toponym",
        "ref": ref,
        "modern_name": modern_name,
        "country": country,
    }
    if region:
        toponym_event["region"] = region

    # source_doc carries enough provenance to find the source entry
    # (hundred + county) when reviewing attestation rows.
    parts = ["Hundred Rolls 1279"]
    if hundred:
        parts.append(f"Hundred of {hundred}")
    if region:
        parts.append(region)
    source_doc = "; ".join(parts)

    attestation_event = {
        "_type": "attestation",
        "toponym_ref": ref,
        "form": vill,
        "date_year": HUNDRED_ROLLS_YEAR,
        "source_doc": source_doc,
    }
    return toponym_event, attestation_event


def iter_csv_rows(csv_path: str | Path) -> Iterable[dict[str, str]]:
    """Yield CSV rows as dicts. Expects a header row naming at least
    ``vill``; raises :class:`CsvSchemaError` otherwise.

    Unlike OS Open Names (headerless, fixed-column-order), the
    Hundred Rolls CSV is operator-built and uses named columns so
    transcribers can omit fields they don't have data for.
    """
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise CsvSchemaError(f"{csv_path}: empty file (no header row)")
        missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise CsvSchemaError(
                f"{csv_path}: required columns missing: {missing}. "
                f"Expected at least {_REQUIRED_COLUMNS} in the header; "
                f"optional: {_OPTIONAL_COLUMNS}"
            )
        yield from reader


def build_events(
    csv_rows: Iterable[dict[str, str]],
    *,
    include_source_row: bool = True,
) -> list[dict[str, Any]]:
    """Hundred Rolls: CSV rows → JSONL events, via the shared builder fed this
    source's metadata row + row mapper (orchestration in _csv_events)."""
    return _csv_events.build_events(
        csv_rows,
        source_row=_SOURCE_ROW,
        normalize_row=_normalize_row,
        include_source_row=include_source_row,
    )


def ingest(csv_path: str | Path, out_path: str | Path, *, apply: bool = False) -> dict[str, int]:
    """Hundred Rolls: read CSV → write JSONL events (dry-run by default), via the
    shared ingester fed this source's CSV reader + event builder (orchestration in _csv_events)."""
    return _csv_events.ingest(
        csv_path,
        out_path,
        iter_csv_rows=iter_csv_rows,
        build_events=build_events,
        apply=apply,
    )
