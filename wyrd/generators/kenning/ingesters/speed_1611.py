"""John Speed (1611) ingester — wyrd-myh1 / Phase 3d.

Reads a transcribed extract of John Speed's *The Theatre of the
Empire of Great Britaine* (1611) — the county atlas that gives every
shire its first systematic published map — and emits ``toponym`` +
``attestation`` events with ``date_year=1611``.

Same operator-CSV template as the Hundred Rolls ingester (wyrd-3atv).
Multiple PD editions / Wikisource transcriptions are available; the
operator extracts entries into a CSV with named columns and feeds via
``lexicon ingest-speed-1611``.

CSV columns the ingester reads (named header row):

- ``place_name`` (required) — toponym as engraved on Speed's map.
- ``county`` — modern county equivalent → toponym.region.
- ``parish`` — parish container (optional).
- ``country`` — England / Wales / Scotland (defaults to England).
- ``modern_name`` — present-day form when different from Speed's;
  falls back to ``place_name`` when blank (most early-modern forms
  survive unchanged).

Cross-era cross-linking is automatic via toponym ref
(``modern_name@region``) — a Speed entry whose ``modern_name``
matches a Hundred Rolls or Domesday or OS Open Names entry attaches
all four attestations to the same toponym row.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.ingesters import _csv_events
from wyrd.generators.kenning.jsonl.log import write_jsonl

SPEED_1611_SOURCE_ID = "speed_1611"
SPEED_1611_YEAR = 1611

_REQUIRED_COLUMNS = ("place_name",)
_OPTIONAL_COLUMNS = ("county", "parish", "country", "modern_name")


class CsvSchemaError(ValueError):
    """Required column missing from the operator-supplied CSV."""


_SOURCE_ROW = {
    "_type": "source",
    "ref": SPEED_1611_SOURCE_ID,
    "author": "John Speed",
    "title": "The Theatre of the Empire of Great Britaine",
    "year": SPEED_1611_YEAR,
    "region": "Great Britain",
    "language_focus": "early-modern-english",
    "notes": (
        "John Speed's 1611 county atlas — the first systematic "
        "published mapping of every English shire. PD; multiple "
        "Wikisource / Internet Archive transcriptions. Operator "
        "extracts place names per county map into a CSV (columns: "
        "place_name, county, parish, country, modern_name) and feeds "
        "via `lexicon ingest-speed-1611`. Fills the early-modern "
        "attestation layer (1500-1700) between the Hundred Rolls "
        "(1279) and OS Open Names (present)."
    ),
}


def _toponym_ref(modern_name: str, region: str | None) -> str:
    return f"{modern_name}@{region or '-'}"


def _normalize_row(row: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """One CSV row → (toponym, attestation) event pair, or None if
    the row is unusable (blank place_name)."""
    place_name = (row.get("place_name") or "").strip()
    if not place_name:
        return None

    modern_name = (row.get("modern_name") or "").strip() or place_name
    region = (row.get("county") or "").strip() or None
    country = (row.get("country") or "").strip() or "England"
    parish = (row.get("parish") or "").strip() or None

    ref = _toponym_ref(modern_name, region)
    toponym_event: dict[str, Any] = {
        "_type": "toponym",
        "ref": ref,
        "modern_name": modern_name,
        "country": country,
    }
    if region:
        toponym_event["region"] = region

    parts = ["Speed 1611"]
    if parish:
        parts.append(f"Parish of {parish}")
    if region:
        parts.append(region)
    source_doc = "; ".join(parts)

    attestation_event = {
        "_type": "attestation",
        "toponym_ref": ref,
        "form": place_name,
        "date_year": SPEED_1611_YEAR,
        "source_doc": source_doc,
    }
    return toponym_event, attestation_event


def iter_csv_rows(csv_path: str | Path) -> Iterable[dict[str, str]]:
    """Yield CSV rows as dicts. Header row must name at least
    ``place_name``."""
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
    """Speed 1611: CSV rows → JSONL events, via the shared builder fed this
    source's metadata row + row mapper (orchestration in _csv_events)."""
    return _csv_events.build_events(
        csv_rows,
        source_row=_SOURCE_ROW,
        normalize_row=_normalize_row,
        include_source_row=include_source_row,
    )


def ingest(csv_path: str | Path, out_path: str | Path, *, apply: bool = False) -> dict[str, int]:
    """Read CSV → write JSONL events. Dry-run by default."""
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    counts = {
        "csv_rows_scanned": 0,
        "toponym_events": 0,
        "attestation_events": 0,
    }

    def _counting_rows() -> Iterable[dict[str, str]]:
        for row in iter_csv_rows(csv_path):
            counts["csv_rows_scanned"] += 1
            yield row

    events = build_events(_counting_rows())
    for event in events:
        if event["_type"] == "toponym":
            counts["toponym_events"] += 1
        elif event["_type"] == "attestation":
            counts["attestation_events"] += 1

    if apply:
        write_jsonl(out_path, events)
        counts["written"] = 1
    return counts
