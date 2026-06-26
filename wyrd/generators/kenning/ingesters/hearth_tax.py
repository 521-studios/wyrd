"""Hearth Tax returns (1660s-70s) ingester — wyrd-myh1 / Phase 3d.

Reads transcribed Hearth Tax returns (Charles II's 1662 tax on
hearths; collected ~1662-1674 in printed Record-Office editions) and
emits ``toponym`` + ``attestation`` events. The Hearth Tax gives the
finest-grained settlement coverage of the early-modern era — every
parish + every taxable township appears, not just market towns.

Unlike Speed 1611 (single date), the Hearth Tax was collected in
multiple rounds across the 1660s-70s; ``year_specific`` per-row
captures the actual collection year. The source-level
``date_year`` (1665) is the median; ``attestation.date_year`` is the
specific year of the return when known.

Same operator-CSV template as Hundred Rolls (wyrd-3atv) and Speed
(this PR). CSV columns the ingester reads:

- ``place_name`` (required) — taxed settlement as it appears in
  the return (parish, township, or hamlet).
- ``parish`` — parish container when ``place_name`` is a township
  or hamlet (optional; adds richer ``source_doc`` provenance).
- ``county`` — modern county.
- ``year_specific`` — actual collection year (1662-1674); falls
  back to ``DEFAULT_HEARTH_TAX_YEAR`` (1665) when blank or
  unparseable. Out-of-range integers warn via
  :class:`OutOfRangeYearWarning` so transcription errors surface
  to the operator without aborting the ingest.
- ``country`` — defaults to England (Hearth Tax was English-only).
- ``modern_name`` — present-day form when different; falls back
  to ``place_name``.

Cross-era cross-linking: a Hearth Tax modern_name matching a Hundred
Rolls or Speed modern_name attaches all attestations to one toponym.
"""

from __future__ import annotations

import csv
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.ingesters import _csv_events
from wyrd.generators.kenning.jsonl.log import write_jsonl

HEARTH_TAX_SOURCE_ID = "hearth_tax_1660s"
DEFAULT_HEARTH_TAX_YEAR = 1665
HEARTH_TAX_YEAR_RANGE = (1662, 1674)

_REQUIRED_COLUMNS = ("place_name",)
_OPTIONAL_COLUMNS = ("county", "parish", "year_specific", "country", "modern_name")


class OutOfRangeYearWarning(UserWarning):
    """A ``year_specific`` cell parsed as a valid int but fell outside
    the Hearth Tax collection window (1662-1674). Emitted at warning
    level (not raised) so the ingest completes, but the operator gets
    a signal that a transcription value is suspect."""


class CsvSchemaError(ValueError):
    """Required column missing from the operator-supplied CSV."""


_SOURCE_ROW = {
    "_type": "source",
    "ref": HEARTH_TAX_SOURCE_ID,
    "author": "Crown (Exchequer; transcribed by various county editors)",
    "title": "Hearth Tax Returns",
    "year": DEFAULT_HEARTH_TAX_YEAR,
    "region": "England",
    "language_focus": "early-modern-english",
    "notes": (
        "Hearth Tax returns 1662-1674 (Charles II) — England-wide "
        "household tax recorded per parish + township. PD; printed "
        "editions per county (Yorkshire, Essex, Surrey, Wiltshire, "
        "etc.) by county record societies. Operator extracts "
        "settlement-level entries into a CSV (columns: place_name, "
        "parish, county, year_specific, country, modern_name) and "
        "feeds via `lexicon ingest-hearth-tax`. Finer-grained than "
        "Speed 1611 "
        "because every taxable township appears, not just map-scale "
        "places. Fills early-modern attestation gap alongside Speed."
    ),
}


def _toponym_ref(modern_name: str, region: str | None) -> str:
    return f"{modern_name}@{region or '-'}"


def _parse_year(raw: str) -> int:
    """Parse a year_specific cell. Falls back to the default when
    blank or unparseable. Strict-mode (raise on garbage) was
    rejected: operator-supplied transcriptions sometimes have
    'c.1665' / '1662?' that we'd rather accept-with-default than
    abort on. Out-of-range integers DO emit a warning — the operator
    likely has a transcription error to investigate."""
    raw = (raw or "").strip()
    if not raw:
        return DEFAULT_HEARTH_TAX_YEAR
    try:
        year = int(raw)
    except ValueError:
        return DEFAULT_HEARTH_TAX_YEAR
    low, high = HEARTH_TAX_YEAR_RANGE
    if low <= year <= high:
        return year
    warnings.warn(
        f"year_specific={year} outside Hearth Tax collection window "
        f"{low}-{high}; using default {DEFAULT_HEARTH_TAX_YEAR}. "
        f"Likely a transcription error in the operator CSV.",
        OutOfRangeYearWarning,
        stacklevel=2,
    )
    return DEFAULT_HEARTH_TAX_YEAR


def _normalize_row(row: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """One CSV row → (toponym, attestation) event pair, or None."""
    place_name = (row.get("place_name") or "").strip()
    if not place_name:
        return None

    modern_name = (row.get("modern_name") or "").strip() or place_name
    region = (row.get("county") or "").strip() or None
    country = (row.get("country") or "").strip() or "England"
    parish = (row.get("parish") or "").strip() or None
    year = _parse_year(row.get("year_specific") or "")

    ref = _toponym_ref(modern_name, region)
    toponym_event: dict[str, Any] = {
        "_type": "toponym",
        "ref": ref,
        "modern_name": modern_name,
        "country": country,
    }
    if region:
        toponym_event["region"] = region

    parts = [f"Hearth Tax {year}"]
    if parish:
        parts.append(f"Parish of {parish}")
    if region:
        parts.append(region)
    source_doc = "; ".join(parts)

    attestation_event = {
        "_type": "attestation",
        "toponym_ref": ref,
        "form": place_name,
        "date_year": year,
        "source_doc": source_doc,
    }
    return toponym_event, attestation_event


def iter_csv_rows(csv_path: str | Path) -> Iterable[dict[str, str]]:
    """Yield CSV rows as dicts. Header must name at least
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
    """Hearth Tax: CSV rows → JSONL events, via the shared builder fed this
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
