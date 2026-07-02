"""OS Open Names (Ordnance Survey) ingester — wyrd-3ypp / Phase 3b.

Reads the OS Open Names CSV dataset and emits ``toponym`` +
``attestation`` events to ``data/mining/os_open_names.jsonl`` (an L2
file). Operator downloads the source CSV separately (~200MB, OGL v3)
and points the ingester at it; the ingester filters to populated
places and writes JSONL events.

Why JSONL events vs. direct DB insert:

The OS Open Names rowcount is too large for git (~1M rows; ~200K
after filtering to populated places ≈ 30-50MB JSONL). The JSONL
file is gitignored and regenerated on demand from the L1 CSV — same
pattern as the bulk wiktextract sources. Operators run:

    lexicon ingest-os-open-names sources/os_open_names.csv --apply
    lexicon rebuild-from-jsonl

after a fresh checkout. The resulting toponym + attestation rows in
the DB are the same shape any scholar-mined source produces; the
matcher (lexicon decompose) treats them identically.

CSV schema (per OS Open Names data product spec):

Columns this ingester reads:
- ``ID`` — internal OS identifier (unused; kept for future ref).
- ``NAMES_URI`` — stable URI for the place; goes into source_doc.
- ``NAME1`` — primary place name (the modern_name).
- ``TYPE`` — high-level category. We filter to ``populatedPlace``.
- ``LOCAL_TYPE`` — fine-grained type (City, Town, Village, Hamlet,
  Suburban Area, etc.). Carried in the attestation source_doc.
- ``COUNTY_UNITARY`` — administrative unit; goes to toponym.region.
- ``COUNTRY`` — England / Scotland / Wales; goes to toponym.country.

License: Open Government Licence v3 (commercial-friendly). The
ingester records the OS attribution string in the source row's
notes field.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.ingesters import _csv_events
from wyrd.generators.kenning.jsonl.dump import toponym_ref
from wyrd.generators.kenning.jsonl.log import write_jsonl

# Source identity. Excluded from default dump-jsonl (see
# jsonl.dump.DEFAULT_BULK_EXCLUDED_SOURCES) — the JSONL is regenerated
# from the L1 CSV rather than round-tripped from the DB.
OS_OPEN_NAMES_SOURCE_ID = "os_open_names"

# CSV column names per the OS Open Names data product (CSV variant).
# Codified here so a column-name drift in a new OS release surfaces
# at one site rather than scattered through the loop.
_COLUMNS = (
    "ID",
    "NAMES_URI",
    "NAME1",
    "NAME1_LANG",
    "NAME2",
    "NAME2_LANG",
    "TYPE",
    "LOCAL_TYPE",
    "GEOMETRY_X",
    "GEOMETRY_Y",
    "MOST_DETAIL_VIEW_RES",
    "LEAST_DETAIL_VIEW_RES",
    "MBR_XMIN",
    "MBR_YMIN",
    "MBR_XMAX",
    "MBR_YMAX",
    "POSTCODE_DISTRICT",
    "POSTCODE_DISTRICT_URI",
    "POPULATED_PLACE",
    "POPULATED_PLACE_URI",
    "POPULATED_PLACE_TYPE",
    "DISTRICT_BOROUGH",
    "DISTRICT_BOROUGH_URI",
    "DISTRICT_BOROUGH_TYPE",
    "COUNTY_UNITARY",
    "COUNTY_UNITARY_URI",
    "COUNTY_UNITARY_TYPE",
    "REGION",
    "REGION_URI",
    "COUNTRY",
    "COUNTRY_URI",
    "RELATED_SPATIAL_OBJECT",
    "SAME_AS_DBPEDIA",
    "SAME_AS_GEONAMES",
)

# OS LOCAL_TYPE values that count as populated places worth ingesting.
# Filters out hydrography, transport networks, woodland features, etc.
_POPULATED_LOCAL_TYPES = frozenset(
    {
        "City",
        "Town",
        "Village",
        "Hamlet",
        "Suburban Area",
        "Other Settlement",
    }
)


_SOURCE_ROW = {
    "_type": "source",
    "ref": OS_OPEN_NAMES_SOURCE_ID,
    "author": "Ordnance Survey",
    "title": "OS Open Names",
    "region": "Great Britain",
    "language_focus": "mixed",
    "notes": (
        "Ordnance Survey OS Open Names data product. Open Government "
        "Licence v3.0. Source CSV downloaded separately to "
        "sources/os_open_names/ (gitignored, ~200MB); this JSONL is "
        "regenerated on demand via `lexicon ingest-os-open-names`. "
        "© Crown copyright and database right; "
        "see https://www.ordnancesurvey.co.uk/products/os-open-names"
    ),
}


def _format_source_doc(uri: str, local_type: str) -> str:
    """Stable source_doc string the build can use to identify the row
    came from OS Open Names. Embeds the LOCAL_TYPE (City/Town/etc.)
    so downstream queries can filter without re-parsing the URI."""
    return f"OS Open Names ({local_type}) {uri}"


def _row_is_populated_place(row: dict[str, str]) -> bool:
    """Filter test: TYPE=populatedPlace AND LOCAL_TYPE is one of the
    settlement classes (excludes transport, hydrography, etc.).
    Skipping reduces ~1M total rows to ~200K populated-place rows."""
    return row.get("TYPE") == "populatedPlace" and row.get("LOCAL_TYPE") in _POPULATED_LOCAL_TYPES


def _normalize_row(row: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Convert one OS Open Names CSV row into a (toponym, attestation)
    event pair. Returns ``None`` if the row should be skipped (wrong
    type, missing required fields)."""
    if not _row_is_populated_place(row):
        return None
    name = row.get("NAME1", "").strip()
    if not name:
        return None
    # COUNTRY / COUNTY_UNITARY sit AFTER the populated-place gate's columns
    # (TYPE/LOCAL_TYPE), so a row that passes the gate but is truncated before
    # them gets ``None`` from DictReader's restval (the drift guard only catches
    # EXTRA columns, not fewer). ``.get(col, "")`` returns that None as-is (the
    # default applies to absent keys, not present-None) — ``None.strip()`` would
    # crash the whole ingest. ``(... or "")`` coerces a present-None to "".
    country = (row.get("COUNTRY") or "").strip() or None
    region = (row.get("COUNTY_UNITARY") or "").strip() or None
    ref = toponym_ref(name, region)
    uri = row.get("NAMES_URI", "").strip()
    local_type = row.get("LOCAL_TYPE", "").strip() or "unknown"

    toponym_event = {
        "_type": "toponym",
        "ref": ref,
        "modern_name": name,
    }
    if country:
        toponym_event["country"] = country
    if region:
        toponym_event["region"] = region

    attestation_event = {
        "_type": "attestation",
        "toponym_ref": ref,
        "form": name,
        "source_doc": _format_source_doc(uri, local_type),
    }
    return toponym_event, attestation_event


class ColumnDriftError(ValueError):
    """Raised when the OS Open Names CSV's column count doesn't match
    :data:`_COLUMNS`. Per OS release notes, the layout has evolved
    across data product versions (e.g. ``CENSUS_CODE`` added in one
    revision), so we fail loudly rather than letting csv.DictReader
    silently misalign the surplus columns under the ``None`` key."""


def iter_csv_rows(csv_path: str | Path) -> Iterable[dict[str, str]]:
    """Yield CSV rows as dicts. Yields raw rows — caller filters.

    OS Open Names CSVs ship WITHOUT a header line (header lives in a
    separate doc spec), so the reader supplies column names explicitly
    from :data:`_COLUMNS`.

    Guards against silent column-count drift: if the first row has a
    different field count than ``_COLUMNS``, raises
    :class:`ColumnDriftError`. Without this check, a new OS release
    that adds a column (e.g. ``CENSUS_CODE``) would silently
    misalign every column after the drift point.
    """
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, fieldnames=list(_COLUMNS))
        first = True
        for row in reader:
            if first:
                first = False
                # The "extra columns" key DictReader uses to capture
                # surplus values is the literal None — checking for
                # its presence catches drift cheaply.
                extras = row.get(None)
                if extras:
                    raise ColumnDriftError(
                        f"OS Open Names CSV has more columns than expected "
                        f"({len(_COLUMNS) + len(extras)} vs {len(_COLUMNS)}). "
                        f"Schema may have drifted in a new OS release — "
                        f"update _COLUMNS in os_open_names.py."
                    )
            yield row


def build_events(
    csv_rows: Iterable[dict[str, str]],
    *,
    include_source_row: bool = True,
) -> list[dict[str, Any]]:
    """OS Open Names: CSV rows → JSONL events, via the shared builder fed this
    source's metadata row + row mapper (orchestration in _csv_events)."""
    return _csv_events.build_events(
        csv_rows,
        source_row=_SOURCE_ROW,
        normalize_row=_normalize_row,
        include_source_row=include_source_row,
    )


def ingest(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    apply: bool = False,
) -> dict[str, int]:
    """Read OS Open Names CSV, write JSONL events to ``out_path``.

    Returns counts. Dry-run (``apply=False``) computes the counts but
    doesn't write the file — useful for previewing before committing
    to a multi-minute parse.
    """
    csv_path = Path(csv_path)
    out_path = Path(out_path)

    counts = {
        "csv_rows_scanned": 0,
        "populated_places_kept": 0,
        "toponym_events": 0,
        "attestation_events": 0,
    }

    def _counting_rows() -> Iterable[dict[str, str]]:
        for row in iter_csv_rows(csv_path):
            counts["csv_rows_scanned"] += 1
            yield row

    events = build_events(_counting_rows())
    # Tally per-type. The source row is one of the events; subtract
    # it so the toponym/attestation counts match what the build will
    # actually insert.
    for event in events:
        if event["_type"] == "toponym":
            counts["toponym_events"] += 1
            counts["populated_places_kept"] += 1
        elif event["_type"] == "attestation":
            counts["attestation_events"] += 1

    if apply:
        write_jsonl(out_path, events)
        counts["written"] = 1
    return counts
