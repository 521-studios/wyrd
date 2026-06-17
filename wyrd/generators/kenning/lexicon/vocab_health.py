"""Controlled-vocabulary health scan (wyrd-cgl4).

Counts non-canonical ``country`` / ``region`` / ``language`` values in the lexicon
DB against the Class-A controlled vocabularies (D49) — the automatic stale-DB /
dirty-data canary. A DB built before a normalization merged, or a future ingest
path that bypasses the guards, shows up here as ``stale`` (a value that *should
have folded* to its canonical) or ``dirty`` (a value not modeled at all).

Read-only. Pure logic — the CLI wrapper (``cli/lexicon/vocab_health.py``) handles
presentation. Severity buckets:

- ``ok``    — canonical; nothing to do.
- ``stale`` — a known alias that was never folded (e.g. ``Republic of Ireland``);
  fixed by re-running normalization / rebuilding from the clean L2.
- ``dirty`` — not modeled at all (unknown country/language, quarantined or
  unknown region, country-as-region); needs a vocab decision or a data fix.
- ``info``  — expected / out-of-scope: NULLs, non-England regions (no model yet),
  and deferred zones (``England (Danelaw)`` → awaits wyrd-hytz).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

from wyrd.generators.kenning.lexicon.controlled_vocab import (
    CountryValidationError,
    LanguageValidationError,
    canonicalize_country,
    canonicalize_language,
)
from wyrd.generators.kenning.lexicon.region_model import RegionStatus, classify_region

# The closed set of severities. Declared once here; the CLI sort order + the
# stale/dirty violation filter both reference these same four.
Severity = Literal["ok", "stale", "dirty", "info"]

# region-classification status -> severity. Total over RegionStatus; the lookup
# below defaults to "dirty" so a future classify_region bucket surfaces as a
# violation rather than crashing the canary.
_REGION_SEVERITY: dict[RegionStatus, Severity] = {
    "node": "ok",
    "alias": "stale",
    "zone": "info",  # deferred cultural zone — expected (wyrd-hytz)
    "non-england": "info",  # no model yet (wyrd-3q6m.5 territory)
    "empty": "info",
    "quarantine": "dirty",
    "country-root": "dirty",
    "unknown": "dirty",
}


@dataclass(frozen=True)
class VocabFinding:
    dimension: str  # "country" | "region" | "language"
    value: str | None
    count: int  # rows carrying this value
    status: str  # dimension-specific classification
    severity: Severity
    context: str | None = None  # region findings carry their country (scope disambiguator)


def _classify_country(value: str | None) -> tuple[str, Severity]:
    if value is None or not value.strip():
        return "null", "info"  # whitespace-only treated as missing (cf. region/language)
    stripped = value.strip()
    try:
        canon = canonicalize_country(stripped)
    except CountryValidationError:
        return "unknown", "dirty"
    # compare against the stripped value: padded-but-canonical (" England ") is ok,
    # not a spurious alias — canonicalize_* strips internally.
    return ("ok", "ok") if canon == stripped else ("alias", "stale")


def _classify_language(value: str | None) -> tuple[str, Severity]:
    if value is None or not value.strip():
        return "null", "info"
    stripped = value.strip()
    try:
        canon = canonicalize_language(stripped)
    except LanguageValidationError:
        return "non-canonical", "dirty"
    return ("ok", "ok") if canon == stripped else ("alias", "stale")


def _rows(conn: sqlite3.Connection, sql: str) -> list[tuple]:
    """Execute ``sql`` returning plain tuples regardless of the connection's
    ``row_factory`` (forced off on a local cursor) — so the scans don't depend on,
    and can't be broken by, however the caller configured the connection (Gemini)."""
    cur = conn.cursor()
    cur.row_factory = None
    try:
        return cur.execute(sql).fetchall()
    finally:
        cur.close()


def scan_countries(conn: sqlite3.Connection) -> list[VocabFinding]:
    out = []
    for value, n in _rows(
        conn, "SELECT country, COUNT(*) FROM toponym GROUP BY country ORDER BY COUNT(*) DESC"
    ):
        status, severity = _classify_country(value)
        out.append(VocabFinding("country", value, n, status, severity))
    return out


def scan_regions(conn: sqlite3.Connection) -> list[VocabFinding]:
    out = []
    for value, country, n in _rows(
        conn,
        "SELECT region, country, COUNT(*) FROM toponym "
        "GROUP BY region, country ORDER BY COUNT(*) DESC",
    ):
        status = classify_region(value, country=country)
        out.append(
            VocabFinding(
                "region", value, n, status, _REGION_SEVERITY.get(status, "dirty"), context=country
            )
        )
    return out


def scan_languages(conn: sqlite3.Connection) -> list[VocabFinding]:
    out = []
    for value, n in _rows(
        conn, "SELECT language, COUNT(*) FROM etymon GROUP BY language ORDER BY COUNT(*) DESC"
    ):
        status, severity = _classify_language(value)
        out.append(VocabFinding("language", value, n, status, severity))
    return out


def scan(conn: sqlite3.Connection) -> dict[str, list[VocabFinding]]:
    """All three dimensions, each a list of per-value findings (rows aggregated)."""
    return {
        "country": scan_countries(conn),
        "region": scan_regions(conn),
        "language": scan_languages(conn),
    }


def violations(findings: dict[str, list[VocabFinding]]) -> list[VocabFinding]:
    """The stale + dirty findings across all dimensions — what ``--strict`` gates on."""
    return [f for fs in findings.values() for f in fs if f.severity in ("stale", "dirty")]
