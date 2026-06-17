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

from wyrd.generators.kenning.lexicon.controlled_vocab import (
    CountryValidationError,
    LanguageValidationError,
    canonicalize_country,
    canonicalize_language,
)
from wyrd.generators.kenning.lexicon.region_model import classify_region

# region-classification status -> severity bucket
_REGION_SEVERITY = {
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
    severity: str  # ok | stale | dirty | info


def _classify_country(value: str | None) -> tuple[str, str]:
    if value is None:
        return "null", "info"
    try:
        canon = canonicalize_country(value)
    except CountryValidationError:
        return "unknown", "dirty"
    return ("ok", "ok") if canon == value else ("alias", "stale")


def _classify_language(value: str | None) -> tuple[str, str]:
    if value is None or not value.strip():
        return "null", "info"
    try:
        canon = canonicalize_language(value)
    except LanguageValidationError:
        return "non-canonical", "dirty"
    return ("ok", "ok") if canon == value else ("alias", "stale")


def scan_countries(conn: sqlite3.Connection) -> list[VocabFinding]:
    rows = conn.execute(
        "SELECT country AS v, COUNT(*) AS n FROM toponym GROUP BY country ORDER BY n DESC"
    ).fetchall()
    out = []
    for r in rows:
        status, severity = _classify_country(r["v"])
        out.append(VocabFinding("country", r["v"], r["n"], status, severity))
    return out


def scan_regions(conn: sqlite3.Connection) -> list[VocabFinding]:
    rows = conn.execute(
        "SELECT region AS v, country AS c, COUNT(*) AS n "
        "FROM toponym GROUP BY region, country ORDER BY n DESC"
    ).fetchall()
    out = []
    for r in rows:
        status = classify_region(r["v"], country=r["c"])
        out.append(VocabFinding("region", r["v"], r["n"], status, _REGION_SEVERITY[status]))
    return out


def scan_languages(conn: sqlite3.Connection) -> list[VocabFinding]:
    rows = conn.execute(
        "SELECT language AS v, COUNT(*) AS n FROM etymon GROUP BY language ORDER BY n DESC"
    ).fetchall()
    out = []
    for r in rows:
        status, severity = _classify_language(r["v"])
        out.append(VocabFinding("language", r["v"], r["n"], status, severity))
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
