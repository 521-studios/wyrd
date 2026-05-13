"""Tests for the cross-era timeline + coverage report — wyrd-ub76."""

from __future__ import annotations

import sqlite3

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.era_timeline import (
    ERA_ANGLO_SAXON,
    ERA_EARLY_MODERN,
    ERA_MIDDLE_ENGLISH,
    ERA_MODERN,
    ERA_ORDER,
    ERA_UNDATED,
    bucket_year,
    fetch_era_coverage,
    fetch_era_timeline,
    format_era_coverage,
    format_era_timeline,
)

# ---------------------------------------------------------------------------
# bucket_year — pure function, the most important unit
# ---------------------------------------------------------------------------


def test_bucket_year_anglo_saxon_pre_1100():
    """1086 is Domesday; anything before 1100 buckets as Anglo-Saxon
    even though strictly that label runs to 1066. The Domesday
    landscape is what the bucket means in practice."""
    assert bucket_year(700) == ERA_ANGLO_SAXON
    assert bucket_year(1086) == ERA_ANGLO_SAXON
    assert bucket_year(1099) == ERA_ANGLO_SAXON


def test_bucket_year_middle_english_1100_to_1500():
    """1100 is the boundary — included in Middle English (not
    Anglo-Saxon). Hundred Rolls 1279 should land here."""
    assert bucket_year(1100) == ERA_MIDDLE_ENGLISH
    assert bucket_year(1279) == ERA_MIDDLE_ENGLISH
    assert bucket_year(1499) == ERA_MIDDLE_ENGLISH


def test_bucket_year_early_modern_1500_to_1800():
    """Speed 1611 and Hearth Tax 1664 should land here."""
    assert bucket_year(1500) == ERA_EARLY_MODERN
    assert bucket_year(1611) == ERA_EARLY_MODERN
    assert bucket_year(1664) == ERA_EARLY_MODERN
    assert bucket_year(1799) == ERA_EARLY_MODERN


def test_bucket_year_modern_1800_plus():
    """OS Open Names (present-day) buckets here. So does any
    Wikipedia-era reference."""
    assert bucket_year(1800) == ERA_MODERN
    assert bucket_year(2026) == ERA_MODERN


def test_bucket_year_undated_for_none():
    """No date_year → UNDATED bucket. Coverage metrics exclude this
    bucket since it can't contribute to chronological spread."""
    assert bucket_year(None) == ERA_UNDATED


# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(
        """
        CREATE TABLE toponym (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modern_name TEXT NOT NULL,
            country TEXT,
            region TEXT
        );
        CREATE TABLE toponym_attestation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            toponym_id INTEGER NOT NULL REFERENCES toponym(id) ON DELETE CASCADE,
            form TEXT NOT NULL,
            date_year INTEGER,
            source_doc TEXT
        );
        """
    )
    yield c
    c.close()


def _insert_toponym(conn, modern_name, region=None, country="England"):
    cur = conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
        (modern_name, country, region),
    )
    return cur.lastrowid


def _insert_attestation(conn, toponym_id, form, date_year, source_doc=None):
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (?, ?, ?, ?)",
        (toponym_id, form, date_year, source_doc),
    )


# ---------------------------------------------------------------------------
# fetch_era_timeline
# ---------------------------------------------------------------------------


def test_fetch_era_timeline_full_four_eras(conn):
    """A toponym with attestations across all four eras populates
    every bucket; era_count == 4."""
    tid = _insert_toponym(conn, "Acton", "Cheshire")
    _insert_attestation(conn, tid, "Aclitone", 1086, "Domesday")
    _insert_attestation(conn, tid, "Acton", 1279, "Hundred Rolls 1279")
    _insert_attestation(conn, tid, "Acton", 1611, "Speed 1611")
    _insert_attestation(conn, tid, "Acton", 2025, "OS Open Names")

    timeline = fetch_era_timeline(conn, "Acton@Cheshire")
    assert timeline is not None
    assert timeline.modern_name == "Acton"
    assert timeline.region == "Cheshire"
    assert timeline.era_count == 4
    assert timeline.total_attestations == 4
    assert len(timeline.attestations_by_era[ERA_ANGLO_SAXON]) == 1
    assert timeline.attestations_by_era[ERA_ANGLO_SAXON][0].form == "Aclitone"


def test_fetch_era_timeline_partial_coverage(conn):
    """A toponym with only modern + early-modern eras: era_count == 2,
    missing-era buckets stay empty (not absent)."""
    tid = _insert_toponym(conn, "Stourbridge", "Worcestershire")
    _insert_attestation(conn, tid, "Stourbridge", 1664, "Hearth Tax 1664")
    _insert_attestation(conn, tid, "Stourbridge", 2025, "OS Open Names")

    timeline = fetch_era_timeline(conn, "Stourbridge@Worcestershire")
    assert timeline is not None
    assert timeline.era_count == 2
    assert timeline.attestations_by_era[ERA_ANGLO_SAXON] == []
    assert timeline.attestations_by_era[ERA_MIDDLE_ENGLISH] == []


def test_fetch_era_timeline_undated_excluded_from_era_count(conn):
    """An undated attestation goes to UNDATED; era_count excludes it
    (coverage means chronological spread, not 'have an attestation')."""
    tid = _insert_toponym(conn, "Mystery", "Suffolk")
    _insert_attestation(conn, tid, "Mistere", None, "scholar notes")
    _insert_attestation(conn, tid, "Mystery", 2025, "OS Open Names")

    timeline = fetch_era_timeline(conn, "Mystery@Suffolk")
    assert timeline is not None
    assert timeline.era_count == 1  # only Modern is dated
    assert len(timeline.attestations_by_era[ERA_UNDATED]) == 1


def test_fetch_era_timeline_missing_returns_none(conn):
    """Unknown ref → None (not exception). CLI surfaces the absence."""
    assert fetch_era_timeline(conn, "Nowhere@Nowhereshire") is None


def test_fetch_era_timeline_bare_name_picks_first_region(conn):
    """Bare-name lookup with multiple regional matches: deterministic
    first-match (ASCII-sorted region) with a fallback to null-region
    last."""
    _insert_toponym(conn, "Acton", "Cheshire")
    _insert_toponym(conn, "Acton", "Worcestershire")
    timeline = fetch_era_timeline(conn, "Acton")
    assert timeline is not None
    assert timeline.region == "Cheshire"  # ASCII-first


def test_fetch_era_timeline_null_region_sentinel(conn):
    """@- (or @) means "specifically null region"."""
    _insert_toponym(conn, "Generic", region=None)
    _insert_toponym(conn, "Generic", region="Hampshire")
    timeline = fetch_era_timeline(conn, "Generic@-")
    assert timeline is not None
    assert timeline.region is None


def test_fetch_era_timeline_bare_name_regional_beats_null_region(conn):
    """Bare-name lookup matching one null-region + one regional row:
    the regional row wins. The ORDER BY in _resolve_toponym_id puts
    non-null regions before nulls, so this is the documented
    preference — bare-name lookup is "any region, prefer specific
    over null." Operator who wants the null-region row uses @-."""
    _insert_toponym(conn, "Acton", region=None)
    _insert_toponym(conn, "Acton", region="Cheshire")
    timeline = fetch_era_timeline(conn, "Acton")
    assert timeline is not None
    assert timeline.region == "Cheshire"  # regional, not None


def test_fetch_era_timeline_bare_name_only_null_region(conn):
    """When the only matching row IS the null-region one, bare-name
    lookup still finds it (the IS NULL ordering pushes it to last
    but it's the sole candidate)."""
    _insert_toponym(conn, "Solo", region=None)
    timeline = fetch_era_timeline(conn, "Solo")
    assert timeline is not None
    assert timeline.region is None


# ---------------------------------------------------------------------------
# fetch_era_coverage
# ---------------------------------------------------------------------------


def test_fetch_era_coverage_empty_db(conn):
    """Empty corpus is well-defined: zeros everywhere, 0% coverage."""
    report = fetch_era_coverage(conn)
    assert report.total_toponyms == 0
    assert report.toponyms_with_attestations == 0
    assert report.coverage_3plus_pct == 0.0
    assert all(report.per_era_counts[e] == 0 for e in ERA_ORDER)


def test_fetch_era_coverage_mixed_corpus(conn):
    """Two well-attested + one 1-era-only + one no-attestation = the
    full distribution shape. coverage_3plus_pct should reflect the
    well-attested ones."""
    t1 = _insert_toponym(conn, "Acton")
    t2 = _insert_toponym(conn, "Beverley")
    _insert_toponym(conn, "Cottesmore")  # one-era-only
    _insert_toponym(conn, "Empty")  # no attestations at all

    # Acton: all 4 dated eras → 4
    for y in (1086, 1279, 1611, 2025):
        _insert_attestation(conn, t1, "form", y)
    # Beverley: 3 dated eras → 3
    for y in (1086, 1279, 2025):
        _insert_attestation(conn, t2, "form", y)
    # Cottesmore: 1 dated era → 1
    _insert_attestation(conn, _insert_toponym(conn, "Cottesmore2"), "form", 2025)

    report = fetch_era_coverage(conn)
    assert report.total_toponyms == 5
    assert report.toponyms_with_attestations == 3
    # Acton (4) + Beverley (3) → 2 of 5 = 40%
    assert report.coverage_3plus_pct == pytest.approx(40.0)
    # 0 eras: Empty + Cottesmore (no attestations) = 2
    # 1 era: Cottesmore2 = 1
    # 3 eras: Beverley = 1
    # 4 eras: Acton = 1
    assert report.era_count_distribution == {0: 2, 1: 1, 3: 1, 4: 1}


def test_fetch_era_coverage_undated_only_toponym(conn):
    """A toponym with only undated attestations has 0 dated eras
    even though it has attestations."""
    tid = _insert_toponym(conn, "Mystery")
    _insert_attestation(conn, tid, "Mistere", None)
    report = fetch_era_coverage(conn)
    assert report.total_toponyms == 1
    assert report.toponyms_with_attestations == 1
    assert report.era_count_distribution[0] == 1
    assert report.per_era_counts[ERA_UNDATED] == 1


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_format_era_timeline_renders_buckets(conn):
    tid = _insert_toponym(conn, "Acton", "Cheshire")
    _insert_attestation(conn, tid, "Aclitone", 1086, "Domesday")
    _insert_attestation(conn, tid, "Acton", 1279, "Hundred Rolls")
    timeline = fetch_era_timeline(conn, "Acton@Cheshire")
    out = format_era_timeline(timeline)
    assert "# Acton @Cheshire (England)" in out
    assert "Eras with coverage: 2/4" in out
    assert "## Anglo-Saxon (1)" in out
    assert "1086: `Aclitone` — Domesday" in out
    assert "## Middle English (1)" in out


def test_format_era_timeline_empty_attestations(conn):
    """A toponym row with zero attestations renders the empty-state
    marker rather than a sea of empty buckets."""
    _insert_toponym(conn, "Empty", "Nowhereshire")
    timeline = fetch_era_timeline(conn, "Empty@Nowhereshire")
    out = format_era_timeline(timeline)
    assert "_no attestations on this toponym_" in out


def test_format_era_coverage_table(conn):
    """Coverage report renders both per-era and histogram tables."""
    t1 = _insert_toponym(conn, "Acton")
    for y in (1086, 1279, 2025):
        _insert_attestation(conn, t1, "form", y)
    report = fetch_era_coverage(conn)
    out = format_era_coverage(report)
    assert "# Cross-era coverage report" in out
    assert "Toponyms with ≥3 dated eras: 100.0%" in out
    assert "## Per-era coverage" in out
    assert "## Dated-era-count distribution" in out


# ---------------------------------------------------------------------------
# CLI integration — uses a tmp_path SQLite + the existing _readonly_lexicon
# wiring so the file-URI/quoted-path code path is exercised.
# ---------------------------------------------------------------------------


@pytest.fixture()
def cli_db(tmp_path):
    """Create a tmp SQLite with minimal schema + a sample row, return
    its path."""
    db_path = tmp_path / "lexicon.db"
    c = sqlite3.connect(db_path)
    c.executescript(
        """
        CREATE TABLE toponym (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modern_name TEXT NOT NULL,
            country TEXT,
            region TEXT
        );
        CREATE TABLE toponym_attestation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            toponym_id INTEGER NOT NULL REFERENCES toponym(id) ON DELETE CASCADE,
            form TEXT NOT NULL,
            date_year INTEGER,
            source_doc TEXT
        );
        """
    )
    c.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES ('Acton', 'England', 'Cheshire')"
    )
    c.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (1, 'Aclitone', 1086, 'Domesday')"
    )
    c.commit()
    c.close()
    return db_path


def test_cli_era_timeline_renders(cli_db):
    runner = CliRunner()
    result = runner.invoke(
        cli_root, ["lexicon", "era-timeline", "Acton@Cheshire", "--db", str(cli_db)]
    )
    assert result.exit_code == 0, result.output
    assert "Anglo-Saxon" in result.output
    assert "Aclitone" in result.output


def test_cli_era_timeline_missing_exits_1(cli_db):
    runner = CliRunner()
    result = runner.invoke(
        cli_root, ["lexicon", "era-timeline", "Nowhere@Nowhereshire", "--db", str(cli_db)]
    )
    assert result.exit_code == 1
    assert "No toponym found" in result.output


def test_cli_era_coverage_renders(cli_db):
    runner = CliRunner()
    result = runner.invoke(cli_root, ["lexicon", "era-coverage", "--db", str(cli_db)])
    assert result.exit_code == 0, result.output
    assert "Cross-era coverage report" in result.output
    assert "Per-era coverage" in result.output
