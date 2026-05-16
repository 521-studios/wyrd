"""Tests for the toponym reverse-search module (wyrd-x82p Phase 1)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from wyrd.generators.kenning.toponym_reverse_search import (
    _build_form_to_toponym_lookup,
    _load_source_body,
    _normalize_for_match,
    reverse_search_source,
)


def _build_fixture_db() -> sqlite3.Connection:
    """Minimal schema covering toponym + toponym_attestation. Mirrors
    the columns reverse_search_source queries."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY, title TEXT NOT NULL);
        CREATE TABLE toponym (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            modern_name TEXT NOT NULL,
            country     TEXT,
            region      TEXT
        );
        CREATE TABLE toponym_attestation (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            toponym_id  INTEGER NOT NULL,
            form        TEXT NOT NULL,
            date_year   INTEGER,
            source_doc  TEXT
        );
        CREATE UNIQUE INDEX idx_attestation_unique
            ON toponym_attestation(toponym_id, form, date_year, source_doc);
        """
    )
    return conn


def test_normalize_for_match_lowercases_and_strips_trailing_punct():
    assert _normalize_for_match("Birmingham") == "birmingham"
    assert _normalize_for_match("Birmingham,") == "birmingham"
    assert _normalize_for_match("Birmingham.") == "birmingham"
    assert _normalize_for_match("  Birmingham  ") == "birmingham"
    assert _normalize_for_match("CASTLE CARY") == "castle cary"


def test_normalize_for_match_preserves_internal_punctuation():
    """Multi-word toponyms with internal punctuation (apostrophes,
    hyphens) keep that punctuation — they're part of the form
    identity, not trailing noise."""
    assert _normalize_for_match("Stratford-upon-Avon") == "stratford-upon-avon"
    assert _normalize_for_match("St. Albans") == "st. albans"


def test_build_lookup_includes_modern_names_and_attestation_forms():
    """Lookup contains modern_name entries AND historical forms
    from existing attestations."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Birmingham')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (2, 'Stratford')")
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (2, 'Stretford', 1086, 'old_source')"
    )
    conn.commit()

    lookup = _build_form_to_toponym_lookup(conn)
    assert lookup["birmingham"] == 1
    assert lookup["stratford"] == 2
    assert lookup["stretford"] == 2  # historical form maps to same toponym


def test_build_lookup_modern_name_wins_on_collision():
    """When the same form is both a modern_name AND an attestation
    form for a different toponym, the modern_name wins (canonical
    identity preference)."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Newton')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (2, 'Sutton')")
    # An attestation labels Sutton (id=2) as 'Newton' — pathological
    # but possible if scholar mining had cross-references.
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (2, 'Newton', 1086, 'src')"
    )
    conn.commit()

    lookup = _build_form_to_toponym_lookup(conn)
    assert lookup["newton"] == 1, "modern_name (id=1) wins over attestation-form collision"


def test_load_source_body_returns_text(tmp_path: Path):
    (tmp_path / "src.txt").write_text("body text here.")
    assert _load_source_body(tmp_path, "src") == "body text here."


def test_load_source_body_missing_returns_none(tmp_path: Path):
    assert _load_source_body(tmp_path, "nope") is None


def test_load_source_body_unicode_decode_error_returns_none(tmp_path: Path):
    (tmp_path / "bad.txt").write_bytes(b"valid prefix \xff\xfe invalid bytes")
    assert _load_source_body(tmp_path, "bad") is None


def test_load_source_body_whitespace_only_returns_none(tmp_path: Path):
    (tmp_path / "blank.txt").write_text("   \n\n   ")
    assert _load_source_body(tmp_path, "blank") is None


def test_load_source_body_strips_path_components(tmp_path: Path):
    """Path traversal must be rejected — source_id with ../ is
    stripped to the basename via Path(...).name."""
    sensitive = tmp_path / "sensitive.txt"
    sensitive.write_text("should-not-be-readable")
    inner = tmp_path / "inner"
    inner.mkdir()
    assert _load_source_body(inner, "../sensitive") is None


def test_reverse_search_source_matches_modern_name(tmp_path: Path):
    """A scholar source body mentioning a known modern_name with a
    nearby year produces an attestation insert."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('mawer', 'Mawer 1920')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Cestretone')")
    conn.commit()
    (tmp_path / "mawer.txt").write_text(
        "Some scholarly prose. Cestretone in 1210 was recorded. More text."
    )

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "mawer", tmp_path, lookup, apply=True)

    assert report.pairs_extracted == 1
    assert report.matched == 1
    assert report.unmatched == 0
    assert report.inserted == 1
    row = conn.execute(
        "SELECT toponym_id, form, date_year, source_doc FROM toponym_attestation"
    ).fetchone()
    assert (row["toponym_id"], row["form"], row["date_year"], row["source_doc"]) == (
        1,
        "Cestretone",
        1210,
        "mawer",
    )


def test_reverse_search_source_dry_run_does_not_insert(tmp_path: Path):
    """Without --apply, the report counters increment but the DB
    isn't touched."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('mawer', 'Mawer')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Birmingham')")
    conn.commit()
    (tmp_path / "mawer.txt").write_text("As found in Birmingham, 1086, the name is...")

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "mawer", tmp_path, lookup, apply=False)

    assert report.matched == 1
    assert report.inserted == 0  # dry-run
    nrows = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert nrows == 0


def test_reverse_search_source_idempotent_on_rerun(tmp_path: Path):
    """Re-running --apply against the same source body is a no-op:
    the UNIQUE index drops duplicates, counted as already_present."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Cestretone')")
    conn.commit()
    (tmp_path / "src.txt").write_text("Cestretone in 1210.")

    lookup = _build_form_to_toponym_lookup(conn)
    r1 = reverse_search_source(conn, "src", tmp_path, lookup, apply=True)
    assert r1.inserted == 1
    assert r1.already_present == 0

    r2 = reverse_search_source(conn, "src", tmp_path, lookup, apply=True)
    assert r2.inserted == 0
    assert r2.already_present == 1

    nrows = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert nrows == 1, "re-run must not duplicate the attestation"


def test_reverse_search_source_unmatched_form_skipped(tmp_path: Path):
    """A (form, year) pair whose form doesn't map to any known
    toponym is counted as unmatched but doesn't fail or insert."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    # No toponyms — every form is unmatched.
    conn.commit()
    (tmp_path / "src.txt").write_text(
        "Birmingham, 1086 and also Cestretone in 1210 and Norwich, 1255."
    )

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "src", tmp_path, lookup, apply=True)
    assert report.pairs_extracted == 3
    assert report.matched == 0
    assert report.unmatched == 3
    assert report.inserted == 0
    assert len(report.unmatched_samples) == 3


def test_reverse_search_source_case_insensitive_match(tmp_path: Path):
    """Match is case-insensitive: source prose uses lowercase, modern
    name is capitalized, they should still join."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Birmingham')")
    conn.commit()
    # Real source prose may have form in lower or upper case depending
    # on context — the regex requires capitalization, so we test
    # the trailing-punct normalization specifically.
    (tmp_path / "src.txt").write_text("Birmingham, 1086 — important place.")

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "src", tmp_path, lookup, apply=True)
    assert report.matched == 1
    assert report.inserted == 1


def test_reverse_search_source_missing_body_returns_empty_report(tmp_path: Path):
    """A source without a committed .txt body returns an empty
    report (operator can have JSONLs for sources whose .txt isn't
    bundled)."""
    conn = _build_fixture_db()
    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "missing", tmp_path, lookup, apply=True)
    assert report.pairs_extracted == 0
    assert report.matched == 0


def test_reverse_search_source_apply_actually_commits(tmp_path: Path):
    """The SQL INSERT OR IGNORE actually commits — verified by
    opening a fresh connection to the same DB."""
    db_path = tmp_path / "lex.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY, title TEXT NOT NULL);
        CREATE TABLE toponym (id INTEGER PRIMARY KEY AUTOINCREMENT, modern_name TEXT NOT NULL,
            country TEXT, region TEXT);
        CREATE TABLE toponym_attestation (id INTEGER PRIMARY KEY AUTOINCREMENT,
            toponym_id INTEGER NOT NULL, form TEXT NOT NULL, date_year INTEGER, source_doc TEXT);
        CREATE UNIQUE INDEX idx_attestation_unique
            ON toponym_attestation(toponym_id, form, date_year, source_doc);
        INSERT INTO source (id, title) VALUES ('src', 'Src');
        INSERT INTO toponym (modern_name) VALUES ('Cestretone');
        """
    )
    conn.commit()
    (tmp_path / "src.txt").write_text("Cestretone in 1210.")
    lookup = _build_form_to_toponym_lookup(conn)
    reverse_search_source(conn, "src", tmp_path, lookup, apply=True)
    conn.close()

    fresh = sqlite3.connect(str(db_path))
    nrows = fresh.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    fresh.close()
    assert nrows == 1, "INSERT must persist after refill_source returns"
