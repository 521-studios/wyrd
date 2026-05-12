"""Tests for the Domesday ingester (wyrd-el93).

The real Hull dataset is a 4MB Microsoft Access .mdb file; tests
bypass that by constructing fixture ``places`` / ``placeforms`` dicts
in the shape ``access_parser.parse_table`` produces (column-name →
list-of-row-values) and routing them through
``_ingest_from_tables`` directly. Keeps the tests fast and
schema-stable without committing the .mdb into the repo.
"""

from __future__ import annotations

import sqlite3

from wyrd.generators.kenning.domesday_ingester import (
    COUNTY_CODE_TO_NAME,
    DOMESDAY_SOURCE_ID,
    _ingest_from_tables,
    upsert_domesday_source,
)


def _build_fixture_db() -> sqlite3.Connection:
    """Schema-equivalent in-memory DB for the toponym + source +
    toponym_attestation tables. Minimal — enough columns to exercise
    the ingester's insertion path."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE source (
            id TEXT PRIMARY KEY,
            author TEXT,
            title TEXT NOT NULL,
            year INTEGER,
            region TEXT,
            language_focus TEXT,
            notes TEXT
        );
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
    return conn


def _fixture_tables() -> tuple[dict[str, list], dict[str, list]]:
    """Two-place fixture: Abberley (Worcestershire) — confident
    identification — and a 'speculative' row that should be skipped
    by default."""
    places = {
        "PlacesIdx": [1, 6, 41],
        "County": ["WOR", "ESS", "HAM"],
        "Phillimore": ["15,8", "20,20. 24,51", "29,2"],
        "Hundred": ["`Doddingtree'", "`Winstree'", "Bowcombe"],
        "Vill": ["Abberley", "Abberton", "Abedestone"],
        "Area": [None, None, None],
        "XRefs": [None, None, None],
        "OSrefs": ["SO7567", "TL9919", "SZ5888"],
        "OScodes": [None, None, "speculative"],
    }
    placeforms = {
        "PlacesIdx": [1, 6, 41],
        "PlaceFormSub": [1, 1, 1],
        "County": ["WOR", "ESS", "HAM"],
        "Hundred": ["`Doddingtree'", "`Winstree'", "Bowcombe"],
        "Vill": ["Abberley", "Abberton", "Abedestone"],
        "OSref": ["SO7567", "TL9919", "SZ5888"],
        "OScodes": [None, None, "speculative"],
    }
    return places, placeforms


def test_upsert_domesday_source_creates_row_once() -> None:
    """The source row carries the required attribution (J.J.N. Palmer
    + Hull team, CC-BY-NC-SA reference). Idempotent — re-running is
    a no-op."""
    conn = _build_fixture_db()
    upsert_domesday_source(conn)
    upsert_domesday_source(conn)  # idempotent
    rows = list(conn.execute("SELECT * FROM source WHERE id = ?", (DOMESDAY_SOURCE_ID,)))
    assert len(rows) == 1
    row = rows[0]
    assert row["year"] == 1086
    assert "Palmer" in row["author"]
    assert "CC-BY-NC-SA" in row["notes"]


def test_ingest_writes_toponym_and_attestation_rows() -> None:
    """Apply path inserts a toponym row per (Vill, county) and one
    attestation row per PlaceForm entry with date_year=1086."""
    conn = _build_fixture_db()
    places, placeforms = _fixture_tables()
    counts = _ingest_from_tables(conn, places, placeforms, apply=True)
    assert counts["toponym_inserted"] == 2  # speculative one skipped
    assert counts["skipped_uncertain"] == 1
    assert counts["attestation_inserted"] == 2

    # Verify the toponym rows.
    rows = list(conn.execute("SELECT modern_name, country, region FROM toponym ORDER BY id"))
    assert [(r["modern_name"], r["country"], r["region"]) for r in rows] == [
        ("Abberley", "England", "Worcestershire"),
        ("Abberton", "England", "Essex"),
    ]
    # Verify the attestation rows carry 1086 + Phillimore in source_doc.
    atst = list(
        conn.execute("SELECT form, date_year, source_doc FROM toponym_attestation ORDER BY id")
    )
    assert all(a["date_year"] == 1086 for a in atst)
    assert "Phillimore 15,8" in atst[0]["source_doc"]
    assert "Hundred=" in atst[0]["source_doc"]
    assert "OS=SO7567" in atst[0]["source_doc"]


def test_ingest_skips_speculative_by_default() -> None:
    """OScodes='speculative' / 'unknown' rows are uncertain modern
    identifications; the default ingest omits them so the toponym
    corpus stays high-confidence."""
    conn = _build_fixture_db()
    places, placeforms = _fixture_tables()
    counts = _ingest_from_tables(conn, places, placeforms, apply=True)
    # The third row ('Abedestone', speculative) should NOT appear.
    cnt = conn.execute(
        "SELECT COUNT(*) FROM toponym WHERE modern_name = ?", ("Abedestone",)
    ).fetchone()[0]
    assert cnt == 0
    assert counts["skipped_uncertain"] == 1


def test_ingest_include_uncertain_keeps_speculative() -> None:
    """Operators can opt in to speculative + unknown identifications
    (lost vills, scholarly disagreements about modern equivalent)
    via include_uncertain=True."""
    conn = _build_fixture_db()
    places, placeforms = _fixture_tables()
    counts = _ingest_from_tables(conn, places, placeforms, apply=True, include_uncertain=True)
    assert counts["toponym_inserted"] == 3
    assert counts["skipped_uncertain"] == 0
    cnt = conn.execute(
        "SELECT COUNT(*) FROM toponym WHERE modern_name = ?", ("Abedestone",)
    ).fetchone()[0]
    assert cnt == 1


def test_ingest_is_idempotent() -> None:
    """Re-running over the same data inserts zero new rows. Both
    toponym and attestation paths dedupe (toponym by
    (modern_name, region); attestation by
    (toponym_id, form, year, source_doc))."""
    conn = _build_fixture_db()
    places, placeforms = _fixture_tables()
    _ingest_from_tables(conn, places, placeforms, apply=True)
    counts2 = _ingest_from_tables(conn, places, placeforms, apply=True)
    assert counts2["toponym_inserted"] == 0
    assert counts2["toponym_existing"] == 2
    assert counts2["attestation_inserted"] == 0
    assert counts2["attestation_existing"] == 2


def test_ingest_dry_run_does_not_write() -> None:
    """Without apply=True the call must not touch the DB. Counts the
    would-be-inserts so operators can preview before committing."""
    conn = _build_fixture_db()
    places, placeforms = _fixture_tables()
    counts = _ingest_from_tables(conn, places, placeforms, apply=False)
    # Would-be-inserted: 2 (speculative skipped).
    assert counts["toponym_inserted"] == 2
    assert counts["attestation_inserted"] == 2
    # ...but DB is empty.
    assert conn.execute("SELECT COUNT(*) FROM toponym").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM source").fetchone()[0] == 0


def test_ingest_resolves_county_code_to_full_name() -> None:
    """County codes (WOR, ESS, etc.) are resolved to full county
    names (Worcestershire, Essex) before insert. Keeps the DB
    human-readable rather than carrying Hull's abbreviation
    convention into downstream consumers."""
    conn = _build_fixture_db()
    places, placeforms = _fixture_tables()
    _ingest_from_tables(conn, places, placeforms, apply=True)
    regions = [r[0] for r in conn.execute("SELECT region FROM toponym ORDER BY id")]
    assert "Worcestershire" in regions
    assert "Essex" in regions
    # Unmapped code passes through unchanged (degrades gracefully).
    assert COUNTY_CODE_TO_NAME["WOR"] == "Worcestershire"


def test_ingest_progress_callback_fires() -> None:
    """Optional progress_callback gets (done, total) tuples; tests pin
    that it fires AT LEAST at the end of the walk so CLI throttling
    isn't trivially broken."""
    conn = _build_fixture_db()
    places, placeforms = _fixture_tables()
    callbacks: list[tuple[int, int]] = []
    _ingest_from_tables(
        conn,
        places,
        placeforms,
        apply=True,
        progress_callback=lambda d, t: callbacks.append((d, t)),
    )
    assert callbacks, "expected at least one progress callback"
    # Final emission reports completion.
    assert callbacks[-1] == (3, 3)


def test_ingest_writes_source_row_on_apply() -> None:
    """The first apply pass auto-inserts the open_domesday_hull source
    row so subsequent FK references resolve. Verifies the source row
    carries year=1086 + the attribution notes."""
    conn = _build_fixture_db()
    places, placeforms = _fixture_tables()
    _ingest_from_tables(conn, places, placeforms, apply=True)
    src = conn.execute("SELECT * FROM source WHERE id = ?", (DOMESDAY_SOURCE_ID,)).fetchone()
    assert src is not None
    assert src["year"] == 1086
    assert src["region"] == "England"
