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

import pytest

from wyrd.generators.kenning.ingesters.domesday import (
    COUNTY_CODE_TO_NAME,
    DOMESDAY_SOURCE_ID,
    _compute_source_doc,
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
    # COUNTY_CODE_TO_NAME is the source of truth for the code→name resolution.
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


def test_ingest_skips_rows_missing_vill_or_county() -> None:
    """Rows without ``Vill`` or ``County`` are counted into separate
    skip buckets so operators can spot data-quality issues in source
    MDB files (e.g. Hull's marginal Welsh entries with malformed
    rows). Neither bucket triggers an insert."""
    conn = _build_fixture_db()
    places = {
        "PlacesIdx": [1, 2, 3],
        "County": ["WOR", None, "ESS"],
        "Phillimore": ["15,8", "9,9", "20,20"],
        "Hundred": ["A", "B", "C"],
        "Vill": ["Good", "NoCounty", None],  # row 2 missing Vill
        "Area": [None, None, None],
        "XRefs": [None, None, None],
        "OSrefs": [None, None, None],
        "OScodes": [None, None, None],
    }
    placeforms = {
        "PlacesIdx": [1, 2, 3],
        "PlaceFormSub": [1, 1, 1],
        "County": ["WOR", None, "ESS"],
        "Hundred": ["A", "B", "C"],
        "Vill": ["Good", "NoCounty", None],
        "OSref": [None, None, None],
        "OScodes": [None, None, None],
    }
    counts = _ingest_from_tables(conn, places, placeforms, apply=True)
    assert counts["skipped_no_county"] == 1  # row 2 (None county)
    assert counts["skipped_no_vill"] == 1  # row 3 (None vill)
    assert counts["toponym_inserted"] == 1  # only "Good" survives


def test_ingest_attributes_welsh_counties_to_wales() -> None:
    """wyrd-el93 fix: Monmouthshire (MON) and Flintshire (FLN) are
    Welsh marches — Hull's dataset lists them but doesn't formally
    label country. Default 'England' would mis-attribute. Pin that
    the country column = 'Wales' for those codes."""
    conn = _build_fixture_db()
    places = {
        "PlacesIdx": [1, 2],
        "County": ["MON", "WOR"],
        "Phillimore": ["W,1", "15,8"],
        "Hundred": ["x", "y"],
        "Vill": ["WelshVill", "EnglishVill"],
        "Area": [None, None],
        "XRefs": [None, None],
        "OSrefs": [None, None],
        "OScodes": [None, None],
    }
    placeforms = {
        "PlacesIdx": [1, 2],
        "PlaceFormSub": [1, 1],
        "County": ["MON", "WOR"],
        "Hundred": ["x", "y"],
        "Vill": ["WelshVill", "EnglishVill"],
        "OSref": [None, None],
        "OScodes": [None, None],
    }
    _ingest_from_tables(conn, places, placeforms, apply=True)
    rows = list(conn.execute("SELECT modern_name, country FROM toponym ORDER BY modern_name"))
    by_name = {r["modern_name"]: r["country"] for r in rows}
    assert by_name["WelshVill"] == "Wales"
    assert by_name["EnglishVill"] == "England"


def test_ingest_dry_run_reports_existing_db_state() -> None:
    """wyrd-el93 fix: dry-run pre-populates the toponym + attestation
    caches from the current DB state, so it reports accurate
    inserted vs existing counts (not just 'every parsed row is new').
    Pin against a fixture where one toponym is already in the DB
    before the ingest starts."""
    conn = _build_fixture_db()
    # Pre-existing toponym + attestation from an earlier scholar
    # ingest. Same (modern_name, region) the Domesday ingest will
    # touch.
    conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES ('Abberley', 'England', 'Worcestershire')"
    )
    pre_id = conn.execute("SELECT id FROM toponym WHERE modern_name = 'Abberley'").fetchone()["id"]
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (?, ?, ?, ?)",
        (pre_id, "Abberley", 1086, "Phillimore 15,8; Hundred=`Doddingtree'; OS=SO7567"),
    )
    conn.commit()

    places, placeforms = _fixture_tables()
    counts = _ingest_from_tables(conn, places, placeforms, apply=False)

    # Of the 2 valid rows (speculative skipped), 1 already exists
    # in the DB and 1 is new. Dry-run must reflect both.
    assert counts["toponym_existing"] == 1
    assert counts["toponym_inserted"] == 1
    # Same for attestations: 1 already exists, 1 new.
    assert counts["attestation_existing"] == 1
    assert counts["attestation_inserted"] == 1


def test_ingest_rejects_unmapped_county_code() -> None:
    """An unknown county code (not in COUNTY_CODE_TO_NAME) falls back to the raw
    code as region, which the fail-closed Class-A guard rejects (wyrd-fxxf) —
    surfacing the unmapped code instead of silently inserting a dirty region.
    A new Hull code is a conscious one-line addition to COUNTY_CODE_TO_NAME.
    (Was: passed through dirty — the anti-pattern D49 removes.)"""
    from wyrd.generators.kenning.lexicon.region_model import RegionValidationError

    conn = _build_fixture_db()
    places = {
        "PlacesIdx": [1],
        "County": ["ZZZ"],
        "Phillimore": ["1,1"],
        "Hundred": ["x"],
        "Vill": ["Xville"],
        "Area": [None],
        "XRefs": [None],
        "OSrefs": [None],
        "OScodes": [None],
    }
    placeforms = {
        "PlacesIdx": [1],
        "PlaceFormSub": [1],
        "County": ["ZZZ"],
        "Hundred": ["x"],
        "Vill": ["Xville"],
        "OSref": [None],
        "OScodes": [None],
    }
    with pytest.raises(RegionValidationError):
        _ingest_from_tables(conn, places, placeforms, apply=True)


def test_ingest_dry_run_matches_apply_with_duplicate_vill_county_pairs() -> None:
    """wyrd-el93 dry-run bug regression: when PlaceForm has multiple
    rows pointing at the same (Vill, county) (PlaceFormSub > 1
    modern-name alternates), the apply path dedupes to ONE toponym
    row, and the dry-run report must match — otherwise operators
    over-estimate the insert volume before committing.

    Fixture: 'Abbotskerswell' appears with PlaceFormSub=1 AND
    PlaceFormSub=2; both for the same Devon county code. apply
    inserts 1 toponym, attests 2 times; dry-run must report the
    same."""
    places = {
        "PlacesIdx": [26],
        "County": ["DEV"],
        "Phillimore": ["5,1"],
        "Hundred": ["Kerswell"],
        "Vill": ["Abbotskerswell"],
        "Area": [None],
        "XRefs": [None],
        "OSrefs": ["SX8568"],
        "OScodes": [None],
    }
    # Use distinct OSrefs so the apply path's
    # (toponym_id, form, year, source_doc) attestation dedupe doesn't
    # collapse the two rows — mirrors real Hull data where
    # PlaceFormSub variants are distinct manor entries at different
    # OS grid refs (e.g. Great/Little Abington).
    placeforms = {
        "PlacesIdx": [26, 26],
        "PlaceFormSub": [1, 2],
        "County": ["DEV", "DEV"],
        "Hundred": ["Kerswell", "Kerswell"],
        "Vill": ["Abbotskerswell", "Abbotskerswell"],
        "OSref": ["SX8568", "SX8569"],
        "OScodes": [None, None],
    }
    # Apply pass.
    conn_apply = _build_fixture_db()
    apply_counts = _ingest_from_tables(conn_apply, places, placeforms, apply=True)
    # Dry-run pass.
    conn_dry = _build_fixture_db()
    dry_counts = _ingest_from_tables(conn_dry, places, placeforms, apply=False)
    # The would-be-toponym count and attestation count must match
    # between dry-run and apply so dry-run is a faithful preview.
    assert dry_counts["toponym_inserted"] == apply_counts["toponym_inserted"] == 1
    assert dry_counts["attestation_inserted"] == apply_counts["attestation_inserted"] == 2


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


def test_compute_source_doc_assembly_and_fallbacks() -> None:
    """`_compute_source_doc` joins the present Phillimore / Hundred / OS parts
    with '; ' in that fixed order, and falls back to DOMESDAY_SOURCE_ID when
    none are present (so citation-less rows still dedup against the index).
    Pins the wyrd-8uvi-extracted helper across the present/absent matrix."""
    assert (
        _compute_source_doc("15,8", "Doddingtree", "SO7567")
        == "Phillimore 15,8; Hundred=Doddingtree; OS=SO7567"
    )
    assert _compute_source_doc("15,8", None, None) == "Phillimore 15,8"
    assert _compute_source_doc(None, "Doddingtree", None) == "Hundred=Doddingtree"
    assert _compute_source_doc("", None, "SO7567") == "OS=SO7567"
    # All absent (None Phillimore, empty Hundred/OS) → the source-id fallback.
    assert _compute_source_doc(None, None, None) == DOMESDAY_SOURCE_ID


def test_dry_run_distinct_negative_ids_for_same_name_different_region() -> None:
    """Two DISTINCT new toponyms sharing a modern_name + identical source_doc
    but a different region must get DISTINCT dry-run negative ids, so their
    attestations don't collapse in the attestation_index. With a shared id the
    second attestation would be miscounted as existing. Pins the
    dry_run_id_counter uniqueness on _DomesdayIngestState (wyrd-8uvi).

    Both rows have no Phillimore/Hundred/OS → identical source_doc
    (DOMESDAY_SOURCE_ID); only the negative id distinguishes the att_keys."""
    places = {"PlacesIdx": [100, 200], "Phillimore": [None, None]}
    placeforms = {
        "PlacesIdx": [100, 200],
        "Vill": ["Newton", "Newton"],
        "County": ["DEV", "ESS"],  # different regions → distinct toponyms
        "OSref": [None, None],
        "OScodes": [None, None],
        "Hundred": [None, None],
    }
    apply_counts = _ingest_from_tables(_build_fixture_db(), places, placeforms, apply=True)
    dry_counts = _ingest_from_tables(_build_fixture_db(), places, placeforms, apply=False)
    # Two distinct toponyms, two distinct attestations — dry-run matches apply.
    assert dry_counts["toponym_inserted"] == apply_counts["toponym_inserted"] == 2
    assert dry_counts["attestation_inserted"] == apply_counts["attestation_inserted"] == 2


def test_preexisting_toponym_hit_twice_counts_existing_once() -> None:
    """A pre-existing DB toponym hit by 2+ PlaceForm rows is counted in
    `toponym_existing` exactly ONCE (the in-run seen_this_run guard), while
    each row's distinct-source_doc attestation still inserts. Pins the
    multi-count guard on _DomesdayIngestState.seen_this_run (wyrd-8uvi)."""
    conn = _build_fixture_db()
    # Pre-seed the toponym 'Town' in Worcestershire (WOR → Worcestershire).
    conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
        ("Town", "England", COUNTY_CODE_TO_NAME["WOR"]),
    )
    conn.commit()
    places = {"PlacesIdx": [1, 2], "Phillimore": ["1,1", "1,2"]}
    placeforms = {
        "PlacesIdx": [1, 2],
        "Vill": ["Town", "Town"],
        "County": ["WOR", "WOR"],  # same (name, region) → same pre-existing toponym
        "OSref": ["SO1", "SO2"],  # distinct → distinct source_doc → both attest
        "OScodes": [None, None],
        "Hundred": [None, None],
    }
    counts = _ingest_from_tables(conn, places, placeforms, apply=True)
    assert counts["toponym_inserted"] == 0
    assert counts["toponym_existing"] == 1  # counted once, not twice
    assert counts["attestation_inserted"] == 2
