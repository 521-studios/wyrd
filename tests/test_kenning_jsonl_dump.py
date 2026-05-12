"""Tests for the SQLite → per-source JSONL dumper — wyrd-f295.

Fixture DB carries a minimal subset of the lexicon schema — just the
tables the dump touches. Each test exercises one row-type slice
through ``dump_source_to_rows``, asserting both the row shape and the
:func:`jsonl_log.replay` round-trip (dumped rows must replay back to
the same canonical state without errors).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from wyrd.generators.kenning.jsonl_dump import (
    dump_all_sources,
    dump_source_to_file,
    dump_source_to_rows,
    list_source_ids,
    etymon_ref,
    toponym_ref,
)
from wyrd.generators.kenning.jsonl_log import (
    ReplayState,
    read_jsonl,
    replay,
    replay_file,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _build_fixture_db() -> sqlite3.Connection:
    """In-memory DB matching the L2-touching slice of lexicon.sql.

    Schema is intentionally minimal — only the columns the dumper reads.
    Drops L3-derived columns (lemma_id, merged_into_id, etc.) since the
    dump pretends they don't exist."""
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
        CREATE TABLE etymon (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL,
            language TEXT NOT NULL,
            modifier_type TEXT,
            position_pref TEXT,
            notes TEXT,
            pronunciation_ipa TEXT,
            pronunciation_dialect TEXT,
            original_script TEXT,
            transliteration TEXT,
            -- L3 columns the dumper should ignore
            lemma_id INTEGER,
            merged_into_id INTEGER,
            cognate_id INTEGER,
            stratum TEXT,
            english_shaped TEXT,
            UNIQUE (canonical_form, language)
        );
        CREATE TABLE etymon_gloss (
            etymon_id INTEGER NOT NULL,
            gloss TEXT NOT NULL,
            PRIMARY KEY (etymon_id, gloss)
        );
        CREATE TABLE etymon_tag (
            etymon_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (etymon_id, tag)
        );
        CREATE TABLE etymon_citation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            etymon_id INTEGER NOT NULL,
            source_id TEXT NOT NULL,
            page TEXT,
            short_quote TEXT,
            context_snippet TEXT
        );
        CREATE TABLE etymon_descent (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER NOT NULL,
            child_id INTEGER NOT NULL,
            edge_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            confidence TEXT,
            notes TEXT
        );
        CREATE TABLE mining_run (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            mode TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            parsed_count INTEGER NOT NULL DEFAULT 0,
            accepted INTEGER NOT NULL DEFAULT 0,
            declined INTEGER NOT NULL DEFAULT 0,
            rejected INTEGER NOT NULL DEFAULT 0,
            by_failure TEXT,
            notes TEXT
        );
        CREATE TABLE toponym (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modern_name TEXT NOT NULL,
            country TEXT,
            region TEXT
        );
        CREATE TABLE toponym_etymology (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            toponym_id INTEGER NOT NULL,
            source_id TEXT NOT NULL,
            page TEXT,
            historical_form TEXT,
            confidence TEXT,
            notes TEXT,
            attested_year INTEGER
        );
        CREATE TABLE toponym_etymology_element (
            toponym_etymology_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            etymon_id INTEGER NOT NULL,
            inflection TEXT,
            surface_in_modern TEXT,
            PRIMARY KEY (toponym_etymology_id, ordinal)
        );
        """
    )
    return conn


def _add_source(conn: sqlite3.Connection, **kwargs) -> str:
    cols = ", ".join(kwargs.keys())
    placeholders = ", ".join("?" * len(kwargs))
    conn.execute(f"INSERT INTO source ({cols}) VALUES ({placeholders})", tuple(kwargs.values()))
    return kwargs["id"]


def _add_etymon(conn: sqlite3.Connection, language: str, canonical_form: str, **kwargs) -> int:
    cols = ["language", "canonical_form", *kwargs.keys()]
    vals = [language, canonical_form, *kwargs.values()]
    placeholders = ", ".join("?" * len(cols))
    cur = conn.execute(
        f"INSERT INTO etymon ({', '.join(cols)}) VALUES ({placeholders})", tuple(vals)
    )
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Source row dump
# ---------------------------------------------------------------------------


def test_dump_source_row_minimal():
    conn = _build_fixture_db()
    _add_source(
        conn, id="skeat_1901", title="Place-Names of Cambs", author="W. W. Skeat", year=1901
    )
    rows = dump_source_to_rows(conn, "skeat_1901")
    assert rows[0] == {
        "_type": "source",
        "ref": "skeat_1901",
        "title": "Place-Names of Cambs",
        "author": "W. W. Skeat",
        "year": 1901,
    }


def test_dump_source_row_missing_raises():
    conn = _build_fixture_db()
    import pytest

    with pytest.raises(ValueError, match="no source row"):
        dump_source_to_rows(conn, "nope")


def test_dump_source_strips_null_columns():
    """Region / language_focus / notes left null at insert should not
    appear in the dumped row — keeps the JSONL diff-clean."""
    conn = _build_fixture_db()
    _add_source(conn, id="x", title="X", year=1900)
    rows = dump_source_to_rows(conn, "x")
    assert rows[0] == {"_type": "source", "ref": "x", "title": "X", "year": 1900}


# ---------------------------------------------------------------------------
# Etymon dump
# ---------------------------------------------------------------------------


def test_dump_etymon_with_citation():
    """An etymon cited by the source gets one canonical-state etymon
    row + one citation list row. L3 columns are dropped."""
    conn = _build_fixture_db()
    _add_source(conn, id="skeat", title="X")
    eid = _add_etymon(
        conn,
        "old-english",
        "cot",
        modifier_type="noun",
        notes="dwelling-place",
        # L3 columns — should NOT appear in the dump
        lemma_id=None,
        stratum="common",
        english_shaped="cot",
    )
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, ?, ?)",
        (eid, "skeat", "15"),
    )
    conn.execute("INSERT INTO etymon_gloss VALUES (?, ?)", (eid, "cottage"))
    conn.execute("INSERT INTO etymon_gloss VALUES (?, ?)", (eid, "hut"))
    conn.execute("INSERT INTO etymon_tag VALUES (?, ?)", (eid, "architecture"))
    rows = dump_source_to_rows(conn, "skeat")
    etymon = next(r for r in rows if r["_type"] == "etymon")
    assert etymon == {
        "_type": "etymon",
        "ref": "old-english:cot",
        "canonical_form": "cot",
        "language": "old-english",
        "modifier_type": "noun",
        "notes": "dwelling-place",
        "glosses": ["cottage", "hut"],
        "tags": ["architecture"],
    }
    # L3 columns absent
    assert "stratum" not in etymon
    assert "english_shaped" not in etymon
    assert "lemma_id" not in etymon


def test_dump_etymon_only_cited_by_target_source():
    """An etymon cited by source A but NOT by source B should appear
    in A's dump but not B's. Each source's dump is independent."""
    conn = _build_fixture_db()
    _add_source(conn, id="a", title="A")
    _add_source(conn, id="b", title="B")
    a_only = _add_etymon(conn, "old-english", "cot")
    b_only = _add_etymon(conn, "old-english", "tun")
    conn.execute("INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, ?)", (a_only, "a"))
    conn.execute("INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, ?)", (b_only, "b"))
    a_rows = dump_source_to_rows(conn, "a")
    b_rows = dump_source_to_rows(conn, "b")
    a_etymons = {r["ref"] for r in a_rows if r["_type"] == "etymon"}
    b_etymons = {r["ref"] for r in b_rows if r["_type"] == "etymon"}
    assert a_etymons == {"old-english:cot"}
    assert b_etymons == {"old-english:tun"}


def test_dump_etymon_distinct_per_source_even_with_multiple_citations():
    """Two citations from the same source on the same etymon = one
    etymon row, two citation rows. DISTINCT in the etymon query."""
    conn = _build_fixture_db()
    _add_source(conn, id="skeat", title="X")
    eid = _add_etymon(conn, "old-english", "cot")
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, 'skeat', '15')", (eid,)
    )
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, 'skeat', '16')", (eid,)
    )
    rows = dump_source_to_rows(conn, "skeat")
    n_etymons = sum(1 for r in rows if r["_type"] == "etymon")
    n_citations = sum(1 for r in rows if r["_type"] == "citation")
    assert (n_etymons, n_citations) == (1, 2)


# ---------------------------------------------------------------------------
# Descent dump
# ---------------------------------------------------------------------------


def test_dump_descent_edges():
    conn = _build_fixture_db()
    _add_source(conn, id="wiki", title="Wiktionary")
    parent = _add_etymon(conn, "proto-germanic", "*tunaz")
    child = _add_etymon(conn, "old-english", "tun")
    conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id, confidence) "
        "VALUES (?, ?, 'inheritance', 'wiki', 'high')",
        (parent, child),
    )
    rows = dump_source_to_rows(conn, "wiki")
    descent = next(r for r in rows if r["_type"] == "etymon_descent")
    assert descent == {
        "_type": "etymon_descent",
        "parent_ref": "proto-germanic:*tunaz",
        "child_ref": "old-english:tun",
        "edge_type": "inheritance",
        "confidence": "high",
    }


# ---------------------------------------------------------------------------
# Mining-run dump
# ---------------------------------------------------------------------------


def test_dump_mining_run():
    conn = _build_fixture_db()
    _add_source(conn, id="skeat", title="X")
    conn.execute(
        """INSERT INTO mining_run
           (source_id, provider, model, mode, started_at, completed_at,
            parsed_count, accepted, declined, rejected)
           VALUES ('skeat', 'anthropic', 'claude-opus-4-7', 'mine',
                   '2026-05-01', '2026-05-01', 100, 95, 3, 2)"""
    )
    rows = dump_source_to_rows(conn, "skeat")
    run = next(r for r in rows if r["_type"] == "mining_run")
    assert run == {
        "_type": "mining_run",
        "provider": "anthropic",
        "model": "claude-opus-4-7",
        "mode": "mine",
        "started_at": "2026-05-01",
        "completed_at": "2026-05-01",
        "parsed_count": 100,
        "accepted": 95,
        "declined": 3,
        "rejected": 2,
    }


# ---------------------------------------------------------------------------
# Toponym + etymology element dump
# ---------------------------------------------------------------------------


def test_dump_toponym_and_etymology_elements():
    conn = _build_fixture_db()
    _add_source(conn, id="mawer", title="Northumberland and Durham")
    cot = _add_etymon(conn, "old-english", "cot")
    tun = _add_etymon(conn, "old-english", "tun")
    cur = conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
        ("Cotton", "England", "Norfolk"),
    )
    tid = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, historical_form, attested_year, confidence) "
        "VALUES (?, 'mawer', 'Cotuna', 1086, 'high')",
        (tid,),
    )
    eid = cur.lastrowid
    conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) VALUES (?, 1, ?)",
        (eid, cot),
    )
    conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id, inflection) "
        "VALUES (?, 2, ?, 'oblique')",
        (eid, tun),
    )
    rows = dump_source_to_rows(conn, "mawer")
    topo = next(r for r in rows if r["_type"] == "toponym")
    assert topo == {
        "_type": "toponym",
        "ref": "Cotton@Norfolk",
        "modern_name": "Cotton",
        "country": "England",
        "region": "Norfolk",
    }
    el = next(r for r in rows if r["_type"] == "etymology_element")
    assert el == {
        "_type": "etymology_element",
        "toponym_ref": "Cotton@Norfolk",
        "elements": [
            {"ordinal": 1, "etymon_ref": "old-english:cot"},
            {"ordinal": 2, "etymon_ref": "old-english:tun", "inflection": "oblique"},
        ],
        "historical_form": "Cotuna",
        "confidence": "high",
        "attested_year": 1086,
    }


def test_toponym_ref_handles_null_region():
    assert toponym_ref("Acton", None) == "Acton@-"
    assert toponym_ref("Acton", "Cheshire") == "Acton@Cheshire"


def test_etymon_ref_format():
    assert etymon_ref("old-english", "cot") == "old-english:cot"


# ---------------------------------------------------------------------------
# Round-trip: dump → replay produces a valid ReplayState with no errors
# ---------------------------------------------------------------------------


def test_dump_round_trips_through_replay():
    """A dump's output replays cleanly — no add-duplicates, no missing
    refs. The replayed state contains the expected entities."""
    conn = _build_fixture_db()
    _add_source(conn, id="skeat", title="X")
    eid = _add_etymon(conn, "old-english", "cot")
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, 'skeat', '15')",
        (eid,),
    )
    conn.execute("INSERT INTO etymon_gloss VALUES (?, 'cottage')", (eid,))
    rows = dump_source_to_rows(conn, "skeat")
    state = replay(rows)
    assert state.get("source", "skeat") == {"title": "X"}
    assert state.get("etymon", "old-english:cot") == {
        "canonical_form": "cot",
        "language": "old-english",
        "glosses": ["cottage"],
    }
    assert len(state.lists["citation"]) == 1


# ---------------------------------------------------------------------------
# File-level dump
# ---------------------------------------------------------------------------


def test_dump_source_to_file(tmp_path: Path):
    conn = _build_fixture_db()
    _add_source(conn, id="skeat", title="X")
    _add_etymon(conn, "old-english", "cot")
    path, count = dump_source_to_file(conn, "skeat", tmp_path)
    assert path == tmp_path / "skeat.jsonl"
    assert count == 1  # just the source row (no citations linked)
    # File replays cleanly.
    state = replay_file(path)
    assert state.get("source", "skeat") == {"title": "X"}


def test_dump_all_sources_one_file_per_source(tmp_path: Path):
    conn = _build_fixture_db()
    _add_source(conn, id="a", title="A")
    _add_source(conn, id="b", title="B")
    counts = dump_all_sources(conn, tmp_path, exclude=())
    assert set(counts) == {"a", "b"}
    assert (tmp_path / "a.jsonl").exists()
    assert (tmp_path / "b.jsonl").exists()


def test_dump_all_sources_skips_excluded(tmp_path: Path):
    """A source named in ``exclude`` is silently skipped — no file
    written, no key in returned counts."""
    conn = _build_fixture_db()
    _add_source(conn, id="wiktionary", title="W")
    _add_source(conn, id="skeat", title="S")
    counts = dump_all_sources(conn, tmp_path, exclude={"wiktionary"})
    assert set(counts) == {"skeat"}
    assert not (tmp_path / "wiktionary.jsonl").exists()


def test_dump_all_sources_default_excludes_bulk_wiktextract(tmp_path: Path):
    """Without an explicit ``exclude``, the wiktionary-* bulk sources
    are skipped — their dumps are too large for git and re-derivable
    from L1 raw inputs."""
    conn = _build_fixture_db()
    _add_source(conn, id="wiktionary", title="W")
    _add_source(conn, id="skeat", title="S")
    counts = dump_all_sources(conn, tmp_path)
    assert "wiktionary" not in counts
    assert "skeat" in counts


def test_list_source_ids_returns_sorted():
    conn = _build_fixture_db()
    _add_source(conn, id="zeta", title="Z")
    _add_source(conn, id="alpha", title="A")
    assert list_source_ids(conn) == ["alpha", "zeta"]


def test_list_source_ids_honors_exclude():
    conn = _build_fixture_db()
    _add_source(conn, id="a", title="A")
    _add_source(conn, id="b", title="B")
    _add_source(conn, id="c", title="C")
    assert list_source_ids(conn, exclude={"b"}) == ["a", "c"]
