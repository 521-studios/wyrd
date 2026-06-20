"""Tests for the SQLite → per-source JSONL dumper — wyrd-f295.

Fixture DB carries a minimal subset of the lexicon schema — just the
tables the dump touches. Each test exercises one row-type slice
through ``dump_source_to_rows``, asserting both the row shape and the
:func:`jsonl.log.replay` round-trip (dumped rows must replay back to
the same canonical state without errors).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from wyrd.generators.kenning.jsonl.dump import (
    _attestation_source_doc_filter,
    _source_for_attestation_doc,
    count_orphan_attestations,
    dump_all_sources,
    dump_fantasy_morphemes_to_file,
    dump_fantasy_morphemes_to_rows,
    dump_source_to_file,
    dump_source_to_rows,
    etymon_ref,
    list_source_ids,
    toponym_ref,
)
from wyrd.generators.kenning.jsonl.log import (
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
            -- L3 columns the dumper ignores, EXCEPT merged_into_id, which the
            -- fantasy dump reads to follow a referenced loser to its winner
            -- (wyrd-ruvk).
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
            -- wyrd-2n1: per-element confidence; matches schema/0013.
            confidence TEXT,
            PRIMARY KEY (toponym_etymology_id, ordinal)
        );
        CREATE TABLE toponym_attestation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            toponym_id INTEGER NOT NULL,
            form TEXT NOT NULL,
            date_year INTEGER,
            source_doc TEXT
        );
        CREATE TABLE fantasy_morpheme (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            input_name          TEXT NOT NULL COLLATE NOCASE,
            input_description   TEXT,
            usable              INTEGER NOT NULL CHECK (usable IN (0, 1)),
            etymon_id           INTEGER REFERENCES etymon(id) ON DELETE SET NULL,
            bar_reason          TEXT,
            resolution_method   TEXT NOT NULL,
            approach_version    TEXT NOT NULL,
            confidence          TEXT,
            citation            TEXT,
            reasoning           TEXT,
            unapproved_language TEXT,
            unapproved_form     TEXT,
            processed_at        TEXT NOT NULL,
            UNIQUE (input_name, approach_version)
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


def test_dump_cited_etymon_merged_loser_follows_to_winner(tmp_path: Path):
    """wyrd-q6ro: when a source cites an OCR-cluster LOSER (merged_into_id
    set), the per-source dump emits the WINNER etymon row AND a citation
    referencing the winner — never the tombstone. build_from_jsonl drops
    merged_into_id, so emitting the loser would resurrect it as a live
    unmerged etymon (D22) and orphan the citation witness (D21). Mirrors the
    fantasy guard (test_fantasy_morpheme_merged_loser_etymon_follows_to_winner)."""
    from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in

    pre = _build_fixture_db()
    _add_source(pre, id="skeat", title="X")
    winner = _add_etymon(pre, "old-english", "wic")
    loser = _add_etymon(pre, "old-english", "wych", merged_into_id=winner)
    pre.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, 'skeat', '15')",
        (loser,),
    )
    rows = dump_source_to_rows(pre, "skeat")
    # Etymon dump: only the winner, never the tombstone.
    assert {r["ref"] for r in rows if r["_type"] == "etymon"} == {"old-english:wic"}
    # The citation follows to the winner ref too (no dangling loser ref).
    citation = next(r for r in rows if r["_type"] == "citation")
    assert citation["etymon_ref"] == "old-english:wic"
    # Hardening: the loser ref must not appear in ANY emitted row (ref or
    # etymon_ref), so a partial-fix regression on either dump can't slip by.
    loser_ref = "old-english:wych"
    assert all(r.get("ref") != loser_ref and r.get("etymon_ref") != loser_ref for r in rows)
    dump_source_to_file(pre, "skeat", tmp_path)
    pre.close()

    rebuilt = _build_fixture_db()
    counts = build_from_jsonl(rebuilt, jsonl_paths_in(tmp_path))
    # No tombstone resurrected: the only etymon is the unmerged winner.
    assert (
        rebuilt.execute("SELECT count(*) FROM etymon WHERE merged_into_id IS NOT NULL").fetchone()[
            0
        ]
        == 0
    )
    forms = {r[0] for r in rebuilt.execute("SELECT canonical_form FROM etymon").fetchall()}
    assert forms == {"wic"}
    # Citation evidence preserved + attached to the winner (not orphaned).
    assert counts.get("citation_orphans", 0) == 0
    cited = rebuilt.execute(
        "SELECT e.canonical_form FROM etymon_citation c JOIN etymon e ON e.id = c.etymon_id "
        "WHERE c.source_id = 'skeat'"
    ).fetchall()
    assert [r[0] for r in cited] == ["wic"]
    rebuilt.close()


def test_dump_cited_etymon_winner_and_loser_both_cited_dedups_etymon_keeps_all_witnesses(
    tmp_path: Path,
):
    """wyrd-q6ro: a source citing BOTH the winner directly AND its merge
    loser (plus a second citation on the loser) emits exactly ONE winner
    etymon row (the DISTINCT + COALESCE collapse), and every citation
    follows to the winner ref so all three witnesses round-trip onto the
    winner — no duplicate etymon, no lost witness, no resurrected loser."""
    from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in

    pre = _build_fixture_db()
    _add_source(pre, id="skeat", title="X")
    winner = _add_etymon(pre, "old-english", "wic")
    loser = _add_etymon(pre, "old-english", "wych", merged_into_id=winner)
    # Winner cited directly; loser cited twice (distinct pages).
    pre.executemany(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, 'skeat', ?)",
        [(winner, "10"), (loser, "15"), (loser, "16")],
    )
    rows = dump_source_to_rows(pre, "skeat")
    # Exactly one winner etymon row (loser collapses into it via COALESCE).
    assert [r["ref"] for r in rows if r["_type"] == "etymon"] == ["old-english:wic"]
    # All three citations emit, every one referencing the winner.
    cite_refs = [r["etymon_ref"] for r in rows if r["_type"] == "citation"]
    assert cite_refs == ["old-english:wic", "old-english:wic", "old-english:wic"]
    dump_source_to_file(pre, "skeat", tmp_path)
    pre.close()

    rebuilt = _build_fixture_db()
    counts = build_from_jsonl(rebuilt, jsonl_paths_in(tmp_path))
    assert counts.get("citation_orphans", 0) == 0
    # One etymon (the winner), all three witnesses attached to it.
    assert [r[0] for r in rebuilt.execute("SELECT canonical_form FROM etymon").fetchall()] == [
        "wic"
    ]
    n_cites = rebuilt.execute(
        "SELECT count(*) FROM etymon_citation c JOIN etymon e ON e.id = c.etymon_id "
        "WHERE e.canonical_form = 'wic'"
    ).fetchone()[0]
    assert n_cites == 3
    rebuilt.close()


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


def test_dump_descent_edge_merged_endpoint_follows_to_winner(tmp_path: Path):
    """wyrd-q6ro: a descent edge whose endpoint is an OCR-cluster loser emits
    the WINNER ref (matching the winner etymon row _dump_cited_etymons emits),
    so the edge references an etymon the file carries and round-trips through
    build_from_jsonl without orphaning (no etymon_descent_orphans)."""
    from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in

    conn = _build_fixture_db()
    _add_source(conn, id="wiki", title="Wiktionary")
    parent = _add_etymon(conn, "proto-germanic", "*tunaz")
    winner = _add_etymon(conn, "old-english", "tun")
    loser = _add_etymon(conn, "old-english", "tunne", merged_into_id=winner)
    conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
        "VALUES (?, ?, 'inheritance', 'wiki')",
        (parent, loser),
    )
    rows = dump_source_to_rows(conn, "wiki")
    descent = next(r for r in rows if r["_type"] == "etymon_descent")
    assert descent["child_ref"] == "old-english:tun"
    assert descent["parent_ref"] == "proto-germanic:*tunaz"
    # The loser ref appears nowhere.
    assert all(
        r.get("child_ref") != "old-english:tunne" and r.get("parent_ref") != "old-english:tunne"
        for r in rows
    )
    dump_source_to_file(conn, "wiki", tmp_path)
    conn.close()

    # Round-trip: the edge resolves onto the winner, no orphan, no resurrected
    # tombstone — the D21/D22 bar the citation arm already meets.
    rebuilt = _build_fixture_db()
    counts = build_from_jsonl(rebuilt, jsonl_paths_in(tmp_path))
    assert counts.get("etymon_descent_orphans", 0) == 0
    assert (
        rebuilt.execute("SELECT count(*) FROM etymon WHERE merged_into_id IS NOT NULL").fetchone()[
            0
        ]
        == 0
    )
    edge = rebuilt.execute(
        "SELECT ce.canonical_form FROM etymon_descent d JOIN etymon ce ON ce.id = d.child_id"
    ).fetchone()
    assert edge["canonical_form"] == "tun"
    rebuilt.close()


def test_dump_descent_edge_collapsing_to_self_loop_is_skipped():
    """wyrd-q6ro: a descent edge between two OCR variants of the SAME morpheme
    (both merge to one winner) degenerates to a parent==child self-loop after
    follow-to-winner and is dropped — it carries no real descent signal and
    build's _insert_descent has no self-edge guard."""
    conn = _build_fixture_db()
    _add_source(conn, id="wiki", title="Wiktionary")
    winner = _add_etymon(conn, "old-english", "tun")
    loser_a = _add_etymon(conn, "old-english", "tunne", merged_into_id=winner)
    loser_b = _add_etymon(conn, "old-english", "tvn", merged_into_id=winner)
    conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
        "VALUES (?, ?, 'inheritance', 'wiki')",
        (loser_a, loser_b),
    )
    rows = dump_source_to_rows(conn, "wiki")
    assert not any(r["_type"] == "etymon_descent" for r in rows)


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


def test_dump_etymology_element_merged_etymon_follows_to_winner(tmp_path: Path):
    """wyrd-q6ro: a toponym etymology element referencing an OCR-cluster loser
    emits the WINNER etymon_ref, matching the winner etymon row, so the element
    round-trips through build_from_jsonl without orphaning."""
    from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in

    conn = _build_fixture_db()
    _add_source(conn, id="mawer", title="X")
    winner = _add_etymon(conn, "old-english", "tun")
    loser = _add_etymon(conn, "old-english", "tunne", merged_into_id=winner)
    cur = conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
        ("Acton", "England", "Cheshire"),
    )
    tid = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id) VALUES (?, 'mawer')",
        (tid,),
    )
    eid = cur.lastrowid
    conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) "
        "VALUES (?, 1, ?)",
        (eid, loser),
    )
    rows = dump_source_to_rows(conn, "mawer")
    el = next(r for r in rows if r["_type"] == "etymology_element")
    assert el["elements"] == [{"ordinal": 1, "etymon_ref": "old-english:tun"}]
    dump_source_to_file(conn, "mawer", tmp_path)
    conn.close()

    # Round-trip: the element resolves onto the winner, no orphan.
    rebuilt = _build_fixture_db()
    counts = build_from_jsonl(rebuilt, jsonl_paths_in(tmp_path))
    assert counts.get("etymology_element_orphans", 0) == 0
    linked = rebuilt.execute(
        "SELECT e.canonical_form FROM toponym_etymology_element el "
        "JOIN etymon e ON e.id = el.etymon_id"
    ).fetchall()
    assert [r["canonical_form"] for r in linked] == ["tun"]
    rebuilt.close()


def test_toponym_ref_handles_null_region():
    assert toponym_ref("Acton", None) == "Acton@-"
    assert toponym_ref("Acton", "Cheshire") == "Acton@Cheshire"


def test_dump_emits_per_element_confidence_when_present():
    """wyrd-2n1: per-element confidence flows through the dump's
    elements list alongside inflection / surface_in_modern. NULL
    elements drop the key (_drop_nulls convention). Mixed-confidence
    elements within one breakdown are the whole point of the per-
    element migration — a scholar can mark the prefix 'high' and the
    suffix 'low' on the same toponym."""
    conn = _build_fixture_db()
    _add_source(conn, id="mawer", title="English Place-Names")
    cot = _add_etymon(conn, "old-english", "cot")
    tun = _add_etymon(conn, "old-english", "tun")
    cur = conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
        ("Mixedton", "England", "Norfolk"),
    )
    tid = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
        "VALUES (?, 'mawer', 'medium')",
        (tid,),
    )
    eid = cur.lastrowid
    # Prefix marked high (scholars confident on cot-); suffix low
    # (scholars hedging on -tun); aggregate stays 'medium' on parent.
    conn.execute(
        "INSERT INTO toponym_etymology_element "
        "(toponym_etymology_id, ordinal, etymon_id, confidence) VALUES (?, 1, ?, 'high')",
        (eid, cot),
    )
    conn.execute(
        "INSERT INTO toponym_etymology_element "
        "(toponym_etymology_id, ordinal, etymon_id, confidence) VALUES (?, 2, ?, 'low')",
        (eid, tun),
    )
    rows = dump_source_to_rows(conn, "mawer")
    el = next(r for r in rows if r["_type"] == "etymology_element")
    assert el["elements"] == [
        {"ordinal": 1, "etymon_ref": "old-english:cot", "confidence": "high"},
        {"ordinal": 2, "etymon_ref": "old-english:tun", "confidence": "low"},
    ]
    # Parent aggregate flows through unchanged.
    assert el["confidence"] == "medium"


def test_dump_drops_null_per_element_confidence_key():
    """Elements with NULL confidence drop the key entirely (matches
    the _drop_nulls convention used for inflection / surface_in_modern).
    Without this, every pre-migration element would carry a redundant
    `confidence: null` in the JSONL."""
    conn = _build_fixture_db()
    _add_source(conn, id="mawer", title="English Place-Names")
    cot = _add_etymon(conn, "old-english", "cot")
    cur = conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
        ("Cot", "England", "Norfolk"),
    )
    tid = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id) VALUES (?, 'mawer')",
        (tid,),
    )
    eid = cur.lastrowid
    conn.execute(
        "INSERT INTO toponym_etymology_element "
        "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, 1, ?)",
        (eid, cot),
    )
    rows = dump_source_to_rows(conn, "mawer")
    el = next(r for r in rows if r["_type"] == "etymology_element")
    assert "confidence" not in el["elements"][0]


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


def test_attestation_round_trips_through_dump_and_build(tmp_path: Path):
    """End-to-end pin for wyrd-6gpy: a Domesday-shaped fixture (toponym
    with attestation but no etymology) survives the full dump-jsonl →
    rebuild-from-jsonl cycle. Pre-wyrd-6gpy the attestation rows AND
    the orphan toponyms were silently lost — this test guards against
    regression of either gap."""
    from wyrd.generators.kenning.jsonl.build import (
        build_from_jsonl,
        jsonl_paths_in,
    )
    from wyrd.generators.kenning.jsonl.log import write_jsonl

    # PRE: dump-source DB has the Domesday shape — a source row, two
    # toponyms attested via Phillimore source_doc, NO etymology rows.
    pre = _build_fixture_db()
    _add_source(pre, id="open_domesday_hull", title="Open Domesday Hull")
    lincoln_id = _add_toponym(pre, "Lincoln", country="England", region="Lincolnshire")
    york_id = _add_toponym(pre, "York", country="England", region=None)
    _add_attestation(pre, lincoln_id, "Lindum", date_year=1086, source_doc="Phillimore 1L1.")
    _add_attestation(pre, lincoln_id, "Lindum colonia", date_year=900, source_doc="Phillimore 1L2.")
    _add_attestation(pre, york_id, "Eboracum", date_year=1086, source_doc="Phillimore 2L1.")

    # Dump
    dump_path = tmp_path / "open_domesday_hull.jsonl"
    rows = dump_source_to_rows(pre, "open_domesday_hull")
    write_jsonl(dump_path, rows)
    pre.close()

    # Build into a fresh DB (same schema)
    rebuilt = _build_fixture_db()
    counts = build_from_jsonl(rebuilt, jsonl_paths_in(tmp_path))
    assert counts.get("attestation_orphans", 0) == 0, (
        "attestation rows orphaned — toponym rows must round-trip first"
    )
    assert counts["toponym"] == 2
    assert counts["attestation"] == 3

    # Toponyms survive
    topo_rows = list(
        rebuilt.execute("SELECT modern_name, region FROM toponym ORDER BY modern_name")
    )
    assert [(r["modern_name"], r["region"]) for r in topo_rows] == [
        ("Lincoln", "Lincolnshire"),
        ("York", None),
    ]

    # Attestations survive with full metadata
    att_rows = list(
        rebuilt.execute(
            "SELECT t.modern_name, ta.form, ta.date_year, ta.source_doc "
            "FROM toponym_attestation ta JOIN toponym t ON t.id = ta.toponym_id "
            "ORDER BY t.modern_name, ta.date_year"
        )
    )
    assert (att_rows[0]["modern_name"], att_rows[0]["form"], att_rows[0]["date_year"]) == (
        "Lincoln",
        "Lindum colonia",
        900,
    )
    assert att_rows[0]["source_doc"] == "Phillimore 1L2."
    assert (
        att_rows[2]["modern_name"],
        att_rows[2]["form"],
        att_rows[2]["date_year"],
        att_rows[2]["source_doc"],
    ) == ("York", "Eboracum", 1086, "Phillimore 2L1.")


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


# ---------------------------------------------------------------------------
# Attestation dump (wyrd-6gpy)
# ---------------------------------------------------------------------------


def _add_toponym(
    conn: sqlite3.Connection,
    modern_name: str,
    country: str | None = None,
    region: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
        (modern_name, country, region),
    )
    return cur.lastrowid


def _add_attestation(
    conn: sqlite3.Connection,
    toponym_id: int,
    form: str,
    date_year: int | None = None,
    source_doc: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (?, ?, ?, ?)",
        (toponym_id, form, date_year, source_doc),
    )


# --- _source_for_attestation_doc resolver --------------------------------


def test_source_for_attestation_doc_returns_none_for_null():
    assert _source_for_attestation_doc(None, {"open_domesday_hull"}) is None


def test_source_for_attestation_doc_direct_source_id_match():
    """Most scholar ingesters write source_doc = '<source.id>' directly."""
    known = {"mawer_1920_northumberland_durham", "open_domesday_hull"}
    assert (
        _source_for_attestation_doc("mawer_1920_northumberland_durham", known)
        == "mawer_1920_northumberland_durham"
    )


def test_source_for_attestation_doc_phillimore_prefix_routes_to_open_domesday():
    """Open Domesday Hull ingester stamps 'Phillimore <citation>' into
    source_doc; the dumper must route it back to open_domesday_hull."""
    known = {"open_domesday_hull"}
    assert (
        _source_for_attestation_doc("Phillimore 1L1. See also CHS Y2; Hundred=Amounderness", known)
        == "open_domesday_hull"
    )
    assert (
        _source_for_attestation_doc("Phillimore 3,105. 6,298. 13,6.", known) == "open_domesday_hull"
    )


def test_source_for_attestation_doc_unknown_returns_none():
    """Free-text source_doc that doesn't match a known source id AND
    doesn't match a registered prefix → None → silently skipped by the
    per-source dump (operator surfaces orphans separately if needed)."""
    known = {"open_domesday_hull"}
    assert _source_for_attestation_doc("Hundred=Toseland; OS=TL1860", known) is None


# --- _dump_attestations_for_source --------------------------------------


def test_dump_emits_attestations_for_direct_source_id_match():
    """Attestation with source_doc == 'mawer_1920...' lands in that
    source's dump under _type=attestation."""
    conn = _build_fixture_db()
    _add_source(conn, id="mawer", title="Northumberland and Durham")
    tid = _add_toponym(conn, "Acton", country="England", region="Northumberland")
    # Etymology so toponym_etymology JOIN includes the row (separate
    # from the attestation-only path tested below).
    conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
        "VALUES (?, 'mawer', 'high')",
        (tid,),
    )
    _add_attestation(conn, tid, "Actone", date_year=1245, source_doc="mawer")

    rows = dump_source_to_rows(conn, "mawer")
    attestations = [r for r in rows if r["_type"] == "attestation"]
    assert len(attestations) == 1
    assert attestations[0] == {
        "_type": "attestation",
        "toponym_ref": "Acton@Northumberland",
        "form": "Actone",
        "date_year": 1245,
        "source_doc": "mawer",
    }


def test_dump_emits_attestations_for_phillimore_prefix():
    """Open Domesday's Phillimore-prefixed attestations route to
    open_domesday_hull's JSONL even though source_doc is free-text."""
    conn = _build_fixture_db()
    _add_source(conn, id="open_domesday_hull", title="Open Domesday")
    tid = _add_toponym(conn, "Lincoln", country="England", region="Lincolnshire")
    _add_attestation(conn, tid, "Lindum", date_year=1086, source_doc="Phillimore 1L1.")

    rows = dump_source_to_rows(conn, "open_domesday_hull")
    attestations = [r for r in rows if r["_type"] == "attestation"]
    assert len(attestations) == 1
    assert attestations[0]["source_doc"] == "Phillimore 1L1."
    assert attestations[0]["toponym_ref"] == "Lincoln@Lincolnshire"


def test_dump_orphans_attestation_with_unknown_source_doc():
    """An attestation with no known source_id match and no recognized
    prefix doesn't show up in ANY source's dump — silently filtered."""
    conn = _build_fixture_db()
    _add_source(conn, id="open_domesday_hull", title="ODH")
    _add_source(conn, id="mawer", title="Mawer")
    tid = _add_toponym(conn, "Somewhere", region="Cheshire")
    _add_attestation(conn, tid, "Sumare", source_doc="Hundred=Toseland; OS=TL1860")

    for sid in ("open_domesday_hull", "mawer"):
        rows = dump_source_to_rows(conn, sid)
        assert all(r["_type"] != "attestation" for r in rows), (
            f"orphan attestation leaked into {sid} dump"
        )


def test_dump_emits_attestation_only_toponym_row():
    """A toponym referenced ONLY by attestations (no toponym_etymology
    row) must STILL appear in the source's dump — otherwise the
    rebuild's _insert_attestation_rows would orphan-skip the
    attestation since no toponym_ref resolves."""
    conn = _build_fixture_db()
    _add_source(conn, id="open_domesday_hull", title="ODH")
    tid = _add_toponym(conn, "Wessex", country="England", region=None)
    _add_attestation(conn, tid, "Wessax", date_year=1086, source_doc="Phillimore 1L1.")

    rows = dump_source_to_rows(conn, "open_domesday_hull")
    toponyms = [r for r in rows if r["_type"] == "toponym"]
    assert len(toponyms) == 1
    assert toponyms[0]["modern_name"] == "Wessex"
    assert toponyms[0]["ref"] == "Wessex@-"
    # And the attestation that referenced it appears below.
    attestations = [r for r in rows if r["_type"] == "attestation"]
    assert len(attestations) == 1


def test_dump_does_not_double_emit_toponym_with_etymology_and_attestation():
    """A toponym with BOTH an etymology AND an attestation that route
    to the same source must appear exactly once in the dump. The
    etymology path emits it via _dump_toponyms_and_etymologies; the
    attestation-only path excludes it via its ``NOT IN (SELECT
    toponym_id FROM toponym_etymology WHERE source_id = ?)`` clause
    so the same toponym doesn't get re-emitted.

    Pinned at the dump layer so a future refactor that drops the
    NOT IN guard can't silently start double-emitting (the build's
    _merge_toponym handles cross-row dedup so a double would be
    invisible at the JSONL replay level — only the row-count
    assertion below catches it)."""
    conn = _build_fixture_db()
    _add_source(conn, id="open_domesday_hull", title="ODH")
    tid = _add_toponym(conn, "York", country="England", region=None)
    # Etymology side → emits one toponym row via _dump_toponyms_and_etymologies
    conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
        "VALUES (?, 'open_domesday_hull', 'high')",
        (tid,),
    )
    # Attestation side: routes to same source via Phillimore prefix
    _add_attestation(conn, tid, "Eboracum", source_doc="Phillimore 2L1.")

    rows = dump_source_to_rows(conn, "open_domesday_hull")
    yorks = [r for r in rows if r["_type"] == "toponym" and r["modern_name"] == "York"]
    # The etymology path emits it; the attestation-only-toponyms path
    # skips it because the etymology JOIN catches it first. So exactly
    # one toponym row.
    assert len(yorks) == 1


# --- _attestation_source_doc_filter — direct unit tests --------------


def test_attestation_source_doc_filter_no_prefixes_uses_direct_match_only():
    """A source with no registered prefix in _ATTESTATION_DOC_PREFIX_TO_SOURCE
    gets a single-clause filter (the direct source_id match)."""
    clause, params = _attestation_source_doc_filter("mawer_1920_northumberland_durham")
    assert clause == "(ta.source_doc = ?)"
    assert params == ["mawer_1920_northumberland_durham"]


def test_attestation_source_doc_filter_with_prefix_uses_glob_for_case_sensitivity():
    """A source with a registered prefix gets a ``GLOB`` clause (not
    ``LIKE``) so the SQL filter is case-sensitive. SQLite's LIKE is
    ASCII-case-INsensitive by default, and COLLATE BINARY doesn't
    override that — LIKE's case rule is separate. GLOB is
    case-sensitive by default, matching the Python resolver's
    ``str.startswith()`` semantics.

    Also pins the ``NOT IN (SELECT id FROM source)`` guard which
    enforces the Python oracle's exact-match-wins priority in SQL —
    without it, a future invariant-violating registration would
    silently double-route rows whose source_doc both equals a
    source.id and matches a prefix."""
    clause, params = _attestation_source_doc_filter("open_domesday_hull")
    assert "ta.source_doc = ?" in clause
    assert "ta.source_doc GLOB ?" in clause
    assert "LIKE" not in clause, "LIKE has case-insensitive default — use GLOB instead"
    assert "NOT IN (SELECT id FROM source)" in clause, (
        "prefix clause must enforce exact-match-wins priority in SQL"
    )
    assert params[0] == "open_domesday_hull"
    assert "Phillimore*" in params


def test_attestation_source_doc_filter_resolver_parity():
    """The SQL filter and the Python resolver must agree on which
    source_doc values bucket to a given source_id. If a future change
    drifts them — e.g. the SQL gains case-insensitivity without the
    resolver, or the resolver gains a precedence rule the SQL skips —
    the test oracle (the Python resolver) and production behavior
    (the SQL filter) diverge silently. Pin them with a small fixture
    DB so the parity is exercised end-to-end."""
    conn = _build_fixture_db()
    known = {"open_domesday_hull", "mawer_1920_northumberland_durham"}
    samples = [
        "open_domesday_hull",  # direct source_id match
        "mawer_1920_northumberland_durham",  # direct match — different source
        "Phillimore 1L1.",  # prefix match → open_domesday_hull
        "Phillimore 3,105.",  # prefix match — same source
        "phillimore 1L1.",  # lowercase — Python orphans (case-sensitive)
        "PhillimoreCompany Ltd.",  # technically starts with prefix → routes
        "Hundred=Toseland",  # orphan
        None,  # NULL orphan
    ]

    # Seed: one toponym, all sample attestations attached to it.
    cur = conn.execute(
        "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
        ("Anywhere", None),
    )
    tid = cur.lastrowid
    for doc in samples:
        conn.execute(
            "INSERT INTO toponym_attestation (toponym_id, form, source_doc) VALUES (?, ?, ?)",
            (tid, "Forma", doc),
        )

    for source_id in ("open_domesday_hull", "mawer_1920_northumberland_durham"):
        # SQL filter — what production sees
        clause, params = _attestation_source_doc_filter(source_id)
        sql_routed = {
            r["source_doc"]
            for r in conn.execute(
                f"SELECT source_doc FROM toponym_attestation ta WHERE {clause}",  # noqa: S608
                params,
            )
        }
        # Python resolver — what the test oracle says
        py_routed = {doc for doc in samples if _source_for_attestation_doc(doc, known) == source_id}
        assert sql_routed == py_routed, (
            f"resolver/SQL drift for {source_id!r}: "
            f"SQL routed {sql_routed!r}, Python routed {py_routed!r}"
        )


def test_attestation_source_doc_filter_is_case_sensitive():
    """GLOB (case-sensitive by default) makes the SQL filter accept
    only ``Phillimore``, not ``phillimore`` — matching the Python
    resolver's str.startswith() semantics. Pins the wyrd-6gpy round-1
    fix — LIKE (the previous attempt) is ASCII-case-INsensitive and
    would silently accept what the Python oracle orphans."""
    conn = _build_fixture_db()
    _add_source(conn, id="open_domesday_hull", title="ODH")
    tid = _add_toponym(conn, "Anywhere", region=None)
    _add_attestation(conn, tid, "Caps", source_doc="Phillimore 1L1.")  # routes
    _add_attestation(conn, tid, "Lower", source_doc="phillimore 1L1.")  # orphan

    rows = dump_source_to_rows(conn, "open_domesday_hull")
    forms = [r["form"] for r in rows if r["_type"] == "attestation"]
    assert forms == ["Caps"], (
        f"case-sensitive filter should accept only the capitalized prefix; got {forms!r}"
    )


def test_attestation_source_doc_filter_enforces_exact_match_wins_priority(monkeypatch):
    """If a future operator violates the
    _ATTESTATION_DOC_PREFIX_TO_SOURCE invariant by registering a
    prefix that equals an actual source.id, the SQL filter must
    still route per the Python resolver's exact-match-wins priority.

    Without the ``NOT IN (SELECT id FROM source)`` guard on the
    prefix clause, the row ``source_doc='Mawer'`` would be double-
    routed: once via the ``Mawer`` source's exact-match clause and
    once via the ``mawer_volumes`` source's prefix clause. The build
    side would then see two emit paths.

    Pins Gemini round-2 finding."""
    from wyrd.generators.kenning.jsonl import dump as jsonl_dump

    # Inject the invariant-violating registration: source.id 'Mawer'
    # exists AND it's also a registered prefix routing to a different
    # source ('mawer_volumes').
    monkeypatch.setattr(
        jsonl_dump,
        "_ATTESTATION_DOC_PREFIX_TO_SOURCE",
        (("Phillimore", "open_domesday_hull"), ("Mawer", "mawer_volumes")),
    )

    conn = _build_fixture_db()
    _add_source(conn, id="Mawer", title="Mawer's volumes")
    _add_source(conn, id="mawer_volumes", title="Mawer volumes index")
    _add_source(conn, id="open_domesday_hull", title="ODH")
    tid = _add_toponym(conn, "Anywhere", region=None)
    # The collision row: exact match for 'Mawer', also matches 'Mawer*' prefix.
    _add_attestation(conn, tid, "Form-A", source_doc="Mawer")
    # A clean prefix match — should still route to mawer_volumes.
    _add_attestation(conn, tid, "Form-B", source_doc="Mawer page 12")

    # mawer_volumes must NOT claim the 'Mawer' row (exact match wins for Mawer).
    mawer_volumes_rows = dump_source_to_rows(conn, "mawer_volumes")
    mawer_volumes_forms = [r["form"] for r in mawer_volumes_rows if r["_type"] == "attestation"]
    assert mawer_volumes_forms == ["Form-B"], (
        f"prefix clause must not hijack source_doc='Mawer' (its source.id); got {mawer_volumes_forms!r}"
    )

    # 'Mawer' source should claim its own exact-match row only.
    mawer_rows = dump_source_to_rows(conn, "Mawer")
    mawer_forms = [r["form"] for r in mawer_rows if r["_type"] == "attestation"]
    assert mawer_forms == ["Form-A"], (
        f"exact-match clause must claim source_doc='Mawer'; got {mawer_forms!r}"
    )


# --- E2E round-trip for direct source_id match (was untested) ---------


def test_attestation_round_trips_through_dump_and_build_direct_source_id(tmp_path: Path):
    """Sibling of test_attestation_round_trips_through_dump_and_build —
    that one exercises the Phillimore prefix branch; this one exercises
    the direct-source_id-match branch (most scholar ingesters write
    source_doc = '<source_id>' directly). Without this test the
    direct-match SQL clause could be broken silently while the
    prefix-branch round-trip continues to pass."""
    from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in
    from wyrd.generators.kenning.jsonl.log import write_jsonl

    pre = _build_fixture_db()
    _add_source(pre, id="mawer", title="Mawer — Northumberland and Durham")
    acton = _add_toponym(pre, "Acton", country="England", region="Northumberland")
    # Etymology + attestation, both attributed to mawer via direct source_id
    pre.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
        "VALUES (?, 'mawer', 'high')",
        (acton,),
    )
    _add_attestation(pre, acton, "Actone", date_year=1245, source_doc="mawer")

    dump_path = tmp_path / "mawer.jsonl"
    write_jsonl(dump_path, dump_source_to_rows(pre, "mawer"))
    pre.close()

    rebuilt = _build_fixture_db()
    counts = build_from_jsonl(rebuilt, jsonl_paths_in(tmp_path))
    assert counts.get("attestation_orphans", 0) == 0
    assert counts["attestation"] == 1

    row = rebuilt.execute(
        "SELECT t.modern_name, ta.form, ta.date_year, ta.source_doc "
        "FROM toponym_attestation ta JOIN toponym t ON t.id = ta.toponym_id"
    ).fetchone()
    assert (row["modern_name"], row["form"], row["date_year"], row["source_doc"]) == (
        "Acton",
        "Actone",
        1245,
        "mawer",
    )


# ---------------------------------------------------------------------------
# fantasy_morpheme dump (wyrd-2thc)
# ---------------------------------------------------------------------------


def _add_fantasy_morpheme(conn: sqlite3.Connection, **kwargs) -> int:
    """Insert one fantasy_morpheme row with required NOT NULL defaults
    pre-populated; caller overrides via kwargs."""
    defaults: dict[str, object] = {
        "input_name": "Default",
        "input_description": None,
        "usable": 1,
        "etymon_id": None,
        "bar_reason": None,
        "resolution_method": "descent_lookup",
        "approach_version": "fantasy-v1",
        "confidence": "high",
        "citation": None,
        "reasoning": None,
        "unapproved_language": None,
        "unapproved_form": None,
        "processed_at": "2026-05-15T00:00:00+00:00",
    }
    defaults.update(kwargs)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join("?" * len(defaults))
    cur = conn.execute(
        f"INSERT INTO fantasy_morpheme ({cols}) VALUES ({placeholders})",
        tuple(defaults.values()),
    )
    return cur.lastrowid


def test_fantasy_morpheme_uncited_etymon_round_trips_through_dump_and_build(tmp_path: Path):
    """wyrd-ruvk: a fantasy_morpheme links to an etymon that mine-fantasy-name
    created with NO citation, so no per-source dump carries it. The fantasy
    dump must emit that etymon's canonical-state row itself — otherwise the
    rebuild can't resolve the etymon_ref and drops the morpheme as an orphan
    (the gap that left only 8/1,072 fantasy morphemes resolving on rebuild)."""
    from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in

    pre = _build_fixture_db()
    # An UNCITED etymon (no etymon_citation row) — the gap's signature.
    eid = _add_etymon(pre, "middle-english", "-ard")
    pre.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, 'agent suffix')", (eid,))
    _add_fantasy_morpheme(pre, input_name="Bækard", etymon_id=eid, reasoning="decomposed -ard")
    dump_fantasy_morphemes_to_file(pre, tmp_path)
    pre.close()

    rebuilt = _build_fixture_db()
    counts = build_from_jsonl(rebuilt, jsonl_paths_in(tmp_path))
    # No orphan: the etymon was recreated from the fantasy file + the FK resolved.
    assert counts.get("fantasy_morpheme_orphans", 0) == 0
    assert counts["fantasy_morpheme"] == 1
    row = rebuilt.execute(
        "SELECT e.language, e.canonical_form, "
        "       (SELECT gloss FROM etymon_gloss WHERE etymon_id = e.id) AS gloss "
        "FROM fantasy_morpheme fm JOIN etymon e ON e.id = fm.etymon_id "
        "WHERE fm.input_name = 'Bækard'"
    ).fetchone()
    # Language, form AND the gloss carried on the emitted etymon row survive.
    assert (row["language"], row["canonical_form"], row["gloss"]) == (
        "middle-english",
        "-ard",
        "agent suffix",
    )
    rebuilt.close()


def test_fantasy_morpheme_merged_loser_etymon_follows_to_winner(tmp_path: Path):
    """wyrd-ruvk: when a fantasy_morpheme references an OCR-cluster LOSER
    (merged_into_id set — e.g. diacritic variant 'aeon' merged into 'æon'), the
    dump emits the WINNER's ref + the WINNER etymon row, never the tombstone.
    The morpheme round-trips linked to the (unmerged) winner — full parity
    without resurrecting a merged etymon (D22)."""
    from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in

    pre = _build_fixture_db()
    winner = _add_etymon(pre, "latin", "æon")
    loser = _add_etymon(pre, "latin", "aeon", merged_into_id=winner)
    _add_fantasy_morpheme(pre, input_name="Aeonar", etymon_id=loser)
    rows = dump_fantasy_morphemes_to_rows(pre)
    # The dump emits ONLY the winner etymon (no tombstone), and the morpheme
    # carries the winner ref.
    assert {r["ref"] for r in rows if r["_type"] == "etymon"} == {"latin:æon"}
    morpheme = next(r for r in rows if r["_type"] == "fantasy_morpheme")
    assert morpheme["etymon_ref"] == "latin:æon"
    dump_fantasy_morphemes_to_file(pre, tmp_path)
    pre.close()

    rebuilt = _build_fixture_db()
    counts = build_from_jsonl(rebuilt, jsonl_paths_in(tmp_path))
    assert counts.get("fantasy_morpheme_orphans", 0) == 0
    # No tombstone resurrected: the only etymon is the unmerged winner.
    assert (
        rebuilt.execute("SELECT count(*) FROM etymon WHERE merged_into_id IS NOT NULL").fetchone()[
            0
        ]
        == 0
    )
    row = rebuilt.execute(
        "SELECT e.canonical_form FROM fantasy_morpheme fm JOIN etymon e ON e.id = fm.etymon_id "
        "WHERE fm.input_name = 'Aeonar'"
    ).fetchone()
    assert row["canonical_form"] == "æon"
    rebuilt.close()


def test_fantasy_etymon_shared_by_two_morphemes_emits_one_row(tmp_path: Path):
    """wyrd-ruvk: two fantasy_morphemes sharing one etymon emit a single
    DISTINCT etymon row (guards the DISTINCT in _dump_fantasy_etymons)."""
    pre = _build_fixture_db()
    eid = _add_etymon(pre, "old-english", "draca")
    _add_fantasy_morpheme(pre, input_name="Dracos", etymon_id=eid)
    _add_fantasy_morpheme(pre, input_name="Dracar", etymon_id=eid)
    rows = dump_fantasy_morphemes_to_rows(pre)
    pre.close()
    etymon_rows = [r for r in rows if r["_type"] == "etymon"]
    assert [r["ref"] for r in etymon_rows] == ["old-english:draca"]
    assert len([r for r in rows if r["_type"] == "fantasy_morpheme"]) == 2


def test_fantasy_etymon_also_cited_by_real_source_merges_to_one(tmp_path: Path):
    """wyrd-ruvk: an etymon that is BOTH fantasy-referenced AND cited by a real
    per-source dump is emitted by both files; the build's cross-file merge
    (by ref) must unify them into a single etymon on rebuild."""
    from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in

    pre = _build_fixture_db()
    _add_source(pre, id="skeat", title="Skeat")
    eid = _add_etymon(pre, "old-english", "cot")
    pre.execute("INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, 'skeat')", (eid,))
    _add_fantasy_morpheme(pre, input_name="Cotling", etymon_id=eid)
    dump_source_to_file(pre, "skeat", tmp_path)  # emits old-english:cot (cited)
    dump_fantasy_morphemes_to_file(pre, tmp_path)  # also emits old-english:cot (fantasy)
    pre.close()

    rebuilt = _build_fixture_db()
    build_from_jsonl(rebuilt, jsonl_paths_in(tmp_path))
    # One etymon, not two — the cross-file merge unified the duplicate refs.
    assert (
        rebuilt.execute(
            "SELECT count(*) FROM etymon WHERE canonical_form='cot' AND language='old-english'"
        ).fetchone()[0]
        == 1
    )
    rebuilt.close()


def test_dump_fantasy_morphemes_returns_empty_when_no_rows():
    """No rows → no source declaration emitted (file shouldn't be
    written). Keeps the source-row contract from creating an
    empty-but-meaningful artifact on DBs that never ran fantasy
    mining."""
    conn = _build_fixture_db()
    assert dump_fantasy_morphemes_to_rows(conn) == []


def test_dump_fantasy_morphemes_emits_source_declaration_and_rows():
    """The synthetic ``fantasy-mining`` source row is emitted first,
    followed by one ``fantasy_morpheme`` list row per DB row, ordered
    by input_name."""
    conn = _build_fixture_db()
    angel = _add_etymon(conn, "old-english", "engel")
    _add_fantasy_morpheme(
        conn,
        input_name="Angel",
        input_description="celestial host",
        etymon_id=angel,
        citation="wiktionary-descent-chain",
        reasoning="matched OE engel via direct",
    )
    _add_fantasy_morpheme(
        conn,
        input_name="Military",
        usable=0,
        bar_reason="attested_but_not_in_corpus",
        resolution_method="llm_full_research",
        unapproved_language="latin",
        unapproved_form="mīlitārius",
    )

    rows = dump_fantasy_morphemes_to_rows(conn)
    assert rows[0]["_type"] == "source"
    assert rows[0]["ref"] == "fantasy-mining"
    # wyrd-ruvk: referenced (uncited) etymons are emitted as canonical-state
    # rows so the morphemes' etymon_refs resolve on rebuild.
    etymon_rows = [r for r in rows if r["_type"] == "etymon"]
    assert {r["ref"] for r in etymon_rows} == {"old-english:engel"}
    morphemes = [r for r in rows if r["_type"] == "fantasy_morpheme"]
    assert len(morphemes) == 2
    # Ordered by input_name (case-insensitive collation collapses, but
    # ORDER BY is text — "Angel" < "Military" still).
    assert morphemes[0]["input_name"] == "Angel"
    assert morphemes[0]["etymon_ref"] == "old-english:engel"
    assert morphemes[0]["usable"] == 1
    # Null fields are omitted, not emitted as null.
    assert "bar_reason" not in morphemes[0]
    assert "unapproved_language" not in morphemes[0]
    # Barred row carries no etymon_ref (etymon_id was NULL).
    assert morphemes[1]["input_name"] == "Military"
    assert "etymon_ref" not in morphemes[1]
    assert morphemes[1]["usable"] == 0
    assert morphemes[1]["bar_reason"] == "attested_but_not_in_corpus"
    assert morphemes[1]["unapproved_language"] == "latin"


def test_dump_fantasy_morphemes_handles_missing_table():
    """A DB without the fantasy_morpheme table (older DBs predating
    wyrd-ami) yields an empty list rather than raising."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Bare DB, no fantasy_morpheme table.
    assert dump_fantasy_morphemes_to_rows(conn) == []


def test_dump_fantasy_morphemes_to_file_writes_no_file_when_empty(tmp_path: Path):
    """Don't create a 0-byte file when there's nothing to dump."""
    conn = _build_fixture_db()
    path, count = dump_fantasy_morphemes_to_file(conn, tmp_path)
    assert count == 0
    assert not path.exists()


def test_dump_fantasy_morphemes_to_file_writes_file_when_populated(tmp_path: Path):
    conn = _build_fixture_db()
    angel = _add_etymon(conn, "old-english", "engel")
    _add_fantasy_morpheme(conn, input_name="Angel", etymon_id=angel)

    path, count = dump_fantasy_morphemes_to_file(conn, tmp_path)
    assert path.name == "_fantasy_morphemes.jsonl"
    assert path.exists()
    assert count == 3  # source + referenced etymon row (wyrd-ruvk) + fantasy_morpheme row


def test_dump_fantasy_morphemes_orders_by_input_name_then_approach(tmp_path: Path):
    """Diff-stable ordering — two rows with the same input_name but
    different approach_versions appear in approach_version order."""
    conn = _build_fixture_db()
    eid = _add_etymon(conn, "old-english", "engel")
    _add_fantasy_morpheme(conn, input_name="Angel", approach_version="fantasy-v2", etymon_id=eid)
    _add_fantasy_morpheme(conn, input_name="Angel", approach_version="fantasy-v1", etymon_id=eid)

    rows = dump_fantasy_morphemes_to_rows(conn)
    morphemes = [r for r in rows if r["_type"] == "fantasy_morpheme"]
    assert [m["approach_version"] for m in morphemes] == ["fantasy-v1", "fantasy-v2"]


def test_default_bulk_excluded_sources_excludes_fantasy_mining_from_dump_all():
    """A rebuilt DB carries a ``fantasy-mining`` row in the ``source``
    table (inserted by build_from_jsonl from the synthetic source
    declaration in ``_fantasy_morphemes.jsonl``). Without the
    DEFAULT_BULK_EXCLUDED_SOURCES entry, dump_all_sources would emit
    a competing per-source ``fantasy-mining.jsonl`` alongside the
    real ``_fantasy_morphemes.jsonl`` — pins the wyrd-2thc exclusion
    against accidental removal."""
    from wyrd.generators.kenning.jsonl.dump import DEFAULT_BULK_EXCLUDED_SOURCES

    assert "fantasy-mining" in DEFAULT_BULK_EXCLUDED_SOURCES

    conn = _build_fixture_db()
    _add_source(conn, id="fantasy-mining", title="Fantasy morpheme mining (rebuilt DB)")
    _add_source(conn, id="skeat", title="Skeat")

    sids = list_source_ids(conn, exclude=DEFAULT_BULK_EXCLUDED_SOURCES)
    assert "fantasy-mining" not in sids
    assert "skeat" in sids


def test_wiktionary_empirical_is_dumpable_and_round_trips(tmp_path: Path):
    """wyrd-x33t: wiktionary-empirical was excluded from dump as an L3-only
    re-mine, but the re-mine is non-convergent, so its citations now round-trip
    through L2. It must NOT be in DEFAULT_BULK_EXCLUDED_SOURCES, and dump → build
    must reproduce its citations (while the true bulk `wiktionary` stays
    excluded)."""
    from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in
    from wyrd.generators.kenning.jsonl.dump import DEFAULT_BULK_EXCLUDED_SOURCES

    assert "wiktionary-empirical" not in DEFAULT_BULK_EXCLUDED_SOURCES
    assert "wiktionary" in DEFAULT_BULK_EXCLUDED_SOURCES  # the 5.8M-row bulk stays out

    pre = _build_fixture_db()
    _add_source(pre, id="wiktionary-empirical", title="Wiktionary (empirical corpus mining)")
    eid = _add_etymon(pre, "old-english", "worþ")
    pre.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) "
        "VALUES (?, 'wiktionary-empirical', 'corpus')",
        (eid,),
    )
    dump_source_to_file(pre, "wiktionary-empirical", tmp_path)
    pre.close()

    rebuilt = _build_fixture_db()
    counts = build_from_jsonl(rebuilt, jsonl_paths_in(tmp_path))
    assert counts.get("citation", 0) == 1  # the citation replays, not silently dropped
    row = rebuilt.execute(
        "SELECT e.canonical_form, e.language FROM etymon_citation c "
        "JOIN etymon e ON e.id = c.etymon_id WHERE c.source_id = 'wiktionary-empirical'"
    ).fetchone()
    assert row is not None, "No citation found for wiktionary-empirical after rebuild"
    assert (row["canonical_form"], row["language"]) == ("worþ", "old-english")
    rebuilt.close()


def test_wiktionary_empirical_citation_merges_with_scholar_source(tmp_path: Path):
    """wyrd-x33t: an etymon cited by BOTH wiktionary-empirical AND a real scholar
    source must rebuild to ONE etymon carrying both citations (the realistic
    worth-gate overlap shape) — the cross-file merge-by-ref unifies them."""
    from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in

    pre = _build_fixture_db()
    _add_source(pre, id="wiktionary-empirical", title="Wiktionary (empirical corpus mining)")
    _add_source(pre, id="skeat", title="Skeat")
    eid = _add_etymon(pre, "old-english", "worþ")
    pre.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, 'wiktionary-empirical')",
        (eid,),
    )
    pre.execute("INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, 'skeat')", (eid,))
    dump_source_to_file(pre, "wiktionary-empirical", tmp_path)
    dump_source_to_file(pre, "skeat", tmp_path)
    pre.close()

    rebuilt = _build_fixture_db()
    build_from_jsonl(rebuilt, jsonl_paths_in(tmp_path))
    # One etymon, both citations.
    assert (
        rebuilt.execute(
            "SELECT count(*) FROM etymon WHERE canonical_form='worþ' AND language='old-english'"
        ).fetchone()[0]
        == 1
    )
    assert {
        r[0]
        for r in rebuilt.execute(
            "SELECT source_id FROM etymon_citation c JOIN etymon e ON e.id=c.etymon_id "
            "WHERE e.canonical_form='worþ'"
        )
    } == {"wiktionary-empirical", "skeat"}
    rebuilt.close()


def test_default_bulk_excluded_sources_excludes_modern_reflex_curation():
    """A rebuilt DB carries a ``modern-reflex-curation`` row in ``source``
    (inserted by ``import-modern-reflexes`` from ``_modern_reflexes.jsonl``).
    Without the exclusion, dump_all_sources would emit a competing per-source
    ``modern-reflex-curation.jsonl`` (lacking the enrichment-derived cognate_id)
    alongside the curated file — pins the wyrd-vewk exclusion."""
    from wyrd.generators.kenning.jsonl.dump import DEFAULT_BULK_EXCLUDED_SOURCES

    assert "modern-reflex-curation" in DEFAULT_BULK_EXCLUDED_SOURCES

    conn = _build_fixture_db()
    _add_source(conn, id="modern-reflex-curation", title="Curated modern-English reflexes")
    _add_source(conn, id="skeat", title="Skeat")

    sids = list_source_ids(conn, exclude=DEFAULT_BULK_EXCLUDED_SOURCES)
    assert "modern-reflex-curation" not in sids
    assert "skeat" in sids


# ---------------------------------------------------------------------
# wyrd-2t28: orphan-attestation telemetry
# ---------------------------------------------------------------------


def test_count_orphan_attestations_zero_when_all_routed() -> None:
    """Attestations whose source_doc exactly matches a source.id route
    cleanly and aren't counted as orphans."""
    conn = _build_fixture_db()
    _add_source(conn, id="skeat", title="X")
    tid = _add_toponym(conn, "Stratford")
    _add_attestation(conn, toponym_id=tid, form="Streatfort", source_doc="skeat")
    assert count_orphan_attestations(conn) == 0


def test_count_orphan_attestations_counts_unrouted_source_doc() -> None:
    """A source_doc that neither matches a source.id nor a registered
    prefix is counted as an orphan."""
    conn = _build_fixture_db()
    _add_source(conn, id="skeat", title="X")
    tid = _add_toponym(conn, "Stratford")
    _add_attestation(conn, toponym_id=tid, form="Streatfort", source_doc="unknown_ingester")
    assert count_orphan_attestations(conn) == 1


def test_count_orphan_attestations_ignores_null_source_doc() -> None:
    """Rows with NULL source_doc aren't orphans — they're un-attributed
    by design, not misrouted."""
    conn = _build_fixture_db()
    tid = _add_toponym(conn, "Stratford")
    _add_attestation(conn, toponym_id=tid, form="Streatfort", source_doc=None)
    assert count_orphan_attestations(conn) == 0


def test_count_orphan_attestations_excludes_registered_prefix() -> None:
    """A source_doc matching a prefix in _ATTESTATION_DOC_PREFIX_TO_SOURCE
    (e.g. 'Phillimore ...') routes to its target source and is NOT an
    orphan. Pins the prefix-exclusion path."""
    conn = _build_fixture_db()
    _add_source(conn, id="open_domesday_hull", title="Hull")
    tid = _add_toponym(conn, "Settlement")
    _add_attestation(
        conn,
        toponym_id=tid,
        form="Settlentum",
        source_doc="Phillimore 1L1.2",
    )
    assert count_orphan_attestations(conn) == 0


def test_dump_all_sources_warns_on_orphan_attestations(tmp_path: Path, caplog) -> None:
    """dump_all_sources logs a warning when orphan_count > 0 so the
    operator notices silently-dropped rows."""
    import logging

    conn = _build_fixture_db()
    _add_source(conn, id="skeat", title="X")
    tid = _add_toponym(conn, "Stratford")
    _add_attestation(conn, toponym_id=tid, form="Streatfort", source_doc="unknown_ingester")
    with caplog.at_level(logging.WARNING, logger="wyrd.generators.kenning.jsonl.dump"):
        dump_all_sources(conn, tmp_path, exclude=())
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("toponym_attestation row(s)" in r.message for r in warnings), (
        f"expected orphan-attestation warning, got: {[r.message for r in warnings]}"
    )


def test_dump_all_sources_does_not_warn_when_all_attestations_routed(
    tmp_path: Path, caplog
) -> None:
    """When every source_doc routes cleanly, no orphan warning fires
    — pins the silent-default path so a regression that started
    spuriously warning would surface here."""
    import logging

    conn = _build_fixture_db()
    _add_source(conn, id="skeat", title="X")
    tid = _add_toponym(conn, "Stratford")
    _add_attestation(conn, toponym_id=tid, form="Streatfort", source_doc="skeat")
    with caplog.at_level(logging.WARNING, logger="wyrd.generators.kenning.jsonl.dump"):
        dump_all_sources(conn, tmp_path, exclude=())
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "toponym_attestation" in r.message
    ]
    assert warnings == []
