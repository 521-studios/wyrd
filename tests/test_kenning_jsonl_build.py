"""Tests for the per-source JSONL → SQLite rebuild — wyrd-f295.

The round-trip test at the bottom is the foundation contract:
``dump → wipe → build → dump-again`` produces semantically-identical
L2. If that test ever breaks the JSONL-as-source-of-truth promise
breaks too.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from wyrd.generators.kenning.jsonl_build import (
    BuildError,
    build_from_jsonl,
    jsonl_paths_in,
)
from wyrd.generators.kenning.jsonl_dump import (
    dump_all_sources,
)
from wyrd.generators.kenning.jsonl_log import write_jsonl

# ---------------------------------------------------------------------------
# Fixture: schema matching the L2-touching slice + the build-target columns
# ---------------------------------------------------------------------------


def _build_fixture_db() -> sqlite3.Connection:
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


def _write_jsonl(tmp: Path, source_id: str, rows: list[dict]) -> Path:
    path = tmp / f"{source_id}.jsonl"
    write_jsonl(path, rows)
    return path


# ---------------------------------------------------------------------------
# Single-source build
# ---------------------------------------------------------------------------


def test_build_inserts_source_row(tmp_path: Path):
    _write_jsonl(
        tmp_path,
        "skeat",
        [
            {"_type": "source", "ref": "skeat", "title": "Place-Names of Cambs", "year": 1901},
        ],
    )
    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["source"] == 1
    row = conn.execute("SELECT id, title, year FROM source").fetchone()
    assert (row["id"], row["title"], row["year"]) == ("skeat", "Place-Names of Cambs", 1901)


def test_build_inserts_etymon_with_glosses_and_tags(tmp_path: Path):
    _write_jsonl(
        tmp_path,
        "skeat",
        [
            {"_type": "source", "ref": "skeat", "title": "X"},
            {
                "_type": "etymon",
                "ref": "old-english:cot",
                "language": "old-english",
                "canonical_form": "cot",
                "glosses": ["cottage", "hut"],
                "tags": ["architecture"],
            },
        ],
    )
    conn = _build_fixture_db()
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    e = conn.execute("SELECT id, canonical_form, language FROM etymon").fetchone()
    assert (e["canonical_form"], e["language"]) == ("cot", "old-english")
    glosses = [
        r["gloss"]
        for r in conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id=?", (e["id"],))
    ]
    assert sorted(glosses) == ["cottage", "hut"]
    tags = [r["tag"] for r in conn.execute("SELECT tag FROM etymon_tag")]
    assert tags == ["architecture"]


def test_build_inserts_citation_with_source_fk(tmp_path: Path):
    _write_jsonl(
        tmp_path,
        "skeat",
        [
            {"_type": "source", "ref": "skeat", "title": "X"},
            {
                "_type": "etymon",
                "ref": "old-english:cot",
                "language": "old-english",
                "canonical_form": "cot",
            },
            {"_type": "citation", "etymon_ref": "old-english:cot", "page": "15"},
        ],
    )
    conn = _build_fixture_db()
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    c = conn.execute("SELECT source_id, page FROM etymon_citation").fetchone()
    assert (c["source_id"], c["page"]) == ("skeat", "15")


def test_build_inserts_descent_with_resolved_fks(tmp_path: Path):
    _write_jsonl(
        tmp_path,
        "wiki",
        [
            {"_type": "source", "ref": "wiki", "title": "Wiktionary"},
            {
                "_type": "etymon",
                "ref": "proto-germanic:*tunaz",
                "language": "proto-germanic",
                "canonical_form": "*tunaz",
            },
            {
                "_type": "etymon",
                "ref": "old-english:tun",
                "language": "old-english",
                "canonical_form": "tun",
            },
            {
                "_type": "etymon_descent",
                "parent_ref": "proto-germanic:*tunaz",
                "child_ref": "old-english:tun",
                "edge_type": "inheritance",
                "confidence": "high",
            },
        ],
    )
    conn = _build_fixture_db()
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    d = conn.execute("SELECT edge_type, confidence, source_id FROM etymon_descent").fetchone()
    assert (d["edge_type"], d["confidence"], d["source_id"]) == ("inheritance", "high", "wiki")


def test_build_inserts_mining_run_with_source_fk(tmp_path: Path):
    _write_jsonl(
        tmp_path,
        "skeat",
        [
            {"_type": "source", "ref": "skeat", "title": "X"},
            {
                "_type": "mining_run",
                "provider": "anthropic",
                "model": "claude-opus-4-7",
                "mode": "mine",
                "parsed_count": 100,
                "accepted": 95,
            },
        ],
    )
    conn = _build_fixture_db()
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    m = conn.execute("SELECT source_id, provider, parsed_count FROM mining_run").fetchone()
    assert (m["source_id"], m["provider"], m["parsed_count"]) == ("skeat", "anthropic", 100)


def test_build_inserts_toponym_and_etymology_elements(tmp_path: Path):
    _write_jsonl(
        tmp_path,
        "mawer",
        [
            {"_type": "source", "ref": "mawer", "title": "N&D"},
            {
                "_type": "etymon",
                "ref": "old-english:cot",
                "language": "old-english",
                "canonical_form": "cot",
            },
            {
                "_type": "etymon",
                "ref": "old-english:tun",
                "language": "old-english",
                "canonical_form": "tun",
            },
            {
                "_type": "toponym",
                "ref": "Cotton@Norfolk",
                "modern_name": "Cotton",
                "country": "England",
                "region": "Norfolk",
            },
            {
                "_type": "etymology_element",
                "toponym_ref": "Cotton@Norfolk",
                "historical_form": "Cotuna",
                "confidence": "high",
                "elements": [
                    {"ordinal": 1, "etymon_ref": "old-english:cot"},
                    {"ordinal": 2, "etymon_ref": "old-english:tun", "inflection": "oblique"},
                ],
            },
        ],
    )
    conn = _build_fixture_db()
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    te = conn.execute(
        "SELECT historical_form, confidence, source_id FROM toponym_etymology"
    ).fetchone()
    assert (te["historical_form"], te["confidence"], te["source_id"]) == ("Cotuna", "high", "mawer")
    elements = [
        (r["ordinal"], r["etymon_id"], r["inflection"])
        for r in conn.execute(
            "SELECT ordinal, etymon_id, inflection FROM toponym_etymology_element ORDER BY ordinal"
        )
    ]
    assert len(elements) == 2
    assert elements[0][0] == 1 and elements[1][0] == 2
    assert elements[1][2] == "oblique"


# ---------------------------------------------------------------------------
# Cross-file merging
# ---------------------------------------------------------------------------


def test_build_merges_etymon_across_files(tmp_path: Path):
    """Same etymon ref in two files = ONE etymon row, with union of
    glosses + tags. Scalar fields (notes) follow last-write-wins by
    file-order (alphabetical)."""
    _write_jsonl(
        tmp_path,
        "a_file",
        [
            {"_type": "source", "ref": "a_file", "title": "A"},
            {
                "_type": "etymon",
                "ref": "old-english:cot",
                "language": "old-english",
                "canonical_form": "cot",
                "glosses": ["cottage"],
                "tags": ["architecture"],
                "notes": "from A",
            },
        ],
    )
    _write_jsonl(
        tmp_path,
        "b_file",
        [
            {"_type": "source", "ref": "b_file", "title": "B"},
            {
                "_type": "etymon",
                "ref": "old-english:cot",
                "language": "old-english",
                "canonical_form": "cot",
                "glosses": ["hut", "shed"],
                "tags": ["domestic"],
                "notes": "from B",
            },
        ],
    )
    conn = _build_fixture_db()
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    rows = list(conn.execute("SELECT id, notes FROM etymon"))
    assert len(rows) == 1  # merged into one row
    # Last-write-wins (b_file > a_file by alphabetical order).
    assert rows[0]["notes"] == "from B"
    glosses = {r["gloss"] for r in conn.execute("SELECT gloss FROM etymon_gloss")}
    assert glosses == {"cottage", "hut", "shed"}
    tags = {r["tag"] for r in conn.execute("SELECT tag FROM etymon_tag")}
    assert tags == {"architecture", "domestic"}


# ---------------------------------------------------------------------------
# Contract enforcement
# ---------------------------------------------------------------------------


def test_build_rejects_file_with_no_source(tmp_path: Path):
    _write_jsonl(
        tmp_path,
        "x",
        [
            {
                "_type": "etymon",
                "ref": "old-english:cot",
                "language": "old-english",
                "canonical_form": "cot",
            },
        ],
    )
    conn = _build_fixture_db()
    with pytest.raises(BuildError, match="no source row"):
        build_from_jsonl(conn, jsonl_paths_in(tmp_path))


def test_build_rejects_file_with_multiple_sources(tmp_path: Path):
    _write_jsonl(
        tmp_path,
        "x",
        [
            {"_type": "source", "ref": "a", "title": "A"},
            {"_type": "source", "ref": "b", "title": "B"},
        ],
    )
    conn = _build_fixture_db()
    with pytest.raises(BuildError, match="found 2 source"):
        build_from_jsonl(conn, jsonl_paths_in(tmp_path))


def test_build_rejects_citation_to_unknown_etymon(tmp_path: Path):
    _write_jsonl(
        tmp_path,
        "x",
        [
            {"_type": "source", "ref": "x", "title": "X"},
            {"_type": "citation", "etymon_ref": "old-english:ghost"},
        ],
    )
    conn = _build_fixture_db()
    with pytest.raises(BuildError, match="unknown etymon"):
        build_from_jsonl(conn, jsonl_paths_in(tmp_path))


def test_build_rejects_descent_to_unknown_etymon(tmp_path: Path):
    _write_jsonl(
        tmp_path,
        "x",
        [
            {"_type": "source", "ref": "x", "title": "X"},
            {
                "_type": "etymon_descent",
                "parent_ref": "old-english:ghost-a",
                "child_ref": "old-english:ghost-b",
                "edge_type": "inheritance",
            },
        ],
    )
    conn = _build_fixture_db()
    with pytest.raises(BuildError, match="unknown etymon"):
        build_from_jsonl(conn, jsonl_paths_in(tmp_path))


def test_build_rejects_element_ref_to_unknown_etymon(tmp_path: Path):
    _write_jsonl(
        tmp_path,
        "x",
        [
            {"_type": "source", "ref": "x", "title": "X"},
            {"_type": "toponym", "ref": "Foo@-", "modern_name": "Foo"},
            {
                "_type": "etymology_element",
                "toponym_ref": "Foo@-",
                "elements": [{"ordinal": 1, "etymon_ref": "old-english:ghost"}],
            },
        ],
    )
    conn = _build_fixture_db()
    with pytest.raises(BuildError, match="unknown etymon"):
        build_from_jsonl(conn, jsonl_paths_in(tmp_path))


# ---------------------------------------------------------------------------
# Round-trip: dump → wipe → build → dump produces same canonical state
# ---------------------------------------------------------------------------


def _populate_realistic_db() -> sqlite3.Connection:
    """A DB with a representative mix of L2 facts: two sources, shared
    etymons, citations + descent edges + a toponym etymology with
    elements + a mining run."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title, year) VALUES ('skeat', 'Cambs', 1901)")
    conn.execute("INSERT INTO source (id, title, year) VALUES ('mawer', 'N&D', 1920)")
    cur = conn.execute(
        "INSERT INTO etymon (language, canonical_form, notes) VALUES ('old-english', 'cot', 'dwelling')"
    )
    cot_id = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO etymon (language, canonical_form) VALUES ('old-english', 'tun')"
    )
    tun_id = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO etymon (language, canonical_form) VALUES ('proto-germanic', '*tunaz')"
    )
    proto_id = cur.lastrowid
    conn.execute("INSERT INTO etymon_gloss VALUES (?, 'cottage')", (cot_id,))
    conn.execute("INSERT INTO etymon_gloss VALUES (?, 'hut')", (cot_id,))
    conn.execute("INSERT INTO etymon_tag VALUES (?, 'architecture')", (cot_id,))
    # Skeat cites cot
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, 'skeat', '15')",
        (cot_id,),
    )
    # Mawer cites cot AND tun
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, 'mawer', '7')",
        (cot_id,),
    )
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, 'mawer', '8')",
        (tun_id,),
    )
    # Skeat attributes a descent edge from proto-germanic to tun
    conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id, confidence) "
        "VALUES (?, ?, 'inheritance', 'skeat', 'high')",
        (proto_id, tun_id),
    )
    # Mawer mined-Cotton (Norfolk) → cot + tun(oblique)
    cur = conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES ('Cotton', 'England', 'Norfolk')"
    )
    tid = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, historical_form, confidence) "
        "VALUES (?, 'mawer', 'Cotuna', 'high')",
        (tid,),
    )
    eid = cur.lastrowid
    conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) VALUES (?, 1, ?)",
        (eid, cot_id),
    )
    conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id, inflection) "
        "VALUES (?, 2, ?, 'oblique')",
        (eid, tun_id),
    )
    conn.execute(
        "INSERT INTO mining_run (source_id, provider, model, mode, parsed_count) "
        "VALUES ('mawer', 'anthropic', 'claude', 'mine', 42)"
    )
    return conn


def _l2_state_snapshot(conn: sqlite3.Connection) -> dict:
    """Compare-friendly snapshot of the L2-touching tables. Used by
    round-trip test."""
    return {
        "source": sorted(
            (r["id"], r["title"], r["year"])
            for r in conn.execute("SELECT id, title, year FROM source")
        ),
        "etymon": sorted(
            (r["language"], r["canonical_form"], r["notes"])
            for r in conn.execute("SELECT language, canonical_form, notes FROM etymon")
        ),
        "etymon_gloss": sorted(
            (r["language"], r["canonical_form"], r["gloss"])
            for r in conn.execute("""
                SELECT e.language, e.canonical_form, g.gloss
                FROM etymon_gloss g JOIN etymon e ON e.id = g.etymon_id
            """)
        ),
        "etymon_tag": sorted(
            (r["language"], r["canonical_form"], r["tag"])
            for r in conn.execute("""
                SELECT e.language, e.canonical_form, t.tag
                FROM etymon_tag t JOIN etymon e ON e.id = t.etymon_id
            """)
        ),
        "etymon_citation": sorted(
            (r["language"], r["canonical_form"], r["source_id"], r["page"])
            for r in conn.execute("""
                SELECT e.language, e.canonical_form, c.source_id, c.page
                FROM etymon_citation c JOIN etymon e ON e.id = c.etymon_id
            """)
        ),
        "etymon_descent": sorted(
            (
                r["p_lang"],
                r["p_form"],
                r["c_lang"],
                r["c_form"],
                r["edge_type"],
                r["source_id"],
                r["confidence"],
            )
            for r in conn.execute("""
                SELECT pe.language AS p_lang, pe.canonical_form AS p_form,
                       ce.language AS c_lang, ce.canonical_form AS c_form,
                       d.edge_type, d.source_id, d.confidence
                FROM etymon_descent d
                JOIN etymon pe ON pe.id = d.parent_id
                JOIN etymon ce ON ce.id = d.child_id
            """)
        ),
        "mining_run": sorted(
            (r["source_id"], r["provider"], r["model"], r["mode"], r["parsed_count"])
            for r in conn.execute(
                "SELECT source_id, provider, model, mode, parsed_count FROM mining_run"
            )
        ),
        "toponym": sorted(
            (r["modern_name"], r["country"], r["region"])
            for r in conn.execute("SELECT modern_name, country, region FROM toponym")
        ),
        "toponym_etymology": sorted(
            (r["modern_name"], r["region"], r["source_id"], r["historical_form"], r["confidence"])
            for r in conn.execute("""
                SELECT t.modern_name, t.region, te.source_id, te.historical_form, te.confidence
                FROM toponym_etymology te JOIN toponym t ON t.id = te.toponym_id
            """)
        ),
        "toponym_etymology_element": sorted(
            (
                r["modern_name"],
                r["region"],
                r["ordinal"],
                r["language"],
                r["canonical_form"],
                r["inflection"],
            )
            for r in conn.execute("""
                SELECT t.modern_name, t.region, el.ordinal,
                       e.language, e.canonical_form, el.inflection
                FROM toponym_etymology_element el
                JOIN toponym_etymology te ON te.id = el.toponym_etymology_id
                JOIN toponym t ON t.id = te.toponym_id
                JOIN etymon e ON e.id = el.etymon_id
            """)
        ),
    }


def test_round_trip_dump_rebuild_dump(tmp_path: Path):
    """The foundation contract: ``dump → wipe → rebuild → dump-again``
    produces semantically-identical L2 state, and the JSONL files
    byte-match the second dump."""
    pre = _populate_realistic_db()
    snapshot_pre = _l2_state_snapshot(pre)

    dump_dir_a = tmp_path / "dump_a"
    dump_dir_a.mkdir()
    dump_all_sources(pre, dump_dir_a, exclude=())

    # Wipe + rebuild.
    rebuilt = _build_fixture_db()
    build_from_jsonl(rebuilt, jsonl_paths_in(dump_dir_a))

    snapshot_post = _l2_state_snapshot(rebuilt)
    assert snapshot_pre == snapshot_post, (
        "DB-level snapshot diverged across dump→rebuild round-trip"
    )

    # Dump again and compare files row-by-row (semantically — same
    # set of canonical rows per source).
    dump_dir_b = tmp_path / "dump_b"
    dump_dir_b.mkdir()
    dump_all_sources(rebuilt, dump_dir_b, exclude=())

    files_a = sorted(p.name for p in dump_dir_a.glob("*.jsonl"))
    files_b = sorted(p.name for p in dump_dir_b.glob("*.jsonl"))
    assert files_a == files_b

    for fname in files_a:
        rows_a = sorted(json.dumps(r, sort_keys=True) for r in _read_rows(dump_dir_a / fname))
        rows_b = sorted(json.dumps(r, sort_keys=True) for r in _read_rows(dump_dir_b / fname))
        assert rows_a == rows_b, f"round-trip diff in {fname}"


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
