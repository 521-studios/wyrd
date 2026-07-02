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

from wyrd.generators.kenning.jsonl.build import (
    BuildError,
    build_from_jsonl,
    jsonl_paths_in,
)
from wyrd.generators.kenning.jsonl.dump import (
    dump_all_sources,
)
from wyrd.generators.kenning.jsonl.log import write_jsonl

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
            merged_into_id INTEGER,
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


def test_build_merges_etymon_when_row_already_exists(tmp_path: Path):
    """L1 wiktextract bulk ingest writes etymon rows BEFORE the L2 JSONL
    replay runs (when ``rebuild-from-jsonl`` is invoked without
    ``--skip-bulk``). L2 JSONL files often reference those etymons by
    ref (the wyrd-4hx7 wiktionary-empirical mining path is the obvious
    source), so the replay hits ``UNIQUE constraint failed:
    etymon.canonical_form, etymon.language`` on the plain INSERT.

    Regression test for wyrd-pcsj: pre-populate an etymon (simulating
    L1 ingest), then run ``build_from_jsonl`` with a JSONL referencing
    the same (canonical_form, language). Must not raise, must merge
    glosses + tags onto the existing row, and must keep the L1 row's
    id stable so downstream citations / descent edges resolve."""
    conn = _build_fixture_db()
    # Simulate L1: pre-populate one etymon with an L1-shaped payload
    # (canonical_form + language + pronunciation_ipa, but no glosses
    # or tags yet — those are L2's contribution).
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language, pronunciation_ipa) VALUES (?, ?, ?)",
        ("cot", "old-english", "/kɒt/"),
    )
    pre_eid = cur.lastrowid
    conn.commit()

    # Now run L2 replay against a JSONL that names the same etymon.
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
    # Must not raise.
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))

    # Exactly one etymon row (merged, not duplicated).
    rows = list(conn.execute("SELECT id, canonical_form, language, pronunciation_ipa FROM etymon"))
    assert len(rows) == 1
    row = rows[0]
    # L1 row's id stays stable so anything that references it via FK still works.
    assert row["id"] == pre_eid
    # L1's pronunciation_ipa survives (L2 didn't set it).
    assert row["pronunciation_ipa"] == "/kɒt/"

    # L2's glosses + tags landed on the existing row.
    glosses = sorted(
        r["gloss"]
        for r in conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id=?", (pre_eid,))
    )
    assert glosses == ["cottage", "hut"]
    tags = [
        r["tag"] for r in conn.execute("SELECT tag FROM etymon_tag WHERE etymon_id=?", (pre_eid,))
    ]
    assert tags == ["architecture"]


def test_build_merge_dedupes_glosses_and_tags_on_l1_overlap(tmp_path: Path):
    """When L1 pre-populated the etymon AND its glosses/tags, and L2
    JSONL has overlapping glosses/tags, the post-merge state must be
    set-union (no duplicates). This is the gloss/tag side of the
    L1+L2 merge contract — separate from the test that exercises
    L1-with-no-glosses + L2-adds-glosses.

    Without ``INSERT OR IGNORE`` on the (etymon_id, gloss) PK, an L1
    row that already had ``barn`` would conflict when L2 also tries
    to write ``barn``. Pinned here so a future change can't regress
    the dedup."""
    conn = _build_fixture_db()
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        ("cot", "old-english"),
    )
    pre_eid = cur.lastrowid
    # L1-style pre-population: a couple of glosses + a tag already on disk.
    for g in ("barn", "shed"):
        conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
            (pre_eid, g),
        )
    conn.execute("INSERT INTO etymon_tag (etymon_id, tag) VALUES (?, ?)", (pre_eid, "architecture"))
    conn.commit()

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
                # ``barn`` overlaps with L1; ``cottage`` is new.
                "glosses": ["barn", "cottage"],
                # ``architecture`` overlaps with L1; ``settlement`` is new.
                "tags": ["architecture", "settlement"],
            },
        ],
    )
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))

    glosses = sorted(
        r["gloss"]
        for r in conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id=?", (pre_eid,))
    )
    # Set-union: barn (L1) + cottage (L2) + shed (L1). No duplicate barn.
    assert glosses == ["barn", "cottage", "shed"]
    tags = sorted(
        r["tag"] for r in conn.execute("SELECT tag FROM etymon_tag WHERE etymon_id=?", (pre_eid,))
    )
    # Set-union: architecture (L1+L2 overlap) + settlement (L2).
    assert tags == ["architecture", "settlement"]


def test_insert_etymon_raises_build_error_when_select_misses_after_conflict():
    """Defensive branch in _insert_etymon: when ``INSERT OR IGNORE``
    fails AND the subsequent SELECT can't find the matching row,
    raise a clear ``BuildError`` rather than letting None propagate
    to TypeError on row indexing.

    SQLite's ``INSERT OR IGNORE`` suppresses every constraint failure
    (UNIQUE, NOT NULL, CHECK, FK), not just UNIQUE conflicts. The
    docstring on _insert_etymon explains that for the etymon table
    only the UNIQUE constraint can fire in practice (canonical_form +
    language are validated non-null above), but a NOT NULL failure
    here is the cheapest way to exercise the defensive branch
    deterministically: feed an explicit ``None`` past the early
    validation guard, watch INSERT OR IGNORE swallow it, then assert
    the BuildError pins the offending payload context."""
    from wyrd.generators.kenning.jsonl.build import BuildError, _insert_etymon

    conn = _build_fixture_db()
    # Payload satisfies the early ``key in payload`` validation but
    # carries None values — INSERT OR IGNORE swallows the NOT NULL
    # violation, no row is created, and the SELECT for the (None, None)
    # pair returns None.
    with pytest.raises(BuildError) as exc_info:
        _insert_etymon(
            conn,
            {
                "canonical_form": None,
                "language": None,
            },
        )
    # The error message must include enough payload context to find
    # the offending row in the JSONL.
    msg = str(exc_info.value)
    assert "could not be located" in msg


def test_build_merge_overwrites_scalar_when_l2_specifies_it(tmp_path: Path):
    """When the existing (L1) row has a non-NULL scalar value AND the
    L2 payload explicitly sets the same field, L2 wins (matches the
    existing _merge_etymon last-write-wins semantics for scalars).

    Pinned separately from the basic merge test so a future change
    that swapped the precedence (e.g. preserve-L1-when-non-null)
    would be caught immediately."""
    conn = _build_fixture_db()
    conn.execute(
        "INSERT INTO etymon (canonical_form, language, notes) VALUES (?, ?, ?)",
        ("cot", "old-english", "L1 note"),
    )
    conn.commit()
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
                "notes": "L2 override",  # explicit override
            },
        ],
    )
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    row = conn.execute("SELECT notes FROM etymon WHERE canonical_form='cot'").fetchone()
    assert row["notes"] == "L2 override"


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


def test_build_cited_source_registers_and_attributes_citations(tmp_path: Path):
    """wyrd-2b50: a file declares a secondary source via ``cited_source``
    and a ``citation`` row attributes itself to it with an explicit
    ``source_id``. The secondary source is registered in ``source`` and
    the citation lands against it; a citation with no ``source_id`` falls
    back to the file's primary source. This is the 'one source cites on
    behalf of others' path (no separate JSONL per source)."""
    _write_jsonl(
        tmp_path,
        "primary",
        [
            {"_type": "source", "ref": "primary", "title": "Primary"},
            {"_type": "cited_source", "ref": "secondary", "title": "Secondary"},
            {
                "_type": "etymon",
                "ref": "old-english:x",
                "language": "old-english",
                "canonical_form": "x",
            },
            {"_type": "citation", "etymon_ref": "old-english:x"},  # → primary
            {"_type": "citation", "etymon_ref": "old-english:x", "source_id": "secondary"},
        ],
    )
    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    registered = {r["id"] for r in conn.execute("SELECT id FROM source")}
    assert {"primary", "secondary"} <= registered
    attributed = {r["source_id"] for r in conn.execute("SELECT source_id FROM etymon_citation")}
    assert attributed == {"primary", "secondary"}
    assert counts["citation_source_orphans"] == 0


def test_build_unregistered_citation_source_is_counted_not_swallowed(tmp_path: Path):
    """wyrd-2b50: a citation whose explicit ``source_id`` names a source
    no file registered is counted in ``citation_source_orphans`` and
    skipped — not silently dropped by the FK + INSERT OR IGNORE (which
    would also have falsely incremented the citation count)."""
    _write_jsonl(
        tmp_path,
        "primary",
        [
            {"_type": "source", "ref": "primary", "title": "Primary"},
            {
                "_type": "etymon",
                "ref": "old-english:x",
                "language": "old-english",
                "canonical_form": "x",
            },
            {"_type": "citation", "etymon_ref": "old-english:x", "source_id": "never-declared"},
        ],
    )
    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["citation"] == 0
    assert counts["citation_source_orphans"] == 1
    assert conn.execute("SELECT COUNT(*) FROM etymon_citation").fetchone()[0] == 0


def test_build_empty_source_id_is_orphan_not_file_fallback(tmp_path: Path):
    """wyrd-2b50: a present-but-empty ``source_id`` is kept verbatim
    (``.get`` keys on absence, not falsiness) and surfaces as an orphan
    — it does NOT silently fall back to the file's primary source."""
    _write_jsonl(
        tmp_path,
        "primary",
        [
            {"_type": "source", "ref": "primary", "title": "Primary"},
            {
                "_type": "etymon",
                "ref": "old-english:x",
                "language": "old-english",
                "canonical_form": "x",
            },
            {"_type": "citation", "etymon_ref": "old-english:x", "source_id": ""},
        ],
    )
    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["citation"] == 0
    assert counts["citation_source_orphans"] == 1
    # Not misattributed to "primary".
    assert conn.execute("SELECT COUNT(*) FROM etymon_citation").fetchone()[0] == 0


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


def test_build_round_trips_per_element_confidence_when_present(tmp_path: Path):
    """wyrd-2n1: per-element confidence from the JSONL lands on the
    rebuilt toponym_etymology_element rows. Pins the build-side
    INSERT parameter directly — without it, a regression that always
    inserted NULL would pass the existing test (which doesn't read
    el.confidence)."""
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
                "confidence": "medium",  # parent aggregate
                "elements": [
                    # Mixed-confidence elements — the scenario the
                    # whole migration exists for.
                    {"ordinal": 1, "etymon_ref": "old-english:cot", "confidence": "high"},
                    {"ordinal": 2, "etymon_ref": "old-english:tun", "confidence": "low"},
                ],
            },
        ],
    )
    conn = _build_fixture_db()
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    elements = [
        (r["ordinal"], r["confidence"])
        for r in conn.execute(
            "SELECT ordinal, confidence FROM toponym_etymology_element ORDER BY ordinal"
        )
    ]
    assert elements == [(1, "high"), (2, "low")]


def test_build_backfills_per_element_confidence_from_parent_when_jsonl_omits_it(
    tmp_path: Path,
):
    """wyrd-2n1 reconstructibility gate: rebuilding from pre-2n1 JSONL
    (which doesn't carry per-element confidence) lands the parent's
    value on every element, matching the migration's lossless-
    starting-point promise. Without the post-build hook, a rebuilt
    DB would silently differ from a migrated-legacy DB on the new
    column."""
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
                "_type": "toponym",
                "ref": "Cot@Norfolk",
                "modern_name": "Cot",
                "country": "England",
                "region": "Norfolk",
            },
            {
                "_type": "etymology_element",
                "toponym_ref": "Cot@Norfolk",
                "confidence": "high",  # parent only — no per-element field
                "elements": [
                    {"ordinal": 1, "etymon_ref": "old-english:cot"},
                ],
            },
        ],
    )
    conn = _build_fixture_db()
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    confidence = conn.execute("SELECT confidence FROM toponym_etymology_element").fetchone()[0]
    # Parent value propagated to the per-element row via the post-
    # build backfill hook.
    assert confidence == "high"


# ---------------------------------------------------------------------------
# Cross-file merging
# ---------------------------------------------------------------------------


def test_table_counts_returns_zero_for_empty_table(tmp_path: Path):
    """table_counts reports 0 for empty tables (distinct from missing)."""
    from wyrd.generators.kenning.jsonl.build import table_counts

    conn = _build_fixture_db()
    # Empty schema — every L2 table is present but empty.
    counts = table_counts(conn, ["source", "etymon", "toponym"])
    assert counts == {"source": 0, "etymon": 0, "toponym": 0}


def test_table_counts_returns_negative_one_for_missing_table(tmp_path: Path):
    """Missing tables surface as -1 so schema mismatch shows up clearly
    instead of conflating with zero."""
    from wyrd.generators.kenning.jsonl.build import table_counts

    conn = _build_fixture_db()
    counts = table_counts(conn, ["source", "does_not_exist"])
    assert counts["source"] == 0
    assert counts["does_not_exist"] == -1


def test_diff_table_counts_zero_delta_for_identical(tmp_path: Path):
    from wyrd.generators.kenning.jsonl.build import diff_table_counts

    before = {"source": 5, "etymon": 100}
    after = {"source": 5, "etymon": 100}
    rows = diff_table_counts(before, after)
    assert all(r["delta"] == 0 for r in rows)


def test_diff_table_counts_signed_delta(tmp_path: Path):
    from wyrd.generators.kenning.jsonl.build import diff_table_counts

    before = {"etymon": 100, "toponym": 50}
    after = {"etymon": 102, "toponym": 48}
    rows = diff_table_counts(before, after)
    rows_by_table = {r["table"]: r for r in rows}
    assert rows_by_table["etymon"]["delta"] == 2
    assert rows_by_table["toponym"]["delta"] == -2


def test_diff_table_counts_unions_tables(tmp_path: Path):
    """A table present only in one snapshot still shows up (with the
    missing side as -1). Catches schema-evolution drift."""
    from wyrd.generators.kenning.jsonl.build import diff_table_counts

    before = {"etymon": 100}
    after = {"etymon": 100, "new_table": 5}
    rows = diff_table_counts(before, after)
    new_row = next(r for r in rows if r["table"] == "new_table")
    assert new_row["before"] == -1
    assert new_row["after"] == 5
    # Sentinel -1 must NOT poison the delta arithmetic — mixing it
    # with a real count would produce misleading deltas like
    # 5 - (-1) = 6 when the truth is "missing on one side".
    assert new_row["delta"] is None


def test_diff_table_counts_missing_table_yields_none_delta():
    """When a table is missing (-1) on the source side but present on
    the rebuild side, delta is None (explicit "can't compare") rather
    than the misleading after - (-1)."""
    from wyrd.generators.kenning.jsonl.build import diff_table_counts

    rows = diff_table_counts({"etymon": -1}, {"etymon": 100})
    assert rows[0]["delta"] is None


def test_has_any_delta():
    from wyrd.generators.kenning.jsonl.build import has_any_delta

    assert not has_any_delta([{"table": "x", "before": 5, "after": 5, "delta": 0}])
    assert has_any_delta([{"table": "x", "before": 5, "after": 6, "delta": 1}])
    # None delta (missing table) counts as a delta — schema mismatch
    # shouldn't pass the CI gate.
    assert has_any_delta([{"table": "x", "before": -1, "after": 5, "delta": None}])


def test_format_diff_rebuild_shows_signed_deltas():
    from wyrd.generators.kenning.jsonl.build import format_diff_rebuild

    rows = [
        {"table": "etymon", "before": 100, "after": 102, "delta": 2},
        {"table": "toponym", "before": 50, "after": 48, "delta": -2},
        {"table": "source", "before": 5, "after": 5, "delta": 0},
    ]
    md = format_diff_rebuild(rows)
    assert "etymon" in md and "+2" in md
    assert "toponym" in md and "-2" in md
    # No-change rows render as plain '0' (no sign).
    zero_lines = [ln for ln in md.split("\n") if "source" in ln]
    assert "+" not in zero_lines[0].split("|")[-2]
    assert "-" not in zero_lines[0].split("|")[-2]


def test_format_diff_rebuild_handles_wiktionary_scale_counts():
    """Number columns sized for 9-digit counts so the live wiktionary
    slice (2.37M etymons, 1.76M descent edges) renders without
    overflowing the column and corrupting the markdown table."""
    from wyrd.generators.kenning.jsonl.build import format_diff_rebuild

    rows = [{"table": "etymon", "before": 2373985, "after": 6318, "delta": -2367667}]
    md = format_diff_rebuild(rows)
    # Number stays in its cell — pipe before/after should still align.
    line = [ln for ln in md.split("\n") if "etymon" in ln][0]
    # 4 pipes + 5 cells (table + before + after + delta + trailing)
    assert line.count("|") == 5
    assert "2373985" in line
    assert "-2367667" in line


def test_rebuild_from_jsonl_summary_handles_list_valued_counts(tmp_path: Path, monkeypatch):
    """Regression test for wyrd-f4nl: rebuild-from-jsonl summary used
    to crash with `TypeError: unsupported format string passed to
    list.__format__` because the counts dict picked up list-valued
    fields (orphan_refs from wyrd-lene) but the summary loop used
    `{n:>8}` which only works on numbers.

    The fix renders list fields as `(N samples)` and scalar fields as
    right-aligned numbers. This test verifies the CLI runs to
    completion when the counts dict contains both kinds."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    # Minimal JSONL that builds without errors but produces empty
    # orphan_refs lists in the counts dict.
    src = tmp_path / "skeat.jsonl"
    src.write_text('{"_type": "source", "ref": "skeat", "title": "X"}\n')

    db_path = tmp_path / "test.db"
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["lexicon", "rebuild-from-jsonl", "--db", str(db_path), "--jsonl-dir", str(tmp_path)],
    )
    # The fix: command runs to completion without TypeError. Pre-fix
    # this raised the format error and exit code 1.
    assert result.exit_code == 0, f"rebuild crashed:\n{result.output}"
    # List-handling renders as "(N samples)".
    assert "samples" in result.output or "samples" in (result.stderr_bytes or b"").decode()


# ---------------------------------------------------------------------------
# Per-row-type helper tests (wyrd-3vh3)
#
# build_from_jsonl Pass 3 was extracted into four helpers. They share
# a signature shape (conn, source_id, rows, [maps], counts) so the test
# fixtures are reusable. Each helper owns its orphan-skip + dedup rules,
# so the tests target those semantics directly without rebuilding the
# full pipeline.
# ---------------------------------------------------------------------------


def _empty_counts() -> dict:
    """Counts dict shape that the helpers mutate. Mirrors the dict
    build_from_jsonl initializes."""
    return {
        "citation": 0,
        "citation_orphans": 0,
        "citation_orphan_refs": [],
        "citation_source_orphans": 0,
        "citation_source_orphan_refs": [],
        "etymon_descent": 0,
        "etymon_descent_orphans": 0,
        "etymon_descent_orphan_refs": [],
        "mining_run": 0,
        "etymology_element": 0,
        "etymology_element_dupes_skipped": 0,
        "etymology_element_orphans": 0,
        "etymology_element_orphan_refs": [],
    }


def _seed_etymon(conn, language: str, canonical_form: str) -> int:
    cur = conn.execute(
        "INSERT INTO etymon (language, canonical_form) VALUES (?, ?)",
        (language, canonical_form),
    )
    return cur.lastrowid


def _seed_source(conn, source_id: str) -> None:
    conn.execute("INSERT INTO source (id, title) VALUES (?, ?)", (source_id, "X"))


def _seed_toponym(conn, name: str, region: str | None = None) -> int:
    cur = conn.execute("INSERT INTO toponym (modern_name, region) VALUES (?, ?)", (name, region))
    return cur.lastrowid


def test_insert_citation_rows_skips_unknown_etymon_ref():
    """Orphan citation → counted in citation_orphans + ref captured."""
    from wyrd.generators.kenning.jsonl.build import _insert_citation_rows

    conn = _build_fixture_db()
    _seed_source(conn, "skeat")
    cot_id = _seed_etymon(conn, "old-english", "cot")
    counts = _empty_counts()
    rows = [
        {"etymon_ref": "old-english:cot", "page": "15"},
        {"etymon_ref": "old-english:ghost", "page": "99"},
    ]
    _insert_citation_rows(conn, "skeat", rows, {"old-english:cot": cot_id}, {"skeat"}, counts)
    assert counts["citation"] == 1
    assert counts["citation_orphans"] == 1
    assert counts["citation_orphan_refs"] == ["old-english:ghost"]


def test_insert_descent_rows_skips_either_endpoint_missing():
    """Orphan descent edge → counted + ref pair captured."""
    from wyrd.generators.kenning.jsonl.build import _insert_descent_rows

    conn = _build_fixture_db()
    _seed_source(conn, "wiki")
    proto_id = _seed_etymon(conn, "proto-germanic", "*tunaz")
    oe_id = _seed_etymon(conn, "old-english", "tun")
    counts = _empty_counts()
    rows = [
        {
            "parent_ref": "proto-germanic:*tunaz",
            "child_ref": "old-english:tun",
            "edge_type": "inheritance",
        },
        {
            "parent_ref": "old-english:ghost-a",
            "child_ref": "old-english:tun",
            "edge_type": "inheritance",
        },
    ]
    _insert_descent_rows(
        conn,
        "wiki",
        rows,
        {"proto-germanic:*tunaz": proto_id, "old-english:tun": oe_id},
        counts,
    )
    assert counts["etymon_descent"] == 1
    assert counts["etymon_descent_orphans"] == 1
    assert counts["etymon_descent_orphan_refs"] == ["old-english:ghost-a → old-english:tun"]


def test_insert_mining_run_rows_no_orphan_path():
    """mining_run has no FK refs — every row inserts unconditionally."""
    from wyrd.generators.kenning.jsonl.build import _insert_mining_run_rows

    conn = _build_fixture_db()
    _seed_source(conn, "skeat")
    counts = _empty_counts()
    rows = [
        {"provider": "anthropic", "model": "claude", "mode": "mine"},
        {"provider": "gemini", "model": "flash", "mode": "review"},
    ]
    _insert_mining_run_rows(conn, "skeat", rows, counts)
    assert counts["mining_run"] == 2


def test_insert_attestation_rows_basic(tmp_path: Path):
    """Per wyrd-3ypp: attestation rows from OS Open Names + future
    historical sources resolve their toponym_ref and insert into
    toponym_attestation. Unknown ref → orphan-skip pattern."""
    from wyrd.generators.kenning.jsonl.build import _insert_attestation_rows

    # Extended fixture with the toponym_attestation table.
    conn = _build_fixture_db()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS toponym_attestation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            toponym_id INTEGER NOT NULL,
            form TEXT NOT NULL,
            date_year INTEGER,
            source_doc TEXT
        )"""
    )
    cotton_id = _seed_toponym(conn, "Cotton", "Norfolk")
    counts = _empty_counts()
    counts["attestation"] = 0
    counts["attestation_orphans"] = 0
    counts["attestation_orphan_refs"] = []
    rows = [
        {
            "toponym_ref": "Cotton@Norfolk",
            "form": "Cotuna",
            "date_year": 1086,
            "source_doc": "Domesday Book",
        },
        {
            "toponym_ref": "Ghost@Nowhere",  # orphan
            "form": "Ghost",
            "date_year": 2024,
        },
    ]
    _insert_attestation_rows(conn, rows, {"Cotton@Norfolk": cotton_id}, counts)
    assert counts["attestation"] == 1
    assert counts["attestation_orphans"] == 1
    assert counts["attestation_orphan_refs"] == ["Ghost@Nowhere"]
    # DB has the one attestation
    row = conn.execute(
        "SELECT toponym_id, form, date_year, source_doc FROM toponym_attestation"
    ).fetchone()
    assert (row["form"], row["date_year"]) == ("Cotuna", 1086)


def test_insert_etymology_element_rows_per_source_dedup_and_orphan():
    """Helper handles BOTH dedup (wyrd-tzf2) and toponym-orphan skip
    (wyrd-lene) in one place."""
    from wyrd.generators.kenning.jsonl.build import _insert_etymology_element_rows

    conn = _build_fixture_db()
    _seed_source(conn, "mawer")
    cot_id = _seed_etymon(conn, "old-english", "cot")
    cotton_id = _seed_toponym(conn, "Cotton", "Norfolk")
    counts = _empty_counts()
    element = {"ordinal": 1, "etymon_ref": "old-english:cot"}
    rows = [
        {
            "toponym_ref": "Cotton@Norfolk",
            "elements": [element],
            "historical_form": "Cotuna",
            "confidence": "high",
        },
        # Byte-identical dupe
        {
            "toponym_ref": "Cotton@Norfolk",
            "elements": [element],
            "historical_form": "Cotuna",
            "confidence": "high",
        },
        # Orphan: unknown toponym
        {
            "toponym_ref": "Ghost@-",
            "elements": [element],
        },
    ]
    _insert_etymology_element_rows(
        conn,
        "mawer",
        rows,
        {"Cotton@Norfolk": cotton_id},
        {"old-english:cot": cot_id},
        counts,
    )
    assert counts["etymology_element"] == 1
    assert counts["etymology_element_dupes_skipped"] == 1
    assert counts["etymology_element_orphans"] == 1


def test_format_diff_rebuild_renders_missing_and_none_delta():
    """When a table is missing on one side, the count column shows
    '(missing)' and the delta column shows '—' so neither can be
    mistaken for zero."""
    from wyrd.generators.kenning.jsonl.build import format_diff_rebuild

    rows = [{"table": "new_table", "before": -1, "after": 5, "delta": None}]
    md = format_diff_rebuild(rows)
    assert "(missing)" in md
    assert "—" in md


def test_build_dedupes_content_equal_etymology_element_rows(tmp_path: Path):
    """Per wyrd-tzf2: a content-equal etymology_element row in the same
    file inserts ONCE. Re-run LLM mining used to leave byte-identical
    dupes (toponym_etymology has no UNIQUE constraint); the build now
    fingerprints + dedupes per source."""
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
                    {"ordinal": 2, "etymon_ref": "old-english:tun"},
                ],
            },
            # Byte-identical duplicate row.
            {
                "_type": "etymology_element",
                "toponym_ref": "Cotton@Norfolk",
                "historical_form": "Cotuna",
                "confidence": "high",
                "elements": [
                    {"ordinal": 1, "etymon_ref": "old-english:cot"},
                    {"ordinal": 2, "etymon_ref": "old-english:tun"},
                ],
            },
        ],
    )
    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["etymology_element"] == 1
    assert counts["etymology_element_dupes_skipped"] == 1
    n_te = conn.execute("SELECT COUNT(*) FROM toponym_etymology").fetchone()[0]
    assert n_te == 1
    n_el = conn.execute("SELECT COUNT(*) FROM toponym_etymology_element").fetchone()[0]
    assert n_el == 2  # the two element children of the one etymology


def test_build_dedup_scope_is_per_source_file(tmp_path: Path):
    """Two source files with content-equal etymology_element rows
    each insert their row — that's scholarly agreement, not a dupe."""
    cot_row = {
        "_type": "etymon",
        "ref": "old-english:cot",
        "language": "old-english",
        "canonical_form": "cot",
    }
    topo_row = {
        "_type": "toponym",
        "ref": "Cotton@Norfolk",
        "modern_name": "Cotton",
        "country": "England",
        "region": "Norfolk",
    }
    element_row = {
        "_type": "etymology_element",
        "toponym_ref": "Cotton@Norfolk",
        "historical_form": "Cotuna",
        "confidence": "high",
        "elements": [{"ordinal": 1, "etymon_ref": "old-english:cot"}],
    }
    _write_jsonl(
        tmp_path,
        "skeat",
        [{"_type": "source", "ref": "skeat", "title": "Skeat"}, cot_row, topo_row, element_row],
    )
    _write_jsonl(
        tmp_path,
        "mawer",
        [{"_type": "source", "ref": "mawer", "title": "Mawer"}, cot_row, topo_row, element_row],
    )
    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    # Both sources contribute their row — no cross-source dedup.
    assert counts["etymology_element"] == 2
    assert counts["etymology_element_dupes_skipped"] == 0


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


def test_build_orphan_citation_skipped_not_raised(tmp_path: Path):
    """Per wyrd-lene: a citation referencing a removed etymon is
    counted as an orphan and skipped, not raised. Lets ``remove``
    events on etymons be applied without manually cleaning every
    referencing fact-row first."""
    _write_jsonl(
        tmp_path,
        "x",
        [
            {"_type": "source", "ref": "x", "title": "X"},
            {"_type": "citation", "etymon_ref": "old-english:ghost"},
        ],
    )
    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["citation"] == 0
    assert counts["citation_orphans"] == 1


def test_build_orphan_descent_skipped_not_raised(tmp_path: Path):
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
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["etymon_descent"] == 0
    assert counts["etymon_descent_orphans"] == 1


def test_build_orphan_etymology_element_skipped_not_raised(tmp_path: Path):
    """An etymology_element referencing a REMOVED toponym is skipped
    + counted. (Element-ref to a missing etymon is still a BuildError
    because that path runs inside _insert_etymology_element; coarse
    orphan check at the toponym level only.)"""
    _write_jsonl(
        tmp_path,
        "x",
        [
            {"_type": "source", "ref": "x", "title": "X"},
            {
                "_type": "etymology_element",
                "toponym_ref": "Ghost@-",
                "elements": [],
            },
        ],
    )
    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["etymology_element"] == 0
    assert counts["etymology_element_orphans"] == 1


def test_build_etymology_element_with_missing_etymon_ref_still_raises(tmp_path: Path):
    """Coverage for the still-raising path: an etymology_element with
    a known toponym but an element pointing at a missing etymon. The
    toponym-level orphan check above doesn't catch this — the build
    proceeds to _insert_etymology_element which raises BuildError.
    Deliberate inversion of the orphan-skip change."""
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


def test_build_orphan_refs_captured_as_sample(tmp_path: Path):
    """Orphan-skip records the specific orphaned refs (not just counts)
    so operators can spot LLM typos vs expected post-remove orphans
    without diffing the JSONL."""
    _write_jsonl(
        tmp_path,
        "x",
        [
            {"_type": "source", "ref": "x", "title": "X"},
            {"_type": "citation", "etymon_ref": "old-english:typo1"},
            {"_type": "citation", "etymon_ref": "old-english:typo2"},
        ],
    )
    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["citation_orphans"] == 2
    assert "old-english:typo1" in counts["citation_orphan_refs"]
    assert "old-english:typo2" in counts["citation_orphan_refs"]


def test_build_orphan_refs_capped_at_sample_limit(tmp_path: Path):
    """Orphan-ref lists cap at ORPHAN_SAMPLE_LIMIT (50) so a bulk
    prune with hundreds of orphans doesn't bloat the result dict.
    Count remains accurate; only the ref-sample is bounded."""
    from wyrd.generators.kenning.jsonl.build import ORPHAN_SAMPLE_LIMIT

    rows: list[dict] = [{"_type": "source", "ref": "x", "title": "X"}]
    n_orphans = ORPHAN_SAMPLE_LIMIT + 10
    for i in range(n_orphans):
        rows.append({"_type": "citation", "etymon_ref": f"old-english:typo{i}"})
    _write_jsonl(tmp_path, "x", rows)
    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["citation_orphans"] == n_orphans
    # Sample size capped; count unaffected.
    assert len(counts["citation_orphan_refs"]) == ORPHAN_SAMPLE_LIMIT


def test_build_remove_event_drops_etymon_and_skips_citation_orphan(tmp_path: Path):
    """End-to-end: add an etymon + citation, then a 'remove' event on
    the etymon. After replay+build the etymon is gone and the citation
    is counted as an orphan."""
    _write_jsonl(
        tmp_path,
        "x",
        [
            {"_type": "source", "ref": "x", "title": "X"},
            {
                "_type": "etymon",
                "ref": "old-english:ghost",
                "language": "old-english",
                "canonical_form": "ghost",
            },
            {"_type": "citation", "etymon_ref": "old-english:ghost"},
            {"_op": "remove", "_type": "etymon", "ref": "old-english:ghost"},
        ],
    )
    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    # Etymon removed at replay; citation orphaned at insert.
    assert counts["etymon"] == 0
    assert counts["citation_orphans"] == 1


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
    # wyrd-2n1: seed per-element confidence to match the parent's
    # 'high' so the dump→rebuild→dump round-trip is byte-identical.
    # Without this seed, the post-build backfill hook would inherit
    # the parent's value during rebuild and the second dump would
    # surface a confidence key the first dump didn't (breaks the
    # round-trip equality).
    conn.execute(
        "INSERT INTO toponym_etymology_element "
        "(toponym_etymology_id, ordinal, etymon_id, confidence) VALUES (?, 1, ?, 'high')",
        (eid, cot_id),
    )
    conn.execute(
        "INSERT INTO toponym_etymology_element "
        "(toponym_etymology_id, ordinal, etymon_id, inflection, confidence) "
        "VALUES (?, 2, ?, 'oblique', 'high')",
        (eid, tun_id),
    )
    conn.execute(
        "INSERT INTO mining_run (source_id, provider, model, mode, parsed_count) "
        "VALUES ('mawer', 'anthropic', 'claude', 'mine', 42)"
    )
    # wyrd-jr37: at least one attestation row so the round-trip test
    # actually exercises the attestation dump+rebuild path. Without it,
    # toponym_attestation enters the DIFF_REBUILD_TABLES set with zero
    # rows on both sides — a regression that silently drops attestations
    # would still produce a 0-vs-0 diff.
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (?, 'Cotuna', 1086, 'mawer')",
        (tid,),
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
        # wyrd-jr37: attestation rows in the snapshot so dump→rebuild→
        # dump pins (form, date_year, source_doc) identity. Without
        # this, a regression dropping all attestations would pass the
        # snapshot equality check.
        "toponym_attestation": sorted(
            (r["modern_name"], r["region"], r["form"], r["date_year"], r["source_doc"])
            for r in conn.execute("""
                SELECT t.modern_name, t.region, ta.form, ta.date_year, ta.source_doc
                FROM toponym_attestation ta JOIN toponym t ON t.id = ta.toponym_id
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
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# ---------------------------------------------------------------------------
# fantasy_morpheme build (wyrd-2thc)
# ---------------------------------------------------------------------------


def test_build_inserts_fantasy_morpheme_row_with_etymon_resolved(tmp_path: Path):
    """A standard usable=1 row with etymon_ref resolving to a known
    etymon inserts with the resolved etymon_id."""
    _write_jsonl(
        tmp_path,
        "skeat",
        [
            {"_type": "source", "ref": "skeat", "title": "Place-Names of Cambs"},
            {
                "_type": "etymon",
                "ref": "old-english:engel",
                "canonical_form": "engel",
                "language": "old-english",
            },
        ],
    )
    _write_jsonl(
        tmp_path,
        "_fantasy_morphemes",
        [
            {
                "_type": "source",
                "ref": "fantasy-mining",
                "title": "Fantasy morpheme mining",
            },
            {
                "_type": "fantasy_morpheme",
                "input_name": "Angel",
                "input_description": "celestial host",
                "usable": 1,
                "resolution_method": "descent_lookup",
                "approach_version": "fantasy-v1",
                "confidence": "high",
                "citation": "wiktionary",
                "reasoning": "matched OE engel",
                "processed_at": "2026-05-15T00:00:00+00:00",
                "etymon_ref": "old-english:engel",
            },
        ],
    )

    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["fantasy_morpheme"] == 1
    assert counts["fantasy_morpheme_orphans"] == 0

    row = conn.execute(
        "SELECT fm.input_name, fm.input_description, fm.usable, "
        "       fm.resolution_method, fm.approach_version, fm.confidence, "
        "       e.canonical_form, e.language "
        "  FROM fantasy_morpheme fm JOIN etymon e ON e.id = fm.etymon_id"
    ).fetchone()
    assert row["input_name"] == "Angel"
    assert row["input_description"] == "celestial host"
    assert row["canonical_form"] == "engel"
    assert row["language"] == "old-english"


def test_build_inserts_fantasy_morpheme_with_null_etymon_id_when_no_ref(tmp_path: Path):
    """A barred row (usable=0, no etymon_ref field) inserts with
    etymon_id=NULL and is counted as a normal fantasy_morpheme row,
    not an orphan."""
    _write_jsonl(
        tmp_path,
        "_fantasy_morphemes",
        [
            {"_type": "source", "ref": "fantasy-mining", "title": "Fantasy morpheme mining"},
            {
                "_type": "fantasy_morpheme",
                "input_name": "Military",
                "usable": 0,
                "bar_reason": "attested_but_not_in_corpus",
                "resolution_method": "llm_full_research",
                "approach_version": "fantasy-v1",
                "processed_at": "2026-05-15T00:00:00+00:00",
                "unapproved_language": "latin",
                "unapproved_form": "mīlitārius",
            },
        ],
    )

    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["fantasy_morpheme"] == 1
    assert counts["fantasy_morpheme_orphans"] == 0

    row = conn.execute(
        "SELECT input_name, usable, etymon_id, bar_reason, unapproved_language "
        "FROM fantasy_morpheme"
    ).fetchone()
    assert row["input_name"] == "Military"
    assert row["etymon_id"] is None
    assert row["bar_reason"] == "attested_but_not_in_corpus"
    assert row["unapproved_language"] == "latin"


def test_build_orphans_fantasy_morpheme_when_etymon_ref_unknown(tmp_path: Path):
    """A row whose etymon_ref doesn't resolve (the cited etymon was
    removed via a kernel ``remove`` event, or never dumped) is
    counted as an orphan and skipped rather than inserted with
    etymon_id=NULL — silently nulling the FK would lose the operator's
    intent."""
    _write_jsonl(
        tmp_path,
        "_fantasy_morphemes",
        [
            {"_type": "source", "ref": "fantasy-mining", "title": "Fantasy morpheme mining"},
            {
                "_type": "fantasy_morpheme",
                "input_name": "Phoenix",
                "usable": 1,
                "resolution_method": "descent_lookup",
                "approach_version": "fantasy-v1",
                "processed_at": "2026-05-15T00:00:00+00:00",
                "etymon_ref": "ancient-greek:phoinix",  # not dumped anywhere
            },
        ],
    )

    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["fantasy_morpheme"] == 0
    assert counts["fantasy_morpheme_orphans"] == 1
    assert counts["fantasy_morpheme_orphan_refs"] == ["ancient-greek:phoinix"]

    nrows = conn.execute("SELECT COUNT(*) FROM fantasy_morpheme").fetchone()[0]
    assert nrows == 0


def test_build_fantasy_morpheme_uses_insert_or_ignore_on_duplicate_key(tmp_path: Path):
    """The (input_name, approach_version) UNIQUE means a re-replay
    against an already-populated DB shouldn't fail. INSERT OR IGNORE
    means the existing row wins; the duplicate is silently skipped
    (idempotent re-replay). Counter increments only on actual inserts —
    see test_build_fantasy_morpheme_counter_reflects_inserts_only."""
    _write_jsonl(
        tmp_path,
        "skeat",
        [
            {"_type": "source", "ref": "skeat", "title": "X"},
            {
                "_type": "etymon",
                "ref": "old-english:engel",
                "canonical_form": "engel",
                "language": "old-english",
            },
        ],
    )
    _write_jsonl(
        tmp_path,
        "_fantasy_morphemes",
        [
            {"_type": "source", "ref": "fantasy-mining", "title": "Fantasy mining"},
            {
                "_type": "fantasy_morpheme",
                "input_name": "Angel",
                "usable": 1,
                "resolution_method": "descent_lookup",
                "approach_version": "fantasy-v1",
                "processed_at": "2026-05-15T00:00:00+00:00",
                "etymon_ref": "old-english:engel",
            },
        ],
    )

    conn = _build_fixture_db()
    # Pre-populate the same row.
    conn.execute(
        "INSERT INTO fantasy_morpheme "
        "(input_name, usable, resolution_method, approach_version, processed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("Angel", 1, "descent_lookup", "fantasy-v1", "2026-05-01T00:00:00+00:00"),
    )
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    # No IntegrityError; INSERT OR IGNORE skipped the duplicate.
    nrows = conn.execute("SELECT COUNT(*) FROM fantasy_morpheme").fetchone()[0]
    assert nrows == 1
    # The pre-existing processed_at is preserved (INSERT OR IGNORE).
    row = conn.execute(
        "SELECT processed_at FROM fantasy_morpheme WHERE input_name='Angel'"
    ).fetchone()
    assert row["processed_at"] == "2026-05-01T00:00:00+00:00"
    # Counter does NOT increment — the IGNORE'd row was not actually
    # inserted, and counting it would inflate operator-facing telemetry.
    assert counts["fantasy_morpheme"] == 0


def test_fantasy_morpheme_round_trips_through_dump_and_build(tmp_path: Path):
    """End-to-end: seed DB → dump → build into fresh DB → assert
    fantasy_morpheme rows round-trip with etymon attribution
    preserved. Load-bearing test for wyrd-2thc."""
    from wyrd.generators.kenning.jsonl.dump import (
        dump_fantasy_morphemes_to_file,
        dump_source_to_file,
    )

    pre = _build_fixture_db()
    pre.execute("INSERT INTO source (id, title) VALUES ('skeat', 'Skeat')")
    cur = pre.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('engel', 'old-english')"
    )
    eid_engel = cur.lastrowid
    pre.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, page) VALUES (?, 'skeat', '12')",
        (eid_engel,),
    )
    # Fully populate every nullable column so the round-trip exercises
    # the entire scalar surface. Dropping any column from the single-sourced
    # _FANTASY_MORPHEME_COLUMNS (jsonl/_l2_columns — aliased as the dump SCALAR
    # and build INSERT lists) makes this test fail.
    pre.execute(
        "INSERT INTO fantasy_morpheme "
        "(input_name, input_description, usable, etymon_id, "
        " bar_reason, resolution_method, approach_version, "
        " confidence, citation, reasoning, "
        " unapproved_language, unapproved_form, processed_at) "
        "VALUES (?, ?, 1, ?, NULL, 'descent_lookup', 'fantasy-v1', "
        " 'high', 'wiktionary', 'matched OE engel via direct', "
        " NULL, NULL, '2026-05-15T00:00:00+00:00')",
        ("Angel", "celestial host of divine messengers", eid_engel),
    )
    pre.execute(
        "INSERT INTO fantasy_morpheme "
        "(input_name, input_description, usable, etymon_id, "
        " bar_reason, resolution_method, approach_version, "
        " confidence, citation, reasoning, "
        " unapproved_language, unapproved_form, processed_at) "
        "VALUES ('Military', 'soldiers and warriors', 0, NULL, "
        "        'attested_but_not_in_corpus', 'llm_full_research', "
        "        'fantasy-v1', 'high', 'Etymonline', "
        "        'attested in latin as mīlitārius', "
        "        'latin', 'mīlitārius', '2026-05-15T00:00:00+00:00')",
    )
    pre.commit()

    # Dump both the etymon-owning source AND the fantasy_morpheme file.
    dump_source_to_file(pre, "skeat", tmp_path)
    dump_fantasy_morphemes_to_file(pre, tmp_path)
    pre.close()

    rebuilt = _build_fixture_db()
    counts = build_from_jsonl(rebuilt, jsonl_paths_in(tmp_path))
    assert counts["fantasy_morpheme"] == 2
    assert counts["fantasy_morpheme_orphans"] == 0

    rows = rebuilt.execute(
        "SELECT fm.input_name, fm.input_description, fm.usable, "
        "       fm.bar_reason, fm.resolution_method, fm.approach_version, "
        "       fm.confidence, fm.citation, fm.reasoning, "
        "       fm.unapproved_language, fm.unapproved_form, "
        "       fm.processed_at, e.canonical_form "
        "  FROM fantasy_morpheme fm "
        "  LEFT JOIN etymon e ON e.id = fm.etymon_id "
        " ORDER BY fm.input_name"
    ).fetchall()
    angel, military = rows[0], rows[1]
    assert dict(angel) == {
        "input_name": "Angel",
        "input_description": "celestial host of divine messengers",
        "usable": 1,
        "bar_reason": None,
        "resolution_method": "descent_lookup",
        "approach_version": "fantasy-v1",
        "confidence": "high",
        "citation": "wiktionary",
        "reasoning": "matched OE engel via direct",
        "unapproved_language": None,
        "unapproved_form": None,
        "processed_at": "2026-05-15T00:00:00+00:00",
        "canonical_form": "engel",
    }
    assert dict(military) == {
        "input_name": "Military",
        "input_description": "soldiers and warriors",
        "usable": 0,
        "bar_reason": "attested_but_not_in_corpus",
        "resolution_method": "llm_full_research",
        "approach_version": "fantasy-v1",
        "confidence": "high",
        "citation": "Etymonline",
        "reasoning": "attested in latin as mīlitārius",
        "unapproved_language": "latin",
        "unapproved_form": "mīlitārius",
        "processed_at": "2026-05-15T00:00:00+00:00",
        "canonical_form": None,  # NULL etymon_id preserved
    }


def test_build_fantasy_morpheme_raises_on_missing_required_field(tmp_path: Path):
    """Required NOT NULL fields (input_name, usable, resolution_method,
    approach_version, processed_at) are pre-validated by
    _insert_fantasy_morpheme — INSERT OR IGNORE would otherwise swallow
    the IntegrityError and let the caller count a phantom row. A
    missing field surfaces as BuildError so operators get an actionable
    diagnostic, not a silent count discrepancy."""
    _write_jsonl(
        tmp_path,
        "_fantasy_morphemes",
        [
            {"_type": "source", "ref": "fantasy-mining", "title": "Fantasy mining"},
            {
                "_type": "fantasy_morpheme",
                "input_name": "Phoenix",
                # usable missing — NOT NULL violation if we didn't pre-validate
                "resolution_method": "descent_lookup",
                "approach_version": "fantasy-v1",
                "processed_at": "2026-05-15T00:00:00+00:00",
            },
        ],
    )
    conn = _build_fixture_db()
    import pytest

    with pytest.raises(BuildError, match=r"missing required NOT NULL field.*usable"):
        build_from_jsonl(conn, jsonl_paths_in(tmp_path))


def test_build_fantasy_morpheme_counter_reflects_inserts_only(tmp_path: Path):
    """Per silent-failure-hunter on PR #206: when INSERT OR IGNORE
    skips a UNIQUE-conflict duplicate, the counter must NOT increment
    — otherwise a re-replay against a populated DB inflates the
    fantasy_morpheme count by the number of duplicates and the dump
    summary lies to operators.

    Distinct from test_build_fantasy_morpheme_uses_insert_or_ignore_on_duplicate_key
    above (which checks no IntegrityError fires); this test pins the
    counter semantics."""
    _write_jsonl(
        tmp_path,
        "skeat",
        [
            {"_type": "source", "ref": "skeat", "title": "X"},
            {
                "_type": "etymon",
                "ref": "old-english:engel",
                "canonical_form": "engel",
                "language": "old-english",
            },
        ],
    )
    _write_jsonl(
        tmp_path,
        "_fantasy_morphemes",
        [
            {"_type": "source", "ref": "fantasy-mining", "title": "Fantasy mining"},
            {
                "_type": "fantasy_morpheme",
                "input_name": "Angel",
                "usable": 1,
                "resolution_method": "descent_lookup",
                "approach_version": "fantasy-v1",
                "processed_at": "2026-05-15T00:00:00+00:00",
                "etymon_ref": "old-english:engel",
            },
        ],
    )
    conn = _build_fixture_db()
    conn.execute(
        "INSERT INTO fantasy_morpheme "
        "(input_name, usable, resolution_method, approach_version, processed_at) "
        "VALUES ('Angel', 1, 'descent_lookup', 'fantasy-v1', '2026-05-01T00:00:00+00:00')",
    )
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    assert counts["fantasy_morpheme"] == 0, (
        "duplicate UNIQUE conflict was IGNORE'd; counter must not double-count"
    )


# ---------------------------------------------------------------------------
# wyrd-1ed9: regression tests that exercise the REAL init_schema-produced
# schema, not the hand-rolled _build_fixture_db. The wyrd-rrse bug
# (fantasy_morpheme missing from init_schema-produced DBs) shipped under
# 848 green tests precisely because all existing fantasy_morpheme tests
# above use _build_fixture_db, which creates the table directly via
# CREATE TABLE — bypassing init_schema and masking the alembic gap.
# ---------------------------------------------------------------------------


def test_build_fantasy_morpheme_via_init_schema(tmp_path: Path):
    """End-to-end regression for wyrd-rrse: the literal crash scenario
    was ``init_schema(tmp) → build_from_jsonl(_fantasy_morphemes.jsonl)``
    failing with ``no such table: fantasy_morpheme`` because the table
    lived only in the legacy ``migrate_schema`` helper, not in alembic.

    This test exercises the real init_schema path (which runs alembic
    upgrade head to 0011_fantasy_morpheme) and writes one
    fantasy_morpheme row. Crashes immediately if init_schema ever
    stops producing the table — the gap _build_fixture_db masked."""
    from wyrd.generators.kenning.lexicon import init_schema

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)

    jsonl_dir = tmp_path / "mining"
    jsonl_dir.mkdir()
    _write_jsonl(
        jsonl_dir,
        "_fantasy_morphemes",
        [
            {"_type": "source", "ref": "fantasy-mining", "title": "Fantasy mining"},
            {
                "_type": "fantasy_morpheme",
                "input_name": "Angel",
                "usable": 1,
                "resolution_method": "descent_lookup",
                "approach_version": "fantasy-v1",
                "processed_at": "2026-05-15T00:00:00+00:00",
            },
        ],
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        counts = build_from_jsonl(conn, jsonl_paths_in(jsonl_dir))
        assert counts["fantasy_morpheme"] == 1
        row = conn.execute("SELECT input_name FROM fantasy_morpheme").fetchone()
        assert row is not None, "expected one fantasy_morpheme row after build"
        assert row["input_name"] == "Angel"
    finally:
        conn.close()


def test_init_schema_fantasy_morpheme_input_name_is_nocase(tmp_path: Path):
    """Pin COLLATE NOCASE at the column level via PRAGMA, independent
    of the UNIQUE(input_name, approach_version) constraint shape.

    The behavioral dedup test below catches a NOCASE drop only via
    its observable consequence under the CURRENT UNIQUE shape. If a
    future change widens / re-orders / renames the UNIQUE (e.g. adds
    a tertiary column or splits the constraint), the conflict no
    longer fires the same way and a NOCASE regression on the column
    silently slips through the behavioral test. This PRAGMA pin
    catches it regardless of UNIQUE shape."""
    from wyrd.generators.kenning.lexicon import init_schema

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)

    # PRAGMA table_info doesn't expose collation directly — read it
    # from sqlite_master's CREATE TABLE DDL, which preserves the
    # column-level COLLATE clause verbatim.
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='fantasy_morpheme'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "fantasy_morpheme table missing from sqlite_master"
    ddl = row[0]
    assert "input_name" in ddl
    assert "COLLATE NOCASE" in ddl, (
        f"input_name COLLATE NOCASE missing from init_schema DDL:\n{ddl}"
    )


def test_build_fantasy_morpheme_via_init_schema_dedups_case_insensitively(tmp_path: Path):
    """Pin the COLLATE NOCASE invariant added in PR #281 round-1.
    The fantasy_morpheme.input_name column is COLLATE NOCASE so a
    later "harpy" doesn't double-research an earlier "Harpy". The
    UNIQUE(input_name, approach_version) constraint relies on
    this collation for the dedup to fire.

    Uses init_schema (not _build_fixture_db) so a future regression
    that drops the COLLATE NOCASE from the alembic migration / SA
    MetaData fails this test — the fixture DB has its own COLLATE
    NOCASE hardcoded and would mask such a regression."""
    from wyrd.generators.kenning.lexicon import init_schema

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)

    # Pre-populate with title-case "Angel".
    pre = sqlite3.connect(db_path)
    try:
        pre.execute(
            "INSERT INTO fantasy_morpheme "
            "(input_name, usable, resolution_method, approach_version, processed_at) "
            "VALUES ('Angel', 1, 'descent_lookup', 'fantasy-v1', '2026-05-01T00:00:00+00:00')"
        )
        pre.commit()
    finally:
        pre.close()

    # JSONL replay with lowercase "angel" — case-insensitive UNIQUE
    # should treat this as a duplicate and IGNORE the insert,
    # preserving the title-case row.
    jsonl_dir = tmp_path / "mining"
    jsonl_dir.mkdir()
    _write_jsonl(
        jsonl_dir,
        "_fantasy_morphemes",
        [
            {"_type": "source", "ref": "fantasy-mining", "title": "Fantasy mining"},
            {
                "_type": "fantasy_morpheme",
                "input_name": "angel",
                "usable": 1,
                "resolution_method": "descent_lookup",
                "approach_version": "fantasy-v1",
                "processed_at": "2026-05-20T00:00:00+00:00",
            },
        ],
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        counts = build_from_jsonl(conn, jsonl_paths_in(jsonl_dir))
        assert counts["fantasy_morpheme"] == 0  # IGNORE'd
        rows = conn.execute("SELECT input_name, processed_at FROM fantasy_morpheme").fetchall()
        assert len(rows) == 1
        assert rows[0]["input_name"] == "Angel", (
            "INSERT OR IGNORE must preserve the original-case row"
        )
        assert rows[0]["processed_at"] == "2026-05-01T00:00:00+00:00"
    finally:
        conn.close()


def test_build_fantasy_morpheme_via_init_schema_barred_row(tmp_path: Path):
    """wyrd-bk4b: companion to test_build_fantasy_morpheme_via_init_schema.
    The wyrd-rrse "no such table" crash would have hit the BARRED path
    (usable=0) equally, so pin it through the real init_schema schema too.
    A barred row carries usable=0 + bar_reason, has no resolved etymon
    (etymon_id NULL, no etymon_ref — so it is NOT an orphan, per the
    legitimately-no-ref path), and records the LLM's finding in a
    non-approved language via unapproved_language / unapproved_form."""
    from wyrd.generators.kenning.lexicon import init_schema

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)

    jsonl_dir = tmp_path / "mining"
    jsonl_dir.mkdir()
    _write_jsonl(
        jsonl_dir,
        "_fantasy_morphemes",
        [
            {"_type": "source", "ref": "fantasy-mining", "title": "Fantasy mining"},
            {
                "_type": "fantasy_morpheme",
                "input_name": "Vrock",
                "usable": 0,
                "bar_reason": "modern_coinage",
                "resolution_method": "llm_full_research",
                "approach_version": "fantasy-v1",
                "unapproved_language": "japanese",
                "unapproved_form": "vurokku",
                "processed_at": "2026-05-15T00:00:00+00:00",
            },
        ],
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        counts = build_from_jsonl(conn, jsonl_paths_in(jsonl_dir))
        assert counts["fantasy_morpheme"] == 1
        assert counts["fantasy_morpheme_orphans"] == 0
        row = conn.execute(
            "SELECT input_name, usable, bar_reason, etymon_id, resolution_method, "
            "approach_version, unapproved_language, unapproved_form FROM fantasy_morpheme"
        ).fetchone()
        assert row is not None, "expected one barred fantasy_morpheme row after build"
        assert row["input_name"] == "Vrock"
        assert row["usable"] == 0
        assert row["bar_reason"] == "modern_coinage"
        assert row["etymon_id"] is None
        # The required NOT NULL scalars must round-trip on the barred /
        # etymon_id-NULL path too (a distinct code path from usable=1).
        assert row["resolution_method"] == "llm_full_research"
        assert row["approach_version"] == "fantasy-v1"
        assert row["unapproved_language"] == "japanese"
        assert row["unapproved_form"] == "vurokku"
    finally:
        conn.close()


def test_build_fantasy_morpheme_dedup_respects_approach_version(tmp_path: Path):
    """wyrd-bk4b: the UNIQUE is composite (input_name, approach_version),
    NOT input_name alone — migration 0011's re-processing invariant. A
    later pass under a bumped approach_version must NOT dedupe against the
    earlier version's row (re-researching all inputs under a stronger
    pipeline is the intended use). Pre-populate (Angel, fantasy-v1),
    replay (angel, fantasy-v2), and assert TWO rows survive: the input
    matches case-insensitively but the versions differ, so it is not a
    UNIQUE conflict."""
    from wyrd.generators.kenning.lexicon import init_schema

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)

    pre = sqlite3.connect(db_path)
    try:
        pre.execute(
            "INSERT INTO fantasy_morpheme "
            "(input_name, usable, resolution_method, approach_version, processed_at) "
            "VALUES ('Angel', 1, 'descent_lookup', 'fantasy-v1', '2026-05-01T00:00:00+00:00')"
        )
        pre.commit()
    finally:
        pre.close()

    jsonl_dir = tmp_path / "mining"
    jsonl_dir.mkdir()
    _write_jsonl(
        jsonl_dir,
        "_fantasy_morphemes",
        [
            {"_type": "source", "ref": "fantasy-mining", "title": "Fantasy mining"},
            {
                "_type": "fantasy_morpheme",
                "input_name": "angel",
                "usable": 1,
                # Distinct from the pre-populated v1 row so the assertions can
                # confirm the v2 row is the freshly-replayed one, not a mutation.
                "resolution_method": "llm_full_research",
                "approach_version": "fantasy-v2",
                "processed_at": "2026-05-20T00:00:00+00:00",
            },
        ],
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        counts = build_from_jsonl(conn, jsonl_paths_in(jsonl_dir))
        assert counts["fantasy_morpheme"] == 1  # v2 is a NEW row, not a dupe of v1
        rows = conn.execute(
            "SELECT approach_version, resolution_method FROM fantasy_morpheme "
            "ORDER BY approach_version"
        ).fetchall()
        assert [r["approach_version"] for r in rows] == ["fantasy-v1", "fantasy-v2"], (
            "composite UNIQUE must let a bumped approach_version re-process the input"
        )
        # Each surviving row carries its OWN fields — the v2 row is the
        # replayed one, not an in-place edit of the v1 row.
        assert [r["resolution_method"] for r in rows] == [
            "descent_lookup",
            "llm_full_research",
        ]
    finally:
        conn.close()


def test_build_fantasy_morpheme_insert_or_ignore_does_not_replace_columns(tmp_path: Path):
    """wyrd-bk4b: an INSERT OR IGNORE on a (input_name, approach_version)
    conflict must leave the EXISTING row untouched — IGNORE, not REPLACE.
    The sibling case-insensitive test pins processed_at; this pins a
    SECOND column (unapproved_language) so a future switch to
    INSERT OR REPLACE is caught on more than one field. Pre-populate a
    barred row carrying unapproved_language='latin'; replay a same-key
    (case-insensitive) row with unapproved_language='greek'; assert the
    pre-populated value survives."""
    from wyrd.generators.kenning.lexicon import init_schema

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)

    pre = sqlite3.connect(db_path)
    try:
        pre.execute(
            "INSERT INTO fantasy_morpheme "
            "(input_name, usable, bar_reason, resolution_method, approach_version, "
            "unapproved_language, processed_at) "
            "VALUES ('Sphinx', 0, 'outside_language_family', 'llm_full_research', "
            "'fantasy-v1', 'latin', '2026-05-01T00:00:00+00:00')"
        )
        pre.commit()
    finally:
        pre.close()

    jsonl_dir = tmp_path / "mining"
    jsonl_dir.mkdir()
    _write_jsonl(
        jsonl_dir,
        "_fantasy_morphemes",
        [
            {"_type": "source", "ref": "fantasy-mining", "title": "Fantasy mining"},
            {
                "_type": "fantasy_morpheme",
                "input_name": "sphinx",
                "usable": 0,
                "bar_reason": "outside_language_family",
                "resolution_method": "llm_full_research",
                "approach_version": "fantasy-v1",
                "unapproved_language": "greek",
                "processed_at": "2026-05-20T00:00:00+00:00",
            },
        ],
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        counts = build_from_jsonl(conn, jsonl_paths_in(jsonl_dir))
        assert counts["fantasy_morpheme"] == 0  # IGNORE'd, not replaced
        rows = conn.execute(
            "SELECT unapproved_language, processed_at FROM fantasy_morpheme"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["unapproved_language"] == "latin", (
            "INSERT OR IGNORE must not overwrite the existing row's columns"
        )
        assert rows[0]["processed_at"] == "2026-05-01T00:00:00+00:00"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# wyrd-ned5: seed reflex layer L2 round-trip
# ---------------------------------------------------------------------------


def _init_real_schema(tmp_path: Path) -> Path:
    """A real L3 schema (not the minimal fixture) so reflex/reflex_etymon
    exist for the reflex round-trip tests."""
    from wyrd.generators.kenning.lexicon import init_schema

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def test_build_replays_reflex_rows_links_modern_usage_to_etymon(tmp_path: Path):
    """wyrd-ned5: a ``_reflexes.jsonl`` reflex row restores the
    modern_usage→etymon link (-ton → OE tūn) on rebuild — the mapping a
    full rebuild-from-jsonl used to drop."""
    db_path = _init_real_schema(tmp_path)
    _write_jsonl(
        tmp_path,
        "rando-port",
        [
            {"_type": "source", "ref": "rando-port", "title": "Rando"},
            {
                "_type": "etymon",
                "ref": "old-english:tūn",
                "language": "old-english",
                "canonical_form": "tūn",
                "glosses": ["estate"],
            },
        ],
    )
    write_jsonl(
        tmp_path / "_reflexes.jsonl",
        [
            {"_type": "source", "ref": "reflex-seed", "title": "Seed reflex"},
            {
                "_type": "reflex",
                "surface_form": "-ton",
                "position": "post",
                "etymon_refs": ["old-english:tūn"],
            },
        ],
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
        assert counts["reflex"] == 1
        assert counts["reflex_link_orphans"] == 0
        row = conn.execute(
            """
            SELECT r.surface_form, r.position, e.canonical_form, e.language
            FROM reflex r
            JOIN reflex_etymon re ON re.reflex_id = r.id
            JOIN etymon e ON e.id = re.etymon_id
            WHERE r.surface_form = '-ton'
            """
        ).fetchone()
        assert (row["surface_form"], row["position"], row["canonical_form"], row["language"]) == (
            "-ton",
            "post",
            "tūn",
            "old-english",
        )
    finally:
        conn.close()


def test_build_reflex_unresolved_ref_is_orphan_but_reflex_upserts(tmp_path: Path):
    """An etymon_ref that doesn't resolve is counted as a link orphan +
    skipped; the reflex row itself still upserts (empty link set is a
    valid intermediate state)."""
    db_path = _init_real_schema(tmp_path)
    _write_jsonl(
        tmp_path,
        "rando-port",
        [{"_type": "source", "ref": "rando-port", "title": "Rando"}],
    )
    write_jsonl(
        tmp_path / "_reflexes.jsonl",
        [
            {"_type": "source", "ref": "reflex-seed", "title": "Seed reflex"},
            {
                "_type": "reflex",
                "surface_form": "-ham",
                "position": "post",
                "etymon_refs": ["old-english:hām"],  # etymon never declared
            },
        ],
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))
        assert counts["reflex"] == 1
        assert counts["reflex_link_orphans"] == 1
        assert "old-english:hām" in counts["reflex_link_orphan_refs"]
        # Reflex upserted; no links.
        assert (
            conn.execute("SELECT COUNT(*) FROM reflex WHERE surface_form='-ham'").fetchone()[0] == 1
        )
        assert conn.execute("SELECT COUNT(*) FROM reflex_etymon").fetchone()[0] == 0
    finally:
        conn.close()


def test_dump_reflexes_round_trips_through_rebuild(tmp_path: Path):
    """Seed a DB with a reflex, dump ``_reflexes.jsonl``, rebuild a fresh
    DB from the dumped L2, and confirm the reflex link survives."""
    from wyrd.generators.kenning.jsonl.dump import dump_reflexes_to_file
    from wyrd.generators.kenning.lexicon import init_schema
    from wyrd.generators.kenning.lexicon.sql.queries.reflex import (
        LINK_REFLEX_ETYMON_OR_IGNORE,
        UPSERT_REFLEX,
    )

    # --- source DB: one etymon + one linked reflex.
    src_db = tmp_path / "src.db"
    init_schema(src_db)
    sconn = sqlite3.connect(src_db)
    sconn.row_factory = sqlite3.Row
    eid = sconn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('tūn','old-english') RETURNING id"
    ).fetchone()[0]
    rid = sconn.execute(UPSERT_REFLEX, ("-ton", "post")).fetchone()[0]
    sconn.execute(LINK_REFLEX_ETYMON_OR_IGNORE, (rid, eid))
    sconn.commit()

    # --- dump reflexes; also write a rando-port file so the etymon exists on rebuild.
    out_dir = tmp_path / "l2"
    out_dir.mkdir()
    _, n = dump_reflexes_to_file(sconn, out_dir)
    sconn.close()
    assert n == 2  # synthetic source row + one reflex row
    _write_jsonl(
        out_dir,
        "rando-port",
        [
            {"_type": "source", "ref": "rando-port", "title": "Rando"},
            {
                "_type": "etymon",
                "ref": "old-english:tūn",
                "language": "old-english",
                "canonical_form": "tūn",
            },
        ],
    )

    # --- rebuild fresh DB from the dumped L2.
    dst_db = tmp_path / "dst.db"
    init_schema(dst_db)
    dconn = sqlite3.connect(dst_db)
    dconn.row_factory = sqlite3.Row
    try:
        counts = build_from_jsonl(dconn, jsonl_paths_in(out_dir))
        assert counts["reflex"] == 1
        row = dconn.execute(
            """
            SELECT e.canonical_form, e.language
            FROM reflex r
            JOIN reflex_etymon re ON re.reflex_id = r.id
            JOIN etymon e ON e.id = re.etymon_id
            WHERE r.surface_form = '-ton'
            """
        ).fetchone()
        assert (row["canonical_form"], row["language"]) == ("tūn", "old-english")
    finally:
        dconn.close()


def test_dump_reflexes_round_trips_orphan_reflex(tmp_path: Path):
    """wyrd-br5o: an ORPHAN reflex (no etymon link — the generation-surface
    layer) must round-trip too. wyrd-ned5's INNER-JOIN dump silently dropped
    these, so a clean rebuild lost ~4k of the ~16k orphan surfaces. The dump
    now LEFT-JOINs and emits them with ``etymon_refs: []``; the replay upserts
    the bare reflex."""
    from wyrd.generators.kenning.jsonl.dump import (
        dump_reflexes_to_file,
        dump_reflexes_to_rows,
    )
    from wyrd.generators.kenning.lexicon import init_schema
    from wyrd.generators.kenning.lexicon.sql.queries.reflex import UPSERT_REFLEX

    # --- source DB: a bare reflex with NO reflex_etymon link.
    src_db = tmp_path / "src.db"
    init_schema(src_db)
    sconn = sqlite3.connect(src_db)
    sconn.row_factory = sqlite3.Row
    sconn.execute(UPSERT_REFLEX, ("worth", "post")).fetchone()
    sconn.commit()

    # the dump emits the orphan with an empty etymon_refs list.
    rows = dump_reflexes_to_rows(sconn)
    reflex_rows = [r for r in rows if r["_type"] == "reflex"]
    assert reflex_rows == [
        {"_type": "reflex", "surface_form": "worth", "position": "post", "etymon_refs": []}
    ]

    out_dir = tmp_path / "l2"
    out_dir.mkdir()
    _, n = dump_reflexes_to_file(sconn, out_dir)
    sconn.close()
    assert n == 2  # synthetic source row + the orphan reflex row

    # --- rebuild fresh DB: the orphan reflex survives (no link, but present).
    dst_db = tmp_path / "dst.db"
    init_schema(dst_db)
    dconn = sqlite3.connect(dst_db)
    dconn.row_factory = sqlite3.Row
    try:
        counts = build_from_jsonl(dconn, jsonl_paths_in(out_dir))
        assert counts["reflex"] == 1
        # An empty etymon_refs list resolves zero refs — no spurious orphan links.
        assert counts["reflex_link_orphans"] == 0
        present = dconn.execute(
            """
            SELECT r.surface_form, r.position,
                   (SELECT count(*) FROM reflex_etymon re WHERE re.reflex_id = r.id) AS links
            FROM reflex r WHERE r.surface_form = 'worth'
            """
        ).fetchone()
        assert (present["surface_form"], present["position"], present["links"]) == (
            "worth",
            "post",
            0,
        )
    finally:
        dconn.close()


def test_dump_reflexes_merged_loser_only_link_emits_orphan(tmp_path: Path):
    """wyrd-br5o: a reflex whose ONLY link is to a merged-loser etymon
    (merged_into_id NOT NULL) is a structurally different path from a bare
    orphan — the reflex_etymon row exists, but the LEFT JOIN's
    ``AND e.merged_into_id IS NULL`` predicate fails, yielding NULL etymon
    columns. It must still emit (as ``etymon_refs: []``), NOT be dropped.
    This guards the merged_into_id filter staying on the JOIN: a WHERE on the
    nullable side would collapse the LEFT JOIN back to an inner one and drop
    this reflex (the wyrd-ned5 bug)."""
    from wyrd.generators.kenning.jsonl.dump import dump_reflexes_to_rows
    from wyrd.generators.kenning.lexicon import init_schema
    from wyrd.generators.kenning.lexicon.sql.queries.reflex import (
        LINK_REFLEX_ETYMON_OR_IGNORE,
        UPSERT_REFLEX,
    )

    src_db = tmp_path / "src.db"
    init_schema(src_db)
    sconn = sqlite3.connect(src_db)
    sconn.row_factory = sqlite3.Row
    # winner + loser (loser merged INTO winner), reflex links only to the loser.
    winner = sconn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('tūn','old-english') RETURNING id"
    ).fetchone()[0]
    loser = sconn.execute(
        "INSERT INTO etymon (canonical_form, language, merged_into_id) "
        "VALUES ('tun','old-english',?) RETURNING id",
        (winner,),
    ).fetchone()[0]
    rid = sconn.execute(UPSERT_REFLEX, ("-ton", "post")).fetchone()[0]
    sconn.execute(LINK_REFLEX_ETYMON_OR_IGNORE, (rid, loser))
    sconn.commit()

    reflex_rows = [r for r in dump_reflexes_to_rows(sconn) if r["_type"] == "reflex"]
    sconn.close()
    # Emitted as an orphan (loser excluded by the JOIN predicate), not dropped.
    assert reflex_rows == [
        {"_type": "reflex", "surface_form": "-ton", "position": "post", "etymon_refs": []}
    ]


def test_dump_reflexes_mixed_live_and_merged_loser_keeps_live_ref(tmp_path: Path):
    """wyrd-br5o: a reflex linked to BOTH a live etymon and a merged loser
    must dump with ``etymon_refs`` holding ONLY the live ref — the per-row
    NULL-skip in the grouper drops the loser's NULL row without dropping the
    whole reflex or the live ref."""
    from wyrd.generators.kenning.jsonl.dump import dump_reflexes_to_rows
    from wyrd.generators.kenning.lexicon import init_schema
    from wyrd.generators.kenning.lexicon.sql.queries.reflex import (
        LINK_REFLEX_ETYMON_OR_IGNORE,
        UPSERT_REFLEX,
    )

    src_db = tmp_path / "src.db"
    init_schema(src_db)
    sconn = sqlite3.connect(src_db)
    sconn.row_factory = sqlite3.Row
    live = sconn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('hām','old-english') RETURNING id"
    ).fetchone()[0]
    loser = sconn.execute(
        "INSERT INTO etymon (canonical_form, language, merged_into_id) "
        "VALUES ('ham','old-english',?) RETURNING id",
        (live,),
    ).fetchone()[0]
    rid = sconn.execute(UPSERT_REFLEX, ("-ham", "post")).fetchone()[0]
    sconn.execute(LINK_REFLEX_ETYMON_OR_IGNORE, (rid, live))
    sconn.execute(LINK_REFLEX_ETYMON_OR_IGNORE, (rid, loser))
    sconn.commit()

    reflex_rows = [r for r in dump_reflexes_to_rows(sconn) if r["_type"] == "reflex"]
    sconn.close()
    assert reflex_rows == [
        {
            "_type": "reflex",
            "surface_form": "-ham",
            "position": "post",
            "etymon_refs": ["old-english:hām"],
        }
    ]


# ---------------------------------------------------------------------------
# wyrd-5qg7: audit verdict logs must be excluded from the replay set
# ---------------------------------------------------------------------------


def test_jsonl_paths_in_excludes_replay_excluded_ledgers(tmp_path: Path):
    """The replay-excluded ledgers are excluded from the replay set; conforming files
    — curated sources, _collapses, and _tags/_merge_audit (which replay inert) —
    are kept."""
    from wyrd.generators.kenning.jsonl.build import REPLAY_EXCLUDED_LEDGERS

    kept = {"skeat.jsonl", "_tags.jsonl", "_merge_audit.jsonl", "_collapses.jsonl"}
    for name in kept | set(REPLAY_EXCLUDED_LEDGERS):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert {p.name for p in jsonl_paths_in(tmp_path)} == kept


def test_no_unhandled_nonconforming_ledgers():
    """Drift guard, both directions (wyrd-5qg7): every real data/mining/*.jsonl is
    EITHER replay-conforming (all rows carry a _type in ALL_TYPES) OR listed in
    REPLAY_EXCLUDED_LEDGERS; AND every listed name still exists and is still
    non-conforming. A new non-conforming ledger added without listing it — or a
    stale/over-broad exclusion that became conforming or vanished — fails THIS test
    in CI instead of silently breaking (or silently over-trimming) the rebuild."""
    from wyrd.generators.kenning.jsonl.build import REPLAY_EXCLUDED_LEDGERS
    from wyrd.generators.kenning.jsonl.log import ALL_TYPES

    mining = Path(__file__).resolve().parents[1] / "data" / "mining"

    def first_nonconforming(path: Path) -> str | None:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if json.loads(line).get("_type") not in ALL_TYPES:
                return repr(json.loads(line).get("_type"))
        return None

    # forward: a non-conforming file must be listed
    unlisted = [
        f"{f.name}: _type={t}"
        for f in sorted(mining.glob("*.jsonl"))
        if f.name not in REPLAY_EXCLUDED_LEDGERS and (t := first_nonconforming(f)) is not None
    ]
    assert not unlisted, (
        f"Non-conforming data/mining ledgers not in REPLAY_EXCLUDED_LEDGERS "
        f"(add them there or fix their schema): {unlisted}"
    )
    # inverse: every listed name must still exist AND still be non-conforming
    stale = [
        name
        for name in sorted(REPLAY_EXCLUDED_LEDGERS)
        if not (mining / name).is_file() or first_nonconforming(mining / name) is None
    ]
    assert not stale, (
        f"Stale REPLAY_EXCLUDED_LEDGERS entries (missing, or now replay-conforming so "
        f"they'd be wrongly dropped from replay — remove them): {stale}"
    )


def test_build_skips_replay_excluded_ledgers(tmp_path: Path):
    """Regression: a directory containing both failure modes rebuilds cleanly —
    the type-less ``_reflex_audit.jsonl`` (bare rows, no ``_type``) AND the
    bespoke-``_type`` ``_pronunciation.jsonl`` (``_type: pronunciation`` not in
    ALL_TYPES). Either, swept into the generic replay, used to raise ReplayError
    (from-scratch rebuild broken)."""
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
        ],
    )
    # bare audit rows with NO _type — would raise "row missing '_type'" if replayed
    (tmp_path / "_reflex_audit.jsonl").write_text(
        json.dumps({"surface": "-ock", "canon": "aecern", "valid": False}) + "\n",
        encoding="utf-8",
    )
    # bespoke _type not in ALL_TYPES — would raise "unknown _type 'pronunciation'"
    (tmp_path / "_pronunciation.jsonl").write_text(
        json.dumps({"_type": "source", "ref": "pron", "title": "P"})
        + "\n"
        + json.dumps({"_type": "pronunciation", "ref": "old-english:cot", "ipa": "kɒt"})
        + "\n",
        encoding="utf-8",
    )
    conn = _build_fixture_db()
    counts = build_from_jsonl(conn, jsonl_paths_in(tmp_path))  # must not raise
    assert counts["source"] == 1


def test_l2_round_trip_columns_are_single_sourced():
    """Reconstruction-fidelity guard (D21): build's insert columns are DERIVED from
    the dumped L2 columns (single-sourced in ``jsonl/_l2_columns``), so a dump column
    can never drift out of the inserter and get silently dropped on
    ``rebuild-from-jsonl``. Pin both the round-trip subset invariant (every dumped
    column is insertable) and the exact derivation, so a future re-hardcoding of
    build's lists — the drift this single-sourcing exists to prevent — is caught.
    """
    # Import the dumped-column lists from the single source directly (not via dump's
    # re-export) — the invariant under test is that build derives from _l2_columns.
    from wyrd.generators.kenning.jsonl._l2_columns import _ETYMON_L2_COLUMNS, _SOURCE_COLUMNS
    from wyrd.generators.kenning.jsonl.build import (
        _ETYMON_INSERT_COLUMNS,
        _SOURCE_INSERT_COLUMNS,
    )

    # Every dumped etymon column is insertable → no round-trip data loss.
    assert set(_ETYMON_L2_COLUMNS) <= set(_ETYMON_INSERT_COLUMNS)
    assert _ETYMON_INSERT_COLUMNS == _ETYMON_L2_COLUMNS  # single-sourced, verbatim
    # Source insert = the synthetic row id + the dumped source columns.
    assert set(_SOURCE_COLUMNS) <= set(_SOURCE_INSERT_COLUMNS)
    assert ("id", *_SOURCE_COLUMNS) == _SOURCE_INSERT_COLUMNS
