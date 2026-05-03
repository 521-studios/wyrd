"""Tests for the wiktextract ingester (wyrd-hun / wyrd-4rt.1).

Synthetic JSONL fixtures only — these exercise the parser logic
without needing the multi-GB kaikki.org dump. Real-corpus end-to-end
verification happens out-of-band when the dump lands.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as kenning_cli
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.wiktextract_ingester import (
    _canonical_language,
    ingest_wiktextract_path,
    ingest_wiktextract_stream,
)


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def _wiktextract_entry(
    *,
    word: str,
    lang_code: str,
    etymology_templates: list[dict] | None = None,
    descendants: list[dict] | None = None,
) -> str:
    """Build one JSONL line matching the wiktextract entry shape:
    word + lang_code are required, etymology_templates and descendants
    are the two sections we actually read."""
    record = {"word": word, "lang_code": lang_code, "pos": "noun"}
    if etymology_templates is not None:
        record["etymology_templates"] = etymology_templates
    if descendants is not None:
        record["descendants"] = descendants
    return json.dumps(record) + "\n"


def _stream(*lines: str) -> io.StringIO:
    return io.StringIO("".join(lines))


def _all_descent_edges(db: LexiconDB) -> list[tuple[str, str, str, str]]:
    """Return (parent_form, child_form, edge_type, source_id) tuples
    for every row in etymon_descent. Sorted for stable assertions."""
    rows = db.conn.execute(
        """
        SELECT p.canonical_form AS parent, c.canonical_form AS child,
               d.edge_type, d.source_id
        FROM etymon_descent d
        JOIN etymon p ON p.id = d.parent_id
        JOIN etymon c ON c.id = d.child_id
        ORDER BY parent, child, edge_type
        """
    ).fetchall()
    return [(r["parent"], r["child"], r["edge_type"], r["source_id"]) for r in rows]


# --- canonical language mapping ---------------------------------------


def test_canonical_language_maps_known_codes_to_wyrd_strings() -> None:
    """Spot-check the most common lang code mappings — these are the
    ones place-name etymology chains will surface most often."""
    assert _canonical_language("en") == "modern-english"
    assert _canonical_language("ang") == "old-english"
    assert _canonical_language("non") == "old-norse"
    assert _canonical_language("is") == "icelandic"
    assert _canonical_language("gem-pro") == "proto-germanic"
    assert _canonical_language("cy") == "welsh"
    assert _canonical_language("la") == "latin"
    assert _canonical_language("ine-pro") == "proto-indo-european"


def test_canonical_language_passes_unmapped_codes_through_raw() -> None:
    """Unmapped codes pass through unchanged so the data is still
    loadable; operator can extend the map later. No KeyError."""
    assert _canonical_language("xyz-unknown") == "xyz-unknown"


# --- etymology_templates → upward edges -------------------------------


def test_inh_template_emits_inheritance_upward_edge(fresh_db: Path) -> None:
    """{{inh|<this>|<parent_lang>|<parent_word>}} produces an
    inheritance edge from the named parent down to this entry."""
    line = _wiktextract_entry(
        word="tūn",
        lang_code="ang",
        etymology_templates=[
            {
                "name": "inh",
                "args": {"1": "ang", "2": "gem-pro", "3": "*tūnaz"},
                "expansion": "Proto-Germanic *tūnaz",
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    assert result["upward_edges"] == 1
    assert result["downward_edges"] == 0
    assert edges == [("*tūnaz", "tūn", "inheritance", "wiktionary")]


@pytest.mark.parametrize(
    "template_name,expected_edge_type",
    [
        ("inh", "inheritance"),
        ("inherited", "inheritance"),
        ("bor", "borrowing"),
        ("borrowed", "borrowing"),
        ("der", "derivation"),
        ("derived", "derivation"),
        ("cal", "calque"),
        ("calque", "calque"),
    ],
)
def test_each_upward_template_kind_maps_to_correct_edge_type(
    fresh_db: Path, template_name: str, expected_edge_type: str
) -> None:
    """The wiktextract template name → D27 edge_type map must be
    bit-stable across all eight upward kinds. Pin each so a refactor
    can't silently drop or rename a mapping."""
    line = _wiktextract_entry(
        word="child",
        lang_code="en",
        etymology_templates=[
            {
                "name": template_name,
                "args": {"1": "en", "2": "ang", "3": "parent"},
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edge_type = db.conn.execute("SELECT edge_type FROM etymon_descent").fetchone()["edge_type"]
    assert edge_type == expected_edge_type


def test_cog_template_is_skipped_not_emitted(fresh_db: Path) -> None:
    """Cognate templates are peer relations, not directional chains.
    Per D27 they don't bridge synsets and should not produce edges
    here either (would over-unify clustering downstream)."""
    line = _wiktextract_entry(
        word="tūn",
        lang_code="ang",
        etymology_templates=[{"name": "cog", "args": {"1": "non", "2": "tún"}}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    assert result["upward_edges"] == 0
    assert result["skipped_templates"] == 1
    assert edges == []


def test_unrecognized_template_increments_unsupported_counter(
    fresh_db: Path,
) -> None:
    """Templates we don't know about (some niche wiktextract kind)
    pass through as unsupported_templates, surfaced in the operator
    output so the maps can be extended."""
    line = _wiktextract_entry(
        word="tūn",
        lang_code="ang",
        etymology_templates=[{"name": "made-up-template", "args": {}}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
    assert result["unsupported_templates"] == 1


def test_upward_template_with_missing_args_is_skipped(fresh_db: Path) -> None:
    """A malformed template (missing parent_lang or parent_word) is
    skipped without exception — the rest of the entry processes."""
    line = _wiktextract_entry(
        word="tūn",
        lang_code="ang",
        etymology_templates=[
            {"name": "inh", "args": {"1": "ang"}},  # missing args 2 + 3
            {"name": "inh", "args": {"1": "ang", "2": "gem-pro", "3": "*tūnaz"}},
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)

    # Only the well-formed template produces an edge.
    assert result["upward_edges"] == 1


# --- descendants section → downward edges -----------------------------


def test_descendants_flat_list_emits_downward_edges_from_head(
    fresh_db: Path,
) -> None:
    """Two depth=1 descendants attach directly to the head entry."""
    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "ang", "2": "tūn"}}],
            },
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "non", "2": "tún"}}],
            },
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    assert result["downward_edges"] == 2
    assert ("*tūnaz", "tūn", "inheritance", "wiktionary") in edges
    assert ("*tūnaz", "tún", "inheritance", "wiktionary") in edges


def test_descendants_nested_depth_chains_to_preceding_entry(
    fresh_db: Path,
) -> None:
    """Depth=2 descendants attach to the most recent depth=1 entry, not
    to the head. Pin the parent_stack walk so a regression in nesting
    surfaces immediately.

    Tree:
      *tūnaz
        ├── tūn (OE)        depth=1
        │     └── town (en) depth=2
        └── tún (ON)        depth=1
              └── tún (is)  depth=2
    """
    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "ang", "2": "tūn"}}],
            },
            {
                "depth": 2,
                "templates": [{"name": "desc", "args": {"1": "en", "2": "town"}}],
            },
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "non", "2": "tún_on"}}],
            },
            {
                "depth": 2,
                "templates": [{"name": "desc", "args": {"1": "is", "2": "tún_is"}}],
            },
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    assert result["downward_edges"] == 4
    # Direct depth-1 edges from the head.
    assert ("*tūnaz", "tūn", "inheritance", "wiktionary") in edges
    assert ("*tūnaz", "tún_on", "inheritance", "wiktionary") in edges
    # Depth-2 attaches to the most recent depth-1 entry (NOT the head).
    assert ("tūn", "town", "inheritance", "wiktionary") in edges
    assert ("tún_on", "tún_is", "inheritance", "wiktionary") in edges
    # Sanity: the head should NOT have a direct edge to the depth=2 entries.
    assert ("*tūnaz", "town", "inheritance", "wiktionary") not in edges
    assert ("*tūnaz", "tún_is", "inheritance", "wiktionary") not in edges


def test_descendants_depth_jump_recovers_without_crash(fresh_db: Path) -> None:
    """A malformed depth jump (depth=3 with no preceding depth=2) is
    recovered by attaching to the deepest available parent and
    incrementing depth_jumps_recovered. The ingest doesn't crash."""
    line = _wiktextract_entry(
        word="*root",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "ang", "2": "child1"}}],
            },
            {
                # Skipping depth=2 directly to depth=3 — malformed.
                "depth": 3,
                "templates": [{"name": "desc", "args": {"1": "en", "2": "child3"}}],
            },
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    assert result["depth_jumps_recovered"] == 1
    assert result["downward_edges"] == 2
    # depth=3 entry attached to the deepest available parent (child1).
    assert ("child1", "child3", "inheritance", "wiktionary") in edges


def test_descendants_non_desc_templates_inside_entry_are_skipped(
    fresh_db: Path,
) -> None:
    """Descendant entries can carry non-desc templates (qualifiers,
    glosses). These should be skipped without producing edges."""
    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [
                    {"name": "desc", "args": {"1": "ang", "2": "tūn"}},
                    {"name": "qualifier", "args": {"1": "rare"}},
                ],
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    assert result["downward_edges"] == 1
    assert result["skipped_templates"] == 1
    assert edges == [("*tūnaz", "tūn", "inheritance", "wiktionary")]


# --- idempotency / dry-run / counters ---------------------------------


def test_ingest_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False parses + counts but writes zero rows; the source
    'wiktionary' row isn't even created."""
    line = _wiktextract_entry(
        word="tūn",
        lang_code="ang",
        etymology_templates=[{"name": "inh", "args": {"1": "ang", "2": "gem-pro", "3": "*tūnaz"}}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=False)
        edge_count = db.conn.execute("SELECT COUNT(*) AS n FROM etymon_descent").fetchone()["n"]
        etymon_count = db.conn.execute("SELECT COUNT(*) AS n FROM etymon").fetchone()["n"]
        source_exists = db.conn.execute(
            "SELECT COUNT(*) AS n FROM source WHERE id = 'wiktionary'"
        ).fetchone()["n"]

    assert result["upward_edges"] == 1  # counted
    assert edge_count == 0  # not written
    assert etymon_count == 0
    assert source_exists == 0


def test_ingest_idempotent_on_rerun(fresh_db: Path) -> None:
    """Re-running on the same JSONL is a no-op: etymon upserts dedupe
    on (canonical_form, language) and edge inserts dedupe on the
    UNIQUE (parent_id, child_id, edge_type, source_id) constraint."""
    line = _wiktextract_entry(
        word="tūn",
        lang_code="ang",
        etymology_templates=[{"name": "inh", "args": {"1": "ang", "2": "gem-pro", "3": "*tūnaz"}}],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges_after_first = _all_descent_edges(db)
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges_after_second = _all_descent_edges(db)

    assert edges_after_first == edges_after_second
    assert len(edges_after_first) == 1


def test_ingest_skips_malformed_lines(fresh_db: Path) -> None:
    """JSON parse failures and missing word/lang_code skip cleanly,
    incrementing the counter — they don't crash the run."""
    valid = _wiktextract_entry(
        word="tūn",
        lang_code="ang",
        etymology_templates=[{"name": "inh", "args": {"1": "ang", "2": "gem-pro", "3": "*tūnaz"}}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(
            db,
            _stream(
                "not json at all\n",
                json.dumps({"missing_word": True}) + "\n",
                json.dumps({"word": "no_lang"}) + "\n",
                "\n",  # blank line: not malformed, just skipped
                valid,
            ),
            apply=True,
        )

    assert result["entries_skipped_malformed"] == 3
    assert result["entries_parsed"] == 1
    assert result["upward_edges"] == 1


def test_ingest_limit_stops_after_n_entries(fresh_db: Path) -> None:
    """--limit N stops processing after N parsed entries; remaining
    lines stay unread (lines_read may still tick because the limit is
    checked AFTER the line read)."""
    lines = [_wiktextract_entry(word=f"w{i}", lang_code="en") for i in range(10)]
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(*lines), apply=True, limit=3)
        etymon_count = db.conn.execute("SELECT COUNT(*) AS n FROM etymon").fetchone()["n"]

    assert result["entries_parsed"] == 3
    assert etymon_count == 3


def test_ingest_since_line_skips_first_n_lines(fresh_db: Path) -> None:
    """--since-line N skips the first N lines (resumability across
    multi-hour ingest sessions)."""
    lines = [_wiktextract_entry(word=f"w{i}", lang_code="en") for i in range(5)]
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(*lines), apply=True, since_line=3)
        forms = sorted(
            row["canonical_form"] for row in db.conn.execute("SELECT canonical_form FROM etymon")
        )

    # Skipped w0, w1, w2; processed w3, w4.
    assert result["entries_parsed"] == 2
    assert forms == ["w3", "w4"]


# --- ingest_wiktextract_path: file vs gzip ---------------------------


def test_ingest_path_reads_plain_jsonl(fresh_db: Path, tmp_path: Path) -> None:
    """A .jsonl file opens as plain text."""
    line = _wiktextract_entry(
        word="tūn",
        lang_code="ang",
        etymology_templates=[{"name": "inh", "args": {"1": "ang", "2": "gem-pro", "3": "*tūnaz"}}],
    )
    path = tmp_path / "wiktextract.jsonl"
    path.write_text(line)

    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_path(db, path, apply=True)

    assert result["upward_edges"] == 1


def test_ingest_path_reads_gzipped_jsonl(fresh_db: Path, tmp_path: Path) -> None:
    """A .jsonl.gz file opens via gzip — kaikki.org dumps are typically
    gzipped, so this is the common case."""
    line = _wiktextract_entry(
        word="tūn",
        lang_code="ang",
        etymology_templates=[{"name": "inh", "args": {"1": "ang", "2": "gem-pro", "3": "*tūnaz"}}],
    )
    path = tmp_path / "wiktextract.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(line)

    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_path(db, path, apply=True)

    assert result["upward_edges"] == 1


# --- end-to-end with cluster_cognates --------------------------------


def test_ingest_then_cluster_cognates_assigns_synsets_across_languages(
    fresh_db: Path,
) -> None:
    """The full chain: ingest a Proto-Germanic entry's Etymology +
    Descendants → run cluster_cognates → every reachable etymon
    carries the same synset_id pointing at the proto root.

    This is the load-bearing integration test — it proves the full
    Wiktionary→cluster pipeline works end-to-end on realistic data."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "ang", "2": "tūn"}}],
            },
            {
                "depth": 2,
                "templates": [{"name": "desc", "args": {"1": "en", "2": "town"}}],
            },
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "non", "2": "tún_on"}}],
            },
            {
                "depth": 2,
                "templates": [{"name": "desc", "args": {"1": "is", "2": "tún_is"}}],
            },
        ],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        cluster_result = cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["synset_id"]
            for row in db.conn.execute("SELECT canonical_form, synset_id FROM etymon")
        }
        proto_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = '*tūnaz'"
        ).fetchone()["id"]

    assert cluster_result["roots"] == 1
    # All five etymons (proto + 4 descendants) carry the proto-root's id.
    assert rows["*tūnaz"] == proto_id
    assert rows["tūn"] == proto_id
    assert rows["town"] == proto_id
    assert rows["tún_on"] == proto_id
    assert rows["tún_is"] == proto_id


# --- CLI tests ---------------------------------------------------------


def test_ingest_wiktionary_cli_dry_run_reports_without_writing(
    fresh_db: Path, tmp_path: Path
) -> None:
    """End-to-end CLI smoke: dry-run reports parse counts and leaves
    the DB untouched."""
    line = _wiktextract_entry(
        word="tūn",
        lang_code="ang",
        etymology_templates=[{"name": "inh", "args": {"1": "ang", "2": "gem-pro", "3": "*tūnaz"}}],
    )
    path = tmp_path / "wiktextract.jsonl"
    path.write_text(line)

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "ingest-wiktionary", str(path), "--db", str(fresh_db)],
    )

    assert result.exit_code == 0, result.output
    assert "1 entry/entries parsed" in result.output
    assert "upward_edges    = 1" in result.output
    assert "dry-run" in result.output

    with LexiconDB(fresh_db) as db:
        edge_count = db.conn.execute("SELECT COUNT(*) AS n FROM etymon_descent").fetchone()["n"]
    assert edge_count == 0


def test_ingest_wiktionary_cli_apply_writes_to_db(fresh_db: Path, tmp_path: Path) -> None:
    """--apply persists etymons + descent edges and the synthetic
    'wiktionary' source row."""
    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "ang", "2": "tūn"}}],
            }
        ],
    )
    path = tmp_path / "wiktextract.jsonl"
    path.write_text(line)

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "ingest-wiktionary", str(path), "--db", str(fresh_db), "--apply"],
    )

    assert result.exit_code == 0, result.output
    assert "downward_edges  = 1" in result.output
    assert "dry-run" not in result.output

    with LexiconDB(fresh_db) as db:
        sources = [r["id"] for r in db.conn.execute("SELECT id FROM source")]
        edges = _all_descent_edges(db)
    assert "wiktionary" in sources
    assert ("*tūnaz", "tūn", "inheritance", "wiktionary") in edges


# --- wyrd-hun round-1 review follow-ups ---------------------------------


@pytest.mark.parametrize(
    "flag,expected_edge_type",
    [
        ("bor", "borrowing"),
        ("der", "derivation"),
        ("cal", "calque"),
    ],
)
def test_desc_template_with_relationship_flag_emits_correct_edge_type(
    fresh_db: Path, flag: str, expected_edge_type: str
) -> None:
    """Wiktionary editors flag descendants that were borrowed (or
    derived/calqued) from the parent rather than directly inherited
    via a `bor=1`/`der=1`/`cal=1` arg on the {{desc}} template. Without
    this, the edge_type column lies for those rows. Pin each flag so a
    refactor of _DESC_FLAG_TO_EDGE surfaces here."""
    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [
                    {
                        "name": "desc",
                        "args": {"1": "ang", "2": "tūn", flag: "1"},
                    }
                ],
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edge_type = db.conn.execute("SELECT edge_type FROM etymon_descent").fetchone()["edge_type"]
    assert edge_type == expected_edge_type


def test_desc_template_without_flag_defaults_to_inheritance(
    fresh_db: Path,
) -> None:
    """Pin the default behavior: a {{desc}} template with no flag is
    inheritance. This was the only behavior tested before round 1; now
    asserted alongside the flagged variants so the default-vs-flag
    branching is explicit."""
    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "ang", "2": "tūn"}}],
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edge_type = db.conn.execute("SELECT edge_type FROM etymon_descent").fetchone()["edge_type"]
    assert edge_type == "inheritance"


def test_desc_flag_precedence_bor_wins_over_der(fresh_db: Path) -> None:
    """When multiple flags are present (theoretical but defensible
    against malformed templates), bor wins per the documented order
    in _DESC_FLAG_TO_EDGE. Pin the precedence so a tuple-reorder
    surfaces here."""
    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [
                    {
                        "name": "desc",
                        "args": {"1": "ang", "2": "tūn", "bor": "1", "der": "1"},
                    }
                ],
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edge_type = db.conn.execute("SELECT edge_type FROM etymon_descent").fetchone()["edge_type"]
    assert edge_type == "borrowing"


def test_walk_descendants_multiple_templates_per_entry_uses_last_as_parent(
    fresh_db: Path,
) -> None:
    """A descendants entry with multiple {{desc}} templates emits one
    edge per template, but only the LAST upserted child becomes the
    parent for subsequent depth+1 entries. Pin this LAST-wins choice
    so a refactor of _walk_descendants surfaces if it changes.

    Tree:
      *root → [variant_a, variant_canonical]   (both depth=1, same entry)
        → child  (depth=2 → must attach to variant_canonical, NOT variant_a)
    """
    line = _wiktextract_entry(
        word="*root",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [
                    {"name": "desc", "args": {"1": "ang", "2": "variant_a"}},
                    {"name": "desc", "args": {"1": "ang", "2": "variant_canonical"}},
                ],
            },
            {
                "depth": 2,
                "templates": [{"name": "desc", "args": {"1": "en", "2": "child"}}],
            },
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    assert result["downward_edges"] == 3
    # Both variants attach to the head as direct children.
    assert ("*root", "variant_a", "inheritance", "wiktionary") in edges
    assert ("*root", "variant_canonical", "inheritance", "wiktionary") in edges
    # depth=2 child attaches to the LAST template's child (variant_canonical),
    # NOT the first (variant_a). Pin the LAST-wins behavior.
    assert ("variant_canonical", "child", "inheritance", "wiktionary") in edges
    assert ("variant_a", "child", "inheritance", "wiktionary") not in edges


def test_entry_combines_etymology_and_descendants_emits_both(
    fresh_db: Path,
) -> None:
    """Real Wiktionary entries almost always have BOTH etymology_templates
    AND descendants populated. Round 1 had no test pinning the cross-
    section interaction in _process_entry — fix here. Tree:

      *PIE-root  ← UPWARD edge from etymology (this entry inherits from root)
         ↓
        this_word  (the entry being ingested)
         ↓
        descendant  ← DOWNWARD edge from descendants
    """
    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        etymology_templates=[
            {
                "name": "inh",
                "args": {"1": "gem-pro", "2": "ine-pro", "3": "*dewh₂-"},
            }
        ],
        descendants=[
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "ang", "2": "tūn"}}],
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    assert result["upward_edges"] == 1
    assert result["downward_edges"] == 1
    # Upward edge: PIE root → this entry.
    assert ("*dewh₂-", "*tūnaz", "inheritance", "wiktionary") in edges
    # Downward edge: this entry → its OE descendant.
    assert ("*tūnaz", "tūn", "inheritance", "wiktionary") in edges


def test_walk_descendants_skips_non_dict_entries(fresh_db: Path) -> None:
    """Defensive: a malformed descendants list with a non-dict entry
    (string, None, integer) skips that entry without crashing. Pinned
    to keep the defensive branch from being dead-on-paper code."""
    # Build the JSON manually since _wiktextract_entry expects clean dicts.
    record = {
        "word": "*tūnaz",
        "lang_code": "gem-pro",
        "pos": "noun",
        "descendants": [
            "this is a malformed string entry",  # non-dict
            None,  # also non-dict
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "ang", "2": "tūn"}}],
            },
        ],
    }
    line = json.dumps(record) + "\n"
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    # Non-dict entries silently skipped; the well-formed entry processes.
    assert result["downward_edges"] == 1
    assert ("*tūnaz", "tūn", "inheritance", "wiktionary") in edges


def test_walk_descendants_normalizes_invalid_depth_to_one(fresh_db: Path) -> None:
    """Defensive: a descendants entry with depth=0 or a non-int depth
    falls back to depth=1 (direct child of head). Pin so the
    `if not isinstance(depth, int) or depth < 1: depth = 1` guard
    isn't dead code."""
    record = {
        "word": "*root",
        "lang_code": "gem-pro",
        "pos": "noun",
        "descendants": [
            {
                "depth": 0,  # invalid — should normalize to 1
                "templates": [{"name": "desc", "args": {"1": "ang", "2": "child0"}}],
            },
            {
                "depth": "not-a-number",  # non-int — should normalize to 1
                "templates": [{"name": "desc", "args": {"1": "ang", "2": "child_str"}}],
            },
        ],
    }
    line = json.dumps(record) + "\n"
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    # Both descendants attach to the head as if depth=1.
    assert ("*root", "child0", "inheritance", "wiktionary") in edges
    assert ("*root", "child_str", "inheritance", "wiktionary") in edges


def test_ingest_wiktionary_cli_surfaces_unsupported_templates(
    fresh_db: Path, tmp_path: Path
) -> None:
    """The CLI's conditional 'unsupported_templates = N' echo branch
    fires only when the counter > 0 — round 1 had no test pinning the
    format string. A typo in cli.py would slip through silently."""
    line = _wiktextract_entry(
        word="tūn",
        lang_code="ang",
        etymology_templates=[{"name": "totally-made-up-template", "args": {}}],
    )
    path = tmp_path / "wiktextract.jsonl"
    path.write_text(line)

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "ingest-wiktionary", str(path), "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "unsupported_templates = 1" in result.output


def test_ingest_wiktionary_cli_surfaces_depth_jumps(fresh_db: Path, tmp_path: Path) -> None:
    """The CLI's conditional 'depth_jumps_recovered = N' echo branch
    fires only when the counter > 0. Pin the format string so a typo
    doesn't slip past."""
    line = _wiktextract_entry(
        word="*root",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "ang", "2": "child1"}}],
            },
            {
                # Skipping depth=2 directly to depth=3 — malformed.
                "depth": 3,
                "templates": [{"name": "desc", "args": {"1": "en", "2": "child3"}}],
            },
        ],
    )
    path = tmp_path / "wiktextract.jsonl"
    path.write_text(line)

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "ingest-wiktionary", str(path), "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "depth_jumps_recovered = 1" in result.output


def test_ingest_wiktionary_cli_surfaces_malformed_skip_count(
    fresh_db: Path, tmp_path: Path
) -> None:
    """The CLI's 'skipped N malformed line(s)' branch fires when
    entries_skipped_malformed > 0. Pin the format string."""
    valid = _wiktextract_entry(
        word="tūn",
        lang_code="ang",
        etymology_templates=[{"name": "inh", "args": {"1": "ang", "2": "gem-pro", "3": "*tūnaz"}}],
    )
    path = tmp_path / "wiktextract.jsonl"
    path.write_text("not json at all\n" + valid)

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "ingest-wiktionary", str(path), "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "skipped 1 malformed line(s)" in result.output


def test_desc_flag_precedence_der_wins_over_cal(fresh_db: Path) -> None:
    """Round-2 test-coverage gap: precedence chain bor > der > cal is
    documented but only bor>der was pinned. Adding der>cal so a swap
    of cal ahead of der would surface here."""
    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [
                    {
                        "name": "desc",
                        "args": {"1": "ang", "2": "tūn", "der": "1", "cal": "1"},
                    }
                ],
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edge_type = db.conn.execute("SELECT edge_type FROM etymon_descent").fetchone()["edge_type"]
    assert edge_type == "derivation"


def test_walk_descendants_all_skipped_templates_does_not_advance_parent_stack(
    fresh_db: Path,
) -> None:
    """Round-2 test-coverage gap: the LAST-wins logic only updates
    last_child_id inside the valid-edge branch, so a descendants entry
    whose templates are ALL skipped (e.g. only a {{qualifier}} or
    {{m}}) must NOT push None onto the parent_stack and must NOT
    overwrite the previous depth-N anchor.

    Tree (the load-bearing case):
      *root → variant   (depth=1, valid desc → becomes parent for depth=2)
            → [qualifier-only entry]   (depth=1, no edge — must NOT
                                         displace 'variant' as the parent
                                         for the next depth=2 entry)
            → child   (depth=2 → must attach to 'variant', NOT crash on
                                  a None parent from the qualifier-only row)
    """
    line = _wiktextract_entry(
        word="*root",
        lang_code="gem-pro",
        descendants=[
            {
                "depth": 1,
                "templates": [{"name": "desc", "args": {"1": "ang", "2": "variant"}}],
            },
            {
                "depth": 1,
                "templates": [
                    {"name": "qualifier", "args": {"1": "rare"}},
                ],
            },
            {
                "depth": 2,
                "templates": [{"name": "desc", "args": {"1": "en", "2": "child"}}],
            },
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    # 2 valid downward edges (variant + child); the qualifier row
    # contributed 0 edges and bumped skipped_templates instead.
    assert result["downward_edges"] == 2
    assert result["skipped_templates"] == 1
    # Critical: depth=2 'child' attached to 'variant', NOT to the
    # qualifier-only entry (which has no etymon and would crash a
    # naive implementation that pushed None onto parent_stack).
    assert ("variant", "child", "inheritance", "wiktionary") in edges
    assert ("*root", "variant", "inheritance", "wiktionary") in edges
