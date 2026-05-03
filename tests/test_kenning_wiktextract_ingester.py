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


# --- descendants section → downward edges (wyrd-c9t) -------------------
#
# Real wiktextract descendants are a NESTED TREE: each node has
# `lang_code` and `word` directly, with optional `descendants` (recursive)
# for sub-trees. v1 misread this as flat-with-depth-markers + templates;
# rewritten in wyrd-c9t against real kaikki.org data.


def test_descendants_flat_list_emits_downward_edges_from_head(
    fresh_db: Path,
) -> None:
    """Two top-level descendants attach directly to the head entry."""
    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        descendants=[
            {"lang_code": "ang", "word": "tūn"},
            {"lang_code": "non", "word": "tún"},
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    assert result["downward_edges"] == 2
    assert ("*tūnaz", "tūn", "inheritance", "wiktionary") in edges
    assert ("*tūnaz", "tún", "inheritance", "wiktionary") in edges


def test_descendants_nested_recursion_chains_through_subtree(
    fresh_db: Path,
) -> None:
    """Nested `descendants` arrays produce edges from each node to its
    own children — the recursion is the depth-tracking. Pin the tree
    walk so a regression in recursion surfaces immediately.

    Tree:
      *tūnaz
        ├── tūn (OE)
        │     └── town (en)
        └── tún (ON)
              └── tún_is (is)
    """
    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        descendants=[
            {
                "lang_code": "ang",
                "word": "tūn",
                "descendants": [
                    {"lang_code": "en", "word": "town"},
                ],
            },
            {
                "lang_code": "non",
                "word": "tún",
                "descendants": [
                    {"lang_code": "is", "word": "tún_is"},
                ],
            },
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    assert result["downward_edges"] == 4
    # Direct top-level edges from the head.
    assert ("*tūnaz", "tūn", "inheritance", "wiktionary") in edges
    assert ("*tūnaz", "tún", "inheritance", "wiktionary") in edges
    # Recursive edges hang off each node, NOT the head.
    assert ("tūn", "town", "inheritance", "wiktionary") in edges
    assert ("tún", "tún_is", "inheritance", "wiktionary") in edges
    # Head must NOT have direct edges to the deeper nodes.
    assert ("*tūnaz", "town", "inheritance", "wiktionary") not in edges
    assert ("*tūnaz", "tún_is", "inheritance", "wiktionary") not in edges


def test_descendants_calque_tag_overrides_inheritance(fresh_db: Path) -> None:
    """tags=['calque'] on a descendant node overrides the default
    'inheritance' edge_type. This is the real shape — the OE wiktextract
    slice has 99 such tags. Pin so a refactor of _DESCENDANT_TAG_TO_EDGE
    surfaces here."""
    line = _wiktextract_entry(
        word="*tūnaz",
        lang_code="gem-pro",
        descendants=[
            {"lang_code": "ang", "word": "tūn", "tags": ["calque"]},
        ],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edge_type = db.conn.execute("SELECT edge_type FROM etymon_descent").fetchone()["edge_type"]
    assert edge_type == "calque"


@pytest.mark.parametrize(
    "tag,expected_edge_type",
    [
        ("calque", "calque"),
        ("borrowed", "borrowing"),
        ("borrowing", "borrowing"),
        ("derived", "derivation"),
        ("derivation", "derivation"),
    ],
)
def test_descendants_tag_to_edge_type_mapping(
    fresh_db: Path, tag: str, expected_edge_type: str
) -> None:
    """Each entry in _DESCENDANT_TAG_TO_EDGE pinned individually so a
    map shrink can't silently drop one."""
    line = _wiktextract_entry(
        word="*head",
        lang_code="gem-pro",
        descendants=[{"lang_code": "ang", "word": "child", "tags": [tag]}],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edge_type = db.conn.execute("SELECT edge_type FROM etymon_descent").fetchone()["edge_type"]
    assert edge_type == expected_edge_type


def test_descendants_node_without_word_is_skipped_but_recursion_continues(
    fresh_db: Path,
) -> None:
    """Real wiktextract has lang-only header rows (lang+lang_code with
    no `word`) that introduce a sub-tree without a specific attested
    form. Skip the header itself (no edge to anchor) but recurse — the
    sub-tree's children attach to the parent of the header.

    Tree:
      *root
        └── [Proto-Brythonic header, no word]
              └── ɨn (Welsh) — should attach to *root, not to the header
    """
    line = _wiktextract_entry(
        word="*root",
        lang_code="gem-pro",
        descendants=[
            {
                "lang_code": "cel-bry-pro",
                # no `word`
                "descendants": [
                    {"lang_code": "cy", "word": "ɨn"},
                ],
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    assert result["downward_edges"] == 1
    # The Welsh child attached to *root because the proto-brythonic
    # header had no word to anchor through.
    assert ("*root", "ɨn", "inheritance", "wiktionary") in edges


def test_descendants_node_without_lang_code_is_skipped(fresh_db: Path) -> None:
    """A node missing lang_code is malformed wiktextract output. Skip
    silently rather than crashing or upserting a bogus etymon."""
    line = _wiktextract_entry(
        word="*root",
        lang_code="gem-pro",
        descendants=[
            {"word": "no_lang_code"},  # malformed
            {"lang_code": "ang", "word": "valid"},
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)
    assert result["downward_edges"] == 1
    assert ("*root", "valid", "inheritance", "wiktionary") in edges


def test_descendants_skips_non_dict_entries_in_array(fresh_db: Path) -> None:
    """Defensive: a descendants array with a string or None entry skips
    that entry without crashing — the well-formed siblings still
    process."""
    record = {
        "word": "*root",
        "lang_code": "gem-pro",
        "pos": "noun",
        "descendants": [
            "this is malformed",  # non-dict
            None,  # also non-dict
            {"lang_code": "ang", "word": "valid"},
        ],
    }
    line = json.dumps(record) + "\n"
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)
    assert result["downward_edges"] == 1
    assert ("*root", "valid", "inheritance", "wiktionary") in edges


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
                "lang_code": "ang",
                "word": "tūn",
                "descendants": [{"lang_code": "en", "word": "town"}],
            },
            {
                "lang_code": "non",
                "word": "tún_on",
                "descendants": [{"lang_code": "is", "word": "tún_is"}],
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
        descendants=[{"lang_code": "ang", "word": "tūn"}],
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


# --- combined etymology + descendants + CLI conditional output -----


def test_entry_combines_etymology_and_descendants_emits_both(
    fresh_db: Path,
) -> None:
    """Real Wiktionary entries almost always have BOTH etymology_templates
    AND descendants populated. Pin the cross-section interaction —
    upward edge from PIE root via etymology, downward edge to OE child
    via descendants tree. Tree:

      *PIE-root  ← UPWARD edge from etymology
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
        descendants=[{"lang_code": "ang", "word": "tūn"}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)

    assert result["upward_edges"] == 1
    assert result["downward_edges"] == 1
    assert ("*dewh₂-", "*tūnaz", "inheritance", "wiktionary") in edges
    assert ("*tūnaz", "tūn", "inheritance", "wiktionary") in edges


def test_ingest_wiktionary_cli_surfaces_unsupported_templates(
    fresh_db: Path, tmp_path: Path
) -> None:
    """The CLI's conditional 'unsupported_templates = N' echo branch
    fires only when the counter > 0. Pin the format string so a typo
    doesn't slip through silently."""
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


def test_ingest_wiktionary_cli_surfaces_malformed_skip_count(
    fresh_db: Path, tmp_path: Path
) -> None:
    """The CLI's 'skipped N malformed line(s)' branch fires when
    entries_skipped_malformed > 0."""
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


# --- wyrd-prv: extended template kind support --------------------------


def test_root_template_emits_inheritance_edge_to_pie_root(fresh_db: Path) -> None:
    """{{root|<this>|<ancestor_lang>|<root_word>}} produces an
    inheritance edge from the named PIE root down to this entry."""
    line = _wiktextract_entry(
        word="word",
        lang_code="ang",
        etymology_templates=[
            {
                "name": "root",
                "args": {"1": "ang", "2": "ine-pro", "3": "*werh₁-"},
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)
    assert ("*werh₁-", "word", "inheritance", "wiktionary") in edges


def test_root_template_with_multiple_parallel_roots_emits_one_edge_each(
    fresh_db: Path,
) -> None:
    """Wiktionary editors sometimes cite parallel PIE roots when the
    chain is contested ({{root|ang|ine-pro|*werh₁-|*dʰeh₁-}}). Each
    root word gets its own edge, so the descent graph carries both
    candidate ancestries."""
    line = _wiktextract_entry(
        word="word",
        lang_code="ang",
        etymology_templates=[
            {
                "name": "root",
                "args": {"1": "ang", "2": "ine-pro", "3": "*werh₁-", "4": "*dʰeh₁-"},
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)
    assert ("*werh₁-", "word", "inheritance", "wiktionary") in edges
    assert ("*dʰeh₁-", "word", "inheritance", "wiktionary") in edges


@pytest.mark.parametrize(
    "compound_template",
    ["compound", "com", "prefix", "pre", "suffix", "suf", "af", "affix"],
)
def test_compound_template_emits_one_edge_per_constituent(
    fresh_db: Path, compound_template: str
) -> None:
    """{{compound|ang|Sċott|land}} produces TWO edges: each constituent
    becomes a 'compound' parent of the resulting entry. Pin each of the
    8 known compound-template aliases so a name drop in
    _COMPOUND_TEMPLATE_NAMES surfaces here."""
    line = _wiktextract_entry(
        word="Scotland",
        lang_code="ang",
        etymology_templates=[
            {
                "name": compound_template,
                "args": {"1": "ang", "2": "Sċott", "3": "land"},
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)
    assert result["upward_edges"] == 2
    assert ("Sċott", "Scotland", "compound", "wiktionary") in edges
    assert ("land", "Scotland", "compound", "wiktionary") in edges


def test_compound_template_with_three_constituents_emits_three_edges(
    fresh_db: Path,
) -> None:
    """Three-part compounds (rare but real) produce one edge per part."""
    line = _wiktextract_entry(
        word="threepart",
        lang_code="ang",
        etymology_templates=[
            {
                "name": "compound",
                "args": {"1": "ang", "2": "alpha", "3": "beta", "4": "gamma"},
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)
    assert result["upward_edges"] == 3
    for part in ("alpha", "beta", "gamma"):
        assert (part, "threepart", "compound", "wiktionary") in edges


def test_compound_edge_does_not_bridge_synsets(fresh_db: Path) -> None:
    """Compound is correctly classified outside the bridging set
    (per D27 + wyrd-81n: only inheritance and borrowing bridge
    synset_id). Pin so re-running cluster-cognates after a compound-
    only ingest produces zero clusters from those edges alone."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    line = _wiktextract_entry(
        word="Scotland",
        lang_code="ang",
        etymology_templates=[
            {
                "name": "compound",
                "args": {"1": "ang", "2": "Sċott", "3": "land"},
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        result = cluster_cognates(db, apply=True)
    # 0 roots because compound edges aren't bridging; no canonical-edge
    # set exists from the compound-only ingest.
    assert result["roots"] == 0
    assert result["candidates"] == 0


@pytest.mark.parametrize(
    "skipped_template",
    ["dercat", "etymon", "surf", "unk", "m+", "noncog", "glossary"],
)
def test_each_new_skipped_template_does_not_count_unsupported(
    fresh_db: Path, skipped_template: str
) -> None:
    """The wyrd-prv expansion added several template kinds to
    _SKIPPED_TEMPLATE_NAMES (kinds that are real but un-extractable
    or peer-not-edge). Pin each so a refactor that moves one out of
    the skipped set surfaces it as unsupported_templates instead."""
    line = _wiktextract_entry(
        word="x",
        lang_code="ang",
        etymology_templates=[{"name": skipped_template, "args": {}}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
    assert result["skipped_templates"] == 1
    assert result["unsupported_templates"] == 0
