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
from wyrd.generators.kenning.lexicon.wiktextract_ingester import (
    _canonical_language,
    _classify_form_variant,
    _extract_head_template_renderings,
    _extract_pronunciation,
    _is_clean_native_form,
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
        ("inh+", "inheritance"),
        ("inh-lite", "inheritance"),
        ("bor", "borrowing"),
        ("borrowed", "borrowing"),
        ("bor+", "borrowing"),
        ("lbor", "borrowing"),
        ("ubor", "borrowing"),
        ("der", "derivation"),
        ("derived", "derivation"),
        ("der+", "derivation"),
        ("der-lite/lang", "derivation"),
        ("uder", "derivation"),
        ("cal", "calque"),
        ("calque", "calque"),
        ("clq", "calque"),
        ("semantic loan", "calque"),
        ("sl", "calque"),
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
    map shrink can't silently drop one. NB: of these, only ``calque`` actually
    occurs in ``tags`` in real wiktextract data — ``borrowed`` / ``derived`` live in
    ``raw_tags`` (see test_descendants_raw_tags_to_edge_type_mapping). These cases
    via ``tags`` are synthetic contract pins for the map, not the real shape."""
    line = _wiktextract_entry(
        word="*head",
        lang_code="gem-pro",
        descendants=[{"lang_code": "ang", "word": "child", "tags": [tag]}],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        edge_type = db.conn.execute("SELECT edge_type FROM etymon_descent").fetchone()["edge_type"]
    assert edge_type == expected_edge_type


@pytest.mark.parametrize(
    "raw_tag,expected_edge_type",
    [
        ("borrowed", "borrowing"),
        ("borrowing", "borrowing"),
        ("derived", "derivation"),
    ],
)
def test_descendants_raw_tags_to_edge_type_mapping(
    fresh_db: Path, raw_tag: str, expected_edge_type: str
) -> None:
    """Regression: the ``borrowed`` / ``derived`` markers land in wiktextract's
    ``raw_tags`` (un-normalized) array, NOT ``tags`` — only ``calque`` appears in
    ``tags``. ``_descendant_edge_type`` read only ``tags``, so every borrowed
    descendant defaulted to ``inheritance`` (16k+ nodes in the English slice; the
    501 modern-english→welsh "inheritance" edges of wyrd-fljy were exactly this —
    obvious loans like phone→ffôn). Now reads both arrays. The tags-based test
    above keeps passing (additive); this pins the REAL data shape."""
    line = _wiktextract_entry(
        word="*head",
        lang_code="gem-pro",
        descendants=[{"lang_code": "ang", "word": "child", "raw_tags": [raw_tag]}],
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
    carries the same cognate_id pointing at the proto root.

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
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
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
    # The PIE root is stored de-dashed (wyrd-aicu.8, D45): the boundary dash on
    # the raw template arg '*dewh₂-' is affix-position decoration and is
    # stripped, while the '*' reconstruction sigil survives.
    assert ("*dewh₂", "*tūnaz", "inheritance", "wiktionary") in edges
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
    # Stored de-dashed (wyrd-aicu.8, D45): '*werh₁-' → '*werh₁'.
    assert ("*werh₁", "word", "inheritance", "wiktionary") in edges


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
    # Stored de-dashed (wyrd-aicu.8, D45): boundary dash stripped, '*' kept.
    assert ("*werh₁", "word", "inheritance", "wiktionary") in edges
    assert ("*dʰeh₁", "word", "inheritance", "wiktionary") in edges


def test_junk_de_dashed_words_are_dropped_and_counted(fresh_db: Path) -> None:
    """wyrd-aicu.8 (D45): a word that de-dashes to empty junk (a bare '-', or a
    '*-' root that is only sigil+dash) must be DROPPED at the ingest boundary —
    never handed to the upsert choke, which would raise — and COUNTED so the
    drop is visible to an operator (not a silent skip). Covers all three word
    sites: the head word, an etymology-template parent ref, and a descendants
    child word."""
    junk_head = _wiktextract_entry(word="-", lang_code="ang")  # head strips to empty
    mixed = _wiktextract_entry(
        word="word",
        lang_code="ang",
        # '*-' is sigil+boundary-dash only → de-dashes to empty (junk parent).
        etymology_templates=[{"name": "root", "args": {"1": "ang", "2": "ine-pro", "3": "*-"}}],
        descendants=[{"lang_code": "ang", "word": "-"}],  # junk descendant child
    )
    with LexiconDB(fresh_db) as db:
        counts = ingest_wiktextract_stream(db, _stream(junk_head, mixed), apply=True)
        forms = {r[0] for r in db.conn.execute("SELECT canonical_form FROM etymon")}
    # Only the real head was stored — no empty/junk etymon row, no crash.
    assert forms == {"word"}
    # Each junk drop is counted on its own path.
    assert counts["entries_skipped_malformed"] == 1  # the '-' head entry
    assert counts["upward_edges_skipped_junk_parent"] == 1  # the '*-' root parent
    assert counts["descendant_junk_word"] == 1  # the '-' descendant child
    # No edge survived the junk endpoints.
    assert counts["upward_edges"] == 0
    assert counts["downward_edges"] == 0


@pytest.mark.parametrize(
    "compound_template",
    [
        "compound",
        "compound+",
        "com",
        "com+",
        "prefix",
        "pre",
        "suffix",
        "suf",
        "af",
        "affix",
        "confix",
        "blend",
        "univerbation",
    ],
)
def test_compound_template_emits_one_edge_per_constituent(
    fresh_db: Path, compound_template: str
) -> None:
    """{{compound|ang|Sċott|land}} produces TWO edges: each constituent
    becomes a 'compound' parent of the resulting entry. Pin each of the
    compound-template aliases so a name drop in _COMPOUND_TEMPLATE_NAMES
    surfaces here."""
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


def test_compound_template_walrus_loop_terminates_on_arg_gap(fresh_db: Path) -> None:
    """The walrus while-loop in _compound_template_edges advertises
    'walks args[2..N] until a positional arg is missing'. Pin that a
    sparse args dict ({2: 'a', 4: 'c'}) yields exactly one edge for
    args[2] and stops there — args[4] is not reached because args[3]
    is absent. Without this pin a regression to a fixed-range loop
    (or an iterator that skipped gaps) wouldn't surface."""
    line = _wiktextract_entry(
        word="sparse",
        lang_code="ang",
        etymology_templates=[
            {
                "name": "compound",
                "args": {"1": "ang", "2": "a", "4": "c"},
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)
    assert result["upward_edges"] == 1
    assert ("a", "sparse", "compound", "wiktionary") in edges
    assert ("c", "sparse", "compound", "wiktionary") not in edges


def test_compound_template_with_no_constituent_args_emits_no_edges(
    fresh_db: Path,
) -> None:
    """{{univerbation|gem-pro}} occurs in PG slice as a bare category
    marker with no args[2..]. _compound_template_edges defensively
    returns [] when no parts are found — pin that this case neither
    raises nor produces edges."""
    line = _wiktextract_entry(
        word="bare",
        lang_code="gem-pro",
        etymology_templates=[{"name": "univerbation", "args": {"1": "gem-pro"}}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
    assert result["upward_edges"] == 0
    assert result["unsupported_templates"] == 0  # univerbation is in known set


def test_root_template_walrus_loop_terminates_on_arg_gap(fresh_db: Path) -> None:
    """Same walrus-loop gap-termination contract as
    _compound_template_edges, but for _root_template_edges walking
    args[3..N]. Pin that a sparse args dict ({3: 'r1', 5: 'r3'})
    yields exactly one edge for args[3] and stops there."""
    line = _wiktextract_entry(
        word="rooted",
        lang_code="en",
        etymology_templates=[
            {
                "name": "root",
                "args": {"1": "en", "2": "ine-pro", "3": "*r1", "5": "*r3"},
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)
    assert result["upward_edges"] == 1
    assert ("*r1", "rooted", "inheritance", "wiktionary") in edges
    assert ("*r3", "rooted", "inheritance", "wiktionary") not in edges


@pytest.mark.parametrize(
    "same_lang_derivation_template",
    [
        "clipping",
        "bf",
        "back-formation",
        "back-form",
        "contraction",
        "deverbal",
        "nom",
        "reduplication",
        "contr",
        "sync",
        "apocopic form",
        "metathesis",
        "past participle of",
        "alt form",
        "abbrev",
        "vrd",
        "nominalization",
        "syncope",
        "alternative form of",
        "abbreviation",
    ],
)
def test_same_lang_derivation_template_emits_one_derivation_edge(
    fresh_db: Path, same_lang_derivation_template: str
) -> None:
    """wyrd-wse: the back-formation/clipping family of templates has a
    different arg shape from inh/bor/der/cal — args[1] = this_lang AND
    parent's lang, args[2] = parent_word (NOT args[3]). Each emits one
    'derivation' edge with parent and child sharing the language. Pin
    each alias so a refactor of _SAME_LANG_DERIVATION_TEMPLATE_NAMES
    surfaces immediately."""
    line = _wiktextract_entry(
        word="elp",
        lang_code="ang",
        etymology_templates=[
            {
                "name": same_lang_derivation_template,
                "args": {"1": "ang", "2": "elpend"},
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        edges = _all_descent_edges(db)
    assert result["upward_edges"] == 1
    assert ("elpend", "elp", "derivation", "wiktionary") in edges


def test_same_lang_derivation_with_missing_parent_word_is_skipped(
    fresh_db: Path,
) -> None:
    """A clipping/bf template missing args[2] (the parent word) is not
    edge-producing — the parent isn't named. Skipped without exception
    so the rest of the entry's templates still process."""
    line = _wiktextract_entry(
        word="elp",
        lang_code="ang",
        etymology_templates=[
            {"name": "clipping", "args": {"1": "ang"}},  # missing args[2]
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)

    assert result["upward_edges"] == 0


def test_compound_edge_does_not_bridge_synsets(fresh_db: Path) -> None:
    """Compound is correctly classified outside the bridging set
    (per D27 + wyrd-81n: only inheritance and borrowing bridge
    cognate_id). Pin so re-running cluster-cognates after a compound-
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
    [
        "dercat",
        "etymon",
        "surf",
        "unk",
        "unc",
        "unknown",
        "m+",
        "m-g",
        "noncog",
        "ncog",
        "glossary",
        "gl",
        "yesno",
        "etymid",
        "nonlemma",
        "col-top",
        "doublet",
        "dbt",
        "piecewise doublet",
        "PIE word",
        "onomatopoeic",
        "sound-symbolic",
        "wp",
        "normalized",
        "surface analysis",
        "str left",
        "str_index-lite",
        "str index-lite/logic",
        "str len-lite",
        "str len-lite/core",
        "str sub-lite",
        "str sub-lite/2",
        "ety",
        "lit",
        "langname",
        "word",
        "lg",
        "q",
        "nl",
        "ref",
        "sup",
        "smallsup",
        "cat",
        "or else",
        "desc",
        ",",
        # wyrd-wse: long-name + abbreviation variants of already-skipped
        "nonlemmas",
        "mention-gloss",
        "noncognate",
        "onomatopoeia",
        "onom",
        "m-lite",
        # wyrd-wse: no-arg category-only markers
        "pre-Germanic",
        "vrddhi",
        # wyrd-wse: small formatting / unknown-context templates
        "g",
        "?",
        "s",
        "IPAfont",
        "uncertain",
        "anchor",
        "number box",
        "sno",
        "qinfl",
        "coin",
        "lang",
        "ISSN",
        "tea",
        "pw dbt",
        # wyrd-wse: long-tail single-occurrence kinds
        "senseno",
        "C.E.",
        "smallcaps",
        "angbr",
        "PIE root box",
        "non-gloss",
        "displaced",
        "alter",
        "etydate",
    ],
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


# --- wyrd-fqil: forms array → etymon_variant rows --------------------


def _wiktextract_entry_with_forms(
    *,
    word: str,
    lang_code: str,
    forms: list[dict],
) -> str:
    record = {"word": word, "lang_code": lang_code, "pos": "noun", "forms": forms}
    return json.dumps(record) + "\n"


def _all_etymon_variants(db: LexiconDB) -> list[tuple[str, str, str, str]]:
    """Return (lemma_form, variant_form, variant_class, tags_json) tuples.
    Sorted for stable assertions."""
    rows = db.conn.execute(
        """SELECT e.canonical_form AS lemma, v.form, v.variant_class, v.tags
           FROM etymon_variant v
           JOIN etymon e ON e.id = v.etymon_id
           ORDER BY lemma, v.form, v.variant_class"""
    ).fetchall()
    return [(r["lemma"], r["form"], r["variant_class"], r["tags"]) for r in rows]


def test_forms_alternative_tag_emits_alternative_variant(fresh_db: Path) -> None:
    """`tags: ['alternative']` → variant_class='alternative'."""
    line = _wiktextract_entry_with_forms(
        word="harpe",
        lang_code="enm",
        forms=[{"form": "harp", "tags": ["alternative"]}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        variants = _all_etymon_variants(db)
    assert result["forms_emitted"] == 1
    assert ("harpe", "harp", "alternative", '["alternative"]') in variants


def test_forms_inflection_tag_emits_inflection_variant(fresh_db: Path) -> None:
    """A grammatical-feature tag like `dative_or_pl` or `genitive` →
    variant_class='inflection'."""
    line = _wiktextract_entry_with_forms(
        word="word",
        lang_code="ang",
        forms=[
            {"form": "wordes", "tags": ["genitive", "singular"]},
            {"form": "worde", "tags": ["dative", "singular"]},
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        variants = _all_etymon_variants(db)
    assert result["forms_emitted"] == 2
    classes = {v[2] for v in variants}
    assert classes == {"inflection"}
    forms = {v[1] for v in variants}
    assert forms == {"wordes", "worde"}


def test_forms_romanization_tag_emits_romanization_variant(fresh_db: Path) -> None:
    line = _wiktextract_entry_with_forms(
        word="ἅρπυια",
        lang_code="grc",
        forms=[{"form": "hárpuia", "tags": ["romanization"]}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        variants = _all_etymon_variants(db)
    assert result["forms_emitted"] == 1
    assert variants == [("ἅρπυια", "hárpuia", "romanization", '["romanization"]')]


def test_forms_pure_metadata_rows_are_skipped(fresh_db: Path) -> None:
    """Forms tagged only with metadata (`table-tags`, `inflection-template`,
    `class`) are NOT emitted — they're not real word variants."""
    line = _wiktextract_entry_with_forms(
        word="word",
        lang_code="ang",
        forms=[
            {"form": "no-table-tags", "tags": ["table-tags"]},
            {"form": "ang-decl-noun-a-n", "tags": ["inflection-template"]},
            {"form": "Strong neuter", "tags": ["class"]},
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        variants = _all_etymon_variants(db)
    assert result["forms_emitted"] == 0
    assert result["forms_skipped_metadata"] == 3
    assert variants == []


def test_forms_self_reference_to_lemma_is_skipped(fresh_db: Path) -> None:
    """Wiktextract sometimes emits a form row equal to the lemma word
    itself (often tagged `canonical`). Skip — already represented by
    the etymon row's own canonical_form."""
    line = _wiktextract_entry_with_forms(
        word="word",
        lang_code="ang",
        forms=[{"form": "word", "tags": ["canonical"]}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        variants = _all_etymon_variants(db)
    assert result["forms_emitted"] == 0
    assert result["forms_skipped_unemittable"] == 1
    assert variants == []


def test_forms_empty_form_string_is_skipped(fresh_db: Path) -> None:
    """Defensive: empty/whitespace `form` strings are dropped."""
    line = _wiktextract_entry_with_forms(
        word="word",
        lang_code="ang",
        forms=[{"form": "", "tags": ["alternative"]}, {"form": "   ", "tags": ["plural"]}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
    assert result["forms_emitted"] == 0
    assert result["forms_skipped_unemittable"] == 2


def test_forms_dry_run_counts_but_doesnt_write(fresh_db: Path) -> None:
    """apply=False counts forms but doesn't insert into etymon_variant."""
    line = _wiktextract_entry_with_forms(
        word="harpe",
        lang_code="enm",
        forms=[{"form": "harp", "tags": ["alternative"]}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=False)
        n = db.conn.execute("SELECT COUNT(*) FROM etymon_variant").fetchone()[0]
    assert result["forms_emitted"] == 1
    assert n == 0


def test_forms_re_ingest_is_idempotent(fresh_db: Path) -> None:
    """Re-running the same JSONL line over the etymon_variant UNIQUE
    constraint produces no duplicate rows."""
    line = _wiktextract_entry_with_forms(
        word="harpe",
        lang_code="enm",
        forms=[{"form": "harp", "tags": ["alternative"]}],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        n = db.conn.execute("SELECT COUNT(*) FROM etymon_variant").fetchone()[0]
    assert n == 1


def test_forms_lookup_is_case_insensitive(fresh_db: Path) -> None:
    """COLLATE NOCASE on form: 'harp' and 'HARP' match the same row."""
    line = _wiktextract_entry_with_forms(
        word="harpe",
        lang_code="enm",
        forms=[{"form": "harp", "tags": ["alternative"]}],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        rows = db.conn.execute("SELECT COUNT(*) FROM etymon_variant WHERE form = 'HARP'").fetchone()
    assert rows[0] == 1


def test_forms_classify_variant_function() -> None:
    """Unit-test the tag-classification function directly."""
    assert _classify_form_variant(["alternative"]) == "alternative"
    assert _classify_form_variant(["romanization"]) == "romanization"
    assert _classify_form_variant(["canonical"]) == "canonical"
    assert _classify_form_variant(["plural"]) == "inflection"
    assert _classify_form_variant(["genitive", "singular"]) == "inflection"
    assert _classify_form_variant(["table-tags"]) is None
    assert _classify_form_variant(["inflection-template"]) is None
    assert _classify_form_variant([]) == "other"
    # Multi-tag with both alternative and inflectional: 'alternative' wins
    assert _classify_form_variant(["alternative", "plural"]) == "alternative"


def test_forms_template_marker_prefix_is_skipped(fresh_db: Path) -> None:
    """`#`-prefixed and `-`-prefixed form strings (template-marker
    artifacts) are dropped before INSERT — _is_emittable_form filter."""
    line = _wiktextract_entry_with_forms(
        word="word",
        lang_code="ang",
        forms=[
            {"form": "#hash-prefix", "tags": ["alternative"]},
            {"form": "-dash-prefix", "tags": ["alternative"]},
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
    assert result["forms_emitted"] == 0
    assert result["forms_skipped_unemittable"] == 2


def test_forms_canonical_class_round_trips_through_insert(fresh_db: Path) -> None:
    """A 'canonical' tag on a form whose value differs from the lemma
    word is emitted with variant_class='canonical' and survives the
    CHECK constraint at INSERT time."""
    line = _wiktextract_entry_with_forms(
        word="ἅρπυια",
        lang_code="grc",
        # Canonical-tagged spelling variant (e.g. macron'd vs not).
        forms=[{"form": "ἅρπυιᾰ", "tags": ["canonical"]}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        variants = _all_etymon_variants(db)
    assert result["forms_emitted"] == 1
    assert variants == [("ἅρπυια", "ἅρπυιᾰ", "canonical", '["canonical"]')]


def test_forms_other_class_round_trips_through_insert(fresh_db: Path) -> None:
    """A form with tags that don't match alternative/romanization/
    canonical/inflection lands as variant_class='other' and survives
    the CHECK constraint."""
    line = _wiktextract_entry_with_forms(
        word="word",
        lang_code="ang",
        # 'sense:foo' is not in the inflection set or any of the named
        # alternative/romanization/canonical buckets.
        forms=[{"form": "wordlet", "tags": ["sense:foo"]}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        variants = _all_etymon_variants(db)
    assert result["forms_emitted"] == 1
    assert variants == [("word", "wordlet", "other", '["sense:foo"]')]


# ---------------------------------------------------------------------
# wyrd-ha9q Phase 2a: pronunciation + multi-script capture
# ---------------------------------------------------------------------


def _wiktextract_entry_full(
    *,
    word: str,
    lang_code: str,
    sounds: list[dict] | None = None,
    head_templates: list[dict] | None = None,
    senses: list[dict] | None = None,
) -> str:
    """Build a wiktextract JSONL line carrying optional `sounds`,
    `head_templates`, and `senses` arrays — the sections the ingester
    reads for Phase 2a (pronunciation + multi-script) and wyrd-vsvi
    (sense-category tag extraction)."""
    record: dict = {"word": word, "lang_code": lang_code, "pos": "noun"}
    if sounds is not None:
        record["sounds"] = sounds
    if head_templates is not None:
        record["head_templates"] = head_templates
    if senses is not None:
        record["senses"] = senses
    return json.dumps(record) + "\n"


def _etymon_tags(db: LexiconDB, canonical_form: str, language: str) -> list[str]:
    """Fetch the tags attached to an etymon (wyrd-vsvi assertion helper)."""
    row = db.conn.execute(
        "SELECT id FROM etymon WHERE canonical_form = ? AND language = ?",
        (canonical_form, language),
    ).fetchone()
    if row is None:
        return []
    return sorted(
        r["tag"]
        for r in db.conn.execute("SELECT tag FROM etymon_tag WHERE etymon_id = ?", (row["id"],))
    )


def _etymon_phase2a_cols(
    db: LexiconDB, canonical_form: str, language: str
) -> tuple[str | None, str | None, str | None, str | None]:
    row = db.conn.execute(
        """SELECT pronunciation_ipa, pronunciation_dialect,
                  original_script, transliteration
           FROM etymon
           WHERE canonical_form = ? AND language = ?""",
        (canonical_form, language),
    ).fetchone()
    if row is None:
        return (None, None, None, None)
    return (
        row["pronunciation_ipa"],
        row["pronunciation_dialect"],
        row["original_script"],
        row["transliteration"],
    )


def test_extract_pronunciation_prefers_untagged_ipa() -> None:
    """When the sounds array contains both an undialected entry and
    dialected ones, the undialected entry wins as the canonical
    pronunciation. dialect tag returns None in that case."""
    sounds = [
        {"ipa": "/ʤɪnː/"},  # untagged
        {"tags": ["Modern Standard Arabic"], "ipa": "/d͡ʒɪnː/"},
    ]
    assert _extract_pronunciation(sounds) == ("/ʤɪnː/", None)


def test_extract_pronunciation_falls_back_to_first_dialected_when_no_untagged() -> None:
    """Common wave-2 case: every sound entry is dialect-tagged. First
    one wins, with the first tag captured as pronunciation_dialect."""
    sounds = [
        {"tags": ["Biblical-Hebrew"], "ipa": "/kalb/"},
        {"tags": ["Yemenite-Hebrew"], "ipa": "/ˈka.lav/"},
    ]
    assert _extract_pronunciation(sounds) == ("/kalb/", "Biblical-Hebrew")


def test_extract_pronunciation_skips_entries_without_ipa() -> None:
    """A sounds array entry without an `ipa` key is just metadata
    (audio file, rhyme listing) — skip past it to the first IPA."""
    sounds = [
        {"audio": "Ar-jinn.ogg"},  # no ipa
        {"tags": ["MSA"], "ipa": "/d͡ʒɪnː/"},
    ]
    assert _extract_pronunciation(sounds) == ("/d͡ʒɪnː/", "MSA")


def test_extract_pronunciation_empty_returns_none_pair() -> None:
    assert _extract_pronunciation([]) == (None, None)


def test_extract_head_template_renderings_pulls_wv_and_tr() -> None:
    """Hebrew shape: wv = vocalized native, tr = academic Latin."""
    head_templates = [
        {
            "name": "he-noun",
            "args": {
                "g": "m",
                "tr": "kɛ́lɛḇ, kélev",
                "wv": "כֶּלֶב",
                "pl": "כְּלָבִים",
            },
            "expansion": "...",
        }
    ]
    assert _extract_head_template_renderings(head_templates, "כלב") == (
        "כֶּלֶב",
        "kɛ́lɛḇ, kélev",
    )


def test_extract_head_template_renderings_drops_wv_equal_to_lemma() -> None:
    """When wv equals the lemma word verbatim there's no extra info
    to surface — return None for original_script so the SPA falls
    back to canonical_form (which is identical)."""
    head_templates = [{"name": "he-noun", "args": {"wv": "כלב", "tr": "kelev"}}]
    assert _extract_head_template_renderings(head_templates, "כלב") == (None, "kelev")


def test_extract_head_template_renderings_handles_missing_args() -> None:
    """Templates that lack `args` (or have a non-dict args field —
    seen rarely in malformed entries) return (None, None)."""
    assert _extract_head_template_renderings([{"name": "he-noun"}], "lemma") == (None, None)
    assert _extract_head_template_renderings([{"name": "he-noun", "args": ["bad"]}], "lemma") == (
        None,
        None,
    )


def test_extract_head_template_renderings_falls_back_to_head_when_no_wv() -> None:
    """Arabic / Egyptian shape: no `wv`, but `head` carries the
    vocalized native form (Arabic harakat) or hieroglyphic markup."""
    # Arabic: `head` = vocalized form different from lemma.
    arabic = [{"name": "ar-noun", "args": {"head": "الْعَرَبِيَّة"}}]
    assert _extract_head_template_renderings(arabic, "العربية") == ("الْعَرَبِيَّة", None)
    # Egyptian: `head` = hieroglyphic markup.
    egyptian = [{"name": "egy-noun", "args": {"head": "<hiero>g:p-mw</hiero>"}}]
    assert _extract_head_template_renderings(egyptian, "gp") == ("<hiero>g:p-mw</hiero>", None)


def test_extract_head_template_renderings_wv_beats_head_when_both_present() -> None:
    """If a template carries both `wv` and `head`, the cleaner `wv`
    convention wins (it's the unambiguous canonical form)."""
    head_templates = [{"name": "he-noun", "args": {"wv": "כֶּלֶב", "head": "OTHER"}}]
    orig, _ = _extract_head_template_renderings(head_templates, "כלב")
    assert orig == "כֶּלֶב"


def test_extract_head_template_renderings_rejects_multiform_head() -> None:
    """Arabic editors sometimes pack multiple alt forms into one
    `head` string with `/` separators (e.g. `'و / و'`). Don't surface
    those as a single form — return None and let the SPA fall back to
    canonical_form."""
    head_templates = [{"name": "ar-noun", "args": {"head": "و / و"}}]
    assert _extract_head_template_renderings(head_templates, "و") == (None, None)


def test_extract_head_template_renderings_rejects_affix_marker_head() -> None:
    """Affix markers (leading/trailing tatweel `ـ` or hyphen) signal
    a partial form, not a complete native-script word."""
    head_templates_pre = [{"name": "ar-noun", "args": {"head": "ـت"}}]
    assert _extract_head_template_renderings(head_templates_pre, "ت")[0] is None
    head_templates_post = [{"name": "ar-noun", "args": {"head": "test-"}}]
    assert _extract_head_template_renderings(head_templates_post, "test")[0] is None


def test_extract_head_template_renderings_rejects_reconstructed_form() -> None:
    """Reconstructed forms (`*tūnaz` style) shouldn't land in
    original_script — they have no native script to begin with, and
    the `*` would confuse a SPA panel that expects renderable text."""
    head_templates = [{"name": "gem-pro-noun", "args": {"head": "*tūnaz"}}]
    assert _extract_head_template_renderings(head_templates, "*tūnaz")[0] is None


def test_extract_head_template_renderings_first_template_wins() -> None:
    """Multiple head_templates entries (alternative POS / sense
    headers) — only the first one is read. Avoids merging metadata
    across distinct senses that wiktextract files separately."""
    head_templates = [
        {"name": "he-noun", "args": {"wv": "כֶּלֶב", "tr": "kélev"}},
        {"name": "he-noun", "args": {"wv": "OTHER", "tr": "other"}},
    ]
    orig, translit = _extract_head_template_renderings(head_templates, "כלב")
    assert orig == "כֶּלֶב"
    assert translit == "kélev"


def test_extract_head_template_renderings_empty_list() -> None:
    assert _extract_head_template_renderings([], "lemma") == (None, None)


def test_ingest_extracts_tags_from_sense_categories(fresh_db: Path) -> None:
    """wyrd-vsvi: a wiktextract entry with senses+categories must
    surface mapped tags on the etymon. Pin: the mapping mirrors the
    corpus-miner path's _map_categories_to_tags so the two ingesters
    produce comparable tag coverage."""
    line = _wiktextract_entry_full(
        word="bridge",
        lang_code="en",
        senses=[
            {
                "glosses": ["a structure"],
                "categories": [{"name": "Architecture"}, {"name": "Topography"}],
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        tags = _etymon_tags(db, "bridge", "modern-english")
    # 'Architecture' → architecture tag; 'Topography' → topography tag.
    assert tags == ["architecture", "topography"]
    assert result["tags_added"] == 2
    assert result["entries_with_tags"] == 1


def test_ingest_unions_tags_across_multiple_senses(fresh_db: Path) -> None:
    """An entry with multiple senses spanning different semantic
    classes (e.g. 'mill' = grinding building AND a unit of measure)
    gets the union of all mapped tags. Distinct from the corpus-miner
    path which picks ONE canonical sense; the full ingester unions
    across all senses for richer tag coverage."""
    line = _wiktextract_entry_full(
        word="mill",
        lang_code="en",
        senses=[
            {
                "glosses": ["a grinding building"],
                "categories": [{"name": "Buildings"}, {"name": "Agriculture"}],
            },
            {
                "glosses": ["1/1000 of a unit"],
                "categories": [{"name": "Units of measure"}],
            },
        ],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        tags = _etymon_tags(db, "mill", "modern-english")
    # Buildings → architecture; Agriculture → agriculture; Units of
    # measure isn't in the current pattern table (correctly absent).
    # The point: tags from BOTH senses appear, not just sense [0].
    assert "architecture" in tags
    assert "agriculture" in tags


def test_ingest_skips_tag_writes_for_entries_without_categories(
    fresh_db: Path,
) -> None:
    """Entries without senses, or with senses but no categories, leave
    the etymon untagged. Counter stays at 0. Common case for etymology-
    template stub rows that the wiktextract dump references via the
    etym-templates section but doesn't carry a full sense list for."""
    line = _wiktextract_entry_full(
        word="placeholder",
        lang_code="en",
        senses=[{"glosses": ["a placeholder"]}],  # no categories
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        tags = _etymon_tags(db, "placeholder", "modern-english")
    assert tags == []
    assert result["tags_added"] == 0
    assert result["entries_with_tags"] == 0


def test_ingest_idempotent_on_tag_rerun(fresh_db: Path) -> None:
    """Re-running ingest with the same entry doesn't duplicate tag rows.
    Add-tag uses INSERT OR IGNORE under the hood; pin so a future
    schema change can't silently allow duplicates."""
    line = _wiktextract_entry_full(
        word="forest",
        lang_code="en",
        senses=[{"glosses": ["wooded area"], "categories": [{"name": "Trees"}]}],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        ingest_wiktextract_stream(db, _stream(line), apply=True)
        tags = _etymon_tags(db, "forest", "modern-english")
    # 'Trees' maps to ['plant', 'tree']; two runs but only one row each.
    assert tags == ["plant", "tree"]


def test_ingest_persists_pronunciation_and_renderings_end_to_end(fresh_db: Path) -> None:
    """End-to-end: a wiktextract entry with sounds + head_templates lands
    on the etymon row with all four Phase 2a columns populated. Counters
    in the result dict track per-field capture."""
    line = _wiktextract_entry_full(
        word="כלב",
        lang_code="he",
        sounds=[{"tags": ["Biblical-Hebrew"], "ipa": "/kalb/"}],
        head_templates=[
            {
                "name": "he-noun",
                "args": {"g": "m", "tr": "kɛ́lɛḇ, kélev", "wv": "כֶּלֶב"},
            }
        ],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        ipa, dialect, orig, translit = _etymon_phase2a_cols(db, "כלב", "he")
    assert ipa == "/kalb/"
    assert dialect == "Biblical-Hebrew"
    assert orig == "כֶּלֶב"
    assert translit == "kɛ́lɛḇ, kélev"
    assert result["pronunciation_captured"] == 1
    assert result["original_script_captured"] == 1
    assert result["transliteration_captured"] == 1


def test_ingest_pronunciation_columns_default_null_for_latin_entries(fresh_db: Path) -> None:
    """Latin / OE / etc. wiktextract entries usually carry no `sounds`
    array and no head_templates with wv/tr — those rows must end up
    with NULL in the new columns, not crash on missing keys."""
    line = _wiktextract_entry_full(
        word="tūn",
        lang_code="ang",
        sounds=None,
        head_templates=[{"name": "ang-noun", "args": {"head": "tūn", "g": "n"}}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=True)
        ipa, dialect, orig, translit = _etymon_phase2a_cols(db, "tūn", "old-english")
    assert (ipa, dialect, orig, translit) == (None, None, None, None)
    assert result["pronunciation_captured"] == 0
    assert result["original_script_captured"] == 0


def test_ingest_upsert_does_not_overwrite_richer_existing_pronunciation(fresh_db: Path) -> None:
    """COALESCE-on-conflict invariant: if an etymon already has IPA
    populated (e.g. from a prior richer source), a re-ingest of an
    entry that lacks IPA must NOT clobber the existing value with NULL.
    Symmetrically, an existing NULL gets filled by a later non-NULL
    re-ingest. This is the wave-2 backfill story."""
    # First ingest: poor-data entry, no sounds.
    poor = _wiktextract_entry_full(
        word="جن",
        lang_code="ar",
        sounds=None,
        head_templates=[{"name": "ar-noun", "args": {}}],
    )
    # Second ingest: rich-data entry on the SAME (canonical_form, language).
    rich = _wiktextract_entry_full(
        word="جن",
        lang_code="ar",
        sounds=[{"tags": ["MSA"], "ipa": "/d͡ʒɪnː/"}],
        head_templates=[{"name": "ar-noun", "args": {"tr": "jinn"}}],
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(poor), apply=True)
        # Pre-second-ingest state: NULLs.
        assert _etymon_phase2a_cols(db, "جن", "ar") == (None, None, None, None)
        ingest_wiktextract_stream(db, _stream(rich), apply=True)
        # Post-second-ingest: rich values backfilled.
        assert _etymon_phase2a_cols(db, "جن", "ar") == ("/d͡ʒɪnː/", "MSA", None, "jinn")
        # Now a THIRD ingest of the poor version — must NOT clobber
        # the rich values.
        ingest_wiktextract_stream(db, _stream(poor), apply=True)
        assert _etymon_phase2a_cols(db, "جن", "ar") == ("/d͡ʒɪnː/", "MSA", None, "jinn")


def test_is_clean_native_form_rejects_comma_separated_alt_forms() -> None:
    """`'كلب, كلاب'` (singular + plural in one head string) is one of
    the multi-form display patterns `_NATIVE_FORM_REJECT_TOKENS`
    blocks. Slash-separated is covered by another test; this pins the
    comma path."""
    assert _is_clean_native_form("كلب, كلاب", "كلب") is False


def test_is_clean_native_form_rejects_reconstructed_form_anywhere() -> None:
    """The `*` rejection isn't restricted to position 0 — any `*`
    in the value disqualifies it (reconstructed forms have no native
    script render). Lemma is a non-`*` value here so the equality
    short-circuit doesn't fire and we exercise the token-rejection
    path directly."""
    assert _is_clean_native_form("*tūnaz", "tūnaz") is False
    # Embedded `*` (not at position 0) — same rejection applies.
    assert _is_clean_native_form("foo*bar", "foo") is False


def test_is_clean_native_form_accepts_clean_value() -> None:
    """Sanity: a non-empty value, distinct from the lemma, with no
    affix markers and no reject tokens, passes the filter."""
    assert _is_clean_native_form("الْعَرَبِيَّة", "العربية") is True


def test_extract_pronunciation_skips_non_dict_sound_entries() -> None:
    """Defensive style mirroring `_walk_descendants`: a malformed
    `sounds` array containing a non-dict (string, None, list) should
    not crash the ingester — skip the bad element and keep walking."""
    sounds = [
        "not-a-dict",  # noqa: E501
        None,
        {"ipa": "/d͡ʒɪnː/"},
    ]
    assert _extract_pronunciation(sounds) == ("/d͡ʒɪnː/", None)


def test_extract_head_template_renderings_skips_non_dict_first_entry() -> None:
    """Same defensive style: head_templates[0] is sometimes None or
    a string in malformed wiktextract output. Return (None, None)
    rather than crashing the ingester."""
    assert _extract_head_template_renderings([None, {"args": {"wv": "כֶּלֶב"}}], "כלב") == (
        None,
        None,
    )
    assert _extract_head_template_renderings(["bad"], "lemma") == (None, None)


def test_extract_head_template_renderings_first_template_no_wv_returns_none() -> None:
    """First-template-wins extends to the case where template[0]
    has neither `wv` NOR `head` populated, while template[1] has them.
    We must NOT fall through — that would risk merging metadata across
    distinct senses that wiktextract files separately."""
    head_templates = [
        {"name": "he-noun", "args": {"g": "m"}},  # no wv, no head
        {"name": "he-noun", "args": {"wv": "OTHER", "head": "ALT", "tr": "alt"}},
    ]
    # Both fields come from template[0]'s args; since template[0] has
    # neither wv nor head, original_script is None. Same for tr.
    assert _extract_head_template_renderings(head_templates, "lemma") == (None, None)


def test_ingest_pronunciation_ipa_and_dialect_update_atomically(fresh_db: Path) -> None:
    """The IPA + dialect pair updates as a unit. Adversarial flow: a
    first ingest writes IPA-only (no tags → dialect=NULL); a second
    ingest brings BOTH a different IPA and a dialect tag.

    Without atomicity (per-column COALESCE), the second pass would
    leave the OLD IPA in place but write the NEW dialect, producing a
    row whose dialect describes an IPA we don't store. With the CASE-
    on-existing-IPA in upsert_etymon, the second pass sees an existing
    IPA → keeps BOTH the existing IPA and existing (NULL) dialect,
    preserving the pair."""
    # First ingest: IPA only, no dialect.
    ipa_only = _wiktextract_entry_full(
        word="כלב",
        lang_code="he",
        sounds=[{"ipa": "/kalb/"}],  # untagged
        head_templates=None,
    )
    # Second ingest: different IPA + a dialect tag.
    different = _wiktextract_entry_full(
        word="כלב",
        lang_code="he",
        sounds=[{"tags": ["Yemenite-Hebrew"], "ipa": "/ˈka.lav/"}],
        head_templates=None,
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(ipa_only), apply=True)
        assert _etymon_phase2a_cols(db, "כלב", "he") == ("/kalb/", None, None, None)
        ingest_wiktextract_stream(db, _stream(different), apply=True)
        # Existing IPA wins; dialect stays NULL because the existing
        # IPA was untagged. The new (IPA, dialect) pair is dropped
        # together rather than letting the dialect leak in alone.
        ipa, dialect, _, _ = _etymon_phase2a_cols(db, "כלב", "he")
        assert ipa == "/kalb/"
        assert dialect is None


def test_ingest_pronunciation_pair_fills_when_existing_ipa_is_null(fresh_db: Path) -> None:
    """Symmetric case: when the existing row has no IPA at all, both
    fields take the incoming pair atomically — backfills the NULL."""
    no_pron = _wiktextract_entry_full(
        word="جن",
        lang_code="ar",
        sounds=None,
        head_templates=None,
    )
    with_pair = _wiktextract_entry_full(
        word="جن",
        lang_code="ar",
        sounds=[{"tags": ["MSA"], "ipa": "/d͡ʒɪnː/"}],
        head_templates=None,
    )
    with LexiconDB(fresh_db) as db:
        ingest_wiktextract_stream(db, _stream(no_pron), apply=True)
        assert _etymon_phase2a_cols(db, "جن", "ar") == (None, None, None, None)
        ingest_wiktextract_stream(db, _stream(with_pair), apply=True)
        ipa, dialect, _, _ = _etymon_phase2a_cols(db, "جن", "ar")
        assert ipa == "/d͡ʒɪnː/"
        assert dialect == "MSA"


def test_ingest_dry_run_does_not_set_any_phase2a_counters(fresh_db: Path) -> None:
    """Dry-run mode (apply=False): all three Phase 2a counters stay
    at 0, AND no etymon row is written. Pins both contracts in one
    place so neither can regress silently."""
    line = _wiktextract_entry_full(
        word="כלב",
        lang_code="he",
        sounds=[{"ipa": "/kalb/"}],
        head_templates=[{"name": "he-noun", "args": {"wv": "כֶּלֶב", "tr": "kelev"}}],
    )
    with LexiconDB(fresh_db) as db:
        result = ingest_wiktextract_stream(db, _stream(line), apply=False)
        # No row written — dry-run skips the upsert entirely.
        row = db.conn.execute("SELECT 1 FROM etymon WHERE canonical_form = ?", ("כלב",)).fetchone()
    assert row is None
    assert result["pronunciation_captured"] == 0
    assert result["original_script_captured"] == 0
    assert result["transliteration_captured"] == 0


def test_phase2a_columns_present_on_fresh_db(fresh_db: Path) -> None:
    """Schema sanity: the five Phase 2a columns exist on a freshly-
    initialized DB. Catches a missing entry in lexicon.sql / migration."""
    with LexiconDB(fresh_db) as db:
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
    for col in (
        "pronunciation_ipa",
        "pronunciation_dialect",
        "original_script",
        "transliteration",
        "english_shaped",
    ):
        assert col in cols, f"{col} missing from etymon schema"
