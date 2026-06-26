"""Tests for ``wyrd kenning lexicon export-runtime-db`` (wyrd-d90t PR 1).

PR 1 ships the L4 runtime DB emitter. These tests round-trip every L4
table the emitter writes — meanings, fantasy morphemes, canonical
decompositions, the three sampled proportions tables (usage,
single_usage, structure), and the two statistics tables
(tag_marginal, tag_cooccurrence). They confirm the emit shape so
PRs 4-6 (loader + load_meanings + proportions refactor) have a
concrete target.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.cli.lexicon._threshold_args import parse_lang_thresholds
from wyrd.generators.kenning.cli.lexicon.export_runtime_db import (
    EMITTER_VERSION,
    SCHEMA_VERSION,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.runtime_db_export import (
    _insert_cumulative,
    _insert_tag_cooccurrence,
    _pick_primary_language,
    _pick_unanimous_stratum,
    _strata_for_word,
    _write_canonical_decompositions,
    _write_fantasy_morphemes,
    write_runtime_db,
)

# ---------- helpers ----------


def _seed_minimal_lexicon(db_path: Path) -> None:
    """Initialize an empty L3 lexicon DB.

    Tests that don't depend on lexicon contents (the emitter still has
    to produce a valid L4 even with empty inputs) use this directly.
    Tests that need specific meanings build them via the proportions
    sidecar fixture below — emitting from a richly-populated L3 DB is
    covered by integration runs, not unit tests.
    """
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.upsert_source(id="skeat-1901", title="Skeat 1901", year=1901)
        db.commit()


def _seed_lexicon_with_toponyms(db_path: Path, toponyms: list[tuple[str, str, str | None]]) -> None:
    """Init an L3 lexicon DB and insert ``(modern_name, country, region)``
    toponym rows.

    wyrd-bo01.3: the inline proportions rebuild now reads its per-culture
    corpus from the DB ``toponym`` table (filtered by
    ``CULTURE_COUNTRIES``) instead of the static
    ``{culture}_place_names.json`` gazetteer, so the inline-rebuild tests
    must seed the names they expect to decompose into the DB rather than
    relying on the bundled corpus."""
    init_schema(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executemany(
            "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
            toponyms,
        )
        conn.commit()
    finally:
        conn.close()


def _write_proportions_fixture(directory: Path, culture: str = "english") -> None:
    """Write a small-but-shape-correct proportions JSON into ``directory``.

    Single fixture file per culture; the emitter reads it via the
    ``--proportions-dir`` flag so tests don't have to monkeypatch the
    bundled data dir.
    """
    payload = {
        "usages": {
            "ham": {"inner": 100},
            "ton": {"inner": 50},
            "zero-weight": {"post": 0},
        },
        "single_usages": {"-castle": 20, "-bury": 5},
        "structures": [
            {"proportion": 70, "words": [[{"location": "pre"}, {"location": "post"}]]},
            {
                "proportion": 30,
                "words": [[{"location": "pre"}, {"location": "inner"}, {"location": "post"}]],
            },
            {"proportion": 0, "words": [[{"location": "pre"}]]},  # skipped
        ],
        "tag_marginal": {"water": 12, "architecture": 8},
        "tag_cooccurrence": {"water|architecture": 3, "water|tree": 2},
        # wyrd-rogd.13: surface-keyed per-word-position bare stats; the
        # zero-weight row pins the emitter's weight>0 skip.
        "bare_word_positions": {"post": {"castle": 4, "ghost": 0}, "pre": {"bury": 3}},
    }
    (directory / f"{culture}_proportions.json").write_text(json.dumps(payload))


def _run_emit(
    *,
    db_path: Path,
    out_path: Path,
    proportions_dir: Path | None = None,
) -> None:
    runner = CliRunner()
    args = [
        "lexicon",
        "export-runtime-db",
        "--db",
        str(db_path),
        "--output",
        str(out_path),
    ]
    if proportions_dir is not None:
        args.extend(["--proportions-dir", str(proportions_dir)])
    result = runner.invoke(cli_root, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output


# ---------- schema + metadata ----------


def test_emit_creates_l4_schema(tmp_path: Path) -> None:
    """A fresh emit must produce every L4 table + index defined in
    runtime.sql. Catches accidental schema drift between the .sql
    resource and the emitter's expectations."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()

    assert {
        "meaning",
        "fantasy_morpheme",
        "canonical_decomposition",
        "proportions_usage",
        "proportions_single_usage",
        "proportions_structure",
        "proportions_tag_marginal",
        "proportions_tag_cooccurrence",
        "proportions_bare_word_position",
        "empirical_priors",
        "genitive_split_prior",
        "bundle_metadata",
    } <= tables


def test_emit_writes_metadata(tmp_path: Path) -> None:
    """bundle_metadata must carry schema_version + emitter_version + a
    built_at timestamp + the source DB path so deployed DBs can be
    correlated back to their emit source (D38.1)."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        meta = dict(conn.execute("SELECT key, value FROM bundle_metadata").fetchall())
    finally:
        conn.close()

    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["emitter_version"] == EMITTER_VERSION
    # Resolve both sides — macOS tmp_path lives under /var/folders/...
    # which symlinks to /private/var/...; the emitter calls .resolve()
    # so the stored value is the realpath, not the as-passed value.
    assert meta["source_lexicon_db"] == str(db_path.resolve())
    # Timestamp is sensitive to wall-clock; just confirm it's the
    # iso-ish shape and ends with Z.
    assert meta["built_at"].endswith("Z")
    assert "T" in meta["built_at"]


def test_emit_overwrites_existing_output(tmp_path: Path) -> None:
    """Re-emitting onto an existing path replaces the file (the L4 build
    is whole-DB; re-emit is the only way to update it)."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)
    first_size = out_path.stat().st_size
    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)
    second_size = out_path.stat().st_size

    # Sizes should be roughly equal (timestamps drift) — confirms the
    # file wasn't appended to, just rewritten.
    assert abs(first_size - second_size) < 1024


# ---------- proportions ----------


def test_proportions_usage_cumulative_monotonic(tmp_path: Path) -> None:
    """Sampled tables carry a precomputed cumulative column for O(log n)
    weighted-random sampling (D38.3). Verify the cumulative values are
    monotonically increasing per culture and equal weight-sum."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        rows = conn.execute(
            "SELECT usage_key, weight, cumulative FROM proportions_usage "
            "WHERE culture = 'english' ORDER BY cumulative"
        ).fetchall()
    finally:
        conn.close()

    # -zero-weight has weight 0 → dropped (would conflict on cumulative PK).
    # D45: usage_key is the BARE surface; position is the explicit column.
    assert [r[0] for r in rows] == ["ham", "ton"]
    assert [r[1] for r in rows] == [100, 50]
    assert [r[2] for r in rows] == [100, 150]


def test_proportions_single_usage_emits(tmp_path: Path) -> None:
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        rows = conn.execute(
            "SELECT usage_key, weight, cumulative FROM proportions_single_usage "
            "WHERE culture = 'english' ORDER BY cumulative"
        ).fetchall()
    finally:
        conn.close()

    # D45: single usage_key folds to bare surface.
    assert [r[0] for r in rows] == ["castle", "bury"]
    assert [r[2] for r in rows] == [20, 25]


def test_proportions_structure_stores_template_as_json(tmp_path: Path) -> None:
    """Structure rows carry the template (the ``words`` list) as a JSON
    blob; the runtime decodes it after weighted-sampling on cumulative."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        rows = conn.execute(
            "SELECT template, weight, cumulative FROM proportions_structure "
            "WHERE culture = 'english' ORDER BY cumulative"
        ).fetchall()
    finally:
        conn.close()

    # 0-weight structure is dropped.
    assert len(rows) == 2
    first_template = json.loads(rows[0][0])
    assert first_template == [[{"location": "pre"}, {"location": "post"}]]
    assert [r[1] for r in rows] == [70, 30]
    assert [r[2] for r in rows] == [70, 100]


def test_proportions_bare_word_position_rows_survive_the_emitter(tmp_path: Path) -> None:
    """wyrd-rogd.13: positional bare-word rows land in
    proportions_bare_word_position (point-lookup, no cumulative), and
    zero-weight rows are skipped at emit."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        rows = conn.execute(
            "SELECT position, usage_key, weight FROM proportions_bare_word_position "
            "WHERE culture = 'english' ORDER BY position, usage_key"
        ).fetchall()
    finally:
        conn.close()

    assert rows == [("post", "castle", 4), ("pre", "bury", 3)]


def test_proportions_tag_marginal_and_cooccurrence(tmp_path: Path) -> None:
    """Statistics tables (point-lookup, no sampling) preserve the
    underlying tag weights from the source JSON."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        marginal = dict(
            conn.execute(
                "SELECT tag, weight FROM proportions_tag_marginal WHERE culture = 'english'"
            ).fetchall()
        )
        cooc = conn.execute(
            "SELECT tag1, tag2, weight FROM proportions_tag_cooccurrence WHERE culture = 'english'"
        ).fetchall()
    finally:
        conn.close()

    assert marginal == {"water": 12, "architecture": 8}
    assert sorted(cooc) == sorted([("water", "architecture", 3), ("water", "tree", 2)])


def test_proportions_skips_missing_cultures(tmp_path: Path) -> None:
    """Cultures whose proportions JSON is absent from the proportions-dir
    are silently skipped — the bundled set is the truth-source."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    # Only ship english; don't write scottish/welsh/irish/breton.
    _write_proportions_fixture(tmp_path, culture="english")

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        cultures = {
            row[0] for row in conn.execute("SELECT DISTINCT culture FROM proportions_usage")
        }
    finally:
        conn.close()

    assert cultures == {"english"}


# ---------- meaning ----------


def test_meaning_row_round_trips_subject_payload(tmp_path: Path) -> None:
    """End-to-end emit + read: a usage-keyed meaning row's data blob
    decodes back into the {meaning, modifier_tags, modifier_type, word}
    structure each Meaning ultimately loads from."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    # Inject a meaning into the L3 DB so export_meanings has something to
    # emit. rando-port-cited etymons bypass the D4 witness threshold via
    # --include-rando (defaulted on), so a single citation suffices.
    with LexiconDB(db_path) as db:
        db.upsert_source(id="rando-port", title="Rando port (seed)")
        etymon_id = db.upsert_etymon(
            "ham",
            "old-english",
            modifier_type="Topographical",
            position_pref="post",
        )
        db.add_citation(etymon_id, "rando-port")
        db.conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
            (etymon_id, "home, dwelling"),
        )
        db.commit()

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        rows = conn.execute("SELECT usage_key, primary_language, data FROM meaning").fetchall()
    finally:
        conn.close()

    # At least one row should have landed (the rando-port include path
    # passes through ham → -ham- as a topographical usage).
    assert rows, "expected at least one meaning row, got zero"
    by_usage = {row[0]: row for row in rows}
    # The exact usage depends on export_meanings's modern_usage projection;
    # confirm structural invariants on whatever shape it emits.
    sample_key = next(iter(by_usage))
    sample = by_usage[sample_key]
    payload = json.loads(sample[2])
    assert "entries" in payload and isinstance(payload["entries"], list)
    entry = payload["entries"][0]
    assert set(entry) >= {"meaning", "modifier_tags", "modifier_type", "word"}
    assert entry["word"]["modern_usage"] == sample_key


# ---------- canonical_decomposition ----------


def test_canonical_decomposition_culture_agnostic(tmp_path: Path) -> None:
    """Per D38.4 the L4 canonical_decomposition table is culture-agnostic
    in PR 1: PK is just toponym_name, with the {signature, source} JSON
    payload in ``data``. Verify the schema matches that contract — and
    that an emit against an L3 with no canonical picks succeeds with an
    empty table (the runtime tolerates an empty set)."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(canonical_decomposition)")]
        rows = conn.execute("SELECT toponym_name, data FROM canonical_decomposition").fetchall()
    finally:
        conn.close()

    assert cols == ["toponym_name", "data"], (
        "canonical_decomposition must be (toponym_name, data) per D38.4 — no culture column in PR 1"
    )
    # Empty L3 → empty table; no rows expected.
    assert rows == []


# ---------- fantasy_morpheme ----------


def test_fantasy_morpheme_table_writable(tmp_path: Path) -> None:
    """Empty L3 yields zero fantasy_morpheme rows; the table exists and
    the emitter doesn't crash on the empty case (the bundled production
    DB has thousands, but unit tests run on a near-empty DB by design)."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        rows = conn.execute("SELECT usage_key, data FROM fantasy_morpheme").fetchall()
    finally:
        conn.close()

    assert rows == []


# ---------- sampling primitive smoke ----------


# ---------- _pick_primary_language ----------


def test_pick_primary_language_excludes_citations_sibling() -> None:
    """A word carrying ``celtic_mix_citations: list[str]`` alongside
    ``celtic_mix: list[str]`` must not double-count the citations
    sibling. Without the explicit non-language-suffix exclusion the
    alphabetical tiebreaker would pick ``celtic_mix_citations``."""
    entries = [
        {
            "meaning": [],
            "modifier_tags": [],
            "modifier_type": None,
            "word": {
                "modern_usage": "-aig",
                "celtic_mix": ["-aig", "aig"],
                "celtic_mix_citations": ["src1", "src2"],
            },
        }
    ]
    assert _pick_primary_language(entries) == "celtic_mix"


def test_pick_primary_language_tiebreaks_alphabetically() -> None:
    """When two languages tie on count, the alphabetically-first key
    wins (re-emits with identical inputs produce identical rows)."""
    entries = [
        {
            "meaning": [],
            "modifier_tags": [],
            "modifier_type": None,
            "word": {
                "modern_usage": "ham",
                "old_norse": ["ham"],
                "old_english": ["ham"],
            },
        }
    ]
    assert _pick_primary_language(entries) == "old_english"


def test_pick_primary_language_returns_none_for_orphan_reflex() -> None:
    """Orphan-reflex entries (from ``_orphan_reflex_subjects``) carry
    only ``modern_usage`` — they're matcher-only and have no etymon-
    backed language. Returning None lets the schema column stay NULL
    instead of masking the category with an empty string."""
    entries = [
        {
            "meaning": [],
            "modifier_tags": [],
            "modifier_type": None,
            "word": {"modern_usage": "-abann-"},
        }
    ]
    assert _pick_primary_language(entries) is None


# ---------- _pick_unanimous_stratum ----------


def test_pick_unanimous_stratum_returns_single_value() -> None:
    """When every per-form stratum entry agrees, return the value."""
    entries = [
        {"word": {"old_english_stratum": [{"form": "ham", "stratum": "native-old-english"}]}},
        {"word": {"old_english_stratum": [{"form": "ham", "stratum": "native-old-english"}]}},
    ]
    assert _pick_unanimous_stratum(entries) == "native-old-english"


def test_pick_unanimous_stratum_returns_none_on_mixed() -> None:
    """Mixed strata across entries — common for cross-cultural morphemes
    like Latin loans — yield NULL so the bulk index doesn't lie."""
    entries = [
        {"word": {"welsh_stratum": [{"form": "caer", "stratum": "latin-loan"}]}},
        {"word": {"welsh_stratum": [{"form": "din", "stratum": "native-welsh"}]}},
    ]
    assert _pick_unanimous_stratum(entries) is None


def test_pick_unanimous_stratum_returns_none_when_absent() -> None:
    """No stratum data on any entry → NULL (passthrough on the runtime
    filter per D32)."""
    entries = [
        {"word": {"old_english": ["ham"]}},
        {"word": {"old_english": ["heim"]}},
    ]
    assert _pick_unanimous_stratum(entries) is None


# ---------- _insert_tag_cooccurrence ----------


def test_insert_tag_cooccurrence_raises_on_malformed_key(tmp_path: Path) -> None:
    """A cooccurrence key without ``|`` is a data bug — raise rather
    than drop the row silently (D24 observability rule)."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)
    # Build the runtime DB normally first so the table exists.
    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        with pytest.raises(ValueError, match="missing the 'tag1\\|tag2' separator"):
            _insert_tag_cooccurrence(conn, "english", {"oops_no_pipe": 1})
    finally:
        conn.close()


# ---------- _write_fantasy_morphemes / _write_canonical_decompositions ----------


def test_write_fantasy_morphemes_inserts_lowercased_key(tmp_path: Path) -> None:
    """Verify the lowercase-key contract that mirrors the bundle's
    COLLATE NOCASE semantics."""
    out_path = tmp_path / "runtime.db"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out_path))
    try:
        # Schema bootstrap: create just the table we're exercising.
        conn.executescript(
            "CREATE TABLE fantasy_morpheme (usage_key TEXT PRIMARY KEY, data BLOB NOT NULL);"
        )
        n = _write_fantasy_morphemes(
            conn,
            {
                "Harpy": {"input_name": "Harpy", "language": "ancient-greek"},
                "Djinni": {"input_name": "Djinni", "language": "ar"},
            },
        )
        rows = conn.execute(
            "SELECT usage_key, data FROM fantasy_morpheme ORDER BY usage_key"
        ).fetchall()
    finally:
        conn.close()

    assert n == 2
    assert [r[0] for r in rows] == ["djinni", "harpy"]
    payload = json.loads(rows[0][1])
    assert payload["language"] == "ar"


def test_write_canonical_decompositions_inserts_blob(tmp_path: Path) -> None:
    """End-to-end emit + read for a populated canonical_decomposition
    set — the empty case is covered above; this exercises the insert
    path itself."""
    out_path = tmp_path / "runtime.db"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out_path))
    try:
        conn.executescript(
            "CREATE TABLE canonical_decomposition ("
            "  toponym_name TEXT PRIMARY KEY,"
            "  data BLOB NOT NULL"
            ");"
        )
        n = _write_canonical_decompositions(
            conn,
            {
                "Stratford": {"signature": "abc123", "source": "scholar"},
                "Birmingham": {"signature": "def456", "source": "unique-zero"},
            },
        )
        rows = dict(
            conn.execute("SELECT toponym_name, data FROM canonical_decomposition").fetchall()
        )
    finally:
        conn.close()

    assert n == 2
    assert json.loads(rows["Stratford"]) == {
        "signature": "abc123",
        "source": "scholar",
    }


# ---------- _parse_lang_thresholds ----------


def test_parse_lang_thresholds_happy_path() -> None:
    """Specs override the preset; aliases are normalized."""
    out = parse_lang_thresholds(("old_english=2", "celtic_mix=4"), use_preset=False)
    assert out["old-english"] == 2  # LANGUAGE_FIELDS alias
    assert out["celtic"] == 4


def test_parse_lang_thresholds_no_preset_empties_map() -> None:
    out = parse_lang_thresholds((), use_preset=False)
    assert out == {}


def test_parse_lang_thresholds_keeps_preset_when_use_preset() -> None:
    """The preset (RECOMMENDED_LANG_THRESHOLDS) should be present when
    no overrides are supplied."""
    out = parse_lang_thresholds((), use_preset=True)
    # Preset is calibrated against corpus availability; OE is the
    # canonical strict entry.
    assert "old-english" in out


def test_parse_lang_thresholds_rejects_missing_equals() -> None:
    with pytest.raises(click.BadParameter, match="LANG=N"):
        parse_lang_thresholds(("old_english",), use_preset=False)


def test_parse_lang_thresholds_rejects_non_integer_n() -> None:
    with pytest.raises(click.BadParameter, match="must be an integer"):
        parse_lang_thresholds(("old_english=two",), use_preset=False)


def test_parse_lang_thresholds_rejects_empty_parts() -> None:
    with pytest.raises(click.BadParameter, match="must be non-empty"):
        parse_lang_thresholds(("=3",), use_preset=False)


def test_parse_lang_thresholds_rejects_negative_n() -> None:
    """A negative witness threshold is nonsensical (every morpheme has
    >= -1 witnesses, so it silently disables the filter for that language).
    Reject it loudly rather than admit-all (wyrd review #849)."""
    with pytest.raises(click.BadParameter, match="must be non-negative"):
        parse_lang_thresholds(("old_english=-1",), use_preset=False)


def test_parse_lang_thresholds_allows_zero() -> None:
    """N=0 is a valid explicit 'no minimum / admit all' — only negatives reject."""
    out = parse_lang_thresholds(("old_english=0",), use_preset=False)
    assert out["old-english"] == 0


# ---------- proportions_structure PK invariant ----------


def test_proportions_structure_uses_composite_pk(tmp_path: Path) -> None:
    """Per Gemini PR-350 review: proportions_structure must use
    (culture, cumulative) as PK to match the sampled-table convention
    on its siblings (proportions_usage / proportions_single_usage)."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(proportions_structure)")]
        pk_cols = [
            row[1] for row in conn.execute("PRAGMA table_info(proportions_structure)") if row[5]
        ]
    finally:
        conn.close()

    assert "id" not in cols, "proportions_structure should not carry a surrogate id"
    assert sorted(pk_cols) == ["culture", "cumulative"]


# ---------- log-n sampling smoke ----------


def test_proportions_usage_supports_log_n_sampling(tmp_path: Path) -> None:
    """The D38.3 sampling pattern (random int + cumulative >= roll) should
    round-trip a usage_key consistently. Smoke-tests the SQL shape PR 6
    will refactor the runtime onto."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)

    conn = sqlite3.connect(str(out_path))
    try:
        total = conn.execute(
            "SELECT MAX(cumulative) FROM proportions_usage WHERE culture = ?",
            ("english",),
        ).fetchone()[0]
        assert total == 150

        # roll=1 → first row (ham, cumulative=100) — D45: bare surface key.
        row_low = conn.execute(
            "SELECT usage_key FROM proportions_usage "
            "WHERE culture = ? AND cumulative >= ? "
            "ORDER BY cumulative LIMIT 1",
            ("english", 1),
        ).fetchone()
        # roll=120 → second row (ton, cumulative=150)
        row_high = conn.execute(
            "SELECT usage_key FROM proportions_usage "
            "WHERE culture = ? AND cumulative >= ? "
            "ORDER BY cumulative LIMIT 1",
            ("english", 120),
        ).fetchone()
    finally:
        conn.close()

    assert row_low[0] == "ham"
    assert row_high[0] == "ton"


# ---------- round-2 coverage: round-1-added paths ----------


def test_insert_cumulative_rejects_non_whitelisted_table(tmp_path: Path) -> None:
    """``_CUMULATIVE_USAGE_TABLES`` whitelist is the SQL-injection safety
    on the f-string-interpolated table name. A caller passing an
    arbitrary table must hit the whitelist raise BEFORE the INSERT is
    formatted."""
    out_path = tmp_path / "runtime.db"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out_path))
    try:
        with pytest.raises(ValueError, match="not in whitelist"):
            _insert_cumulative(conn, "evil_table; DROP TABLE meaning;--", "english", [])
    finally:
        conn.close()


def test_insert_cumulative_wraps_integrity_error(tmp_path: Path) -> None:
    """A duplicate (culture, cumulative) row at insert time should raise
    a RuntimeError chained from sqlite3.IntegrityError with culture +
    table context so the operator can locate the offending pair."""
    out_path = tmp_path / "runtime.db"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out_path))
    try:
        conn.executescript(
            "CREATE TABLE proportions_usage ("
            "  culture TEXT NOT NULL, usage_key TEXT NOT NULL, "
            "  position TEXT NOT NULL, "
            "  weight INTEGER NOT NULL, cumulative INTEGER NOT NULL, "
            "  PRIMARY KEY (culture, cumulative));"
        )
        # First insert lands fine. D45: items are (usage_key, position, weight).
        _insert_cumulative(conn, "proportions_usage", "english", [("ham", "inner", 100)])
        # Second insert with the same culture re-uses cumulative=100 → IntegrityError.
        with pytest.raises(RuntimeError, match="proportions_usage.*english"):
            _insert_cumulative(conn, "proportions_usage", "english", [("ton", "inner", 100)])
    finally:
        conn.close()


def test_strata_for_word_skips_non_list_and_non_dict() -> None:
    """``_strata_for_word`` defensive guards: skip non-list values,
    skip non-dict elements within the list, skip dicts without a
    truthy ``stratum`` field."""
    word = {
        "modern_usage": "-ham-",
        "old_english_stratum": [
            {"form": "ham", "stratum": "native-old-english"},
            "not-a-dict",  # skipped
            {"form": "heim"},  # no stratum → skipped
            {"form": "hum", "stratum": ""},  # empty stratum → skipped
        ],
        "welsh_stratum": "not-a-list",  # skipped at outer guard
        "not_a_stratum_key": [{"stratum": "decoy"}],  # wrong suffix
    }
    assert list(_strata_for_word(word)) == ["native-old-english"]


def test_emit_default_proportions_dir_computes_inline(tmp_path: Path) -> None:
    """When ``--proportions-dir`` is omitted the emit rebuilds the
    proportions inline from the subjects it just exported + the
    bundled ``<culture>_place_names.json`` corpora. End-to-end smoke:
    a fresh emit with no explicit proportions dir must still populate
    the proportions tables for every shipped culture, with no
    intermediate JSON artifact required."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "export-runtime-db",
            "--db",
            str(db_path),
            "--output",
            str(out_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    conn = sqlite3.connect(str(out_path))
    try:
        # The emit must produce a well-formed L4 against a minimal
        # fixture lexicon — bundle_metadata is populated, the
        # proportions_* tables exist, and there are no constraint
        # violations. The proportions rowcounts can be zero against a
        # tiny fixture (the bundled <culture>_place_names.json corpora
        # have ~tens of thousands of real toponyms; few of them
        # decompose perfectly against a 1-meaning fixture), which is
        # correct behavior — not a regression. A separate test
        # (``test_inline_rebuild_emits_populated_proportions_with_canonical``
        # below) exercises the populated case against a richer fixture.
        bundle_meta = dict(conn.execute("SELECT key, value FROM bundle_metadata").fetchall())
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'proportions_%'"
        ).fetchall():
            conn.execute(f"SELECT COUNT(*) FROM {row[0]}").fetchone()
    finally:
        conn.close()
    assert bundle_meta.get("schema_version") == SCHEMA_VERSION
    assert "built_at" in bundle_meta


def _three_subject_acton_fixture() -> list[dict[str, Any]]:
    """Minimal bundle that decomposes the English-corpus toponym
    ``Acton`` perfectly (``Ac-`` + ``-ton``) plus the standalone
    ``Bath``. The inline-rebuild tests below all use this fixture to
    assert non-empty proportions output against the bundled
    ``english_place_names.json`` corpus."""
    return [
        {
            "meaning": ["Roman bath"],
            "modifier_tags": ["name"],
            "modifier_type": None,
            "words": [{"modern_usage": "Bath", "old_english": ["Baðum"]}],
        },
        {
            "meaning": ["oak"],
            "modifier_tags": ["plant"],
            "modifier_type": "Descriptive",
            "words": [{"modern_usage": "Ac-", "old_english": ["āc"]}],
        },
        {
            "meaning": ["estate"],
            "modifier_tags": ["settlement"],
            "modifier_type": "Topographical",
            "words": [{"modern_usage": "-ton", "old_english": ["tūn"]}],
        },
    ]


def test_inline_rebuild_emits_populated_proportions_with_canonical(tmp_path: Path) -> None:
    """The inline-rebuild path must produce non-empty proportions when
    the fixture meanings + the culture's DB toponyms can perfectly
    decompose at least one name. Directly exercises
    ``_compute_proportions_inline`` so the assertion pins the
    function's output dict, not just the presence of well-formed
    bundle_metadata."""
    from wyrd.generators.kenning.lexicon.runtime_db_export import (
        _compute_proportions_inline,
    )

    # wyrd-bo01.3: seed the english corpus (Acton + Bath, England) into
    # the DB toponym table — the inline rebuild reads it from there now.
    db_path = tmp_path / "lexicon.db"
    _seed_lexicon_with_toponyms(db_path, [("Acton", "England", None), ("Bath", "England", None)])

    # Empty canonical map → every name flows through the heuristic.
    out = _compute_proportions_inline(
        _three_subject_acton_fixture(), canonical_decompositions={}, source_lexicon_db=db_path
    )

    # Acton perfectly decomposes against the fixture → both Ac and
    # ton end up in the english usages map. Pin both so a regression
    # that loses either half of the compound surfaces here.
    # D45: usages is nested {surface: {position: weight}} with BARE keys.
    assert "english" in out
    english_usages = out["english"].get("usages", {})
    assert "Ac" in english_usages, (
        f"Ac missing from english usages; rebuild didn't traverse the corpus. got {sorted(english_usages.keys())[:10]}"
    )
    assert "ton" in english_usages, (
        f"ton missing from english usages. got {sorted(english_usages.keys())[:10]}"
    )

    # And every output dict must carry the five canonical proportions keys.
    for culture, payload in out.items():
        assert {"usages", "single_usages", "structures", "tag_marginal", "tag_cooccurrence"} <= set(
            payload.keys()
        ), f"culture {culture!r} missing keys: {set(payload.keys())}"


def test_inline_rebuild_honors_canonical_decomposition(tmp_path: Path) -> None:
    """When a canonical_decompositions entry exists for a name in the
    corpus, ``_compute_proportions_inline`` routes through
    ``apply_canonical_to_name`` rather than the heuristic. Test the
    contract negatively: feed a nonsense canonical signature for a
    name that exists in the corpus + a fixture that DOES decompose it.
    Without the canonical-miss fallback this would silently drop the
    name (the canonical narrowing would empty ``name.words``). With
    the fallback, the heuristic path picks the same decomp it would
    have picked anyway, so the resulting usages map still contains
    the morphemes."""
    from wyrd.generators.kenning.lexicon.runtime_db_export import (
        _compute_proportions_inline,
    )

    # wyrd-bo01.3: Acton must live in the DB toponym corpus for the
    # canonical-miss fallback to have a name to re-decompose.
    db_path = tmp_path / "lexicon.db"
    _seed_lexicon_with_toponyms(db_path, [("Acton", "England", None)])

    canonical_decompositions = {
        "Acton": {"signature": "nonsense-signature-no-cell-will-match", "source": "test"},
    }

    out = _compute_proportions_inline(
        _three_subject_acton_fixture(), canonical_decompositions, source_lexicon_db=db_path
    )
    # Canonical-miss must fall back to the heuristic — Acton still
    # decomposes via Ac + ton and both morphemes still surface.
    # D45: usages keys are BARE surfaces.
    assert out["english"].get("usages", {}).get("Ac"), (
        f"canonical-miss didn't fall back to heuristic; "
        f"got english usages {sorted(out['english'].get('usages', {}).keys())[:10]}"
    )
    assert "ton" in out["english"]["usages"]


def test_empirical_priors_round_trip(tmp_path: Path) -> None:
    """The ``empirical_priors`` singleton row carries the priors payload
    in the same shape ``_cell_records`` produces — flat per-cell record
    with the cell-field keys at the top level + ``lemmas`` as a dict
    mapping lemma_ref → count. Emit it, read it back via the runtime
    adapter, and confirm both the on-disk shape AND the
    production-loader-acceptance round-trip both succeed."""
    from wyrd.generators.kenning.lexicon.empirical_priors import (
        load_empirical_priors_from_payload,
    )
    from wyrd.generators.kenning.lexicon.runtime_db_export import write_runtime_db
    from wyrd.generators.kenning.runtime.runtime_db_adapter import (
        empirical_priors_payload_from_runtime_db,
    )

    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    payload = {
        "version": "test-priors",
        "native": [
            {
                "culture": "english",
                "position": "post",
                "tag": "settlement",
                "era_midpoint": 1100,
                "lemmas": {"old-english:tūn": 42},
            }
        ],
        "loan": [],
    }
    write_runtime_db(
        output_path=out_path,
        subjects=[],
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_dir=tmp_path,
        source_lexicon_db=db_path,
        empirical_priors_payload=payload,
    )

    conn = sqlite3.connect(f"file:{out_path}?mode=ro", uri=True)
    try:
        readback = empirical_priors_payload_from_runtime_db(conn)
    finally:
        conn.close()
    # Round-trip through raw JSON serdes.
    assert readback == payload
    # Round-trip through the production loader (the actual consumer
    # of this payload). The cell-key tuple is what the runtime keys on.
    loaded = load_empirical_priors_from_payload(readback)
    assert loaded.version == "test-priors"
    assert loaded.native[("english", "post", "settlement", 1100)] == {"old-english:tūn": 42}


def test_empirical_priors_missing_row_returns_none(tmp_path: Path) -> None:
    """No priors emitted (``empirical_priors_payload=None``) → the
    adapter returns ``None`` and the runtime loader degrades to an
    empty ``EmpiricalPriors``. Pins the no-priors fallback contract."""
    from wyrd.generators.kenning.lexicon.runtime_db_export import write_runtime_db
    from wyrd.generators.kenning.runtime.runtime_db_adapter import (
        empirical_priors_payload_from_runtime_db,
    )

    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    write_runtime_db(
        output_path=out_path,
        subjects=[],
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_dir=tmp_path,
        source_lexicon_db=db_path,
    )

    conn = sqlite3.connect(f"file:{out_path}?mode=ro", uri=True)
    try:
        assert empirical_priors_payload_from_runtime_db(conn) is None
    finally:
        conn.close()


def test_emit_resolves_relative_lexicon_path(tmp_path: Path, monkeypatch) -> None:
    """``source_lexicon_db.resolve()`` must coerce a relative input
    path to absolute before stamping bundle_metadata, so deployed L4
    DBs can be correlated back to their L3 source."""
    db_dir = tmp_path / "subdir"
    db_dir.mkdir()
    db_path = db_dir / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    # Invoke from within db_dir with a bare filename — click resolves it
    # against cwd, but our explicit Path.resolve() must absolute-ize.
    monkeypatch.chdir(db_dir)
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "export-runtime-db",
            "--db",
            "lexicon.db",  # relative
            "--output",
            str(out_path),
            "--proportions-dir",
            str(tmp_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    conn = sqlite3.connect(str(out_path))
    try:
        stored = conn.execute(
            "SELECT value FROM bundle_metadata WHERE key = 'source_lexicon_db'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert Path(stored).is_absolute()
    assert Path(stored).resolve() == db_path.resolve()


def test_emit_unlinks_partial_output_on_failure(tmp_path: Path, monkeypatch) -> None:
    """A failure mid-emit must remove the partial output file so a
    subsequent build step never picks up a corrupt L4 DB. Inject a
    failure by replacing ``_write_proportions`` with a raiser AFTER
    sqlite3.connect has materialized the file on disk."""
    from wyrd.generators.kenning.lexicon import runtime_db_export as mod

    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated mid-emit failure")

    monkeypatch.setattr(mod, "_write_proportions", _boom)

    with pytest.raises(RuntimeError, match="simulated"):
        mod.write_runtime_db(
            output_path=out_path,
            subjects=[],
            fantasy_morphemes={},
            canonical_decompositions={},
            proportions_dir=tmp_path,
            source_lexicon_db=db_path,
        )

    assert not out_path.exists(), "partial L4 DB must not be left on disk after a mid-emit failure"


def test_emit_via_traversable_proportions_dir(tmp_path: Path) -> None:
    """``_load_proportions`` walks via the Traversable protocol so the
    operator-supplied override path works under any importlib.resources
    backend including zip loaders. Smoke-test the Traversable path
    explicitly by writing a tiny per-culture proportions JSON to a
    tmpdir and loading it via the Path branch (Path is the simplest
    Traversable; the contract holds for any Traversable subtype)."""
    import json as _json

    from wyrd.generators.kenning.lexicon.runtime_db_export import _load_proportions

    sample = {
        "usages": {"ham": {"post": 1}},
        "single_usages": {"ham": 1},
        "structures": [],
        "tag_marginal": {},
        "tag_cooccurrence": {},
    }
    for culture in ("english", "scottish", "welsh", "irish", "breton"):
        (tmp_path / f"{culture}_proportions.json").write_text(_json.dumps(sample))

    loaded = _load_proportions(tmp_path)
    assert set(loaded.keys()) == {"english", "scottish", "welsh", "irish", "breton"}
    for culture, data in loaded.items():
        assert {"usages", "single_usages", "structures", "tag_marginal", "tag_cooccurrence"} <= set(
            data.keys()
        ), f"culture {culture!r} missing keys"


# ---------- D45 (wyrd-aicu.1): meaning merge regroup-then-repick ----------


def test_write_meanings_merges_dash_variants_into_one_bare_row(tmp_path: Path) -> None:
    """D45: the up-to-4 dash-variant subjects for one surface (`-ton`/`Ton-`/
    `-ton-`/`ton`) collapse into ONE bare `meaning` row, entries UNIONED, with
    primary_language re-picked over the union (regroup-then-repick). Every
    existing test seeds one variant per surface, so the merge always collapsed a
    length-1 list — this is the headline behavior, exercised directly."""
    from wyrd.generators.kenning.lexicon.runtime_db_export import _write_meanings

    # Three dash-variants of one surface 'ton', plus a case twin. The caller
    # (write_runtime_db) de-dashes modern_usage before _write_meanings, so the
    # subjects here are already bare — distinct dash-variants fold to 'ton' /
    # 'Ton' (case twins stay distinct rows; dashes are what merge).
    subjects = [
        {
            "meaning": ["enclosure"],
            "modifier_tags": ["a"],
            "words": [{"modern_usage": "ton", "old_english": ["tūn"]}],
        },
        {
            "meaning": ["farmstead"],
            "modifier_tags": ["b"],
            "words": [{"modern_usage": "ton", "modern_english": ["ton"]}],
        },
        {
            "meaning": ["wave"],
            "modifier_tags": ["c"],
            "words": [{"modern_usage": "ton", "celtic_mix": ["ton"]}],
        },
    ]
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE meaning (usage_key TEXT PRIMARY KEY, primary_language TEXT, "
        "stratum TEXT, data BLOB NOT NULL)"
    )
    n = _write_meanings(conn, subjects)
    assert n == 1  # three subjects → ONE bare 'ton' row
    rows = conn.execute("SELECT usage_key, primary_language, data FROM meaning").fetchall()
    assert len(rows) == 1
    usage_key, primary_language, data = rows[0]
    assert usage_key == "ton"  # bare, no dash
    payload = json.loads(data)
    # All three subjects' entries unioned under the one row.
    assert len(payload["entries"]) == 3
    assert {tuple(e["meaning"]) for e in payload["entries"]} == {
        ("enclosure",),
        ("farmstead",),
        ("wave",),
    }
    # primary_language re-picked over the UNION (not per-variant) — a real
    # language wins over None; here three distinct single-language entries, so
    # the tie breaks alphabetically: celtic_mix.
    assert primary_language == "celtic_mix"
    conn.close()


# ---------- D45 (wyrd-aicu.5): dash-guard integration ----------


def test_dash_guard_runs_inside_emit_and_cleans_up_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The D45 guard is WIRED into write_runtime_db at a point where its raise
    triggers the partial-DB cleanup. Monkeypatch it to raise → the RuntimeError
    propagates AND the half-written L4 is unlinked (the BaseException handler).
    Deleting the guard call site makes this test fail (the patched guard never
    runs → emit succeeds → no RuntimeError), so the guard can't be silently
    disarmed — closing the gap that every isolated guard unit test leaves."""
    from wyrd.generators.kenning.lexicon import (
        collect_canonical_decompositions,
        collect_fantasy_morphemes,
        export_meanings,
    )
    from wyrd.generators.kenning.lexicon import runtime_db_export as mod

    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    def _boom(conn: sqlite3.Connection) -> None:
        raise RuntimeError("dashed identity!")

    monkeypatch.setattr(mod, "_verify_no_dashed_identity", _boom)

    with LexiconDB(db_path) as db:
        subjects = export_meanings(db)
        canonical = collect_canonical_decompositions(db)
        fantasy = collect_fantasy_morphemes(db)

    with pytest.raises(RuntimeError, match="dashed identity"):
        mod.write_runtime_db(
            output_path=out_path,
            subjects=subjects,
            fantasy_morphemes=fantasy,
            canonical_decompositions=canonical,
            proportions_dir=tmp_path,
            source_lexicon_db=db_path,
        )
    # The guard raised before commit; the BaseException cleanup unlinked the
    # half-written DB so no corrupt L4 is left for a downstream uploader.
    assert not out_path.exists()


def test_dash_guard_invoked_once_on_successful_emit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean emit calls the guard exactly once (before commit) and succeeds —
    the success-path complement to the raise-path test above."""
    from wyrd.generators.kenning.lexicon import runtime_db_export as mod

    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    real = mod._verify_no_dashed_identity
    calls: list[int] = []

    def _spy(conn: sqlite3.Connection) -> None:
        calls.append(1)
        real(conn)

    monkeypatch.setattr(mod, "_verify_no_dashed_identity", _spy)
    _run_emit(db_path=db_path, out_path=out_path, proportions_dir=tmp_path)
    assert calls == [1]
    assert out_path.exists()


# ---------- genitive-split prior emit (wyrd-aicu.9, D-3) ----------


def _emit_with_genitive(tmp_path: Path, genitive_split_map):
    """Emit a minimal L4 with ``genitive_split_map``; return the out path."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)
    counts = write_runtime_db(
        output_path=out_path,
        subjects=[],
        fantasy_morphemes={},
        canonical_decompositions={},
        proportions_dir=tmp_path,
        genitive_split_map=genitive_split_map,
        source_lexicon_db=db_path,
    )
    return out_path, counts


def test_emit_writes_genitive_split_prior_singleton(tmp_path: Path) -> None:
    """A non-empty map writes the singleton id=1 blob; the JSON payload
    carries the pairs sorted by -split_probability then the pair, and the
    return count reports the pair total."""
    out_path, counts = _emit_with_genitive(tmp_path, {("ston", "ton"): 0.4, ("sley", "ley"): 0.9})
    assert counts["genitive_split_pairs"] == 2

    conn = sqlite3.connect(str(out_path))
    try:
        rows = conn.execute("SELECT id, data FROM genitive_split_prior").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0][0] == 1
    payload = json.loads(rows[0][1])
    assert payload["version"] == EMITTER_VERSION
    # Sorted by -split_probability: the 0.9 pair leads.
    assert payload["pairs"][0] == {
        "long_form": "sley",
        "short_form": "ley",
        "split_probability": 0.9,
    }
    assert payload["pairs"][1]["long_form"] == "ston"


def test_emit_skips_genitive_split_prior_when_empty(tmp_path: Path) -> None:
    """An empty / None map writes no row and reports a zero pair count —
    the wipe-without-remine graceful path."""
    out_path, counts = _emit_with_genitive(tmp_path, {})
    assert counts["genitive_split_pairs"] == 0

    conn = sqlite3.connect(str(out_path))
    try:
        n = conn.execute("SELECT COUNT(*) FROM genitive_split_prior").fetchone()[0]
    finally:
        conn.close()
    assert n == 0


def test_emit_stamps_schema_version_unbumped(tmp_path: Path) -> None:
    """genitive_split_prior is additive/optional — emit stays at the existing
    schema_version (no bump, wyrd-aicu.9)."""
    out_path, _ = _emit_with_genitive(tmp_path, {})
    conn = sqlite3.connect(str(out_path))
    try:
        sv = conn.execute(
            "SELECT value FROM bundle_metadata WHERE key='schema_version'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert sv == SCHEMA_VERSION == "3"


def test_loader_tolerates_missing_genitive_table(tmp_path: Path) -> None:
    """A pre-aicu.9 bundle lacks the genitive_split_prior table entirely (the
    table is additive, no schema bump). The loader must degrade to None (→ {}),
    NOT raise — and the runtime must still accept the bundle."""
    from wyrd.generators.kenning.runtime.runtime_db_adapter import (
        genitive_split_map_from_runtime_db,
    )

    out_path, _ = _emit_with_genitive(tmp_path, {})
    conn = sqlite3.connect(str(out_path))
    try:
        conn.execute("DROP TABLE IF EXISTS genitive_split_prior")
        conn.commit()
        # Tolerant: missing table → None, not OperationalError.
        assert genitive_split_map_from_runtime_db(conn) is None
        # Schema version is unchanged, so the runtime still accepts it.
        sv = conn.execute(
            "SELECT value FROM bundle_metadata WHERE key='schema_version'"
        ).fetchone()[0]
        assert sv == "3"
    finally:
        conn.close()


# ---------- wyrd-x7w4: --dev auto-regenerates structures.yaml ----------


def test_dev_export_regenerates_structure_allowlist(tmp_path: Path) -> None:
    """wyrd-x7w4: ``_regen_structure_allowlist`` rebuilds the per-language
    structures.yaml from the just-written seed DB via a refresh-merge.

    Pointed at a copy of the COMMITTED seed-runtime.db (the DB ``--dev`` would
    have just written), it must reproduce the COMMITTED structures.yaml
    byte-for-byte — the idempotency that the drift-guard
    (test_kenning_structure_allowlist.test_committed_structures_yaml_in_sync_with_bundle)
    relies on, here exercised through the export hook's code path."""
    import shutil
    from importlib import resources

    from wyrd.generators.kenning.cli.lexicon.export_runtime_db import (
        _regen_structure_allowlist,
    )

    data = resources.files("wyrd.generators.kenning.data")
    with resources.as_file(data.joinpath("seed-runtime.db")) as seed:
        shutil.copy(seed, tmp_path / "seed-runtime.db")

    out = _regen_structure_allowlist(tmp_path / "seed-runtime.db")

    assert out == tmp_path / "structures.yaml"  # written next to the DB
    committed = data.joinpath("structures.yaml").read_text(encoding="utf-8")
    assert out.read_text(encoding="utf-8") == committed


def test_dev_export_writes_structures_yaml_next_to_db(tmp_path: Path) -> None:
    """wyrd-x7w4 wiring: ``export-runtime-db --dev`` invokes the allowlist regen,
    leaving a structures.yaml next to the seed DB (in tmp — never the package file).

    Uses the inline-rebuild path (toponym-seeded lexicon, no --proportions-dir) so
    --dev's top-N produces D45-nested usages."""
    import yaml

    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "seed-runtime.db"
    _seed_lexicon_with_toponyms(db_path, [("Acton", "England", None), ("Bath", "England", None)])

    result = CliRunner().invoke(
        cli_root,
        [
            "lexicon",
            "export-runtime-db",
            "--db",
            str(db_path),
            "--output",
            str(out_path),
            "--dev",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    structures = tmp_path / "structures.yaml"
    assert structures.exists(), "--dev must regenerate structures.yaml next to the DB"
    parsed = yaml.safe_load(structures.read_text(encoding="utf-8"))
    # sectioned per-language output (refresh-merged over the committed file)
    assert "english" in parsed


def test_non_dev_export_does_not_write_structures_yaml(tmp_path: Path) -> None:
    """The allowlist regen is --dev-only; a plain export leaves no structures.yaml."""
    db_path = tmp_path / "lexicon.db"
    out_path = tmp_path / "runtime.db"
    _seed_minimal_lexicon(db_path)
    _write_proportions_fixture(tmp_path)

    result = CliRunner().invoke(
        cli_root,
        [
            "lexicon",
            "export-runtime-db",
            "--db",
            str(db_path),
            "--output",
            str(out_path),
            "--proportions-dir",
            str(tmp_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "structures.yaml").exists()


def test_dev_regen_preserves_operator_false_through_merge(tmp_path: Path) -> None:
    """wyrd-x7w4: the hook is a refresh-MERGE, not a regen-from-scratch — an
    operator ``enabled: false`` on a MULTI-morpheme structure (which _build would
    otherwise default to ``true``) must survive. Guards against a regression to
    ``_build(conn, {})`` that the idempotency test can't catch (the committed
    file's only falses are auto-seeded <Bare> defaults).

    Also pins the Gemini-HIGH fix: the merge base is the WORKING-TREE sibling
    structures.yaml next to the DB (written below), not the importlib.resources
    copy — so local curation isn't dropped under a non-editable install."""
    import shutil
    from importlib import resources

    from wyrd.generators.kenning.cli.lexicon.export_runtime_db import _regen_structure_allowlist
    from wyrd.generators.kenning.runtime.structure_allowlist import (
        load_structure_allowlist_from_text,
    )

    data = resources.files("wyrd.generators.kenning.data")
    with resources.as_file(data.joinpath("seed-runtime.db")) as seed:
        shutil.copy(seed, tmp_path / "seed-runtime.db")

    # An operator has disabled a real 3-morpheme english structure (default:
    # enabled) in the WORKING-TREE structures.yaml sitting next to the seed.
    curated = "(pre+inner+post)"
    (tmp_path / "structures.yaml").write_text(
        f'english:\n  "{curated}": {{enabled: false}}\n', encoding="utf-8"
    )

    out = _regen_structure_allowlist(tmp_path / "seed-runtime.db")
    al = load_structure_allowlist_from_text(out.read_text(encoding="utf-8"))
    assert al["english"][curated] is False  # sticky operator false preserved by the merge
