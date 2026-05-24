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

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.cli.lexicon.export_runtime_db import (
    EMITTER_VERSION,
    SCHEMA_VERSION,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


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


def _write_proportions_fixture(directory: Path, culture: str = "english") -> None:
    """Write a small-but-shape-correct proportions JSON into ``directory``.

    Single fixture file per culture; the emitter reads it via the
    ``--proportions-dir`` flag so tests don't have to monkeypatch the
    bundled data dir.
    """
    payload = {
        "usages": {"-ham-": 100, "-ton-": 50, "-zero-weight": 0},
        "single_usages": {"-castle": 20, "-bury": 5},
        "structures": [
            {"proportion": 70, "words": [[{"location": "pre"}, {"location": "post"}]]},
            {
                "proportion": 30,
                "words": [
                    [{"location": "pre"}, {"location": "inner"}, {"location": "post"}]
                ],
            },
            {"proportion": 0, "words": [[{"location": "pre"}]]},  # skipped
        ],
        "tag_marginal": {"water": 12, "architecture": 8},
        "tag_cooccurrence": {"water|architecture": 3, "water|tree": 2},
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
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
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
    assert meta["source_lexicon_db"] == str(db_path)
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
    assert [r[0] for r in rows] == ["-ham-", "-ton-"]
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

    assert [r[0] for r in rows] == ["-castle", "-bury"]
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
                "SELECT tag, weight FROM proportions_tag_marginal "
                "WHERE culture = 'english'"
            ).fetchall()
        )
        cooc = conn.execute(
            "SELECT tag1, tag2, weight FROM proportions_tag_cooccurrence "
            "WHERE culture = 'english'"
        ).fetchall()
    finally:
        conn.close()

    assert marginal == {"water": 12, "architecture": 8}
    assert sorted(cooc) == sorted(
        [("water", "architecture", 3), ("water", "tree", 2)]
    )


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
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT culture FROM proportions_usage"
            )
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
        rows = conn.execute(
            "SELECT usage_key, primary_language, data FROM meaning"
        ).fetchall()
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
        cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(canonical_decomposition)")
        ]
        rows = conn.execute(
            "SELECT toponym_name, data FROM canonical_decomposition"
        ).fetchall()
    finally:
        conn.close()

    assert cols == ["toponym_name", "data"], (
        "canonical_decomposition must be (toponym_name, data) per D38.4 — "
        "no culture column in PR 1"
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
        rows = conn.execute(
            "SELECT usage_key, data FROM fantasy_morpheme"
        ).fetchall()
    finally:
        conn.close()

    assert rows == []


# ---------- sampling primitive smoke ----------


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

        # roll=1 → first row (-ham-, cumulative=100)
        row_low = conn.execute(
            "SELECT usage_key FROM proportions_usage "
            "WHERE culture = ? AND cumulative >= ? "
            "ORDER BY cumulative LIMIT 1",
            ("english", 1),
        ).fetchone()
        # roll=120 → second row (-ton-, cumulative=150)
        row_high = conn.execute(
            "SELECT usage_key FROM proportions_usage "
            "WHERE culture = ? AND cumulative >= ? "
            "ORDER BY cumulative LIMIT 1",
            ("english", 120),
        ).fetchone()
    finally:
        conn.close()

    assert row_low[0] == "-ham-"
    assert row_high[0] == "-ton-"
