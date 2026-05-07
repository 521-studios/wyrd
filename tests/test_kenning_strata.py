"""Tests for within-language stratum tagging (wyrd-lr4).

Covers:
- Schema: ``etymon.stratum`` column + index land via fresh-install +
  migration paths.
- ``classify_welsh``: per-stratum heuristic, including priority order
  when an etymon has multiple ancestor languages.
- ``lexicon classify-stratum`` CLI: dry-run reports counts without
  writing; ``--apply`` writes; ``--force`` overrides existing values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, migrate_schema
from wyrd.generators.kenning.strata import WELSH_STRATA, classify_welsh


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def _seed_source(db: LexiconDB) -> None:
    """Every etymon_descent row needs a source — minimal source seed."""
    db.upsert_source(id="wiktionary", title="Wiktionary")
    db.commit()


def _add_descent(
    db: LexiconDB, *, parent_id: int, child_id: int, edge_type: str = "inheritance"
) -> None:
    db.conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
        "VALUES (?, ?, ?, 'wiktionary')",
        (parent_id, child_id, edge_type),
    )
    db.commit()


# --- schema ---------------------------------------------------------------


def test_init_schema_creates_etymon_stratum_column(fresh_db: Path) -> None:
    """Fresh installs pick up the ``etymon.stratum`` column from
    data/lexicon.sql. Pinning here so a future schema-rewrite that
    drops the column would fail this test loudly."""
    with LexiconDB(fresh_db) as db:
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        assert "stratum" in cols


def test_init_schema_creates_etymon_stratum_index(fresh_db: Path) -> None:
    """The ``stratum`` filter (deferred follow-up) needs an index on
    the column — pin its presence on fresh installs."""
    with LexiconDB(fresh_db) as db:
        indexes = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
    assert "idx_etymon_stratum" in indexes


def test_migrate_schema_adds_stratum_to_legacy_db(tmp_path: Path) -> None:
    """A legacy DB that was created BEFORE this column existed gets
    the column added by ``migrate_schema``. Simulated by dropping the
    column post-init and re-running migrate.

    Verifies: (a) the column is added, (b) the index is created,
    (c) the migration is idempotent (second run is a no-op).
    """
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    # Simulate a legacy DB by dropping the column. SQLite supports
    # DROP COLUMN since 3.35 (2021) — fine for the pinned test runner.
    with LexiconDB(db_path) as db:
        db.conn.execute("DROP INDEX IF EXISTS idx_etymon_stratum")
        db.conn.execute("ALTER TABLE etymon DROP COLUMN stratum")
        db.commit()

    with LexiconDB(db_path) as db:
        applied = migrate_schema(db)
    assert applied["etymon.stratum"] is True
    assert applied["idx_etymon_stratum"] is True

    with LexiconDB(db_path) as db:
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        assert "stratum" in cols

    # Idempotent: a second migrate flips the flags back to False (no
    # work needed) and doesn't error.
    with LexiconDB(db_path) as db:
        applied2 = migrate_schema(db)
    assert applied2["etymon.stratum"] is False
    assert applied2["idx_etymon_stratum"] is False


# --- classify_welsh -------------------------------------------------------


def test_classify_welsh_assigns_native_welsh_when_no_descent(fresh_db: Path) -> None:
    """A welsh etymon with NO ``etymon_descent`` parents falls into
    the default ``native-welsh`` bucket. This is the most common shape
    in the empirical bundle (compound formations from native morphemes
    that the wiktextract miner couldn't trace further back)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("rhydoldfoel", "welsh")
        proposals = classify_welsh(db)
    assert proposals == {eid: "native-welsh"}


def test_classify_welsh_latin_loan_via_descent(fresh_db: Path) -> None:
    """A welsh etymon descending from a Latin parent gets
    ``latin-loan``."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        latin_id = db.upsert_etymon("pons", "latin")
        welsh_id = db.upsert_etymon("pont", "welsh")
        _add_descent(db, parent_id=latin_id, child_id=welsh_id)
        proposals = classify_welsh(db)
    assert proposals[welsh_id] == "latin-loan"
    # Latin parent itself isn't in the welsh family — not classified.
    assert latin_id not in proposals


def test_classify_welsh_english_loan_via_descent(fresh_db: Path) -> None:
    """A welsh etymon descending from any English variety gets
    ``english-loan``."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eng_id = db.upsert_etymon("manor", "modern-english")
        welsh_id = db.upsert_etymon("maenor", "welsh")
        _add_descent(db, parent_id=eng_id, child_id=welsh_id, edge_type="borrowing")
        proposals = classify_welsh(db)
    assert proposals[welsh_id] == "english-loan"


def test_classify_welsh_brittonic_substrate_via_descent(fresh_db: Path) -> None:
    """A welsh etymon descending from cel-bry-pro / proto-celtic /
    old-welsh ancestor → ``brittonic-substrate``."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        bry_id = db.upsert_etymon("*kaston-", "cel-bry-pro")
        welsh_id = db.upsert_etymon("cest", "welsh")
        _add_descent(db, parent_id=bry_id, child_id=welsh_id)
        proposals = classify_welsh(db)
    assert proposals[welsh_id] == "brittonic-substrate"


def test_classify_welsh_medieval_via_descent(fresh_db: Path) -> None:
    """A welsh etymon whose ONLY ancestor is middle-welsh →
    ``medieval-welsh``."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        mw_id = db.upsert_etymon("dyffryn", "middle-welsh")
        welsh_id = db.upsert_etymon("dyffryn", "welsh")
        _add_descent(db, parent_id=mw_id, child_id=welsh_id)
        proposals = classify_welsh(db)
    assert proposals[welsh_id] == "medieval-welsh"
    # The middle-welsh ancestor itself gets the medieval-welsh stratum
    # via the self-language path (it doesn't descend FROM 'welsh').
    assert proposals[mw_id] == "medieval-welsh"


def test_classify_welsh_priority_latin_over_brittonic(fresh_db: Path) -> None:
    """Priority order: a welsh etymon with BOTH a Latin and a Brittonic
    ancestor classifies as ``latin-loan``, not ``brittonic-substrate``.
    Encodes the scholarly convention that an explicit loan pathway
    displaces inherited form for register classification."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        latin_id = db.upsert_etymon("ecclesia", "latin")
        bry_id = db.upsert_etymon("*ekklesia", "cel-bry-pro")
        welsh_id = db.upsert_etymon("eglwys", "welsh")
        _add_descent(db, parent_id=latin_id, child_id=welsh_id, edge_type="borrowing")
        _add_descent(db, parent_id=bry_id, child_id=welsh_id)
        proposals = classify_welsh(db)
    assert proposals[welsh_id] == "latin-loan"


def test_classify_welsh_priority_english_over_medieval(fresh_db: Path) -> None:
    """Priority: english-loan beats medieval-welsh (rare combo, but
    the rule needs to be deterministic so callers can trust the
    register-tag)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eng_id = db.upsert_etymon("market", "modern-english")
        mw_id = db.upsert_etymon("marchnad", "middle-welsh")
        welsh_id = db.upsert_etymon("marchnad", "welsh")
        _add_descent(db, parent_id=eng_id, child_id=welsh_id, edge_type="borrowing")
        _add_descent(db, parent_id=mw_id, child_id=welsh_id)
        proposals = classify_welsh(db)
    assert proposals[welsh_id] == "english-loan"


def test_classify_welsh_self_language_for_ancestor_varieties(fresh_db: Path) -> None:
    """An etymon whose own ``language`` is a Welsh-family ancestor
    (cel-bry-pro / old-welsh / middle-welsh) gets classified by self-
    language, not by walking descent. This is a Welsh-family etymon
    even before it became 'welsh' proper."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        bry_id = db.upsert_etymon("*kambo-", "cel-bry-pro")
        ow_id = db.upsert_etymon("camm", "old-welsh")
        mw_id = db.upsert_etymon("kamm", "middle-welsh")
        proposals = classify_welsh(db)
    assert proposals[bry_id] == "brittonic-substrate"
    assert proposals[ow_id] == "brittonic-substrate"
    assert proposals[mw_id] == "medieval-welsh"


def test_classify_welsh_skips_merged_rows(fresh_db: Path) -> None:
    """Etymons with ``merged_into_id`` set are OCR-clustering losers
    — they've been folded into a canonical winner. Skipping them in
    the classifier avoids carrying a stale stratum on a row that
    callers already filter out."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        winner = db.upsert_etymon("dyffryn", "welsh")
        loser = db.upsert_etymon("dyffrn", "welsh")  # OCR variant
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (winner, loser),
        )
        db.commit()
        proposals = classify_welsh(db)
    assert winner in proposals
    assert loser not in proposals


# --- CLI ------------------------------------------------------------------


def test_cli_classify_stratum_dry_run_does_not_write(fresh_db: Path) -> None:
    """``classify-stratum --language welsh`` (no --apply) prints the
    bucket counts but leaves the stratum column NULL on every row."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        latin_id = db.upsert_etymon("pons", "latin")
        welsh_id = db.upsert_etymon("pont", "welsh")
        _add_descent(db, parent_id=latin_id, child_id=welsh_id)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "classify-stratum", "--db", str(fresh_db), "--language", "welsh"],
    )
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert "latin-loan" in combined
    assert "dry-run" in combined.lower()

    with LexiconDB(fresh_db) as db:
        row = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (welsh_id,)).fetchone()
    assert row["stratum"] is None


def test_cli_classify_stratum_apply_writes_column(fresh_db: Path) -> None:
    """``--apply`` writes the proposed stratum to the etymon row."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        latin_id = db.upsert_etymon("pons", "latin")
        welsh_id = db.upsert_etymon("pont", "welsh")
        _add_descent(db, parent_id=latin_id, child_id=welsh_id)

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "classify-stratum",
            "--db",
            str(fresh_db),
            "--language",
            "welsh",
            "--apply",
        ],
    )
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined

    with LexiconDB(fresh_db) as db:
        row = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (welsh_id,)).fetchone()
    assert row["stratum"] == "latin-loan"


def test_cli_classify_stratum_skips_existing_without_force(fresh_db: Path) -> None:
    """Idempotent re-run: rows that already have a stratum are skipped
    by default, so a second --apply doesn't overwrite operator hand-
    edits."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        welsh_id = db.upsert_etymon("rhydoldfoel", "welsh")
        # Pre-populate with a manual stratum that DIFFERS from what
        # the classifier would assign (the classifier would say
        # 'native-welsh' since there are no descent edges).
        db.conn.execute("UPDATE etymon SET stratum = 'manual-override' WHERE id = ?", (welsh_id,))
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "classify-stratum",
            "--db",
            str(fresh_db),
            "--language",
            "welsh",
            "--apply",
        ],
    )
    assert result.exit_code == 0

    with LexiconDB(fresh_db) as db:
        row = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (welsh_id,)).fetchone()
    # Manual override preserved.
    assert row["stratum"] == "manual-override"


def test_cli_classify_stratum_force_overwrites(fresh_db: Path) -> None:
    """``--force`` overwrites existing stratum values. Useful after
    re-running the heuristic with new rules — operators don't have to
    NULL the column manually first."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        welsh_id = db.upsert_etymon("rhydoldfoel", "welsh")
        db.conn.execute("UPDATE etymon SET stratum = 'old-bucket' WHERE id = ?", (welsh_id,))
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "classify-stratum",
            "--db",
            str(fresh_db),
            "--language",
            "welsh",
            "--apply",
            "--force",
        ],
    )
    assert result.exit_code == 0

    with LexiconDB(fresh_db) as db:
        row = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (welsh_id,)).fetchone()
    assert row["stratum"] == "native-welsh"


def test_cli_classify_stratum_rejects_unknown_language(fresh_db: Path) -> None:
    """Click's Choice validator blocks unknown languages at parse
    time so we don't pretend to classify something we don't have a
    vocabulary for. French / OE / ON are follow-up tickets."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "classify-stratum",
            "--db",
            str(fresh_db),
            "--language",
            "french",
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "french" in combined.lower() or "invalid value" in combined.lower()


def test_cli_classify_stratum_no_etymons_short_circuits(fresh_db: Path) -> None:
    """Empty DB → 'No etymons found' message, exit 0. Pins the early-
    return so an empty corpus doesn't print a cryptic empty bucket
    table."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "classify-stratum", "--db", str(fresh_db), "--language", "welsh"],
    )
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert "no etymons found" in combined.lower()


def test_welsh_strata_priority_order_includes_all_buckets() -> None:
    """``WELSH_STRATA`` is the source-of-truth ordering. The CLI
    summary iterates it; pin that all five buckets are present and in
    the load-bearing priority order."""
    assert WELSH_STRATA == (
        "latin-loan",
        "english-loan",
        "brittonic-substrate",
        "medieval-welsh",
        "native-welsh",
    )
