"""Tests for within-language stratum tagging (wyrd-lr4).

Covers:
- Schema: ``etymon.stratum`` column + index land via fresh-install +
  migration paths.
- ``classify_welsh``: per-stratum heuristic, including priority order
  when an etymon has multiple ancestor languages.
- ``classify_french``: per-stratum heuristic for French, including
  the load-bearing Latin-absence invariant (every French word
  descends from Latin so 'latin in parents' is intentionally not a
  stratum signal).
- ``classify_old_english``: per-stratum heuristic for OE, including
  the parallel Germanic-descent absence invariant (gmw-pro /
  proto-germanic in parents is not a stratum signal — every OE
  word descends from those, would collapse the bundle).
- ``lexicon classify-stratum`` CLI: dry-run reports counts without
  writing; ``--apply`` writes; ``--force`` overrides existing values;
  exercised under Welsh / French / Old English dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, migrate_schema
from wyrd.generators.kenning.strata import (
    FRENCH_STRATA,
    OLD_ENGLISH_STRATA,
    WELSH_STRATA,
    classify_french,
    classify_old_english,
    classify_welsh,
)


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
    """The Phase 3 ``--stratum`` filter scans the etymon table by
    stratum value during keep-set construction; pin the index's
    presence on fresh installs so the scan stays cheap."""
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
    vocabulary for. Old Norse is the still-unsupported follow-up."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "classify-stratum",
            "--db",
            str(fresh_db),
            "--language",
            "old-norse",
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "old-norse" in combined.lower() or "invalid value" in combined.lower()


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


# --- classify_french (wyrd-lr4 Phase 4a) ----------------------------------


def test_classify_french_assigns_native_french_when_no_descent(fresh_db: Path) -> None:
    """A french etymon with NO ``etymon_descent`` parents falls into
    the default ``native-french`` bucket. Most-common shape — French
    descends from Latin via the standard Romance path, but the bulk
    of bundle entries lack explicit descent links."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("ville", "french")
        proposals = classify_french(db)
    assert proposals == {eid: "native-french"}


def test_classify_french_frankish_substrate_via_descent(fresh_db: Path) -> None:
    """A french etymon descending from a Frankish (frk) parent gets
    ``frankish-substrate``. Frankish is the Germanic source for
    French personal-name suffixes -hard / -old and toponymic
    elements visible in post-Conquest English place names."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        frk_id = db.upsert_etymon("hard", "frk")
        fr_id = db.upsert_etymon("hard", "french")
        _add_descent(db, parent_id=frk_id, child_id=fr_id)
        proposals = classify_french(db)
    assert proposals[fr_id] == "frankish-substrate"
    # The frk parent itself is in the family — classified by self-language.
    assert proposals[frk_id] == "frankish-substrate"


def test_classify_french_gaulish_substrate_via_descent(fresh_db: Path) -> None:
    """A french etymon descending from Gaulish (cel-gau) gets
    ``gaulish-substrate``. Gaulish is the pre-Latin Celtic substrate
    visible in toponyms like -dunum / -briga."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        gau_id = db.upsert_etymon("dunon", "cel-gau")
        fr_id = db.upsert_etymon("dun", "french")
        _add_descent(db, parent_id=gau_id, child_id=fr_id)
        proposals = classify_french(db)
    assert proposals[fr_id] == "gaulish-substrate"
    # Self-language path covers cel-gau too.
    assert proposals[gau_id] == "gaulish-substrate"


def test_classify_french_latin_descent_does_not_collapse_into_one_bucket(
    fresh_db: Path,
) -> None:
    """Critical correctness pin: Latin in the parent set MUST NOT
    classify a french etymon as gallo-roman, otherwise every French
    word would land in one bucket and the substrate distinctions
    would vanish. ``_FRENCH_ANCESTOR_TO_STRATUM`` deliberately
    excludes 'latin' / 'vulgar-latin' for this reason."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        latin_id = db.upsert_etymon("villa", "latin")
        fr_id = db.upsert_etymon("ville", "french")
        _add_descent(db, parent_id=latin_id, child_id=fr_id)
        proposals = classify_french(db)
    # Latin parent → falls through to native-french, NOT gallo-roman.
    assert proposals[fr_id] == "native-french"


def test_classify_french_priority_frankish_over_gaulish(fresh_db: Path) -> None:
    """Priority order: a french etymon with BOTH Frankish and
    Gaulish ancestors classifies as ``frankish-substrate``.
    Encodes the convention that the Germanic post-Roman layer
    displaces the pre-Roman Celtic layer when both signals are
    present (rare combo, but the rule needs to be deterministic
    so callers can trust the register tag)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        frk_id = db.upsert_etymon("garda", "frk")
        gau_id = db.upsert_etymon("garto", "cel-gau")
        fr_id = db.upsert_etymon("jardin", "french")
        _add_descent(db, parent_id=frk_id, child_id=fr_id)
        _add_descent(db, parent_id=gau_id, child_id=fr_id)
        proposals = classify_french(db)
    assert proposals[fr_id] == "frankish-substrate"


def test_classify_french_self_language_for_period_varieties(fresh_db: Path) -> None:
    """An etymon whose own ``language`` is a French-family period
    variety (old-french / middle-french / norman-french) gets
    classified as ``medieval-french`` directly. These ARE the
    ancestor period; classifying by descent would describe what fed
    INTO them, not their own register."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        of_id = db.upsert_etymon("vile", "old-french")
        mf_id = db.upsert_etymon("ville", "middle-french")
        nf_id = db.upsert_etymon("vile", "norman-french")
        proposals = classify_french(db)
    assert proposals[of_id] == "medieval-french"
    assert proposals[mf_id] == "medieval-french"
    assert proposals[nf_id] == "medieval-french"


def test_classify_french_self_language_for_gallo_roman(fresh_db: Path) -> None:
    """An etymon whose own ``language`` is ``vulgar-latin`` gets
    ``gallo-roman``. This is the ancestor stage between Classical
    Latin and Old French — the only non-default path that
    populates the gallo-roman bucket today (Latin parents on a
    'french'-language row deliberately fall through, see
    test_classify_french_latin_descent_does_not_collapse...)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        vl_id = db.upsert_etymon("villa", "vulgar-latin")
        proposals = classify_french(db)
    assert proposals[vl_id] == "gallo-roman"


def test_classify_french_skips_merged_rows(fresh_db: Path) -> None:
    """Etymons with ``merged_into_id`` set are OCR-clustering losers
    — skipped in both passes (self-language + ancestor walk) so we
    don't carry a stale stratum on a row callers already filter out."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        winner = db.upsert_etymon("ville", "french")
        loser = db.upsert_etymon("vyle", "french")  # OCR variant
        # Also a loser in an ancestor-variety language so both passes
        # are exercised.
        period_loser = db.upsert_etymon("vyle", "old-french")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id IN (?, ?)",
            (winner, loser, period_loser),
        )
        db.commit()
        proposals = classify_french(db)
    assert winner in proposals
    assert loser not in proposals
    assert period_loser not in proposals


def test_french_strata_priority_order_includes_all_buckets() -> None:
    """``FRENCH_STRATA`` is the source-of-truth ordering. Pin all
    five buckets are present in the load-bearing priority order so a
    refactor that re-orders them (and changes which stratum wins on
    multi-ancestor etymons) gets caught."""
    assert FRENCH_STRATA == (
        "frankish-substrate",
        "gaulish-substrate",
        "gallo-roman",
        "medieval-french",
        "native-french",
    )


def test_cli_classify_stratum_french_dry_run_reports_counts(
    fresh_db: Path,
) -> None:
    """End-to-end CLI sanity: ``--language french`` dispatches to
    classify_french, prints the per-stratum counts, and doesn't write
    without ``--apply`` (mirrors the welsh dry-run test)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        # One row per French stratum so the summary table covers each.
        db.upsert_etymon("hard", "frk")  # frankish-substrate
        db.upsert_etymon("dunon", "cel-gau")  # gaulish-substrate
        db.upsert_etymon("villa", "vulgar-latin")  # gallo-roman
        db.upsert_etymon("vile", "old-french")  # medieval-french
        db.upsert_etymon("amour", "french")  # native-french (default)
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "classify-stratum", "--db", str(fresh_db), "--language", "french"],
    )
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert "french stratum classification: 5 etymons" in combined
    for stratum in FRENCH_STRATA:
        assert stratum in combined
    # Dry-run — no writes.
    with LexiconDB(fresh_db) as db:
        non_null = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE stratum IS NOT NULL"
        ).fetchone()[0]
    assert non_null == 0


def test_classify_french_frankish_beats_vulgar_latin_parent(fresh_db: Path) -> None:
    """Defense-in-depth on the Latin-absence invariant. A french
    etymon with BOTH a frk parent (real substrate signal) AND a
    vulgar-latin parent (NOT in the ancestor map by design) must
    classify as frankish-substrate, not gallo-roman. Pin that
    vulgar-latin in the ancestor pass doesn't leak into a stratum
    when paired with a genuine substrate signal — a regression that
    accidentally added vulgar-latin to _FRENCH_ANCESTOR_TO_STRATUM
    would change this etymon's stratum and be caught."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        frk_id = db.upsert_etymon("baro", "frk")
        vl_id = db.upsert_etymon("baro", "vulgar-latin")
        fr_id = db.upsert_etymon("baron", "french")
        _add_descent(db, parent_id=frk_id, child_id=fr_id)
        _add_descent(db, parent_id=vl_id, child_id=fr_id)
        proposals = classify_french(db)
    # frk in parents → frankish-substrate; vulgar-latin in parents
    # is intentionally not in the ancestor map so it doesn't pull
    # toward gallo-roman.
    assert proposals[fr_id] == "frankish-substrate"
    # vulgar-latin self-language still classifies the parent itself.
    assert proposals[vl_id] == "gallo-roman"


def test_cli_classify_stratum_french_force_overwrites(fresh_db: Path) -> None:
    """``--apply --force`` overwrites an existing stratum value when
    the user has dispatched to the French classifier. The force/skip
    logic lives in the CLI body (outside the per-language
    classifier), so it's the same code path as Welsh — but pin a
    French-specific invocation so a regression that wires --force
    through only one branch of the dispatch dict gets caught."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        fr_id = db.upsert_etymon("amour", "french")
        # Pre-populate with a wrong stratum so we can verify the
        # overwrite. classify_french would assign 'native-french' to
        # this row (no descent → default).
        db.conn.execute(
            "UPDATE etymon SET stratum = ? WHERE id = ?",
            ("manual-override", fr_id),
        )
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
            "french",
            "--apply",
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (fr_id,)).fetchone()[
            "stratum"
        ]
    assert stratum == "native-french"


def test_cli_classify_stratum_french_apply_writes_column(fresh_db: Path) -> None:
    """``--apply`` actually writes the proposed stratum values for
    French. Mirrors the welsh apply test — pinned independently
    because the dispatch via the classifiers dict could regress."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        frk_id = db.upsert_etymon("hard", "frk")
        of_id = db.upsert_etymon("vile", "old-french")
        fr_id = db.upsert_etymon("amour", "french")
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
            "french",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        rows = db.conn.execute(
            "SELECT id, stratum FROM etymon WHERE id IN (?, ?, ?) ORDER BY id",
            (frk_id, of_id, fr_id),
        ).fetchall()
    by_id = {r["id"]: r["stratum"] for r in rows}
    assert by_id[frk_id] == "frankish-substrate"
    assert by_id[of_id] == "medieval-french"
    assert by_id[fr_id] == "native-french"


# --- classify_old_english (wyrd-lr4 Phase 4b) -----------------------------


def test_classify_old_english_assigns_native_when_no_descent(fresh_db: Path) -> None:
    """An OE etymon with NO ``etymon_descent`` parents falls into the
    ``native-old-english`` default. Most-common shape — gmw-pro and
    proto-germanic ancestors are intentionally NOT signal carriers
    (would collapse the bundle), so the default catches all standard-
    descent etymons."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("dūn", "old-english")
        proposals = classify_old_english(db)
    assert proposals == {eid: "native-old-english"}


def test_classify_old_english_latin_loan_via_descent(fresh_db: Path) -> None:
    """An OE etymon descending from a Latin parent gets ``latin-loan``.
    Encodes the post-Christianization (~597 CE onward) layer of
    learned-borrowing — church / literacy / liturgical vocabulary."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        latin_id = db.upsert_etymon("monasterium", "latin")
        oe_id = db.upsert_etymon("mynster", "old-english")
        _add_descent(db, parent_id=latin_id, child_id=oe_id, edge_type="borrowing")
        proposals = classify_old_english(db)
    assert proposals[oe_id] == "latin-loan"
    # Latin parent itself isn't in the OE family — not classified.
    assert latin_id not in proposals


def test_classify_old_english_latin_variants_all_route_to_latin_loan(
    fresh_db: Path,
) -> None:
    """All Latin variants in the data (la-lat / la-med / la-ecc /
    vulgar-latin) route to ``latin-loan``. Ecclesiastical Latin
    (la-ecc) is the most semantically specific Christianization
    signal but the bundle is too small for its own bucket; folded
    in here. ancient-greek also routes here — Christian-era Greek
    loans came via Latin / clerical channels."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        cases = [
            ("la-lat-parent", "la-lat", "la-lat-child"),
            ("la-med-parent", "la-med", "la-med-child"),
            ("la-ecc-parent", "la-ecc", "la-ecc-child"),
            ("vulgar-latin-parent", "vulgar-latin", "vulgar-latin-child"),
            ("greek-parent", "ancient-greek", "greek-child"),
        ]
        oe_ids = {}
        for parent_form, parent_lang, child_form in cases:
            parent_id = db.upsert_etymon(parent_form, parent_lang)
            child_id = db.upsert_etymon(child_form, "old-english")
            _add_descent(db, parent_id=parent_id, child_id=child_id, edge_type="borrowing")
            oe_ids[child_form] = child_id
        proposals = classify_old_english(db)
    for child_form, child_id in oe_ids.items():
        assert proposals[child_id] == "latin-loan", child_form


def test_classify_old_english_norse_loan_via_descent(fresh_db: Path) -> None:
    """An OE etymon descending from Old Norse gets ``norse-loan``.
    Encodes the Danelaw period (~800-1000 CE) lexical influence —
    the late-OE settler / mercantile / legal layer."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        on_id = db.upsert_etymon("þorp", "old-norse")
        oe_id = db.upsert_etymon("þrop", "old-english")
        _add_descent(db, parent_id=on_id, child_id=oe_id, edge_type="borrowing")
        proposals = classify_old_english(db)
    assert proposals[oe_id] == "norse-loan"


def test_classify_old_english_celtic_substrate_via_descent(fresh_db: Path) -> None:
    """An OE etymon descending from a Celtic ancestor (proto-celtic /
    cel-bry-pro / old-irish / cel) gets ``celtic-substrate``. Rare
    in the live DB (~60 etymons total) but distinctive — pre-Anglo-
    Saxon Brittonic substrate or post-conversion Goidelic influence
    via Northumbrian Christianity."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        cel_id = db.upsert_etymon("*kuninga", "proto-celtic")
        oe_id = db.upsert_etymon("cyning", "old-english")
        _add_descent(db, parent_id=cel_id, child_id=oe_id)
        proposals = classify_old_english(db)
    assert proposals[oe_id] == "celtic-substrate"


def test_classify_old_english_germanic_descent_does_not_collapse_into_one_bucket(
    fresh_db: Path,
) -> None:
    """Critical correctness pin (parallel to the French Latin-absence
    test). gmw-pro / proto-germanic / proto-indo-european in a
    parent set MUST NOT classify an OE etymon into any non-default
    bucket. Every OE word descends from those via the standard
    Germanic path; treating them as 'germanic-inheritance' would
    collapse the bundle into one stratum and erase the loan
    distinctions. ``_OLD_ENGLISH_ANCESTOR_TO_STRATUM`` deliberately
    excludes them for this reason."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        gmw_id = db.upsert_etymon("*haimaz", "gmw-pro")
        gem_id = db.upsert_etymon("*xaimaz", "proto-germanic")
        pie_id = db.upsert_etymon("*tḱoy-mo-", "proto-indo-european")
        oe_id = db.upsert_etymon("hām", "old-english")
        _add_descent(db, parent_id=gmw_id, child_id=oe_id)
        _add_descent(db, parent_id=gem_id, child_id=oe_id)
        _add_descent(db, parent_id=pie_id, child_id=oe_id)
        proposals = classify_old_english(db)
    # All three Germanic-descent ancestors → falls through to default.
    assert proposals[oe_id] == "native-old-english"


def test_classify_old_english_priority_latin_over_norse(fresh_db: Path) -> None:
    """Priority order: an OE etymon with BOTH a Latin and an Old Norse
    parent classifies as ``latin-loan``, not ``norse-loan``. Encodes
    the convention that the learned-loan layer (Christianization)
    dominates the Danelaw layer when both signals are present (rare
    combo but the rule needs to be deterministic)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        latin_id = db.upsert_etymon("scrīnium", "latin")
        on_id = db.upsert_etymon("skrín", "old-norse")
        oe_id = db.upsert_etymon("scrīn", "old-english")
        _add_descent(db, parent_id=latin_id, child_id=oe_id, edge_type="borrowing")
        _add_descent(db, parent_id=on_id, child_id=oe_id, edge_type="borrowing")
        proposals = classify_old_english(db)
    assert proposals[oe_id] == "latin-loan"


def test_classify_old_english_priority_norse_over_celtic(fresh_db: Path) -> None:
    """Priority: norse-loan beats celtic-substrate when both are
    present. Defensive — pins the deterministic ordering even though
    this combo is exceedingly rare in real data."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        on_id = db.upsert_etymon("kross", "old-norse")
        cel_id = db.upsert_etymon("*kross-", "proto-celtic")
        oe_id = db.upsert_etymon("cros", "old-english")
        _add_descent(db, parent_id=on_id, child_id=oe_id, edge_type="borrowing")
        _add_descent(db, parent_id=cel_id, child_id=oe_id)
        proposals = classify_old_english(db)
    assert proposals[oe_id] == "norse-loan"


def test_classify_old_english_germanic_parent_with_latin_loan_still_classifies(
    fresh_db: Path,
) -> None:
    """Defense-in-depth on the Germanic-absence invariant. An OE
    etymon with BOTH a gmw-pro parent (intentionally not in the
    ancestor map) AND a latin parent (in the map) MUST classify as
    ``latin-loan``. Pins that gmw-pro doesn't accidentally take
    priority over latin even when both are in the parent set —
    proves the absence is genuine, not a 'lower-priority alternative'."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        gmw_id = db.upsert_etymon("*spīkari", "gmw-pro")
        latin_id = db.upsert_etymon("specularia", "latin")
        oe_id = db.upsert_etymon("spīcere", "old-english")
        _add_descent(db, parent_id=gmw_id, child_id=oe_id)
        _add_descent(db, parent_id=latin_id, child_id=oe_id, edge_type="borrowing")
        proposals = classify_old_english(db)
    assert proposals[oe_id] == "latin-loan"


def test_classify_old_english_skips_merged_rows(fresh_db: Path) -> None:
    """OCR-clustering losers (merged_into_id IS NOT NULL) are skipped
    so we don't carry a stale stratum on a row callers already filter
    out. Mirrors the French / Welsh skip-merged tests."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        winner = db.upsert_etymon("dūn", "old-english")
        loser = db.upsert_etymon("dvn", "old-english")  # OCR variant
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (winner, loser),
        )
        db.commit()
        proposals = classify_old_english(db)
    assert winner in proposals
    assert loser not in proposals


def test_old_english_strata_priority_order_includes_all_buckets() -> None:
    """``OLD_ENGLISH_STRATA`` is the source-of-truth ordering. Pin
    all four buckets in the load-bearing priority order so a
    refactor that re-orders them (and changes which stratum wins on
    multi-ancestor etymons) gets caught. Note: 4 buckets, one fewer
    than Welsh / French — no dialect / period axis is supported by
    the live DB today."""
    assert OLD_ENGLISH_STRATA == (
        "latin-loan",
        "norse-loan",
        "celtic-substrate",
        "native-old-english",
    )


def test_cli_classify_stratum_old_english_dry_run_reports_counts(
    fresh_db: Path,
) -> None:
    """End-to-end CLI sanity: ``--language old-english`` dispatches to
    classify_old_english, prints per-stratum counts, doesn't write
    without ``--apply``. Mirrors the welsh / french dry-run tests."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        # One row per stratum so the summary table covers each.
        latin_id = db.upsert_etymon("scrīnium", "latin")
        latin_oe = db.upsert_etymon("scrīn", "old-english")
        _add_descent(db, parent_id=latin_id, child_id=latin_oe, edge_type="borrowing")
        on_id = db.upsert_etymon("þorp", "old-norse")
        norse_oe = db.upsert_etymon("þrop", "old-english")
        _add_descent(db, parent_id=on_id, child_id=norse_oe, edge_type="borrowing")
        cel_id = db.upsert_etymon("*kuninga", "proto-celtic")
        cel_oe = db.upsert_etymon("cyning", "old-english")
        _add_descent(db, parent_id=cel_id, child_id=cel_oe)
        db.upsert_etymon("dūn", "old-english")  # native default
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
            "old-english",
        ],
    )
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert "old-english stratum classification: 4 etymons" in combined
    for stratum in OLD_ENGLISH_STRATA:
        assert stratum in combined
    # Dry-run — no writes.
    with LexiconDB(fresh_db) as db:
        non_null = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE stratum IS NOT NULL"
        ).fetchone()[0]
    assert non_null == 0


def test_cli_classify_stratum_old_english_apply_writes_column(fresh_db: Path) -> None:
    """``--apply`` writes the proposed stratum values for OE. Pinned
    independently from welsh / french so a regression in the
    classifiers dispatch dict (e.g. wiring --apply through only one
    branch) gets caught."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        latin_id = db.upsert_etymon("monasterium", "latin")
        oe_id = db.upsert_etymon("mynster", "old-english")
        _add_descent(db, parent_id=latin_id, child_id=oe_id, edge_type="borrowing")
        native_id = db.upsert_etymon("dūn", "old-english")
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
            "old-english",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        rows = db.conn.execute(
            "SELECT id, stratum FROM etymon WHERE id IN (?, ?) ORDER BY id",
            (oe_id, native_id),
        ).fetchall()
    by_id = {r["id"]: r["stratum"] for r in rows}
    assert by_id[oe_id] == "latin-loan"
    assert by_id[native_id] == "native-old-english"
