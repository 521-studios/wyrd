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
- ``classify_old_norse``: per-stratum heuristic for ON, including
  the East-Norse self-language pass (gmq-osw / gmq-oda → east-norse)
  and the same Germanic-descent absence invariant.
- ``lexicon classify-stratum`` CLI: dry-run reports counts without
  writing; ``--apply`` writes; ``--force`` overrides existing values;
  exercised under Welsh / French / Old English / Old Norse dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, migrate_schema
from wyrd.generators.kenning.lexicon.strata import (
    BRITTONIC_STRATA,
    CELTIC_STRATA,
    FRENCH_STRATA,
    GOIDELIC_STRATA,
    OLD_ENGLISH_STRATA,
    OLD_NORSE_STRATA,
    WELSH_STRATA,
    _classify_family,
    classify_brittonic,
    classify_celtic,
    classify_french,
    classify_goidelic,
    classify_old_english,
    classify_old_norse,
    classify_stratum_all,
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
    data/seed/lexicon.sql. Pinning here so a future schema-rewrite that
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
        # wyrd-fljy: a Welsh←Latin loan is a BORROWING (cross-family; Welsh is
        # Celtic, so it cannot inherit from Latin) — a loan stratum now counts
        # only loan edge_types, so use borrowing rather than the default inheritance.
        _add_descent(db, parent_id=latin_id, child_id=welsh_id, edge_type="borrowing")
        proposals = classify_welsh(db)
    assert proposals[welsh_id] == "latin-loan"
    # Latin parent itself isn't in the welsh family — not classified.
    assert latin_id not in proposals


def test_classify_welsh_latin_variants_all_route_to_latin_loan(fresh_db: Path) -> None:
    """All period-specific Latin variants left un-normalized by the ingester
    (la-lat / la-med / la-ecc / vulgar-latin) route a Welsh loan to ``latin-loan``,
    matching the OE / ON classifiers. Pre-fix the Welsh map only listed plain
    ``latin``, so a Welsh←la-lat/la-med/la-ecc/vulgar-latin loan fell through to
    ``native-welsh`` — 75 such etymons were mis-bucketed in the live DB."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        welsh_ids = {}
        for parent_form, parent_lang in (
            ("la-lat-parent", "la-lat"),
            ("la-med-parent", "la-med"),
            ("la-ecc-parent", "la-ecc"),
            ("vulgar-latin-parent", "vulgar-latin"),
        ):
            parent_id = db.upsert_etymon(parent_form, parent_lang)
            child_id = db.upsert_etymon(f"{parent_lang}-child", "welsh")
            # wyrd-fljy: Welsh←Latin loans are borrowings (loan strata now
            # count only loan edge_types, not the default inheritance).
            _add_descent(db, parent_id=parent_id, child_id=child_id, edge_type="borrowing")
            welsh_ids[parent_lang] = child_id
        proposals = classify_welsh(db)
    for parent_lang, child_id in welsh_ids.items():
        assert proposals[child_id] == "latin-loan", parent_lang


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


def test_classify_welsh_english_variants_all_route_to_english_loan(fresh_db: Path) -> None:
    """The period/dialect/regional English variants the ingester leaves un-normalized
    (en-ear Early Modern English / enm-wmi a Middle English dialect / en-GB British
    English) route a Welsh loan to ``english-loan``, exactly as the la-* Latin variants
    do. Pre-fix the Welsh map listed only modern-/middle-/old-english + english, so a
    Welsh loan whose sole English-bearing parent used one of these fell through to
    ``native-welsh`` — 3 such etymons (sole en-ear parent) were mis-bucketed in the
    live DB. (``sco`` Scots is deliberately excluded — a distinct sister language.)

    Edge types mirror the live DB: en-ear/en-GB arrive via ``borrowing``, enm-wmi via
    ``derivation`` (e.g. Welsh ``siom`` ← ME ``schome``). All three are LOAN edge_types,
    so they route to ``english-loan`` under the wyrd-fljy loan-edge filter — an english
    INHERITANCE edge would NOT (see
    :func:`test_classify_welsh_english_inheritance_parent_not_loan`), which is correct:
    en-ear/enm-wmi/en-GB all post-date the Brythonic/English split, so any real edge from
    them into Welsh is a loan, never inheritance."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        welsh_ids = {}
        for parent_lang, edge_type in (
            ("en-ear", "borrowing"),
            ("enm-wmi", "derivation"),
            ("en-GB", "borrowing"),
        ):
            parent_id = db.upsert_etymon(f"{parent_lang}-parent", parent_lang)
            child_id = db.upsert_etymon(f"{parent_lang}-child", "welsh")
            _add_descent(db, parent_id=parent_id, child_id=child_id, edge_type=edge_type)
            welsh_ids[parent_lang] = child_id
        proposals = classify_welsh(db)
    for parent_lang, child_id in welsh_ids.items():
        assert proposals[child_id] == "english-loan", parent_lang


def test_classify_welsh_english_inheritance_parent_not_loan(fresh_db: Path) -> None:
    """wyrd-fljy: english-loan is a LOAN stratum, so it counts only loan edge_types.
    A welsh etymon reached from an english parent by INHERITANCE (mis-traced mining
    noise — no genuine cross-family inheritance) is NOT english-loan (falls through to
    native-welsh); the SAME parent via BORROWING IS english-loan. Guards the 501
    mis-typed modern-english→welsh inheritance edges in the live DB."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eng_id = db.upsert_etymon("borrow-src", "modern-english")
        borrowed = db.upsert_etymon("borrowed-child", "welsh")
        inherited = db.upsert_etymon("inherited-child", "welsh")
        _add_descent(db, parent_id=eng_id, child_id=borrowed, edge_type="borrowing")
        _add_descent(db, parent_id=eng_id, child_id=inherited, edge_type="inheritance")
        proposals = classify_welsh(db)
    assert proposals[borrowed] == "english-loan"
    assert proposals[inherited] == "native-welsh"  # inheritance excluded from the loan stratum


def test_classify_welsh_substrate_fires_via_inheritance(fresh_db: Path) -> None:
    """wyrd-fljy: substrate strata are NOT loan-class, so they keep ALL edge types — a
    substrate legitimately rides inheritance. A welsh etymon INHERITING from a
    proto-celtic (brittonic-substrate) parent is still brittonic-substrate."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        pc_id = db.upsert_etymon("proto-celtic-root", "proto-celtic")
        welsh_id = db.upsert_etymon("welsh-child", "welsh")
        _add_descent(db, parent_id=pc_id, child_id=welsh_id, edge_type="inheritance")
        proposals = classify_welsh(db)
    assert proposals[welsh_id] == "brittonic-substrate"


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


def test_classify_welsh_priority_latin_variant_over_brittonic(fresh_db: Path) -> None:
    """Defense-in-depth for the variant fix: a Latin VARIANT loan (la-lat) still
    wins over a Brittonic ancestor by priority order — not just plain ``latin``.
    Without the variant in the ancestor map the la-lat parent would be unmatched,
    the brittonic parent would win, and this would mis-classify brittonic-substrate."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        la_lat_id = db.upsert_etymon("ecclesia", "la-lat")
        bry_id = db.upsert_etymon("*ekklesia", "cel-bry-pro")
        welsh_id = db.upsert_etymon("eglwys", "welsh")
        _add_descent(db, parent_id=la_lat_id, child_id=welsh_id, edge_type="borrowing")
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
        # wyrd-fljy: borrowing (a loan edge) so latin-loan fires under the filter.
        _add_descent(db, parent_id=latin_id, child_id=welsh_id, edge_type="borrowing")

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
    vocabulary for. Use a clearly-fictional language so the test
    remains stable as the classifier's Choice list grows — picking
    a real out-of-scope language (e.g. modern-english, icelandic)
    would silently start failing the day that language enters the
    Choice list."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "classify-stratum",
            "--db",
            str(fresh_db),
            "--language",
            "klingon",
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "klingon" in combined.lower() or "invalid value" in combined.lower()


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


def test_classify_old_english_christian_era_greek_loan_folds_into_latin_loan(
    fresh_db: Path,
) -> None:
    """Defense for the Christianization-via-Latin-channels semantic
    decision. ancient-greek parents on an OE etymon route to
    ``latin-loan`` because pre-modern Greek loans into OE arrived
    through Latin / clerical pathways (Anglo-Saxon scriptoria, the
    7th-c. Theodore-of-Tarsus school at Canterbury). Pinned as a
    dedicated test rather than just a batch entry so a refactor that
    drops 'ancient-greek' from _OLD_ENGLISH_ANCESTOR_TO_STRATUM
    doesn't silently survive (the parameterized 5-case test would
    still pass with 4 cases)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        greek_id = db.upsert_etymon("ekklesia", "ancient-greek")
        oe_id = db.upsert_etymon("cirice", "old-english")
        _add_descent(db, parent_id=greek_id, child_id=oe_id, edge_type="borrowing")
        proposals = classify_old_english(db)
    assert proposals[oe_id] == "latin-loan"


def test_classify_old_english_unmapped_dialect_self_language_not_classified(
    fresh_db: Path,
) -> None:
    """The umbrella ticket scoped dialect strata (West Saxon vs
    Mercian vs Anglian) but the live DB has only 33 dialect-coded
    rows (31 ang-ang Anglian + 2 ang-wsx West Saxon, all orphans),
    so _OLD_ENGLISH_SELF_LANGUAGE_TO_STRATUM stays empty. Pin that
    etymons seeded with the actual dialect codes used in the live DB
    are NOT silently classified — confirms the empty-map invariant
    and pins the family-membership scope to language='old-english'
    alone.

    If wave-2 mining ever ingests dialect-coded OE corpora at scale,
    this test needs updating in lockstep with the self-language map
    population — a deliberate regression-flagging point for the
    follow-up work."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        # Use the actual dialect codes from the live DB (ang-ang for
        # Anglian, ang-wsx for West Saxon — Wiktionary's wave-2
        # convention) rather than the ang-x-* hyphen-form codes that
        # exist in some scholarly references but not in our data.
        anglian_id = db.upsert_etymon("hām", "ang-ang")
        wsx_id = db.upsert_etymon("hām", "ang-wsx")
        proposals = classify_old_english(db)
    # Both dialect-coded rows fall outside the classifier's scope
    # (modern_lang='old-english' only, empty self-language map).
    assert anglian_id not in proposals
    assert wsx_id not in proposals


def test_cli_classify_stratum_old_english_force_overwrites(fresh_db: Path) -> None:
    """``--apply --force`` overwrites an existing stratum value when
    dispatched to the OE classifier. Mirror of the French force
    test (the force/skip logic shares a code path, but pin the OE
    invocation so a regression that wires --force through only some
    branches of the dispatch dict gets caught)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        oe_id = db.upsert_etymon("dūn", "old-english")
        # Pre-populate with a wrong stratum so we can verify the
        # overwrite. classify_old_english would assign
        # 'native-old-english' to this row (no descent → default).
        db.conn.execute(
            "UPDATE etymon SET stratum = ? WHERE id = ?",
            ("manual-override", oe_id),
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
            "old-english",
            "--apply",
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (oe_id,)).fetchone()[
            "stratum"
        ]
    assert stratum == "native-old-english"


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


# --- classify_old_norse (wyrd-lr4 Phase 4c) -------------------------------


def test_classify_old_norse_assigns_native_when_no_descent(fresh_db: Path) -> None:
    """An ON etymon with NO ``etymon_descent`` parents falls into the
    ``native-old-norse`` default. Most-common shape — the standard
    Germanic descent path is intentionally absent from the ancestor
    map (would collapse the bundle), so the default catches all
    standard-descent ON etymons."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("þorp", "old-norse")
        proposals = classify_old_norse(db)
    assert proposals == {eid: "native-old-norse"}


def test_classify_old_norse_latin_loan_via_descent(fresh_db: Path) -> None:
    """An ON etymon descending from Latin gets ``latin-loan`` —
    Christianization-era learned vocabulary. Same convention as OE."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        latin_id = db.upsert_etymon("ecclesia", "latin")
        on_id = db.upsert_etymon("kirkja", "old-norse")
        _add_descent(db, parent_id=latin_id, child_id=on_id, edge_type="borrowing")
        proposals = classify_old_norse(db)
    assert proposals[on_id] == "latin-loan"


def test_classify_old_norse_low_german_loan_via_descent(fresh_db: Path) -> None:
    """An ON etymon descending from Middle Low German (gml) gets
    ``low-german-loan``. Encodes the Hanseatic League trade-contact
    layer (~1100-1300 CE) — the merchant register that distinguishes
    later Old Norse vocabulary."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        gml_id = db.upsert_etymon("klippe", "gml")
        on_id = db.upsert_etymon("klippa", "old-norse")
        _add_descent(db, parent_id=gml_id, child_id=on_id, edge_type="borrowing")
        proposals = classify_old_norse(db)
    assert proposals[on_id] == "low-german-loan"


def test_classify_old_norse_english_loan_via_descent(fresh_db: Path) -> None:
    """An ON etymon descending from Old English or Old Saxon gets
    ``english-loan``. Encodes Viking-Age cross-North-Sea contact
    (Anglo-Saxon and Continental Saxon borrowings into ON, ~800-1100
    CE)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        oe_id = db.upsert_etymon("scōl", "old-english")
        oe_borrow = db.upsert_etymon("skóli", "old-norse")
        _add_descent(db, parent_id=oe_id, child_id=oe_borrow, edge_type="borrowing")
        osx_id = db.upsert_etymon("skuld", "osx")
        osx_borrow = db.upsert_etymon("skuld", "old-norse")
        _add_descent(db, parent_id=osx_id, child_id=osx_borrow, edge_type="borrowing")
        proposals = classify_old_norse(db)
    assert proposals[oe_borrow] == "english-loan"
    assert proposals[osx_borrow] == "english-loan"


def test_classify_old_norse_gaelic_substrate_via_descent(fresh_db: Path) -> None:
    """An ON etymon descending from a Goidelic / Brittonic ancestor
    gets ``gaelic-substrate``. Rare in the live DB (~64 etymons
    total) but distinctive — Norse-Gaelic settlement contact in the
    Irish Sea region (Dublin, Mann, Western Isles)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        oi_id = db.upsert_etymon("airigid", "old-irish")
        on_id = db.upsert_etymon("ǽrgi", "old-norse")
        _add_descent(db, parent_id=oi_id, child_id=on_id, edge_type="borrowing")
        proposals = classify_old_norse(db)
    assert proposals[on_id] == "gaelic-substrate"


def test_classify_old_norse_east_norse_via_self_language(fresh_db: Path) -> None:
    """The umbrella ticket's Eastern (Swedish/Danish) vs Western
    (Norwegian/Icelandic) axis maps to a self-language pass: an
    etymon whose own language is gmq-osw (Old Swedish) or gmq-oda
    (Old Danish) classifies as ``east-norse`` directly. These ARE
    the Eastern Norse varieties; classifying by descent would
    describe what fed INTO them rather than their own register."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        osw_id = db.upsert_etymon("kalla", "gmq-osw")
        oda_id = db.upsert_etymon("kalde", "gmq-oda")
        proposals = classify_old_norse(db)
    assert proposals[osw_id] == "east-norse"
    assert proposals[oda_id] == "east-norse"


def test_classify_old_norse_germanic_descent_does_not_collapse_into_one_bucket(
    fresh_db: Path,
) -> None:
    """Critical correctness pin (parallel to French Latin-absence and
    OE Germanic-absence). proto-germanic / proto-indo-european /
    gmq-pro / gmw-pro in the parent set MUST NOT classify an ON
    etymon into any non-default bucket. Every ON word descends from
    those via the standard Germanic path; treating them as
    'germanic-inheritance' would collapse the bundle into one
    stratum and erase the loan distinctions. Pinned by including
    ALL FOUR standard ancestors in the parent set with no loan
    signals."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        gem_id = db.upsert_etymon("*þurpą", "proto-germanic")
        gmq_id = db.upsert_etymon("*þurpa", "gmq-pro")
        gmw_id = db.upsert_etymon("*þurpą", "gmw-pro")
        pie_id = db.upsert_etymon("*treb-", "proto-indo-european")
        on_id = db.upsert_etymon("þorp", "old-norse")
        _add_descent(db, parent_id=gem_id, child_id=on_id)
        _add_descent(db, parent_id=gmq_id, child_id=on_id)
        _add_descent(db, parent_id=gmw_id, child_id=on_id)
        _add_descent(db, parent_id=pie_id, child_id=on_id)
        proposals = classify_old_norse(db)
    # All four standard Germanic-descent ancestors → falls through to default.
    assert proposals[on_id] == "native-old-norse"


def test_classify_old_norse_priority_latin_over_low_german(fresh_db: Path) -> None:
    """Priority order: an ON etymon with BOTH a Latin and a Middle
    Low German parent classifies as ``latin-loan``, not
    ``low-german-loan``. Encodes the convention that the
    institutional / clerical Christianization signal dominates the
    merchant / Hanseatic signal when both coexist (rare combo, but
    deterministic priority is load-bearing)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        latin_id = db.upsert_etymon("schola", "latin")
        gml_id = db.upsert_etymon("schole", "gml")
        on_id = db.upsert_etymon("skóli", "old-norse")
        _add_descent(db, parent_id=latin_id, child_id=on_id, edge_type="borrowing")
        _add_descent(db, parent_id=gml_id, child_id=on_id, edge_type="borrowing")
        proposals = classify_old_norse(db)
    assert proposals[on_id] == "latin-loan"


def test_classify_old_norse_priority_low_german_over_english(fresh_db: Path) -> None:
    """Priority: low-german-loan beats english-loan when both are
    present. Defensive — pins the deterministic ordering even though
    the combo is rare."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        gml_id = db.upsert_etymon("kjøbe", "gml")
        oe_id = db.upsert_etymon("ċēap", "old-english")
        on_id = db.upsert_etymon("kaupa", "old-norse")
        _add_descent(db, parent_id=gml_id, child_id=on_id, edge_type="borrowing")
        _add_descent(db, parent_id=oe_id, child_id=on_id, edge_type="borrowing")
        proposals = classify_old_norse(db)
    assert proposals[on_id] == "low-german-loan"


def test_classify_old_norse_priority_english_over_gaelic(fresh_db: Path) -> None:
    """Priority: english-loan beats gaelic-substrate when both are
    present. Completes the loan-priority chain (latin > low-german
    > english > gaelic) so a refactor that swaps any adjacent pair
    in OLD_NORSE_STRATA gets caught at the resolution-behavior
    layer rather than only by the tuple-shape pin."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        oe_id = db.upsert_etymon("scōl", "old-english")
        oi_id = db.upsert_etymon("scol", "old-irish")
        on_id = db.upsert_etymon("skóli", "old-norse")
        _add_descent(db, parent_id=oe_id, child_id=on_id, edge_type="borrowing")
        _add_descent(db, parent_id=oi_id, child_id=on_id, edge_type="borrowing")
        proposals = classify_old_norse(db)
    assert proposals[on_id] == "english-loan"


def test_classify_old_norse_east_norse_self_lang_holds_against_ancestor_walk(
    fresh_db: Path,
) -> None:
    """The self-language pass runs first in _classify_family; the
    ancestor-walk pass runs second over modern_lang rows only. For
    ON, gmq-osw / gmq-oda rows are NOT in modern_ids (modern_lang is
    'old-norse'), so a gmq-osw etymon's east-norse assignment from
    the self-language pass is structurally protected from being
    overwritten by the ancestor walk. Pin that even when a gmq-osw
    row carries a latin parent, the east-norse classification holds.

    Catches a future refactor that changes _classify_family's
    two-pass shape (e.g. running the ancestor walk over self-lang
    rows too) — the silent overwrite would change which stratum
    wins for East Norse rows with foreign loan ancestors."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        latin_id = db.upsert_etymon("ecclesia", "latin")
        osw_id = db.upsert_etymon("kirkia", "gmq-osw")
        # Add a latin parent edge — would route to latin-loan if the
        # ancestor walk fired on this row.
        _add_descent(db, parent_id=latin_id, child_id=osw_id, edge_type="borrowing")
        proposals = classify_old_norse(db)
    # Self-language pass assigned east-norse; ancestor walk did NOT
    # touch this row (modern_lang='old-norse' only).
    assert proposals[osw_id] == "east-norse"


def test_classify_old_norse_germanic_parent_with_low_german_loan_still_classifies(
    fresh_db: Path,
) -> None:
    """Defense-in-depth on the Germanic-absence invariant for ON
    (parallel to the OE gmw-pro+latin test). An ON etymon with BOTH
    a proto-germanic parent (intentionally not in the ancestor map)
    AND a gml parent (in the map) MUST classify as
    ``low-german-loan``. Pins that proto-germanic doesn't
    accidentally take priority over a real loan signal — proves the
    absence is genuine, not a 'lower-priority alternative'."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        gem_id = db.upsert_etymon("*kaupą", "proto-germanic")
        gml_id = db.upsert_etymon("kopen", "gml")
        on_id = db.upsert_etymon("kaupa", "old-norse")
        _add_descent(db, parent_id=gem_id, child_id=on_id)
        _add_descent(db, parent_id=gml_id, child_id=on_id, edge_type="borrowing")
        proposals = classify_old_norse(db)
    assert proposals[on_id] == "low-german-loan"


def test_classify_old_norse_christian_era_greek_loan_folds_into_latin_loan(
    fresh_db: Path,
) -> None:
    """Same Christianization-via-Latin-channels semantic as OE: an ON
    etymon with an ancient-greek parent routes to ``latin-loan``
    because pre-modern Greek loans into ON arrived through Latin /
    clerical pathways. Pinned dedicated rather than implicit so a
    refactor that drops 'ancient-greek' from
    _OLD_NORSE_ANCESTOR_TO_STRATUM gets caught."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        greek_id = db.upsert_etymon("kyrios", "ancient-greek")
        on_id = db.upsert_etymon("kirkja", "old-norse")
        _add_descent(db, parent_id=greek_id, child_id=on_id, edge_type="borrowing")
        proposals = classify_old_norse(db)
    assert proposals[on_id] == "latin-loan"


def test_classify_old_norse_skips_merged_rows(fresh_db: Path) -> None:
    """OCR-clustering losers (merged_into_id IS NOT NULL) are skipped
    in BOTH passes (self-language for east-norse + ancestor walk for
    old-norse), so we don't carry a stale stratum on a row callers
    already filter out."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        winner = db.upsert_etymon("þorp", "old-norse")
        loser = db.upsert_etymon("þrop", "old-norse")  # OCR variant
        # Also a loser in the East Norse self-language pass.
        east_loser = db.upsert_etymon("kallé", "gmq-osw")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id IN (?, ?)",
            (winner, loser, east_loser),
        )
        db.commit()
        proposals = classify_old_norse(db)
    assert winner in proposals
    assert loser not in proposals
    assert east_loser not in proposals


def test_classify_old_norse_proto_norse_falls_through_to_default(fresh_db: Path) -> None:
    """gmq-pro (Proto-Norse, ~423 rows) is intentionally NOT in the
    self-language map — it's the common ancestor of both East and
    West Norse traditions and folding into either bucket would
    bias the East/West dichotomy. Pin that gmq-pro etymons fall
    outside the classifier's modern_lang scope (modern_lang is
    'old-norse' only) and aren't classified at all."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        gmq_id = db.upsert_etymon("*þurpa", "gmq-pro")
        proposals = classify_old_norse(db)
    # gmq-pro is outside modern_lang scope and outside self-language
    # map — silently absent from proposals.
    assert gmq_id not in proposals


def test_old_norse_strata_priority_order_includes_all_buckets() -> None:
    """``OLD_NORSE_STRATA`` is the source-of-truth ordering. Six
    buckets — one more than Welsh / French because the Eastern /
    Western dialect axis adds a regional dimension orthogonal to
    the loan strata. Pin the load-bearing priority order so a
    refactor that re-orders them (and changes which stratum wins
    on multi-ancestor etymons) gets caught."""
    assert OLD_NORSE_STRATA == (
        "latin-loan",
        "low-german-loan",
        "english-loan",
        "gaelic-substrate",
        "east-norse",
        "native-old-norse",
    )


def test_cli_classify_stratum_old_norse_dry_run_reports_counts(
    fresh_db: Path,
) -> None:
    """End-to-end CLI sanity: ``--language old-norse`` dispatches to
    classify_old_norse, prints per-stratum counts, doesn't write
    without ``--apply``. Mirrors the dry-run tests for the other
    language families."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        # One row per stratum so the summary table covers each.
        latin_id = db.upsert_etymon("ecclesia", "latin")
        latin_on = db.upsert_etymon("kirkja", "old-norse")
        _add_descent(db, parent_id=latin_id, child_id=latin_on, edge_type="borrowing")
        gml_id = db.upsert_etymon("klippe", "gml")
        gml_on = db.upsert_etymon("klippa", "old-norse")
        _add_descent(db, parent_id=gml_id, child_id=gml_on, edge_type="borrowing")
        oe_id = db.upsert_etymon("scōl", "old-english")
        oe_on = db.upsert_etymon("skóli", "old-norse")
        _add_descent(db, parent_id=oe_id, child_id=oe_on, edge_type="borrowing")
        oi_id = db.upsert_etymon("airigid", "old-irish")
        oi_on = db.upsert_etymon("ǽrgi", "old-norse")
        _add_descent(db, parent_id=oi_id, child_id=oi_on, edge_type="borrowing")
        db.upsert_etymon("kalla", "gmq-osw")  # east-norse via self-lang
        db.upsert_etymon("þorp", "old-norse")  # native default
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
            "old-norse",
        ],
    )
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert "old-norse stratum classification: 6 etymons" in combined
    for stratum in OLD_NORSE_STRATA:
        assert stratum in combined
    # Dry-run — no writes.
    with LexiconDB(fresh_db) as db:
        non_null = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE stratum IS NOT NULL"
        ).fetchone()[0]
    assert non_null == 0


def test_cli_classify_stratum_old_norse_apply_writes_column(fresh_db: Path) -> None:
    """``--apply`` writes the proposed stratum values for ON. Pinned
    independently so a regression in the classifiers dispatch dict
    (e.g. wiring through only some branches) gets caught."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        gml_id = db.upsert_etymon("klippe", "gml")
        on_id = db.upsert_etymon("klippa", "old-norse")
        _add_descent(db, parent_id=gml_id, child_id=on_id, edge_type="borrowing")
        east_id = db.upsert_etymon("kalla", "gmq-osw")
        native_id = db.upsert_etymon("þorp", "old-norse")
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
            "old-norse",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        rows = db.conn.execute(
            "SELECT id, stratum FROM etymon WHERE id IN (?, ?, ?) ORDER BY id",
            (on_id, east_id, native_id),
        ).fetchall()
    by_id = {r["id"]: r["stratum"] for r in rows}
    assert by_id[on_id] == "low-german-loan"
    assert by_id[east_id] == "east-norse"
    assert by_id[native_id] == "native-old-norse"


def test_cli_classify_stratum_old_norse_force_overwrites(fresh_db: Path) -> None:
    """``--apply --force`` overwrites an existing stratum value when
    dispatched to the ON classifier. Mirror of the welsh / french /
    OE force tests — pin the ON branch of the dispatch dict
    end-to-end."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        on_id = db.upsert_etymon("þorp", "old-norse")
        db.conn.execute(
            "UPDATE etymon SET stratum = ? WHERE id = ?",
            ("manual-override", on_id),
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
            "old-norse",
            "--apply",
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (on_id,)).fetchone()[
            "stratum"
        ]
    assert stratum == "native-old-norse"


# --- classify_brittonic / classify_goidelic / classify_celtic (wyrd-de77) -
#
# The Celtic-family classifiers cover the coarse 'brittonic' /
# 'goidelic' / 'celtic' leaf tags. Three load-bearing properties
# distinguish them from Welsh / OE / ON: (1) 87% of the cohort are
# orphans → native-* default, (2) NO loan buckets (data carries no
# loan ancestries), (3) proto-germanic parents are mining noise and
# fold to default.


def test_classify_brittonic_assigns_native_when_no_descent(fresh_db: Path) -> None:
    """A brittonic etymon with NO descent parents → native-brittonic
    (the dominant case — 87% of the Celtic cohort are orphans)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("*brigantī", "brittonic")
        proposals = classify_brittonic(db)
    assert proposals == {eid: "native-brittonic"}


def test_classify_brittonic_proto_celtic_substrate_via_descent(fresh_db: Path) -> None:
    """A brittonic etymon descending from proto-celtic / cel-bry-pro →
    proto-celtic-substrate."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        pc_id = db.upsert_etymon("*brigant-", "proto-celtic")
        br_id = db.upsert_etymon("*brigantī", "brittonic")
        _add_descent(db, parent_id=pc_id, child_id=br_id)
        cbp_id = db.upsert_etymon("*brɨɣant", "cel-bry-pro")
        br2_id = db.upsert_etymon("*breɣant", "brittonic")
        _add_descent(db, parent_id=cbp_id, child_id=br2_id)
        proposals = classify_brittonic(db)
    assert proposals[br_id] == "proto-celtic-substrate"
    assert proposals[br2_id] == "proto-celtic-substrate"


def test_classify_brittonic_proto_germanic_parent_is_noise_stays_default(
    fresh_db: Path,
) -> None:
    """proto-germanic parents are mining NOISE and intentionally NOT
    in the ancestor map (mirrors every other classifier) — a brittonic
    etymon with only a proto-germanic parent falls through to the
    native-brittonic default."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        gem_id = db.upsert_etymon("*berhtaz", "proto-germanic")
        br_id = db.upsert_etymon("*brigantī", "brittonic")
        _add_descent(db, parent_id=gem_id, child_id=br_id)
        proposals = classify_brittonic(db)
    assert proposals[br_id] == "native-brittonic"


def test_classify_brittonic_skips_merged_rows(fresh_db: Path) -> None:
    """OCR-clustering losers (merged_into_id IS NOT NULL) are skipped."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        winner = db.upsert_etymon("*brigantī", "brittonic")
        loser = db.upsert_etymon("*brigant1", "brittonic")  # OCR variant
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (winner, loser),
        )
        db.commit()
        proposals = classify_brittonic(db)
    assert winner in proposals
    assert loser not in proposals


def test_brittonic_strata_priority_order_includes_all_buckets() -> None:
    """``BRITTONIC_STRATA`` source-of-truth ordering: substrate first,
    native default last. Two buckets (no loan strata — the data carries
    no loan ancestries on Brittonic rows)."""
    assert BRITTONIC_STRATA == (
        "proto-celtic-substrate",
        "native-brittonic",
    )


def test_classify_goidelic_assigns_native_when_no_descent(fresh_db: Path) -> None:
    """A goidelic etymon with NO descent parents → native-goidelic."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("*windos", "goidelic")
        proposals = classify_goidelic(db)
    assert proposals == {eid: "native-goidelic"}


def test_classify_goidelic_proto_celtic_substrate_via_descent(fresh_db: Path) -> None:
    """A goidelic etymon descending from proto-celtic →
    proto-celtic-substrate."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        pc_id = db.upsert_etymon("*windos", "proto-celtic")
        go_id = db.upsert_etymon("finn", "goidelic")
        _add_descent(db, parent_id=pc_id, child_id=go_id)
        proposals = classify_goidelic(db)
    assert proposals[go_id] == "proto-celtic-substrate"


def test_classify_goidelic_old_irish_via_descent(fresh_db: Path) -> None:
    """A goidelic etymon descending from old-irish → old-irish stratum."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        oi_id = db.upsert_etymon("dún", "old-irish")
        go_id = db.upsert_etymon("dún", "goidelic")
        _add_descent(db, parent_id=oi_id, child_id=go_id)
        proposals = classify_goidelic(db)
    assert proposals[go_id] == "old-irish"


def test_classify_goidelic_middle_irish_via_descent(fresh_db: Path) -> None:
    """A goidelic etymon descending from middle-irish →
    middle-irish stratum."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        mi_id = db.upsert_etymon("baile", "middle-irish")
        go_id = db.upsert_etymon("baile", "goidelic")
        _add_descent(db, parent_id=mi_id, child_id=go_id)
        proposals = classify_goidelic(db)
    assert proposals[go_id] == "middle-irish"


def test_classify_goidelic_priority_substrate_over_gaelic_stages(fresh_db: Path) -> None:
    """Priority order pins the tuple: a goidelic etymon with BOTH a
    proto-celtic and an old-irish ancestor classifies as the
    substrate (it leads OLD_GOIDELIC priority order)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        pc_id = db.upsert_etymon("*windos", "proto-celtic")
        oi_id = db.upsert_etymon("finn", "old-irish")
        go_id = db.upsert_etymon("fionn", "goidelic")
        _add_descent(db, parent_id=pc_id, child_id=go_id)
        _add_descent(db, parent_id=oi_id, child_id=go_id)
        proposals = classify_goidelic(db)
    assert proposals[go_id] == "proto-celtic-substrate"


def test_classify_goidelic_proto_germanic_parent_is_noise_stays_default(
    fresh_db: Path,
) -> None:
    """proto-germanic parents are mining noise → native-goidelic default."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        gem_id = db.upsert_etymon("*gardaz", "proto-germanic")
        go_id = db.upsert_etymon("gort", "goidelic")
        _add_descent(db, parent_id=gem_id, child_id=go_id)
        proposals = classify_goidelic(db)
    assert proposals[go_id] == "native-goidelic"


def test_classify_goidelic_skips_merged_rows(fresh_db: Path) -> None:
    """OCR-clustering losers are skipped."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        winner = db.upsert_etymon("dún", "goidelic")
        loser = db.upsert_etymon("dvn", "goidelic")  # OCR variant
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (winner, loser),
        )
        db.commit()
        proposals = classify_goidelic(db)
    assert winner in proposals
    assert loser not in proposals


def test_goidelic_strata_priority_order_includes_all_buckets() -> None:
    """``GOIDELIC_STRATA`` source-of-truth ordering: substrate first,
    then the attested Gaelic stages (old-irish, middle-irish), native
    default last. Four buckets, no loan strata."""
    assert GOIDELIC_STRATA == (
        "proto-celtic-substrate",
        "old-irish",
        "middle-irish",
        "native-goidelic",
    )


def test_classify_celtic_assigns_native_when_no_descent(fresh_db: Path) -> None:
    """A celtic etymon with NO descent parents → native-celtic."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("*dūnom", "celtic")
        proposals = classify_celtic(db)
    assert proposals == {eid: "native-celtic"}


def test_classify_celtic_proto_celtic_substrate_via_descent(fresh_db: Path) -> None:
    """A celtic etymon descending from proto-celtic / cel-bry-pro →
    proto-celtic-substrate."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        pc_id = db.upsert_etymon("*dūnom", "proto-celtic")
        ce_id = db.upsert_etymon("dun", "celtic")
        _add_descent(db, parent_id=pc_id, child_id=ce_id)
        cbp_id = db.upsert_etymon("*dīnas", "cel-bry-pro")
        ce2_id = db.upsert_etymon("din", "celtic")
        _add_descent(db, parent_id=cbp_id, child_id=ce2_id)
        proposals = classify_celtic(db)
    assert proposals[ce_id] == "proto-celtic-substrate"
    assert proposals[ce2_id] == "proto-celtic-substrate"


def test_classify_celtic_mixed_branch_ancestry_folds_to_default(fresh_db: Path) -> None:
    """The generic celtic tag deliberately does NOT map the mixed
    old-irish / old-welsh ancestries some rows carry — it can't honestly
    claim a branch-specific stage, so they fold to native-celtic."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        oi_id = db.upsert_etymon("cathair", "old-irish")
        ow_id = db.upsert_etymon("cair", "old-welsh")
        ce_id = db.upsert_etymon("caer", "celtic")
        _add_descent(db, parent_id=oi_id, child_id=ce_id)
        _add_descent(db, parent_id=ow_id, child_id=ce_id)
        proposals = classify_celtic(db)
    assert proposals[ce_id] == "native-celtic"


def test_classify_celtic_proto_germanic_parent_is_noise_stays_default(
    fresh_db: Path,
) -> None:
    """proto-germanic parents are mining noise → native-celtic default."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        gem_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        ce_id = db.upsert_etymon("dun", "celtic")
        _add_descent(db, parent_id=gem_id, child_id=ce_id)
        proposals = classify_celtic(db)
    assert proposals[ce_id] == "native-celtic"


def test_classify_celtic_skips_merged_rows(fresh_db: Path) -> None:
    """OCR-clustering losers are skipped."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        winner = db.upsert_etymon("dun", "celtic")
        loser = db.upsert_etymon("dnu", "celtic")  # OCR variant
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (winner, loser),
        )
        db.commit()
        proposals = classify_celtic(db)
    assert winner in proposals
    assert loser not in proposals


def test_celtic_strata_priority_order_includes_all_buckets() -> None:
    """``CELTIC_STRATA`` source-of-truth ordering: substrate first,
    native default last. Two buckets, no loan strata."""
    assert CELTIC_STRATA == (
        "proto-celtic-substrate",
        "native-celtic",
    )


def test_cli_classify_stratum_celtic_families_dispatch(fresh_db: Path) -> None:
    """End-to-end CLI sanity: the three Celtic-family --language
    choices dispatch to their classifiers and write via --apply.
    Pins the Choice list + dispatch table + STRATA-tuple wiring for
    all three at once (one row each, orphan → native-*)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        br_id = db.upsert_etymon("*brigantī", "brittonic")
        go_id = db.upsert_etymon("*windos", "goidelic")
        ce_id = db.upsert_etymon("*dūnom", "celtic")
        db.commit()

    runner = CliRunner()
    for language, eid, expected in (
        ("brittonic", br_id, "native-brittonic"),
        ("goidelic", go_id, "native-goidelic"),
        ("celtic", ce_id, "native-celtic"),
    ):
        result = runner.invoke(
            cli_root,
            [
                "lexicon",
                "classify-stratum",
                "--db",
                str(fresh_db),
                "--language",
                language,
                "--apply",
            ],
        )
        assert result.exit_code == 0, result.output + (result.stderr or "")
        with LexiconDB(fresh_db) as db:
            stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (eid,)).fetchone()[
                "stratum"
            ]
        assert stratum == expected, language


# --- lexicon set-stratum (wyrd-lr4 Phase 4d) ------------------------------


def test_cli_set_stratum_by_etymon_id_writes_value(fresh_db: Path) -> None:
    """Happy path — pass --etymon-id and --stratum, verify the row's
    stratum is updated."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("caer", "welsh")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(eid),
            "--stratum",
            "latin-loan",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (eid,)).fetchone()[
            "stratum"
        ]
    assert stratum == "latin-loan"


def test_cli_set_stratum_by_canonical_form_and_language(fresh_db: Path) -> None:
    """Happy path — pass --canonical-form + --language for the
    pair-lookup variant. The lookup must resolve to exactly one row."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("ville", "french")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--canonical-form",
            "ville",
            "--language",
            "french",
            "--stratum",
            "frankish-substrate",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (eid,)).fetchone()[
            "stratum"
        ]
    assert stratum == "frankish-substrate"


def test_cli_set_stratum_clear_sets_null(fresh_db: Path) -> None:
    """``--clear`` writes NULL to the row's stratum (escape hatch
    when the classifier wrote something the operator now disagrees
    with — re-running classify-stratum --apply will then re-classify
    the cleared row on the next pass)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("caer", "welsh")
        db.conn.execute("UPDATE etymon SET stratum = ? WHERE id = ?", ("latin-loan", eid))
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(eid),
            "--clear",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (eid,)).fetchone()[
            "stratum"
        ]
    assert stratum is None


def test_cli_set_stratum_reports_old_and_new_to_stderr(fresh_db: Path) -> None:
    """The success path emits a stderr line of the form
    'etymon N (canonical_form, language): <old> → <new>' so the
    operator can verify the change. Pin both old and new values
    appear so a regression that drops the diff signal is caught."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("caer", "welsh")
        db.conn.execute("UPDATE etymon SET stratum = ? WHERE id = ?", ("native-welsh", eid))
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(eid),
            "--stratum",
            "latin-loan",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    combined = result.output + (result.stderr or "")
    assert f"etymon {eid}" in combined
    assert "caer" in combined
    assert "welsh" in combined
    assert "native-welsh" in combined
    assert "latin-loan" in combined
    # Defense against a regression where the stderr message is
    # constructed but the UPDATE is skipped — verify the row was
    # actually rewritten with the new value.
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (eid,)).fetchone()[
            "stratum"
        ]
    assert stratum == "latin-loan"


def test_cli_set_stratum_rejects_cross_family_stratum(fresh_db: Path) -> None:
    """wyrd-j3gy family-aware validation: setting a Welsh stratum on
    a French row is rejected with a friendly error listing the
    French family's valid strata. Replaces the Phase 4d behavior of
    silently allowing cross-family writes."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("ville", "french")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(eid),
            "--stratum",
            "native-welsh",  # Welsh stratum on French etymon.
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "native-welsh" in combined
    assert "french" in combined.lower()
    # The error message should list the French family's valid strata
    # so the operator can pick the right one.
    assert "frankish-substrate" in combined
    # Original row stratum stays NULL — failed validation didn't write.
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (eid,)).fetchone()[
            "stratum"
        ]
    assert stratum is None


def test_cli_set_stratum_unclassified_language_falls_back_to_all_strata(
    fresh_db: Path,
) -> None:
    """Languages without a classifier (no entry in
    LANGUAGE_TO_FAMILY) skip the family-aware check and fall back to
    the ALL_STRATA typo-protection only. Pin so an operator can
    still hand-correct on unclassified rows. Will need updating when
    wyrd-lr4 Phase 4e+ adds a classifier for the chosen language."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        # 'old-frisian' (ofs) has no Phase 4 classifier — sister
        # West Germanic language outside the OE / ON / French scopes.
        eid = db.upsert_etymon("hús", "ofs")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(eid),
            "--stratum",
            "latin-loan",  # in ALL_STRATA, not tied to any specific family
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (eid,)).fetchone()[
            "stratum"
        ]
    assert stratum == "latin-loan"


def test_cli_set_stratum_clear_when_already_null_succeeds(fresh_db: Path) -> None:
    """``--clear`` on a row whose stratum is already NULL succeeds
    cleanly with the diff line showing 'None → None'. Pin so a
    'skip if already NULL' optimization that suppressed the stderr
    audit line doesn't slip in undetected."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("caer", "welsh")
        # No stratum write — row stays NULL.
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(eid),
            "--clear",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    combined = result.output + (result.stderr or "")
    # Both the old and new sides render as 'None'.
    assert "None → None" in combined
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (eid,)).fetchone()[
            "stratum"
        ]
    assert stratum is None


def test_cli_set_stratum_rejects_unknown_stratum(fresh_db: Path) -> None:
    """Stratum value validated against the union of all four families'
    STRATA tuples (ALL_STRATA frozenset). Catches typos like
    'lattin-loan' before they silently write garbage to the column."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("caer", "welsh")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(eid),
            "--stratum",
            "lattin-loan",
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "unknown stratum" in combined.lower()
    # The error message should list known strata so the operator
    # can fix the typo without re-reading the source.
    assert "latin-loan" in combined
    # Original row stratum stays NULL — the failed write didn't take.
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (eid,)).fetchone()[
            "stratum"
        ]
    assert stratum is None


def test_cli_set_stratum_rejects_missing_etymon(fresh_db: Path) -> None:
    """A nonexistent --etymon-id returns a friendly error rather than
    silently no-op'ing on zero rows. Catches operator typos."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            "999999",
            "--stratum",
            "latin-loan",
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "no etymon" in combined.lower()


def test_cli_set_stratum_rejects_canonical_form_without_language(
    fresh_db: Path,
) -> None:
    """--canonical-form alone is ambiguous (the same form can exist
    across multiple languages — e.g. 'caer' in welsh and old-welsh).
    Require both pair components together."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--canonical-form",
            "caer",
            "--stratum",
            "latin-loan",
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "must both be passed" in combined.lower() or "canonical_form" in combined.lower()


def test_cli_set_stratum_rejects_language_without_canonical_form(
    fresh_db: Path,
) -> None:
    """Mirror of the canonical-form-without-language test — ensures
    the by_pair guard catches both directions of half-supplied pair
    args. A regression that loosened ``or`` to ``and`` in the
    by_pair predicate would silently fail one direction."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--language",
            "welsh",
            "--stratum",
            "latin-loan",
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "must both be passed" in combined.lower() or "canonical_form" in combined.lower()


def test_cli_set_stratum_rejects_id_and_pair_together(fresh_db: Path) -> None:
    """Mutually-exclusive identification: --etymon-id and
    --canonical-form/--language pick different lookup paths; passing
    both is a usage error."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("caer", "welsh")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(eid),
            "--canonical-form",
            "caer",
            "--language",
            "welsh",
            "--stratum",
            "latin-loan",
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "not both" in combined.lower() or "mutually" in combined.lower()


def test_cli_set_stratum_rejects_neither_id_nor_pair(fresh_db: Path) -> None:
    """No identification mode at all → error explaining the two
    available lookup modes."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--stratum",
            "latin-loan",
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "must specify" in combined.lower() or "etymon-id" in combined.lower()


def test_cli_set_stratum_rejects_stratum_and_clear_together(fresh_db: Path) -> None:
    """``--stratum`` writes a value and ``--clear`` writes NULL —
    mutually exclusive."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("caer", "welsh")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(eid),
            "--stratum",
            "latin-loan",
            "--clear",
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "mutually exclusive" in combined.lower()


def test_cli_set_stratum_rejects_neither_stratum_nor_clear(fresh_db: Path) -> None:
    """Must specify either --stratum VALUE or --clear (one is
    required so the command isn't a silent no-op)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("caer", "welsh")
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(eid),
        ],
    )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "must specify" in combined.lower()


def test_cli_set_stratum_warns_on_merged_cluster_loser(fresh_db: Path) -> None:
    """A hand-correction on a merged-cluster loser
    (merged_into_id IS NOT NULL) succeeds but emits a warning to
    stderr — the classifier and bundle export both filter these
    rows out, so the operator should know the correction may not
    surface as expected. Doesn't block (operator may have a legit
    reason — e.g. planning to undo the merge later)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        winner = db.upsert_etymon("caer", "welsh")
        loser = db.upsert_etymon("ceer", "welsh")  # OCR variant
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (winner, loser),
        )
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(loser),
            "--stratum",
            "latin-loan",
        ],
    )
    # Succeeds (write goes through) ...
    assert result.exit_code == 0, result.output + (result.stderr or "")
    combined = result.output + (result.stderr or "")
    # ... but warns the operator that the row is a merged loser.
    assert "warning" in combined.lower()
    assert "merged" in combined.lower()
    # And the write actually happened.
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (loser,)).fetchone()[
            "stratum"
        ]
    assert stratum == "latin-loan"


def test_cli_set_stratum_survives_classify_stratum_apply_without_force(
    fresh_db: Path,
) -> None:
    """End-to-end idempotency check: a hand-correction (set-stratum)
    survives a subsequent ``classify-stratum --apply`` (without
    --force). The default classify behavior skips rows that already
    have a non-NULL stratum, so hand-corrections aren't blown away
    by the next bulk-classification pass.

    This is the load-bearing contract for Phase 4d's value: the
    operator's manual override stays manual until they explicitly
    ask for it to be re-classified (via --clear, or by re-running
    classify-stratum --force)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("caer", "welsh")
        # Add an ancestor that classify_welsh would normally use to
        # assign 'latin-loan' — the hand-correction overrides this.
        latin_id = db.upsert_etymon("castrum", "latin")
        _add_descent(db, parent_id=latin_id, child_id=eid, edge_type="borrowing")
        db.commit()

    runner = CliRunner()
    # Step 1: hand-correct to brittonic-substrate (NOT what the
    # classifier would assign — the descent says latin-loan).
    result1 = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(eid),
            "--stratum",
            "brittonic-substrate",
        ],
    )
    assert result1.exit_code == 0, result1.output + (result1.stderr or "")

    # Step 2: re-run classify-stratum --apply (without --force).
    result2 = runner.invoke(
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
    assert result2.exit_code == 0, result2.output + (result2.stderr or "")

    # Hand-correction survives — stratum is still 'brittonic-substrate'
    # (not the 'latin-loan' the classifier would have written).
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (eid,)).fetchone()[
            "stratum"
        ]
    assert stratum == "brittonic-substrate"


def test_all_strata_includes_every_family_bucket() -> None:
    """``ALL_STRATA`` is the source-of-truth registry consumed by
    ``set-stratum`` for typo protection. Pin that every family's
    full STRATA tuple is included so a Phase 4e+ classifier addition
    that forgets to extend ALL_STRATA gets caught (currently
    ALL_STRATA is built from the four constants directly, but a
    refactor that swaps to an explicit listing would silently drop
    coverage)."""
    from wyrd.generators.kenning.lexicon.strata import ALL_STRATA

    for stratum in WELSH_STRATA + FRENCH_STRATA + OLD_ENGLISH_STRATA + OLD_NORSE_STRATA:
        assert stratum in ALL_STRATA, stratum
    # Frozen — caller cannot mutate.
    assert isinstance(ALL_STRATA, frozenset)


def test_family_for_language_classifies_every_known_language() -> None:
    """Every entry in ``LANGUAGE_TO_FAMILY`` must map to a family
    that's also a key of ``STRATA_BY_FAMILY``. Pins the cross-map
    invariant: a Phase 4e classifier addition that adds to one map
    but not the other would surface here as a KeyError."""
    from wyrd.generators.kenning.lexicon.strata import (
        LANGUAGE_TO_FAMILY,
        STRATA_BY_FAMILY,
        family_for_language,
    )

    for lang, family in LANGUAGE_TO_FAMILY.items():
        assert family_for_language(lang) == family
        assert family in STRATA_BY_FAMILY, (lang, family)


def test_family_for_language_returns_none_for_unclassified() -> None:
    """Languages without a classifier yet (real lexicon codes that
    appear in the live DB but aren't in any ``_*_SELF_LANGUAGE_TO_STRATUM``
    map) return None, signalling 'fall back to ALL_STRATA typo-check'
    in set-stratum."""
    from wyrd.generators.kenning.lexicon.strata import family_for_language

    # ofs / goh / ga are real Wiktextract codes for Old Frisian /
    # Old High German / Irish — none classified today.
    assert family_for_language("ofs") is None
    assert family_for_language("goh") is None
    assert family_for_language("ga") is None
    # Bogus values also return None.
    assert family_for_language("klingon") is None


def test_valid_strata_for_culture_known_and_unknown() -> None:
    """``valid_strata_for_culture`` returns a non-empty frozenset for
    every real culture (english / scottish / welsh, plus irish /
    breton since wyrd-de77 shipped the Celtic-family classifiers), and
    an empty frozenset only for an unknown culture (via dict.get's
    default) — which signals 'no per-culture restriction, fall back to
    the ALL_STRATA typo-check'."""
    from wyrd.generators.kenning.lexicon.strata import valid_strata_for_culture

    assert valid_strata_for_culture("english")  # non-empty
    assert valid_strata_for_culture("welsh")  # non-empty
    # wyrd-de77: irish / breton now carry Celtic-family restrictions.
    assert valid_strata_for_culture("irish")  # non-empty
    assert valid_strata_for_culture("breton")  # non-empty
    # Unknown cultures still return frozenset() — signals 'no
    # per-culture restriction'.
    assert valid_strata_for_culture("klingon") == frozenset()


def test_language_to_family_values_are_known_families() -> None:
    """Cross-map structural invariant: every value in
    ``LANGUAGE_TO_FAMILY`` is a key of ``STRATA_BY_FAMILY``. A
    Phase 4e+ classifier addition that adds a self-language entry
    pointing at an unregistered family would fail this pin (drift
    catch). Complements ``test_family_for_language_classifies_every_known_language``
    by validating the structural invariant directly rather than via
    the helper."""
    from wyrd.generators.kenning.lexicon.strata import LANGUAGE_TO_FAMILY, STRATA_BY_FAMILY

    families_used = set(LANGUAGE_TO_FAMILY.values())
    families_known = set(STRATA_BY_FAMILY.keys())
    assert families_used <= families_known, families_used - families_known


def test_cli_set_stratum_clear_skips_family_check_on_classified_language(
    fresh_db: Path,
) -> None:
    """``set-stratum --clear`` on a classified-language row (one whose
    language IS in LANGUAGE_TO_FAMILY) succeeds without hitting the
    family-aware check. _validate_set_stratum_family_match early-
    returns when stratum is None (the --clear path); pin so a
    refactor that drops the early-return doesn't accidentally
    require a family-valid stratum value to clear a row."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        # Welsh is classified — exercises the 'family is not None'
        # path without --clear-skip, the helper would hit the ALL_STRATA
        # check on stratum=None.
        eid = db.upsert_etymon("caer", "welsh")
        db.conn.execute("UPDATE etymon SET stratum = ? WHERE id = ?", ("latin-loan", eid))
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "set-stratum",
            "--db",
            str(fresh_db),
            "--etymon-id",
            str(eid),
            "--clear",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        stratum = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (eid,)).fetchone()[
            "stratum"
        ]
    assert stratum is None


# --- wyrd-b43r batch UPDATE + progress line -------------------------------


def test_cli_classify_stratum_batches_writes_above_chunk_size(fresh_db: Path) -> None:
    """wyrd-b43r CASE-batched UPDATE: at >300 rows, the write splits
    across multiple batches but produces the same result as the prior
    per-row loop. Pin by seeding 350 welsh etymons (above the
    _STRATUM_WRITE_BATCH_SIZE = 300 ceiling) and verifying every row
    gets the expected stratum.

    Catches a regression where the batch boundary off-by-one drops
    or duplicates rows."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        # Seed 350 distinct welsh etymons — none have descent edges, so
        # all should classify as native-welsh.
        ids = [db.upsert_etymon(f"caer{n:03d}", "welsh") for n in range(350)]
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
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined

    with LexiconDB(fresh_db) as db:
        rows = db.conn.execute(
            f"SELECT id, stratum FROM etymon WHERE id IN ({','.join('?' * len(ids))}) ORDER BY id",
            ids,
        ).fetchall()
    assert len(rows) == 350
    for r in rows:
        assert r["stratum"] == "native-welsh", r["id"]


def test_cli_classify_stratum_emits_progress_line(fresh_db: Path) -> None:
    """wyrd-b43r progress line per CLAUDE.md mining-progress
    convention: each batch emits ``[N/total]  written=X skipped=Y
    (R s/entry)`` to stderr so operators can extrapolate ETA on
    multi-minute classify runs.

    Pin the line shape (count brackets + s/entry rate) so a refactor
    that drops the periodic signal is caught."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        # Two welsh etymons — single-batch run, but the progress line
        # still fires after the batch.
        db.upsert_etymon("caer1", "welsh")
        db.upsert_etymon("caer2", "welsh")
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
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined
    # Two welsh rows → batch processes both in one pass.
    assert "[2/2]" in combined
    assert "written=" in combined
    assert "skipped=" in combined
    assert "s/entry" in combined


def test_build_case_update_force_off_includes_stratum_null_guard() -> None:
    """``_build_case_update`` with ``force=False`` emits the
    load-bearing ``AND stratum IS NULL`` clause that protects
    set-stratum hand-corrections (wyrd-lr4 Phase 4d idempotency
    contract). Pin the SQL shape directly so a refactor that
    accidentally drops the clause is caught at the SQL-level rather
    than only by the end-to-end test."""
    from wyrd.generators.kenning.cli import _build_case_update

    chunk = [(1, "latin-loan"), (2, "native-welsh")]
    sql, params = _build_case_update(chunk, force=False)
    assert "AND stratum IS NULL" in sql
    assert "CASE id WHEN ? THEN ? WHEN ? THEN ? END" in sql
    # Params: 2 (id, stratum) per row + 1 id per row in the IN clause.
    assert params == [1, "latin-loan", 2, "native-welsh", 1, 2]


def test_build_case_update_force_on_omits_stratum_null_guard() -> None:
    """``--force`` writes regardless of existing stratum, so the
    AND-clause is omitted. Pin its absence so a regression that
    accidentally re-adds the gate (and silently no-ops --force on
    pre-classified rows) is caught."""
    from wyrd.generators.kenning.cli import _build_case_update

    chunk = [(1, "latin-loan")]
    sql, _params = _build_case_update(chunk, force=True)
    assert "AND stratum IS NULL" not in sql
    assert "WHERE id IN (?)" in sql


def test_cli_classify_stratum_batched_force_overwrites_above_chunk_size(
    fresh_db: Path,
) -> None:
    """``--apply --force`` over >300 rows correctly overwrites every
    pre-existing stratum value. Catches a regression in the batch
    UPDATE's CASE-WHEN parameter ordering — if the (id, stratum)
    pairs drift from the WHEN-clause expectation, rows would land
    with the wrong stratum value."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        ids = [db.upsert_etymon(f"caer{n:03d}", "welsh") for n in range(350)]
        # Pre-populate with a placeholder stratum across all 350 rows.
        db.conn.execute(
            "UPDATE etymon SET stratum = 'placeholder' WHERE id IN ("
            + ",".join("?" * len(ids))
            + ")",
            ids,
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
            "welsh",
            "--apply",
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        rows = db.conn.execute(
            f"SELECT id, stratum FROM etymon WHERE id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
    for r in rows:
        # Every row was overwritten from 'placeholder' to 'native-welsh'.
        assert r["stratum"] == "native-welsh", r["id"]


def test_cli_classify_stratum_emits_progress_line_per_batch(fresh_db: Path) -> None:
    """wyrd-b43r progress line fires AFTER EACH batch, not just at
    the end. Pin via a 350-row run (2 batches at chunk_size=300):
    both ``[300/350]`` and ``[350/350]`` must appear in stderr.

    A regression that moved the progress emit out of the loop into a
    final-only echo would silently kill the ETA-extrapolation
    contract on long classify runs."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        for n in range(350):
            db.upsert_etymon(f"caer{n:03d}", "welsh")
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
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined
    # First batch boundary at 300, then final at 350.
    assert "[300/350]" in combined
    assert "[350/350]" in combined


def test_cli_classify_stratum_empty_proposals_short_circuits_no_write(
    fresh_db: Path,
) -> None:
    """When the classifier returns no proposals (no etymons in the
    target language family), the CLI short-circuits with a friendly
    'No etymons found' message and exit 0 — without issuing an
    UPDATE. Pin so a refactor that always runs the write loop (even
    on an empty dict) is caught."""
    runner = CliRunner()
    # Fresh DB has no welsh etymons → classify_welsh returns {} →
    # short-circuit branch fires.
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
    assert "no etymons found" in combined.lower()
    # And the empty etymon table stays empty.
    with LexiconDB(fresh_db) as db:
        count = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon WHERE stratum IS NOT NULL"
        ).fetchone()["n"]
    assert count == 0


def test_cli_classify_stratum_progress_rate_is_nonnegative(fresh_db: Path) -> None:
    """Progress line's ``s/entry`` rate token must be a non-negative
    float — pin so a refactor that swaps the rate calculation's
    numerator and denominator (or drops the per-row normalization)
    surfaces here. Extracts the rate via regex and asserts >= 0."""
    import re

    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        db.upsert_etymon("caer", "welsh")
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
    combined = result.output + (result.stderr or "")
    assert result.exit_code == 0, combined
    # Match the rate token: e.g. "(0.0001s/entry)".
    match = re.search(r"\((\d+\.\d+)s/entry\)", combined)
    assert match, f"progress rate token missing in stderr: {combined!r}"
    assert float(match.group(1)) >= 0


def test_build_case_update_single_row_chunk_force_off_executes_against_db(
    fresh_db: Path,
) -> None:
    """The single-row chunk shape (one ``WHEN ? THEN ?`` clause, one
    ``IN (?)``) for ``force=False`` must execute cleanly. Pin by
    running the SQL against a real seeded row — catches a syntax
    regression in the CASE-clause join when the chunk has no
    multi-clause whitespace context."""
    from wyrd.generators.kenning.cli import _build_case_update

    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        eid = db.upsert_etymon("caer", "welsh")
        db.commit()

    chunk = [(eid, "native-welsh")]
    sql, params = _build_case_update(chunk, force=False)

    with LexiconDB(fresh_db) as db:
        cur = db.conn.execute(sql, params)
        db.commit()
    assert cur.rowcount == 1
    with LexiconDB(fresh_db) as db:
        row = db.conn.execute("SELECT stratum FROM etymon WHERE id = ?", (eid,)).fetchone()
    assert row["stratum"] == "native-welsh"


def test_cli_classify_stratum_batched_apply_is_idempotent(fresh_db: Path) -> None:
    """Two back-to-back ``classify-stratum --apply`` runs over >300
    rows: the first writes everything, the second writes 0 (all rows
    already have non-NULL stratum, so the AND-clause filters them
    out). Pin idempotency at the batched-write layer.

    Catches a regression in the batch path's WHERE-clause guard that
    would re-write rows on subsequent runs."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        for n in range(350):
            db.upsert_etymon(f"caer{n:03d}", "welsh")
        db.commit()

    runner = CliRunner()
    common_args = [
        "lexicon",
        "classify-stratum",
        "--db",
        str(fresh_db),
        "--language",
        "welsh",
        "--apply",
    ]
    # First run: all 350 rows write.
    first = runner.invoke(cli_root, common_args)
    assert first.exit_code == 0, first.output + (first.stderr or "")
    first_combined = first.output + (first.stderr or "")
    assert "Wrote 350 rows" in first_combined
    assert "Skipped 0" in first_combined

    # Second run: 0 writes, all skipped (stratum already set).
    second = runner.invoke(cli_root, common_args)
    assert second.exit_code == 0, second.output + (second.stderr or "")
    second_combined = second.output + (second.stderr or "")
    assert "Wrote 0 rows" in second_combined
    assert "Skipped 350" in second_combined


# --- classify_stratum_all wrapper (L3 enrichment chain entry point) -------
#
# The CLI tests above exercise the per-language dispatcher with --apply
# bookkeeping. classify_stratum_all is the wrapper used by
# run_full_enrichment, and its dry-run gate (added in PR #199 round 2)
# needs its own coverage because the orchestrator dry-run test seeds an
# empty DB, so the gate's filtering effect is never observed there.


def test_classify_stratum_all_dry_run_excludes_already_classified(
    fresh_db: Path,
) -> None:
    """A welsh etymon whose stratum is already set must be skipped, not
    counted in by_stratum/written. Pins the dry-run gate parity with the
    apply=True branch."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        # Two welsh etymons, neither has a descent row → both classifier
        # proposals will be 'native-welsh'.
        eid_blank = db.upsert_etymon("rhydoldfoel", "welsh")
        eid_preset = db.upsert_etymon("aberystwyth", "welsh")
        # Manually pre-classify the second one. The gate should now
        # exclude it from the dry-run preview.
        db.conn.execute(
            "UPDATE etymon SET stratum = 'native-welsh' WHERE id = ?",
            (eid_preset,),
        )
        db.commit()

        result = classify_stratum_all(db, apply=False)

        # Eid_blank's stratum is still NULL (dry-run) — no actual UPDATE.
        rows = {
            row["id"]: row["stratum"]
            for row in db.conn.execute(
                "SELECT id, stratum FROM etymon WHERE id IN (?, ?)",
                (eid_blank, eid_preset),
            )
        }

    welsh = result["languages"]["welsh"]
    assert welsh["proposed"] == 2, "both etymons matched the welsh classifier"
    assert welsh["written"] == 1, "only the NULL-stratum row would be written"
    assert welsh["skipped"] == 1, "the pre-classified row is skipped"
    assert welsh["by_stratum"] == {"native-welsh": 1}, (
        "by_stratum must reflect would-be-written rows only"
    )
    # Dry-run didn't touch the DB.
    assert rows[eid_blank] is None
    assert rows[eid_preset] == "native-welsh"


def test_classify_stratum_all_apply_is_idempotent_on_rerun(
    fresh_db: Path,
) -> None:
    """apply=True is idempotent — re-running on a fully-classified DB
    writes zero new rows and marks every prior row as skipped. Pins the
    wyrd-s9z3 acceptance: idempotency gate parity between dry-run and
    apply branches."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        db.upsert_etymon("rhydoldfoel", "welsh")
        db.upsert_etymon("aberystwyth", "welsh")
        first = classify_stratum_all(db, apply=True)
        second = classify_stratum_all(db, apply=True)
    first_welsh = first["languages"]["welsh"]
    second_welsh = second["languages"]["welsh"]
    assert first_welsh["written"] == 2
    assert first_welsh["skipped"] == 0
    # Second run: every prior write trips the "stratum already set"
    # skip — no new writes.
    assert second_welsh["written"] == 0
    assert second_welsh["skipped"] == 2
    # by_stratum on the second run reflects WOULD-BE writes only; since
    # nothing was written, the dict is empty.
    assert second_welsh.get("by_stratum", {}) == {}


def test_classify_stratum_all_by_stratum_dict_keyed_by_assignment(
    fresh_db: Path,
) -> None:
    """by_stratum maps stratum-name → count-of-writes for that name.
    Distinct strata each get their own key; total across keys equals
    ``written``. Pins the renderer's iteration contract."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        # Two welsh etymons, both classify as 'native-welsh' (the
        # classifier defaults when no descent row pulls them off-path).
        db.upsert_etymon("rhydoldfoel", "welsh")
        db.upsert_etymon("aberystwyth", "welsh")
        result = classify_stratum_all(db, apply=True)
    welsh = result["languages"]["welsh"]
    assert welsh["written"] == 2
    assert welsh.get("by_stratum") == {"native-welsh": 2}
    # by_stratum sums to written.
    assert sum(welsh["by_stratum"].values()) == welsh["written"]


def test_classify_stratum_all_routes_celtic_families(fresh_db: Path) -> None:
    """wyrd-de77: the chain entry point dispatches the three Celtic-family
    classifiers (brittonic / goidelic / celtic), not just the original four.
    Guards against a regression that drops a loop entry — the enrichment
    chain is the path that re-derives stratum on every rebuild, so a missing
    entry would silently leave the Celtic admit cohort unclassified.
    Seeds one orphan per family (→ native-*) plus a proto-celtic-descended
    goidelic etymon (→ proto-celtic-substrate)."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        db.upsert_etymon("ynys", "brittonic")  # orphan → native-brittonic
        db.upsert_etymon("dun", "celtic")  # orphan → native-celtic
        db.upsert_etymon("achadh", "goidelic")  # orphan → native-goidelic
        child = db.upsert_etymon("baile", "goidelic")
        parent = db.upsert_etymon("*baljom", "proto-celtic")
        _add_descent(db, parent_id=parent, child_id=child)
        result = classify_stratum_all(db, apply=True)

    langs = result["languages"]
    assert langs["brittonic"]["by_stratum"] == {"native-brittonic": 1}
    assert langs["celtic"]["by_stratum"] == {"native-celtic": 1}
    # goidelic: one orphan default + one proto-celtic-substrate via descent.
    assert langs["goidelic"]["by_stratum"] == {
        "native-goidelic": 1,
        "proto-celtic-substrate": 1,
    }


# --- _classify_family precondition + proposal-order invariants -------------


def test_classify_family_rejects_modern_lang_in_self_language_map(fresh_db: Path) -> None:
    """Guard (comment-analyzer #757): if ``modern_lang`` is also a key in
    ``self_lang_to_stratum`` the self-language and ancestor-walk passes select
    overlapping etymons, so pass 2 would silently overwrite pass 1 and break the
    documented proposal order. The classifier rejects that misconfiguration
    loudly instead — the ``classify_<lang>`` wrappers are an explicit extension
    point, so the precondition is enforced, not merely documented."""
    with (
        LexiconDB(fresh_db) as db,
        pytest.raises(ValueError, match="must not also be a self_lang_to_stratum key"),
    ):
        _classify_family(
            db,
            modern_lang="welsh",
            strata_order=WELSH_STRATA,
            ancestor_to_stratum={},
            self_lang_to_stratum={"welsh": "native-welsh"},
            default_stratum="native-welsh",
        )


def test_classify_welsh_proposal_order_is_self_language_then_modern(fresh_db: Path) -> None:
    """The proposals dict lists all self-language (ancestor-variety) ids before
    all modern (welsh) ids — the order produced by applying the self-language
    pass then ``.update()``-ing the ancestor-walk pass. This is load-bearing:
    ``classify_stratum_all`` iterates ``proposals.items()`` in insertion order to
    build ``by_stratum``, which feeds the enrichment report (Phase 3 byte-diff).
    Ids are interleaved on insert so the order proves the two-pass structure, not
    physical row order. (pr-test-analyzer #757.)"""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        w1 = db.upsert_etymon("waaa", "welsh")  # modern variety
        mw = db.upsert_etymon("mwww", "middle-welsh")  # self-language (ancestor)
        w2 = db.upsert_etymon("wbbb", "welsh")  # modern variety
        cbp = db.upsert_etymon("cccc", "cel-bry-pro")  # self-language (ancestor)
        proposals = classify_welsh(db)
    keys = list(proposals.keys())
    # Cross-pass split: the two self-language ids precede the two modern ids
    # (the property the .update() merge guarantees), and the modern block is in
    # ascending id order.
    assert set(keys[:2]) == {cbp, mw}
    assert keys[2:] == [w1, w2]
