"""Tests for the toponym reverse-search module (wyrd-x82p Phase 1)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from wyrd.generators.kenning.toponym_reverse_search import (
    _build_form_to_toponym_lookup,
    _load_source_body,
    _normalize_for_match,
    reverse_search_source,
)

# Shared fixture DDL so both in-memory and on-disk tests use the
# SAME schema (fixture-data-reviewer PR #212 finding: an inline
# duplicate in a single test was at risk of silent drift if the
# main fixture helper changes).
_FIXTURE_DDL = """
    CREATE TABLE source (id TEXT PRIMARY KEY, title TEXT NOT NULL);
    CREATE TABLE toponym (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        modern_name TEXT NOT NULL,
        country     TEXT,
        region      TEXT
    );
    CREATE TABLE toponym_attestation (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        toponym_id  INTEGER NOT NULL,
        form        TEXT NOT NULL,
        date_year   INTEGER,
        source_doc  TEXT
    );
    CREATE UNIQUE INDEX idx_attestation_unique
        ON toponym_attestation(toponym_id, form, date_year, source_doc);
"""


def _build_fixture_db() -> sqlite3.Connection:
    """In-memory fixture DB. Schema lives in :data:`_FIXTURE_DDL` so
    on-disk tests can mirror it without copy-pasting."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_FIXTURE_DDL)
    return conn


def test_normalize_for_match_lowercases_and_strips_trailing_punct():
    assert _normalize_for_match("Birmingham") == "birmingham"
    assert _normalize_for_match("Birmingham,") == "birmingham"
    assert _normalize_for_match("Birmingham.") == "birmingham"
    assert _normalize_for_match("  Birmingham  ") == "birmingham"
    assert _normalize_for_match("CASTLE CARY") == "castle cary"


def test_normalize_for_match_preserves_internal_punctuation():
    """Multi-word toponyms with internal punctuation (apostrophes,
    hyphens) keep that punctuation — they're part of the form
    identity, not trailing noise."""
    assert _normalize_for_match("Stratford-upon-Avon") == "stratford-upon-avon"
    assert _normalize_for_match("St. Albans") == "st. albans"


def test_normalize_for_match_strips_leading_punctuation():
    """Gemini PR #212 round-2: scholar prose often quotes / brackets /
    parenthesizes a form (``"Birmingham``, ``(Birmingham``,
    ``[Birmingham``). Leading-punct strip catches these so the
    form-to-toponym join still works."""
    assert _normalize_for_match('"Birmingham"') == "birmingham"
    assert _normalize_for_match("(Birmingham)") == "birmingham"
    assert _normalize_for_match("[Birmingham]") == "birmingham"
    assert _normalize_for_match("'Stratford'") == "stratford"
    # Mixed leading + trailing.
    assert _normalize_for_match("(Birmingham,") == "birmingham"


def test_normalize_for_match_strips_unicode_quotes():
    """OCR / LLM prose quotes with the typographic Unicode variants far more than
    ASCII — curly “” ‘’, guillemets « » ‹ ›, low „. Stripping only ASCII left a
    curly-quoted form silently never matching a known toponym (same silent-miss
    the ASCII leading-strip was added to prevent)."""
    assert _normalize_for_match("“Birmingham”") == "birmingham"  # curly double
    assert _normalize_for_match("‘Stratford’") == "stratford"  # curly single
    assert _normalize_for_match("«Birmingham»") == "birmingham"  # guillemets
    assert _normalize_for_match("‹Stratford›") == "stratford"  # single guillemets
    assert _normalize_for_match("„Birmingham“") == "birmingham"  # low-open + curly
    # A closing quote AFTER trailing punctuation still fully unwraps.
    assert _normalize_for_match("“Birmingham”.") == "birmingham"
    # An interior curly apostrophe is identity, NOT stripped (boundary-only).
    assert _normalize_for_match("King’s Lynn") == "king’s lynn"


def test_normalize_for_match_every_strip_codepoint_is_covered():
    """Data-driven lock over both strip constants: every boundary char unwraps to
    the bare form, so a future edit dropping one (e.g. the low-quote ‚ that has no
    representative case above) fails here."""
    from wyrd.generators.kenning.toponym_reverse_search import (
        _MATCH_STRIP_ENDS,
        _MATCH_STRIP_TRAILING,
    )

    for ch in _MATCH_STRIP_ENDS:
        # Wrapped both ends (the both-ends strip set).
        assert _normalize_for_match(f"{ch}Birmingham{ch}") == "birmingham", (
            f"{ch!r} (U+{ord(ch):04X}) not stripped at both ends"
        )
    for ch in _MATCH_STRIP_TRAILING:
        # Trailing only.
        assert _normalize_for_match(f"Birmingham{ch}") == "birmingham", (
            f"{ch!r} (U+{ord(ch):04X}) not stripped trailing"
        )


def test_build_lookup_includes_modern_names_and_attestation_forms():
    """Lookup contains modern_name entries AND historical forms
    from existing attestations."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Birmingham')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (2, 'Stratford')")
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (2, 'Stretford', 1086, 'old_source')"
    )
    conn.commit()

    lookup = _build_form_to_toponym_lookup(conn)
    assert lookup["birmingham"] == 1
    assert lookup["stratford"] == 2
    assert lookup["stretford"] == 2  # historical form maps to same toponym


def test_build_lookup_ambiguous_form_excluded_from_lookup():
    """Gemini PR #212 round-2 (HIGH): when the same form maps to
    MULTIPLE toponym_ids (homonyms — Newton across counties; OR
    cross-source collision: 'Newton' as both a modern_name and an
    attestation form for a different toponym), the form is EXCLUDED
    from the lookup. Phase 1 pattern-based search has no context to
    disambiguate; Phase 2's LLM extraction uses surrounding text."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Newton')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (2, 'Sutton')")
    # Cross-source collision: attestation labels Sutton (id=2) as 'Newton'.
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (2, 'Newton', 1086, 'src')"
    )
    conn.commit()

    lookup = _build_form_to_toponym_lookup(conn)
    assert "newton" not in lookup, "ambiguous form must be excluded; Phase 2 LLM disambiguates"
    # Sutton is unambiguous (only its own modern_name).
    assert lookup["sutton"] == 2


def test_build_lookup_homonym_modern_names_excluded():
    """Two distinct toponym rows with the SAME modern_name (Newton
    across counties is the canonical scholar-prose homonym) are
    excluded from the lookup. Phase 1 can't tell from prose alone
    which Newton the source means; Phase 2 LLM gets the surrounding
    context."""
    conn = _build_fixture_db()
    conn.execute(
        "INSERT INTO toponym (id, modern_name, region) VALUES (1, 'Newton', 'Northumberland')"
    )
    conn.execute("INSERT INTO toponym (id, modern_name, region) VALUES (2, 'Newton', 'Yorkshire')")
    conn.commit()
    lookup = _build_form_to_toponym_lookup(conn)
    assert "newton" not in lookup, "homonym modern_names must be excluded as ambiguous"


def test_load_source_body_returns_text(tmp_path: Path):
    (tmp_path / "src.txt").write_text("body text here.")
    assert _load_source_body(tmp_path, "src") == "body text here."


def test_load_source_body_missing_returns_none(tmp_path: Path):
    assert _load_source_body(tmp_path, "nope") is None


def test_load_source_body_unicode_decode_error_returns_none(tmp_path: Path):
    (tmp_path / "bad.txt").write_bytes(b"valid prefix \xff\xfe invalid bytes")
    assert _load_source_body(tmp_path, "bad") is None


def test_load_source_body_whitespace_only_returns_none(tmp_path: Path):
    (tmp_path / "blank.txt").write_text("   \n\n   ")
    assert _load_source_body(tmp_path, "blank") is None


def test_load_source_body_strips_path_components(tmp_path: Path):
    """Path traversal must be rejected — source_id with ../ is
    stripped to the basename via Path(...).name."""
    sensitive = tmp_path / "sensitive.txt"
    sensitive.write_text("should-not-be-readable")
    inner = tmp_path / "inner"
    inner.mkdir()
    assert _load_source_body(inner, "../sensitive") is None


def test_reverse_search_source_matches_modern_name(tmp_path: Path):
    """A scholar source body mentioning a known modern_name with a
    nearby year produces an attestation insert."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('mawer', 'Mawer 1920')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Cestretone')")
    conn.commit()
    (tmp_path / "mawer.txt").write_text(
        "Some scholarly prose. Cestretone in 1210 was recorded. More text."
    )

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "mawer", tmp_path, lookup, apply=True)

    assert report.pairs_extracted == 1
    assert report.matched == 1
    assert report.unmatched == 0
    assert report.inserted == 1
    row = conn.execute(
        "SELECT toponym_id, form, date_year, source_doc FROM toponym_attestation"
    ).fetchone()
    assert (row["toponym_id"], row["form"], row["date_year"], row["source_doc"]) == (
        1,
        "Cestretone",
        1210,
        "mawer",
    )


def test_reverse_search_source_dry_run_predicts_apply_accurately(tmp_path: Path):
    """Without --apply, the DB isn't touched but the report's
    inserted/already_present counts ACCURATELY predict what
    apply=True would do (preflight uniqueness check via SELECT).
    silent-failure-hunter PR #212 finding: dry-run with always-zero
    already_present masks the all-dupes case from operators."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('mawer', 'Mawer')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Birmingham')")
    conn.commit()
    (tmp_path / "mawer.txt").write_text("As found in Birmingham, 1086, the name is...")

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "mawer", tmp_path, lookup, apply=False)

    assert report.matched == 1
    # Predicted: this row would be a new insert (not in DB yet).
    assert report.inserted == 1
    assert report.already_present == 0
    # But the DB is still unchanged — dry-run is read-only.
    nrows = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert nrows == 0


def test_reverse_search_source_dry_run_predicts_all_dupes(tmp_path: Path):
    """Dry-run on a source whose matched rows ALREADY exist in the DB
    correctly shows inserted=0, already_present=N — so operators can
    tell 'all rows are already there, applying is a no-op' before
    spending --apply time. Pin for the round-1 silent-failure-hunter
    'dry-run dup=0 always' finding."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('mawer', 'Mawer')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Birmingham')")
    # Pre-populate the attestation row that the source body would produce.
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (1, 'Birmingham', 1086, 'mawer')"
    )
    conn.commit()
    (tmp_path / "mawer.txt").write_text("As found in Birmingham, 1086, the name is...")

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "mawer", tmp_path, lookup, apply=False)

    assert report.matched == 1
    # Predicted: all dupes; applying would be a no-op.
    assert report.inserted == 0
    assert report.already_present == 1


def test_reverse_search_source_idempotent_on_rerun(tmp_path: Path):
    """Re-running --apply against the same source body is a no-op:
    the UNIQUE index drops duplicates, counted as already_present."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Cestretone')")
    conn.commit()
    (tmp_path / "src.txt").write_text("Cestretone in 1210.")

    lookup = _build_form_to_toponym_lookup(conn)
    r1 = reverse_search_source(conn, "src", tmp_path, lookup, apply=True)
    assert r1.inserted == 1
    assert r1.already_present == 0

    r2 = reverse_search_source(conn, "src", tmp_path, lookup, apply=True)
    assert r2.inserted == 0
    assert r2.already_present == 1

    nrows = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert nrows == 1, "re-run must not duplicate the attestation"


def test_reverse_search_source_unmatched_form_skipped(tmp_path: Path):
    """A (form, year) pair whose form doesn't map to any known
    toponym is counted as unmatched but doesn't fail or insert."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    # No toponyms — every form is unmatched.
    conn.commit()
    (tmp_path / "src.txt").write_text(
        "Birmingham, 1086 and also Cestretone in 1210 and Norwich, 1255."
    )

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "src", tmp_path, lookup, apply=True)
    assert report.pairs_extracted == 3
    assert report.matched == 0
    assert report.unmatched == 3
    assert report.inserted == 0
    assert len(report.unmatched_samples) == 3


def test_reverse_search_source_normalized_lookup_join(tmp_path: Path):
    """The form→toponym join is case-insensitive via
    _normalize_for_match. The mine-attestations regex requires the
    form to start with a capital, so the only case-folding that's
    exercised here is the trailing-comma trim — but pin both
    explicitly so a future regex change doesn't silently break
    the normalization invariant. Renamed from
    test_reverse_search_source_case_insensitive_match per
    pr-test-analyzer + fixture-data-reviewer PR #212 findings
    (the original name overstated the test's reach)."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    # Modern name stored title-case; source mentions exact title-case
    # form with trailing comma.
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Birmingham')")
    conn.commit()
    (tmp_path / "src.txt").write_text("Birmingham, 1086 — important place.")

    lookup = _build_form_to_toponym_lookup(conn)
    # Sanity: lookup key is the lowercased form.
    assert "birmingham" in lookup
    report = reverse_search_source(conn, "src", tmp_path, lookup, apply=True)
    assert report.matched == 1
    assert report.inserted == 1


def test_reverse_search_source_matches_via_existing_historical_form(tmp_path: Path):
    """The lookup admits BOTH ``toponym.modern_name`` AND existing
    ``toponym_attestation.form``. A scholar source body that mentions
    a historical form (Cestretone) with a year should map to the
    correct toponym (Chester), even though Chester's modern_name
    doesn't appear anywhere in the source body — pr-test-analyzer
    PR #212 finding."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (5, 'Chester')")
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (5, 'Cestretone', 1086, 'prior_source')"
    )
    conn.commit()
    (tmp_path / "src.txt").write_text("And Cestretone, 1210 — see also Domesday entry.")

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "src", tmp_path, lookup, apply=True)
    assert report.matched == 1
    row = conn.execute(
        "SELECT toponym_id, form, date_year FROM toponym_attestation WHERE source_doc = 'src'"
    ).fetchone()
    assert (row["toponym_id"], row["form"], row["date_year"]) == (5, "Cestretone", 1210)


def test_reverse_search_source_mixed_matched_and_unmatched(tmp_path: Path):
    """A single source body produces both matched and unmatched
    pairs in one call — verify per-pair branching works correctly
    (test-coverage-reviewer PR #212 finding)."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Birmingham')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (2, 'Coventry')")
    conn.commit()
    (tmp_path / "src.txt").write_text(
        "Birmingham, 1086 is known. So is Coventry, 1100. "
        "But Glasshamburton, 1150 is fictional and Snorklewick, 1086 is too."
    )

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "src", tmp_path, lookup, apply=True)
    assert report.pairs_extracted == 4
    assert report.matched == 2  # Birmingham, Coventry
    assert report.unmatched == 2  # Glasshamburton, Snorklewick
    assert report.inserted == 2
    assert len(report.unmatched_samples) == 2


def test_reverse_search_source_unmatched_samples_capped(tmp_path: Path):
    """When unmatched count exceeds _UNMATCHED_SAMPLE_LIMIT (10),
    the samples list is capped but the counter is accurate
    (test-coverage-reviewer PR #212 finding).

    Form-pattern requires letters only (no digits) so the fake
    place names use letter-suffix differentiators."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    conn.commit()
    suffixes = ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj", "kk", "ll"]
    pairs = " ".join(f"Foozz{s}berg, {1000 + i * 10} is a place." for i, s in enumerate(suffixes))
    (tmp_path / "src.txt").write_text(pairs)

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "src", tmp_path, lookup, apply=False)
    assert report.unmatched == 12
    assert len(report.unmatched_samples) == 10  # capped


def test_build_lookup_attestation_homonym_excluded():
    """Two toponyms each claiming the same attestation form is the
    cross-source-homonym shape — same form across two distinct
    toponym ids. Gemini PR #212 round-2 fix: excluded from lookup."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Older')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (2, 'Newer')")
    # Both toponyms claim 'Grenewic' as an attestation form.
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (2, 'Grenewic', 900, 'src')"
    )
    conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year, source_doc) "
        "VALUES (1, 'Grenewic', 1086, 'src')"
    )
    conn.commit()
    lookup = _build_form_to_toponym_lookup(conn)
    assert "grenewic" not in lookup, (
        "ambiguous attestation-form must be excluded; both 'Older' and 'Newer' claim it"
    )


def test_build_lookup_empty_db():
    """Empty DB → empty lookup (test-coverage-reviewer PR #212)."""
    conn = _build_fixture_db()
    assert _build_form_to_toponym_lookup(conn) == {}


def test_normalize_for_match_empty_string():
    """Empty input is the falsy-key gate's invariant (test-coverage-reviewer)."""
    assert _normalize_for_match("") == ""
    assert _normalize_for_match("   ") == ""
    assert _normalize_for_match(",.;:") == ""


def test_reverse_search_source_body_with_no_attestation_patterns(tmp_path: Path):
    """A non-empty source body that contains zero (form, year)
    patterns returns pairs_extracted=0 — exercises the CLI's
    early-continue path (test-coverage-reviewer PR #212)."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Birmingham')")
    conn.commit()
    (tmp_path / "narrative.txt").write_text(
        "This is a narrative passage with no place name attestation patterns. "
        "Just words and clauses without the FORM-followed-by-year shape."
    )
    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "narrative", tmp_path, lookup, apply=True)
    assert report.pairs_extracted == 0
    assert report.matched == 0
    assert report.unmatched == 0


def test_reverse_search_source_missing_body_returns_empty_report(tmp_path: Path):
    """A source without a committed .txt body returns an empty
    report (operator can have JSONLs for sources whose .txt isn't
    bundled)."""
    conn = _build_fixture_db()
    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "missing", tmp_path, lookup, apply=True)
    assert report.pairs_extracted == 0
    assert report.matched == 0


def test_reverse_search_source_does_not_commit_internally(tmp_path: Path):
    """code-reviewer PR #212 finding: reverse_search_source must
    NOT call conn.commit() internally — caller owns transaction
    granularity. Verify the INSERT is uncommitted by reading via
    a fresh second connection: the new attestation should NOT be
    visible until caller calls commit() explicitly."""
    db_path = tmp_path / "lex.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Reuse the fixture schema by attaching the in-memory copy's
    # DDL — keeps schema drift from the helper in check.
    conn.executescript(_FIXTURE_DDL)
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    conn.execute("INSERT INTO toponym (modern_name) VALUES ('Cestretone')")
    conn.commit()
    (tmp_path / "src.txt").write_text("Cestretone in 1210.")
    lookup = _build_form_to_toponym_lookup(conn)
    reverse_search_source(conn, "src", tmp_path, lookup, apply=True)

    # Fresh connection: only committed data visible. The function
    # didn't commit, so 0 rows visible from a parallel session.
    fresh = sqlite3.connect(str(db_path))
    nrows_pre_commit = fresh.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    fresh.close()
    assert nrows_pre_commit == 0, (
        "reverse_search_source must NOT commit internally; caller owns transaction control"
    )

    # Now have the caller commit and verify the row IS visible.
    conn.commit()
    fresh = sqlite3.connect(str(db_path))
    nrows_post_commit = fresh.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    fresh.close()
    assert nrows_post_commit == 1


# ---------------------------------------------------------------------------
# CLI tests (per multiple agent PR #212 findings: zero CLI coverage)
# ---------------------------------------------------------------------------


def _build_on_disk_db(tmp_path: Path) -> Path:
    """Build a fresh on-disk DB using the shared schema. Used by
    CLI tests that need to hit the CLI as a black box."""
    db_path = tmp_path / "lex.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_FIXTURE_DDL)
    conn.commit()
    conn.close()
    return db_path


def test_cli_reverse_search_toponyms_validates_unknown_source(tmp_path: Path):
    """A misspelled --source raises ClickException with the unknown
    list (silent-failure-hunter PR #212 pattern; same as wyrd-1hpc
    CLI). Pin the user-facing error contract."""
    from click.testing import CliRunner

    db_path = _build_on_disk_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO source (id, title) VALUES ('legit', 'Legit')")
    conn.commit()
    conn.close()

    from wyrd.generators.kenning.cli import cli as cli_root

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "reverse-search-toponyms",
            "--db",
            str(db_path),
            "--sources-dir",
            str(tmp_path),
            "--source",
            "legit",
            "--source",
            "doesnotexist",
        ],
    )
    assert result.exit_code != 0
    assert "unknown source_id" in result.output
    assert "doesnotexist" in result.output


def test_cli_reverse_search_toponyms_dry_run_emits_total(tmp_path: Path):
    """Smoke test the dry-run end-to-end: TOTAL line shows '(dry-run)'
    marker, predicted inserts visible. Explicit --source so we
    don't need etymon_citation in the fixture schema."""
    from click.testing import CliRunner

    db_path = _build_on_disk_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Birmingham')")
    conn.commit()
    conn.close()
    (tmp_path / "src.txt").write_text("Birmingham, 1086 was recorded.")

    from wyrd.generators.kenning.cli import cli as cli_root

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "reverse-search-toponyms",
            "--db",
            str(db_path),
            "--sources-dir",
            str(tmp_path),
            "--source",
            "src",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "(dry-run)" in result.stdout
    assert "matched=    1" in result.output


def test_cli_reverse_search_toponyms_apply_round_trips(tmp_path: Path):
    """--apply round-trip: the row appears in the DB after the CLI
    exits (verifies the CLI commits — function no longer does)."""
    from click.testing import CliRunner

    db_path = _build_on_disk_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    conn.execute("INSERT INTO toponym (modern_name) VALUES ('Birmingham')")
    conn.commit()
    conn.close()
    (tmp_path / "src.txt").write_text("Birmingham, 1086 was recorded.")

    from wyrd.generators.kenning.cli import cli as cli_root

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "reverse-search-toponyms",
            "--db",
            str(db_path),
            "--sources-dir",
            str(tmp_path),
            "--source",
            "src",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "(APPLIED)" in result.stdout

    fresh = sqlite3.connect(str(db_path))
    nrows = fresh.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    fresh.close()
    assert nrows == 1, "CLI must commit after walking all sources"


def test_cli_reverse_search_toponyms_verbose_shows_all_samples(tmp_path: Path):
    """--verbose surfaces the unmatched samples per source — the
    module collects up to 10, and the CLI displays ALL of them (not
    a smaller [:5] slice — fixed per code-reviewer + comment-analyzer
    PR #212)."""
    from click.testing import CliRunner

    db_path = _build_on_disk_db(tmp_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO source (id, title) VALUES ('src', 'Src')")
    conn.commit()
    conn.close()
    # 8 unmatched pairs — below the 10-cap so we expect all 8 in
    # output. FORM regex requires letters only.
    suffixes = ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh"]
    body = " ".join(f"Bozzz{s}berg, {1000 + i * 10} hi." for i, s in enumerate(suffixes))
    (tmp_path / "src.txt").write_text(body)

    from wyrd.generators.kenning.cli import cli as cli_root

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "reverse-search-toponyms",
            "--db",
            str(db_path),
            "--sources-dir",
            str(tmp_path),
            "--source",
            "src",
            "--verbose",
        ],
    )
    assert result.exit_code == 0, result.output
    # All 8 unmatched should appear (CLI no longer slices to 5).
    for s in suffixes:
        assert f"Bozzz{s}berg" in result.output


# ---------------------------------------------------------------------
# wyrd-9ekl: body-text source-chain guard wiring
# ---------------------------------------------------------------------
#
# The extractor-level tests in test_kenning_lexicon.py pin
# _extract_attestation_pairs(context_is_body=True) in isolation.
# These pin the wiring through reverse_search_source — a regression
# that dropped the context_is_body kwarg (or stopped reading the
# telemetry into ReverseSearchReport) would still pass those isolated
# tests but trip these.


def test_reverse_search_source_recovers_year_form_year_chain(tmp_path: Path):
    """Body text with the ``<year> FORM, <year>`` shape — which the
    default extractor guard suppresses for short notes strings — must
    yield the inner attestation when called via reverse_search_source.
    Pins that reverse_search_source passes context_is_body=True. Without
    that, the chain pattern silently drops the row and matched stays 0."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('mawer', 'Mawer')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Suttone')")
    conn.commit()
    # The ``1086 Suttone, 1210`` shape is the canonical body-text chain
    # the default guard over-suppresses. With context_is_body=True the
    # year-anchored extractor admits Suttone(1210) as a legitimate
    # attestation.
    (tmp_path / "mawer.txt").write_text(
        "Earlier attestations of the place include 1086 Suttone, 1210 as recorded."
    )

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "mawer", tmp_path, lookup, apply=False)

    assert report.matched == 1, (
        "Suttone(1210) should survive body-mode extraction; matched=0 means "
        "the source-chain guard is still suppressing the chain attestation"
    )


def test_reverse_search_source_reports_pattern_match_count(tmp_path: Path):
    """ReverseSearchReport.suppressed_by_source_chain is populated from
    the extractor's telemetry dict and reflects pattern-match
    prevalence — bumps whenever the `<year> FORM, <year>` shape occurs
    even in body mode where the guard admits the row (per Gemini
    round-2 decoupling). Pins the wiring + the new semantics."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('mawer', 'Mawer')")
    conn.execute("INSERT INTO toponym (id, modern_name) VALUES (1, 'Suttone')")
    conn.commit()
    (tmp_path / "mawer.txt").write_text("1086 Suttone, 1210 reference.")

    lookup = _build_form_to_toponym_lookup(conn)
    report = reverse_search_source(conn, "mawer", tmp_path, lookup, apply=False)

    # The field must exist on the dataclass (regression catcher).
    assert hasattr(report, "suppressed_by_source_chain")
    # Pattern matched once → counter bumped (even though body mode
    # admitted the row, the prevalence is still surfaced).
    assert report.suppressed_by_source_chain == 1
