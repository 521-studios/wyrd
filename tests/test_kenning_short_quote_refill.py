"""Tests for the truncated-short_quote refill module (wyrd-1hpc)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from wyrd.generators.kenning.short_quote_refill import (
    _normalize_whitespace,
    refill_short_quote,
    refill_source,
)


def test_normalize_whitespace_collapses_runs():
    assert _normalize_whitespace("a  b\n\nc\t\td") == "a b c d"


def test_normalize_whitespace_empty():
    assert _normalize_whitespace("") == ""
    assert _normalize_whitespace("   ") == ""


def test_refill_short_quote_post_pipe_anchor_located():
    """Bannister-style 'commentary | source-context' shape — the
    post-pipe segment is the LLM's source quote, which got cut off.
    Refill locates the tail in source body and extends forward,
    trimming to the last terminal char so the new quote ends cleanly."""
    quote = "commentary | source-context says Ab"
    source = (
        "The book opens. source-context says Abber-place (Mawer) 1268 Ipm Akum. Then more prose."
    )
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=80)
    assert new is not None
    assert recovered > 0
    # Original prefix preserved; recovered tail appended
    assert new.startswith("commentary | source-context says Ab")
    assert "Akum" in new
    assert new.rstrip()[-1] in ".!?\"')]}"


def test_refill_short_quote_no_pipe_uses_whole_quote_as_anchor():
    """When there's no pipe, the entire quote is the anchor (so the
    full quote text must appear in the source body for refill to fire).
    Source must contain a terminal character in the recovered window
    so the new short_quote ends cleanly (idempotency)."""
    quote = "Plain truncated text about Bedfordshire 1086 Ak"
    source = "Plain truncated text about Bedfordshire 1086 Akum is recorded as the entry."
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=80)
    assert new is not None
    assert "Akum" in new
    assert recovered > 0
    # Must end at a terminal char so re-audit doesn't re-flag.
    assert new.rstrip()[-1] in ".!?\"')]}"


def test_refill_short_quote_falls_back_to_shorter_anchor():
    """80-char anchor fails but the shorter 40-char fallback matches
    (LLM lightly paraphrased the early part of the tail). The tail
    must be >80 chars so the long anchor is distinct from the short
    fallback.

    Tail length: 95 chars (50 paraphrased prefix + 45 matched suffix).
    Long anchor (-80) spans both regions and won't match in source;
    short anchor (-40) hits the verbatim suffix."""
    quote = (
        "commentary | "
        "WAS-PARAPHRASED-BY-LLM-SO-WONT-MATCH-IN-SOURCE-XX "  # 50 chars
        "matched suffix that exists in source verbatim"  # 45 chars
    )
    source = "irrelevant prose matched suffix that exists in source verbatim then more content."
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=40)
    assert new is not None
    assert "more content" in new


def test_refill_short_quote_returns_none_when_hallucinated():
    """The LLM made up commentary that doesn't exist in the source —
    refill correctly returns None rather than guessing."""
    quote = "commentary | the LLM hallucinated some prose that isnt in source"
    source = "completely unrelated source text mentions different things entirely"
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=200)
    assert new is None
    assert recovered == 0


def test_refill_short_quote_returns_none_on_empty_tail():
    quote = "commentary |   "  # post-pipe is empty
    source = "anything"
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=200)
    assert new is None
    assert recovered == 0


def test_refill_short_quote_returns_none_when_forward_is_blank():
    """Anchor matches at the END of the source body — no forward
    context to recover. Return None rather than appending whitespace
    or empty content.

    Anchor must be ≥ _MIN_ANCHOR_LEN (20 chars) — otherwise the test
    short-circuits at the length guard and never exercises the
    actual blank-forward branch (round-5 fix)."""
    quote = "commentary | this twenty-plus char anchor exactly at the end"
    # Anchor literally at the end of source — forward window is empty.
    source = "complete prose then this twenty-plus char anchor exactly at the end"
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=200)
    assert new is None  # nothing to recover
    assert recovered == 0


def _build_fixture_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE source (
            id TEXT PRIMARY KEY,
            author TEXT, title TEXT NOT NULL, year INTEGER,
            region TEXT, language_focus TEXT, notes TEXT
        );
        CREATE TABLE etymon (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL,
            language TEXT NOT NULL,
            UNIQUE(canonical_form, language)
        );
        CREATE TABLE etymon_citation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            etymon_id INTEGER NOT NULL,
            source_id TEXT NOT NULL,
            page TEXT,
            short_quote TEXT,
            context_snippet TEXT
        );
        """
    )
    return conn


# Truncated short_quote that should refill. Pattern: LLM commentary +
# pipe + bannister-style source tail cut off mid-place-name. Must be
# >MIN_TRUNCATION_LENGTH (200 chars) for looks_truncated to fire.
_TRUNCATED = (
    "Mawer discusses the OE element 'acum' as the place name suffix component "
    "in northern English toponyms, with parallel forms appearing in Yorkshire "
    "and Northumberland gazetteers dating to the medieval period. | Acomb "
    "(Bywell St Peter) 1268"
)


def test_refill_source_dry_run_doesnt_write(tmp_path: Path):
    """Without --apply, refill_source returns counts but doesn't
    touch the DB."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('book', 'Book')")
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        ("acum", "old-english"),
    )
    eid = cur.lastrowid
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
        (eid, "book", _TRUNCATED),
    )
    conn.commit()
    (tmp_path / "book.txt").write_text(
        "history. Acomb (Bywell St Peter) 1268 Ipm Akum. More prose follows here."
    )

    report = refill_source(conn, "book", tmp_path, apply=False)
    assert report.total_truncated == 1
    assert report.refilled == 1
    # DB unchanged
    row = conn.execute("SELECT short_quote FROM etymon_citation").fetchone()
    assert row["short_quote"] == _TRUNCATED


def test_refill_source_apply_writes_extended_short_quote(tmp_path: Path):
    """With --apply, the SQL UPDATE rewrites the short_quote to
    include the recovered forward context."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('book', 'Book')")
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        ("acum", "old-english"),
    )
    eid = cur.lastrowid
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
        (eid, "book", _TRUNCATED),
    )
    conn.commit()
    (tmp_path / "book.txt").write_text(
        "history. Acomb (Bywell St Peter) 1268 Ipm Akum. More prose follows here."
    )

    report = refill_source(conn, "book", tmp_path, apply=True)
    assert report.refilled == 1
    row = conn.execute("SELECT short_quote FROM etymon_citation").fetchone()
    assert "Akum" in row["short_quote"]
    assert row["short_quote"] != _TRUNCATED  # actually changed


def test_refill_source_hallucinated_not_written(tmp_path: Path):
    """An LLM-hallucinated short_quote whose tail isn't in the source
    body is counted as 'hallucinated', NOT written, and stays as-is."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('book', 'Book')")
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        ("acum", "old-english"),
    )
    eid = cur.lastrowid
    # Tail is "Acomb (Bywell St Peter) 1268" but the source doesn't have it
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
        (eid, "book", _TRUNCATED),
    )
    conn.commit()
    (tmp_path / "book.txt").write_text("completely unrelated prose about other things")

    report = refill_source(conn, "book", tmp_path, apply=True)
    assert report.total_truncated == 1
    assert report.refilled == 0
    assert report.hallucinated == 1
    # DB unchanged
    row = conn.execute("SELECT short_quote FROM etymon_citation").fetchone()
    assert row["short_quote"] == _TRUNCATED


def test_refill_source_skips_non_truncated(tmp_path: Path):
    """Citations whose short_quote isn't flagged by looks_truncated
    skip without inspection. Re-runs are safe (already-refilled rows
    typically end in terminal punctuation and skip)."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('book', 'Book')")
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        ("a", "old-english"),
    )
    eid = cur.lastrowid
    # Short + terminal-punctuation → not flagged
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
        (eid, "book", "A clean short quote ending with a period."),
    )
    conn.commit()
    (tmp_path / "book.txt").write_text("anything")
    report = refill_source(conn, "book", tmp_path, apply=True)
    assert report.total_truncated == 0
    assert report.refilled == 0


def test_refill_source_missing_source_txt_returns_empty(tmp_path: Path):
    """When sources/<source_id>.txt doesn't exist AND the source has
    no citations either, refill_source returns an empty report
    rather than crashing."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('missing', 'M')")
    conn.commit()
    report = refill_source(conn, "missing", tmp_path, apply=True)
    assert report.total_truncated == 0
    assert report.refilled == 0


def test_refill_source_missing_source_txt_still_counts_truncated(tmp_path: Path):
    """Gemini PR #211 round-2: when the .txt is missing but the source
    has truncated citations in the DB, refill_source still reports
    total_truncated so the operator sees the gap. Without this, a
    missing source body would mask the existence of truncations.

    refilled=0 and hallucinated=0 (can't classify without the body
    to search against)."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('book', 'Book')")
    eid = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('acum', 'old-english')"
    ).lastrowid
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
        (eid, "book", _TRUNCATED),
    )
    conn.commit()
    # tmp_path has no book.txt — source body missing
    report = refill_source(conn, "book", tmp_path, apply=True)
    assert report.total_truncated == 1, "missing source body must not hide truncation count"
    assert report.refilled == 0
    assert report.hallucinated == 0
    # DB unchanged.
    row = conn.execute("SELECT short_quote FROM etymon_citation").fetchone()
    assert row["short_quote"] == _TRUNCATED


# ---------------------------------------------------------------------------
# Sentence-boundary trimming (idempotency)
# ---------------------------------------------------------------------------


def test_refill_short_quote_trims_to_terminal_char():
    """The recovered forward window is trimmed back to the last
    terminal character (.!?'")]}) so subsequent audit runs don't
    re-flag the refilled row as truncated."""
    # Anchor must be ≥ _MIN_ANCHOR_LEN (20 chars) — use a long-enough tail.
    quote = "commentary | the early scholar citation about Ab"
    source = (
        "irrelevant. the early scholar citation about Abber 1086 Akum. "
        "Then much more prose without terminal punct"
    )
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=80)
    assert new is not None
    # Must end at the terminal char (".") found within the window.
    assert new.rstrip()[-1] in ".!?\"')]}"
    # Specifically: the trim should land at "Akum." not at the end
    # of the window mid-word in "without terminal punct".
    assert new.endswith("Akum.")


def test_refill_short_quote_returns_none_when_no_terminal_in_window():
    """If the forward window has no terminal character, the chunk is
    too fragmentary to safely extend the citation. Return None
    rather than emit a partial refill that would itself look
    truncated on the next audit run.

    Anchor must be ≥ _MIN_ANCHOR_LEN (20 chars) — otherwise the test
    short-circuits at the length guard and never exercises the
    no-terminal-char branch (round-5 fix)."""
    # 20-char anchor that appears uniquely in source.
    quote = "commentary | this is a long enough anchor here"
    # Source has the anchor exactly once, followed by 80+ chars with
    # no terminal char at all (just words and a comma).
    source = (
        "irrelevant this is a long enough anchor here then a long "
        "stream of words without any terminal punctuation at all not even one"
    )
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=80)
    assert new is None
    assert recovered == 0


def test_refill_source_idempotent_on_rerun(tmp_path: Path):
    """A second --apply run on an already-refilled DB is a no-op.
    The first run's output ends at a terminal character per
    _trim_to_terminal_char, so looks_truncated returns False for it
    and the row is skipped before the anchor search.

    Pin from pr-test-analyzer PR #211 finding: the original PR
    docstring claimed idempotency but the implementation didn't
    deliver it (no terminal-char trim) and a re-run would
    double-extend."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('book', 'Book')")
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        ("acum", "old-english"),
    )
    eid = cur.lastrowid
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
        (eid, "book", _TRUNCATED),
    )
    conn.commit()
    (tmp_path / "book.txt").write_text(
        "history. Acomb (Bywell St Peter) 1268 Ipm Akum. More prose follows here."
    )

    r1 = refill_source(conn, "book", tmp_path, apply=True)
    assert r1.refilled == 1
    after_first = conn.execute("SELECT short_quote FROM etymon_citation").fetchone()[0]

    # Second run: looks_truncated should return False on the
    # already-refilled row, so refill_source skips it entirely.
    r2 = refill_source(conn, "book", tmp_path, apply=True)
    assert r2.total_truncated == 0, (
        "re-run flagged an already-refilled row — _trim_to_terminal_char "
        "didn't actually leave a terminal char, OR looks_truncated's "
        "heuristic re-flagged the extended row"
    )
    assert r2.refilled == 0

    after_second = conn.execute("SELECT short_quote FROM etymon_citation").fetchone()[0]
    assert after_first == after_second, "re-run double-extended an already-refilled row"


# ---------------------------------------------------------------------------
# Multi-row aggregate counters
# ---------------------------------------------------------------------------


def test_refill_source_aggregates_counts_across_multiple_citations(tmp_path: Path):
    """Multiple citations in one source — refill_source's counters
    must accurately accumulate. pr-test-analyzer PR #211 finding:
    every prior test had exactly 1 citation row, so a bug in the
    accumulator loop would slip through."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('book', 'B')")
    eid = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('acum', 'old-english')"
    ).lastrowid
    # 3 truncated rows: 2 with locatable anchors (will refill), 1 with
    # a hallucinated tail (won't locate).
    locatable_a = _TRUNCATED  # tail "Acomb (Bywell St Peter) 1268"
    locatable_b = _TRUNCATED.replace("(Bywell St Peter)", "(St John Lee)")  # different tail
    hallucinated = _TRUNCATED.replace("Acomb (Bywell St Peter) 1268", "Wholly-Invented-By-LLM 1900")
    for sq in (locatable_a, locatable_b, hallucinated):
        conn.execute(
            "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
            (eid, "book", sq),
        )
    conn.commit()
    (tmp_path / "book.txt").write_text(
        "history. Acomb (Bywell St Peter) 1268 Ipm Akum. "
        "More prose. Acomb (St John Lee) 1268 N.vi.119 Akum. "
        "Then more text."
    )

    report = refill_source(conn, "book", tmp_path, apply=True, sample_limit=5)
    assert report.total_truncated == 3
    assert report.refilled == 2
    assert report.hallucinated == 1
    # Samples should include all 3 (sample_limit=5 > 3 total).
    statuses = sorted(s.status for s in report.samples)
    assert statuses == ["hallucinated", "refilled", "refilled"]


def test_refill_source_samples_capped_at_limit(tmp_path: Path):
    """report.samples is capped at sample_limit (default 3) regardless
    of how many citations exist. Counts above the cap stay accurate."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('book', 'B')")
    eid = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('acum', 'old-english')"
    ).lastrowid
    # 5 hallucinated rows (no locatable tail) — exceeds sample_limit=2.
    hallucinated = _TRUNCATED.replace("Acomb (Bywell St Peter) 1268", "Wholly-Invented-By-LLM 1900")
    for _ in range(5):
        conn.execute(
            "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
            (eid, "book", hallucinated),
        )
    conn.commit()
    (tmp_path / "book.txt").write_text("unrelated source text.")

    report = refill_source(conn, "book", tmp_path, apply=True, sample_limit=2)
    assert report.total_truncated == 5
    assert report.hallucinated == 5
    # Counter accurate, samples capped at 2.
    assert len(report.samples) == 2


# ---------------------------------------------------------------------------
# CLI surface (CliRunner)
# ---------------------------------------------------------------------------


def test_cli_refill_short_quotes_validates_unknown_source(tmp_path: Path):
    """A misspelled --source raises a ClickException rather than
    silently producing zero output. silent-failure-hunter PR #211
    finding."""
    from click.testing import CliRunner

    # Build a tiny DB the CLI can hit.
    db_path = tmp_path / "lex.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE source (
            id TEXT PRIMARY KEY, author TEXT, title TEXT NOT NULL,
            year INTEGER, region TEXT, language_focus TEXT, notes TEXT
        );
        CREATE TABLE etymon (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL, language TEXT NOT NULL,
            UNIQUE(canonical_form, language)
        );
        CREATE TABLE etymon_citation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            etymon_id INTEGER NOT NULL, source_id TEXT NOT NULL,
            page TEXT, short_quote TEXT, context_snippet TEXT
        );
        INSERT INTO source (id, title) VALUES ('mawer_1920', 'Mawer 1920');
        """
    )
    conn.commit()
    conn.close()
    (tmp_path / "mawer_1920.txt").write_text("anything")

    from wyrd.generators.kenning.cli import cli as cli_root

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "refill-short-quotes",
            "--db",
            str(db_path),
            "--sources-dir",
            str(tmp_path),
            "--source",
            "mawer_1920",
            "--source",
            "doesnotexist",
        ],
    )
    assert result.exit_code != 0
    assert "unknown source_id" in result.output
    assert "doesnotexist" in result.output


def test_cli_refill_short_quotes_dry_run_emits_total_line(tmp_path: Path):
    """Smoke test the dry-run path end-to-end: summary line shows
    (dry-run) marker and TOTAL accounting."""
    from click.testing import CliRunner

    db_path = tmp_path / "lex.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY, author TEXT, title TEXT NOT NULL,
            year INTEGER, region TEXT, language_focus TEXT, notes TEXT);
        CREATE TABLE etymon (id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL, language TEXT NOT NULL,
            UNIQUE(canonical_form, language));
        CREATE TABLE etymon_citation (id INTEGER PRIMARY KEY AUTOINCREMENT,
            etymon_id INTEGER NOT NULL, source_id TEXT NOT NULL, page TEXT,
            short_quote TEXT, context_snippet TEXT);
        INSERT INTO source (id, title) VALUES ('book', 'Book');
        INSERT INTO etymon (canonical_form, language) VALUES ('acum', 'old-english');
        """
    )
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (1, 'book', ?)",
        (_TRUNCATED,),
    )
    conn.commit()
    conn.close()
    (tmp_path / "book.txt").write_text(
        "history. Acomb (Bywell St Peter) 1268 Ipm Akum. More prose."
    )

    from wyrd.generators.kenning.cli import cli as cli_root

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "refill-short-quotes",
            "--db",
            str(db_path),
            "--sources-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # Report goes to stdout (Gemini PR #211 round-4 finding).
    assert "(dry-run)" in result.stdout
    assert "truncated=   1" in result.stdout
    assert "refilled=   1" in result.stdout


def test_cli_refill_short_quotes_warns_on_missing_source_body(tmp_path: Path):
    """Gemini PR #211 round-2: with explicit --source AND truncated
    rows AND a missing source .txt, the CLI emits a warning line so
    the operator understands why refilled+hallucinated are zero."""
    from click.testing import CliRunner

    db_path = tmp_path / "lex.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY, author TEXT, title TEXT NOT NULL,
            year INTEGER, region TEXT, language_focus TEXT, notes TEXT);
        CREATE TABLE etymon (id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL, language TEXT NOT NULL,
            UNIQUE(canonical_form, language));
        CREATE TABLE etymon_citation (id INTEGER PRIMARY KEY AUTOINCREMENT,
            etymon_id INTEGER NOT NULL, source_id TEXT NOT NULL, page TEXT,
            short_quote TEXT, context_snippet TEXT);
        INSERT INTO source (id, title) VALUES ('book', 'Book');
        INSERT INTO etymon (canonical_form, language) VALUES ('acum', 'old-english');
        """
    )
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (1, 'book', ?)",
        (_TRUNCATED,),
    )
    conn.commit()
    conn.close()
    # NB: deliberately NOT writing book.txt — that's the test condition.

    from wyrd.generators.kenning.cli import cli as cli_root

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "refill-short-quotes",
            "--db",
            str(db_path),
            "--sources-dir",
            str(tmp_path),
            "--source",
            "book",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "truncated=   1" in result.output
    assert "refilled=   0" in result.output
    assert "warning" in result.output.lower()
    assert "book.txt" in result.output


def test_cli_refill_short_quotes_verbose_surfaces_samples(tmp_path: Path):
    """--verbose prints before/after samples per source so operators
    can spot-check refill quality. Gemini PR #211 round-1 finding."""
    from click.testing import CliRunner

    db_path = tmp_path / "lex.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY, author TEXT, title TEXT NOT NULL,
            year INTEGER, region TEXT, language_focus TEXT, notes TEXT);
        CREATE TABLE etymon (id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL, language TEXT NOT NULL,
            UNIQUE(canonical_form, language));
        CREATE TABLE etymon_citation (id INTEGER PRIMARY KEY AUTOINCREMENT,
            etymon_id INTEGER NOT NULL, source_id TEXT NOT NULL, page TEXT,
            short_quote TEXT, context_snippet TEXT);
        INSERT INTO source (id, title) VALUES ('book', 'Book');
        INSERT INTO etymon (canonical_form, language) VALUES ('acum', 'old-english');
        """
    )
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (1, 'book', ?)",
        (_TRUNCATED,),
    )
    conn.commit()
    conn.close()
    (tmp_path / "book.txt").write_text(
        "history. Acomb (Bywell St Peter) 1268 Ipm Akum. More prose."
    )

    from wyrd.generators.kenning.cli import cli as cli_root

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "refill-short-quotes",
            "--db",
            str(db_path),
            "--sources-dir",
            str(tmp_path),
            "--verbose",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "[refilled]" in result.output
    assert "old-english:acum" in result.output
    # Pin the format-dense old/new sample lines emitted by
    # _echo_refill_samples (wyrd-8uvi C901 extraction owns this) — the
    # branch was executed but its format went unasserted.
    assert "old (" in result.output
    assert "new (+" in result.output


# ---------------------------------------------------------------------------
# Multi-pipe + edge-case shapes
# ---------------------------------------------------------------------------


def test_refill_source_sql_filters_below_truncation_floor(tmp_path: Path):
    """Gemini PR #211 round-3 perf: short_quotes shorter than
    MIN_TRUNCATION_LENGTH (200 chars) are guaranteed not flagged
    by looks_truncated. The SQL filter ``length(short_quote) >= 200``
    skips them at the DB layer rather than fetching+Python-filtering.

    This test seeds 2 short citations + 1 truncated citation and
    asserts only the truncated one shows up in total_truncated."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('book', 'Book')")
    eid = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('acum', 'old-english')"
    ).lastrowid
    # 2 short healthy quotes (below 200 chars) — would be filtered by
    # looks_truncated in Python anyway, but the SQL filter skips them
    # without round-trip.
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
        (eid, "book", "Short healthy citation that ends in a period."),
    )
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
        (eid, "book", "Another short clean citation. 42 chars."),
    )
    # 1 truncated citation (>= 200 chars).
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
        (eid, "book", _TRUNCATED),
    )
    conn.commit()
    (tmp_path / "book.txt").write_text(
        "history. Acomb (Bywell St Peter) 1268 Ipm Akum. More prose."
    )

    report = refill_source(conn, "book", tmp_path, apply=True)
    # Only the one above-floor citation is considered.
    assert report.total_truncated == 1
    assert report.refilled == 1


def test_refill_short_quote_rejects_ambiguous_anchor():
    """Gemini PR #211 round-4: when the anchor appears in multiple
    places in the source body, `find()` returns the FIRST match —
    which may not be the citation's actual location. A wrong-location
    match would silently write incorrect forward context. Refuse to
    refill rather than risk corruption."""
    quote = "commentary | a common scholarly phrase here"
    # Source has the same anchor in two distinct places.
    source = (
        "First occurrence. a common scholarly phrase here is one place. "
        "Later text. a common scholarly phrase here appears again."
    )
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=80)
    assert new is None, "ambiguous anchor (multiple matches) must be rejected"
    assert recovered == 0


def test_refill_short_quote_accepts_unique_anchor():
    """Boundary: an anchor that appears EXACTLY once is accepted
    (the success path of _unique_find)."""
    quote = "commentary | a unique scholarly phrase here"
    source = "First text. a unique scholarly phrase here is the only occurrence."
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=80)
    assert new is not None
    assert recovered > 0


def test_refill_short_quote_rejects_too_short_anchor():
    """Gemini PR #211 round-3: an anchor shorter than _MIN_ANCHOR_LEN
    (20 chars) is too unspecific in typical scholar prose — could match
    in indexes, running headers, cross-references. Rejected with None
    rather than risk a wrong-location refill that silently writes
    incorrect forward context."""
    # Tail is 10 chars; whole quote is 10 chars (no pipe). Anchor would
    # be the whole quote — below the 20-char threshold.
    quote = "by 12 Akum"
    source = "by 12 Akum is here and elsewhere also by 12 Akum appears."
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=80)
    assert new is None, (
        "10-char anchor must be rejected as too short — could match wrong occurrence"
    )
    assert recovered == 0


def test_refill_short_quote_post_pipe_anchor_at_minimum_length():
    """Anchor exactly at the minimum (20 chars) refills successfully —
    pin the boundary."""
    # 20-char anchor — at the threshold, should succeed.
    quote = "commentary | abcdefghij12345_anchor"  # post-pipe is 22 chars
    source = "irrelevant abcdefghij12345_anchor and then more content here."
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=80)
    assert new is not None
    assert "content here." in new


def test_refill_short_quote_multi_pipe_uses_last_segment():
    """A short_quote with multiple `|` delimiters uses the LAST
    segment as the anchor (per rsplit('|', 1)). pr-test-analyzer
    PR #211 finding: realistic nested-commentary shapes can produce
    multiple pipes."""
    # Anchor must be ≥ _MIN_ANCHOR_LEN (20 chars) — use a long-enough final tail.
    quote = "first commentary | nested commentary | the final-segment tail anchor"
    source = "irrelevant. the final-segment tail anchor extends with more content here."
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=80)
    assert new is not None
    # The full original quote prefix is preserved (including both pipes).
    assert new.startswith("first commentary | nested commentary | the final-segment tail anchor")
    # The recovered portion came from after the LAST pipe's anchor.
    assert "content here." in new


# ---------------------------------------------------------------------------
# Round-5 fanout coverage additions
# ---------------------------------------------------------------------------


def test_load_source_body_unicode_decode_error_returns_none(tmp_path: Path):
    """Some OCR-produced source bodies have stray non-UTF-8 bytes;
    _load_source_body must surface as None rather than crashing the
    refill run. test-coverage-reviewer round-5 gap."""
    from wyrd.generators.kenning.short_quote_refill import _load_source_body

    (tmp_path / "bad.txt").write_bytes(b"valid utf-8 prefix \xff\xfe invalid bytes")
    assert _load_source_body(tmp_path, "bad") is None


def test_load_source_body_permission_error_returns_none(tmp_path: Path):
    """code-reviewer round-5: a file we can't open (PermissionError
    or other OSError subclass) must not crash the refill run."""
    import os

    from wyrd.generators.kenning.short_quote_refill import _load_source_body

    p = tmp_path / "locked.txt"
    p.write_text("body")
    os.chmod(p, 0)  # no read permission
    try:
        assert _load_source_body(tmp_path, "locked") is None
    finally:
        os.chmod(p, 0o644)  # restore for cleanup


def test_refill_short_quote_fallback_anchor_respects_min_length():
    """silent-failure-hunter round-5 round-2: the fallback path
    (40-char anchor) ALSO checks _MIN_ANCHOR_LEN after .strip().
    Without this guard, a 40-char tail ending in mostly whitespace
    could collapse to a 1-2 char anchor that bypasses the safety
    rail."""
    # Construct a tail >80 chars so the long anchor fires.
    # The long anchor (last 80 chars) has no match in source.
    # The fallback last-40 with .strip() collapses to a too-short string.
    long_pad = "a" * 60  # paraphrased prefix
    # Last 40 chars of tail are mostly whitespace; after .strip() it's "x"
    short_strip_segment = "                                       x"  # 40 chars
    quote = f"commentary | {long_pad}{short_strip_segment}"
    # Source contains the stripped fragment but the function should
    # reject because the stripped anchor is below _MIN_ANCHOR_LEN.
    source = "irrelevant prose x and then more content here."
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=80)
    assert new is None, "stripped fallback anchor below _MIN_ANCHOR_LEN must be rejected"
    assert recovered == 0


def test_refill_source_apply_actually_commits(tmp_path: Path):
    """pr-test-analyzer round-5: the apply test was reading back the
    updated row on the SAME connection, which sees uncommitted data.
    Verify the SQL UPDATE actually commits by opening a SECOND
    connection to the same DB file."""
    db_path = tmp_path / "lex.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY, title TEXT NOT NULL);
        CREATE TABLE etymon (id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL, language TEXT NOT NULL,
            UNIQUE(canonical_form, language));
        CREATE TABLE etymon_citation (id INTEGER PRIMARY KEY AUTOINCREMENT,
            etymon_id INTEGER NOT NULL, source_id TEXT NOT NULL, page TEXT,
            short_quote TEXT, context_snippet TEXT);
        INSERT INTO source (id, title) VALUES ('book', 'Book');
        """
    )
    eid = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('acum', 'old-english')"
    ).lastrowid
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
        (eid, "book", _TRUNCATED),
    )
    conn.commit()
    (tmp_path / "book.txt").write_text(
        "history. Acomb (Bywell St Peter) 1268 Ipm Akum. More prose follows here."
    )

    refill_source(conn, "book", tmp_path, apply=True)
    conn.close()

    # Open a fresh connection: only committed data is visible.
    fresh = sqlite3.connect(str(db_path))
    row = fresh.execute("SELECT short_quote FROM etymon_citation").fetchone()
    fresh.close()
    assert row[0] != _TRUNCATED, "refill_source did not commit the SQL UPDATE"
    assert "Akum" in row[0]


def test_cli_refill_short_quotes_apply_round_trips(tmp_path: Path):
    """test-coverage-reviewer round-5: every CLI test was dry-run.
    Exercise the --apply path end-to-end: verify the APPLIED marker
    appears in stdout AND the DB row was actually changed."""
    from click.testing import CliRunner

    db_path = tmp_path / "lex.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY, title TEXT NOT NULL);
        CREATE TABLE etymon (id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL, language TEXT NOT NULL,
            UNIQUE(canonical_form, language));
        CREATE TABLE etymon_citation (id INTEGER PRIMARY KEY AUTOINCREMENT,
            etymon_id INTEGER NOT NULL, source_id TEXT NOT NULL, page TEXT,
            short_quote TEXT, context_snippet TEXT);
        INSERT INTO source (id, title) VALUES ('book', 'Book');
        """
    )
    eid = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('acum', 'old-english')"
    ).lastrowid
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
        (eid, "book", _TRUNCATED),
    )
    conn.commit()
    conn.close()
    (tmp_path / "book.txt").write_text(
        "history. Acomb (Bywell St Peter) 1268 Ipm Akum. More prose follows here."
    )

    from wyrd.generators.kenning.cli import cli as cli_root

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "refill-short-quotes",
            "--db",
            str(db_path),
            "--sources-dir",
            str(tmp_path),
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "(APPLIED)" in result.stdout
    # Verify the DB row was actually changed.
    fresh = sqlite3.connect(str(db_path))
    row = fresh.execute("SELECT short_quote FROM etymon_citation").fetchone()
    fresh.close()
    assert row[0] != _TRUNCATED
    assert "Akum" in row[0]


def test_load_source_body_strips_path_components_from_source_id(tmp_path: Path):
    """Gemini PR #211 round-7 (security-high): if a source_id ever
    contains path separators (slashes, ../, etc.), _load_source_body
    must not traverse outside sources_dir. Path(source_id).name
    strips directory components, leaving only the basename."""
    from wyrd.generators.kenning.short_quote_refill import _load_source_body

    # Put a benign file at sources_dir/legit.txt and a sensitive file
    # OUTSIDE sources_dir. A malicious source_id ../sensitive must
    # NOT reach outside.
    sensitive = tmp_path / "sensitive.txt"
    sensitive.write_text("should-not-be-readable")
    inner = tmp_path / "sources"
    inner.mkdir()
    (inner / "legit.txt").write_text("legitimate source body content here.")

    # A source_id with traversal — Path(source_id).name → 'sensitive',
    # so we look for sources/sensitive.txt, which doesn't exist.
    # Result: None, not the contents of ../sensitive.txt.
    result = _load_source_body(inner, "../sensitive")
    assert result is None, "path traversal must not return file outside sources_dir"


def test_cli_refill_short_quotes_default_mode_uses_distinct_source_ids(tmp_path: Path):
    """Gemini PR #211 round-7 (efficiency): when no --source is given,
    the CLI iterates only sources that have at least one citation.
    Sources in the source table with zero citations are skipped."""
    from click.testing import CliRunner

    db_path = tmp_path / "lex.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY, title TEXT NOT NULL);
        CREATE TABLE etymon (id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL, language TEXT NOT NULL,
            UNIQUE(canonical_form, language));
        CREATE TABLE etymon_citation (id INTEGER PRIMARY KEY AUTOINCREMENT,
            etymon_id INTEGER NOT NULL, source_id TEXT NOT NULL, page TEXT,
            short_quote TEXT, context_snippet TEXT);
        -- Two sources, only one has citations.
        INSERT INTO source (id, title) VALUES ('with_cites', 'Has cites');
        INSERT INTO source (id, title) VALUES ('empty_source', 'No cites');
        INSERT INTO etymon (canonical_form, language) VALUES ('acum', 'old-english');
        """
    )
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (1, 'with_cites', ?)",
        (_TRUNCATED,),
    )
    conn.commit()
    conn.close()
    (tmp_path / "with_cites.txt").write_text(
        "history. Acomb (Bywell St Peter) 1268 Ipm Akum. More prose."
    )
    # NB: deliberately not creating empty_source.txt — if the CLI
    # were iterating ALL sources it would log a missing-body warning
    # for empty_source. With the DISTINCT-citations filter, it skips.

    from wyrd.generators.kenning.cli import cli as cli_root

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "refill-short-quotes", "--db", str(db_path), "--sources-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    # with_cites is processed.
    assert "with_cites" in result.output
    # empty_source is NOT processed (no warning, no progress line).
    assert "empty_source" not in result.output


def test_cli_refill_short_quotes_warns_on_unreadable_source_body(tmp_path: Path):
    """test-coverage-reviewer round-5: the 'file exists but unreadable'
    branch of the missing-source warning was untested. Covers the
    empty-body case (after round-4, _load_source_body normalizes
    empty/whitespace-only bodies to None, so the warning fires)."""
    from click.testing import CliRunner

    db_path = tmp_path / "lex.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY, title TEXT NOT NULL);
        CREATE TABLE etymon (id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL, language TEXT NOT NULL,
            UNIQUE(canonical_form, language));
        CREATE TABLE etymon_citation (id INTEGER PRIMARY KEY AUTOINCREMENT,
            etymon_id INTEGER NOT NULL, source_id TEXT NOT NULL, page TEXT,
            short_quote TEXT, context_snippet TEXT);
        INSERT INTO source (id, title) VALUES ('book', 'Book');
        """
    )
    eid = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('acum', 'old-english')"
    ).lastrowid
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id, short_quote) VALUES (?, ?, ?)",
        (eid, "book", _TRUNCATED),
    )
    conn.commit()
    conn.close()
    # File exists but is whitespace-only — _load_source_body returns None.
    (tmp_path / "book.txt").write_text("   \n\n   ")

    from wyrd.generators.kenning.cli import cli as cli_root

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "refill-short-quotes",
            "--db",
            str(db_path),
            "--sources-dir",
            str(tmp_path),
            "--source",
            "book",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "warning" in result.output.lower()
    # File exists, so we should NOT see "not found".
    assert "not found" not in result.output
    # Should see the "exists but empty/unreadable" branch.
    assert "empty/unreadable" in result.output
