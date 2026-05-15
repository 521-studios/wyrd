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
    or empty content."""
    quote = "commentary | The end of source"
    source = "complete prose. The end of source"  # anchor matches at the very end
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


def test_refill_short_quote_trims_to_terminal_char(tmp_path: Path):
    """The recovered forward window is trimmed back to the last
    terminal character (.!?'")]}) so subsequent audit runs don't
    re-flag the refilled row as truncated."""
    quote = "commentary | early citation Ab"
    source = (
        "irrelevant. early citation Abber 1086 Akum. Then much more prose without terminal punct"
    )
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=80)
    assert new is not None
    # Must end at the terminal char (".") found within the window.
    assert new.rstrip()[-1] in ".!?\"')]}"
    # Specifically: the trim should land at "Akum." not at the end
    # of the window mid-word in "without terminal punct".
    assert new.endswith("Akum.")


def test_refill_short_quote_returns_none_when_no_terminal_in_window(tmp_path: Path):
    """If the forward window has no terminal character, the chunk is
    too fragmentary to safely extend the citation. Return None
    rather than emit a partial refill that would itself look
    truncated on the next audit run."""
    quote = "commentary | anchor"
    source = "irrelevant anchor then a long stream of words without any terminal punctuation at all"
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
    import subprocess

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

    _ = subprocess  # silence unused-import lint


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
    assert "(dry-run)" in result.output
    assert "truncated=1" in result.output
    assert "refilled=1" in result.output


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


# ---------------------------------------------------------------------------
# Multi-pipe + edge-case shapes
# ---------------------------------------------------------------------------


def test_refill_short_quote_multi_pipe_uses_last_segment(tmp_path: Path):
    """A short_quote with multiple `|` delimiters uses the LAST
    segment as the anchor (per rsplit('|', 1)). pr-test-analyzer
    PR #211 finding: realistic nested-commentary shapes can produce
    multiple pipes."""
    quote = "first commentary | nested commentary | final tail anchor"
    source = "irrelevant. final tail anchor extends with more content here."
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=80)
    assert new is not None
    # The full original quote prefix is preserved (including both pipes).
    assert new.startswith("first commentary | nested commentary | final tail anchor")
    # The recovered portion came from after the LAST pipe's anchor.
    assert "content here." in new
