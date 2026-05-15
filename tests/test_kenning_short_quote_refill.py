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
    Refill locates the tail in source body and extends forward."""
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


def test_refill_short_quote_no_pipe_uses_whole_quote_as_anchor():
    """When there's no pipe, the entire quote is the anchor (so the
    full quote text must appear in the source body for refill to fire)."""
    quote = "Plain truncated text about Bedfordshire 1086 Ak"
    source = "Plain truncated text about Bedfordshire 1086 Akum is recorded as the entry"
    new, recovered = refill_short_quote(quote, _normalize_whitespace(source), window=40)
    assert new is not None
    assert "Akum" in new
    assert recovered > 0


def test_refill_short_quote_falls_back_to_shorter_anchor():
    """80-char anchor fails but the shorter 40-char fallback matches
    (LLM lightly paraphrased the early part of the tail). The tail
    must be >80 chars so the long anchor is distinct from the short
    fallback."""
    # 100-char tail. The first 60 chars are paraphrased; the last 40
    # match the source verbatim. The 80-char anchor (chars -80:)
    # spans both regions and won't match; the 40-char fallback hits.
    quote = (
        "commentary | "
        "WAS-PARAPHRASED-BY-LLM-SO-WONT-MATCH-IN-SOURCE-XX "  # 50 chars
        "matched suffix that exists in source verbatim"  # 45 chars
    )
    source = "irrelevant prose matched suffix that exists in source verbatim then more content"
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
        "history. Acomb (Bywell St Peter) 1268 Ipm Akum and more prose follows here"
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
        "history. Acomb (Bywell St Peter) 1268 Ipm Akum and more prose follows here"
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
    """When sources/<source_id>.txt doesn't exist, refill_source
    returns an empty report rather than crashing (operator may have
    audits across sources whose .txt isn't on disk)."""
    conn = _build_fixture_db()
    conn.execute("INSERT INTO source (id, title) VALUES ('missing', 'M')")
    conn.commit()
    report = refill_source(conn, "missing", tmp_path, apply=True)
    assert report.total_truncated == 0
    assert report.refilled == 0
