"""Tests for the report's country/region/language distribution sections (wyrd-c95y).

Fixture-only: minimal in-memory toponym+etymon DB, sections called directly and
their click.echo output captured. Never touches the live lexicon.db.
"""

from __future__ import annotations

import sqlite3

from wyrd.generators.kenning.cli.lexicon.report import (
    _section_countries,
    _section_languages,
    _section_regions,
)

_DDL = """
CREATE TABLE toponym (id INTEGER PRIMARY KEY, modern_name TEXT, country TEXT, region TEXT);
CREATE TABLE etymon (id INTEGER PRIMARY KEY, canonical_form TEXT, language TEXT);
"""


def _conn(toponyms=(), etymons=()) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row  # mirror production (_readonly_lexicon sets Row)
    conn.executescript(_DDL)
    conn.executemany("INSERT INTO toponym(modern_name, country, region) VALUES (?,?,?)", toponyms)
    conn.executemany("INSERT INTO etymon(canonical_form, language) VALUES (?,?)", etymons)
    return conn


def test_section_countries_distribution_and_marks(capsys):
    conn = _conn(
        toponyms=[
            ("A", "England", None),
            ("B", "England", None),
            ("C", "Republic of Ireland", None),  # stale alias
        ]
    )
    _section_countries(conn)
    out = capsys.readouterr().out
    assert "England" in out
    assert "Republic of Ireland" in out
    assert "[stale]" in out  # the alias is flagged inline
    # canonical England carries no marker
    england_line = next(line for line in out.splitlines() if line.strip().startswith("England"))
    assert "[" not in england_line


def test_section_regions_shows_country_context_and_marks(capsys):
    conn = _conn(
        toponyms=[
            ("A", "England", "Yorkshire"),  # ok node
            ("B", "England", "BUK"),  # stale alias
            ("C", "Wales", "Anglesey"),  # info: non-England, no model yet
            ("D", None, "Surrey"),  # ok node, NULL country → no "@" context
        ]
    )
    _section_regions(conn, top=15)
    out = capsys.readouterr().out
    assert "Yorkshire @ England" in out  # region carries its country context
    assert "BUK @ England" in out
    assert "[stale]" in out
    assert "1 non-canonical" in out  # only BUK counts; Anglesey(info)/Surrey(ok) excluded
    # info region renders unmarked
    anglesey = next(line for line in out.splitlines() if "Anglesey" in line)
    assert "[" not in anglesey
    # NULL-country region renders without the "@ <country>" suffix (no-context branch)
    surrey = next(line for line in out.splitlines() if "Surrey" in line)
    assert "@" not in surrey


def test_section_regions_caps_to_top(capsys):
    conn = _conn(
        toponyms=[
            ("A", "England", "Kent"),
            ("B", "England", "Kent"),  # Kent x2 — highest count
            ("C", "England", "Surrey"),
            ("D", "England", "Devon"),
        ]
    )
    _section_regions(conn, top=2)
    out = capsys.readouterr().out
    value_lines = [line for line in out.splitlines() if line.startswith("  ")]
    assert len(value_lines) == 2  # only the top 2 of 3 distinct regions
    assert "3 distinct" in out


def test_section_countries_renders_null(capsys):
    conn = _conn(toponyms=[("A", None, None)])
    _section_countries(conn)
    out = capsys.readouterr().out
    assert "(null)" in out


def test_section_negative_top_shows_none_not_tail(capsys):
    conn = _conn(etymons=[("a", "old-english"), ("b", "klingon")])
    _section_languages(conn, top=-1)  # must not slice from the end
    out = capsys.readouterr().out
    value_lines = [line for line in out.splitlines() if line.startswith("  ")]
    assert value_lines == []


def test_section_regions_negative_top_shows_none(capsys):
    # same max(0, top) guard as _section_languages — pin it for regions too
    conn = _conn(toponyms=[("A", "England", "Kent"), ("B", "England", "Surrey")])
    _section_regions(conn, top=-1)
    out = capsys.readouterr().out
    value_lines = [line for line in out.splitlines() if line.startswith("  ")]
    assert value_lines == []


def test_section_languages_caps_and_flags_noncanonical(capsys):
    conn = _conn(
        etymons=[("a", "old-english"), ("b", "klingon"), ("c", "dothraki"), ("d", "celtic")]
    )
    _section_languages(conn, top=2)
    out = capsys.readouterr().out
    assert "4 distinct, 2 non-canonical" in out  # klingon + dothraki
    # capped to top 2 by count → only 2 value rows under the header
    value_lines = [line for line in out.splitlines() if line.startswith("  ")]
    assert len(value_lines) == 2


def test_section_dirty_value_renders_marker(capsys):
    """A dirty value renders the [dirty] marker on its row (not just counted in
    the header) — pins _mark on the dirty path, distinct from [stale]."""
    conn = _conn(etymons=[("a", "klingon"), ("b", "old-english")])
    _section_languages(conn, top=15)
    out = capsys.readouterr().out
    klingon = next(line for line in out.splitlines() if "klingon" in line)
    assert "[dirty]" in klingon


def test_section_ok_values_unmarked(capsys):
    conn = _conn(etymons=[("a", "old-english")])
    _section_languages(conn, top=15)
    out = capsys.readouterr().out
    assert "old-english" in out and "[" not in out.split("===")[-1]
