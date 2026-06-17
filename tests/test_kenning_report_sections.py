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
        ]
    )
    _section_regions(conn, top=15)
    out = capsys.readouterr().out
    assert "Yorkshire @ England" in out  # region carries its country context
    assert "BUK @ England" in out
    assert "[stale]" in out
    assert "non-canonical" in out  # the header summary


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


def test_section_ok_values_unmarked(capsys):
    conn = _conn(etymons=[("a", "old-english")])
    _section_languages(conn, top=15)
    out = capsys.readouterr().out
    assert "old-english" in out and "[" not in out.split("===")[-1]
