"""Tests for the controlled-vocab health scan (wyrd-cgl4).

Fixture-only: an in-memory toponym + etymon DB seeded with clean / stale / dirty /
info values. Never touches the live lexicon.db.
"""

from __future__ import annotations

import sqlite3

from click.testing import CliRunner

from wyrd.generators.kenning.cli.lexicon.vocab_health import lexicon_vocab_health
from wyrd.generators.kenning.lexicon.region_model import classify_region
from wyrd.generators.kenning.lexicon.vocab_health import scan, violations

_DDL = """
CREATE TABLE toponym (id INTEGER PRIMARY KEY, modern_name TEXT, country TEXT, region TEXT);
CREATE TABLE etymon (id INTEGER PRIMARY KEY, canonical_form TEXT, language TEXT);
"""


def _conn(toponyms, etymons) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.executemany(
        "INSERT INTO toponym(modern_name, country, region) VALUES (?, ?, ?)",
        toponyms,
    )
    conn.executemany("INSERT INTO etymon(canonical_form, language) VALUES (?, ?)", etymons)
    return conn


# --- classify_region (the read-only model classifier) -----------------------


def test_classify_region_buckets():
    assert classify_region("Yorkshire", country="England") == "node"
    assert classify_region("BUK", country="England") == "alias"
    assert classify_region("England (Danelaw)", country="England") == "zone"
    assert classify_region("Cumberland and Westmorland", country="England") == "quarantine"
    assert classify_region("England", country="England") == "country-root"
    assert classify_region("Freedonia", country="England") == "unknown"
    assert classify_region("Cornwall", country="Wales") == "non-england"
    assert classify_region(None) == "empty"
    assert classify_region("   ") == "empty"


# --- scan severity classification -------------------------------------------


def _find(findings, dimension, value):
    return next(f for f in findings[dimension] if f.value == value)


def test_scan_country_severities():
    conn = _conn(
        [
            ("A", "England", None),
            ("B", "Republic of Ireland", None),  # stale alias → Ireland
            ("C", "Freedonia", None),  # dirty unknown
            ("D", None, None),  # info NULL
        ],
        [],
    )
    f = scan(conn)
    assert _find(f, "country", "England").severity == "ok"
    assert _find(f, "country", "Republic of Ireland").severity == "stale"
    assert _find(f, "country", "Freedonia").severity == "dirty"
    assert _find(f, "country", None).severity == "info"


def test_scan_region_severities():
    conn = _conn(
        [
            ("A", "England", "Yorkshire"),  # ok node
            ("B", "England", "BUK"),  # stale alias
            ("C", "England", "England (Danelaw)"),  # info (deferred zone)
            ("D", "England", "Cumberland and Westmorland"),  # dirty quarantine
            ("E", "England", "Freedonia"),  # dirty unknown
            ("F", "Wales", "Anglesey"),  # info non-england
        ],
        [],
    )
    f = scan(conn)
    sev = {x.value: x.severity for x in f["region"]}
    assert sev["Yorkshire"] == "ok"
    assert sev["BUK"] == "stale"
    assert sev["England (Danelaw)"] == "info"
    assert sev["Cumberland and Westmorland"] == "dirty"
    assert sev["Freedonia"] == "dirty"
    assert sev["Anglesey"] == "info"


def test_scan_language_severities():
    conn = _conn([], [("a", "old-english"), ("b", "klingon"), ("c", None)])
    f = scan(conn)
    sev = {x.value: x.severity for x in f["language"]}
    assert sev["old-english"] == "ok"
    assert sev["klingon"] == "dirty"
    assert sev[None] == "info"


def test_violations_collects_stale_and_dirty_only():
    conn = _conn(
        [
            ("A", "England", "Yorkshire"),  # all ok
            ("B", "Republic of Ireland", None),  # stale country
            ("C", "England", "Freedonia"),  # dirty region (England-scoped unknown)
        ],
        [("a", "old-english"), ("b", "klingon")],
    )
    bad = violations(scan(conn))
    sevs = {f.severity for f in bad}
    assert sevs == {"stale", "dirty"}
    # Republic of Ireland (stale) + Freedonia (dirty) + klingon (dirty)
    assert {f.value for f in bad} == {"Republic of Ireland", "Freedonia", "klingon"}


def test_counts_aggregate_rows():
    conn = _conn(
        [("A", "Republic of Ireland", None), ("B", "Republic of Ireland", None)],
        [],
    )
    f = scan(conn)
    assert _find(f, "country", "Republic of Ireland").count == 2


# --- CLI: --strict exit code ------------------------------------------------


def test_cli_strict_exits_nonzero_on_violations(tmp_path):
    db = tmp_path / "lex.db"
    conn = sqlite3.connect(db)
    conn.executescript(_DDL)
    conn.execute("INSERT INTO toponym(modern_name, country, region) VALUES ('B','Brittany',NULL)")
    conn.commit()
    conn.close()
    res = CliRunner().invoke(lexicon_vocab_health, ["--db", str(db), "--strict"])
    assert res.exit_code == 1
    assert "Brittany" in res.output


def test_cli_clean_db_exits_zero(tmp_path):
    db = tmp_path / "lex.db"
    conn = sqlite3.connect(db)
    conn.executescript(_DDL)
    conn.execute(
        "INSERT INTO toponym(modern_name, country, region) VALUES ('A','England','Yorkshire')"
    )
    conn.execute("INSERT INTO etymon(canonical_form, language) VALUES ('x','old-english')")
    conn.commit()
    conn.close()
    res = CliRunner().invoke(lexicon_vocab_health, ["--db", str(db), "--strict"])
    assert res.exit_code == 0
    assert "canonical ✓" in res.output
