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
    # deliberately a PLAIN connection (no row_factory) — scan() must not depend on it.
    conn = sqlite3.connect(":memory:")
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
            ("G", "England", "England"),  # dirty country-root
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
    assert sev["England"] == "dirty"  # country-root used as a region


def test_scan_region_no_country_uses_model_recognition():
    """With no country on the row, England-scope falls back to the model: a known
    alias is still caught as stale (pins classify_region's country=None path)."""
    conn = _conn([("A", None, "BUK"), ("B", None, "Yorkshire")], [])
    sev = {x.value: x.severity for x in scan(conn)["region"]}
    assert sev["BUK"] == "stale"  # recognized alias even without a country hint
    assert sev["Yorkshire"] == "ok"


def test_scan_language_severities():
    conn = _conn([], [("a", "old-english"), ("b", "klingon"), ("c", None), ("d", "   ")])
    f = scan(conn)
    sev = {x.value: x.severity for x in f["language"]}
    assert sev["old-english"] == "ok"
    assert sev["klingon"] == "dirty"
    assert sev[None] == "info"
    assert sev["   "] == "info"  # whitespace-only treated as missing, not dirty


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


def _file_db(tmp_path, toponyms=(), etymons=()):
    db = tmp_path / "lex.db"
    conn = sqlite3.connect(db)
    conn.executescript(_DDL)
    conn.executemany("INSERT INTO toponym(modern_name, country, region) VALUES (?,?,?)", toponyms)
    conn.executemany("INSERT INTO etymon(canonical_form, language) VALUES (?,?)", etymons)
    conn.commit()
    conn.close()
    return db


def test_cli_strict_exits_nonzero_on_violations(tmp_path):
    # Freedonia is a genuinely-unknown country (dirty), exercising the dirty→exit path.
    db = _file_db(tmp_path, toponyms=[("B", "Freedonia", None)])
    res = CliRunner().invoke(lexicon_vocab_health, ["--db", str(db), "--strict"])
    assert res.exit_code == 1
    assert "Freedonia" in res.output


def test_cli_top_caps_and_reports_remainder(tmp_path):
    # 3 distinct unknown languages; --top 1 shows one + "… and 2 more".
    db = _file_db(tmp_path, etymons=[("a", "klingon"), ("b", "dothraki"), ("c", "sindarin")])
    res = CliRunner().invoke(lexicon_vocab_health, ["--db", str(db), "--top", "1"])
    assert "… and 2 more" in res.output
    # --top 0 = no cap: all three listed, no remainder line.
    res_all = CliRunner().invoke(lexicon_vocab_health, ["--db", str(db), "--top", "0"])
    assert "more" not in res_all.output
    assert all(lang in res_all.output for lang in ("klingon", "dothraki", "sindarin"))


def test_cli_all_lists_ok_values(tmp_path):
    db = _file_db(tmp_path, toponyms=[("A", "England", "Yorkshire")])
    plain = CliRunner().invoke(lexicon_vocab_health, ["--db", str(db)])
    assert "Yorkshire" not in plain.output  # ok values hidden by default
    verbose = CliRunner().invoke(lexicon_vocab_health, ["--db", str(db), "--all"])
    assert "Yorkshire" in verbose.output and "[ok" in verbose.output


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
