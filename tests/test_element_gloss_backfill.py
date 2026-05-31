"""wyrd-u9k6: grounded element-gloss backfill — derive, collect, apply, adjudicate."""

from __future__ import annotations

import json
import sqlite3
import types

from wyrd.generators.kenning.lexicon.element_gloss_backfill import (
    _has_vowel,
    _normform,
    _position,
    accept_adjudication,
    apply_element_glosses,
    build_adjudication_user,
    collect_element_glosses,
    derive_element_glosses,
    toponym_context,
)


def _fake_db(conn: sqlite3.Connection):
    return types.SimpleNamespace(conn=conn)


def _grounding_db() -> sqlite3.Connection:
    """A minimal lexicon with one glossed suffix element (-ick → OE wīc) plus
    three toponyms whose final element is that etymon, and the orphan reflex."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE etymon (id INTEGER PRIMARY KEY, language TEXT, canonical_form TEXT);
        CREATE TABLE etymon_gloss (etymon_id INTEGER, gloss TEXT);
        CREATE TABLE toponym (id INTEGER PRIMARY KEY, modern_name TEXT);
        CREATE TABLE toponym_etymology (id INTEGER PRIMARY KEY, toponym_id INTEGER);
        CREATE TABLE toponym_etymology_element (
            toponym_etymology_id INTEGER, ordinal INTEGER, etymon_id INTEGER);
        CREATE TABLE reflex (id INTEGER PRIMARY KEY, surface_form TEXT, position TEXT);
        CREATE TABLE reflex_etymon (reflex_id INTEGER, etymon_id INTEGER,
            PRIMARY KEY (reflex_id, etymon_id));
        """
    )
    conn.execute("INSERT INTO etymon VALUES (1, 'old-english', 'wīc')")
    conn.execute("INSERT INTO etymon VALUES (2, 'old-english', 'hard')")  # a first element
    conn.executemany(
        "INSERT INTO etymon_gloss VALUES (?,?)",
        [(1, "dwelling"), (1, "specialized farm"), (2, "hard")],
    )
    # Hardwick / Chiswick / Berwick: each is (first, wīc)
    for tid, name in [(1, "Hardwick"), (2, "Chiswick"), (3, "Berwick")]:
        conn.execute("INSERT INTO toponym VALUES (?,?)", (tid, name))
        conn.execute("INSERT INTO toponym_etymology VALUES (?,?)", (tid, tid))
        conn.execute("INSERT INTO toponym_etymology_element VALUES (?,0,2)", (tid,))
        conn.execute("INSERT INTO toponym_etymology_element VALUES (?,1,1)", (tid,))
    conn.execute("INSERT INTO reflex VALUES (10, '-ick', 'post')")
    conn.commit()
    return conn


def _census_db() -> sqlite3.Connection:
    """Census bundle with one unglossed english usage (-ick)."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE proportions_usage (culture TEXT, usage_key TEXT, weight INTEGER);
        CREATE TABLE meaning (usage_key TEXT PRIMARY KEY, data BLOB);
        """
    )
    conn.execute("INSERT INTO proportions_usage VALUES ('english', '-ick', 50)")
    # an empty-meaning entry → still 'unglossed'
    conn.execute(
        "INSERT INTO meaning VALUES ('-ick', ?)", (json.dumps({"entries": [{"meaning": []}]}),)
    )
    conn.commit()
    return conn


def test_pure_helpers():
    assert _normform("-Tūn-") == "tun"
    assert _position("-ick") == "suffix"
    assert _position("Bow-") == "prefix"
    assert _position("-ing-") == "infix"
    assert _position("Bourne") == "standalone"
    assert _has_vowel("-ick") and not _has_vowel("nt")


def test_derive_element_glosses_grounded_consensus():
    rows = derive_element_glosses(_census_db(), _grounding_db(), "english")
    assert len(rows) == 1
    row = rows[0]
    assert row["surface"] == "-ick"
    assert row["is_element"] is True  # 3/3 support for wīc → high
    top = row["candidates"][0]
    # candidate form is the normalized vote key (deaccented), not the raw "wīc"
    assert top["form"] == "wic" and top["language"] == "old-english"
    assert top["support"] == 3 and "dwelling" in top["glosses"]


def test_collect_and_apply_round_trip(tmp_path):
    # consensus jsonl: one is_element row; adjudication jsonl: one accepted pick
    cons = tmp_path / "_element_glosses.jsonl"
    cons.write_text(
        json.dumps(
            {
                "surface": "-ick",
                "position": "suffix",
                "is_element": True,
                "candidates": [{"language": "old-english", "form": "wic", "glosses": ["dwelling"]}],
            }
        )
        + "\n"
    )
    adj = tmp_path / "_adj.jsonl"
    adj.write_text(
        json.dumps(
            {
                "surface": "nt",  # consonant fragment — collect's vowel guard drops it
                "position": "suffix",
                "accepted": True,
                "candidate": {"language": "old-english", "form": "tun", "glosses": ["farm"]},
            }
        )
        + "\n"
    )
    state = collect_element_glosses(str(cons), str(adj))
    surfaces = {s["surface"] for s in state}
    assert "-ick" in surfaces and "nt" not in surfaces  # vowel guard at collect

    conn = _grounding_db()
    res = apply_element_glosses(_fake_db(conn), state, apply=True)
    assert res["linked_surfaces"] == 1 and res.get("no_reflex", 0) == 0
    linked = conn.execute("SELECT etymon_id FROM reflex_etymon WHERE reflex_id=10").fetchall()
    assert linked == [(1,)]  # -ick → wīc


def test_apply_dry_run_writes_nothing():
    conn = _grounding_db()
    state = [{"surface": "-ick", "position": "suffix", "language": "old-english", "form": "wic"}]
    res = apply_element_glosses(_fake_db(conn), state, apply=False)
    assert res["linked_surfaces"] == 1
    assert "links" not in res  # nothing inserted
    assert conn.execute("SELECT COUNT(*) FROM reflex_etymon").fetchone()[0] == 0


def test_accept_adjudication():
    row = {"candidates": [{"form": "a"}, {"form": "b"}]}
    assert accept_adjudication(row, {"choice": 1, "confidence": "high"})["form"] == "a"
    assert accept_adjudication(row, {"choice": 2, "confidence": "medium"})["form"] == "b"
    assert accept_adjudication(row, {"choice": 0, "confidence": "high"}) is None  # none/junk
    assert accept_adjudication(row, {"choice": 1, "confidence": "low"}) is None  # too unsure
    assert accept_adjudication(row, {"choice": "x", "confidence": "high"}) is None  # bad int


def test_toponym_context_and_prompt():
    topos = [("Hardwick", "hardwick"), ("Chiswick", "chiswick"), ("Oxford", "oxford")]
    ctx = toponym_context("-ick", "suffix", topos)
    assert ctx == ["Hardwick", "Chiswick"]
    user = build_adjudication_user(
        "-ick",
        "suffix",
        ctx,
        [
            {
                "language": "old-english",
                "form": "wic",
                "glosses": ["dwelling"],
                "support": 3,
                "share": 0.5,
            }
        ],
    )
    assert "Hardwick" in user and "wic" in user and "[1]" in user
