"""wyrd-vm8t Loop 3: pure helpers of the LLM pronunciation miner."""

from __future__ import annotations

import sqlite3

from wyrd.generators.kenning.lexicon.pronunciation_mining import (
    METHOD,
    build_messages,
    parse_response,
    record,
    select_targets,
)


def test_parse_response():
    assert parse_response('{"ipa":"/knɔk/","confidence":"high"}') == ("/knɔk/", "high")
    assert parse_response('{"ipa":"/x/"}') == ("/x/", "low")  # missing → low
    assert parse_response('{"ipa":"/x/","confidence":"wild"}') == ("/x/", "low")  # bad → low
    assert parse_response('{"ipa":"knɔk"}') is None  # no slash
    assert parse_response('{"confidence":"high"}') is None  # no ipa
    assert parse_response("not json") is None


def test_build_messages_primes_language():
    msgs = build_messages("old-french", "ville")
    assert msgs[0]["role"] == "system" and "Old French" in msgs[0]["content"]
    assert "ville" in msgs[1]["content"]


def test_record():
    hit = record("irish", "cnoc", ("/knɔk/", "high"), "qwen2.5:7b")
    assert hit["ref"] == "irish:cnoc" and hit["ipa"] == "/knɔk/" and hit["method"] == METHOD
    miss = record("irish", "x", None, "qwen2.5:7b")
    assert miss["skipped"] == "no-ipa" and "ipa" not in miss


def test_select_targets(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE etymon (id INTEGER PRIMARY KEY, language TEXT, canonical_form TEXT, "
        "pronunciation_ipa TEXT);"
        "CREATE TABLE reflex_etymon (reflex_id INTEGER, etymon_id INTEGER);"
    )
    conn.executemany(
        "INSERT INTO etymon VALUES (?,?,?,?)",
        [
            (1, "irish", "baile", None),  # target
            (2, "irish", "cnoc", "/knɔk/"),  # has IPA → excluded
            (3, "irish", "a b", None),  # space → excluded
            (4, "irish", "*recon", None),  # star → excluded
            (5, "old-french", "ville", None),  # target (other lang)
            (6, "irish", "orphan", None),  # no reflex link → excluded
        ],
    )
    conn.executemany(
        "INSERT INTO reflex_etymon VALUES (?,?)", [(10, 1), (11, 2), (12, 3), (13, 4), (14, 5)]
    )
    conn.commit()
    conn.close()
    assert select_targets(str(db), ("irish", "old-french")) == [
        ("irish", "baile"),
        ("old-french", "ville"),
    ]
    assert select_targets(str(db), ("old-french",)) == [("old-french", "ville")]
