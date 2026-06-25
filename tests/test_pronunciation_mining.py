"""wyrd-vm8t Loop 3: pure helpers of the LLM pronunciation miner."""

from __future__ import annotations

import sqlite3

from wyrd.generators.kenning.cli.lexicon.mine_pronunciation_llm import _load_done
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


def test_load_done_retries_errored_rows(tmp_path):
    p = tmp_path / "_pronunciation.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"language":"irish","form":"baile","ipa":"/bˠalʲə/","confidence":"high"}',
                '{"language":"irish","form":"x","skipped":"no-ipa"}',  # real outcome → done
                '{"language":"irish","form":"cnoc","error":"timeout"}',  # transient → retry
            ]
        ),
        encoding="utf-8",
    )
    # The errored row is NOT counted done, so a resumed run retries it.
    assert _load_done(p) == {("irish", "baile"), ("irish", "x")}
    assert _load_done(tmp_path / "missing.jsonl") == set()


def test_select_targets(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE etymon (id INTEGER PRIMARY KEY, language TEXT, canonical_form TEXT, "
        "pronunciation_ipa TEXT, merged_into_id INTEGER);"
        "CREATE TABLE reflex_etymon (reflex_id INTEGER, etymon_id INTEGER);"
    )
    conn.executemany(
        "INSERT INTO etymon VALUES (?,?,?,?,?)",
        [
            (1, "irish", "baile", None, None),  # target
            (2, "irish", "cnoc", "/knɔk/", None),  # has IPA → excluded
            (3, "irish", "a b", None, None),  # space → excluded
            (4, "irish", "*recon", None, None),  # star → excluded
            (5, "old-french", "ville", None, None),  # target (other lang)
            (6, "irish", "orphan", None, None),  # no reflex link → excluded
            (7, "irish", "tombst", None, 1),  # merged OCR-loser (tombstone) → excluded
        ],
    )
    conn.executemany(
        "INSERT INTO reflex_etymon VALUES (?,?)",
        [
            (10, 1),
            (11, 2),
            (12, 3),
            (13, 4),
            (14, 5),
            (15, 7),
        ],  # row 7 HAS a reflex but is a tombstone
    )
    conn.commit()
    conn.close()
    assert select_targets(str(db), ("irish", "old-french")) == [
        ("irish", "baile"),
        ("old-french", "ville"),
    ]
    assert select_targets(str(db), ("old-french",)) == [("old-french", "ville")]
