"""wyrd-u9k6: grounded element-gloss backfill — derive, collect, apply, adjudicate."""

from __future__ import annotations

import json
import sqlite3
import types

from wyrd.generators.kenning.lexicon.element_gloss_backfill import (
    _confidence,
    _has_vowel,
    _merge_into_ledger,
    _normform,
    _position,
    accept_adjudication,
    apply_element_glosses,
    build_adjudication_user,
    collect_element_glosses,
    derive_element_glosses,
    load_toponym_lowers,
    mine_to_jsonl,
    toponym_context,
    write_jsonl,
)


def _fake_db(conn: sqlite3.Connection):
    return types.SimpleNamespace(conn=conn)


def _grounding_db(path: str = ":memory:") -> sqlite3.Connection:
    """A minimal lexicon with one glossed suffix element (-ick → OE wīc) plus
    three toponyms whose final element is that etymon, and the orphan reflex.
    Also a prefix element (Chur- → OE cirice) for the prefix-path test."""
    conn = sqlite3.connect(path)
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
    conn.execute("INSERT INTO etymon VALUES (3, 'old-english', 'cirice')")  # prefix element
    conn.execute("INSERT INTO etymon_gloss VALUES (3, 'a church')")
    # Hardwick / Chiswick / Berwick: each is (first=hard, last=wīc)
    for tid, name in [(1, "Hardwick"), (2, "Chiswick"), (3, "Berwick")]:
        conn.execute("INSERT INTO toponym VALUES (?,?)", (tid, name))
        conn.execute("INSERT INTO toponym_etymology VALUES (?,?)", (tid, tid))
        conn.execute("INSERT INTO toponym_etymology_element VALUES (?,0,2)", (tid,))
        conn.execute("INSERT INTO toponym_etymology_element VALUES (?,1,1)", (tid,))
    # Churchill / Churchdown: each is (first=cirice, last=...)
    for tid, name in [(4, "Churchill"), (5, "Churchdown"), (6, "Churchend")]:
        conn.execute("INSERT INTO toponym VALUES (?,?)", (tid, name))
        conn.execute("INSERT INTO toponym_etymology VALUES (?,?)", (tid, tid))
        conn.execute("INSERT INTO toponym_etymology_element VALUES (?,0,3)", (tid,))
        conn.execute("INSERT INTO toponym_etymology_element VALUES (?,1,2)", (tid,))
    conn.execute("INSERT INTO reflex VALUES (10, '-ick', 'post')")
    conn.commit()
    return conn


def _census_db(path: str = ":memory:") -> sqlite3.Connection:
    """Census bundle with two unglossed english usages (-ick suffix, Chur- prefix)."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE proportions_usage (culture TEXT, usage_key TEXT, weight INTEGER);
        CREATE TABLE meaning (usage_key TEXT PRIMARY KEY, data BLOB);
        """
    )
    empty = json.dumps({"entries": [{"meaning": []}]})
    conn.execute("INSERT INTO proportions_usage VALUES ('english', '-ick', 50)")
    conn.execute("INSERT INTO proportions_usage VALUES ('english', 'Chur-', 9)")
    conn.execute("INSERT INTO meaning VALUES ('-ick', ?)", (empty,))  # empty meaning → unglossed
    conn.execute("INSERT INTO meaning VALUES ('Chur-', ?)", (empty,))
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
    rows = {
        r["surface"]: r for r in derive_element_glosses(_census_db(), _grounding_db(), "english")
    }
    # suffix path: -ick → last element (wīc)
    ick = rows["-ick"]
    assert ick["is_element"] is True  # 3/3 support → high
    top = ick["candidates"][0]
    # candidate form is the normalized vote key (deaccented), not the raw "wīc"
    assert top["form"] == "wic" and top["language"] == "old-english"
    assert top["support"] == 3 and "dwelling" in top["glosses"]
    # prefix path: Chur- → FIRST element (cirice), not the last
    chur = rows["Chur-"]
    assert chur["candidates"][0]["form"] == "cirice"
    assert "a church" in chur["candidates"][0]["glosses"]


def test_derive_no_match_yields_no_candidates():
    census = sqlite3.connect(":memory:")
    census.executescript(
        "CREATE TABLE proportions_usage (culture TEXT, usage_key TEXT, weight INTEGER);"
        "CREATE TABLE meaning (usage_key TEXT PRIMARY KEY, data BLOB);"
    )
    census.execute("INSERT INTO proportions_usage VALUES ('english', '-zz', 3)")
    census.commit()
    rows = derive_element_glosses(census, _grounding_db(), "english")
    assert rows[0]["is_element"] is False and rows[0]["candidates"] == []


def test_confidence_bands():
    assert _confidence(8, 0.45) == "high"
    assert _confidence(3, 0.25) == "medium"
    assert _confidence(2, 0.9) == "low"  # support below 3
    assert _confidence(20, 0.20) == "low"  # share below 0.25


def test_apply_rejection_branches():
    conn = _grounding_db()
    state = [
        {
            "surface": "-ick",
            "position": "suffix",
            "language": "old-english",
            "form": "wic",
        },  # links
        {
            "surface": "-ick",
            "position": "suffix",
            "language": "old-english",
            "form": "nope",
        },  # no_etymon
        {
            "surface": "-zzz",
            "position": "suffix",
            "language": "old-english",
            "form": "wic",
        },  # no_reflex
        {
            "surface": "Bourne",
            "position": "standalone",
            "language": "old-english",
            "form": "wic",
        },  # no_reflex (no pos)
    ]
    res = apply_element_glosses(_fake_db(conn), state, apply=True)
    assert res["linked_surfaces"] == 1
    assert res["no_etymon"] == 1
    assert res["no_reflex"] == 2


def test_mine_to_jsonl_and_load_toponym_lowers(tmp_path):
    census_path = str(tmp_path / "census.db")
    db_path = str(tmp_path / "lex.db")
    _census_db(census_path).close()
    _grounding_db(db_path).close()
    out = str(tmp_path / "_element_glosses.jsonl")
    result = mine_to_jsonl(census_path, db_path, out)  # fresh ledger → write == derived
    surfaces = {r["surface"] for r in result.rows}
    assert "-ick" in surfaces and "Chur-" in surfaces
    assert result.new_added == len(result.rows) and result.existing_kept == 0
    with open(out, encoding="utf-8") as fh:
        written = [json.loads(line) for line in fh]
    assert {r["surface"] for r in written} == surfaces  # round-trips to disk

    lowers = load_toponym_lowers(db_path)
    assert ("Hardwick", "hardwick") in lowers and ("Churchill", "churchill") in lowers


def test_merge_into_ledger_keeps_existing_and_appends_new(tmp_path):
    """wyrd-5f5w: merging unions by (surface, position); existing rows win on
    conflict and are NEVER dropped — only genuinely-new keys append."""
    path = str(tmp_path / "_element_glosses.jsonl")
    write_jsonl(
        [
            {
                "surface": "-ick",
                "position": "suffix",
                "is_element": True,
                "candidates": [{"form": "wic"}],
            },
            {
                "surface": "-old",
                "position": "suffix",
                "is_element": True,
                "candidates": [{"form": "ald"}],
            },
        ],
        path,
    )
    new_rows = [
        # conflict on (-ick, suffix): existing must win (don't rewrite established gloss)
        {
            "surface": "-ick",
            "position": "suffix",
            "is_element": True,
            "candidates": [{"form": "DIFFERENT"}],
        },
        {
            "surface": "-new",
            "position": "suffix",
            "is_element": True,
            "candidates": [{"form": "niwe"}],
        },
    ]
    merged, new_added = _merge_into_ledger(new_rows, path)
    by_surface = {r["surface"]: r for r in merged}
    assert set(by_surface) == {"-ick", "-old", "-new"}  # -old not in the census, NOT dropped
    assert new_added == 1  # only -new
    assert by_surface["-ick"]["candidates"][0]["form"] == "wic"  # existing won the conflict


def test_merge_into_ledger_no_existing_file_is_passthrough(tmp_path):
    new_rows = [{"surface": "-a", "position": "suffix"}]
    merged, new_added = _merge_into_ledger(new_rows, str(tmp_path / "absent.jsonl"))
    assert merged == new_rows and new_added == 1


def test_mine_to_jsonl_merges_by_default_overwrite_replaces(tmp_path):
    """wyrd-5f5w: the footgun fix end-to-end. A fresh (partial) census MERGES —
    a pre-existing surface the census doesn't reproduce survives. --overwrite is
    the deliberate full-replace escape hatch."""
    census_path = str(tmp_path / "census.db")
    db_path = str(tmp_path / "lex.db")
    _census_db(census_path).close()
    _grounding_db(db_path).close()
    out = str(tmp_path / "_element_glosses.jsonl")
    # Seed an accumulated row the census will NOT re-derive.
    write_jsonl(
        [
            {
                "surface": "-zzz",
                "position": "suffix",
                "is_element": True,
                "candidates": [{"form": "zzz"}],
            }
        ],
        out,
    )

    merged = mine_to_jsonl(census_path, db_path, out)  # default merge
    assert merged.overwritten is False
    assert "-zzz" in {r["surface"] for r in merged.written}  # NOT dropped (the fix)
    assert merged.existing_kept >= 1

    replaced = mine_to_jsonl(census_path, db_path, out, overwrite=True)  # escape hatch
    assert replaced.overwritten is True
    assert "-zzz" not in {r["surface"] for r in replaced.written}  # deliberately gone
    assert replaced.existing_kept == 0


def test_run_full_enrichment_threads_element_gloss_state(tmp_path):
    from wyrd.generators.kenning.enrichment import (
        format_enrichment_run,
        run_full_enrichment,
    )
    from wyrd.generators.kenning.lexicon import LexiconDB, init_schema

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    eid = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('wic', 'old-english')"
    ).lastrowid
    conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, 'dwelling')", (eid,))
    conn.execute("INSERT INTO reflex (surface_form, position) VALUES ('-ick', 'post')")
    conn.commit()
    conn.close()

    state = [{"surface": "-ick", "position": "suffix", "language": "old-english", "form": "wic"}]
    with LexiconDB(db_path) as db:
        result = run_full_enrichment(
            db, apply=True, element_gloss_state=state, skip_l3_derivations=True
        )
    assert "apply-element-glosses" in result["order"]
    assert result["element_glosses"]["linked_surfaces"] == 1
    assert "Element-gloss backfill" in format_enrichment_run(result)


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
