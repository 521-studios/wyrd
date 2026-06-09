"""wyrd-vm8t: positional OE h-rule (G2P) + the IPA-backfill enrichment pass."""

from __future__ import annotations

import sqlite3
import types

from wyrd.generators.kenning.lexicon.pronunciation_backfill import (
    _clean_form,
    _initial_h_is_x,
    _is_real_ipa,
    collect_pronunciation,
    derive_pronunciation_ipa,
    surface_ipa,
)
from wyrd.generators.kenning.registers.phonology import to_ipa


def _fake_db(conn):
    return types.SimpleNamespace(conn=conn)


# --- the positional h-rule in the G2P -------------------------------------
def test_oe_h_is_positional():
    # onset h (before a vowel) → [h]
    assert to_ipa("hām", "old-english") == "/hɑːm/"
    assert to_ipa("hill", "old-english") == "/hɪll/"
    # coda h (before a consonant or word-final) → velar fricative [x]
    assert to_ipa("burh", "old-english") == "/bʊrx/"
    assert to_ipa("niht", "old-english") == "/nɪxt/"
    # both in one word: onset [h], coda [x]
    assert to_ipa("heah", "old-english") == "/hɛɑx/"
    # hl/hr/hn/hw clusters are matched first — NOT the single-h rule
    assert to_ipa("hlāw", "old-english") == "/l̥ɑːw/"
    assert to_ipa("hwæt", "old-english") == "/ʍæt/"


def test_on_h_stays_plain():
    # Old Norse has no coda-h rule — h is always /h/
    assert to_ipa("hof", "old-norse") == "/hɔf/"


# --- backfill helpers ------------------------------------------------------
def test_clean_form():
    assert _clean_form("*hristian") == "hristian"  # reconstruction marker stripped
    assert _clean_form("hām") == "hām"
    assert _clean_form("a b") is None  # space → not a single word
    assert _clean_form("h2o") is None  # digit
    assert _clean_form("  ") is None


def test_is_real_ipa_and_initial_h():
    assert _is_real_ipa("/tʊn/", "tun") is True
    assert _is_real_ipa("/crom/", "crom") is False  # passthrough (celtic-style)
    assert _initial_h_is_x("ham", "/xɑːm/") is True  # onset-h wrongly /x/
    assert _initial_h_is_x("ham", "/hɑːm/") is False  # already /h/
    assert _initial_h_is_x("burh", "/bʊrx/") is False  # coda h, not onset


# --- the enrichment pass ---------------------------------------------------
def _etymon_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE etymon (id INTEGER PRIMARY KEY, language TEXT, "
        "canonical_form TEXT, pronunciation_ipa TEXT)"
    )
    rows = [
        (1, "old-norse", "haugr", None),  # FILL (ON gap; au→aʊ converts)
        (2, "old-english", "ham", "/xɑːm/"),  # FIX initial-h
        (3, "old-english", "tun", "/tʊn/"),  # KEEP (already good)
        (4, "old-english", "burh", "/bʊrx/"),  # KEEP (coda h, not onset)
        (5, "celtic", "bedd", None),  # FILL via welsh table (Brittonic)
        (6, "old-english", "*x y", None),  # unclean → skipped
        (7, "irish", "baile", None),  # Goidelic → not in _G2P_LANG → not selected
    ]
    conn.executemany("INSERT INTO etymon VALUES (?,?,?,?)", rows)
    conn.commit()
    return conn


def test_derive_pronunciation_ipa_fill_and_fix():
    conn = _etymon_db()
    res = derive_pronunciation_ipa(_fake_db(conn), apply=True)
    assert res["filled"] == 2  # haugr (ON) + bedd (celtic via welsh)
    assert res["fixed_initial_h"] == 1  # ham
    assert res["kept_existing"] == 2  # tun + burh
    assert res["skipped_unclean"] == 1  # "*x y"
    ipa = dict(conn.execute("SELECT canonical_form, pronunciation_ipa FROM etymon"))
    assert ipa["haugr"] == "/haʊgr/"
    assert ipa["bedd"] == "/bɛð/"  # celtic filled via welsh table
    assert ipa["ham"] == "/hɑm/"  # fixed: /x/ → /h/
    assert ipa["tun"] == "/tʊn/"  # untouched
    assert ipa["baile"] is None  # irish (Goidelic) not selected


def test_dry_run_writes_nothing():
    conn = _etymon_db()
    res = derive_pronunciation_ipa(_fake_db(conn), apply=False)
    assert res["filled"] == 2 and res["fixed_initial_h"] == 1
    assert conn.execute("SELECT pronunciation_ipa FROM etymon WHERE id=1").fetchone()[0] is None


# --- export-side surface fallback (Loop 4) ---------------------------------
def test_surface_ipa():
    # table language (bundle underscore name) → G2P IPA
    assert surface_ipa("burh", "old_english") == "/bʊrx/"  # coda-h via G2P
    assert surface_ipa("bedd", "celtic") == "/bɛð/"  # celtic → welsh table
    # non-table bundle field → None (LLM tier handles those, via etymon column)
    assert surface_ipa("baile", "irish") is None
    assert surface_ipa("ville", "old_french") is None
    # unclean form → None
    assert surface_ipa("*x y", "old_english") is None


# --- the LLM jsonl tier ----------------------------------------------------
def test_collect_pronunciation_confidence_filter(tmp_path):
    p = tmp_path / "_pronunciation.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"language":"irish","form":"baile","ipa":"/bˠalʲə/","confidence":"high"}',
                '{"language":"irish","form":"cnoc","ipa":"/knɔk/","confidence":"medium"}',
                '{"language":"irish","form":"aint","ipa":"/iːnt/","confidence":"low"}',  # dropped
                '{"language":"irish","form":"x","confidence":"high"}',  # no ipa → dropped
                '{"language":"old-french","form":"trop","ipa":"/troP/","confidence":"high"}',  # ASCII artifact
                '{"language":"old-french","form":"plume","ipa":"/pλym/","confidence":"high"}',  # Greek artifact
                "not json",  # tolerated
            ]
        ),
        encoding="utf-8",
    )
    state = collect_pronunciation(str(p))
    assert state == {("irish", "baile"): "/bˠalʲə/", ("irish", "cnoc"): "/knɔk/"}
    assert collect_pronunciation(str(tmp_path / "missing.jsonl")) == {}


def test_run_full_enrichment_threads_pronunciation_state(tmp_path):
    """The pronunciation_state kwarg flows through run_full_enrichment into the
    pass and fills a no-G2P-table etymon (mirrors the sibling thread-through
    tests for addition/suppression/split state)."""
    from wyrd.generators.kenning.enrichment import run_full_enrichment
    from wyrd.generators.kenning.lexicon import LexiconDB, init_schema

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO etymon (canonical_form, language) VALUES ('baile', 'irish')")
    conn.commit()
    conn.close()

    with LexiconDB(db_path) as db:
        result = run_full_enrichment(
            db, apply=True, pronunciation_state={("irish", "baile"): "/bˠalʲə/"}
        )
    assert "derive-pronunciation-ipa" in result["order"]
    assert result["pronunciation_ipa"]["filled_llm"] == 1
    check = sqlite3.connect(db_path)
    assert (
        check.execute(
            "SELECT pronunciation_ipa FROM etymon WHERE canonical_form='baile'"
        ).fetchone()[0]
        == "/bˠalʲə/"
    )
    check.close()


def test_llm_state_fills_no_table_languages_gaps_only():
    conn = _etymon_db()  # has irish "baile" with no IPA
    conn.execute("INSERT INTO etymon VALUES (8,'irish','cnoc','/EXISTING/')")
    llm = {("irish", "baile"): "/bˠalʲə/", ("irish", "cnoc"): "/SHOULD-NOT-OVERRIDE/"}
    res = derive_pronunciation_ipa(_fake_db(conn), apply=True, llm_state=llm)
    assert res["filled_llm"] == 1  # only baile (cnoc already had IPA)
    ipa = dict(conn.execute("SELECT canonical_form, pronunciation_ipa FROM etymon"))
    assert ipa["baile"] == "/bˠalʲə/"  # filled from LLM
    assert ipa["cnoc"] == "/EXISTING/"  # never overridden
