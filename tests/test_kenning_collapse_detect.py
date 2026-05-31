"""Deterministic collapse detection — wyrd-y651 P2b."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from wyrd.generators.kenning.lexicon import init_schema
from wyrd.generators.kenning.lexicon.collapse_detect import (
    POINTER_PARSE_METHOD,
    VARIANT_GLOSS_METHOD,
    detect_deterministic_collapses,
)


def _conn(db_path: Path) -> sqlite3.Connection:
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO source (id, title) VALUES ('wikt', 'Wiktionary')")
    return conn


def _ety(conn, eid, form, glosses):
    conn.execute(
        "INSERT INTO etymon (id, canonical_form, language) VALUES (?, ?, 'old-english')",
        (eid, form),
    )
    for g in glosses:
        conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, g))


def _variant(conn, lemma_id, form, klass="alternative"):
    conn.execute(
        "INSERT INTO etymon_variant (etymon_id, form, variant_class, source_id) "
        "VALUES (?, ?, ?, 'wikt')",
        (lemma_id, form, klass),
    )


def test_variant_gloss_overlap_detected(tmp_path):
    """A variant-doubling that shares a gloss with its parent → collapse."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "burh", ["fort", "a stronghold"])  # doubling, shares 'fort'
    _ety(conn, 2, "burg", ["fort", "borough"])  # the lemma
    _variant(conn, 2, "burh")  # burh registered as a variant of burg
    rows = detect_deterministic_collapses(conn)
    assert {r["ref"]: r["into"] for r in rows} == {"old-english:burh": "old-english:burg"}
    assert rows[0]["method"] == VARIANT_GLOSS_METHOD


def test_variant_without_shared_gloss_left_for_llm(tmp_path):
    """Variant-doubling but NO shared gloss → NOT deterministic (P3's job)."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "fen", ["marsh"])
    _ety(conn, 2, "fenn", ["a different sense entirely"])
    _variant(conn, 2, "fen")
    assert detect_deterministic_collapses(conn) == []


def test_pure_pointer_resolved_detected(tmp_path):
    """A pure form-of etymon whose pointer names a resolvable lemma."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "Hea", ["h-prothesized form of ea"])  # pure pointer
    _ety(conn, 2, "ea", ["river", "stream"])  # the target lemma
    rows = detect_deterministic_collapses(conn)
    assert {r["ref"]: r["into"] for r in rows} == {"old-english:Hea": "old-english:ea"}
    assert rows[0]["method"] == POINTER_PARSE_METHOD


def test_pointer_plus_real_gloss_left_for_llm(tmp_path):
    """A pointer + a real gloss is NOT pure form-of → left for P3."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "bil", ["alternative form of bill", "a billhook"])  # mixed
    _ety(conn, 2, "bill", ["a sword"])
    assert detect_deterministic_collapses(conn) == []


def test_form_of_mid_sentence_not_a_pointer(tmp_path):
    """A real definition that merely CONTAINS 'form of' deep in prose is
    NOT a form-of pointer (dreng: '...a particular form of free tenure')."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "dreng", ["A man holding land by a particular form of free tenure."])
    _ety(conn, 2, "free", ["unbound"])
    assert detect_deterministic_collapses(conn) == []


def test_pointer_to_missing_target_skipped(tmp_path):
    """Pure pointer but the named target doesn't exist → skipped."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "x", ["alternative form of ghostword"])
    assert detect_deterministic_collapses(conn) == []


def test_bidirectional_variants_dont_cycle(tmp_path):
    """wode↔wood mutually registered as variants → only ONE direction
    emitted (no merged_into_id cycle)."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "wode", ["wood"])
    _ety(conn, 2, "wood", ["wood"])
    _variant(conn, 2, "wode")  # wode is a variant of wood
    _variant(conn, 1, "wood")  # AND wood is a variant of wode
    rows = detect_deterministic_collapses(conn)
    assert len(rows) == 1
    # deterministic: (ref, into)-sorted, so 'wode'->'wood' survives
    assert rows[0]["ref"] == "old-english:wode"
    assert rows[0]["into"] == "old-english:wood"


def test_chain_flattened_to_terminal_survivor(tmp_path):
    """wella -> well -> wiell collapses flatten so both point straight at
    wiell (no apply-order-fragile intermediate tombstone)."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "wella", ["spring"])
    _ety(conn, 2, "well", ["spring"])
    _ety(conn, 3, "wiell", ["spring"])
    _variant(conn, 2, "wella")  # wella is a variant of well
    _variant(conn, 3, "well")  # well is a variant of wiell
    rows = detect_deterministic_collapses(conn)
    intos = {r["ref"]: r["into"] for r in rows}
    assert intos == {
        "old-english:wella": "old-english:wiell",  # re-pointed past `well`
        "old-english:well": "old-english:wiell",
    }


def test_merged_rows_excluded(tmp_path):
    """Already-tombstoned etymons are not re-detected."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "burh", ["fort"])
    _ety(conn, 2, "burg", ["fort"])
    _variant(conn, 2, "burh")
    conn.execute("UPDATE etymon SET merged_into_id = 2 WHERE id = 1")
    assert detect_deterministic_collapses(conn) == []
