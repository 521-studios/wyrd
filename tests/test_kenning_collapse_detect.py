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


def test_pointer_target_dashed_resolves_to_bare_lemma(tmp_path):
    """wyrd-aicu.8 (D45) read-path sweep: a pointer gloss naming a DASHED target
    ('form of -ton') de-dashes to resolve the BARE-stored lemma ('ton'), so the
    collapse is still detected (and the emitted 'into' ref is bare) after PR 1
    made etymon storage bare. Pre-sweep, '-ton' missed the bare row and the
    collapse was silently dropped."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "tun", ["alternative form of -ton"])  # pointer to a dashed target
    _ety(conn, 2, "ton", ["enclosure, farmstead"])  # bare-stored lemma
    rows = detect_deterministic_collapses(conn)
    assert {r["ref"]: r["into"] for r in rows} == {"old-english:tun": "old-english:ton"}
    assert rows[0]["method"] == POINTER_PARSE_METHOD


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


def test_is_form_of_pointer():
    from wyrd.generators.kenning.lexicon.collapse_detect import is_form_of_pointer

    assert is_form_of_pointer("alternative form of burg")
    assert is_form_of_pointer("h-prothesized form of ea")
    assert is_form_of_pointer("variant of X")
    # a real definition that merely contains the phrase mid-sentence is NOT
    assert not is_form_of_pointer("A man holding land by a particular form of free tenure.")
    assert not is_form_of_pointer("fort")


def test_cli_detect_collapses_appends_and_is_idempotent(tmp_path):
    """The detect-collapses CLI scans the DB, writes new collapse rows (plus the
    one synthetic 'source' row) to the ledger, and a re-run adds nothing —
    covers _resolve_lexicon_db / _load_recorded_collapse_refs / _append_collapse_rows."""
    import json

    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon.detect_collapses import lexicon_detect_collapses

    db = tmp_path / "lex.db"
    conn = _conn(db)
    _ety(conn, 1, "burh", ["fort", "a stronghold"])  # variant-doubling
    _ety(conn, 2, "burg", ["fort", "borough"])  # the lemma
    _variant(conn, 2, "burh")
    conn.commit()
    conn.close()

    ledger = tmp_path / "_collapses.jsonl"
    runner = CliRunner()
    r = runner.invoke(lexicon_detect_collapses, ["--db", str(db), "--collapse-file", str(ledger)])
    assert r.exit_code == 0, r.output

    rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert [row for row in rows if row.get("_type") == "source"][0]["ref"] == "collapse"
    collapses = [row for row in rows if row.get("_type") == "collapse"]
    assert {c["ref"]: c["into"] for c in collapses} == {"old-english:burh": "old-english:burg"}

    # Idempotent: a second run records nothing new and leaves the ledger unchanged.
    r2 = runner.invoke(lexicon_detect_collapses, ["--db", str(db), "--collapse-file", str(ledger)])
    assert r2.exit_code == 0
    assert "Nothing new to emit" in r2.output
    rows2 = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len([row for row in rows2 if row.get("_type") == "collapse"]) == 1


def test_cli_detect_collapses_missing_db_errors(tmp_path):
    """A non-existent DB raises a ClickException (exit 1) — _resolve_lexicon_db."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon.detect_collapses import lexicon_detect_collapses

    r = CliRunner().invoke(
        lexicon_detect_collapses,
        ["--db", str(tmp_path / "nope.db"), "--collapse-file", str(tmp_path / "x.jsonl")],
    )
    assert r.exit_code != 0
    assert "Lexicon DB not found" in r.output
