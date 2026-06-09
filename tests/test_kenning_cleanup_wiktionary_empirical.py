"""Tests for ``wyrd kenning lexicon cleanup-wiktionary-empirical`` (wyrd-a106).

The command purges etymons cited ONLY by wiktionary-empirical that are either
derivative-gloss-only or modern-english-only, while keeping semantic+historical
ones and any etymon with an additional scholarly source. These pin the
classification buckets, the NO-ACTION-FK pre-handling, and the dry-run gate of
the wyrd-8uvi-extracted helpers + the thin command.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli.lexicon.cleanup_wiktionary_empirical import (
    _collect_wikt_only_cleanup,
    _delete_cleanup_etymons,
    lexicon_cleanup_wiktionary_empirical,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


def _seed(db_path: Path) -> dict[str, int]:
    """Five wikt-only etymons (one per classification outcome) + a child whose
    lemma_id points at a deleted parent + a toponym element citing a deleted
    etymon (the two NO-ACTION FKs). Returns the etymon ids by role."""
    init_schema(db_path)
    c = sqlite3.connect(db_path)
    for s in ("wiktionary-empirical", "mawer_1920"):
        c.execute("INSERT INTO source(id, title) VALUES (?, ?)", (s, s))

    def ety(form: str, lang: str) -> int:
        return c.execute(
            "INSERT INTO etymon(canonical_form, language) VALUES (?, ?)", (form, lang)
        ).lastrowid

    def gloss(eid: int, g: str) -> None:
        c.execute("INSERT INTO etymon_gloss(etymon_id, gloss) VALUES (?, ?)", (eid, g))

    def cite(eid: int, s: str) -> None:
        c.execute("INSERT INTO etymon_citation(etymon_id, source_id) VALUES (?, ?)", (eid, s))

    ids: dict[str, int] = {}
    # derivative-only: wikt-only + every gloss derivative -> DELETE
    ids["deriv"] = ety("foom", "old-english")
    gloss(ids["deriv"], "plural of foo")
    cite(ids["deriv"], "wiktionary-empirical")
    # modern-only: wikt-only + modern language + semantic gloss -> DELETE
    ids["modern"] = ety("bar", "modern-english")
    gloss(ids["modern"], "a tavern")
    cite(ids["modern"], "wiktionary-empirical")
    # keep: wikt-only + historical language + semantic gloss -> STAY
    ids["keep"] = ety("burh", "old-english")
    gloss(ids["keep"], "fortified place")
    cite(ids["keep"], "wiktionary-empirical")
    # glossless wikt-only -> DELETE (derivative bucket)
    ids["nogloss"] = ety("zzz", "old-english")
    cite(ids["nogloss"], "wiktionary-empirical")
    # has a scholarly source too -> excluded up front (stays regardless)
    ids["scholar"] = ety("tun", "old-english")
    gloss(ids["scholar"], "enclosure")
    cite(ids["scholar"], "wiktionary-empirical")
    cite(ids["scholar"], "mawer_1920")
    # NO-ACTION FK targets: a child lemma-linked to 'deriv'; an element citing 'modern'.
    ids["child"] = ety("fooms", "old-english")
    c.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (ids["deriv"], ids["child"]))
    c.execute("INSERT INTO toponym(id, modern_name) VALUES (1, 'T')")
    c.execute(
        "INSERT INTO toponym_etymology(id, toponym_id, source_id) VALUES (1, 1, 'mawer_1920')"
    )
    c.execute(
        "INSERT INTO toponym_etymology_element(toponym_etymology_id, ordinal, etymon_id) "
        "VALUES (1, 0, ?)",
        (ids["modern"],),
    )
    c.commit()
    c.close()
    return ids


def test_collect_classifies_wikt_only_buckets(tmp_path: Path) -> None:
    ids = _seed(tmp_path / "lex.db")
    with LexiconDB(tmp_path / "lex.db") as db:
        wikt_only, derivative_only, modern_only = _collect_wikt_only_cleanup(db)
    wikt_ids = set(wikt_only)
    # 'scholar' has a second (non-wiktionary) source -> excluded from wikt-only.
    assert ids["scholar"] not in wikt_ids
    assert {ids["deriv"], ids["modern"], ids["keep"], ids["nogloss"]} <= wikt_ids
    deriv_ids = {r[0] for r in derivative_only}
    modern_ids = {r[0] for r in modern_only}
    assert deriv_ids == {ids["deriv"], ids["nogloss"]}  # derivative gloss + glossless
    assert modern_ids == {ids["modern"]}  # modern lang, non-derivative gloss
    # 'keep' (semantic gloss + historical lang) is in neither delete bucket.
    assert ids["keep"] not in deriv_ids and ids["keep"] not in modern_ids


def test_delete_cleanup_handles_no_action_fks(tmp_path: Path) -> None:
    ids = _seed(tmp_path / "lex.db")
    with LexiconDB(tmp_path / "lex.db") as db:
        # Disable FK enforcement so this pins the helper's EXPLICIT pre-handle
        # statements, not the schema's ON DELETE cascades — i.e. removing the
        # helper's `UPDATE ... lemma_id = NULL` / `DELETE ... element` lines
        # would now leave a dangling lemma_id + orphan element and fail below.
        db.conn.execute("PRAGMA foreign_keys = OFF")
        _delete_cleanup_etymons(db, [ids["deriv"], ids["modern"], ids["nogloss"]])
        db.commit()
        remaining = {r[0] for r in db.conn.execute("SELECT id FROM etymon")}
        child_lemma = db.conn.execute(
            "SELECT lemma_id FROM etymon WHERE id = ?", (ids["child"],)
        ).fetchone()[0]
        elems = db.conn.execute("SELECT COUNT(*) FROM toponym_etymology_element").fetchone()[0]
    assert remaining == {ids["keep"], ids["scholar"], ids["child"]}
    assert child_lemma is None  # NO-ACTION self-FK nulled, child survives
    assert elems == 0  # element citing the deleted 'modern' was removed


def test_command_dry_run_writes_nothing(tmp_path: Path) -> None:
    db_path = tmp_path / "lex.db"
    _seed(db_path)
    result = CliRunner().invoke(lexicon_cleanup_wiktionary_empirical, ["--db", str(db_path)])
    assert result.exit_code == 0
    with LexiconDB(db_path) as db:
        n = db.conn.execute("SELECT COUNT(*) FROM etymon").fetchone()[0]
    assert n == 6  # nothing deleted without --apply


def test_command_apply_deletes_garbage_keeps_rest(tmp_path: Path) -> None:
    db_path = tmp_path / "lex.db"
    ids = _seed(db_path)
    result = CliRunner().invoke(
        lexicon_cleanup_wiktionary_empirical, ["--db", str(db_path), "--apply"]
    )
    assert result.exit_code == 0
    with LexiconDB(db_path) as db:
        remaining = {r[0] for r in db.conn.execute("SELECT id FROM etymon")}
    assert remaining == {ids["keep"], ids["scholar"], ids["child"]}
