"""Tests for ``recompute_reflex_productivity`` (wyrd-14p).

Pins the per-reflex productivity-count derivation from
``toponym_etymology_element`` joins, including position-matching
semantics, idempotence, and the bulk-zero-then-recompute contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.reflex_productivity import (
    recompute_reflex_productivity,
)
from wyrd.generators.kenning.lexicon.schema import init_schema


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def _seed_corpus(db: LexiconDB) -> dict[str, int]:
    """Build a minimal corpus: 2 etymons, 3 reflexes, 3 toponyms.

    Layout:
      etymon ham: -ham reflex (post) — used by 2 toponyms (Bingham, Oldham)
      etymon ac:  Ac- reflex (pre)   — used by 1 toponym (Acton)
      etymon ton: -ton reflex (post) — used by 2 toponyms (Acton, Brighton)

    Toponyms:
      Acton    = [Ac (ordinal 0, pre), ton (ordinal 1, post)] — 2 elements
      Bingham  = [bing (ordinal 0, pre), ham (ordinal 1, post)] — 2 elements
      Oldham   = [Old (ordinal 0, pre), ham (ordinal 1, post)] — 2 elements
      Brighton = [bright (ordinal 0, pre), ton (ordinal 1, post)] — 2 elements

    Returns a dict of reflex names → reflex_id for the assertions.
    """
    db.upsert_source(id="test-src", title="Test")
    db.commit()

    eth_ham = db.upsert_etymon("ham", "old-english")
    eth_ac = db.upsert_etymon("aecern", "old-english")
    eth_ton = db.upsert_etymon("tun", "old-english")
    # Filler etymons for the "bing", "Old", "bright" prefixes that don't
    # have reflexes wired — they exist as toponym_etymology_element rows
    # but do not contribute to productivity counts.
    eth_bing = db.upsert_etymon("bing", "old-english")
    eth_old = db.upsert_etymon("eald", "old-english")
    eth_bright = db.upsert_etymon("beorht", "old-english")

    r_ham_post = db.upsert_reflex("-ham", "post")
    r_ac_pre = db.upsert_reflex("Ac-", "pre")
    r_ton_post = db.upsert_reflex("-ton", "post")

    conn = db.conn
    conn.execute("INSERT OR IGNORE INTO reflex_etymon VALUES (?, ?)", (r_ham_post, eth_ham))
    conn.execute("INSERT OR IGNORE INTO reflex_etymon VALUES (?, ?)", (r_ac_pre, eth_ac))
    conn.execute("INSERT OR IGNORE INTO reflex_etymon VALUES (?, ?)", (r_ton_post, eth_ton))

    for name in ("Acton", "Bingham", "Oldham", "Brighton"):
        conn.execute(
            "INSERT INTO toponym (modern_name) VALUES (?)",
            (name,),
        )
    toponym_ids = {
        row[0]: row[1] for row in conn.execute("SELECT modern_name, id FROM toponym").fetchall()
    }

    def _add_breakdown(toponym_name: str, elements: list[tuple[int, int]]) -> None:
        """elements = list of (ordinal, etymon_id)"""
        cur = conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id) VALUES (?, ?)",
            (toponym_ids[toponym_name], "test-src"),
        )
        te_id = cur.lastrowid
        for ordinal, etymon_id in elements:
            conn.execute(
                "INSERT INTO toponym_etymology_element "
                "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
                (te_id, ordinal, etymon_id),
            )

    _add_breakdown("Acton", [(0, eth_ac), (1, eth_ton)])
    _add_breakdown("Bingham", [(0, eth_bing), (1, eth_ham)])
    _add_breakdown("Oldham", [(0, eth_old), (1, eth_ham)])
    _add_breakdown("Brighton", [(0, eth_bright), (1, eth_ton)])

    db.commit()
    return {"ham_post": r_ham_post, "ac_pre": r_ac_pre, "ton_post": r_ton_post}


def test_recompute_counts_distinct_toponyms_per_reflex(fresh_db: Path) -> None:
    """ham appears as post-element in 2 toponyms → productivity 2.
    ton appears as post-element in 2 toponyms → productivity 2.
    Ac- appears as pre-element in 1 toponym → productivity 1."""
    with LexiconDB(fresh_db) as db:
        reflex_ids = _seed_corpus(db)
        stats = recompute_reflex_productivity(db)
        rows = {
            row[0]: row[1]
            for row in db.conn.execute("SELECT id, productivity FROM reflex").fetchall()
        }
    assert rows[reflex_ids["ham_post"]] == 2
    assert rows[reflex_ids["ton_post"]] == 2
    assert rows[reflex_ids["ac_pre"]] == 1
    assert stats["updated"] == 3  # 3 reflexes saw corpus support
    assert stats["max_productivity"] == 2
    assert stats["sum_productivity"] == 5


def test_recompute_is_idempotent(fresh_db: Path) -> None:
    """Re-running on an unchanged corpus produces identical row values."""
    with LexiconDB(fresh_db) as db:
        _seed_corpus(db)
        first = recompute_reflex_productivity(db)
        before = sorted(
            (row[0], row[1])
            for row in db.conn.execute("SELECT id, productivity FROM reflex").fetchall()
        )
        second = recompute_reflex_productivity(db)
        after = sorted(
            (row[0], row[1])
            for row in db.conn.execute("SELECT id, productivity FROM reflex").fetchall()
        )
    assert before == after
    assert first == second


def test_recompute_zeros_reflexes_without_corpus_support(fresh_db: Path) -> None:
    """A reflex whose etymon has no toponym observations reads 0, even
    if it carried a stale non-zero count from a previous recompute on a
    larger corpus. Pins the bulk-zero-then-recompute contract."""
    with LexiconDB(fresh_db) as db:
        _seed_corpus(db)
        # Add an unused reflex for an etymon that no breakdown touches.
        orphan_etymon = db.upsert_etymon("orphan", "old-english")
        orphan_reflex = db.upsert_reflex("-orph", "post")
        db.conn.execute("INSERT INTO reflex_etymon VALUES (?, ?)", (orphan_reflex, orphan_etymon))
        # Manually seed a stale productivity value to verify it gets reset.
        db.conn.execute("UPDATE reflex SET productivity = 99 WHERE id = ?", (orphan_reflex,))
        db.commit()
        recompute_reflex_productivity(db)
        productivity = db.conn.execute(
            "SELECT productivity FROM reflex WHERE id = ?", (orphan_reflex,)
        ).fetchone()[0]
    assert productivity == 0


def test_recompute_respects_position_match(fresh_db: Path) -> None:
    """A reflex registered with position='pre' does NOT pick up
    toponyms where the etymon appears in a post or inner element,
    even if the etymon_id matches. Position is part of the join key."""
    with LexiconDB(fresh_db) as db:
        _seed_corpus(db)
        # Add a misregistered reflex: ham as PRE (the corpus has it
        # in post-position only). Productivity should be 0.
        eth_ham_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'ham'"
        ).fetchone()[0]
        mis_pre = db.upsert_reflex("ham-", "pre")
        db.conn.execute("INSERT INTO reflex_etymon VALUES (?, ?)", (mis_pre, eth_ham_id))
        db.commit()
        recompute_reflex_productivity(db)
        productivity = db.conn.execute(
            "SELECT productivity FROM reflex WHERE id = ?", (mis_pre,)
        ).fetchone()[0]
    assert productivity == 0


def test_recompute_single_element_breakdown_counts_as_pre(fresh_db: Path) -> None:
    """A single-element toponym contributes to its etymon's 'pre'
    reflex (matches the _position_label convention in
    empirical_priors). Pins the convention so the runtime weighting
    matches the priors-extraction position labeling."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        eth = db.upsert_etymon("solo", "old-english")
        r_pre = db.upsert_reflex("Solo-", "pre")
        r_post = db.upsert_reflex("-solo", "post")
        db.conn.execute("INSERT OR IGNORE INTO reflex_etymon VALUES (?, ?)", (r_pre, eth))
        db.conn.execute("INSERT OR IGNORE INTO reflex_etymon VALUES (?, ?)", (r_post, eth))
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("Solo",))
        toponym_id = cur.lastrowid
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id) VALUES (?, ?)",
            (toponym_id, "test-src"),
        )
        te_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
            (te_id, 0, eth),
        )
        db.commit()
        recompute_reflex_productivity(db)
        rows = {
            row[0]: row[1]
            for row in db.conn.execute("SELECT id, productivity FROM reflex").fetchall()
        }
    assert rows[r_pre] == 1  # single-element → counts as pre
    assert rows[r_post] == 0  # post sibling does not pick up the solo
