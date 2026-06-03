"""wyrd-rogd.15 — reflex-LINK op in the collapse applier.

A ``inherits`` row asserts that ``ref`` is a later-era reflex of an ancestor
(same morpheme across eras). Unlike the ``into`` fold, it adds an inheritance
descent edge and keeps BOTH etymons distinct, so the era forms survive for the
grid and the rollup follows the edge. Replayed from _collapses.jsonl on rebuild.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.enrichment import apply_collapses
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def test_reflex_link_adds_inheritance_edge_without_tombstoning(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        tun = db.upsert_etymon("tūn", "old-english", modifier_type="Habitative")
        ton = db.upsert_etymon("-ton", "modern-english")
        db.commit()
        state = {
            "modern-english:-ton": {"inherits": "old-english:tūn", "confidence": "high"},
        }
        counts = apply_collapses(db, state)
        assert counts["links_processed"] == 1
        edge = db.conn.execute(
            "SELECT edge_type FROM etymon_descent WHERE parent_id=? AND child_id=?",
            (tun, ton),
        ).fetchone()
        assert edge is not None and edge["edge_type"] == "inheritance"
        # the link keeps both etymons distinct — neither is tombstoned.
        for eid in (tun, ton):
            mi = db.conn.execute("SELECT merged_into_id FROM etymon WHERE id=?", (eid,)).fetchone()[0]
            assert mi is None


def test_reflex_link_is_idempotent(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        tun = db.upsert_etymon("tūn", "old-english", modifier_type="Habitative")
        ton = db.upsert_etymon("-ton", "modern-english")
        db.commit()
        state = {"modern-english:-ton": {"inherits": "old-english:tūn"}}
        apply_collapses(db, state)
        apply_collapses(db, state)  # replay
        n = db.conn.execute(
            "SELECT count(*) FROM etymon_descent WHERE parent_id=? AND child_id=?",
            (tun, ton),
        ).fetchone()[0]
        assert n == 1


def test_reflex_link_unresolved_and_self_are_skipped(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        tun = db.upsert_etymon("tūn", "old-english", modifier_type="Habitative")
        db.commit()
        # unresolved ancestor
        c1 = apply_collapses(db, {"old-english:tūn": {"inherits": "old-english:ghost"}})
        assert c1["unresolved_inherits"] == 1 and c1["links_processed"] == 0
        # self-link
        c2 = apply_collapses(db, {"old-english:tūn": {"inherits": "old-english:tūn"}})
        assert c2["self_link_skipped"] == 1
        assert db.conn.execute("SELECT count(*) FROM etymon_descent").fetchone()[0] == 0
        _ = tun


def test_fold_still_works_alongside_link(fresh_db: Path) -> None:
    # A mixed state (one fold + one link) — the fold tombstones, the link edges.
    with LexiconDB(fresh_db) as db:
        burg = db.upsert_etymon("burg", "old-english", modifier_type="Habitative")
        borough = db.upsert_etymon("borough", "old-english", modifier_type="Habitative")
        tun = db.upsert_etymon("tūn", "old-english", modifier_type="Habitative")
        ton = db.upsert_etymon("-ton", "modern-english")
        db.commit()
        counts = apply_collapses(
            db,
            {
                "old-english:borough": {"into": "old-english:burg", "variant_class": "alternative"},
                "modern-english:-ton": {"inherits": "old-english:tūn"},
            },
        )
        assert counts["collapses_processed"] == 1 and counts["links_processed"] == 1
        # fold tombstoned borough; link left tūn/-ton distinct + edged
        assert (
            db.conn.execute("SELECT merged_into_id FROM etymon WHERE id=?", (borough,)).fetchone()[0]
            == burg
        )
        assert (
            db.conn.execute("SELECT merged_into_id FROM etymon WHERE id=?", (ton,)).fetchone()[0]
            is None
        )
        assert db.conn.execute(
            "SELECT count(*) FROM etymon_descent WHERE parent_id=? AND child_id=?", (tun, ton)
        ).fetchone()[0] == 1
