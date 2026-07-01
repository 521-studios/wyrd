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
from wyrd.generators.kenning.lexicon.reflex_link_detect import _reflex_pair_eligible


def _e(lang: str) -> dict:
    return {"lang": lang, "form": "x"}


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # wyrd-s48q: latin's cells are classical … vulgar-latin … medieval …
        # renaissance, so `latin` (rank=classical) and `vulgar-latin` (the 200-700
        # stage between classical and medieval latin) cannot be ordered by a single
        # first-seen rank — the pair is ambiguous and must be skipped in BOTH
        # directions (a backwards reflex link corrupts; a missed one is harmless).
        ("latin", "vulgar-latin"),
        ("vulgar-latin", "latin"),
    ],
)
def test_reflex_pair_eligible_skips_straddling_language_pairs(a: str, b: str) -> None:
    assert _reflex_pair_eligible(_e(a), _e(b)) is None


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # Contiguous polychronic / distinct-stage same-family pairs stay eligible —
        # no intervening stage falls inside either language's cell span, so the
        # first-seen rank orders them correctly (the s48q guard must not over-reject).
        ("old-english", "middle-english"),
        ("old-english", "modern-english"),
    ],
)
def test_reflex_pair_eligible_keeps_orderable_same_family_pairs(a: str, b: str) -> None:
    assert _reflex_pair_eligible(_e(a), _e(b)) is not None


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def test_reflex_link_adds_inheritance_edge_without_tombstoning(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        tun = db.upsert_etymon("tūn", "old-english", modifier_type="Habitative")
        # Modern morpheme stored BARE (wyrd-aicu.8, D45) — position is the
        # separate axis, not a dash on the identity; refs use the bare form.
        ton = db.upsert_etymon("ton", "modern-english")
        db.commit()
        state = {
            "modern-english:ton": {"inherits": "old-english:tūn", "confidence": "high"},
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
            mi = db.conn.execute("SELECT merged_into_id FROM etymon WHERE id=?", (eid,)).fetchone()[
                0
            ]
            assert mi is None


def test_reflex_link_is_idempotent(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        tun = db.upsert_etymon("tūn", "old-english", modifier_type="Habitative")
        ton = db.upsert_etymon("ton", "modern-english")  # bare (wyrd-aicu.8, D45)
        db.commit()
        state = {"modern-english:ton": {"inherits": "old-english:tūn"}}
        apply_collapses(db, state)
        apply_collapses(db, state)  # replay
        n = db.conn.execute(
            "SELECT count(*) FROM etymon_descent WHERE parent_id=? AND child_id=?",
            (tun, ton),
        ).fetchone()[0]
        assert n == 1


def test_reflex_link_dashed_refs_resolve_to_bare_etymons(fresh_db: Path) -> None:
    """wyrd-aicu.8 (D45) read-path sweep: a still-dashed collapse ref
    ('modern-english:-ton' / 'old-english:-tūn') resolves to the BARE-stored
    etymon via _resolve_live_etymon's de-dash, so the inheritance edge is wired
    even when the L2 ref hasn't been re-dumped bare yet."""
    with LexiconDB(fresh_db) as db:
        tun = db.upsert_etymon("tūn", "old-english", modifier_type="Habitative")
        ton = db.upsert_etymon("ton", "modern-english")  # stored bare
        db.commit()
        # BOTH refs carry an affix dash the live DB no longer stores.
        state = {"modern-english:-ton": {"inherits": "old-english:-tūn"}}
        counts = apply_collapses(db, state)
        assert counts["links_processed"] == 1
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
        # unresolved CHILD (from_ref) on a link row — distinct code path from
        # the fold's unresolved_from; the ancestor exists but the child doesn't.
        c3 = apply_collapses(db, {"old-english:ghostchild": {"inherits": "old-english:tūn"}})
        assert c3["unresolved_from"] == 1 and c3["links_processed"] == 0
        assert db.conn.execute("SELECT count(*) FROM etymon_descent").fetchone()[0] == 0
        _ = tun


def test_reflex_link_empty_inherits_is_rejection_not_fold(fresh_db: Path) -> None:
    """A row carrying ``inherits: ""`` (a recorded LLM rejection / revert) is a
    no-op counted as link_rejections — it MUST NOT fall through to the ``into``
    fold (no tombstone, no edge, not even empty_into_skipped). Pins the
    dispatch short-circuit the wyrd-8uvi C901 split moved into a returning
    helper (_apply_reflex_link)."""
    with LexiconDB(fresh_db) as db:
        tun = db.upsert_etymon("tūn", "old-english", modifier_type="Habitative")
        db.commit()
        counts = apply_collapses(db, {"old-english:tūn": {"inherits": ""}})
        assert counts["link_rejections"] == 1
        assert counts["links_processed"] == 0
        assert counts["collapses_processed"] == 0
        assert counts["empty_into_skipped"] == 0  # NOT treated as an empty fold
        # no edge written, and tūn is NOT tombstoned (no fold fall-through)
        assert db.conn.execute("SELECT count(*) FROM etymon_descent").fetchone()[0] == 0
        assert (
            db.conn.execute("SELECT merged_into_id FROM etymon WHERE id=?", (tun,)).fetchone()[0]
            is None
        )


def test_fold_still_works_alongside_link(fresh_db: Path) -> None:
    # A mixed state (one fold + one link) — the fold tombstones, the link edges.
    with LexiconDB(fresh_db) as db:
        burg = db.upsert_etymon("burg", "old-english", modifier_type="Habitative")
        borough = db.upsert_etymon("borough", "old-english", modifier_type="Habitative")
        tun = db.upsert_etymon("tūn", "old-english", modifier_type="Habitative")
        ton = db.upsert_etymon("ton", "modern-english")  # bare (wyrd-aicu.8, D45)
        db.commit()
        counts = apply_collapses(
            db,
            {
                "old-english:borough": {"into": "old-english:burg", "variant_class": "alternative"},
                "modern-english:ton": {"inherits": "old-english:tūn"},
            },
        )
        assert counts["collapses_processed"] == 1 and counts["links_processed"] == 1
        # fold tombstoned borough; link left tūn/-ton distinct + edged
        assert (
            db.conn.execute("SELECT merged_into_id FROM etymon WHERE id=?", (borough,)).fetchone()[
                0
            ]
            == burg
        )
        assert (
            db.conn.execute("SELECT merged_into_id FROM etymon WHERE id=?", (ton,)).fetchone()[0]
            is None
        )
        assert (
            db.conn.execute(
                "SELECT count(*) FROM etymon_descent WHERE parent_id=? AND child_id=?", (tun, ton)
            ).fetchone()[0]
            == 1
        )
