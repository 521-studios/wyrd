"""wyrd-rogd.9 — the family rollup follows inheritance descent edges.

A later-era reflex rolls up into its earliest-era ancestor's family (so cross-era
forms are ONE morpheme and the reflex inherits the ancestor's era_reflexes),
while genuine homographs (no inheritance edge) stay separate, and only
inheritance edges are followed (derivation/compound are NOT — wyrd-rogd.11).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.bundle._export import build_family_rollup


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def _edge(db, parent, child, edge_type, source_id="collapse"):
    # rogd.9 follows ONLY 'collapse'-sourced (rogd.15 reflex-link) inheritance.
    db.conn.execute(
        "INSERT OR IGNORE INTO source (id, title) VALUES (?, ?)", (source_id, source_id)
    )
    db.conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) VALUES (?,?,?,?)",
        (parent, child, edge_type, source_id),
    )


def test_reflex_rolls_into_ancestor_family(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        biscop = db.upsert_etymon("biscop", "old-english")
        bishop = db.upsert_etymon("bishop", "modern-english")
        _edge(db, biscop, bishop, "inheritance")
        db.commit()
        members_by_root, root_of = build_family_rollup(db)
        assert root_of(bishop) == biscop  # reflex joins the ancestor's family
        assert root_of(biscop) == biscop
        assert set(members_by_root[biscop]) == {biscop, bishop}


def test_homographs_without_edge_stay_separate(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        dun = db.upsert_etymon("dūn", "old-english")
        holt = db.upsert_etymon("holt", "old-english")
        db.commit()
        _, root_of = build_family_rollup(db)
        assert root_of(dun) == dun and root_of(holt) == holt


def test_derivation_edge_is_not_followed(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        base = db.upsert_etymon("tūn", "old-english")
        deriv = db.upsert_etymon("tūnscipe", "old-english")
        _edge(db, base, deriv, "derivation")
        db.commit()
        _, root_of = build_family_rollup(db)
        assert root_of(deriv) == deriv  # derived word keeps its own family


def test_multi_era_chain_rolls_to_oldest(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        oe = db.upsert_etymon("hyll", "old-english")
        me = db.upsert_etymon("hil", "middle-english")
        mod = db.upsert_etymon("hill", "modern-english")
        _edge(db, oe, me, "inheritance")
        _edge(db, me, mod, "inheritance")
        db.commit()
        _, root_of = build_family_rollup(db)
        assert root_of(mod) == oe and root_of(me) == oe  # chain to the oldest


def test_inheritance_cycle_terminates(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        a = db.upsert_etymon("aaa", "old-english")
        b = db.upsert_etymon("bbb", "middle-english")
        _edge(db, a, b, "inheritance")
        _edge(db, b, a, "inheritance")  # noisy back-edge
        db.commit()
        _, root_of = build_family_rollup(db)
        assert root_of(a) in (a, b) and root_of(b) in (a, b)  # terminates, no hang


def test_non_collapse_inheritance_is_not_followed(fresh_db: Path) -> None:
    # The raw Wiktionary inheritance graph is too dense to follow (it would
    # collapse ~440K families); only curated 'collapse'-sourced reflex links
    # drive the rollup. A wiktionary-sourced inheritance edge is ignored.
    with LexiconDB(fresh_db) as db:
        anc = db.upsert_etymon("wīc", "old-english")
        ref = db.upsert_etymon("wick", "modern-english")
        _edge(db, anc, ref, "inheritance", source_id="wiktionary")
        db.commit()
        _, root_of = build_family_rollup(db)
        assert root_of(ref) == ref  # NOT rolled into the ancestor


def test_consensus_reflex_promotes_ancestor_not_itself(fresh_db: Path) -> None:
    # The double-emission guard: a reflex that is its OWN consensus lemma must
    # promote its ancestor's root (so it isn't promoted standalone AND rolled
    # into the ancestor's family → emitted twice / surface double-counted).
    from wyrd.generators.kenning.lexicon.bundle._export import (
        build_family_rollup,
        select_promoted_root_ids,
    )

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        db.upsert_source(id="src-b", title="B")
        biscop = db.upsert_etymon("biscop", "old-english")
        bishop = db.upsert_etymon("bishop", "modern-english")
        db.add_citation(bishop, "src-a")  # 2 witnesses → consensus lemma
        db.add_citation(bishop, "src-b")
        _edge(db, biscop, bishop, "inheritance")
        db.commit()
        members_by_root, root_of = build_family_rollup(db)
        promoted = select_promoted_root_ids(
            db,
            lang_thresholds={},
            min_witnesses=1,
            include_rando=False,
            include_wiktionary_empirical=False,
            include_wave2_enriched=False,
            root_of=root_of,
            members_by_root=members_by_root,
        )
        assert biscop in promoted  # ancestor promoted
        assert bishop not in promoted  # reflex NOT promoted standalone


def test_reflex_with_lemma_id_still_finds_ancestor(fresh_db: Path) -> None:
    # The edge sits on a reflex child that itself has a lemma_id; root_of lands
    # on the child's lemma-root first, so the edge must be keyed there too.
    with LexiconDB(fresh_db) as db:
        anc = db.upsert_etymon("biscop", "old-english")
        lemma = db.upsert_etymon("bishop", "modern-english")
        infl = db.upsert_etymon("bishops", "modern-english")
        db.conn.execute("UPDATE etymon SET lemma_id=? WHERE id=?", (lemma, infl))
        _edge(db, anc, infl, "inheritance")  # edge on the inflected reflex
        db.commit()
        _, root_of = build_family_rollup(db)
        assert root_of(infl) == anc  # rolls through its lemma to the ancestor
        assert root_of(lemma) == anc


def test_multi_parent_picks_lowest_content_key(fresh_db: Path) -> None:
    # Two candidate ancestors for one child → keep the smallest
    # (language, canonical_form), a rebuild-stable key, NOT the autoinc id.
    with LexiconDB(fresh_db) as db:
        p1 = db.upsert_etymon("aac", "old-english")
        p2 = db.upsert_etymon("zzz", "old-english")
        child = db.upsert_etymon("aac", "modern-english")
        _edge(db, p1, child, "inheritance")
        _edge(db, p2, child, "inheritance")
        db.commit()
        _, root_of = build_family_rollup(db)
        assert root_of(child) == p1  # ('old-english','aac') < ('old-english','zzz')
