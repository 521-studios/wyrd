"""Tests for same-morpheme identity uplift (wyrd-c7iz, D50 1a).

A shipped morpheme (witness-promoted) and a breakdown-only variant that share a
folded surface + cognate cluster are bound to one canonical_morpheme node
(same-morpheme). A homograph (same surface, different cluster + disjoint gloss) is
left separate (D46).
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.same_morpheme_mining import (
    METHOD,
    bind_assertions,
    mine_same_morpheme_binds,
)


def _etymon(db, form, *, language="old-english", cognate=None):
    eid = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    ).lastrowid
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (cognate or eid, eid))
    return eid


def _ship(db, eid):
    # >=2 distinct citation sources -> witnesses >=2 -> promoted (not via breakdown).
    for src in ("src_a", "src_b"):
        db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES (?, ?)", (src, src))
        db.conn.execute(
            "INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, ?)", (eid, src)
        )


def _gloss(db, eid, gloss):
    db.conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, gloss))


def _breakdown_only(db, eid, name):
    # An etymon attested only by a scholarly toponym breakdown (no citations).
    db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('topo', 'topo')")
    tid = db.conn.execute(
        "INSERT INTO toponym (modern_name, country) VALUES (?, 'England')", (name,)
    ).lastrowid
    teid = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
        "VALUES (?, 'topo', 'high')",
        (tid,),
    ).lastrowid
    db.conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) "
        "VALUES (?, 0, ?)",
        (teid, eid),
    )


def _db(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    return LexiconDB(tmp_path / "lexicon.db")


def test_high_confidence_same_morpheme_bind(tmp_path):
    db = _db(tmp_path)
    ford_shipped = _etymon(db, "ford")  # cluster = c<ford_shipped>
    _ship(db, ford_shipped)
    _gloss(db, ford_shipped, "ford crossing")
    ford_bd = _etymon(db, "Ford", cognate=ford_shipped)  # same folded surface + cluster
    _gloss(db, ford_bd, "ford")
    _breakdown_only(db, ford_bd, "Fordton")
    db.commit()

    groups = mine_same_morpheme_binds(db)
    assert len(groups) == 1
    g = groups[0]
    assert g.shipped_etymon == ford_shipped
    assert g.breakdown_etymons == (ford_bd,)
    assert g.confidence == "high"  # shared cognate cluster

    assertions = bind_assertions(groups, source="test")
    mint = [a for a in assertions if a.predicate == "mint-canonical"]
    binds = [a for a in assertions if a.predicate == "bind"]
    assert len(mint) == 1
    node_id = mint[0].subject.ref
    assert {b.subject.ref for b in binds} == {str(ford_shipped), str(ford_bd)}
    assert all(b.object.ref == node_id for b in binds)
    assert all(b.qualifiers["kind"] == "same-morpheme" for b in binds)
    assert all(a.method == METHOD and a.id for a in assertions)
    db.close()


def test_homograph_not_bound(tmp_path):
    db = _db(tmp_path)
    ton_town = _etymon(db, "ton")  # shipped, cluster A, gloss town
    _ship(db, ton_town)
    _gloss(db, ton_town, "town")
    ton_wave = _etymon(db, "ton", language="welsh")  # homograph: own cluster, gloss wave
    _gloss(db, ton_wave, "wave")
    _breakdown_only(db, ton_wave, "Tonsea")
    db.commit()

    groups = mine_same_morpheme_binds(db)
    db.close()
    assert groups == []  # same surface, different cluster + disjoint gloss -> left separate
