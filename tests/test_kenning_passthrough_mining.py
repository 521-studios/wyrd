"""Tests for stream-1 passthrough mining (wyrd-h5u1, D50/D51).

A toponym decomposed by two scholars — coarse ``[ald, ington]`` vs fine
``[ald, ing, tūn]`` — directly attests the passthrough ``ington → ing + tūn``.
The miner records it (cluster-pooled, etymon-referenced, surface-ordered) and
emits validated D50 ``composed-of`` assertions. A gloss-blind surface segmenter
would instead split a real primitive like ``barton`` into ``bar+ton``; the
cross-scholar evidence is what keeps it gloss-correct.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.decomposition_grader import load_cluster_index
from wyrd.generators.kenning.lexicon.passthrough_mining import (
    METHOD,
    extract_cross_scholar_passthroughs,
    passthrough_assertions,
)


def _etymon(db, form, *, language="old-english"):
    cur = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    )
    eid = cur.lastrowid
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (eid, eid))
    return eid


def _reflex(db, surface, etymon_id, position="post"):
    cur = db.conn.execute(
        "INSERT INTO reflex (surface_form, position) VALUES (?, ?)", (surface, position)
    )
    db.conn.execute(
        "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)",
        (cur.lastrowid, etymon_id),
    )


def _breakdown(db, toponym_id, elements):
    teid = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
        "VALUES (?, 'test_src', 'high')",
        (toponym_id,),
    ).lastrowid
    for ordinal, eid in enumerate(elements):
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
            (teid, ordinal, eid),
        )


def _world(tmp_path):
    db = LexiconDB(tmp_path / "lexicon.db")
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")
    ids = {
        "ald": _etymon(db, "Ald"),
        "ing": _etymon(db, "ing"),
        "tun": _etymon(db, "tūn"),
        "ington": _etymon(db, "ington"),  # the composite
    }
    _reflex(db, "ton", ids["tun"])  # tūn's modern reflex — so ington = ing|ton
    tid = db.conn.execute(
        "INSERT INTO toponym (modern_name, country) VALUES ('Aldington', 'England')"
    ).lastrowid
    # Two scholars: one coarse (ington), one fine (ing + tūn). Same place.
    _breakdown(db, tid, [ids["ald"], ids["ington"]])
    _breakdown(db, tid, [ids["ald"], ids["ing"], ids["tun"]])
    db.commit()
    return db, ids


def test_extracts_cross_scholar_passthrough(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db, ids = _world(tmp_path)
    index = load_cluster_index(db)
    passthroughs = extract_cross_scholar_passthroughs(db, index=index)
    db.close()

    assert len(passthroughs) == 1
    p = passthroughs[0]
    assert p.composite_etymon == ids["ington"]
    # Surface-ordered: ington tiles as ing | ton, so ing precedes tūn.
    assert p.constituent_etymons == (ids["ing"], ids["tun"])
    assert p.composite_form == "ington"
    assert p.constituent_forms == ("ing", "tun")
    assert p.support == 1
    assert p.sample_toponyms == ("Aldington",)


def test_emits_validated_composed_of_assertions(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db, ids = _world(tmp_path)
    index = load_cluster_index(db)
    passthroughs = extract_cross_scholar_passthroughs(db, index=index)
    db.close()

    assertions = passthrough_assertions(passthroughs, source="test")
    assert len(assertions) == 2
    assert all(a.predicate == "composed-of" for a in assertions)
    assert all(a.subject.ref == str(ids["ington"]) for a in assertions)
    assert all(a.method == METHOD and a.id for a in assertions)
    by_ordinal = {a.qualifiers["ordinal"]: a.object.ref for a in assertions}
    assert by_ordinal == {0: str(ids["ing"]), 1: str(ids["tun"])}


def test_assertions_are_deterministic(tmp_path):
    # Same corpus -> byte-identical ids (D36.9; no timestamp in the mined record).
    init_schema(tmp_path / "lexicon.db")
    db, _ = _world(tmp_path)
    a1 = passthrough_assertions(extract_cross_scholar_passthroughs(db), source="test")
    a2 = passthrough_assertions(extract_cross_scholar_passthroughs(db), source="test")
    db.close()
    assert [a.id for a in a1] == [a.id for a in a2]
