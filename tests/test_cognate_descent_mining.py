"""Tests for cognate_descent_mining (wyrd-zrce.1) — placing unclustered breakdown
morphemes into existing cognate clusters via relational descends-from edges.

Invariants under test: a fold-match to a clustered etymon yields a descends-from to
the cluster ROOT (not the matched member); gloss corroboration tiers confidence
(medium) vs surface-only (low); ambiguity across clusters is left-separate (D46);
and the emitted assertions are valid Family-B descends-from records (never bind).
"""

from __future__ import annotations

from wyrd.generators.kenning.canonicalization import validate
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.cognate_descent_mining import (
    descent_assertions,
    mine_cognate_descents,
)


def _db(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    # Unit-testing the miner's queries, not referential integrity — skip the
    # toponym→toponym_etymology→element parent chain and insert elements directly.
    db.conn.execute("PRAGMA foreign_keys = OFF")
    return db


def _etymon(db, form, *, language="old-english", cognate_id=None):
    eid = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    ).lastrowid
    if cognate_id is not None:
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (cognate_id, eid))
    return eid


def _breakdown(db, eid):
    # toponym_etymology_id = eid keeps (toponym_etymology_id, ordinal) unique per call.
    db.conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) "
        "VALUES (?, 0, ?)",
        (eid, eid),
    )


def _gloss(db, eid, *glosses):
    for g in glosses:
        db.conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, g))


def _cluster(db, root_form, member_form, *, gloss=None):
    """A two-etymon cognate cluster: a root (cognate_id = itself) + one clustered
    member in another language. Returns (root_id, member_id)."""
    root = _etymon(db, root_form, language="proto-germanic")
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, root))
    member = _etymon(db, member_form, language="old-norse", cognate_id=root)
    if gloss:
        _gloss(db, member, gloss)
    return root, member


def test_places_cohort_into_gloss_corroborated_cluster(tmp_path):
    db = _db(tmp_path)
    root, _member = _cluster(db, "tunaz", "tún", gloss="enclosure")
    x = _etymon(db, "tun", language="old-english")  # folds to "tun" like tún; cognate_id NULL
    _breakdown(db, x)
    _gloss(db, x, "enclosure")
    db.commit()

    edges = mine_cognate_descents(db)
    assert len(edges) == 1
    assert edges[0].child_etymon == x
    assert edges[0].cluster_root == root  # the ROOT, not the matched member
    assert edges[0].confidence == "medium"  # gloss-corroborated
    db.close()


def test_surface_only_match_is_low_confidence(tmp_path):
    db = _db(tmp_path)
    root, _member = _cluster(db, "tunaz", "tún")  # no gloss on the cluster
    x = _etymon(db, "tun", language="old-english")
    _breakdown(db, x)
    _gloss(db, x, "enclosure")  # cohort has a gloss, but the cluster doesn't overlap
    db.commit()

    edges = mine_cognate_descents(db)
    assert len(edges) == 1
    assert edges[0].confidence == "low"
    db.close()


def test_ambiguous_across_clusters_left_separate(tmp_path):
    db = _db(tmp_path)
    # Two DIFFERENT clusters whose members both fold to "tun"; no gloss to disambiguate.
    _cluster(db, "tunaz", "tún")
    _cluster(db, "tunos", "tun")
    x = _etymon(db, "tun", language="old-english")
    _breakdown(db, x)
    db.commit()

    assert mine_cognate_descents(db) == []  # D46 leave-separate
    db.close()


def test_ambiguous_resolved_by_unique_gloss(tmp_path):
    db = _db(tmp_path)
    root_a, _ = _cluster(db, "tunaz", "tún", gloss="enclosure")
    _cluster(db, "tunos", "tunn", gloss="barrel")  # different sense
    x = _etymon(db, "tun", language="old-english")
    _breakdown(db, x)
    _gloss(db, x, "enclosure")  # overlaps only cluster A
    db.commit()

    edges = mine_cognate_descents(db)
    assert len(edges) == 1
    assert edges[0].cluster_root == root_a
    assert edges[0].confidence == "medium"
    db.close()


def test_no_clustered_match_yields_no_edge(tmp_path):
    db = _db(tmp_path)
    x = _etymon(db, "lonelyform", language="old-english")
    _breakdown(db, x)
    db.commit()
    assert mine_cognate_descents(db) == []
    db.close()


def test_non_breakdown_unclustered_is_ignored(tmp_path):
    # Only admitted BREAKDOWN morphemes are the cohort; a plain unclustered etymon
    # that happens to fold-match a cluster is not 1b's concern.
    db = _db(tmp_path)
    _cluster(db, "tunaz", "tún", gloss="enclosure")
    x = _etymon(db, "tun", language="old-english")  # NOT a breakdown element
    _gloss(db, x, "enclosure")
    db.commit()
    assert mine_cognate_descents(db) == []
    db.close()


def test_assertions_are_valid_descends_from(tmp_path):
    db = _db(tmp_path)
    root, _ = _cluster(db, "tunaz", "tún", gloss="enclosure")
    x = _etymon(db, "tun", language="old-english")
    _breakdown(db, x)
    _gloss(db, x, "enclosure")
    db.commit()

    assertions = descent_assertions(mine_cognate_descents(db), source="test")
    assert len(assertions) == 1
    a = assertions[0]
    assert a.predicate == "descends-from"
    assert a.subject.type == "etymon" and a.subject.ref == str(x)
    assert a.object.type == "etymon" and a.object.ref == str(root)
    assert a.qualifiers["edge_type"] == "inheritance"
    validate(a)  # raises if the record violates the Family-B spec
    db.close()


def test_deterministic_ids(tmp_path):
    db = _db(tmp_path)
    _cluster(db, "tunaz", "tún", gloss="enclosure")
    x = _etymon(db, "tun", language="old-english")
    _breakdown(db, x)
    _gloss(db, x, "enclosure")
    db.commit()

    first = [a.id for a in descent_assertions(mine_cognate_descents(db), source="test")]
    second = [a.id for a in descent_assertions(mine_cognate_descents(db), source="test")]
    assert first == second  # no timestamp in the id (D36.9 idempotent re-mining)
    db.close()


def test_clustered_or_merged_morphemes_excluded_from_cohort(tmp_path):
    # The cohort is unclustered (cognate_id NULL) + non-merged breakdown morphemes.
    # A breakdown morpheme that is already clustered, or a tombstoned duplicate, must
    # not be re-placed (the latter is the D22 corruption guard).
    db = _db(tmp_path)
    root, _ = _cluster(db, "tunaz", "tún", gloss="enclosure")
    already = _etymon(db, "tun", language="old-english")  # folds to the cluster…
    db.conn.execute(
        "UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, already)
    )  # …but clustered
    _breakdown(db, already)
    tomb = _etymon(db, "tun", language="middle-english")  # folds too…
    db.conn.execute(
        "UPDATE etymon SET merged_into_id = ? WHERE id = ?", (root, tomb)
    )  # …but merged away
    _breakdown(db, tomb)
    db.commit()
    assert mine_cognate_descents(db) == []
    db.close()


def test_multiple_edges_sorted_by_child(tmp_path):
    db = _db(tmp_path)
    root, _ = _cluster(db, "tunaz", "tún", gloss="enclosure")
    # Insert cohort out of fold order; edges must come back sorted by child_etymon.
    b = _etymon(db, "tun", language="old-english")
    _breakdown(db, b)
    _gloss(db, b, "enclosure")
    a = _etymon(db, "tūn", language="mercian")  # also folds to "tun"
    _breakdown(db, a)
    _gloss(db, a, "enclosure")
    db.commit()
    edges = mine_cognate_descents(db)
    assert [e.child_etymon for e in edges] == sorted(e.child_etymon for e in edges)
    assert {e.child_etymon for e in edges} == {a, b}
    assert all(e.cluster_root == root for e in edges)
    db.close()


def test_same_cluster_multiple_members_one_edge(tmp_path):
    # A cohort morpheme folding to TWO members of the SAME cluster → one edge to the
    # shared root (grouped by cluster, not by matched member).
    db = _db(tmp_path)
    root = _etymon(db, "tunaz", language="proto-germanic")
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, root))
    m1 = _etymon(db, "tún", language="old-norse", cognate_id=root)
    _gloss(db, m1, "enclosure")
    m2 = _etymon(
        db, "tun", language="icelandic", cognate_id=root
    )  # second member, same fold+cluster
    _gloss(db, m2, "enclosure")
    x = _etymon(db, "tun", language="old-english")
    _breakdown(db, x)
    _gloss(db, x, "enclosure")
    db.commit()
    edges = mine_cognate_descents(db)
    assert len(edges) == 1
    assert edges[0].child_etymon == x and edges[0].cluster_root == root
    db.close()
