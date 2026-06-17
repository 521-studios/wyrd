"""Tests for relational_projection (wyrd-zrce.1) — descends-from → etymon_descent,
plus the end-to-end proof that mined cognate descents flow through cluster_cognates
into cognate_id.

Invariants: the gate admits medium / excludes low by default; a refute drops a
matching affirmed edge (D50.4); apply is idempotent + scoped to its own source_id;
dry-run writes nothing; and the full mine→project→cluster pipeline lands the cohort
morpheme in the matched cluster.
"""

from __future__ import annotations

from wyrd.generators.kenning.canonicalization import Assertion, NodeRef, append_assertion
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.cognate_cluster import cluster_cognates
from wyrd.generators.kenning.lexicon.cognate_descent_mining import (
    descent_assertions,
    mine_cognate_descents,
)
from wyrd.generators.kenning.lexicon.relational_projection import (
    DESCENT_SOURCE_ID,
    project_descent_assertions,
)


def _db(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    db.conn.execute("PRAGMA foreign_keys = OFF")  # insert breakdown elements directly
    return db


def _etymon(db, form, *, language="old-english"):
    return db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    ).lastrowid


def _breakdown(db, eid):
    db.conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) "
        "VALUES (1, 0, ?)",
        (eid,),
    )


def _gloss(db, eid, gloss):
    db.conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, gloss))


def _descends(child, parent, *, confidence="medium", polarity="affirm"):
    return Assertion(
        predicate="descends-from",
        subject=NodeRef("etymon", str(child)),
        object=NodeRef("etymon", str(parent)),
        qualifiers={"edge_type": "inheritance"},
        confidence=confidence,
        polarity=polarity,
        method="test",
        source="test",
    ).with_id()


def _edges(db, source_id=DESCENT_SOURCE_ID):
    return {
        (r["parent_id"], r["child_id"], r["edge_type"])
        for r in db.conn.execute(
            "SELECT parent_id, child_id, edge_type FROM etymon_descent WHERE source_id = ?",
            (source_id,),
        )
    }


def test_medium_projects_low_gated_out_by_default(tmp_path):
    db = _db(tmp_path)
    root, mid, lo = _etymon(db, "r"), _etymon(db, "m"), _etymon(db, "l")
    append_assertion(tmp_path, _descends(mid, root, confidence="medium"))
    append_assertion(tmp_path, _descends(lo, root, confidence="low"))
    db.commit()

    result = project_descent_assertions(db, mining_dir=tmp_path, apply=True)
    assert result.inserted == 1
    assert result.gated_out == 1
    assert _edges(db) == {(root, mid, "inheritance")}  # parent=object, child=subject
    db.close()


def test_low_projects_when_gate_lowered(tmp_path):
    db = _db(tmp_path)
    root, lo = _etymon(db, "r"), _etymon(db, "l")
    append_assertion(tmp_path, _descends(lo, root, confidence="low"))
    db.commit()

    result = project_descent_assertions(db, mining_dir=tmp_path, apply=True, confidence_gate="low")
    assert result.inserted == 1
    assert _edges(db) == {(root, lo, "inheritance")}
    db.close()


def test_refute_drops_matching_affirm(tmp_path):
    db = _db(tmp_path)
    root, child = _etymon(db, "r"), _etymon(db, "c")
    append_assertion(tmp_path, _descends(child, root, confidence="medium"))
    append_assertion(tmp_path, _descends(child, root, confidence="medium", polarity="refute"))
    db.commit()

    result = project_descent_assertions(db, mining_dir=tmp_path, apply=True)
    assert result.inserted == 0
    assert result.refuted == 1
    assert _edges(db) == set()
    db.close()


def test_dry_run_writes_nothing(tmp_path):
    db = _db(tmp_path)
    root, child = _etymon(db, "r"), _etymon(db, "c")
    append_assertion(tmp_path, _descends(child, root, confidence="medium"))
    db.commit()

    result = project_descent_assertions(db, mining_dir=tmp_path, apply=False)
    assert result.inserted == 1  # reports what WOULD project
    assert _edges(db) == set()  # but wrote nothing
    db.close()


def test_idempotent_and_scoped_to_source(tmp_path):
    db = _db(tmp_path)
    root, child = _etymon(db, "r"), _etymon(db, "c")
    # A foreign descent edge under a different source must survive the rebuild.
    db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('wiktionary', 'w')")
    db.conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
        "VALUES (?, ?, 'inheritance', 'wiktionary')",
        (root, child),
    )
    append_assertion(tmp_path, _descends(child, root, confidence="medium"))
    db.commit()

    first = project_descent_assertions(db, mining_dir=tmp_path, apply=True)
    assert first.inserted == 1 and first.cleared == 0
    second = project_descent_assertions(db, mining_dir=tmp_path, apply=True)
    assert second.inserted == 1 and second.cleared == 1  # cleared its own row, re-inserted
    assert _edges(db) == {(root, child, "inheritance")}
    assert _edges(db, "wiktionary") == {(root, child, "inheritance")}  # untouched
    db.close()


def test_end_to_end_mine_project_cluster(tmp_path):
    # The proof: a mined cognate descent flows through cluster_cognates into cognate_id.
    db = _db(tmp_path)
    # An existing cluster, derived the normal way (a descent edge + cluster_cognates).
    root = _etymon(db, "tunaz", language="proto-germanic")
    member = _etymon(db, "tún", language="old-norse")
    _gloss(db, member, "enclosure")
    db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('wiktionary', 'w')")
    db.conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
        "VALUES (?, ?, 'inheritance', 'wiktionary')",
        (root, member),
    )
    db.commit()
    cluster_cognates(db, apply=True)
    assert (
        db.conn.execute("SELECT cognate_id FROM etymon WHERE id=?", (member,)).fetchone()[0] == root
    )

    # The unclustered breakdown morpheme that folds to the clustered member.
    x = _etymon(db, "tun", language="old-english")
    _breakdown(db, x)
    _gloss(db, x, "enclosure")
    db.commit()
    assert db.conn.execute("SELECT cognate_id FROM etymon WHERE id=?", (x,)).fetchone()[0] is None

    # Mine → author → project → re-cluster.
    edges = mine_cognate_descents(db)
    assert len(edges) == 1
    for a in descent_assertions(edges, source="test"):
        append_assertion(tmp_path, a)
    db.commit()
    project_descent_assertions(db, mining_dir=tmp_path, apply=True)
    cluster_cognates(db, apply=True)

    # X now shares the cluster root with the member it matched.
    x_cog = db.conn.execute("SELECT cognate_id FROM etymon WHERE id=?", (x,)).fetchone()[0]
    member_cog = db.conn.execute("SELECT cognate_id FROM etymon WHERE id=?", (member,)).fetchone()[
        0
    ]
    assert x_cog == member_cog == root
    db.close()
