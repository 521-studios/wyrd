"""Tests for the L2->L3 canonicalization projection (wyrd-u6fn.3, D50.6).

Proves the projection's load-bearing properties: confidence gating (D46
leave-separate), refute/retract resolution, merge-canonical chain-flatten
(D22 smallest-id-wins), the conflict rule (>1 distinct hub -> unbound+flagged),
place default+override binding, label tie-break, determinism, idempotency,
reversibility, and the observable skip of not-yet-supported kinds.
"""

from __future__ import annotations

from wyrd.generators.kenning.canonicalization import (
    Assertion,
    NodeRef,
    append_assertion,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.canonicalization_projection import (
    clear_canonical_graph,
    project_canonical,
)


def _db(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    return LexiconDB(tmp_path / "lexicon.db")


def _etymon(db, form, *, language="old-english"):
    return db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    ).lastrowid


def _toponym(db, name):
    return db.conn.execute(
        "INSERT INTO toponym (modern_name, country) VALUES (?, 'England')", (name,)
    ).lastrowid


def _toponym_etymology(db, toponym_id):
    db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('s', 's')")
    return db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) VALUES (?, 's', 'high')",
        (toponym_id,),
    ).lastrowid


def _mint(node_id, node_type="canonical_morpheme"):
    return Assertion(predicate="mint-canonical", subject=NodeRef(node_type, node_id))


def _bind(
    etymon_id, node_id, *, confidence="high", kind="same-morpheme", polarity="affirm", retracts=None
):
    return Assertion(
        predicate="bind",
        subject=NodeRef("etymon", str(etymon_id)),
        object=NodeRef("canonical_morpheme", node_id),
        qualifiers={"kind": kind},
        confidence=confidence,
        polarity=polarity,
        retracts=retracts,
    )


def _author(tmp_path, *assertions):
    return [append_assertion(tmp_path, a) for a in assertions]


def _morpheme_id(db, etymon_id):
    return db.conn.execute(
        "SELECT canonical_morpheme_id FROM etymon WHERE id = ?", (etymon_id,)
    ).fetchone()["canonical_morpheme_id"]


def test_mint_and_bind_same_morpheme(tmp_path):
    db = _db(tmp_path)
    a, b = _etymon(db, "niwe"), _etymon(db, "nīwe")
    db.commit()
    _author(tmp_path, _mint("CM-new"), _bind(a, "CM-new"), _bind(b, "CM-new"))
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert res.minted == {"canonical_morpheme": 1}
    assert res.bound == {"etymon": 2}
    assert _morpheme_id(db, a) == "CM-new"
    assert _morpheme_id(db, b) == "CM-new"
    # The synthetic node row exists.
    assert db.conn.execute("SELECT COUNT(*) n FROM canonical_morpheme").fetchone()["n"] == 1
    db.close()


def test_confidence_gate_leaves_below_gate_unbound(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "leah")
    db.commit()
    _author(tmp_path, _mint("CM-leah"), _bind(a, "CM-leah", confidence="medium"))
    # Default high gate: medium bind does NOT apply (D46 leave-separate).
    project_canonical(db, mining_dir=tmp_path, apply=True)
    assert _morpheme_id(db, a) is None
    # Lower the gate -> it binds.
    project_canonical(db, mining_dir=tmp_path, apply=True, confidence_gate="medium")
    assert _morpheme_id(db, a) == "CM-leah"
    db.close()


def test_refute_drops_the_bind(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "ton")
    db.commit()
    _author(
        tmp_path,
        _mint("CM-ton"),
        _bind(a, "CM-ton"),
        _bind(a, "CM-ton", polarity="refute"),
    )
    project_canonical(db, mining_dir=tmp_path, apply=True)
    assert _morpheme_id(db, a) is None
    db.close()


def test_retract_drops_the_bind(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "cot")
    db.commit()
    [_, bind, _] = _author(tmp_path, _mint("CM-cot"), _bind(a, "CM-cot"), _mint("CM-x"))
    _author(
        tmp_path,
        Assertion(
            predicate="bind",
            subject=NodeRef("etymon", str(a)),
            object=NodeRef("canonical_morpheme", "CM-cot"),
            qualifiers={"kind": "same-morpheme"},
            polarity="retract",
            retracts=bind.id,
        ),
    )
    project_canonical(db, mining_dir=tmp_path, apply=True)
    assert _morpheme_id(db, a) is None
    db.close()


def test_merge_canonical_chain_flatten_smallest_id_wins(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "tun")
    db.commit()
    # Mint three hubs, merge in a chain C<-B, B<-A; root is the lexicographic min.
    _author(
        tmp_path,
        _mint("CM-a"),
        _mint("CM-b"),
        _mint("CM-c"),
        Assertion(
            predicate="merge-canonical",
            subject=NodeRef("canonical_morpheme", "CM-c"),
            object=NodeRef("canonical_morpheme", "CM-b"),
        ),
        Assertion(
            predicate="merge-canonical",
            subject=NodeRef("canonical_morpheme", "CM-b"),
            object=NodeRef("canonical_morpheme", "CM-a"),
        ),
        _bind(a, "CM-c"),
    )
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    rows = dict(db.conn.execute("SELECT id, merged_into FROM canonical_morpheme").fetchall())
    assert rows["CM-a"] is None  # root
    assert rows["CM-b"] == "CM-a"  # flattened, not -> CM... chain
    assert rows["CM-c"] == "CM-a"
    assert res.merged == 2
    # The bind stores the declared node; rollup via merged_into is a read-time step.
    assert _morpheme_id(db, a) == "CM-c"
    db.close()


def test_conflict_two_distinct_hubs_left_unbound_and_flagged(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "mere")
    db.commit()
    _author(tmp_path, _mint("CM-1"), _mint("CM-2"), _bind(a, "CM-1"), _bind(a, "CM-2"))
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert _morpheme_id(db, a) is None  # left UNBOUND
    assert res.conflicts == [("etymon", str(a), ["CM-1", "CM-2"])]
    db.close()


def test_conflict_dissolves_when_hubs_are_merged(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "mere")
    db.commit()
    _author(
        tmp_path,
        _mint("CM-1"),
        _mint("CM-2"),
        Assertion(
            predicate="merge-canonical",
            subject=NodeRef("canonical_morpheme", "CM-2"),
            object=NodeRef("canonical_morpheme", "CM-1"),
        ),
        _bind(a, "CM-1"),
        _bind(a, "CM-2"),
    )
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert res.conflicts == []  # both hubs are one identity post-merge
    assert _morpheme_id(db, a) in {"CM-1", "CM-2"}  # bound to one (both roll up to CM-1)
    db.close()


def test_place_default_and_etymology_override_bind_independently(tmp_path):
    db = _db(tmp_path)
    t = _toponym(db, "Newton")
    te = _toponym_etymology(db, t)
    db.commit()
    _author(
        tmp_path,
        _mint("CP-default", "canonical_place"),
        _mint("CP-override", "canonical_place"),
        Assertion(
            predicate="bind",
            subject=NodeRef("toponym", str(t)),
            object=NodeRef("canonical_place", "CP-default"),
            qualifiers={"kind": "same-place"},
            confidence="high",
        ),
        Assertion(
            predicate="bind",
            subject=NodeRef("toponym_etymology", str(te)),
            object=NodeRef("canonical_place", "CP-override"),
            qualifiers={"kind": "same-place"},
            confidence="high",
        ),
    )
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert res.bound == {"toponym": 1, "toponym_etymology": 1}
    assert (
        db.conn.execute("SELECT canonical_place_id FROM toponym WHERE id=?", (t,)).fetchone()[0]
        == "CP-default"
    )
    assert (
        db.conn.execute(
            "SELECT canonical_place_id FROM toponym_etymology WHERE id=?", (te,)
        ).fetchone()[0]
        == "CP-override"
    )
    db.close()


def test_canonical_label_highest_confidence_wins(tmp_path):
    db = _db(tmp_path)
    _author(
        tmp_path,
        _mint("CP-newton", "canonical_place"),
        Assertion(
            predicate="canonical-label",
            subject=NodeRef("canonical_place", "CP-newton"),
            qualifiers={"stratum": "domesday", "value": "Neutune"},
            confidence="medium",
        ),
        Assertion(
            predicate="canonical-label",
            subject=NodeRef("canonical_place", "CP-newton"),
            qualifiers={"stratum": "domesday", "value": "Newentone"},
            confidence="high",
        ),
    )
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert res.labels == 1
    row = db.conn.execute(
        "SELECT value FROM canonical_label WHERE canonical_id='CP-newton' AND stratum='domesday'"
    ).fetchone()
    assert row["value"] == "Newentone"  # high beat medium
    db.close()


def test_same_sense_bind_is_skipped_observably(tmp_path):
    db = _db(tmp_path)
    _author(
        tmp_path,
        _mint("CS-1", "canonical_sense"),
        Assertion(
            predicate="bind",
            subject=NodeRef("etymon_gloss", "1:new"),
            object=NodeRef("canonical_sense", "CS-1"),
            qualifiers={"kind": "same-sense"},
            confidence="high",
        ),
    )
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert res.skipped_unsupported == {"same-sense": 1}  # flagged, not silently dropped
    db.close()


def test_bind_to_unminted_node_is_skipped_with_warning(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "ford")
    db.commit()
    _author(tmp_path, _bind(a, "CM-never-minted"))  # no mint for it
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert _morpheme_id(db, a) is None
    assert any("unminted" in w for w in res.warnings)
    db.close()


def test_dry_run_writes_nothing(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "wic")
    db.commit()
    _author(tmp_path, _mint("CM-wic"), _bind(a, "CM-wic"))
    res = project_canonical(db, mining_dir=tmp_path, apply=False)
    assert res.bound == {"etymon": 1}  # would-bind reported
    assert _morpheme_id(db, a) is None  # but nothing written
    assert db.conn.execute("SELECT COUNT(*) n FROM canonical_morpheme").fetchone()["n"] == 0
    db.close()


def test_idempotent_and_reversible(tmp_path):
    db = _db(tmp_path)
    a, b = _etymon(db, "burh"), _etymon(db, "byrig")
    db.commit()
    _author(tmp_path, _mint("CM-burh"), _bind(a, "CM-burh"), _bind(b, "CM-burh"))

    def snapshot():
        return (
            sorted(
                tuple(r) for r in db.conn.execute("SELECT id, canonical_morpheme_id FROM etymon")
            ),
            sorted(
                tuple(r) for r in db.conn.execute("SELECT id, merged_into FROM canonical_morpheme")
            ),
        )

    project_canonical(db, mining_dir=tmp_path, apply=True)
    first = snapshot()
    project_canonical(db, mining_dir=tmp_path, apply=True)  # re-project
    assert snapshot() == first  # idempotent

    clear_canonical_graph(db)
    db.commit()
    assert _morpheme_id(db, a) is None
    assert db.conn.execute("SELECT COUNT(*) n FROM canonical_morpheme").fetchone()["n"] == 0
    db.close()


def test_projection_is_order_independent(tmp_path):
    # Authoring assertions in different orders yields the same projected graph
    # (load + project both sort by stable id).
    def run(order):
        d = tmp_path / order
        d.mkdir()
        db = _db(d)
        a, b = _etymon(db, "dun"), _etymon(db, "down")
        db.commit()
        asserts = [_mint("CM-dun"), _bind(a, "CM-dun"), _bind(b, "CM-dun")]
        if order == "rev":
            # Appends binds before the mint; the projection sorts by id, so the
            # result must be identical regardless of on-disk append order.
            asserts = list(reversed(asserts))
        _author(d, *asserts)
        project_canonical(db, mining_dir=d, apply=True)
        snap = (
            sorted(
                tuple(r) for r in db.conn.execute("SELECT id, canonical_morpheme_id FROM etymon")
            ),
            sorted(
                tuple(r) for r in db.conn.execute("SELECT id, merged_into FROM canonical_morpheme")
            ),
        )
        db.close()
        return snap

    assert run("fwd") == run("rev")


def test_clear_enrichment_canonical_stage(tmp_path):
    from wyrd.generators.kenning.lexicon.enrichment.clear import clear_enrichment

    db = _db(tmp_path)
    a = _etymon(db, "stoc")
    db.commit()
    _author(tmp_path, _mint("CM-stoc"), _bind(a, "CM-stoc"))
    project_canonical(db, mining_dir=tmp_path, apply=True)
    assert _morpheme_id(db, a) == "CM-stoc"

    counts = clear_enrichment(db, stage="canonical", apply=True)
    assert counts["canonical_nodes_to_clear"] == 1
    assert _morpheme_id(db, a) is None
    assert db.conn.execute("SELECT COUNT(*) n FROM canonical_morpheme").fetchone()["n"] == 0
    db.close()


def test_run_full_enrichment_runs_projection(tmp_path):
    from wyrd.generators.kenning.enrichment import run_full_enrichment

    db = _db(tmp_path)
    a = _etymon(db, "halh")
    db.commit()
    _author(tmp_path, _mint("CM-halh"), _bind(a, "CM-halh"))
    result = run_full_enrichment(db, apply=True, canonicalization_dir=tmp_path)
    assert "project-canonical" in result["order"]
    assert result["canonical"]["minted"] == {"canonical_morpheme": 1}
    assert result["canonical"]["bound"] == {"etymon": 1}
    assert _morpheme_id(db, a) == "CM-halh"
    db.close()
