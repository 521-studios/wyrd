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
    ProjectionResult,
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


def _ref(db, etymon_id):
    """The etymon's stable natural-key ref (wyrd-c6wu), as the emitters now author."""
    r = db.conn.execute(
        "SELECT language, canonical_form FROM etymon WHERE id = ?", (etymon_id,)
    ).fetchone()
    return f"{r['language'] or ''}:{r['canonical_form']}"


def _bind(
    db,
    etymon_id,
    node_id,
    *,
    confidence="high",
    kind="same-morpheme",
    polarity="affirm",
    retracts=None,
):
    return Assertion(
        predicate="bind",
        subject=NodeRef("etymon", _ref(db, etymon_id)),
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
    _author(tmp_path, _mint("CM-new"), _bind(db, a, "CM-new"), _bind(db, b, "CM-new"))
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
    _author(tmp_path, _mint("CM-leah"), _bind(db, a, "CM-leah", confidence="medium"))
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
        _bind(db, a, "CM-ton"),
        _bind(db, a, "CM-ton", polarity="refute"),
    )
    project_canonical(db, mining_dir=tmp_path, apply=True)
    assert _morpheme_id(db, a) is None
    db.close()


def test_retract_drops_the_bind(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "cot")
    db.commit()
    [_, bind, _] = _author(tmp_path, _mint("CM-cot"), _bind(db, a, "CM-cot"), _mint("CM-x"))
    _author(
        tmp_path,
        Assertion(
            predicate="bind",
            subject=NodeRef("etymon", _ref(db, a)),
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
        _bind(db, a, "CM-c"),
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
    _author(tmp_path, _mint("CM-1"), _mint("CM-2"), _bind(db, a, "CM-1"), _bind(db, a, "CM-2"))
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert _morpheme_id(db, a) is None  # left UNBOUND
    assert res.conflicts == [("etymon", _ref(db, a), ["CM-1", "CM-2"])]
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
        _bind(db, a, "CM-1"),
        _bind(db, a, "CM-2"),
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
    _author(tmp_path, _bind(db, a, "CM-never-minted"))  # no mint for it
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert _morpheme_id(db, a) is None
    assert any("unminted" in w for w in res.warnings)
    db.close()


def test_dry_run_writes_nothing(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "wic")
    db.commit()
    _author(tmp_path, _mint("CM-wic"), _bind(db, a, "CM-wic"))
    res = project_canonical(db, mining_dir=tmp_path, apply=False)
    assert res.bound == {"etymon": 1}  # would-bind reported
    assert _morpheme_id(db, a) is None  # but nothing written
    assert db.conn.execute("SELECT COUNT(*) n FROM canonical_morpheme").fetchone()["n"] == 0
    db.close()


def test_idempotent_and_reversible(tmp_path):
    db = _db(tmp_path)
    a, b = _etymon(db, "burh"), _etymon(db, "byrig")
    db.commit()
    _author(tmp_path, _mint("CM-burh"), _bind(db, a, "CM-burh"), _bind(db, b, "CM-burh"))

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
        asserts = [_mint("CM-dun"), _bind(db, a, "CM-dun"), _bind(db, b, "CM-dun")]
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
    _author(tmp_path, _mint("CM-stoc"), _bind(db, a, "CM-stoc"))
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
    _author(tmp_path, _mint("CM-halh"), _bind(db, a, "CM-halh"))
    result = run_full_enrichment(db, apply=True, canonicalization_dir=tmp_path)
    assert "project-canonical" in result["order"]
    assert result["canonical"]["minted"] == {"canonical_morpheme": 1}
    assert result["canonical"]["bound"] == {"etymon": 1}
    assert _morpheme_id(db, a) == "CM-halh"
    db.close()


def test_confidence_gate_validation(tmp_path):
    import pytest

    db = _db(tmp_path)
    with pytest.raises(ValueError, match="unknown confidence_gate"):
        project_canonical(db, mining_dir=tmp_path, confidence_gate="hgih")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        project_canonical(db, mining_dir=tmp_path, confidence_gate=1.5)
    # A float gate arriving as a string (the CLI path) must be parsed, not rejected.
    project_canonical(db, mining_dir=tmp_path, confidence_gate="0.8")  # no raise
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        project_canonical(db, mining_dir=tmp_path, confidence_gate="1.5")
    db.close()


def test_unresolvable_ref_skipped_not_fatal(tmp_path):
    # wyrd-c6wu: a bind whose etymon natural-key ref doesn't resolve to an
    # observation in this build must be skipped with a warning BEFORE the
    # destructive write — never abort the projection.
    db = _db(tmp_path)
    a = _etymon(db, "stan")
    db.commit()
    _author(
        tmp_path,
        _mint("CM-stan"),
        _bind(db, a, "CM-stan"),  # valid (old-english:stan resolves)
        # No separator -> not a natural-key ref at all (e.g. a stale bare-int).
        Assertion(
            predicate="bind",
            subject=NodeRef("etymon", "not-an-int"),
            object=NodeRef("canonical_morpheme", "CM-stan"),
            qualifiers={"kind": "same-morpheme"},
            confidence="high",
        ),
        # Well-formed natural key, but no such etymon in this build.
        Assertion(
            predicate="bind",
            subject=NodeRef("etymon", "old-english:ghost"),
            object=NodeRef("canonical_morpheme", "CM-stan"),
            qualifiers={"kind": "same-morpheme"},
            confidence="high",
        ),
    )
    res = project_canonical(db, mining_dir=tmp_path, apply=True)  # must not raise
    assert _morpheme_id(db, a) == "CM-stan"  # the valid bind still applied
    assert sum("did not resolve" in w for w in res.warnings) == 2  # both bad refs skipped
    db.close()


def test_float_confidence_gating(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "wella")
    db.commit()
    _author(
        tmp_path,
        _mint("CM-wella"),
        _bind(db, a, "CM-wella", confidence=0.7),
    )
    project_canonical(db, mining_dir=tmp_path, apply=True, confidence_gate=0.9)
    assert _morpheme_id(db, a) is None  # 0.7 < 0.9 gate
    project_canonical(db, mining_dir=tmp_path, apply=True, confidence_gate=0.6)
    assert _morpheme_id(db, a) == "CM-wella"  # 0.7 >= 0.6 gate
    db.close()


def test_merge_to_unminted_node_skipped(tmp_path):
    db = _db(tmp_path)
    _author(
        tmp_path,
        _mint("CM-1"),
        Assertion(
            predicate="merge-canonical",
            subject=NodeRef("canonical_morpheme", "CM-1"),
            object=NodeRef("canonical_morpheme", "CM-never"),
        ),
    )
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert res.merged == 0
    assert any("unminted" in w for w in res.warnings)
    db.close()


def test_cross_type_merge_skipped(tmp_path):
    db = _db(tmp_path)
    _author(
        tmp_path,
        _mint("CM-1", "canonical_morpheme"),
        _mint("CP-1", "canonical_place"),
        Assertion(
            predicate="merge-canonical",
            subject=NodeRef("canonical_morpheme", "CM-1"),
            object=NodeRef("canonical_place", "CP-1"),
        ),
    )
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert res.merged == 0
    assert any("crosses node types" in w for w in res.warnings)
    db.close()


def test_orphan_label_skipped(tmp_path):
    db = _db(tmp_path)
    _author(
        tmp_path,
        Assertion(
            predicate="canonical-label",
            subject=NodeRef("canonical_place", "CP-never-minted"),
            qualifiers={"stratum": "modern", "value": "Nowhere"},
        ),
    )
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert res.labels == 0
    assert db.conn.execute("SELECT COUNT(*) n FROM canonical_label").fetchone()["n"] == 0
    assert any("unminted" in w for w in res.warnings)
    db.close()


def test_retract_of_winning_label_reverts_to_runner_up(tmp_path):
    db = _db(tmp_path)
    [_, label_hi, _] = _author(
        tmp_path,
        _mint("CP-x", "canonical_place"),
        Assertion(
            predicate="canonical-label",
            subject=NodeRef("canonical_place", "CP-x"),
            qualifiers={"stratum": "modern", "value": "Hightown"},
            confidence="high",
        ),
        Assertion(
            predicate="canonical-label",
            subject=NodeRef("canonical_place", "CP-x"),
            qualifiers={"stratum": "modern", "value": "Midtown"},
            confidence="medium",
        ),
    )
    _author(
        tmp_path,
        Assertion(
            predicate="canonical-label",
            subject=NodeRef("canonical_place", "CP-x"),
            qualifiers={"stratum": "modern", "value": "Hightown"},
            confidence="high",
            polarity="retract",
            retracts=label_hi.id,
        ),
    )
    project_canonical(db, mining_dir=tmp_path, apply=True)
    row = db.conn.execute(
        "SELECT value FROM canonical_label WHERE canonical_id='CP-x' AND stratum='modern'"
    ).fetchone()
    assert row["value"] == "Midtown"  # the high label retracted -> medium runner-up wins
    db.close()


def test_merge_order_invariance_multi_component(tmp_path):
    # Two independent merge components, authored in opposite orders, must yield
    # the same merged_into graph (union-find is order-invariant; D36.9).
    def run(order):
        d = tmp_path / order
        d.mkdir()
        db = _db(d)
        merges = [
            ("CM-c", "CM-b"),
            ("CM-b", "CM-a"),
            ("CM-y", "CM-x"),
        ]
        if order == "rev":
            merges = list(reversed(merges))
        asserts = [_mint(n) for n in ("CM-a", "CM-b", "CM-c", "CM-x", "CM-y")]
        asserts += [
            Assertion(
                predicate="merge-canonical",
                subject=NodeRef("canonical_morpheme", s),
                object=NodeRef("canonical_morpheme", o),
            )
            for s, o in merges
        ]
        _author(d, *asserts)
        project_canonical(db, mining_dir=d, apply=True)
        snap = sorted(
            tuple(r) for r in db.conn.execute("SELECT id, merged_into FROM canonical_morpheme")
        )
        db.close()
        return snap

    assert run("fwd") == run("rev")


# --- wyrd-u6fn.4: folding the deterministic legacy identity clustering ---


def _set_merged(db, loser, winner):
    db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (winner, loser))


def _set_lemma(db, inflection, lemma):
    db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (lemma, inflection))


def test_fold_legacy_identity_family(tmp_path):
    # An OCR variant (merged_into_id) and an inflection (lemma_id) of a root must
    # all land on one canonical_morpheme hub — reproducing today's clustering with
    # no committed assertions on disk (fold_legacy default True).
    db = _db(tmp_path)
    root = _etymon(db, "tun")
    ocr = _etymon(db, "tvn")
    _set_merged(db, ocr, root)
    infl = _etymon(db, "tuns")
    _set_lemma(db, infl, root)
    db.commit()
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    cm = _morpheme_id(db, root)
    assert cm is not None
    assert _morpheme_id(db, ocr) == cm  # OCR variant shares the hub
    assert _morpheme_id(db, infl) == cm  # inflection shares the hub
    assert res.bound == {"etymon": 3}
    db.close()


def test_fold_singleton_left_unbound(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "ford")  # no merge / lemma -> its own identity
    db.commit()
    project_canonical(db, mining_dir=tmp_path, apply=True)
    assert _morpheme_id(db, a) is None  # NULL, exactly like merged_into_id IS NULL today
    db.close()


def test_legacy_bind_kinds(tmp_path):
    from wyrd.generators.kenning.lexicon.canonicalization_projection import (
        _legacy_identity_assertions,
    )

    db = _db(tmp_path)
    root = _etymon(db, "burh")
    ocr = _etymon(db, "buruh")
    _set_merged(db, ocr, root)
    infl = _etymon(db, "burhs")
    _set_lemma(db, infl, root)
    db.commit()
    kinds = {
        a.subject.ref: a.qualifiers["kind"]
        for a in _legacy_identity_assertions(db, ProjectionResult())
        if a.predicate == "bind"
    }
    assert kinds[_ref(db, root)] == "same-morpheme"
    assert kinds[_ref(db, ocr)] == "same-morpheme"  # OCR variant = same morpheme
    assert kinds[_ref(db, infl)] == "inflection-of"  # inflection of the lemma
    db.close()


def test_identity_fidelity_gate(tmp_path):
    from wyrd.generators.kenning.lexicon.canonicalization_projection import (
        assess_identity_fidelity,
    )

    db = _db(tmp_path)
    r1, v1 = _etymon(db, "tun"), _etymon(db, "tvn")
    _set_merged(db, v1, r1)
    r2, i2 = _etymon(db, "ham"), _etymon(db, "hams")
    _set_lemma(db, i2, r2)
    db.commit()
    project_canonical(db, mining_dir=tmp_path, apply=True)
    rep = assess_identity_fidelity(db)
    db.close()
    assert rep.legacy_families == 2
    assert rep.faithful == 2
    assert rep.violations == 0


def test_fold_composes_with_authored_bind(tmp_path):
    # An authored bind (1a-style) to the SAME content-keyed node id as the legacy
    # fold composes onto one hub — not a conflict (same node id => same identity).
    from wyrd.generators.kenning.canonicalization import mint_canonical_id

    db = _db(tmp_path)
    root = _etymon(db, "tun")
    ocr = _etymon(db, "tvn")
    _set_merged(db, ocr, root)
    extra = _etymon(db, "Tun")  # singleton, but authored-bound below
    db.commit()
    node_id = mint_canonical_id("canonical_morpheme", "old-english", "tun")  # root's key
    _author(tmp_path, _mint(node_id), _bind(db, extra, node_id))
    project_canonical(db, mining_dir=tmp_path, apply=True)
    assert _morpheme_id(db, root) == node_id
    assert _morpheme_id(db, ocr) == node_id
    assert _morpheme_id(db, extra) == node_id  # authored bind landed on the legacy hub
    db.close()


def test_fold_legacy_off_is_pure_assertion(tmp_path):
    # fold_legacy=False ignores the columns — only authored assertions apply.
    db = _db(tmp_path)
    root = _etymon(db, "tun")
    ocr = _etymon(db, "tvn")
    _set_merged(db, ocr, root)
    db.commit()
    project_canonical(db, mining_dir=tmp_path, apply=True, fold_legacy=False)
    assert _morpheme_id(db, root) is None  # not folded
    assert _morpheme_id(db, ocr) is None
    db.close()


def test_fidelity_detects_violation(tmp_path):
    # If a legacy family's members do NOT all share a canonical group, the gate
    # must report a violation + sample it (not silently pass).
    from wyrd.generators.kenning.lexicon.canonicalization_projection import (
        assess_identity_fidelity,
    )

    db = _db(tmp_path)
    r, v = _etymon(db, "tun"), _etymon(db, "tvn")
    _set_merged(db, v, r)
    db.commit()
    project_canonical(db, mining_dir=tmp_path, apply=True)
    # Break the binding for one member after the fact -> a split family.
    db.conn.execute("UPDATE etymon SET canonical_morpheme_id = NULL WHERE id = ?", (v,))
    db.commit()
    rep = assess_identity_fidelity(db)
    db.close()
    assert rep.violations == 1
    assert rep.faithful == 0
    assert len(rep.sample_violations) == 1
    assert rep.sample_violations[0].legacy_root == r


def test_fidelity_holds_when_log_merge_joins_families(tmp_path):
    # A logged merge-canonical may JOIN two legacy families onto one hub; fidelity
    # must still pass (canonical may join, never split) — exercises canonical_root.
    from wyrd.generators.kenning.canonicalization import mint_canonical_id
    from wyrd.generators.kenning.lexicon.canonicalization_projection import (
        assess_identity_fidelity,
    )

    db = _db(tmp_path)
    r1, v1 = _etymon(db, "tun"), _etymon(db, "tvn")
    _set_merged(db, v1, r1)
    r2, v2 = _etymon(db, "ham"), _etymon(db, "hvm")
    _set_merged(db, v2, r2)
    db.commit()
    n1 = mint_canonical_id("canonical_morpheme", "old-english", "tun")
    n2 = mint_canonical_id("canonical_morpheme", "old-english", "ham")
    _author(
        tmp_path,
        Assertion(
            predicate="merge-canonical",
            subject=NodeRef("canonical_morpheme", n2),
            object=NodeRef("canonical_morpheme", n1),
        ),
    )
    project_canonical(db, mining_dir=tmp_path, apply=True)
    rep = assess_identity_fidelity(db)
    db.close()
    assert rep.violations == 0  # both families still each map to one (joined) group
    assert rep.faithful == 2


def test_fold_member_with_both_merge_and_lemma_is_same_morpheme(tmp_path):
    # A member carrying BOTH merged_into_id and lemma_id is an OCR variant first
    # (same-morpheme), not an inflection — pin the precedence.
    from wyrd.generators.kenning.lexicon.canonicalization_projection import (
        _legacy_identity_assertions,
    )

    db = _db(tmp_path)
    root = _etymon(db, "wic")
    both = _etymon(db, "wicc")
    _set_merged(db, both, root)
    _set_lemma(db, both, root)
    db.commit()
    res = ProjectionResult()
    kinds = {
        a.subject.ref: a.qualifiers["kind"]
        for a in _legacy_identity_assertions(db, res)
        if a.predicate == "bind"
    }
    assert kinds[_ref(db, both)] == "same-morpheme"
    db.close()


def test_fold_root_without_canonical_form_warns(tmp_path):
    # A multi-member family whose root has an empty canonical_form is skipped, but
    # observably (a warning), not silently.
    from wyrd.generators.kenning.lexicon.canonicalization_projection import (
        _legacy_identity_assertions,
    )

    db = _db(tmp_path)
    root = _etymon(db, "x")
    ocr = _etymon(db, "x2")
    _set_merged(db, ocr, root)
    db.conn.execute("UPDATE etymon SET canonical_form = '' WHERE id = ?", (root,))
    db.commit()
    res = ProjectionResult()
    out = _legacy_identity_assertions(db, res)
    db.close()
    assert out == []  # family dropped
    assert any("no canonical_form" in w for w in res.warnings)  # but observably


def _reflex_link(db, parent, child):
    # A curated reflex-link inheritance edge (COLLAPSE_VARIANT_SOURCE_ID) — the
    # third family mechanism _build_family_rollup follows (cross-era same morpheme).
    db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('collapse', 'collapse')")
    db.conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id, confidence) "
        "VALUES (?, ?, 'inheritance', 'collapse', 'high')",
        (parent, child),
    )


def test_reflex_link_member_is_same_morpheme(tmp_path):
    # A member pulled into the family only via a reflex-link edge (no merged_into,
    # no lemma_id) is a cross-era SAME morpheme — bind kind must be same-morpheme,
    # not inflection-of.
    from wyrd.generators.kenning.lexicon.canonicalization_projection import (
        _legacy_identity_assertions,
    )

    db = _db(tmp_path)
    root = _etymon(db, "burh")
    reflex = _etymon(db, "borough", language="modern-english")
    _reflex_link(db, root, reflex)
    db.commit()
    res = ProjectionResult()
    kinds = {
        a.subject.ref: a.qualifiers["kind"]
        for a in _legacy_identity_assertions(db, res)
        if a.predicate == "bind"
    }
    assert kinds[_ref(db, reflex)] == "same-morpheme"  # reflex, not inflection
    assert kinds[_ref(db, root)] == "same-morpheme"
    db.close()


def test_skip_warning_surfaces_through_project_canonical(tmp_path):
    # The empty-canonical_form warning must reach project_canonical's
    # ProjectionResult.warnings (the seam the CLI echoes), not just the helper.
    db = _db(tmp_path)
    root = _etymon(db, "z")
    ocr = _etymon(db, "z2")
    _set_merged(db, ocr, root)
    db.conn.execute("UPDATE etymon SET canonical_form = '' WHERE id = ?", (root,))
    db.commit()
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    db.close()
    assert any("no canonical_form" in w for w in res.warnings)


def test_bind_natural_key_ref_survives_id_reassignment(tmp_path):
    """wyrd-c6wu: a same-morpheme bind authored against one set of etymon row-ids
    still binds the CORRECT morpheme after a rebuild reassigns every id — the bind
    subject is the stable natural key, not the row-id (an id-keyed bind would bind
    a 'pad' row or nothing)."""
    db = _db(tmp_path)
    a = _etymon(db, "stoc")
    db.commit()
    _author(tmp_path, _mint("CM-stoc"), _bind(db, a, "CM-stoc"))

    # Simulate a from-scratch rebuild: wipe + reinsert the SAME morpheme, padded so
    # AUTOINCREMENT hands it a DIFFERENT id.
    db.conn.execute("DELETE FROM etymon")
    for i in range(5):
        db.conn.execute(
            "INSERT INTO etymon (canonical_form, language) VALUES (?, 'x')", (f"pad{i}",)
        )
    new_a = _etymon(db, "stoc")
    db.commit()
    assert new_a != a  # id genuinely reassigned

    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert res.bound == {"etymon": 1}
    assert _morpheme_id(db, new_a) == "CM-stoc"  # bound at its NEW id via the natural key
    db.close()
