"""Tests for same-morpheme identity uplift (wyrd-c7iz, D50 1a).

A shipped morpheme (witness-promoted) and a breakdown-only variant that share a
folded surface + cognate cluster are bound to one canonical_morpheme node
(same-morpheme). A homograph (same surface, different cluster + disjoint gloss) is
left separate (D46).
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.etymon_refs import etymon_ref
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

    assertions = bind_assertions(groups, conn=db.conn, source="test")
    mint = [a for a in assertions if a.predicate == "mint-canonical"]
    binds = [a for a in assertions if a.predicate == "bind"]
    assert len(mint) == 1
    node_id = mint[0].subject.ref
    # wyrd-c6wu: bind refs are stable natural keys, not etymon row-ids.
    assert {b.subject.ref for b in binds} == {
        etymon_ref("old-english", "ford"),
        etymon_ref("old-english", "Ford"),
    }
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


def _unclustered(db, eid):
    db.conn.execute("UPDATE etymon SET cognate_id = NULL WHERE id = ?", (eid,))


def test_medium_confidence_gloss_only_bind(tmp_path):
    # Same folded surface, DIFFERENT cluster (breakdown unclustered), gloss overlap
    # -> medium-confidence bind.
    db = _db(tmp_path)
    leah_shipped = _etymon(db, "leah")
    _ship(db, leah_shipped)
    _gloss(db, leah_shipped, "clearing")
    leah_bd = _etymon(db, "Leah")
    _unclustered(db, leah_bd)
    _gloss(db, leah_bd, "clearing")
    _breakdown_only(db, leah_bd, "Leahton")
    db.commit()
    groups = mine_same_morpheme_binds(db)
    db.close()
    assert len(groups) == 1
    assert groups[0].confidence == "medium"
    assert groups[0].breakdown_etymons == (leah_bd,)


def test_placeholder_gloss_only_overlap_not_bound(tmp_path):
    """Regression: a medium bind must NOT rest SOLELY on a placeholder gloss.
    'a personal name' is on 8507 etymons and carries no morpheme identity, so two
    distinct same-surface morphemes sharing only it (old-norse ami / old-english Ami)
    were wrongly bound as the same morpheme — identity-collapse (D46/D50.2 leave-
    separate). Same setup as the medium-bind test above but with a placeholder gloss."""
    db = _db(tmp_path)
    ami_shipped = _etymon(db, "Ami")
    _ship(db, ami_shipped)
    _gloss(db, ami_shipped, "a personal name")
    ami_bd = _etymon(db, "ami", language="old-norse")
    _unclustered(db, ami_bd)
    _gloss(db, ami_bd, "a personal name")
    _breakdown_only(db, ami_bd, "Amiton")
    db.commit()
    groups = mine_same_morpheme_binds(db)
    db.close()
    assert groups == []  # placeholder-only overlap → no bind


def test_real_gloss_binds_even_with_shared_placeholder(tmp_path):
    """The placeholder filter only removes the placeholder FROM the overlap — a
    genuine shared gloss ('wolf') still binds even when 'a personal name' is also
    shared. (Guards against the fix over-suppressing real-gloss binds.)"""
    db = _db(tmp_path)
    s = _etymon(db, "wulf")
    _ship(db, s)
    _gloss(db, s, "a personal name")
    _gloss(db, s, "wolf")
    bd = _etymon(db, "Wulf", language="old-norse")
    _unclustered(db, bd)
    _gloss(db, bd, "a personal name")
    _gloss(db, bd, "wolf")
    _breakdown_only(db, bd, "Wulfton")
    db.commit()
    groups = mine_same_morpheme_binds(db)
    db.close()
    assert len(groups) == 1 and groups[0].confidence == "medium"
    assert groups[0].breakdown_etymons == (bd,)


def test_ambiguous_two_shipped_left_separate(tmp_path):
    # Two shipped etymons share the breakdown's surface + cluster -> ambiguous ->
    # left separate (D46), no bind.
    db = _db(tmp_path)
    mere1 = _etymon(db, "mere")
    mere2 = _etymon(db, "Mere", cognate=mere1)  # same cluster as mere1
    _ship(db, mere1)
    _ship(db, mere2)
    mere_bd = _etymon(db, "mere", language="welsh", cognate=mere1)  # same cluster
    _breakdown_only(db, mere_bd, "Meredon")
    db.commit()
    groups = mine_same_morpheme_binds(db)
    db.close()
    assert groups == []


def test_multi_member_group_binds_all(tmp_path):
    # One shipped target with TWO breakdown variants -> one node, 3 binds.
    db = _db(tmp_path)
    cot = _etymon(db, "cot")
    _ship(db, cot)
    _gloss(db, cot, "cottage")
    cot_a = _etymon(db, "Cot", cognate=cot)
    _breakdown_only(db, cot_a, "Cotton")
    cot_b = _etymon(db, "COT", cognate=cot)
    _breakdown_only(db, cot_b, "Cotham")
    db.commit()
    groups = mine_same_morpheme_binds(db)
    assertions = bind_assertions(groups, conn=db.conn, source="t")
    db.close()
    assert len(groups) == 1
    assert set(groups[0].breakdown_etymons) == {cot_a, cot_b}
    binds = [a for a in assertions if a.predicate == "bind"]
    assert len([a for a in assertions if a.predicate == "mint-canonical"]) == 1
    assert {b.subject.ref for b in binds} == {
        etymon_ref("old-english", "cot"),
        etymon_ref("old-english", "Cot"),
        etymon_ref("old-english", "COT"),
    }


def test_cli_apply_is_idempotent(tmp_path):
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli as cli_root

    db = _db(tmp_path)
    ford = _etymon(db, "ford")
    _ship(db, ford)
    _gloss(db, ford, "ford crossing")
    ford_bd = _etymon(db, "Ford", cognate=ford)
    _breakdown_only(db, ford_bd, "Fordton")
    db.commit()
    db.close()

    mining = tmp_path / "mining"
    runner = CliRunner()
    args = [
        "lexicon",
        "mine-same-morpheme-binds",
        "--db",
        str(tmp_path / "lexicon.db"),
        "--mining-dir",
        str(mining),
        "--apply",
    ]
    first = runner.invoke(cli_root, args)
    assert first.exit_code == 0, first.output
    stream = mining / "canonicalization" / "_assert_bind.jsonl"
    lines_after_first = stream.read_text().count("\n")
    assert lines_after_first > 0

    second = runner.invoke(cli_root, args)
    assert second.exit_code == 0, second.output
    # Idempotent: re-running writes nothing new.
    assert stream.read_text().count("\n") == lines_after_first
    assert "skipped" in second.output


def test_mixed_group_per_variant_confidence(tmp_path):
    # One shipped target bound by both a cluster variant (high) and a gloss-only
    # variant (medium). The group summary is high, but each bind must carry its
    # OWN confidence — the medium variant must NOT be promoted to high.
    db = _db(tmp_path)
    wic = _etymon(db, "wic")
    _ship(db, wic)
    _gloss(db, wic, "dwelling")
    wic_cluster = _etymon(db, "Wic", cognate=wic)  # same cluster -> high
    _breakdown_only(db, wic_cluster, "Wicton")
    wic_gloss = _etymon(db, "Wíc")  # different surface string, folds to "wic"
    _unclustered(db, wic_gloss)  # different cluster, gloss-only -> medium
    _gloss(db, wic_gloss, "dwelling")
    _breakdown_only(db, wic_gloss, "Wicham")
    db.commit()

    groups = mine_same_morpheme_binds(db)
    assert len(groups) == 1
    g = groups[0]
    assert g.confidence == "high"  # group summary
    assert g.breakdown_confidences == {wic_cluster: "high", wic_gloss: "medium"}

    assertions = bind_assertions(groups, conn=db.conn, source="t")
    conf = {a.subject.ref: a.confidence for a in assertions if a.predicate == "bind"}
    db.close()
    assert conf[etymon_ref("old-english", "wic")] == "high"  # shipped IS the morpheme
    assert conf[etymon_ref("old-english", "Wic")] == "high"
    assert conf[etymon_ref("old-english", "Wíc")] == "medium"  # NOT leaked up to high
