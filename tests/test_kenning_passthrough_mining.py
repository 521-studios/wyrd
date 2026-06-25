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
    Passthrough,
    _segment_order,
    extract_cross_scholar_passthroughs,
    passthrough_assertions,
)


def test_segment_order_unique_tiling_returns_order():
    """A surface that tiles under exactly ONE cluster order returns that order — the
    constituent (ordinal) ordering, left to right."""
    assert _segment_order("tunton", ["ctun", "cton"], {"ctun": {"tun"}, "cton": {"ton"}}) == (
        "ctun",
        "cton",
    )
    assert _segment_order("xyz", ["ca"], {"ca": {"q"}}) is None  # no tiling


def test_segment_order_ambiguous_tiling_left_separate():
    """Regression: when constituent clusters share folded reflex surfaces, a surface
    can tile under MORE THAN ONE order (both clusters carry 'tun' AND 'ton' → 'tunton'
    tiles as tun|ton OR ton|tun). The old code returned the lexicographically-smallest
    cluster-key permutation — an arbitrary 'ordinal' guess that would commit a WRONG
    split. Ambiguous → None (D46 leave-separate; a missed passthrough is harmless)."""
    out = _segment_order(
        "tunton", ["ctun", "cton"], {"ctun": {"tun", "ton"}, "cton": {"ton", "tun"}}
    )
    assert out is None  # was ('cton','ctun') — an arbitrary, possibly-wrong ordinal


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


def _fresh(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")
    return db


def test_no_passthrough_when_composite_surface_does_not_segment(tmp_path):
    # Gloss-blind guard: a coarse/fine count mismatch whose composite surface does
    # NOT word-break into the constituent reflexes is rejected (not mined). This is
    # what stops barton -> bar+ton: only a surface that actually tiles is recorded.
    db = _fresh(tmp_path)
    spec = _etymon(db, "Spec")
    p1 = _etymon(db, "aa")
    p2 = _etymon(db, "bb")
    comp = _etymon(db, "zzz")  # 'zzz' cannot tile into 'aa' + 'bb'
    tid = db.conn.execute(
        "INSERT INTO toponym (modern_name, country) VALUES ('Zzztown', 'England')"
    ).lastrowid
    _breakdown(db, tid, [spec, comp])  # coarse
    _breakdown(db, tid, [spec, p1, p2])  # fine
    db.commit()
    passthroughs = extract_cross_scholar_passthroughs(db)
    db.close()
    assert passthroughs == []


def test_support_two_yields_high_confidence(tmp_path):
    # Two toponyms attest ington -> ing+tūn -> support 2 -> pooled into one record,
    # confidence 'high' (single attestation would be 'medium').
    db = _fresh(tmp_path)
    ing = _etymon(db, "ing")
    tun = _etymon(db, "tūn")
    ington = _etymon(db, "ington")
    _reflex(db, "ton", tun)
    for name in ("Aldington", "Bilington"):
        spec = _etymon(db, name[:3])
        tid = db.conn.execute(
            "INSERT INTO toponym (modern_name, country) VALUES (?, 'England')", (name,)
        ).lastrowid
        _breakdown(db, tid, [spec, ington])
        _breakdown(db, tid, [spec, ing, tun])
    db.commit()
    passthroughs = extract_cross_scholar_passthroughs(db)
    db.close()
    assert len(passthroughs) == 1
    assert passthroughs[0].support == 2
    assertions = passthrough_assertions(passthroughs, source="t")
    assert assertions and all(a.confidence == "high" for a in assertions)


def test_self_loop_constituent_is_skipped():
    # If a representative composite etymon coincides with a constituent etymon,
    # passthrough_assertions skips that edge (D50 no-self-loop) instead of letting
    # validate() raise.
    p = Passthrough(
        composite_cluster="c1",
        composite_etymon=7,
        constituent_clusters=("c2", "c3"),
        constituent_etymons=(7, 9),  # 7 == the composite etymon -> self-loop
        composite_form="x",
        constituent_forms=("a", "b"),
        support=1,
        sample_toponyms=("T",),
    )
    assertions = passthrough_assertions([p], source="t")
    assert len(assertions) == 1
    assert assertions[0].object.ref == "9"


def test_build_passthrough_map(tmp_path):
    # The grader bridge: composite cluster -> ordered constituent clusters.
    from wyrd.generators.kenning.lexicon.passthrough_mining import build_passthrough_map

    init_schema(tmp_path / "lexicon.db")
    db, ids = _world(tmp_path)
    index = load_cluster_index(db)
    passthroughs = extract_cross_scholar_passthroughs(db, index=index)
    pmap = build_passthrough_map(passthroughs)
    db.close()

    assert pmap == {
        index.cluster_of[ids["ington"]]: (
            index.cluster_of[ids["ing"]],
            index.cluster_of[ids["tun"]],
        )
    }


def test_mine_passthroughs_cli_authors_composed_of(tmp_path):
    from click.testing import CliRunner

    from wyrd.generators.kenning.canonicalization import load_assertions
    from wyrd.generators.kenning.cli.lexicon import mine_passthroughs as cli_mod

    init_schema(tmp_path / "lexicon.db")
    db, ids = _world(tmp_path)
    db.close()
    mining = tmp_path / "mining"
    mining.mkdir()

    result = CliRunner().invoke(
        cli_mod.lexicon_mine_passthroughs,
        ["--db", str(tmp_path / "lexicon.db"), "--mining-dir", str(mining), "--apply"],
    )
    assert result.exit_code == 0, result.output

    composed = [a for a in load_assertions(mining) if a.predicate == "composed-of"]
    assert len(composed) == 2  # ington composed-of ing (ord 0) + tūn (ord 1)
    assert all(a.subject.ref == str(ids["ington"]) for a in composed)
    assert {a.qualifiers["ordinal"]: a.object.ref for a in composed} == {
        0: str(ids["ing"]),
        1: str(ids["tun"]),
    }

    # Idempotent: a second --apply writes nothing new.
    again = CliRunner().invoke(
        cli_mod.lexicon_mine_passthroughs,
        ["--db", str(tmp_path / "lexicon.db"), "--mining-dir", str(mining), "--apply"],
    )
    assert again.exit_code == 0
    assert "wrote 0" in again.output


def test_build_passthrough_map_highest_support_wins():
    # A composite cluster with two attested tilings → the HIGHER-support one wins,
    # regardless of input order (rebuild-stable, not last-write-wins).
    from wyrd.generators.kenning.lexicon.passthrough_mining import (
        Passthrough,
        build_passthrough_map,
    )

    weak = Passthrough("cC", 1, ("cA", "cB"), (2, 3), "ington", ("ing", "ton"), 1, ())
    strong = Passthrough("cC", 1, ("cX", "cY"), (4, 5), "ington", ("i", "ngton"), 9, ())
    assert build_passthrough_map([weak, strong])["cC"] == ("cX", "cY")
    assert build_passthrough_map([strong, weak])["cC"] == ("cX", "cY")  # order-independent


def test_mine_passthroughs_dry_run_writes_nothing(tmp_path):
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon import mine_passthroughs as cli_mod

    init_schema(tmp_path / "lexicon.db")
    db, _ = _world(tmp_path)
    db.close()
    mining = tmp_path / "mining"
    mining.mkdir()
    result = CliRunner().invoke(
        cli_mod.lexicon_mine_passthroughs,
        ["--db", str(tmp_path / "lexicon.db"), "--mining-dir", str(mining)],  # no --apply
    )
    assert result.exit_code == 0
    assert "dry-run" in result.output
    assert not (mining / "canonicalization").exists()  # nothing authored
