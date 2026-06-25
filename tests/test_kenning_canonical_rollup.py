"""Tests for `_build_canonical_rollup` — the identity rollup read from the
authoritative `canonical_morpheme` layer (wyrd-b2mf reader cutover).

The load-bearing property: TODAY it produces the SAME partition + representative
roots as the legacy `build_family_rollup` (because `canonical_morpheme_id`
reproduces the legacy rollup — the u6fn.4 fidelity gate), so switching the export
to it is byte-equivalent. Once 1a's binds merge families, the canonical rollup
reflects the merge and the legacy one does not.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.canonicalization import (
    Assertion,
    NodeRef,
    append_assertion,
    mint_canonical_id,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.bundle._export import (
    _build_canonical_rollup,
    _canonical_hub_root,
    build_family_rollup,
)
from wyrd.generators.kenning.lexicon.canonicalization_projection import project_canonical
from wyrd.generators.kenning.lexicon.etymon_refs import etymon_ref


def _db(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    return LexiconDB(tmp_path / "lexicon.db")


def _etymon(db, form, *, language="old-english"):
    return db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    ).lastrowid


def _set_merged(db, loser, winner):
    db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (winner, loser))


def _set_lemma(db, inflection, lemma):
    db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (lemma, inflection))


def _partition(members_by_root):
    return {frozenset(v) for v in members_by_root.values()}


def _families(tmp_path):
    """A DB with an OCR+lemma family, an OCR family, and a singleton; the
    canonical graph populated by the fold."""
    db = _db(tmp_path)
    tun = _etymon(db, "tun")
    tvn = _etymon(db, "Tun")
    _set_merged(db, tvn, tun)
    tuns = _etymon(db, "tuns")
    _set_lemma(db, tuns, tun)
    ham = _etymon(db, "ham")
    hamm = _etymon(db, "hamm")
    _set_merged(db, hamm, ham)
    _etymon(db, "ford")  # singleton
    db.commit()
    project_canonical(db, mining_dir=tmp_path, apply=True, fold_legacy=True)
    return db, {"tun": tun, "ham": ham}


def test_canonical_rollup_matches_legacy_today(tmp_path):
    db, _ = _families(tmp_path)
    legacy_m, legacy_root = build_family_rollup(db)
    canon_m, canon_root = _build_canonical_rollup(db)

    # Same partition: identical sets of family member-sets.
    assert _partition(canon_m) == _partition(legacy_m)
    # Same representative for every etymon.
    all_etymons = [e for members in legacy_m.values() for e in members]
    assert all_etymons  # sanity
    for e in all_etymons:
        assert canon_root(e) == legacy_root(e), f"root mismatch for {e}"
    db.close()


def test_singleton_is_its_own_family(tmp_path):
    db, ids = _families(tmp_path)
    canon_m, canon_root = _build_canonical_rollup(db)
    ford = db.conn.execute("SELECT id FROM etymon WHERE canonical_form='ford'").fetchone()["id"]
    assert canon_root(ford) == ford  # unbound -> its own family
    assert canon_m[ford] == [ford]
    db.close()


def test_canonical_rollup_reflects_a_bind_legacy_does_not(tmp_path):
    # The realistic 1a scenario: a breakdown-only etymon (its own legacy singleton)
    # is bound to a shipped hub. The canonical rollup must fold it into that family;
    # the legacy rollup keeps it separate. (Binding an already-fold-bound etymon to
    # a different hub is instead a D46 conflict — left unbound — not a merge.)
    db, ids = _families(tmp_path)
    extra = _etymon(db, "Tunbreakdown")  # singleton: no merged_into / lemma
    db.commit()
    tun_hub = mint_canonical_id("canonical_morpheme", "old-english", "tun")
    append_assertion(
        tmp_path,
        Assertion(
            predicate="bind",
            subject=NodeRef("etymon", etymon_ref("old-english", "Tunbreakdown")),
            object=NodeRef("canonical_morpheme", tun_hub),
            qualifiers={"kind": "same-morpheme"},
            confidence="high",
        ),
    )
    project_canonical(db, mining_dir=tmp_path, apply=True, fold_legacy=True)

    _, legacy_root = build_family_rollup(db)
    _, canon_root = _build_canonical_rollup(db)
    assert legacy_root(extra) == extra  # legacy: its own singleton
    assert canon_root(extra) == canon_root(ids["tun"])  # canonical: joined tun's family
    db.close()


def test_falls_back_to_legacy_when_canonical_graph_unpopulated(tmp_path):
    # No fold/projection run -> canonical_morpheme_id all NULL. The export must
    # stay usable: fall back to the legacy rollup (full clustering -> never
    # under-clusters) with a warning, NOT raise. (Production always projects.)
    db = _db(tmp_path)
    a = _etymon(db, "tun")
    b = _etymon(db, "Tun")
    _set_merged(db, b, a)
    db.commit()
    with pytest.warns(UserWarning, match="unpopulated"):
        canon_m, canon_root = _build_canonical_rollup(db)
    legacy_m, legacy_root = build_family_rollup(db)
    assert _partition(canon_m) == _partition(legacy_m)  # identical to legacy
    assert canon_root(b) == legacy_root(b) == a
    db.close()


def _reflex_link(db, parent, child):
    db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('collapse', 'collapse')")
    db.conn.execute(
        "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id, confidence) "
        "VALUES (?, ?, 'inheritance', 'collapse', 'high')",
        (parent, child),
    )


def test_merges_two_multi_member_families(tmp_path):
    # The load-bearing post-1a case: TWO genuine multi-member legacy families,
    # joined by a merge-canonical between their hubs. The canonical rollup must
    # UNION all members under one family and pick the min-content-key legacy root.
    from wyrd.generators.kenning.canonicalization import Assertion, NodeRef, append_assertion

    db = _db(tmp_path)
    tun = _etymon(db, "tun")
    tvn = _etymon(db, "Tun")
    _set_merged(db, tvn, tun)
    tuns = _etymon(db, "tuns")
    _set_lemma(db, tuns, tun)
    ham = _etymon(db, "ham")
    hamm = _etymon(db, "hamm")
    _set_merged(db, hamm, ham)
    db.commit()
    project_canonical(db, mining_dir=tmp_path, apply=True, fold_legacy=True)

    tun_hub = mint_canonical_id("canonical_morpheme", "old-english", "tun")
    ham_hub = mint_canonical_id("canonical_morpheme", "old-english", "ham")
    append_assertion(
        tmp_path,
        Assertion(
            predicate="merge-canonical",
            subject=NodeRef("canonical_morpheme", ham_hub),
            object=NodeRef("canonical_morpheme", tun_hub),
        ),
    )
    project_canonical(db, mining_dir=tmp_path, apply=True, fold_legacy=True)

    canon_m, canon_root = _build_canonical_rollup(db)
    # Both families collapse to one canonical family containing all 5 members.
    assert canon_root(tun) == canon_root(ham)
    fam = canon_m[canon_root(tun)]
    assert set(fam) == {tun, tvn, tuns, ham, hamm}
    # Representative is the min-content-key legacy root ("ham" < "tun").
    assert canon_root(tun) == ham
    db.close()


def test_reflex_link_member_rolls_to_ancestor_canonically(tmp_path):
    # A rogd.9 reflex (bishop) joined to its ancestor (biscop) via a reflex-link
    # inheritance edge must roll to the ancestor under the CANONICAL rollup too.
    db = _db(tmp_path)
    biscop = _etymon(db, "biscop")
    bishop = _etymon(db, "bishop", language="modern-english")
    _reflex_link(db, biscop, bishop)
    db.commit()
    project_canonical(db, mining_dir=tmp_path, apply=True, fold_legacy=True)

    canon_m, canon_root = _build_canonical_rollup(db)
    assert canon_root(bishop) == canon_root(biscop)
    assert bishop in canon_m[canon_root(biscop)]
    db.close()


def test_content_keys_chunks_beyond_999(tmp_path):
    # _content_keys chunks at 900 to dodge the SQLite host-param cap; exercise the
    # boundary so a dropped trailing chunk can't ship green.
    from wyrd.generators.kenning.lexicon.bundle._export import _content_keys

    db = _db(tmp_path)
    ids = {_etymon(db, f"w{i}") for i in range(1000)}
    db.commit()
    keys = _content_keys(db, ids)
    assert set(keys) == ids  # every id returned across the chunk boundary
    assert all(lang == "old-english" for lang, _form in keys.values())
    db.close()


def test_export_subjects_identical_canonical_vs_legacy(tmp_path):
    # End-to-end: export_meanings (which drives _collect_families -> promotion via
    # the canonical root_of) yields IDENTICAL subjects whether the rollup is
    # canonical or legacy — the unit-scale mirror of the live byte-identical proof.
    import json

    import wyrd.generators.kenning.lexicon.bundle._export as exp
    from wyrd.generators.kenning.lexicon import export_meanings

    db = _db(tmp_path)
    tun = _etymon(db, "tun")
    tvn = _etymon(db, "Tun")
    _set_merged(db, tvn, tun)
    for src in ("a", "b"):  # 2 witnesses -> promoted
        db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES (?, ?)", (src, src))
        db.conn.execute(
            "INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, ?)", (tun, src)
        )
    ham = _etymon(db, "ham")  # a second, single-witness family
    db.conn.execute("INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, 'a')", (ham,))
    db.commit()
    project_canonical(db, mining_dir=tmp_path, apply=True, fold_legacy=True)

    canon = json.dumps(export_meanings(db), sort_keys=True, ensure_ascii=False)
    orig = exp._build_canonical_rollup
    try:
        exp._build_canonical_rollup = exp.build_family_rollup
        legacy = json.dumps(export_meanings(db), sort_keys=True, ensure_ascii=False)
    finally:
        exp._build_canonical_rollup = orig
    db.close()
    assert canon == legacy


def test_canonical_hub_root_walks_chain_and_guards_cycles():
    """``_canonical_hub_root`` (the merged_into walk extracted from
    _build_canonical_rollup) follows the chain to the surviving hub, treats a
    no/empty-target node as its own hub, and breaks on a cycle instead of
    looping forever."""
    # Chain a -> b -> c: every node resolves to the terminal hub c.
    chain = {"a": "b", "b": "c"}
    assert _canonical_hub_root(chain, "a") == "c"
    assert _canonical_hub_root(chain, "b") == "c"
    assert _canonical_hub_root(chain, "c") == "c"  # no merged_into target -> own hub
    # A node absent from the map is its own hub.
    assert _canonical_hub_root(chain, "z") == "z"
    # Falsy targets (None / empty string) terminate the walk at that node.
    assert _canonical_hub_root({"x": None}, "x") == "x"
    assert _canonical_hub_root({"p": "q", "q": ""}, "p") == "q"
    # Cycle guard: a <-> b and a 3-cycle both terminate (no infinite loop).
    assert _canonical_hub_root({"a": "b", "b": "a"}, "a") in {"a", "b"}
    assert _canonical_hub_root({"a": "b", "b": "c", "c": "a"}, "a") in {"a", "b", "c"}
