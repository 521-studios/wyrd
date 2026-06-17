"""Tests for `_build_canonical_rollup` — the identity rollup read from the
authoritative `canonical_morpheme` layer (wyrd-b2mf reader cutover).

The load-bearing property: TODAY it produces the SAME partition + representative
roots as the legacy `_build_family_rollup` (because `canonical_morpheme_id`
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
    _build_family_rollup,
)
from wyrd.generators.kenning.lexicon.canonicalization_projection import project_canonical


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
    legacy_m, legacy_root = _build_family_rollup(db)
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
            subject=NodeRef("etymon", str(extra)),
            object=NodeRef("canonical_morpheme", tun_hub),
            qualifiers={"kind": "same-morpheme"},
            confidence="high",
        ),
    )
    project_canonical(db, mining_dir=tmp_path, apply=True, fold_legacy=True)

    _, legacy_root = _build_family_rollup(db)
    _, canon_root = _build_canonical_rollup(db)
    assert legacy_root(extra) == extra  # legacy: its own singleton
    assert canon_root(extra) == canon_root(ids["tun"])  # canonical: joined tun's family
    db.close()


def test_guard_raises_when_canonical_graph_unpopulated(tmp_path):
    # No fold/projection run -> canonical_morpheme_id all NULL -> guard, not
    # silent under-clustering.
    db = _db(tmp_path)
    a = _etymon(db, "tun")
    b = _etymon(db, "Tun")
    _set_merged(db, b, a)
    db.commit()
    with pytest.raises(RuntimeError, match="unpopulated"):
        _build_canonical_rollup(db)
    db.close()
