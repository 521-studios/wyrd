"""Tests for lexicon/etymon_refs (wyrd-c6wu) — stable natural-key etymon refs.

The round-trip (author → reassign ids → resolve) is proven end-to-end in
test_relational_projection; here we unit-test the encoder/resolver edges the
projection path doesn't exercise: the IN-chunk seam, missing-id omission, the
no-colon / not-found None branches, and the descent_assertions fail-loud contract.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.cognate_descent_mining import DescentEdge, descent_assertions
from wyrd.generators.kenning.lexicon.etymon_refs import (
    etymon_ref,
    etymon_refs_for,
    resolve_etymon_ref,
)


def _db(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    return LexiconDB(tmp_path / "lexicon.db")


def _etymon(db, form, language="old-english"):
    return db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    ).lastrowid


def test_etymon_ref_encoding():
    assert etymon_ref("old-english", "tun") == "old-english:tun"
    # A None language (caller-supplied; never from the NOT NULL column) coalesces to "".
    assert etymon_ref(None, "tun") == ":tun"
    # A canonical_form containing ":" is preserved — only the FIRST sep splits.
    assert etymon_ref("x", "a:b") == "x:a:b"


def test_resolve_round_trip_and_colon_in_form(tmp_path):
    db = _db(tmp_path)
    eid = _etymon(db, "tun", "old-english")
    weird = _etymon(db, "a:b", "x")  # canonical_form with an embedded ":"
    db.commit()
    assert resolve_etymon_ref(db.conn, etymon_ref("old-english", "tun")) == eid
    assert resolve_etymon_ref(db.conn, etymon_ref("x", "a:b")) == weird  # FIRST-sep split
    db.close()


def test_resolve_de_dashes_ref_form_keeping_interior_hyphen(tmp_path):
    """wyrd-aicu.8 (D45) read-path sweep: resolve_etymon_ref de-dashes a dashed
    ref form to the BARE-stored etymon, while PRESERVING an interior hyphen
    (al-Quadim must not collapse to alQuadim — the sweep's key invariant)."""
    db = _db(tmp_path)
    ach = _etymon(db, "ach", "celtic")  # bare-stored
    alq = _etymon(db, "al-Quadim", "arabic")  # interior hyphen
    db.commit()
    assert resolve_etymon_ref(db.conn, "celtic:-ach") == ach  # dashed ref → bare
    assert resolve_etymon_ref(db.conn, "arabic:al-Quadim") == alq  # interior kept
    db.close()


def test_resolve_none_branches(tmp_path):
    db = _db(tmp_path)
    _etymon(db, "tun", "old-english")
    db.commit()
    # No separator at all (e.g. a stray bare-int legacy ref) → None, never an int() crash.
    assert resolve_etymon_ref(db.conn, "12345") is None
    # Well-formed but not in this build → None.
    assert resolve_etymon_ref(db.conn, "old-english:ghost") is None
    # wyrd-s964: a legacy id int (the `str | int` input from a pre-natural-key
    # cache) resolves to None, never a TypeError on the `':' not in ref` test.
    assert resolve_etymon_ref(db.conn, 12345) is None
    db.close()


def test_etymon_refs_for_omits_missing_ids(tmp_path):
    db = _db(tmp_path)
    a, b = _etymon(db, "tun"), _etymon(db, "gar")
    db.commit()
    refs = etymon_refs_for(db.conn, {a, b, 999_999})  # 999999 doesn't exist
    assert refs == {a: "old-english:tun", b: "old-english:gar"}  # missing id omitted
    assert etymon_refs_for(db.conn, set()) == {}
    db.close()


def test_etymon_refs_for_chunks_over_sqlite_variable_cap(tmp_path):
    """>900 ids must cross the IN-chunk seam and still map every id (a chunk-seam
    bug would silently drop refs)."""
    db = _db(tmp_path)
    ids = [_etymon(db, f"m{i}") for i in range(2100)]  # > 2 chunks of 900
    db.commit()
    refs = etymon_refs_for(db.conn, ids)
    assert len(refs) == 2100
    assert refs[ids[0]] == "old-english:m0" and refs[ids[-1]] == "old-english:m2099"
    db.close()


def test_descent_assertions_raises_on_incomplete_refs():
    """The fail-loud contract: an endpoint id absent from ``refs`` is a programming
    error (the caller must cover every endpoint), surfaced as KeyError — never a
    silently broken ref."""
    edge = DescentEdge(child_etymon=1, cluster_root=2, confidence="medium", rationale="r")
    with pytest.raises(KeyError):
        descent_assertions([edge], {1: "old-english:tun"}, source="test")  # missing root=2
