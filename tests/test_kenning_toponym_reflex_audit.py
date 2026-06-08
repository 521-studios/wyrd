"""wyrd-1wv2 — per-form toponym-reflex audit (era-grid over-merge long-tail cleaner)."""

from __future__ import annotations

import sqlite3

import pytest

from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.schema import init_schema
from wyrd.generators.kenning.lexicon.toponym_reflex_audit import (
    LLM_TOPONYM_REFLEX_DETACH_METHOD,
    ToponymReflexCandidate,
    ToponymReflexVerdict,
    _surface_similar,
    audit_log_row,
    detach_row,
    detect_toponym_reflex_candidates,
    parse_verdict,
    verdict_from_log,
)


def _mk(db, form, lang):
    return db.upsert_etymon(form, lang)


def _cluster(db, *ids):
    for eid in ids:
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (ids[0], eid))


def _edge(db, parent, child, edge_type="borrowing"):
    db.conn.execute(
        "INSERT INTO etymon_descent(parent_id, child_id, edge_type, source_id) VALUES (?,?,?,'wk')",
        (parent, child, edge_type),
    )


@pytest.fixture
def lex(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    db.conn.row_factory = sqlite3.Row
    db.upsert_source(id="wk", title="Wiktionary")
    db.commit()
    try:
        yield db
    finally:
        db.close()


def test_surface_similar():
    assert _surface_similar("vale", "val")  # shared prefix
    assert _surface_similar("home", "hām")  # ratio
    assert not _surface_similar("vulva", "valley")
    assert not _surface_similar("écuelle", "ess")


def test_detect_flags_dissimilar_modern_keeps_similar(lex):
    """A modern form surface-dissimilar to every English-family morpheme in its
    cluster is a candidate; a surface-similar one (likely a real reflex) is auto-kept."""
    ham = _mk(lex, "hām", "old-english")  # English-family morpheme
    home = _mk(lex, "home", "modern-english")  # surface-similar -> auto-kept
    vulva = _mk(lex, "vulva", "modern-english")  # dissimilar -> candidate
    lat = _mk(lex, "vulva", "latin")
    _cluster(lex, ham, home, vulva, lat)
    _edge(lex, lat, vulva)  # bridging parent for vulva
    _edge(lex, ham, home)
    lex.commit()
    refs = {c.reflex_ref for c in detect_toponym_reflex_candidates(lex.conn)}
    assert "modern-english:vulva" in refs
    assert "modern-english:home" not in refs  # surface-similar to hām -> auto-kept


def test_cluster_without_english_morpheme_skipped(lex):
    a = _mk(lex, "alpha", "latin")
    m = _mk(lex, "zeta", "modern-english")
    _cluster(lex, a, m)
    _edge(lex, a, m)
    lex.commit()
    assert detect_toponym_reflex_candidates(lex.conn) == []


def test_detach_row_and_thresholds():
    row = {
        "reflex_ref": "modern-english:vulva",
        "bridging_parents": ["latin:vulva"],
        "toponymic": False,
        "confidence": "high",
        "reason": "anatomical",
    }
    assert detach_row(row, "medium") == {
        "_type": "collapse",
        "ref": "modern-english:vulva",
        "detach_parents": ["latin:vulva"],
        "method": LLM_TOPONYM_REFLEX_DETACH_METHOD,
        "confidence": "high",
        "reason": "anatomical",
    }
    keep = {**row, "toponymic": True}
    assert detach_row(keep, "medium") is None
    low = {**row, "confidence": "low"}
    assert detach_row(low, "medium") is None
    assert detach_row(low, "low") is not None


def test_verdict_parse_roundtrip():
    assert parse_verdict({"toponymic": False, "confidence": "high", "reason": "x"}) == (
        ToponymReflexVerdict(toponymic=False, confidence="high", reason="x")
    )
    assert parse_verdict({"toponymic": "yes"}).toponymic is True
    assert parse_verdict({"confidence": "high"}) is None
    c = ToponymReflexCandidate("modern-english:vulva", "vulva", ("latin:vulva",))
    v = ToponymReflexVerdict(toponymic=False, confidence="high", reason="x")
    assert verdict_from_log(audit_log_row(c, v)) == v


def test_protected_element_never_detached():
    """A known place-name element is kept even with a non-toponymic detach verdict."""
    row = {
        "reflex_ref": "modern-english:home",
        "bridging_parents": ["old-english:hām"],
        "toponymic": False,
        "confidence": "high",
        "reason": "common noun",
    }
    assert detach_row(row, "medium") is None  # protected
    row2 = {**row, "reflex_ref": "modern-english:vulva"}
    assert detach_row(row2, "medium") is not None  # not protected -> detached
