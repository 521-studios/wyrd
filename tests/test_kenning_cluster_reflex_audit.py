"""wyrd-1wv2 — cluster modern-reflex sense-audit (the working era-grid fix).

Real-DB (D10). The detector flags a modern reflex that doesn't match its
cluster's dominant glossed sense (the val->vulva class), auto-keeps on-sense
reflexes via the gloss pre-screen, and the detach row orphans the leaf by
cutting its in-cluster bridging parents.
"""

from __future__ import annotations

import sqlite3

import pytest

from wyrd.generators.kenning.lexicon.cluster_reflex_audit import (
    LLM_CLUSTER_REFLEX_DETACH_METHOD,
    ClusterReflexVerdict,
    audit_log_row,
    build_judge_prompt,
    detach_row,
    detect_cluster_reflex_candidates,
    parse_verdict,
    verdict_from_log,
)
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.schema import init_schema


def _mk(db, form, lang, *, gloss=None):
    eid = db.upsert_etymon(form, lang)
    if gloss:
        db.add_gloss(eid, gloss)
    return eid


def _cluster(db, *ids):
    rep = ids[0]
    for eid in ids:
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (rep, eid))


def _edge(db, parent, child, edge_type="borrowing"):
    db.conn.execute(
        "INSERT INTO etymon_descent(parent_id, child_id, edge_type, source_id) VALUES (?,?,?,'wiktionary')",
        (parent, child, edge_type),
    )


@pytest.fixture
def lex(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    db.conn.row_factory = sqlite3.Row
    db.upsert_source(id="wiktionary", title="Wiktionary")
    db.commit()
    try:
        yield db
    finally:
        db.close()


def _valley_cluster(lex):
    """A 'valley' cluster with two modern members: vale (on-sense) + vulva
    (glossless, suspect, reached via latin:vulva)."""
    vallis = _mk(lex, "vallis", "latin", gloss="valley")
    val = _mk(lex, "val", "old-french", gloss="valley")
    vale = _mk(lex, "vale", "modern-english", gloss="valley")
    lvulva = _mk(lex, "vulva", "latin")  # deep parent (glossless)
    mvulva = _mk(lex, "vulva", "modern-english")  # the suspect modern leaf (glossless)
    _cluster(lex, vallis, val, vale, lvulva, mvulva)
    _edge(lex, lvulva, mvulva, "borrowing")  # latin:vulva -> modern:vulva
    lex.commit()
    return mvulva


def test_detect_flags_offsense_modern_reflex_keeps_onsense(lex):
    _valley_cluster(lex)
    cands = detect_cluster_reflex_candidates(lex.conn)
    refs = {c.reflex_ref for c in cands}
    assert "modern-english:vulva" in refs  # off-sense + glossless -> suspect
    assert "modern-english:vale" not in refs  # shares the dominant 'valley' sense -> auto-kept
    vulva = next(c for c in cands if c.reflex_ref == "modern-english:vulva")
    assert vulva.bridging_parents == ("latin:vulva",)  # the edge to cut
    assert "valley" in " ".join(vulva.dominant_glosses)


def test_lone_modern_member_not_screened(lex):
    # only one modern member -> below _MIN_MODERN_MEMBERS
    vallis = _mk(lex, "vallis", "latin", gloss="valley")
    val = _mk(lex, "val", "old-french", gloss="valley")
    mvulva = _mk(lex, "vulva", "modern-english")
    lvulva = _mk(lex, "vulva", "latin")
    _cluster(lex, vallis, val, mvulva, lvulva)
    _edge(lex, lvulva, mvulva)
    lex.commit()
    assert detect_cluster_reflex_candidates(lex.conn) == []


def test_no_clear_dominant_sense_not_screened(lex):
    # 1-1 glossed split -> no unique dominant
    a = _mk(lex, "alpha", "latin", gloss="fire")
    b = _mk(lex, "beta", "latin", gloss="water")
    m1 = _mk(lex, "m1", "modern-english")
    m2 = _mk(lex, "m2", "modern-english")
    p = _mk(lex, "p", "latin")
    _cluster(lex, a, b, m1, m2, p)
    _edge(lex, p, m1)
    _edge(lex, p, m2)
    lex.commit()
    assert detect_cluster_reflex_candidates(lex.conn) == []


def test_detach_row_targets_bridging_parents():
    row = {
        "reflex_ref": "modern-english:vulva",
        "bridging_parents": ["latin:vulva"],
        "reflex_of_sense": False,
        "confidence": "high",
        "reason": "vulva is not a valley reflex",
    }
    d = detach_row(row, "medium")
    assert d == {
        "_type": "collapse",
        "ref": "modern-english:vulva",
        "detach_parents": ["latin:vulva"],
        "method": LLM_CLUSTER_REFLEX_DETACH_METHOD,
        "confidence": "high",
        "reason": "vulva is not a valley reflex",
    }


def test_detach_row_skips_kept_and_below_threshold():
    kept_row = {
        "reflex_ref": "a:b",
        "bridging_parents": ["c:d"],
        "reflex_of_sense": True,
        "confidence": "high",
        "reason": "x",
    }
    assert detach_row(kept_row, "medium") is None
    low_row = {
        "reflex_ref": "a:b",
        "bridging_parents": ["c:d"],
        "reflex_of_sense": False,
        "confidence": "low",
        "reason": "x",
    }
    assert detach_row(low_row, "medium") is None
    assert detach_row(low_row, "low") is not None


def test_verdict_parse_and_roundtrip():
    assert parse_verdict({"reflex_of_sense": False, "confidence": "high", "reason": "r"}) == (
        ClusterReflexVerdict(plausible=False, confidence="high", reason="r")
    )
    assert parse_verdict({"reflex_of_sense": "yes"}).plausible is True
    assert parse_verdict({"confidence": "high"}) is None
    assert parse_verdict("nope") is None
    # round-trip through the log row
    from wyrd.generators.kenning.lexicon.cluster_reflex_audit import ClusterReflexCandidate

    c = ClusterReflexCandidate("modern-english:vulva", "vulva", ("valley",), ("latin:vulva",))
    v = ClusterReflexVerdict(plausible=False, confidence="high", reason="r")
    assert verdict_from_log(audit_log_row(c, v)) == v


def test_judge_prompt_has_form_and_sense(lex):
    _valley_cluster(lex)
    c = next(x for x in detect_cluster_reflex_candidates(lex.conn) if x.reflex_form == "vulva")
    system, user = build_judge_prompt(c)
    assert "reflex_of_sense" in system
    assert "vulva" in user and "valley" in user
