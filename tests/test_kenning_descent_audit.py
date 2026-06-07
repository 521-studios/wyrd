"""wyrd-rd69 — descent-edge audit: screen bridge edges in over-merged cognate
clusters, judge, and emit edge-detach rows.

Real-DB (D10). The detector must (a) flag the bridge edges that link a
non-dominant sense-group into an incoherent cluster (the proto fan-out + direct
cross-sense cases), (b) leave coherent clusters alone, and (c) the detach row
must target the right ``parent -> child`` edge for ``apply_collapses``.
"""

from __future__ import annotations

import sqlite3

import pytest

from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.descent_audit import (
    LLM_DESCENT_AUDIT_DETACH_METHOD,
    DescentAuditVerdict,
    build_descent_judge_prompt,
    detach_row,
    detect_descent_audit_candidates,
    parse_descent_verdict,
)
from wyrd.generators.kenning.lexicon.schema import init_schema


def _mk(db: LexiconDB, form: str, lang: str, *, gloss: str | None = None) -> int:
    eid = db.upsert_etymon(form, lang)
    if gloss:
        db.add_gloss(eid, gloss)
    return eid


def _cluster(db: LexiconDB, *ids: int) -> None:
    """Put ``ids`` in one cognate cluster (cognate_id = first id — a real etymon
    row, as the FK requires)."""
    rep = ids[0]
    for eid in ids:
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (rep, eid))


def _edge(db: LexiconDB, parent: int, child: int, edge_type: str = "inheritance") -> None:
    db.conn.execute(
        "INSERT INTO etymon_descent(parent_id, child_id, edge_type, source_id) "
        "VALUES (?, ?, ?, 'wiktionary')",
        (parent, child, edge_type),
    )


@pytest.fixture
def lex(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    db.conn.row_factory = sqlite3.Row
    # the synthetic source the descent edges reference
    db.upsert_source(id="wiktionary", title="Wiktionary")
    db.commit()
    try:
        yield db
    finally:
        db.close()


def test_proto_fanout_bridge_is_flagged_dominant_kept(lex):
    """An un-glossed proto conductor fanning out to two sense-groups: edges into
    the NON-dominant group are candidates; the edge into the dominant (larger)
    group is not."""
    proto = _mk(lex, "*wolwumen", "itc-pro")  # glossless conductor
    vallis = _mk(lex, "vallis", "latin", gloss="a valley")
    val = _mk(lex, "val", "old-french", gloss="valley")
    vale = _mk(lex, "vale", "middle-english", gloss="valley")
    vulva = _mk(lex, "vulva", "latin", gloss="vulva")
    _cluster(lex, proto, vallis, val, vale, vulva)
    _edge(lex, proto, vallis)
    _edge(lex, proto, vulva)
    _edge(lex, vallis, val)
    _edge(lex, vallis, vale)
    lex.commit()

    cands = detect_descent_audit_candidates(lex.conn)
    keys = {c.edge_key for c in cands}
    # the valley group {vallis,val,vale} is dominant (3 > 1); the conductor->vulva
    # bridge is the candidate, conductor->vallis (dominant) is not.
    assert "itc-pro:*wolwumen->latin:vulva" in keys
    assert "itc-pro:*wolwumen->latin:vallis" not in keys
    vulva_cand = next(c for c in cands if c.child_ref == "latin:vulva")
    assert vulva_cand.child_glosses == ("vulva",)
    assert "valley" in " ".join(vulva_cand.cluster_dominant_glosses)


def test_direct_cross_sense_edge_is_flagged(lex):
    """A direct edge between two glossed members in different sense-groups."""
    ham = _mk(lex, "hām", "old-english", gloss="home homestead")
    home = _mk(lex, "home", "modern-english", gloss="home")
    home2 = _mk(lex, "ham", "modern-english", gloss="homestead")
    knee = _mk(lex, "hamme", "old-english", gloss="back of the knee")
    _cluster(lex, ham, home, home2, knee)
    _edge(lex, ham, home)
    _edge(lex, ham, home2)
    _edge(lex, ham, knee)  # spurious: knee is a different sense
    lex.commit()

    cands = detect_descent_audit_candidates(lex.conn)
    keys = {c.edge_key for c in cands}
    assert "old-english:hām->old-english:hamme" in keys  # the knee bridge
    assert "old-english:hām->modern-english:home" not in keys  # same dominant sense


def test_coherent_cluster_yields_no_candidates(lex):
    """A cluster whose glossed members all share a sense → not over-merged."""
    ham = _mk(lex, "hām", "old-english", gloss="home village")
    home = _mk(lex, "home", "modern-english", gloss="home dwelling")
    heim = _mk(lex, "heim", "german", gloss="home")
    _cluster(lex, ham, home, heim)
    _edge(lex, ham, home)
    _edge(lex, ham, heim)
    lex.commit()
    assert detect_descent_audit_candidates(lex.conn) == []


def test_detach_row_targets_parent_child_edge():
    v = DescentAuditVerdict(related=False, confidence="high", reason="valley != vulva")
    row = detach_row("itc-pro:*wolwumen", "latin:vulva", v, "medium")
    assert row == {
        "_type": "collapse",
        "ref": "latin:vulva",  # the 'from' (child)
        "detach_parents": ["itc-pro:*wolwumen"],  # removes parent -> child
        "method": LLM_DESCENT_AUDIT_DETACH_METHOD,
        "confidence": "high",
        "reason": "valley != vulva",
    }


def test_detach_row_skips_kept_and_below_threshold():
    kept = DescentAuditVerdict(related=True, confidence="high", reason="same")
    assert detach_row("a:x", "b:y", kept, "medium") is None
    low = DescentAuditVerdict(related=False, confidence="low", reason="maybe")
    assert detach_row("a:x", "b:y", low, "medium") is None  # below threshold
    assert detach_row("a:x", "b:y", low, "low") is not None  # at threshold


def test_verdict_parse_coercion():
    assert parse_descent_verdict({"same_family": False, "confidence": "high", "reason": "x"}) == (
        DescentAuditVerdict(related=False, confidence="high", reason="x")
    )
    assert parse_descent_verdict({"same_family": "yes"}).related is True
    assert parse_descent_verdict({"confidence": "high"}) is None  # missing verdict
    assert parse_descent_verdict("nope") is None
    # unknown confidence degrades to low
    assert parse_descent_verdict({"same_family": True, "confidence": "bogus"}).confidence == "low"


def test_judge_prompt_includes_forms_and_dominant_sense(lex):
    proto = _mk(lex, "*wolwumen", "itc-pro")
    vallis = _mk(lex, "vallis", "latin", gloss="valley")
    val = _mk(lex, "val", "old-french", gloss="valley")
    vulva = _mk(lex, "vulva", "latin", gloss="vulva")
    _cluster(lex, proto, vallis, val, vulva)
    _edge(lex, proto, vallis)
    _edge(lex, proto, vulva)
    lex.commit()
    cand = next(
        c for c in detect_descent_audit_candidates(lex.conn) if c.child_ref == "latin:vulva"
    )
    system, user = build_descent_judge_prompt(cand)
    assert "same_family" in system
    assert "vulva" in user and "*wolwumen" in user
    assert "valley" in user  # dominant sense context
