"""wyrd-rd69 — descent-edge audit: screen bridge edges in over-merged cognate
clusters, judge, and emit edge-detach rows.

Real-DB (D10). The detector must (a) flag the bridge edges that link a
non-dominant sense-group into an incoherent cluster (the proto fan-out + direct
cross-sense cases), (b) leave coherent clusters alone, and (c) the detach row
must target the right ``parent -> child`` edge for ``apply_collapses``.
"""

from __future__ import annotations

import io
import json
import sqlite3

import pytest

from wyrd.generators.kenning.cli.lexicon import audit_descent as cli
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.descent_audit import (
    LLM_DESCENT_AUDIT_DETACH_METHOD,
    DescentAuditCandidate,
    DescentAuditVerdict,
    audit_log_row,
    build_descent_judge_prompt,
    detach_row,
    detect_descent_audit_candidates,
    parse_descent_verdict,
    verdict_from_log,
)
from wyrd.generators.kenning.lexicon.schema import init_schema


def _logrow(parent: str, child: str, *, same: bool, conf: str = "high") -> dict:
    """An ``audit_log_row``-shaped verdict row keyed for ``_emit_detaches``."""
    return {
        "_type": "descent_audit",
        "edge_key": f"{parent}->{child}",
        "parent_ref": parent,
        "child_ref": child,
        "edge_type": "inheritance",
        "same_family": same,
        "confidence": conf,
        "reason": "r",
    }


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


# --- detector: diamond + tie-break recall (reviewer P2) ---


def test_diamond_both_in_edges_of_intruder_are_flagged(lex):
    """A non-dominant member reachable by TWO paths (conductor + cross-sense)
    has BOTH in-edges flagged, so detaching them isolates it."""
    proto = _mk(lex, "*wolwumen", "itc-pro")
    vallis = _mk(lex, "vallis", "latin", gloss="valley")
    val = _mk(lex, "val", "old-french", gloss="valley")
    vulva = _mk(lex, "vulva", "latin", gloss="vulva")
    _cluster(lex, proto, vallis, val, vulva)
    _edge(lex, proto, vulva)  # conductor path
    _edge(lex, vallis, vulva)  # cross-sense path (valley -> vulva)
    _edge(lex, proto, vallis)
    _edge(lex, vallis, val)
    lex.commit()
    keys = {c.edge_key for c in detect_descent_audit_candidates(lex.conn)}
    assert "itc-pro:*wolwumen->latin:vulva" in keys
    assert "latin:vallis->latin:vulva" in keys


def test_even_split_screens_both_conductor_edges(lex):
    """On a 1-1 sense tie there is no dominant group, so EVERY conductor edge is
    screened (neither sense silently skipped)."""
    proto = _mk(lex, "*x", "ine-pro")
    a = _mk(lex, "alpha", "latin", gloss="fire flame")
    b = _mk(lex, "beta", "latin", gloss="water river")
    _cluster(lex, proto, a, b)
    _edge(lex, proto, a)
    _edge(lex, proto, b)
    lex.commit()
    cands = detect_descent_audit_candidates(lex.conn)
    keys = {c.edge_key for c in cands}
    assert "ine-pro:*x->latin:alpha" in keys
    assert "ine-pro:*x->latin:beta" in keys
    # On a tie there is no single dominant sense, so dom_glosses widens to the
    # union of ALL tied top sense-groups — pins _dominant_sense's tie path
    # (a `[top[0]]`/`[unique_dom]` would drop one sense or KeyError on None).
    assert set(cands[0].cluster_dominant_glosses) == {"fire flame", "water river"}


# --- round-trip (reviewer P2) ---


def test_audit_log_row_round_trips():
    cand = DescentAuditCandidate(
        parent_ref="a:x",
        child_ref="b:y",
        edge_type="inheritance",
        parent_form="x",
        child_form="y",
        parent_lang="a",
        child_lang="b",
        parent_glosses=(),
        child_glosses=("y",),
        cluster_dominant_glosses=("z",),
    )
    v = DescentAuditVerdict(related=False, confidence="high", reason="distinct")
    row = audit_log_row(cand, v)
    assert verdict_from_log(row) == v
    assert verdict_from_log({"same_family": "notbool"}) is None  # legacy/no raw verdict
    assert verdict_from_log({"same_family": True, "confidence": "bogus"}).confidence == "low"


# --- CLI helpers (reviewer P1) ---


def test_emit_detaches_groups_parents_into_one_row():
    """Two bad parents of the SAME child collapse into ONE detach row (last-write-
    wins safe), not two rows that would clobber each other on rebuild."""
    log = {
        "p1->c": _logrow("lat:p1", "lat:c", same=False),
        "p2->c": _logrow("lat:p2", "lat:c", same=False),
        "p3->d": _logrow("lat:p3", "lat:d", same=True),  # kept
    }
    fh = io.StringIO()
    counts = cli._emit_detaches(log, "medium", {}, set(), fh, dry_run=False)
    rows = [json.loads(line) for line in fh.getvalue().splitlines()]
    assert counts == {"detaches": 2, "kept": 1, "already": 0}
    assert len(rows) == 1  # one merged row for child 'c'
    assert rows[0]["ref"] == "lat:c"
    assert rows[0]["detach_parents"] == ["lat:p1", "lat:p2"]


def test_emit_detaches_preserves_existing_fold():
    """Merging a detach into a child that already has an into-fold preserves the
    fold (into / detach_children) rather than clobbering it."""
    log = {"p->c": _logrow("lat:p", "lat:c", same=False)}
    collapse_state = {
        "lat:c": {
            "_type": "collapse",
            "ref": "lat:c",
            "into": "lat:lemma",
            "detach_children": ["lat:z"],
        }
    }
    fh = io.StringIO()
    cli._emit_detaches(log, "medium", collapse_state, set(), fh, dry_run=False)
    row = json.loads(fh.getvalue())
    assert row["into"] == "lat:lemma"
    assert row["detach_children"] == ["lat:z"]
    assert row["detach_parents"] == ["lat:p"]


def test_emit_detaches_dedups_existing_and_thresholds():
    log = {
        "p1->c": _logrow("lat:p1", "lat:c", same=False, conf="high"),
        "p2->c": _logrow("lat:p2", "lat:c", same=False, conf="low"),  # below medium
    }
    fh = io.StringIO()
    # p1 already detached in a prior run; p2 below threshold → nothing emitted.
    counts = cli._emit_detaches(log, "medium", {}, {("lat:c", "lat:p1")}, fh, dry_run=False)
    assert counts == {"detaches": 0, "kept": 1, "already": 1}
    assert fh.getvalue() == ""


def test_load_log_filters_and_skips_malformed(tmp_path):
    f = tmp_path / "_descent_audit.jsonl"
    f.write_text(
        "\n".join(
            [
                json.dumps(_logrow("a:p", "a:c", same=False)),
                json.dumps({"_type": "other", "edge_key": "x"}),  # wrong type
                json.dumps({"_type": "descent_audit", "edge_key": "y"}),  # no raw verdict
                "{not json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log = cli._load_log(f)
    assert set(log) == {"a:p->a:c"}
    assert cli._load_log(tmp_path / "absent.jsonl") == {}


def test_existing_detach_pairs_from_lww_state():
    """Pairs come from the LWW net collapse state (not a raw file scan), so a
    detach a later row dropped is correctly absent."""
    collapse_state = {
        "a:c": {"_type": "collapse", "ref": "a:c", "detach_parents": ["a:p1", "a:p2"]},
        "a:d": {"_type": "collapse", "ref": "a:d", "into": "a:e"},  # fold, no detach
    }
    assert cli._existing_detach_pairs(collapse_state) == {("a:c", "a:p1"), ("a:c", "a:p2")}


def test_judge_fresh_aborts_after_five_consecutive_failures(monkeypatch):
    import click

    monkeypatch.setattr(cli, "_judge_with_retry", lambda client, c: None)
    cands = [
        DescentAuditCandidate(
            parent_ref=f"a:p{i}",
            child_ref="a:c",
            edge_type="inheritance",
            parent_form="p",
            child_form="c",
            parent_lang="a",
            child_lang="a",
            parent_glosses=(),
            child_glosses=(),
            cluster_dominant_glosses=(),
        )
        for i in range(10)
    ]
    log: dict = {}
    # 5 consecutive failures (Ollama down) raises non-zero, not a silent exit-0.
    with pytest.raises(click.ClickException):
        cli._judge_fresh(None, cands, log, None, "url", "model")
    assert log == {}


def test_judge_fresh_resets_counter_on_success(monkeypatch):
    seq = iter(
        [
            None,
            None,
            DescentAuditVerdict(related=False, confidence="high", reason="x"),
            None,
            None,
            None,
            None,
        ]
    )
    monkeypatch.setattr(cli, "_judge_with_retry", lambda client, c: next(seq))
    cands = [
        DescentAuditCandidate(
            parent_ref=f"a:p{i}",
            child_ref="a:c",
            edge_type="inheritance",
            parent_form="p",
            child_form="c",
            parent_lang="a",
            child_lang="a",
            parent_glosses=(),
            child_glosses=(),
            cluster_dominant_glosses=(),
        )
        for i in range(7)
    ]
    log: dict = {}
    judged, skipped = cli._judge_fresh(None, cands, log, None, "url", "model")
    assert judged == 1  # the one success recorded; counter reset prevented early abort
    assert len(log) == 1


# --- Phase 2b: glossless-proto-conductor fan-out (wyrd-2wml) ---


def test_phase2b_flags_glossless_offsense_child_of_proto_conductor(lex):
    """A coherent-looking cluster (one glossed 'valley' sense) whose proto
    conductor also fans out to a GLOSSLESS off-sense child (vulva): 2b flags the
    conductor->vulva edge; 2a (gloss screen) does not see it."""
    proto = _mk(lex, "*wolwumen", "itc-pro")  # glossless conductor
    vallis = _mk(lex, "vallis", "latin", gloss="valley")
    val = _mk(lex, "val", "old-french", gloss="valley")
    vale = _mk(lex, "vale", "middle-english", gloss="valley")
    vulva = _mk(lex, "vulva", "latin")  # GLOSSLESS off-sense pollutant
    _cluster(lex, proto, vallis, val, vale, vulva)
    _edge(lex, proto, vallis)
    _edge(lex, proto, vale)
    _edge(lex, proto, vulva)  # conductor out-degree 3
    lex.commit()

    keys_2b = {c.edge_key for c in detect_descent_audit_candidates(lex.conn, scope="2b")}
    assert "itc-pro:*wolwumen->latin:vulva" in keys_2b
    assert "itc-pro:*wolwumen->latin:vallis" not in keys_2b  # into the dominant sense
    # 2a alone can't see it — the cluster is gloss-coherent (one sense-group).
    assert detect_descent_audit_candidates(lex.conn, scope="2a") == []
    # the candidate carries the dominant sense for the judge.
    cand = next(
        c
        for c in detect_descent_audit_candidates(lex.conn, scope="2b")
        if c.child_ref == "latin:vulva"
    )
    assert "valley" in " ".join(cand.cluster_dominant_glosses)


def test_phase2b_requires_proto_conductor_and_fanout(lex):
    """A non-proto parent, or a proto with out-degree < 3, is not a conductor."""
    # non-proto parent with a glossless off-sense child -> NOT flagged by 2b
    head = _mk(lex, "headword", "latin", gloss="valley")
    h2 = _mk(lex, "headtwo", "latin", gloss="valley")
    glossless = _mk(lex, "zzz", "latin")
    _cluster(lex, head, h2, glossless)
    _edge(lex, head, h2)
    _edge(lex, head, glossless)
    lex.commit()
    assert detect_descent_audit_candidates(lex.conn, scope="2b") == []

    # a proto conductor with only 2 children is below _MIN_PROTO_FANOUT
    proto = _mk(lex, "*x", "ine-pro")
    a = _mk(lex, "aaa", "latin", gloss="valley")
    b = _mk(lex, "bbb", "latin", gloss="valley")
    off = _mk(lex, "off", "latin")  # glossless off-sense
    _cluster(lex, proto, a, b, off)
    _edge(lex, proto, a)
    _edge(lex, proto, off)  # out-degree 2 only
    lex.commit()
    assert detect_descent_audit_candidates(lex.conn, scope="2b") == []


def test_is_proto_branches():
    from wyrd.generators.kenning.lexicon.descent_audit import _is_proto

    assert _is_proto("proto-germanic", "haimaz") is True  # _PROTO_LANGS set (no -pro suffix)
    assert _is_proto("xyz-pro", "f") is True  # -pro suffix
    assert _is_proto("latin", "*recon") is True  # *-prefixed reconstruction
    assert _is_proto("latin", "vallis") is False  # plain attested word


def test_scope_both_dedups_2a_and_2b_on_same_edge(lex):
    """An incoherent cluster (2a fires) with a glossless-proto conductor to a
    glossless child (2b fires): scope=both yields the conductor edge exactly once;
    2a alone misses the glossless child, 2b alone catches it."""
    proto = _mk(lex, "*p", "itc-pro")
    v1 = _mk(lex, "vallis", "latin", gloss="valley")
    v2 = _mk(lex, "val", "old-french", gloss="valley")
    other = _mk(lex, "vulgus", "latin", gloss="crowd")  # 2nd glossed sense -> incoherent (2a runs)
    glossless = _mk(lex, "vulva", "latin")  # glossless off-sense (only 2b sees it)
    _cluster(lex, proto, v1, v2, other, glossless)
    _edge(lex, proto, v1)
    _edge(lex, proto, v2)
    _edge(lex, proto, glossless)  # conductor out-degree 3
    _edge(lex, proto, other)
    lex.commit()
    both = detect_descent_audit_candidates(lex.conn, scope="both")
    key = "itc-pro:*p->latin:vulva"
    assert sum(1 for c in both if c.edge_key == key) == 1  # union deduped
    keys_2a = {c.edge_key for c in detect_descent_audit_candidates(lex.conn, scope="2a")}
    keys_2b = {c.edge_key for c in detect_descent_audit_candidates(lex.conn, scope="2b")}
    assert key not in keys_2a and key in keys_2b  # 2b added recall


def test_phase2b_skipped_on_dominant_tie(lex):
    """A 1-1 sense tie has no clear dominant, so 2b screens nothing."""
    proto = _mk(lex, "*x", "ine-pro")
    a = _mk(lex, "alpha", "latin", gloss="fire")
    b = _mk(lex, "beta", "latin", gloss="water")
    glossless = _mk(lex, "gl", "latin")
    _cluster(lex, proto, a, b, glossless)
    _edge(lex, proto, a)
    _edge(lex, proto, b)
    _edge(lex, proto, glossless)
    lex.commit()
    assert detect_descent_audit_candidates(lex.conn, scope="2b") == []


def test_load_cluster_edges_groups_and_excludes_cross_cluster(lex):
    from wyrd.generators.kenning.lexicon.descent_audit import _load_cluster_edges

    a1 = _mk(lex, "a1", "latin")
    a2 = _mk(lex, "a2", "latin")
    b1 = _mk(lex, "b1", "latin")
    b2 = _mk(lex, "b2", "latin")
    _cluster(lex, a1, a2)  # cluster A (cognate_id = a1)
    _cluster(lex, b1, b2)  # cluster B (cognate_id = b1)
    _edge(lex, a1, a2)  # intra-A
    _edge(lex, b1, b2)  # intra-B
    _edge(lex, a1, b1)  # cross-cluster — must be excluded
    lex.commit()
    ebc = _load_cluster_edges(lex.conn)
    assert ebc[a1] == [(a1, a2, "inheritance")]
    assert ebc[b1] == [(b1, b2, "inheritance")]
    assert all((a1, b1) != (p, c) for edges in ebc.values() for p, c, _ in edges)
