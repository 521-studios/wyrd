"""Tests for cognate_descent_llm (wyrd-zrce.2) — candidate generation, the two-pass
prompt parsers, and verdict-log → edge derivation. LLM-free: the parsers take raw
dicts and the candidate builder takes a DB; the actual LLM calls live in the CLI
(tested there with a fake client).
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.cognate_descent_llm import (
    audit_log_row,
    build_candidates,
    build_propose_prompt,
    build_refute_prompt,
    edge_from_log,
    parse_proposal,
    parse_verdict,
)


def _db(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    db.conn.execute("PRAGMA foreign_keys = OFF")
    return db


def _etymon(db, form, *, language="old-english", cognate_id=None):
    eid = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    ).lastrowid
    if cognate_id is not None:
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (cognate_id, eid))
    return eid


def _breakdown(db, eid):
    db.conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) "
        "VALUES (?, 0, ?)",
        (eid, eid),
    )


def _gloss(db, eid, *glosses):
    for g in glosses:
        db.conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, g))


def _clustered(db, form, *, language, root):
    return _etymon(db, form, language=language, cognate_id=root)


# --- candidate generation --------------------------------------------------


def test_candidate_built_for_shared_prefix(tmp_path):
    db = _db(tmp_path)
    root = _etymon(db, "stanaz", language="proto-germanic")
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, root))
    bridge = _clustered(db, "stane", language="old-norse", root=root)  # fold "stane"
    _gloss(db, bridge, "stone")
    x = _etymon(db, "stant", language="old-english")  # fold "stant", shares prefix "stan"
    _breakdown(db, x)
    _gloss(db, x, "stony place")
    db.commit()

    items = build_candidates(db, prefix_len=4)
    assert len(items) == 1
    it = items[0]
    assert it.etymon_id == x and it.glosses == ("stony place",)
    assert len(it.candidates) == 1
    assert it.candidates[0].cluster_root == root
    # wyrd-s964: build_candidates resolves the root's durable natural key (via the
    # batch etymon_refs_for lookup) onto each candidate — this is what audit_log_row
    # records, so pin it at the source.
    assert it.candidates[0].root_ref == "proto-germanic:stanaz"
    assert it.candidates[0].bridge_form == "stane"
    db.close()


def test_glossless_residue_skipped(tmp_path):
    db = _db(tmp_path)
    root = _etymon(db, "stanaz", language="proto-germanic")
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, root))
    _gloss(db, _clustered(db, "stane", language="old-norse", root=root), "stone")
    x = _etymon(db, "stant", language="old-english")
    _breakdown(db, x)  # no gloss on x
    db.commit()
    assert build_candidates(db, prefix_len=4) == []  # no semantic signal → skip
    db.close()


def test_no_stem_mate_no_item(tmp_path):
    db = _db(tmp_path)
    root = _etymon(db, "stanaz", language="proto-germanic")
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, root))
    _gloss(db, _clustered(db, "stane", language="old-norse", root=root), "stone")
    x = _etymon(db, "qrxyz", language="old-english")  # shares no prefix
    _breakdown(db, x)
    _gloss(db, x, "whatever")
    db.commit()
    assert build_candidates(db, prefix_len=4) == []
    db.close()


def test_candidates_capped_and_one_per_cluster(tmp_path):
    db = _db(tmp_path)
    roots = []
    for i in range(7):
        r = _etymon(db, f"stanr{i}", language="proto-germanic")
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (r, r))
        # two members per cluster, both share the "stan" prefix
        _gloss(db, _clustered(db, f"stanea{i}", language="old-norse", root=r), "stone")
        _gloss(db, _clustered(db, f"staneb{i}", language="icelandic", root=r), "stone")
        roots.append(r)
    x = _etymon(db, "stant", language="old-english")
    _breakdown(db, x)
    _gloss(db, x, "stony")
    db.commit()
    items = build_candidates(db, prefix_len=4, max_candidates=5)
    assert len(items) == 1
    cands = items[0].candidates
    assert len(cands) == 5  # capped
    assert len({c.cluster_root for c in cands}) == 5  # one per cluster, no dupes
    db.close()


# --- parsers ---------------------------------------------------------------


def test_parse_proposal_choice_and_none():
    assert parse_proposal({"choice": 2, "confidence": "high", "reason": "x"}, 3).chosen == 1
    assert parse_proposal({"choice": 0, "confidence": "low"}, 3).chosen is None  # 0 = none
    assert parse_proposal({"choice": 9}, 3) is None  # out of range
    assert parse_proposal({"choice": "bad"}, 3) is None
    assert parse_proposal({}, 3) is None
    # unknown confidence label floors to low
    assert parse_proposal({"choice": 1, "confidence": "??"}, 3).confidence == "low"


def test_parse_verdict():
    assert parse_verdict({"confirmed": True, "confidence": "medium"}).confirmed is True
    assert parse_verdict({"confirmed": "yes"}).confirmed is True
    assert parse_verdict({"confirmed": False}).confirmed is False
    assert parse_verdict({"confidence": "high"}) is None  # missing confirmed
    assert parse_verdict("nope") is None


# --- verdict log → edge ----------------------------------------------------


# wyrd-s964: the verdict log stores stable NATURAL KEYS, so edge_from_log resolves
# them to the current build's ids via a conn (not int()).
def _log(confirmed, pconf, vconf, *, root_ref="proto-germanic:stanaz", ref="old-english:stant"):
    return {
        "morpheme_ref": ref,
        "morpheme_form": "stant",
        "cluster_root": root_ref if confirmed else None,
        "bridge_form": "stane",
        "proposal_confidence": pconf,
        "confirmed": confirmed,
        "verify_confidence": vconf,
    }


def _edge_db(tmp_path):
    """A DB holding the child + root etymons the verdict-log natural keys resolve to,
    so edge_from_log can map "old-english:stant" / "proto-germanic:stanaz" → ids."""
    db = _db(tmp_path)
    child = _etymon(db, "stant", language="old-english")  # ref "old-english:stant"
    root = _etymon(db, "stanaz", language="proto-germanic")  # ref "proto-germanic:stanaz"
    db.commit()
    return db, child, root


def test_edge_from_log_confirmed_above_gate(tmp_path):
    db, child, root = _edge_db(tmp_path)
    e = edge_from_log(_log(True, "medium", "high"), conn=db.conn, min_confidence="medium")
    assert e is not None
    assert e.child_etymon == child and e.cluster_root == root and e.confidence == "medium"
    db.close()


def test_edge_from_log_effective_is_min_and_capped(tmp_path):
    db, _child, _root = _edge_db(tmp_path)
    c = db.conn
    # proposal high + verify medium → effective medium
    assert (
        edge_from_log(_log(True, "high", "medium"), conn=c, min_confidence="medium").confidence
        == "medium"
    )
    # high + high → capped to medium (LLM never high-confidence identity)
    assert (
        edge_from_log(_log(True, "high", "high"), conn=c, min_confidence="medium").confidence
        == "medium"
    )
    # verify low floors it → below medium gate → no edge
    assert edge_from_log(_log(True, "medium", "low"), conn=c, min_confidence="medium") is None
    # ...but emits at a lowered gate
    assert (
        edge_from_log(_log(True, "medium", "low"), conn=c, min_confidence="low").confidence == "low"
    )
    db.close()


def test_edge_from_log_refuted_or_none(tmp_path):
    db, _child, _root = _edge_db(tmp_path)
    c = db.conn
    assert edge_from_log(_log(False, "high", "high"), conn=c, min_confidence="low") is None
    assert (
        edge_from_log({"confirmed": True, "cluster_root": None}, conn=c, min_confidence="low")
        is None
    )  # no chosen cluster
    db.close()


def test_edge_from_log_resolves_against_reassigned_ids(tmp_path):
    """wyrd-s964 core promise: a verdict cached against one id-assignment must resolve
    correctly after a rebuild REASSIGNS etymon ids. We replay the SAME natural-key row
    against a second build where the identical (form, language) pair lands on a
    DIFFERENT row id — the edge must follow the natural key, not a frozen id. A
    regression to int(ref) would silently pass whenever the ids happened to coincide;
    this test fails it by forcing them apart."""
    db_a, child_a, root_a = _edge_db(tmp_path / "a")
    row = _log(True, "medium", "high")  # natural keys: old-english:stant / proto-germanic:stanaz

    # Build B: insert filler etymons FIRST so the same two morphemes get higher ids.
    db_b = _db(tmp_path / "b")
    for i in range(5):
        _etymon(db_b, f"filler{i}", language="gothic")
    child_b = _etymon(db_b, "stant", language="old-english")
    root_b = _etymon(db_b, "stanaz", language="proto-germanic")
    db_b.commit()

    assert (child_b, root_b) != (child_a, root_a)  # ids genuinely reassigned
    e = edge_from_log(row, conn=db_b.conn, min_confidence="medium")
    assert e is not None
    assert e.child_etymon == child_b and e.cluster_root == root_b  # resolved to BUILD B's ids
    db_a.close()
    db_b.close()


def test_edge_from_log_unresolvable_key_skipped(tmp_path):
    """wyrd-s964: a confirmed verdict whose endpoint key is absent from THIS build
    (a stale entry from a different corpus, or a legacy id-keyed row) resolves to no
    edge — skipped + re-judged, never a wrong/orphaned edge."""
    db, _child, _root = _edge_db(tmp_path)
    c = db.conn
    # stale natural key (no such etymon in this build)
    assert (
        edge_from_log(
            _log(True, "high", "high", root_ref="old-norse:ghost"), conn=c, min_confidence="low"
        )
        is None
    )
    # legacy id-keyed row (bare "10" has no ':' → resolve_etymon_ref → None)
    assert (
        edge_from_log(
            _log(True, "high", "high", ref="10", root_ref="42"), conn=c, min_confidence="low"
        )
        is None
    )
    db.close()


# --- prompts ---------------------------------------------------------------


def test_prompts_render(tmp_path):
    db = _db(tmp_path)
    root = _etymon(db, "stanaz", language="proto-germanic")
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (root, root))
    _gloss(db, _clustered(db, "stane", language="old-norse", root=root), "stone")
    x = _etymon(db, "stant", language="old-english")
    _breakdown(db, x)
    _gloss(db, x, "stony place")
    db.commit()
    item = build_candidates(db, prefix_len=4)[0]

    _sys, user = build_propose_prompt(item)
    assert "stant" in user and "stane" in user and "1." in user and "stony place" in user
    _sys2, user2 = build_refute_prompt(item, item.candidates[0])
    assert "stant" in user2 and "stane" in user2 and "Refute" in user2

    # log row round-trips through edge_from_log
    from wyrd.generators.kenning.lexicon.cognate_descent_llm import Proposal, Verdict

    row = audit_log_row(
        item, Proposal(chosen=0, confidence="medium", reason="r"), Verdict(True, "medium", "ok")
    )
    # wyrd-s964: the log records the root's NATURAL KEY, not its id…
    assert row["confirmed"] is True and row["cluster_root"] == "proto-germanic:stanaz"
    # …which edge_from_log resolves back to the current build's root id.
    assert edge_from_log(row, conn=db.conn, min_confidence="medium").cluster_root == root
    db.close()


# --- two-pass judge_item (fake judge) --------------------------------------


def _item():
    from wyrd.generators.kenning.lexicon.cognate_descent_llm import ClusterCand, ResidueItem

    return ResidueItem(
        etymon_id=5,
        form="stant",
        language="old-english",
        glosses=("stony place",),
        candidates=(
            ClusterCand(
                cluster_root=10,
                root_ref="proto-germanic:stanaz",
                bridge_form="stane",
                bridge_lang="old-norse",
                bridge_glosses=("stone",),
            ),
        ),
    )


def test_judge_item_confirmed_path(tmp_path):
    from wyrd.generators.kenning.lexicon.cognate_descent_llm import judge_item

    db, _child, root = _edge_db(tmp_path)
    calls = []

    def judge(system, user):
        calls.append(user)
        if '"choice"' not in system:  # refute system requests {"confirmed": ...}
            return {"confirmed": True, "confidence": "medium", "reason": "real cognate"}
        return {"choice": 1, "confidence": "medium", "reason": "matches"}

    row = judge_item(_item(), judge)
    assert len(calls) == 2  # propose + verify
    # wyrd-s964: the row records the root's natural key; edge_from_log resolves it.
    assert row["confirmed"] is True and row["cluster_root"] == "proto-germanic:stanaz"
    assert edge_from_log(row, conn=db.conn, min_confidence="medium").cluster_root == root
    db.close()


def test_judge_item_none_proposal_skips_verify(tmp_path):
    from wyrd.generators.kenning.lexicon.cognate_descent_llm import judge_item

    db, _child, _root = _edge_db(tmp_path)
    calls = []

    def judge(system, user):
        calls.append(user)
        return {"choice": 0, "confidence": "low", "reason": "no match"}

    row = judge_item(_item(), judge)
    assert len(calls) == 1  # NONE proposal → no verify call
    assert row["confirmed"] is False and row["cluster_root"] is None
    assert edge_from_log(row, conn=db.conn, min_confidence="low") is None
    db.close()


def test_judge_item_refuted_yields_no_edge(tmp_path):
    from wyrd.generators.kenning.lexicon.cognate_descent_llm import judge_item

    db, _child, _root = _edge_db(tmp_path)

    def judge(system, user):
        if '"choice"' not in system:
            return {"confirmed": False, "confidence": "high", "reason": "look-alike"}
        return {"choice": 1, "confidence": "high", "reason": "matches"}

    row = judge_item(_item(), judge)
    assert row["confirmed"] is False
    assert edge_from_log(row, conn=db.conn, min_confidence="low") is None
    db.close()


def test_judge_item_unusable_response_returns_none():
    from wyrd.generators.kenning.lexicon.cognate_descent_llm import judge_item

    row = judge_item(_item(), lambda s, u: {"garbage": 1})
    assert row is None  # caller retries
