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


# --- P1 fix: family-proto parents bridge (only PIE is non-bridging) ---


def test_bridging_parents_includes_family_proto_excludes_pie_and_cross_cluster(lex):
    """A leaf's in-cluster family-proto parent (proto-germanic) MUST be a detach
    target (it bridges the cluster, like cluster_cognates); a PIE parent and a
    different-cluster parent must NOT be."""
    vallis = _mk(lex, "vallis", "latin", gloss="valley")
    val = _mk(lex, "val", "old-french", gloss="valley")
    leaf = _mk(lex, "leafword", "modern-english")  # glossless suspect
    sibling = _mk(lex, "vale", "modern-english", gloss="valley")  # 2nd modern member
    fam_proto = _mk(lex, "*famproto", "proto-germanic")  # family proto — BRIDGES
    pie = _mk(lex, "*pieroot", "proto-indo-european")  # PIE — non-bridging
    _cluster(lex, vallis, val, leaf, sibling, fam_proto)  # leaf in this cluster
    other = _mk(lex, "otherword", "latin", gloss="unrelated")
    other2 = _mk(lex, "otherword2", "latin", gloss="unrelated")
    _cluster(lex, other, other2)  # a different cluster
    _edge(lex, fam_proto, leaf, "inheritance")  # family-proto -> leaf (bridges)
    _edge(lex, pie, leaf, "inheritance")  # PIE -> leaf (non-bridging)
    _edge(lex, other, leaf, "borrowing")  # cross-cluster parent
    lex.commit()
    cands = detect_cluster_reflex_candidates(lex.conn)
    leaf_c = next(c for c in cands if c.reflex_ref == "modern-english:leafword")
    assert leaf_c.bridging_parents == ("proto-germanic:*famproto",)


# --- CLI helpers (mirror the audit_descent coverage) ---


def _logrow(ref, parents, *, plausible, conf="high"):
    return {
        "_type": "cluster_reflex_audit",
        "reflex_ref": ref,
        "bridging_parents": parents,
        "reflex_of_sense": plausible,
        "confidence": conf,
        "reason": "r",
    }


def test_cli_emit_detaches_groups_merges_and_dedups():
    import io

    from wyrd.generators.kenning.cli.lexicon import audit_cluster_reflexes as cli

    log = {
        "modern-english:vulva": _logrow("modern-english:vulva", ["latin:vulva"], plausible=False),
        "modern-english:vale": _logrow("modern-english:vale", ["old-french:val"], plausible=True),
    }
    # vale kept; vulva detached -> one row
    fh = io.StringIO()
    counts = cli._emit_detaches(log, "medium", {}, set(), fh, dry_run=False)
    import json as _json

    rows = [_json.loads(x) for x in fh.getvalue().splitlines()]
    assert counts == {"detaches": 1, "kept": 1, "already": 0}
    assert rows == [
        {
            "_type": "collapse",
            "ref": "modern-english:vulva",
            "detach_parents": ["latin:vulva"],
            "method": "llm-cluster-reflex-detach-v1",
            "confidence": "high",
            "reason": "r",
        }
    ]
    # merge-with-existing-fold preserves into/detach_children
    state = {
        "modern-english:vulva": {
            "_type": "collapse",
            "ref": "modern-english:vulva",
            "into": "x:y",
            "detach_children": ["z:w"],
        }
    }
    fh2 = io.StringIO()
    cli._emit_detaches(log, "medium", state, set(), fh2, dry_run=False)
    merged = _json.loads(fh2.getvalue())
    assert merged["into"] == "x:y" and merged["detach_children"] == ["z:w"]
    assert merged["detach_parents"] == ["latin:vulva"]
    # dedup against existing pairs
    fh3 = io.StringIO()
    c3 = cli._emit_detaches(
        log, "medium", {}, {("modern-english:vulva", "latin:vulva")}, fh3, dry_run=False
    )
    assert c3["already"] == 1 and fh3.getvalue() == ""


def test_cli_load_log_filters_malformed(tmp_path):
    import json as _json

    from wyrd.generators.kenning.cli.lexicon import audit_cluster_reflexes as cli

    f = tmp_path / "_cluster_reflex_audit.jsonl"
    f.write_text(
        "\n".join(
            [
                _json.dumps(_logrow("a:b", ["c:d"], plausible=False)),
                _json.dumps({"_type": "other", "reflex_ref": "x"}),
                _json.dumps(
                    {"_type": "cluster_reflex_audit", "reflex_ref": "y"}
                ),  # no bool verdict
                "{bad json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert set(cli._load_log(f)) == {"a:b"}
    assert cli._load_log(tmp_path / "absent.jsonl") == {}


def test_cli_judge_fresh_aborts_after_five_failures(monkeypatch):
    import click

    from wyrd.generators.kenning.cli.lexicon import audit_cluster_reflexes as cli
    from wyrd.generators.kenning.lexicon.cluster_reflex_audit import ClusterReflexCandidate

    monkeypatch.setattr(cli, "_judge_with_retry", lambda client, c: None)
    cands = [
        ClusterReflexCandidate(f"modern-english:w{i}", f"w{i}", ("s",), ("p:q",)) for i in range(8)
    ]
    log: dict = {}
    with pytest.raises(click.ClickException):
        cli._judge_fresh(None, cands, log, None, "url", "model")
    assert log == {}
