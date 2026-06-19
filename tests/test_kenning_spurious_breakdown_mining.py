"""wyrd-u6fn.6 — spurious-breakdown finder (detector + LLM-judge plumbing + assertion authoring)."""

from __future__ import annotations

import sqlite3

import pytest

from wyrd.generators.kenning.canonicalization.assertions import validate
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.schema import init_schema
from wyrd.generators.kenning.lexicon.spurious_breakdown_mining import (
    METHOD,
    BreakdownCandidate,
    BreakdownVerdict,
    build_judge_prompt,
    detect_candidates,
    parse_verdict,
    spurious_assertion,
)


def _toponym(db, name):
    return db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", (name,)).lastrowid


def _etymology(db, tid, *, hf, notes, elements):
    """Insert one toponym_etymology row + its element rows under toponym ``tid``."""
    te_id = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, historical_form, notes, confidence) "
        "VALUES (?,?,?,?,'high')",
        (tid, "wk", hf, notes),
    ).lastrowid
    for ordinal, (form, lang) in enumerate(elements):
        eid = db.upsert_etymon(form, lang)
        db.conn.execute(
            "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) "
            "VALUES (?,?,?)",
            (te_id, ordinal, eid),
        )
    return te_id


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


def test_detect_prescreen_flags_low_overlap_excludes_overlapping(lex):
    """Pre-screen (recall-biased) flags breakdowns sharing no >=3-char run with the
    toponym name — the ford+botl-for-Newton contamination AND, by design, legit OE
    decompositions whose surface diverged (nīwe+tūn vs 'Newton'); the LLM judge,
    not the pre-screen, separates those. A breakdown that DOES share a run with the
    name (ford -> Fordham) is excluded outright."""
    newton = _toponym(lex, "Newton")
    spurious = _etymology(
        lex,
        newton,
        hf="ford-botl",
        notes="Newton (v.): Newtona 1191-8 FC, Neuton 1190.",
        elements=[("ford", "old-english"), ("botl", "old-english")],
    )
    legit_oe = _etymology(
        lex,
        newton,
        hf="Neutune",
        notes="Newton (v.): Neutune DB. 'The new tun.'",
        elements=[("nīwe", "old-english"), ("tūn", "old-english")],
    )
    overlapping = _etymology(
        lex,
        _toponym(lex, "Fordham"),
        hf="ford-ham",
        notes="Fordham : Fordham DB.",
        elements=[
            ("ford", "old-english"),
            ("hām", "old-english"),
        ],  # 'ford' shares 'ford' with name
    )
    lex.commit()
    ids = {c.te_id for c in detect_candidates(lex.conn)}
    assert spurious in ids  # ford+botl shares nothing with "Newton" -> candidate
    assert legit_oe in ids  # OE surface diverged -> also a candidate (LLM decides, not surface)
    assert overlapping not in ids  # 'ford' overlaps 'Fordham' -> pre-screened out


def test_detect_pisu_holm_neighbour_contamination(lex):
    """Newtown tagged with the neighbour entry 'Peaseholmes' breakdown is flagged."""
    te = _etymology(
        lex,
        _toponym(lex, "Newtown"),
        hf="pisu-holm",
        notes="Newtown (in the S.): Newtowne 1537 LR. Peaseholmes : Pesholme 1509.",
        elements=[("pisu", "old-english"), ("holm", "old-norse")],
    )
    lex.commit()
    assert te in {c.te_id for c in detect_candidates(lex.conn)}


def test_parse_verdict():
    assert parse_verdict({"spurious": True, "confidence": "high", "reason": "x"}) == (
        BreakdownVerdict(spurious=True, confidence="high", reason="x")
    )
    assert parse_verdict({"spurious": "yes"}).spurious is True
    assert parse_verdict({"spurious": False, "confidence": "bogus"}).confidence == "low"
    assert parse_verdict({"confidence": "high"}) is None  # no spurious field
    assert parse_verdict("nope") is None


def test_build_judge_prompt_carries_name_breakdown_note():
    c = BreakdownCandidate(
        te_id=780,
        modern_name="Newton",
        historical_form="ford-botl",
        notes="Newton: Newtona, Neuton.",
        element_refs=("old-english:ford", "old-english:botl"),
    )
    system, user = build_judge_prompt(c)
    assert "place-name" in system.lower()
    assert "Newton" in user and "ford + botl" in user and "Newtona" in user


def test_spurious_assertion_is_valid_decomposition_spurious(lex):
    c = BreakdownCandidate(
        te_id=780,
        modern_name="Newton",
        historical_form="ford-botl",
        notes="…",
        element_refs=("old-english:ford", "old-english:botl"),
    )
    a = spurious_assertion(
        c, BreakdownVerdict(spurious=True, confidence="high", reason="belongs to a fordbotl name")
    )
    validate(a)  # raises if the record violates the predicate contract
    assert a.predicate == "decomposition-spurious"
    assert a.subject.type == "toponym_etymology" and a.subject.ref == "780"
    assert a.object is None and a.polarity == "affirm"
    assert a.qualifiers == {"reason": "belongs to a fordbotl name"}
    assert a.method == METHOD and a.id  # content-hash id minted
