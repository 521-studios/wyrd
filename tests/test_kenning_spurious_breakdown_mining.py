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


def test_fold_maps_oe_on_letters_before_ascii_strip():
    """þ/ð/æ/œ must be mapped before encode('ascii') drops them (else 'þorn'→'orn')."""
    from wyrd.generators.kenning.lexicon.spurious_breakdown_mining import _fold

    assert _fold("Þorn") == "thorn"
    assert _fold("ðæl") == "thael"
    assert _fold("œ") == "oe"
    assert _fold("nīwe") == "niwe"  # macron decomposed + stripped via NFKD


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


# --- CLI orchestration (monkeypatched judge — no live Ollama) ---------------


def test_load_judged_tolerates_corrupt_and_filters(tmp_path):
    import json

    from wyrd.generators.kenning.cli.lexicon import mine_spurious_breakdowns as cli

    cdir = tmp_path / "canonicalization"
    cdir.mkdir()
    (cdir / "_decomposition_spurious_audit.jsonl").write_text(
        json.dumps(
            {
                "_type": "decomposition_spurious_audit",
                "te_id": 780,
                "spurious": True,
                "confidence": "high",
                "reason": "x",
            }
        )
        + "\n{bad json\n"
        + json.dumps({"_type": "other", "te_id": 999})  # wrong _type → skipped
        + "\n"
        + json.dumps(
            {"_type": "decomposition_spurious_audit", "te_id": "nope"}
        )  # non-int → skipped
        + "\n",
        encoding="utf-8",
    )
    assert set(cli._load_judged(tmp_path)) == {780}


def test_judge_loop_authors_spurious_gates_low_and_dedups(tmp_path, monkeypatch):
    from wyrd.generators.kenning.canonicalization import load_assertions
    from wyrd.generators.kenning.cli.lexicon import mine_spurious_breakdowns as cli

    hi = BreakdownCandidate(
        780, "Newton", "ford-botl", "n", ("old-english:ford", "old-english:botl")
    )
    low = BreakdownCandidate(781, "X", "a-b", "n", ("old-english:a", "old-english:b"))
    keep = BreakdownCandidate(782, "Y", "c-d", "n", ("old-english:c", "old-english:d"))
    verdicts = {
        780: BreakdownVerdict(True, "high", "x"),
        781: BreakdownVerdict(True, "low", "y"),  # below medium gate → not authored
        782: BreakdownVerdict(False, "high", "z"),  # not spurious → kept
    }
    monkeypatch.setattr(cli, "_judge_with_retry", lambda client, c: verdicts[c.te_id])
    counts = cli._judge_loop(
        None,
        [hi, low, keep],
        mining_dir=tmp_path,
        source="t",
        min_confidence="medium",
        apply=True,
        existing_ids=set(),
        audit_fh=None,
        base_url="u",
        model="m",
    )
    assert counts == {"authored": 1, "spurious": 1, "kept": 2, "skipped": 0}
    authored = list(load_assertions(tmp_path))
    assert [a.subject.ref for a in authored] == ["780"]
    # dedup: re-run with the id already present → no second author
    counts2 = cli._judge_loop(
        None,
        [hi],
        mining_dir=tmp_path,
        source="t",
        min_confidence="medium",
        apply=True,
        existing_ids={authored[0].id},
        audit_fh=None,
        base_url="u",
        model="m",
    )
    assert counts2["authored"] == 0 and len(list(load_assertions(tmp_path))) == 1


def test_judge_loop_dry_run_authors_nothing(tmp_path, monkeypatch):
    from wyrd.generators.kenning.canonicalization import load_assertions
    from wyrd.generators.kenning.cli.lexicon import mine_spurious_breakdowns as cli

    c = BreakdownCandidate(
        780, "Newton", "ford-botl", "n", ("old-english:ford", "old-english:botl")
    )
    monkeypatch.setattr(
        cli, "_judge_with_retry", lambda client, c: BreakdownVerdict(True, "high", "x")
    )
    counts = cli._judge_loop(
        None,
        [c],
        mining_dir=tmp_path,
        source="t",
        min_confidence="medium",
        apply=False,
        existing_ids=set(),
        audit_fh=None,
        base_url="u",
        model="m",
    )
    assert counts["spurious"] == 1 and counts["authored"] == 0
    assert list(load_assertions(tmp_path)) == []  # dry-run writes nothing


def test_judge_with_retry_recovers_then_gives_up():
    from wyrd.generators.kenning.cli.lexicon import mine_spurious_breakdowns as cli

    class _Client:
        def __init__(self, seq):
            self.seq, self.calls = list(seq), 0

        def chat_json(self, system, user, schema):
            r = self.seq[self.calls]
            self.calls += 1
            if isinstance(r, Exception):
                raise r
            return r

    c = BreakdownCandidate(1, "N", "h", "x", ("old-english:a",))
    # attempt 1 raises, attempt 2 parses → verdict returned
    ok = cli._judge_with_retry(
        _Client([RuntimeError("boom"), {"spurious": True, "confidence": "high", "reason": "r"}]), c
    )
    assert ok is not None and ok.spurious is True
    # both attempts unparseable → None (skipped, not crashed)
    assert cli._judge_with_retry(_Client([{"bad": 1}, {"bad": 2}]), c) is None


def test_judge_loop_aborts_after_consecutive_failures(monkeypatch):
    import click

    from wyrd.generators.kenning.cli.lexicon import mine_spurious_breakdowns as cli

    monkeypatch.setattr(cli, "_judge_with_retry", lambda client, c: None)  # always fail
    cands = [BreakdownCandidate(i, "n", "h", "x", ("old-english:a",)) for i in range(6)]
    with pytest.raises(click.ClickException):
        cli._judge_loop(
            None,
            cands,
            mining_dir=None,
            source="t",
            min_confidence="medium",
            apply=False,
            existing_ids=set(),
            audit_fh=None,
            base_url="u",
            model="m",
        )


def test_cli_apply_authors_assertion_end_to_end(lex, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from wyrd.generators.kenning.canonicalization import load_assertions
    from wyrd.generators.kenning.cli.lexicon import mine_spurious_breakdowns as cli

    te = _etymology(
        lex,
        _toponym(lex, "Newton"),
        hf="ford-botl",
        notes="Newton (v.): Newtona 1191-8 FC, Neuton 1190.",
        elements=[("ford", "old-english"), ("botl", "old-english")],
    )
    lex.commit()
    monkeypatch.setattr(
        cli,
        "_judge_with_retry",
        lambda client, c: BreakdownVerdict(True, "high", "belongs to a fordbotl name"),
    )
    mining = tmp_path / "mining"
    res = CliRunner().invoke(
        cli.lexicon_mine_spurious_breakdowns,
        ["--db", str(lex.path), "--mining-dir", str(mining), "--apply"],
    )
    assert res.exit_code == 0, res.output
    authored = list(load_assertions(mining))
    assert [a.subject.ref for a in authored] == [str(te)]
