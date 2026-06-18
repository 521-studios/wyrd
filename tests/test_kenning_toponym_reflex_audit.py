"""wyrd-1wv2 — per-form toponym-reflex audit (era-grid over-merge long-tail cleaner)."""

from __future__ import annotations

import sqlite3

import pytest

from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.schema import init_schema
from wyrd.generators.kenning.lexicon.toponym_reflex_audit import (
    _CELTIC_LANGS,
    LLM_TOPONYM_REFLEX_DETACH_METHOD,
    ToponymReflexCandidate,
    ToponymReflexVerdict,
    _surface_similar,
    audit_log_row,
    build_judge_prompt,
    detach_row,
    detect_toponym_reflex_candidates,
    deterministic_proper_noun_verdict,
    looks_like_proper_noun,
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


def test_over_merged_judges_surface_similar_default_autokeeps(lex):
    """Default prescreen auto-keeps a surface-similar modern form; over-merged
    mode (prescreen=False, >=2 modern members) judges it."""
    val = _mk(lex, "val", "old-french")
    vale = _mk(lex, "vale", "modern-english")  # surface-similar to val
    valgus = _mk(lex, "valgus", "modern-english")  # surface-similar ('val' prefix) but noise
    lat = _mk(lex, "valgus", "latin")
    _cluster(lex, val, vale, valgus, lat)
    _edge(lex, lat, valgus)
    _edge(lex, val, vale)
    lex.commit()
    default = {c.reflex_ref for c in detect_toponym_reflex_candidates(lex.conn)}
    assert "modern-english:valgus" not in default  # surface-similar -> auto-kept by default
    broad = {
        c.reflex_ref
        for c in detect_toponym_reflex_candidates(lex.conn, prescreen=False, min_modern_members=2)
    }
    assert "modern-english:valgus" in broad  # over-merged mode judges it


def test_min_modern_members_excludes_single_modern_cluster(lex):
    val = _mk(lex, "val", "old-french")
    valgus = _mk(lex, "valgus", "modern-english")  # only ONE modern member
    lat = _mk(lex, "valgus", "latin")
    _cluster(lex, val, valgus, lat)
    _edge(lex, lat, valgus)
    lex.commit()
    assert detect_toponym_reflex_candidates(lex.conn, prescreen=False, min_modern_members=2) == []


def test_cli_emit_detaches_and_load_log(tmp_path):
    import io
    import json as _json

    from wyrd.generators.kenning.cli.lexicon import audit_toponym_reflexes as cli

    log = {
        "modern-english:vulva": {
            "_type": "toponym_reflex_audit",
            "reflex_ref": "modern-english:vulva",
            "bridging_parents": ["latin:vulva"],
            "toponymic": False,
            "confidence": "high",
            "reason": "anatomical",
        },
        "modern-english:vale": {
            "_type": "toponym_reflex_audit",
            "reflex_ref": "modern-english:vale",
            "bridging_parents": ["old-french:val"],
            "toponymic": True,
            "confidence": "high",
            "reason": "valley",
        },
    }
    fh = io.StringIO()
    counts = cli._emit_detaches(log, "medium", {}, set(), fh, dry_run=False)
    assert counts == {"detaches": 1, "kept": 1, "already": 0}
    assert _json.loads(fh.getvalue())["ref"] == "modern-english:vulva"
    # _load_log filters _type + missing-bool + malformed
    f = tmp_path / "a.jsonl"
    f.write_text(
        _json.dumps(log["modern-english:vulva"])
        + "\n"
        + _json.dumps({"_type": "other"})
        + "\n{bad\n",
        encoding="utf-8",
    )
    assert set(cli._load_log(f)) == {"modern-english:vulva"}


def test_cli_judge_fresh_aborts(monkeypatch):
    import click

    from wyrd.generators.kenning.cli.lexicon import audit_toponym_reflexes as cli
    from wyrd.generators.kenning.lexicon.toponym_reflex_audit import ToponymReflexCandidate

    monkeypatch.setattr(cli, "_judge_with_retry", lambda client, c: None)
    cands = [ToponymReflexCandidate(f"modern-english:w{i}", f"w{i}", ("p:q",)) for i in range(8)]
    with pytest.raises(click.ClickException):
        cli._judge_fresh(None, cands, {}, None, "url", "model")


def test_protected_match_normalizes_case():
    row = {
        "reflex_ref": "modern-english:New",
        "bridging_parents": ["old-english:nīwe"],
        "toponymic": False,
        "confidence": "high",
        "reason": "adjective",
    }
    assert detach_row(row, "medium") is None  # 'New' normalizes to protected 'new'


# --- wyrd-xdam.5: Celtic scope + deterministic proper-noun screen ----------


def test_looks_like_proper_noun():
    assert looks_like_proper_noun("Macedonia")
    assert looks_like_proper_noun("Libya")
    assert looks_like_proper_noun("Philip")
    assert looks_like_proper_noun("Caitrìona")  # accented capital (Gaelic personal name)
    assert looks_like_proper_noun("Dùn Bheagain")  # multi-word specific place name
    assert not looks_like_proper_noun("afon")  # lowercase element
    assert not looks_like_proper_noun("aber")  # lowercase Celtic element
    assert not looks_like_proper_noun("Aber")  # capitalized but whitelisted Celtic element
    assert not looks_like_proper_noun("Ham")  # capitalized but whitelisted English element
    assert not looks_like_proper_noun("-aber")  # leading hyphen stripped -> lowercase element
    assert not looks_like_proper_noun("")


def test_deterministic_proper_noun_verdict():
    proper = ToponymReflexCandidate("welsh:Macedonia", "Macedonia", ("old-english:hām",))
    v = deterministic_proper_noun_verdict(proper)
    assert v is not None and v.toponymic is False and v.confidence == "high"
    ambiguous = ToponymReflexCandidate("welsh:afon", "afon", ("old-english:hām",))
    assert deterministic_proper_noun_verdict(ambiguous) is None  # lowercase -> LLM tail


def test_celtic_scope_detects_celtic_members_keeps_similar(lex):
    """--scope celtic picks the Celtic members bridging into an English-anchored
    cluster; a Celtic form surface-similar to its English morpheme is auto-kept;
    the default English scope ignores the Celtic member entirely."""
    oe = _mk(lex, "hām", "old-english")  # English-family anchor
    _me = _mk(lex, "Macedonia", "modern-english")  # english side of the bad merge
    w_proper = _mk(lex, "Macedonia", "welsh")  # foreign proper noun -> candidate
    w_similar = _mk(lex, "hamlet", "welsh")  # surface-similar to hām -> auto-kept
    _cluster(lex, oe, _me, w_proper, w_similar)
    _edge(lex, oe, w_proper)  # bridging parent
    _edge(lex, oe, w_similar)
    lex.commit()
    celtic = {
        c.reflex_ref for c in detect_toponym_reflex_candidates(lex.conn, target_langs=_CELTIC_LANGS)
    }
    assert "welsh:Macedonia" in celtic
    assert "welsh:hamlet" not in celtic  # surface-similar -> auto-kept
    english = {c.reflex_ref for c in detect_toponym_reflex_candidates(lex.conn)}
    assert "welsh:Macedonia" not in english  # default scope never picks Celtic members


def test_celtic_element_protected_from_detach():
    row = {
        "reflex_ref": "welsh:llan",
        "bridging_parents": ["old-english:hām"],
        "toponymic": False,
        "confidence": "high",
        "reason": "judge said no",
    }
    assert detach_row(row, "medium") is None  # CELTIC_PROTECTED_ELEMENTS guard
    # the guard normalizes, so a capitalized/accented form of a known element is kept too
    capitalized = {**row, "reflex_ref": "welsh:Llan"}
    assert detach_row(capitalized, "medium") is None


def test_celtic_judge_prompt_is_celtic_aware():
    celtic = ToponymReflexCandidate("welsh:Macedonia", "Macedonia", ())
    sysp, user = build_judge_prompt(celtic)
    assert "Celtic" in sysp and "Macedonia" in user
    english = ToponymReflexCandidate("modern-english:vulva", "vulva", ())
    sysp_en, _ = build_judge_prompt(english)
    assert "English place-name" in sysp_en


def test_judge_deterministic_partitions_and_logs():
    """The CLI orchestrator logs proper-noun detaches and returns the rest."""
    from wyrd.generators.kenning.cli.lexicon import audit_toponym_reflexes as cli

    proper = ToponymReflexCandidate("welsh:Macedonia", "Macedonia", ("old-english:hām",))
    ambiguous = ToponymReflexCandidate("welsh:afon", "afon", ("old-english:hām",))
    log: dict = {}
    remaining, flagged = cli._judge_deterministic([proper, ambiguous], log, audit_fh=None)
    assert flagged == 1
    assert [c.reflex_ref for c in remaining] == ["welsh:afon"]
    assert log["welsh:Macedonia"]["toponymic"] is False
    assert "welsh:afon" not in log  # ambiguous -> deferred to LLM tail, not logged


def test_cli_deterministic_only_requires_celtic_scope():
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon.audit_toponym_reflexes import (
        lexicon_audit_toponym_reflexes,
    )

    res = CliRunner().invoke(
        lexicon_audit_toponym_reflexes, ["--scope", "english", "--deterministic-only"]
    )
    assert res.exit_code != 0
    assert "deterministic-only" in res.output.lower()


def test_cli_scope_celtic_deterministic_detaches_proper_noun(tmp_path):
    """End-to-end: --scope celtic --deterministic-only flags a welsh proper noun
    bridging into an English-anchored cluster, with no LLM call."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon.audit_toponym_reflexes import (
        lexicon_audit_toponym_reflexes,
    )

    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    db.conn.row_factory = sqlite3.Row
    db.upsert_source(id="wk", title="Wiktionary")
    oe = db.upsert_etymon("hām", "old-english")  # English-family anchor
    me = db.upsert_etymon("Macedonia", "modern-english")
    w = db.upsert_etymon("Macedonia", "welsh")  # foreign proper noun
    for eid in (oe, me, w):
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (oe, eid))
    db.conn.execute(
        "INSERT INTO etymon_descent(parent_id, child_id, edge_type, source_id) "
        "VALUES (?,?,'borrowing','wk')",
        (oe, w),  # bridging parent for the welsh leaf
    )
    db.commit()
    db.close()

    res = CliRunner().invoke(
        lexicon_audit_toponym_reflexes,
        [
            "--db",
            str(tmp_path / "lexicon.db"),
            "--scope",
            "celtic",
            "--deterministic-only",
            "--dry-run",
            "--collapse-file",
            str(tmp_path / "c.jsonl"),
            "--audit-file",
            str(tmp_path / "a.jsonl"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "welsh:Macedonia" in res.output
    assert "deterministic=1" in res.output


def test_cli_scope_celtic_llm_tail_judges_ambiguous(tmp_path, monkeypatch):
    """celtic without --deterministic-only: the proper noun is caught
    deterministically AND the lowercase form not resolved deterministically goes to
    the LLM tail."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon import audit_toponym_reflexes as cli

    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    db.conn.row_factory = sqlite3.Row
    db.upsert_source(id="wk", title="Wiktionary")
    oe = db.upsert_etymon("hām", "old-english")  # English-family anchor
    proper = db.upsert_etymon("Macedonia", "welsh")  # deterministic detach
    ambiguous = db.upsert_etymon("vwlgws", "welsh")  # lowercase -> LLM tail
    for eid in (oe, proper, ambiguous):
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (oe, eid))
    for child in (proper, ambiguous):
        db.conn.execute(
            "INSERT INTO etymon_descent(parent_id, child_id, edge_type, source_id) "
            "VALUES (?,?,'borrowing','wk')",
            (oe, child),
        )
    db.commit()
    db.close()

    monkeypatch.setattr(
        cli,
        "_judge_with_retry",
        lambda client, c: ToponymReflexVerdict(
            toponymic=False, confidence="high", reason="foreign word"
        ),
    )
    res = CliRunner().invoke(
        cli.lexicon_audit_toponym_reflexes,
        [
            "--db",
            str(tmp_path / "lexicon.db"),
            "--scope",
            "celtic",
            "--dry-run",
            "--collapse-file",
            str(tmp_path / "c.jsonl"),
            "--audit-file",
            str(tmp_path / "a.jsonl"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "deterministic=1" in res.output  # the proper noun, no LLM
    assert "judged=1" in res.output  # the ambiguous form, via the LLM tail
    assert "welsh:Macedonia" in res.output and "welsh:vwlgws" in res.output
