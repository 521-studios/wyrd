"""LLM same-meaning merge — wyrd-6xzs P3. (LLM mocked; no Ollama in CI.)"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from wyrd.generators.kenning.lexicon import init_schema
from wyrd.generators.kenning.lexicon.collapse_merge import (
    LLM_MERGE_METHOD,
    LLM_REJECT_METHOD,
    MergeCandidate,
    Verdict,
    build_judge_prompt,
    detect_merge_candidates,
    parse_verdict,
    verdict_to_row,
)


def _conn(db_path: Path) -> sqlite3.Connection:
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO source (id, title) VALUES ('wikt', 'Wiktionary')")
    return conn


def _ety(conn, eid, form, glosses):
    conn.execute(
        "INSERT INTO etymon (id, canonical_form, language) VALUES (?, ?, 'old-english')",
        (eid, form),
    )
    for g in glosses:
        conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, g))


def _variant(conn, lemma_id, form):
    conn.execute(
        "INSERT INTO etymon_variant (etymon_id, form, variant_class, source_id) "
        "VALUES (?, ?, 'alternative', 'wikt')",
        (lemma_id, form),
    )


def _reflex_link(conn, etymon_id, surface):
    cur = conn.execute(
        "INSERT INTO reflex (surface_form, position, productivity) VALUES (?, 'post', 0)",
        (surface,),
    )
    conn.execute(
        "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (cur.lastrowid, etymon_id)
    )


def test_detect_only_no_shared_gloss_doublings(tmp_path):
    conn = _conn(tmp_path / "lex.db")
    # candidate: doubling, reflex-linked, both glossed, NO shared gloss
    _ety(conn, 1, "burh", ["alternative form of burg"])
    _ety(conn, 2, "burg", ["fort"])
    _variant(conn, 2, "burh")
    _reflex_link(conn, 1, "-burh")
    # NOT a candidate: shares a gloss (deterministic detector's job)
    _ety(conn, 3, "cald", ["cold"])
    _ety(conn, 4, "cold", ["cold"])
    _variant(conn, 4, "cald")
    _reflex_link(conn, 3, "-cald")
    # NOT a candidate: from is unglossed
    _ety(conn, 5, "x", [])
    _ety(conn, 6, "y", ["something"])
    _variant(conn, 6, "x")
    _reflex_link(conn, 5, "-x")

    cands = detect_merge_candidates(conn)
    assert [c.from_ref for c in cands] == ["old-english:burh"]
    c = cands[0]
    assert c.into_ref == "old-english:burg"
    assert c.from_glosses == ("alternative form of burg",)
    assert c.into_glosses == ("fort",)


def test_detect_mixed_pointer_candidates(tmp_path):
    """A MIXED form-of pointer (real gloss + 'alternative form of X', X
    glossed) is a candidate; a PURE pointer is the deterministic
    detector's, and an unglossed target is skipped."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "wall", ["wall", "alternative form of weall"])
    _ety(conn, 2, "weall", ["wall, rampart"])
    _ety(conn, 3, "Hea", ["h-prothesized form of ea"])  # PURE → deterministic's
    _ety(conn, 4, "ea", ["river"])
    _ety(conn, 5, "x", ["thing", "variant of y"])  # target unglossed → skip
    _ety(conn, 6, "y", [])
    cands = {c.from_ref: c.into_ref for c in detect_merge_candidates(conn)}
    assert cands == {"old-english:wall": "old-english:weall"}


def test_mixed_pointer_multiword_tail_targets_real_lemma(tmp_path):
    """The mixed-pointer candidate target now uses the shared _resolve_pointer_target
    (cycle-85): a multi-word pointer tail resolves the real lemma, not the first word.
    'alternative form of Old English burg' targets 'burg', skipping the capitalized
    language qualifier (a bare-lower-case canonical_form never matches 'Old'). The old
    first-word extraction captured 'Old' → no resolve → the candidate was MISSED."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "ealdburh", ["a stronghold", "alternative form of Old English burg"])
    _ety(conn, 2, "burg", ["fort"])
    _ety(conn, 3, "old", ["aged"])  # capital 'Old' in the gloss must NOT match this lemma
    cands = {c.from_ref: c.into_ref for c in detect_merge_candidates(conn)}
    assert cands == {"old-english:ealdburh": "old-english:burg"}


def test_mixed_pointer_ignores_form_of_in_prose_gloss(tmp_path):
    """Guard (cycle-85): only the START-ANCHORED pointer glosses feed target
    resolution. A real PROSE gloss that merely contains 'form of' deep in the sentence
    is not a pointer (fails the ≤3-word anchor), so it must never steer the target —
    the candidate targets 'weall' (the actual pointer's lemma), never 'stone' from the
    prose. Pins that the helper receives the filtered pointer glosses, not the full
    gloss list (whose unanchored tail regex would mis-resolve the prose)."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "wal", ["a wall of a particular form of stone", "alternative form of weall"])
    _ety(conn, 2, "weall", ["wall, rampart"])
    _ety(conn, 3, "stone", ["rock"])  # appears in 'form of stone' prose — must NOT be the target
    cands = {c.from_ref: c.into_ref for c in detect_merge_candidates(conn)}
    assert cands == {"old-english:wal": "old-english:weall"}


def _cand():
    return MergeCandidate(
        from_ref="old-english:burh",
        into_ref="old-english:burg",
        from_form="burh",
        into_form="burg",
        from_glosses=("alternative form of burg",),
        into_glosses=("fort",),
        variant_class="alternative",
    )


def test_build_prompt_contains_forms_and_glosses():
    system, user = build_judge_prompt(_cand())
    assert "JSON" in system and "same_meaning" in system
    assert "burh" in user and "burg" in user and "fort" in user


def test_parse_verdict_variants():
    assert parse_verdict({"same_meaning": True, "confidence": "high"}).same_meaning is True
    # string bool coercion
    assert parse_verdict({"same_meaning": "yes", "confidence": "medium"}).same_meaning is True
    # unrecognized confidence → low (conservative)
    assert parse_verdict({"same_meaning": True, "confidence": "pretty sure"}).confidence == "low"
    # unusable → None (caller skips, doesn't crash)
    assert parse_verdict({"reason": "no verdict"}) is None
    assert parse_verdict("not a dict") is None


def test_verdict_to_row_fold_vs_reject():
    c = _cand()
    # confident same-meaning → real fold
    row = verdict_to_row(c, Verdict(True, "high", "spelling variant"), min_confidence="medium")
    assert row["into"] == "old-english:burg" and row["method"] == LLM_MERGE_METHOD
    # same-meaning but BELOW threshold → no-op rejection row (records judgment)
    row = verdict_to_row(c, Verdict(True, "low", "unsure"), min_confidence="medium")
    assert row["into"] == "" and row["method"] == LLM_REJECT_METHOD
    # different meaning → no-op rejection
    row = verdict_to_row(c, Verdict(False, "high", "homograph"), min_confidence="medium")
    assert row["into"] == "" and row["method"] == LLM_REJECT_METHOD
    # the no-op rows still carry the audit fields
    assert row["confidence"] == "high" and row["reason"] == "homograph"
