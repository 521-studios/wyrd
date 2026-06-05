"""LLM merge-decision audit — wyrd-6hbv.

Detection + provenance-routing are deterministic and tested here; the LLM judge
itself is exercised by constructing verdicts directly (no Ollama in CI).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from wyrd.generators.kenning.lexicon import init_schema
from wyrd.generators.kenning.lexicon.merge_audit import (
    LLM_MERGE_AUDIT_REVERT_METHOD,
    PROV_COLLAPSE,
    PROV_LEMMA,
    PROV_OCR,
    MergeAuditCandidate,
    MergeAuditVerdict,
    audit_verdict_to_rows,
    build_audit_judge_prompt,
    detect_merge_audit_candidates,
    parse_audit_verdict,
)


def _conn(db_path: Path) -> sqlite3.Connection:
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _ety(conn, eid, form, glosses, *, lang="modern-english", merged_into=None, lemma=None):
    conn.execute(
        "INSERT INTO etymon (id, canonical_form, language, merged_into_id, lemma_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (eid, form, lang, merged_into, lemma),
    )
    for g in glosses:
        conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, g))


def _cand(member_ref, anchor_ref, provenance):
    return MergeAuditCandidate(
        anchor_ref=anchor_ref,
        member_ref=member_ref,
        provenance=provenance,
        anchor_form=anchor_ref.split(":", 1)[1],
        member_form=member_ref.split(":", 1)[1],
        anchor_glosses=("anchor sense",),
        member_glosses=(),
    )


# --- detection -------------------------------------------------------------


def test_detect_glossless_affix_into_glossed_bare_anchor_is_ocr(tmp_path):
    """The -ton class: a glossless dashed affix OCR-merged into a glossed bare
    word surfaces as an OCR-provenance candidate."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "ton", ["Any of various units of mass"])  # bare weight unit
    _ety(conn, 2, "-ton", [], merged_into=1)  # glossless toponym suffix, OCR-merged
    cands = detect_merge_audit_candidates(conn, {}, scope="affixes")
    assert len(cands) == 1
    c = cands[0]
    assert c.member_ref == "modern-english:-ton"
    assert c.anchor_ref == "modern-english:ton"
    assert c.provenance == PROV_OCR


def test_detect_benign_affix_into_root_still_surfaces_for_llm(tmp_path):
    """A correct affix↔root pair (-land/land) is still emitted as a candidate —
    the LLM decides; the detector only screens."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "land", ["The part of Earth not covered by water"])
    _ety(conn, 2, "-land", [], merged_into=1)
    cands = detect_merge_audit_candidates(conn, {}, scope="affixes")
    assert [c.member_ref for c in cands] == ["modern-english:-land"]


def test_detect_member_level_disjoint_sense_intruder(tmp_path):
    """The -ing class: an anchor whose glossed members split into disjoint
    sense-groups flags the non-dominant member as an intruder."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "-ing", ["patronymic suffix son descendants"], lang="old-english")
    _ety(
        conn, 2, "-ingas", ["patronymic son descendants people"], lang="old-english", merged_into=1
    )
    _ety(conn, 3, "ing", ["water-meadow grassland"], lang="old-english", merged_into=1)
    cands = detect_merge_audit_candidates(conn, {}, scope="anchors")
    # -ingas shares the patronymic sense (dominant, with anchor) → not flagged;
    # ing (water-meadow) is the disjoint intruder.
    refs = {c.member_ref for c in cands}
    assert "old-english:ing" in refs
    assert "old-english:-ingas" not in refs


def test_detect_lemma_link_provenance(tmp_path):
    """A disjoint intruder reached via lemma_id (not merged_into) is lemma-link
    provenance."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "stream", ["a flowing watercourse brook"])
    _ety(conn, 2, "streamX", ["unrelated machine process"], lemma=1)
    cands = detect_merge_audit_candidates(conn, {}, scope="anchors")
    assert cands and cands[0].provenance == PROV_LEMMA


def test_detect_collapse_fold_provenance_and_scope(tmp_path):
    """A live collapse fold surfaces under scope=collapses with collapse
    provenance (even when its glosses aren't disjoint — full ledger re-audit)."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "john", ["a masculine personal name"])
    _ety(conn, 2, "johan", ["a masculine personal name"], merged_into=1)
    collapse_state = {"modern-english:johan": {"into": "modern-english:john"}}
    cands = detect_merge_audit_candidates(conn, collapse_state, scope="collapses")
    assert len(cands) == 1
    assert cands[0].member_ref == "modern-english:johan"
    assert cands[0].provenance == PROV_COLLAPSE
    # a collapse-fold member is collapse-provenance even if also merged in the DB
    assert detect_merge_audit_candidates(conn, collapse_state, scope="anchors") == []


def test_detect_dedup_across_sources(tmp_path):
    """A member that is BOTH a collapse fold and an affix candidate is judged
    once (deduped by member ref)."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "ton", ["units of mass"])
    _ety(conn, 2, "-ton", [], merged_into=1)
    collapse_state = {"modern-english:-ton": {"into": "modern-english:ton"}}
    cands = detect_merge_audit_candidates(conn, collapse_state, scope="both")
    members = [c.member_ref for c in cands]
    assert members.count("modern-english:-ton") == 1
    # collapse-fold provenance wins (it IS a live fold)
    assert cands[0].provenance == PROV_COLLAPSE


# --- verdict routing -------------------------------------------------------


def test_revert_collapse_fold_goes_to_collapse_ledger():
    c = _cand("oe:ton", "oe:tonweight", PROV_COLLAPSE)
    v = MergeAuditVerdict(correct=False, confidence="high", reason="distinct")
    rows = audit_verdict_to_rows(c, v, "medium")
    assert rows.curation is None
    assert rows.collapse == {
        "_type": "collapse",
        "ref": "oe:ton",
        "into": "",
        "method": LLM_MERGE_AUDIT_REVERT_METHOD,
        "confidence": "high",
        "reason": "distinct",
    }
    assert rows.audit_log["verdict"] == "revert"


def test_revert_ocr_goes_to_curation_merged_into():
    c = _cand("modern-english:-ton", "modern-english:ton", PROV_OCR)
    v = MergeAuditVerdict(correct=False, confidence="high", reason="suffix not weight")
    rows = audit_verdict_to_rows(c, v, "medium")
    assert rows.collapse is None
    assert rows.curation == {
        "_type": "etymon_curation",
        "ref": "modern-english:-ton",
        "reason": "suffix not weight",
        "merged_into_ref": None,
    }


def test_revert_lemma_goes_to_curation_lemma_ref():
    c = _cand("oe:streamX", "oe:stream", PROV_LEMMA)
    v = MergeAuditVerdict(correct=False, confidence="high", reason="distinct")
    rows = audit_verdict_to_rows(c, v, "medium")
    assert rows.collapse is None
    assert rows.curation["lemma_ref"] is None
    assert "merged_into_ref" not in rows.curation


def test_keep_verdict_emits_only_audit_log():
    c = _cand("modern-english:-land", "modern-english:land", PROV_OCR)
    v = MergeAuditVerdict(correct=True, confidence="high", reason="same morpheme")
    rows = audit_verdict_to_rows(c, v, "medium")
    assert rows.collapse is None and rows.curation is None
    assert rows.audit_log["verdict"] == "keep"


def test_low_confidence_revert_is_held_as_keep():
    """A wrong-merge verdict below --min-confidence does NOT emit a revert."""
    c = _cand("modern-english:-ton", "modern-english:ton", PROV_OCR)
    v = MergeAuditVerdict(correct=False, confidence="low", reason="maybe")
    rows = audit_verdict_to_rows(c, v, "high")
    assert rows.collapse is None and rows.curation is None
    assert rows.audit_log["verdict"] == "keep"


# --- LLM-output parsing (deterministic coercion boundary) -------------------


def test_parse_audit_verdict_bool_and_confidence():
    v = parse_audit_verdict({"same_morpheme": True, "confidence": "high", "reason": "x"})
    assert v.correct is True and v.confidence == "high" and v.reason == "x"


def test_parse_audit_verdict_string_bool_coercion():
    assert parse_audit_verdict({"same_morpheme": "yes", "confidence": "low"}).correct is True
    assert parse_audit_verdict({"same_morpheme": "false", "confidence": "low"}).correct is False


def test_parse_audit_verdict_unknown_confidence_defaults_low():
    assert parse_audit_verdict({"same_morpheme": False, "confidence": "wat"}).confidence == "low"


def test_parse_audit_verdict_rejects_unusable():
    assert parse_audit_verdict("not a dict") is None
    assert parse_audit_verdict({"confidence": "high"}) is None  # missing same_morpheme
    assert parse_audit_verdict({"same_morpheme": 3}) is None  # non-bool, non-str


def test_parse_audit_verdict_truncates_reason():
    v = parse_audit_verdict({"same_morpheme": True, "reason": "z" * 500})
    assert len(v.reason) == 200


def test_build_audit_judge_prompt_includes_forms_and_handles_empty_glosses():
    c = MergeAuditCandidate(
        anchor_ref="modern-english:ton",
        member_ref="modern-english:-ton",
        provenance=PROV_OCR,
        anchor_form="ton",
        member_form="-ton",
        anchor_glosses=("units of mass",),
        member_glosses=(),
    )
    system, user = build_audit_judge_prompt(c)
    assert "ton" in user and "-ton" in user
    assert "(none)" in user  # empty member glosses render as (none)
    assert "JSON object" in system


# --- collapse-fold resolve-away edge -----------------------------------------


def test_collapse_fold_unresolved_ref_skipped(tmp_path):
    """A collapse_state fold whose ref/into no longer resolve in the DB is
    skipped (already reverted / typo), not crashed on."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "john", ["a name"])  # 'into' exists, but the folded ref doesn't
    collapse_state = {"modern-english:ghostform": {"into": "modern-english:john"}}
    assert detect_merge_audit_candidates(conn, collapse_state, scope="collapses") == []


# --- rebuild round-trip: the audit log must replay, not crash -----------------


def test_merge_audit_jsonl_replays_inert(tmp_path):
    """wyrd-6hbv reconstructibility: _merge_audit.jsonl must replay through
    rebuild-from-jsonl's glob without raising (the merge_audit _type is a
    registered inert LIST_TYPE). Regression for the unregistered-type crash."""
    from wyrd.generators.kenning.cli.lexicon.audit_merges import _MERGE_AUDIT_SOURCE_ROW
    from wyrd.generators.kenning.jsonl.log import replay_file

    path = tmp_path / "_merge_audit.jsonl"
    rows = [
        _MERGE_AUDIT_SOURCE_ROW,
        {
            "_type": "merge_audit",
            "member_ref": "modern-english:-ton",
            "anchor_ref": "modern-english:ton",
            "provenance": PROV_OCR,
            "verdict": "revert",
            "confidence": "high",
            "reason": "suffix not weight",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    state = replay_file(path)  # must not raise ReplayError
    assert [r["member_ref"] for r in state.lists["merge_audit"]] == ["modern-english:-ton"]


# --- CLI helpers (idempotency gate + revert-write contract) -------------------


def test_cli_load_judged_filters(tmp_path):
    from wyrd.generators.kenning.cli.lexicon.audit_merges import _load_judged

    path = tmp_path / "_merge_audit.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"_type": "source", "ref": "merge-audit"},
                {"_type": "merge_audit", "member_ref": "a:x", "verdict": "keep"},
                {"_type": "merge_audit", "member_ref": "a:y", "verdict": "revert"},
                {"_type": "merge_audit"},  # missing member_ref → ignored
                {"_type": "other", "member_ref": "a:z"},  # wrong type → ignored
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert _load_judged(path) == {"a:x", "a:y"}
    assert _load_judged(tmp_path / "absent.jsonl") == set()  # no file → empty


def test_cli_write_rows_routes_to_handles():
    import io

    from wyrd.generators.kenning.cli.lexicon.audit_merges import _write_rows

    audit_fh, collapse_fh, curation_fh = io.StringIO(), io.StringIO(), io.StringIO()
    # a keep verdict writes only the audit log
    keep = audit_verdict_to_rows(
        _cand("m:-land", "m:land", PROV_OCR),
        MergeAuditVerdict(correct=True, confidence="high", reason="same"),
        "medium",
    )
    _write_rows(keep, audit_fh, collapse_fh, curation_fh)
    assert audit_fh.getvalue() and not collapse_fh.getvalue() and not curation_fh.getvalue()
    # an ocr revert also writes the curation ledger, not the collapse ledger
    revert = audit_verdict_to_rows(
        _cand("m:-ton", "m:ton", PROV_OCR),
        MergeAuditVerdict(correct=False, confidence="high", reason="distinct"),
        "medium",
    )
    _write_rows(revert, audit_fh, collapse_fh, curation_fh)
    assert curation_fh.getvalue() and not collapse_fh.getvalue()
