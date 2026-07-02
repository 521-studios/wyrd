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
    _sub_sense_token_sets,
    build_audit_judge_prompt,
    detect_merge_audit_candidates,
    parse_audit_verdict,
    revert_rows,
    verdict_from_log,
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


def test_glossless_anchor_disjoint_members_are_skipped_not_guessed(tmp_path):
    """wyrd-bk69: a GLOSSLESS anchor gives no evidence for the family's correct
    sense, so the disjoint-sense screen SKIPS it rather than falling back to the
    largest member cluster as a guessed 'dominant' and over-flagging the rest.
    Here the anchor is glossless and its two glossed members carry disjoint
    senses; no candidate is emitted (pre-fix, the smaller member was flagged an
    intruder off a coin-flip dominant)."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "ton", [])  # glossless family head — no anchor evidence
    _ety(conn, 2, "tonne", ["Any of various units of mass"], merged_into=1)
    _ety(conn, 3, "tun", ["An enclosure; a farmstead, a village"], merged_into=1)
    assert detect_merge_audit_candidates(conn, {}, scope="anchors") == []


def test_glossed_anchor_still_flags_disjoint_intruder(tmp_path):
    """The glossless guard is narrow: a GLOSSED anchor still drives the
    disjoint-sense screen normally (regression guard for the wyrd-bk69 skip)."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "ing", ["patronymic descendants people"], lang="old-english")
    _ety(conn, 2, "ingas", ["patronymic son descendants"], lang="old-english", merged_into=1)
    _ety(conn, 3, "eng", ["water-meadow grassland"], lang="old-english", merged_into=1)
    refs = {c.member_ref for c in detect_merge_audit_candidates(conn, {}, scope="anchors")}
    assert "old-english:eng" in refs  # topographic intruder flagged
    assert "old-english:ingas" not in refs  # patronymic core (with anchor) spared


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


def test_detect_disjoint_intruder_despite_mixed_gloss_on_anchor(tmp_path):
    """wyrd-uumc: the real -ing blob — the ANCHOR carries a MIXED gloss
    ('patronymic descendants; or water-meadow') whose tokens span BOTH senses.
    Pre-fix, the gloss-token union-find unioned the anchor's combined tokens with
    both the patronymic members AND the topographic intruder → ONE component →
    not flagged disjoint → 'ing' (water-meadow) silently survived on -ing.
    Splitting the gloss on ';'/'or' into sub-senses lets the mixed anchor vote for
    each cluster WITHOUT bridging them, so the topographic intruder is flagged."""
    conn = _conn(tmp_path / "lex.db")
    # Anchor with a conflated mixed gloss (the bridging gloss) + a clean one.
    _ety(
        conn,
        1,
        "-ing",
        ["patronymic descendants; or water-meadow", "patronymic suffix"],
        lang="old-english",
    )
    # Patronymic core (dominant: more members) — must NOT be flagged.
    _ety(conn, 2, "-ingas", ["patronymic descendants people"], lang="old-english", merged_into=1)
    _ety(conn, 4, "-ung", ["patronymic descendants"], lang="old-english", merged_into=1)
    # Pure topographic intruder — the one to revert off -ing.
    _ety(conn, 3, "ing", ["water-meadow grassland"], lang="old-english", merged_into=1)
    cands = detect_merge_audit_candidates(conn, {}, scope="anchors")
    refs = {c.member_ref for c in cands}
    assert "old-english:ing" in refs  # topographic intruder flagged despite the mixed anchor
    assert "old-english:-ingas" not in refs  # patronymic core spared
    assert "old-english:-ung" not in refs
    assert "old-english:-ing" not in refs  # the mixed anchor itself is never an intruder


def test_sub_sense_token_sets_splits_on_semicolon_and_or_not_comma():
    """wyrd-uumc: the sense-split boundaries are ';' and the WORD 'or' — NOT comma
    (comma phrases are one sense), and not 'or' inside a word."""
    # ';' splits → 2 sub-senses
    assert len(_sub_sense_token_sets({"patronymic descendants; water-meadow"})) == 2
    # ' or ' splits → 2 sub-senses
    assert len(_sub_sense_token_sets({"patronymic descendants or water-meadow"})) == 2
    # comma does NOT split — one conflated wetland list stays ONE sub-sense
    assert len(_sub_sense_token_sets({"meadow, water meadow, marsh"})) == 1
    # 'or' inside a word ('border', 'corn') must not false-split
    assert len(_sub_sense_token_sets({"border fortress stronghold"})) == 1
    # empty sub-senses (a trailing ';') are dropped, not emitted as empty sets
    assert all(s for s in _sub_sense_token_sets({"hill;"}))


def test_detect_disjoint_intruder_via_or_only_boundary(tmp_path):
    """wyrd-uumc: the ' or ' boundary in isolation (no ';') also un-bridges a mixed
    gloss so the topographic intruder is flagged. Patronymic is the majority sense
    (2 members vs 1) so it's unambiguously dominant."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "-ing", ["patronymic descendants or water-meadow"], lang="old-english")
    _ety(conn, 2, "-ingas", ["patronymic descendants people"], lang="old-english", merged_into=1)
    _ety(conn, 4, "-ung", ["patronymic descendants"], lang="old-english", merged_into=1)
    _ety(conn, 3, "ing", ["water-meadow grassland"], lang="old-english", merged_into=1)
    refs = {c.member_ref for c in detect_merge_audit_candidates(conn, {}, scope="anchors")}
    assert "old-english:ing" in refs
    assert "old-english:-ingas" not in refs
    assert "old-english:-ung" not in refs


def test_disjoint_dominant_tie_is_deterministic(tmp_path):
    """wyrd-uumc: when two disjoint senses are EQUAL size (anchor in both), the
    dominant pick is structurally ambiguous — but it must be DETERMINISTIC, not
    dependent on incidental ordering. The real order signal is the ANCHOR's gloss
    SENSE-ORDER: _anchor_intruders always puts the anchor first, so cluster return
    order follows which sub-sense the anchor lists first. A plain key=len max would
    flag whichever side the anchor happens to mention first (flaky); the
    (len, sorted-eids) secondary key flags the SAME side regardless. The LLM judge
    arbitrates the flagged side."""

    def flagged(anchor_gloss):
        slug = anchor_gloss.replace(" ", "").replace(";", "")[:12]
        conn = _conn(tmp_path / f"lex_{slug}.db")
        # distinct fillers (people / grassland) so the two members DON'T share a
        # token — only the anchor's mixed gloss spans both senses → a clean 2-2 tie.
        _ety(conn, 1, "-x", [anchor_gloss], lang="old-english")
        _ety(conn, 2, "xa", ["alpha beta people"], lang="old-english", merged_into=1)
        _ety(conn, 3, "xb", ["gamma delta grassland"], lang="old-english", merged_into=1)
        return {c.member_ref for c in detect_merge_audit_candidates(conn, {}, scope="anchors")}

    # Same two equal clusters {1,2}(alpha/beta) and {1,3}(gamma/delta), but the
    # anchor lists the senses in OPPOSITE order — which flips the cluster return
    # order. A key=len max would flag different sides; the sorted-eids key must not.
    one = flagged("alpha beta; or gamma delta")
    two = flagged("gamma delta; or alpha beta")
    assert one == two  # deterministic despite the anchor's sense-order flip
    assert len(one) == 1  # exactly one side flagged (recall screen; LLM arbitrates)


def test_detect_mixed_non_anchor_member_sharing_dominant_is_spared(tmp_path):
    """wyrd-uumc: the ``- dominant`` term — a NON-anchor member whose mixed gloss
    carries BOTH the dominant (patronymic) sense AND a topographic one BELONGS (it
    shares the dominant sense), so it must NOT be flagged; only the pure
    topographic intruder is."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "-ing", ["patronymic descendants"], lang="old-english")
    _ety(conn, 2, "-ingas", ["patronymic descendants people"], lang="old-english", merged_into=1)
    # mixed CHILD: shares the dominant patronymic sense + a water-meadow sub-sense
    _ety(
        conn,
        5,
        "-ling",
        ["patronymic descendants; or water-meadow"],
        lang="old-english",
        merged_into=1,
    )
    # pure topographic intruder
    _ety(conn, 3, "ing", ["water-meadow grassland"], lang="old-english", merged_into=1)
    refs = {c.member_ref for c in detect_merge_audit_candidates(conn, {}, scope="anchors")}
    assert "old-english:ing" in refs  # pure intruder flagged
    assert "old-english:-ling" not in refs  # mixed member sharing dominant → spared
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
    """A member produced by BOTH surviving sources — the disjoint-sense anchor
    screen AND a live collapse fold — is judged once (deduped by member ref), and
    collapse-fold provenance wins (it IS a live fold, layered first). 'eng'
    (water-meadow) is a disjoint intruder off the glossed 'ing' anchor and is also
    a live fold into it."""
    conn = _conn(tmp_path / "lex.db")
    _ety(conn, 1, "ing", ["patronymic descendants people"], lang="old-english")
    _ety(conn, 2, "ingas", ["patronymic son descendants"], lang="old-english", merged_into=1)
    _ety(conn, 3, "eng", ["water-meadow grassland"], lang="old-english", merged_into=1)
    collapse_state = {"old-english:eng": {"into": "old-english:ing"}}
    cands = detect_merge_audit_candidates(conn, collapse_state, scope="both")
    members = [c.member_ref for c in cands]
    assert members.count("old-english:eng") == 1  # deduped across the two sources
    eng = next(c for c in cands if c.member_ref == "old-english:eng")
    assert eng.provenance == PROV_COLLAPSE  # collapse-fold wins on dedup


# --- verdict routing -------------------------------------------------------


def test_revert_rows_lemma_provenance_keys_lemma_ref():
    """PROV_LEMMA wrong-merge revert routes to a curation row keyed
    ``lemma_ref`` (not ``merged_into_ref``), with no collapse row. Re-pins the
    provenance ternary in the live ``revert_rows`` after ``audit_verdict_to_rows``
    — its former sole test caller — was removed in the PR #498 dead-code audit."""
    v = MergeAuditVerdict(correct=False, confidence="high", reason="distinct")
    collapse, curation = revert_rows("oe:streamX", PROV_LEMMA, v, "medium")
    assert collapse is None
    assert curation["lemma_ref"] is None
    assert "merged_into_ref" not in curation


# --- raw-verdict round-trip: emit high now, re-derive medium later -----------


def test_verdict_from_log_rejects_legacy_row_without_raw_verdict():
    assert verdict_from_log({"member_ref": "m:x", "confidence": "high"}) is None


def test_verdict_from_log_defaults_bad_confidence_low():
    """A raw bool with a junk/absent confidence (schema drift) defaults to low."""
    assert verdict_from_log({"same_morpheme": False, "confidence": "bogus"}).confidence == "low"
    assert verdict_from_log({"same_morpheme": True}).confidence == "low"


def test_revert_rows_re_derive_at_lower_threshold():
    """The recorded raw verdict yields NO revert at high but a curation revert at
    medium — the 'emit medium later' path, no LLM re-run."""
    v = MergeAuditVerdict(correct=False, confidence="medium", reason="distinct")
    assert revert_rows("m:-ton", PROV_OCR, v, "high") == (None, None)
    collapse, curation = revert_rows("m:-ton", PROV_OCR, v, "medium")
    assert collapse is None and curation["merged_into_ref"] is None


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
            "same_morpheme": False,
            "confidence": "high",
            "reason": "suffix not weight",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    state = replay_file(path)  # must not raise ReplayError
    assert [r["member_ref"] for r in state.lists["merge_audit"]] == ["modern-english:-ton"]


# --- CLI helpers (idempotency gate + revert-write contract) -------------------


def test_cli_load_log_returns_rows_by_ref(tmp_path):
    from wyrd.generators.kenning.cli.lexicon.audit_merges import _load_log

    path = tmp_path / "_merge_audit.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"_type": "source", "ref": "merge-audit"},
                {"_type": "merge_audit", "member_ref": "a:x", "same_morpheme": True},
                {"_type": "merge_audit", "member_ref": "a:y", "same_morpheme": False},
                # legacy threshold-gated row (no same_morpheme) → excluded so it
                # gets re-judged rather than silently lost.
                {"_type": "merge_audit", "member_ref": "a:legacy", "verdict": "revert"},
                {"_type": "merge_audit"},  # missing member_ref → ignored
                {"_type": "other", "member_ref": "a:z"},  # wrong type → ignored
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log = _load_log(path)
    assert set(log) == {"a:x", "a:y"}  # a:legacy excluded → will be re-judged
    assert log["a:y"]["same_morpheme"] is False
    assert _load_log(tmp_path / "absent.jsonl") == {}  # no file → empty


def test_cli_already_reverted_dedup():
    """A re-run must not re-append a revert already recorded in the net ledger
    state (collapse into:'' for collapse-fold; cleared ref for ocr/lemma)."""
    from wyrd.generators.kenning.cli.lexicon.audit_merges import _already_reverted

    # collapse-fold: net into already empty → already reverted
    assert _already_reverted("m:x", PROV_COLLAPSE, {"m:x": {"into": ""}}, {})
    assert not _already_reverted("m:x", PROV_COLLAPSE, {"m:x": {"into": "m:y"}}, {})
    # ocr: curation already clears merged_into_ref
    assert _already_reverted("m:-ton", PROV_OCR, {}, {"m:-ton": {"merged_into_ref": None}})
    assert not _already_reverted("m:-ton", PROV_OCR, {}, {})
    # lemma: curation already clears lemma_ref
    assert _already_reverted("m:s", PROV_LEMMA, {}, {"m:s": {"lemma_ref": None}})


def test_cli_emit_reverts_from_log_dedups_and_thresholds():
    """_emit_reverts derives reverts from recorded raw verdicts at a threshold,
    skips below-threshold (kept) + already-reverted, and re-derives at a lower
    threshold — the emit-high-now/emit-medium-later contract, no LLM."""
    import io

    from wyrd.generators.kenning.cli.lexicon.audit_merges import _emit_reverts

    log = {
        "m:-ton": {
            "member_ref": "m:-ton",
            "anchor_ref": "m:ton",
            "provenance": PROV_OCR,
            "same_morpheme": False,
            "confidence": "medium",
            "reason": "distinct",
        },
        "m:-land": {
            "member_ref": "m:-land",
            "anchor_ref": "m:land",
            "provenance": PROV_OCR,
            "same_morpheme": True,
            "confidence": "high",
            "reason": "same",
        },
    }
    # at HIGH: -ton (medium) is below threshold → kept, nothing emitted
    cur = io.StringIO()
    counts = _emit_reverts(log, "high", {}, {}, io.StringIO(), cur, dry_run=False)
    assert counts["reverts"] == 0 and counts["kept"] == 2 and not cur.getvalue()
    # at MEDIUM: -ton now emits a curation revert (re-derived from the SAME log)
    cur = io.StringIO()
    counts = _emit_reverts(log, "medium", {}, {}, io.StringIO(), cur, dry_run=False)
    assert counts["reverts"] == 1 and counts["ocr-cluster"] == 1
    assert json.loads(cur.getvalue())["ref"] == "m:-ton"
    # already-reverted in the net curation state → skipped, not re-appended
    cur = io.StringIO()
    counts = _emit_reverts(
        log, "medium", {}, {"m:-ton": {"merged_into_ref": None}}, io.StringIO(), cur, dry_run=False
    )
    assert counts["reverts"] == 0 and counts["already"] == 1 and not cur.getvalue()


def test_cli_emit_reverts_collapse_fold_and_dry_run():
    """A collapse-fold revert writes the COLLAPSE ledger (into:''); dry-run counts
    but writes nothing."""
    import io

    from wyrd.generators.kenning.cli.lexicon.audit_merges import _emit_reverts

    log = {
        "m:fold": {
            "member_ref": "m:fold",
            "anchor_ref": "m:keep",
            "provenance": PROV_COLLAPSE,
            "same_morpheme": False,
            "confidence": "high",
            "reason": "distinct",
        },
    }
    # the fold is still live (net into set) → revert emitted to the collapse ledger
    coll = io.StringIO()
    counts = _emit_reverts(
        log, "high", {"m:fold": {"into": "m:keep"}}, {}, coll, io.StringIO(), dry_run=False
    )
    assert counts["reverts"] == 1 and counts["collapse-fold"] == 1
    assert json.loads(coll.getvalue()) == {
        "_type": "collapse",
        "ref": "m:fold",
        "into": "",
        "method": LLM_MERGE_AUDIT_REVERT_METHOD,
        "confidence": "high",
        "reason": "distinct",
    }
    # dry-run: counts the revert but writes nothing
    coll = io.StringIO()
    counts = _emit_reverts(
        log, "high", {"m:fold": {"into": "m:keep"}}, {}, coll, io.StringIO(), dry_run=True
    )
    assert counts["reverts"] == 1 and not coll.getvalue()


def test_judge_fresh_aborts_after_consecutive_failures(monkeypatch):
    """PASS 1 aborts after 5 consecutive judge failures (Ollama down) instead of
    burning every candidate; records nothing for the failures."""
    from wyrd.generators.kenning.cli.lexicon import audit_merges

    monkeypatch.setattr(audit_merges, "_judge_with_retry", lambda client, c: None)
    cands = [_cand(f"m:c{i}", "m:a", PROV_OCR) for i in range(10)]
    log: dict = {}
    judged, skipped = audit_merges._judge_fresh(None, cands, log, None, "url", "model")
    assert judged == 0 and skipped == 5 and log == {}  # broke after the 5th failure


def test_judge_fresh_resets_counter_and_records(monkeypatch):
    """A success resets the consecutive-failure counter (so scattered failures
    don't trip the abort) and records the raw verdict into the log."""
    from wyrd.generators.kenning.cli.lexicon import audit_merges

    ok = MergeAuditVerdict(correct=False, confidence="high", reason="distinct")
    # fail, fail, SUCCESS, fail, fail, fail, fail → max-consecutive 4, no abort
    seq = iter([None, None, ok, None, None, None, None])
    monkeypatch.setattr(audit_merges, "_judge_with_retry", lambda client, c: next(seq))
    cands = [_cand(f"m:c{i}", "m:a", PROV_OCR) for i in range(7)]
    log: dict = {}
    judged, skipped = audit_merges._judge_fresh(None, cands, log, None, "url", "model")
    assert judged == 1 and skipped == 6
    assert log["m:c2"]["same_morpheme"] is False  # the success was recorded raw
