"""wyrd-u6fn.5 — duplicate-canonical finder (detector + two-pass judge + Family-A
authoring + the niwe≈ne projection round-trip regression)."""

from __future__ import annotations

import sqlite3

import pytest

from wyrd.generators.kenning.canonicalization.assertions import validate
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.duplicate_canonical_mining import (
    DuplicateCandidate,
    MergeVerdict,
    _Propose,
    _Refute,
    build_propose_prompt,
    build_refute_prompt,
    combine_verdict,
    detect_candidates,
    parse_propose,
    parse_refute,
    same_morpheme_assertions,
)
from wyrd.generators.kenning.lexicon.etymon_refs import etymon_ref
from wyrd.generators.kenning.lexicon.schema import init_schema


def _etymon(db, form, lang, glosses):
    eid = db.upsert_etymon(form, lang)
    for g in glosses:
        db.add_gloss(eid, g)
    return eid


def _ocr_variant(db, root_id, form, lang):
    """An etymon merged into ``root_id`` (an OCR-variant tombstone) — makes root_id a
    legacy multi-member family root."""
    vid = db.upsert_etymon(form, lang)
    db.conn.execute("UPDATE etymon SET merged_into_id=? WHERE id=?", (root_id, vid))
    return vid


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


def _cand(a_id=1, b_id=2, lang="old-english", a_form="niwe", b_form="ne"):
    return DuplicateCandidate(
        a_id=a_id,
        a_form=a_form,
        b_id=b_id,
        b_form=b_form,
        language=lang,
        a_glosses=("new",),
        b_glosses=("new",),
        gloss_overlap=1.0,
    )


# --- detector ---------------------------------------------------------------


def test_detect_flags_niwe_ne_excludes_low_overlap_and_cross_language(lex):
    """The regression pair (both glossed 'new', same language) is flagged; a pair
    sharing no gloss token, and a same-gloss pair in a DIFFERENT language, are not."""
    niwe = _etymon(lex, "niwe", "old-english", ["new"])
    ne = _etymon(lex, "ne", "old-english", ["new"])
    _etymon(lex, "ford", "old-english", ["ford", "river crossing"])  # no gloss overlap
    _etymon(lex, "nieuw", "dutch", ["new"])  # same gloss, different language
    lex.commit()
    pairs = {(c.a_id, c.b_id) for c in detect_candidates(lex.conn).candidates}
    assert (min(niwe, ne), max(niwe, ne)) in pairs
    # only the OE niwe/ne pair — no cross-language, no zero-overlap pairing
    assert len(pairs) == 1


def test_detect_skips_already_collapsed_pair(lex):
    """A pair already sharing a canonical_morpheme_id (collapsed by a prior run) is
    not re-proposed."""
    niwe = _etymon(lex, "niwe", "old-english", ["new"])
    ne = _etymon(lex, "ne", "old-english", ["new"])
    lex.conn.execute("INSERT INTO canonical_morpheme (id) VALUES ('H')")
    lex.conn.execute("UPDATE etymon SET canonical_morpheme_id='H' WHERE id IN (?,?)", (niwe, ne))
    lex.commit()
    assert detect_candidates(lex.conn).candidates == []


def test_detect_jaccard_threshold(lex):
    """Partial gloss overlap below the Jaccard threshold is excluded; raising it in."""
    _etymon(lex, "aa", "x", ["new", "fresh", "young", "recent"])
    _etymon(lex, "bb", "x", ["new"])  # overlap 1/4 = 0.25
    lex.commit()
    assert detect_candidates(lex.conn, min_gloss_overlap=0.5).candidates == []
    assert len(detect_candidates(lex.conn, min_gloss_overlap=0.2).candidates) == 1


def test_detect_reports_oversized_bucket(lex):
    """A gloss-token bucket larger than the cap is skipped and counted (no silent drop)."""
    for i in range(5):
        _etymon(lex, f"f{i}", "x", ["common"])
    lex.commit()
    det = detect_candidates(lex.conn, max_bucket=3)
    assert det.dropped_buckets == 1 and det.candidates == []


# --- two-pass judge ---------------------------------------------------------


def test_parse_propose_and_refute():
    assert parse_propose({"same_morpheme": True, "confidence": "high", "reason": "x"}) == _Propose(
        True, "high", "x"
    )
    assert parse_propose({"same_morpheme": "yes"}).same is True
    assert parse_propose({"same_morpheme": False, "confidence": "bogus"}).confidence == "low"
    assert parse_propose({"confidence": "high"}) is None  # no verdict field
    assert parse_propose("nope") is None
    assert parse_refute({"refuted": True, "confidence": "medium", "reason": "y"}) == _Refute(
        True, "medium", "y"
    )
    assert parse_refute({}) is None


def test_combine_verdict_leaves_separate_unless_proposed_and_unrefuted():
    hi_same = _Propose(True, "high", "same word")
    # not proposed → separate
    assert combine_verdict(_Propose(False, "high", "distinct"), None).same is False
    # proposed but refuted → separate
    assert combine_verdict(hi_same, _Refute(True, "medium", "different root")).same is False
    # proposed, verify call failed (None) → separate (leave-separate on doubt)
    assert combine_verdict(hi_same, None).same is False
    # no proposal at all → separate
    assert combine_verdict(None, None).same is False
    # proposed high, survives a MEDIUM refute → same, floored to medium
    v = combine_verdict(hi_same, _Refute(False, "medium", "holds"))
    assert v.same is True and v.confidence == "medium"
    # proposed high, survives a HIGH refute → same at high
    assert combine_verdict(hi_same, _Refute(False, "high", "clearly one word")).confidence == "high"


def test_prompts_carry_both_forms_and_glosses():
    c = _cand()
    ps, pu = build_propose_prompt(c)
    rs, ru = build_refute_prompt(c)
    assert "place-name" in ps.lower() and "niwe" in pu and "ne" in pu and "new" in pu
    assert "refute" in rs.lower() and "niwe" in ru and "ne" in ru


# --- Family-A authoring -----------------------------------------------------


def test_same_morpheme_assertions_are_valid_family_a():
    c = _cand(a_id=671826, b_id=369498)  # winner = smaller id = 369498
    v = MergeVerdict(True, "high", "both 'new'")
    asserts = same_morpheme_assertions(c, v, source="t")
    for a in asserts:
        validate(a)  # raises on contract violation
    preds = [a.predicate for a in asserts]
    assert preds.count("mint-canonical") == 2
    assert preds.count("bind") == 2
    assert preds.count("merge-canonical") == 1
    binds = [a for a in asserts if a.predicate == "bind"]
    assert all(a.qualifiers == {"kind": "same-morpheme"} for a in binds)
    # wyrd-c6wu: bind refs are stable natural keys, not etymon row-ids.
    assert {a.subject.ref for a in binds} == {
        etymon_ref("old-english", "niwe"),
        etymon_ref("old-english", "ne"),
    }
    assert all(a.confidence == "high" for a in binds)


# --- the regression: author → project → collapsed (niwe ≈ ne) ---------------


def _canonical_root(db, etymon_id):
    """Resolve an etymon's canonical_morpheme_id through the merged_into chain."""
    row = db.conn.execute(
        "SELECT canonical_morpheme_id FROM etymon WHERE id=?", (etymon_id,)
    ).fetchone()
    node = row[0]
    seen = set()
    while node is not None and node not in seen:
        seen.add(node)
        nxt = db.conn.execute(
            "SELECT merged_into FROM canonical_morpheme WHERE id=?", (node,)
        ).fetchone()
        if nxt is None or nxt[0] is None:
            break
        node = nxt[0]
    return node


def test_projection_collapses_multi_member_families_preserving_fidelity(lex, tmp_path):
    """niwe and ne each anchor a legacy OCR family; authoring + projection collapses
    BOTH families into ONE canonical root (the merge-canonical brings each family
    along), and identity fidelity holds (canonical may JOIN legacy families, never
    SPLIT). Uses real multi-member families so the fidelity check is non-vacuous."""
    from wyrd.generators.kenning.canonicalization import append_assertion
    from wyrd.generators.kenning.lexicon.canonicalization_projection import (
        assess_identity_fidelity,
        project_canonical,
    )

    niwe = _etymon(lex, "niwe", "old-english", ["new"])
    niwa = _ocr_variant(lex, niwe, "niwa", "old-english")  # OCR variant of niwe
    ne = _etymon(lex, "ne", "old-english", ["new"])
    nee = _ocr_variant(lex, ne, "nee", "old-english")  # OCR variant of ne
    lex.commit()
    c = DuplicateCandidate(
        a_id=niwe,
        a_form="niwe",
        b_id=ne,
        b_form="ne",
        language="old-english",
        a_glosses=("new",),
        b_glosses=("new",),
        gloss_overlap=1.0,
    )
    for a in same_morpheme_assertions(c, MergeVerdict(True, "high", "both 'new'"), source="t"):
        append_assertion(tmp_path, a)

    project_canonical(lex, mining_dir=tmp_path, apply=True, confidence_gate="high")
    roots = {_canonical_root(lex, e) for e in (niwe, niwa, ne, nee)}
    assert len(roots) == 1 and None not in roots  # all four JOINED into one node (the merge)
    # fidelity checks the no-SPLIT invariant: each legacy family maps to one canonical
    # group (non-vacuous here — two real 2-member families exist).
    fid = assess_identity_fidelity(lex)
    assert fid.legacy_families >= 2 and fid.violations == 0


def test_detect_excludes_non_root_survivor(lex):
    """An inflection survivor (merged_into_id NULL but lemma_id set → its legacy hub is
    the lemma's, not its own) is excluded — pairing it would mint a divergent hub."""
    niwe = _etymon(lex, "niwe", "old-english", ["new"])
    infl = _etymon(lex, "niwum", "old-english", ["new"])  # an inflected form of a lemma
    lemma = _etymon(lex, "neowe", "old-english", ["new"])
    lex.conn.execute("UPDATE etymon SET lemma_id=? WHERE id=?", (lemma, infl))
    lex.commit()
    ids = {(c.a_id, c.b_id) for c in detect_candidates(lex.conn).candidates}
    assert all(infl not in pair for pair in ids)  # the inflection child is never paired
    assert (min(niwe, lemma), max(niwe, lemma)) in ids  # the two roots still pair


# --- CLI orchestration (two-pass judge mocked — no live Ollama) -------------


class _Client:
    """Fake OllamaClient: dispatches propose vs refute by the 'refute' marker present
    in the refute prompt (system+user), robust to either being reworded."""

    def __init__(self, propose, refute=None):
        self.propose, self.refute = propose, refute

    def chat_json(self, system, user, schema):
        return self.refute if "refute" in (system + user).lower() else self.propose


def test_load_judged_tolerates_corrupt_and_filters(tmp_path):
    import json

    from wyrd.generators.kenning.cli.lexicon import mine_duplicate_canonicals as cli

    cdir = tmp_path / "canonicalization"
    cdir.mkdir()
    (cdir / "_duplicate_canonical_audit.jsonl").write_text(
        json.dumps({"_type": "duplicate_canonical_audit", "pair": "1:2", "same": True})
        + "\n{bad json\n"
        + json.dumps({"_type": "other", "pair": "9:9"})  # wrong _type → skipped
        + "\n"
        + json.dumps({"_type": "duplicate_canonical_audit", "pair": 5})  # non-str → skipped
        + "\n",
        encoding="utf-8",
    )
    assert cli._load_judged(tmp_path) == {"1:2"}


def test_judge_pair_two_pass():
    from wyrd.generators.kenning.cli.lexicon import mine_duplicate_canonicals as cli

    c = _cand()
    # propose says not-same → leave-separate; refute never consulted
    v = cli._judge_pair(
        _Client({"same_morpheme": False, "confidence": "high", "reason": "distinct"}), c
    )
    assert v.same is False
    # proposed same but refuted → separate
    v = cli._judge_pair(
        _Client(
            {"same_morpheme": True, "confidence": "high", "reason": "x"},
            {"refuted": True, "confidence": "medium", "reason": "diff root"},
        ),
        c,
    )
    assert v.same is False
    # proposed same, survives a medium refute → same, floored to medium
    v = cli._judge_pair(
        _Client(
            {"same_morpheme": True, "confidence": "high", "reason": "x"},
            {"refuted": False, "confidence": "medium", "reason": "holds"},
        ),
        c,
    )
    assert v.same is True and v.confidence == "medium"
    # propose call never parses → None (skip / counts toward abort)
    assert cli._judge_pair(_Client("garbage"), c) is None
    # proposed same, but the refute call fails transiently (None) → None (retryable skip,
    # NOT a permanent 'separate' that would bury the pair in the judged-log)
    assert (
        cli._judge_pair(
            _Client({"same_morpheme": True, "confidence": "high", "reason": "x"}, "garbage"), c
        )
        is None
    )


def test_judge_loop_authors_high_queues_medium_separates_and_dedups(tmp_path, monkeypatch):
    from wyrd.generators.kenning.canonicalization import load_assertions
    from wyrd.generators.kenning.cli.lexicon import mine_duplicate_canonicals as cli

    hi = DuplicateCandidate(1, "niwe", 2, "ne", "oe", ("new",), ("new",), 1.0)
    med = DuplicateCandidate(3, "aa", 4, "bb", "oe", ("hill",), ("hill",), 1.0)
    sep = DuplicateCandidate(5, "ea", 6, "eg", "oe", ("water",), ("water",), 1.0)
    verdicts = {
        1: MergeVerdict(True, "high", "both new"),
        3: MergeVerdict(True, "medium", "maybe"),
        5: MergeVerdict(False, "high", "distinct"),
    }
    monkeypatch.setattr(cli, "_judge_pair", lambda client, c: verdicts[c.a_id])
    counts = cli._judge_loop(
        None,
        [hi, med, sep],
        mining_dir=tmp_path,
        source="t",
        min_confidence="high",
        apply=True,
        existing_ids=set(),
        audit_fh=None,
        base_url="u",
        model="m",
    )
    assert counts == {"authored": 1, "same": 1, "queued": 1, "separate": 1, "skipped": 0}
    assert len(list(load_assertions(tmp_path))) == 5  # the high pair's mint×2+bind×2+merge
    # dedup: re-run the high pair with its ids present → nothing new authored
    existing = {a.id for a in load_assertions(tmp_path)}
    counts2 = cli._judge_loop(
        None,
        [hi],
        mining_dir=tmp_path,
        source="t",
        min_confidence="high",
        apply=True,
        existing_ids=existing,
        audit_fh=None,
        base_url="u",
        model="m",
    )
    assert counts2["authored"] == 0 and len(list(load_assertions(tmp_path))) == 5


def test_judge_loop_aborts_after_consecutive_failures(monkeypatch):
    import click

    from wyrd.generators.kenning.cli.lexicon import mine_duplicate_canonicals as cli

    monkeypatch.setattr(cli, "_judge_pair", lambda client, c: None)
    cands = [DuplicateCandidate(i, "a", i + 100, "b", "oe", ("g",), ("g",), 1.0) for i in range(6)]
    with pytest.raises(click.ClickException):
        cli._judge_loop(
            None,
            cands,
            mining_dir=None,
            source="t",
            min_confidence="high",
            apply=False,
            existing_ids=set(),
            audit_fh=None,
            base_url="u",
            model="m",
        )


def test_cli_apply_collapses_pair_end_to_end(lex, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon import mine_duplicate_canonicals as cli
    from wyrd.generators.kenning.lexicon.canonicalization_projection import project_canonical

    niwe = _etymon(lex, "niwe", "old-english", ["new"])
    ne = _etymon(lex, "ne", "old-english", ["new"])
    lex.commit()
    monkeypatch.setattr(
        cli, "_judge_pair", lambda client, c: MergeVerdict(True, "high", "both 'new'")
    )
    mining = tmp_path / "mining"
    argv = ["--db", str(lex.path), "--mining-dir", str(mining), "--apply"]
    res = CliRunner().invoke(cli.lexicon_mine_duplicate_canonicals, argv)
    assert res.exit_code == 0, res.output
    # audit row reads back; second --apply judges nothing (idempotent round-trip)
    assert cli._load_judged(mining) == {f"{min(niwe, ne)}:{max(niwe, ne)}"}
    res2 = CliRunner().invoke(cli.lexicon_mine_duplicate_canonicals, argv)
    assert "to-judge=0" in res2.output
    # the authored assertions actually collapse the pair when projected
    project_canonical(lex, mining_dir=mining, apply=True, confidence_gate="high")
    assert _canonical_root(lex, niwe) is not None
    assert _canonical_root(lex, niwe) == _canonical_root(lex, ne)


def test_maybe_pair_skips_glossless_and_already_collapsed():
    from wyrd.generators.kenning.lexicon.duplicate_canonical_mining import _maybe_pair

    # by_id entries always carry a precomputed "fold" (detect_candidates sets it).
    full = {
        "cm_id": None,
        "form": "a",
        "fold": "a",
        "lang": "x",
        "glosses": {"new"},
        "tokens": {"new"},
    }
    glossless = {
        "cm_id": None,
        "form": "b",
        "fold": "b",
        "lang": "x",
        "glosses": set(),
        "tokens": set(),
    }
    assert _maybe_pair(full, glossless, 1, 2, 0.5) is None  # no shared tokens to compare
    same_hub_a = {**full, "cm_id": "H"}
    same_hub_b = {**full, "cm_id": "H", "form": "b"}
    assert _maybe_pair(same_hub_a, same_hub_b, 1, 2, 0.5) is None  # already collapsed
    assert _maybe_pair(full, {**full, "form": "b"}, 1, 2, 0.5) is not None  # genuine pair


def test_call_retries_then_gives_up():
    from wyrd.generators.kenning.cli.lexicon import mine_duplicate_canonicals as cli
    from wyrd.generators.kenning.lexicon.duplicate_canonical_mining import parse_propose

    class _Flaky:
        def __init__(self, seq):
            self.seq, self.calls = list(seq), 0

        def chat_json(self, system, user, schema):
            r = self.seq[self.calls]
            self.calls += 1
            if isinstance(r, Exception):
                raise r
            return r

    # attempt 1 raises, attempt 2 parses → verdict returned
    ok = _Flaky(
        [RuntimeError("boom"), {"same_morpheme": True, "confidence": "high", "reason": "r"}]
    )
    assert cli._call(ok, "s", "u", parse_propose).same is True
    # both attempts unparseable → None (skipped, not crashed)
    assert cli._call(_Flaky([{"bad": 1}, {"bad": 2}]), "s", "u", parse_propose) is None


def test_judge_loop_skip_leaves_no_audit_row(tmp_path, monkeypatch):
    """A None verdict (transient failure) must NOT be written to the judged-log — the
    pair stays retryable. Guards the continue/_write_audit ordering in the loop."""
    from wyrd.generators.kenning.cli.lexicon import mine_duplicate_canonicals as cli

    c = DuplicateCandidate(1, "niwe", 2, "ne", "oe", ("new",), ("new",), 1.0)
    monkeypatch.setattr(cli, "_judge_pair", lambda client, ca: None)
    audit = tmp_path / "canonicalization" / "_duplicate_canonical_audit.jsonl"
    audit.parent.mkdir(parents=True)
    with audit.open("a", encoding="utf-8") as fh:
        counts = cli._judge_loop(
            None,
            [c],
            mining_dir=tmp_path,
            source="t",
            min_confidence="high",
            apply=True,
            existing_ids=set(),
            audit_fh=fh,
            base_url="u",
            model="m",
        )
    assert counts["skipped"] == 1
    assert cli._load_judged(tmp_path) == set()  # nothing recorded → pair retried next run


def test_load_judged_tolerates_non_dict_line(tmp_path):
    from wyrd.generators.kenning.cli.lexicon import mine_duplicate_canonicals as cli

    cdir = tmp_path / "canonicalization"
    cdir.mkdir()
    (cdir / "_duplicate_canonical_audit.jsonl").write_text(
        '[1, 2]\n"a string"\ntrue\n'  # valid JSON, but not dicts → must not crash
        '{"_type": "duplicate_canonical_audit", "pair": "1:2"}\n',
        encoding="utf-8",
    )
    assert cli._load_judged(tmp_path) == {"1:2"}


# --- wyrd-szyd: dash compound pre-filter + prompt reject-class guidance -------


def test_detect_dash_compound_excluded_but_foldequal_kept(lex):
    """A dashed form whose fold differs from its partner is a compound↔constituent
    (gōs vs gos-wic) and is pre-filtered out; a dashed form that folds EQUAL to its
    partner is the same etymon under a punctuation/diacritic variant (wulfpytt vs
    wulf-pytt; kaup-maðr vs kaup-madr — note ð folds to d) and is kept."""
    _etymon(lex, "gos", "old-english", ["goose"])
    _etymon(lex, "gos-wic", "old-english", ["goose", "farm"])  # compound (goose-farm)
    _etymon(lex, "gos-tun", "old-english", ["goose", "farm"])  # ANOTHER compound (both dashed)
    _etymon(lex, "wulfpytt", "old-english", ["wolf", "pit"])
    _etymon(lex, "wulf-pytt", "old-english", ["wolf", "pit"])  # same word, dash spelling
    _etymon(lex, "kaup-maðr", "old-norse", ["merchant"])
    _etymon(lex, "kaup-madr", "old-norse", ["merchant"])  # ð/d diacritic variant, both dashed
    lex.commit()
    pairs = {frozenset((c.a_form, c.b_form)) for c in detect_candidates(lex.conn).candidates}
    assert frozenset(("gos", "gos-wic")) not in pairs  # compound↔constituent excluded
    assert frozenset(("gos-wic", "gos-tun")) not in pairs  # both dashed, folds differ → excluded
    assert frozenset(("wulfpytt", "wulf-pytt")) in pairs  # fold-equal dash variant kept
    assert frozenset(("kaup-maðr", "kaup-madr")) in pairs  # ð folds to d → kept (was the #687 miss)


def test_maybe_pair_excludes_unicode_dash_compound():
    """wyrd-szyd sibling (#778 surface-as-identity class): the compound pre-filter
    keyed on the ASCII hyphen only (``"-" in form``), so an interior Unicode dash —
    en-dash / U+2010, which mining yields from PDF/Wiktionary and
    ``normalize_morpheme_surface`` PRESERVES (boundary-only trim) — slipped past it.
    ``gos–wic`` (en-dash) vs its constituent ``gos`` folds differently
    (goswic != gos) and MUST be excluded exactly like the ASCII ``gos-wic``; a
    fold-EQUAL Unicode-dash variant (same etymon, different dash spelling) stays kept.
    """
    from wyrd.generators.kenning.lexicon.duplicate_canonical_mining import _maybe_pair
    from wyrd.generators.kenning.lexicon.genitive_priors import fold_surface

    def _row(form: str, tokens: set[str]) -> dict:
        return {
            "cm_id": None,
            "form": form,
            "fold": fold_surface(form),
            "lang": "old-english",
            "glosses": set(tokens),
            "tokens": set(tokens),
        }

    constituent = _row("gos", {"goose", "farm"})  # identical tokens → Jaccard 1.0
    # en-dash (U+2013) and U+2010 compounds vs the constituent: fold differs → excluded
    assert _maybe_pair(_row("gos–wic", {"goose", "farm"}), constituent, 1, 2, 0.5) is None
    assert _maybe_pair(_row("gos‐wic", {"goose", "farm"}), constituent, 1, 2, 0.5) is None
    # ASCII control is unchanged (already excluded pre-fix)
    assert _maybe_pair(_row("gos-wic", {"goose", "farm"}), constituent, 1, 2, 0.5) is None
    # a fold-EQUAL Unicode-dash variant is still KEPT for the LLM to judge
    assert (
        _maybe_pair(
            _row("wulf–pytt", {"wolf", "pit"}),
            _row("wulfpytt", {"wolf", "pit"}),
            1,
            2,
            0.5,
        )
        is not None
    )


def test_prompts_reject_compound_derivation_name_classes():
    """Both prompts must steer the judge away from the three false-positive classes
    the u6fn.5 apply-run surfaced, while still keeping genuine variants/inflections."""
    ps, _ = build_propose_prompt(_cand())
    rs, _ = build_refute_prompt(_cand())
    for s in (ps.lower(), rs.lower()):
        assert "compound" in s  # constituent↔compound
        assert "deriv" in s  # derivation (noun↔verb etc.)
        assert "hypocorist" in s or "nickname" in s or "pet-form" in s  # distinct names
    # and BOTH prompts still affirm the genuine-keep classes (inflection + name variant)
    for s in (ps.lower(), rs.lower()):
        assert "inflection" in s and "katharine" in s
