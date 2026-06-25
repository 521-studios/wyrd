"""wyrd-rogd.15 — cross-era reflex-link candidate detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.reflex_link_detect import detect_reflex_link_candidates


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    p = tmp_path / "lexicon.db"
    init_schema(p)
    return p


def _e(db: LexiconDB, form: str, lang: str, *glosses: str) -> int:
    eid = db.upsert_etymon(form, lang, modifier_type="Habitative")
    for g in glosses:
        db.add_gloss(eid, g)
    return eid


def test_cross_era_reflex_is_a_candidate(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        tun = _e(db, "tūn", "old-english", "farmstead")
        # Modern morpheme stored BARE (wyrd-aicu.8, D45) — the position is the
        # separate axis, not a dash on the identity.
        ton = _e(db, "ton", "modern-english", "farmstead")
        db.commit()
        cands = detect_reflex_link_candidates(db.conn)
        assert len(cands) == 1
        c = cands[0]
        assert c.parent_ref == "old-english:tūn"  # earlier era = parent
        assert c.child_ref == "modern-english:ton"  # later era = child (reflex)
        assert c.shared_glosses == ("farmstead",)
        _ = (tun, ton)


def test_form_dissimilar_same_gloss_is_not_a_candidate(fresh_db: Path) -> None:
    # holt (wood) and hill share a gloss but are not form-similar → no link.
    with LexiconDB(fresh_db) as db:
        _e(db, "holt", "old-english", "Grove")
        _e(db, "hill", "modern-english", "Grove")
        db.commit()
        assert detect_reflex_link_candidates(db.conn) == []


def test_same_language_variant_is_not_emitted_here(fresh_db: Path) -> None:
    # OE hill vs OE hyll: same-language variant = a FOLD (existing detector),
    # deliberately NOT a reflex-LINK.
    with LexiconDB(fresh_db) as db:
        _e(db, "hill", "old-english", "hill")
        _e(db, "hyll", "old-english", "hill")
        db.commit()
        assert detect_reflex_link_candidates(db.conn) == []


def test_already_edged_pair_is_excluded(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        hyll = _e(db, "hyll", "old-english", "hill")
        hill = _e(db, "hill", "modern-english", "hill")
        db.upsert_source(id="s", title="s")
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 's')",
            (hyll, hill),
        )
        db.commit()
        assert detect_reflex_link_candidates(db.conn) == []


def test_similarity_threshold_filters(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        _e(db, "hyll", "old-english", "hill")
        _e(db, "hill", "modern-english", "hill")
        db.commit()
        assert len(detect_reflex_link_candidates(db.conn, min_similarity=0.6)) == 1
        # bump the bar past hyll~hill's ratio → drops out
        assert detect_reflex_link_candidates(db.conn, min_similarity=0.99) == []


def test_best_per_child_tiebreak_is_deterministic() -> None:
    # Regression: two parents tying on (parent_rank, similarity) for one child must
    # resolve input-order-independently, else the proposed parent flips across a DB
    # rebuild / VACUUM (unordered SQL scan), breaking the LLM ledger's byte-stable-
    # rebuild contract (D36.9). The pick is keyed on the unique parent_ref.
    from wyrd.generators.kenning.lexicon.reflex_link_detect import _best_per_child

    child = "modern-english:grave"
    # tuple: (child_ref, parent_ref, child_form, parent_form, shared, sim, parent_rank)
    p1 = (child, "old-english:gratf", "grave", "gratf", ("x",), 0.6, 0)
    p2 = (child, "old-english:greot", "grave", "greot", ("x",), 0.6, 0)
    forward = _best_per_child([p1, p2])[child][1]
    reverse = _best_per_child([p2, p1])[child][1]
    assert forward == reverse  # was gratf vs greot (first-seen) before the parent_ref tiebreak


def _cand():
    from wyrd.generators.kenning.lexicon.reflex_link_detect import ReflexLinkCandidate

    return ReflexLinkCandidate(
        child_ref="modern-english:-ton",
        parent_ref="old-english:tūn",
        child_form="-ton",
        parent_form="tūn",
        shared_glosses=("farmstead",),
        similarity=0.67,
    )


def test_build_reflex_prompt_frames_earlier_and_later() -> None:
    from wyrd.generators.kenning.lexicon.reflex_link_detect import build_reflex_judge_prompt

    system, user = build_reflex_judge_prompt(_cand())
    assert "REFLEX" in system
    assert 'Earlier form "tūn"' in user and 'Later form   "-ton"' in user
    assert "farmstead" in user


def test_parse_reflex_verdict() -> None:
    from wyrd.generators.kenning.lexicon.reflex_link_detect import parse_reflex_verdict

    v = parse_reflex_verdict({"is_reflex": True, "confidence": "high", "reason": "tūn>ton"})
    assert v.is_reflex and v.confidence == "high"
    # string yes/no + bad confidence coerced
    assert parse_reflex_verdict({"is_reflex": "yes", "confidence": "??"}).confidence == "low"
    assert parse_reflex_verdict({"is_reflex": "no"}).is_reflex is False
    assert parse_reflex_verdict({"confidence": "high"}) is None  # missing verdict
    assert parse_reflex_verdict("nope") is None


def test_reflex_verdict_to_row_links_or_rejects() -> None:
    from wyrd.generators.kenning.lexicon.reflex_link_detect import (
        LLM_REFLEX_LINK_METHOD,
        LLM_REFLEX_REJECT_METHOD,
        ReflexVerdict,
        reflex_verdict_to_row,
    )

    c = _cand()
    # confident reflex → an inherits LINK row keyed by the child
    link = reflex_verdict_to_row(c, ReflexVerdict(True, "high", "r"), "medium")
    assert link["ref"] == "modern-english:-ton"
    assert link["inherits"] == "old-english:tūn"
    assert link["method"] == LLM_REFLEX_LINK_METHOD
    # not-a-reflex → no-op rejection (records the judgment, empty inherits)
    rej = reflex_verdict_to_row(c, ReflexVerdict(False, "high", "cognate"), "medium")
    assert rej["inherits"] == "" and rej["method"] == LLM_REFLEX_REJECT_METHOD
    # below min confidence → also a no-op
    low = reflex_verdict_to_row(c, ReflexVerdict(True, "low", "maybe"), "high")
    assert low["inherits"] == ""


def test_cross_family_pair_is_not_a_candidate(fresh_db: Path) -> None:
    # english vs brythonic: different families → never a reflex link (the fix
    # that dropped the cross-family cognate noise).
    with LexiconDB(fresh_db) as db:
        _e(db, "dun", "old-english", "hill")
        _e(db, "din", "old-welsh", "hill")
        db.commit()
        assert detect_reflex_link_candidates(db.conn, min_similarity=0.5) == []


def test_unranked_language_pair_is_not_a_candidate(fresh_db: Path) -> None:
    # "celtic" is a proto/non-stage label absent from the era-rank → excluded.
    with LexiconDB(fresh_db) as db:
        _e(db, "broc", "celtic", "badger")
        _e(db, "broc", "irish", "badger")
        db.commit()
        assert detect_reflex_link_candidates(db.conn, min_similarity=0.5) == []


def test_dedup_keeps_the_most_immediate_ancestor(fresh_db: Path) -> None:
    # modern 'hoyle' descends from both OE 'hol' and ME 'hole' (same gloss,
    # similar form). Only ONE candidate survives — the closest era (ME).
    with LexiconDB(fresh_db) as db:
        _e(db, "hol", "old-english", "hollow")
        _e(db, "hole", "middle-english", "hollow")
        _e(db, "hoyle", "modern-english", "hollow")
        db.commit()
        cands = detect_reflex_link_candidates(db.conn, min_similarity=0.6)
        hoyle = [c for c in cands if c.child_ref == "modern-english:hoyle"]
        assert len(hoyle) == 1
        assert hoyle[0].parent_ref == "middle-english:hole"  # immediate, not OE
        assert hoyle[0].shared_glosses == ("hollow",)  # never empty


def test_cli_link_reflexes_writes_inherits_rows(
    fresh_db: Path, tmp_path: Path, monkeypatch
) -> None:
    # End-to-end CLI loop with a STUB LLM (always "yes, high") — exercises
    # detect → judge → ledger append + the dedup/dry-run-free path.
    import json as _json

    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon.link_reflexes import lexicon_link_reflexes

    with LexiconDB(fresh_db) as db:
        _e(db, "biscop", "old-english", "bishop")
        _e(db, "bishop", "modern-english", "bishop")
        db.commit()

    class _StubClient:
        def __init__(self, *a, **k):
            pass

        def chat_json(self, system, user, schema):
            return {"is_reflex": True, "confidence": "high", "reason": "stub"}

    monkeypatch.setattr("wyrd.generators.kenning.extractors.llm.OllamaClient", _StubClient)
    ledger = tmp_path / "_collapses.jsonl"
    res = CliRunner().invoke(
        lexicon_link_reflexes,
        ["--db", str(fresh_db), "--collapse-file", str(ledger), "--min-similarity", "0.6"],
    )
    assert res.exit_code == 0, res.output
    rows = [
        _json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    link = next(r for r in rows if r.get("inherits"))
    assert link["ref"] == "modern-english:bishop" and link["inherits"] == "old-english:biscop"


def test_pair_sharing_two_glosses_emits_one_candidate(fresh_db: Path) -> None:
    """A cross-era pair that shares TWO glosses surfaces in both gloss groups,
    but the ``seen`` guard + best-per-child dedup yield exactly ONE candidate,
    whose shared_glosses lists both (sorted)."""
    with LexiconDB(fresh_db) as db:
        _e(db, "tūn", "old-english", "farmstead", "enclosure")
        _e(db, "-ton", "modern-english", "farmstead", "enclosure")
        db.commit()
        cands = detect_reflex_link_candidates(db.conn)
        assert len(cands) == 1
        assert cands[0].shared_glosses == ("enclosure", "farmstead")


def test_max_per_gloss_skips_oversized_groups(fresh_db: Path) -> None:
    """A gloss group larger than ``max_per_gloss`` is skipped wholesale (the
    O(n^2) pairing never runs inside it)."""
    with LexiconDB(fresh_db) as db:
        _e(db, "tūn", "old-english", "farmstead")
        _e(db, "toun", "middle-english", "farmstead")
        _e(db, "-ton", "modern-english", "farmstead")
        db.commit()
        # group size 3 > cap 2 → skipped entirely
        assert detect_reflex_link_candidates(db.conn, max_per_gloss=2) == []
        # within cap → the chain still yields a candidate
        assert len(detect_reflex_link_candidates(db.conn, max_per_gloss=3)) == 1
