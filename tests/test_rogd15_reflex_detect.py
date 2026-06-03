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
        ton = _e(db, "-ton", "modern-english", "farmstead")
        db.commit()
        cands = detect_reflex_link_candidates(db.conn)
        assert len(cands) == 1
        c = cands[0]
        assert c.parent_ref == "old-english:tūn"  # earlier era = parent
        assert c.child_ref == "modern-english:-ton"  # later era = child (reflex)
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


def _cand() -> "ReflexLinkCandidate":
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
