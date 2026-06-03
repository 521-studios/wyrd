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
