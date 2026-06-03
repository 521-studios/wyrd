"""wyrd-rogd.11 — the era-grid modern stage must carry the morpheme's own
common-noun reflex, not the cluster tier's derived proper nouns (toponyms,
anthroponyms) or mistagged foreign cognates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.bundle._family import (
    _fetch_root_era_reflexes,
    _is_derived_name_pollution,
)


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def test_pollution_predicate_flags_proper_nouns_and_multiword() -> None:
    for bad in (
        "West Ham",
        "Notting Hill",
        "Wesley",
        "Derek",
        "Hank",
        "Coldingham",
        "Düne",
        "Ouest",
    ):
        assert _is_derived_name_pollution(bad), bad
    for good in ("ton", "hill", "-bury", "ceaster", "wīc", "thorpe"):
        assert not _is_derived_name_pollution(good), good


def _mate(db, root_id: int, form: str, lang: str) -> int:
    eid = db.upsert_etymon(form, lang)
    db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root_id, eid))
    return eid


def test_modern_stage_drops_derived_names_keeps_reflex(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        root = db.upsert_etymon("tūn", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root, root))
        _mate(db, root, "ton", "modern-english")  # genuine reflex
        _mate(db, root, "West Ham", "modern-english")  # multi-word toponym
        _mate(db, root, "Wesley", "modern-english")  # anthroponym
        _mate(db, root, "Düne", "modern-english")  # capitalized foreign cognate
        db.commit()
        out = _fetch_root_era_reflexes(db, root, "old-english")
        assert [e["form"] for e in out.get("modern-english", [])] == ["ton"]


def test_historical_stage_is_not_filtered(fresh_db: Path) -> None:
    # The filter is modern-only; a capitalized OE cluster mate (an attested
    # period spelling) must survive in the old-english stage.
    with LexiconDB(fresh_db) as db:
        root = db.upsert_etymon("tūn", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root, root))
        _mate(db, root, "Tuun", "old-english")  # capitalized, but historical → kept
        db.commit()
        out = _fetch_root_era_reflexes(db, root, "old-english")
        oe = {e["form"] for e in out.get("old-english", [])}
        assert "Tuun" in oe and "tūn" in oe


def test_all_modern_pollution_empties_the_stage(fresh_db: Path) -> None:
    # When nothing but derived names survive, the modern stage is omitted
    # entirely (no empty bucket manufactured).
    with LexiconDB(fresh_db) as db:
        root = db.upsert_etymon("hyll", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root, root))
        _mate(db, root, "Notting Hill", "modern-english")
        _mate(db, root, "Hillary", "modern-english")
        db.commit()
        out = _fetch_root_era_reflexes(db, root, "old-english")
        assert "modern-english" not in out


def test_pollution_predicate_edge_cases() -> None:
    # empty / whitespace-only → pollution (err toward empty)
    assert _is_derived_name_pollution("")
    assert _is_derived_name_pollution("   ")
    # non-space whitespace (tab, non-breaking space) is still multi-word
    assert _is_derived_name_pollution("West\tHam")
    assert _is_derived_name_pollution("West Ham")
    # leading punctuation before an uppercase letter is still a proper noun
    assert _is_derived_name_pollution(".Wesley")
    assert _is_derived_name_pollution("(Hank)")
    # a leading-dash affix whose first alpha char is lowercase is kept
    assert not _is_derived_name_pollution("-ton")


def test_filtered_forms_do_not_reach_forms_by_lang(fresh_db: Path) -> None:
    # The seam guarantee: a dropped modern proper noun must not survive into
    # forms_by_lang (what the generator samples), not just the era-grid view.
    from wyrd.generators.kenning.lexicon.bundle._family import (
        _merge_era_reflexes_into_forms_by_lang,
    )

    with LexiconDB(fresh_db) as db:
        root = db.upsert_etymon("tūn", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root, root))
        _mate(db, root, "ton", "modern-english")
        _mate(db, root, "West Ham", "modern-english")
        db.commit()
        era_reflexes = _fetch_root_era_reflexes(db, root, "old-english")
        forms_by_lang: dict[str, list[str]] = {}
        _merge_era_reflexes_into_forms_by_lang(forms_by_lang, era_reflexes)
        assert forms_by_lang.get("modern-english") == ["ton"]


def test_fantasy_export_also_filters_modern_pollution(fresh_db: Path) -> None:
    from wyrd.generators.kenning.lexicon.fantasy_export import collect_fantasy_morphemes

    with LexiconDB(fresh_db) as db:
        root = db.upsert_etymon("tūn", "old-english")
        db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (root, root))
        _mate(db, root, "ton", "modern-english")
        _mate(db, root, "Pemberton", "modern-english")  # derived toponym
        db.conn.execute(
            "INSERT INTO fantasy_morpheme "
            "(input_name, usable, etymon_id, resolution_method, approach_version, processed_at) "
            "VALUES ('Tunhold', 1, ?, 'test', 'v1', '2026-01-01')",
            (root,),
        )
        db.commit()
        out = collect_fantasy_morphemes(db)
        modern = [e["form"] for e in out["Tunhold"]["era_reflexes"].get("modern-english", [])]
        assert modern == ["ton"]
