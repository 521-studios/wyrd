"""Tests for the flat controlled-vocabulary guards (wyrd-3q6m.5, D49):
country + curated-etymon language canonicalize/validate, and their ingest hooks.
Fixture-free for the pure guards; tmp-path DB for the ingest integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.controlled_vocab import (
    COUNTRY_ALIASES,
    COUNTRY_CANONICAL,
    CountryValidationError,
    LanguageValidationError,
    canonicalize_country,
    canonicalize_language,
)
from wyrd.generators.kenning.lexicon.ingest import _upsert_toponym, ingest_parsed_entries
from wyrd.generators.kenning.lexicon.regions import _REGION_TO_COUNTRY

# --- country -----------------------------------------------------------------


def test_country_aliases_fold():
    assert canonicalize_country("Republic of Ireland") == "Ireland"
    assert canonicalize_country("The Isle of Man") == "Isle of Man"
    assert canonicalize_country("Brittany") == "France"


def test_country_canonical_passes():
    for c in (
        "England",
        "Scotland",
        "Wales",
        "Ireland",
        "Northern Ireland",
        "Isle of Man",
        "France",
    ):
        assert canonicalize_country(c) == c


def test_country_none_empty_whitespace_to_none():
    assert canonicalize_country(None) is None
    assert canonicalize_country("") is None
    assert canonicalize_country("   ") is None


def test_country_strips_whitespace():
    assert canonicalize_country("  England  ") == "England"
    assert canonicalize_country(" Republic of Ireland ") == "Ireland"


def test_country_unknown_is_rejected():
    with pytest.raises(CountryValidationError):
        canonicalize_country("Atlantis")


def test_country_alias_targets_are_canonical():
    """Every alias must resolve to a canonical country — else a typo'd target
    would bypass the guard (alias resolved before the canonical check)."""
    assert set(COUNTRY_ALIASES.values()) <= COUNTRY_CANONICAL


def test_every_region_derived_country_is_canonical():
    """Consistency: every country regions.py can derive must be in the canonical
    enum, so the region-derived path never produces an unknown country."""
    derived = set(_REGION_TO_COUNTRY.values())
    assert derived <= COUNTRY_CANONICAL, derived - COUNTRY_CANONICAL


# --- language ----------------------------------------------------------------


def test_language_canonical_passes():
    for lang in ("old-english", "celtic", "old-norse", "norman-french", "unknown"):
        assert canonicalize_language(lang) == lang


def test_language_strips_whitespace():
    assert canonicalize_language("  old-english  ") == "old-english"


def test_language_none_or_empty_raises():
    # language is part of the etymon dedup key — never None; empty fails closed.
    for bad in (None, "", "   "):
        with pytest.raises(LanguageValidationError):
            canonicalize_language(bad)


def test_language_unknown_is_rejected():
    # a wiktextract code zoo value, and a not-yet-added place-name language
    for lang in ("gmw-cfr", "af", "welsh"):
        with pytest.raises(LanguageValidationError):
            canonicalize_language(lang)


# --- ingest integration ------------------------------------------------------


def test_upsert_folds_country_alias(tmp_path: Path):
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        tid = _upsert_toponym(db, "Dublin", region=None, country="Republic of Ireland")
        db.commit()
        row = db.conn.execute("SELECT country FROM toponym WHERE id = ?", (tid,)).fetchone()
    assert row["country"] == "Ireland"


def test_upsert_rejects_unknown_country(tmp_path: Path):
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db, pytest.raises(CountryValidationError):
        _upsert_toponym(db, "Nowhere", region=None, country="Freedonia")


def test_ingest_rejects_unknown_etymon_language(tmp_path: Path):
    from wyrd.generators.kenning.parsers.skeat import ParsedElement, ParsedEntry

    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    entry = ParsedEntry(
        toponym="Placeholder",
        section_suffix=None,
        historical_form=None,
        elements=[ParsedElement(form="x", language="gmw-cfr", gloss=None, inflection=None)],
        confidence="high",
    )
    with LexiconDB(db_path) as db, pytest.raises(LanguageValidationError):
        ingest_parsed_entries(db, [entry], source_id="test", region="Suffolk")


def test_upsert_region_derived_country_is_canonical(tmp_path: Path):
    """The region→country_for_region→canonicalize_country path (country=None)
    stores a canonical country end-to-end."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        tid = _upsert_toponym(db, "Ipswich", region="Suffolk")  # no country → derive
        db.commit()
        row = db.conn.execute("SELECT country FROM toponym WHERE id = ?", (tid,)).fetchone()
    assert row["country"] == "England"


def _entry(toponym, elems):
    from wyrd.generators.kenning.parsers.skeat import ParsedElement, ParsedEntry

    return ParsedEntry(
        toponym=toponym,
        section_suffix=None,
        historical_form=None,
        elements=[ParsedElement(form=f, language=lang) for f, lang in elems],
        confidence="high",
    )


def test_ingest_happy_path_multi_element(tmp_path: Path):
    """A valid multi-element entry writes the rows, counts correctly, and maps
    each element to its (ordinal-correct) canonical language."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    entry = _entry("Aberford", [("ford", "old-english"), ("á", "old-norse")])
    with LexiconDB(db_path) as db:
        db.conn.execute("INSERT INTO source (id, title) VALUES ('test', 'T')")
        counts = ingest_parsed_entries(db, [entry], source_id="test", region="Suffolk")
        db.commit()
        by_form = {
            r["canonical_form"]: r["language"]
            for r in db.conn.execute("SELECT canonical_form, language FROM etymon").fetchall()
        }
    assert counts["etymologies"] == 1 and counts["elements"] == 2
    # per-element pin (not just a set): catches a transposed elem_languages[ordinal]
    assert by_form["ford"] == "old-english"
    assert by_form["á"] == "old-norse"


def test_ingest_bad_language_is_atomic(tmp_path: Path):
    """A bad language on the SECOND element rejects the whole entry before any
    write — no toponym, no etymology row persists."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    entry = _entry("Aberford", [("ford", "old-english"), ("x", "gmw-cfr")])
    with LexiconDB(db_path) as db:
        db.conn.execute("INSERT INTO source (id, title) VALUES ('test', 'T')")
        with pytest.raises(LanguageValidationError):
            ingest_parsed_entries(db, [entry], source_id="test", region="Suffolk")
        db.commit()
        assert db.conn.execute("SELECT count(*) c FROM toponym").fetchone()["c"] == 0
        assert db.conn.execute("SELECT count(*) c FROM toponym_etymology").fetchone()["c"] == 0


def test_ingest_low_confidence_writes_toponym_only(tmp_path: Path):
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    from wyrd.generators.kenning.parsers.skeat import ParsedEntry

    entry = ParsedEntry(
        toponym="Placeholder", section_suffix=None, historical_form=None, confidence="low"
    )
    with LexiconDB(db_path) as db:
        counts = ingest_parsed_entries(db, [entry], source_id="test", region="Suffolk")
        db.commit()
    assert counts["toponyms"] == 1 and counts["etymologies"] == 0 and counts["elements"] == 0
