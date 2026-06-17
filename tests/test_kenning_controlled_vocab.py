"""Tests for the flat controlled-vocabulary guards (wyrd-3q6m.5, D49):
country + curated-etymon language canonicalize/validate, and their ingest hooks.
Fixture-free for the pure guards; tmp-path DB for the ingest integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.controlled_vocab import (
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


def test_country_none_and_empty_to_none():
    assert canonicalize_country(None) is None
    assert canonicalize_country("") is None


def test_country_unknown_is_rejected():
    with pytest.raises(CountryValidationError):
        canonicalize_country("Atlantis")


def test_every_region_derived_country_is_canonical():
    """Consistency: every country regions.py can derive must be in the canonical
    enum, so the region-derived path never produces an unknown country."""
    derived = set(_REGION_TO_COUNTRY.values())
    assert derived <= COUNTRY_CANONICAL, derived - COUNTRY_CANONICAL


# --- language ----------------------------------------------------------------


def test_language_canonical_passes():
    for lang in ("old-english", "celtic", "old-norse", "norman-french", "unknown"):
        assert canonicalize_language(lang) == lang


def test_language_none_to_none():
    assert canonicalize_language(None) is None
    assert canonicalize_language("") is None


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
