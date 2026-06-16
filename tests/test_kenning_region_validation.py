"""Tests for the fail-closed region canonicalizer (wyrd-3q6m.3, D49).

Covers the pure :func:`canonicalize_region` contract and its integration at the
ingest boundary (``_upsert_toponym``). Fixture-free for the pure function (it
reads only the committed ``regions_england.yaml``); the ingest tests use a
tmp-path DB.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.ingest import _upsert_toponym
from wyrd.generators.kenning.lexicon.region_model import (
    RegionValidationError,
    canonicalize_region,
)

# --- pure canonicalizer ------------------------------------------------------


def test_none_passes_through():
    assert canonicalize_region(None) is None


def test_canonical_nodes_pass_unchanged():
    for region in ("Suffolk", "Yorkshire", "West Riding", "Durham", "Isle of Wight"):
        assert canonicalize_region(region) == region


def test_modern_node_is_a_valid_region_value():
    """A modern-administrative node (Cumbria) is a known, valid region value —
    it passes unchanged (resolving it to a historic county is lossy Class-B
    succession, out of scope), unlike the non-lossy E/W Sussex normalize-up."""
    assert canonicalize_region("Cumbria") == "Cumbria"


def test_aliases_fold_to_canonical():
    cases = {
        "SUF": "Suffolk",
        "LEC": "Leicestershire",
        "BUK": "Buckinghamshire",
        "SUR": "Surrey",
        "SUS": "Sussex",
        "County Durham": "Durham",
        "Isle Of Wight": "Isle of Wight",
        "West Riding of Yorkshire": "West Riding",
        "East Riding of Yorkshire": "East Riding",
        "East Sussex": "Sussex",
        "West Sussex": "Sussex",
    }
    for variant, canonical in cases.items():
        assert canonicalize_region(variant) == canonical, variant


def test_country_root_is_rejected():
    with pytest.raises(RegionValidationError):
        canonicalize_region("England")


def test_deferred_zone_is_rejected():
    with pytest.raises(RegionValidationError) as exc:
        canonicalize_region("England (Danelaw)")
    assert "zone" in str(exc.value)


def test_quarantined_codings_are_rejected():
    for region in (
        "Northumberland and Durham",
        "Cumberland and Westmorland",
        "South-West Yorkshire",
    ):
        with pytest.raises(RegionValidationError):
            canonicalize_region(region)


def test_unknown_england_region_is_rejected_when_country_is_england():
    """An England-scoped (country=England) value that is neither node nor alias
    is the load-bearing fail-closed case — it must not silently become a new
    bucket."""
    with pytest.raises(RegionValidationError) as exc:
        canonicalize_region("Borsetshire", country="England")
    assert "unknown" in str(exc.value)


def test_non_england_regions_pass_through():
    """Out of scope here (their models are wyrd-3q6m.5): known non-England
    regions and country-overridden values flow unchanged."""
    assert canonicalize_region("Argyll") == "Argyll"  # Scotland (regions.py)
    assert canonicalize_region("Brittany") == "Brittany"  # France
    assert canonicalize_region("British Isles", country="Wales") == "British Isles"


def test_explicit_country_is_authoritative_for_scope():
    """An explicit non-England country wins over England-recognition: a caller
    asserting country='Scotland' gets the value passed through unchanged, even
    for an England-looking code/quarantine string (the contradiction is the
    caller's country assertion to own, not this region guard's)."""
    assert canonicalize_region("SUF", country="Scotland") == "SUF"
    assert canonicalize_region("Northumberland and Durham", country="Scotland") == (
        "Northumberland and Durham"
    )
    # ...but an explicit England country still folds/validates as normal:
    assert canonicalize_region("SUF", country="England") == "Suffolk"


def test_model_recognition_alone_scopes_rejection(monkeypatch):
    """Forward-protection branch: even if a quarantine value were NOT in the
    region→country map (today they coincidentally all are), the model's own
    recognition still England-scopes it so it fails closed. Isolated by stubbing
    country_for_region to None so recognition is the sole scoping signal."""
    monkeypatch.setattr(
        "wyrd.generators.kenning.lexicon.region_model.country_for_region",
        lambda _region: None,
    )
    with pytest.raises(RegionValidationError):
        canonicalize_region("Northumberland and Durham")  # no country; map stubbed to None


def test_unknown_with_no_england_signal_passes_through():
    """A genuinely unknown value with no England signal (no country, unmapped)
    is not England-scoped, so it is deferred rather than rejected — the guard
    fires only once a value is known to be England's."""
    assert canonicalize_region("Atlantis") == "Atlantis"


# --- ingest-boundary integration --------------------------------------------


def test_upsert_canonicalizes_coded_region(tmp_path: Path):
    """A coded region at ingest is folded to canonical AND its country derives
    from the canonical form (SUF -> Suffolk -> England)."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        tid = _upsert_toponym(db, "Ipswich", region="SUF")
        db.commit()
        row = db.conn.execute("SELECT country, region FROM toponym WHERE id = ?", (tid,)).fetchone()
    assert row["region"] == "Suffolk"
    assert row["country"] == "England"


def test_upsert_rejects_quarantined_region(tmp_path: Path):
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db, pytest.raises(RegionValidationError):
        _upsert_toponym(db, "Somewhere", region="Northumberland and Durham")


def test_upsert_rejects_zone_region(tmp_path: Path):
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db, pytest.raises(RegionValidationError):
        _upsert_toponym(db, "Somewhere", region="England (Danelaw)")


def test_upsert_riding_long_form_keeps_country(tmp_path: Path):
    """Regression guard: a Riding long-form folds to the short canonical AND
    still derives country=England (the short forms are in the region→country map,
    so canonicalize-before-derivation doesn't strand country=NULL)."""
    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        tid = _upsert_toponym(db, "Wakefield", region="West Riding of Yorkshire")
        db.commit()
        row = db.conn.execute("SELECT country, region FROM toponym WHERE id = ?", (tid,)).fetchone()
    assert row["region"] == "West Riding"
    assert row["country"] == "England"


def test_ingest_parsed_entries_propagates_validation_error(tmp_path: Path):
    """The guard fires through the public ingest entrypoint too (not just the
    private upsert), and a bad region aborts rather than being swallowed."""
    from wyrd.generators.kenning.lexicon.ingest import ingest_parsed_entries
    from wyrd.generators.kenning.parsers.skeat import ParsedEntry

    db_path = tmp_path / "lex.db"
    init_schema(db_path)
    entries = [ParsedEntry(toponym="Placeholder", section_suffix=None, historical_form=None)]
    with LexiconDB(db_path) as db, pytest.raises(RegionValidationError):
        ingest_parsed_entries(db, entries, source_id="test", region="Northumberland and Durham")
