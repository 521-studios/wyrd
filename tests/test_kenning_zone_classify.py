"""Tests for the cultural/linguistic-zone classifier spike (wyrd-hytz, D52).

Fixture-free: reads only the committed zones.yaml. Covers the morpheme→zone and
place→zone rule-based classification + the model's well-formedness.
"""

from __future__ import annotations

from importlib import resources

import pytest
import yaml

from wyrd.generators.kenning.lexicon.zone_classify import (
    _build_model,
    morpheme_zone,
    place_zones,
    zone_names,
)


def _raw():
    text = (
        resources.files("wyrd.generators.kenning.data")
        .joinpath("zones.yaml")
        .read_text(encoding="utf-8")
    )
    return yaml.safe_load(text)


def test_zone_vocabulary():
    assert set(zone_names()) == {"danelaw", "celtic", "roman"}


def test_morpheme_zone_by_language():
    assert morpheme_zone("old-norse") == "danelaw"
    assert morpheme_zone("celtic") == "celtic"
    assert morpheme_zone("latin") == "roman"
    assert morpheme_zone("medieval-latin") == "roman"


def test_morpheme_zone_celtic_branches():
    # both branch-level Celtic languages map to the celtic zone (wyrd-xdam.1)
    assert morpheme_zone("brittonic") == "celtic"
    assert morpheme_zone("goidelic") == "celtic"


def test_celtic_languages_classify_to_celtic_zone():
    """wyrd-xdam.1 sync tripwire: EVERY brythonic/goidelic language in the era
    source-of-truth (cells.LANGUAGE_TO_FAMILY) must classify to the celtic zone —
    otherwise morpheme_zone silently drops it to the anglo-saxon default. Pins the
    zones.yaml morpheme_languages list against LANGUAGE_TO_FAMILY so the two can't
    drift (this would have caught the missing manx / old-breton / middle-breton)."""
    from wyrd.generators.kenning.era.cells import LANGUAGE_TO_FAMILY

    celtic_langs = {
        lang
        for lang, family in LANGUAGE_TO_FAMILY.items()
        if family in ("brythonic", "goidelic") and lang != "celtic_mix"
    }
    assert celtic_langs, "expected some brythonic/goidelic languages"
    for lang in celtic_langs:
        assert morpheme_zone(lang) == "celtic", lang


def test_morpheme_zone_defaults_to_anglo_saxon():
    # native + unclaimed languages fall to the un-zoned default
    for lang in ("old-english", "middle-english", "modern-english", "unknown", None):
        assert morpheme_zone(lang) == "anglo-saxon"


def test_place_zones_by_region_and_country():
    assert "danelaw" in place_zones(region="Yorkshire", country="England")
    assert "danelaw" in place_zones(region="Lincolnshire", country="England")
    assert "celtic" in place_zones(country="Wales")
    assert "celtic" in place_zones(region="Cornwall", country="England")


def test_place_zones_unzoned_is_empty():
    # a non-Danelaw English shire is in no zone (the un-zoned majority)
    assert place_zones(region="Devon", country="England") == frozenset()
    assert place_zones() == frozenset()


def test_place_zones_multi_membership():
    """The region and country legs union — a place can fall in more than one
    zone (the axis cuts across admin boundaries)."""
    result = place_zones(region="Yorkshire", country="Wales")
    assert result == {"danelaw", "celtic"}


# --- _build_model fail-loud guards (test seam) -------------------------------


def test_build_model_rejects_non_mapping():
    with pytest.raises(ValueError, match="did not parse as a mapping"):
        _build_model([1, 2, 3])


def test_build_model_requires_default_zone():
    with pytest.raises(ValueError, match="missing default_zone"):
        _build_model({"zones": [{"name": "danelaw"}]})


def test_build_model_rejects_duplicate_zone_name():
    with pytest.raises(ValueError, match="duplicate zone names"):
        _build_model({"default_zone": "anglo-saxon", "zones": [{"name": "x"}, {"name": "x"}]})


def test_build_model_rejects_default_zone_as_named_zone():
    """The un-zoned default must not also be an explicit zone."""
    with pytest.raises(ValueError, match="must not also be a named zone"):
        _build_model({"default_zone": "anglo-saxon", "zones": [{"name": "anglo-saxon"}]})


def test_build_model_rejects_malformed_zone_entry():
    """A zone entry missing 'name', or one that isn't a mapping at all, fails with
    a clear error rather than a cryptic KeyError/TypeError/AttributeError."""
    with pytest.raises(ValueError, match="malformed zone entry"):
        _build_model({"default_zone": "anglo-saxon", "zones": [{"morpheme_languages": ["x"]}]})
    with pytest.raises(ValueError, match="malformed zone entry"):
        _build_model({"default_zone": "anglo-saxon", "zones": ["not-a-mapping"]})


def test_build_model_rejects_language_claimed_by_two_zones():
    bad = {
        "default_zone": "anglo-saxon",
        "zones": [
            {"name": "a", "morpheme_languages": ["x"]},
            {"name": "b", "morpheme_languages": ["x"]},
        ],
    }
    with pytest.raises(ValueError, match="claimed by two zones"):
        _build_model(bad)


def test_model_is_wellformed():
    """No language claimed by two zones; no duplicate zone names; default present."""
    m = _raw()
    assert m["default_zone"] == "anglo-saxon"
    names = [z["name"] for z in m["zones"]]
    assert len(set(names)) == len(names)
    seen_langs: set[str] = set()
    for z in m["zones"]:
        for lang in z.get("morpheme_languages") or []:
            assert lang not in seen_langs, f"{lang} claimed by two zones"
            seen_langs.add(lang)


def test_default_zone_is_not_a_named_zone():
    """The default bucket is distinct from the explicit zones (a morpheme is
    'anglo-saxon' by absence, never by being listed)."""
    assert "anglo-saxon" not in zone_names()
