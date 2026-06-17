"""Tests for the cultural/linguistic-zone classifier spike (wyrd-hytz, D52).

Fixture-free: reads only the committed zones.yaml. Covers the morpheme→zone and
place→zone rule-based classification + the model's well-formedness.
"""

from __future__ import annotations

from importlib import resources

import yaml

from wyrd.generators.kenning.lexicon.zone_classify import (
    morpheme_zone,
    place_zones,
    zone_names,
)


def _raw():
    text = resources.files("wyrd.generators.kenning.data").joinpath("zones.yaml").read_text()
    return yaml.safe_load(text)


def test_zone_vocabulary():
    assert set(zone_names()) == {"danelaw", "celtic", "roman"}


def test_morpheme_zone_by_language():
    assert morpheme_zone("old-norse") == "danelaw"
    assert morpheme_zone("celtic") == "celtic"
    assert morpheme_zone("latin") == "roman"
    assert morpheme_zone("medieval-latin") == "roman"


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
