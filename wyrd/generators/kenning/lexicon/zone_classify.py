"""Cultural/linguistic-zone classifier — rule-based spike (wyrd-hytz, D52).

Derives zone membership from EXISTING signals (no mining): a morpheme's zone from
its etymon ``language``, a place's zone(s) from its ``region`` / ``country``. This
is the cheap, deterministic first cut that the D52 design recommends over an
LLM/enrichment classifier — the existing language/region signals map to zones
cleanly enough (see D52 for the coverage measurements).

This is the data-model + classifier ONLY. The generator ``--zone`` knob, runtime
bundle, and SPA are deferred (D52). Loaded like the other package-data YAMLs
(``register_effects.yaml`` / ``regions_england.yaml``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType

import yaml

_DATA_PACKAGE = "wyrd.generators.kenning.data"
_FILENAME = "zones.yaml"


@dataclass(frozen=True)
class _ZoneModel:
    default_zone: str
    names: tuple[str, ...]
    # reverse indexes: language -> zone, region -> {zones}, country -> {zones}
    by_language: MappingProxyType[str, str]
    by_region: MappingProxyType[str, frozenset[str]]
    by_country: MappingProxyType[str, frozenset[str]]


@lru_cache(maxsize=1)
def _model() -> _ZoneModel:
    """Parse + index zones.yaml once per process (immutable; static config)."""
    text = resources.files(_DATA_PACKAGE).joinpath(_FILENAME).read_text(encoding="utf-8")
    return _build_model(yaml.safe_load(text))


def _build_model(m: object) -> _ZoneModel:
    """Validate + index a parsed zones mapping. Split out from :func:`_model` as
    a pure (uncached, no-I/O) function so the fail-loud guards are testable
    without a malformed YAML fixture."""
    if not isinstance(m, dict):
        raise ValueError(f"{_FILENAME} did not parse as a mapping")
    default_zone = m.get("default_zone")
    if not default_zone:
        raise ValueError(f"{_FILENAME} is missing default_zone")
    zones = m.get("zones") or []
    try:
        names = tuple(z["name"] for z in zones)
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"{_FILENAME} has a malformed zone entry: {e}") from e
    if len(set(names)) != len(names):
        raise ValueError(f"{_FILENAME} has duplicate zone names: {names}")
    if default_zone in names:
        # the default is "un-zoned by absence" — it must not also be a named zone,
        # or morpheme_zone's fallback would collide with an explicit one.
        raise ValueError(f"{_FILENAME} default_zone {default_zone!r} must not also be a named zone")
    by_language, by_region, by_country = _index_zones(zones)
    return _ZoneModel(
        default_zone=default_zone,
        names=names,
        by_language=MappingProxyType(by_language),
        by_region=MappingProxyType({k: frozenset(v) for k, v in by_region.items()}),
        by_country=MappingProxyType({k: frozenset(v) for k, v in by_country.items()}),
    )


def _index_zones(zones: list) -> tuple[dict[str, str], dict[str, set[str]], dict[str, set[str]]]:
    """Build the language/region/country reverse indexes; raise on a language
    claimed by two zones or a malformed (missing-key) entry."""
    by_language: dict[str, str] = {}
    by_region: dict[str, set[str]] = {}
    by_country: dict[str, set[str]] = {}
    try:
        for z in zones:
            for lang in z.get("morpheme_languages") or []:
                if lang in by_language:
                    raise ValueError(f"{_FILENAME}: language {lang!r} claimed by two zones")
                by_language[lang] = z["name"]
            for region in z.get("place_regions") or []:
                by_region.setdefault(region, set()).add(z["name"])
            for country in z.get("place_countries") or []:
                by_country.setdefault(country, set()).add(z["name"])
    except (KeyError, TypeError, AttributeError) as e:
        # an entry is missing an expected key or isn't a mapping — clear error
        # instead of a cryptic KeyError (zones.yaml is operator-edited).
        raise ValueError(f"{_FILENAME} has a malformed zone entry: {e}") from e
    return by_language, by_region, by_country


def zone_names() -> tuple[str, ...]:
    """The defined zone names (excludes the default/un-zoned bucket)."""
    return _model().names


def morpheme_zone(language: str | None) -> str:
    """The zone a morpheme belongs to, from its etymon ``language``. Languages
    claimed by no zone (Old/Middle/Modern English, ``unknown``, …) fall to the
    default ``anglo-saxon`` bucket. This is the generation-relevant signal — a
    zone's flavor is the morphemes whose language maps to it."""
    model = _model()
    if not language:
        return model.default_zone
    return model.by_language.get(language, model.default_zone)


def place_zones(region: str | None = None, country: str | None = None) -> frozenset[str]:
    """The zone(s) a place falls in, from its ``region`` / ``country`` — possibly
    several (the axis cuts across admin boundaries), possibly none. Secondary to
    :func:`morpheme_zone`; for analysis / empirical priors, not generation."""
    model = _model()
    zones: set[str] = set()
    if region and region in model.by_region:
        zones |= model.by_region[region]
    if country and country in model.by_country:
        zones |= model.by_country[country]
    return frozenset(zones)
