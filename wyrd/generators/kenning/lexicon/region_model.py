"""Class-A region model loader + fail-closed ingest canonicalizer (wyrd-3q6m).

Loads ``regions_england.yaml`` (the canonical node vocabulary + alias map +
deferred zones + quarantine — wyrd-3q6m.1/.2) and exposes
:func:`canonicalize_region`, the ingest-boundary guard (wyrd-3q6m.3, D49):

- an alias variant (``SUF``, ``County Durham``, ``East Sussex``) is folded to its
  canonical node;
- a canonical region node (a county or subdivision) passes unchanged;
- any *other* England-scoped value — the country root, a deferred zone, a
  quarantined coding, or an unknown string — is a HARD error
  (:class:`RegionValidationError`), never a silent new row. This is the guard
  that stops the controlled-vocabulary mess re-accumulating after the one-time
  repair (wyrd-3q6m.4).

Scope: **England only.** A region that resolves to another country (Scotland,
Wales, …) or is unknown with no England signal passes through unchanged — those
dimensions get their own models in wyrd-3q6m.5. England-scoping uses the
existing region→country map (:func:`country_for_region`) plus recognition by the
England model itself, so a known England zone/quarantine value is caught even
when the caller didn't pass ``country``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType

import yaml

from wyrd.generators.kenning.lexicon.regions import country_for_region

_DATA_PACKAGE = "wyrd.generators.kenning.data"
_FILENAME = "regions_england.yaml"
_REGION_LEVELS = ("county", "subdivision")


class RegionValidationError(ValueError):
    """Raised at the ingest boundary when an England-scoped region value is
    neither a canonical node nor an alias (fail-closed; wyrd-3q6m.3 / D49)."""

    def __init__(self, region: str, reason: str):
        self.region = region
        self.reason = reason
        super().__init__(f"invalid England region {region!r}: {reason}")


@dataclass(frozen=True)
class _EnglandModel:
    valid_nodes: frozenset[str]  # county + subdivision canonicals — valid region values
    country_root: str  # "England"
    aliases: MappingProxyType  # variant -> canonical node
    zones: frozenset[str]  # deferred cultural/linguistic zones (wyrd-hytz)
    quarantine: frozenset[str]  # known-bad codings requiring re-code

    def recognizes(self, region: str) -> bool:
        """Whether the England model knows ``region`` in any bucket — used to
        England-scope a value even when the caller passed no country."""
        return (
            region in self.valid_nodes
            or region == self.country_root
            or region in self.aliases
            or region in self.zones
            or region in self.quarantine
        )


@lru_cache(maxsize=1)
def _model() -> _EnglandModel:
    """Parse the bundled England region model once per process. Read-only
    (immutable dataclass + ``MappingProxyType``): this is shared, long-lived
    cached static config, never request-derived state."""
    text = resources.files(_DATA_PACKAGE).joinpath(_FILENAME).read_text(encoding="utf-8")
    m = yaml.safe_load(text)
    nodes = m["nodes"]
    valid = frozenset(n["canonical"] for n in nodes if n["level"] in _REGION_LEVELS)
    root = next(n["canonical"] for n in nodes if n["level"] == "country")
    aliases = MappingProxyType({a["from"]: a["to"] for a in m["aliases"]})
    zones = frozenset(z["value"] for z in m["deferred_zones"])
    quarantine = frozenset(q["value"] for q in m["quarantine"])
    return _EnglandModel(valid, root, aliases, zones, quarantine)


def canonicalize_region(region: str | None, *, country: str | None = None) -> str | None:
    """Validate + canonicalize a region value at the ingest boundary.

    Returns the canonical region (an alias variant folded to its node; a
    canonical node unchanged; a non-England / unknown-unscoped value unchanged),
    or raises :class:`RegionValidationError` for an England-scoped value that is
    not a valid region node.

    ``country`` (the caller's explicit country, if any) helps decide England
    scope; when omitted it is derived from ``region`` via
    :func:`country_for_region`.
    """
    if region is None:
        return None
    model = _model()
    effective_country = country or country_for_region(region)
    england_scoped = effective_country == "England" or model.recognizes(region)
    if not england_scoped:
        # Non-England dimension (Scotland/Wales/…) or an unknown with no England
        # signal — out of scope here; its model is wyrd-3q6m.5.
        return region
    if region in model.aliases:
        return model.aliases[region]
    if region in model.valid_nodes:
        return region
    # England-scoped but not a valid region node: classify the failure.
    if region == model.country_root:
        reason = "the country root is not a region; use a county or leave region empty"
    elif region in model.zones:
        reason = "a cultural/linguistic zone, not an admin region (migrate to wyrd-hytz)"
    elif region in model.quarantine:
        reason = "a quarantined coding; re-code to a single historic county"
    else:
        reason = "unknown England region; add it to regions_england.yaml as a node or alias"
    raise RegionValidationError(region, reason)
