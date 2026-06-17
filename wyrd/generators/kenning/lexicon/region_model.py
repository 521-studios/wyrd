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
dimensions get their own models in wyrd-3q6m.5. An explicit ``country`` argument
is authoritative for scope; absent one, England-scope falls back to the existing
region→country map (:func:`country_for_region`) plus recognition by the England
model itself, so a known England zone/quarantine value is caught even when the
caller didn't pass ``country`` (see :func:`canonicalize_region`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Literal

import yaml

from wyrd.generators.kenning.lexicon.regions import country_for_region

_DATA_PACKAGE = "wyrd.generators.kenning.data"
_FILENAME = "regions_england.yaml"
_REGION_LEVELS = ("county", "subdivision")

# The closed set of buckets classify_region sorts a value into.
RegionStatus = Literal[
    "empty", "non-england", "node", "alias", "zone", "quarantine", "country-root", "unknown"
]


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
    aliases: MappingProxyType[str, str]  # variant -> canonical node
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
    """Parse + validate the bundled England region model once per process.

    Read-only (immutable dataclass + ``MappingProxyType``): shared, long-lived
    cached static config, never request-derived state.

    Fails LOUD on a malformed model rather than degrading to a silently-empty
    (guard-disabled) one: the file must parse as a mapping, carry exactly one
    country-level node, and every alias target must be a valid region node — a
    config typo that pointed an alias at a zone/quarantine/missing node would
    otherwise let an invalid region slip through :func:`canonicalize_region`.
    """
    text = resources.files(_DATA_PACKAGE).joinpath(_FILENAME).read_text(encoding="utf-8")
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as e:  # name the file — _model() is cached, so this is permanent
        raise ValueError(f"{_FILENAME} is not valid YAML: {e}") from e
    return _build_model(parsed)


def _build_model(m: object) -> _EnglandModel:
    """Validate a parsed region-model mapping into an immutable ``_EnglandModel``.

    Split out from :func:`_model` as a pure (uncached, no-I/O) function so the
    fail-loud guards are directly testable without a malformed YAML fixture. The
    guards (all raise ``ValueError``): mapping shape; exactly one country node;
    unique alias sources (a dict would silently drop a duplicate); alias sources
    disjoint from nodes/zones/quarantine (an overlap is a contradiction — a node
    folded away, or a quarantine value folded instead of rejected); alias targets
    are valid region nodes. Any of these would corrupt ``canonicalize_region``.
    """
    if not isinstance(m, dict):
        raise ValueError(f"{_FILENAME} did not parse as a mapping")
    try:
        nodes = m.get("nodes") or []
        valid = frozenset(n["canonical"] for n in nodes if n["level"] in _REGION_LEVELS)
        country_nodes = [n["canonical"] for n in nodes if n["level"] == "country"]
        alias_pairs = [(a["from"], a["to"]) for a in (m.get("aliases") or [])]
        zones = frozenset(z["value"] for z in (m.get("deferred_zones") or []))
        quarantine = frozenset(q["value"] for q in (m.get("quarantine") or []))
    except (KeyError, TypeError) as e:
        # A node/alias/zone/quarantine entry is missing an expected key or isn't
        # a mapping — surface a clear error instead of a cryptic KeyError.
        raise ValueError(f"{_FILENAME} has a malformed entry (missing key {e})") from e
    if len(country_nodes) != 1:
        raise ValueError(
            f"{_FILENAME} must have exactly one country-level node, got {country_nodes}"
        )
    sources = [frm for frm, _ in alias_pairs]
    dup_sources = {s for s in sources if sources.count(s) > 1}
    if dup_sources:
        raise ValueError(f"{_FILENAME} has duplicate alias sources: {dup_sources}")
    aliases = dict(alias_pairs)
    bad_sources = set(aliases) & (valid | zones | quarantine)
    if bad_sources:
        raise ValueError(
            f"{_FILENAME} alias sources collide with nodes/zones/quarantine: {bad_sources}"
        )
    bad_targets = {frm: to for frm, to in aliases.items() if to not in valid}
    if bad_targets:
        raise ValueError(f"{_FILENAME} alias targets are not valid region nodes: {bad_targets}")
    return _EnglandModel(valid, country_nodes[0], MappingProxyType(aliases), zones, quarantine)


def canonicalize_region(region: str | None, *, country: str | None = None) -> str | None:
    """Validate + canonicalize a region value at the ingest boundary.

    Returns:

    - ``None`` for ``None`` / empty / whitespace-only (no region);
      a non-empty value is whitespace-stripped before lookup;
    - the canonical node for an alias variant (folded);
    - a canonical region node unchanged;
    - a non-England / out-of-scope value unchanged (passed through);

    or raises :class:`RegionValidationError` for an England-scoped value that is
    not a valid region node (country root / deferred zone / quarantine / unknown).

    Note the return value does NOT encode provenance: a string came back either
    because it is a *validated* England node or because it was *deferred* as
    out-of-scope. The only England-validity signal is "did this raise?". A future
    multi-country caller must not treat a passed-through value as validated.

    Scope precedence: an explicit ``country`` is authoritative — when it is
    supplied, the value is England-scoped iff ``country == "England"`` (a caller
    asserting a non-England country is respected, even for an England-looking
    string). Only when ``country`` is omitted does England-scope fall back to the
    region→country map plus the model's own recognition (so a known England
    zone/quarantine value is still caught with no country hint).
    """
    if region is not None:
        region = region.strip()  # tolerate stray whitespace from source data
    if not region:  # None / "" / whitespace-only — normalize a missing region to NULL
        return None
    model = _model()
    if country is not None:
        england_scoped = country == "England"
    else:
        england_scoped = country_for_region(region) == "England" or model.recognizes(region)
    if not england_scoped:
        # Non-England dimension (Scotland/Wales/…) or an unknown with no England
        # signal — out of scope here; its model is wyrd-3q6m.5.
        return region
    if region in model.aliases:
        # Safe: _model() validates at load that every alias target is a valid
        # region node, so this can never return a zone/quarantine/root value.
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


def classify_region(region: str | None, *, country: str | None = None) -> RegionStatus:
    """Read-only counterpart to :func:`canonicalize_region` — classify a region
    value against the England model WITHOUT raising, for health reports (wyrd-cgl4).

    Returns one of: ``empty`` (None/blank), ``non-england`` (out of scope, passed
    through), ``node`` (valid canonical), ``alias`` (variant that *would fold* —
    its presence in stored data means it was never normalized), ``zone`` (a
    deferred cultural/linguistic zone — expected, awaits wyrd-hytz), ``quarantine``
    (a known-bad coding needing re-code), ``country-root`` (the country used as a
    region), or ``unknown`` (England-scoped but not modeled at all).

    Shares the scope logic + the model's buckets with :func:`canonicalize_region`,
    so the gate and the report can't disagree on what a value is.
    """
    if region is not None:
        region = region.strip()
    if not region:
        return "empty"
    model = _model()
    if country is not None:
        england_scoped = country == "England"
    else:
        england_scoped = country_for_region(region) == "England" or model.recognizes(region)
    if not england_scoped:
        return "non-england"
    if region in model.aliases:
        return "alias"
    if region in model.valid_nodes:
        return "node"
    if region == model.country_root:
        return "country-root"
    if region in model.zones:
        return "zone"
    if region in model.quarantine:
        return "quarantine"
    return "unknown"
