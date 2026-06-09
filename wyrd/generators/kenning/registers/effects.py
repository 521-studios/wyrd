"""Register-effect catalog loader (wyrd-kq7w.2 Phase B).

Reads the operator-edited
``wyrd/generators/kenning/data/register_effects.yaml`` catalog and
returns a ``dict[str, RegisterEffect]`` keyed by effect name.

The catalog is the operator's path for adding/editing register
effects without code changes (D36 + the rip-and-replace mood-system
refactor wyrd-kq7w). At catalog load time the loader validates:

* Top-level shape is ``mapping[name -> mapping{phonological,
  semantic_tags, position_bias}]``.
* Phonological feature names are in ``PhonologicalFeatureName``
  (the canonical 14 v1 dimensions in ``vectors/schemas.py``). Typos
  raise loudly here rather than silently no-op via the dot()
  forgiveness path at scoring time.
* Every weight is a finite float in ``[-1.0, +1.0]``. NaN / Inf /
  out-of-range values raise loudly.
* Effect names are non-empty and unique. PyYAML's default behavior
  is silent last-wins on duplicate keys (not the
  ``DuplicateKeyError`` from strictyaml), so this module uses a
  custom ``SafeLoader`` subclass that raises
  ``RegisterEffectCatalogError`` on any duplicate top-level OR
  nested mapping key — a copy-paste mistake on an effect block
  surfaces immediately rather than silently dropping data.
* Each per-effect block has the three required dict fields (one
  can be ``{}`` or bare ``key:`` (YAML null), but the key must be
  present so a typo in the field name surfaces at load time).

The runtime path is read-once / cache-forever: the catalog file is
small (~12 entries on v1) and immutable across a single process
lifetime. The first ``_load_bundled_cached()`` call parses and
validates; subsequent calls return the cached canonical form, and
``get_register_effect`` deep-copies on the way out so callers can
mutate without polluting the cache.

Per the wyrd-kq7w epic, the runtime CLI's ``--register`` flag will
parse colon-suffix graduation (``harsh:0.5``) at the request-vector
construction layer (Phase C, wyrd-kq7w.3) by looking up the named
effect here and applying ``RegisterEffect.scaled(weight)``. This
module exposes the catalog only — the CLI surface is layered on
top.
"""

from __future__ import annotations

import math
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml

from wyrd.generators.kenning.vectors.schemas import (
    _DIMENSION_NAMES,
    WEIGHT_TIERS,
    RegisterEffect,
)

# The three required per-effect dict keys.
_REQUIRED_FIELDS: frozenset[str] = frozenset({"phonological", "semantic_tags", "position_bias"})

# Permitted weight range. Component-wise sum + clamp happens at
# compose time; out-of-range source values are a catalog authoring
# error (you can't compose your way OUT of clamping).
_WEIGHT_MIN: float = -1.0
_WEIGHT_MAX: float = 1.0


class RegisterEffectCatalogError(ValueError):
    """Catalog-load failure. Distinct from generic ValueError so a
    catalog-validation failure surfaces clearly in operator output.
    """


def _validate_weight(value: Any, label: str) -> float:
    """Coerce ``value`` to a float in ``[-1, +1]``; raise if invalid.

    ``label`` identifies the source location for the error message
    (e.g. ``"harsh.phonological.cluster_density"``).
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RegisterEffectCatalogError(
            f"{label}: weight must be a number, got {type(value).__name__} {value!r}"
        )
    weight = float(value)
    if math.isnan(weight) or math.isinf(weight):
        raise RegisterEffectCatalogError(f"{label}: weight must be finite, got {weight!r}")
    if not _WEIGHT_MIN <= weight <= _WEIGHT_MAX:
        raise RegisterEffectCatalogError(
            f"{label}: weight {weight!r} is outside [{_WEIGHT_MIN}, {_WEIGHT_MAX}]"
        )
    return weight


def _validate_tier(value: Any, label: str) -> str:
    """wyrd-we1u: validate a tier-string is one of WEIGHT_TIERS.

    A typo'd tier (``universla`` for ``universal``) is loud-failure
    rather than silent-untagged so an editor reorganization of
    REGISTERS.md → YAML doesn't quietly drop the grounding metadata.
    """
    if not isinstance(value, str):
        raise RegisterEffectCatalogError(
            f"{label}: tier must be a string, got {type(value).__name__} {value!r}"
        )
    if value not in WEIGHT_TIERS:
        raise RegisterEffectCatalogError(
            f"{label}: unknown tier {value!r}; expected one of {sorted(WEIGHT_TIERS)}"
        )
    return value


def _validate_weight_entry(value: Any, label: str) -> tuple[float, str | None]:
    """wyrd-we1u: parse one per-weight value supporting both shapes.

    * Bare numeric (``cluster_density: 0.6``) — back-compat shape.
      Returns ``(weight, None)`` meaning "no tier metadata; caller
      uses the parallel tier-dict's default-on-missing behavior".
    * Mapping with ``value`` + optional ``tier`` keys
      (``cluster_density: {value: 0.6, tier: universal}``) — the
      tier-tagged shape. Returns ``(weight, tier_string)``.

    Returning ``None`` for the bare-float path (rather than
    ``UNTAGGED_TIER``) keeps the parallel-dict authoring sparse:
    only explicitly-tagged keys land in the tier dict, so a
    ``RegisterEffect`` with no migrated weights at all carries
    empty tier dicts rather than every-key-mapped-to-untagged
    bookkeeping. Consumers default missing keys to ``UNTAGGED_TIER``
    anyway.
    """
    if isinstance(value, dict):
        # Tier-tagged shape: must have 'value', optional 'tier'.
        extra = set(value.keys()) - {"value", "tier"}
        if extra:
            raise RegisterEffectCatalogError(
                f"{label}: unexpected per-weight keys {sorted(extra)}; "
                "expected 'value' [+ optional 'tier']"
            )
        if "value" not in value:
            raise RegisterEffectCatalogError(f"{label}: per-weight mapping must include 'value'")
        weight = _validate_weight(value["value"], f"{label}.value")
        tier = _validate_tier(value["tier"], f"{label}.tier") if "tier" in value else None
        return weight, tier
    return _validate_weight(value, label), None


def _validate_weight_dict(
    value: Any, label: str, allowed_keys: frozenset[str] | None = None
) -> tuple[dict[str, float], dict[str, str]]:
    """Validate one of the three per-effect weight dicts.

    ``value`` must be a (possibly empty) mapping. Each entry is
    either a bare finite float in [-1, +1] (back-compat shape) or
    a mapping ``{value: float, tier: str}`` (wyrd-we1u shape).
    ``allowed_keys`` restricts which keys are permitted (used for
    phonological); when ``None``, any string key is accepted
    (semantic_tags / position_bias are open-set).

    Returns ``(weights, tiers)`` — two parallel dicts. Tier dict
    contains only entries whose YAML carried an explicit tier;
    bare-float entries are absent from the tier dict (consumers
    default-on-missing).
    """
    if value is None:
        return {}, {}
    if not isinstance(value, dict):
        raise RegisterEffectCatalogError(f"{label}: must be a mapping, got {type(value).__name__}")
    out_weights: dict[str, float] = {}
    out_tiers: dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise RegisterEffectCatalogError(
                f"{label}: keys must be non-empty strings, got {key!r}"
            )
        if allowed_keys is not None and key not in allowed_keys:
            raise RegisterEffectCatalogError(
                f"{label}: unknown key {key!r}; expected one of {sorted(allowed_keys)}"
            )
        weight, tier = _validate_weight_entry(raw, f"{label}.{key}")
        out_weights[key] = weight
        if tier is not None:
            out_tiers[key] = tier
    return out_weights, out_tiers


def _validate_entry(name: str, entry: Any) -> RegisterEffect:
    """Build a ``RegisterEffect`` from one catalog entry; raise on
    any shape / dimension / weight violation."""
    if not isinstance(name, str) or not name:
        raise RegisterEffectCatalogError(f"effect name must be a non-empty string, got {name!r}")
    if not isinstance(entry, dict):
        raise RegisterEffectCatalogError(
            f"{name}: catalog entry must be a mapping, got {type(entry).__name__}"
        )
    extra = set(entry.keys()) - _REQUIRED_FIELDS
    missing = _REQUIRED_FIELDS - set(entry.keys())
    if missing or extra:
        problem_parts: list[str] = []
        if missing:
            problem_parts.append(f"missing fields {sorted(missing)}")
        if extra:
            problem_parts.append(f"unexpected fields {sorted(extra)}")
        raise RegisterEffectCatalogError(f"{name}: " + "; ".join(problem_parts))
    phon_weights, phon_tiers = _validate_weight_dict(
        entry["phonological"], f"{name}.phonological", _DIMENSION_NAMES
    )
    sem_weights, sem_tiers = _validate_weight_dict(entry["semantic_tags"], f"{name}.semantic_tags")
    pos_weights, pos_tiers = _validate_weight_dict(entry["position_bias"], f"{name}.position_bias")
    return RegisterEffect(
        name=name,
        phonological=phon_weights,
        semantic_tags=sem_weights,
        position_bias=pos_weights,
        phonological_tiers=phon_tiers,
        semantic_tag_tiers=sem_tiers,
        position_bias_tiers=pos_tiers,
    )


class _NoDuplicateKeysLoader(yaml.SafeLoader):
    """SafeLoader subclass that raises on duplicate mapping keys.

    PyYAML's default behavior is silent last-wins on duplicate keys,
    which would let a copy-paste mistake on a register-effect entry
    silently lose data. Raising here is the operator-visible
    fail-loud path.
    """


def _construct_mapping_no_duplicates(loader: yaml.Loader, node: yaml.nodes.MappingNode) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            raise RegisterEffectCatalogError(
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_NoDuplicateKeysLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_no_duplicates,
)


def load_register_effects_from_text(text: str) -> dict[str, RegisterEffect]:
    """Parse and validate a YAML catalog string.

    Useful for tests that want to exercise specific shapes without
    a file round-trip. Raises ``RegisterEffectCatalogError`` on any
    shape / value violation or duplicate top-level keys.

    Bare ``key:`` (YAML null) on a weight-dict field (``phonological``,
    ``semantic_tags``, ``position_bias``) is accepted as an empty
    dict — operator can write ``semantic_tags:`` for "no semantic
    contribution" rather than the explicit ``semantic_tags: {}``.
    """
    parsed = yaml.load(text, Loader=_NoDuplicateKeysLoader)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise RegisterEffectCatalogError(
            f"catalog root must be a mapping, got {type(parsed).__name__}"
        )
    return {name: _validate_entry(name, entry) for name, entry in parsed.items()}


def _copy_effect(effect: RegisterEffect) -> RegisterEffect:
    """Return a fresh ``RegisterEffect`` with copies of all weight
    dicts.

    ``RegisterEffect`` is frozen but holds mutable ``dict[str, float]``
    fields; this rebuild gives each caller fresh dicts to mutate
    without affecting the source instance. Used by ``get_register_effect``
    to enforce mutation isolation against the cached canonical form.
    """
    return RegisterEffect(
        name=effect.name,
        phonological=dict(effect.phonological),
        semantic_tags=dict(effect.semantic_tags),
        position_bias=dict(effect.position_bias),
        phonological_tiers=dict(effect.phonological_tiers),
        semantic_tag_tiers=dict(effect.semantic_tag_tiers),
        position_bias_tiers=dict(effect.position_bias_tiers),
    )


@lru_cache(maxsize=1)
def _load_bundled_cached() -> dict[str, RegisterEffect]:
    """Read + parse + validate the bundled catalog. Cached because the
    parse cost is real and the catalog file is immutable per process.

    Returns the canonical parsed form; callers should NOT mutate the
    returned dict or its nested ``RegisterEffect`` fields.
    ``get_register_effect`` deep-copies on the way out to hand out
    fresh, mutation-safe copies.
    """
    catalog_text = (
        resources.files("wyrd.generators.kenning.data")
        .joinpath("register_effects.yaml")
        .read_text(encoding="utf-8")
    )
    return load_register_effects_from_text(catalog_text)


def get_register_effect(name: str) -> RegisterEffect:
    """Look up a single register effect by name from the bundled
    catalog. Raises ``KeyError`` (with the available-names list in
    the message) if absent.

    Returns a fresh ``RegisterEffect`` instance with copied weight
    dicts so the caller can mutate without polluting the cache.
    """
    catalog = _load_bundled_cached()
    if name not in catalog:
        raise KeyError(f"unknown register effect {name!r}; catalog has {sorted(catalog.keys())}")
    return _copy_effect(catalog[name])


def available_register_effects() -> list[str]:
    """Return the sorted list of register-effect names in the bundled
    catalog. Used by CLI / JSON-schema layers that need to surface
    the operator-facing mood vocabulary without depending on the
    legacy MOODS dict."""
    return sorted(_load_bundled_cached().keys())


def parse_mood_spec(spec: str) -> RegisterEffect:
    """Resolve a single mood spec (``"harsh"`` or ``"harsh:0.5"``) to
    a graduated :class:`RegisterEffect` from the catalog.

    The bare-name form returns the catalog entry at full strength
    (multiplier 1.0). The colon-suffix form parses the value as a
    float and returns the catalog entry scaled by it. Negative or
    >1.0 multipliers are NOT rejected here — graduation past the
    catalog's per-dimension weights is the caller's choice; the
    final ``compose_register_effects`` clamp keeps composed
    requests bounded. Replaces the wyrd-kq7w pre-rip MOODS-dict
    lookups; the catalog is now the single source of truth for
    mood-name resolution.

    Raises:
        KeyError: if ``name`` (the part before any colon) is not in
            the catalog.
        ValueError: if the colon-suffix is present but not parseable
            as a float (matches the legacy ``_apply_mood`` error
            shape — a typo like ``harsh:x`` is loud-failure rather
            than silently no-op'd).
    """
    if ":" in spec:
        name, _, value_str = spec.partition(":")
        try:
            multiplier = float(value_str)
        except ValueError as exc:
            raise ValueError(
                f"mood spec {spec!r} graduation suffix {value_str!r} is not a float"
            ) from exc
    else:
        name, multiplier = spec, 1.0
    effect = get_register_effect(name)
    if multiplier == 1.0:
        return effect
    return effect.scaled(multiplier)
