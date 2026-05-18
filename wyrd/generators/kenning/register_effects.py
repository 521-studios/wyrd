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
  (the canonical 14 v1 dimensions in ``vector_schemas.py``). Typos
  raise loudly here rather than silently no-op via the dot()
  forgiveness path at scoring time.
* Every weight is a finite float in ``[-1.0, +1.0]``. NaN / Inf /
  out-of-range values raise loudly.
* Effect names are non-empty and unique. (PyYAML mapping parse
  already rejects duplicate keys with ``DuplicateKeyError`` on
  ``yaml.safe_load``.)
* Each per-effect block has the three required dict fields (one
  can be ``{}``, but the key must be present so a typo in the
  field name surfaces at load time).

The runtime path is read-once / cache-forever: the catalog file is
small (~12 entries on v1) and immutable across a single process
lifetime. The first ``load_register_effects()`` call parses and
validates; subsequent calls return the cached mapping.

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
from pathlib import Path
from typing import Any

import yaml

from wyrd.generators.kenning.vector_schemas import (
    _DIMENSION_NAMES,
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


def _validate_weight_dict(
    value: Any, label: str, allowed_keys: frozenset[str] | None = None
) -> dict[str, float]:
    """Validate one of the three per-effect weight dicts.

    ``value`` must be a (possibly empty) mapping of string-keyed
    finite floats in [-1, +1]. ``allowed_keys`` restricts which
    keys are permitted (used for phonological); when ``None``, any
    string key is accepted (semantic_tags / position_bias are
    open-set).
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RegisterEffectCatalogError(f"{label}: must be a mapping, got {type(value).__name__}")
    out: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise RegisterEffectCatalogError(
                f"{label}: keys must be non-empty strings, got {key!r}"
            )
        if allowed_keys is not None and key not in allowed_keys:
            raise RegisterEffectCatalogError(
                f"{label}: unknown key {key!r}; expected one of {sorted(allowed_keys)}"
            )
        out[key] = _validate_weight(raw, f"{label}.{key}")
    return out


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
    return RegisterEffect(
        name=name,
        phonological=_validate_weight_dict(
            entry["phonological"], f"{name}.phonological", _DIMENSION_NAMES
        ),
        semantic_tags=_validate_weight_dict(entry["semantic_tags"], f"{name}.semantic_tags"),
        position_bias=_validate_weight_dict(entry["position_bias"], f"{name}.position_bias"),
    )


def _default_catalog_path() -> Path:
    """Resolve the bundled catalog path via importlib.resources."""
    return Path(
        str(resources.files("wyrd.generators.kenning.data").joinpath("register_effects.yaml"))
    )


def load_register_effects_from_text(text: str) -> dict[str, RegisterEffect]:
    """Parse and validate a YAML catalog string.

    Useful for tests that want to exercise specific shapes without
    a file round-trip. Raises ``RegisterEffectCatalogError`` on any
    shape / value violation.
    """
    parsed = yaml.safe_load(text)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise RegisterEffectCatalogError(
            f"catalog root must be a mapping, got {type(parsed).__name__}"
        )
    return {name: _validate_entry(name, entry) for name, entry in parsed.items()}


def load_register_effects(path: Path | None = None) -> dict[str, RegisterEffect]:
    """Read + validate the register-effect catalog from a YAML file.

    ``path``: optional override (used by tests). When ``None``, reads
    the bundled ``wyrd/generators/kenning/data/register_effects.yaml``
    via importlib.resources so the catalog ships with the package.

    Cached via the sibling ``_load_bundled()`` for the default path.
    Explicit-path loads bypass the cache so callers can iterate on a
    test catalog without import-state contamination.
    """
    if path is None:
        return _load_bundled()
    return load_register_effects_from_text(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_bundled() -> dict[str, RegisterEffect]:
    """Cache-once load of the bundled catalog. Returns the dict
    directly — callers MUST treat it as immutable (no in-place
    mutation; use ``RegisterEffect.scaled`` for graduation copies)."""
    return load_register_effects_from_text(_default_catalog_path().read_text(encoding="utf-8"))


def get_register_effect(name: str) -> RegisterEffect:
    """Look up a single register effect by name from the bundled
    catalog. Raises ``KeyError`` (with the available-names list in
    the message) if absent.
    """
    catalog = _load_bundled()
    if name not in catalog:
        raise KeyError(f"unknown register effect {name!r}; catalog has {sorted(catalog.keys())}")
    return catalog[name]
