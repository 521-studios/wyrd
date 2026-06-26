"""Feature-flag + default-value resolution for the SPA config UI (wyrd-0gou).

The SPA gates each advanced configuration option behind a feature flag so
untested options stay hidden in production and can be turned on one-by-one as
they're validated. This is a UI concern only — the API stays permissive; the
flags only decide what the SPA *renders*.

Resolution is env-driven, so flags + default values are settable per
environment with no rebuild (the resolved config rides on /api/manifest):

  WYRD_FF_ALL=true            master override → every flag on (staging sets
                              this so a forgotten flag can't hide an option)
  WYRD_FF_<NAME>=true|false   per-flag override; NAME is the flag name
                              upper-cased with '.' / '-' replaced by '_'
                              (e.g. flag ``culture.welsh`` → WYRD_FF_CULTURE_WELSH)
  WYRD_DEFAULT_<OPTION>=<v>   override an option's default VALUE (e.g.
                              WYRD_DEFAULT_CULTURE=english)

Every flag defaults OFF. The shipped shape is

  {"all": bool, "flags": {<ENVIFIED_NAME>: bool}, "defaults": {<option>: str}}

and the SPA resolves a flag as ``all OR flags[envify(name)]`` (absent → off).
The canonical flag-name list lives on the SPA side (it owns the flag→field
mapping); the server stays registry-free, so adding a flag is just a new env
var + a new SPA mapping entry — no change here.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping

# Leading optional-sign integer, mirroring JS ``parseInt(raw, 10)``: it parses
# the leading run of digits and ignores any trailing fractional/garbage part
# (``parseInt('5.0', 10) === 5``, ``parseInt('5px', 10) === 5``). ASCII ``[0-9]``,
# NOT ``\d``: ``\d`` also matches Unicode digits (Arabic-Indic ``١٢٣``, Devanagari)
# which ``int()`` would then accept, but JS ``parseInt`` is ASCII-only and yields
# NaN there — so ``\d`` would re-introduce a server/SPA divergence, not close one.
_LEADING_INT_RE = re.compile(r"[+-]?[0-9]+")

_FF_PREFIX = "WYRD_FF_"
_DEFAULT_PREFIX = "WYRD_DEFAULT_"
_ALL_VAR = "WYRD_FF_ALL"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in _TRUE_VALUES


def resolve_feature_config(env: Mapping[str, str] | None = None) -> dict:
    """Resolve the SPA feature-flag + default-value config from ``env``.

    Returns ``{"all": bool, "flags": {envified_name: bool}, "defaults":
    {option: str}}``:

    * ``all`` — the master override (``WYRD_FF_ALL``). When true the SPA
      treats every flag as on regardless of individual settings.
    * ``flags`` — every explicitly-set ``WYRD_FF_<NAME>`` (excluding
      ``WYRD_FF_ALL``), keyed by its envified suffix, parsed to bool.
    * ``defaults`` — every ``WYRD_DEFAULT_<OPTION>``, keyed by the
      lower-cased option name, value kept as the raw string (the SPA
      coerces to the field's schema type when seeding).

    Pure: reads only the passed mapping (``os.environ`` by default), so it
    is trivially testable with a synthetic env.
    """
    src = os.environ if env is None else env
    all_on = _parse_bool(src.get(_ALL_VAR, ""))
    flags: dict[str, bool] = {}
    defaults: dict[str, str] = {}
    for key, value in src.items():
        if key == _ALL_VAR:
            continue
        if key.startswith(_FF_PREFIX):
            # Keep the envified suffix as the key; the SPA computes the same
            # suffix from the flag name, so no ambiguous reverse-parsing of
            # 'CULTURE_WELSH' → 'culture.welsh' is ever needed.
            flags[key[len(_FF_PREFIX) :]] = _parse_bool(value)
        elif key.startswith(_DEFAULT_PREFIX):
            defaults[key[len(_DEFAULT_PREFIX) :].lower()] = value
    return {"all": all_on, "flags": flags, "defaults": defaults}


def _coerce_to_schema_type(raw: str, prop: Mapping) -> object | None:
    """Coerce a ``WYRD_DEFAULT_*`` string to an option's JSON-schema type.

    The server-side mirror of the SPA's ``featureFlags.coerceToType``. Returns
    ``None`` for an empty/uncoercible value or an ``array`` option, so the
    caller falls back to the option's own schema default rather than seeding
    garbage (matches the SPA, which returns ``undefined`` in those cases)."""
    schema_type = (prop or {}).get("type")
    if schema_type == "array":
        return None
    if schema_type == "integer":
        # Mirror the SPA's ``parseInt(raw, 10)``: take the leading optional-sign
        # integer, ignoring a trailing fractional/garbage part ("5.0"→5, "5px"→5).
        # Plain ``int(text)`` rejects those (ValueError) and falls back to the
        # schema default while the SPA seeds the parsed number — exactly the
        # wyrd-7f22 SPA/server mismatch this module exists to prevent.
        m = _LEADING_INT_RE.match(str(raw).strip())
        return int(m.group()) if m is not None else None
    if schema_type == "number":
        text = str(raw).strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        # float() accepts "nan"/"inf"; reject non-finite so an operator typo
        # (WYRD_DEFAULT_NOVELTY=inf) can't poison the score blend or serialize as
        # invalid-JSON `NaN`/`Infinity` in the echoed params. Mirrors the SPA's
        # Number.isFinite guard (featureFlags.coerceToType — which also rejects an
        # overflow like Number('1e999') → Infinity).
        if not math.isfinite(value):
            return None
        return value
    if schema_type == "boolean":
        return _parse_bool(str(raw))
    return raw


def apply_env_defaults(
    params: dict, properties: Mapping, env: Mapping[str, str] | None = None
) -> dict:
    """Fill option values from ``WYRD_DEFAULT_<OPTION>`` for options ABSENT from
    a generation request, coerced to each option's schema type. Mutates and
    returns ``params``.

    This is the server-side half of the ``WYRD_DEFAULT`` contract (CLAUDE.md:
    it overrides an option's default *value*, not just its SPA visibility). The
    SPA seeds the same overrides for display, but it omits any field still at
    its seeded default from the request (the wyrd-14hn "default = don't include"
    rule), so without this the server would fall back to the SCHEMA default and
    a value-changing override like ``WYRD_DEFAULT_NOVELTY=0.1`` would be a no-op
    on generation while the slider showed 0.1 (wyrd-7f22). Only fills keys that
    are real options for this generator and absent from the request, so an
    explicit request value (incl. one equal to the schema default) always wins
    and bit-stability holds when no ``WYRD_DEFAULT_*`` is set."""
    defaults = resolve_feature_config(env)["defaults"]
    for key, raw in defaults.items():
        if key in params:
            continue
        prop = properties.get(key)
        if prop is None:
            continue
        coerced = _coerce_to_schema_type(raw, prop)
        if coerced is not None:
            params[key] = coerced
    return params
