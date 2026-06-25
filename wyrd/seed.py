"""Seed handling: accept, default, echo back."""

from __future__ import annotations

import random
import secrets

# wyrd-aof8: cap generated seeds at the JS Number safe-integer
# range (2**53 - 1 = 9007199254740991). JSON doesn't distinguish
# Number from integer, so a 64-bit Python seed becomes a JS Number
# on the SPA — which only preserves 53 bits of integer precision.
# Above 2**53, the displayed seed is truncated; copying it back
# would send a DIFFERENT value to the server (user-visible bug:
# 'copy seed from result, paste, re-roll → different name'). 2**53
# still gives ~9 quadrillion seeds, plenty of entropy.
#
# Operator-supplied seeds beyond MAX_SAFE_INTEGER pass through —
# they're explicitly chosen and the operator presumably knows.
# Only auto-generated seeds (random + sub-seeds) get the cap.
MAX_SAFE_INTEGER = 2**53 - 1


def resolve_seed(value: int | str | None) -> int:
    """Coerce a request-supplied seed to an int, or generate a random
    JS-safe seed when blank.

    Raises ``ValueError`` for a non-coercible seed — a non-numeric string
    (``"abc"``, ``"1.5"``) or a non-string/non-int (a list/object) — so the
    dispatcher maps it to a 400 bad_params. (A bare ``int(value)`` would raise
    ValueError for a bad string but TypeError for a list; both surface as
    ValueError here for a single, catchable bad-param signal.)"""
    if value is None or value == "":
        return secrets.randbelow(MAX_SAFE_INTEGER + 1)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"seed must be an integer, got {value!r}") from e


def rng_for(seed: int) -> random.Random:
    """A dedicated Random instance for a given seed.

    Generators MUST use this rather than the global random module so the same
    seed produces the same output across requests.
    """
    return random.Random(seed)
