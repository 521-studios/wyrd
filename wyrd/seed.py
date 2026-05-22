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
    JS-safe seed when blank."""
    if value is None or value == "":
        return secrets.randbelow(MAX_SAFE_INTEGER + 1)
    if isinstance(value, int):
        return value
    return int(value)


def rng_for(seed: int) -> random.Random:
    """A dedicated Random instance for a given seed.

    Generators MUST use this rather than the global random module so the same
    seed produces the same output across requests.
    """
    return random.Random(seed)
