"""Seed handling: accept, default, echo back."""

from __future__ import annotations

import random
import secrets


def resolve_seed(value: int | str | None) -> int:
    """Coerce a request-supplied seed to an int, or generate a random 64-bit seed."""
    if value is None or value == "":
        return secrets.randbits(64)
    if isinstance(value, int):
        return value
    return int(value)


def rng_for(seed: int) -> random.Random:
    """A dedicated Random instance for a given seed.

    Generators MUST use this rather than the global random module so the same
    seed produces the same output across requests.
    """
    return random.Random(seed)
