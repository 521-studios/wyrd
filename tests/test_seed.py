"""Unit tests for wyrd.seed: seed coercion and reproducible RNG construction."""

from __future__ import annotations

import random

from wyrd.seed import MAX_SAFE_INTEGER, resolve_seed, rng_for


def test_resolve_seed_none_returns_random_js_safe():
    """wyrd-aof8: auto-generated seeds must fit in the JS Number
    safe-integer range so they round-trip through copy/paste in
    the SPA. Pre-fix they were 64-bit, lost precision on JS
    Number, and copy/paste yielded a different seed → different
    name."""
    a = resolve_seed(None)
    b = resolve_seed(None)
    assert isinstance(a, int)
    assert 0 <= a <= MAX_SAFE_INTEGER
    assert a != b


def test_resolve_seed_empty_string_treated_as_none():
    a = resolve_seed("")
    b = resolve_seed("")
    assert isinstance(a, int)
    assert 0 <= a <= MAX_SAFE_INTEGER
    assert a != b


def test_resolve_seed_int_passes_through():
    """Operator-supplied seeds (incl. beyond MAX_SAFE_INTEGER)
    pass through unchanged — only auto-generation is capped."""
    assert resolve_seed(42) == 42
    assert resolve_seed(0) == 0
    assert resolve_seed(2**63 - 1) == 2**63 - 1


def test_max_safe_integer_matches_js_constant():
    assert MAX_SAFE_INTEGER == 9007199254740991  # Number.MAX_SAFE_INTEGER


def test_resolve_seed_string_int_is_coerced():
    assert resolve_seed("42") == 42
    assert resolve_seed("0") == 0


def test_resolve_seed_non_coercible_raises_value_error():
    """A non-coercible seed raises ValueError (a bad-param signal the dispatcher
    maps to 400) — never a bare TypeError. A non-numeric string raises ValueError
    on int(); a list/object raises TypeError on int() — both surface as ValueError
    here so a single catch covers them."""
    import pytest

    for bad in ("abc", "1.5", "  ", [1, 2], {"x": 1}):
        with pytest.raises(ValueError, match="seed must be an integer"):
            resolve_seed(bad)


def test_rng_for_returns_random_instance():
    assert isinstance(rng_for(42), random.Random)


def test_rng_for_is_reproducible():
    a = rng_for(42)
    b = rng_for(42)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_rng_for_different_seeds_diverge():
    a = rng_for(1)
    b = rng_for(2)
    assert a.random() != b.random()


def test_rng_for_does_not_affect_global_random():
    """rng_for(seed) must not perturb the process-wide RNG."""
    random.seed(99)
    expected = random.random()
    random.seed(99)
    rng_for(7).random()
    assert random.random() == expected
