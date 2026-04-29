"""Unit tests for wyrd.seed: seed coercion and reproducible RNG construction."""

from __future__ import annotations

import random

from wyrd.seed import resolve_seed, rng_for


def test_resolve_seed_none_returns_random_64bit():
    a = resolve_seed(None)
    b = resolve_seed(None)
    assert isinstance(a, int)
    assert 0 <= a < 2**64
    # Astronomically unlikely to collide.
    assert a != b


def test_resolve_seed_empty_string_treated_as_none():
    a = resolve_seed("")
    b = resolve_seed("")
    assert isinstance(a, int)
    assert a != b


def test_resolve_seed_int_passes_through():
    assert resolve_seed(42) == 42
    assert resolve_seed(0) == 0
    assert resolve_seed(2**63 - 1) == 2**63 - 1


def test_resolve_seed_string_int_is_coerced():
    assert resolve_seed("42") == 42
    assert resolve_seed("0") == 0


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
