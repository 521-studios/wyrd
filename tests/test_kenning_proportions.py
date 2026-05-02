"""Unit tests for kenning's weighted_choice — the core weighted sampler."""

from __future__ import annotations

import random
from collections import Counter

from wyrd.generators.kenning.proportions import _blend_uniform, weighted_choice


def test_weighted_choice_all_zero_weights_returns_none():
    assert weighted_choice(random.Random(0), [("a", 0), ("b", 0)]) is None


def test_weighted_choice_empty_returns_none():
    assert weighted_choice(random.Random(0), []) is None


def test_weighted_choice_single_positive_weight_returned():
    assert weighted_choice(random.Random(0), [("a", 1), ("b", 0), ("c", 0)]) == "a"


def test_weighted_choice_skips_zero_weight_items():
    rng = random.Random(0)
    for _ in range(50):
        assert weighted_choice(rng, [("a", 0), ("b", 1), ("c", 0)]) == "b"


def test_weighted_choice_uses_weights_to_pick():
    """Monte Carlo: 90/10 weights should produce ~90% A picks over many trials."""
    rng = random.Random(42)
    counts = {"a": 0, "b": 0}
    for _ in range(1000):
        counts[weighted_choice(rng, [("a", 90), ("b", 10)])] += 1
    # Generous slack — we're checking the weighting is real, not exact ratios.
    assert counts["a"] > 750
    assert counts["b"] > 50


def test_weighted_choice_reproducible_with_same_rng_seed():
    a = random.Random(123)
    b = random.Random(123)
    seq_a = [weighted_choice(a, [("x", 1), ("y", 2), ("z", 3)]) for _ in range(20)]
    seq_b = [weighted_choice(b, [("x", 1), ("y", 2), ("z", 3)]) for _ in range(20)]
    assert seq_a == seq_b


# --- D17 mixture / novelty knob -------------------------------------------


def test_blend_uniform_at_novelty_one_yields_pure_uniform():
    """novelty=1 wipes empirical weights — every key gets 1/n share."""
    blended = _blend_uniform([("a", 90), ("b", 10), ("c", 0)], 1.0)
    weights = dict(blended)
    assert abs(weights["a"] - 1 / 3) < 1e-9
    assert abs(weights["b"] - 1 / 3) < 1e-9
    assert abs(weights["c"] - 1 / 3) < 1e-9


def test_blend_uniform_intermediate_novelty_softens_distribution():
    """At novelty=0.5, the heavy-weight key still leads but the gap shrinks."""
    blended = _blend_uniform([("a", 99), ("b", 1)], 0.5)
    weights = dict(blended)
    # Empirical: a=0.99 b=0.01. Uniform: 0.5 each. Blend: a=0.745, b=0.255.
    assert weights["a"] == 0.5 * (99 / 100) + 0.5 * 0.5
    assert weights["b"] == 0.5 * (1 / 100) + 0.5 * 0.5
    # Sums to 1 (within float rounding).
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_blend_uniform_with_all_zero_weights_yields_uniform():
    """If every empirical weight is zero, the blend defaults to pure uniform
    so all keys still get a share rather than the bucket collapsing to None."""
    blended = _blend_uniform([("a", 0), ("b", 0)], 0.5)
    weights = dict(blended)
    # Each key gets novelty / n = 0.5 / 2 = 0.25 (not normalized but
    # weighted_choice handles fractional weights).
    assert weights["a"] == 0.25
    assert weights["b"] == 0.25


def test_blend_uniform_distribution_via_monte_carlo():
    """At novelty=1, sampling over many trials produces roughly equal counts
    across keys regardless of their original empirical weight."""
    blended = _blend_uniform([("a", 99), ("b", 1)], 1.0)
    rng = random.Random(0)
    counts = Counter(weighted_choice(rng, blended) for _ in range(2000))
    # Wide tolerance band against RNG variance — both keys should be picked
    # roughly equally despite the 99:1 empirical imbalance.
    assert 800 < counts["a"] < 1200
    assert 800 < counts["b"] < 1200


def test_blend_uniform_empty_input_returns_input_unchanged():
    assert _blend_uniform([], 0.5) == []
