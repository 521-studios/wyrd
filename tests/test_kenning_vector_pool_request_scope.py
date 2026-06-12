"""wyrd-dag4: the vector eligibility pool is built ONCE per request (reused
across the count loop's names), not once per name — while never persisting on
the shared NameGenerator across requests (the wyrd-i7uy rule).

The pool cache lives in the request's ``params`` dict: the app reuses one
params dict across all N names of a count=N request, so a count loop sharing
that dict builds the pool once; a caller with a fresh dict per call (single
result, tests) rebuilds each time — the prior behavior.
"""

from __future__ import annotations

import pytest

from wyrd.app import MAX_SAFE_INTEGER, rng_for
from wyrd.generators.kenning.generators.kenning import Kenning
from wyrd.generators.kenning.runtime import proportions


@pytest.fixture
def count_build_calls(monkeypatch):
    """Spy that counts NameGenerator._build_vector_pools invocations."""
    calls = {"n": 0}
    orig = proportions.NameGenerator._build_vector_pools

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(proportions.NameGenerator, "_build_vector_pools", counting)
    return calls


def _names_via_count_loop(gen, params, *, seed, count):
    """Mimic app.py's count loop: ONE params dict reused across N sub-seeds."""
    seed_rng = rng_for(seed)
    return [
        gen.generate(params, seed_rng.randrange(MAX_SAFE_INTEGER + 1)).result for _ in range(count)
    ]


def test_pool_built_once_across_the_count_loop(count_build_calls):
    gen = Kenning()
    params = {"culture": "english"}  # one request dict, reused across names
    names = _names_via_count_loop(gen, params, seed=12345, count=5)
    assert len(names) == 5
    assert count_build_calls["n"] == 1, (
        f"pool rebuilt {count_build_calls['n']}x across a count=5 request; expected once"
    )
    # the cache lived in the request's params (request-scoped, discarded with it)
    assert "_vector_pool_cache" in params
    assert "vector_pools" in params["_vector_pool_cache"]


def test_reuse_does_not_change_output(count_build_calls):
    """The once-per-request pool yields the SAME names as a fresh-per-name
    build would (deterministic, gate-derived pool)."""
    gen = Kenning()
    shared = _names_via_count_loop(gen, {"culture": "english"}, seed=999, count=5)
    # fresh params each call → pool rebuilt per name (prior behavior)
    seed_rng = rng_for(999)
    fresh = [
        gen.generate({"culture": "english"}, seed_rng.randrange(MAX_SAFE_INTEGER + 1)).result
        for _ in range(5)
    ]
    assert shared == fresh


def test_pool_not_shared_across_separate_requests(count_build_calls):
    """A fresh params dict per call (separate requests) rebuilds the pool each
    time — no cross-request state leaks via the shared NameGenerator."""
    gen = Kenning()
    seed_rng = rng_for(42)
    for _ in range(3):
        gen.generate({"culture": "english"}, seed_rng.randrange(MAX_SAFE_INTEGER + 1))
    assert count_build_calls["n"] == 3  # one build per separate request
