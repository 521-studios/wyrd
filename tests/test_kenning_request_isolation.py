"""wyrd-i7uy: request isolation for the vector path.

The eligibility pool is filtered by the request's gate (incl. --tag's
required_tags). It used to be memoized on the shared, long-lived NameGenerator
under a key that omitted required_tags, so the FIRST tag's filtered pool stuck
for every later different-tag request on the warm container (tags silently broke
after the first one). These tests pin that switching tags on the SAME generator
always reflects the current request — i.e. no request-derived state survives a
call. Seed-deterministic → non-flaky.
"""

from __future__ import annotations

from wyrd.generators.kenning.generators.kenning import Kenning


def _names(k, tag, n=8):
    return [k.generate({"culture": "english", "tags": [tag]}, seed=i).result for i in range(n)]


def _tag_hit_rate(k, tag, n=8):
    """Fraction of names whose morphemes carry the requested tag (the --tag hard
    gate filters the pool to tag-carrying morphemes, so this should be ~1.0)."""
    hits = 0
    for i in range(n):
        comps = k.generate({"culture": "english", "tags": [tag]}, seed=i).components or []
        if any(tag in (c.get("tags") or []) for c in comps):
            hits += 1
    return hits / n


def test_tag_switch_is_not_sticky_across_calls():
    """Warm the shared generator with one tag, switch to another, switch back —
    each must reflect its own tag, not a cached first-tag pool."""
    k = Kenning()
    water1 = _names(k, "water")
    death = _names(k, "death")
    water2 = _names(k, "water")
    assert death != water1, "death request returned the cached water pool (sticky cache)"
    assert water2 == water1, "same (params, seed) must be deterministic"


def test_each_tag_filters_to_its_own_tag_even_after_warming():
    """The gate actually filters per request — both tags hit their tag at a high
    rate, even though they share the long-lived generator."""
    k = Kenning()
    # interleave so a stale cache from one would corrupt the other
    k.generate({"culture": "english", "tags": ["water"]}, seed=0)
    assert _tag_hit_rate(k, "death") >= 0.9
    assert _tag_hit_rate(k, "water") >= 0.9


def test_no_request_derived_caches_persist_on_the_generator():
    """The vector path must not stash request-derived state on the shared
    NameGenerator (wyrd-i7uy). Guards against the instance-cache pattern coming
    back — bundle/immutable data is fine, request-keyed dicts are not."""
    from wyrd.generators.kenning import _load_culture

    ng, _ = _load_culture("english")
    assert not hasattr(ng, "_vector_eligible_cache")
    assert not hasattr(ng, "_vector_slot_score_cache")
