"""End-to-end tests for Kenning.generate with scoring_mode='vector'
(wyrd-ecjp.5 PR C).

The dispatch path: Kenning.generate reads scoring_mode, translates
the per-call knobs into a RequestVector via the adapter, optionally
loads priors from a JSON sidecar, calls
NameGenerator.select_via_vector, and returns a GenerationResult.

These tests cover the wiring end-to-end against the LIVE bundle —
unlike the vector_name_select unit tests which use synthetic
meaning_db fixtures, the dispatch needs the real bundle's
structures + meanings to round-trip the NewName output surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyrd.generators.kenning.generators.kenning import Kenning


def test_kenning_scoring_mode_proportions_is_default():
    """Default scoring_mode is 'proportions' — bit-stable with the
    legacy path."""
    k = Kenning()
    # Generate without specifying scoring_mode — should not raise
    result = k.generate({"culture": "english"}, seed=42)
    assert result.result  # non-empty surface string


def test_kenning_scoring_mode_proportions_explicit():
    """Explicit scoring_mode='proportions' produces the same output as
    the implicit default (bit-stable)."""
    k = Kenning()
    a = k.generate({"culture": "english"}, seed=42)
    b = k.generate({"culture": "english", "scoring_mode": "proportions"}, seed=42)
    assert a.result == b.result


def test_kenning_scoring_mode_vector_smoke():
    """scoring_mode='vector' without priors — vector path runs but
    might return empty / raise depending on bundle coverage. We just
    check it doesn't crash with an unhandled exception; if it raises
    the documented ValueError, that's fine — operator's fault for
    requesting vector mode without setting up the data."""
    k = Kenning()
    # Pass a register-relevant tag so the sem axis has signal
    try:
        result = k.generate(
            {
                "culture": "english",
                "tags": ["water"],
                "harshness": 0.5,
                "scoring_mode": "vector",
            },
            seed=42,
        )
        # If we got here, the vector path produced a non-empty pick
        assert result.result  # non-empty surface
    except ValueError as e:
        # Documented loud-failure path: vector returned None because
        # the bundle/data combo couldn't produce a non-zero score
        # for any slot. Acceptable in v1 (bundle doesn't yet ship
        # phonological_vector / priors via ecjp.8).
        assert "vector" in str(e).lower()


def test_kenning_scoring_mode_vector_with_priors(tmp_path: Path):
    """scoring_mode='vector' with priors_path pointing at a JSON
    sidecar. The priors load + the baseline axis contributes."""
    # Build a minimal valid priors sidecar
    priors_file = tmp_path / "priors.json"
    priors_file.write_text(
        json.dumps(
            {
                "version": "test-v1",
                "native": [
                    {
                        "culture": "english",
                        "position": "pre",
                        "tag": "water",
                        "era_midpoint": 0,
                        "lemmas": {"bourne": 50, "brook": 30},
                    }
                ],
                "loan": [],
            }
        )
    )
    k = Kenning()
    try:
        result = k.generate(
            {
                "culture": "english",
                "tags": ["water"],
                "harshness": 0.3,
                "scoring_mode": "vector",
                "priors_path": str(priors_file),
            },
            seed=42,
        )
        # Vector path produced something (priors + sem + phon all
        # contribute non-zero for water-tagged lemmas)
        assert result.result
    except ValueError as e:
        # Acceptable — even with priors, the bundle may not have
        # phonological_vector for water-tagged pre-slot lemmas, and
        # the gate might filter everything.
        assert "vector" in str(e).lower()


def test_kenning_scoring_mode_vector_default_uses_uniform_tag_weights():
    """Vector mode with no explicit semantic signal (no tags, no
    mood, no harshness) now falls back to uniform-tag-weights via
    ``build_request_vector``'s no-opinion-default branch — so the
    request still expresses semantic interest (1.0 over the full
    tag universe) and ``baseline_score_native`` returns non-zero
    for any lemma the priors data covers. Pin that the dispatch
    produces a name rather than raising the legacy "no eligible
    name" error.

    Pre-fix contract was: empty register → no eligible name (vector
    mode only worked with a mood). That was the gate on adopting
    vector as a default scoring mode; this test pins the new
    works-by-default contract."""
    k = Kenning()
    result = k.generate(
        {
            "culture": "english",
            "scoring_mode": "vector",
        },
        seed=42,
    )
    assert result.result, "vector mode default-call must produce a non-empty name"


def test_kenning_seed_stable_within_mode():
    """Same seed + same mode → same result (bit-stable contract per
    GenerationResult)."""
    k = Kenning()
    r1 = k.generate({"culture": "english"}, seed=7)
    r2 = k.generate({"culture": "english"}, seed=7)
    assert r1.result == r2.result


def test_kenning_seed_stable_within_vector_mode():
    """Same seed + scoring_mode='vector' → same result. The vector
    path's bit-stability is critical for ecjp.6/7 drift measurement —
    if two runs at the same seed diverge, the drift signal is noise.
    Round-1 reviewer caught the gap that legacy was pinned but the
    vector path wasn't."""
    k = Kenning()
    params = {
        "culture": "english",
        "tags": ["water"],
        "harshness": 0.5,
        "scoring_mode": "vector",
    }
    # Either both raise OR both succeed with identical output. The
    # contract isn't "vector mode succeeds against the live bundle"
    # (that depends on data not yet shipped); it's "same inputs →
    # same outputs whatever those outputs are".
    try:
        r1 = k.generate(params, seed=7)
        r2 = k.generate(params, seed=7)
        assert r1.result == r2.result
    except ValueError:
        # If the first raised, the second must raise identically too.
        with pytest.raises(ValueError):
            k.generate(params, seed=7)


def test_kenning_scoring_mode_unknown_value_validated_by_schema():
    """Per the schema's enum, 'banana' is invalid. The generator
    itself doesn't validate (relies on the schema layer); but a
    typo'd value falls through the else branch and runs the legacy
    path — pinning that fallback behavior."""
    k = Kenning()
    # Unknown scoring_mode → falls through to else → legacy path runs
    result = k.generate(
        {"culture": "english", "scoring_mode": "banana"},
        seed=42,
    )
    assert result.result


def _reset_vector_caches():
    """Clear both vector caches on the english NameGenerator. The
    cache is process-wide (lru_cached _load_culture instance) so
    other tests can leak state in; reset before asserting cache
    contents."""
    from wyrd.generators.kenning import _load_culture

    name_gen, _ = _load_culture("english")
    name_gen._vector_eligible_cache.clear()
    name_gen._vector_slot_score_cache.clear()
    return name_gen


def test_vector_eligibility_pool_cache_reuses_across_dispatch():
    """The NameGenerator caches the non-position eligibility pool
    on first ``select_via_vector`` call. Subsequent calls with the
    same (gate, exclude_tags, packs) reuse the cached pool — the
    O(N=meaning_db_size) scan only runs ONCE per filter combo per
    container lifetime. Pin so a regression that re-keys the cache
    or invalidates between calls trips the test."""
    name_gen = _reset_vector_caches()
    k = Kenning()

    # First call populates the cache.
    k.generate({"culture": "english", "scoring_mode": "vector"}, seed=42)
    assert len(name_gen._vector_eligible_cache) == 1
    cached_pool_id = id(next(iter(name_gen._vector_eligible_cache.values())))
    # Second call with identical params must reuse the same list.
    k.generate({"culture": "english", "scoring_mode": "vector"}, seed=99)
    assert len(name_gen._vector_eligible_cache) == 1
    assert id(next(iter(name_gen._vector_eligible_cache.values()))) == cached_pool_id


def test_vector_slot_score_cache_reuses_across_sub_seeds():
    """Per-slot ``(meaning, base_score)`` lists are cached on the
    NameGenerator keyed by (filter_key, request_signature,
    era_midpoint, priors_id). Sub-seeds in a count>1 dispatch (or
    repeated identical dispatches) hit the cached lists — the O(P)
    score loop per slot drops to one build per (request, slot)
    pair. Pin so a regression that breaks the cache key trips the
    perf-critical contract."""
    name_gen = _reset_vector_caches()
    k = Kenning()

    # Two calls with identical params (different seed for sub-seed
    # variance; cache key doesn't include seed).
    k.generate({"culture": "english", "scoring_mode": "vector"}, seed=42)
    # One slot-score entry per (request, era, priors) tuple.
    assert len(name_gen._vector_slot_score_cache) == 1
    _, slot_dict = next(iter(name_gen._vector_slot_score_cache.items()))
    # slot_dict accumulates as the dispatch walks the structure;
    # at least one slot_position should be cached after a single roll.
    assert len(slot_dict) >= 1
    cached_slot_id = id(slot_dict)
    # Second identical call: same outer key, same inner dict object.
    k.generate({"culture": "english", "scoring_mode": "vector"}, seed=99)
    assert len(name_gen._vector_slot_score_cache) == 1
    assert id(next(iter(name_gen._vector_slot_score_cache.values()))) == cached_slot_id


def test_vector_slot_score_cache_evicts_on_overflow():
    """The slot-score cache is bounded to prevent unbounded growth
    across warm-Lambda lifetime. Past the cap, FIFO-evict the
    oldest entry on each new insertion."""
    name_gen = _reset_vector_caches()
    k = Kenning()
    # Force the cap small so the test doesn't have to issue 16+ calls
    # with distinct registers. Restore after the test.
    original_max = name_gen._VECTOR_SLOT_CACHE_MAX
    name_gen._VECTOR_SLOT_CACHE_MAX = 2
    try:
        # Three calls with distinct request signatures (harshness > 0
        # puts the adapter into a different register — easiest way to
        # vary the request signature without touching gate or weights).
        for h in (0.1, 0.3, 0.6):
            k.generate(
                {"culture": "english", "scoring_mode": "vector", "harshness": h},
                seed=42,
            )
        # Past the cap of 2 → cache holds only the 2 newest entries.
        assert len(name_gen._vector_slot_score_cache) == 2
    finally:
        name_gen._VECTOR_SLOT_CACHE_MAX = original_max
        name_gen._vector_slot_score_cache.clear()
