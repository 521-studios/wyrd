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
from wyrd.seed import rng_for


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


# ---- wyrd-izcr: vector path honors structure qualifier flags ---------------


def test_vector_seed_1170_no_longer_produces_split_affix_port_all():
    """wyrd-izcr regression: pre-fix seed 1170 produced ``Port All``
    at sub-seed 5 — Port- + -all rendered as two separate words
    because the vector path dropped the name/saint flags from
    qualifier-flagged structures (e.g. [pre+saint, post+name]).
    Post-fix the same struct triggers the qualifier filter; if
    the saint/name pool is empty under the live request, the
    NameGenerator retries with a different struct.

    ``Kenning.generate`` produces one name per seed; the original
    bug surfaced via the CLI's count=N dispatch which derives
    sub-seeds from the master seed. Replay the same per-sub-seed
    pattern here so the regression catches the actual buggy slot
    rather than just seed=1170 itself."""
    k = Kenning()
    master = rng_for(1170)
    sub_seeds = [master.randint(0, 2**63) for _ in range(10)]
    names: list[str] = []
    for sub in sub_seeds:
        result = k.generate(
            {"culture": "english", "scoring_mode": "vector"},
            seed=sub,
        )
        names.append(result.result)
    assert "Port All" not in names, f"Port All bug regressed: {names}"


def test_vector_slot_qualifier_cache_separates_qualifier_variants():
    """wyrd-izcr: the slot-score cache key is now (position,
    qualifier) so a (pre, "name") slot and a (pre, None) slot
    populate distinct entries. Without the qualifier in the key the
    smaller name-filtered list would shadow the larger no-filter
    list for any later slot that wanted the full pre pool."""
    name_gen = _reset_vector_caches()
    k = Kenning()
    k.generate({"culture": "english", "scoring_mode": "vector"}, seed=1170)
    # Every cached key is a (slot_position, slot_qualifier) tuple.
    _, slot_dict = next(iter(name_gen._vector_slot_score_cache.items()))
    for key in slot_dict.keys():
        assert isinstance(key, tuple) and len(key) == 2, (
            f"slot_base_scores cache key must be (position, qualifier) tuple: got {key!r}"
        )
        position, qualifier = key
        assert position in ("pre", "inner", "post")
        assert qualifier in (None, "name", "saint")


def _build_synthetic_vector_name_gen(structs: dict, meaning_db: dict):
    """Build a NameGenerator with synthetic data for direct
    select_via_vector testing — bypasses the live bundle so the
    retry loop, qualifier filter, and exhaust path can be exercised
    deterministically."""
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    proportions = dict.fromkeys(meaning_db, 1)
    mg = MeaningGenerator(meaning_db, {}, proportions)
    mg.load_parts(proportions, "single")
    return NameGenerator(meaning_db, mg, structs)


def _vector_request(culture: str = "english"):
    """Build a RequestVector that produces non-zero scores against
    a synthetic Meaning tagged 'urban' — so eligibility, not
    scoring, determines whether the dispatch picks a meaning."""
    from wyrd.generators.kenning.vectors.schemas import (
        EligibilityGate,
        RegisterEffect,
        RequestVector,
        ScoringWeights,
    )

    return RequestVector(
        gate=EligibilityGate(culture=culture),
        register=RegisterEffect(name="test", semantic_tags={"urban": 1.0}),
        weights=ScoringWeights(),
    )


def test_select_via_vector_retry_exhausts_when_no_qualifier_pool():
    """wyrd-izcr: when every struct in the pool requires a qualifier
    slot but the meaning_db has zero qualifier-flagged morphemes,
    NameGenerator.select_via_vector exhausts the retry budget and
    returns None — caller (Kenning.generate) then raises the
    operator-visible ValueError."""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    # meaning_db has only bare morphemes — no is_name() True, no
    # literal Saint- usage. Any qualifier-flagged struct slot is
    # unsatisfiable.
    db = {
        "Port-": [Meaning(usage="Port-", tags=["urban"], meanings=[], sources=[])],
        "-all": [Meaning(usage="-all", tags=["urban"], meanings=[], sources=[])],
    }
    # Two structs, both with qualifier-flagged slots — exactly the
    # shape that produced "Port All" pre-fix.
    structs = {
        ((("pre", "saint", "single"),), (("post", "name", "single"),)): 4,
        ((("pre", "saint", "single"),), (("pre", "name", "single"),)): 11,
    }
    name_gen = _build_synthetic_vector_name_gen(structs, db)
    result = name_gen.select_via_vector(
        random.Random(0),
        request=_vector_request(),
        priors=EmpiricalPriors(),
    )
    assert result is None


def test_select_via_vector_retry_exclues_tried_structs():
    """wyrd-izcr: after an attempt fails (qualifier pool empty),
    that struct is excluded from subsequent retries. Without
    exclusion, weighted_choice could re-pick the same un-satisfiable
    struct and burn the retry budget without exploring alternatives.
    Pin via a synthetic 2-struct pool where one is unsatisfiable
    and one is — the satisfiable struct MUST be picked within the
    retry budget regardless of initial weighted-choice outcome."""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    # One usable bare morpheme + one name-flagged morpheme.
    smith = Meaning(usage="Smith-", tags=["family name", "urban"], meanings=[], sources=[])
    bare = Meaning(usage="Port-", tags=["urban"], meanings=[], sources=[])
    end = Meaning(usage="-end", tags=["urban"], meanings=[], sources=[])
    db = {"Smith-": [smith], "Port-": [bare], "-end": [end]}
    # Heavy weight on the unsatisfiable struct so it dominates the
    # initial pick; the satisfiable compound struct survives only
    # via retry-with-exclusion.
    unsatisfiable = ((("pre", "saint", "single"),), (("pre", "name", "single"),))
    satisfiable = ((("pre",), ("post",)),)
    structs = {unsatisfiable: 999, satisfiable: 1}
    name_gen = _build_synthetic_vector_name_gen(structs, db)
    result = name_gen.select_via_vector(
        random.Random(0),
        request=_vector_request(),
        priors=EmpiricalPriors(),
    )
    # Must succeed: retry should drop the unsatisfiable struct and
    # land on the compound one.
    assert result is not None


def test_select_via_vector_retry_limit_bounds_attempts(monkeypatch):
    """wyrd-izcr: _VECTOR_STRUCT_RETRY_LIMIT bounds the retry budget.
    Pin the constant by counting weighted_choice calls when every
    struct is unsatisfiable — should be at most the limit."""
    import random

    from wyrd.generators.kenning.runtime import proportions as proportions_mod
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    db = {"-all": [Meaning(usage="-all", tags=["urban"], meanings=[], sources=[])]}
    structs = {
        ((("pre", "saint", "single"),), (("post", "name", "single"),)): 4,
        ((("pre", "saint", "single"),), (("pre", "name", "single"),)): 11,
    }
    name_gen = _build_synthetic_vector_name_gen(structs, db)

    call_count = 0
    real_weighted_choice = proportions_mod.weighted_choice

    def counting_weighted_choice(rng, items):
        nonlocal call_count
        call_count += 1
        return real_weighted_choice(rng, items)

    monkeypatch.setattr(proportions_mod, "weighted_choice", counting_weighted_choice)
    # Force the limit down so the test can pin the exact bound without
    # depending on how many structs survive exclusion.
    monkeypatch.setattr(proportions_mod, "_VECTOR_STRUCT_RETRY_LIMIT", 3)

    result = name_gen.select_via_vector(
        random.Random(0),
        request=_vector_request(),
        priors=EmpiricalPriors(),
    )
    assert result is None
    # Each retry consumes one weighted_choice; after the per-struct
    # exclusion drains the 2-element pool, the retry loop's
    # ``if not remaining: return None`` exits early — so at most 2
    # weighted_choice calls were issued, less than the limit of 3.
    assert call_count <= 3


def test_select_via_vector_saint_filter_matches_legacy_literal_usage_only():
    """wyrd-izcr: saint-qualifier filter uses the legacy literal
    usage check (Meaning.usage.replace("-", "").lower() == "saint"),
    NOT m.is_saint() (the tag check). Andrew- is saint-tagged but
    its usage is "Andrew-", not "Saint-". Legacy bucket-keying
    routes it through (pre, "name") via Meaning.key()'s if/elif
    precedence; the vector qualifier filter must do the same to
    avoid behavioral divergence between modes."""
    import random

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    # Andrew- is saint-tagged but is also is_name() True (male
    # name) — legacy routes it to the name bucket, not saint.
    andrew = Meaning(
        usage="Andrew-",
        tags=["male name", "saint", "urban"],
        meanings=[],
        sources=[],
    )
    # Literal Saint- — the only morpheme that fills the saint slot
    # in legacy mode.
    saint_literal = Meaning(
        usage="Saint-",
        tags=["saint", "urban"],
        meanings=[],
        sources=[],
    )
    db = {"Andrew-": [andrew], "Saint-": [saint_literal]}
    # Saint-flagged single-element 1-word struct.
    structs = {((("pre", "saint", "single"),),): 1}
    name_gen = _build_synthetic_vector_name_gen(structs, db)
    # Over many seeds, only Saint- should ever fill the saint slot
    # — never Andrew-. NewName.name is list-of-words; the slot
    # struct is a 1-word 1-element compound so name[0][0] is the
    # picked morpheme's usage.
    picked: set[str] = set()
    for seed in range(20):
        result = name_gen.select_via_vector(
            random.Random(seed),
            request=_vector_request(),
            priors=EmpiricalPriors(),
        )
        if result is None:
            continue
        picked.add(result.name[0][0])
    assert picked == {"Saint-"}, f"saint slot admitted non-literal usage: {picked}"
