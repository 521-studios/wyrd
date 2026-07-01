"""wyrd-y0lx: the `kenning-regenerate-morpheme` endpoint — re-roll ONE
morpheme of a generated name while holding the others fixed.

Pins the operator-decided contract (2026-06-11):
  * deterministic per seed (the SPA bakes the seed into the transform step,
    so pipeline re-runs / restored workspaces replay the same pick);
  * the replacement NEVER duplicates the replaced morpheme or any morpheme
    already in use elsewhere in the name (no "Hill Hill" by construction),
    checked on BOTH the modern-usage fold and the native-form fold;
  * the same hard gates as a fresh roll apply (the --tag per-lemma OR-gate
    must keep a tag-filtered name tag-satisfying);
  * held slots pass through verbatim;
  * malformed input surfaces as ValueError → the dispatcher's 400; pool
    exhaustion surfaces as NoEligibleReplacementError (a KenningError /
    EmptyEligiblePool subclass → the dispatcher's 422, wyrd-hh2m — the same
    valid-but-unsatisfiable mapping a fresh roll's empty pool gets).

Runs against the committed seed-runtime.db (no live lexicon DB — CI has
none)."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import pytest

from wyrd.app import create_app
from wyrd.generators.kenning import _load_culture
from wyrd.generators.kenning.errors import EmptyEligiblePool
from wyrd.generators.kenning.generators import Kenning, KenningRegenerateMorpheme
from wyrd.generators.kenning.generators.kenning_regenerate import (
    NoEligibleReplacementError,
    _apply_mood_restriction,
    _collides,
    _derived_position,
    _slot_qualifier,
)
from wyrd.generators.kenning.registers.effects import parse_mood_spec
from wyrd.generators.kenning.runtime import vector_name_select
from wyrd.generators.kenning.runtime.proportions import _native_form_for_morpheme_id
from wyrd.registry import GenerationResult

# This module exercises the kenning runtime, which reads the committed
# seed-runtime.db via get_runtime_db() — it never touches the operator's
# ~/.wyrd/lexicon.db. The module-scoped english_roll fixture initializes
# OUTSIDE the function-scoped autouse isolation fixtures, so opt out
# explicitly to document the bypass (matching the conftest contract).
pytestmark = pytest.mark.no_lexicon_isolation


def _fold(usage: str) -> str:
    return usage.replace("-", "").lower()


@pytest.fixture(scope="module")
def english_roll() -> GenerationResult:
    """One stable english roll to regenerate against (module-scoped — the
    roll itself is not under test here)."""
    result = Kenning().generate({"culture": "english"}, seed=42)
    assert result.morphemes_by_word
    return result


def _regen(
    words: list[list[dict[str, Any]]], wi: int, mi: int, seed: int, **params: Any
) -> GenerationResult:
    payload = {"words": words, "word_index": wi, "morpheme_index": mi, **params}
    results = KenningRegenerateMorpheme().generate_all(payload, seed)
    assert len(results) == 1
    return results[0]


def test_replaces_only_the_target_slot(english_roll):
    words = english_roll.morphemes_by_word
    wi, mi = 0, len(words[0]) - 1
    out = _regen(words, wi, mi, seed=7, culture="english")
    new_words = out.morphemes_by_word
    assert len(new_words) == len(words)
    for w, (old_word, new_word) in enumerate(zip(words, new_words, strict=True)):
        assert len(new_word) == len(old_word)
        for e, (old_m, new_m) in enumerate(zip(old_word, new_word, strict=True)):
            if (w, e) == (wi, mi):
                # Structural, not lucky-seed: the replaced morpheme's own
                # fold is in the exclusion set, so the pick can't equal it.
                assert _fold(new_m["usage"]) != _fold(old_m["usage"])
            else:
                # Held slots pass through with their supplied surfaces.
                assert new_m["usage"] == old_m["usage"]


def test_deterministic_per_seed(english_roll):
    words = english_roll.morphemes_by_word
    wi, mi = 0, len(words[0]) - 1
    first = _regen(words, wi, mi, seed=7, culture="english")
    second = _regen(words, wi, mi, seed=7, culture="english")
    assert first.morphemes_by_word[wi][mi]["usage"] == second.morphemes_by_word[wi][mi]["usage"]
    assert first.result == second.result


def test_never_duplicates_an_in_use_morpheme(english_roll):
    words = english_roll.morphemes_by_word
    wi, mi = 0, len(words[0]) - 1
    used = {_fold(m["usage"]) for word in words for m in word}
    for seed in range(30):
        out = _regen(words, wi, mi, seed=seed, culture="english")
        assert _fold(out.morphemes_by_word[wi][mi]["usage"]) not in used


def test_collides_excludes_by_native_form_fold():
    """The D41 half of the no-duplicate rule: a candidate whose NATIVE form
    folds to an in-use surface is excluded even when its modern usage
    doesn't collide (a fresh `tūn` against a held `tūn` on era/native
    renders)."""
    name_gen, _ = _load_culture("english")
    # Find a real meaning whose native form differs from its modern usage.
    candidate = None
    for meanings in name_gen.meaning_db.values():
        for m in meanings:
            mid = getattr(m, "morpheme_id", None)
            if not mid:
                continue
            native = _native_form_for_morpheme_id(mid)
            if native and _fold(native) != _fold(m.usage):
                candidate = (m, native)
                break
        if candidate:
            break
    assert candidate is not None, "seed bundle should carry a native≠modern morpheme"
    meaning, native = candidate
    # Modern fold not in use, native fold in use → still a collision.
    assert _collides(meaning, {_fold(native)}, _native_form_for_morpheme_id) is True
    # Neither fold in use → no collision.
    assert _collides(meaning, {"zzz-not-a-fold"}, _native_form_for_morpheme_id) is False
    # Modern fold in use → collision regardless of native.
    assert _collides(meaning, {_fold(meaning.usage)}, _native_form_for_morpheme_id) is True


def test_pool_exhaustion_raises_no_eligible_replacement(english_roll, monkeypatch):
    """When the pool is empty after gates + exclusions, the endpoint fails
    loudly with NoEligibleReplacementError (→ the dispatcher's 422, wyrd-hh2m),
    never a silent no-op or duplicate."""
    monkeypatch.setattr(vector_name_select, "_slot_weighted_pool", lambda *a, **k: [])
    words = english_roll.morphemes_by_word
    with pytest.raises(NoEligibleReplacementError, match="no eligible replacement"):
        _regen(words, 0, 0, seed=1, culture="english")


def test_bucket_key_miss_falls_back_to_unweighted_pool(english_roll, monkeypatch, caplog):
    """When the reconstructed bucket key has no frequency-table entry, the
    endpoint degrades to the unweighted score-only pool — loudly (WARNING)
    — instead of failing."""
    real = vector_name_select._slot_weighted_pool

    def fake(*args, **kwargs):
        # Simulate the frequency table missing this bucket: the keyed call
        # returns nothing; the fallback (bucket_key=None, no frequency
        # layer) resolves normally.
        if kwargs.get("slot_bucket_key") is not None:
            return []
        return real(*args, **kwargs)

    monkeypatch.setattr(vector_name_select, "_slot_weighted_pool", fake)
    words = english_roll.morphemes_by_word
    with caplog.at_level(logging.WARNING):
        out = _regen(words, 0, 0, seed=3, culture="english")
    assert out.morphemes_by_word[0][0]["usage"]
    assert any("bucket_key_miss" in r.message for r in caplog.records)


def _carriers(words, tag):
    return [
        (wi, mi)
        for wi, word in enumerate(words)
        for mi, m in enumerate(word)
        if tag in (m.get("tags") or [])
    ]


# Tags to probe for a sole-carrier re-roll scenario. The behavior under test is
# tag-agnostic; we search for whichever tag the committed --dev seed can
# actually exercise (wyrd-x5y4.5: the case-fold reseed thinned some pools —
# e.g. english 'water' fell to 2 morphemes with no position-compatible
# alternative — so a single hardcoded tag is fixture-brittle. The full bundle
# is unaffected; this is a --dev subset limit.)
_REROLL_CANDIDATE_TAGS = ("topography", "tree", "water", "hill", "wood", "stone")


def test_reroll_of_sole_tag_carrier_stays_tagged():
    """wyrd-c6o1.4: a --tag reserves ONE slot (not every slot, the wv85 bug). On
    a re-roll, only the name's SOLE tag-carrier is restricted to tagged
    candidates — so re-rolling it can't silently drop the name's >=1-tagged
    guarantee. (A free slot, with the tag satisfied elsewhere, samples freely.)

    The guarantee is "stays tagged OR fails loudly, never a silent untagged
    swap": a sole-carrier re-roll must EITHER return another tagged morpheme OR
    raise NoEligibleReplacementError (the honest empty-pool signal). On the
    exercised scenario we assert the tagged replacement across a RANGE of
    re-roll seeds (not one pinned seed — a broken c6o1.4 gate drops the tag on
    only a fraction of seeds, so a single seed can coincidentally miss it), and
    treat NoEligible as a valid guarantee-honoring outcome — not a bug the
    search masks. We keep searching seeds (then tags) until the positive
    tagged-replacement path is exercised at least once, else fail."""
    exercised = False
    for tag in _REROLL_CANDIDATE_TAGS:
        for s in range(50):
            try:
                rolled = Kenning().generate({"culture": "english", "tags": [tag]}, seed=s)
            except EmptyEligiblePool:
                # No eligible english name for THIS (tag, seed): the structure
                # sampled at this seed may be unfillable even when the tag has
                # morphemes, so keep trying other seeds before giving up on the
                # tag (continue, not break).
                continue
            carriers = _carriers(rolled.morphemes_by_word, tag)
            assert carriers, f"roll tag={tag} seed={s} must satisfy the --tag guarantee"
            if len(carriers) != 1:
                continue  # want an unambiguous SOLE-carrier name to re-roll
            wi, mi = carriers[0]
            # Re-roll the sole carrier across a RANGE of seeds: EVERY produced
            # replacement must stay tagged. A wv85-class silent untagged swap
            # drops the tag on a large fraction of seeds, so pinning ONE seed
            # can coincidentally miss it (that pick happens to stay tagged even
            # with the c6o1.4 gate removed) — iterate so the guard actually
            # bites the regression. NoEligible on a given seed is the honest
            # empty-pool signal, not a silent drop; skip it and keep going.
            produced = 0
            for rs in range(30):
                try:
                    out = _regen(
                        rolled.morphemes_by_word, wi, mi, seed=rs, culture="english", tags=[tag]
                    )
                except NoEligibleReplacementError:
                    continue
                # A replacement WAS produced — it MUST carry the tag (the sole
                # carrier is restricted to tagged candidates; never a silent drop).
                assert tag in (out.morphemes_by_word[wi][mi].get("tags") or []), (
                    f"re-roll of the sole {tag}-carrier (seed={rs}) silently dropped the tag"
                )
                # ...and the name as a whole still carries the tag.
                assert _carriers(out.morphemes_by_word, tag)
                produced += 1
            if produced:
                exercised = True
                break
        if exercised:
            break

    assert exercised, (
        "expected a candidate tag to yield a sole-carrier roll whose carrier "
        "re-rolls to another tagged morpheme (none did)"
    )


def test_reroll_of_free_slot_under_tag_samples_freely():
    """wyrd-c6o1.4: re-rolling a NON-sole-carrier slot under --tag is NOT
    restricted to tagged morphemes (the tag is satisfied by another held slot), so
    it samples freely — the inverse of the sole-carrier case. Pre-c6o1.4 the
    pool-wide gate forced EVERY re-roll tagged; this asserts a free slot can yield
    an un-tagged morpheme."""
    # Find a multi-morpheme roll with exactly one water carrier, then re-roll a
    # DIFFERENT (free) slot — the carrier still satisfies the tag, so the free slot
    # is NOT pool-gated to water. The property is that a free slot CAN yield a
    # non-water pick; but a given example's free slot may have a water-ONLY pool
    # (free sampling ≠ guaranteed non-water when the pool is all water), so assert
    # the property holds for SOME example, failing only if EVERY free slot is
    # water-locked (the real gating regression). wyrd-aicu.7: the previously
    # first-found example became a water-only-pool slot after the seed refresh;
    # scanning keeps this a property test, not a seed-pinned one.
    saw_free_sampling = False
    examples = 0
    for s in range(80):
        words = (
            Kenning().generate({"culture": "english", "tags": ["water"]}, seed=s).morphemes_by_word
        )
        flat = [(wi, mi) for wi, w in enumerate(words) for mi, _ in enumerate(w)]
        carriers = _carriers(words, "water")
        if len(flat) >= 2 and len(carriers) == 1:
            examples += 1
            wi, mi = next(slot for slot in flat if slot != carriers[0])
            if any(
                "water"
                not in (
                    _regen(words, wi, mi, seed=rs, culture="english", tags=["water"])
                    .morphemes_by_word[wi][mi]
                    .get("tags")
                    or []
                )
                for rs in range(12)
            ):
                saw_free_sampling = True
                break
    if examples == 0:
        pytest.skip("no multi-morpheme single-water-carrier roll found in range")
    assert saw_free_sampling, (
        "a free slot under --tag must sample freely (not force water) — every free "
        "slot was water-locked across all scanned single-carrier rolls"
    )


def test_mood_theme_survives_when_target_is_the_only_carrier():
    """wyrd-4rp8 context rule: re-rolling the single mood-carrying slot of a
    mood roll restricts the pool to mood-tagged candidates, so the theme
    survives the re-roll. Scans seeds for a roll with exactly one carrier so
    the precondition is structural, not pinned to one seed's bundle
    behavior."""
    mood_tags = set(parse_mood_spec("grim").semantic_tags)
    target = None
    for seed in range(40):
        rolled = Kenning().generate({"culture": "english", "mood": ["grim"]}, seed=seed)
        words = rolled.morphemes_by_word
        carriers = [
            (w, e)
            for w, word in enumerate(words)
            for e, m in enumerate(word)
            if mood_tags & set(m.get("tags", []))
        ]
        if len(carriers) == 1:
            target = (words, carriers[0])
            break
    assert target is not None, "no seed in range produced a single-mood-carrier roll"
    words, (wi, mi) = target
    out = _regen(words, wi, mi, seed=5, culture="english", mood=["grim"])
    assert mood_tags & set(out.morphemes_by_word[wi][mi].get("tags", []))


def test_apply_mood_restriction_falls_back_when_no_candidate_carries_mood():
    """The graceful-degradation half of the mood rule: when NO candidate
    carries a mood tag, the full pool passes through unchanged (the re-roll
    still succeeds, just un-themed)."""

    class _FakeMeaning:
        def __init__(self, tags):
            self.tags = tags

    untagged = [(_FakeMeaning([]), 1.0), (_FakeMeaning(["water"]), 2.0)]
    out = _apply_mood_restriction(untagged, {"death": 0.7}, frozenset())
    assert out == untagged
    # And the restricting half: a mood-tagged candidate wins the pool.
    tagged = _FakeMeaning(["death"])
    mixed = [(_FakeMeaning(["water"]), 1.0), (tagged, 1.0)]
    out = _apply_mood_restriction(mixed, {"death": 0.7}, frozenset())
    assert [m for m, _ in out] == [tagged]
    # No-op when a HELD slot already carries the mood (prior_tags overlap).
    out = _apply_mood_restriction(mixed, {"death": 0.7}, frozenset({"death"}))
    assert out == mixed


def test_derived_position_covers_all_four_slots():
    """D40: position is derived from where the slot lands in its word."""
    assert _derived_position(1, 0) == "bare"
    assert _derived_position(2, 0) == "pre"
    assert _derived_position(2, 1) == "post"
    assert _derived_position(3, 1) == "inner"
    assert _derived_position(3, 2) == "post"


def test_slot_qualifier_name_wins_over_saint_usage():
    """Regression for the round-1 P1: _slot_qualifier must mirror
    Meaning.key's if/elif priority — a name-tagged morpheme routes through
    "name" even when its surface is literally "Saint-"; the saint string
    check only decides for unresolvable / non-name morphemes."""

    class _FakeMeaning:
        def __init__(self, name):
            self._name = name

        def is_name(self):
            return self._name

    assert _slot_qualifier(_FakeMeaning(True), "Saint-") == "name"
    assert _slot_qualifier(_FakeMeaning(False), "Saint-") == "saint"
    assert _slot_qualifier(None, "Saint-") == "saint"
    assert _slot_qualifier(_FakeMeaning(True), "Hill") == "name"
    assert _slot_qualifier(_FakeMeaning(False), "Hill") is None
    assert _slot_qualifier(None, "Hill") is None


def test_era_request_renders_the_replacement_at_era():
    """A replacement under an era request must never carry a foreign era
    render; at least one seed must produce an actual era-rendered
    replacement so the assertion can't pass vacuously."""
    rolled = Kenning().generate({"culture": "english", "era": "old-english"}, seed=11)
    words = rolled.morphemes_by_word
    old_fold = _fold(words[0][0]["usage"])
    saw_rendered = False
    for seed in range(10):
        out = _regen(words, 0, 0, seed=seed, culture="english", era="old-english")
        fresh = out.morphemes_by_word[0][0]
        # Always-asserted invariants: the slot changed, and any render
        # carried is the requested era's (a missing render is the
        # documented ~10% no-reflex modern fallback).
        assert _fold(fresh["usage"]) != old_fold
        if fresh.get("rendered"):
            assert fresh.get("rendered_language") == "old-english"
            saw_rendered = True
    assert saw_rendered, "no seed in range produced an era-rendered replacement"


def test_bad_inputs_raise_value_error(english_roll):
    words = english_roll.morphemes_by_word
    rg = KenningRegenerateMorpheme()
    with pytest.raises(ValueError, match="words is required"):
        rg.generate_all({"words": [], "word_index": 0, "morpheme_index": 0}, seed=1)
    with pytest.raises(ValueError, match="non-empty list of non-empty morpheme lists"):
        rg.generate_all({"words": [[]], "word_index": 0, "morpheme_index": 0}, seed=1)
    with pytest.raises(ValueError, match="out of range"):
        rg.generate_all({"words": words, "word_index": 99, "morpheme_index": 0}, seed=1)
    with pytest.raises(ValueError, match="must be integers"):
        rg.generate_all({"words": words, "word_index": 0, "morpheme_index": "x"}, seed=1)
    with pytest.raises(ValueError, match="non-empty usage"):
        rg.generate_all({"words": [[{"usage": ""}]], "word_index": 0, "morpheme_index": 0}, seed=1)


def test_bounded_era_regeneration_draws_from_the_record(english_roll):
    """D46 on the REGENERATE path: replacing a slot under a bounded era
    must draw only from morphemes in the record by the era's end — the
    Silicon rule applies to rerolls, not just full generation. Fold-aware
    sibling check (D40/D45)."""
    from wyrd.generators.kenning import _load_culture
    from wyrd.generators.kenning.runtime.proportions import _resolve_surface

    name_gen, _ = _load_culture("english")
    db = name_gen.meaning_db
    words = english_roll.morphemes_by_word
    rg = KenningRegenerateMorpheme()
    checked = 0
    for seed in range(25):
        result = rg.generate_all(
            {
                "words": words,
                "word_index": 0,
                "morpheme_index": 0,
                "culture": "english",
                "era": "me",
            },
            seed=seed,
        )[0]
        for word in result.morphemes_by_word:
            for morpheme in word:
                siblings = _resolve_surface(db, morpheme.get("usage"))
                starts = [s.record_start() for s in siblings]
                checked += 1
                assert not starts or any(s is None or s < 1500 for s in starts), (
                    f"seed {seed}: {morpheme.get('usage')!r} in an era=me "
                    f"regeneration but every sibling's record_start >= 1500: {starts}"
                )
    assert checked > 20


def test_era_midpoint_threaded_into_slot_weighted_pool(english_roll, monkeypatch):
    """wyrd-951i: the single-slot re-roll threads the priors-baseline era_midpoint
    into _slot_weighted_pool (the same cell the main path leans toward, wyrd-3tvd)
    — a bounded era → its midpoint; no era / present-day → the wildcard 0. Covers
    BOTH the keyed pool and the bucket-miss fallback. Spies on the kwarg the way
    test_bucket_key_miss does."""
    from wyrd.generators.kenning import _resolve_era_param
    from wyrd.generators.kenning.generators.kenning import _request_era_midpoint

    captured: list[tuple[bool, int | None]] = []  # (is_fallback, era_midpoint)
    real = vector_name_select._slot_weighted_pool
    force_miss = {"on": False}

    def spy(*args, **kwargs):
        is_fallback = kwargs.get("slot_bucket_key") is None
        captured.append((is_fallback, kwargs.get("era_midpoint")))
        # Force the bucket-key miss so the fallback _slot_weighted_pool call fires.
        if force_miss["on"] and not is_fallback:
            return []
        return real(*args, **kwargs)

    monkeypatch.setattr(vector_name_select, "_slot_weighted_pool", spy)
    words = english_roll.morphemes_by_word

    # Bounded era 'me' (Middle English, (1100, 1500)) → midpoint 1300, keyed path.
    expected = _request_era_midpoint(_resolve_era_param("me", "english"))
    assert expected == 1300
    captured.clear()
    _regen(words, 0, 0, seed=7, culture="english", era="me")
    assert captured and all(mp == expected for _, mp in captured)

    # Forced bucket miss → both the keyed call AND the fallback thread era_midpoint.
    captured.clear()
    force_miss["on"] = True
    _regen(words, 0, 0, seed=7, culture="english", era="me")
    force_miss["on"] = False
    assert any(is_fb for is_fb, _ in captured)  # the fallback actually fired
    assert all(mp == expected for _, mp in captured)  # both call sites threaded

    # No era AND the deployed default present-day → 0 (reached via different
    # _request_era_midpoint branches: era_range is None vs end is None).
    for era_params in ({}, {"era": "present-day"}):
        captured.clear()
        _regen(words, 0, 0, seed=7, culture="english", **era_params)
        assert captured and all(mp == 0 for _, mp in captured)


def test_bad_era_raises_value_error(english_roll):
    """The era input must surface a typo as ValueError → the dispatcher's
    400 (D46: the resolved range's END also feeds the record-entry gate;
    the validation contract is unchanged)."""
    words = english_roll.morphemes_by_word
    rg = KenningRegenerateMorpheme()
    with pytest.raises(ValueError, match="invalid 'era'"):
        rg.generate_all(
            {"words": words, "word_index": 0, "morpheme_index": 0, "era": "victorian"},
            seed=1,
        )


def test_registered_and_dispatchable_via_api(english_roll):
    """End-to-end through the Flask dispatcher: the generator is registered,
    multi_result short-circuits count, and ValueError maps to 400."""
    app = create_app()
    words = english_roll.morphemes_by_word
    with app.test_client() as client:
        ok = client.post(
            "/api/kenning-regenerate-morpheme",
            json={
                "words": words,
                "word_index": 0,
                "morpheme_index": len(words[0]) - 1,
                "culture": "english",
                "seed": 7,
            },
        )
        assert ok.status_code == 200
        body = ok.get_json()
        assert body["generator"] == "kenning-regenerate-morpheme"
        assert len(body["results"]) == 1
        assert body["results"][0]["morphemes_by_word"]

        bad = client.post(
            "/api/kenning-regenerate-morpheme",
            json={"words": words, "word_index": 99, "morpheme_index": 0},
        )
        assert bad.status_code == 400
        assert bad.get_json()["error"] == "bad_params"


def test_reroll_bare_word_carries_word_position_bucket(monkeypatch):
    """wyrd-c6o1.2 (D43 / wyrd-rogd.13): re-rolling a BARE word draws from the
    per-WORD-position bare pool — its slot bucket key is extended with wp-pre /
    wp-inner / wp-post by the word's index, mirroring _flatten_struct_slots. Without
    it a re-rolled bare word samples the un-positioned pool and can land at a
    word-position it's never attested in ('Parva Wigston' shapes). A NON-bare slot
    and a SINGLE-word name carry no wp- extension.

    Builds a synthetic multi-word name from real bare morphemes so all three
    positions + the single-word (word_position None) branch are covered
    deterministically (not pinned to a particular roll's layout)."""
    from wyrd.generators.kenning.runtime.word import WORD_POSITION_KEY_PREFIX

    captured: list = []
    real = vector_name_select._slot_weighted_pool

    def capture(*args, **kwargs):
        captured.append(kwargs.get("slot_bucket_key"))
        return real(*args, **kwargs)

    monkeypatch.setattr(vector_name_select, "_slot_weighted_pool", capture)

    # Gather 3 distinct BARE morphemes (single-morpheme words) + one multi-morpheme
    # word from real rolls — trivially available, so no layout-scan flakiness.
    k = Kenning()
    bare: dict[str, dict] = {}
    multi_word = None
    for s in range(60):
        for word in k.generate({"culture": "english"}, seed=s).morphemes_by_word:
            if len(word) == 1:
                bare.setdefault(word[0]["usage"], word[0])
            elif multi_word is None:
                multi_word = word
        if len(bare) >= 3 and multi_word is not None:
            break
    assert len(bare) >= 3, "expected >=3 distinct bare morphemes from the bundle"
    a, b, c = list(bare.values())[:3]

    def keyed_for(words, wi, mi):
        # We assert on the SELECTION bucket key (captured in _slot_weighted_pool,
        # before any pick) — whether the synthetic name then exhausts candidates
        # after the in-use exclusions is irrelevant to the key-derivation logic.
        captured.clear()
        with contextlib.suppress(NoEligibleReplacementError):
            _regen(words, wi, mi, seed=7, culture="english")
        return next((bk for bk in captured if bk is not None), None)

    # A 3-bare-word name → each word's re-roll keys ITS word-position.
    three = [[a], [b], [c]]
    for wi, expected in ((0, "pre"), (1, "inner"), (2, "post")):
        key = keyed_for(three, wi, 0)
        assert key is not None and key[0] == "bare", (wi, key)
        assert key[-1] == f"{WORD_POSITION_KEY_PREFIX}{expected}", (wi, key, expected)

    # Single-word bare name → no word-sequence evidence → NO wp- extension.
    solo = keyed_for([[a]], 0, 0)
    if solo is not None:
        assert not any(str(p).startswith(WORD_POSITION_KEY_PREFIX) for p in solo), solo

    # NON-bare slot (multi-morpheme word) → NO wp- extension (gated on position==bare).
    if multi_word is not None:
        nb = keyed_for([multi_word], 0, len(multi_word) - 1)
        if nb is not None:
            assert not any(str(p).startswith(WORD_POSITION_KEY_PREFIX) for p in nb), nb
