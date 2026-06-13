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
    exhaustion surfaces as NoEligibleReplacementError (a ValueError
    subclass, same 400 mapping).

Runs against the committed seed-runtime.db (no live lexicon DB — CI has
none)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from wyrd.app import create_app
from wyrd.generators.kenning import _load_culture
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
    loudly with NoEligibleReplacementError (→ the dispatcher's 400), never
    a silent no-op or duplicate."""
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


def test_tag_gate_holds_on_the_replacement():
    rolled = Kenning().generate({"culture": "english", "tags": ["water"]}, seed=42)
    words = rolled.morphemes_by_word
    out = _regen(words, 0, 0, seed=3, culture="english", tags=["water"])
    assert "water" in out.morphemes_by_word[0][0].get("tags", [])


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
