"""wyrd-y0lx: the `kenning-regenerate-morpheme` endpoint — re-roll ONE
morpheme of a generated name while holding the others fixed.

Pins the operator-decided contract (2026-06-11):
  * deterministic per seed (the SPA bakes the seed into the transform step,
    so pipeline re-runs / restored workspaces replay the same pick);
  * the replacement NEVER duplicates the replaced morpheme or any morpheme
    already in use elsewhere in the name (no "Hill Hill" by construction);
  * the same hard gates as a fresh roll apply (the --tag per-lemma OR-gate
    must keep a tag-filtered name tag-satisfying);
  * held slots pass through verbatim;
  * malformed input surfaces as ValueError → the dispatcher's 400.

Runs against the committed seed-runtime.db (no live lexicon DB — CI has
none)."""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.generators import Kenning, KenningRegenerateMorpheme


def _fold(usage: str) -> str:
    return usage.replace("-", "").lower()


@pytest.fixture(scope="module")
def english_roll():
    """One stable english roll to regenerate against (module-scoped — the
    roll itself is not under test here)."""
    result = Kenning().generate({"culture": "english"}, seed=42)
    assert result.morphemes_by_word
    return result


def _regen(words, wi, mi, seed, **params):
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


def test_tag_gate_holds_on_the_replacement():
    rolled = Kenning().generate({"culture": "english", "tags": ["water"]}, seed=42)
    words = rolled.morphemes_by_word
    out = _regen(words, 0, 0, seed=3, culture="english", tags=["water"])
    assert "water" in out.morphemes_by_word[0][0].get("tags", [])


def test_mood_theme_survives_when_target_is_the_only_carrier():
    """wyrd-4rp8 context rule: re-rolling the single mood-carrying slot of a
    mood roll restricts the pool to mood-tagged candidates, so the theme
    survives the re-roll."""
    from wyrd.generators.kenning.registers.effects import parse_mood_spec

    mood_tags = set(parse_mood_spec("grim").semantic_tags)
    rolled = Kenning().generate({"culture": "english", "mood": ["grim"]}, seed=21)
    words = rolled.morphemes_by_word
    carriers = [
        (w, e)
        for w, word in enumerate(words)
        for e, m in enumerate(word)
        if mood_tags & set(m.get("tags", []))
    ]
    assert len(carriers) == 1, "fixture roll should have exactly one mood carrier"
    wi, mi = carriers[0]
    out = _regen(words, wi, mi, seed=5, culture="english", mood=["grim"])
    assert mood_tags & set(out.morphemes_by_word[wi][mi].get("tags", []))


def test_era_request_renders_the_replacement_at_era():
    rolled = Kenning().generate({"culture": "english", "era": "old-english"}, seed=11)
    words = rolled.morphemes_by_word
    out = _regen(words, 0, 0, seed=5, culture="english", era="old-english")
    fresh = out.morphemes_by_word[0][0]
    # The replacement either carries an OE-rendered form, or had no reflex
    # for the era (the documented ~10% modern-fallback) — but when a render
    # IS present it must be the requested era's.
    if fresh.get("rendered"):
        assert fresh.get("rendered_language") == "old-english"


def test_bad_inputs_raise_value_error(english_roll):
    words = english_roll.morphemes_by_word
    rg = KenningRegenerateMorpheme()
    with pytest.raises(ValueError, match="words is required"):
        rg.generate_all({"words": [], "word_index": 0, "morpheme_index": 0}, seed=1)
    with pytest.raises(ValueError, match="out of range"):
        rg.generate_all({"words": words, "word_index": 99, "morpheme_index": 0}, seed=1)
    with pytest.raises(ValueError, match="must be integers"):
        rg.generate_all({"words": words, "word_index": 0, "morpheme_index": "x"}, seed=1)
    with pytest.raises(ValueError, match="non-empty usage"):
        rg.generate_all({"words": [[{"usage": ""}]], "word_index": 0, "morpheme_index": 0}, seed=1)


def test_registered_and_dispatchable_via_api(english_roll):
    """End-to-end through the Flask dispatcher: the generator is registered,
    multi_result short-circuits count, and ValueError maps to 400."""
    from wyrd.app import create_app

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
