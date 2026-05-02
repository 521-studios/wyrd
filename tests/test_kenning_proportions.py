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
    """With no empirical mass to blend against, the novelty knob has no axis.
    Return a normalized uniform distribution (sum to 1) so the result still
    matches the docstring contract."""
    blended = _blend_uniform([("a", 0), ("b", 0)], 0.5)
    weights = dict(blended)
    assert abs(weights["a"] - 0.5) < 1e-9
    assert abs(weights["b"] - 0.5) < 1e-9
    assert abs(sum(weights.values()) - 1.0) < 1e-9


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


# --- Generator.select novelty plumbing ------------------------------------


def test_generator_select_novelty_zero_takes_fast_path():
    """At novelty=0, Generator.select hits weighted_choice directly with
    the unmodified empirical weights — bit-stable with the pre-D17 path."""
    from wyrd.generators.kenning.proportions import Generator

    rng_a = random.Random(42)
    rng_b = random.Random(42)
    g = Generator(tag_db={}, elements={"a": 90, "b": 10})
    pre_d17 = weighted_choice(rng_a, list(g.elements.items()))
    via_select = g.select(rng_b, novelty=0.0)
    assert pre_d17 == via_select


def test_generator_select_novelty_one_picks_uniformly():
    """At novelty=1, picks should distribute roughly uniformly across keys
    regardless of their original empirical weight. Monte Carlo with a wide
    tolerance band against RNG variance."""
    from collections import Counter

    from wyrd.generators.kenning.proportions import Generator

    g = Generator(tag_db={}, elements={"a": 99, "b": 1})
    counts = Counter(g.select(random.Random(i), novelty=1.0) for i in range(2000))
    assert 800 < counts["a"] < 1200
    assert 800 < counts["b"] < 1200


def test_meaning_generator_select_threads_novelty():
    """MeaningGenerator.select(key, *tags, novelty=...) forwards novelty
    into Generator.select for the chosen bucket."""
    from collections import Counter

    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import MeaningGenerator

    # Two usages keyed by the same Meaning.key() = ('post',). Empirical
    # weights skewed 99:1; with novelty=1 the picks should split ~50:50.
    m_a = Meaning("-a", [], [], {})
    m_b = Meaning("-b", [], [], {})
    meaning_db = {"-a": [m_a], "-b": [m_b]}
    proportions = {"-a": 99, "-b": 1}
    mg = MeaningGenerator(meaning_db, {}, proportions)
    counts = Counter(mg.select(random.Random(i), ("post",), novelty=1.0) for i in range(2000))
    assert 800 < counts["-a"] < 1200
    assert 800 < counts["-b"] < 1200


# --- NameGenerator._render_substitutions + NewName.__str__ rendered branch ------


def _build_minimal_name_generator(meaning_db):
    """Build a NameGenerator with a single trivial structure for tests.
    Uses only one usage from meaning_db so the structure walk is deterministic."""
    from wyrd.generators.kenning.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    proportions = dict.fromkeys(meaning_db, 1)
    mg = MeaningGenerator(meaning_db, {}, proportions)
    structs = {(((next(iter(meaning_db.values()))[0].location,),)): 1}
    return NameGenerator(meaning_db, mg, structs)


def test_render_substitutions_falls_back_to_canonical_when_no_pool():
    """A meaning with no variant or inflection pool renders as the
    dash-stripped usage even at spelling_variety=1.0."""
    from wyrd.generators.kenning.meaning import Meaning

    m = Meaning("-cot", [], [], {"old_english": ["cot"]})
    name_gen = _build_minimal_name_generator({"-cot": [m]})
    rendered, labels = name_gen._render_substitutions(random.Random(0), [["-cot"]], 1.0, 0.0)
    assert rendered == [["cot"]]
    assert labels == [[None]]


def test_render_substitutions_substitutes_variant_with_case_mimic():
    """At spelling_variety=1 with a non-empty pool, the rendered surface
    form is the case-mimicked variant rather than the canonical."""
    from wyrd.generators.kenning.meaning import Meaning

    m = Meaning(
        "Bridg-",
        [],
        [],
        {"old_english": ["brycg"]},
        variants={"old_english": [("brycg", 10)]},
    )
    name_gen = _build_minimal_name_generator({"Bridg-": [m]})
    rendered, labels = name_gen._render_substitutions(random.Random(0), [["Bridg-"]], 1.0, 0.0)
    # Title-case template projects onto the variant.
    assert rendered == [["Brycg"]]
    assert labels == [[None]]


def test_render_substitutions_substitutes_inflection_with_label():
    """At inflection_density=1 with a non-empty inflection pool, the
    rendered form is the inflected child and the label is preserved."""
    from wyrd.generators.kenning.meaning import Meaning

    m = Meaning(
        "-cot",
        [],
        [],
        {"old_english": ["cot", "cotum"]},
        inflections={"old_english": [("cotum", "dative_or_pl")]},
    )
    name_gen = _build_minimal_name_generator({"-cot": [m]})
    rendered, labels = name_gen._render_substitutions(random.Random(0), [["-cot"]], 0.0, 1.0)
    assert rendered == [["cotum"]]
    assert labels == [["dative_or_pl"]]


def test_render_substitutions_inflection_wins_over_variant():
    """When both knobs would fire on the same morpheme, inflection wins
    (more specific morphological data)."""
    from wyrd.generators.kenning.meaning import Meaning

    m = Meaning(
        "-cot",
        [],
        [],
        {"old_english": ["cot"]},
        variants={"old_english": [("cotte", 10)]},
        inflections={"old_english": [("cotum", "dative_or_pl")]},
    )
    name_gen = _build_minimal_name_generator({"-cot": [m]})
    rendered, labels = name_gen._render_substitutions(random.Random(0), [["-cot"]], 1.0, 1.0)
    assert rendered == [["cotum"]]
    assert labels == [["dative_or_pl"]]


def test_render_substitutions_handles_none_usage():
    """Tag-filter passes can leave None entries when no candidate matched the
    structure slot. _render_substitutions must propagate None on both lists."""
    name_gen = _build_minimal_name_generator(
        {
            "-x": [
                __import__("wyrd.generators.kenning.meaning", fromlist=["Meaning"]).Meaning(
                    "-x", [], [], {}
                )
            ]
        }
    )
    rendered, labels = name_gen._render_substitutions(random.Random(0), [[None]], 1.0, 1.0)
    assert rendered == [[None]]
    assert labels == [[None]]


def test_newname_str_uses_rendered_when_set():
    """When NewName.rendered is populated, __str__ emits the pre-rendered
    surface forms instead of stripping dashes from name."""
    from wyrd.generators.kenning.proportions import NewName

    new_name = NewName(
        struct=None,
        meaning_db={},
        name=[["Bridg-", "-water"]],
        rendered=[["Brycg", "wattyr"]],
    )
    assert str(new_name) == "Brycgwattyr"


def test_newname_str_falls_back_to_dash_stripped_when_rendered_none():
    """Without rendered set, __str__ keeps the historic dash-stripping path."""
    from wyrd.generators.kenning.proportions import NewName

    new_name = NewName(
        struct=None,
        meaning_db={},
        name=[["Bridg-", "-water"]],
    )
    assert str(new_name) == "Bridgwater"


def test_newname_str_falls_back_per_element_when_rendered_entry_is_none():
    """rendered is per-element optional; a None entry in the rendered list
    triggers per-element fallback to the dash-stripped usage."""
    from wyrd.generators.kenning.proportions import NewName

    new_name = NewName(
        struct=None,
        meaning_db={},
        name=[["Bridg-", "-water"]],
        rendered=[["Brycg", None]],
    )
    assert str(new_name) == "Brycgwater"
