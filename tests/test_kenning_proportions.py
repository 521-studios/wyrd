"""Unit tests for kenning's weighted_choice — the core weighted sampler."""

from __future__ import annotations

import os
import random
import textwrap
from collections import Counter

from wyrd.generators.kenning.runtime.proportions import (
    _blend_harsh,
    _blend_uniform,
    _harshness_score,
    weighted_choice,
)


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
    from wyrd.generators.kenning.runtime.proportions import Generator

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

    from wyrd.generators.kenning.runtime.proportions import Generator

    g = Generator(tag_db={}, elements={"a": 99, "b": 1})
    counts = Counter(g.select(random.Random(i), novelty=1.0) for i in range(2000))
    assert 800 < counts["a"] < 1200
    assert 800 < counts["b"] < 1200


def test_meaning_generator_select_threads_novelty():
    """MeaningGenerator.select(key, *tags, novelty=...) forwards novelty
    into Generator.select for the chosen bucket."""
    from collections import Counter

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

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


def test_meaning_generator_select_threads_harshness():
    """MeaningGenerator.select(key, *tags, harshness=...) forwards harshness
    into Generator.select for the chosen bucket. At harshness=1, the empirical
    99:1 weight gets re-weighted by the harsh phonology score, shifting picks
    toward the stop-final key (-shuck) over the soft -baron bucket-mate."""
    from collections import Counter

    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    # Two usages keyed by the same Meaning.key() = ('post',). Empirical
    # weight 99 on the soft form, 1 on the harsh form; harshness=1 should
    # claw back the gap because soft scores ~0 and harsh scores ~0.7+.
    m_baron = Meaning("-baron", [], [], {})
    m_shuck = Meaning("-shuck", [], [], {})
    meaning_db = {"-baron": [m_baron], "-shuck": [m_shuck]}
    proportions = {"-baron": 99, "-shuck": 1}
    mg = MeaningGenerator(meaning_db, {}, proportions)
    plain = Counter(mg.select(random.Random(i), ("post",)) for i in range(2000))
    harsh = Counter(mg.select(random.Random(i), ("post",), harshness=1.0) for i in range(2000))
    # Sanity: plain has -baron almost always.
    assert plain["-baron"] > 1800
    # Harsh: -shuck (stop-final) gets a meaningful share even against the
    # 99:1 empirical headwind. The exact amount depends on the score gap;
    # any non-trivial shift confirms the kwarg threaded through.
    assert harsh["-shuck"] > plain["-shuck"]


# --- NameGenerator._render_substitutions + NewName.__str__ rendered branch ------


def _build_minimal_name_generator(meaning_db):
    """Build a NameGenerator with a single trivial structure for tests.

    Uses only one usage from meaning_db so the structure walk is
    deterministic. Mirrors the production load_proportions() shape: every
    usage is registered both with its bare location key (multi-element
    words) and the (location, "single") key (single-element words). The
    test fixture uses single-element structures so the runtime can resolve
    the bucket via the "single"-suffixed key.
    """
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    proportions = dict.fromkeys(meaning_db, 1)
    mg = MeaningGenerator(meaning_db, {}, proportions)
    mg.load_parts(proportions, "single")
    location = next(iter(meaning_db.values()))[0].location
    structs = {(((location, "single"),),): 1}
    return NameGenerator(meaning_db, mg, structs)


def test_render_substitutions_falls_back_to_canonical_when_no_pool():
    """A meaning with no variant or inflection pool renders as the
    dash-stripped usage even at spelling_variety=1.0."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    m = Meaning("-cot", [], [], {"old_english": ["cot"]})
    name_gen = _build_minimal_name_generator({"-cot": [m]})
    rendered, labels = name_gen._render_substitutions(random.Random(0), [["-cot"]], 1.0, 0.0)
    assert rendered == [["cot"]]
    assert labels == [[None]]


def test_render_substitutions_substitutes_variant_with_case_mimic():
    """At spelling_variety=1 with a non-empty pool, the rendered surface
    form is the case-mimicked variant rather than the canonical."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

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
    from wyrd.generators.kenning.runtime.meaning import Meaning

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
    from wyrd.generators.kenning.runtime.meaning import Meaning

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


def test_select_populates_inflection_labels_at_high_density():
    """End-to-end: NameGenerator.select(inflection_density=1.0) on a
    meaning_db with inflection metadata returns a NewName whose
    inflection_labels carry the picked label per element. Pins the
    integration boundary that _render_substitutions tests can't reach
    on their own."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    m = Meaning(
        "-cot",
        [],
        [],
        {"old_english": ["cot", "cotum"]},
        inflections={"old_english": [("cotum", "dative_or_pl")]},
    )
    name_gen = _build_minimal_name_generator({"-cot": [m]})
    new_name = name_gen.select(random.Random(0), inflection_density=1.0)
    assert new_name.inflection_labels == [["dative_or_pl"]]
    assert new_name.rendered == [["cotum"]]


def test_select_default_skips_render_pass_entirely():
    """At default knobs (variety=0, density=0), select() doesn't populate
    rendered or inflection_labels — they stay None. Cheap fast-path
    confirmation that protects bit-stability."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    m = Meaning(
        "-cot",
        [],
        [],
        {"old_english": ["cot"]},
        variants={"old_english": [("kotte", 5)]},
        inflections={"old_english": [("cotum", "dative_or_pl")]},
    )
    name_gen = _build_minimal_name_generator({"-cot": [m]})
    new_name = name_gen.select(random.Random(0))
    assert new_name.rendered is None
    assert new_name.inflection_labels is None


def test_render_substitutions_handles_none_usage():
    """Tag-filter passes can leave None entries when no candidate matched the
    structure slot. _render_substitutions must propagate None on both lists."""
    name_gen = _build_minimal_name_generator(
        {
            "-x": [
                __import__("wyrd.generators.kenning.runtime.meaning", fromlist=["Meaning"]).Meaning(
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
    from wyrd.generators.kenning.runtime.proportions import NewName

    new_name = NewName(
        struct=None,
        meaning_db={},
        name=[["Bridg-", "-water"]],
        rendered=[["Brycg", "wattyr"]],
    )
    assert str(new_name) == "Brycgwattyr"


def test_newname_str_falls_back_to_dash_stripped_when_rendered_none():
    """Without rendered set, __str__ keeps the historic dash-stripping path."""
    from wyrd.generators.kenning.runtime.proportions import NewName

    new_name = NewName(
        struct=None,
        meaning_db={},
        name=[["Bridg-", "-water"]],
    )
    assert str(new_name) == "Bridgwater"


def test_newname_str_falls_back_per_element_when_rendered_entry_is_none():
    """rendered is per-element optional; a None entry in the rendered list
    triggers per-element fallback to the dash-stripped usage."""
    from wyrd.generators.kenning.runtime.proportions import NewName

    new_name = NewName(
        struct=None,
        meaning_db={},
        name=[["Bridg-", "-water"]],
        rendered=[["Brycg", None]],
    )
    assert str(new_name) == "Brycgwater"


# --- D6 --harsh: phonological harshness skew -----------------------------


def test_harshness_score_stop_finals_score_higher_than_vowel_finals():
    """A stop-final cluster-heavy morpheme scores strictly higher than a
    vowel-final one — the load-bearing relative ordering for the knob."""
    assert _harshness_score("shuck") > _harshness_score("baron")
    assert _harshness_score("crag") > _harshness_score("ley")
    assert _harshness_score("fork") > _harshness_score("borough")


def test_harshness_score_dashes_stripped():
    """Usage keys carry leading/trailing dashes (-ham, Bridg-, -y); the
    scorer ignores them so the score reflects the morpheme itself."""
    assert _harshness_score("-ham") == _harshness_score("ham")
    assert _harshness_score("Bridg-") == _harshness_score("bridg")


def test_harshness_score_in_unit_range():
    """All scores fall within [0, 1]. Caller's reweight assumes this range."""
    for usage in ["a", "i", "ham", "shuck", "crag", "Bridg-", "-water", "Saint", ""]:
        score = _harshness_score(usage)
        assert 0.0 <= score <= 1.0, f"{usage!r} → {score}"


def test_blend_harsh_at_zero_returns_input_weights_unchanged():
    """harshness=0 leaves every weight identical to the input. Bit-stable
    fast-path contract for the --harsh knob."""
    items = [("shuck", 10), ("ham", 5), ("baron", 3)]
    blended = _blend_harsh(items, 0.0)
    assert blended == items


def test_blend_harsh_at_one_zeroes_pure_vowel_keys():
    """harshness=1 with the multiplier 2*score sends pure-vowel ("a") to
    weight 0 because its score is 0. A stop-final key keeps positive weight."""
    items = [("a", 100), ("shuck", 10)]
    blended = _blend_harsh(items, 1.0)
    weights = dict(blended)
    assert weights["a"] == 0.0
    assert weights["shuck"] > 0.0


def test_blend_harsh_empty_input_returns_input():
    assert _blend_harsh([], 0.5) == []


def test_blend_harsh_distribution_skews_toward_harsh_via_monte_carlo():
    """At harshness=1 with empirical weights split between a soft key and a
    harsh key, sampling lands the harsh key far more often than empirical
    alone would predict. Wide tolerance against RNG variance."""
    items = [("shuck", 1), ("baron", 9)]  # baron is 9x in empirical
    rng = random.Random(0)
    plain_counts = Counter(weighted_choice(rng, items) for _ in range(2000))
    rng = random.Random(0)
    harsh_counts = Counter(weighted_choice(rng, _blend_harsh(items, 1.0)) for _ in range(2000))
    # Plain: baron dominates (~90%). Harsh: shuck's score is much higher,
    # which counteracts baron's empirical lead — at harshness=1 shuck
    # should appear noticeably more often than the plain ~10%.
    assert plain_counts["baron"] > plain_counts["shuck"]
    assert harsh_counts["shuck"] > plain_counts["shuck"]


def test_generator_select_harshness_zero_takes_fast_path():
    """At harshness=0, Generator.select hits weighted_choice directly with
    unmodified empirical weights — bit-stable with the pre-D6 path."""
    from wyrd.generators.kenning.runtime.proportions import Generator

    rng_a = random.Random(42)
    rng_b = random.Random(42)
    g = Generator(tag_db={}, elements={"-ham": 90, "-shuck": 10})
    pre_d6 = weighted_choice(rng_a, list(g.elements.items()))
    via_select = g.select(rng_b, harshness=0.0)
    assert pre_d6 == via_select


def test_generator_select_harshness_one_skews_toward_harsh_keys():
    """At harshness=1, monte-carlo over 2000 picks shifts the empirical
    90:10 split (ham:shuck) toward shuck enough to detect."""
    from wyrd.generators.kenning.runtime.proportions import Generator

    g = Generator(tag_db={}, elements={"-ham": 90, "-shuck": 10})
    counts = Counter(g.select(random.Random(i), harshness=1.0) for i in range(2000))
    plain_counts = Counter(g.select(random.Random(i), harshness=0.0) for i in range(2000))
    # Sanity: plain has -ham dominant.
    assert plain_counts["-ham"] > 1500
    # Harsh: -shuck (stop-final) gets a meaningful share.
    assert counts["-shuck"] > plain_counts["-shuck"]


def test_generator_select_composes_harsh_and_novelty():
    """harshness=1 + novelty=1 → uniform over the bucket (novelty wipes
    empirical, including the harsh-skew). Same monte-carlo shape as the
    novelty-alone test."""
    from wyrd.generators.kenning.runtime.proportions import Generator

    g = Generator(tag_db={}, elements={"-ham": 99, "-shuck": 1})
    counts = Counter(g.select(random.Random(i), novelty=1.0, harshness=1.0) for i in range(2000))
    assert 800 < counts["-ham"] < 1200
    assert 800 < counts["-shuck"] < 1200


# --- D8 explainer: <lemma>@<label> surfacing -----------------------------


def test_description_no_inflection_labels_unchanged():
    """At default (inflection_labels=None), description() emits the historic
    `lemma (sources gloss)` form per element. Bit-stable with pre-D8."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m = Meaning("-cot", [], ["cottage"], {"old_english": ["cot"]})
    new_name = NewName(
        struct=None,
        meaning_db={"-cot": [m]},
        name=[["-cot"]],
    )
    assert new_name.description() == "cot (EN cottage)"


def test_description_emits_at_label_when_inflection_picked():
    """When inflection_labels carries a non-None label for an element, the
    explainer surfaces `lemma@label (sources gloss)`."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m = Meaning("-cot", [], ["cottage"], {"old_english": ["cot", "cotum"]})
    new_name = NewName(
        struct=None,
        meaning_db={"-cot": [m]},
        name=[["-cot"]],
        rendered=[["cotum"]],
        inflection_labels=[["dative_or_pl"]],
    )
    assert new_name.description() == "cot@dative_or_pl (EN cottage)"


def test_description_handles_mixed_labels_within_word():
    """A multi-element word where only one position has an inflection label
    surfaces @label only on that position. The other element stays bare."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    bridg = Meaning("Bridg-", [], ["bridge"], {"old_english": ["brycg"]})
    water = Meaning("-water", [], ["water"], {"old_english": ["wæter"]})
    new_name = NewName(
        struct=None,
        meaning_db={"Bridg-": [bridg], "-water": [water]},
        name=[["Bridg-", "-water"]],
        inflection_labels=[[None, "genitive_strong"]],
    )
    # Multi-element words keep dashes in the head per the existing
    # description() shape. wyrd-c0xn: each etymon on its own line
    # (newline-separated) and no inline citations — citations live
    # in components() for the SPA's expandable box.
    assert new_name.description() == "Bridg- (EN bridge)\n-water@genitive_strong (EN water)"


def test_description_inflection_labels_shorter_than_name_does_not_crash():
    """Defensive: if inflection_labels has a shorter inner list than its
    matching name word — possible only via direct construction or a bug
    elsewhere — description() must fall through to the unlabelled form
    rather than raising IndexError. Pins the IndexError catch in
    _inflection_label_for."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m = Meaning("-cot", [], ["cottage"], {"old_english": ["cot"]})
    new_name = NewName(
        struct=None,
        meaning_db={"-cot": [m]},
        name=[["-cot"]],
        # Outer list is correct length but inner is empty — element index 0
        # raises IndexError, which the helper must swallow.
        inflection_labels=[[]],
    )
    assert new_name.description() == "cot (EN cottage)"


# --- wyrd-c0xn: description() is now newline-joined gloss-only. Citations
# stay in components() for the SPA's expandable citation box (see
# test_components_includes_citation_list below). The previous wyrd-9kh.1
# tests that pinned 'cited by ...' INSIDE description() were removed
# along with that codepath; coverage now lives in the components-side
# tests below.


def test_description_omits_citation_block():
    """description() carries gloss only — no inline citations regardless
    of whether the Meaning has them. wyrd-c0xn."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m = Meaning(
        "-cot",
        [],
        ["cottage"],
        {"old_english": ["cot"]},
        citations={"old_english": ["mawer_1920", "skeat_1901"]},
    )
    new_name = NewName(
        struct=None,
        meaning_db={"-cot": [m]},
        name=[["-cot"]],
    )
    assert new_name.description() == "cot (EN cottage)"


def test_components_carries_full_citation_list_unchanged():
    """The components() envelope still surfaces the full citation list
    (not truncated) so the SPA can render attribution per element. The
    description-side truncation that wyrd-9kh.1 introduced was removed
    along with description-side citations entirely (wyrd-c0xn)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    sources = [f"src_{i:02}" for i in range(7)]
    m = Meaning(
        "-ham",
        [],
        ["home"],
        {"old_english": ["ham"]},
        citations={"old_english": sources},
    )
    new_name = NewName(
        struct=None,
        meaning_db={"-ham": [m]},
        name=[["-ham"]],
    )
    components = new_name.components()
    assert components[0]["citations"] == sources


def test_components_includes_citation_list():
    """The API envelope's components carry the same citation list so the
    SPA can render attribution per element."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m = Meaning(
        "-cot",
        [],
        ["cottage"],
        {"old_english": ["cot"]},
        citations={"old_english": ["mawer_1920"]},
    )
    new_name = NewName(
        struct=None,
        meaning_db={"-cot": [m]},
        name=[["-cot"]],
    )
    components = new_name.components()
    assert len(components) == 1
    assert components[0]["citations"] == ["mawer_1920"]


def test_components_citation_is_empty_list_when_no_citations():
    """Components always carry a 'citations' key so consumers don't have to
    branch on its presence — empty list signals 'no scholarly attestation
    yet' (the legacy rando-only case)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m = Meaning("-cot", [], ["cottage"], {"old_english": ["cot"]})
    new_name = NewName(
        struct=None,
        meaning_db={"-cot": [m]},
        name=[["-cot"]],
    )
    components = new_name.components()
    assert components[0]["citations"] == []


def test_components_renderings_is_empty_dict_when_no_phase2d_data():
    """wyrd-qhs0 Phase 2d: components always carry a 'renderings' key
    so the SPA panel can iterate uniformly. Empty dict signals 'no
    wyrd-ha9q rendering data' (Latin-script source langs, older
    bundles). The SPA's _renderProvenancePanel skips when ALL
    components have empty renderings."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m = Meaning("-cot", [], ["cottage"], {"old_english": ["cot"]})
    new_name = NewName(
        struct=None,
        meaning_db={"-cot": [m]},
        name=[["-cot"]],
    )
    components = new_name.components()
    assert components[0]["renderings"] == {}


def test_components_renderings_aggregates_four_phase2d_columns():
    """A Meaning with Phase 2d data on all four columns surfaces them
    in components() under the renderings dict, keyed by lang_field
    then canonical_form. Each form gets a slot dict with whichever
    of (original_script, transliteration, english_shaped, ipa,
    dialect) the lexicon supplied. Pinned because this is the wire
    contract the SPA's _renderProvenancePanel reads."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m = Meaning(
        "-golem",
        [],
        ["golem"],
        {"hebrew": ["גולם"]},
        original_script={"hebrew": {"גולם": "גוֹלֶם"}},
        transliteration={"hebrew": {"גולם": "gōlem"}},
        english_shaped={"hebrew": {"גולם": "golem"}},
        pronunciation={"hebrew": {"גולם": {"ipa": "/ɡoːlɛm/", "dialect": "Modern-Hebrew"}}},
    )
    new_name = NewName(
        struct=None,
        meaning_db={"-golem": [m]},
        name=[["-golem"]],
    )
    components = new_name.components()
    assert components[0]["renderings"] == {
        "hebrew": {
            "גולם": {
                "original_script": "גוֹלֶם",
                "transliteration": "gōlem",
                "english_shaped": "golem",
                "ipa": "/ɡoːlɛm/",
                "dialect": "Modern-Hebrew",
            }
        }
    }


def test_components_renderings_partial_data_only_surfaces_present_keys():
    """If a Meaning carries only SOME of the four rendering columns
    (e.g. transliteration + english_shaped but no IPA / native script),
    the slot dict has only the keys that actually exist. Consumers
    rely on this: the SPA panel iterates known keys with .get() and
    skips missing ones rather than rendering 'None'."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m = Meaning(
        "-jinn",
        [],
        ["spirit"],
        {"arabic": ["جن"]},
        transliteration={"arabic": {"جن": "ǧinn"}},
        english_shaped={"arabic": {"جن": "jinn"}},
    )
    new_name = NewName(
        struct=None,
        meaning_db={"-jinn": [m]},
        name=[["-jinn"]],
    )
    slot = new_name.components()[0]["renderings"]["arabic"]["جن"]
    assert slot == {"transliteration": "ǧinn", "english_shaped": "jinn"}
    # Absent keys ARE absent (not None-valued) so the SPA's truthy
    # check on `value` correctly skips them.
    assert "original_script" not in slot
    assert "ipa" not in slot
    assert "dialect" not in slot


# --- wyrd-yan: fiction-tag exclusion gate -------------------------------


def test_generator_select_exclude_tags_drops_matching_keys():
    """Generator.select with exclude_tags=('fiction',) removes any usage
    that the tag_db lists under 'fiction'. Picks must come from the
    remaining keys only — pin via Monte Carlo with a tag_db that marks
    one key as fiction."""
    from wyrd.generators.kenning.runtime.proportions import Generator

    # tag_db reverse-index: key 'fake-' is fiction, 'real-' is not.
    g = Generator(
        tag_db={"fiction": ["fake-"]},
        elements={"real-": 50, "fake-": 50},
    )
    picks = {g.select(random.Random(i), exclude_tags=("fiction",)) for i in range(50)}
    assert picks == {"real-"}


def test_generator_select_default_exclude_tags_is_noop():
    """Default exclude_tags=() must not change behavior — bit-stable with
    the pre-wyrd-yan path. A fiction-tagged key still draws when no
    exclusion is requested."""
    from wyrd.generators.kenning.runtime.proportions import Generator

    g = Generator(
        tag_db={"fiction": ["fake-"]},
        elements={"real-": 1, "fake-": 99},
    )
    counts = Counter(g.select(random.Random(i)) for i in range(500))
    # 99:1 weight → fake- should dominate. The exact ratio doesn't
    # matter; we just need to confirm fake- WAS picked, proving exclude
    # didn't fire by default.
    assert counts["fake-"] > counts["real-"]


def test_generator_select_exclude_composes_with_positive_tag_filter():
    """exclude_tags applies AFTER the positive tag include-filter, so a
    usage tagged BOTH 'tree' and 'fiction' is dropped from a --tag tree
    selection. Pin so a refactor can't silently invert the order."""
    from wyrd.generators.kenning.runtime.proportions import Generator

    g = Generator(
        tag_db={
            "tree": ["oak-", "fake-tree-"],
            "fiction": ["fake-tree-"],
        },
        elements={"oak-": 50, "fake-tree-": 50, "irrelevant-": 100},
    )
    picks = {g.select(random.Random(i), "tree", exclude_tags=("fiction",)) for i in range(50)}
    # 'tree' filter narrows to {oak-, fake-tree-}; fiction exclude drops
    # fake-tree-; only oak- remains.
    assert picks == {"oak-"}


def test_generator_select_exclude_returns_none_when_pool_empties():
    """If the exclude set covers every available key, Generator.select
    returns None — no infinite loop, no IndexError, just a clean
    'nothing to pick'."""
    from wyrd.generators.kenning.runtime.proportions import Generator

    g = Generator(
        tag_db={"fiction": ["only-key-"]},
        elements={"only-key-": 1},
    )
    assert g.select(random.Random(0), exclude_tags=("fiction",)) is None


def test_meaning_generator_select_threads_exclude_tags():
    """MeaningGenerator.select forwards exclude_tags into Generator.select
    so the gate works end-to-end through the bucket dispatch."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    m_real = Meaning("-real", [], [], {})
    m_fake = Meaning("-fake", [], [], {})
    meaning_db = {"-real": [m_real], "-fake": [m_fake]}
    proportions = {"-real": 1, "-fake": 99}
    tag_db = {"fiction": ["-fake"]}
    mg = MeaningGenerator(meaning_db, tag_db, proportions)
    picks = {mg.select(random.Random(i), ("post",), exclude_tags=("fiction",)) for i in range(50)}
    assert picks == {"-real"}


def test_name_generator_select_excludes_fiction_end_to_end():
    """End-to-end through NameGenerator: a synthetic single-morpheme
    culture where the only available usage is fiction-tagged returns
    no morpheme under exclude_tags=('fiction',) and the morpheme under
    exclude_tags=()."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator, NameGenerator

    m = Meaning("-mythron", [], ["constructed"], {"old_english": ["mythron"]})
    meaning_db = {"-mythron": [m]}
    tag_db = {"fiction": ["-mythron"]}
    proportions = {"-mythron": 1}
    mg = MeaningGenerator(meaning_db, tag_db, proportions)
    # Same shape as production load_proportions: single-element words
    # register usages under (location, "single").
    mg.load_parts(proportions, "single")
    structs = {(((m.location, "single"),),): 1}
    name_gen = NameGenerator(meaning_db, mg, structs)

    # Default mode: fiction excluded → no morpheme in the slot.
    new_name = name_gen.select(random.Random(0), exclude_tags=("fiction",))
    assert new_name.name == [[None]]

    # Include-fiction mode: morpheme draws cleanly.
    new_name = name_gen.select(random.Random(0), exclude_tags=())
    assert new_name.name == [["-mythron"]]


def test_kenning_input_schema_exposes_include_fiction():
    """The wyrd-yan flag must be visible in input_schema so the SPA and
    API consumers can render it. Default=False keeps realistic mode the
    out-of-box behavior."""
    from wyrd.generators.kenning import Kenning

    schema = Kenning().input_schema()
    assert "include_fiction" in schema["properties"]
    assert schema["properties"]["include_fiction"]["type"] == "boolean"
    assert schema["properties"]["include_fiction"]["default"] is False


def test_available_tags_hides_fiction_from_dropdown():
    """'fiction' is a metadata marker (opted into via include_fiction),
    NOT a positive selection — it must not surface in the SPA tag
    dropdown via available_tags(). Pin so a refactor of _INTERNAL_TAGS
    can't accidentally re-expose it."""
    from wyrd.generators.kenning import _INTERNAL_TAGS, available_tags

    assert "fiction" in _INTERNAL_TAGS
    assert "fiction" not in available_tags()


# --- D5-2 / wyrd-lyp era filter plumbing ----------------------------------


def test_generator_select_keep_keys_drops_excluded_usages():
    """Generator.select with keep_keys={'a'} should never return 'b' even
    though 'b' has positive empirical weight. Pins the bucket-level filter
    that MeaningGenerator uses to apply the era keep-set."""
    from wyrd.generators.kenning.runtime.proportions import Generator

    g = Generator(tag_db={}, elements={"a": 50, "b": 50})
    for i in range(50):
        assert g.select(random.Random(i), keep_keys=frozenset({"a"})) == "a"


def test_generator_select_keep_keys_none_disables_filter():
    """keep_keys=None is the bit-stable 'no filter' signal — both keys
    remain reachable. Pin so a future refactor can't accidentally
    treat None as an empty set."""
    from collections import Counter

    from wyrd.generators.kenning.runtime.proportions import Generator

    g = Generator(tag_db={}, elements={"a": 50, "b": 50})
    counts = Counter(g.select(random.Random(i), keep_keys=None) for i in range(200))
    assert counts["a"] > 0 and counts["b"] > 0


def test_generator_select_keep_keys_empty_set_returns_none():
    """An empty keep_keys is the legitimate 'no usage matches the era'
    signal — Generator.select should return None rather than crash on
    an empty items_list."""
    from wyrd.generators.kenning.runtime.proportions import Generator

    g = Generator(tag_db={}, elements={"a": 50, "b": 50})
    assert g.select(random.Random(0), keep_keys=frozenset()) is None


def test_meaning_generator_keep_keys_for_era_returns_none_for_none_range():
    """MeaningGenerator.keep_keys_for_era(None) returns None (the
    'no filter' signal) — the runtime threads this straight through to
    Generator.select."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    m = Meaning("-cot", [], [], {})
    mg = MeaningGenerator({"-cot": [m]}, {}, {"-cot": 1})
    assert mg.keep_keys_for_era(None) is None


def test_meaning_generator_keep_keys_for_era_filters_by_attestation():
    """A morpheme with year evidence outside the window is excluded;
    one with evidence inside the window is admitted; one with NO
    evidence passes through (the documented 'pass-through' rule)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    in_window = Meaning("-in", [], [], {}, attested_years={"old_english": [("in", 950)]})
    out_window = Meaning("-out", [], [], {}, attested_years={"old_english": [("out", 1500)]})
    no_data = Meaning("-none", [], [], {})
    meaning_db = {"-in": [in_window], "-out": [out_window], "-none": [no_data]}
    mg = MeaningGenerator(meaning_db, {}, dict.fromkeys(meaning_db, 1))
    keep = mg.keep_keys_for_era((800, 1100))
    assert keep == frozenset({"-in", "-none"})


def test_meaning_generator_keep_keys_for_era_full_coverage_returns_none():
    """When the computed keep-set covers EVERY usage in meaning_db,
    keep_keys_for_era collapses to None — the bit-stable no-filter
    signal for Generator.select. Steady-state today: zero usages
    carry attested_years data, so every era_range is full-coverage
    and the runtime takes the fast path. Pinned because two reviewers
    converged on the un-pinned-branch concern. wyrd-lyp."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    # Mix: one in-era morpheme with attestation, one with NO data.
    # Both pass the (800, 1100) window — the no-data one passes by the
    # 'no attestation → pass through' rule. So coverage is total → None.
    m_in = Meaning("-in", [], [], {}, attested_years={"old_english": [("in", 950)]})
    m_no_data = Meaning("-no-data", [], [], {})
    mg = MeaningGenerator({"-in": [m_in], "-no-data": [m_no_data]}, {}, {"-in": 1, "-no-data": 1})
    assert mg.keep_keys_for_era((800, 1100)) is None


def test_meaning_generator_keep_keys_for_era_caches_per_range():
    """Two calls with the same era_range share the precomputed set so
    the meaning_db isn't re-walked per bucket. Pin via identity (the
    cache returns the same frozenset object) so a perf regression
    that recomputes per call is caught."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    m = Meaning("-cot", [], [], {}, attested_years={"old_english": [("cot", 950)]})
    mg = MeaningGenerator({"-cot": [m]}, {}, {"-cot": 1})
    a = mg.keep_keys_for_era((800, 1100))
    b = mg.keep_keys_for_era((800, 1100))
    assert a is b


def test_meaning_generator_select_threads_keep_keys():
    """MeaningGenerator.select forwards keep_keys into Generator.select
    so an out-of-era usage never wins, even when its empirical weight
    dominates."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    m_ok = Meaning("-ok", [], [], {}, attested_years={"old_english": [("ok", 950)]})
    m_too_late = Meaning("-too-late", [], [], {}, attested_years={"old_english": [("late", 1500)]})
    meaning_db = {"-ok": [m_ok], "-too-late": [m_too_late]}
    proportions = {"-ok": 1, "-too-late": 99}
    mg = MeaningGenerator(meaning_db, {}, proportions)
    keep = mg.keep_keys_for_era((800, 1100))
    for i in range(100):
        assert mg.select(random.Random(i), ("post",), keep_keys=keep) == "-ok"


def test_name_generator_select_drops_out_of_era_morphemes_at_pick_time():
    """End-to-end: NameGenerator.select(era_range=...) constrains the
    sampled morpheme to the era's keep-set. The single-element bucket
    has both an in-era and an out-of-era usage; with empirical weights
    that favor the out-of-era one, the era_range argument should still
    force every pick onto the in-era usage."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    m_in = Meaning("-in", [], [], {}, attested_years={"old_english": [("in", 950)]})
    m_out = Meaning("-out", [], [], {}, attested_years={"old_english": [("out", 1500)]})
    meaning_db = {"-in": [m_in], "-out": [m_out]}
    proportions = {"-in": 1, "-out": 99}
    mg = MeaningGenerator(meaning_db, {}, proportions)
    mg.load_parts(proportions, "single")
    structs = {(((m_in.location, "single"),),): 1}
    name_gen = NameGenerator(meaning_db, mg, structs)
    for i in range(50):
        new_name = name_gen.select(random.Random(i), era_range=(800, 1100))
        assert new_name.name == [["-in"]]


def test_name_generator_select_era_range_threads_through_positive_tag_path():
    """`_select_tag` and `_select_tags` (positive-tag branches) must
    forward keep_keys into MeaningGenerator.select the same way
    `_select_no_tag` does. Two usages both tagged 'tree', one in-era and
    one out-of-era with the out-of-era usage carrying 99x the empirical
    weight: with `tags=('tree',)` AND era_range set, every pick must
    still land on the in-era usage. Pins the `_select_tag`/`_select_tags`
    branches that `test_..._drops_out_of_era_morphemes_at_pick_time`
    above doesn't reach. wyrd-lyp."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    m_in = Meaning("-in", ["tree"], [], {}, attested_years={"old_english": [("in", 950)]})
    m_out = Meaning("-out", ["tree"], [], {}, attested_years={"old_english": [("out", 1500)]})
    meaning_db = {"-in": [m_in], "-out": [m_out]}
    proportions = {"-in": 1, "-out": 99}
    tag_db = {"tree": ["-in", "-out"]}
    mg = MeaningGenerator(meaning_db, tag_db, proportions)
    mg.load_parts(proportions, "single")
    structs = {(((m_in.location, "single"),),): 1}
    name_gen = NameGenerator(meaning_db, mg, structs)
    for i in range(50):
        new_name = name_gen.select(random.Random(i), "tree", era_range=(800, 1100))
        assert new_name.name == [["-in"]]


def test_name_generator_select_era_range_none_is_bit_stable():
    """era_range=None is the runtime's 'no --era passed' signal — the
    keep-set is None and Generator.select takes its bit-stable fast
    path. Pin by comparing seeded picks with vs without era_range=None
    to the historic no-kwarg call."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    m_a = Meaning("-a", [], [], {})
    m_b = Meaning("-b", [], [], {})
    meaning_db = {"-a": [m_a], "-b": [m_b]}
    proportions = {"-a": 50, "-b": 50}
    mg = MeaningGenerator(meaning_db, {}, proportions)
    mg.load_parts(proportions, "single")
    structs = {((("post", "single"),),): 1}
    name_gen = NameGenerator(meaning_db, mg, structs)
    seq_default = [name_gen.select(random.Random(i)).name for i in range(20)]
    seq_explicit_none = [name_gen.select(random.Random(i), era_range=None).name for i in range(20)]
    assert seq_default == seq_explicit_none


def test_name_generator_select_no_era_matches_pre_pr_weighted_choice():
    """Stronger bit-stability pin: a sample drawn through the new
    NameGenerator.select code path matches what raw weighted_choice
    would have produced over the same items pre-PR. The default
    era_range=None test above only proves the default kwarg matches
    the explicit kwarg — both go through the new code path. This test
    locks the actual sequence against the pre-PR sampler so a regression
    that, say, sorted items differently before passing to weighted_choice
    would surface as a name-sequence mismatch.

    Mirrors `test_generator_select_novelty_zero_takes_fast_path` for the
    NameGenerator-level bit-stability claim. Note that
    `NameGenerator.select` consumes one rng draw to pick the structure
    before reaching the bucket; the comparison RNG is advanced the same
    way to stay aligned with the seeded picks."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
        weighted_choice,
    )

    # Skewed weights so a regression-induced order swap surfaces as a
    # sequence change rather than a 50/50 noise wash.
    m_a = Meaning("-a", [], [], {})
    m_b = Meaning("-b", [], [], {})
    meaning_db = {"-a": [m_a], "-b": [m_b]}
    proportions = {"-a": 30, "-b": 70}
    mg = MeaningGenerator(meaning_db, {}, proportions)
    mg.load_parts(proportions, "single")
    structs = {((("post", "single"),),): 1}
    name_gen = NameGenerator(meaning_db, mg, structs)

    bucket = mg.generators[("post", "single")]
    items = list(bucket.elements.items())
    struct_items = list(structs.items())

    seq_via_select = [name_gen.select(random.Random(i)).name[0][0] for i in range(20)]
    seq_via_weighted = []
    for i in range(20):
        rng = random.Random(i)
        # Replay the structure pick that NameGenerator.select does first,
        # so the rng state is aligned with what reaches the bucket.
        weighted_choice(rng, struct_items)
        seq_via_weighted.append(weighted_choice(rng, items))
    assert seq_via_select == seq_via_weighted


def test_kenning_generate_accepts_era_param():
    """Smoke test: Kenning.generate({'era': 'me'}) doesn't raise and
    produces a name. Bundled meanings.json has no attested_years data
    today, so the filter is a documented no-op — but the wiring
    needs to be in place for the next bundle re-emit."""
    from wyrd.generators.kenning import Kenning

    result = Kenning().generate({"culture": "english", "era": "me"}, seed=42)
    assert result.result
    assert result.explanation


def test_kenning_input_schema_exposes_era():
    """The era param surfaces in input_schema so the SPA renders an
    input control for it. Default empty string maps to 'no filter' in
    _resolve_era_param."""
    from wyrd.generators.kenning import Kenning

    schema = Kenning().input_schema()
    assert "era" in schema["properties"]
    assert schema["properties"]["era"]["type"] == "string"
    assert schema["properties"]["era"]["default"] == ""


def test_kenning_input_schema_era_carries_dependent_select_options():
    """wyrd-awo: the era field carries an x-options-by-culture map so
    the SPA renders a culture-keyed dependent select. Each culture's
    options must include the empty 'no filter' string and the cell
    labels of its era family. Pinning the structure prevents a
    regression that drops the metadata or breaks the per-culture key
    set."""
    from wyrd.generators.kenning import CULTURES, Kenning

    schema = Kenning().input_schema()
    options = schema["properties"]["era"]["x-options-by-culture"]
    assert set(options.keys()) == set(CULTURES)
    for culture, labels in options.items():
        assert labels[0] == "", f"{culture}: empty 'no filter' option must come first"
        assert len(labels) > 1, f"{culture}: must have at least one cell label"


def test_kenning_input_schema_era_options_match_era_family_per_culture():
    """The per-culture option set must equal the culture's era family
    cells (plus the empty 'no filter' option). Drives the source-of-truth
    invariant: adding a new cell to era.ERA_CELLS automatically surfaces
    in the SPA dropdown without touching the input_schema."""
    from wyrd.generators.kenning import _CULTURE_TO_ERA_FAMILY, Kenning
    from wyrd.generators.kenning.era.cells import era_cells_for_family

    schema = Kenning().input_schema()
    options = schema["properties"]["era"]["x-options-by-culture"]
    for culture, family in _CULTURE_TO_ERA_FAMILY.items():
        expected = ("", *era_cells_for_family(family))
        assert tuple(options[culture]) == expected, (
            f"{culture} options must equal era_cells_for_family({family!r}) plus empty"
        )


def test_resolve_era_param_treats_empty_string_as_no_filter():
    """The SPA now actively ships era="" when the user picks the
    'no filter' dropdown option (vs. previously omitting the key). Pin
    the resolver's empty-string handling so that path remains a no-op
    rather than ever raising or being interpreted as a literal year."""
    from wyrd.generators.kenning import _resolve_era_param

    assert _resolve_era_param("", "english") is None
    assert _resolve_era_param(None, "english") is None


# --- wyrd-j3gy stratum-param validator + culture gating -------------------


def test_resolve_stratum_param_treats_empty_string_as_no_filter():
    """Empty string and None both signal 'no filter' — same shape
    as _resolve_era_param. Pinned because the SPA actively ships
    stratum="" when the user picks the 'no filter' dropdown option."""
    from wyrd.generators.kenning import _resolve_stratum_param

    assert _resolve_stratum_param("", "english") is None
    assert _resolve_stratum_param(None, "english") is None
    # Same behavior across cultures.
    assert _resolve_stratum_param("", "welsh") is None
    assert _resolve_stratum_param("", "irish") is None


def test_resolve_stratum_param_returns_valid_stratum_for_culture():
    """A stratum that's in the culture's allowed-set passes through
    unchanged."""
    from wyrd.generators.kenning import _resolve_stratum_param

    # English culture spans all four classified families' strata.
    assert _resolve_stratum_param("native-welsh", "english") == "native-welsh"
    assert _resolve_stratum_param("frankish-substrate", "english") == "frankish-substrate"
    assert _resolve_stratum_param("east-norse", "english") == "east-norse"
    # Welsh culture: WELSH_STRATA + FRENCH_STRATA only.
    assert _resolve_stratum_param("brittonic-substrate", "welsh") == "brittonic-substrate"
    assert _resolve_stratum_param("medieval-french", "welsh") == "medieval-french"


def test_resolve_stratum_param_rejects_culturally_incoherent_value():
    """A stratum that's in ALL_STRATA but NOT in the culture's
    allowed-set raises ValueError naming the culture and listing
    valid options. Pin the cultural-incoherence catch — the
    primary value-add of wyrd-j3gy beyond simple typo protection."""
    import pytest

    from wyrd.generators.kenning import _resolve_stratum_param

    # east-norse is a real stratum (in OLD_NORSE_STRATA) but not
    # in welsh culture's allowed-set (welsh = WELSH + FRENCH only).
    with pytest.raises(ValueError, match="welsh"):
        _resolve_stratum_param("east-norse", "welsh")
    # The error mentions valid options.
    with pytest.raises(ValueError, match="brittonic-substrate"):
        _resolve_stratum_param("east-norse", "welsh")


def test_resolve_stratum_param_rejects_typo_in_known_culture():
    """A typo'd stratum (not in ALL_STRATA at all) on a culture
    with a per-culture restriction surfaces with the same culture-
    specific error path."""
    import pytest

    from wyrd.generators.kenning import _resolve_stratum_param

    with pytest.raises(ValueError, match="native-welch"):
        _resolve_stratum_param("native-welch", "welsh")  # typo


def test_resolve_stratum_param_falls_back_to_all_strata_for_unrestricted_culture():
    """Cultures with no per-culture restriction (irish / breton —
    no classifier yet) fall back to ALL_STRATA typo-check. Valid
    cross-family strata pass through; typos still raise."""
    import pytest

    from wyrd.generators.kenning import _resolve_stratum_param

    # 'native-welsh' is in ALL_STRATA — passes through for irish.
    assert _resolve_stratum_param("native-welsh", "irish") == "native-welsh"
    # 'frankish-substrate' — same.
    assert _resolve_stratum_param("frankish-substrate", "breton") == "frankish-substrate"
    # Typo — caught by ALL_STRATA fallback.
    with pytest.raises(ValueError, match="lattin-loan"):
        _resolve_stratum_param("lattin-loan", "irish")


def test_kenning_generate_propagates_resolve_stratum_error():
    """End-to-end: bad --stratum surfaces from Kenning.generate as
    a ValueError. The CLI's existing ValueError handler converts it
    to a friendly stderr message + non-zero exit."""
    import pytest

    from wyrd.generators.kenning import Kenning

    k = Kenning()
    with pytest.raises(ValueError, match="stratum"):
        k.generate(
            {"culture": "welsh", "stratum": "east-norse"},  # culturally incoherent
            seed=42,
        )


def test_kenning_input_schema_stratum_carries_x_options_by_culture():
    """The SPA dependent-select needs ``x-options-by-culture`` on the
    stratum schema entry, mirroring era. Pin its presence + shape so
    a refactor that drops the metadata silently breaks the SPA UX."""
    from wyrd.generators.kenning import CULTURES, Kenning

    schema = Kenning().input_schema()
    stratum_def = schema["properties"]["stratum"]
    assert "x-options-by-culture" in stratum_def
    options = stratum_def["x-options-by-culture"]
    # Every CULTURE has an entry.
    for culture in CULTURES:
        assert culture in options
        # Empty string is always first ('no filter' option).
        assert options[culture][0] == ""


def test_resolve_stratum_param_unknown_culture_falls_back_to_all_strata():
    """A culture that's not in ``_CULTURE_TO_VALID_STRATA`` at all
    (vs configured-but-empty like irish/breton) gets the ALL_STRATA
    typo-check fallback via dict.get's frozenset() default. Both
    sub-cases of the empty-set branch are now pinned: configured-
    empty + missing-entirely → same fallback."""
    import pytest

    from wyrd.generators.kenning import _resolve_stratum_param

    # Bogus culture — falls through to ALL_STRATA typo-check.
    assert _resolve_stratum_param("native-welsh", "klingon") == "native-welsh"
    # And typos still raise on this fallback path.
    with pytest.raises(ValueError, match="lattin-loan"):
        _resolve_stratum_param("lattin-loan", "klingon")


def test_kenning_input_schema_stratum_default_round_trips_to_no_filter():
    """The schema default is the empty string ('no filter' option in
    the SPA dropdown). Pin that feeding the default through
    ``_resolve_stratum_param`` returns None — the runtime 'no filter'
    sentinel. A future refactor that changes the default to None or
    a non-empty value would silently break the SPA's no-filter path
    without this test."""
    from wyrd.generators.kenning import Kenning, _resolve_stratum_param

    schema = Kenning().input_schema()
    default = schema["properties"]["stratum"]["default"]
    assert default == ""
    # SPA roundtrip — the default value must validate to None.
    assert _resolve_stratum_param(default, "english") is None
    assert _resolve_stratum_param(default, "irish") is None  # unrestricted culture


def test_kenning_input_schema_stratum_options_match_per_culture_allowed_set():
    """The schema's per-culture stratum lists must match what
    _resolve_stratum_param accepts — so picking a value from the
    SPA dropdown can't surface as a 4xx at submit time. Pin a
    representative culture (welsh, narrow set) + a representative
    fallback culture (irish, empty set → just the no-filter option)."""
    from wyrd.generators.kenning import (
        FRENCH_STRATA,
        WELSH_STRATA,
        Kenning,
    )

    schema = Kenning().input_schema()
    options = schema["properties"]["stratum"]["x-options-by-culture"]

    # Welsh allows WELSH + FRENCH strata; sorted by the schema helper.
    welsh_expected = [""] + sorted(set(WELSH_STRATA + FRENCH_STRATA))
    assert options["welsh"] == welsh_expected

    # Irish has no per-culture restriction → just 'no filter'.
    assert options["irish"] == [""]


# --- wyrd-lr4 Phase 3 stratum filter --------------------------------------


def test_meaning_generator_keep_keys_for_stratum_returns_none_for_none_arg():
    """MeaningGenerator.keep_keys_for_stratum(None) returns None — the
    'no filter' bit-stable signal, mirrors keep_keys_for_era(None)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    m = Meaning("-cot", [], [], {})
    mg = MeaningGenerator({"-cot": [m]}, {}, {"-cot": 1})
    assert mg.keep_keys_for_stratum(None) is None


def test_meaning_generator_keep_keys_for_stratum_filters_by_classified_data():
    """A morpheme with stratum data NOT containing the target tag is
    excluded; one with matching data is admitted; one with NO stratum
    data passes through (the documented Phase 3 'pass-through' rule —
    Welsh is the only family classified today, so unclassified
    languages must not be silently dropped)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    in_stratum = Meaning("-in", [], [], {}, stratum={"celtic_mix": {"caer": "native-welsh"}})
    other_stratum = Meaning("-out", [], [], {}, stratum={"celtic_mix": {"caer": "latin-loan"}})
    no_data = Meaning("-none", [], [], {})
    meaning_db = {"-in": [in_stratum], "-out": [other_stratum], "-none": [no_data]}
    mg = MeaningGenerator(meaning_db, {}, dict.fromkeys(meaning_db, 1))
    keep = mg.keep_keys_for_stratum("native-welsh")
    assert keep == frozenset({"-in", "-none"})


def test_meaning_generator_keep_keys_for_stratum_full_coverage_returns_none():
    """When every usage admits (because every Meaning has either no
    stratum data or matching data), the keep-set covers the whole
    meaning_db and collapses to None — Generator.select takes its
    bit-stable fast path. Steady-state today: zero usages carry
    stratum data outside the welsh-family slice, so most --stratum
    values are full-coverage."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    m_match = Meaning("-match", [], [], {}, stratum={"celtic_mix": {"caer": "native-welsh"}})
    m_no_data = Meaning("-no-data", [], [], {})
    mg = MeaningGenerator(
        {"-match": [m_match], "-no-data": [m_no_data]},
        {},
        {"-match": 1, "-no-data": 1},
    )
    assert mg.keep_keys_for_stratum("native-welsh") is None


def test_meaning_generator_keep_keys_for_stratum_caches_per_tag():
    """Two calls with the same stratum tag share the precomputed set
    (identity check) so the meaning_db isn't re-walked per bucket.

    Pin via identity on a NON-None result — needs at least one
    non-matching Meaning so the filter doesn't collapse to the
    full-coverage None fast path (where ``None is None`` would pass
    trivially regardless of caching). A perf regression that recomputes
    per call would reconstruct a fresh frozenset each time and break
    the identity check."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    m_in = Meaning("-in", [], [], {}, stratum={"celtic_mix": {"caer": "native-welsh"}})
    m_out = Meaning("-out", [], [], {}, stratum={"celtic_mix": {"din": "latin-loan"}})
    mg = MeaningGenerator({"-in": [m_in], "-out": [m_out]}, {}, {"-in": 1, "-out": 1})
    a = mg.keep_keys_for_stratum("native-welsh")
    b = mg.keep_keys_for_stratum("native-welsh")
    assert a is not None  # if this collapses to None the identity test is meaningless
    assert a is b


def test_name_generator_select_drops_out_of_stratum_morphemes_at_pick_time():
    """End-to-end: NameGenerator.select(stratum=...) constrains the
    sampled morpheme to the stratum's keep-set. Both candidates carry
    stratum data so neither benefits from the no-data passthrough; the
    one with mismatching tag is filtered out even when its empirical
    weight dominates."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    m_in = Meaning("-in", [], [], {}, stratum={"celtic_mix": {"caer": "native-welsh"}})
    m_out = Meaning("-out", [], [], {}, stratum={"celtic_mix": {"din": "latin-loan"}})
    meaning_db = {"-in": [m_in], "-out": [m_out]}
    proportions = {"-in": 1, "-out": 99}
    mg = MeaningGenerator(meaning_db, {}, proportions)
    mg.load_parts(proportions, "single")
    structs = {(((m_in.location, "single"),),): 1}
    name_gen = NameGenerator(meaning_db, mg, structs)
    for i in range(50):
        new_name = name_gen.select(random.Random(i), stratum="native-welsh")
        assert new_name.name == [["-in"]]


def test_keep_keys_for_stratum_returns_empty_when_no_usage_admits():
    """When --stratum is set and EVERY Meaning has stratum data that
    DOESN'T match, keep_keys_for_stratum returns an empty frozenset
    (NOT None — the full-coverage→None collapse only fires when
    coverage is total). The downstream Generator.select must accept
    an empty keep_keys without crashing — it's the documented 'no
    usage matches the filter' signal that just produces no name.

    Pinned because the full-coverage tests above mask this branch:
    they include at least one no-data Meaning that admits via
    passthrough. Phase 4 may produce realistic 'all classified, none
    matching' bundles that hit this exact edge."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    only_other = Meaning("-other", [], [], {}, stratum={"celtic_mix": {"caer": "latin-loan"}})
    mg = MeaningGenerator({"-other": [only_other]}, {}, {"-other": 1})
    keep = mg.keep_keys_for_stratum("native-welsh")
    assert keep == frozenset()
    assert keep is not None


def test_generator_select_with_empty_keep_keys_returns_none():
    """The base Generator.select handles empty keep_keys by returning
    None — already covered for era at line 940, but pin it explicitly
    in the stratum context so a refactor that accidentally diverges
    the two paths is caught. Reuses the existing
    'empty keep_keys → None' contract."""
    from wyrd.generators.kenning.runtime.proportions import Generator

    g = Generator(tag_db={}, elements={"a": 50, "b": 50})
    assert g.select(random.Random(0), keep_keys=frozenset()) is None


def test_name_generator_select_stratum_threads_through_positive_tag_path():
    """Mirror of test_name_generator_select_era_range_threads_through_positive_tag_path
    for stratum. _select_tag / _select_tags branch threads keep_keys
    independently from _select_no_tag, so a tag + stratum combo must
    be pinned: two usages tagged 'tree', one in-stratum and one not,
    with empirical weight on the wrong one. With tags=('tree',) AND
    stratum set, every pick must still land on the in-stratum usage."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    m_in = Meaning("-in", ["tree"], [], {}, stratum={"celtic_mix": {"caer": "native-welsh"}})
    m_out = Meaning("-out", ["tree"], [], {}, stratum={"celtic_mix": {"din": "latin-loan"}})
    meaning_db = {"-in": [m_in], "-out": [m_out]}
    proportions = {"-in": 1, "-out": 99}
    tag_db = {"tree": ["-in", "-out"]}
    mg = MeaningGenerator(meaning_db, tag_db, proportions)
    mg.load_parts(proportions, "single")
    structs = {(((m_in.location, "single"),),): 1}
    name_gen = NameGenerator(meaning_db, mg, structs)
    for i in range(50):
        new_name = name_gen.select(random.Random(i), "tree", stratum="native-welsh")
        assert new_name.name == [["-in"]]


def test_kenning_generate_treats_empty_stratum_as_no_filter():
    """``params['stratum'] = ""`` (the SPA's empty-dropdown shape AND
    the input_schema default) reaches the Kenning layer where
    ``stratum or None`` coerces it to None — the 'no filter' signal.
    Pin bit-stability with the no-key call so a regression that drops
    the coercion (and crashes on the empty string at the filter
    layer) is caught."""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    seed = 12345
    a = k.generate({"culture": "english", "stratum": ""}, seed)
    b = k.generate({"culture": "english"}, seed)
    assert a.result == b.result


def test_kenning_schema_default_round_trips_through_generate():
    """The SPA reads ``input_schema().properties.stratum.default``
    and feeds it back through ``generate()``. Pin that the schema
    default (the empty string) is a valid input — without this, a
    refactor that tightens the coercion (e.g. requires a non-empty
    string) would silently break SPA usage where no test fires."""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    default = k.input_schema()["properties"]["stratum"]["default"]
    seed = 42
    with_default = k.generate({"culture": "english", "stratum": default}, seed)
    baseline = k.generate({"culture": "english"}, seed)
    assert with_default.result == baseline.result


def test_name_generator_select_stratum_none_is_bit_stable():
    """stratum=None matches the no-arg call exactly — the kwarg adds
    a None default rather than a behavior change. Pin via seeded
    sequence comparison."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    m_a = Meaning("-a", [], [], {})
    m_b = Meaning("-b", [], [], {})
    meaning_db = {"-a": [m_a], "-b": [m_b]}
    proportions = {"-a": 50, "-b": 50}
    mg = MeaningGenerator(meaning_db, {}, proportions)
    mg.load_parts(proportions, "single")
    structs = {((("post", "single"),),): 1}
    name_gen = NameGenerator(meaning_db, mg, structs)
    seq_default = [name_gen.select(random.Random(i)).name for i in range(20)]
    seq_explicit_none = [name_gen.select(random.Random(i), stratum=None).name for i in range(20)]
    assert seq_default == seq_explicit_none


def test_name_generator_select_era_and_stratum_compose_via_intersection():
    """When BOTH --era and --stratum are set, a usage must clear both
    gates. Three usages: A in-era + in-stratum, B in-era + wrong-stratum,
    C wrong-era + in-stratum. With both filters set, only A survives;
    B and C are dropped despite passing one gate each. Pins the
    intersection semantics in _intersect_keep_keys."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    m_ok = Meaning(
        "-ok",
        [],
        [],
        {},
        attested_years={"old_english": [("ok", 950)]},
        stratum={"celtic_mix": {"ok": "native-welsh"}},
    )
    m_wrong_stratum = Meaning(
        "-wrong-stratum",
        [],
        [],
        {},
        attested_years={"old_english": [("ws", 950)]},
        stratum={"celtic_mix": {"ws": "latin-loan"}},
    )
    m_wrong_era = Meaning(
        "-wrong-era",
        [],
        [],
        {},
        attested_years={"old_english": [("we", 1500)]},
        stratum={"celtic_mix": {"we": "native-welsh"}},
    )
    meaning_db = {
        "-ok": [m_ok],
        "-wrong-stratum": [m_wrong_stratum],
        "-wrong-era": [m_wrong_era],
    }
    proportions = {"-ok": 1, "-wrong-stratum": 99, "-wrong-era": 99}
    mg = MeaningGenerator(meaning_db, {}, proportions)
    mg.load_parts(proportions, "single")
    structs = {(((m_ok.location, "single"),),): 1}
    name_gen = NameGenerator(meaning_db, mg, structs)
    for i in range(50):
        new_name = name_gen.select(
            random.Random(i),
            era_range=(800, 1100),
            stratum="native-welsh",
        )
        assert new_name.name == [["-ok"]]


def test_intersect_keep_keys_handles_none_and_set_combinations():
    """Direct test of the helper — both None → None; one None → other;
    both set → intersection. Pinning each branch independently because
    a regression on the 'one None' path would silently widen the filter
    (producing more morphemes than intended) without breaking
    end-to-end seed-stability tests."""
    from wyrd.generators.kenning.runtime.proportions import _intersect_keep_keys

    assert _intersect_keep_keys(None, None) is None
    assert _intersect_keep_keys(frozenset({"a", "b"}), None) == frozenset({"a", "b"})
    assert _intersect_keep_keys(None, frozenset({"a", "b"})) == frozenset({"a", "b"})
    assert _intersect_keep_keys(frozenset({"a", "b"}), frozenset({"b", "c"})) == frozenset({"b"})


def test_kenning_generate_accepts_stratum_param():
    """Kenning.generate threads ``stratum`` through to NameGenerator.select.
    Pin via a smoke run that the param doesn't crash and the result is
    a non-empty string. Bit-stability with stratum=None vs unset is
    covered by test_name_generator_select_stratum_none_is_bit_stable
    upstream."""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    result = k.generate(
        {"culture": "english", "stratum": "native-welsh"},
        seed=12345,
    )
    assert result.result
    assert isinstance(result.result, str)


def test_kenning_input_schema_exposes_stratum():
    """The SPA reads ``input_schema`` to render form inputs. Pin that
    ``stratum`` is exposed with the expected default + description so
    the SPA doesn't silently drop the new knob."""
    from wyrd.generators.kenning import Kenning

    schema = Kenning().input_schema()
    assert "stratum" in schema["properties"]
    stratum_def = schema["properties"]["stratum"]
    assert stratum_def["type"] == "string"
    assert stratum_def["default"] == ""
    assert "wyrd-lr4" in stratum_def["description"]


# --- wyrd-mj2 cohesion (tag co-occurrence bias) ---------------------------


def _build_cohesion_test_generator(meaning_db, proportions, structs, cooc, marg):
    """Construct a NameGenerator pre-loaded with a synthetic tag-cooccurrence
    bundle. Used by the cohesion tests to drive the bias deterministically
    rather than relying on whatever the bundled corpora happen to encode."""
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator, NameGenerator

    mg = MeaningGenerator(meaning_db, {}, proportions)
    mg.load_parts(proportions, "single")
    return NameGenerator(meaning_db, mg, structs, tag_cooccurrence=cooc, tag_marginal=marg)


def test_name_generator_cohesion_zero_is_bit_stable_with_no_cooccurrence():
    """cohesion=0 must bypass the boost computation entirely. Pin via
    sequence equality across 30 seeds against the no-kwarg call."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    m_a = Meaning("-a", ["water"], [], {})
    m_b = Meaning("-b", ["plant"], [], {})
    meaning_db = {"-a": [m_a], "-b": [m_b]}
    proportions = {"-a": 50, "-b": 50}
    structs = {((("post", "single"),),): 1}
    name_gen = _build_cohesion_test_generator(
        meaning_db,
        proportions,
        structs,
        cooc={"water|plant": 100, "plant|water": 100},
        marg={"water": 100, "plant": 100},
    )
    seq_default = [name_gen.select(random.Random(i)).name for i in range(30)]
    seq_zero = [name_gen.select(random.Random(i), cohesion=0.0).name for i in range(30)]
    assert seq_default == seq_zero


def test_name_generator_cohesion_one_biases_second_slot_toward_cooccurring_tag():
    """At cohesion=1 with a strong asymmetric cooccurrence signal, the
    second slot must skew toward the candidate whose tags co-occur with
    the first slot's tags. Two-slot structure: prefix carries ['water'],
    two post candidates compete: -in tagged 'plant' (strongly co-occurs
    with water in our synthetic stats), -out tagged 'religion' (zero
    co-occurrence). Equal empirical weights so any skew comes from the
    cohesion bias alone."""
    from collections import Counter

    from wyrd.generators.kenning.runtime.meaning import Meaning

    river = Meaning("River-", ["water"], [], {})
    in_word = Meaning("-in", ["plant"], [], {})
    out_word = Meaning("-out", ["religion"], [], {})
    meaning_db = {"River-": [river], "-in": [in_word], "-out": [out_word]}
    proportions = {"River-": 1, "-in": 50, "-out": 50}
    # Two-slot structure: pre + post, no 'single' marker.
    structs = {((("pre",), ("post",)),): 1}
    cooc = {
        "water|plant": 500,  # Strong co-occurrence.
        "water|religion": 0,  # Never seen together.
    }
    marg = {"water": 500, "plant": 500, "religion": 1}
    name_gen = _build_cohesion_test_generator(meaning_db, proportions, structs, cooc, marg)
    counts_default = Counter()
    counts_cohesion = Counter()
    for i in range(200):
        n_default = name_gen.select(random.Random(i)).name
        n_cohesion = name_gen.select(random.Random(i), cohesion=1.0).name
        counts_default[n_default[0][1]] += 1
        counts_cohesion[n_cohesion[0][1]] += 1
    # At cohesion=0 (default), the post slot is ~50/50 between -in and -out.
    assert 70 < counts_default["-in"] < 130
    # At cohesion=1, the post slot strongly favors -in (the co-occurring
    # tag); -out gets pushed below the equal-weight baseline.
    assert counts_cohesion["-in"] > counts_default["-in"]
    assert counts_cohesion["-in"] > 150


def test_name_generator_cohesion_no_cooccurrence_data_is_no_op():
    """When the bundle carries no tag-cooccurrence data, even cohesion=1
    must be a no-op — no zero-divide, no crash, just bit-stable with
    cohesion=0. Legacy bundles (no co-occurrence keys) ride this path."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    m_a = Meaning("-a", ["water"], [], {})
    m_b = Meaning("-b", ["plant"], [], {})
    meaning_db = {"-a": [m_a], "-b": [m_b]}
    proportions = {"-a": 50, "-b": 50}
    structs = {((("post", "single"),),): 1}
    name_gen = _build_cohesion_test_generator(meaning_db, proportions, structs, cooc={}, marg={})
    seq_zero = [name_gen.select(random.Random(i), cohesion=0.0).name for i in range(20)]
    seq_one = [name_gen.select(random.Random(i), cohesion=1.0).name for i in range(20)]
    assert seq_zero == seq_one


def test_name_generator_cohesion_no_prior_tags_first_slot_unaffected():
    """The first slot has no prior context, so cohesion can't bias it.
    Pin: with cohesion=1 but only one slot, the output sequence is
    identical to cohesion=0. Catches a regression that would try to
    apply the boost to the first pick (zero-divide on mean_raw)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    m_a = Meaning("-a", ["water"], [], {})
    m_b = Meaning("-b", ["plant"], [], {})
    meaning_db = {"-a": [m_a], "-b": [m_b]}
    proportions = {"-a": 50, "-b": 50}
    structs = {((("post", "single"),),): 1}
    name_gen = _build_cohesion_test_generator(
        meaning_db,
        proportions,
        structs,
        cooc={"water|plant": 100},
        marg={"water": 100, "plant": 100},
    )
    seq_zero = [name_gen.select(random.Random(i), cohesion=0.0).name for i in range(20)]
    seq_one = [name_gen.select(random.Random(i), cohesion=1.0).name for i in range(20)]
    assert seq_zero == seq_one


def test_name_generator_cohesion_no_signal_returns_none_boost():
    """Internal: when no candidate tag has any co-occurrence with the
    prior context tags, _cohesion_boost returns None so the bucket
    sampling skips the multiplication entirely. Pinned via direct call
    to the private helper because the no-signal path is hard to spot
    in end-to-end output."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    m_a = Meaning("-a", ["plant"], [], {})
    m_b = Meaning("-b", ["religion"], [], {})
    meaning_db = {"-a": [m_a], "-b": [m_b]}
    proportions = {"-a": 1, "-b": 1}
    structs = {((("post", "single"),),): 1}
    # marg has 'water' but no co-occurrence between water and any
    # candidate tag → raw scores are all zero → boost is None.
    name_gen = _build_cohesion_test_generator(
        meaning_db,
        proportions,
        structs,
        cooc={},
        marg={"water": 100},
    )
    boost = name_gen._cohesion_boost(("post", "single"), {"water"}, cohesion=1.0)
    assert boost is None


def test_name_generator_cohesion_unknown_bucket_returns_none_boost():
    """A key referenced by a structure but absent from MeaningGenerator's
    bucket map returns None — the caller hits the bit-stable path
    rather than crashing on an empty candidates iteration."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    m = Meaning("-a", ["water"], [], {})
    meaning_db = {"-a": [m]}
    proportions = {"-a": 1}
    structs = {((("post", "single"),),): 1}
    name_gen = _build_cohesion_test_generator(
        meaning_db,
        proportions,
        structs,
        cooc={"water|plant": 1},
        marg={"water": 1, "plant": 1},
    )
    boost = name_gen._cohesion_boost(("post", "nonexistent"), {"water"}, cohesion=1.0)
    assert boost is None


def test_kenning_input_schema_exposes_cohesion():
    """The cohesion knob surfaces in input_schema so the SPA renders a
    slider. Default 0.0, range [0,1], number type."""
    from wyrd.generators.kenning import Kenning

    schema = Kenning().input_schema()
    assert "cohesion" in schema["properties"]
    prop = schema["properties"]["cohesion"]
    assert prop["type"] == "number"
    assert prop["default"] == 0.0
    assert prop["minimum"] == 0.0
    assert prop["maximum"] == 1.0


def test_kenning_generate_cohesion_zero_bit_stable_against_default():
    """End-to-end: Kenning.generate with cohesion=0 produces identical
    output to default (no cohesion key) for the same seed across the
    bundled English culture."""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    seq_default = [k.generate({"culture": "english"}, seed=s).result for s in range(20)]
    seq_zero = [
        k.generate({"culture": "english", "cohesion": 0.0}, seed=s).result for s in range(20)
    ]
    assert seq_default == seq_zero


def test_name_generator_cohesion_threads_through_single_positive_tag_path():
    """test-coverage P2: the `_select_tag` branch (single positive tag
    passed to NameGenerator.select) was not exercised by the original
    cohesion tests. Mirror the asymmetric-cooccurrence test but invoke
    via `name_gen.select(rng, 'tree', cohesion=1.0)` so the
    `_select_tag` cohesion plumbing gets pinned."""
    from collections import Counter

    from wyrd.generators.kenning.runtime.meaning import Meaning

    river = Meaning("River-", ["water", "tree"], [], {})
    in_word = Meaning("-in", ["plant", "tree"], [], {})
    out_word = Meaning("-out", ["religion", "tree"], [], {})
    meaning_db = {"River-": [river], "-in": [in_word], "-out": [out_word]}
    proportions = {"River-": 1, "-in": 50, "-out": 50}
    structs = {((("pre",), ("post",)),): 1}
    cooc = {"water|plant": 500, "water|religion": 0}
    marg = {"water": 500, "plant": 500, "religion": 1, "tree": 100}
    name_gen = _build_cohesion_test_generator(meaning_db, proportions, structs, cooc, marg)
    counts_default = Counter()
    counts_cohesion = Counter()
    for i in range(200):
        n_default = name_gen.select(random.Random(i), "tree").name
        n_cohesion = name_gen.select(random.Random(i), "tree", cohesion=1.0).name
        counts_default[n_default[0][1]] += 1
        counts_cohesion[n_cohesion[0][1]] += 1
    # Sanity: the no-cohesion baseline is roughly even.
    assert 70 < counts_default["-in"] < 130
    # With cohesion=1, -in (co-occurring with water) wins much more often.
    assert counts_cohesion["-in"] > counts_default["-in"]
    assert counts_cohesion["-in"] > 150


def test_name_generator_cohesion_threads_through_multi_tag_pool_path():
    """test-coverage P2: `_select_tags` (multi-tag pool) threads cohesion
    through both `_select_no_tag` AND `_select_tag` branches, then
    rng-merges the per-pool picks. Pin the multi-tag-pool path with two
    positional tags + cohesion=1, and confirm sequence determinism by
    checking two same-seed calls produce identical names."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    river = Meaning("River-", ["water", "tree", "plant"], [], {})
    in_word = Meaning("-in", ["plant", "tree"], [], {})
    out_word = Meaning("-out", ["religion", "tree", "plant"], [], {})
    meaning_db = {"River-": [river], "-in": [in_word], "-out": [out_word]}
    proportions = {"River-": 1, "-in": 50, "-out": 50}
    structs = {((("pre",), ("post",)),): 1}
    cooc = {"water|plant": 500, "tree|plant": 500}
    marg = {"water": 500, "plant": 500, "religion": 1, "tree": 500}
    name_gen = _build_cohesion_test_generator(meaning_db, proportions, structs, cooc, marg)
    a = name_gen.select(random.Random(7), "tree", "plant", cohesion=1.0).name
    b = name_gen.select(random.Random(7), "tree", "plant", cohesion=1.0).name
    assert a == b  # determinism
    # And the multi-tag path must not crash with cohesion=0 either.
    c = name_gen.select(random.Random(7), "tree", "plant", cohesion=0.0).name
    assert c is not None


def test_name_generator_cohesion_composes_with_novelty():
    """test-coverage P3: cohesion=0 + novelty=X must still match a
    pre-PR novelty=X sample (proves the new key_boost is None branch in
    Generator.select doesn't disturb the existing novelty path)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    m_a = Meaning("-a", ["water"], [], {})
    m_b = Meaning("-b", ["plant"], [], {})
    meaning_db = {"-a": [m_a], "-b": [m_b]}
    proportions = {"-a": 70, "-b": 30}
    structs = {((("post", "single"),),): 1}
    name_gen = _build_cohesion_test_generator(
        meaning_db,
        proportions,
        structs,
        cooc={"water|plant": 100},
        marg={"water": 100, "plant": 100},
    )
    seq_pure_novelty = [name_gen.select(random.Random(i), novelty=0.5).name for i in range(20)]
    seq_with_zero_cohesion = [
        name_gen.select(random.Random(i), novelty=0.5, cohesion=0.0).name for i in range(20)
    ]
    assert seq_pure_novelty == seq_with_zero_cohesion


def test_name_generator_cohesion_one_normalization_respects_keep_keys():
    """The cohesion mean-normalization must be computed over the surviving
    keep_keys subset, not the full bucket — otherwise the era-filtered
    pool's effective mean drifts from 1.0 and the boost is over-
    weighted toward leftover candidates. Pinned at the helper level by
    passing keep_keys directly."""
    from wyrd.generators.kenning.runtime.meaning import Meaning

    # Three candidates: -in (in-era, scores 1.0), -out (in-era, scores
    # 0.0), -filtered (era-filtered out, would have scored 9.0 → without
    # keep_keys-aware normalization, mean would drag toward 3.33).
    river = Meaning("River-", ["water"], [], {})
    in_word = Meaning("-in", ["plant"], [], {})
    out_word = Meaning("-out", ["religion"], [], {})
    # -filtered carries the same 'plant' tag as -in, so it scores the
    # same raw 1.0. Without keep_keys, having it in the denominator
    # drags the bucket mean up to (1 + 0 + 1)/3 = 0.667; with keep_keys,
    # the mean is (1 + 0)/2 = 0.5. The shift in mean directly affects
    # -in's multiplier: 1.5 (without) vs 2.0 (with).
    filtered = Meaning("-filtered", ["plant"], [], {})
    meaning_db = {
        "River-": [river],
        "-in": [in_word],
        "-out": [out_word],
        "-filtered": [filtered],
    }
    proportions = {"River-": 1, "-in": 1, "-out": 1, "-filtered": 1}
    structs = {((("pre",), ("post",)),): 1}
    cooc = {"water|plant": 100, "water|religion": 0}
    marg = {"water": 100, "plant": 100, "religion": 1}
    name_gen = _build_cohesion_test_generator(meaning_db, proportions, structs, cooc, marg)
    keep = frozenset({"-in", "-out"})
    # With keep_keys, the mean is computed only over -in and -out (the
    # surviving subset). -in scores 1.0, -out scores 0.0 → mean=0.5 →
    # multipliers ×2 and ×0 at cohesion=1.
    boost_filtered = name_gen._cohesion_boost(("post",), {"water"}, 1.0, keep_keys=keep)
    assert boost_filtered is not None
    assert "-filtered" not in boost_filtered  # excluded from normalization
    assert boost_filtered["-in"] > boost_filtered["-out"]
    # Without keep_keys, -filtered is in the denominator → mean is higher
    # → -in's boost is correspondingly smaller. Pin the relationship.
    boost_full = name_gen._cohesion_boost(("post",), {"water"}, 1.0)
    assert boost_full is not None
    assert "-filtered" in boost_full
    assert boost_filtered["-in"] > boost_full["-in"]


def test_meaning_generator_bucket_keys_returns_registered_usages():
    """test-coverage P3: pin the new MeaningGenerator.bucket_keys helper
    directly. Returns the tuple of usages registered under a key, or ()
    for an unknown key."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    m_a = Meaning("-a", [], [], {})
    m_b = Meaning("-b", [], [], {})
    mg = MeaningGenerator({"-a": [m_a], "-b": [m_b]}, {}, {"-a": 1, "-b": 1})
    keys = mg.bucket_keys(("post",))
    assert set(keys) == {"-a", "-b"}
    # Unknown bucket → empty tuple, not a crash.
    assert mg.bucket_keys(("nonexistent",)) == ()


def test_raw_class_score_empty_inputs_return_zero():
    """test-coverage P3: pin the early-return branches in _raw_class_score."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator, NameGenerator

    mg = MeaningGenerator({"-a": [Meaning("-a", [], [], {})]}, {}, {"-a": 1})
    name_gen = NameGenerator(
        {}, mg, {}, tag_cooccurrence={"water|plant": 100}, tag_marginal={"water": 100}
    )
    assert name_gen._raw_class_score(set(), {"plant"}) == 0.0
    assert name_gen._raw_class_score({"water"}, set()) == 0.0
    # Prior tag with marginal=0 is skipped.
    assert name_gen._raw_class_score({"unknown_tag"}, {"plant"}) == 0.0


def test_raw_class_score_sums_across_tag_cartesian():
    """test-coverage P3: pin the sum-vs-mean intent — two prior × two
    candidate tags additively contribute (sum, not max, not mean)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator, NameGenerator

    mg = MeaningGenerator({"-a": [Meaning("-a", [], [], {})]}, {}, {"-a": 1})
    cooc = {
        "water|plant": 50,
        "water|tree": 30,
        "fire|plant": 20,
        "fire|tree": 10,
    }
    marg = {"water": 100, "fire": 100}
    name_gen = NameGenerator({}, mg, {}, tag_cooccurrence=cooc, tag_marginal=marg)
    # Sum over all 4 (prior, candidate) pairs:
    #   50/100 + 30/100 + 20/100 + 10/100 = 1.10
    score = name_gen._raw_class_score({"water", "fire"}, {"plant", "tree"})
    assert abs(score - 1.10) < 1e-9


def test_raw_class_score_is_bit_stable_across_python_hash_seed():
    """seed-reproducibility hazard fix: PYTHONHASHSEED randomizes set
    iteration order, and float += is non-associative. Without sorted
    iteration in _raw_class_score, the same input data produces ULP-
    level different scores across processes (verified empirically:
    41.97856592189611 vs 41.978565921896106 vs 41.97856592189612).

    Within ONE process, PYTHONHASHSEED is fixed at startup so
    set-iteration order is stable across calls — a same-process loop
    can't catch the hazard. Spawn subprocesses with varied
    PYTHONHASHSEED env vars and assert the score is bit-identical
    across them. Without sorted iteration, this test fails.
    """
    import subprocess
    import sys

    script = textwrap.dedent("""
        from wyrd.generators.kenning.runtime.meaning import Meaning
        from wyrd.generators.kenning.runtime.proportions import MeaningGenerator, NameGenerator

        mg = MeaningGenerator({"-a": [Meaning("-a", [], [], {})]}, {}, {"-a": 1})
        # Dense 20x20 cooccurrence so the sum has many addends — more
        # terms = more chances for non-associative drift to surface.
        tags = [f"t{i}" for i in range(20)]
        cooc = {f"{a}|{b}": (i + j + 1) * 7
                for i, a in enumerate(tags) for j, b in enumerate(tags)}
        marg = dict.fromkeys(tags, 100)
        name_gen = NameGenerator({}, mg, {}, tag_cooccurrence=cooc, tag_marginal=marg)
        score = name_gen._raw_class_score(set(tags), set(tags))
        # repr to lock the full float precision across the subprocess boundary.
        print(repr(score))
    """)
    scores = set()
    # PYTHONHASHSEED=0 disables randomization; positive ints select a
    # fixed hash seed. A spread of values ensures we hit several distinct
    # set-iteration orders.
    for seed_value in ("1", "2", "3", "5", "7", "11", "13", "17"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={"PYTHONHASHSEED": seed_value, "PATH": os.environ.get("PATH", "")},
            capture_output=True,
            text=True,
            check=True,
        )
        scores.add(result.stdout.strip())
    assert len(scores) == 1, f"non-deterministic score across PYTHONHASHSEED: {scores}"


def test_filter_for_tag_is_bit_stable_across_python_hash_seed():
    """seed-reproducibility hazard (wyrd-8ga): Generator.filter_for_tag
    builds a dict by iterating a set of tag-matched usages. The dict's
    iteration order then feeds weighted_choice's cumulative-threshold
    construction, where order changes the boundary on which a given
    rng draw lands. Without sorted iteration, the same (tags, seed)
    tuple can produce different weighted_choice picks across processes
    with different PYTHONHASHSEED.

    Same approach as test_raw_class_score_is_bit_stable_across_python_hash_seed:
    spawn subprocesses with varied PYTHONHASHSEED env vars and assert
    the picked key is bit-identical across them. Without the sort in
    filter_for_tag, this test fails — empirically by producing 6
    distinct keys across the 8 hash seeds below.
    """
    import subprocess
    import sys

    script = textwrap.dedent("""
        import random
        from wyrd.generators.kenning.runtime.proportions import Generator, weighted_choice

        # 20 keys on the same tag — enough distinct strings to exercise
        # several hash-bucket orderings. Skewed weights so the cumulative
        # threshold lands at a position where order matters: the rng draw
        # falls between adjacent items, and which item is "before" the
        # boundary depends on iteration order.
        keys = [f"-key{i:02d}" for i in range(20)]
        elements = {k: i + 1 for i, k in enumerate(keys)}
        tag_db = {"t": keys}
        g = Generator(tag_db=tag_db, elements=elements)
        # Fixed rng seed pins the cumulative-threshold draw to one
        # specific position; a different iteration order routes that
        # draw to a different key.
        picked = g.select(random.Random(42), "t")
        print(picked)
    """)
    picks = set()
    for seed_value in ("1", "2", "3", "5", "7", "11", "13", "17"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={"PYTHONHASHSEED": seed_value, "PATH": os.environ.get("PATH", "")},
            capture_output=True,
            text=True,
            check=True,
        )
        picks.add(result.stdout.strip())
    assert len(picks) == 1, f"non-deterministic pick across PYTHONHASHSEED: {picks}"


def test_generator_select_key_boost_multiplies_weights():
    """test-coverage P3: direct unit pin of Generator.select(key_boost=...).
    Strongly skewed boost should flip the empirical winner."""
    from collections import Counter

    from wyrd.generators.kenning.runtime.proportions import Generator

    g = Generator(tag_db={}, elements={"a": 70, "b": 30})
    # Without boost, 'a' should dominate ~70/30.
    plain = Counter(g.select(random.Random(i)) for i in range(2000))
    assert plain["a"] > 1200
    # With a 10x boost on 'b', 'b' should now dominate ~30*10 vs 70.
    boost = {"a": 1.0, "b": 10.0}
    boosted = Counter(g.select(random.Random(i), key_boost=boost) for i in range(2000))
    assert boosted["b"] > boosted["a"]


def test_load_proportions_handles_missing_cooccurrence_keys():
    """test-coverage P3: legacy bundles without the new tag_cooccurrence
    / tag_marginal keys must load cleanly — load_proportions defaults
    both to empty dicts and the resulting NameGenerator has the no-op
    cohesion path. Pinned at the loader to catch silent rename drift
    between bundle keys and the loader's `.get(...)` lookup."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import load_proportions

    meaning_db = {"-a": [Meaning("-a", [], [], {})]}
    tag_db = {}
    legacy_data = {
        "usages": {"-a": 1},
        "single_usages": {"-a": 1},
        "structures": [{"proportion": 1, "words": [[{"location": "post"}]]}],
        # No tag_cooccurrence, no tag_marginal — legacy shape.
    }
    name_gen = load_proportions(legacy_data, meaning_db, tag_db)
    assert name_gen.tag_cooccurrence == {}
    assert name_gen.tag_marginal == {}


def test_load_proportions_passes_cooccurrence_keys_through():
    """test-coverage P3: present-keys path. Confirms the loader actually
    reads from the bundle dict and threads to NameGenerator."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import load_proportions

    meaning_db = {"-a": [Meaning("-a", [], [], {})]}
    tag_db = {}
    data = {
        "usages": {"-a": 1},
        "single_usages": {"-a": 1},
        "structures": [{"proportion": 1, "words": [[{"location": "post"}]]}],
        "tag_cooccurrence": {"water|plant": 7},
        "tag_marginal": {"water": 7, "plant": 7},
    }
    name_gen = load_proportions(data, meaning_db, tag_db)
    assert name_gen.tag_cooccurrence == {"water|plant": 7}
    assert name_gen.tag_marginal == {"water": 7, "plant": 7}


def test_kenning_generate_cohesion_one_shifts_distribution_against_seeds():
    """End-to-end: cohesion=1 shifts the empirical distribution against
    the no-cohesion baseline. Across 30 seeds at least some names
    diverge — the bias is real, not a no-op."""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    plain = set()
    cohered = set()
    for seed in range(30):
        plain.add(k.generate({"culture": "english"}, seed=seed).result)
        cohered.add(k.generate({"culture": "english", "cohesion": 1.0}, seed=seed).result)
    assert plain != cohered
