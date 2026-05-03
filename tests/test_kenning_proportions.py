"""Unit tests for kenning's weighted_choice — the core weighted sampler."""

from __future__ import annotations

import random
from collections import Counter

from wyrd.generators.kenning.proportions import (
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


def test_meaning_generator_select_threads_harshness():
    """MeaningGenerator.select(key, *tags, harshness=...) forwards harshness
    into Generator.select for the chosen bucket. At harshness=1, the empirical
    99:1 weight gets re-weighted by the harsh phonology score, shifting picks
    toward the stop-final key (-shuck) over the soft -baron bucket-mate."""
    from collections import Counter

    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import MeaningGenerator

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
    from wyrd.generators.kenning.proportions import (
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


def test_select_populates_inflection_labels_at_high_density():
    """End-to-end: NameGenerator.select(inflection_density=1.0) on a
    meaning_db with inflection metadata returns a NewName whose
    inflection_labels carry the picked label per element. Pins the
    integration boundary that _render_substitutions tests can't reach
    on their own."""
    from wyrd.generators.kenning.meaning import Meaning

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
    from wyrd.generators.kenning.meaning import Meaning

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
    from wyrd.generators.kenning.proportions import Generator

    rng_a = random.Random(42)
    rng_b = random.Random(42)
    g = Generator(tag_db={}, elements={"-ham": 90, "-shuck": 10})
    pre_d6 = weighted_choice(rng_a, list(g.elements.items()))
    via_select = g.select(rng_b, harshness=0.0)
    assert pre_d6 == via_select


def test_generator_select_harshness_one_skews_toward_harsh_keys():
    """At harshness=1, monte-carlo over 2000 picks shifts the empirical
    90:10 split (ham:shuck) toward shuck enough to detect."""
    from wyrd.generators.kenning.proportions import Generator

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
    from wyrd.generators.kenning.proportions import Generator

    g = Generator(tag_db={}, elements={"-ham": 99, "-shuck": 1})
    counts = Counter(g.select(random.Random(i), novelty=1.0, harshness=1.0) for i in range(2000))
    assert 800 < counts["-ham"] < 1200
    assert 800 < counts["-shuck"] < 1200


# --- D8 explainer: <lemma>@<label> surfacing -----------------------------


def test_description_no_inflection_labels_unchanged():
    """At default (inflection_labels=None), description() emits the historic
    `lemma (sources gloss)` form per element. Bit-stable with pre-D8."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import NewName

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
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import NewName

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
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import NewName

    bridg = Meaning("Bridg-", [], ["bridge"], {"old_english": ["brycg"]})
    water = Meaning("-water", [], ["water"], {"old_english": ["wæter"]})
    new_name = NewName(
        struct=None,
        meaning_db={"Bridg-": [bridg], "-water": [water]},
        name=[["Bridg-", "-water"]],
        inflection_labels=[[None, "genitive_strong"]],
    )
    # Multi-element words keep dashes in the head per the existing
    # description() shape.
    assert new_name.description() == "Bridg- (EN bridge) -water@genitive_strong (EN water)"


def test_description_inflection_labels_shorter_than_name_does_not_crash():
    """Defensive: if inflection_labels has a shorter inner list than its
    matching name word — possible only via direct construction or a bug
    elsewhere — description() must fall through to the unlabelled form
    rather than raising IndexError. Pins the IndexError catch in
    _inflection_label_for."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import NewName

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


# --- wyrd-9kh.1: citation surfacing in description() / components() ----


def test_description_surfaces_citations_when_present():
    """When a Meaning carries scholarly citations, description() appends
    'cited by <list>' to the breakdown line."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import NewName

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
    assert new_name.description() == "cot (EN cottage; cited by mawer_1920, skeat_1901)"


def test_description_omits_citation_block_when_absent():
    """A Meaning with no citations gets the historic shape — no trailing
    'cited by' clutters the explainer for legacy rando-only morphemes."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import NewName

    m = Meaning("-cot", [], ["cottage"], {"old_english": ["cot"]})
    new_name = NewName(
        struct=None,
        meaning_db={"-cot": [m]},
        name=[["-cot"]],
    )
    assert new_name.description() == "cot (EN cottage)"


def test_description_dedupes_citations_across_languages():
    """A Meaning with citations under multiple languages (rare but possible
    when the family spans both — e.g., a Welsh-Marches morpheme cited for
    both its OE and Welsh attestations) emits each source once, sorted."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import NewName

    m = Meaning(
        "-allt",
        [],
        ["wood"],
        {"old_english": ["allt"], "celtic_mix": ["allt"]},
        citations={
            "old_english": ["bannister_1916", "morgan_1912"],
            "celtic_mix": ["morgan_1912", "watson_1926"],
        },
    )
    new_name = NewName(
        struct=None,
        meaning_db={"-allt": [m]},
        name=[["-allt"]],
    )
    # bannister_1916, morgan_1912 (deduped), watson_1926 — sorted alpha,
    # at the 3-citation limit so all three render without truncation.
    assert new_name.description() == (
        "allt (EN/CL wood; cited by bannister_1916, morgan_1912, watson_1926)"
    )


def test_description_truncates_long_citation_lists_with_count():
    """Words like '-ham' carry 18+ citations from the well-mined OE corpus.
    The explainer surfaces the first three with '(+N more)' so the
    breakdown stays readable; components() keeps the full list for SPA
    rendering."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import NewName

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
    # 7 citations → top 3 + '(+4 more)'.
    assert new_name.description() == "ham (EN home; cited by src_00, src_01, src_02 (+4 more))"
    # components() carries the full list — the truncation is display-only.
    components = new_name.components()
    assert components[0]["citations"] == sources


def test_components_includes_citation_list():
    """The API envelope's components carry the same citation list so the
    SPA can render attribution per element."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import NewName

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
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import NewName

    m = Meaning("-cot", [], ["cottage"], {"old_english": ["cot"]})
    new_name = NewName(
        struct=None,
        meaning_db={"-cot": [m]},
        name=[["-cot"]],
    )
    components = new_name.components()
    assert components[0]["citations"] == []


# --- wyrd-yan: fiction-tag exclusion gate -------------------------------


def test_generator_select_exclude_tags_drops_matching_keys():
    """Generator.select with exclude_tags=('fiction',) removes any usage
    that the tag_db lists under 'fiction'. Picks must come from the
    remaining keys only — pin via Monte Carlo with a tag_db that marks
    one key as fiction."""
    from wyrd.generators.kenning.proportions import Generator

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
    from wyrd.generators.kenning.proportions import Generator

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
    from wyrd.generators.kenning.proportions import Generator

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
    from wyrd.generators.kenning.proportions import Generator

    g = Generator(
        tag_db={"fiction": ["only-key-"]},
        elements={"only-key-": 1},
    )
    assert g.select(random.Random(0), exclude_tags=("fiction",)) is None


def test_meaning_generator_select_threads_exclude_tags():
    """MeaningGenerator.select forwards exclude_tags into Generator.select
    so the gate works end-to-end through the bucket dispatch."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import MeaningGenerator

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
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import MeaningGenerator, NameGenerator

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
    from wyrd.generators.kenning.proportions import Generator

    g = Generator(tag_db={}, elements={"a": 50, "b": 50})
    for i in range(50):
        assert g.select(random.Random(i), keep_keys=frozenset({"a"})) == "a"


def test_generator_select_keep_keys_none_disables_filter():
    """keep_keys=None is the bit-stable 'no filter' signal — both keys
    remain reachable. Pin so a future refactor can't accidentally
    treat None as an empty set."""
    from collections import Counter

    from wyrd.generators.kenning.proportions import Generator

    g = Generator(tag_db={}, elements={"a": 50, "b": 50})
    counts = Counter(g.select(random.Random(i), keep_keys=None) for i in range(200))
    assert counts["a"] > 0 and counts["b"] > 0


def test_generator_select_keep_keys_empty_set_returns_none():
    """An empty keep_keys is the legitimate 'no usage matches the era'
    signal — Generator.select should return None rather than crash on
    an empty items_list."""
    from wyrd.generators.kenning.proportions import Generator

    g = Generator(tag_db={}, elements={"a": 50, "b": 50})
    assert g.select(random.Random(0), keep_keys=frozenset()) is None


def test_meaning_generator_keep_keys_for_era_returns_none_for_none_range():
    """MeaningGenerator.keep_keys_for_era(None) returns None (the
    'no filter' signal) — the runtime threads this straight through to
    Generator.select."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import MeaningGenerator

    m = Meaning("-cot", [], [], {})
    mg = MeaningGenerator({"-cot": [m]}, {}, {"-cot": 1})
    assert mg.keep_keys_for_era(None) is None


def test_meaning_generator_keep_keys_for_era_filters_by_attestation():
    """A morpheme with year evidence outside the window is excluded;
    one with evidence inside the window is admitted; one with NO
    evidence passes through (the documented 'pass-through' rule)."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import MeaningGenerator

    in_window = Meaning("-in", [], [], {}, attested_years={"old_english": [("in", 950)]})
    out_window = Meaning("-out", [], [], {}, attested_years={"old_english": [("out", 1500)]})
    no_data = Meaning("-none", [], [], {})
    meaning_db = {"-in": [in_window], "-out": [out_window], "-none": [no_data]}
    mg = MeaningGenerator(meaning_db, {}, dict.fromkeys(meaning_db, 1))
    keep = mg.keep_keys_for_era((800, 1100))
    assert keep == frozenset({"-in", "-none"})


def test_meaning_generator_keep_keys_for_era_caches_per_range():
    """Two calls with the same era_range share the precomputed set so
    the meaning_db isn't re-walked per bucket. Pin via identity (the
    cache returns the same frozenset object) so a perf regression
    that recomputes per call is caught."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import MeaningGenerator

    m = Meaning("-cot", [], [], {}, attested_years={"old_english": [("cot", 950)]})
    mg = MeaningGenerator({"-cot": [m]}, {}, {"-cot": 1})
    a = mg.keep_keys_for_era((800, 1100))
    b = mg.keep_keys_for_era((800, 1100))
    assert a is b


def test_meaning_generator_select_threads_keep_keys():
    """MeaningGenerator.select forwards keep_keys into Generator.select
    so an out-of-era usage never wins, even when its empirical weight
    dominates."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import MeaningGenerator

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
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import (
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


def test_name_generator_select_era_range_none_is_bit_stable():
    """era_range=None is the runtime's 'no --era passed' signal — the
    keep-set is None and Generator.select takes its bit-stable fast
    path. Pin by comparing seeded picks with vs without era_range=None
    to the historic no-kwarg call."""
    from wyrd.generators.kenning.meaning import Meaning
    from wyrd.generators.kenning.proportions import (
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
    input control for it. Default-less: the absence of an era key
    means 'no filter', which the runtime maps to era_range=None."""
    from wyrd.generators.kenning import Kenning

    schema = Kenning().input_schema()
    assert "era" in schema["properties"]
    assert schema["properties"]["era"]["type"] == "string"
