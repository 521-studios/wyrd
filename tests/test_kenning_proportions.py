"""Unit tests for kenning's weighted_choice — the core weighted sampler."""

from __future__ import annotations

import random

import pytest

from wyrd.generators.kenning.runtime.proportions import (
    NewName,
    weighted_choice,
)


def _render(name, rendered=None):
    return str(NewName(struct=None, meaning_db=None, name=name, rendered=rendered))


# wyrd-5z5j / DECISIONS D39: the render is the single owner of positional case —
# the SLOT (word position) decides, not the morpheme's stored case.
def test_render_lowercases_non_initial_morphemes():
    # a name morpheme used mid-word renders lowercase — no stray mid-word capital
    assert _render([["Corn-", "na", "mul", "-lac-", "Buna", "-rath"]]) == "Cornnamullacbunarath"


def test_render_splits_words_with_space_and_caps_each():
    assert _render([["Corn-", "na", "mul", "-lac-"], ["Buna", "-rath"]]) == "Cornnamullac Bunarath"


def test_render_front_caps_lowercase_initial():
    assert _render([["buna", "-rath"]]) == "Bunarath"


def test_render_preserves_word_initial_internal_caps():
    # McLeod / O'Brien-style internal caps survive at a word start (front-cap,
    # do NOT lowercase first); a non-initial McLeod still lowercases.
    assert _render([["McLeod", "-ton"]]) == "McLeodton"
    assert _render([["Glen"], ["McLeod"]]) == "Glen McLeod"


def test_render_variants_obey_the_slot():
    # a substituted variant (self.rendered) dropped into a non-initial slot
    # lowercases like the base; word-initial keeps its case.
    assert _render([["Whit-", "x"]], rendered=[["Whit", "BUNA"]]) == "Whitbuna"
    assert _render([["x"]], rendered=[["McLeod"]]) == "McLeod"


def test_diversify_repeats_picks_cross_language_synonym():
    """wyrd-vd6y: a repeated surface ("Hill Hill") renders its later
    occurrence as a same-meaning synonym in a DIFFERENT language (Old Norse
    "haeth" glossed "Hill"), not the canonical repeat — and NOT a
    different-meaning sibling (Celtic "coille" = wood). The breakdown
    (to_dict) follows, so name + morphemes stay consistent."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    hill = Meaning("hill", tags=[], meanings=["hill"], sources={"old_english": ["hill"]})
    norse = Meaning("hill", tags=[], meanings=["Hill"], sources={"old_scandinavian": ["haeth"]})
    wood = Meaning("hill", tags=[], meanings=["Wood"], sources={"celtic_mix": ["coille"]})
    meaning_db = {"hill": [hill, norse, wood]}

    nn = NewName(struct=None, meaning_db=meaning_db, name=[["hill"], ["hill"]])
    assert str(nn) == "Hill Haeth"  # second diversified to the Norse synonym
    d = nn.to_dict()
    assert d["words"][0][0]["usage"] == "hill"
    assert d["words"][1][0]["usage"] == "haeth"  # breakdown follows the name
    assert d["words"][1][0]["sources"] == {"old_scandinavian": ["haeth"]}


def test_diversify_synonym_graft_uses_era_form():
    """wyrd-68ye: under an active era render, a repeat resolved by a cross-language
    synonym grafts the synonym's ERA reflex (period-pure), not its raw modern form
    — so an oe-early name no longer shows a stray modern token in the repeat slot.
    Contrast test_diversify_repeats_picks_cross_language_synonym (no era → raw)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    hill = Meaning("hill", tags=[], meanings=["hill"], sources={"old_english": ["hill"]})
    norse = Meaning(
        "hill",
        tags=[],
        meanings=["Hill"],
        sources={"old_scandinavian": ["haeth"]},
        era_reflexes={"old-english": [("haethr", "src")]},
    )
    nn = NewName(struct=None, meaning_db={"hill": [hill, norse]}, name=[["hill"], ["hill"]])
    nn.era_render_language = "old-english"  # _apply_render sets this in production
    assert str(nn) == "Hill Haethr"  # era reflex, not the raw 'Haeth'


def test_diversify_repick_uses_era_form():
    """wyrd-68ye: a re-picked repeat (no cross-language synonym) renders the
    re-pick's ERA reflex under an active era render, instead of falling back to the
    modern usage. Contrast test_diversify_repeats_repicks_* (no era → 'Park Dale')."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    park = Meaning("park", tags=[], meanings=["park"], sources={"old_english": ["park"]})
    dale = Meaning(
        "dale",
        tags=[],
        meanings=["valley"],
        sources={"old_english": ["dale"]},
        era_reflexes={"old-english": [("dael", "src")]},
    )
    # rendered is non-None in an era request (set by _apply_render before diversify);
    # the re-pick's render branch only runs then — for non-era (rendered=None) the
    # path is unchanged (see test_diversify_repeats_repicks_different_morpheme...).
    nn = NewName(
        struct=None,
        meaning_db={"park": [park], "dale": [dale]},
        name=[["park"], ["park"]],
        rendered=[["park"], ["park"]],
    )
    nn.era_render_language = "old-english"
    assert str(nn) == "Park Dael"  # re-picked AND era-rendered (era reflex of dale)


def test_diversify_era_active_falls_back_when_no_reflex():
    """wyrd-68ye: when an era render is active but the synonym/re-pick morpheme has
    NO reflex at that era (the common case — most morphemes lack era reflexes), the
    grafted/re-picked slot falls back to the prior form, NOT empty. Site (a) → raw
    synonym form; site (b) → native form. Pins the `... or form`/`... or native`
    fallback so an era render never blanks a resolved repeat."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    # site (a): synonym has reflexes only for a DIFFERENT era → no oe reflex.
    hill = Meaning("hill", tags=[], meanings=["hill"], sources={"old_english": ["hill"]})
    norse = Meaning(
        "hill",
        tags=[],
        meanings=["Hill"],
        sources={"old_scandinavian": ["haeth"]},
        era_reflexes={"middle-english": [("haethe", "src")]},
    )
    a = NewName(struct=None, meaning_db={"hill": [hill, norse]}, name=[["hill"], ["hill"]])
    a.era_render_language = "old-english"
    assert str(a) == "Hill Haeth"  # falls back to the raw synonym form, not blank

    # site (b): re-pick morpheme has no reflex at the active era → native fallback.
    park = Meaning("park", tags=[], meanings=["park"], sources={"old_english": ["park"]})
    dale = Meaning("dale", tags=[], meanings=["valley"], sources={"old_english": ["dale"]})
    b = NewName(
        struct=None,
        meaning_db={"park": [park], "dale": [dale]},
        name=[["park"], ["park"]],
        rendered=[["park"], ["park"]],
    )
    b.era_render_language = "old-english"
    assert str(b) == "Park Dale"  # native/modern fallback (dale has no oe reflex)


def test_diversify_repeats_leaves_dupe_when_nothing_else_available():
    """wyrd-72q9: a repeat with NO cross-language synonym AND no other
    same-position morpheme to re-pick is left as-is (last resort)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    park = Meaning("park", tags=[], meanings=["park"], sources={"old_english": ["park"]})
    nn = NewName(struct=None, meaning_db={"park": [park]}, name=[["park"], ["park"]])
    assert str(nn) == "Park Park"  # only 'park' exists — nothing else to pick


def test_diversify_repeats_repicks_different_morpheme_when_no_synonym():
    """wyrd-72q9: a repeat with NO cross-language synonym is REJECTED and
    re-picked as a different same-position morpheme ("Park Park" → "Park Dale")
    rather than left as a duplicate."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    park = Meaning("park", tags=[], meanings=["park"], sources={"old_english": ["park"]})
    dale = Meaning("dale", tags=[], meanings=["valley"], sources={"old_english": ["dale"]})
    meaning_db = {"park": [park], "dale": [dale]}
    nn = NewName(struct=None, meaning_db=meaning_db, name=[["park"], ["park"]])
    assert str(nn) == "Park Dale"  # second re-picked to the only other bare morpheme
    d = nn.to_dict()
    assert d["words"][1][0]["usage"] == "dale"  # breakdown follows


def test_modern_name_uses_override_modern_usage_after_diversification():
    """wyrd-24s6 (D41): for a cross-language-synonym diversification, the NATIVE
    render shows the synonym's source form ('Haeth') but the MODERN companion
    shows the override Meaning's OWN modern usage. Here the Norse synonym is
    bucketed under 'hill' (it glosses 'hill'), so its modern usage is 'hill' —
    modern_name() collapses to 'Hill Hill'. The native render keeps the OE/ON
    distinction the modern English reflexes don't. (Documents the known
    name-level vs per-morpheme modern gap the general reviewer flagged.)"""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    hill = Meaning("hill", tags=[], meanings=["hill"], sources={"old_english": ["hill"]})
    norse = Meaning("hill", tags=[], meanings=["Hill"], sources={"old_scandinavian": ["haeth"]})
    nn = NewName(struct=None, meaning_db={"hill": [hill, norse]}, name=[["hill"], ["hill"]])
    assert str(nn) == "Hill Haeth"  # native: source forms, diversified
    assert nn.modern_name() == "Hill Hill"  # modern: override's own usage ('hill')


def test_diversify_repick_renders_native_form_when_rendered_active():
    """wyrd-24s6 (D41) Branch A: a re-pick that happens while a native render is
    active renders the REPLACEMENT in its own native (source) form, and the
    modern companion uses the replacement's modern usage. Pre-D41 the re-picked
    slot's rendered was left None (→ modern surface even in the native render)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    park = Meaning("park", tags=[], meanings=["park"], sources={"old_english": ["park"]})
    park.morpheme_id = "old-english:park"
    dale = Meaning("dale", tags=[], meanings=["valley"], sources={"old_english": ["dale"]})
    dale.morpheme_id = "old-english:dæl"
    meaning_db = {"park": [park], "dale": [dale]}
    # rendered pre-set (native render already ran) to the colliding 'pearroc'/park
    # native form for both slots; diversification re-picks slot 1 to 'dale'.
    nn = NewName(
        struct=None,
        meaning_db=meaning_db,
        name=[["park"], ["park"]],
        rendered=[["pearroc"], ["pearroc"]],
    )
    assert str(nn) == "Pearroc Dæl"  # re-pick rendered in its NATIVE form (dæl)
    assert nn.modern_name() == "Park Dale"  # modern uses the replacement's modern usage
    assert nn.name[1][0] == "dale"  # name key swapped to the re-pick


def test_diversify_native_collision_falls_back_to_modern():
    """wyrd-24s6 (D41) Branch B: two DISTINCT morphemes whose NATIVE forms
    collide ('Biscop Biscop') but whose modern usages differ — with no synonym
    and no re-pick available — break the native duplicate by falling one slot
    back to its modern usage. The modern companion (always distinct here) is
    unaffected."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m1 = Meaning("biscop", tags=[], meanings=["bishop"], sources={"old_english": ["biscop"]})
    m2 = Meaning("bishop", tags=[], meanings=["bishop"], sources={"old_english": ["biscop"]})
    nn = NewName(
        struct=None,
        meaning_db={"biscop": [m1], "bishop": [m2]},
        name=[["biscop"], ["bishop"]],
        rendered=[["Biscop"], ["Biscop"]],  # native render collided on 'Biscop'
    )
    assert str(nn) == "Biscop Bishop"  # collision broken — slot 2 fell back to modern
    assert nn.rendered[1][0] is None  # the colliding native render was dropped
    assert nn.modern_name() == "Biscop Bishop"  # modern was always distinct
    # D44: pass 2 is render-ONLY — a render collision must never rewrite the
    # picked morphemes (name keys), only fall a slot's render back to modern.
    assert nn.name == [["biscop"], ["bishop"]]


def test_break_native_duplicate_falls_back_first_slot_and_stale_rseen_is_safe():
    """The reversed-role collision: the SECOND slot's modern equals the
    collision fold ('biscop' — an archaic morpheme whose modern == its
    native), so _break_native_duplicate falls the FIRST slot back to
    modern instead. The three-way tail pins the stale-rseen safety: the
    third collider pairs against the broken slot 1, which the
    rendered-None candidate guard skips, and correctly leaves the
    genuine biscop/biscop dupe — all without the skeleton ever changing
    (D44 pass 2 is render-only)."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    m1 = Meaning("bishop", tags=[], meanings=["bishop"], sources={"old_english": ["biscop"]})
    m2 = Meaning("biscop", tags=[], meanings=["bishop"], sources={"old_english": ["biscop"]})
    m3 = Meaning("biscop", tags=[], meanings=["bishop"], sources={"old_english": ["biscop"]})
    nn = NewName(
        struct=None,
        meaning_db={"bishop": [m1], "biscop": [m2, m3]},
        name=[["bishop"], ["biscop"], ["biscop"]],
        rendered=[["Biscop"], ["Biscop"], ["Biscop"]],  # three-way native collision
    )
    rendered_out = str(nn)
    # Slot 1 (modern 'bishop' != fold 'biscop') is the breakable one — it
    # fell back to modern; slot 2 stayed native (its modern == the fold).
    assert nn.rendered[0][0] is None
    assert rendered_out.split(" ")[0] == "Bishop"
    # Slots 2 + 3 keep their native renders: both genuinely ARE 'biscop'
    # (modern == fold → unbreakable), so the dupe between them is left.
    assert nn.rendered[1][0] == "Biscop"
    assert nn.rendered[2][0] == "Biscop"
    # Render-only: the picked morphemes never change.
    assert nn.name == [["bishop"], ["biscop"], ["biscop"]]


def test_diversify_repeats_skips_base_pool_excluded_repick():
    """wyrd-57d8: a base-pool-excluded morpheme (here a synthesized manorial
    subject, same bare position + same language as the dupe) is NOT a valid
    re-pick target. With no other candidate, the dupe is left as-is rather than
    grafting in the manorial 'Malet' — the original 'Corner Malet' leak, where
    the re-pick read meaning_db raw and admitted it."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    park = Meaning("park", tags=[], meanings=["park"], sources={"old_english": ["park"]})
    malet = Meaning(
        "Malet",
        tags=["manorial", "norman"],
        meanings=["Norman manorial family: Malet"],
        sources={"old_english": ["malet"]},
    )
    meaning_db = {"park": [park], "Malet": [malet]}
    nn = NewName(struct=None, meaning_db=meaning_db, name=[["park"], ["park"]])
    # 'Malet' is the only other bare same-language morpheme, but it's excluded —
    # so the dupe stays rather than becoming "Park Malet".
    assert str(nn) == "Park Park"


def test_diversify_repeats_keeps_mixed_key_with_surviving_place_sense():
    """wyrd-57d8 carve-out: a usage key whose senses include BOTH an excluded
    manorial subject AND a legitimate place element stays eligible for re-pick —
    only keys whose EVERY sense is base-pool-excluded are dropped."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import NewName

    park = Meaning("park", tags=[], meanings=["park"], sources={"old_english": ["park"]})
    dale_place = Meaning("dale", tags=[], meanings=["valley"], sources={"old_english": ["dale"]})
    dale_manor = Meaning(
        "dale",
        tags=["manorial", "norman"],
        meanings=["Norman manorial family: Dale"],
        sources={"old_english": ["dale"]},
    )
    meaning_db = {"park": [park], "dale": [dale_place, dale_manor]}
    nn = NewName(struct=None, meaning_db=meaning_db, name=[["park"], ["park"]])
    # 'dale' survives because its plain place sense is not excluded.
    assert str(nn) == "Park Dale"


# wyrd-5z5j force-structure: structure label <-> key round-trip + listing.
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


# --- NameGenerator._render_substitutions + NewName.__str__ rendered branch ----


def _build_minimal_name_generator(meaning_db):
    """Build a NameGenerator with a single trivial structure for tests.

    The fixture structure is a 1-word compound `((("pre",), ("post",)),)`
    — one word with two element slots — which passes the wyrd-zzli +
    wyrd-80ib grammaticality filter (multi-element words don't trip
    :func:`_is_ungrammatical_word_template`). The tests using this
    helper exercise the rendering machinery via direct
    ``_render_substitutions`` calls and never run the
    structure-driven dispatch, so the struct doesn't need to match
    ``meaning_db`` — it just needs ONE entry that survives the
    filter at NameGenerator init.

    MeaningGenerator.__init__ already calls ``load_parts(proportions)``
    so the bare-location bucket is populated by construction. The
    additional ``load_parts(proportions, "single")`` call here populates
    the ``(location, "single")`` bucket flavor for symmetry with the
    production loader (which also calls both forms). Tests under this
    helper that need single-element-word resolution should switch to
    an inline NameGenerator construction with a name-flagged single-
    element struct.
    """
    from wyrd.generators.kenning.runtime.proportions import (
        MeaningGenerator,
        NameGenerator,
    )

    proportions = dict.fromkeys(meaning_db, 1)
    mg = MeaningGenerator(meaning_db, {}, proportions)
    mg.load_parts(proportions, "single")
    structs = {((("pre",), ("post",)),): 1}
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


def test_kenning_input_schema_era_options_match_family_stage_order_per_culture():
    """wyrd-rogd.2: the per-culture option set must equal the culture's family
    STAGE order (compressed canonical languages) plus the empty 'no filter'
    option — NOT the raw cells. Drives the source-of-truth invariant: the
    Configure dropdown offers the same axis as the col-3 grid, and a stage
    resolves to the union range of its cells in the generate path."""
    from wyrd.generators.kenning import _CULTURE_TO_ERA_FAMILY, Kenning
    from wyrd.generators.kenning.era.cells import family_stage_order

    schema = Kenning().input_schema()
    era_prop = schema["properties"]["era"]
    options = era_prop["x-options-by-culture"]
    assert era_prop.get("x-option-language") is True  # SPA labels via languageLabel
    for culture, family in _CULTURE_TO_ERA_FAMILY.items():
        expected = ("", *family_stage_order(family))
        assert tuple(options[culture]) == expected, (
            f"{culture} options must equal family_stage_order({family!r}) plus empty"
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

    from wyrd.generators.kenning import _resolve_stratum_param

    with pytest.raises(ValueError, match="native-welch"):
        _resolve_stratum_param("native-welch", "welsh")  # typo


def test_resolve_stratum_param_falls_back_to_all_strata_for_unrestricted_culture():
    """Cultures with no per-culture restriction fall back to the
    ALL_STRATA typo-check. Since wyrd-de77 every real culture has a
    restriction (the Celtic-family classifiers gave irish / breton
    their allowed-sets), so this fallback path is now exercised only
    by an unknown culture (dict.get's frozenset() default). Valid
    cross-family strata pass through; typos still raise."""

    from wyrd.generators.kenning import _resolve_stratum_param

    # 'native-welsh' is in ALL_STRATA — passes through for an
    # unrestricted (unknown) culture.
    assert _resolve_stratum_param("native-welsh", "klingon") == "native-welsh"
    # 'frankish-substrate' — same.
    assert _resolve_stratum_param("frankish-substrate", "klingon") == "frankish-substrate"
    # Typo — caught by ALL_STRATA fallback.
    with pytest.raises(ValueError, match="lattin-loan"):
        _resolve_stratum_param("lattin-loan", "klingon")


def test_resolve_stratum_param_accepts_celtic_family_strata_for_irish_breton():
    """wyrd-de77: irish / breton now have non-empty per-culture
    restrictions (Goidelic / Brittonic / Celtic classifiers shipped),
    so their family strata pass through and out-of-family values are
    rejected as culturally incoherent."""

    from wyrd.generators.kenning import _resolve_stratum_param

    # irish = Goidelic + Celtic.
    assert _resolve_stratum_param("native-goidelic", "irish") == "native-goidelic"
    assert _resolve_stratum_param("old-irish", "irish") == "old-irish"
    assert _resolve_stratum_param("native-celtic", "irish") == "native-celtic"
    # breton = Brittonic + Celtic + French.
    assert _resolve_stratum_param("native-brittonic", "breton") == "native-brittonic"
    assert _resolve_stratum_param("frankish-substrate", "breton") == "frankish-substrate"
    # Out-of-family now rejected (was previously allowed via the
    # ALL_STRATA fallback when irish/breton were unrestricted).
    with pytest.raises(ValueError, match="east-norse"):
        _resolve_stratum_param("east-norse", "irish")


def test_kenning_generate_propagates_resolve_stratum_error():
    """End-to-end: bad --stratum surfaces from Kenning.generate as
    a ValueError. The CLI's existing ValueError handler converts it
    to a friendly stderr message + non-zero exit."""

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
    gets the ALL_STRATA typo-check fallback via dict.get's
    frozenset() default. Since wyrd-de77 every real culture has a
    restriction, so an unknown culture (klingon) is the only remaining
    way to reach this empty-set branch."""

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
    assert _resolve_stratum_param(default, "irish") is None  # empty == no filter


def test_kenning_input_schema_stratum_options_match_per_culture_allowed_set():
    """The schema's per-culture stratum lists must match what
    _resolve_stratum_param accepts — so picking a value from the
    SPA dropdown can't surface as a 4xx at submit time. Pin a
    representative culture (welsh) + a Celtic-family culture (irish,
    wyrd-de77 gave it a Goidelic + Celtic restriction)."""
    from wyrd.generators.kenning import (
        BRITTONIC_STRATA,
        CELTIC_STRATA,
        FRENCH_STRATA,
        GOIDELIC_STRATA,
        WELSH_STRATA,
        Kenning,
    )

    schema = Kenning().input_schema()
    options = schema["properties"]["stratum"]["x-options-by-culture"]

    # wyrd-de77: welsh allows WELSH + FRENCH + BRITTONIC + CELTIC strata;
    # sorted by the schema helper.
    welsh_expected = [""] + sorted(
        set(WELSH_STRATA + FRENCH_STRATA + BRITTONIC_STRATA + CELTIC_STRATA)
    )
    assert options["welsh"] == welsh_expected

    # Irish allows GOIDELIC + CELTIC strata.
    irish_expected = [""] + sorted(set(GOIDELIC_STRATA + CELTIC_STRATA))
    assert options["irish"] == irish_expected


# --- wyrd-lr4 Phase 3 stratum filter --------------------------------------


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


def test_kenning_input_schema_exposes_cohesion():
    """The cohesion knob surfaces in input_schema so the SPA renders a
    slider. Default 0.0, range [0,1], number type."""
    from wyrd.generators.kenning import Kenning

    schema = Kenning().input_schema()
    assert "cohesion" in schema["properties"]
    prop = schema["properties"]["cohesion"]
    assert prop["type"] == "number"
    assert (
        prop["default"] == 0.0
    )  # schema default stays 0 (bit-stable); env override sets prod/staging
    assert prop["minimum"] == 0.0
    assert prop["maximum"] == 1.0
    # wyrd-e2b4: must render as a slider in the SPA (Field.svelte gates on this).
    assert prop["x-ui-widget"] == "slider"


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


def test_kenning_generate_cohesion_one_diverges_from_zero():
    """wyrd-e2b4: cohesion is no longer a no-op. Generating English with
    cohesion=1 differs from cohesion=0 on at least one seed. Pre-fix the
    flat ``{"a|b": count}`` table was handed to a multiplier that indexed
    it as nested ``{"a": {"b": prob}}``, so every lookup missed and the
    multiplier always returned 1.0 (byte-identical output regardless of
    cohesion). This pins the inverse so a regression back to the no-op
    fails loudly."""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    # wyrd-yzul: 20 seeds suffices — cohesion=1 diverges from 0 on ~65% of seeds,
    # so P(zero diffs) ≈ 0.35**20 ≈ 8e-10. The test only needs ONE divergence to
    # disprove the no-op regression.
    diffs = sum(
        k.generate({"culture": "english", "cohesion": 0.0}, seed=s).result
        != k.generate({"culture": "english", "cohesion": 1.0}, seed=s).result
        for s in range(20)
    )
    assert diffs > 0


def test_kenning_generate_cohesion_one_is_deterministic():
    """wyrd-e2b4: cohesion>0 still honours the ``(params, seed) → same
    output`` contract. The cohesion path sums float co-occurrence probs over
    tag sets; ``_cohesion_multiplier`` sorts both tag sets so set-iteration
    order (PYTHONHASHSEED-dependent) can't perturb the sum. Repeat the same
    seed and assert byte-identical output — a guard on that sort."""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    params = {"culture": "english", "cohesion": 1.0}
    runs = [[k.generate(params, seed=s).result for s in range(20)] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_kenning_generate_cohesion_preserves_diversity_non_english():
    """wyrd-e2b4: the prod default (WYRD_DEFAULT_COHESION=0.6) rests on cohesion
    staying diversity-safe across cultures, not just english. Generate a
    non-english culture at full cohesion=1.0 and assert the output does not
    collapse to a handful of repeated names — the mean-normalized multiplier
    redistributes a slot's weight without pruning the pool down to one pick.
    (Empirically ~187/200 distinct at cohesion=1; the floor here is
    conservative against seed variance while still catching a real collapse,
    which would drop distinct into the single digits.)"""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    names = [
        k.generate({"culture": "welsh", "count": 1, "cohesion": 1.0}, seed=s).result
        for s in range(60)
    ]
    distinct = len(set(names))
    assert distinct >= 40, f"cohesion=1.0 collapsed welsh diversity: {distinct}/60 distinct"


def test_load_proportions_handles_missing_cooccurrence_keys():
    """test-coverage P3: legacy bundles without the new tag_cooccurrence
    key must load cleanly — load_proportions defaults it to an empty dict
    and the resulting NameGenerator has the no-op cohesion path. Pinned at
    the loader to catch silent rename drift between bundle keys and the
    loader's `.get(...)` lookup."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import load_proportions

    # A 2-morpheme compound (pre+post) survives both the wyrd-zzli/80ib
    # grammaticality filter AND the wyrd-g1hj single-morpheme filter; the test
    # is about loader plumbing, not structure validity.
    meaning_db = {"-a": [Meaning("-a", [], [], {})]}
    tag_db = {}
    legacy_data = {
        "usages": {"-a": 1},
        "single_usages": {"-a": 1},
        "structures": [
            {"proportion": 1, "words": [[{"location": "pre"}, {"location": "post", "name": True}]]}
        ],
        # No tag_cooccurrence — legacy shape.
    }
    name_gen = load_proportions(legacy_data, meaning_db, tag_db)
    assert name_gen.tag_cooccurrence == {}
    # No counts + no marginal → empty derived table → bit-stable no-op cohesion.
    assert name_gen.cohesion_table == {}


def test_load_proportions_passes_cooccurrence_keys_through():
    """test-coverage P3: present-keys path. Confirms the loader actually
    reads from the bundle dict and threads to NameGenerator."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import load_proportions

    # A 2-morpheme compound (pre+post) survives both the wyrd-zzli/80ib
    # grammaticality filter AND the wyrd-g1hj single-morpheme filter; the test
    # is about loader plumbing, not structure validity.
    meaning_db = {"-a": [Meaning("-a", [], [], {})]}
    tag_db = {}
    data = {
        "usages": {"-a": 1},
        "single_usages": {"-a": 1},
        "structures": [
            {"proportion": 1, "words": [[{"location": "pre"}, {"location": "post", "name": True}]]}
        ],
        "tag_cooccurrence": {"water|plant": 7},
        "tag_marginal": {"water": 7, "plant": 7},
    }
    name_gen = load_proportions(data, meaning_db, tag_db)
    assert name_gen.tag_cooccurrence == {"water|plant": 7}
    # tag_marginal must also thread through (it's the normalizer denominator).
    assert name_gen.tag_marginal == {"water": 7, "plant": 7}


def test_load_proportions_builds_nested_cohesion_table():
    """wyrd-e2b4: the loader normalizes the flat counts + ``tag_marginal``
    into the nested conditional-probability table the cohesion multiplier
    consumes. Raw counts stay on ``.tag_cooccurrence`` (the JSON↔SQLite
    round-trip surface); the derived nested table lands on
    ``.cohesion_table`` keyed ``{candidate: {prior: prob}}``."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import load_proportions

    meaning_db = {"-a": [Meaning("-a", [], [], {})]}
    tag_db = {}
    data = {
        "usages": {"-a": 1},
        "single_usages": {"-a": 1},
        "structures": [
            {"proportion": 1, "words": [[{"location": "pre"}, {"location": "post", "name": True}]]}
        ],
        # "water|plant" = water (earlier/prior) → plant (later/candidate).
        "tag_cooccurrence": {"water|plant": 7},
        "tag_marginal": {"water": 7, "plant": 7},
    }
    name_gen = load_proportions(data, meaning_db, tag_db)
    # Inverted to candidate-keyed, normalized by the prior tag's marginal:
    # P(plant | water) = 7 / marginal[water] = 7 / 7 = 1.0.
    assert name_gen.cohesion_table == {"plant": {"water": 1.0}}


class TestBuildCohesionTable:
    """wyrd-e2b4: ``build_cohesion_table`` turns the bundle's flat tag
    co-occurrence counts + per-tag marginals into the nested conditional-
    probability table the vector path's ``_cohesion_multiplier`` reads."""

    def test_conditional_normalization_and_inversion(self):
        from wyrd.generators.kenning.runtime.proportions import build_cohesion_table

        # "a|b" = a (prior/earlier) → b (candidate/later). P(b|a) = 3/6 = 0.5.
        table = build_cohesion_table({"a|b": 3}, {"a": 6, "b": 6})
        assert table == {"b": {"a": 0.5}}

    def test_all_probs_bounded_and_directional(self):
        from wyrd.generators.kenning.runtime.proportions import build_cohesion_table

        flat = {"water|plant": 5, "water|tree": 2, "plant|water": 1}
        marg = {"water": 10, "plant": 4, "tree": 3}
        table = build_cohesion_table(flat, marg)
        # candidate-keyed, prior-valued
        assert table["plant"]["water"] == 0.5  # 5 / marginal[water]=10
        assert table["tree"]["water"] == 0.2  # 2 / 10
        assert table["water"]["plant"] == 0.25  # 1 / marginal[plant]=4
        assert all(0.0 <= p <= 1.0 for inner in table.values() for p in inner.values())

    def test_clamps_to_one_on_marginal_inconsistency(self):
        from wyrd.generators.kenning.runtime.proportions import build_cohesion_table

        # 10 / 5 = 2.0 would exceed 1.0; the clamp keeps each cell a probability
        # in [0,1] so _cohesion_raw's per-candidate mean stays bounded.
        table = build_cohesion_table({"a|b": 10}, {"a": 5})
        assert table == {"b": {"a": 1.0}}

    def test_missing_inputs_return_empty(self):
        from wyrd.generators.kenning.runtime.proportions import build_cohesion_table

        assert build_cohesion_table(None, None) == {}
        assert build_cohesion_table({}, {}) == {}
        assert build_cohesion_table({"a|b": 1}, {}) == {}  # no marginals
        assert build_cohesion_table({}, {"a": 1}) == {}  # no co-occurrence

    def test_skips_malformed_keys_and_missing_denominators(self):
        from wyrd.generators.kenning.runtime.proportions import build_cohesion_table

        # "nopipe" has no separator → skipped (no bare-string mis-key).
        # "x|y" has no marginal for x (denom 0) → skipped.
        table = build_cohesion_table({"nopipe": 5, "x|y": 4, "a|b": 2}, {"nopipe": 5, "a": 4})
        assert table == {"b": {"a": 0.5}}


def test_new_name_str_title_cases_lowercase_morpheme_compound() -> None:
    """wyrd-3xdb: a word built entirely from lowercase morphemes
    (post-modifiers / inner-modifiers) gets its first letter title-
    cased so the output reads as a proper noun. Pre-PR behavior:
    'holmr' + 'wood' -> 'holmrwood' (lowercase, broken). Post-PR:
    'Holmrwood'."""
    from wyrd.generators.kenning.runtime.proportions import NewName

    new_name = NewName(struct=None, meaning_db={}, name=[["-holmr", "-wood"]])
    assert str(new_name) == "Holmrwood"


def test_new_name_str_title_cases_each_word_in_multi_word_name() -> None:
    """wyrd-3xdb: multi-word names get title-casing applied per word.
    'ward' + ' ' + 'green' -> 'Ward Green', not 'ward green'. Pins
    the per-word capitalization invariant that fixes the demo's
    'ward green' / 'holmrwood common' broken outputs."""
    from wyrd.generators.kenning.runtime.proportions import NewName

    new_name = NewName(struct=None, meaning_db={}, name=[["-ward"], ["green"]])
    assert str(new_name) == "Ward Green"


def test_new_name_str_preserves_already_capitalized_word() -> None:
    """wyrd-3xdb: a word that already opens with an uppercase
    letter passes through unchanged (ch.upper() == ch for already-
    uppercase letters). 'Hunlock' + 'Forest' stays 'Hunlock Forest',
    NOT mangled."""
    from wyrd.generators.kenning.runtime.proportions import NewName

    new_name = NewName(struct=None, meaning_db={}, name=[["Hunlock"], ["Forest"]])
    assert str(new_name) == "Hunlock Forest"


def test_new_name_str_skips_none_chunks_in_word() -> None:
    """wyrd-3xdb regression: word slots can carry None entries
    (when a joiner slot is empty). The renderer must skip None
    without crashing or emitting a stray space, and the title-
    casing still applies to the first non-None chunk's first
    letter."""
    from wyrd.generators.kenning.runtime.proportions import NewName

    new_name = NewName(struct=None, meaning_db={}, name=[["bridge-", None, "-water"]])
    assert str(new_name) == "Bridgewater"


def test_new_name_str_uses_rendered_when_present() -> None:
    """wyrd-3xdb: the optional ``rendered`` parameter (D18 variant /
    D8 inflection substitution) still wins over the dash-stripped
    morpheme string, and the substituted form gets the same title-
    case treatment. ``rendered`` strings flow through verbatim
    (legacy behavior — no dash stripping; rendered values are
    expected to already be in surface form); the title-case only
    touches the first letter of the concatenated word."""
    from wyrd.generators.kenning.runtime.proportions import NewName

    new_name = NewName(
        struct=None,
        meaning_db={},
        name=[["cot-", "-an"]],
        rendered=[["cotum", "an"]],
    )
    assert str(new_name) == "Cotuman"


# ---- wyrd-gzvr: no adjacent-duplicate words (end-to-end) ----------------


def test_name_generator_no_adjacent_duplicate_words_across_a_seed_sweep() -> None:
    """wyrd-gzvr end-to-end: across a 40-seed × 5-culture sweep
    (count=10 each → ~2000 names) the generator produces zero outputs
    containing adjacent-duplicate words. Pre-PR scan surfaced 'North
    North' / 'East East' / 'Green Green' / 'Pentre Pentre' / 'Les Les'
    patterns — the exclude_keys filter eliminates them. (wyrd-yzul: 40
    seeds, down from 50; a returned regression removes the filter
    wholesale, so the duplicates flood back. The pre-fix rate was
    materially higher than the triple-letter case below, so 2000 names
    catch it with even more margin than that sibling's ~e^-9.6 P(miss).)"""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    duplicates: list[str] = []
    for culture in ("english", "welsh", "scottish", "irish", "breton"):
        for seed in range(1, 41):
            result = k.generate({"culture": culture, "count": 10}, seed=seed)
            for line in result.result.splitlines():
                line = line.strip()
                if not line:
                    continue
                words = line.split()
                for i in range(len(words) - 1):
                    if words[i] == words[i + 1]:
                        duplicates.append(f"{culture}/{seed}: {line}")
                        break
    assert not duplicates, (
        f"adjacent-duplicate words in {len(duplicates)} outputs: {duplicates[:5]}"
    )


def test_name_generator_no_adjacent_duplicate_words_under_tag_filter() -> None:
    """wyrd-gzvr: the surface-form dedup also applies on the multi-
    tag path (_select_tags / _select_tag). Pre-PR the dedup was only
    in _select_no_tag, so a call with --tag would still surface
    'X X' adjacents. Tag-filter restricts vocabulary so the
    probability of a dup is higher — making the gap especially
    visible at tag-driven generation."""
    from wyrd.generators.kenning import Kenning

    k = Kenning()
    duplicates: list[str] = []
    # Run a smaller sweep — tag-filtered generation is slower per
    # call and the bucket vocabularies are smaller, so 25 seeds is
    # enough to catch the regression if it returns.
    # wyrd-i7uy: these MUST be real corpus tags — a tag with no eligible
    # morphemes (e.g. the old "religion", which doesn't exist; it's "religious")
    # filters the pool empty and raises. Until the per-request cache fix, the
    # first valid tag's pool was wrongly reused for the rest, masking that.
    for tag in ("water", "religious", "topography"):
        for seed in range(1, 26):
            result = k.generate({"culture": "english", "tags": [tag], "count": 10}, seed=seed)
            for line in result.result.splitlines():
                line = line.strip()
                if not line:
                    continue
                words = line.split()
                for i in range(len(words) - 1):
                    if words[i] == words[i + 1]:
                        duplicates.append(f"tag={tag}/seed={seed}: {line}")
                        break
    assert not duplicates, (
        f"adjacent-duplicate words on tag-filtered path "
        f"in {len(duplicates)} outputs: {duplicates[:5]}"
    )


# ---- wyrd-ovsv: collapse triple-letter join artifacts ------------------


def test_collapse_triple_letters_collapses_triple_run_to_double() -> None:
    """wyrd-ovsv: a run of 3+ identical letters from morpheme join
    collapses down to 2. Pins 'Killlen' (Kill + llen) → 'Killen'."""
    from wyrd.generators.kenning.runtime.proportions import _collapse_triple_letters

    assert _collapse_triple_letters("Killlen") == "Killen"
    assert _collapse_triple_letters("Bronferrrhedyn") == "Bronferrhedyn"
    assert _collapse_triple_letters("Nettton") == "Netton"


def test_collapse_triple_letters_preserves_legitimate_doubles() -> None:
    """wyrd-ovsv conservative-fix invariant: morphemes that already
    have legitimate double letters (Kill, Ball, Cromwell, Newcastle)
    pass through unchanged when they stand alone."""
    from wyrd.generators.kenning.runtime.proportions import _collapse_triple_letters

    for word in ("Kill", "Ball", "Cromwell", "Newcastle", "Ellen", "Llanelli"):
        assert _collapse_triple_letters(word) == word


def test_collapse_triple_letters_collapses_quadruple_or_longer_runs() -> None:
    """wyrd-ovsv defensive: a 4+ run (e.g. Kill + llll) collapses
    all the way down to 2 in one pass. Pathological but pinned so a
    bundle edit that creates a 4+ join doesn't silently leak past
    the regex's quantifier."""
    from wyrd.generators.kenning.runtime.proportions import _collapse_triple_letters

    # 'Killlll' = K + i + 5 l's; collapse to K + i + 2 l's = 'Kill'.
    assert _collapse_triple_letters("Killlll") == "Kill"


def test_new_name_str_applies_triple_letter_collapse() -> None:
    """wyrd-ovsv end-to-end through __str__: a NewName whose
    morphemes join to produce a triple-letter run renders with the
    collapse applied."""
    from wyrd.generators.kenning.runtime.proportions import NewName

    new_name = NewName(struct=None, meaning_db={}, name=[["Kill-", "-llen"]])
    assert str(new_name) == "Killen"


def test_name_generator_no_triple_letter_runs_across_a_seed_sweep() -> None:
    """wyrd-ovsv end-to-end: across a 40-seed × 5-culture sweep
    (count=10 each → ~2000 names) the generator produces zero outputs
    containing 3+-letter runs. Pre-PR scan surfaced 12 such outputs
    (Killlen, Alllas, Bronferrrhedyn, ...) — the collapse pass
    eliminates them. (wyrd-yzul: 40 seeds, down from 50; the pre-fix
    density was ~12/2500 names, so a returned regression yields ~9.6
    hits across the 40-seed sweep — P(miss) ≈ e^-9.6 ≈ 0.007%. This
    zero-tolerance scan has no built-in slack, so it keeps a fatter
    sample than the rate-based tests in this PR.)"""
    import re as _re

    from wyrd.generators.kenning import Kenning

    triple = _re.compile(r"(.)\1\1")
    k = Kenning()
    violations: list[str] = []
    for culture in ("english", "welsh", "scottish", "irish", "breton"):
        for seed in range(1, 41):
            result = k.generate({"culture": culture, "count": 10}, seed=seed)
            for line in result.result.splitlines():
                line = line.strip()
                if line and triple.search(line):
                    violations.append(f"{culture}/{seed}: {line}")
    assert not violations, f"triple-letter runs in {len(violations)} outputs: {violations[:5]}"


def test_name_generator_no_triple_letter_runs_under_tag_filter() -> None:
    """wyrd-ovsv: the collapse pass also applies on the tag-filtered
    generation path (NewName.__str__ is shared). Tag-filter narrows
    vocabulary so collision probability rises — mirror the wyrd-gzvr
    pattern of pinning both paths."""
    import re as _re

    from wyrd.generators.kenning import Kenning

    triple = _re.compile(r"(.)\1\1")
    k = Kenning()
    violations: list[str] = []
    # wyrd-i7uy: these MUST be real corpus tags — a tag with no eligible
    # morphemes (e.g. the old "religion", which doesn't exist; it's "religious")
    # filters the pool empty and raises. Until the per-request cache fix, the
    # first valid tag's pool was wrongly reused for the rest, masking that.
    for tag in ("water", "religious", "topography"):
        for seed in range(1, 26):
            result = k.generate({"culture": "english", "tags": [tag], "count": 10}, seed=seed)
            for line in result.result.splitlines():
                line = line.strip()
                if line and triple.search(line):
                    violations.append(f"tag={tag}/seed={seed}: {line}")
    assert not violations, (
        f"triple-letter runs on tag-filtered path in {len(violations)} outputs: {violations[:5]}"
    )


def test_new_name_str_collapse_is_per_word_not_cross_space() -> None:
    """wyrd-ovsv: the collapse runs per-word at the morpheme-join step
    (BEFORE words are space-joined). A name where word 1 ends in 'll'
    and word 2 starts with 'l' renders as '... Ll...' (space-
    separated, no triple) — the space prevents the visual triple,
    so per-word collapse is sufficient. Pins the design intent so a
    future refactor that joins-then-collapses doesn't accidentally
    fire across word boundaries (which would alter both words'
    surface forms, e.g. 'Bell Lake' → '...wat? collapse weirdness')."""
    from wyrd.generators.kenning.runtime.proportions import NewName

    # Word 1 morphemes concatenate to 'Bell'; word 2 to 'Lake'. The
    # 'll' (word 1 end) + 'L' (word 2 start) is space-separated —
    # no triple to collapse. Output: 'Bell Lake' unchanged.
    new_name = NewName(struct=None, meaning_db={}, name=[["Bell-"], ["Lake-"]])
    assert str(new_name) == "Bell Lake"


def test_new_name_str_drops_fully_empty_words() -> None:
    """wyrd-3xdb edge case: a word slot containing only None entries
    (no surviving morpheme after joiner-pruning) emits nothing —
    not a stray space. The ``if w`` guard at the join boundary
    handles this. Mixed empty + populated words produce the
    populated word in isolation, no leading/trailing blanks."""
    from wyrd.generators.kenning.runtime.proportions import NewName

    # All-None word slot
    empty_only = NewName(struct=None, meaning_db={}, name=[[None, None]])
    assert str(empty_only) == ""

    # Empty word slot adjacent to a populated one
    mixed = NewName(struct=None, meaning_db={}, name=[[None], ["-wood"]])
    assert str(mixed) == "Wood"

    # Fully-empty input
    empty_all = NewName(struct=None, meaning_db={}, name=[])
    assert str(empty_all) == ""


def test_keep_keys_for_gloss_policy():
    """wyrd-glos: MeaningGenerator.keep_keys_for_gloss is the proportions-
    side twin of the vector pool filter — same _gloss_eligible predicate,
    so both scoring modes apply identical generation-pool rules."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    def m(usage, glossed):
        return Meaning(usage, [], (["g"] if glossed else []), {})

    meaning_db = {
        "-ton": [m("-ton", True)],  # glossed multi-char
        "-xyz": [m("-xyz", False)],  # unglossed multi-char
        "-q": [m("-q", False)],  # unglossed 1-char
        "á": [m("á", True)],  # glossed 1-char (survives)
    }
    mg = MeaningGenerator(meaning_db, {}, {})

    # Default: glossed-only. (wyrd-eyjk/D40: keep-sets keyed by bare SURFACE.)
    assert mg.keep_keys_for_gloss(include_unglossed=False) == frozenset({"ton", "á"})
    # Opt-in: + multi-char unglossed, never the 1-char unglossed.
    assert mg.keep_keys_for_gloss(include_unglossed=True) == frozenset({"ton", "xyz", "á"})


def test_keep_keys_for_gloss_full_coverage_returns_none():
    """All-glossed (or all glossed-or-multichar under the flag) bundle →
    keep-set covers every usage → None, the no-filter fast path, so a
    fully-glossed bundle stays bit-stable with the pre-wyrd-glos sampler."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator

    meaning_db = {
        "-ton": [Meaning("-ton", [], ["estate"], {})],
        "-ham": [Meaning("-ham", [], ["village"], {})],
    }
    mg = MeaningGenerator(meaning_db, {}, {})
    assert mg.keep_keys_for_gloss(include_unglossed=False) is None
    assert mg.keep_keys_for_gloss(include_unglossed=True) is None
