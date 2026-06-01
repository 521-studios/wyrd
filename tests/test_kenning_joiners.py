"""Tests for joiner schema + matcher hook (wyrd-q0g6 Phase 1).

Covers:
- Bundle schema: ``load_meanings`` accepts both legacy list shape AND
  new dict shape ``{"subjects": [...], "joiners": {...}}``.
- ``load_joiners``: extracts joiners dict from new shape, returns
  empty for legacy shape.
- ``Joiner`` sentinel: not str / not Meaning, surface preserved via
  __str__, equality by (surface, lang_field).
- Matcher hook: ``Name.find_meaning(joiners=...)`` consumes joiner
  forms in unaccounted slots; ``count_unaccounted`` drops the joiner
  chars; ``Word.__str__`` keeps the surface intact.
- No-op fast path: ``joiners=None`` (default) preserves bit-stable
  behavior for legacy callers.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.runtime.meaning import (
    Joiner,
    Meaning,
    _bundle_subjects,
    load_joiners,
    load_meanings,
)
from wyrd.generators.kenning.runtime.name import (
    Name,
    _build_joiner_lookup,
    _consume_joiners,
)

# --- Joiner sentinel ------------------------------------------------------


def test_joiner_is_not_str_or_meaning() -> None:
    """A Joiner is its OWN type, not a str or Meaning. The whole
    matcher-hook design depends on this discriminator: code that
    walks decompositions on the str-vs-Meaning axis must see Joiners
    as 'matched-but-no-semantic'."""
    j = Joiner("en", lang_field="old_english")
    assert not isinstance(j, str)
    assert not isinstance(j, Meaning)


def test_joiner_str_returns_surface() -> None:
    """``Joiner.__str__`` returns the consumed surface form so
    ``Word.__str__`` (which joins ``str(slot)`` over the slot list)
    keeps the joiner chars in the surface output."""
    j = Joiner("en", lang_field="old_english")
    assert str(j) == "en"


def test_joiner_equality_by_surface_and_lang() -> None:
    """Two joiners are equal iff their surface AND lang_field match.
    Distinct lang_fields with the same surface (rare but possible
    cross-family) should NOT compare equal — they're different
    runtime objects."""
    j1 = Joiner("en", lang_field="old_english")
    j2 = Joiner("en", lang_field="old_english")
    j3 = Joiner("en", lang_field="celtic_mix")
    assert j1 == j2
    assert j1 != j3
    assert hash(j1) == hash(j2)
    assert hash(j1) != hash(j3)


def test_joiner_repr_includes_lang_field() -> None:
    """``__repr__`` exposes both surface and lang_field for debugging
    decomposition output."""
    j = Joiner("en", lang_field="old_english")
    assert "en" in repr(j)
    assert "old_english" in repr(j)


# --- bundle shape parsing ------------------------------------------------


def test_bundle_subjects_passes_through_legacy_list() -> None:
    """Legacy bundles are bare lists of subjects. ``_bundle_subjects``
    returns the list unchanged."""
    legacy = [{"meaning": ["X"], "modifier_tags": [], "modifier_type": "T", "words": []}]
    assert _bundle_subjects(legacy) == legacy


def test_bundle_subjects_extracts_subjects_from_dict() -> None:
    """New-shape bundles wrap subjects in a dict; ``_bundle_subjects``
    pulls the list out so the rest of ``load_meanings`` doesn't need
    to know which shape was used."""
    new_shape = {
        "subjects": [
            {"meaning": ["X"], "modifier_tags": [], "modifier_type": "T", "words": []},
        ],
        "joiners": {},
    }
    assert _bundle_subjects(new_shape) == new_shape["subjects"]


def test_bundle_subjects_handles_dict_without_subjects() -> None:
    """A dict bundle missing ``subjects`` is treated as zero subjects
    (rather than raising). Defensive — keeps the loader robust against
    partial-export bundles."""
    assert _bundle_subjects({"joiners": {}}) == []


def test_load_meanings_accepts_dict_shape() -> None:
    """The full ``load_meanings`` pipeline runs against a dict bundle
    and produces the same meaning_db / tags_db it would for the
    equivalent list bundle. Pinned so a future bundle re-emit that
    switches to dict shape doesn't silently break the runtime
    loader."""
    subjects = [
        {
            "meaning": ["Bridge"],
            "modifier_tags": ["topography"],
            "modifier_type": "Topographical",
            "words": [{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        }
    ]
    list_word_db, list_tags = load_meanings(subjects)
    dict_word_db, dict_tags = load_meanings({"subjects": subjects, "joiners": {}})
    assert set(list_word_db.keys()) == set(dict_word_db.keys())
    assert list_tags == dict_tags


# --- load_joiners ---------------------------------------------------------


def test_load_joiners_returns_empty_for_legacy_list() -> None:
    """Legacy bundles can't carry joiners (no top-level dict).
    ``load_joiners`` returns an empty dict so the matcher's
    ``joiner_lookup`` short-circuits."""
    assert load_joiners([]) == {}
    assert (
        load_joiners([{"meaning": ["X"], "modifier_tags": [], "modifier_type": "T", "words": []}])
        == {}
    )


def test_load_joiners_returns_empty_for_dict_without_joiners_key() -> None:
    """A new-shape dict bundle that didn't bother to emit ``joiners``
    (e.g. Phase 1 export before any audit lands) reads as empty."""
    assert load_joiners({"subjects": []}) == {}


def test_load_joiners_parses_per_language_pool() -> None:
    """Joiner entries become ``(form, weight)`` tuples keyed by
    lang_field. Multiple languages, multiple forms each."""
    bundle = {
        "subjects": [],
        "joiners": {
            "old_english": [
                {"form": "en", "weight": 100},
                {"form": "es", "weight": 80},
            ],
            "celtic_mix": [
                {"form": "y", "weight": 50},
            ],
        },
    }
    joiners = load_joiners(bundle)
    assert joiners == {
        "old_english": [("en", 100), ("es", 80)],
        "celtic_mix": [("y", 50)],
    }


# --- _build_joiner_lookup ------------------------------------------------


def test_build_joiner_lookup_flattens_to_surface_to_lang() -> None:
    """The matcher only needs 'is this surface a registered joiner,
    and which family did it come from?'. Flatten the per-language
    pool into a single surface→lang lookup."""
    joiners = {
        "old_english": [("en", 100)],
        "celtic_mix": [("y", 50)],
    }
    lookup = _build_joiner_lookup(joiners)
    assert lookup == {"en": "old_english", "y": "celtic_mix"}


def test_build_joiner_lookup_lowercases_surface() -> None:
    """Surfaces are normalized to lowercase so the matcher's case-
    insensitive comparison hits."""
    joiners = {"old_english": [("EN", 100)]}
    lookup = _build_joiner_lookup(joiners)
    assert "en" in lookup


def test_build_joiner_lookup_first_lang_wins_on_collision() -> None:
    """Same surface in two languages — the FIRST lang_field encountered
    in iteration order wins. Phase 1 ships with zero populated joiners
    so the collision can't actually surface yet, but pin the policy
    so Phase 2 audit decisions are reviewable."""
    joiners = {
        "old_english": [("y", 100)],
        "celtic_mix": [("y", 50)],
    }
    lookup = _build_joiner_lookup(joiners)
    assert lookup["y"] == "old_english"


def test_build_joiner_lookup_handles_empty() -> None:
    """Empty / None inputs → empty lookup (caller short-circuits)."""
    assert _build_joiner_lookup(None) == {}
    assert _build_joiner_lookup({}) == {}
    assert _build_joiner_lookup({"old_english": []}) == {}


def test_build_joiner_lookup_skips_empty_form() -> None:
    """An empty-string form would otherwise match every str slot — drop
    it defensively at lookup-build time."""
    lookup = _build_joiner_lookup({"old_english": [("", 100), ("en", 50)]})
    assert lookup == {"en": "old_english"}


# --- _consume_joiners -----------------------------------------------------


def _meaning_for_anchor(usage: str = "Bridg-") -> Meaning:
    """Cheap Meaning instance for anchor-gate test fixtures."""
    return Meaning(usage, tags=[], meanings=[usage], sources={"old_english": ["brycg"]})


def test_consume_joiners_replaces_str_slot_when_sandwiched() -> None:
    """A str slot matching a joiner form becomes a Joiner ONLY when
    it sits between two Meaning slots (anchor gate)."""
    left = _meaning_for_anchor("Bridg-")
    right = _meaning_for_anchor("-water")
    decomp = [left, "en", right]
    consumed = _consume_joiners(decomp, {"en": "old_english"})
    assert consumed[0] is left
    assert isinstance(consumed[1], Joiner)
    assert consumed[1].surface == "en"
    assert consumed[1].lang_field == "old_english"
    assert consumed[2] is right


def test_consume_joiners_leaves_unmatched_strings_alone() -> None:
    """A str slot whose surface isn't in the joiner lookup stays a
    plain str — count_unaccounted will charge those chars, which is
    the correct behavior (genuinely unaccounted)."""
    left = _meaning_for_anchor("Bridg-")
    right = _meaning_for_anchor("-water")
    decomp = [left, "xyz", right]
    consumed = _consume_joiners(decomp, {"en": "old_english"})
    assert consumed[1] == "xyz"


def test_consume_joiners_no_op_with_empty_lookup() -> None:
    """Empty lookup means no joiners registered; the decomposition is
    returned unchanged."""
    left = _meaning_for_anchor("Bridg-")
    right = _meaning_for_anchor("-water")
    decomp = [left, "en", right]
    consumed = _consume_joiners(decomp, {})
    assert consumed == decomp


def test_consume_joiners_case_insensitive_match() -> None:
    """A str slot 'EN' between two Meanings matches a registered
    joiner 'en' — case independence matches the lookup's lowercase
    normalization."""
    left = _meaning_for_anchor("Bridg-")
    right = _meaning_for_anchor("-water")
    decomp = [left, "EN", right]
    consumed = _consume_joiners(decomp, {"en": "old_english"})
    assert isinstance(consumed[1], Joiner)
    assert consumed[1].surface == "EN"  # ORIGINAL case preserved


def test_consume_joiners_anchor_required_no_left_meaning() -> None:
    """A str matching a joiner form at position 0 (no left neighbor)
    stays unconsumed even when right is a Meaning. Phase 2 data alone
    could otherwise launder garbage: with joiner 'en' registered,
    'enbridge' would falsely report 0 unaccounted via leading
    Joiner('en') with no morphological evidence for it."""
    right = _meaning_for_anchor("Bridg-")
    consumed = _consume_joiners(["en", right], {"en": "old_english"})
    assert consumed[0] == "en"
    assert not isinstance(consumed[0], Joiner)


def test_consume_joiners_anchor_required_no_right_meaning() -> None:
    """Trailing joiner-form str with no right neighbor stays
    unconsumed."""
    left = _meaning_for_anchor("Bridg-")
    consumed = _consume_joiners([left, "en"], {"en": "old_english"})
    assert consumed[1] == "en"


def test_consume_joiners_anchor_required_both_str_neighbors() -> None:
    """A joiner-form str sandwiched between two unaccounted strs is
    NOT consumed — joiners connect morphemes, not unaccounted
    fragments."""
    consumed = _consume_joiners(["xx", "en", "yy"], {"en": "old_english"})
    assert consumed == ["xx", "en", "yy"]


def test_consume_joiners_anchor_required_joiner_neighbor_does_not_count() -> None:
    """A Joiner adjacent to another joiner-form str does NOT satisfy
    the Meaning-anchor requirement — only Meaning instances anchor.
    Pinned so a future ``isinstance(slot, str)`` change doesn't
    inadvertently let consumed-Joiner anchors cascade."""
    # Construct a synthetic decomposition with a pre-existing Joiner
    # next to a joiner-form str.
    left_joiner = Joiner("en", lang_field="old_english")
    consumed = _consume_joiners([left_joiner, "es"], {"es": "old_english"})
    assert consumed[1] == "es"
    assert not isinstance(consumed[1], Joiner)


# --- find_meaning integration --------------------------------------------


def _word_db_with_two_morphemes() -> dict:
    """word_db with two real morphemes: Bridge- and -water."""
    subjects = [
        {
            "meaning": ["Bridge"],
            "modifier_tags": [],
            "modifier_type": "T",
            "words": [{"modern_usage": "Bridge-", "old_english": ["brycg"]}],
        },
        {
            "meaning": ["Water"],
            "modifier_tags": [],
            "modifier_type": "T",
            "words": [{"modern_usage": "-water", "old_english": ["wæter"]}],
        },
    ]
    word_db, _ = load_meanings(subjects)
    return word_db


def test_find_meaning_no_joiners_kwarg_is_bit_stable() -> None:
    """Calling ``find_meaning(word_db)`` without ``joiners`` produces
    the identical decomposition shape as before this change. Pinned
    so the no-joiners default path stays bit-stable."""
    word_db = _word_db_with_two_morphemes()
    name = Name("Bridgexywater")  # 'xy' between Bridge and water is unaccounted
    name.find_meaning(word_db, reduce=False)
    # Verify NO Joiner sentinels appeared in the decomposition.
    for words in name.words.values():
        for w in words:
            assert not any(isinstance(slot, Joiner) for slot in w.word)


def test_find_meaning_with_joiners_consumes_matching_str() -> None:
    """When the unaccounted bridge between two morphemes matches a
    registered joiner form, the matcher post-processing converts it
    to a Joiner. count_unaccounted drops to zero for that slot."""
    word_db = _word_db_with_two_morphemes()
    joiners = {"old_english": [("xy", 100)]}
    name = Name("Bridgexywater")
    name.find_meaning(word_db, reduce=False, joiners=joiners)
    # At least one decomposition should carry a Joiner instance now.
    found_joiner = False
    for words in name.words.values():
        for w in words:
            for slot in w.word:
                if isinstance(slot, Joiner) and slot.surface == "xy":
                    found_joiner = True
                    assert slot.lang_field == "old_english"
    assert found_joiner


def test_find_meaning_joiner_drops_unaccounted_count() -> None:
    """A joiner-matched 'xy' bridge contributes ZERO to the word's
    ``count_unaccounted`` (Joiner is non-str so the sum-of-len
    accumulator skips it). Without joiners the same slot would
    contribute 2."""
    word_db = _word_db_with_two_morphemes()
    name_no_joiners = Name("Bridgexywater")
    name_no_joiners.find_meaning(word_db, reduce=False)
    base_unaccounted = name_no_joiners.count_unaccounted()
    assert base_unaccounted >= 2  # 'xy' would be unaccounted

    name_with_joiners = Name("Bridgexywater")
    name_with_joiners.find_meaning(word_db, reduce=False, joiners={"old_english": [("xy", 100)]})
    joiner_unaccounted = name_with_joiners.count_unaccounted()
    # The joiner-matched decomposition has fewer unaccounted chars
    # than the equivalent no-joiner decomposition. (Other parses with
    # different unaccounted shapes may still be in the result set; the
    # min-unaccounted parse is what reduce=True would pick.)
    assert joiner_unaccounted < base_unaccounted


def test_find_meaning_preserves_surface_through_word_str() -> None:
    """The joiner's chars stay in ``Word.__str__``'s output — Joiner
    is non-str but its ``__str__`` returns the surface, so the surface
    representation of 'Bridgexywater' isn't truncated to 'Bridgewater'
    by joiner consumption."""
    word_db = _word_db_with_two_morphemes()
    name = Name("Bridgexywater")
    name.find_meaning(word_db, reduce=False, joiners={"old_english": [("xy", 100)]})
    surfaces: list[str] = []
    for words in name.words.values():
        for w in words:
            surfaces.append(str(w))
    # Every emitted surface still contains the joiner chars.
    assert all("xy" in s.lower() for s in surfaces)


def test_find_meaning_with_reduce_picks_joiner_decomposition_when_better() -> None:
    """``reduce=True`` keeps only canonical (lowest-unaccounted)
    decompositions. With joiners, a parse that consumed the bridge
    chars beats a parse that left them unaccounted — pin this so a
    future Phase 2.5 'joiner-aware canonical scoring' refactor doesn't
    regress the empirical effect."""
    word_db = _word_db_with_two_morphemes()
    name = Name("Bridgexywater")
    name.find_meaning(word_db, reduce=True, joiners={"old_english": [("xy", 100)]})
    # The reduce-mode pick should have ZERO unaccounted (joiner
    # consumed 'xy', morphemes covered the rest).
    assert name.count_unaccounted() == 0


def test_find_meaning_with_unmatched_joiner_no_change() -> None:
    """When the joiner pool doesn't match any unaccounted slot, the
    decomposition is unchanged — bit-stable with the no-joiners path."""
    word_db = _word_db_with_two_morphemes()
    name_baseline = Name("Bridgexywater")
    name_baseline.find_meaning(word_db, reduce=False)
    baseline_total_unaccounted = name_baseline.count_unaccounted()

    name_unmatched = Name("Bridgexywater")
    name_unmatched.find_meaning(word_db, reduce=False, joiners={"old_english": [("zzz", 100)]})
    assert name_unmatched.count_unaccounted() == baseline_total_unaccounted


def test_find_meaning_joiners_empty_dict_is_no_op() -> None:
    """Passing an empty joiners dict is identical to omitting the kwarg."""
    word_db = _word_db_with_two_morphemes()
    name_empty = Name("Bridgexywater")
    name_empty.find_meaning(word_db, reduce=False, joiners={})
    name_omitted = Name("Bridgexywater")
    name_omitted.find_meaning(word_db, reduce=False)
    assert name_empty.count_unaccounted() == name_omitted.count_unaccounted()


# --- has_name / get_structure invariants ----------------------------------


def test_joiner_does_not_pollute_word_get_structure() -> None:
    """Word.get_structure() should NOT see the joiner — it only emits
    structure tuples for Meanings. A joiner-containing parse should
    produce the same structure as the same parse without the joiner."""
    word_db = _word_db_with_two_morphemes()
    name = Name("Bridgexywater")
    name.find_meaning(word_db, reduce=True, joiners={"old_english": [("xy", 100)]})
    # All Words for this name should have a 2-tuple structure
    # (pre + post Meanings) — the joiner doesn't add a third tuple.
    for word_list in name.words.values():
        for w in word_list:
            structure = w.get_structure()
            assert len(structure) == 2  # Bridge- + -water, joiner skipped


def test_get_structure_derives_position_from_index_not_dashes() -> None:
    """wyrd-5z5j/D39: a morpheme's structure position comes from its INDEX
    among the word's Meanings, NOT the matched form's dashes.

    - A sole morpheme matched via a dashed (`-pleasant`) form is structurally
      BARE, not post (this is the two-word-loss bug fix).
    - A 3-morpheme word is pre / inner / post by index, regardless of each
      stored form's own dashes.
    - The name/saint flag rides alongside the derived position."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.word import Word

    # Sole morpheme, matched via a post-shaped (`-pleasant`) form → bare.
    assert Word([Meaning("-pleasant", [], [], {})]).get_structure() == (("bare",),)

    # Three morphemes → pre / inner / post by index, ignoring stored dashes.
    three = Word(
        [Meaning("al", [], [], {}), Meaning("-ham-", [], [], {}), Meaning("Ton-", [], [], {})]
    )
    assert three.get_structure() == (("pre",), ("inner",), ("post",))

    # Name flag rides alongside (sole male-name morpheme → bare + name).
    assert Word([Meaning("-andrew", ["male name"], [], {})]).get_structure() == (("bare", "name"),)


def test_joiner_does_not_count_in_word_has_name() -> None:
    """Word.has_name acts on Meaning instances; a Joiner is non-Meaning
    so it can't accidentally satisfy has_name()."""
    j = Joiner("en", lang_field="old_english")
    from wyrd.generators.kenning.runtime.word import Word

    w = Word([j])  # only a Joiner, no Meanings
    assert w.has_name() is False
    assert w.has_saint() is False


# --- behavior across cross-language joiners (forward-looking) ------------


def test_consume_joiners_handles_multiple_lang_fields() -> None:
    """Multiple lang_fields populated: the right joiner-form fires
    for each, the lang_field on each Joiner reflects the source pool."""
    joiners = {
        "old_english": [("en", 100)],
        "celtic_mix": [("y", 50)],
    }
    lookup = _build_joiner_lookup(joiners)
    a = _meaning_for_anchor("a-")
    b = _meaning_for_anchor("-b-")
    c = _meaning_for_anchor("-c")
    decomp = [a, "en", b, "y", c]
    consumed = _consume_joiners(decomp, lookup)
    assert isinstance(consumed[1], Joiner)
    assert consumed[1].lang_field == "old_english"
    assert isinstance(consumed[3], Joiner)
    assert consumed[3].lang_field == "celtic_mix"


# --- Joiner equality boundary cases (test-coverage gap fill) -------------


def test_joiner_eq_returns_false_for_non_joiner() -> None:
    """Sentinel-discriminator code paths depend on
    ``isinstance(other, Joiner)``. Pin the contract so a bare
    ``j == "en"`` doesn't surprise downstream consumers."""
    j = Joiner("en", lang_field="old_english")
    assert (j == "en") is False
    assert (j == None) is False  # noqa: E711 — testing the equality directly
    m = Meaning("en", tags=[], meanings=["en"], sources={"old_english": ["en"]})
    assert (j == m) is False


def test_joiner_default_lang_field_is_none() -> None:
    """Constructing without a lang_field defaults to None — the
    public constructor shape that callers without lang context can
    use."""
    j = Joiner("en")
    assert j.lang_field is None
    assert str(j) == "en"


def test_joiner_with_default_lang_field_compares_equal() -> None:
    """Two Joiners constructed with default lang_field are equal."""
    a = Joiner("en")
    b = Joiner("en")
    assert a == b
    assert hash(a) == hash(b)


# --- load_joiners boundary cases (test-coverage gap fill) ----------------


def test_load_joiners_handles_null_per_language_entries() -> None:
    """A bundle with ``{"old_english": null}`` (e.g. an export
    setting the per-lang slot before populating it) skips the entry
    rather than raising. Defensive — keeps load_joiners robust against
    partially-emitted bundles."""
    bundle = {
        "subjects": [],
        "joiners": {"old_english": None, "celtic_mix": [{"form": "y", "weight": 50}]},
    }
    joiners = load_joiners(bundle)
    assert "old_english" not in joiners
    assert joiners == {"celtic_mix": [("y", 50)]}


def test_load_meanings_with_populated_joiners_loads_subjects_correctly() -> None:
    """End-to-end: a dict bundle with BOTH subjects and a populated
    joiners pool produces the same meaning_db as the bare-list shape.
    Pinned so the dict-shape parsing doesn't accidentally couple
    joiners and subjects."""
    subjects = [
        {
            "meaning": ["Bridge"],
            "modifier_tags": [],
            "modifier_type": "T",
            "words": [{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        }
    ]
    bundle = {
        "subjects": subjects,
        "joiners": {"old_english": [{"form": "en", "weight": 100}]},
    }
    word_db, _ = load_meanings(bundle)
    assert "Bridg-" in word_db
    # And load_joiners on the SAME bundle returns the populated pool.
    joiners = load_joiners(bundle)
    assert joiners == {"old_english": [("en", 100)]}


# --- reduce=True canonical filter caveat (Phase 2.5 limitation pin) ------


def test_find_meaning_with_reduce_canonical_filters_before_joiners() -> None:
    """Phase 2.5 LIMITATION pinned: under ``reduce=True``, the
    canonical filter scores RAW matcher output BEFORE joiner
    consumption. So a non-joiner parse that already reaches 0
    unaccounted wins canonical over a joiner-needing parse — even
    when the joiner-needing parse would tie on score after
    consumption.

    Setup: 'Bridgexywater' admits BOTH:
      A. [Bridge-, -xywater] — 0 unaccounted, 2 morphemes (raw)
      B. [Bridge-, "xy", -water] — 2 unaccounted, 2 morphemes (raw)

    Canonical filter sees A wins on (0, 2) before consumption. After
    consumption, B becomes [Bridge-, Joiner("xy"), -water] — 0 unacc,
    3 'morphemes' counting the Joiner. A still wins on Occam.

    Concretely: the surface picked by reduce=True is 'Bridgexywater'
    decomposed as [Bridge-, -xywater], NOT as [Bridge-, joiner,
    -water]. A future Phase 2.5 may push joiner-awareness into the
    canonical scoring; until then this is the documented behavior.
    """
    subjects = [
        {
            "meaning": ["Bridge"],
            "modifier_tags": [],
            "modifier_type": "T",
            "words": [{"modern_usage": "Bridge-", "old_english": ["brycg"]}],
        },
        {
            "meaning": ["Water"],
            "modifier_tags": [],
            "modifier_type": "T",
            "words": [{"modern_usage": "-water", "old_english": ["wæter"]}],
        },
        {
            "meaning": ["XYWater"],  # synthetic morpheme covering 'xywater' suffix
            "modifier_tags": [],
            "modifier_type": "T",
            "words": [{"modern_usage": "-xywater", "old_english": ["xywæter"]}],
        },
    ]
    word_db, _ = load_meanings(subjects)
    name = Name("Bridgexywater")
    name.find_meaning(word_db, reduce=True, joiners={"old_english": [("xy", 100)]})
    # Both parses reach 0 unaccounted but the non-joiner parse wins.
    assert name.count_unaccounted() == 0
    # Confirm: NO Joiner ended up in the canonical set — the parse
    # that would have needed one was filtered out.
    found_joiner = any(
        isinstance(slot, Joiner) for words in name.words.values() for w in words for slot in w.word
    )
    assert not found_joiner, (
        "Phase 2.5 limitation: canonical filter should pick the no-joiner "
        "parse [Bridge-, -xywater] over [Bridge-, joiner('xy'), -water]"
    )


# --- compose-time joiner insertion (wyrd-0l2g Phase 1.5) -----------------


def test_kenning_input_schema_exposes_joiner_density() -> None:
    """The Kenning generator's input_schema declares a
    ``joiner_density`` knob (0..1, default 0). Pinned so the SPA's
    schema-driven form picks it up automatically."""
    from wyrd.generators.kenning import Kenning

    schema = Kenning().input_schema()
    assert "joiner_density" in schema["properties"]
    knob = schema["properties"]["joiner_density"]
    assert knob["default"] == 0.0
    assert knob["minimum"] == 0.0
    assert knob["maximum"] == 1.0


def test_kenning_generate_default_joiner_density_is_bit_stable() -> None:
    """``Kenning.generate`` with joiner_density=0 (default) produces
    identical output to a call without the param. Bit-stability gate."""
    from wyrd.generators.kenning import Kenning

    kenning = Kenning()
    out_default = kenning.generate({"culture": "english"}, seed=42)
    out_explicit = kenning.generate({"culture": "english", "joiner_density": 0.0}, seed=42)
    assert out_default.result == out_explicit.result
    assert out_default.explanation == out_explicit.explanation
    assert out_default.components == out_explicit.components


def test_kenning_generate_no_joiners_in_bundle_is_no_op() -> None:
    """Even with joiner_density=1.0, today's bundle ships no joiners
    so the surface form is identical to the default-knob output.
    Pinned so a future joiner population can't silently regress
    legacy callers."""
    from wyrd.generators.kenning import Kenning

    kenning = Kenning()
    out_zero = kenning.generate({"culture": "english"}, seed=42)
    out_full = kenning.generate({"culture": "english", "joiner_density": 1.0}, seed=42)
    # No joiners in the bundle → no insertion happens regardless of
    # density. Both calls produce the same name string.
    assert out_zero.result == out_full.result


def test_apply_joiner_insertion_skips_when_no_shared_lang() -> None:
    """``_apply_joiner_insertion`` is a no-op when adjacent morphemes
    don't share any lang_field with a populated joiner pool."""
    import random

    from wyrd.generators.kenning import _apply_joiner_insertion
    from wyrd.generators.kenning.runtime.proportions import NewName

    m1 = Meaning("Bridge-", tags=[], meanings=["Bridge"], sources={"old_english": ["brycg"]})
    m2 = Meaning("-water", tags=[], meanings=["Water"], sources={"celtic_mix": ["dwr"]})
    meaning_db = {"Bridge-": [m1], "-water": [m2]}
    new_name = NewName(struct=None, meaning_db=meaning_db, name=[["Bridge-", "-water"]])

    # Joiner pool only has 'old_english' but right meaning is celtic.
    joiners = {"old_english": [("en", 100)]}
    rng = random.Random(0)
    surface, _, components = _apply_joiner_insertion(new_name, joiners, rng, density=1.0)
    assert "en" not in surface  # no shared lang → no joiner
    assert all(c["location"] != "joiner" for c in components)


def test_apply_joiner_insertion_inserts_joiner_when_shared_lang() -> None:
    """When two adjacent morphemes share a lang_field with populated
    joiners, density=1.0 deterministically inserts the joiner. Surface
    + components reflect the insertion."""
    import random

    from wyrd.generators.kenning import _apply_joiner_insertion
    from wyrd.generators.kenning.runtime.proportions import NewName

    m1 = Meaning("Bridge-", tags=[], meanings=["Bridge"], sources={"old_english": ["brycg"]})
    m2 = Meaning("-water", tags=[], meanings=["Water"], sources={"old_english": ["wæter"]})
    meaning_db = {"Bridge-": [m1], "-water": [m2]}
    new_name = NewName(struct=None, meaning_db=meaning_db, name=[["Bridge-", "-water"]])

    joiners = {"old_english": [("en", 100)]}
    rng = random.Random(0)
    surface, explanation, components = _apply_joiner_insertion(new_name, joiners, rng, density=1.0)
    assert "en" in surface
    assert "+joiner: en" in explanation
    joiner_components = [c for c in components if c["location"] == "joiner"]
    assert len(joiner_components) == 1
    assert joiner_components[0]["usage"] == "en"
    assert "joiner" in joiner_components[0]["tags"]


def test_apply_joiner_insertion_zero_density_is_no_op() -> None:
    """At density=0.0, even with shared lang + populated pool, no
    joiner inserts."""
    import random

    from wyrd.generators.kenning import _apply_joiner_insertion
    from wyrd.generators.kenning.runtime.proportions import NewName

    m1 = Meaning("Bridge-", tags=[], meanings=["Bridge"], sources={"old_english": ["brycg"]})
    m2 = Meaning("-water", tags=[], meanings=["Water"], sources={"old_english": ["wæter"]})
    meaning_db = {"Bridge-": [m1], "-water": [m2]}
    new_name = NewName(struct=None, meaning_db=meaning_db, name=[["Bridge-", "-water"]])

    joiners = {"old_english": [("en", 100)]}
    rng = random.Random(0)
    surface, _, components = _apply_joiner_insertion(new_name, joiners, rng, density=0.0)
    assert all(c["location"] != "joiner" for c in components)
    assert "bridgewater" in surface.lower()


def test_shared_lang_fields_returns_intersection_with_populated_pools() -> None:
    """``_shared_lang_fields_with_joiners`` returns lang_fields that
    BOTH morpheme groups carry AND that have a non-empty joiner pool."""
    from wyrd.generators.kenning import _shared_lang_fields_with_joiners

    m1 = Meaning(
        "x",
        tags=[],
        meanings=["x"],
        sources={"old_english": ["xx"], "celtic_mix": ["yy"]},
    )
    m2 = Meaning("y", tags=[], meanings=["y"], sources={"old_english": ["zz"]})
    joiners = {"old_english": [("en", 100)], "celtic_mix": [("y", 50)]}
    shared = _shared_lang_fields_with_joiners([m1], [m2], joiners)
    assert shared == {"old_english"}


def test_shared_lang_fields_empty_when_pool_is_empty() -> None:
    """A shared lang_field with an EMPTY joiner pool doesn't qualify
    — no joiner to insert."""
    from wyrd.generators.kenning import _shared_lang_fields_with_joiners

    m1 = Meaning("x", tags=[], meanings=["x"], sources={"old_english": ["xx"]})
    m2 = Meaning("y", tags=[], meanings=["y"], sources={"old_english": ["zz"]})
    joiners: dict = {"old_english": []}
    shared = _shared_lang_fields_with_joiners([m1], [m2], joiners)
    assert shared == set()


def test_weighted_joiner_choice_respects_weights() -> None:
    """Over many draws, the weighted choice converges on the higher-
    weight option."""
    import random

    from wyrd.generators.kenning import _weighted_joiner_choice

    rng = random.Random(42)
    pool = [("a", 10), ("b", 90)]
    counts = {"a": 0, "b": 0}
    for _ in range(1000):
        counts[_weighted_joiner_choice(pool, rng)] += 1
    assert counts["b"] > 800


def test_weighted_joiner_choice_falls_back_to_uniform_when_zero_weights() -> None:
    """A pool where every weight is zero falls back to uniform random
    choice."""
    import random

    from wyrd.generators.kenning import _weighted_joiner_choice

    rng = random.Random(0)
    pool = [("a", 0), ("b", 0)]
    result = _weighted_joiner_choice(pool, rng)
    assert result in {"a", "b"}


def test_load_joiners_runtime_helper_returns_empty_for_l4_bundle() -> None:
    """``_load_joiners`` reads the L4 runtime DB. The L4 schema
    doesn't carry joiners today, so the runtime helper returns an
    empty dict."""
    from wyrd.generators.kenning import _load_joiners

    _load_joiners.cache_clear()
    joiners = _load_joiners()
    assert joiners == {}


def test_load_joiners_caches_result() -> None:
    """``_load_joiners`` is `@lru_cache(maxsize=1)`. Two consecutive
    calls return the SAME dict object (no re-parse of the bundle
    file)."""
    from wyrd.generators.kenning import _load_joiners

    _load_joiners.cache_clear()
    a = _load_joiners()
    b = _load_joiners()
    assert a is b


def test_weighted_joiner_choice_raises_on_empty_pool() -> None:
    """Defensive: an empty joiner pool should raise ``ValueError``,
    not crash with ``IndexError`` deep inside ``rng.choice([])``.
    Every call site filters empty pools out via
    ``_shared_lang_fields_with_joiners``, but a future caller
    bypassing that filter gets a clear error."""
    import random

    import pytest

    from wyrd.generators.kenning import _weighted_joiner_choice

    with pytest.raises(ValueError):
        _weighted_joiner_choice([], random.Random(0))


@pytest.mark.parametrize("seed", [0, 1, 42, 1000, 2026])
def test_kenning_generate_density_zero_bit_stable_across_seeds(seed: int) -> None:
    """Multi-seed bit-stability gate. Pre-PR Kenning.generate at
    seed=N must produce identical output to post-PR generate at
    seed=N when joiner_density=0. Pinned across multiple seeds so
    a single coincidentally-stable seed can't mask a regression."""
    from wyrd.generators.kenning import Kenning

    kenning = Kenning()
    out_default = kenning.generate({"culture": "english"}, seed=seed)
    out_explicit = kenning.generate({"culture": "english", "joiner_density": 0.0}, seed=seed)
    assert out_default.result == out_explicit.result, f"seed={seed}"
    assert out_default.explanation == out_explicit.explanation, f"seed={seed}"


def test_apply_joiner_insertion_skips_none_elements() -> None:
    """A NewName word can contain ``None`` placeholder slots (the
    selector emitting None for empty slots). The walker must skip
    them rather than crash on attribute access."""
    import random

    from wyrd.generators.kenning import _apply_joiner_insertion
    from wyrd.generators.kenning.runtime.proportions import NewName

    m1 = Meaning("Bridge-", tags=[], meanings=["Bridge"], sources={"old_english": ["brycg"]})
    m2 = Meaning("-water", tags=[], meanings=["Water"], sources={"old_english": ["wæter"]})
    meaning_db = {"Bridge-": [m1], "-water": [m2]}
    # Word with a None slot mid-list — only the non-None elements
    # should be considered for adjacency.
    new_name = NewName(struct=None, meaning_db=meaning_db, name=[["Bridge-", None, "-water"]])

    joiners = {"old_english": [("en", 100)]}
    rng = random.Random(0)
    surface, _, components = _apply_joiner_insertion(new_name, joiners, rng, density=1.0)
    # Joiner still inserted between the two non-None morphemes.
    assert "en" in surface
    assert any(c["location"] == "joiner" for c in components)


def test_apply_joiner_insertion_single_element_word_no_op() -> None:
    """A word with a single morpheme has no adjacent pair; joiner
    insertion is a no-op."""
    import random

    from wyrd.generators.kenning import _apply_joiner_insertion
    from wyrd.generators.kenning.runtime.proportions import NewName

    m1 = Meaning("Bridge-", tags=[], meanings=["Bridge"], sources={"old_english": ["brycg"]})
    meaning_db = {"Bridge-": [m1]}
    new_name = NewName(struct=None, meaning_db=meaning_db, name=[["Bridge-"]])

    joiners = {"old_english": [("en", 100)]}
    rng = random.Random(0)
    _, _, components = _apply_joiner_insertion(new_name, joiners, rng, density=1.0)
    assert all(c["location"] != "joiner" for c in components)


def test_apply_joiner_insertion_uses_rendered_substitutions() -> None:
    """When NewName has ``rendered`` substitutions (D18 spelling-
    variant or D8 inflection picks), the joiner insertion uses the
    rendered surfaces, not the dash-stripped raw usage."""
    import random

    from wyrd.generators.kenning import _apply_joiner_insertion
    from wyrd.generators.kenning.runtime.proportions import NewName

    m1 = Meaning("Bridge-", tags=[], meanings=["Bridge"], sources={"old_english": ["brycg"]})
    m2 = Meaning("-water", tags=[], meanings=["Water"], sources={"old_english": ["wæter"]})
    meaning_db = {"Bridge-": [m1], "-water": [m2]}
    # Rendered substitutions: 'brycg' (variant) + 'water' (canonical).
    new_name = NewName(
        struct=None,
        meaning_db=meaning_db,
        name=[["Bridge-", "-water"]],
        rendered=[["brycg", None]],  # variant for slot 0, default for slot 1
    )

    joiners = {"old_english": [("en", 100)]}
    rng = random.Random(0)
    surface, _, _ = _apply_joiner_insertion(new_name, joiners, rng, density=1.0)
    # Surface should carry the variant 'brycg' (not 'bridge'), the joiner
    # 'en', and the default 'water'.
    assert "brycg" in surface
    assert "en" in surface
    assert "water" in surface


def test_apply_joiner_insertion_multi_word_within_each_word() -> None:
    """For a multi-word toponym, joiner insertion only fires WITHIN
    each word, not across the whitespace boundary."""
    import random

    from wyrd.generators.kenning import _apply_joiner_insertion
    from wyrd.generators.kenning.runtime.proportions import NewName

    m1 = Meaning("Bridge-", tags=[], meanings=["Bridge"], sources={"old_english": ["brycg"]})
    m2 = Meaning("-water", tags=[], meanings=["Water"], sources={"old_english": ["wæter"]})
    m3 = Meaning("Saint", tags=[], meanings=["Saint"], sources={"old_english": ["sanct"]})
    meaning_db = {"Bridge-": [m1], "-water": [m2], "Saint": [m3]}
    # Two-word name: word 1 is 'Saint' (single morpheme), word 2 is
    # 'Bridge- + -water'. Joiners can only fire within word 2.
    new_name = NewName(
        struct=None,
        meaning_db=meaning_db,
        name=[["Saint"], ["Bridge-", "-water"]],
    )

    joiners = {"old_english": [("en", 100)]}
    rng = random.Random(0)
    surface, _, components = _apply_joiner_insertion(new_name, joiners, rng, density=1.0)
    # Surface has exactly one joiner — between Bridge and water.
    assert surface.count("en") >= 1
    joiner_components = [c for c in components if c["location"] == "joiner"]
    assert len(joiner_components) == 1
