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

from wyrd.generators.kenning.meaning import (
    Joiner,
    Meaning,
    _bundle_subjects,
    load_joiners,
    load_meanings,
)
from wyrd.generators.kenning.name import (
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


def test_consume_joiners_replaces_str_slot_with_joiner() -> None:
    """A decomposition slot that's a bare str matching a registered
    joiner form becomes a ``Joiner`` instance carrying the same
    surface + originating lang_field."""
    m = Meaning("Bridg-", tags=[], meanings=["Bridge"], sources={"old_english": ["brycg"]})
    decomp = [m, "en"]
    consumed = _consume_joiners(decomp, {"en": "old_english"})
    assert consumed[0] is m  # Meaning passes through
    assert isinstance(consumed[1], Joiner)
    assert consumed[1].surface == "en"
    assert consumed[1].lang_field == "old_english"


def test_consume_joiners_leaves_unmatched_strings_alone() -> None:
    """A str slot whose surface isn't in the joiner lookup stays a
    plain str — count_unaccounted will charge those chars, which is
    the correct behavior (genuinely unaccounted)."""
    m = Meaning("Bridg-", tags=[], meanings=["Bridge"], sources={"old_english": ["brycg"]})
    decomp = [m, "xyz"]
    consumed = _consume_joiners(decomp, {"en": "old_english"})
    assert consumed[1] == "xyz"


def test_consume_joiners_no_op_with_empty_lookup() -> None:
    """Empty lookup means no joiners registered; the decomposition is
    returned unchanged."""
    m = Meaning("Bridg-", tags=[], meanings=["Bridge"], sources={"old_english": ["brycg"]})
    decomp = [m, "en"]
    consumed = _consume_joiners(decomp, {})
    assert consumed == decomp


def test_consume_joiners_case_insensitive_match() -> None:
    """A str slot 'EN' matches a registered joiner 'en' — case
    independence matches the lookup's lowercase normalization."""
    decomp = ["EN"]
    consumed = _consume_joiners(decomp, {"en": "old_english"})
    assert isinstance(consumed[0], Joiner)
    assert consumed[0].surface == "EN"  # ORIGINAL case preserved on the Joiner


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


def test_joiner_does_not_count_in_word_has_name() -> None:
    """Word.has_name acts on Meaning instances; a Joiner is non-Meaning
    so it can't accidentally satisfy has_name()."""
    j = Joiner("en", lang_field="old_english")
    from wyrd.generators.kenning.word import Word

    w = Word([j])  # only a Joiner, no Meanings
    assert w.has_name() is False
    assert w.has_saint() is False


# --- behavior across cross-language joiners (forward-looking) ------------


def test_consume_joiners_handles_multiple_lang_fields() -> None:
    """The matcher hook with multiple lang_fields populated still
    consumes correctly per-form; lang_field carried on each Joiner
    reflects the source pool."""
    joiners = {
        "old_english": [("en", 100)],
        "celtic_mix": [("y", 50)],
    }
    lookup = _build_joiner_lookup(joiners)
    decomp = ["en", "y", "xx"]
    consumed = _consume_joiners(decomp, lookup)
    assert isinstance(consumed[0], Joiner)
    assert consumed[0].lang_field == "old_english"
    assert isinstance(consumed[1], Joiner)
    assert consumed[1].lang_field == "celtic_mix"
    assert consumed[2] == "xx"  # unmatched, stays str
