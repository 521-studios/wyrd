"""Unit tests for kenning's Meaning model and meaning DB loader."""

from __future__ import annotations

import random
from collections import Counter

from wyrd.generators.kenning.meaning import Meaning, _mimic_case, load_meanings


def make_meaning(usage, tags=None, meanings=None, sources=None):
    return Meaning(usage, tags or [], meanings or [], sources or {})


def test_location_pre_when_trailing_dash():
    assert make_meaning("Bre-").location == "pre"


def test_location_post_when_leading_dash():
    assert make_meaning("-combe").location == "post"


def test_location_inner_when_dashes_on_both_sides():
    assert make_meaning("-by-").location == "inner"


def test_location_post_when_no_dash():
    assert make_meaning("ham").location == "post"


def test_test_pre_returns_meaning_then_remainder():
    m = make_meaning("Bre-")
    assert m.test("Brecombe") == [m, "combe"]


def test_test_pre_no_match_returns_none():
    assert make_meaning("Bre-").test("Westcombe") is None


def test_test_post_returns_remainder_then_meaning():
    m = make_meaning("-ton")
    assert m.test("Easton") == ["Eas", m]


def test_test_inner_returns_left_meaning_right():
    m = make_meaning("-by-")
    assert m.test("Westbycombe") == ["West", m, "combe"]


def test_test_post_strips_match_case_insensitively():
    # Regression: case-sensitive str.replace used to leave 'Hill' as residue
    # when matching post-Meaning('-hill'), inflating unaccounted counts.
    m = make_meaning("-hill")
    assert m.test("Hill") == ["", m]
    assert m.test("Bushhill") == ["Bush", m]


def test_test_pre_preserves_remainder_casing():
    m = make_meaning("Bre-")
    # Token comparison is lowercase, but the residue keeps the original case.
    assert m.test("BreCombe") == [m, "Combe"]


def test_test_inner_preserves_residue_casing():
    m = make_meaning("-by-")
    assert m.test("WestByCombe") == ["West", m, "Combe"]


def test_is_name_true_for_male_female_or_family():
    assert make_meaning("Alf-", tags=["male name"]).is_name()
    assert make_meaning("Alf-", tags=["female name"]).is_name()
    assert make_meaning("Alf-", tags=["family name"]).is_name()


def test_is_name_false_for_other_tags():
    assert not make_meaning("Bre-", tags=["water"]).is_name()


def test_is_saint_detection():
    assert make_meaning("saint", tags=["saint"]).is_saint()
    assert not make_meaning("ham", tags=["water"]).is_saint()


def test_key_simple_post():
    assert make_meaning("-ton").key() == ("post",)


def test_key_with_name_tag():
    assert make_meaning("Alf-", tags=["male name"]).key() == ("pre", "name")


def test_key_with_saint_usage():
    # Saint key path triggers when usage text matches "saint" exactly
    # (with dashes stripped, lowercase) — separate from is_name().
    assert make_meaning("saint", tags=[]).key() == ("post", "saint")


def test_load_meanings_indexes_by_usage_and_tag():
    data = [
        {
            "modifier_tags": ["water"],
            "meaning": ["stream"],
            "words": [{"modern_usage": "-bourne", "old_english": "burna"}],
        }
    ]
    meaning_db, tag_db = load_meanings(data)
    assert "-bourne" in meaning_db
    assert meaning_db["-bourne"][0].sources == {"old_english": "burna"}
    assert "water" in tag_db
    assert "-bourne" in tag_db["water"]


def test_load_meanings_pluralizes_name_tagged_entries():
    data = [
        {
            "modifier_tags": ["male name"],
            "meaning": ["Alfred (Name)"],
            "words": [{"modern_usage": "Alf-"}],
        }
    ]
    meaning_db, tag_db = load_meanings(data)
    assert "Alf-" in meaning_db
    # Plural form added automatically for non-s-ending names.
    assert "Alf-s" in meaning_db
    assert "Alf-s" in tag_db["male name"]


def test_load_meanings_does_not_pluralize_already_plural_names():
    data = [
        {
            "modifier_tags": ["family name"],
            "meaning": ["Tribe (Name)"],
            "words": [{"modern_usage": "Saxons"}],
        }
    ]
    meaning_db, _ = load_meanings(data)
    assert "Saxons" in meaning_db
    assert "Saxonss" not in meaning_db


# --- D18 spelling variants -------------------------------------------------


def _meaning_with_variants(variants):
    return Meaning(
        usage="Bridg-",
        tags=["water"],
        meanings=["Bridge"],
        sources={"old_english": ["brycg"]},
        variants=variants,
    )


def test_pick_variant_returns_none_when_pool_empty():
    """No variants → no substitution, regardless of spelling_variety."""
    m = make_meaning("Bridg-", sources={"old_english": ["brycg"]})
    assert m.pick_variant(random.Random(0), 1.0) is None


def test_pick_variant_returns_none_when_variety_zero():
    """spelling_variety=0 always preserves the canonical surface form."""
    m = _meaning_with_variants({"old_english": [("brycg", 5), ("brigge", 3)]})
    assert m.pick_variant(random.Random(0), 0.0) is None


def test_pick_variant_returns_a_variant_when_variety_one():
    """At spelling_variety=1, a non-empty pool always yields a variant."""
    m = _meaning_with_variants({"old_english": [("brycg", 5), ("brigge", 3)]})
    picked = m.pick_variant(random.Random(0), 1.0)
    assert picked in {"brycg", "brigge"}


def test_pick_variant_weighted_distribution_favors_heavier_weight():
    """Over many draws, variants are picked roughly in proportion to weight."""
    m = _meaning_with_variants({"old_english": [("a", 90), ("b", 10)]})
    counts = Counter(m.pick_variant(random.Random(i), 1.0) for i in range(2000))
    # 'a' should dominate at roughly 9:1. Allow a wide band against RNG variance.
    assert counts["a"] > counts["b"] * 5


def test_load_meanings_extracts_variants_from_variant_field():
    """``<lang>_variants`` keys are split out of ``sources`` into the
    Meaning's variants dict; canonical language form lists stay in sources
    unchanged."""
    data = [
        {
            "modifier_tags": ["water"],
            "meaning": ["Bridge"],
            "words": [
                {
                    "modern_usage": "Bridg-",
                    "old_english": ["brycg"],
                    "old_english_variants": [
                        {"form": "brigge", "weight": 3},
                        {"form": "brige", "weight": 1},
                    ],
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    m = meaning_db["Bridg-"][0]
    assert m.variants == {"old_english": [("brigge", 3), ("brige", 1)]}
    # Canonical language array stays in sources, variant field stripped.
    assert m.sources == {"old_english": ["brycg"]}


def test_load_meanings_omits_variants_field_when_absent():
    """Legacy meanings.json data without the variant field still loads cleanly
    — every Meaning gets an empty variants dict."""
    data = [
        {
            "modifier_tags": [],
            "meaning": ["x"],
            "words": [{"modern_usage": "X-", "old_english": ["x"]}],
        }
    ]
    meaning_db, _ = load_meanings(data)
    assert meaning_db["X-"][0].variants == {}


def test_mimic_case_capitalizes_at_word_start():
    """A pre-position canonical 'Bridg-' should re-cap a variant."""
    assert _mimic_case("Bridg-", "brycg") == "Brycg"


def test_mimic_case_lowercases_at_word_end():
    """A post-position canonical '-water' should keep the variant lowercase
    even when the variant comes in upper-cased."""
    assert _mimic_case("-water", "WATYR") == "watyr"


def test_mimic_case_handles_empty_template():
    """A template with no letters (just dashes) returns the variant verbatim."""
    assert _mimic_case("-", "x") == "x"
