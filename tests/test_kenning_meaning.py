"""Unit tests for kenning's Meaning model and meaning DB loader."""

from __future__ import annotations

from wyrd.generators.kenning.meaning import Meaning, load_meanings


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
