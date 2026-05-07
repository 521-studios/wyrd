"""Tests for wyrd-ha9q Phase 2b: english_shaped derivation."""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.english_shaping import (
    KNOWN_FORM_OVERRIDES,
    _apply_digraphs,
    _apply_single_chars,
    _ipa_to_english_fallback,
    _looks_english_readable,
    _strip_transliteration,
    derive_english_shaped,
)

# ---------------------------------------------------------------------
# Diacritic-stripping primitives
# ---------------------------------------------------------------------


def test_apply_digraphs_replaces_semitic_two_letter_marks() -> None:
    """Semitic-transliteration digraphs (ṯ→th, ḫ→kh, š→sh, ǧ→j, ġ→gh,
    ḏ→dh) produce two-letter output. str.replace replaces ALL
    occurrences within a string."""
    assert _apply_digraphs("ṯalāṯah") == "thalāthah"
    assert _apply_digraphs("ḫamis") == "khamis"
    assert _apply_digraphs("šayṭān") == "shayṭān"  # ṭ stays (handled by single-char)
    assert _apply_digraphs("ǧinn") == "jinn"
    assert _apply_digraphs("ġūl") == "ghūl"  # ū stays for single-char pass
    assert _apply_digraphs("ḏikr") == "dhikr"


def test_apply_digraphs_replaces_sanskrit_sibilants_to_sh() -> None:
    """Both ś and ṣ map to 'sh' (and 'Sh') — Sanskrit retroflex /
    palatal sibilants. rakṣasa convention is rakshasa, not rakssasa."""
    assert _apply_digraphs("rakṣasa") == "rakshasa"
    assert _apply_digraphs("śiva") == "shiva"
    assert _apply_digraphs("Śakti") == "Shakti"


def test_apply_digraphs_replaces_sanskrit_nasals_to_ng_and_ny() -> None:
    """Palatal ñ→ny, velar ṅ→ng. Sanskrit-canonical."""
    assert _apply_digraphs("añjali") == "anyjali"
    assert _apply_digraphs("liṅga") == "lingga"


def test_apply_single_chars_strips_emphatics_and_long_vowels() -> None:
    """Single-char map: emphatics (ḍ→d, ḥ→h, ṭ→t), long vowels (ā→a,
    ī→i, ū→u), retroflex single-chars (ṛ→r, ṇ→n, ḷ→l). The single-char
    pass intentionally does NOT touch digraph chars (ṣ, š, ǧ, ...) —
    those are handled by `_apply_digraphs` upstream."""
    assert _apply_single_chars("ḍāl") == "dal"
    assert _apply_single_chars("ḥikma") == "hikma"
    assert _apply_single_chars("ṭāhir") == "tahir"
    assert _apply_single_chars("ṛ ṇ ḷ ā ī ū ē ō") == "r n l a i u e o"


def test_apply_single_chars_silences_glottals() -> None:
    """ʿ ʾ ʔ → empty string. Acute / grave accents → bare letter."""
    assert _apply_single_chars("ʿifrīt") == "ifrit"
    assert _apply_single_chars("ʾāl") == "al"
    assert _apply_single_chars("rāʔ") == "ra"
    # Hebrew stress-acute drops:
    assert _apply_single_chars("káp") == "kap"
    assert _apply_single_chars("bósem") == "bosem"


def test_strip_transliteration_runs_full_pipeline() -> None:
    """End-to-end strip: digraphs → single-chars → cleanup.
    Verify a Hebrew + Arabic + Sanskrit sample each."""
    assert _strip_transliteration("ʿifrīt") == "ifrit"
    assert _strip_transliteration("rakṣasa") == "rakshasa"
    assert _strip_transliteration("kɛ́lɛḇ, kélev") == "kelev, kelev"  # comma stays; English-readable
    assert _strip_transliteration("Ba'al Zvuv") == "Baal Zvuv"
    # ṯ → 'th' is the rule output ("shabbath" is the older English form).
    # The modern English "shabbat" convention is a known-form override
    # rather than a rule case — kept here as the rule baseline.
    assert _strip_transliteration("šabbāṯ") == "shabbath"


def test_strip_transliteration_collapses_ipa_brackets_and_subscripts() -> None:
    """IPA-style brackets / slashes / length / stress markers are
    stripped. Akkadian subscripts (ar-ga-man-nu, GIR₄) too. The
    d͡ʒ ligature collapses to 'dzh' since ʒ goes through the digraph
    map to 'zh' before the tie-bar gets stripped."""
    assert _strip_transliteration("/d͡ʒɪn/") == "dzhin"
    assert _strip_transliteration("ar-ga-man-nu₂") == "ar-ga-man-nu"
    # ħ→h, ɔ→o, θ→th, brackets+stress stripped → "hothul"
    assert _strip_transliteration("[ħɔˈθul]") == "hothul"


# ---------------------------------------------------------------------
# _looks_english_readable predicate
# ---------------------------------------------------------------------


def test_looks_english_readable_accepts_clean_ascii() -> None:
    assert _looks_english_readable("rakshasa") is True
    assert _looks_english_readable("baal zvuv") is True
    assert _looks_english_readable("ba'al-zvuv") is True


def test_looks_english_readable_rejects_residual_diacritics() -> None:
    """If our maps missed a character, the residual non-ASCII char
    means the output isn't ready — return False so the caller gets
    None and falls back to canonical_form."""
    assert _looks_english_readable("ifrit") is True
    assert _looks_english_readable("ʿifrīt") is False  # both ʿ and ī survive
    assert _looks_english_readable("kelev_") is False  # underscore not in allowed set
    assert _looks_english_readable("123") is False  # digits-only, no letter
    assert _looks_english_readable("") is False


# ---------------------------------------------------------------------
# IPA fallback
# ---------------------------------------------------------------------


def test_ipa_to_english_fallback_strips_brackets_and_stress() -> None:
    """`/d͡ʒɪnː/` → 'dzhin' (length mark and tie-bar removed; ʒ→zh)."""
    assert _ipa_to_english_fallback("/d͡ʒɪnː/") == "dzhin"
    assert _ipa_to_english_fallback("[kalb]") == "kalb"


def test_ipa_to_english_fallback_returns_none_on_unstrippable() -> None:
    """If even after stripping the result has non-ASCII residuals,
    return None — caller falls back to canonical_form."""
    # An IPA string with phonemes we don't map: returns None
    assert _ipa_to_english_fallback("/ʕʕʕ/") is None  # only ayin, all silenced → empty


# ---------------------------------------------------------------------
# Top-level derive_english_shaped
# ---------------------------------------------------------------------


def test_derive_skips_latin_script_languages() -> None:
    """Old English / Latin / Welsh / Old French / Proto-Germanic etc.
    return None — canonical_form is already English-readable."""
    for lang in ("old-english", "latin", "old-french", "welsh", "proto-germanic"):
        assert (
            derive_english_shaped(
                canonical_form="tūn",
                language=lang,
                transliteration="tun",
                pronunciation_ipa=None,
            )
            is None
        ), f"failed for {lang}"


def test_derive_known_form_override_wins_over_rule_strip() -> None:
    """Even when the diacritic-strip would produce a valid output,
    the cultural-precedent override fires first. rakṣasa → rakshasa
    (the override) not 'rakshasa' (which happens to be the same here)
    or rakshasa via single-char rules. Use a case where the override
    differs: 'jinn' override is short 'jinn', not the digraph-stripped
    'jinn' (which would also be jinn — same answer). Use 'apsaras'
    override (override = 'apsara', singular)."""
    assert (
        derive_english_shaped(
            canonical_form="अप्सरस्",
            language="sa",
            transliteration="apsaras",
            pronunciation_ipa=None,
        )
        == "apsara"
    )


def test_derive_known_form_override_case_insensitive() -> None:
    """Override match is case-insensitive: 'RAKṢASA', 'Rakṣasa',
    'rakṣasa' all resolve."""
    for case in ("rakṣasa", "Rakṣasa", "RAKṢASA"):
        assert (
            derive_english_shaped(
                canonical_form="रक्षस",
                language="sa",
                transliteration=case,
                pronunciation_ipa=None,
            )
            == "rakshasa"
        ), f"failed for {case}"


def test_derive_falls_through_to_strip_when_no_override() -> None:
    """A transliteration with no override goes through the strip
    pipeline. Sample: Hebrew 'shabát' → 'shabat'."""
    assert (
        derive_english_shaped(
            canonical_form="שבת",
            language="he",
            transliteration="shabát",
            pronunciation_ipa=None,
        )
        == "shabat"
    )


def test_derive_falls_through_to_ipa_when_no_transliteration() -> None:
    """When transliteration is absent, IPA is the fallback."""
    assert (
        derive_english_shaped(
            canonical_form="جن",
            language="ar",
            transliteration=None,
            pronunciation_ipa="/d͡ʒɪnː/",
        )
        == "dzhin"
    )


def test_derive_returns_none_on_no_inputs() -> None:
    """All sources NULL → None."""
    assert (
        derive_english_shaped(
            canonical_form="جن",
            language="ar",
            transliteration=None,
            pronunciation_ipa=None,
        )
        is None
    )


def test_derive_returns_none_when_strip_leaves_residuals() -> None:
    """If the transliteration uses characters our maps don't cover,
    the strip output won't pass _looks_english_readable, and we
    fall through to IPA / None instead of surfacing a half-stripped
    value."""
    # Greek θ is in the digraph map (→th), so use a character that
    # really isn't in any map. Combining accents that survived NFD:
    assert (
        derive_english_shaped(
            canonical_form="x",
            language="he",
            transliteration="́́",  # bare combining accents, no base char
            pronunciation_ipa=None,
        )
        is None
    )


# ---------------------------------------------------------------------
# Per-language smoke tests against the override + strip pipeline
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "translit,language,expected",
    [
        # Sanskrit creature canon
        ("rakṣasa", "sa", "rakshasa"),
        ("nāga", "sa", "naga"),
        ("garuḍa", "sa", "garuda"),
        ("yakṣa", "sa", "yaksha"),
        # Arabic
        ("ʿifrīt", "ar", "ifrit"),
        ("marīd", "ar", "marid"),
        ("ǧinn", "ar", "jinn"),
        ("šayṭān", "ar", "shaitan"),
        ("ġūl", "ar", "ghoul"),
        # Hebrew
        ("gōlem", "he", "golem"),
        ("behēmōṯ", "he", "behemoth"),
        ("līwyāṯān", "he", "leviathan"),
        ("lilīṯ", "he", "lilith"),
        # Persian
        ("parī", "fa", "peri"),
        ("dīv", "fa", "div"),
        ("sīmurġ", "fa", "simurgh"),
        # Akkadian
        ("tiāmat", "akk", "tiamat"),
        # Egyptian (Faulkner-style)
        ("anpw", "egy", "anpu"),
    ],
)
def test_derive_canonical_creature_names_match_known_forms(
    translit: str, language: str, expected: str
) -> None:
    """Spot-check the well-known English forms for each wave-2 language.
    These are the names that drove the design (rakshasa, jinn, golem,
    ifrit, marid, shaitan, sphinx) — they must come out exactly."""
    actual = derive_english_shaped(
        canonical_form="…",
        language=language,
        transliteration=translit,
        pronunciation_ipa=None,
    )
    assert actual == expected, f"{translit!r} → {actual!r} (expected {expected!r})"


def test_known_form_overrides_keys_are_lowercase_or_typeable() -> None:
    """All KNOWN_FORM_OVERRIDES keys must be lowercase (the lookup
    lowercases before matching). Catches typos that would make an
    override silently never fire."""
    for key in KNOWN_FORM_OVERRIDES:
        assert key == key.lower(), f"override key {key!r} has uppercase letters"


def test_known_form_overrides_values_are_clean_ascii() -> None:
    """Override VALUES must be ASCII (the whole point is English-
    readable) and look like reasonable English words."""
    for key, value in KNOWN_FORM_OVERRIDES.items():
        assert value.isascii(), f"override value for {key!r} has non-ASCII: {value!r}"
        assert value, f"override for {key!r} is empty"
        assert _looks_english_readable(value), (
            f"override value {value!r} for {key!r} doesn't pass readability check"
        )
