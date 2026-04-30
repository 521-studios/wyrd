"""Tests for the per-language phonology tables.

These exercise the GM-facing respelling and IPA outputs for representative
etymons in each supported language. The expected outputs are conservative
(roughly what a SAMPA-lite respelling would say) — when scholarly sources
disagree, we go with the most readable English approximation.
"""

from __future__ import annotations

from wyrd.generators.kenning.phonology import (
    PHONOLOGY,
    to_ipa,
    to_respelling,
)

# --- registry sanity ------------------------------------------------------


def test_phonology_registry_covers_expected_languages() -> None:
    for lang in ("old-english", "old-norse", "welsh"):
        assert lang in PHONOLOGY
        assert len(PHONOLOGY[lang]) > 10


# --- Old English ----------------------------------------------------------


def test_old_english_long_vowels_get_doubled_respelling() -> None:
    """Long vowels (macron-marked) should respell as digraph English vowels."""
    assert "AH" in to_respelling("hām", "old-english")  # 'home'
    assert "OO" in to_respelling("tūn", "old-english")  # 'enclosure'
    assert "AY" in to_respelling("lēah", "old-english")  # 'clearing'


def test_old_english_handles_aesc_and_eth() -> None:
    assert "a" in to_respelling("hædan", "old-english")
    assert "th" in to_respelling("þorn", "old-english")
    assert "th" in to_respelling("ðæt", "old-english")


def test_old_english_cg_digraph() -> None:
    """OE 'cg' should respell as 'j' (Brycg = /brɪdʒ/)."""
    assert "j" in to_respelling("brycg", "old-english")


def test_old_english_sc_digraph() -> None:
    assert "sh" in to_respelling("sceap", "old-english")


# --- Old Norse ------------------------------------------------------------


def test_old_norse_acute_vowels_lengthen() -> None:
    """Old Norse: ó → OH, í → EE, etc."""
    assert "OH" in to_respelling("Jórvík", "old-norse")
    assert "EE" in to_respelling("Jórvík", "old-norse")
    assert "AH" in to_respelling("dálr", "old-norse")


def test_old_norse_j_is_y() -> None:
    """Old Norse j is consonantal /j/, English speakers say 'y'."""
    out = to_respelling("Jórvík", "old-norse")
    assert out.startswith("y")


def test_old_norse_thorn_and_eth() -> None:
    assert "th" in to_respelling("þorpr", "old-norse")
    assert "th" in to_respelling("víðr", "old-norse")


# --- Welsh ----------------------------------------------------------------


def test_welsh_dd_digraph() -> None:
    """Welsh 'dd' = English 'th' as in 'this'."""
    assert "th" in to_respelling("eddi", "welsh")


def test_welsh_ll_digraph() -> None:
    """Welsh 'll' is the voiceless lateral; respelled 'hl'."""
    assert "hl" in to_respelling("llan", "welsh")


def test_welsh_f_is_v() -> None:
    """Welsh single 'f' = English 'v'; double 'ff' = English 'f'."""
    assert "v" in to_respelling("nef", "welsh")
    assert "f" in to_respelling("ffordd", "welsh")


def test_welsh_ch_is_back_fricative() -> None:
    """Welsh 'ch' = /x/, respelled 'kh' for English speakers."""
    assert "kh" in to_respelling("Bach", "welsh")


# --- IPA output -----------------------------------------------------------


def test_to_ipa_returns_slashed_form() -> None:
    """IPA output is wrapped in /slashes/ by convention."""
    out = to_ipa("hām", "old-english")
    assert out.startswith("/") and out.endswith("/")


def test_to_ipa_uses_proper_symbols() -> None:
    """OE 'þ' → /θ/, OE 'ð' → /ð/, in their conventional IPA forms."""
    assert "θ" in to_ipa("þorn", "old-english")
    assert "ð" in to_ipa("ðæt", "old-english")
    assert "ɬ" in to_ipa("llan", "welsh")
    assert "ʃ" in to_ipa("sceap", "old-english")


# --- pass-through for unknown languages -----------------------------------


def test_unknown_language_returns_form_unchanged() -> None:
    """Languages not in PHONOLOGY should pass through respelling
    unchanged rather than crashing."""
    assert to_respelling("foobar", "klingon") == "foobar"
    # IPA pass-through still returns the form (no /slashes/ wrap since we
    # haven't actually converted). Caller can detect by absence of slashes.
    assert "foobar" in to_ipa("foobar", "klingon")
