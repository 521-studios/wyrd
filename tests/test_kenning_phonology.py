"""Tests for the per-language phonology tables.

These exercise the GM-facing respelling and IPA outputs for representative
etymons in each supported language. The expected outputs are conservative
(roughly what a SAMPA-lite respelling would say) — when scholarly sources
disagree, we go with the most readable English approximation.
"""

from __future__ import annotations

from wyrd.generators.kenning.registers.phonology import (
    PHONOLOGY,
    to_ipa,
)

# --- registry sanity ------------------------------------------------------


def test_phonology_registry_covers_expected_languages() -> None:
    for lang in ("old-english", "old-norse", "welsh"):
        assert lang in PHONOLOGY
        assert len(PHONOLOGY[lang]) > 10


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
    """Languages not in PHONOLOGY should pass through IPA conversion
    unchanged rather than crashing."""
    # IPA pass-through still returns the form (no /slashes/ wrap since we
    # haven't actually converted). Caller can detect by absence of slashes.
    assert "foobar" in to_ipa("foobar", "klingon")
