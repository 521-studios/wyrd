"""Unit tests for ``normalize_morpheme_surface`` — de-dash etymon.canonical_form
(wyrd-aicu.8, D45).

The normalizer strips ONLY boundary (leading/trailing) dashes from a stored
morpheme surface — the affix-position decoration — while preserving interior
hyphens (legitimate in-word, ``al-Quadim``), the leading ``*`` reconstruction
sigil (its folding is deferred to wyrd-qoy8), and returning ``None`` for
strip-to-empty junk so the ingest boundary can drop the record.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.lexicon.morpheme_surface import normalize_morpheme_surface


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # --- boundary affix markers are trimmed --------------------------------
        ("-ach", "ach"),  # bound suffix
        ("ton-", "ton"),  # prefix
        ("-ar-", "ar"),  # infix (both boundaries)
        ("--ach", "ach"),  # doubled boundary dash
        ("  -ach  ", "ach"),  # surrounding whitespace stripped first
        # --- already bare is a no-op (idempotent) ------------------------------
        ("ton", "ton"),
        ("giles", "giles"),
        # --- interior hyphens are PRESERVED (not position decoration) ----------
        ("al-Quadim", "al-Quadim"),
        ("'s-Hertogenbosch", "'s-Hertogenbosch"),
        ("-al-Adha", "al-Adha"),  # boundary trimmed, interior kept
        # --- reconstruction sigil PRESERVED (wyrd-qoy8 owns its folding) -------
        ("*tūn", "*tūn"),
        ("*-at-", "*at"),  # sigil kept, boundary dash between sigil + stem trimmed
        ("*(H)réh₁-ti-s", "*(H)réh₁-ti-s"),  # PIE interior proto-segmentation kept
        # --- strip-to-empty junk → None (drop the record) ---------------------
        ("-", None),
        ("--", None),
        ("*-", None),
        ("*", None),
        ("", None),
        ("   ", None),
        (None, None),
    ],
)
def test_normalize_morpheme_surface(raw, expected):
    assert normalize_morpheme_surface(raw) == expected


def test_normalize_is_idempotent():
    """Re-normalizing a bare survivor is a no-op — so wiring it at the central
    write choke can't drift a form that already passed through."""
    for raw in ["-ach", "ton-", "al-Quadim", "*-at-", "*tūn"]:
        once = normalize_morpheme_surface(raw)
        assert normalize_morpheme_surface(once) == once
