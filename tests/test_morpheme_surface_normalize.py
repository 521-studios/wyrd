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
        ("-  ach  -", "ach"),  # nested spaces around boundary dashes
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
        ("-*-ach-", "*ach"),  # leading dash before sigil + boundary dashes
        ("-*ach-", "*ach"),  # leading dash before sigil
        ("* -at-", "*at"),  # whitespace between sigil and stem is re-stripped
        ("* -", None),  # sigil + whitespace + dash strips to junk → None
        ("*(H)réh₁-ti-s", "*(H)réh₁-ti-s"),  # PIE interior proto-segmentation kept
        ("*al-Quadim", "*al-Quadim"),  # sigil + ASCII interior hyphen (documented shape)
        # --- NON-ASCII boundary dashes are decoration too (D45 fork bug) -------
        ("–ach", "ach"),  # U+2013 EN DASH boundary marker
        ("—ach", "ach"),  # U+2014 EM DASH
        ("‐ach", "ach"),  # U+2010 HYPHEN (common in PDF/Wiktionary paste)
        ("‑ach", "ach"),  # U+2011 NON-BREAKING HYPHEN
        ("−ach", "ach"),  # U+2212 MINUS SIGN
        ("ach–", "ach"),  # trailing en-dash
        ("–*ach–", "*ach"),  # en-dash boundary markers around the sigil
        ("–", None),  # lone Unicode dash → junk
        ("al–Quadim", "al–Quadim"),  # interior en-dash is identity, NOT trimmed
        # --- non-ASCII boundary WHITESPACE is trimmed too (docstring promise) --
        ("\tach\t", "ach"),  # tabs
        ("\xa0ach\xa0", "ach"),  # NBSP (U+00A0) — realistic from web scrapes
        ("\xa0-ach-\xa0", "ach"),  # NBSP shields ASCII dashes → both must go
        # --- strip-to-empty junk → None (drop the record) ---------------------
        ("-", None),
        ("--", None),
        ("*-", None),
        ("-*", None),  # dash + sigil only → junk
        ("**", None),  # sigil-only stem → junk
        ("*", None),
        ("", None),
        ("   ", None),
        (None, None),
        (123, None),  # non-string input → None (defensive)
    ],
)
def test_normalize_morpheme_surface(raw, expected):
    assert normalize_morpheme_surface(raw) == expected


def test_normalize_is_idempotent():
    """Re-normalizing a bare survivor is a no-op — so wiring it at the central
    write choke can't drift a form that already passed through."""
    for raw in [
        "-ach",
        "ton-",
        "al-Quadim",
        "*-at-",
        "*tūn",
        "*al-Quadim",  # sigil + interior hyphen together
        "–ach",  # Unicode-dash boundary marker
        "\xa0-ach-\xa0",  # NBSP-shielded ASCII dashes
    ]:
        once = normalize_morpheme_surface(raw)
        assert normalize_morpheme_surface(once) == once


def test_every_boundary_dash_codepoint_is_trimmed():
    """Data-driven lock over the full ``_BOUNDARY_DASHES`` set: every codepoint is
    boundary decoration, so it trims at BOTH ends and a lone one is junk. Guards
    against a silent member-drop in a future cleanup — the obscure small/fullwidth
    variants (U+FE58/FE63/FF0D) have no representative case in the table above."""
    from wyrd.generators.kenning.lexicon.morpheme_surface import _BOUNDARY_DASHES

    for ch in _BOUNDARY_DASHES:
        tag = f"{ch!r} (U+{ord(ch):04X})"
        assert normalize_morpheme_surface(f"{ch}ach{ch}") == "ach", f"{tag} not trimmed"
        assert normalize_morpheme_surface(ch) is None, f"lone {tag} not junk"
