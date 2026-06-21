"""wyrd-f233: reflex-less families synthesize a CLEAN ASCII modern surface.

A breakdown morpheme with no linked reflex falls back to canonical-form synthesis
at bundle export. Before f233 the raw canonical form (macron / proto ``*`` /
OE letters) leaked into the matcher key + rendered output (``tūn`` → key ``tūn``).
The fold turns it into a real modern surface; position decoration (D45) is applied
separately, so dashes are never folded into identity.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.bundle._emit import (
    _fold_to_modern_surface,
    _synthesize_modern_usage,
)


def test_fold_drops_macrons_proto_and_oe_letters():
    assert _fold_to_modern_surface("tūn") == "tun"  # OE macron
    assert _fold_to_modern_surface("dūn") == "dun"
    assert _fold_to_modern_surface("bȳ") == "by"  # ON y-macron
    assert _fold_to_modern_surface("*beuganą") == "beugana"  # proto marker + ogonek
    assert _fold_to_modern_surface("ǣcer") == "aecer"  # æ-macron → ae
    assert _fold_to_modern_surface("þorp") == "thorp"  # thorn
    assert _fold_to_modern_surface("ðæc") == "thaec"  # eth + ash
    assert _fold_to_modern_surface("søkkva") == "sokkva"  # ON ø
    assert _fold_to_modern_surface("bri*or") == "brior"  # internal proto marker too


def test_fold_preserves_case_and_plain_ascii():
    assert _fold_to_modern_surface("Lēah") == "Leah"  # case preserved
    assert _fold_to_modern_surface("ford") == "ford"  # already clean → unchanged
    assert _fold_to_modern_surface("") == ""


def test_fold_does_not_touch_dashes():
    # Dashes are position decoration (D45), applied AFTER the fold — never identity.
    assert _fold_to_modern_surface("-tūn") == "-tun"
    assert _fold_to_modern_surface("ǣ-cer") == "ae-cer"


def test_synthesize_applies_fold_then_position_decoration():
    # bare (no position) → folded surface, no dashes
    assert _synthesize_modern_usage({"root_canonical_form": "tūn"}) == "tun"
    # pre / inner / post wrap the FOLDED surface in dashes
    assert (
        _synthesize_modern_usage({"root_canonical_form": "ǣcer", "position_pref": "pre"})
        == "aecer-"
    )
    assert (
        _synthesize_modern_usage({"root_canonical_form": "tūn", "position_pref": "post"}) == "-tun"
    )
    assert (
        _synthesize_modern_usage({"root_canonical_form": "bȳ", "position_pref": "inner"}) == "-by-"
    )
