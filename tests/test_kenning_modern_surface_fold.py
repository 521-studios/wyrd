"""wyrd-f233: reflex-less families synthesize a clean modern surface.

A breakdown morpheme with no linked reflex falls back to canonical-form synthesis
at bundle export. Before f233 the raw canonical form (macron / proto ``*``) leaked
into the matcher key + rendered output (``tūn`` → key ``tūn``). The fold strips
macrons + proto markers; position decoration (D45) is applied separately, so dashes
are never folded into identity.

Scope is macron+proto ONLY (not OE/ON letter mapping æ/þ/ð/ø) — see the consistency
test below for why: the fold must agree with the matcher's own fold so a synthesized
surface still binds to its reflex cell (the wyrd-mook class). OE-letter rendering
cleanup is wyrd-7fs2.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.bundle._emit import (
    _fold_to_modern_surface,
    _synthesize_modern_usage,
)
from wyrd.generators.kenning.lexicon.genitive_priors import fold_surface


def test_fold_strips_macrons_and_proto_markers():
    assert _fold_to_modern_surface("tūn") == "tun"  # OE macron
    assert _fold_to_modern_surface("dūn") == "dun"
    assert _fold_to_modern_surface("bȳ") == "by"  # ON y-macron
    assert _fold_to_modern_surface("*beuganą") == "beugana"  # proto marker + ogonek
    assert _fold_to_modern_surface("bri*or") == "brior"  # internal proto marker too
    assert _fold_to_modern_surface("Lēah") == "Leah"  # case preserved
    assert _fold_to_modern_surface("ford") == "ford"  # already clean → unchanged
    assert _fold_to_modern_surface("") == ""


def test_fold_keeps_oe_on_letters_and_non_latin():
    # macron+proto only: OE/ON letters are NOT ASCII-mapped here (that would
    # disagree with the matcher fold — see the consistency test). þ/ð/ø pass
    # through; Semitic ʕ/ʔ pass through. ǣ is kept RAW (with macron): stripping it
    # to æ would fold to 'ae' while the raw ǣ folds away, so self-verify keeps raw.
    assert _fold_to_modern_surface("ǣcer") == "ǣcer"  # kept raw (fold-inconsistent to strip)
    assert _fold_to_modern_surface("þorp") == "þorp"
    assert _fold_to_modern_surface("ðæc") == "ðæc"
    assert _fold_to_modern_surface("søkkva") == "søkkva"
    assert _fold_to_modern_surface("ʔanti") == "ʔanti"  # non-Latin passthrough


def test_fold_is_matcher_consistent():
    """The wyrd-mook invariant: a synthesized surface must fold (via the runtime
    matcher fold) to the SAME key as its canonical form, so it binds to its own
    reflex cell. Macron+proto stripping preserves this for every letter class —
    ASCII-mapping OE letters here would break it (ð→d in the matcher vs th here)."""
    for canonical in ("tūn", "ǣcer", "þorp", "ðæc", "søkkva", "*beuganą", "bȳ"):
        assert fold_surface(_fold_to_modern_surface(canonical)) == fold_surface(canonical)


def test_fold_does_not_touch_dashes():
    # Dashes are position decoration (D45), applied AFTER the fold — never identity.
    assert _fold_to_modern_surface("-tūn") == "-tun"
    assert _fold_to_modern_surface("þ-orp") == "þ-orp"


def test_synthesize_applies_fold_then_position_decoration():
    # bare (no position) → folded surface, no dashes
    assert _synthesize_modern_usage({"root_canonical_form": "tūn"}) == "tun"
    # pre / inner / post wrap the FOLDED surface in dashes
    assert (
        _synthesize_modern_usage({"root_canonical_form": "dūn", "position_pref": "pre"}) == "dun-"
    )
    assert (
        _synthesize_modern_usage({"root_canonical_form": "tūn", "position_pref": "post"}) == "-tun"
    )
    assert (
        _synthesize_modern_usage({"root_canonical_form": "bȳ", "position_pref": "inner"}) == "-by-"
    )
