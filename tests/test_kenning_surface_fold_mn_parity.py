"""wyrd-nndd: cross-language surface-fold parity.

The two Python surface folds — ``bundle._subject._surface_fold`` and
``runtime.proportions._grid_match_key`` — and the SPA's ``accentFold`` are
DOCUMENTED to be in parity. Previously they diverged on combining-mark SCOPE:
the SPA dropped only the main Combining Diacritical Marks block (U+0300-U+036F)
while Python dropped by combining-class. This pins the unified contract — all
three drop category-**Mn** combining marks in ANY Unicode block, and all three
KEEP spacing (**Mc**) marks (e.g. Devanagari matras; unifying on ``\\p{M}``
instead would corrupt them). The matching JS vector is in
``spa-next/src/lib/accents.test.js``.

Marks are written as ``\\u`` escapes, never raw combining chars (which an
editor/git can normalize away).
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.lexicon.bundle._subject import _surface_fold
from wyrd.generators.kenning.runtime.proportions import _grid_match_key

# (input, expected) — identical to the accents.test.js vector.
_VECTOR = [
    ("\u00e9", "e"),  # e-acute (Mn U+0301, inside U+0300-036F)
    ("\u0101", "a"),  # a-macron (Mn U+0304)
    ("n\u05b4", "n"),  # Hebrew point hiriq (Mn, OUTSIDE U+0300-036F) - old JS kept, now dropped
    ("b\u064e", "b"),  # Arabic fatha (Mn)
    ("c\u1ab0", "c"),  # Combining Diacritical Marks Extended (Mn)
    ("k\u093e", "k\u093e"),  # Devanagari vowel sign AA (Mc, spacing) - KEPT by both
]


@pytest.mark.parametrize(("raw", "expected"), _VECTOR)
def test_surface_fold_drops_mn_any_block_keeps_mc(raw, expected):
    assert _surface_fold(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), _VECTOR)
def test_grid_match_key_matches_surface_fold_on_vector(raw, expected):
    assert _grid_match_key(raw) == expected
