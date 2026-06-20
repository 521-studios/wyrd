"""wyrd-an8u: surrounding whitespace is never part of a morpheme's identity.

A dirty usage like ``'Oak- '`` (trailing space) used to fold to a distinct
bare surface (``'Oak '``) from the clean ``'Oak-'`` → ``'Oak'``, duplicating
the meaning row and splitting proportion weight across the space/no-space
variants. The three surface-key creation sites now strip surrounding
whitespace alongside the dash-fold:

  * ``word._bare_surface`` — the proportions usages / single_usages /
    bare_word_position key path (and the render key).
  * ``runtime_db_export._bare_modern_usage`` — the meaning / dormant-morpheme
    blob ``modern_usage`` key path.
  * ``proportions_builder._accumulate_attested_languages`` — the
    proportions_attested_language key path.
"""

from __future__ import annotations

from types import SimpleNamespace

from wyrd.generators.kenning.lexicon.proportions_builder import (
    _accumulate_attested_languages,
)
from wyrd.generators.kenning.lexicon.runtime_db_export import _bare_modern_usage
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.word import Word, _bare_surface


def _meaning(usage: str) -> Meaning:
    return Meaning(
        usage=usage,
        tags=["x"],
        meanings=["gloss"],
        sources={"old_english": [usage.lower().replace("-", "").strip()]},
    )


# ---- word._bare_surface (proportions usages + render key) ------------------


def test_bare_surface_strips_trailing_whitespace():
    assert _bare_surface(_meaning("Oak- ")) == "Oak"
    assert _bare_surface(_meaning("Har ")) == "Har"


def test_bare_surface_dirty_and_clean_fold_to_same_identity():
    # The merge guarantee: a whitespace-dirty surface keys identically to the
    # clean one, so their proportion counts sum instead of splitting.
    assert _bare_surface(_meaning("Oak- ")) == _bare_surface(_meaning("Oak-"))


def test_word_lone_sample_uses_whitespace_clean_surface():
    assert Word([_meaning("Oak- ")]).get_lone_samples() == {"Oak"}


def test_word_compound_samples_use_whitespace_clean_surface():
    samples = Word([_meaning("Oak- "), _meaning("ton")]).get_samples()
    assert ("Oak", "pre") in samples
    assert all(" " not in surface for surface, _position in samples)


# ---- runtime_db_export._bare_modern_usage (meaning blob key) ---------------


def test_bare_modern_usage_strips_dash_and_whitespace():
    assert _bare_modern_usage("Oak- ") == "Oak"
    assert _bare_modern_usage("Har ") == "Har"


def test_bare_modern_usage_dirty_and_clean_collide():
    # Same key → _write_meanings groups them into ONE meaning row (unioned
    # entries) rather than two whitespace-variant rows.
    assert _bare_modern_usage("Oak- ") == _bare_modern_usage("Oak-")


# ---- proportions_builder attested-language key -----------------------------


def test_attested_language_surface_strips_whitespace():
    name = SimpleNamespace(words={"w": [Word([_meaning("Oak- ")])]})
    attested: dict[str, set[str]] = {}
    _accumulate_attested_languages(name, attested)
    # Keyed by the clean bare surface, not 'oak ' with a trailing space.
    assert "oak" in attested
    assert all(" " not in surface for surface in attested)
