"""Shared pytest fixtures and helpers for the wyrd test suite.

Notable: ``swap_bundle`` — a context manager that temporarily
replaces the runtime ``meanings.json`` with a fixture so tests
that exercise generator behaviour (decomposition picks, language
tag assignments, decomposition counts) can pin their assertions
to a SMALL stable fixture rather than the live bundle.

Without this, every bundle re-emit (mining adds a few hundred
new morphemes; cluster expansion adds a few thousand subjects)
ripples through the test suite as a cascade of brittle re-tunings:
'Bridgewater' resolves to a different language pick because the
new bundle has more morphemes for 'bridge'; decomposition counts
shift; tag co-occurrence pairs change. The fix is to decouple
behavioural tests from bundle churn — pin them to a fixture
crafted to produce the exact decomposition the test asserts.

Live-bundle smoke tests (``test_find_meaning_runs_full_bundled_corpus_without_crashing``,
``test_sidecar_lifts_irish_corpus_perfect_rate``) intentionally
stay against the real bundle — they're broad-stroke regression
guards, not behavioural pins.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import wyrd.generators.kenning as kenning_mod


@contextmanager
def swap_bundle(fixture_data: dict[str, Any] | list) -> Iterator[None]:
    """Temporarily replace the runtime ``meanings.json`` bundle with
    ``fixture_data`` for the duration of a test.

    Writes the fixture to ``wyrd/generators/kenning/data/meanings.json``
    (the same path ``importlib.resources`` resolves), clears the
    coupled bundle caches (``_load_meanings``, ``_load_culture``,
    ``_load_joiners``, ``_load_canonical_decompositions``,
    ``_load_fantasy_morphemes``), runs the test body, then restores
    the original on exit.

    Use the dict-shape (``{"subjects": [...], ...}``) per wyrd-c1vq;
    list-shape input also accepted for legacy back-compat.
    """
    bundle_path = Path(kenning_mod.__file__).parent / "data" / "meanings.json"
    original = bundle_path.read_text()
    try:
        bundle_path.write_text(json.dumps(fixture_data))
        kenning_mod._coupled_cache_clear()
        yield
    finally:
        bundle_path.write_text(original)
        kenning_mod._coupled_cache_clear()


def _word(modern_usage: str, **lang_forms: list[str]) -> dict:
    """Convenience: build a single word dict for a fixture subject."""
    return {"modern_usage": modern_usage, **lang_forms}


def _subject(meaning: list[str], modifier_tags: list[str], words: list[dict]) -> dict:
    """Convenience: build a single subject for a fixture bundle."""
    return {
        "meaning": meaning,
        "modifier_tags": modifier_tags,
        "modifier_type": None,
        "words": words,
    }


# Reusable fixture for explain / decomposition tests. Built around the
# canonical test names (Bridgewater, Saint Albans, Ashton, Aberystwyth,
# Westminster) so test assertions stay stable as the live bundle
# evolves. Keep this list focused — adding morphemes here can shift
# every fixture-using test's decomposition.
# Note on location encoding: ``Meaning._set_location`` derives
# location from the usage's dash positions:
#
#   * usage starts AND ends with ``-`` → ``inner``  (matches anywhere)
#   * usage ends with ``-`` only       → ``pre``    (must start at pos 0)
#   * else                              → ``post``   (must end at len(word))
#
# So a single-word match like Saint→"Saint" works as ``post`` (the
# match consumes the whole word, end == len). For compound names like
# "Bridgewater" the leading element needs ``-`` suffix → ``pre`` so
# the matcher allows starting at pos 0 with arbitrary end. Trailing
# elements need ``-`` prefix → ``post`` so end must == len.
EXPLAIN_TEST_BUNDLE: dict[str, Any] = {
    "subjects": [
        _subject(
            ["bridge"],
            ["topography", "architecture"],
            [_word("Bridge-", old_english=["brycg"])],
        ),
        _subject(
            ["water", "river", "stream"],
            ["water", "topography"],
            [_word("-water", old_english=["wæter"])],
        ),
        _subject(
            ["saint"],
            ["religious", "saint"],
            [_word("Saint", old_french=["seint"])],
        ),
        _subject(
            ["Alban (Name)"],
            ["male name", "saint"],
            [_word("Alban", latin=["Albanus"])],
        ),
        _subject(
            ["Alban (Name)"],
            ["male name", "saint"],
            [_word("Albans", latin=["Albanus"])],
        ),
        _subject(
            ["ash tree"],
            ["plant", "tree"],
            [_word("Ash-", old_english=["æsc"])],
        ),
        _subject(
            ["enclosure", "settlement"],
            ["architecture", "social"],
            [_word("-ton", old_english=["tūn"])],
        ),
        _subject(
            ["west", "western"],
            ["direction"],
            [_word("West-", old_english=["west"])],
        ),
        _subject(
            ["minster", "monastery"],
            ["religious", "architecture"],
            [_word("-minster", old_english=["mynster"])],
        ),
    ]
}


# Pytest fixtures — exposed via parameter injection so tests don't
# need to import from this module directly. Cross-module imports
# (``from tests.conftest import …``) work locally but break in CI's
# flat sys.path layout (no ``tests/__init__.py``); fixture injection
# sidesteps that entirely.


@pytest.fixture
def explain_test_bundle() -> dict[str, Any]:
    """Returns the canonical EXPLAIN_TEST_BUNDLE fixture data."""
    return EXPLAIN_TEST_BUNDLE


@pytest.fixture
def bundle_swapper():
    """Returns the ``swap_bundle`` context manager. Use as
    ``with bundle_swapper(fixture): ...`` inside a test body."""
    return swap_bundle
