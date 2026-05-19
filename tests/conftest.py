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

import contextlib
import functools
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest

import wyrd.generators.kenning as kenning_mod
from wyrd.generators.kenning.runtime.meaning import (
    load_canonical_decompositions,
    load_fantasy_morphemes,
    load_joiners,
    load_meanings,
)

# ---------------------------------------------------------------------------
# Isolation autouses
#
# Two autouse fixtures keep tests off the operator's live environment:
#
# 1. ``_isolate_default_lexicon_db`` — redirects ``_DEFAULT_LEXICON_PATH``
#    to a per-test ``tmp_path/test.db``. Tests that need a populated DB
#    create their own schema inside ``test.db`` (or pass ``--db`` to a
#    different tmp file). Tests that don't touch the DB don't care.
#
# 2. ``_isolate_bulk_manifest`` — redirects the bulk-source manifest
#    path to a per-test empty file. Without this, ``rebuild-from-jsonl``
#    tests would trigger the L1 wiktextract ingest against the
#    operator's full ``~/.wyrd/sources/`` (~5GB) — minutes-long locally
#    and broken on CI (no local cache).
#
# A test that needs the real lexicon DB or real bulk manifest can opt
# out via ``@pytest.mark.no_lexicon_isolation`` (defined below).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_default_lexicon_db(request, tmp_path, monkeypatch):
    """Point ``_DEFAULT_LEXICON_PATH`` at a per-test ``test.db``."""
    if request.node.get_closest_marker("no_lexicon_isolation") is not None:
        yield
        return
    test_db = tmp_path / "test.db"
    from wyrd.generators.kenning import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_DEFAULT_LEXICON_PATH", test_db, raising=False)
    yield test_db


@pytest.fixture(autouse=True)
def _isolate_bulk_manifest(request, tmp_path, monkeypatch):
    """Point ``bulk_sources.MANIFEST_PATH`` at a per-test empty
    manifest so the L1 wiktextract ingest path in
    ``rebuild-from-jsonl`` is a no-op unless a test deliberately
    populates the manifest. Mirrors the production manifest's
    schema_version=1 + zero slices."""
    if request.node.get_closest_marker("no_lexicon_isolation") is not None:
        yield
        return
    manifest = tmp_path / "_bulk_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bucket": "test-bucket",
                "region": "us-east-2",
                "s3_prefix": "wiktextract/v1",
                "compression": "zstd",
                "slices": [],
            }
        )
    )
    from wyrd.generators.kenning import bulk_sources as bs

    monkeypatch.setattr(bs, "MANIFEST_PATH", manifest, raising=False)
    yield manifest


def pytest_configure(config):
    """Register the opt-out marker so pytest doesn't warn on it."""
    config.addinivalue_line(
        "markers",
        "no_lexicon_isolation: opt out of the autouse default-DB and "
        "bulk-manifest isolation fixtures. Use sparingly — tests that "
        "need this almost always have a better fixture-based design.",
    )


@contextmanager
def swap_bundle(fixture_data: dict[str, Any] | list) -> Iterator[None]:
    """Run a test against ``fixture_data`` instead of the live
    runtime ``meanings.json`` bundle.

    Patches the lru-cached top-level loaders (``_load_meanings``,
    ``_load_joiners``, ``_load_canonical_decompositions``,
    ``_load_fantasy_morphemes``) with stub functions whose return
    values come from ``load_meanings(fixture_data)`` etc. directly.
    Doesn't write anywhere on disk — earlier versions of this helper
    overwrote the source-tree ``meanings.json`` then restored it in a
    ``finally`` block, but a hard interrupt between the write and the
    restore could leave the repo with the fixture corrupt-overwriting
    the real bundle. Mock-patching avoids that whole class of risk.

    Stubs are wrapped in ``functools.lru_cache(maxsize=1)`` so they
    expose a real ``cache_clear`` attribute — preserves the
    ``_load_meanings.cache_clear()`` call protocol that tests use to
    invalidate caches between scenarios. Production code does not
    call ``cache_clear`` (verified via grep), so the patch is fully
    transparent to runtime callers.

    Use the dict-shape (``{"subjects": [...], ...}``) per wyrd-c1vq;
    list-shape input also accepted for legacy back-compat by
    ``load_meanings``.
    """
    fake_meanings = load_meanings(fixture_data)
    fake_joiners = load_joiners(fixture_data)
    fake_canonical = load_canonical_decompositions(fixture_data)
    fake_fantasy = load_fantasy_morphemes(fixture_data)

    fake_load_meanings = functools.lru_cache(maxsize=1)(lambda: fake_meanings)
    fake_load_joiners = functools.lru_cache(maxsize=1)(lambda: fake_joiners)
    fake_load_canonical = functools.lru_cache(maxsize=1)(lambda: fake_canonical)
    fake_load_fantasy = functools.lru_cache(maxsize=1)(lambda: fake_fantasy)

    # wyrd-o9qi: the Generator subclasses (Kenning, KenningExplain, etc.)
    # moved to wyrd.generators.kenning.generators.<name> and each module
    # imports the loaders into its own namespace via
    # `from wyrd.generators.kenning import _load_meanings`. Patching only
    # `kenning_mod._load_meanings` would set the shim attribute but leave
    # the local-bound references in the consumer modules pointing at the
    # original function (the wyrd-g143 slice-5 monkey-patch class of
    # regression). Patch every consumer module's namespace as well so
    # the fixture reaches the call sites the Generator methods actually
    # resolve at runtime.
    from wyrd.generators.kenning.generators import (
        kenning as _kenning_mod,
    )
    from wyrd.generators.kenning.generators import (
        kenning_creature as _kenning_creature_mod,
    )
    from wyrd.generators.kenning.generators import (
        kenning_explain as _kenning_explain_mod,
    )
    from wyrd.generators.kenning.generators import (
        kenning_rewind as _kenning_rewind_mod,
    )

    patches = [
        # shim namespace — kept for back-compat with any caller that grabs
        # the helper via `from wyrd.generators.kenning import _load_meanings`
        # OUTSIDE a generator module (e.g. tests or future runtime callers).
        patch.object(kenning_mod, "_load_meanings", fake_load_meanings),
        patch.object(kenning_mod, "_load_joiners", fake_load_joiners),
        patch.object(kenning_mod, "_load_canonical_decompositions", fake_load_canonical),
        patch.object(kenning_mod, "_load_fantasy_morphemes", fake_load_fantasy),
        # per-generator consumer namespaces (the load-bearing patches)
        patch.object(_kenning_mod, "_load_joiners", fake_load_joiners),
        patch.object(_kenning_explain_mod, "_load_meanings", fake_load_meanings),
        patch.object(_kenning_explain_mod, "_load_canonical_decompositions", fake_load_canonical),
        patch.object(_kenning_rewind_mod, "_load_meanings", fake_load_meanings),
        patch.object(_kenning_creature_mod, "_load_fantasy_morphemes", fake_load_fantasy),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


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
