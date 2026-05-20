"""Tests for ``Name.find_meaning`` (the trie matcher, wyrd-k8e).

After wyrd-zhhz / Phase 3 the trie is the only matcher — the legacy
iterator and the ``use_trie`` flag both went away. These tests pin
the trie path's documented contracts:

* default ``find_meaning`` populates the trie cache (proves the trie
  path is on the hot path);
* multi-parse semantics (multiple senses for one surface, multiple
  boundary placements) survive the canonical filter;
* the trie runs cleanly across the bundled corpus and produces a
  positive perfect-decomposition count (smoke test that catches
  trivial regressions like an empty bundle or a crashing matcher).
"""

from __future__ import annotations

import json
from importlib import resources

import pytest

from wyrd.generators.kenning.runtime.meaning import load_meanings
from wyrd.generators.kenning.runtime.name import Name, _trie_cache, load_names


@pytest.fixture(autouse=True)
def _clear_trie_cache(request):
    """Each test gets a fresh trie cache. Without this, a prior test's
    word_db id() can collide with a freshly-built one and serve a stale
    trie, making failures look like spooky action.

    Skip the clear when ``bundle_word_db`` is in scope: that fixture is
    module-scoped, so its dict id is stable across the parametrized
    runs of ``test_find_meaning_runs_full_bundled_corpus_without_crashing``
    and clearing the cache between them just forces 5 redundant trie
    rebuilds against the now-18k-word post-Phase-3 bundle. The stale-id
    safety only matters when test-local word_dbs get GC'd between
    runs."""
    if "bundle_word_db" in request.fixturenames:
        yield
        return
    _trie_cache.clear()
    yield
    _trie_cache.clear()


@pytest.fixture(scope="module")
def bundle_word_db():
    """Module-scoped: load the bundled meanings.json once and share
    the resulting word_db across the corpus-smoke tests. Cuts ~5x the
    load_meanings overhead vs per-test loading on the larger
    post-wyrd-4hx7 bundle."""
    meanings = json.loads(
        resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
    )
    word_db, _ = load_meanings(meanings)
    return word_db


@pytest.fixture
def synthetic_word_db():
    """Tiny word_db with two morphemes:

    * ``Brad-`` (pre, 'Broad')
    * ``-ham`` (post, 'Homestead')

    Enough to drive the basic-decomposition assertion without dragging
    the whole bundled meanings.json through every test.
    """
    return load_meanings(
        [
            {
                "meaning": ["Broad"],
                "modifier_tags": ["descriptive"],
                "modifier_type": "Descriptive",
                "words": [{"modern_usage": "Brad-", "old_english": ["brad"]}],
            },
            {
                "meaning": ["Homestead"],
                "modifier_tags": ["architecture"],
                "modifier_type": "Architectural",
                "words": [{"modern_usage": "-ham", "old_english": ["ham"]}],
            },
        ]
    )[0]


def test_find_meaning_routes_through_trie_cache(synthetic_word_db):
    """``find_meaning`` must build (and cache) a trie for the supplied
    word_db. Pin the cache-population invariant so a regression that
    bypasses ``_trie_for`` would surface as an unexercised cache."""
    n = Name("Bradham")
    n.find_meaning(synthetic_word_db)
    assert n.count_unaccounted() == 0
    assert id(synthetic_word_db) in _trie_cache


def test_find_meaning_preserves_multi_sense_decompositions():
    """When two senses of one usage exist (the explainer's classic
    ``-y`` = 'island' OR 'district'), ``find_meaning(reduce=False)``
    surfaces both decompositions as distinct ``Word`` objects on the
    same word."""
    word_db, _ = load_meanings(
        [
            # Two subjects, both registering '-y' (post-suffix). Sharing
            # surface, distinct meanings — the explainer's multi-parse
            # axis.
            {
                "meaning": ["Island"],
                "modifier_tags": ["topography"],
                "modifier_type": "Topographical",
                "words": [{"modern_usage": "-y", "old_english": ["ieg"]}],
            },
            {
                "meaning": ["District"],
                "modifier_tags": ["social"],
                "modifier_type": "Topographical",
                "words": [{"modern_usage": "-y", "old_english": ["ie"]}],
            },
        ]
    )
    n = Name("Selsey")
    n.find_meaning(word_db, reduce=False)
    # 'sel' + 'sey' is one word to the matcher (no whitespace to split
    # on), but the trie produces parses that include the '-y' at the
    # end. Without other morphemes for 'sel', the parses tied for best
    # should still surface BOTH senses of '-y' as separate
    # decompositions.
    decomp_count = len(n.words["Selsey"])
    assert decomp_count >= 2, f"expected >=2 multi-sense decompositions, got {decomp_count}"


@pytest.mark.parametrize("culture", ["english", "scottish", "welsh", "irish", "breton"])
def test_find_meaning_runs_full_bundled_corpus_without_crashing(culture, bundle_word_db):
    """Smoke test — every name in the bundled per-culture corpus must
    decompose cleanly without raising, and the perfect-decomposition
    rate must stay above a floor. The previous Phase-2 regression
    that lived here compared trie vs legacy decision parity; with
    legacy gone, this rate-floor check is what's left to guard
    against matcher-or-bundle regressions.

    Floor of 0.5% is a deliberate compromise. As of 2026-05-07 actual
    per-culture rates against the bare ``meanings.json`` bundle (no
    sidecars — this fixture loads the bundle directly) are english
    35.2%, scottish 25.3%, welsh 23.6%, irish 17.1%, breton 2.2%.
    Production runtime adds the wyrd-1cjg Irish anglicization sidecar
    via ``_load_meanings``, lifting Irish to 17.7%; the per-sidecar
    lift is pinned in ``test_kenning_irish_anglicizations.py``.
    Breton sets the floor — at 0.5% the test gives Breton ~4x
    headroom while still catching an order-of-magnitude regression
    for any culture (e.g. corrupted trie or empty bundle would drop
    every culture to ~0%). Tighter thresholds risk false positives
    during morpheme-mining churn; looser thresholds miss real
    regressions.

    Sample size: every name in the bundled corpus. Tens of thousands of
    names per culture for english/irish, but the trie path is
    O(match-length) per position so the full pass is sub-second.
    """
    place_names_text = (
        resources.files("wyrd.generators.kenning.data")
        .joinpath(f"{culture}_place_names.json")
        .read_text()
    )
    names = load_names(json.loads(place_names_text))

    perfect = 0
    for name in names:
        n = Name(name.name)
        n.find_meaning(bundle_word_db)
        if n.count_unaccounted() == 0:
            perfect += 1

    rate = perfect / len(names) if names else 0.0
    assert rate > 0.005, (
        f"{culture} corpus perfect-decomposition rate fell below 0.5% "
        f"({perfect}/{len(names)} = {rate * 100:.2f}%) — likely a matcher "
        f"or bundle regression"
    )
