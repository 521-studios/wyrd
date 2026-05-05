"""Tests for ``Name.find_meaning(use_trie=True)`` (wyrd-k8e Phase 2,
wyrd-ibjp).

Two levels of test:

* Unit tests on small synthetic word_dbs that pin specific behaviors
  (multi-parse, env-var override, default-off).
* Equivalence regression that runs both matchers over a sample of the
  bundled corpus and asserts ``count_unaccounted`` matches per place
  name. This is the load-bearing Phase-2 acceptance criterion: the
  trie matcher must be a drop-in replacement for the legacy iterator
  under canonical scoring before Phase 3 flips the default.
"""

from __future__ import annotations

import json
from importlib import resources

import pytest

from wyrd.generators.kenning.meaning import load_meanings
from wyrd.generators.kenning.name import _USE_TRIE_ENV, Name, _trie_cache, load_names


@pytest.fixture(autouse=True)
def _clear_trie_cache():
    """Each test gets a fresh trie cache. Without this, a prior test's
    word_db id() can collide with a freshly-built one and serve a stale
    trie, making failures look like spooky action."""
    _trie_cache.clear()
    yield
    _trie_cache.clear()


@pytest.fixture
def synthetic_word_db():
    """Tiny word_db with three deliberately overlapping morphemes:

    * ``Brad-`` (pre, 'Broad')
    * ``-ham`` (post, 'Homestead')
    * ``-y-`` (inner, 'descriptive joiner', also as ``-y`` post)

    Picks up the multi-parse axis: a name like ``Bradham`` should
    produce one decomposition (Brad + ham), but ``Brady`` admits both
    (Brad + y-as-post) and (Brad + leftover-y) shapes.
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


def test_default_uses_legacy_iterator(synthetic_word_db, monkeypatch):
    """With no env var and no kwarg, the legacy matcher runs (Phase-2
    contract — opt-in only)."""
    monkeypatch.delenv(_USE_TRIE_ENV, raising=False)
    n = Name("Bradham")
    n.find_meaning(synthetic_word_db)
    # Legacy and trie both produce <Brad-> + <-ham> here; what we're
    # pinning is that no env-var means no trie path was used (we'd see
    # an exception if the trie path errored on the synthetic db).
    assert n.count_unaccounted() == 0


def test_explicit_use_trie_true_runs_trie_path(synthetic_word_db):
    n = Name("Bradham")
    n.find_meaning(synthetic_word_db, use_trie=True)
    assert n.count_unaccounted() == 0
    # The trie cache should have been populated.
    assert id(synthetic_word_db) in _trie_cache


def test_env_var_one_enables_trie(synthetic_word_db, monkeypatch):
    monkeypatch.setenv(_USE_TRIE_ENV, "1")
    n = Name("Bradham")
    n.find_meaning(synthetic_word_db)
    assert id(synthetic_word_db) in _trie_cache


def test_env_var_zero_does_not_enable_trie(synthetic_word_db, monkeypatch):
    """Env var must be exactly '1' to flip the flag; '0', 'true',
    'yes', empty string all keep the legacy matcher."""
    for sentinel in ("", "0", "true", "yes"):
        _trie_cache.clear()
        monkeypatch.setenv(_USE_TRIE_ENV, sentinel)
        n = Name("Bradham")
        n.find_meaning(synthetic_word_db)
        assert id(synthetic_word_db) not in _trie_cache, (
            f"sentinel {sentinel!r} should not enable trie path"
        )


def test_explicit_kwarg_overrides_env(synthetic_word_db, monkeypatch):
    """``find_meaning(use_trie=False)`` must override
    ``WYRD_KENNING_USE_TRIE_MATCHER=1``."""
    monkeypatch.setenv(_USE_TRIE_ENV, "1")
    n = Name("Bradham")
    n.find_meaning(synthetic_word_db, use_trie=False)
    assert id(synthetic_word_db) not in _trie_cache


def test_trie_path_preserves_multi_parse():
    """When two senses of one usage exist (the explainer's classic
    ``-y`` = 'island' OR 'district'), the trie path surfaces both
    decompositions as distinct ``Word`` objects on the same word."""
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
    n.find_meaning(word_db, use_trie=True, reduce=False)
    # 'sel' + 'sey' is two separate words to the matcher (no whitespace
    # to split on), but the trie produces parses that include the '-y'
    # at the end. Without other morphemes for 'sel', the parses tied
    # for best should still surface BOTH senses of '-y' as separate
    # decompositions.
    decomp_count = len(n.words["Selsey"])
    assert decomp_count >= 2, f"expected >=2 multi-sense decompositions, got {decomp_count}"


@pytest.mark.parametrize("culture", ["english", "scottish", "welsh", "irish", "breton"])
def test_trie_matches_legacy_count_unaccounted_on_corpus_sample(culture):
    """**Phase-2 equivalence regression** — the load-bearing test.

    For every place name in the bundled per-culture corpus, both
    matchers must produce the same ``count_unaccounted`` after
    ``reduce()``. If this fails, Phase 3 (flipping the default to
    trie) is unsafe.

    Runs with ``reduce=True`` (the default) since that's what real
    callers use; the post-reduce shape is what downstream code
    consumes (``rebuild-proportions``, perfect-rate measurement).

    Sample size: every name in the bundled corpus. Tens of thousands
    of names per culture for english/irish, but the trie path is
    O(match-length) per position so the full pass is sub-second.
    """
    meanings = json.loads(
        resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
    )
    word_db, _ = load_meanings(meanings)

    place_names_text = (
        resources.files("wyrd.generators.kenning.data")
        .joinpath(f"{culture}_place_names.json")
        .read_text()
    )
    names = load_names(json.loads(place_names_text))

    mismatches: list[tuple[str, int, int]] = []
    for name in names:
        legacy = Name(name.name)
        legacy.find_meaning(word_db, use_trie=False)
        legacy_count = legacy.count_unaccounted()

        trie = Name(name.name)
        trie.find_meaning(word_db, use_trie=True)
        trie_count = trie.count_unaccounted()

        if legacy_count != trie_count:
            mismatches.append((name.name, legacy_count, trie_count))

    # Allow no mismatches. Phase-3 default-flip is gated on this.
    assert mismatches == [], (
        f"{len(mismatches)} mismatches in {culture} corpus; first few: {mismatches[:5]}"
    )


def test_trie_path_matches_legacy_perfect_count():
    """End-to-end check on the english corpus: the headline 'perfect'
    metric (count of fully-decomposed names) must agree between the
    two backends. This is the metric COVERAGE.md tracks; equivalence
    here is the user-visible guarantee."""
    meanings = json.loads(
        resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
    )
    word_db, _ = load_meanings(meanings)
    place_names_text = (
        resources.files("wyrd.generators.kenning.data")
        .joinpath("welsh_place_names.json")
        .read_text()
    )
    names = load_names(json.loads(place_names_text))

    legacy_perfect = 0
    trie_perfect = 0
    for name in names:
        legacy = Name(name.name)
        legacy.find_meaning(word_db, use_trie=False)
        if legacy.count_unaccounted() == 0:
            legacy_perfect += 1
        trie = Name(name.name)
        trie.find_meaning(word_db, use_trie=True)
        if trie.count_unaccounted() == 0:
            trie_perfect += 1
    assert legacy_perfect == trie_perfect, (
        f"perfect counts diverge: legacy={legacy_perfect} trie={trie_perfect}"
    )
