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

import pytest

from wyrd.generators.kenning.paths import seed_data_path
from wyrd.generators.kenning.runtime.meaning import load_meanings
from wyrd.generators.kenning.runtime.name import Name, _trie_cache, load_names


@pytest.fixture(autouse=True)
def _clear_trie_cache():
    """Each test gets a fresh trie cache. Without this, a prior test's
    word_db id() can collide with a freshly-built one and serve a stale
    trie, making failures look like spooky action."""
    _trie_cache.clear()
    yield
    _trie_cache.clear()


@pytest.fixture(scope="module")
def bundle_word_db():
    """Module-scoped: load the runtime word_db once and share it
    across the corpus-smoke tests. Reads from the L4 SQLite path
    (defaults to the bundled seed-runtime.db; ``WYRD_RUNTIME_DB``
    overrides to a full bundle when corpus-quality coverage matters)."""
    from wyrd.generators.kenning import _load_meanings

    word_db, _ = _load_meanings()
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


def test_reduce_dedups_identical_words_but_keeps_distinct_senses():
    """wyrd-1bkw: ``Name.reduce``'s dedup keys on ``tuple(meaning.word)`` —
    identity for Meaning slots, value for str slots. Pin the two invariants that
    key rests on, since the perf fix (O(n²) list scan → set) and a future
    ``Meaning.__hash__``/``__eq__`` override could each quietly break them:

    1. Two decompositions that are the SAME Meaning instances collapse to one.
    2. Two decompositions sharing a surface but using DISTINCT Meaning senses
       (the '-y' = 'island' OR 'district' case) both survive — they are NOT
       merged by surface. A surface-based Meaning hash would silently fail this.
    """
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.word import Word

    island = Meaning("-y", [], ["Island"], {"old_english": ["ieg"]})
    district = Meaning("-y", [], ["District"], {"old_english": ["ie"]})

    n = Name("Selsey")
    # Pre-populate with a duplicate (same `island` instance twice) plus the
    # distinct `district` sense. All three are zero-unaccounted, size-1 — so the
    # score filters keep them all and only the dedup decides cardinality.
    n.words = {"Selsey": [Word([island]), Word([island]), Word([district])]}
    n.reduce()

    survivors = n.words["Selsey"]
    assert len(survivors) == 2, (
        f"expected the identical pair to collapse to 2, got {len(survivors)}"
    )
    glosses = {tuple(w.word[0].meanings) for w in survivors}
    assert glosses == {("Island",), ("District",)}, f"both senses must survive, got {glosses}"


# Target sample size per culture for the corpus smoke test below. The full
# english/irish corpora are tens of thousands of names; a deterministic strided
# sample of roughly this size keeps CI fast while still estimating the rate
# floor and walking a broad slice for the crash smoke. It's a target, not a
# hard ceiling: the slice is ``names[::max(1, len(names) // target)]``, so it
# lands near (usually slightly above) the target rather than exactly on it.
_SMOKE_SAMPLE_TARGET = 1000


@pytest.mark.parametrize("culture", ["english", "scottish", "welsh", "irish", "breton"])
def test_find_meaning_runs_full_bundled_corpus_without_crashing(culture, bundle_word_db):
    """Smoke test — every sampled name in the bundled per-culture
    corpus must decompose cleanly without raising, and the
    perfect-decomposition rate must stay above a floor. The previous
    Phase-2 regression
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

    Sample size: a deterministic strided sample targeting roughly
    ``_SMOKE_SAMPLE_TARGET`` names per culture (``names[::stride]`` with
    ``stride = max(1, len(names) // target)``). The full corpora are tens of
    thousands of names for english (~17.9k) and irish (~34k). A full pass
    over irish is ~20 s (wyrd-1bkw fixed the O(n²) dedup in
    ``Name.reduce`` that had made it ~150 s); striding to ~1k names keeps
    the pass well under a second while preserving both this test's jobs:
    the perfect-decomposition rate is a statistical property a
    deterministic ~1k-name sample estimates well above the 0.5% floor,
    and the crash smoke still walks a broad, evenly-spaced slice.
    Systematic (strided) sampling stays representative only while corpus
    order is uncorrelated with decomposition cost — it is today (long,
    cap-stressing names are spread across the alphabetically-sorted
    files), and the assertion below tripwires a future reorder that
    buried them in a stride gap. Cultures under ~2x the target — welsh
    (~1.9k) and breton (~1.2k) — run in full at stride 1; scottish
    (~2.3k) is just over, at stride 2. Sampling is retained even post-fix
    so this never becomes the slowest test (the ~20 s full pass would be
    the xdist per-worker floor).
    """
    place_names_text = seed_data_path(f"{culture}_place_names.json").read_text()
    names = load_names(json.loads(place_names_text))

    stride = max(1, len(names) // _SMOKE_SAMPLE_TARGET)
    sample = names[::stride]

    # Crash-smoke tripwire: the sample must still include the long,
    # decomposition-stressing names (the ones that dominate the full pass,
    # ~20 s post-wyrd-1bkw). With today's alphabetically-sorted corpora they're spread
    # throughout, so striding keeps tens-to-hundreds of them per culture; this
    # fires only if a future reorder clustered them into a single skipped
    # stride gap — which would gut crash coverage while the rate floor below
    # still passed.
    assert any(len(name.name) >= 15 for name in sample), (
        f"{culture} sample has no name >= 15 chars (sampled {len(sample)}/"
        f"{len(names)} at stride {stride}) — corpus order may have changed so "
        f"the strided crash-smoke no longer covers long, cap-stressing names"
    )

    perfect = 0
    for name in sample:
        n = Name(name.name)
        n.find_meaning(bundle_word_db)
        if n.count_unaccounted() == 0:
            perfect += 1

    rate = perfect / len(sample) if sample else 0.0
    assert rate > 0.005, (
        f"{culture} corpus perfect-decomposition rate fell below 0.5% "
        f"({perfect}/{len(sample)} = {rate * 100:.2f}%, sampled "
        f"{len(sample)}/{len(names)} at stride {stride}) — likely a "
        f"matcher or bundle regression"
    )


# ---------------------------------------------------------------------------
# wyrd-pfoo: culture_languages plumbing
# ---------------------------------------------------------------------------


def test_find_meaning_forwards_culture_languages_to_matcher(monkeypatch):
    """``Name.find_meaning`` forwards the new ``culture_languages``
    kwarg through to ``canonical_decompositions``. Pin the wire so a
    refactor that drops the forward silently makes the per-culture
    splitter no-op the tiebreaker."""
    from wyrd.generators.kenning.runtime import name as name_mod
    from wyrd.generators.kenning.runtime.meaning import Meaning

    captured: dict = {}

    def fake_canonical(word, trie, *, culture_languages=None):
        captured["called"] = True
        captured["culture_languages"] = culture_languages
        # Return a minimal valid decomposition so find_meaning's
        # downstream processing doesn't crash.
        return [[word]]

    monkeypatch.setattr(name_mod, "canonical_decompositions", fake_canonical)
    word_db = {"Foo-": [Meaning("Foo-", [], [], {"old_english": ["foo"]})]}
    n = Name("Foo")
    expected = frozenset({"celtic_mix"})
    n.find_meaning(word_db, culture_languages=expected)
    assert captured.get("called") is True
    assert captured["culture_languages"] == expected


def test_find_meaning_default_culture_languages_is_none(monkeypatch):
    """When the caller doesn't supply ``culture_languages``, the
    forwarded value is None — the bit-stable path for the explainer /
    rewind / era-map callers."""
    from wyrd.generators.kenning.runtime import name as name_mod
    from wyrd.generators.kenning.runtime.meaning import Meaning

    captured: dict = {}

    def fake_canonical(word, trie, *, culture_languages=None):
        captured["culture_languages"] = culture_languages
        return [[word]]

    monkeypatch.setattr(name_mod, "canonical_decompositions", fake_canonical)
    word_db = {"Foo-": [Meaning("Foo-", [], [], {"old_english": ["foo"]})]}
    n = Name("Foo")
    n.find_meaning(word_db)
    assert captured["culture_languages"] is None
