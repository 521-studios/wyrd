"""Name: a full place name (potentially multiple words), decomposable into Words.

Ported from Rando (rando/name.py) — Python 2 → 3.

Two matcher backends share the same ``Name.find_meaning`` entry point:

* **Legacy iterator** (``Word.extract_meanings``) — original Rando-port
  matcher. O(M) per word per recursion level; iterates every Meaning at
  every position. Produces every parse, then ``reduce()`` filters to
  the minimum-unaccounted ones.
* **Trie matcher** (``trie_matcher.canonical_decompositions``,
  wyrd-k8e). O(match-length) per position via a character trie, then
  DFS-enumerates the segmentation DAG. ~200x faster on routine corpus
  passes; same multi-parse semantics.

Selected via the ``use_trie`` arg or the ``WYRD_KENNING_USE_TRIE_MATCHER``
env var. Default is the legacy matcher (Phase 2 of wyrd-k8e ships the
trie behind a flag; Phase 3 will flip the default and remove the
legacy iterator). The two should produce equivalent ``count_unaccounted``
across the bundled corpus — see ``tests/test_kenning_trie_equivalence.py``.
"""

from __future__ import annotations

import itertools
import os
from collections import OrderedDict

from wyrd.generators.kenning.trie_matcher import (
    MorphemeTrie,
    all_decompositions,
    build_morpheme_trie,
    canonical_decompositions,
)
from wyrd.generators.kenning.word import Word

_USE_TRIE_ENV = "WYRD_KENNING_USE_TRIE_MATCHER"

# Bounded LRU memoization of the trie per word_db identity. Building
# the trie is O(M) where M is the morpheme count (~4ms on a 2.9K-
# morpheme bundle), which is cheap once but adds up across the per-
# name loop that rebuild-proportions / matcher consumers run (tens of
# thousands of names → minutes of avoidable trie-builds without the
# cache).
#
# Three assumptions / limitations of this approach:
#
# (a) **word_db is treated as immutable** for the duration of its
#     entry in the cache. Mutating a word_db after a trie has been
#     cached for it (e.g. ``word_db['new-key'] = [...]``) will not
#     invalidate the trie; subsequent find_meaning calls would see a
#     stale trie that doesn't reflect the mutation. Callers that
#     need to mutate should call ``_trie_cache.clear()`` to drop the
#     cached trie. Production / test workflows build word_db once
#     from meanings.json and never mutate.
# (b) ``id()`` is reused after garbage collection. If a word_db is
#     freed and a new dict happens to land at the same address, the
#     cache could in theory return a stale trie. In practice
#     word_dbs are loaded once at session start and held for the
#     process lifetime; the test suite clears _trie_cache between
#     tests via the autouse fixture in test_kenning_name_trie_backend.
#     A fully safe alternative would require wrapping word_db in a
#     weakref-able subclass — too invasive for the speedup it'd
#     enable. Documented as a known boundary; ``_trie_cache.clear()``
#     is the explicit reset.
# (c) Unbounded growth in long-running processes. Capped at
#     ``_TRIE_CACHE_MAX`` via OrderedDict-based LRU eviction. Lambda
#     and CLI workflows hit ≤2 distinct word_dbs in practice, so
#     the bound never kicks in but the leak is bounded if it ever
#     mattered.
_TRIE_CACHE_MAX = 4
_trie_cache: OrderedDict[int, MorphemeTrie] = OrderedDict()


def _trie_for(word_db: dict) -> MorphemeTrie:
    """Return a (possibly cached) MorphemeTrie for ``word_db``.

    Memoized on id(word_db) so repeated find_meaning calls on the
    same word_db share one trie. Cache is LRU-bounded at
    ``_TRIE_CACHE_MAX`` entries; eviction happens on insertion past
    the limit.
    """
    key = id(word_db)
    cached = _trie_cache.get(key)
    if cached is not None:
        # LRU: most-recently-used moves to the end.
        _trie_cache.move_to_end(key)
        return cached
    trie = build_morpheme_trie(word_db)
    # New-key assignment on an OrderedDict already inserts at the end;
    # explicit move_to_end isn't needed.
    _trie_cache[key] = trie
    while len(_trie_cache) > _TRIE_CACHE_MAX:
        _trie_cache.popitem(last=False)
    return trie


class Name:
    def __init__(self, name):
        self.name = str(name)
        self.words: dict[str, list[Word]] = {}
        for word in self.name.split(" "):
            self.words[word] = []
        self.chunks: list = []

    def __str__(self):
        return self.name

    def __repr__(self):
        return str({"name": self.name, "words": self.words})

    def __eq__(self, other):
        if not isinstance(other, Name):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def count_unaccounted(self):
        cnt = 0
        for word in self.words:
            for meaning in self.words[word]:
                cnt += meaning.count_unaccounted()
        return cnt

    def find_chunks(self):
        lower = self.name.lower()
        for words in self.word_db.values():
            if words[0].word_has_meaning(lower):
                self.chunks.extend(words)

    def find_meaning(self, word_db, reduce=True, use_trie=None):
        """Decompose every word in this place name against ``word_db``.

        Notes on argument precedence:

        * ``use_trie`` is the explicit kwarg override. If provided,
          it wins over the env var.
        * ``use_trie=None`` (default) reads the
          ``WYRD_KENNING_USE_TRIE_MATCHER`` env var; ``"1"`` enables
          the trie path, anything else (``"0"``, ``"true"``,
          ``"yes"``, empty, unset) keeps the legacy iterator.

        ``reduce=True`` (default) keeps only the lowest-unaccounted
        decompositions (legacy behavior, gated by ``Name.reduce``).
        ``reduce=False`` preserves every parse the matcher emits —
        the trie backend honors this by switching to
        ``all_decompositions`` instead of ``canonical_decompositions``,
        keeping parity with the legacy iterator's
        ``extract_meanings`` (which also yields every parse).

        Phase-2 default is the legacy iterator; Phase 3 (wyrd-zhhz)
        will flip the default to trie after the equivalence
        regression has held in production for one release cycle.
        """
        if use_trie is None:
            use_trie = os.environ.get(_USE_TRIE_ENV) == "1"
        self.word_db = word_db
        if use_trie:
            self._find_meaning_trie(reduce=reduce)
        else:
            self._find_meaning_legacy()
        # Always run reduce() when reduce=True, even on the trie path.
        # The canonical_decompositions filter only covers parses
        # produced by THIS call; if the caller invoked find_meaning
        # multiple times (e.g. reduce=False once, then reduce=True),
        # the prior call's non-optimal parses are still in
        # self.words[word] and need filtering. The marginal cost
        # of a second-pass reduce on already-filtered data is
        # negligible; correctness wins.
        if reduce:
            self.reduce()

    def _find_meaning_legacy(self) -> None:
        """Original Rando-port matcher: collect candidate Meanings via
        ``find_chunks``, then iterate every Meaning at every position
        in the input. O(M) per word per recursion level."""
        self.find_chunks()
        for word in self.words:
            w = Word(word)
            meanings = w.extract_meanings(self.chunks)
            seen: set[Word] = set(self.words[word])
            for meaning in meanings:
                if meaning not in seen:
                    self.words[word].append(meaning)
                    seen.add(meaning)
            if len(self.words[word]) == 0:
                self.words[word].append(w)

    def _find_meaning_trie(self, reduce: bool = True) -> None:
        """Trie-matcher backend (wyrd-k8e Phase 2). For each word in
        the place name, surface every parse the trie produces:

        * ``reduce=True``: ``canonical_decompositions`` returns only
          parses tied for 'best' (lowest unaccounted + fewest morphemes).
          That's the same set ``reduce()`` would keep on the legacy
          matcher's output. ``find_meaning`` still calls ``self.reduce()``
          afterward to handle the case where the caller invoked
          ``find_meaning`` multiple times — pre-existing entries in
          ``self.words[word]`` need filtering against the new
          additions.
        * ``reduce=False``: ``all_decompositions`` returns every parse,
          matching the legacy iterator's pre-reduce contract — useful
          to callers that want to inspect alternates the canonical
          score collapses.

        Multi-parse semantics preserved either way: a word with two
        senses for one surface (``-y`` = 'island' OR 'district') or a
        word the trie matches at multiple boundaries surfaces every
        reading as its own ``Word`` object.

        Dedup uses an O(1) set membership check rather than the
        legacy O(N) ``in list`` to keep ``reduce=False`` cases tractable
        when the trie emits many alternates."""
        trie = _trie_for(self.word_db)
        for word in self.words:
            decompositions = (
                canonical_decompositions(word, trie) if reduce else all_decompositions(word, trie)
            )
            seen: set[Word] = set(self.words[word])
            for d in decompositions:
                w = Word(d)
                if w not in seen:
                    self.words[word].append(w)
                    seen.add(w)
            if len(self.words[word]) == 0:
                self.words[word].append(Word(word))

    def reduce(self):
        self._filter_for_unaccounted()
        self._filter_for_complexity()

    def _filter_for_unaccounted(self):
        for word in self.words:
            meanings = self.words[word]
            max_unaccounted = 100
            for meaning in meanings:
                if meaning.count_unaccounted() < max_unaccounted:
                    max_unaccounted = meaning.count_unaccounted()
            new_meanings: list[Word] = []
            for meaning in meanings:
                if (
                    meaning.count_unaccounted() == max_unaccounted
                    and meaning not in new_meanings
                    and str(meaning) != ""
                ):
                    new_meanings.append(meaning)
            self.words[word] = new_meanings

    def _filter_for_complexity(self):
        for word in self.words:
            meanings = self.words[word]
            max_complexity = 100
            for meaning in meanings:
                if meaning.size() < max_complexity:
                    max_complexity = meaning.size()
            new_meanings: list[Word] = []
            for meaning in meanings:
                if (
                    meaning.size() == max_complexity
                    and meaning not in new_meanings
                    and len(meaning.word) > 0
                ):
                    new_meanings.append(meaning)
            self.words[word] = new_meanings

    def has_name(self):
        return any(m.has_name() for ms in self.words.values() for m in ms)

    def has_saint(self):
        return any(m.has_saint() for ms in self.words.values() for m in ms)

    def get_samples(self):
        usage_set: set = set()
        for word_list in self.words.values():
            for word in word_list:
                usage_set = usage_set.union(word.get_samples())
        return usage_set

    def get_lone_samples(self):
        usage_set: set = set()
        for word_list in self.words.values():
            for word in word_list:
                usage_set = usage_set.union(word.get_lone_samples())
        return usage_set

    def get_structure(self):
        structure = []
        for word in self.name.split(" "):
            group = {w.get_structure() for w in self.words[word]}
            structure.append(group)
        return set(itertools.product(*structure))


def load_names(data):
    names: list[str] = []
    for country in data:
        for region in data[country]:
            names.extend(data[country][region])
    names.sort()
    newnames: list[str] = []
    for name in names:
        if len(newnames) == 0 or newnames[-1] != name:
            newnames.append(str(name))
    return [Name(name) for name in newnames]
