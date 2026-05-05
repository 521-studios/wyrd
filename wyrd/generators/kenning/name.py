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

from wyrd.generators.kenning.trie_matcher import (
    MorphemeTrie,
    build_morpheme_trie,
    canonical_decompositions,
)
from wyrd.generators.kenning.word import Word

_USE_TRIE_ENV = "WYRD_KENNING_USE_TRIE_MATCHER"

# Memoize the trie per word_db identity. Building the trie is O(M)
# where M is the morpheme count — cheap, but the Name.find_meaning
# call site is per-name in a hot loop (rebuild-proportions iterates
# tens of thousands of names). Keying on id() is safe because
# word_dbs are immutable in normal use; tests and CLI sessions
# instantiate a fresh dict per run, which gets a fresh id and
# fresh trie. The cache never grows unbounded in practice.
_trie_cache: dict[int, MorphemeTrie] = {}


def _trie_for(word_db: dict) -> MorphemeTrie:
    """Return a (possibly cached) MorphemeTrie for ``word_db``.

    Memoized on id(word_db) so repeated find_meaning calls on the
    same word_db share one trie. Test/CLI workflows that build a
    fresh word_db per run get a fresh trie automatically; long-
    running processes (Lambda) build once and reuse forever.
    """
    key = id(word_db)
    cached = _trie_cache.get(key)
    if cached is not None:
        return cached
    trie = build_morpheme_trie(word_db)
    _trie_cache[key] = trie
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

        ``use_trie`` selects the matcher backend:
        * ``True`` — trie-indexed segmentation DAG (wyrd-k8e). Faster +
          equivalent under canonical scoring.
        * ``False`` — legacy ``Word.extract_meanings`` iterator.
        * ``None`` (default) — read the ``WYRD_KENNING_USE_TRIE_MATCHER``
          env var; ``"1"`` enables the trie, anything else falls back
          to the legacy iterator. Phase-2 default is OFF so existing
          callers see no behavior change without opt-in.
        """
        if use_trie is None:
            use_trie = os.environ.get(_USE_TRIE_ENV) == "1"
        self.word_db = word_db
        if use_trie:
            self._find_meaning_trie(word_db)
        else:
            self._find_meaning_legacy()
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
            for meaning in meanings:
                if meaning not in self.words[word]:
                    self.words[word].append(meaning)
            if len(self.words[word]) == 0:
                self.words[word].append(w)

    def _find_meaning_trie(self, word_db: dict) -> None:
        """Trie-matcher backend (wyrd-k8e Phase 2). For each word in
        the place name, ``canonical_decompositions`` returns every
        parse tied for 'best' (lowest unaccounted-chars + fewest
        morphemes). That's already the same set ``reduce()`` would
        keep on the legacy matcher's output, so the post-reduce shape
        is equivalent.

        Multi-parse semantics preserved: a word with two senses for
        one surface (``-y`` = 'island' OR 'district') or a word the
        trie matches at multiple boundaries surfaces every reading
        as its own ``Word`` object."""
        trie = _trie_for(word_db)
        for word in self.words:
            decompositions = canonical_decompositions(word, trie)
            for d in decompositions:
                w = Word(d)
                if w not in self.words[word]:
                    self.words[word].append(w)
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
