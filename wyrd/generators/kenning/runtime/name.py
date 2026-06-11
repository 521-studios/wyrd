"""Name: a full place name (potentially multiple words), decomposable into Words.

Ported from Rando (rando/name.py) — Python 2 → 3.

``Name.find_meaning`` decomposes each whitespace-split word against a
``word_db`` via the trie matcher (``trie_matcher.canonical_decompositions``,
wyrd-k8e). O(match-length) per position via a character trie, then
DFS-enumerates the segmentation DAG. ~200x faster than the original
Rando-port iterator the trie superseded in wyrd-zhhz / Phase 3.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict

from .meaning import Joiner, Meaning
from .trie_matcher import (
    MorphemeTrie,
    all_decompositions,
    build_morpheme_trie,
    canonical_decompositions,
)
from .word import Word, word_position_for

# Bounded LRU memoization of the trie per word_db identity. Building
# the trie is O(M) where M is the morpheme count (~4ms on a 2.9K-
# morpheme bundle), which is cheap once but adds up across the per-
# name loop that rebuild-proportions / matcher consumers run (tens of
# thousands of names → minutes of avoidable trie-builds without the
# cache).
#
# Cache shape: ``OrderedDict[id(word_db), tuple[word_db, trie]]``.
# Storing the word_db reference alongside the trie keeps the dict
# alive while it's in the cache, which prevents Python from reusing
# its memory address for a new object. That eliminates the classic
# ``id()``-as-key staleness bug — as long as the cache entry exists,
# the id is bound to the same dict object.
#
# Two remaining assumptions:
#
# (a) **word_db is treated as immutable** for the duration of its
#     entry in the cache. Mutating a cached word_db (e.g.
#     ``word_db['new-key'] = [...]``) does NOT invalidate the trie;
#     subsequent find_meaning calls would see a stale trie that
#     doesn't reflect the mutation. Callers that mutate should call
#     ``_trie_cache.clear()`` to drop the cached trie. Production /
#     test workflows build word_db once from meanings.json and
#     never mutate.
# (b) Bounded growth in long-running processes. Capped at
#     ``_TRIE_CACHE_MAX`` via OrderedDict-based LRU eviction. Lambda
#     and CLI workflows hit ≤2 distinct word_dbs in practice, so
#     the bound never kicks in but memory is bounded if it ever
#     mattered.
_TRIE_CACHE_MAX = 4
_trie_cache: OrderedDict[int, tuple[dict, MorphemeTrie]] = OrderedDict()


def _trie_for(word_db: dict) -> MorphemeTrie:
    """Return a (possibly cached) MorphemeTrie for ``word_db``.

    Memoized on id(word_db); cache value is ``(word_db, trie)`` so
    the dict stays alive (and its id stable) while in the cache.
    LRU-bounded at ``_TRIE_CACHE_MAX``; insertion past the limit
    evicts the oldest entry.
    """
    key = id(word_db)
    cached = _trie_cache.get(key)
    if cached is not None:
        # LRU: most-recently-used moves to the end.
        _trie_cache.move_to_end(key)
        return cached[1]
    trie = build_morpheme_trie(word_db)
    # New-key assignment on an OrderedDict already inserts at the end;
    # explicit move_to_end isn't needed.
    _trie_cache[key] = (word_db, trie)
    while len(_trie_cache) > _TRIE_CACHE_MAX:
        _trie_cache.popitem(last=False)
    return trie


def _build_joiner_lookup(
    joiners: dict[str, list[tuple[str, int]]] | None,
) -> dict[str, str]:
    """Flatten a per-language-field joiner pool into a lowercase
    surface→lang_field lookup.

    The matcher's joiner consumption (wyrd-q0g6 Phase 1) only needs
    'is this str slot a registered joiner, and which family is it
    from?'. Cross-language collisions (e.g. ``'y'`` in welsh AND
    French) are resolved by 'first lang_field wins' — iteration order
    follows the joiner dict's key order, which is lang-sorted at
    bundle-emit time. Phase 2 may need finer disambiguation but Phase
    1 ships zero populated joiners so the conflict can't surface yet.
    """
    if not joiners:
        return {}
    lookup: dict[str, str] = {}
    for lang_field, entries in joiners.items():
        for form, _weight in entries:
            key = form.lower()
            if key and key not in lookup:
                lookup[key] = lang_field
    return lookup


def _consume_joiners(decomposition: list, joiner_lookup: dict[str, str]) -> list:
    """Replace each ``str`` slot whose lowercase surface is a
    registered joiner AND that sits between two Meaning slots with a
    ``Joiner`` sentinel.

    The both-neighbors-must-be-Meaning gate is load-bearing: joiners
    are phonological CONNECTORS, not standalone morphemes. A leading
    or trailing 'en' with no morpheme on the other side is genuinely
    unaccounted — consuming it would silently launder garbage into
    'matched-but-no-semantic'. Same goes for an 'en' between two
    unaccounted fragments. Only firing between two Meanings keeps
    the rule defensible against Phase 2 data alone.
    """
    if not joiner_lookup:
        return decomposition
    n = len(decomposition)
    new_decomp: list = []
    for i, slot in enumerate(decomposition):
        if isinstance(slot, str):
            lang = joiner_lookup.get(slot.lower())
            if lang is not None:
                left_is_meaning = i > 0 and isinstance(decomposition[i - 1], Meaning)
                right_is_meaning = i < n - 1 and isinstance(decomposition[i + 1], Meaning)
                if left_is_meaning and right_is_meaning:
                    new_decomp.append(Joiner(slot, lang_field=lang))
                    continue
        new_decomp.append(slot)
    return new_decomp


class Name:
    def __init__(self, name):
        self.name = str(name)
        self.words: dict[str, list[Word]] = {}
        for word in self.name.split(" "):
            self.words[word] = []

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

    def find_meaning(
        self, word_db, reduce=True, joiners=None, *, culture_languages=None, trie=None
    ):
        """Decompose every word in this place name against ``word_db``
        via the trie matcher.

        ``reduce=True`` (default) keeps only the lowest-unaccounted
        decompositions: ``canonical_decompositions`` returns only the
        parses tied for 'best' (lowest unaccounted + fewest morphemes).
        ``find_meaning`` then calls ``self.reduce()`` afterward to
        handle the case where the caller invoked ``find_meaning``
        multiple times — pre-existing entries in ``self.words[word]``
        need filtering against the new additions.

        ``joiners`` (wyrd-q0g6 Phase 1, optional) is a
        ``dict[lang_field, list[(form, weight)]]`` from
        ``meaning.load_joiners``. When provided, str slots between
        two Meaning slots whose lowercase surface matches a
        registered joiner are demoted to ``Joiner`` sentinels —
        non-str (don't count as unaccounted) and non-Meaning (don't
        pollute structure stats). Caveat: under ``reduce=True``, the
        canonical filter scores the RAW matcher output before joiner
        consumption; a parse that needs a joiner to reach 0
        unaccounted may not survive canonical filtering against a
        no-joiner parse with already-fewer unaccounted chars. Phase
        2.5 may fold joiners into the canonical score.

        ``reduce=False`` switches to ``all_decompositions``, returning
        every parse the trie produces — useful to callers (e.g.
        KenningExplain) that want to inspect alternates the canonical
        score would collapse.

        ``culture_languages`` (wyrd-pfoo, optional) is a frozenset of
        primary source-language bundle fields the caller wants the
        matcher to prefer when ambiguous parses tie on the canonical
        (unaccounted, morpheme_count) score. Forwarded to
        ``canonical_decompositions`` and consulted only on ties; no
        effect when ``reduce=False``. ``None`` (default) skips the
        tiebreaker entirely — the bit-stable path for KenningExplain,
        KenningRewind, and other consumers that don't carry a culture
        context. Per-culture callers (rebuild-proportions,
        runtime_db_export) pass their culture's expected-language set
        (see ``lexicon.proportions_builder.CULTURE_LANGUAGES``) so the
        decomposition recorded for that culture prefers culturally-
        appropriate senses (Welsh ``pen-`` → Celtic 'end/head' over
        OE 'penny').

        Multi-parse semantics: a word with two senses for one surface
        (``-y`` = 'island' OR 'district') or a word the trie matches
        at multiple boundaries surfaces every reading as its own
        ``Word`` object.

        Dedup keys on ``tuple(decomposition)`` — the raw list of
        Meaning|str elements — rather than constructing a Word per
        candidate just to ask the dedup set. Saves both the Word
        instantiation and the string-based Word.__hash__ on every
        candidate; matters for ``reduce=False`` cases where the
        trie emits many alternates.

        Hash invariant: two decompositions hash equal iff they're the
        same list of objects, where Meaning equality falls back to
        identity (Meaning.__hash__ is the default object id()-based
        hash). The trie cache (``_trie_for``) returns the same Meaning
        instances across calls for a given ``word_db``, so trie-emitted
        decompositions remain consistently dedupable within and across
        calls. A future Meaning.__hash__ override that doesn't preserve
        identity equality (e.g. content-hash on usage+language) would
        keep this dedup correct but the legacy iterator's
        ``tuple(word.word)`` shape (which bottomed out on str(Meaning),
        i.e. the lowercased dash-stripped usage) didn't carry the
        invariant — flagging here so a future refactor of Meaning's
        hash doesn't quietly drift this contract.
        """
        self.word_db = word_db
        # wyrd-04oi: callers on the decomposition path may inject a prebuilt
        # trie (e.g. the pickled matcher loaded beside the runtime DB) to skip
        # the per-cold-container rebuild; default rebuilds (+ in-memory caches).
        if trie is None:
            trie = _trie_for(word_db)
        joiner_forms = _build_joiner_lookup(joiners)
        for word in self.words:
            decompositions = (
                canonical_decompositions(word, trie, culture_languages=culture_languages)
                if reduce
                else all_decompositions(word, trie)
            )
            if joiner_forms:
                decompositions = [_consume_joiners(d, joiner_forms) for d in decompositions]
            seen_keys: set[tuple] = {tuple(existing.word) for existing in self.words[word]}
            for d in decompositions:
                key = tuple(d)
                if key not in seen_keys:
                    self.words[word].append(Word(d))
                    seen_keys.add(key)
            if len(self.words[word]) == 0:
                self.words[word].append(Word(word))
        if reduce:
            self.reduce()

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

    def get_bare_word_positions(self) -> set[tuple[str, str]]:
        """wyrd-rogd.13: ``(word_position, surface)`` pairs for every bare
        (single-morpheme) word of a MULTI-word name, where word_position is
        the word's slot in the name's word sequence (first → ``pre``,
        interior → ``inner``, last → ``post``).

        This is the WITHIN-NAME word-position evidence the bare-placement
        stats are built from — ``Saint`` is essentially always the first
        word of ``Saint Albans``, ``Parva`` the last of ``Wigston Parva``.
        Distinct from the WITHIN-WORD morpheme position (D39/D40 dashes).

        Records the morpheme's IDENTITY — the bare lowercased surface (D40)
        — not the matched variant's stored form. The bundle routinely
        stores case twins of one surface (``Ghyll`` / ``ghyll``) that both
        parse the same word; recording the stored forms would double-count
        every sighting AND split the count across two keys, diluting the
        naked↔structured threshold. Set semantics per name: one observation
        per (word_position, surface) regardless of how many parses or case
        twins matched.

        Single-word names return the empty set: there is no solo case in
        the word-sequence model. A word repeated at two positions in one
        name contributes both positions.
        """
        words = self.name.split(" ")
        out: set = set()
        if len(words) < 2:
            return out
        for index, word_text in enumerate(words):
            position = word_position_for(index, len(words))
            for parse in self.words.get(word_text, []):
                for usage in parse.get_lone_samples():
                    out.add((position, usage.lower().replace("-", "")))
        return out

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


def load_names_with_regions(data) -> list[tuple[Name, str | None]]:
    """Like ``load_names`` but tags each Name with its region.

    Used by consumers that pass region context into the canonical-pick
    lookup (wyrd-8m87): same-modern_name-different-region collisions
    in the lexicon DB get disambiguated by region rather than always
    resolving to the lowest-id toponym row.

    Names are deduplicated by surface (matching ``load_names`` so the
    matcher doesn't double-count a 'Stratford' that shows up in two
    counties); the FIRST (country, region) pair encountered in lex
    order wins for the kept entry. Output is sorted alphabetically by
    name. Country is consumed only as part of the deterministic
    tiebreak — only region travels onward to the canonical lookup
    (the lexicon's toponym schema doesn't index country yet).

    Single-region corpora behave identically to ``load_names`` once
    region info is dropped.
    """
    tagged: list[tuple[str, str, str]] = []
    for country in data:
        for region in data[country]:
            for name in data[country][region]:
                tagged.append((str(name), country, region))
    tagged.sort()
    out: list[tuple[Name, str | None]] = []
    seen: set[str] = set()
    for name, _country, region in tagged:
        if name in seen:
            continue
        seen.add(name)
        out.append((Name(name), region))
    return out
