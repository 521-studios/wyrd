"""Name: a full place name (potentially multiple words), decomposable into Words.

Ported from Rando (rando/name.py) — Python 2 → 3.
"""

from __future__ import annotations

import itertools

from wyrd.generators.kenning.word import Word


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

    def find_meaning(self, word_db):
        self.word_db = word_db
        self.find_chunks()
        for word in self.words:
            w = Word(word)
            meanings = w.extract_meanings(self.chunks)
            for meaning in meanings:
                if meaning not in self.words[word]:
                    self.words[word].append(meaning)
            if len(self.words[word]) == 0:
                self.words[word].append(w)
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
