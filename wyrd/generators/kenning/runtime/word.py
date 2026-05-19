"""Word: a sequence of Meanings (and unmatched string fragments) composing one word.

Ported from Rando (rando/word.py) — Python 2 → 3.
"""

from __future__ import annotations

from .meaning import Meaning


class Word:
    def __init__(self, word):
        if isinstance(word, str):
            self.word = [word]
        else:
            self.word = word

    def __str__(self):
        return "".join(str(w) for w in self.word)

    def __repr__(self):
        results = []
        for w in self.word:
            if isinstance(w, str):
                results.append(w)
            else:
                results.append(w.usage)
        return " ".join(results)

    def __eq__(self, other):
        if not isinstance(other, Word):
            return False
        if len(self.word) != len(other.word):
            return False
        return all(self.word[i] == other.word[i] for i in range(len(self.word)))

    def __hash__(self):
        return hash(tuple(str(w) for w in self.word))

    def size(self):
        return len(self.word)

    def count_unaccounted(self):
        size = 0
        for w in self.word:
            if isinstance(w, str):
                size += len(w)
        return size

    def has_name(self):
        return any(isinstance(m, Meaning) and m.is_name() for m in self.word)

    def has_saint(self):
        return any(isinstance(m, Meaning) and m.is_saint() for m in self.word)

    def get_samples(self):
        usage_set = set()
        if len(self.word) > 1:
            for m in self.word:
                if isinstance(m, Meaning):
                    usage_set.add(m.usage)
        return usage_set

    def get_lone_samples(self):
        usage_set = set()
        if len(self.word) == 1:
            for m in self.word:
                if isinstance(m, Meaning):
                    usage_set.add(m.usage)
        return usage_set

    def get_structure(self):
        structure = []
        for m in self.word:
            if not isinstance(m, Meaning):
                continue
            if m.is_name():
                structure.append((m.location, "name"))
            elif m.usage == "Saint-":
                structure.append((m.location, "saint"))
            else:
                structure.append((m.location,))
        return tuple(structure)
