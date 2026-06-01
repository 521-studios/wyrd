"""Word: a sequence of Meanings (and unmatched string fragments) composing one word.

Ported from Rando (rando/word.py) — Python 2 → 3.
"""

from __future__ import annotations

from .meaning import Meaning


def _structural_position(index: int, count: int) -> str:
    """Map a morpheme's index within its word to a position slot (D39).

    Sole morpheme → ``bare``; otherwise first → ``pre``, last → ``post``,
    interior → ``inner``. Matches ``Meaning._set_location``'s vocabulary so
    the tally, the grammaticality filter, and the render all speak the same
    four slots.
    """
    if count <= 1:
        return "bare"
    if index == 0:
        return "pre"
    if index == count - 1:
        return "post"
    return "inner"


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
        # wyrd-5z5j/D39: a morpheme's structural position comes from WHERE it
        # sits among the word's morphemes, NOT from the dash-shape of the form
        # the matcher happened to match (``m.location``). The sole morpheme of
        # a word is ``bare`` even when matched via a dashed form (``pleasant``
        # filled by ``-pleasant``); only genuine compounds split into
        # pre/inner/post. Deriving from the dash-shape was the indirection that
        # mis-tagged standalone words as pre/post, tripped the grammaticality
        # filter, and dropped two-word toponyms (``Mount Pleasant``).
        #
        # The ``name`` / ``saint`` flags stay orthogonal — they ride alongside
        # whatever structural position the morpheme occupies.
        meanings = [m for m in self.word if isinstance(m, Meaning)]
        count = len(meanings)
        structure = []
        for index, m in enumerate(meanings):
            position = _structural_position(index, count)
            if m.is_name():
                structure.append((position, "name"))
            elif m.usage == "Saint-":
                structure.append((position, "saint"))
            else:
                structure.append((position,))
        return tuple(structure)
