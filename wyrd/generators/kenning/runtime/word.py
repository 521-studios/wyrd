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


def _position_form(meaning: Meaning, position: str) -> str:
    """wyrd-eyjk/D40: render a morpheme's bare surface at its DERIVED position
    as the recorded usage form. The morpheme's identity is its dash-stripped
    surface; the dashes encode position and are applied here from the
    span/index-derived ``position``, NOT read off whatever variant the matcher
    happened to fetch. So a word-final ``giles`` records as ``-giles`` even
    when only ``Giles-`` / ``Giles`` are stored. D39 casing: post/inner are
    lowercased; pre/bare keep the surface's stored case (name capitals)."""
    surface = meaning.usage.replace("-", "")
    if position == "pre":
        return f"{surface}-"
    if position == "post":
        return f"-{surface.lower()}"
    if position == "inner":
        return f"-{surface.lower()}-"
    return surface  # bare


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

    def _positioned_usages(self):
        """wyrd-eyjk/D40: each morpheme's usage recorded at its DERIVED
        position (index among the word's morphemes), not the matched variant's
        stored dash-shape. ``Stokegiles`` → ``Stoke-`` (pre) + ``-giles``
        (post). This is the (morpheme, derived-position) increment the
        proportions tally is built from."""
        # wyrd-eyjk round 2 (Gemini): derive position over ALL word elements
        # (count = len(self.word)), not just the matched Meanings, so a morpheme
        # in a partially-decomposed word (with unmatched string fragments) gets
        # its true structural position — a word-final ``giles`` after an
        # unmatched ``Stoke`` is post, not bare. Recorded corpus words are fully
        # decomposed (the exporter keeps only count_unaccounted()==0), so this
        # is a no-op on production data and robust for any partial caller. Kept
        # in lockstep with ``get_structure``.
        count = len(self.word)
        return [
            _position_form(m, _structural_position(i, count))
            for i, m in enumerate(self.word)
            if isinstance(m, Meaning)
        ]

    def get_samples(self):
        # Multi-morpheme words feed the ``usages`` (part) pool.
        if len(self.word) <= 1:
            return set()
        return set(self._positioned_usages())

    def get_lone_samples(self):
        # Single-morpheme words feed the ``single_usages`` (lone/bare) pool;
        # by definition the sole morpheme's derived position is ``bare``.
        if len(self.word) != 1:
            return set()
        return set(self._positioned_usages())

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
        #
        # wyrd-eyjk round 2: position is derived over ALL word elements
        # (count = len(self.word)), in lockstep with ``_positioned_usages`` —
        # so a matched morpheme keeps the same position in the structure tuple
        # as in the recorded usage form even when the word has unmatched string
        # fragments (a no-op for the fully-decomposed corpus words actually
        # recorded).
        count = len(self.word)
        structure = []
        for index, m in enumerate(self.word):
            if not isinstance(m, Meaning):
                continue
            position = _structural_position(index, count)
            if m.is_name():
                structure.append((position, "name"))
            elif m.usage == "Saint-":
                structure.append((position, "saint"))
            else:
                structure.append((position,))
        return tuple(structure)
