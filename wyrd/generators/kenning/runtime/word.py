"""Word: a sequence of Meanings (and unmatched string fragments) composing one word.

Ported from Rando (rando/word.py) — Python 2 → 3.
"""

from __future__ import annotations

from typing import Literal

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


# wyrd-rogd.13: bucket-key element prefix for the WITHIN-NAME word position of a
# bare word (which word slot of a multi-word name), distinct from the
# WITHIN-WORD morpheme position above. A bare slot's bucket key gains
# ``wp-pre`` / ``wp-inner`` / ``wp-post`` so its frequency pool is the
# per-word-position one.
WORD_POSITION_KEY_PREFIX = "wp-"


def word_position_for(word_index: int, word_count: int) -> Literal["pre", "inner", "post"] | None:
    """wyrd-rogd.13: map a word's index within its NAME to a word-position
    label — first word → ``pre``, last → ``post``, interior → ``inner``.

    Returns ``None`` for single-word names: there is no solo case in the
    word-sequence model (a 1-word name carries no evidence about where its
    word sits relative to others), mirroring the build-side tally which only
    walks multi-word names.
    """
    if word_count <= 1:
        return None
    if word_index == 0:
        return "pre"
    if word_index == word_count - 1:
        return "post"
    return "inner"


def _bare_surface(meaning: Meaning) -> str:
    """D45 (wyrd-aicu): a morpheme's STORED IDENTITY is its bare surface —
    dashes are NEVER part of it. The proportions tally (:meth:`Word.get_samples`)
    records ``(bare_surface, position)`` with position as an EXPLICIT axis, so
    nothing dash-encoded ever reaches the L4 tables. Stored case is kept (so an
    initial name's internal caps survive the render's front-cap, D39)."""
    return meaning.usage.replace("-", "")


def _position_form(meaning: Meaning, position: str) -> str:
    """The TRANSIENT render-key for a slot in ``NewName.name`` — NOT stored
    identity (that is the bare surface, D45). ``NewName.__str__`` strips these
    dashes and re-applies positional case from the SLOT (D39), so the dash
    decoration here is purely a within-generation convenience; the rendered
    output is identical whether this carries dashes or a bare surface. Kept
    dashed for continuity with the explainer / grid readers that fold on
    ``replace("-", "")``. D39 casing: post/inner lowercased; pre/bare keep
    stored case."""
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
        """D45 + wyrd-eyjk/D40: each morpheme recorded as a ``(bare_surface,
        derived_position)`` increment — the morpheme's identity (bare surface)
        plus the position it occupies among the word's morphemes, as an EXPLICIT
        axis. ``Stokegiles`` → ``(Stoke, pre)`` + ``(giles, post)``. Position is
        never folded into the surface string (the pre-D45 dash-form
        ``-giles``); the proportions tables carry it as its own column."""
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
            (_bare_surface(m), _structural_position(i, count))
            for i, m in enumerate(self.word)
            if isinstance(m, Meaning)
        ]

    def get_samples(self):
        # Multi-morpheme words feed the ``usages`` (part) pool, as
        # ``(surface, position)`` increments (D45 — position is an explicit
        # axis, not a dash-encoded form).
        if len(self.word) <= 1:
            return set()
        return set(self._positioned_usages())

    def get_lone_samples(self):
        # Single-morpheme words feed the ``single_usages`` (lone/bare) pool,
        # keyed by bare surface (the sole morpheme's derived position is always
        # ``bare``, so the pool needs no position axis).
        if len(self.word) != 1:
            return set()
        return {surface for surface, _position in self._positioned_usages()}

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
            elif m.usage.replace("-", "").lower() == "saint":
                # The "saint" qualifier is the DEDICATION PARTICLE (the word
                # "Saint" in "Saint Albans"), identified by SURFACE — not the
                # saint TAG (which also marks the dedicated name, Mary/Giles).
                # D45: fold the surface (the pre-D45 ``m.usage == "Saint-"``
                # literal breaks once usages are bare); matches Meaning.key().
                structure.append((position, "saint"))
            else:
                structure.append((position,))
        return tuple(structure)
