"""Word: a sequence of Meanings (and unmatched string fragments) composing one word.

Ported from Rando (rando/word.py) — Python 2 → 3.
"""

from __future__ import annotations

from typing import Literal

from wyrd.generators.kenning.lexicon.morpheme_surface import _strip_all_dashes

from .connective import is_connective
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
    initial name's internal caps survive the render's front-cap, D39).

    Surrounding whitespace is stripped too (wyrd-an8u): it's never part of a
    morpheme's identity, and a dirty ``'Oak- '`` would otherwise tally a
    distinct ``'Oak '`` surface that splits proportion weight from the clean
    ``'Oak'``.

    wyrd-p0e4: de-dash via :func:`_strip_all_dashes` — the Unicode-aware drop-in
    for the L4 fold family's ``replace("-", "")`` (SAME all-positions semantics,
    extended to all 11 boundary-dash codepoints: U+2010, en/em dash, …). A
    non-ASCII boundary marker (``'Oak–'``) can no longer fork from ``'Oak'`` in
    the proportions pool. Interior dashes are folded too (keeping this key family
    mutually consistent — the aicu.4 sweep, not this hardening, is where interior
    hyphens would become identity-bearing). Inert on today's fully-de-dashed
    bundle; a guard against a future dash slipping through."""
    return _strip_all_dashes(meaning.usage).strip()


def _position_form(meaning: Meaning, position: str) -> str:
    """The TRANSIENT render-key for a slot in ``NewName.name`` — NOT stored
    identity (that is the bare surface, D45). ``NewName.__str__`` strips these
    dashes and re-applies positional case from the SLOT (D39), so the dash
    decoration here is purely a within-generation convenience; the rendered
    output is identical whether this carries dashes or a bare surface. Kept
    dashed for continuity with the explainer / grid readers that fold on
    ``replace("-", "")``. D39 casing: post/inner lowercased; pre/bare keep
    stored case."""
    surface = _strip_all_dashes(meaning.usage)
    if position == "pre":
        return f"{surface}-"
    if position == "post":
        return f"-{surface.lower()}"
    if position == "inner":
        return f"-{surface.lower()}-"
    return surface  # bare


def _is_content_element(elem: object) -> bool:
    """wyrd-aicu.9: a CONTENT element is everything position is derived over —
    matched ``Meaning`` morphemes AND unaccounted ``str`` fragments — but NOT a
    ``Connective`` (the genitive ``-s-`` glue). A connective is free + non-content
    + position-neutral: it reconstructs the surface but must not shift the
    pre/inner/post slot of the morphemes around it (``Bishop·s·tūn`` → ``Bishop``
    is pre and ``tūn`` is post, exactly as if the connective weren't there).

    Mirrors ``trie_matcher._is_content_morpheme`` semantics on the connective
    exclusion, but KEEPS unaccounted ``str`` fragments in the count: position is
    derived over all word elements that occupy a structural slot, and an
    unaccounted fragment between two morphemes still separates them (the
    pre-existing ``count = len(self.word)`` behaviour for partial parses), whereas
    a connective is glue that collapses out of the position math entirely."""
    return not is_connective(elem)


def _content_index(word: list) -> tuple[list[int | None], int]:
    """wyrd-aicu.9: map each raw word slot to its CONTENT index (its position
    among non-connective elements), plus the total content count.

    Returns ``(content_index_per_slot, content_count)`` where
    ``content_index_per_slot[i]`` is the 0-based index of slot ``i`` within the
    content sequence, or ``None`` when slot ``i`` is a connective (skipped).
    ``_positioned_usages`` and ``get_structure`` both derive position from this
    shared mapping so they stay in LOCKSTEP — a connective collapses out, and the
    surrounding content keeps its true pre/inner/post slot.

    No-op invariant: when no connective is present, ``content_index_per_slot[i]
    == i`` for every slot and ``content_count == len(word)`` — byte-identical to
    the pre-connective ``enumerate(self.word)`` + ``len(self.word)`` math."""
    content_index_per_slot: list[int | None] = []
    next_content = 0
    for elem in word:
        if _is_content_element(elem):
            content_index_per_slot.append(next_content)
            next_content += 1
        else:
            content_index_per_slot.append(None)
    return content_index_per_slot, next_content


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
            elif is_connective(w):
                results.append(w.surface)  # Connective has .surface, not .usage
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
        # wyrd-aicu.9: the complexity tiebreak (Name._filter_for_complexity)
        # must count the same things the matcher's Occam score does — content
        # elements only. A connective (genitive ``-s-`` glue) is free + non-
        # content, so it never adds to complexity; excluding it here keeps the
        # legacy-matcher complexity rank in agreement with the trie matcher's
        # ``_decomposition_score`` morpheme count (which uses
        # ``_is_content_morpheme``). No-op when no connective is present
        # (every element is content → ``len(self.word)``).
        return sum(1 for w in self.word if _is_content_element(w))

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
        # wyrd-eyjk round 2 (Gemini): derive position over ALL word elements,
        # not just the matched Meanings, so a morpheme in a partially-decomposed
        # word (with unmatched string fragments) gets its true structural
        # position — a word-final ``giles`` after an unmatched ``Stoke`` is post,
        # not bare. Recorded corpus words are fully decomposed (the exporter
        # keeps only count_unaccounted()==0), so this is a no-op on production
        # data and robust for any partial caller. Kept in lockstep with
        # ``get_structure``.
        #
        # wyrd-aicu.9: position is derived over CONTENT elements (the shared
        # ``_content_index`` helper drops connectives), so the genitive ``-s-``
        # glue doesn't shift the pre/inner/post slot of the morphemes around it.
        # No-op when no connective is present (content index == raw index).
        content_index_per_slot, count = _content_index(self.word)
        return [
            (_bare_surface(m), _structural_position(content_index_per_slot[i], count))
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
        # wyrd-eyjk round 2: position is derived over ALL word elements, in
        # lockstep with ``_positioned_usages`` — so a matched morpheme keeps the
        # same position in the structure tuple as in the recorded usage form even
        # when the word has unmatched string fragments (a no-op for the fully-
        # decomposed corpus words actually recorded).
        #
        # wyrd-aicu.9: position is derived over CONTENT elements via the shared
        # ``_content_index`` helper (connectives collapse out), so the genitive
        # ``-s-`` glue is position-neutral and ``get_structure`` /
        # ``_positioned_usages`` stay in lockstep. No-op when no connective is
        # present (content index == raw index).
        content_index_per_slot, count = _content_index(self.word)
        structure = []
        for index, m in enumerate(self.word):
            if not isinstance(m, Meaning):
                continue
            position = _structural_position(content_index_per_slot[index], count)
            if m.is_name():
                structure.append((position, "name"))
            elif _strip_all_dashes(m.usage).lower() == "saint":
                # The "saint" qualifier is the DEDICATION PARTICLE (the word
                # "Saint" in "Saint Albans"), identified by SURFACE — not the
                # saint TAG (which also marks the dedicated name, Mary/Giles).
                # D45: fold the surface (the pre-D45 ``m.usage == "Saint-"``
                # literal breaks once usages are bare); matches Meaning.key().
                structure.append((position, "saint"))
            else:
                structure.append((position,))
        return tuple(structure)
