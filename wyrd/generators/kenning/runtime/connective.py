"""Connective decomposition elements (wyrd-aicu.9).

A **connective** is the third kind of decomposition element, beside content
``Meaning`` morphemes and unaccounted ``str`` fragments. It is *linguistic glue*
between morphemes — neither a content head nor noise:

* the **genitive** ``-s-`` — ``Bishop·s·tūn`` (Bishop's town),
* the **formative** ``-ing-`` — ``Padda·ing·tūn`` ("tūn of Padda's people"),
* euphonic **linking** vowels.

The matcher emits a connective at **0 unaccounted + 0 content-morpheme** cost;
it **reconstructs** the word (renders its surface) and is **excluded from
content-morpheme attribution** (proportions/generation count the head, never the
glue). The genitive ``-s-`` is what unblocks ``X·s·head`` decompositions the
matcher currently leaves partially parsed (the ``s`` falling into unaccounted) —
a broad coverage lift across ``-sworth`` / ``-sley`` / ``-sham`` / … , not a
``-ston`` special case. Where the long form is a genuine homograph
(``ston``=stone) the connective split merely *ties* the literal and the genitive
prior tiebreaks (wyrd-aicu.9, ``split_probability``).

This module is the data model only — the matcher's connective branch and the
soft genitive-credibility tiebreaker live in ``trie_matcher``; consumers
discriminate the three element kinds via :func:`is_connective`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ConnectiveKind(StrEnum):
    """The kind of linguistic glue a connective is — a CLOSED set, so a typo'd
    kind can't silently skip the genitive tiebreak. The matcher applies
    kind-specific selection: the genitive prior applies to ``GENITIVE``;
    ``FORMATIVE`` / ``LINKING`` are meaning-neutral. A ``StrEnum`` so each member
    ``== `` its string value (``ConnectiveKind.GENITIVE == "genitive"``), keeping
    any string-keyed comparison / serialization valid."""

    GENITIVE = "genitive"
    FORMATIVE = "formative"
    LINKING = "linking"


# Module-level aliases — the existing public names, now bound to the enum members.
# A member ``== `` its value, so callers doing ``kind == GENITIVE`` or
# ``kind == "genitive"`` keep working unchanged.
GENITIVE = ConnectiveKind.GENITIVE
FORMATIVE = ConnectiveKind.FORMATIVE
LINKING = ConnectiveKind.LINKING


@dataclass(frozen=True)
class Connective:
    """A connective element in a decomposition. ``surface`` is the glue's
    characters (``s``); ``kind`` is a :class:`ConnectiveKind`. Frozen + hashable
    so decompositions remain dedupable (the matcher dedups by
    ``tuple(decomposition)``), and renders as its bare surface so a decomposition
    reconstructs the input word."""

    surface: str
    kind: ConnectiveKind

    def __str__(self) -> str:
        return self.surface


def is_connective(elem: object) -> bool:
    """True iff ``elem`` is a connective — the discriminator every
    decomposition consumer uses to make the ``Meaning`` / ``Connective`` /
    ``str`` switch three-way. A connective is free (0 unaccounted), non-content
    (not a morpheme), and reconstructs (its surface is part of the word)."""
    return isinstance(elem, Connective)


# Data-driven connective inventory: an ordered tuple of :class:`Connective`.
# v1 ships the genitive ``-s-`` only; ``-ing-`` (FORMATIVE) and linking vowels
# extend this as DATA, with no matcher code change. Surfaces are bare
# lowercase; the matcher folds the word the same way before matching. Typed as
# ``Connective`` (not a bare ``(surface, kind)`` tuple) so a transposed entry is
# a construction error, not a silent bug.
ConnectiveInventory = tuple[Connective, ...]

DEFAULT_CONNECTIVE_INVENTORY: ConnectiveInventory = (Connective("s", GENITIVE),)
