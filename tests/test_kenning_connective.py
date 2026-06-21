"""Tests for the connective decomposition element (wyrd-aicu.9)."""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.runtime.connective import (
    DEFAULT_CONNECTIVE_INVENTORY,
    GENITIVE,
    Connective,
    ConnectiveKind,
    is_connective,
)
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.word import Word, _content_index


def _m(usage: str) -> Meaning:
    """Minimal Meaning for position tests — only ``usage`` matters here."""
    return Meaning(usage, tags=[], meanings=[], sources=[])


def test_connective_kind_is_a_strenum_equal_to_its_value():
    # wyrd-buye: the closed-set kind is a StrEnum, so the module-level aliases
    # ARE its members and a member == its str value — keeping every existing
    # `kind == GENITIVE` / `kind == "genitive"` comparison valid.
    assert GENITIVE is ConnectiveKind.GENITIVE
    assert ConnectiveKind.GENITIVE == "genitive"
    assert Connective("s", GENITIVE).kind == "genitive"
    assert isinstance(Connective("s", GENITIVE).kind, ConnectiveKind)


def test_connective_coerces_str_kind_and_rejects_typos():
    # __post_init__ enforces the closed set so the "can't silently skip the
    # tiebreak" guarantee is real: a valid bare string is coerced to the member,
    # an off-set typo raises at construction (not stored silently).
    coerced = Connective("s", "genitive")
    assert coerced.kind is ConnectiveKind.GENITIVE
    assert isinstance(coerced.kind, ConnectiveKind)
    with pytest.raises(ValueError):
        Connective("s", "genitiv")


def test_connective_renders_its_surface():
    # Reconstruction: a connective renders as its bare surface so a
    # decomposition concatenates back to the input word.
    assert str(Connective("s", GENITIVE)) == "s"


def test_connective_is_frozen_and_hashable():
    # Decompositions dedup by tuple(decomposition); connectives must hash.
    a = Connective("s", GENITIVE)
    b = Connective("s", GENITIVE)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_is_connective_discriminates_the_three_kinds():
    assert is_connective(Connective("s", GENITIVE)) is True
    assert is_connective("s") is False  # unaccounted str
    assert is_connective(object()) is False  # a Meaning duck-types here


def test_default_inventory_is_the_genitive_s():
    assert list(DEFAULT_CONNECTIVE_INVENTORY) == [Connective("s", GENITIVE)]


# --- Phase 2: Word position is derived over CONTENT elements (wyrd-aicu.9) ---


def test_content_index_is_a_no_op_without_a_connective():
    # The load-bearing bit-stability invariant: with no connective present every
    # slot's content index equals its raw index and the count equals len(word) —
    # byte-identical to the pre-connective ``enumerate`` + ``len(self.word)``.
    word = [_m("Bishop"), "x", _m("ton")]
    content_index_per_slot, count = _content_index(word)
    assert content_index_per_slot == [0, 1, 2]
    assert count == len(word) == 3
    # And the raw-index identity holds for every non-connective slot.
    assert all(content_index_per_slot[i] == i for i in range(len(word)))


def test_content_index_collapses_a_connective_out():
    # ``Bishop·s·ton`` with the ``s`` as a connective: the connective slot maps to
    # None (skipped) and the surrounding content keeps its true pre/post slot.
    word = [_m("Bishop"), Connective("s", GENITIVE), _m("ton")]
    content_index_per_slot, count = _content_index(word)
    assert content_index_per_slot == [0, None, 1]
    assert count == 2  # two CONTENT elements, the connective collapses out


def test_positioned_usages_unshifted_by_a_connective():
    # ``Bishop`` is pre and ``ton`` is post, exactly as if the connective weren't
    # there — the genitive glue must not shift the morphemes' positions.
    with_conn = Word([_m("Bishop"), Connective("s", GENITIVE), _m("ton")])
    without_conn = Word([_m("Bishop"), _m("ton")])
    assert with_conn._positioned_usages() == [("Bishop", "pre"), ("ton", "post")]
    assert with_conn._positioned_usages() == without_conn._positioned_usages()


def test_get_structure_unshifted_by_a_connective():
    with_conn = Word([_m("Bishop"), Connective("s", GENITIVE), _m("ton")])
    without_conn = Word([_m("Bishop"), _m("ton")])
    assert with_conn.get_structure() == (("pre",), ("post",))
    assert with_conn.get_structure() == without_conn.get_structure()


def test_size_excludes_connectives():
    # Complexity tiebreak must count content only (matcher Occam parity): the
    # connective adds nothing, so ``Bishop·s·ton`` has size 2, not 3.
    assert Word([_m("Bishop"), Connective("s", GENITIVE), _m("ton")]).size() == 2
    # No-op without a connective.
    assert Word([_m("Bishop"), _m("ton")]).size() == 2
    assert Word([_m("Bishop"), "x", _m("ton")]).size() == 3


def test_count_unaccounted_and_name_flags_ignore_connectives():
    # A connective is neither str (unaccounted) nor Meaning (name/saint) — verify
    # the already-correct discriminators don't miscount it.
    word = Word([_m("Bishop"), Connective("s", GENITIVE), _m("ton")])
    assert word.count_unaccounted() == 0
    assert word.has_name() is False
    assert word.has_saint() is False
