"""Tests for the connective decomposition element (wyrd-aicu.9)."""

from __future__ import annotations

from wyrd.generators.kenning.runtime.connective import (
    DEFAULT_CONNECTIVE_INVENTORY,
    GENITIVE,
    Connective,
    ConnectiveKind,
    is_connective,
)


def test_connective_kind_is_a_strenum_equal_to_its_value():
    # wyrd-buye: the closed-set kind is a StrEnum, so the module-level aliases
    # ARE its members and a member == its str value — keeping every existing
    # `kind == GENITIVE` / `kind == "genitive"` comparison valid.
    assert GENITIVE is ConnectiveKind.GENITIVE
    assert ConnectiveKind.GENITIVE == "genitive"
    assert Connective("s", GENITIVE).kind == "genitive"
    assert isinstance(Connective("s", GENITIVE).kind, ConnectiveKind)


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
