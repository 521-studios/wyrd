"""Matcher genitive-connective + prior-tiebreaker behavior (wyrd-aicu.9).

Proves: (1) bit-stable with the connective off; (2) the genitive prior splits
``Bishopston`` to the town (tūn) and keeps a genuine stone literal; (3) the
connective unblocks a non-homograph genitive (`Grimsworth`) as a coverage gain,
no prior needed.
"""

from __future__ import annotations

from wyrd.generators.kenning.runtime.connective import DEFAULT_CONNECTIVE_INVENTORY
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.trie_matcher import (
    build_morpheme_trie,
    canonical_decomposition,
    count_unaccounted,
    iter_morphemes,
)


def _meaning(usage: str) -> Meaning:
    return Meaning(usage, [], [], {})


def _world():
    # ston (stone) vs ton (town) is the homograph; sworth is NOT a morpheme.
    db = {
        "bishop": [_meaning("bishop")],
        "ston": [_meaning("ston")],  # stone
        "ton": [_meaning("ton")],  # town
        "grim": [_meaning("grim")],
        "worth": [_meaning("worth")],
        "rud": [_meaning("rud")],
    }
    return build_morpheme_trie(db)


def _surfaces(decomp):
    return [str(e) for e in decomp]


def test_connective_off_is_bit_stable_literal():
    # No inventory => today's matcher: the genitive s is eaten into the longer
    # 'ston' (stone). The exact behavior this PR must preserve when off.
    d = canonical_decomposition("bishopston", _world())
    assert _surfaces(d) == ["bishop", "ston"]


def test_prior_split_attributes_to_town():
    # split P=0.94 -> the genitive split wins: Bishop + ·s· + ton.
    d = canonical_decomposition(
        "bishopston",
        _world(),
        connective_inventory=DEFAULT_CONNECTIVE_INVENTORY,
        genitive_prior={("ston", "ton"): 0.94},
    )
    assert _surfaces(d) == ["bishop", "s", "ton"]
    # the connective is dropped from content morphemes -> attribution is town.
    assert [m.usage for m in iter_morphemes(d)] == ["bishop", "ton"]


def test_prior_literal_keeps_stone():
    # split P=0.05 (genuine-stone-heavy suffix) -> the literal long form wins.
    d = canonical_decomposition(
        "rudston",
        _world(),
        connective_inventory=DEFAULT_CONNECTIVE_INVENTORY,
        genitive_prior={("ston", "ton"): 0.05},
    )
    assert _surfaces(d) == ["rud", "ston"]
    assert [m.usage for m in iter_morphemes(d)] == ["rud", "ston"]


def test_connective_unblocks_non_homograph_coverage():
    # 'sworth' is not a morpheme, so today 's' is unaccounted: (1 un, 2 morph).
    # The connective makes Grim + ·s· + worth a (0 un, 2 morph) clean parse and
    # it wins on score alone -- no prior needed. The coverage payoff.
    d = canonical_decomposition(
        "grimsworth",
        _world(),
        connective_inventory=DEFAULT_CONNECTIVE_INVENTORY,
    )
    assert _surfaces(d) == ["grim", "s", "worth"]
    assert count_unaccounted(d) == 0
    assert [m.usage for m in iter_morphemes(d)] == ["grim", "worth"]
