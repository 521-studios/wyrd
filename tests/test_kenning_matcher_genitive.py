"""Matcher genitive-connective + prior-tiebreaker behavior (wyrd-aicu.9).

Proves: (1) bit-stable with the connective off; (2) the genitive prior splits
``Bishopston`` to the town (tūn) and keeps a genuine stone literal; (3) the
connective unblocks a non-homograph genitive (`Grimsworth`) as a coverage gain,
no prior needed.
"""

from __future__ import annotations

from wyrd.generators.kenning.runtime.connective import (
    DEFAULT_CONNECTIVE_INVENTORY,
    is_connective,
)
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.name import Name
from wyrd.generators.kenning.runtime.trie_matcher import (
    build_morpheme_trie,
    canonical_decomposition,
    count_unaccounted,
    iter_morphemes,
)


def _meaning(usage: str) -> Meaning:
    return Meaning(usage, [], [], {})


def _world_db():
    # ston (stone) vs ton (town) is the homograph; sworth is NOT a morpheme.
    return {
        "bishop": [_meaning("bishop")],
        "ston": [_meaning("ston")],  # stone
        "ton": [_meaning("ton")],  # town
        "grim": [_meaning("grim")],
        "worth": [_meaning("worth")],
        "rud": [_meaning("rud")],
    }


def _world():
    return build_morpheme_trie(_world_db())


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


# --- Phase 1: Name.find_meaning activation switch (wyrd-aicu.9) ---


def _word_surfaces(name: Name) -> list[str]:
    """The single best Word's element surfaces for a one-word Name."""
    words = name.words["bishopston"]
    assert len(words) == 1, words
    return [str(e) for e in words[0].word]


def test_find_meaning_default_both_none_is_bit_stable():
    # The activation switch defaults to OFF: both params None reproduce the
    # pre-connective matcher (the genitive s eaten into the longer 'ston').
    name = Name("bishopston")
    name.find_meaning(_world_db())
    assert _word_surfaces(name) == ["bishop", "ston"]


def test_find_meaning_with_inventory_and_prior_splits_to_town():
    # WITH inventory + prior: Bishop·s·tūn parses as bishop + Connective('s') +
    # ton, the connective collapses out of content attribution.
    name = Name("bishopston")
    name.find_meaning(
        _world_db(),
        connective_inventory=DEFAULT_CONNECTIVE_INVENTORY,
        genitive_prior={("ston", "ton"): 0.94},
    )
    word = name.words["bishopston"][0]
    assert [str(e) for e in word.word] == ["bishop", "s", "ton"]
    # The middle element is a genuine Connective (genitive glue), not a str.
    assert is_connective(word.word[1])
    # Content attribution is the town head — the connective is dropped.
    assert [m.usage for m in iter_morphemes(word.word)] == ["bishop", "ton"]
    # size() (complexity tiebreak) counts content only → 2, not 3.
    assert word.size() == 2
    assert word.count_unaccounted() == 0
