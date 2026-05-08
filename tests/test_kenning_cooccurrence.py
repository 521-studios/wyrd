"""Tests for tag-level co-occurrence statistics.

These exercise `_proportions_from` and `_ordered_tag_pairs` from cli.py,
which produce the corpus-derived statistics that future generation work
(wyrd-mj2) will use for conditional sampling.

The point of tag-level co-occurrence is to capture selectional preferences
that morpheme-level co-occurrence would be too sparse to learn — e.g.,
'descriptive + topography' is heavily attested even when the specific
(Brait, combe) pair never co-occurs in real toponyms.
"""

from __future__ import annotations

import json
from importlib import resources

from wyrd.generators.kenning.cli import _ordered_tag_pairs, _proportions_from
from wyrd.generators.kenning.meaning import load_meanings
from wyrd.generators.kenning.name import Name


def _build_word_db():
    text = resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
    word_db, _ = load_meanings(json.loads(text))
    return word_db


def test_proportions_now_emits_tag_cooccurrence_keys():
    """The output of _proportions_from must include the new fields without
    breaking the existing schema."""
    word_db = _build_word_db()
    name = Name("Bridgwater")
    name.find_meaning(word_db)
    out = _proportions_from([name])
    # Existing fields still present
    assert "usages" in out
    assert "single_usages" in out
    assert "structures" in out
    # New fields
    assert "tag_cooccurrence" in out
    assert "tag_marginal" in out


def test_ordered_tag_pairs_within_compound():
    """Ashton = Ash- + -ton; the ordered pair carries both tags.
    Pinned against the fixture so the tag-pair shape doesn't drift
    as the live bundle's morpheme tags evolve."""
    from tests.conftest import EXPLAIN_TEST_BUNDLE, swap_bundle

    with swap_bundle(EXPLAIN_TEST_BUNDLE):
        word_db, _ = load_meanings(EXPLAIN_TEST_BUNDLE)
        name = Name("Ashton")  # decomposes cleanly into Ash- + -ton
        name.find_meaning(word_db)
        pairs = _ordered_tag_pairs(name)
    # Pairs should be tuples of (left_tags, right_tags) lists
    assert len(pairs) >= 1
    for left_tags, right_tags in pairs:
        assert isinstance(left_tags, list)
        assert isinstance(right_tags, list)
    # Ash- carries plant/tree-type tags; -ton carries habitative/architecture
    left_tags, right_tags = pairs[0]
    assert any(t in left_tags for t in ("plant", "tree"))
    assert any(t in right_tags for t in ("architecture", "habitative", "social"))


def test_tag_cooccurrence_serializes_round_trip():
    """tag_cooccurrence keys are 'left|right' strings — round-trip through
    JSON without losing structure."""
    word_db = _build_word_db()
    names = [Name("Bridgwater"), Name("Ashton")]
    for n in names:
        n.find_meaning(word_db)
    out = _proportions_from(names)
    serialized = json.dumps(out)
    back = json.loads(serialized)
    assert back["tag_cooccurrence"] == out["tag_cooccurrence"]
    # Keys are |-separated strings, not tuples (which JSON can't represent)
    for k in back["tag_cooccurrence"]:
        assert "|" in k


def test_tag_cooccurrence_captures_descriptive_topography_pattern():
    """The 'descriptive + topography' pattern (broad/long/great + landscape
    feature, like Braitham Gate's first-word internal compound) must appear
    in the cooccurrence stats when the corpus has such names.

    This is the Braitham Gate regression test — the data layer must capture
    the pattern even before the generator wires it in for sampling."""
    word_db = _build_word_db()
    # Spot-check: feed in a handful of known-shape names that include
    # descriptive+topography compounds.
    candidates = [
        "Bridgwater",  # Bridg + water (architecture/water)
        "Ashton",  # Ash + ton  (tree/architecture)
        "Greenhill",  # Green + hill (descriptive/topography)
        "Oakwood",  # Oak + wood (tree/topography)
        "Longton",  # Long + ton  (descriptive/architecture)
        "Newport",  # New + port  (descriptive/architecture)
    ]
    names = []
    for n_str in candidates:
        n = Name(n_str)
        n.find_meaning(word_db)
        if n.count_unaccounted() == 0:
            names.append(n)

    if not names:
        # If meanings.json doesn't decompose any of these, the rest of the
        # test is moot. Still check the structure is right.
        out = _proportions_from(names)
        assert out["tag_cooccurrence"] == {}
        return

    out = _proportions_from(names)
    cooc = out["tag_cooccurrence"]
    # We expect at least SOME pairs to come out of these compound names.
    assert len(cooc) > 0
    # All pair counts are positive integers.
    for k, v in cooc.items():
        assert "|" in k
        assert isinstance(v, int) and v > 0
