"""Tests for wyrd-zzli: structures must not produce 'By Green'-style
ungrammatical two-word output where a bare pre/post morpheme stands
alone as a word.

User-reported 2026-05-22: SPA generation produced 'By Green'
decomposing as '-by · green-'. Both morphemes are attachment-only
(leading or trailing dash) but the structure picked a 2-word template
where each word was a single bare pre/post morpheme, so the runtime
rendered them as standalone words separated by a space.

Fix: filter at rebuild time + defend at runtime. Real qualifier-word
patterns (Bishop's Stortford, Great Yarmouth) survive because their
lead morpheme carries the ``name`` flag.
"""

from __future__ import annotations

from collections import Counter

from wyrd.generators.kenning.cli.rebuild_proportions import _encode_structs
from wyrd.generators.kenning.runtime.proportions import (
    NameGenerator,
    _is_ungrammatical_word_template,
    is_structurally_grammatical,
)

# ---------------------------------------------------------------------------
# _is_ungrammatical_word_template
# ---------------------------------------------------------------------------


def test_bare_pre_single_word_is_ungrammatical():
    """A word_key like ((`pre`,),) — single element, bare pre, no flags
    — is the 'By Green' shape."""
    assert _is_ungrammatical_word_template((("pre",),)) is True


def test_bare_post_single_word_is_ungrammatical():
    assert _is_ungrammatical_word_template((("post",),)) is True


def test_pre_with_name_flag_is_grammatical():
    """Real qualifier-word patterns (Bishop's, Great, Old) carry the
    ``name`` flag and ARE allowed to stand alone as words."""
    assert _is_ungrammatical_word_template((("pre", "name"),)) is False


def test_pre_with_saint_flag_is_grammatical():
    assert _is_ungrammatical_word_template((("pre", "saint"),)) is False


def test_inner_single_word_is_grammatical():
    """'inner' morphemes aren't part of the leading/trailing-dash
    constraint — let those through. (They're rare as standalone
    anyway; not the bug shape.)"""
    assert _is_ungrammatical_word_template((("inner",),)) is False


def test_multi_element_word_is_grammatical():
    """A word with multiple morpheme slots (pre + post) is a real
    compound and is always grammatical."""
    assert _is_ungrammatical_word_template((("pre",), ("post",))) is False


def test_pre_with_single_flag_is_still_ungrammatical():
    """word_to_key adds a ``single`` flag to single-element words for
    bucket-keying purposes; that flag doesn't grant grammatical
    standalone-ness — only ``name`` / ``saint`` do."""
    assert _is_ungrammatical_word_template((("pre", "single"),)) is True


# ---------------------------------------------------------------------------
# is_structurally_grammatical
# ---------------------------------------------------------------------------


def test_single_word_structure_always_passes():
    """A 1-word structure renders as 1 surface name — no 'By Green'
    risk. Even bare pre/post single-word structures pass (they're
    weird but not the bug shape)."""
    assert is_structurally_grammatical((((("pre",),)),)) is True
    assert is_structurally_grammatical(((("pre",), ("post",)),)) is True


def test_post_then_pre_two_word_is_rejected():
    """The exact bug shape — `By Green`."""
    bad = (
        (("post",),),  # word 1: bare -by
        (("pre",),),  # word 2: bare green-
    )
    assert is_structurally_grammatical(bad) is False


def test_pre_then_post_two_word_is_also_rejected():
    """`Green By` is just as ungrammatical as `By Green` — both are
    attachment morphemes split across word boundaries."""
    bad = (
        (("pre",),),
        (("post",),),
    )
    assert is_structurally_grammatical(bad) is False


def test_qualifier_word_plus_compound_survives():
    """Real `Bishop's Stortford` shape: name-flagged pre + compound."""
    good = (
        (("pre", "name"),),  # 'Bishop's' — name-flagged qualifier
        (("pre",), ("post",)),  # 'Stortford' — pre+post compound
    )
    assert is_structurally_grammatical(good) is True


def test_compound_plus_compound_survives():
    """Two-word names where each word is itself a compound are fine —
    e.g. `Greenton Bridge` shape."""
    good = (
        (("pre",), ("post",)),
        (("pre",), ("post",)),
    )
    assert is_structurally_grammatical(good) is True


# ---------------------------------------------------------------------------
# NameGenerator runtime filter
# ---------------------------------------------------------------------------


def test_name_generator_drops_ungrammatical_structs_on_load():
    """Existing bundles built before the rebuild-side filter still
    work correctly because NameGenerator filters at __init__."""
    good_struct = (((("pre",), ("post",)),),)  # 1 word, pre+post
    bad_struct = ((("post",),), (("pre",),))  # the bug shape
    structs = {
        good_struct: 100,
        bad_struct: 475,  # mirroring the actual english_proportions weight
    }
    name_gen = NameGenerator(
        meaning_db={},
        meaning_gen=None,  # never called — we only check the filter
        structs=structs,
    )
    assert good_struct in name_gen.structs
    assert bad_struct not in name_gen.structs
    assert name_gen.structs[good_struct] == 100


def test_name_generator_passes_grammatical_structs_unchanged():
    """No-op when all structs are grammatical — bit-stable for clean
    bundles."""
    structs = {
        (((("pre",), ("post",)),),): 100,
        ((("pre", "name"),), (("pre",), ("post",))): 50,
    }
    name_gen = NameGenerator(meaning_db={}, meaning_gen=None, structs=structs)
    assert name_gen.structs == structs


# ---------------------------------------------------------------------------
# rebuild-side _encode_structs filter
# ---------------------------------------------------------------------------


def test_encode_structs_drops_ungrammatical():
    """Future rebuild-proportions runs don't emit the bad structures."""
    good_key = ((("pre",), ("post",)),)  # single-word compound
    bad_key = ((("post",),), (("pre",),))  # 'By Green' shape
    counter = Counter({good_key: 100, bad_key: 475})
    encoded = _encode_structs(counter)
    # The encoded structures should be a single entry — the good_key.
    encoded_word_shapes = [
        tuple(tuple(e["location"] for e in word) for word in s["words"]) for s in encoded
    ]
    assert (("pre", "post"),) in encoded_word_shapes
    assert (("post",), ("pre",)) not in encoded_word_shapes


def test_encode_structs_preserves_qualifier_word_shapes():
    """Real qualifier-word patterns survive _encode_structs because
    their lead morpheme carries the ``name`` flag."""
    counter = Counter(
        {
            ((("pre", "name"),), (("pre",), ("post",))): 30,  # Bishop's Stortford
            ((("post",),), (("pre",),)): 475,  # By Green — should be dropped
        }
    )
    encoded = _encode_structs(counter)
    assert len(encoded) == 1
    survivor = encoded[0]
    # The survivor is the qualifier-word shape; verify its lead word
    # carries the `name` flag.
    assert survivor["words"][0][0].get("name") is True
