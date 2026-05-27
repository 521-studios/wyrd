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

import pytest

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


def test_multi_element_word_is_grammatical():
    """A word with multiple morpheme slots (pre + post) is a real
    compound and is always grammatical."""
    assert _is_ungrammatical_word_template((("pre",), ("post",))) is False


def test_pre_with_single_flag_is_still_ungrammatical():
    """word_to_key adds a ``single`` flag to single-element words for
    bucket-keying purposes; that flag doesn't grant grammatical
    standalone-ness — only ``name`` / ``saint`` do."""
    assert _is_ungrammatical_word_template((("pre", "single"),)) is True


def test_inner_single_word_is_now_ungrammatical():
    """Round-2 fix (code-reviewer P3): 'inner' morphemes have dashes
    on BOTH sides, so a bare-inner standalone word is the same kind
    of ungrammatical split as bare-pre / bare-post. Adding to the
    filter is future-proofing — no current bundle has the shape, but
    if mining surfaces one, the filter catches it."""
    assert _is_ungrammatical_word_template((("inner",),)) is True


def test_inner_with_name_flag_still_grammatical():
    """name-flagged inner survives like name-flagged pre/post."""
    assert _is_ungrammatical_word_template((("inner", "name"),)) is False


def test_empty_position_tuple_is_grammatical():
    """Empty inner tuple — defensive guard against malformed inputs.
    Returns False so the structure isn't dropped on bad shape."""
    assert _is_ungrammatical_word_template(((),)) is False


# ---------------------------------------------------------------------------
# is_structurally_grammatical
# ---------------------------------------------------------------------------


def test_single_word_compound_structure_passes():
    """A 1-word structure whose sole word is a real compound
    (pre+post, pre+inner+post, etc.) is grammatical — that's the
    normal `Stonebridge` shape."""
    assert is_structurally_grammatical(((("pre",), ("post",)),)) is True


def test_single_word_bare_pre_structure_is_ungrammatical():
    """wyrd-80ib: a 1-word structure whose sole word is a single bare
    prefix renders the morpheme alone after dash-stripping (`High-` →
    `High`, `South-` → `South`). Same `_is_ungrammatical_word_template`
    shape as the multi-word `By Green` bug — just expressed as a 1-word
    structure where the whole name is the affix."""
    assert is_structurally_grammatical(((("pre",),),)) is False


def test_single_word_bare_post_structure_is_ungrammatical():
    """wyrd-80ib: same as bare-pre but for suffix-only morphemes
    (`-bridge` → `Bridge`, `-green` → `Green`)."""
    assert is_structurally_grammatical(((("post",),),)) is False


def test_single_word_bare_pre_with_name_flag_passes():
    """wyrd-80ib: name-flagged single-word structures survive — the
    `name` flag is exactly what marks a qualifier word allowed to
    stand alone (Bishop's, Great, Old)."""
    assert is_structurally_grammatical(((("pre", "name"),),)) is True


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


def test_name_generator_raises_when_filter_empties_structs():
    """Round-2 fix (generator-contract-reviewer P2): if every input
    struct is filtered as ungrammatical, NameGenerator must fail loudly
    at init rather than letting the legacy select() path crash deep in
    weighted_choice on a None struct."""
    structs = {
        ((("post",),), (("pre",),)): 475,  # bad shape — would be dropped
    }
    with pytest.raises(ValueError, match="wyrd-zzli"):
        NameGenerator(meaning_db={}, meaning_gen=None, structs=structs)


def test_name_generator_empty_input_does_not_raise():
    """Empty input structs is a different situation from filter-emptied;
    don't raise on an already-empty dict (some test fixtures pass empty)."""
    name_gen = NameGenerator(meaning_db={}, meaning_gen=None, structs={})
    assert name_gen.structs == {}


def test_name_generator_warns_on_drop(caplog):
    """Round-2 fix (generator-contract-reviewer P2): operators need to
    know the runtime is filtering their bundle so they understand why
    seeded outputs drift from the pre-fix baseline. Warning names the
    dropped count + percentage so the operator can decide whether to
    re-emit the bundle. wyrd-van9 moved this from print(file=stderr)
    to the runtime logger so SPA / Lambda hosts can route it via
    standard logging config."""
    import logging

    good_struct = ((("pre",), ("post",)),)
    bad_struct = ((("post",),), (("pre",),))
    structs = {good_struct: 100, bad_struct: 475}
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="wyrd.generators.kenning.runtime.proportions"):
        NameGenerator(meaning_db={}, meaning_gen=None, structs=structs)
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert "wyrd-zzli" in log_text
    assert "filtered 1" in log_text
    # 475 / (100 + 475) = 82.6%
    assert "82.6%" in log_text


def test_name_generator_silent_when_nothing_dropped(caplog):
    """Clean bundle — no warning."""
    import logging

    structs = {((("pre",), ("post",)),): 100}
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="wyrd.generators.kenning.runtime.proportions"):
        NameGenerator(meaning_db={}, meaning_gen=None, structs=structs)
    assert not any("wyrd-zzli" in r.getMessage() for r in caplog.records)


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
