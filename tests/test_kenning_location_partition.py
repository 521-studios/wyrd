"""wyrd-vpri: position-partitioning — bare keys are their own location.

Pre-fix, ``Meaning._set_location`` classified no-dash (bare) usages as
``post`` via its else-branch. That conflated two distinct things:

* a genuine bare single-word morpheme (``beck`` 'stream') — valid as a
  standalone word, and
* a suffix-only morpheme (``-beck``) — valid only attached to the END
  of a compound.

Both ended up ``location='post'``, so:

1. the proportions grammaticality guard
   (``_is_ungrammatical_word_template``) couldn't tell a legitimate
   single bare word from a suffix-only morpheme rendered alone — it
   filtered (or admitted) both identically; and
2. vector-mode slot eligibility (the old ``_matches_position`` exact-equality
   gate) let a suffix key fill a single-word slot, producing the
   ``-park`` → "Park" standalone the user QA'd.

The fix gives bare keys a distinct ``location='bare'``. These tests pin
the partition invariants that follow — independent of any proportions
rebuild (the rebuild is what makes the fix take effect end-to-end; see
the PR's operator note).

wyrd-eyjk/D40 update: position is no longer a match-time gate at all — both
the trie ``_location_allows`` and the vector ``_matches_position`` gates were
removed. Matching is string-only; bare/pre/-inner-/-post is DERIVED from the
span, and the data-driven restriction is the per-(position) bucket frequency.
``Meaning.location`` survives only as a render/scoring hint. The string-only
matching invariant is pinned by ``test_matching_is_string_only_no_position_gate``
in ``test_kenning_trie_matcher.py``; the slot-position LABEL (still used for
D36 scoring) by ``test_slot_position_label_bare`` below.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.proportions_builder import encode_meaning
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.proportions import (
    _is_ungrammatical_word_template,
    word_to_key,
)
from wyrd.generators.kenning.runtime.vector_name_select import _slot_position_label


def _m(usage: str) -> Meaning:
    return Meaning(usage, [], [], {})


# --- _set_location: bare is its own location -------------------------------


def test_bare_usage_gets_bare_location():
    assert _m("beck").location == "bare"


def test_dash_shapes_unchanged():
    assert _m("mine-").location == "pre"
    assert _m("-ford").location == "post"
    assert _m("-by-").location == "inner"


# wyrd-eyjk/D40: the `_location_allows` match-time position gate was REMOVED —
# position is no longer a constraint on matching (a morpheme is its string and
# may match anywhere; bare/pre/post/inner is derived from the span afterward and
# statistically ranked at build time, never used to reject a match). Its tests
# are gone with it. `Meaning.location` survives as a render hint (above).


# --- grammaticality guard: bare single grammatical, suffix-alone not -------


def test_bare_single_word_is_grammatical():
    """A single-word structure whose sole morpheme is bare is
    grammatical — the guard keeps it. ``word_to_key`` tags single
    words with the 'single' flag; the key is ('bare', 'single')."""
    key = word_to_key([{"location": "bare"}])
    assert key == (("bare", "single"),)
    assert not _is_ungrammatical_word_template(key)


def test_suffix_only_standalone_is_ungrammatical():
    """A single-word structure whose sole morpheme is a suffix (post)
    is ungrammatical — the guard rejects it. This is the ``-park`` →
    "Park" standalone the fix targets; pre-fix it was indistinguishable
    from a bare single because both were 'post'."""
    key = word_to_key([{"location": "post"}])
    assert key == (("post", "single"),)
    assert _is_ungrammatical_word_template(key)


def test_prefix_only_standalone_still_ungrammatical():
    key = word_to_key([{"location": "pre"}])
    assert _is_ungrammatical_word_template(key)


# --- vector-mode slot eligibility: bare slot ⇄ bare meaning only ------------


def test_slot_position_label_bare():
    assert _slot_position_label("Beck") == "bare"
    assert _slot_position_label("Place-") == "pre"
    assert _slot_position_label("-shire") == "post"
    assert _slot_position_label("-inner-") == "inner"


# wyrd-eyjk/D40: the vector-mode `_matches_position` position gate was REMOVED
# (its tests with it). A morpheme may fill any slot; the per-position bucket
# frequency is the data-driven restriction. `_slot_position_label` survives —
# it still labels the slot's position for the D36 position-axis SCORING.


# --- encode_meaning: bare serializes as a location, not a flag -------------


def test_encode_meaning_bare_is_a_location():
    """``bare`` must serialize to {'location': 'bare'}, NOT the flag
    branch {'bare': True} — otherwise word_to_key's element['location']
    lookup KeyErrors on a bare element."""
    assert encode_meaning(["bare"]) == {"location": "bare"}
    assert encode_meaning(["bare", "name"]) == {"location": "bare", "name": True}


def test_encode_meaning_other_locations_unchanged():
    assert encode_meaning(["pre"]) == {"location": "pre"}
    assert encode_meaning(["post"]) == {"location": "post"}
    assert encode_meaning(["inner"]) == {"location": "inner"}
