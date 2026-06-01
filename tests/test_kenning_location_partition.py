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
2. vector-mode slot eligibility (``_matches_position``, exact equality)
   let a suffix key fill a single-word slot, producing the
   ``-park`` → "Park" standalone the user QA'd.

The fix gives bare keys a distinct ``location='bare'``. These tests pin
the partition invariants that follow — independent of any proportions
rebuild (the rebuild is what makes the fix take effect end-to-end; see
the PR's operator note).

wyrd-5z5j/D39 update: the structural-position model supersedes the
vector-mode exact-equality gate for the ``bare`` slot specifically — a
lone word is bare regardless of the matched form's dashes, so a bare
slot now accepts any location and the data-driven restriction moves to
the ``("bare", …, "single")`` bucket-frequency layer. pre/post/inner
slots keep exact equality. See ``test_bare_slot_accepts_any_location_d39``.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.proportions_builder import encode_meaning
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.proportions import (
    _is_ungrammatical_word_template,
    word_to_key,
)
from wyrd.generators.kenning.runtime.trie_matcher import _location_allows
from wyrd.generators.kenning.runtime.vector_name_select import (
    _matches_position,
    _slot_position_label,
)


def _m(usage: str) -> Meaning:
    return Meaning(usage, [], [], {})


# --- _set_location: bare is its own location -------------------------------


def test_bare_usage_gets_bare_location():
    assert _m("beck").location == "bare"


def test_dash_shapes_unchanged():
    assert _m("mine-").location == "pre"
    assert _m("-ford").location == "post"
    assert _m("-by-").location == "inner"


# --- _location_allows: bare is valid at ANY position -----------------------


def test_location_allows_bare_anywhere():
    """A bare key may match at the start, middle, or end of a word —
    unconstrained (the original permissive Rando-port semantics).
    Contrast post, which is end-anchored."""
    bare = _m("beck")
    # word_length 8, try a match at start (0-4), middle (2-6), end (4-8).
    assert _location_allows(bare, 0, 4, 8)
    assert _location_allows(bare, 2, 6, 8)
    assert _location_allows(bare, 4, 8, 8)


def test_location_allows_post_still_end_anchored():
    """Regression guard: the bare change must not loosen post. A
    suffix key still only matches at word-end."""
    post = _m("-beck")
    assert not _location_allows(post, 0, 4, 8)  # start — rejected
    assert _location_allows(post, 4, 8, 8)  # end — allowed


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


def test_bare_slot_accepts_any_location_d39():
    """wyrd-5z5j/D39: a 'bare' (whole-word) slot accepts a meaning of ANY
    location. A lone word is structurally bare regardless of the matched
    form's dashes, so the position gate is permissive for bare; the
    DATA-driven restriction (which morphemes actually appear standalone)
    moved to the ``("bare", …, "single")`` bucket-frequency layer
    (_resolve_slot_usage_frequency), where a never-observed-bare morpheme
    looks up to 0 and is filtered. This replaces the old wyrd-vpri
    exact-equality gate that rejected suffix keys here."""
    assert _matches_position(_m("beck"), "bare")
    assert _matches_position(_m("-beck"), "bare")


def test_post_slot_no_longer_matches_bare_meaning():
    """Symmetric: a genuine suffix slot does not pull in bare keys
    (pre-fix it did, because bare==post)."""
    assert _matches_position(_m("-beck"), "post")
    assert not _matches_position(_m("beck"), "post")


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
