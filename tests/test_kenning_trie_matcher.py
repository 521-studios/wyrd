"""Unit tests for the trie-indexed segmentation DAG morpheme matcher
(wyrd-k8e Phase 1)."""

from __future__ import annotations

import warnings

import pytest

from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.trie_matcher import (
    DecompositionTruncatedWarning,
    MorphemeTrie,
    all_decompositions,
    build_morpheme_trie,
    canonical_decomposition,
    canonical_decompositions,
    count_unaccounted,
    iter_morphemes,
)


def _meaning(usage: str) -> Meaning:
    """Convenience for building a Meaning at the right Meaning.location
    via the dash-marker convention. 'Bridg-' → pre; '-water' → post;
    '-by-' → inner."""
    return Meaning(usage, [], [], {})


# --- build_morpheme_trie ---------------------------------------------------


def test_build_trie_inserts_each_meaning_once():
    """Trie's morpheme_count tracks total Meanings indexed (sense-
    aware: two meanings sharing a usage count twice). Pin via the
    bundled meaning_db fixture shape."""
    db = {
        "Bridg-": [_meaning("Bridg-")],
        "-water": [_meaning("-water")],
        "-y": [_meaning("-y"), _meaning("-y")],  # two senses
    }
    trie = build_morpheme_trie(db)
    assert trie.morpheme_count == 4


def test_build_trie_skips_empty_surface_form():
    """A usage that's only dashes has empty dash-stripped surface and
    must be skipped — otherwise it'd cycle the segmentation DAG."""
    db = {"---": [_meaning("---")], "ham": [_meaning("ham")]}
    trie = build_morpheme_trie(db)
    # Only 'ham' is in the trie.
    assert trie.morpheme_count == 1
    decomp = canonical_decomposition("ham", trie)
    assert len(decomp) == 1
    assert decomp[0].usage == "ham"


def _egyptian_meaning(usage: str) -> Meaning:
    """Builds a Meaning whose only source is the egyptian language
    field — pins the trie's phonogram-transliteration filter."""
    return Meaning(usage, [], [], {"egyptian": ["X"]})


def _english_meaning(usage: str) -> Meaning:
    """Builds a Meaning with an English-family source so the filter
    keeps it even at single-char surface (the ``-y`` = 'island' case)."""
    return Meaning(usage, [], [], {"modern_english": ["y"]})


def test_build_trie_skips_single_char_phonogram_only_morphemes():
    """wyrd-m2ym: single-char ASCII surfaces whose meanings carry
    EXCLUSIVELY phonogram-transliteration sources (egyptian / akkadian
    / sumerian) are skipped at trie-build. The Meaning records remain
    in meaning_db (other consumers — KenningRewind, future explainer-
    tag inventory — can still see them); only the trie matcher
    filters. Egyptian ``s`` and Akkadian ``u`` are the canonical
    collision case."""
    db = {
        "s": [_egyptian_meaning("s")],
        "u": [Meaning("u", [], [], {"akkadian": ["X"]})],
        "ham": [_meaning("ham")],  # legacy empty-sources fixture — KEEP
    }
    trie = build_morpheme_trie(db)
    # 's' and 'u' filtered; 'ham' kept.
    assert trie.morpheme_count == 1
    # The filtered surfaces don't decompose against themselves anymore:
    # 's' alone returns one decomposition of the raw string (unaccounted).
    decomp = canonical_decomposition("s", trie)
    assert decomp == ["s"]
    # 'ham' still decomposes via the trie.
    decomp = canonical_decomposition("ham", trie)
    assert len(decomp) == 1
    assert getattr(decomp[0], "usage", None) == "ham"


def test_build_trie_keeps_single_char_english_morphemes():
    """wyrd-m2ym (filter symmetry): single-char English suffix
    morphemes like ``-y`` ('island' / 'district') survive the filter
    because their Meaning carries an English-family source — the rule
    fires only when sources are EXCLUSIVELY phonogram-transliteration
    languages."""
    db = {
        "-y": [_english_meaning("-y")],
        "ham": [_meaning("ham")],
    }
    trie = build_morpheme_trie(db)
    assert trie.morpheme_count == 2


def test_build_trie_keeps_multichar_phonogram_only_morphemes():
    """wyrd-m2ym (filter scope): the filter fires ONLY on single-char
    surfaces — multi-char phonogram-only morphemes (e.g. Egyptian
    ``ankh``) pass through unchanged because they don't trigger the
    collision-with-random-positions problem the filter exists to fix."""
    db = {
        "ankh": [Meaning("ankh", [], [], {"egyptian": ["X"]})],
    }
    trie = build_morpheme_trie(db)
    assert trie.morpheme_count == 1


@pytest.mark.parametrize(
    "sources_list, expected",
    [
        # Empty meanings list → vacuously True (unreachable from
        # build_morpheme_trie's gate, but pin the contract).
        ([], True),
        # No sources info at all → KEEP (legacy / test fixtures).
        ([{}], False),
        # All-Egyptian → SKIP.
        ([{"egyptian": ["X"]}], True),
        # All-Akkadian → SKIP.
        ([{"akkadian": ["X"]}], True),
        # All-Sumerian (forward-defensive entry) → SKIP.
        ([{"sumerian": ["X"]}], True),
        # Mixed phonogram + English → KEEP.
        ([{"egyptian": ["X"], "modern_english": ["s"]}], False),
        # Two meanings, one phonogram-only, one mixed → KEEP.
        ([{"egyptian": ["X"]}, {"egyptian": ["X"], "modern_english": ["s"]}], False),
        # Two meanings, both phonogram-only → SKIP.
        ([{"egyptian": ["X"]}, {"akkadian": ["Y"]}], True),
    ],
)
def test_is_phonogram_only_collision_contract(sources_list, expected):
    """wyrd-m2ym (direct-helper contract): the
    ``_is_phonogram_only_collision`` helper's full truth table.
    Documents the contract independent of trie-build coupling so the
    live-data 'X of Y filtered' figure in the PR body is reproducible
    from unit tests."""
    from wyrd.generators.kenning.runtime.trie_matcher import (
        _is_phonogram_only_collision,
    )

    meanings = [Meaning("X", [], [], sources) for sources in sources_list]
    assert _is_phonogram_only_collision(meanings) is expected


def test_build_trie_keeps_single_char_non_latin_surfaces():
    """wyrd-m2ym: single-char NON-ASCII surfaces (Akkadian ī with
    macron, Sanskrit Devanagari) survive the filter — the
    ``.isascii()`` guard means non-Latin codepoints don't trigger the
    skip. Non-Latin surfaces don't collide with English target words
    so the filter rationale doesn't apply."""
    db = {
        "ī": [Meaning("ī", [], [], {"akkadian": ["X"]})],
        "अ": [Meaning("अ", [], [], {"sanskrit": ["X"]})],
        "ham": [_meaning("ham")],
    }
    trie = build_morpheme_trie(db)
    assert trie.morpheme_count == 3


def test_build_trie_keeps_mixed_source_single_char_morpheme():
    """wyrd-m2ym (filter precision): when a single-char surface's
    Meaning carries BOTH a phonogram source AND a non-phonogram source
    (e.g. an Egyptian phonogram borrowed into English usage), the
    filter keeps it — the rule is 'pure phonogram only', not
    'phonogram present'. Avoids dropping legitimate borrowings."""
    db = {
        "s": [Meaning("s", [], [], {"egyptian": ["X"], "modern_english": ["s"]})],
    }
    trie = build_morpheme_trie(db)
    assert trie.morpheme_count == 1


def test_canonical_decomposition_clifts_no_phantom_phonogram_attribution():
    """wyrd-m2ym end-to-end: the user-reported regression — ``Clifts``
    decomposed as ``Clift + s`` where the ``s`` was a phantom Egyptian
    phonogram. Post-fix the canonical parse is ``Clift`` + unaccounted
    ``s`` (the actual English plural marker, which the explainer
    leaves unattributed rather than over-attributing to a phonogram
    transliteration)."""
    db = {
        # 'Clift-' as a pre-position morpheme so it matches at word
        # start; mirrors how the real bundle stores word-initial
        # morphemes like 'Bridg-'.
        "Clift-": [_meaning("Clift-")],
        "s": [_egyptian_meaning("s")],  # would have collided pre-fix
    }
    trie = build_morpheme_trie(db)
    decomp = canonical_decomposition("Clifts", trie)
    usages = [getattr(e, "usage", e) for e in decomp]
    assert usages == ["Clift-", "s"]
    # The trailing 's' must surface as raw string (unaccounted), NOT
    # as the Egyptian Meaning that ended up here pre-fix.
    assert not isinstance(decomp[-1], Meaning), (
        f"trailing 's' must surface as unaccounted string, not a Meaning; got {decomp[-1]!r}"
    )


# --- all_decompositions: single-path matches -------------------------------


def test_simple_compound_decomposes():
    """Bridg + water → both morphemes surface; no unaccounted chars."""
    db = {"Bridg-": [_meaning("Bridg-")], "-water": [_meaning("-water")]}
    trie = build_morpheme_trie(db)
    decompositions = all_decompositions("Bridgwater", trie)
    forms = [tuple(getattr(e, "usage", e) for e in d) for d in decompositions]
    # Best parse is Bridg- + -water; should be present.
    assert ("Bridg-", "-water") in forms


def test_no_match_returns_word_as_unaccounted():
    """A word with NO matching morphemes returns one decomposition
    that's just the word as a single unaccounted fragment."""
    db = {"Bridg-": [_meaning("Bridg-")]}
    trie = build_morpheme_trie(db)
    decomps = all_decompositions("xyz", trie)
    assert len(decomps) == 1
    assert decomps[0] == ["xyz"]


def test_empty_word_returns_empty_decomposition():
    """Empty input → one decomposition that's the empty list. Caller
    handles the empty-string case at its own surface."""
    db = {"a": [_meaning("a")]}
    trie = build_morpheme_trie(db)
    assert all_decompositions("", trie) == [[]]


# --- all_decompositions: multi-parse (the load-bearing case) ---------------


def test_multi_decomposition_when_morpheme_has_multiple_senses():
    """One usage, two Meanings (different senses, e.g. -y can mean
    'island' or 'district'). Both must surface as separate
    decompositions — the explainer needs every reading."""
    sense_island = _meaning("-y")
    sense_district = _meaning("-y")
    db = {"-y": [sense_island, sense_district]}
    trie = build_morpheme_trie(db)
    decomps = all_decompositions("y", trie)
    # The matched-Meaning identity differentiates the two — pick out
    # decompositions that match exactly the morpheme branch.
    matched = [d for d in decomps if d and not isinstance(d[0], str)]
    assert len(matched) == 2
    # Both Meaning instances should appear, distinct identities.
    assert {id(m) for d in matched for m in iter_morphemes(d)} == {
        id(sense_island),
        id(sense_district),
    }


def test_multi_decomposition_when_two_morphemes_match_same_position():
    """Trie carries 'ham-' AND 'hamlet' — both can match at pos 0
    (one short, one long). All_decompositions must surface every
    parse: 'ham- + let' (with 'let' unaccounted) AND 'hamlet' (full
    match). 'pre' / no-dash usages so wyrd-zewx's strict-inner
    semantics doesn't suppress the multi-parse axis (which is what's
    actually being tested)."""
    ham = _meaning("ham-")
    hamlet = _meaning("hamlet")
    db = {"ham-": [ham], "hamlet": [hamlet]}
    trie = build_morpheme_trie(db)
    decomps = all_decompositions("hamlet", trie)
    parses = []
    for d in decomps:
        parses.append(tuple((getattr(e, "usage", e), isinstance(e, str)) for e in d))
    assert (("hamlet", False),) in parses
    assert (("ham-", False), ("let", True)) in parses


def test_multi_decomposition_matches_at_different_starts():
    """Bridg + water AND Bri + dgwater. The trie returns both 'Bridg'
    starting at pos 0 AND 'water' starting at pos 5 (after skipping or
    matching prefix), and the DFS enumerates both paths."""
    bridg = _meaning("Bridg-")
    bri = _meaning("Bri-")
    water = _meaning("-water")
    db = {"Bridg-": [bridg], "Bri-": [bri], "-water": [water]}
    trie = build_morpheme_trie(db)
    decomps = all_decompositions("Bridgwater", trie)
    forms = [tuple(getattr(e, "usage", e) for e in d) for d in decomps]
    # Both compound parses should surface.
    assert ("Bridg-", "-water") in forms
    assert ("Bri-", "dgwater") in forms or any(
        # Bri- + something + -water also acceptable depending on skip steps.
        f[0] == "Bri-" and "-water" in f
        for f in forms
    )


def test_multi_decomposition_multiple_morphemes_same_start_different_lengths():
    """Trie has 'aber' AND 'aberdeen'. Walking 'aberdeen' from pos 0
    matches both (pos 4 and pos 8). Every match branches a separate
    decomposition path."""
    aber = _meaning("Aber-")
    aberdeen = _meaning("Aberdeen")
    db = {"Aber-": [aber], "Aberdeen": [aberdeen]}
    trie = build_morpheme_trie(db)
    decomps = all_decompositions("Aberdeen", trie)
    forms = [
        tuple(getattr(e, "usage", e) if not isinstance(e, str) else f"<{e}>" for e in d)
        for d in decomps
    ]
    assert ("Aberdeen",) in forms
    assert ("Aber-", "<deen>") in forms


# --- position constraints --------------------------------------------------


def test_pre_morpheme_only_matches_at_position_zero():
    """Bridg- is location='pre' (trailing dash). It must only match at
    the start of the word; if it appears mid-word, the matcher must
    NOT emit it."""
    bridg = _meaning("Bridg-")
    inner = _meaning("-bridg-")  # 'inner' for contrast
    db = {"Bridg-": [bridg], "-bridg-": [inner]}
    trie = build_morpheme_trie(db)
    # 'XbridgX' — bridg sits at pos 1..6 of len=7: strictly inner
    # (start>0 AND end<len). The 'pre' Meaning must NOT match (start
    # != 0); the 'inner' Meaning IS allowed.
    decomps = all_decompositions("XbridgX", trie)
    matched_meanings = [m for d in decomps for m in iter_morphemes(d) if not isinstance(m, str)]
    # 'pre' Bridg- should never appear in any decomposition.
    assert bridg not in matched_meanings
    # 'inner' -bridg- can appear (strictly inner).
    assert inner in matched_meanings


def test_post_morpheme_only_matches_at_word_end():
    """-water is location='post' (leading dash). It must only match
    when the match ends at len(word). If 'water' appears mid-word, the
    matcher must NOT emit it."""
    water_post = _meaning("-water")
    db = {"-water": [water_post]}
    trie = build_morpheme_trie(db)
    # 'waterX' — 'water' starts at 0, ends at 5, but len=6 → NOT
    # at word-end → must be excluded.
    decomps = all_decompositions("waterX", trie)
    matched = [m for d in decomps for m in iter_morphemes(d)]
    assert water_post not in matched
    # 'water' at the actual end IS allowed.
    decomps_end = all_decompositions("Bridgwater", trie)
    matched_end = [m for d in decomps_end for m in iter_morphemes(d)]
    assert water_post in matched_end


def test_inner_morpheme_only_matches_strictly_inside():
    """wyrd-zewx: an 'inner' Meaning (dashes both sides) must match
    STRICTLY inside the word — start > 0 AND end < word_length.
    Allowing inner morphemes at boundaries produced 'donhole' /
    'nwydmillate' style outputs from the post-wyrd-eni4 bundle's
    expanded inner-morpheme inventory.

    'by' as inner: matches in 'XbyX' (start=1, end=3, strictly
    interior of len=4) but NOT in 'by' (start=0, end=len), 'Xby'
    (end=len), or 'byX' (start=0)."""
    by = _meaning("-by-")
    db = {"-by-": [by]}
    trie = build_morpheme_trie(db)

    # Strictly-interior position — should match.
    decomps = all_decompositions("XbyX", trie)
    matched = [m for d in decomps for m in iter_morphemes(d)]
    assert by in matched, "'-by-' should match strictly inside 'XbyX'"

    # Boundary positions — should NOT match.
    for word in ("by", "Xby", "byX"):
        decomps = all_decompositions(word, trie)
        matched = [m for d in decomps for m in iter_morphemes(d)]
        assert by not in matched, f"'-by-' should NOT match at boundary in {word!r}"


# --- canonical_decomposition (single answer) -------------------------------


def test_canonical_picks_decomposition_with_fewest_unaccounted_chars():
    """Score primary axis: 'Bridg + water' explains all 10 chars,
    while 'Bridg + W + ater' explains 6 + leaves 4 unaccounted (in
    a hypothetical trie that lacks 'water'). The full-explanation
    parse wins."""
    db = {"Bridg-": [_meaning("Bridg-")], "-water": [_meaning("-water")]}
    trie = build_morpheme_trie(db)
    canonical = canonical_decomposition("Bridgwater", trie)
    forms = tuple(getattr(e, "usage", e) for e in canonical)
    assert forms == ("Bridg-", "-water")
    assert count_unaccounted(canonical) == 0


def test_canonical_prefers_fewer_morphemes_when_unaccounted_is_tied():
    """Score secondary axis: when two decompositions both have 0
    unaccounted chars, the one with FEWER morphemes wins. 'hamlet'
    (1 morpheme) beats 'ham + let' (1 morpheme + 1 unaccounted) on
    primary, AND 'hamlet' (1) beats 'h + a + m + l + e + t' (6) when
    every letter happens to also be a morpheme."""
    hamlet = _meaning("hamlet")
    db = {
        "hamlet": [hamlet],
        "h": [_meaning("h")],
        "a": [_meaning("a")],
        "m": [_meaning("m")],
        "l": [_meaning("l")],
        "e": [_meaning("e")],
        "t": [_meaning("t")],
    }
    trie = build_morpheme_trie(db)
    canonical = canonical_decomposition("hamlet", trie)
    assert len(list(iter_morphemes(canonical))) == 1
    assert canonical[0] is hamlet


def test_canonical_returns_word_as_unaccounted_when_no_matches():
    """No matching morphemes → the canonical decomposition is just
    the word as a single unaccounted fragment. Caller can render
    'unrecognized: foo' from this shape."""
    db = {"a": [_meaning("a")]}
    trie = build_morpheme_trie(db)
    assert canonical_decomposition("xyz", trie) == ["xyz"]


# --- canonical_decompositions (plural — preserves ties) --------------------


def test_canonical_plural_returns_all_ties_at_same_score():
    """Two parses tied at minimum score (e.g. one usage with two
    sense Meanings) — both must surface in the plural-canonical
    output, not just one. This is the user-flagged multi-parse
    invariant."""
    sense_a = _meaning("-y")
    sense_b = _meaning("-y")
    db = {"-y": [sense_a, sense_b]}
    trie = build_morpheme_trie(db)
    canonicals = canonical_decompositions("y", trie)
    # Filter to the matched-Meaning branch (skip the
    # all-unaccounted alternative).
    matched = [c for c in canonicals if c and not isinstance(c[0], str)]
    assert len(matched) == 2
    assert {id(m) for c in matched for m in iter_morphemes(c)} == {
        id(sense_a),
        id(sense_b),
    }


def test_canonical_plural_still_filters_inferior_parses():
    """Even when ties at the top score get preserved, parses that
    score WORSE (more unaccounted chars or more morphemes) are
    dropped. Pin: 'ham' canonical against {ham, h, a, m} returns just
    'ham' (1 morpheme, 0 unaccounted) and not 'h + a + m' (3 morphemes,
    0 unaccounted)."""
    ham = _meaning("ham")
    db = {
        "ham": [ham],
        "h": [_meaning("h")],
        "a": [_meaning("a")],
        "m": [_meaning("m")],
    }
    trie = build_morpheme_trie(db)
    canonicals = canonical_decompositions("ham", trie)
    # Only the 1-morpheme parse survives.
    assert all(len(list(iter_morphemes(c))) == 1 for c in canonicals)
    assert all(c[0] is ham for c in canonicals)


# --- helpers ---------------------------------------------------------------


def test_count_unaccounted_sums_string_lengths():
    """Convenience helper for callers that score downstream. Use an
    inner morpheme so position-constraint filtering doesn't interfere
    with the canonical pick."""
    db = {"-ham-": [_meaning("-ham-")]}
    trie = build_morpheme_trie(db)
    canonical = canonical_decomposition("XhamY", trie)
    # Best parse: ['X', -ham-, 'Y'] → 2 unaccounted chars.
    assert count_unaccounted(canonical) == 2


def test_iter_morphemes_yields_only_meaning_objects():
    """Helper drops unaccounted strings, yielding the matched-morpheme
    sequence in order. Inner morpheme to avoid position-constraint
    interaction with the test."""
    ham = _meaning("-ham-")
    db = {"-ham-": [ham]}
    trie = build_morpheme_trie(db)
    canonical = canonical_decomposition("XhamY", trie)
    morphemes = list(iter_morphemes(canonical))
    assert morphemes == [ham]


def test_morpheme_trie_can_be_constructed_empty():
    """Edge case: empty meaning_db produces a usable (no-match) trie."""
    trie = build_morpheme_trie({})
    assert trie.morpheme_count == 0
    decomp = canonical_decomposition("anyword", trie)
    assert decomp == ["anyword"]


def test_build_morpheme_trie_returns_distinct_instances():
    """Sanity: two calls to build_morpheme_trie with equivalent input
    return distinct objects (not a cached singleton). MorphemeTrie is
    declared with eq=False so structural equality is identity, which
    matches the caller's mental model — pass one instance around, not
    rebuild and compare."""
    t1 = build_morpheme_trie({"a": [_meaning("a")]})
    t2 = build_morpheme_trie({"a": [_meaning("a")]})
    assert t1 is not t2
    # Identity-equality follows from dataclass(eq=False) — the auto-
    # structural compare would have walked the trie subtree O(N).
    assert t1 != t2


def test_canonical_decomposition_is_deterministic():
    """Same trie + same input = same single answer across calls
    (uses a position-of-first-meaning tiebreaker)."""
    db = {"ham": [_meaning("ham")]}
    trie = build_morpheme_trie(db)
    a = canonical_decomposition("hamlet", trie)
    b = canonical_decomposition("hamlet", trie)
    assert a == b


def test_morpheme_trie_typechecks_expected_attributes():
    """Smoke that the dataclass exposes the documented surface."""
    t = build_morpheme_trie({"a": [_meaning("a")]})
    assert isinstance(t, MorphemeTrie)
    assert hasattr(t, "forward")
    assert hasattr(t, "morpheme_count")


# --- round-2 follow-ups: test-coverage P3 gaps ----------------------------


def test_compact_unaccounted_merges_consecutive_string_runs():
    """Direct unit pin for _compact_unaccounted: a decomposition with
    runs of single-character unaccounted elements (the raw DFS shape)
    collapses into single contiguous strings (the public output
    shape)."""
    from wyrd.generators.kenning.runtime.trie_matcher import _compact_unaccounted

    ham = _meaning("-ham-")
    # Mixed: leading run 'X','Y', a meaning, mid-run 'A','B','C',
    # a meaning, trailing 'Z'. Expect 'XY', meaning, 'ABC', meaning,
    # 'Z'.
    raw = ["X", "Y", ham, "A", "B", "C", ham, "Z"]
    compact = _compact_unaccounted(raw)
    assert compact == ["XY", ham, "ABC", ham, "Z"]


def test_compact_unaccounted_handles_no_strings_or_no_meanings():
    """Boundary cases: a pure-meaning decomposition (no strings) and a
    pure-string decomposition (no meanings) both pass through
    correctly — the merger only acts on adjacent string elements."""
    from wyrd.generators.kenning.runtime.trie_matcher import _compact_unaccounted

    ham = _meaning("-ham-")
    # All-meanings: no merge work.
    assert _compact_unaccounted([ham, ham]) == [ham, ham]
    # All-string single-char run: collapses to one string.
    assert _compact_unaccounted(["a", "b", "c"]) == ["abc"]
    # Empty: empty.
    assert _compact_unaccounted([]) == []


def test_skip_branch_runs_when_no_morpheme_matches_at_position():
    """The skip-one-character branch is what lets the matcher tolerate
    unrecognized fragments. Pin the branch in isolation: with a trie
    that only recognizes 'ham' (inner) and an input 'XhamY', the only
    decompositions reaching the pos=1 ham match come through the skip
    branch from pos=0 (skipping 'X'). Likewise the trailing 'Y' must
    come from the skip branch at pos=4."""
    ham = _meaning("-ham-")
    db = {"-ham-": [ham]}
    trie = build_morpheme_trie(db)
    decomps = all_decompositions("XhamY", trie)
    # The clean ['X', ham, 'Y'] parse exists (skip-then-match-then-skip).
    forms = [tuple(getattr(e, "usage", e) for e in d) for d in decomps]
    assert ("X", "-ham-", "Y") in forms
    # The all-skip parse also exists (no match at all → ['XhamY']).
    assert ("XhamY",) in forms


def test_canonical_decomposition_first_meaning_position_tiebreaker_fires():
    """The third score-tuple axis (first_meaning_pos = list index of
    the first matched element) is the deterministic tiebreaker after
    unaccounted-chars and morpheme-count tie.

    Setup: 'aab' with 'aa-' (pre, must start at pos 0) and '-ab'
    (post, must end at pos len). Both span 2 of 3 chars but overlap
    on the middle char so AT MOST ONE matches per parse:
      - [aa, 'b']     — match aa- at 0-2, skip b. (1, 1, 0).
      - ['a', ab]     — skip first, match -ab at 1-3. (1, 1, 1).
    Both tie on (1, 1); first_pos picks list-index 0 → [aa, 'b'].

    wyrd-zewx: pre/post used here instead of inner because strict-
    inner semantics rejects boundary positions; the overlap pattern
    is what matters for the tiebreaker, not which location flavour
    surfaces it."""
    aa = _meaning("aa-")
    ab = _meaning("-ab")
    db = {"aa-": [aa], "-ab": [ab]}
    trie = build_morpheme_trie(db)
    canonical = canonical_decomposition("aab", trie)
    matched = list(iter_morphemes(canonical))
    assert len(matched) == 1
    # Tiebreaker picks the one with first_meaning_pos=0 (i.e. the
    # decomposition that starts with a morpheme rather than a string).
    assert canonical[0] is aa, f"expected tiebreaker to pick aa; got {canonical}"


def test_canonical_decomposition_three_way_tie_falls_back_to_dict_order():
    """When (unaccounted, morpheme_count, first_meaning_pos) all tie,
    Python's min() returns the first occurrence in iteration order. The
    DFS visits the 'match' branch before 'skip' at each position, and
    iterates the trie's terminals list in insertion order. Two senses
    sharing one surface ('-y' = sense_a OR sense_b) produce
    decompositions [sense_a] and [sense_b], tying on all three axes;
    canonical picks whichever was inserted first in the meaning_db."""
    sense_a = _meaning("-y")
    sense_b = _meaning("-y")
    db = {"-y": [sense_a, sense_b]}
    trie = build_morpheme_trie(db)
    # Same DB iterated twice produces the same result.
    a = canonical_decomposition("y", trie)
    b = canonical_decomposition("y", trie)
    assert a == b
    # The matched Meaning is sense_a (first in insertion order).
    matched = list(iter_morphemes(a))
    assert matched and matched[0] is sense_a


def test_walk_memoization_preserves_correctness_on_repeated_substrings():
    """Memoization guards an exponential blow-up on inputs where the
    same suffix is reached from many distinct paths. With 'aaaaaaaaaa'
    (10 'a' characters) and 'a' as a 1-char no-dash morpheme (location
    'post' — matches only when end==len), the cache must still
    collapse the DFS without producing wrong answers.

    wyrd-zewx: previously this used '-a-' (inner) for the 'matches
    everywhere' axis, but strict-inner now rejects boundary positions.
    Switched to no-dash 'a' (post) — matches at pos 9 only — which
    still exercises the memoization path because the skip branch
    walks every position."""
    a = _meaning("a")
    db = {"a": [a]}
    trie = build_morpheme_trie(db)
    decomps = all_decompositions("aaaaaaaaaa", trie)
    # Result count is bounded (memoization).
    assert len(decomps) > 0
    forms = [tuple(getattr(e, "usage", e) for e in d) for d in decomps]
    # Some decomposition ends with the 'a' morpheme matched at pos 9.
    assert any(d and getattr(d[-1], "usage", None) == "a" for d in decomps), forms
    # The all-skip decomposition also exists.
    assert ("aaaaaaaaaa",) in forms


# --- wyrd-p8ve: score-pruning + per-position caps -------------------------


def test_canonical_decompositions_completes_on_long_input_with_dense_overlap():
    """wyrd-p8ve: a long word with many overlapping morpheme matches at
    every position used to OOM ``canonical_decompositions`` because the
    underlying ``all_decompositions`` walk cached the cartesian product
    of (match × cached-tail) at every position. The 58-char Welsh
    village 'Llanfairpwllgwyngyllgogerychwyrndrobwyllllantysiliogogogoch'
    blew memory past 16 GB during rebuild-proportions before the fix.

    Synthetic input here: 30 'a' chars with 'a-' (pre) / '-a-' (inner) /
    '-a' (post) all matching, so every position has 1-2 candidate
    matches plus a skip branch. Pre-fix this would explode; post-fix
    completes near-instantly because the score-pruning walk keeps only
    decompositions tied at the current best at each position."""
    import time

    # 'aa-' / '-a-' / '-aa' so we get matches at every position with
    # different end-positions (forces the multi-tail tie scenarios that
    # previously blew up the cache).
    pre = _meaning("aa-")
    inner = _meaning("-a-")
    post = _meaning("-aa")
    db = {"aa-": [pre], "-a-": [inner], "-aa": [post]}
    trie = build_morpheme_trie(db)

    word = "a" * 30
    t0 = time.monotonic()
    decomps = canonical_decompositions(word, trie)
    elapsed = time.monotonic() - t0

    # The fix runs in ms; 5 seconds is a generous ceiling that catches
    # the OOM blowup (which would never finish) without flaking on
    # slow CI runners.
    assert elapsed < 5.0, f"canonical_decompositions took {elapsed:.1f}s (regression?)"
    # Result count is bounded by MAX_TIED_DECOMPOSITIONS_PER_POSITION.
    assert 0 < len(decomps) <= 100


def test_canonical_decompositions_returns_global_minimum_via_score_pruning():
    """The score-pruning walk must produce the SAME global-minimum set
    as the legacy enumerate-then-filter approach. Pin: a deterministic
    multi-parse case where the canonical readings are well-defined.

    'Bridgwater' with 'Bridg-' (pre) and '-water' (post) has exactly
    one perfect parse: (Bridg-, -water). With both 'Bridg-' AND 'B-'
    in the trie, two parses now tie on (0, 2) score: (Bridg-, -water)
    and (B-, ridgwater) — except the latter has unaccounted 'ridgwater',
    so it scores (8, 1). Only (Bridg-, -water) wins."""
    bridg = _meaning("Bridg-")
    b = _meaning("B-")
    water = _meaning("-water")
    db = {"Bridg-": [bridg], "B-": [b], "-water": [water]}
    trie = build_morpheme_trie(db)

    decomps = canonical_decompositions("Bridgwater", trie)
    forms = [tuple(getattr(e, "usage", e) for e in d) for d in decomps]
    # Only the full-explanation parse wins on score (0 unaccounted).
    assert ("Bridg-", "-water") in forms
    # The (B-, ridgwater) parse has 8 unaccounted chars — strictly worse.
    assert all("ridgwater" not in str(f) for f in forms), forms


def test_all_decompositions_caps_per_position_on_pathological_input():
    """wyrd-p8ve: the pre-fix walk could grow the per-position cache
    without bound on long inputs with dense matches. The post-fix walk
    caps each position at MAX_DECOMPOSITIONS_PER_POSITION (1000) so
    KenningExplain stays responsive on pathological inputs even though
    it asks for the full enumeration. Pin: even with 30 'a' chars and
    every position matching multiple ways, the result list is bounded
    AND a DecompositionTruncatedWarning fires so callers can flag the
    non-exhaustive return in their UI."""
    pre = _meaning("aa-")
    inner = _meaning("-a-")
    post = _meaning("-aa")
    db = {"aa-": [pre], "-a-": [inner], "-aa": [post]}
    trie = build_morpheme_trie(db)

    with pytest.warns(DecompositionTruncatedWarning, match="per-position cap"):
        decomps = all_decompositions("a" * 30, trie)
    # Bounded by MAX_DECOMPOSITIONS_PER_POSITION at the top level.
    # Top-level result count tracks the cap directly when truncation
    # fires at position 0; allow up to 5× slack (truncated branches
    # downstream of pos 0 still contribute multiplicatively up to
    # the top-level cap on each branch).
    assert 0 < len(decomps) <= 50000


def test_all_decompositions_does_not_warn_on_normal_inputs():
    """The cap-fired warning must only fire on pathological inputs.
    Normal-length inputs (Bridgwater etc.) stay well under the cap
    and emit no warning — pin so a future change doesn't accidentally
    trip the warning on every call."""
    bridg = _meaning("Bridg-")
    water = _meaning("-water")
    db = {"Bridg-": [bridg], "-water": [water]}
    trie = build_morpheme_trie(db)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DecompositionTruncatedWarning)
        # If the warning fires here, simplefilter('error') turns it
        # into a raised exception — assertion is implicit.
        decomps = all_decompositions("Bridgwater", trie)
    assert len(decomps) > 0


def test_canonical_decomposition_singular_uses_score_pruning():
    """``canonical_decomposition`` (singular) routes through
    ``canonical_decompositions`` post-wyrd-p8ve so it inherits the
    memory bounds. Behavioral pin: the deterministic single-answer
    pick is unchanged."""
    bridg = _meaning("Bridg-")
    water = _meaning("-water")
    db = {"Bridg-": [bridg], "-water": [water]}
    trie = build_morpheme_trie(db)

    canonical = canonical_decomposition("Bridgwater", trie)
    forms = tuple(getattr(e, "usage", e) for e in canonical)
    assert forms == ("Bridg-", "-water")


# ---------------------------------------------------------------------------
# wyrd-pfoo: culture-aligned tagged-Meaning tiebreaker
# ---------------------------------------------------------------------------


def _meaning_tagged(usage: str, *, language: str, tags: list[str]) -> Meaning:
    """Build a Meaning with a specific primary source language + tags
    so the wyrd-pfoo tiebreaker has data to discriminate on. The
    matcher's tiebreaker reads ``meaning.sources`` for primary
    language and ``meaning.tags`` for the place-name-plausibility
    gate."""
    return Meaning(usage, tags, [], {language: [usage.replace("-", "")]})


def test_canonical_no_culture_kwarg_is_bit_stable_with_prior_behavior():
    """Default ``culture_languages=None`` skips the tiebreaker entirely
    → identical output to the pre-wyrd-pfoo matcher for the explainer /
    rewind / era-map paths that don't supply a culture context."""
    oe_town = _meaning_tagged("-ton", language="old_english", tags=["architecture", "social"])
    celtic_tone = _meaning_tagged("-ton", language="celtic_mix", tags=[])
    db = {"-ton": [oe_town, celtic_tone]}
    trie = build_morpheme_trie(db)

    # Both call shapes (no kwarg, explicit None) must produce the same
    # output as each other so callers that don't opt in stay on the
    # bit-stable path.
    no_kwarg = canonical_decompositions("ton", trie)
    explicit_none = canonical_decompositions("ton", trie, culture_languages=None)
    assert no_kwarg == explicit_none


def test_canonical_prefers_culture_aligned_tagged_meaning():
    """When two Meanings tie on (unaccounted, morpheme_count) and the
    culture hint matches one of them, the matcher prefers the aligned
    one. Welsh + `-ton` → Celtic-tagged Meaning wins over OE-tagged."""
    oe_town = _meaning_tagged("-ton", language="old_english", tags=["architecture", "social"])
    celtic_settlement = _meaning_tagged("-ton", language="celtic_mix", tags=["topography"])
    db = {"-ton": [oe_town, celtic_settlement]}
    trie = build_morpheme_trie(db)

    result = canonical_decompositions("ton", trie, culture_languages=frozenset({"celtic_mix"}))
    # The tiebreaker should keep only the Celtic-aligned parse.
    assert len(result) == 1
    assert result[0][0] is celtic_settlement


def test_canonical_skips_culture_aligned_untagged_meaning():
    """The ≥1-tag gate filters Wiktionary grammatical / modern
    homonyms that share a surface but carry no place-name semantic
    tags. Celtic ``-ton`` (the 'tone' / musical sense) has empty tags
    in the live bundle; the matcher must fall through to the OE-tagged
    Meaning instead of preferring the language-aligned but useless one."""
    oe_town = _meaning_tagged("-ton", language="old_english", tags=["architecture", "social"])
    celtic_tone = _meaning_tagged("-ton", language="celtic_mix", tags=[])  # nonsense sense
    db = {"-ton": [oe_town, celtic_tone]}
    trie = build_morpheme_trie(db)

    result = canonical_decompositions("ton", trie, culture_languages=frozenset({"celtic_mix"}))
    # Both Meanings are scored 0 on alignment (Celtic has no tags;
    # OE isn't in {celtic_mix}). Falls back to full list.
    assert len(result) == 2


def test_canonical_culture_tiebreaker_only_fires_on_score_ties():
    """A parse with FEWER unaccounted chars or FEWER morphemes wins
    regardless of culture alignment — the new tiebreaker is the third
    discriminator, applied only on ties from (1) and (2)."""
    # Wide Meaning that covers the whole word in one morpheme:
    full = _meaning_tagged("compound", language="old_english", tags=["architecture"])
    # Narrow Celtic Meanings that would yield more morphemes:
    com = _meaning_tagged("com-", language="celtic_mix", tags=["topography"])
    pound = _meaning_tagged("-pound", language="celtic_mix", tags=["topography"])
    db = {"compound": [full], "com-": [com], "-pound": [pound]}
    trie = build_morpheme_trie(db)

    result = canonical_decompositions("compound", trie, culture_languages=frozenset({"celtic_mix"}))
    # The 1-morpheme OE parse wins on morpheme_count even though the
    # 2-morpheme Celtic parse would have higher alignment.
    assert len(result) == 1
    assert result[0][0] is full


def test_canonical_culture_tiebreaker_returns_input_when_no_alignment():
    """When the culture-aligned set is non-empty but NO parse has any
    aligned-and-tagged Meaning (all parses score 0 on alignment), the
    function returns the full input list unchanged so the existing
    list-index tiebreaker still applies. Pin so a regression doesn't
    silently empty the result."""
    # Only OE Meanings — no celtic_mix candidate to align with.
    oe_a = _meaning_tagged("a-", language="old_english", tags=["topography"])
    oe_b = _meaning_tagged("-b", language="old_english", tags=["topography"])
    oe_ab = _meaning_tagged("ab", language="old_english", tags=["topography"])
    db = {"a-": [oe_a], "-b": [oe_b], "ab": [oe_ab]}
    trie = build_morpheme_trie(db)

    # Welsh culture, none of the OE meanings will align — return all
    # tied parses unchanged.
    welsh_result = canonical_decompositions("ab", trie, culture_languages=frozenset({"celtic_mix"}))
    # Same call without the tiebreaker.
    baseline = canonical_decompositions("ab", trie)
    assert welsh_result == baseline


def test_canonical_culture_tiebreaker_respects_multi_language_culture():
    """Cultures with multiple expected languages (Scottish = celtic +
    OE + ON) admit a Meaning whose primary language matches ANY of
    them. Pin so the set-membership semantics aren't accidentally
    narrowed to exact-string comparison."""
    on_meaning = _meaning_tagged("-by", language="old_scandinavian", tags=["architecture"])
    of_meaning = _meaning_tagged("-by", language="old_french", tags=["architecture"])
    db = {"-by": [on_meaning, of_meaning]}
    trie = build_morpheme_trie(db)

    result = canonical_decompositions(
        "by",
        trie,
        culture_languages=frozenset({"celtic_mix", "old_english", "old_scandinavian"}),
    )
    # ON aligns with the Scottish culture set; OF doesn't.
    assert len(result) == 1
    assert result[0][0] is on_meaning
