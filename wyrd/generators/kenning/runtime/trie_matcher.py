"""Trie-indexed segmentation DAG morpheme matcher (wyrd-k8e Phase 1).

Naming-precision note: this is NOT a pure trie matcher in the classical
'one input → one path → one match' sense. It's a **trie-indexed
segmentation DAG with multi-path enumeration**:

* The TRIE is just the character-level prefix index. Walking it from a
  given starting position is O(match-length) and surfaces EVERY
  morpheme that begins at that position (not just the longest).
* The SEGMENTATION DAG is built on top of the trie's hits: nodes are
  positions in the input word, edges are (start, end, meaning) triples
  produced by the trie. Plus a 'skip one character into the
  unaccounted bucket' edge from every position so the matcher
  tolerates unrecognized fragments.
* The DFS over that DAG enumerates EVERY full path from 0 to
  len(word) — i.e. every valid decomposition. Multi-parse is the
  whole point: a word with two senses for one surface (-y =
  'island' OR 'district') OR two competing breakdowns (the trie has
  both ``-ham-`` and ``-hamlet-`` → walking 'hamlet' surfaces both
  ``ham + let`` and ``hamlet`` as separate decompositions) surfaces
  ALL of them. See the unit tests for the canonical multi-parse axes.

Foundation for wyrd-08m (decomposition multiplicity) and wyrd-cv3
(corpus DB ingest). Superseded the original Rando-port morpheme-
iteration matcher (Word.extract_meanings, removed in Phase 3 /
wyrd-zhhz) which was O(M) per word per recursion level — this trie
form is O(match-length) per position. Phase 3 / wyrd-zhhz made the
trie the only matcher; the use_trie flag and the iterator both went
away.

Design:

* ``build_morpheme_trie(meaning_db)`` walks every Meaning and inserts
  its dash-stripped surface form into a character-trie. Each accepting
  state remembers the Meaning(s) that finish there. Multiple Meanings
  sharing a surface (different senses) all attach to the same state's
  terminals list — the DAG branches each one as its own edge.
* ``all_decompositions(word, trie)`` walks the trie from every
  starting position, builds the segmentation DAG, and DFS-enumerates
  every path from 0 to len(word). Result is a list of decompositions;
  each decomposition is a list whose elements alternate between
  matched Meaning objects and unaccounted string fragments. Tail
  fragments stay in the input's original casing so the rendered output
  preserves it.
* ``canonical_decompositions(word, trie)`` returns every
  decomposition tied for 'best' (lowest unaccounted chars, fewest
  morphemes). Plural for multi-parse callers.
* ``canonical_decomposition(word, trie)`` returns ONE deterministic
  pick. Single-answer convenience for callers that want exactly one.

Position-constraint semantics match the legacy ``Meaning.location``:

* ``pre``   — anchored at position 0 only.
* ``post``  — anchored at position len(word) only.
* ``inner`` / unset — usable anywhere a match starts.

Match characters lowercased — surface form preserves the input casing
in unaccounted fragments via slice (not lower()) extraction, so the
decomposition can be rendered back without losing case info.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


class DecompositionTruncatedWarning(UserWarning):
    """Raised by ``all_decompositions`` when its per-position result
    cap fires — the returned list is a non-exhaustive subset of the
    full enumeration. Callers like KenningExplain that surface 'every
    valid reading' to the user MAY want to flag the truncation in
    their UI; most callers can ignore the warning. The actual cap
    constant is ``MAX_DECOMPOSITIONS_PER_POSITION``."""


@dataclass(eq=False)
class _TrieNode:
    """One node in the character-keyed morpheme trie. ``terminals`` is
    the list of Meanings that finish at this node (empty for non-
    accepting states). ``children`` is the next-character map.

    eq=False because the auto-generated structural equality recurses
    into the children dict — comparing two trie nodes would walk the
    full subtree, O(trie size). Identity equality is the right default
    for an internal index node. Callers that need structural
    equivalence should compare the morpheme inventories instead.
    """

    terminals: list[Any] = field(default_factory=list)
    children: dict[str, _TrieNode] = field(default_factory=dict)


@dataclass(eq=False)
class MorphemeTrie:
    """Top-level wrapper around the character-keyed morpheme trie. Phase
    1 ships a single forward trie (indexes morphemes by their first
    character; walking from any position surfaces every morpheme that
    starts there). Position constraints (pre / post / inner) are
    applied by filtering the match list after the trie returns hits —
    no separate reverse trie is needed.

    Stored separately from MeaningDB so the same trie can be reused
    across many ``all_decompositions`` calls without rebuild.

    eq=False to match _TrieNode — comparing two MorphemeTrie instances
    structurally would walk the full subtree, surprising O(N) cost.
    Identity equality (``is``) is the right semantics for an immutable
    index built once at startup.
    """

    forward: _TrieNode = field(default_factory=_TrieNode)
    # Total morphemes ingested — informational; lets a CLI report
    # 'matcher built from N morphemes' without re-counting the trie.
    morpheme_count: int = 0


def build_morpheme_trie(meaning_db: dict[str, list[Any]]) -> MorphemeTrie:
    """Build a MorphemeTrie from a meaning_db (the runtime's
    ``dict[usage, list[Meaning]]`` index). Each Meaning's surface form
    is the dash-stripped lowercased usage; multiple Meanings sharing
    one surface form (different senses, e.g. ``-y`` = 'island' or
    'district') all attach to the same accepting state.

    Empty surface forms are skipped (would otherwise cause empty-edge
    cycles in the segmentation DAG).

    Determinism: iteration order of ``meaning_db.items()`` is
    preserved through to the order of terminals at each accepting
    state, which in turn determines the order of decompositions
    returned by ``all_decompositions`` and the order ties surface in
    ``canonical_decompositions``. Python 3.7+ dicts preserve insertion
    order, so a deterministically-built meaning_db gives
    deterministic matcher output across runs."""
    trie = MorphemeTrie()
    for usage, meanings in meaning_db.items():
        if not meanings:
            continue
        surface = usage.lower().replace("-", "")
        if not surface:
            continue
        node = trie.forward
        for ch in surface:
            node = node.children.setdefault(ch, _TrieNode())
        # Attach every Meaning sharing this surface — generation paths
        # (sense disambiguation) downstream pick by tag/score.
        for m in meanings:
            node.terminals.append(m)
        trie.morpheme_count += len(meanings)
    return trie


def _location_for(meaning: Any) -> str:
    """Extract a position constraint from a Meaning-like object.
    Tolerates objects that don't carry .location (treats as 'inner')
    so the matcher works on synthetic test fixtures too."""
    return getattr(meaning, "location", "inner")


def _find_matches_at(
    trie: MorphemeTrie,
    word: str,
    start: int,
) -> list[tuple[int, Any]]:
    """Walk the forward trie from word[start:] and return every
    (end_position, meaning) pair where a morpheme matches starting at
    ``start``. Lowercases the input character-by-character so the match
    is case-insensitive while preserving the input's casing for
    unaccounted fragments."""
    matches: list[tuple[int, Any]] = []
    node = trie.forward
    pos = start
    n = len(word)
    while pos < n:
        ch = word[pos].lower()
        nxt = node.children.get(ch)
        if nxt is None:
            break
        pos += 1
        node = nxt
        if node.terminals:
            for m in node.terminals:
                matches.append((pos, m))
    return matches


def _location_allows(meaning: Any, start: int, end: int, word_length: int) -> bool:
    """Filter a candidate match by its Meaning's position constraint.

    * ``pre`` — must start at position 0.
    * ``post`` — must end at ``word_length``.
    * ``inner`` — must be STRICTLY inner: ``start > 0`` AND
      ``end < word_length``. Inner-only morphemes (those with dashes
      on both sides like ``-don-``, ``-stone-``) only make sense as
      compound infixes; allowing them at boundaries produced the
      ``donhole`` / ``nwydmillate`` style outputs that wyrd-zewx
      caught in the post-wyrd-eni4 bundle.
    * unset / unknown — no anchor constraint (same permissive
      fallback the original Rando-port iterator used for synthetic
      test fixtures and edge-case Meaning subclasses).

    The strict-inner rule means a 4-char word whose only matching
    morpheme is inner-only has no decomposition via that morpheme;
    the matcher falls back to the skip-one-char branch and emits
    the chars as unaccounted. That's correct: the morpheme really
    isn't a valid compound element for that input position.
    """
    location = _location_for(meaning)
    if location == "pre":
        return start == 0
    if location == "post":
        return end == word_length
    if location == "inner":
        return start > 0 and end < word_length
    # Unknown / synthetic — no anchor constraint.
    return True


# wyrd-p8ve: hard caps so adversarial / long inputs (the 58-char
# Welsh village 'Llanfairpwllgwyngyllgogerychwyrndrobwyllllantysiliogogogoch'
# was the trigger) can't blow memory. ``all_decompositions`` enumerates
# every parse so it's the cap that actually fires in pathological
# cases — bounded at the per-position result list. Bit-stable for
# normal-length inputs (their actual count never approaches the cap).
# ``canonical_decompositions`` score-prunes during the walk so its
# cache never grows the cartesian product in the first place; the
# tie cap is a defensive fallback for cases where many decompositions
# legitimately tie at the same score.
# Cap chosen to be high enough that NORMAL inputs (typical place
# names against the live ~8500-subject bundle) never trip it —
# 'Westminster' against the live bundle generates several thousand
# decompositions, so 1000 was too low. 10000 gives ~10× headroom
# while still bounding pathological cases (the 58-char Welsh village
# was the OOM trigger). Worst-case memory at cap × max_decomp_len ×
# n_positions is ~30 MB even at the cap on a long input — safe vs
# the 16+ GB pre-fix blowup.
MAX_DECOMPOSITIONS_PER_POSITION = 10000
MAX_TIED_DECOMPOSITIONS_PER_POSITION = 100


def all_decompositions(word: str, trie: MorphemeTrie) -> list[list[Any]]:
    """Enumerate every valid decomposition of ``word`` against the
    trie. Each result is a list whose elements alternate between
    matched Meaning objects and unaccounted string fragments — that
    invariant is load-bearing for downstream callers (``iter_morphemes``,
    ``count_unaccounted``, ``_decomposition_score``) which use
    ``isinstance(elem, str)`` as the unaccounted-vs-morpheme switch.
    Don't put any third element type into the list. Empty fragments
    are not emitted.

    A decomposition that has no matches at all is the single-element
    list ``[word]`` — caller decides whether to display the un-
    decomposed word as a fallback. Empty input returns ``[[]]`` —
    one decomposition that's the empty list.

    Walks the segmentation DAG via DFS. For each position, we emit
    every (match, residue-decomposition) combination, and we also emit
    the 'skip one character into the unaccounted bucket' alternative.
    The skip path is what makes the matcher tolerant of unrecognized
    fragments — a name with one matchable morpheme and unrecognized
    surrounding chars still produces a usable partial decomposition.

    wyrd-p8ve: per-position result list is capped at
    ``MAX_DECOMPOSITIONS_PER_POSITION`` so adversarial / very long
    inputs (the 58-char Welsh village name was the trigger) can't
    blow memory. When the cap fires the returned list is a
    deterministic non-exhaustive subset of the full enumeration —
    a ``DecompositionTruncatedWarning`` is raised so callers like
    KenningExplain that surface 'every valid reading' to the user
    can flag the truncation in their UI. Most callers can ignore
    the warning; the bit-stable subset is still a valid partial
    answer. ``canonical_decompositions`` (the score-pruning path) is
    the right entry point for callers that only need the best parses
    — its memory is bounded by score-pruning regardless of input
    length, so it doesn't trip the cap or emit the warning.
    """
    n = len(word)
    if n == 0:
        return [[]]
    # Memoize results per starting position so the segmentation DAG
    # doesn't re-explore the same suffix exponentially. The key is
    # ``pos`` alone because ``word`` and ``trie`` are loop-invariant
    # in the closure — the suffix is fully determined by the position.
    cache: dict[int, list[list[Any]]] = {}
    truncated = [False]  # Closure-mutable flag — set if the cap ever fired.

    def walk(pos: int) -> list[list[Any]]:
        if pos == n:
            return [[]]
        if pos in cache:
            return cache[pos]
        results: list[list[Any]] = []
        # Branch 1: every trie-match starting at pos that satisfies
        # its meaning's position constraint.
        for end, meaning in _find_matches_at(trie, word, pos):
            if not _location_allows(meaning, pos, end, n):
                continue
            for tail in walk(end):
                if len(results) >= MAX_DECOMPOSITIONS_PER_POSITION:
                    truncated[0] = True
                    break
                results.append([meaning, *tail])
            if len(results) >= MAX_DECOMPOSITIONS_PER_POSITION:
                break
        # Branch 2: skip one character into the unaccounted bucket and
        # recurse. The skip lets the matcher partially-decompose names
        # that have unrecognized fragments. Adjacent skip characters
        # collapse into one unaccounted string at emit time.
        if len(results) < MAX_DECOMPOSITIONS_PER_POSITION:
            for tail in walk(pos + 1):
                if len(results) >= MAX_DECOMPOSITIONS_PER_POSITION:
                    truncated[0] = True
                    break
                results.append([word[pos], *tail])
        cache[pos] = results
        return results

    raw = walk(0)
    if truncated[0]:
        warnings.warn(
            f"all_decompositions on {word!r} (len={n}) hit the "
            f"per-position cap of {MAX_DECOMPOSITIONS_PER_POSITION}; "
            f"returned list is a deterministic but non-exhaustive subset. "
            f"Use canonical_decompositions for memory-bounded best-parse "
            f"enumeration, or raise the cap if exhaustiveness matters here.",
            DecompositionTruncatedWarning,
            stacklevel=2,
        )
    # Compress runs of single-character unaccounted strings into one
    # contiguous string — the legacy matcher's output shape.
    return [_compact_unaccounted(d) for d in raw]


def _compact_unaccounted(decomposition: list[Any]) -> list[Any]:
    """Merge consecutive unaccounted-string elements into single
    strings. The DFS emits one char per skip-step; downstream callers
    expect contiguous unaccounted fragments."""
    out: list[Any] = []
    for elem in decomposition:
        if isinstance(elem, str) and out and isinstance(out[-1], str):
            out[-1] += elem
        else:
            out.append(elem)
    return out


def canonical_decompositions(word: str, trie: MorphemeTrie) -> list[list[Any]]:
    """Return every decomposition tied for 'best' under the canonical
    score. Multiple ties are common when an etymon has multiple senses
    sharing one surface form (-y → 'island' OR 'district') or when the
    trie matches more than one morpheme at the same position
    ('hampton' = ham + p + ton OR ham + pton). Callers that need to
    enumerate all top-tier readings (the kenning explainer, multi-
    decomposition consumers in wyrd-08m) iterate this list; callers
    that only need ONE deterministic answer use
    ``canonical_decomposition``.

    Scoring (lowest score wins):

    1. Total unaccounted characters (fewer is better — more of the
       word explained).
    2. Total morpheme count (fewer is better — Occam preference;
       'Bridg + water' beats 'B + r + i + d + g + water' when the
       lone-letter morphemes happen to also match).

    Empty input returns ``[[]]``. A word with no matches at all
    returns ``[[word]]``.

    wyrd-p8ve: implementation score-prunes during the DAG walk
    rather than enumerating ``all_decompositions`` then filtering.
    The previous shape (enumerate-then-filter) blew up memory on
    long words with many overlapping matches because the walk's
    cache stored the cartesian product of (match × cached-tail)
    at every position — the 58-char Welsh village name
    'Llanfairpwllgwyngyllgogerychwyrndrobwyllllantysiliogogogoch'
    OOM'd at 16+ GB during rebuild-proportions. Score-pruning
    bounds the cache: at each position only the decompositions
    tied at that position's best score survive, so the cache size
    stays linear in the tie count (typically 1-3) instead of
    exponential.

    The walk's recurrence:
      - At ``pos == n``: empty suffix, score = (0, 0).
      - Else: collect candidates from every match-branch (each adds
        +1 to morpheme count, +0 to unaccounted) and the skip-1-char
        branch (+0 morpheme, +1 unaccounted). Each candidate's score
        adds the chosen tail's best-score. Take the global minimum,
        keep ties.

    The kept-tied-list at each position is capped at
    ``MAX_TIED_DECOMPOSITIONS_PER_POSITION`` (defensive fallback for
    cases where many decompositions legitimately tie). For real
    inputs the tie count is small and the cap never fires.
    """
    n = len(word)
    if n == 0:
        return [[]]

    # cache: pos → (best_score_for_suffix, [decompositions tied at best]).
    cache: dict[int, tuple[tuple[int, int], list[list[Any]]]] = {}

    def walk_best(pos: int) -> tuple[tuple[int, int], list[list[Any]]]:
        if pos == n:
            return ((0, 0), [[]])
        if pos in cache:
            return cache[pos]
        candidates: list[tuple[tuple[int, int], list[Any]]] = []
        # Branch 1: each trie-match contributes +1 morpheme, +0 unaccounted.
        for end, meaning in _find_matches_at(trie, word, pos):
            if not _location_allows(meaning, pos, end, n):
                continue
            (tail_un, tail_mor), tails = walk_best(end)
            score = (tail_un, tail_mor + 1)
            for tail in tails:
                candidates.append((score, [meaning, *tail]))
        # Branch 2: skip-one-char contributes +0 morpheme, +1 unaccounted.
        (tail_un, tail_mor), tails = walk_best(pos + 1)
        score = (tail_un + 1, tail_mor)
        for tail in tails:
            candidates.append((score, [word[pos], *tail]))
        # Pick the best score; keep only the candidates tied at it.
        # ``candidates`` is always non-empty: the skip branch produces
        # at least one entry for every pos < n (walk_best(pos+1) always
        # yields at least the skip-the-rest path).
        best_score = min(s for s, _ in candidates)
        best_decomps = [d for s, d in candidates if s == best_score]
        # Defensive cap on the tied-best list. Bit-stable: candidate
        # iteration order is deterministic, so the kept subset is a
        # function of the trie + word.
        if len(best_decomps) > MAX_TIED_DECOMPOSITIONS_PER_POSITION:
            best_decomps = best_decomps[:MAX_TIED_DECOMPOSITIONS_PER_POSITION]
        cache[pos] = (best_score, best_decomps)
        return cache[pos]

    _, decompositions = walk_best(0)
    return [_compact_unaccounted(d) for d in decompositions]


def canonical_decomposition(word: str, trie: MorphemeTrie) -> list[Any]:
    """Return ONE 'best' decomposition for ``word`` against the trie —
    the deterministic single-answer variant of
    ``canonical_decompositions``. Use this when the caller wants one
    pick (e.g. proportions training, single-line CLI output); use the
    plural ``canonical_decompositions`` when ties carry information
    (e.g. the explainer surfaces every reading scholars would).

    Tiebreaker (after the unaccounted-chars + morpheme-count score
    used by ``canonical_decompositions``): position of the first
    matched morpheme — earlier wins. Stable across runs.

    wyrd-p8ve: routes through ``canonical_decompositions`` so it
    inherits the score-pruning walk's memory bounds. The (un, mor)
    tied set is small enough that applying the first-meaning-pos
    tiebreaker via ``min`` over the tied list is O(K) with K << N.
    ``canonical_decompositions`` is guaranteed to return at least one
    decomposition (``[[]]`` for empty input, ``[[word]]`` for no
    matches), so no empty-list guard is needed before ``min``.
    """
    decompositions = canonical_decompositions(word, trie)
    return min(decompositions, key=_decomposition_score)


def _decomposition_score(decomposition: list[Any]) -> tuple[int, int, int]:
    """Score tuple for the canonical picker. Lower wins."""
    unaccounted_chars = sum(len(e) for e in decomposition if isinstance(e, str))
    morpheme_count = sum(1 for e in decomposition if not isinstance(e, str))
    # Tiebreaker: prefer earlier first-meaning. Stable for matchers
    # that produce the same DAG — primarily a determinism guard for
    # tests / serialization.
    first_meaning_pos = next(
        (i for i, e in enumerate(decomposition) if not isinstance(e, str)),
        len(decomposition),
    )
    return (unaccounted_chars, morpheme_count, first_meaning_pos)


def iter_morphemes(decomposition: Iterable[Any]) -> Iterable[Any]:
    """Yield only the Meaning elements of a decomposition, dropping
    unaccounted fragments. Convenience for callers that want the
    matched morpheme sequence (e.g. proportions training)."""
    for elem in decomposition:
        if not isinstance(elem, str):
            yield elem


def count_unaccounted(decomposition: Iterable[Any]) -> int:
    """Total unaccounted character count across a decomposition.
    Mirrors Word.count_unaccounted on the legacy matcher."""
    return sum(len(e) for e in decomposition if isinstance(e, str))
