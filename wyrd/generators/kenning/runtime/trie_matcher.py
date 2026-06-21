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

from wyrd.generators.kenning.runtime.connective import (
    GENITIVE,
    Connective,
    ConnectiveInventory,
    is_connective,
)


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


# wyrd-m2ym: source-language families whose writing systems use signs
# that transliterate to single ASCII Latin letters (Egyptian hieroglyph
# phonograms s/t/n/m/h/j/p/w/z; Akkadian cuneiform u/ī etc.). Single-
# char transliterations from these languages collide with arbitrary
# letter positions in English target words, manufacturing phantom
# attributions like 'Clifts' → 'Clift + s [Egyptian]'. The data is
# correct upstream (those phonograms ARE single consonants per wave-2
# wiktextract mining); the runtime trie just can't usefully match
# transliterations against Latin-script targets.
#
# 'sumerian' is a forward-defensive entry — the current bundle
# emitter (_emit.py) collapses Sumerian (sux) into the 'akkadian'
# bucket so production Meaning.sources never carries a 'sumerian' key
# today. Keeping the entry costs nothing and survives a future bundle
# split that emits Sumerian under its own tag.
_PHONOGRAM_TRANSLITERATION_LANGS: frozenset[str] = frozenset({"egyptian", "akkadian", "sumerian"})


def _is_phonogram_only_collision(meanings: list[Any]) -> bool:
    """All meanings at this surface have sources EXCLUSIVELY in the
    phonogram-transliteration languages — no English-family or other
    Latin-script-target source claims the surface. We keep the surface
    in the trie whenever at least one Meaning has a non-phonogram
    source (or when no source info is present at all — empty-sources
    test fixtures and legacy bundles default to KEEP)."""
    for m in meanings:
        sources = getattr(m, "sources", None)
        if not sources or not sources.keys() <= _PHONOGRAM_TRANSLITERATION_LANGS:
            return False
    return True


def build_morpheme_trie(meaning_db: dict[str, list[Any]]) -> MorphemeTrie:
    """Build a MorphemeTrie from a meaning_db (the runtime's
    ``dict[usage, list[Meaning]]`` index). Each Meaning's surface form
    is the dash-stripped lowercased usage; multiple Meanings sharing
    one surface form (different senses, e.g. ``-y`` = 'island' or
    'district') all attach to the same accepting state.

    Empty surface forms are skipped (would otherwise cause empty-edge
    cycles in the segmentation DAG).

    Single-character bare-ASCII-Latin surfaces from phonogram-
    transliteration languages (Egyptian / Akkadian / Sumerian) are
    skipped (wyrd-m2ym) — see ``_is_phonogram_only_collision`` for the
    exact filter. Single-char English suffixes like ``-y`` survive
    because their Meaning carries an English-family source.

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
        if (
            len(surface) == 1
            and surface.isascii()
            and surface.isalpha()
            and _is_phonogram_only_collision(meanings)
        ):
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


def _find_matches_at(
    trie: MorphemeTrie,
    word: str,
    start: int,
) -> list[tuple[int, Any]]:
    """Walk the forward trie from word[start:] and return every
    (end_position, meaning) pair where a morpheme matches starting at
    ``start``. Lowercases the input character-by-character so the match
    is case-insensitive while preserving the input's casing for
    unaccounted fragments.

    wyrd-eyjk/D40: a morpheme is its STRING — it matches at any span, with NO
    position gate or position-based variant selection. Every terminal at a
    matching surface is emitted; bare/pre/-inner-/-post is derived from the
    span/index downstream (``Word.get_structure`` / ``Word.get_samples``,
    which re-dash the surface to the derived position) and the credible split
    is chosen by the scorer — position is never smuggled into the match here."""
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
        for m in node.terminals:
            matches.append((pos, m))
    return matches


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
# Tied-best cap. Raised 100→256 (wyrd-aicu.3): real toponyms can legitimately
# tie above 100 ('Tonbridge' = 120 zero-unaccounted 2-morpheme parses), and a
# 100-cap truncated genuine parses by arbitrary DAG/export order BEFORE the
# culture tiebreaker could see them — the de-dash reshuffled the export and
# pushed OE 'tūn' past the cap, leaking the celtic 'wave' homograph into english
# 'ton' AND starving the OE morpheme. 256 keeps realistic tie sets whole so the
# tiebreaker works naturally; _cap_tied_decompositions still prefers culture-
# aligned parses for the rare set that exceeds even 256. Memory is trivial — the
# load-bearing OOM bound is MAX_DECOMPOSITIONS_PER_POSITION above (~30 MB @10000).
MAX_TIED_DECOMPOSITIONS_PER_POSITION = 256


def _append_capped(
    results: list[list[Any]],
    head: Any,
    tails: list[list[Any]],
    truncated: list[bool],
) -> bool:
    """Append ``[head, *tail]`` for each tail in ``tails`` until the
    per-position cap (``MAX_DECOMPOSITIONS_PER_POSITION``) is reached.

    Sets ``truncated[0] = True`` and returns ``True`` the moment the cap
    fires (so the caller can stop iterating outer branches); returns
    ``False`` when all tails were appended within the cap.
    """
    for tail in tails:
        if len(results) >= MAX_DECOMPOSITIONS_PER_POSITION:
            truncated[0] = True
            return True
        results.append([head, *tail])
    return False


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
        # Branch 1: every trie-match starting at pos. wyrd-eyjk/D40: position
        # is NOT a match constraint — a morpheme is its string and may match
        # anywhere; bare/pre/post/inner is derived from the span afterward
        # (Word.get_structure) and statistically ranked at build time, never
        # used to reject a match here.
        for end, meaning in _find_matches_at(trie, word, pos):
            if _append_capped(results, meaning, walk(end), truncated):
                break
            # State-check break, faithful to the original: stop scanning
            # further matches once results is full even if this match's
            # _append_capped didn't itself trip the cap (e.g. empty tails
            # at an exactly-full results). Does NOT set truncated[0] —
            # matches the pre-refactor flag semantics exactly.
            if len(results) >= MAX_DECOMPOSITIONS_PER_POSITION:
                break
        # Branch 2: skip one character into the unaccounted bucket and
        # recurse. The skip lets the matcher partially-decompose names
        # that have unrecognized fragments. Adjacent skip characters
        # collapse into one unaccounted string at emit time.
        if len(results) < MAX_DECOMPOSITIONS_PER_POSITION:
            _append_capped(results, word[pos], walk(pos + 1), truncated)
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


def _connective_candidates(
    word: str,
    pos: int,
    trie: MorphemeTrie,
    connective_inventory: ConnectiveInventory | None,
    walk_best: Any,
) -> list[tuple[tuple[int, int], list[Any]]]:
    """walk_best Branch 3 (wyrd-aicu.9): connective glue. A connective surface
    (genitive ``-s-`` …) consumed between morphemes is +0 unaccounted +0
    content-morpheme; a head morpheme must follow. ``pos > 0`` so a connective is
    never word-initial (it needs a preceding specifier; whether that preceding
    span is itself a clean morpheme is settled by the score — an unaccounted
    prefix loses). Returns ``[]`` when no inventory (bit-stable default)."""
    out: list[tuple[tuple[int, int], list[Any]]] = []
    if not connective_inventory or pos == 0:
        return out
    for spec in connective_inventory:
        clen = len(spec.surface)
        if word[pos : pos + clen].lower() != spec.surface:
            continue
        conn = Connective(word[pos : pos + clen], spec.kind)
        for end, head in _find_matches_at(trie, word, pos + clen):
            (h_un, h_mor), h_tails = walk_best(end)
            cscore = (h_un, h_mor + 1)  # connective free; head +1 morpheme
            out.extend((cscore, [conn, head, *tail]) for tail in h_tails)
    return out


def canonical_decompositions(
    word: str,
    trie: MorphemeTrie,
    *,
    culture_languages: frozenset[str] | None = None,
    connective_inventory: ConnectiveInventory | None = None,
    genitive_prior: dict[tuple[str, str], float] | None = None,
) -> list[list[Any]]:
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
    3. wyrd-pfoo: when ``culture_languages`` is supplied, parses tied
       on (1) + (2) are further reduced by preferring those with the
       most Meanings whose primary language is in ``culture_languages``
       AND who carry ≥1 tag (filters Wiktionary grammatical / modern
       homonyms that carry no place-name semantic tags — Celtic
       ``-ton`` → 'tone', ``Saint-`` → 'greed', etc.). When
       ``culture_languages`` is None (default), the tiebreaker is
       skipped — bit-stable with the pre-PR matcher for the explainer
       / rewind / era-map paths.

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
        # wyrd-eyjk/D40: no position gate — every string-match stands; position
        # is derived from the span and statistically ranked, never rejected.
        for end, meaning in _find_matches_at(trie, word, pos):
            (tail_un, tail_mor), tails = walk_best(end)
            score = (tail_un, tail_mor + 1)
            for tail in tails:
                candidates.append((score, [meaning, *tail]))
        # Branch 2: skip-one-char contributes +0 morpheme, +1 unaccounted.
        (tail_un, tail_mor), tails = walk_best(pos + 1)
        score = (tail_un + 1, tail_mor)
        for tail in tails:
            candidates.append((score, [word[pos], *tail]))
        # Branch 3 (wyrd-aicu.9): connective glue (extracted to keep walk_best
        # under the complexity ceiling). Bit-stable when the inventory is None.
        candidates.extend(_connective_candidates(word, pos, trie, connective_inventory, walk_best))
        # Pick the best score; keep only the candidates tied at it.
        # ``candidates`` is always non-empty: the skip branch produces
        # at least one entry for every pos < n (walk_best(pos+1) always
        # yields at least the skip-the-rest path).
        best_score = min(s for s, _ in candidates)
        best_decomps = [d for s, d in candidates if s == best_score]
        best_decomps = _cap_tied_decompositions(best_decomps, culture_languages)
        cache[pos] = (best_score, best_decomps)
        return cache[pos]

    _, decompositions = walk_best(0)
    decompositions = [_compact_unaccounted(d) for d in decompositions]
    if culture_languages and len(decompositions) > 1:
        decompositions = _prefer_culture_aligned(decompositions, culture_languages)
    if genitive_prior and len(decompositions) > 1:
        decompositions = _prefer_genitive_credible(decompositions, genitive_prior)
    return decompositions


def _cap_tied_decompositions(
    best_decomps: list[list[Any]], culture_languages: frozenset[str] | None
) -> list[list[Any]]:
    """Cap the per-position tied-best list at ``MAX_TIED_DECOMPOSITIONS_PER_POSITION``
    (load-bearing: bounds the DAG-walk cache so pathological long Welsh names
    don't OOM). When a culture is supplied, truncate toward the most
    culture-aligned parses FIRST so a correct culture parse can't be dropped by
    arbitrary DAG order — the cap runs BEFORE :func:`_prefer_culture_aligned`,
    so an order-blind ``[:cap]`` could discard the very parse the tiebreaker
    wants (wyrd-aicu.3: the de-dash reordered the export, pushing OE ``tūn`` for
    'Tonbridge' past the cap → the celtic 'wave' homograph leaked into english
    attestation). Python's sort is stable, so equal-alignment parses keep their
    deterministic DAG order → still bit-stable.
    """
    if len(best_decomps) <= MAX_TIED_DECOMPOSITIONS_PER_POSITION:
        return best_decomps
    if culture_languages:
        best_decomps = sorted(best_decomps, key=lambda d: -_alignment_score(d, culture_languages))
    return best_decomps[:MAX_TIED_DECOMPOSITIONS_PER_POSITION]


def _alignment_score(decomp: list[Any], culture_languages: frozenset[str]) -> int:
    """wyrd-pfoo culture-alignment: count the decomposition's Meanings whose
    primary language is in ``culture_languages`` AND carry ≥1 tag (the ≥1-tag
    gate filters Wiktionary nonsense-senses — Celtic ``ton`` → 'tone', etc.).

    Decomposition elements are ``Meaning | str``; only Meanings carry
    sources/tags/primary_language. The sources+tags duck-type check is the
    str-vs-Meaning discriminator; once past it the elem IS a Meaning (the
    matcher emits no other object kind), so primary_language() is safe.
    """
    score = 0
    for elem in decomp:
        if not (hasattr(elem, "sources") and hasattr(elem, "tags")):
            continue
        if not elem.tags:
            continue
        if elem.primary_language() in culture_languages:
            score += 1
    return score


def _prefer_culture_aligned(
    decompositions: list[list[Any]],
    culture_languages: frozenset[str],
) -> list[list[Any]]:
    """wyrd-pfoo tiebreaker: from a list of decompositions already
    tied on (unaccounted, morpheme_count), prefer those whose Meanings
    are primary-language matched to ``culture_languages`` AND carry
    ≥1 tag. The ≥1 tag gate filters Wiktionary nonsense-senses (Celtic
    ``-ton`` → 'tone', ``Saint-`` → 'greed', ``-bridge`` → card game,
    etc.) which empirically all carry empty tag lists.

    Returns the subset of input decompositions tied at the max
    alignment score. If the max is 0 (no aligned-and-tagged Meanings
    in any parse — common when only one language is represented), the
    full input list is returned unchanged so the existing list-index
    tiebreaker still applies downstream.
    """

    scores = [(_alignment_score(d, culture_languages), d) for d in decompositions]
    best = max(s for s, _ in scores)
    if best == 0:
        return decompositions
    return [d for s, d in scores if s == best]


def _bare_surface(elem: Any) -> str | None:
    """The bare folded surface of a decomposition element, or None for an
    unaccounted ``str``. ``Connective`` carries ``.surface``; a ``Meaning``
    carries ``.usage`` (dash-stripped, lowercased — the morpheme's own surface,
    not its positional decoration)."""
    if isinstance(elem, str):
        return None
    if is_connective(elem):
        return elem.surface.lower()
    return elem.usage.replace("-", "").lower()


def _genitive_credibility(decomp: list[Any], genitive_prior: dict[tuple[str, str], float]) -> float:
    """Soft genitive credibility of a decomposition (wyrd-aicu.9): the product,
    over its genitive-homograph decisions, of how the scholarly prior rates that
    decision. Higher = more credible. A genitive **connective** split to head
    ``S`` scores ``split_probability(L='s'+S, S)``; the **literal** long-form
    morpheme ``L`` (where ``(L, L[1:])`` is an active prior pair) scores
    ``1 - split_probability(L, L[1:])``. Pairs absent from the prior contribute
    1 (neutral) — the tiebreaker only weighs genuine homographs. A SOFT D40
    tiebreaker over already-tied parses, never a hard gate."""
    cred = 1.0
    for i, elem in enumerate(decomp):
        if is_connective(elem) and elem.kind == GENITIVE:
            head = decomp[i + 1] if i + 1 < len(decomp) else None
            short = _bare_surface(head)
            if short:
                pair = (elem.surface.lower() + short, short)
                if pair in genitive_prior:
                    cred *= genitive_prior[pair]
        elif not isinstance(elem, str) and not is_connective(elem):
            long = _bare_surface(elem)
            if long and (long, long[1:]) in genitive_prior:
                cred *= 1.0 - genitive_prior[(long, long[1:])]
    return cred


def _prefer_genitive_credible(
    decompositions: list[list[Any]],
    genitive_prior: dict[tuple[str, str], float],
) -> list[list[Any]]:
    """From parses already tied on (unaccounted, morpheme_count), keep those at
    the max genitive credibility. If every parse scores equally (no genitive
    homograph in play — the common case), the full list passes through so the
    downstream first-meaning tiebreaker still applies. Bit-stable when
    ``genitive_prior`` is empty/None (caller-gated)."""
    scored = [(_genitive_credibility(d, genitive_prior), d) for d in decompositions]
    best = max(s for s, _ in scored)
    keep = [d for s, d in scored if s == best]
    return keep if len(keep) < len(decompositions) else decompositions


def canonical_decomposition(  # noqa: V103 — public single-answer decomposition API; tests pin tiebreakers (PR #498 triage)
    word: str,
    trie: MorphemeTrie,
    *,
    culture_languages: frozenset[str] | None = None,
    connective_inventory: ConnectiveInventory | None = None,
    genitive_prior: dict[tuple[str, str], float] | None = None,
) -> list[Any]:
    """Return ONE 'best' decomposition for ``word`` against the trie —
    the deterministic single-answer variant of
    ``canonical_decompositions``. Use this when the caller wants one
    pick (e.g. proportions training, single-line CLI output, the corpus
    grader); use the plural ``canonical_decompositions`` when ties carry
    information (e.g. the explainer surfaces every reading scholars would).

    Tiebreaker (after the unaccounted-chars + morpheme-count score, the
    culture-alignment and genitive-credibility tiebreakers): position of the
    first matched morpheme — earlier wins. Stable across runs.

    wyrd-p8ve: routes through ``canonical_decompositions`` so it
    inherits the score-pruning walk's memory bounds. The (un, mor)
    tied set is small enough that applying the first-meaning-pos
    tiebreaker via ``min`` over the tied list is O(K) with K << N.
    ``canonical_decompositions`` is guaranteed to return at least one
    decomposition (``[[]]`` for empty input, ``[[word]]`` for no
    matches), so no empty-list guard is needed before ``min``.
    """
    decompositions = canonical_decompositions(
        word,
        trie,
        culture_languages=culture_languages,
        connective_inventory=connective_inventory,
        genitive_prior=genitive_prior,
    )
    return min(decompositions, key=_decomposition_score)


def _is_content_morpheme(elem: Any) -> bool:
    """A content morpheme is a Meaning — not an unaccounted ``str`` and not a
    connective (the genitive ``-s-`` glue is free + non-content, wyrd-aicu.9)."""
    return not isinstance(elem, str) and not is_connective(elem)


def _decomposition_score(decomposition: list[Any]) -> tuple[int, int, int, int]:
    """Score tuple for the canonical picker. Lower wins."""
    unaccounted_chars = sum(len(e) for e in decomposition if isinstance(e, str))
    morpheme_count = sum(1 for e in decomposition if _is_content_morpheme(e))
    # Tiebreaker: prefer earlier first-meaning. Stable for matchers
    # that produce the same DAG — primarily a determinism guard for
    # tests / serialization.
    first_meaning_pos = next(
        (i for i, e in enumerate(decomposition) if _is_content_morpheme(e)),
        len(decomposition),
    )
    # wyrd-5z5j: among parses tied on (unaccounted, morpheme_count,
    # first_meaning_pos), prefer the one whose FIRST morpheme is longer.
    # A longer leading match is the more specific reading — 'Gileston'
    # → 'Giles'(5) + 'ton' vs 'Gil'(3) + 'eston' both score (0, 2, 0),
    # and without this the arbitrary DAG/insertion order could pick the
    # 3-char 'Gil' (Welsh 'splendid') over the intended saint 'Giles'.
    # Negated so longer wins under ``min``. Pure refinement of a
    # previously-arbitrary tie — only reorders genuine ties.
    first_meaning = next(
        (e for e in decomposition if _is_content_morpheme(e)),
        None,
    )
    # Dash-stripped surface length — the morpheme's own length, not its
    # positional decoration. Read ``usage`` directly (not ``str(meaning)``) so
    # the tiebreaker doesn't depend on ``Meaning.__str__`` staying a
    # surface-returning override (Gemini review).
    first_meaning_len = (
        -len(first_meaning.usage.replace("-", "")) if first_meaning is not None else 0
    )
    return (unaccounted_chars, morpheme_count, first_meaning_pos, first_meaning_len)


def iter_morphemes(decomposition: Iterable[Any]) -> Iterable[Any]:  # noqa: V103 — public morpheme iterator; tests pin behavior (PR #498 triage)
    """Yield only the content-morpheme (Meaning) elements of a decomposition,
    dropping unaccounted fragments AND connective glue (the genitive ``-s-`` is
    not a content morpheme — wyrd-aicu.9). Convenience for callers that want the
    matched morpheme sequence (e.g. proportions training, the corpus grader)."""
    for elem in decomposition:
        if _is_content_morpheme(elem):
            yield elem


def count_unaccounted(decomposition: Iterable[Any]) -> int:
    """Total unaccounted character count across a decomposition.
    Mirrors Word.count_unaccounted on the legacy matcher."""
    return sum(len(e) for e in decomposition if isinstance(e, str))
