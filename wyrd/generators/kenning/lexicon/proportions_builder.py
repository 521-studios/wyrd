"""Per-culture proportions construction from decomposed Name lists.

These helpers walk a list of Names that already have ``find_meaning``
applied (i.e. each Name carries its trie-derived decompositions) and
emit the per-culture proportions dict the runtime samples from. The
output shape mirrors the historical ``<culture>_proportions.json``
sidecars — d90t cutover collapsed those sidecars into either the L4
emit pipeline (``runtime_db_export._compute_proportions_inline``) or
the ``wyrd kenning rebuild-proportions`` operator-facing CLI; both call
into this module so the layering goes ``cli/`` and ``lexicon/`` →
``lexicon/proportions_builder`` rather than the other way around.

Lives under ``lexicon/`` per the package-boundary rule: it operates on
domain-modeled data (decomposed Names + Meaning tags) and is shared by
multiple consumers (CLI + L4 emit). The CLI module retains thin
adapters re-exporting these names for backward-compatible imports.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from wyrd.generators.kenning.lexicon.morpheme_surface import _strip_all_dashes
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.proportions import is_structurally_grammatical

# wyrd-pfoo: per-culture preferred-language sets for the matcher's
# culture-aligned tiebreaker. When ``canonical_decompositions`` has
# multiple parses tied on (unaccounted, morpheme_count), it prefers
# parses whose Meanings are primary-language matched to the requested
# culture AND carry ≥1 tag. The "≥1 tag" gate filters Wiktionary
# nonsense-senses (Celtic ``-ton`` → "tone", ``Saint-`` → "greed",
# ``-bridge`` → card game) which empirically have empty tag lists
# (0/19 nonsense Celtic Meanings carry tags in the live Welsh-attested
# pool); see scratch/vector-validation/check-tag-density.py.
#
# Cultures missing from this map fall back to no tiebreaker (existing
# matcher behaviour) — the safe default for pack-overlay and future
# cultures that haven't been classified yet.
#
# Hand-curated. Each culture's language set tracks what its
# place_names.json corpus is empirically composed of (per the lang-mix
# analysis in scratch/vector-validation/check-pool-skew.py).
CULTURE_LANGUAGES: dict[str, frozenset[str]] = {
    "english": frozenset({"old_english", "modern_english", "old_scandinavian", "old_french"}),
    "scottish": frozenset({"celtic_mix", "old_english", "old_scandinavian"}),
    "welsh": frozenset({"celtic_mix"}),
    "irish": frozenset({"celtic_mix"}),
    "breton": frozenset({"celtic_mix", "old_french"}),
}


def _accumulate_attested_languages(name, attested_languages: dict[str, set[str]]) -> None:
    """wyrd-pfoo: walk the picked Meanings for one name and record which
    (usage, primary_language) pairs appeared. The downstream filter is
    per-(usage, primary language), so collisions where two Meanings share
    both admit both — over-admitting one extra Meaning per collision is a
    deliberate tradeoff against shipping per-Meaning UIDs."""
    for word_list in name.words.values():
        for word in word_list:
            for elem in word.word:
                if not isinstance(elem, Meaning):
                    continue
                primary = elem.primary_language()
                if primary is None:
                    continue
                # D45 (wyrd-aicu): key by bare SURFACE, not the dash-marked
                # usage. This is naturally surface-keyed now, retiring the
                # wyrd-c6o1.3 load-time fold-union workaround (the exact-key
                # narrowing leaked welsh `ton` into english through the
                # un-folded dash-variants).
                # ``.strip()`` drops surrounding whitespace (wyrd-an8u) so a
                # dirty ``'Oak- '`` keys the same bare surface as ``'Oak-'``,
                # matching _bare_surface (the usages/single_usages key path).
                # wyrd-n8hw: Unicode-aware fold (drop-in for replace("-","")),
                # matching _bare_surface / _bare_modern_usage so the attested key
                # can't fork on a non-ASCII dash.
                surface = _strip_all_dashes(elem.usage).strip().lower()
                attested_languages.setdefault(surface, set()).add(primary)


def _accumulate_tag_cooccurrence(name, tag_marginal: Counter, tag_cooccurrence: Counter) -> None:
    """Tag co-occurrence: ordered consecutive Meaning pairs across the full
    name (within-word AND between-word). Each pair contributes the cartesian
    product of (left.tags × right.tags) — an etymon often has several tags
    (plant, food, tree), and any of them can legitimately drive co-occurrence."""
    for left_tags, right_tags in ordered_tag_pairs(name):
        for tl in left_tags:
            tag_marginal[tl] += 1
            for tr in right_tags:
                tag_cooccurrence[(tl, tr)] += 1
        for tr in right_tags:
            tag_marginal[tr] += 1


def proportions_from(names) -> dict[str, Any]:
    """Build the per-culture proportions dict from a list of
    perfectly-decomposed Names.

    Each Name must have had ``find_meaning(word_db, reduce=True)``
    called on it; the helpers below pull samples / structure / tag-
    cooccurrence off the resulting decompositions.

    Returned keys: ``{usages, single_usages, structures, tag_marginal,
    tag_cooccurrence, attested_languages, bare_word_positions}`` with
    ``structures`` already passed through the wyrd-zzli grammatical
    filter and tag-pair keys serialized as ``"left|right"`` strings.

    D45 (wyrd-aicu) — morpheme identity is the BARE surface, position an
    explicit axis (never a dash-encoded form):
    * ``usages`` is nested ``{surface: {position: count}}``.
    * ``single_usages`` is ``{surface: count}`` (lone words are bare).
    * ``attested_languages`` is keyed by bare ``surface``.

    ``attested_languages`` (wyrd-pfoo) is the per-Meaning attestation
    half of the culture filter: for each surface, the set of primary
    source-languages of the specific Meanings the matcher used when
    decomposing this culture's place names. The runtime vector path
    consumes this to narrow its eligibility pool from the per-surface
    set to a per-(surface, primary_language) set — closing the
    wrong-sense leakage where ``ton``'s Welsh Meaning ('tone', music)
    or ``saint``'s Celtic Meaning ('greed') could otherwise be picked
    for slots a Welsh culture filter admitted at the surface level.
    """
    part_proportions: Counter = Counter()
    lone_proportions: Counter = Counter()
    struct_proportions: Counter = Counter()
    tag_cooccurrence: Counter = Counter()
    tag_marginal: Counter = Counter()
    attested_languages: dict[str, set[str]] = {}
    bare_word_positions: dict[str, Counter] = {}
    for name in names:
        # D45 (wyrd-aicu): get_samples yields (surface, position) — position is
        # an explicit axis, never folded into a dash-form usage string.
        for surface, position in name.get_samples():
            part_proportions[(surface, position)] += 1
        for u in name.get_lone_samples():
            lone_proportions[u] += 1
        for position, u in name.get_bare_word_positions():
            bare_word_positions.setdefault(position, Counter())[u] += 1
        for structure in name.get_structure():
            struct_proportions[structure] += 1
        _accumulate_attested_languages(name, attested_languages)
        _accumulate_tag_cooccurrence(name, tag_marginal, tag_cooccurrence)
    # D45: ``usages`` is nested ``{surface: {position: count}}`` — the bare
    # morpheme identity at top level, the explicit position axis beneath. The
    # L4 writer flattens to ``(culture, surface, position, weight)`` rows.
    usages_nested: dict[str, dict[str, int]] = {}
    for (surface, position), count in part_proportions.items():
        usages_nested.setdefault(surface, {})[position] = count
    return {
        "usages": {
            surface: dict(sorted(usages_nested[surface].items()))
            for surface in sorted(usages_nested)
        },
        "single_usages": dict(lone_proportions),
        "structures": encode_structs(struct_proportions),
        # Serialize tag-pair keys as "left|right" so they survive a JSON
        # round-trip. Generator side splits on the pipe.
        "tag_cooccurrence": {f"{a}|{b}": v for (a, b), v in tag_cooccurrence.items()},
        "tag_marginal": dict(tag_marginal),
        # wyrd-pfoo: sorted-list values so the JSON round-trip is
        # deterministic across re-builds (set iteration order isn't).
        "attested_languages": {k: sorted(v) for k, v in attested_languages.items()},
        # wyrd-rogd.13: per-word-position counts for bare words of multi-word
        # names ({"pre"|"inner"|"post": {usage: count}}). Emitted RAW — the
        # naked↔structured threshold is applied at LOAD time (env-tunable,
        # WYRD_BARE_POSITION_THRESHOLD) so re-tuning never needs a re-export.
        # Sorted keys both levels for deterministic JSON round-trips.
        "bare_word_positions": {
            position: dict(sorted(bare_word_positions[position].items()))
            for position in sorted(bare_word_positions)
        },
    }


def ordered_tag_pairs(name) -> list[tuple[list[str], list[str]]]:
    """Walk a decomposed Name in order, yielding consecutive (left.tags,
    right.tags) pairs across the entire name (within and between words).

    For 'Bridgwater Gate' this yields:
        (Bridg-.tags, -water.tags)   # within word 1
        (-water.tags, Gate.tags)     # between words

    When a word has multiple equivalent decompositions (e.g. mining
    surfaced a bare-form synthesized usage that competes with the
    dashed reflex), prefer the decomp with the most tag-bearing
    Meanings — that's the one that actually feeds co-occurrence
    statistics rather than emitting empty () pairs.
    """

    def _tag_density(word_obj) -> int:
        if not hasattr(word_obj, "word"):
            return -1
        return sum(1 for c in word_obj.word if isinstance(c, Meaning) and c.tags)

    flat_meanings: list[Meaning] = []
    for word_text in name.name.split(" "):
        word_options = name.words.get(word_text, [])
        if not word_options:
            continue
        chosen = max(word_options, key=_tag_density)
        if not hasattr(chosen, "word"):
            continue
        for chunk in chosen.word:
            if isinstance(chunk, Meaning):
                flat_meanings.append(chunk)

    out: list[tuple[list[str], list[str]]] = []
    for i in range(len(flat_meanings) - 1):
        out.append((list(flat_meanings[i].tags), list(flat_meanings[i + 1].tags)))
    return out


def encode_meaning(qualities) -> dict:
    out: dict = {}
    for quality in qualities:
        # wyrd-vpri: 'bare' is a location (the no-dash single-word
        # position), not a flag. Without it here, a bare quality would
        # fall to the flag branch (out['bare']=True) and leave no
        # 'location' key — then word_to_key's element['location']
        # lookup KeyErrors. Keep in sync with Meaning._set_location.
        if quality in ("pre", "post", "inner", "bare"):
            out["location"] = quality
        else:
            out[quality] = True
    return out


def encode_structs(struct: Counter) -> list:
    """Serialize the structure → count map into the proportions-JSON
    shape, sorted for deterministic output (wyrd-dxu2 review). Counter
    iteration order is insertion-order, so the underlying corpus walk
    determines initial order — but two re-runs against the same corpus
    can iterate in slightly different orders if the matcher's
    decomposition outputs change. Sorting by (-proportion, structure
    key) gives a stable canonical order: most-frequent first, with the
    structure tuple as deterministic tiebreaker.

    wyrd-zzli + wyrd-80ib: drops structures where any word slot is a
    bare 'pre' / 'post' / 'inner' morpheme with no name/saint flag.
    Two shapes both trip the filter:
    - multi-word 'By Green' (zzli): -by + green- split across two
      word slots; 46.7% of English structure weight pre-fix.
    - single-word 'Bridge' (80ib): a 1-word structure whose sole
      word is `-bridge`, rendered alone after dash-strip; 6.7% of
      English 1-word structure weight pre-fix.
    Corresponding runtime defense in
    ``proportions.is_structurally_grammatical``. Real qualifier-word
    names — multi-word (Bishop's Stortford, Great Yarmouth, Saint
    Botolph) and single-word (Bishop's, Great) alike — survive
    because their lead morpheme carries the ``name`` or ``saint``
    flag, which is treated as a distinct position key by
    ``word_to_key`` and the runtime."""
    sorted_items = sorted(struct.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "proportion": value,
            "words": [[encode_meaning(meaning) for meaning in word] for word in key],
        }
        for key, value in sorted_items
        if is_structurally_grammatical(key)
    ]
