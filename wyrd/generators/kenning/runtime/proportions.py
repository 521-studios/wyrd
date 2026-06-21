"""Weighted random selection over learned morpheme proportions.

Ported from Rando (rando/proportions.py) — Python 2 → 3.

Key change beyond syntax: weighted_choice now takes an explicit Random instance
so generation is reproducible from the seed param.
"""

from __future__ import annotations

import logging
import os
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator

from ..era.cells import family_stage_order, language_family
from .meaning import Meaning, _mimic_case
from .structure_allowlist import is_structure_enabled
from .word import WORD_POSITION_KEY_PREFIX, _position_form, word_position_for

# wyrd-lftl: family display order for the SPA col-3 reflex grid — English
# first, then Norse (Danelaw), the Celtic families, French, Latin. Mirrors
# the inspector's etymology-surfacing intent (older/closer strata first).
# Families present on a morpheme but absent here are appended alphabetically.
_GRID_FAMILY_ORDER = ("english", "norse", "brythonic", "goidelic", "norman-french", "latin")

# wyrd-van9: runtime warnings (drift, stale bundle, missing keys) flow
# through the logging module per the project's library-code convention
# — operators / SPA / Lambda hosts can dial them via standard Python
# logging config rather than munging stderr. See runtime/decomposition.py
# for the established pattern.
_logger = logging.getLogger(__name__)

# wyrd-izcr: max struct re-picks in ``NameGenerator.select_via_vector``
# when a qualifier-flagged slot's pool is empty (saint/name slot with
# no eligible morpheme under the current gate). RNG advances across
# attempts so each retry picks a different struct deterministically.
_VECTOR_STRUCT_RETRY_LIMIT = 5

# wyrd-rogd.13: naked↔structured threshold for bare word-sequence placement.
# A bare word with >= threshold total positional observations (summed across
# pre/inner/post in the culture's bare_word_positions stats) is "structured" —
# sampled per its per-word-position weights and eligible only at positions
# it's attested in. Below the threshold it's "naked" — eligible at ANY bare
# slot via the general single_usage distribution. Env-tunable per deploy so
# re-tuning needs no re-export; deliberately NOT a request param and NOT an
# SPA advanced-panel option (operator posture, user 2026-06-11).
_BARE_POSITION_THRESHOLD_ENV = "WYRD_BARE_POSITION_THRESHOLD"
_BARE_POSITION_THRESHOLD_DEFAULT = 3


def bare_position_threshold() -> int:
    """Resolve the wyrd-rogd.13 naked↔structured threshold from the
    environment (``WYRD_BARE_POSITION_THRESHOLD``), defaulting to 3.

    Clamped to >= 1: at 0 every bare word would count as "structured" —
    including words with NO positional observations, which would then be
    eligible at no position at all and silently vanish from the bare
    pools. Malformed values warn + fall back to the default rather than
    failing generation over a typo'd env var.
    """
    raw = os.environ.get(_BARE_POSITION_THRESHOLD_ENV)
    if raw is None or not raw.strip():
        return _BARE_POSITION_THRESHOLD_DEFAULT
    try:
        value = int(raw.strip())
    except ValueError:
        _logger.warning(
            "wyrd-rogd.13: %s=%r is not an integer; using default %d",
            _BARE_POSITION_THRESHOLD_ENV,
            raw,
            _BARE_POSITION_THRESHOLD_DEFAULT,
        )
        return _BARE_POSITION_THRESHOLD_DEFAULT
    if value < 1:
        _logger.warning(
            "wyrd-rogd.13: %s=%d clamped to 1 (0 would strand "
            "never-positionally-observed bare words with no eligible slot)",
            _BARE_POSITION_THRESHOLD_ENV,
            value,
        )
        return 1
    return value


def _flatten_struct_slots(candidate_struct):
    """Flatten a struct (tuple-of-words, each word a tuple-of-keys) into parallel
    position / qualifier / bucket-key lists for the vector primitive. Each key's
    element 0 is the position label (per ``word_to_key``); a ``name`` / ``saint``
    flag becomes the qualifier; the FULL key is the bucket key — preserving the
    ``single`` flag ``word_to_key`` adds so the per-slot frequency lookup hits the
    exact bucket the proportions sample from.

    wyrd-rogd.13: a BARE slot in a multi-word struct additionally derives its
    WORD position from the word's index in the struct (first → ``wp-pre``,
    interior → ``wp-inner``, last → ``wp-post``) and extends its bucket key
    with it, so the frequency lookup hits the per-word-position bare pool
    (``Saint``-style placement stats). The struct already encodes word order;
    no template change — the label is derived here, never stored. Bundles
    without the per-word-position stats fall back to the un-extended key
    (``_resolve_slot_usage_frequency``'s wyrd-rogd.13 fallback), keeping
    legacy behavior bit-stable."""
    positions: list[str] = []
    qualifiers: list[str | None] = []
    bucket_keys: list[tuple] = []
    word_count = len(candidate_struct)
    for word_index, word in enumerate(candidate_struct):
        word_position = word_position_for(word_index, word_count)
        for key in word:
            positions.append(key[0])
            if "name" in key:
                qualifiers.append("name")
            elif "saint" in key:
                qualifiers.append("saint")
            else:
                qualifiers.append(None)
            if key[0] == "bare" and word_position is not None:
                bucket_keys.append((*key, f"{WORD_POSITION_KEY_PREFIX}{word_position}"))
            else:
                bucket_keys.append(key)
    return positions, qualifiers, bucket_keys


def _reconstruct_words(struct, picked):
    """Reconstruct the words list-of-lists from the flat ``picked`` list, matching
    the struct's word-grouping so NewName surfaces the same word boundaries as the
    legacy path. wyrd-g1hj: render each pick at its SLOT-derived position
    (``slot[0]``, the eyjk position-form), NOT the stored dash-variant — NewName
    still strips dashes + applies word-initial case. wyrd-tbke: a permissive-
    fallback None pick passes through as a dropped slot (NewName skips None)."""
    words: list[list[str | None]] = []
    idx = 0
    for word in struct:
        word_keys: list[str | None] = []
        for slot in word:
            pick = picked[idx]
            word_keys.append(_position_form(pick, slot[0]) if pick is not None else None)
            idx += 1
        words.append(word_keys)
    return words


def _reconstruct_picked_ids(struct, picked):
    """wyrd-i4jd: the per-slot ``morpheme_id`` of each selected ``pick``, shaped
    exactly like :func:`_reconstruct_words` (same word-grouping + index walk), so
    it stays index-aligned with ``NewName.name``. This is the IDENTITY the
    generator actually selected — carried so render/grid/highlight resolve the
    morpheme by id instead of re-deriving it from the surface string (which
    string-collides + can land on a different sibling). ``None`` for dropped /
    permissive-fallback slots and for picks lacking a morpheme_id (legacy
    bundles) — every downstream consumer then degrades to the surface path."""
    picked_ids: list[list[str | None]] = []
    idx = 0
    for word in struct:
        word_ids: list[str | None] = []
        for _slot in word:
            pick = picked[idx]
            word_ids.append(getattr(pick, "morpheme_id", None) if pick is not None else None)
            idx += 1
        picked_ids.append(word_ids)
    return picked_ids


class Generator:
    def __init__(self, tag_db, elements):
        self.tag_db = tag_db
        self.elements = elements

    def add_item(self, key, proportion):
        self.elements[key] = proportion


_TRIPLE_LETTER_RUN_RE = re.compile(r"(.)\1{2,}")


def _collapse_triple_letters(text: str) -> str:
    """wyrd-ovsv: collapse runs of 3+ identical letters down to 2.

    Used by ``NewName.__str__`` to clean up morpheme-join artifacts
    like 'Kill' + 'llen' → 'Killlen' (triple l) → 'Killen' (double l).
    Real English / Celtic place names always elide the triple at
    a morpheme boundary (Killarney NOT Killlarney; Llanelli NOT
    Llanellli); this is the orthographic-normalization version of
    that convention.

    Conservative: doesn't touch legitimate doubles (Kill, Ball,
    Cromwell, Newcastle, Llanelli stay intact when standalone) —
    only fires on 3+ runs, which only arise from morpheme-join
    concatenation in this codebase.
    """
    return _TRIPLE_LETTER_RUN_RE.sub(r"\1\1", text)


def _gloss_eligible(usage: str, has_gloss: bool, include_unglossed: bool) -> bool:
    """Gloss-policy predicate shared by the keep-key filter
    (:meth:`MeaningGenerator.keep_keys_for_gloss`) and the vector pool
    filter (:func:`vector_name_select.build_non_position_eligible`) so
    both consumers apply identical generation-pool rules (wyrd-glos).

    * has_gloss → eligible (any length; a glossed single-char like Norse
      ``á`` = river is a real morpheme).
    * unglossed single-char (dash/space-stripped core ≤ 1) → NEVER
      eligible. These are the rando 'Silver' letters + stray ``-m-``/
      ``A-`` fragments; junk regardless of the flag.
    * unglossed multi-char → eligible only when ``include_unglossed``.
    """
    if has_gloss:
        return True
    core = usage.replace("-", "").replace(" ", "")
    if len(core) <= 1:
        return False
    return include_unglossed


# Maps id(meaning_db) → (meaning_db, surface_index). Storing the meaning_db
# reference alongside the index is load-bearing: it keeps the dict ALIVE while
# cached, so CPython can't recycle its id() for a different meaning_db (the
# wyrd-eyjk round-2 staleness hazard), and the ``is`` identity check below
# catches any collision defensively. Bounded so ad-hoc test meaning_dbs don't
# accumulate; also dropped wholesale on bundle reload via
# ``_clear_surface_index_cache`` (wired into the kenning coupled cache-clear).
_SURFACE_INDEX_CACHE: dict[int, tuple[dict, dict[str, list]]] = {}
_SURFACE_INDEX_CACHE_MAX = 32


def _surface_index_for(meaning_db: dict) -> dict[str, list]:
    """wyrd-eyjk/D40: bare-surface (dash-stripped, lowercased) → [Meaning]
    index for a meaning_db. Proportions usages are POSITION-FORMS (`-giles`);
    the morpheme is resolved by its surface since the bundle may store it only
    under another dash-variant. Cached by ``id(meaning_db)`` with an identity
    check so an id() recycled after GC can never return a stale index."""
    key = id(meaning_db)
    entry = _SURFACE_INDEX_CACHE.get(key)
    if entry is not None and entry[0] is meaning_db:
        return entry[1]
    idx: dict[str, list] = {}
    for usage, meanings in meaning_db.items():
        idx.setdefault(usage.lower().replace("-", ""), []).extend(meanings)
    # Production uses one long-lived bundle meaning_db; the cap only matters for
    # the many small ad-hoc meaning_dbs tests build. Evict oldest on overflow.
    if key not in _SURFACE_INDEX_CACHE and len(_SURFACE_INDEX_CACHE) >= _SURFACE_INDEX_CACHE_MAX:
        del _SURFACE_INDEX_CACHE[next(iter(_SURFACE_INDEX_CACHE))]
    _SURFACE_INDEX_CACHE[key] = (meaning_db, idx)
    return idx


def _resolve_surface(meaning_db: dict, usage: str) -> list:
    """Resolve a usage (a position-form like ``-giles``, or any dash-variant)
    to its morpheme's Meanings via the bare-surface index."""
    return _surface_index_for(meaning_db).get(usage.lower().replace("-", ""), [])


def _clear_surface_index_cache() -> None:
    """Drop the bare-surface index cache. Wired into the kenning bundle's
    coupled cache-clear so a meaning_db reload can't read a stale index via
    an id() collision (the cache is keyed by ``id(meaning_db)``)."""
    _SURFACE_INDEX_CACHE.clear()
    _MORPHEME_INDEX_CACHE.clear()


# wyrd-rogd.10 Phase 2: morpheme_id → unified morpheme Meaning. Derived from the
# already-loaded meaning_db (the L4 morpheme table stores the same regrouping on
# disk, but deriving here avoids a second blob read). Cached by id(meaning_db)
# exactly like _SURFACE_INDEX_CACHE, and cleared in lockstep above.
_MORPHEME_INDEX_CACHE: dict[int, tuple[dict, dict[str, object]]] = {}
_MORPHEME_INDEX_CACHE_MAX = 32


def _union_lang(genuine: dict, selfs: dict) -> list:
    """wyrd-phww: union one language's self-seeds onto its genuine reflexes —
    genuine listed first (source preserved); a self-seed is added only for a
    form no genuine reflex already covers."""
    out = dict(genuine)  # genuine first, source preserved
    for form, entry in selfs.items():
        out.setdefault(form, entry)  # self-seed only if no genuine form collides
    return list(out.values())


def _collect_reflex_buckets(ordered: list) -> tuple[dict, dict, dict]:
    """Walk the morpheme group, partitioning each language's era_reflexes into
    genuine (non-``self``, attestation-backed) vs self-seed buckets (so genuine
    can win below), and unioning era_reflex_glosses. First-seen-wins per form.
    Returns ``(nonself, selfseed, glosses)``."""
    nonself: dict[str, dict[str, tuple]] = {}
    selfseed: dict[str, dict[str, tuple]] = {}
    glosses: dict[str, dict[str, str]] = {}
    for m in ordered:
        for lang, entries in (m.era_reflexes or {}).items():
            for form, source in entries:
                bucket = (selfseed if source == "self" else nonself).setdefault(lang, {})
                bucket.setdefault(form, (form, source))
        for lang, form_gloss in (m.era_reflex_glosses or {}).items():
            gbucket = glosses.setdefault(lang, {})
            for form, gloss in form_gloss.items():
                gbucket.setdefault(form, gloss)
    return nonself, selfseed, glosses


def _merge_reflex_buckets(nonself: dict, selfseed: dict, glosses: dict) -> tuple[dict, dict]:
    """Build the merged ``(era_reflexes, era_reflex_glosses)`` from the collected
    buckets: per language (sorted, for determinism) union self-seeds onto
    genuine via :func:`_union_lang`, then keep only glosses whose form survived
    (dropping a language whose every glossed form was a filtered-out self-seed
    rather than leaving an empty map)."""
    merged_reflexes = {
        lang: _union_lang(nonself.get(lang, {}), selfseed.get(lang, {}))
        for lang in sorted(nonself.keys() | selfseed.keys())
    }
    kept = {lang: {form for form, _src in entries} for lang, entries in merged_reflexes.items()}
    merged_glosses: dict[str, dict[str, str]] = {}
    for lang, gloss in glosses.items():
        surviving = {f: g for f, g in gloss.items() if f in kept.get(lang, ())}
        if surviving:
            merged_glosses[lang] = surviving
    return merged_reflexes, merged_glosses


def _merge_morpheme_meanings(meanings: list):
    """Collapse the Meanings sharing one morpheme_id (the connective-form
    fragments — ``-ing``/``-ing-``/``Ing-``) into a single Meaning whose
    ``era_reflexes`` + ``era_reflex_glosses`` are the union across the group, so
    the era-grid reflects the whole morpheme rather than one surface variant.

    **Genuine reflexes first; self-seeds unioned in (wyrd-5olv → wyrd-phww).**
    A ``morpheme_id`` groups not only spelling/position variants of the
    morpheme (``-ing``/``Ing-``) but every usage CONSTRUCTED with it as a head:
    every ``-ton`` town (``-bolton``/``-newton``/...) carries
    ``morpheme_id=old-english:tūn`` because Wiktionary records the town's
    etymology as an inheritance edge FROM ``tūn`` — a word built *with* the
    etymon, recorded as a reflex *of* it (bad upstream data). Each such usage
    self-seeds its OWN surface (``source='self'``, the mook self-seed) into
    ``era_reflexes``.

    So per language the group's genuine — non-``self``: cluster / descent /
    period-form / phonology-rule — reflexes are listed FIRST (their attested
    source preserved), then each self-seed is unioned in only for a form no
    genuine reflex already covers (:func:`_union_lang`). An all-``self``
    connective morpheme is thus retained intact — ``-ing`` has no
    cluster/descent reflexes, so its self-seeded variants (including the
    legitimate ``-ling``) survive. The pre-wyrd-phww rule discarded ALL
    self-seeds whenever any genuine reflex existed, which dropped the generated
    surface (the attested modern toponym form) from the grid; the flood risk
    that rule guarded against (every constructed ``-ton`` town self-seeds its
    whole surface, ~120 for ``tūn``) is instead handled downstream — ``_era_grid``
    narrows the present-day stage to the word's OWN surface, so only the one
    relevant self-seed survives next to the genuine reflexes, never the ~120.

    Deterministic: the group is processed in ``usage`` order, forms dedupe
    first-seen-wins, and languages are visited in sorted order. Returns the
    unchanged base when the result equals it (the common single-family case),
    else a shallow copy with the era fields swapped (era_reflex_for/
    sources_for/gloss_for read these live, no stale cache)."""
    import copy

    ordered = sorted(meanings, key=lambda m: m.usage)
    # A lone meaning is its own merge — keep its reflexes verbatim (self-seeds
    # included). The genuine-wins filter below only fires across a real
    # multi-usage merge, where a usage's self-seed of its own surface is a
    # constructed-compound artifact, not the morpheme's attested spelling.
    if len(ordered) == 1:
        return ordered[0]
    # Collect genuine (non-``self``) vs self-seed reflexes + glosses, then union
    # per language. wyrd-phww: genuine listed first, source preserved; a
    # self-seed is added only for a form no genuine reflex already covers. The
    # pre-fix genuine-WINS rule discarded ALL self-seeds whenever any genuine
    # reflex existed, which dropped the GENERATED SURFACE (the attested modern
    # toponym form — 'Park-'/'Paddock-' for pearroc) from the grid -> "X not in
    # its own reflexes". The flood risk the old rule guarded against (every
    # constructed -ton town self-seeds its whole surface, ~120 for tūn) is
    # handled downstream: _era_grid narrows the PRESENT-DAY stage to the word's
    # OWN surface (generate = the generated surface, explain = the decomposed
    # input surface), so only the one relevant self-seed survives next to the
    # genuine reflexes — never the ~120.
    nonself, selfseed, glosses = _collect_reflex_buckets(ordered)
    merged_reflexes, merged_glosses = _merge_reflex_buckets(nonself, selfseed, glosses)
    # Base = richest member, for the pronunciation/respelling context _era_grid
    # also reads off the Meaning.
    base = max(ordered, key=lambda m: sum(len(v) for v in (m.era_reflexes or {}).values()))
    if merged_reflexes == (base.era_reflexes or {}) and merged_glosses == (
        base.era_reflex_glosses or {}
    ):
        return base
    merged = copy.copy(base)
    merged.era_reflexes = merged_reflexes
    merged.era_reflex_glosses = merged_glosses
    return merged


def _morpheme_index_for(meaning_db: dict) -> dict:
    """``morpheme_id`` → merged morpheme Meaning, cached by id(meaning_db)."""
    key = id(meaning_db)
    entry = _MORPHEME_INDEX_CACHE.get(key)
    if entry is not None and entry[0] is meaning_db:
        return entry[1]
    groups: dict[str, list] = {}
    for meanings in meaning_db.values():
        for m in meanings:
            mid = getattr(m, "morpheme_id", None)
            if mid is not None:
                groups.setdefault(mid, []).append(m)
    idx = {mid: _merge_morpheme_meanings(ms) for mid, ms in groups.items()}
    if key not in _MORPHEME_INDEX_CACHE and len(_MORPHEME_INDEX_CACHE) >= _MORPHEME_INDEX_CACHE_MAX:
        del _MORPHEME_INDEX_CACHE[next(iter(_MORPHEME_INDEX_CACHE))]
    _MORPHEME_INDEX_CACHE[key] = (meaning_db, idx)
    return idx


def _resolve_morpheme(meaning_db: dict, morpheme_id):
    """The unified morpheme Meaning for ``morpheme_id`` (era-grid resolution),
    or None when unset / unknown (bundles predating Phase 1 → caller falls back
    to the per-surface Meaning, so the dash-strip path still works)."""
    if not morpheme_id:
        return None
    return _morpheme_index_for(meaning_db).get(morpheme_id)


def _location_from_form(usage: str) -> str:
    """LEGACY (pre-D45) decode: the position a dash-encoded position-form usage
    carries, from its dashes — ``-x-`` inner, ``x-`` pre, ``-x`` post, bare.
    Only the legacy branch of :func:`_iter_part_proportions` calls this (old v2
    bundles + test fixtures built in the flat ``{dashed: weight}`` shape); the
    D45 path carries position as an explicit axis, not in the string."""
    lead = usage.startswith("-")
    trail = usage.endswith("-")
    if lead and trail:
        return "inner"
    if trail:
        return "pre"
    if lead:
        return "post"
    return "bare"


def _iter_part_proportions(
    proportions: dict[str, int | dict[str, int]],
) -> Iterator[tuple[str, str, str, int]]:
    """Normalize the ``usages`` part pool into
    ``(surface_key, position, item_form, weight)`` tuples, accepting both:

    * **D45 (wyrd-aicu)** nested ``{surface: {position: weight}}`` — position is
      an explicit axis; ``item_form`` is the bare surface (stored case kept so
      the render's initial-slot front-cap preserves internal name caps), and
      ``surface_key`` is its lowercase fold for the bare-surface index.
    * **Legacy** flat ``{dashed_usage: weight}`` — old v2 L4 bundles and test
      fixtures; position is decoded from the dashes (:func:`_location_from_form`)
      and ``item_form`` is the dashed string (the render strips it anyway).

    The shape is detected per-entry by the value type (dict → nested).
    """
    for key, value in proportions.items():
        if isinstance(value, dict):
            surface_key = key.lower().replace("-", "")
            for position, weight in value.items():
                yield surface_key, position, key, weight
        else:
            yield key.lower().replace("-", ""), _location_from_form(key), key, value


class MeaningGenerator:
    def __init__(self, meaning_db, tag_db, proportions):
        self.meaning_db = meaning_db
        self.tag_db = tag_db
        self.generators: dict[tuple, Generator] = {}
        # Gloss-policy keep cache: include_unglossed bool → frozenset of
        # allowed usages (or None for the full-coverage / no-filter fast
        # path). Only two keys ever; same immutable-meaning_db contract.
        self._gloss_keep_cache: dict[bool, frozenset[str] | None] = {}
        self.load_parts(proportions)

    def _surface_index(self) -> dict[str, list]:
        # wyrd-eyjk/D40: bare-surface → [Meaning] index for resolving the
        # position-form usages in the proportions. Shared module-level cache.
        return _surface_index_for(self.meaning_db)

    def load_parts(self, proportions, *addkeys):
        # D45 (wyrd-aicu): the ``usages`` part pool carries each morpheme as
        # ``(surface, position)`` — bare identity + explicit position axis — not
        # a dash-encoded position-form. The morpheme is resolved by its bare
        # SURFACE (the bundle stores it under exactly one bare key now), bucketed
        # by the EXPLICIT position plus the morpheme's name/saint flag. Position
        # is never read off ``Meaning.location`` nor decoded from a stored
        # dash-shape. ``_iter_part_proportions`` also tolerates the legacy flat
        # ``{dashed: weight}`` shape (old v2 bundles + flat test fixtures).
        #
        # wyrd-van9: a surface that isn't in meaning_db at all is skipped +
        # counted (bundle/proportions drift) rather than crashing.
        index = self._surface_index()
        missing_count = 0
        for surface, position, usage, proportion in _iter_part_proportions(proportions):
            meanings = index.get(surface)
            if meanings is None:
                missing_count += 1
                continue
            # Flags (name/saint) come from the morpheme; position is explicit.
            # wyrd-57d8: this per-bucket frequency-weight pool (snapshotted into
            # `usage_frequency_by_bucket` and threaded into the live vector
            # scorer) must apply the SAME base-pool class exclusions as the
            # scored eligibility pool — so route through `is_base_pool_excluded`
            # rather than re-spelling the predicate inline. It drops pure-proper-
            # noun SAINT subjects (`Mary`/`Giles` from the St-dedication list)
            # and personal GIVEN names (`John`/`Edmund`) — both dangle as bare
            # personal names in plain generation — synthesized Norman manorial
            # subjects (`Malet`; they reach output only via `manorial_affix`),
            # and connector morphemes (`cum`/`le`/`juxta`, which need a
            # complement). NOT excluded: a place element merely co-tagged with a
            # name (`stān`+`Stan` — not pure) or a real FAMILY-name etymon
            # (`Smith`; forms legitimate toponyms + is what the synthetic test
            # fixtures use), so neither realism nor the fixtures are disturbed.
            keys = {(position, *m.key()[1:]) for m in meanings if not m.is_base_pool_excluded()}
            for key in keys:
                if addkeys:
                    key = (*key, *addkeys)
                gen = self.generators.setdefault(key, Generator(self.tag_db, {}))
                gen.add_item(usage, proportion)
        if missing_count:
            _logger.warning(
                "wyrd-van9: MeaningGenerator: %d proportions usages have no "
                "Meaning in the bundle; skipping. Re-emit the bundle (and "
                "re-rebuild proportions against it) to clear the drift.",
                missing_count,
            )

    def load_bare_word_positions(
        self,
        bare_word_positions: dict[str, dict[str, int]],
        single_usages: dict[str, int],
        threshold: int,
    ) -> None:
        """wyrd-rogd.13: build the per-WORD-position bare pools.

        ``bare_word_positions`` is the raw per-position tally
        (``{position: {usage: count}}`` over pre/inner/post — the word's
        slot within the NAME, not the D39 within-word morpheme position).
        A bare word whose total positional observations reach ``threshold``
        is "structured": it enters only the word-position buckets it's
        attested in, at its per-position weight. Every other bare word is
        "naked": it enters ALL three word-position buckets at its general
        ``single_usages`` weight, so thin-evidence words keep today's
        any-slot behavior.

        Buckets are registered via :meth:`load_parts` with the extra
        ``wp-<position>`` addkey, producing keys like
        ``('bare', 'single', 'wp-pre')`` / ``('bare', 'saint', 'single',
        'wp-pre')`` — exactly the keys ``_flatten_struct_slots`` derives
        for bare slots of multi-word structs. No-op when the stats are
        absent (legacy bundle): the wp-extended keys then miss and the
        vector path falls back to the general ``('bare', …, 'single')``
        bucket, bit-stable with pre-rogd.13 behavior.

        Identity granularity (D40): the naked↔structured decision is per
        bare SURFACE. The build-side tally already records surfaces, but
        operator-supplied proportions JSON may carry stored-form keys
        (``Ghyll`` / ``ghyll`` case twins) — totals are surface-aggregated
        here so split keys can't dilute the threshold. The naked pool
        keeps ``single_usages``' form keys as-is: the slot scorer's
        ``_apply_per_usage_frequency`` sums frequency by surface anyway,
        so case twins compose exactly as they do in the general bucket.
        """
        if not bare_word_positions:
            return

        def _surface(form: str) -> str:
            return form.lower().replace("-", "")

        # Totals sum ONLY the three known positions — same set the pool loop
        # below builds. Summing every key would let a rogue position label in
        # hand-edited data push a word over the threshold ("structured")
        # while its observations feed no pool, stranding it with no eligible
        # slot at all (type-design review, round 1).
        totals: Counter = Counter()
        for position in ("pre", "inner", "post"):
            for usage, count in (bare_word_positions.get(position) or {}).items():
                totals[_surface(usage)] += count
        structured = {surface for surface, total in totals.items() if total >= threshold}
        naked = {u: w for u, w in single_usages.items() if _surface(u) not in structured}
        for position in ("pre", "inner", "post"):
            pool = dict(naked)
            for usage, count in (bare_word_positions.get(position) or {}).items():
                if _surface(usage) in structured:
                    pool[usage] = count
            if pool:
                self.load_parts(pool, "single", f"{WORD_POSITION_KEY_PREFIX}{position}")

    def keep_keys_for_gloss(self, include_unglossed: bool) -> frozenset[str] | None:
        """Resolve the gloss policy to the set of usages eligible for
        GENERATION (wyrd-glos). Glossing is a project pillar — the
        etymology line is the load-bearing feature — so by default the
        generator only draws usages it can explain (``include_unglossed
        =False``, the historical rando-era behavior). The operator opts
        into unglossed morphemes via the SPA/CLI flag, accepting names
        whose components show ``(unglossed)``.

        A usage is gloss-eligible iff it has at least one glossed Meaning,
        OR (``include_unglossed`` AND it is not a single-character
        fragment). Single-character UNGLOSSED usages (the ``t``/``n``/
        ``r`` rando 'Silver' letters, stray ``-m-``/``A-`` fragments) are
        ALWAYS dropped — they are never valid place-name elements and
        carry no meaning to show. A single-char GLOSSED usage (Norse
        ``á`` = river) survives on the has-gloss branch.

        Returns None when the keep-set covers every usage (no-filter fast
        path), else the frozenset; cached per include_unglossed value. Same
        USAGE-level granularity caveat — a usage with one glossed and one
        unglossed sense stays in the pool, and the downstream
        ``_pick_surface`` may surface the unglossed sense; tightening to
        sense level needs ``_render_substitutions`` reworked.
        """
        if include_unglossed in self._gloss_keep_cache:
            return self._gloss_keep_cache[include_unglossed]
        allowed: frozenset[str] | None = frozenset(
            usage.lower().replace("-", "")
            for usage, meanings in self.meaning_db.items()
            if _gloss_eligible(usage, any(m.meanings for m in meanings), include_unglossed)
        )
        if len(allowed) == len(self._surface_index()):
            allowed = None
        self._gloss_keep_cache[include_unglossed] = allowed
        return allowed


def _is_ungrammatical_word_template(word_key: tuple) -> bool:
    """wyrd-zzli + wyrd-80ib: returns True when a word slot is a single
    standalone 'pre' / 'post' / 'inner' (attachment) morpheme with no
    ``name`` / ``saint`` qualifier. The runtime renders such a slot as a
    standalone word with the dash stripped — producing two distinct bug
    shapes depending on the enclosing structure:

    NB (wyrd-vpri): "standalone" here is the bug; do NOT confuse it with
    the ``bare`` LOCATION. A no-dash key has location ``bare`` (∉
    pre/post/inner), so a single bare-LOCATION word is grammatical and
    kept — only the dash-marked attachment morphemes (pre/post/inner)
    are ungrammatical standing alone.

    - Multi-word `By Green` (wyrd-zzli): a 2-word structure where each
      word is a bare attachment morpheme (-by + green-). The two
      attachment-only morphemes get split across word boundaries.
    - Single-word `Bridge` (wyrd-80ib): a 1-word structure whose sole
      word is a bare attachment morpheme (-bridge). The morpheme
      renders alone with the dash stripped.

    Both shapes share this predicate; the caller
    :func:`is_structurally_grammatical` applies it to every word in
    the structure regardless of how many words the structure has.

    A bare pre/post morpheme is designed to ATTACH to another morpheme
    inside the same word (denoted by leading or trailing dash in the
    canonical_form). Real qualifier-word place-names — Bishop's
    Stortford, Great Yarmouth, Saint Botolph — survive because the
    qualifier word's lexical entry carries the ``name`` or ``saint``
    flag, which is treated as a distinct position key.

    A word_key here is a tuple of position tuples; ``key[0]`` is the
    location label and any subsequent elements are flags (``name``,
    ``saint``, ``single``). The ``single`` flag is added by
    ``word_to_key`` for single-element words but doesn't carry
    grammatical meaning — what matters is whether ``name`` or
    ``saint`` is present.
    """
    if len(word_key) != 1:
        return False
    only = word_key[0]
    if not only:
        return False
    location = only[0]
    if location not in ("pre", "post", "inner"):
        # 'inner' added to the filter alongside pre/post (code-reviewer
        # P3, round 1): inner morphemes have dashes on BOTH sides, so a
        # standalone bare-inner word is arguably MORE ungrammatical than
        # the bare-pre/post case. No current bundle has bare-inner
        # multi-word structures, but cheaper to future-proof than to
        # re-debug if mining surfaces one later.
        return False
    flags = set(only[1:])
    return "name" not in flags and "saint" not in flags


def is_structurally_grammatical(struct_key: tuple) -> bool:
    """wyrd-zzli: predicate for a full structure (tuple of word_keys).

    A structure is grammatical when no word inside it is a bare-pre /
    bare-post / bare-inner standalone (see
    :func:`_is_ungrammatical_word_template`). Applies to both multi-
    word structures (the original `By Green` shape) and single-word
    structures (wyrd-80ib): a 1-word structure whose sole word is a
    bare attachment morpheme renders that morpheme alone with the
    dash stripped, producing single-word output like `Bridge` /
    `Green` (from suffix-only morphemes `-bridge` / `-green`) or
    `South` / `High` (from prefix-only morphemes `South-` / `High-`).
    """
    return not any(_is_ungrammatical_word_template(w) for w in struct_key)


def _era_reflex_raw_choice(meanings, era_render_language: str | None) -> str | None:
    """wyrd-3vju.1: the RAW ``era_reflexes`` entry (``*`` reconstruction marker
    PRESERVED) that the era render selects for ``meanings`` at
    ``era_render_language`` — the same first-sense / attested-over-reconstructed
    choice as :func:`_era_form_for_meanings`, but unstripped.

    The grid cell id is ``lang:<raw form>`` (the ``*`` is kept in the id; see
    :func:`_era_stage`), so returning the raw form lets a caller build the EXACT
    cell id of the reflex the generator rendered — FORWARD — instead of folding
    the stripped, case-projected surface back into the grid (which mismatches on
    macrons / reconstruction markers). None when no sense carries a reflex."""
    for meaning in meanings:
        forms = meaning.era_reflex_for(era_render_language)
        if not forms:
            continue
        attested = [f for f in forms if not f.startswith("*")]
        return attested[0] if attested else forms[0]
    return None


def _era_form_for_meanings(meanings, era_render_language: str | None) -> str | None:
    """wyrd-6c8x: the era-appropriate form for a usage's resolved senses, or
    None when none of them carries a reflex at ``era_render_language``.

    Picks the first sense that has a reflex, and from it prefers an ATTESTED
    form: the reflex list is sorted and the ``*`` reconstruction marker sorts
    before letters, so a naive ``forms[0]`` is biased toward reconstructed
    (unattested) forms — we take the first non-starred form, falling back to the
    marker-stripped first form only when every reflex is reconstructed (still a
    plausible period spelling). Deterministic: no rng draw.

    Shares its selection with :func:`_era_reflex_raw_choice` (which keeps the
    ``*`` for cell-id construction); this returns the marker-stripped surface."""
    raw = _era_reflex_raw_choice(meanings, era_render_language)
    return raw.lstrip("*") if raw is not None else None


def _native_form_for_morpheme_id(morpheme_id) -> str | None:
    """wyrd-i4jd: the NATIVE surface encoded in a ``morpheme_id``
    (``old-english:smiþþe`` → ``smiþþe``), or None when the id is unset /
    malformed / has a dash-only canonical. The single source of truth for
    'morpheme_id → native form' shared by the surface-list and id-first paths."""
    src, sep, canon = (morpheme_id or "").partition(":")
    return canon if (sep and src and canon.strip("-")) else None


def _native_form_for_meanings(meanings) -> str | None:
    """wyrd-24s6 (D41): the morpheme's NATIVE surface — its source-language
    headword, taken from the ``morpheme_id`` canonical (``old-english:smiþþe`` →
    ``smiþþe``). This is the form "as selected": for a morpheme sourced in a
    historical language it is that language's form; for a modern-sourced morpheme
    it equals the modern form. Returns None when no resolved sense carries a
    ``morpheme_id`` — the caller then falls back to the modern usage (the ~orphan
    / synthesized morphemes), mirroring the era-render fallback convention."""
    for meaning in meanings:
        form = _native_form_for_morpheme_id(getattr(meaning, "morpheme_id", None))
        if form is not None:
            return form
    return None


def _era_pronunciation(renderings, era_language, era_form, meaning):
    """wyrd-mf2u: the breakdown's pronunciation for an ERA form — its OWN
    pronunciation, not an arbitrary cluster form's.

    Prefers the rich per-form rendering (IPA + reader_pronunciation) the bundle
    carries for ``era_form`` in ``era_language``; falls back to the rule-based
    ``respelling`` (always available for OE / ME / Welsh / …). ``era_language``
    is hyphenated ('old-english'); the renderings map is keyed underscored
    ('old_english'). Returns a sparse dict ({ipa?, reader_pronunciation?,
    original_script?}) or None when neither source has anything."""
    bare = (era_form or "").replace("-", "")
    if not bare:
        return None
    lang_key = era_language.replace("-", "_")
    lang_renders = (renderings or {}).get(lang_key, {})
    for form, data in lang_renders.items():
        # tolerant match: same form modulo case, or the form IS this entry's
        # accented original_script (so 'stan' finds the 'stān (/stɑːn/)' row).
        if form.lower() == bare.lower() or (
            data.get("original_script", "").lower() == bare.lower()
        ):
            pron = {
                k: data[k]
                for k in ("ipa", "reader_pronunciation", "original_script")
                if data.get(k)
            }
            if pron:
                return pron
    respelled = meaning.respelling_for(bare, era_language) if meaning is not None else None
    return {"reader_pronunciation": respelled} if respelled else None


def _era_form_cell(meaning, renderings, lang, form, sources, glosses):
    """wyrd-lftl: one grid cell — a reflex ``form`` + its ``source`` + its own
    pronunciation (rich IPA where the bundle carries it for the form, rule
    respelling else; reader present where available, IPA sparse). Pronunciation
    keys are omitted when absent so the payload stays sparse (no null noise).

    wyrd-rogd.1: also carries the reflex's own ``gloss`` (sparse) so the SPA
    can flag semantic drift when a swapped cognate no longer means what the
    morpheme does.

    ``form`` is guaranteed a key in ``sources``: both come from the same
    ``era_reflexes[lang]`` ``(form, source)`` tuples, so the ``.get`` default
    never fires under a well-formed bundle — it's belt-and-suspenders.

    wyrd-rogd.6: the scholarly ``*`` reconstructed/unattested-form marker is
    stripped from the EMITTED surface + respelled away — it's a citation
    convention, not part of the surface, and reads as a glitch in the SPA grid
    and (on swap) the name. We keep the ORIGINAL ``form`` for the source/gloss
    lookups (those dicts are keyed by it) but display + respell the clean form."""
    cell = {"form": form.replace("*", ""), "source": sources.get(form, "cluster")}
    # Resolve pronunciation against the ORIGINAL starred form — the bundle's
    # renderings key rich IPA under it (canonical_form carries the *), so a
    # clean-form lookup would miss and drop to respelling. Then strip the * from
    # the outputs: the rule respeller has no */-dropping rule, so respell("*ur")
    # would otherwise leak the marker into reader_pronunciation.
    pron = _era_pronunciation(renderings, lang, form, meaning) or {}
    reader = pron.get("reader_pronunciation")
    if reader:
        cell["reader_pronunciation"] = reader.replace("*", "")
    ipa = pron.get("ipa")
    if ipa:
        cell["ipa"] = ipa.replace("*", "")
    gloss = glosses.get(form)
    if gloss:
        cell["gloss"] = gloss
    return cell


def _era_stage(meaning, renderings, lang):
    """wyrd-lftl: one stage column — every reflex form for ``lang`` as cells,
    or None when the language has no forms (so the caller drops the empty
    column).

    wyrd-rogd.7 / wyrd-i4jd: each cell carries a stable ``id`` — ``lang:<form>``
    using the RAW reflex form (``*`` marker kept so the reconstructed ``*ur`` and
    attested ``ur`` stay distinct). Forms dedupe within a stage (first-seen-wins
    in the merge), so ``(lang, form)`` uniquely identifies the cell, and it is
    CONTENT-stable — unaffected by reflex-list reordering / bundle re-emit (the
    old positional ``lang:index`` shifted when the list moved). The SPA tracks
    the selected + active variant by this id rather than by surface-fold, which
    collides for forms that fold equal (bǣre/bære) and lit up two cells at once."""
    sources = meaning.era_reflex_sources_for(lang)
    glosses = meaning.era_reflex_gloss_for(lang)
    forms = []
    for form in meaning.era_reflex_for(lang):
        cell = _era_form_cell(meaning, renderings, lang, form, sources, glosses)
        cell["id"] = f"{lang}:{form}"
        forms.append(cell)
    return {"language": lang, "forms": forms} if forms else None


def _ordered_stage_langs(family, present):
    """wyrd-lftl: the ``present`` reflex languages of ``family`` in canonical
    stage order (oldest→newest), with any present-but-unmapped tag appended —
    defensive; era_reflexes tags are canonical so this tail is normally empty."""
    stage_order = family_stage_order(family)
    return [lang for lang in stage_order if lang in present] + [
        lang for lang in present if lang not in stage_order
    ]


def _grid_surface_key(form: str) -> str:
    """Normalized key for matching a word's own surface to a present-day era
    cell form: case-insensitive, affix dashes + whitespace ignored ('Park-' ==
    'park', '-ton' == 'ton')."""
    return form.strip().strip("-").strip().casefold()


def _grid_match_key(form: str) -> str:
    """Accent-insensitive variant of :func:`_grid_surface_key` — additionally
    strips combining marks (NFD) and the reconstruction ``*`` marker, mirroring
    the SPA's ``accentFold``. Used when reusing / marking grid cells so a lossy
    ASCII surface ('ra') binds to its genuine accented reflex cell ('rá')
    instead of spawning a visual duplicate."""
    decomposed = unicodedata.normalize("NFD", _grid_surface_key(form).replace("*", ""))
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _iter_grid_cells(grid: list[dict], lang: str | None = None) -> Iterator[dict]:
    """Yield every form cell in ``grid`` — optionally only ``lang``'s stage. The
    single shared grid walk for the cell-matching helpers."""
    for section in grid:
        for stage in section.get("stages", []):
            if lang is not None and stage.get("language") != lang:
                continue
            yield from stage.get("forms", [])


def _ensure_cell(
    grid: list[dict], lang: str | None, surface: str | None, source: str
) -> str | None:
    """wyrd-3vju.3: GUARANTEE a clickable cell for ``surface`` in ``lang``'s stage.
    The era grid IS the SPA's swap menu, so the live/identity form MUST be in it
    (with a stable id) or the user can't swap to it AND back — the long-standing UI
    defect. Returns the cell id, or None when grid/lang/surface is missing or
    ``lang`` has no era family (nothing to inject into — the SPA keeps its legacy
    fold fallback for that morpheme).

    If a cell in ``lang``'s stage already matches ``surface`` (exact, then
    case/dash fold, then accent-insensitive fold), return its id (no duplicate).
    Otherwise INJECT ``{id, form, source}`` into that stage, creating the stage /
    family section in canonical order if absent.
    ``source='usage'`` for the identity anchor (the morpheme's canonical usage when
    no reflex carries it); ``source='self'`` for a rendered reflex the sparse
    era_reflexes omit. Needed because era_reflexes can OMIT or CORRUPT the form:
    the modern 'gold' cell holds 'gowth', and '-ston' renders 'ston' while its
    modern cells are 'stone/steen/sten'."""
    if not grid or not lang or not surface:
        return None
    form = surface.strip().strip("-").strip()
    if not form:
        return None
    existing = _cell_id_in_stage(grid, lang, form)
    if existing is not None:
        return existing  # surface already equals a reflex — no duplicate
    if language_family(lang) is None:
        # Unmapped language (proto-*, untracked): _era_grid never builds a section
        # for it, so don't fabricate a {"family": None} one here — fail closed.
        return None
    cid = f"{lang}:{form}"
    _grid_stage_for(grid, lang)["forms"].append({"id": cid, "form": form, "source": source})
    return cid


def _mark_usage_cells(grid: list[dict], usage: str | None) -> None:
    """wyrd-3vju.3: mark EVERY cell whose form folds to the canonical ``usage`` (the
    morpheme's IDENTITY) with ``is_usage=True`` — across all eras, not just the
    anchor, and accent-insensitively (the '-ra' identity marks its 'rá' reflex).
    So when the identity also surfaces as a genuine reflex (modern 'ton' AND
    middle-english 'ton' both folding to the '-ton' identity), every matching
    cell gets the SPA's identity rendering. Independent of ``active_form_id`` (the
    live reflex highlight): a cell can be the identity, the live form, or both."""
    key = _grid_match_key(usage or "")
    if not key:
        return
    for cell in _iter_grid_cells(grid):
        if _grid_match_key(cell.get("form", "")) == key:
            cell["is_usage"] = True


def _present_day_lang(grid: list[dict], mid: str | None) -> str | None:
    """The present-day stage language of the morpheme's family (from ``mid``'s
    source language, else the grid's primary section) — where the identity/usage
    anchor lives (modern-english for the english family)."""
    src = (mid or "").partition(":")[0]
    fam = language_family(src) if src else None
    section = next((s for s in grid if s.get("family") == fam), None) or (grid[0] if grid else None)
    if section is None:
        return None
    order = family_stage_order(section.get("family", ""))
    return order[-1] if order else None


def _cell_id_in_stage(grid: list[dict], lang: str, surface: str) -> str | None:
    """The id of a cell in ``lang``'s stage matching ``surface`` — exact form
    first, then case/dash fold, then accent-insensitive fold (so a lossy ASCII
    surface reuses its genuine accented reflex cell rather than missing). None
    when nothing matches. Scoped to the language stage so a same-spelled
    earlier-era homograph can't match (wyrd-3vju.2)."""
    folded = _grid_surface_key(surface)
    accentless = _grid_match_key(surface)
    fold_hit = accent_hit = None
    for cell in _iter_grid_cells(grid, lang):
        form = cell.get("form", "")
        if form == surface:
            return cell.get("id")
        if fold_hit is None and _grid_surface_key(form) == folded:
            fold_hit = cell.get("id")
        elif accent_hit is None and _grid_match_key(form) == accentless:
            accent_hit = cell.get("id")
    return fold_hit or accent_hit


def _grid_section_sort_key(fam: str) -> tuple[int, int, str]:
    """Section ordering key matching :func:`_era_grid`'s build convention:
    ``_GRID_FAMILY_ORDER`` families first (in that order), then unknown
    families alphabetically."""
    if fam in _GRID_FAMILY_ORDER:
        return (0, _GRID_FAMILY_ORDER.index(fam), "")
    return (1, 0, fam)


def _grid_stage_for(grid: list[dict], lang: str) -> dict:
    """Find — or create, in canonical oldest→newest stage order — the
    ``{language: lang}`` stage of its family section, returning the stage to append
    cells to. Creates the family section too if the grid lacks it — inserted at its
    canonical ``_GRID_FAMILY_ORDER`` position so injection matches the build-path
    section ordering. ``lang`` must map to an era family (the :func:`_ensure_cell`
    caller guards this)."""
    fam = language_family(lang)
    section = next((s for s in grid if s.get("family") == fam), None)
    if section is None:
        section = {"family": fam, "stages": []}
        key = _grid_section_sort_key(fam)
        at = next(
            (i for i, s in enumerate(grid) if _grid_section_sort_key(s.get("family") or "") > key),
            len(grid),
        )
        grid.insert(at, section)
    stage = next((st for st in section["stages"] if st.get("language") == lang), None)
    if stage is not None:
        return stage
    stage = {"language": lang, "forms": []}
    order = family_stage_order(fam)
    insert_at = len(section["stages"])
    if lang in order:
        for i, st in enumerate(section["stages"]):
            other = st.get("language")
            # Insert before a later canonical stage OR an unmapped tail stage —
            # _ordered_stage_langs puts unmapped tags at the end, so a mapped
            # stage always sorts before them.
            if other not in order or order.index(other) > order.index(lang):
                insert_at = i
                break
    section["stages"].insert(insert_at, stage)
    return stage


def _grid_has_cell_id(grid: list[dict], cell_id: str) -> bool:
    """wyrd-3vju.1: True when ``cell_id`` is an exact cell id in ``grid``. Used to
    confirm the FORWARD-computed active-form id (the cell of the reflex the
    generator actually rendered) is present before stamping it — an exact-id
    compare, no surface fold, so it can't mis-bind a macron / reconstruction
    variant the way the since-removed legacy fold search did."""
    if not grid or not cell_id:
        return False
    return any(cell.get("id") == cell_id for cell in _iter_grid_cells(grid))


def _era_grid(meaning, renderings, own_surface=None):
    """wyrd-lftl: per-morpheme family × era reflex grid for the SPA col-3
    inspector.

    wyrd-phww: ``own_surface`` is the word's OWN surface for this instance —
    the generated surface (generate) or the decomposed input surface (explain).
    When given, the PRESENT-DAY stage of each family keeps its genuine reflexes
    plus ONLY the self-seed matching ``own_surface`` — so the surface the name
    actually used appears ('Park' next to etymological 'parrock') without the
    flood of every sibling self-seed (~120 constructed -ton towns) the merge now
    unions in. Genuine reflexes and all non-present-day stages are untouched (the
    morpheme's own-form self-seed in its OWN era stage, e.g. OE 'pearroc', is
    NOT a present-day cell and survives). ``None`` keeps every self-seed (the
    full union — no caller relies on this today but it's the safe default).

    Returns a list of family sections::

        [{"family": "english",
          "stages": [{"language": "old-english",
                      "forms": [{"form": "tūn", "source": "cluster",
                                 "reader_pronunciation": "TOON", "ipa": "/tuːn/",
                                 "gloss": "farm, enclosure"},
                                ...]},
                     {"language": "middle-english", "forms": [...]},
                     ...]},
         {"family": "norse", "stages": [...]}, ...]

    The data is the ranked etymon's ``era_reflexes`` (cluster/descent/
    period-form/phonology-rule cluster mates, computed at bundle build).
    Stages are the DISTINCT canonical language tags per family in oldest→
    newest order (``family_stage_order`` — collapsing era cells that share
    a tag), so the grid never renders duplicate columns. Per-form
    pronunciation reuses ``_era_pronunciation`` (rich IPA where the bundle
    carries it for the form, rule respelling else; reader always, IPA
    sparse). ``source`` is carried through so the SPA can mark/hide the
    inferred ``phonology-rule:v1`` tier.

    Returns ``[]`` when the morpheme has no era_reflexes (the common case
    today — coverage is sparse, wyrd-32t1). Per-reflex GLOSS is intentionally
    absent: the runtime bundle emits ``{form, source}`` only, so drift-gloss
    needs a bundle re-emit (wyrd-rogd.1)."""
    if meaning is None:
        return []
    # A non-None meaning here is contractually a Meaning, which always defines
    # era_reflexes + the three accessors — so plain attribute access (fail-loud
    # on a genuinely malformed object) over a getattr guard that would only
    # mask the missing-accessors case one loop later.
    reflexes = meaning.era_reflexes or {}
    if not reflexes:
        return []
    # Bucket the morpheme's reflex languages by family. era_reflexes keys are
    # canonical language tags (hyphenated: 'old-english'); language_family maps
    # them to the era family. Skip tags with no family (proto-langs, untracked).
    by_family: dict[str, list[str]] = {}
    for lang in reflexes:
        family = language_family(lang)
        if family is not None:
            by_family.setdefault(family, []).append(lang)
    if not by_family:
        return []
    ordered_families = [f for f in _GRID_FAMILY_ORDER if f in by_family] + sorted(
        f for f in by_family if f not in _GRID_FAMILY_ORDER
    )
    sections: list[dict] = []
    for family in ordered_families:
        langs = _ordered_stage_langs(family, by_family[family])
        stages = [
            stage for lang in langs if (stage := _era_stage(meaning, renderings, lang)) is not None
        ]
        if stages:
            sections.append({"family": family, "stages": stages})
    if own_surface is not None:
        own_form = (getattr(meaning, "morpheme_id", None) or "").partition(":")[2]
        sections = _narrow_present_day_self_seeds(sections, own_surface, own_form)
    return sections


def _narrow_present_day_self_seeds(
    sections: list[dict], own_surface: str, own_form: str
) -> list[dict]:
    """wyrd-phww: in each family's PRESENT-DAY stage, drop ``source='self'`` cells
    EXCEPT the one(s) matching ``own_surface`` (the name's generated/input
    surface) or ``own_form`` (the morpheme's OWN canonical form). Every genuine
    cell is kept. Non-present-day stages are untouched, so the own-form self-seed
    in a separate older stage (OE ``pearroc``) survives.

    Keeping ``own_form`` matters when a family's newest stage IS the morpheme's
    own-language stage — e.g. ``norse``/``latin`` have no present-day cell, so
    ``family_stage_order(family)[-1]`` is the historical stage that ALSO holds
    the own-form self-seed (ON ``hryggr``); without this it would be dropped.
    Stages/sections emptied by the filter are dropped."""
    keep_keys = {_grid_surface_key(own_surface), _grid_surface_key(own_form)}
    out: list[dict] = []
    for section in sections:
        stage_order = family_stage_order(section["family"])
        present_day_lang = stage_order[-1] if stage_order else None
        stages = []
        for stage in section["stages"]:
            if stage["language"] == present_day_lang:
                forms = [
                    c
                    for c in stage["forms"]
                    if c.get("source") != "self" or _grid_surface_key(c["form"]) in keep_keys
                ]
                if not forms:
                    continue  # present-day stage emptied (only unmatched self-seeds)
                stage = {**stage, "forms": forms}
            stages.append(stage)
        if stages:
            out.append({"family": section["family"], "stages": stages})
    return out


def build_cohesion_table(
    flat_cooccurrence: dict[str, int] | None,
    tag_marginal: dict[str, int] | None,
) -> dict[str, dict[str, float]]:
    """Convert the bundle's flat tag co-occurrence counts into the nested
    conditional-probability table the vector path's cohesion multiplier
    consumes (wyrd-e2b4 / D17 refinement).

    The bundle ships co-occurrence as flat ``{"left|right": count}`` pairs —
    ordered (earlier-slot tag → later-slot tag), accumulated over each
    culture's decomposed toponyms by
    :func:`lexicon.proportions_builder.ordered_tag_pairs` — plus a per-tag
    ``tag_marginal`` count. The consumer
    (:func:`vector_name_select._cohesion_raw`) instead reads a nested
    ``{candidate_tag: {prior_tag: prob}}`` map, indexing
    ``table[candidate][prior]``: the candidate is the meaning being scored
    for the current (later) slot, the prior is an already-picked (earlier)
    slot's tag.

    So we restructure to an inverted index — the flat key's left tag (prior)
    becomes the inner key, the right tag (candidate) the outer key — and
    normalize per D17's refinement formula —
    ``P(candidate | prior) ≈ cooccur[prior|candidate] / tag_marginal[prior]``
    — yielding a value in ``[0, 1]``. (Approximate, not a strict conditional:
    ``tag_marginal`` counts a tag in BOTH left and right positions while the
    numerator counts only the prior-as-left pairing, so the denominator is
    inflated. The downstream multiplier mean-normalizes within each slot, so
    this uniform deflation cancels — it's the relative ranking that matters,
    not the absolute value.) Iteration is sorted so the float table is
    byte-identical across processes (PYTHONHASHSEED). Returns an empty dict
    when either input is missing, which degrades the cohesion knob to its
    bit-stable no-op.
    """
    if not flat_cooccurrence or not tag_marginal:
        return {}
    table: dict[str, dict[str, float]] = {}
    skipped = 0
    for key in sorted(flat_cooccurrence):
        left, sep, right = key.partition("|")
        if not sep or not right:
            # Malformed key (no pipe separator, or empty right side): skip
            # rather than mis-key the whole tag under a bare string.
            skipped += 1
            continue
        denom = tag_marginal.get(left, 0)
        if denom <= 0:
            # No marginal for the prior tag — every tag in a co-occurrence
            # pair is also counted in tag_marginal at build time, so a zero
            # here means the bundle's two tables drifted apart.
            skipped += 1
            continue
        # min(1.0, …) guards against any marginal/cooccurrence inconsistency
        # so the consumer's avg_p stays bounded in [0, 1].
        prob = min(1.0, flat_cooccurrence[key] / denom)
        table.setdefault(right, {})[left] = prob
    if skipped:
        # Matches the wyrd-van9 / wyrd-zzli house pattern: a bundle that
        # drifted from its builder shouldn't silently lose cohesion signal.
        _logger.warning(
            "wyrd-e2b4: build_cohesion_table skipped %d malformed/orphaned "
            "tag_cooccurrence keys; re-emit the bundle to clear the drift",
            skipped,
        )
    return table


class NameGenerator:
    def __init__(
        self,
        meaning_db,
        meaning_gen,
        structs,
        tag_cooccurrence: dict[str, int] | None = None,
        culture_attested_usages: frozenset[str] | None = None,
        culture_attested_meanings: dict[str, frozenset[str]] | None = None,
        tag_marginal: dict[str, int] | None = None,
        culture: str | None = None,
    ):
        self.meaning_db = meaning_db
        self.meaning_gen = meaning_gen
        # wyrd-hzqs: the culture NAME (e.g. "english"), used to key the
        # per-language structure allowlist (is_structure_enabled). ``None`` for
        # legacy callers that build a generator without a culture → that path's
        # structures are unfiltered by the allowlist (fail-open).
        self.culture: str | None = culture
        # Set of usage_keys attested in this culture's corpus (union of
        # proportions_usage + proportions_single_usage). Used as a
        # culture filter on the vector path's eligibility pool — without
        # it, English-culture vector requests pull in Welsh / Persian /
        # Hebrew morphemes that aren't attested in English place names
        # (the "culture bleed" issue). ``None`` disables the filter
        # (legacy back-compat for callers that don't supply it).
        #
        # Proportions mode doesn't need this — it samples directly from
        # the per-culture proportions table, which is already culture-
        # restricted by construction.
        self.culture_attested_usages: frozenset[str] | None = culture_attested_usages
        # wyrd-pfoo: per-Meaning attestation per culture. The vector
        # path's eligibility filter narrows from "usage attested in
        # this culture" to "(usage, primary_language) attested in this
        # culture" — preventing wrong-sense picks like Celtic
        # ``-ton``→'tone' or ``Saint-``→'greed' that the per-usage
        # filter admitted as collateral damage. Keys are bare SURFACES
        # (dash-folded usage_keys, union-merged across dash-variants in
        # ``load_proportions`` — wyrd-c6o1.3); values are frozensets of
        # primary-language bundle-field names. ``None`` (legacy bundles
        # built pre-wyrd-pfoo) falls back to the per-usage filter only.
        self.culture_attested_meanings: dict[str, frozenset[str]] | None = culture_attested_meanings
        # wyrd-i7uy: the vector path's eligibility pool + per-slot base-score map
        # are built FRESH per generate() call (see _build_vector_pools) and never
        # stored on the instance. The NameGenerator is shared (one per culture via
        # _load_culture's cache) and long-lived behind the Lambda, so persisting
        # request-derived state here leaked across requests (a tag-filtered pool
        # stuck for later different-tag requests). Caching must never outlive a
        # single request; the per-call rebuild is ~6 ms.
        # wyrd-zzli + wyrd-80ib: filter out structures that would render
        # a bare pre/post/inner morpheme as a standalone word. Two
        # shapes both fail this predicate:
        #   - multi-word `By Green` (zzli): -by + green- split across
        #     two word slots; 46.7% of English structure weight pre-fix
        #   - single-word `Bridge` (80ib): a 1-word structure whose
        #     sole word is `-bridge`, rendered alone after dash-strip;
        #     6.7% of English 1-word structure weight pre-fix
        # The data fix in :func:`lexicon.proportions_builder.encode_structs`
        # (re-exported as ``_encode_structs`` from ``cli.rebuild_proportions``)
        # prevents future rebuilds from emitting them; this runtime gate
        # defends against bundles built before that data fix lands.
        # Two AND-ed structure gates, both load-time:
        #   1. is_structurally_grammatical (wyrd-zzli) — a hard grammaticality
        #      gate (no bare-pre/post/inner standalone words); always on.
        #   2. is_structure_enabled (wyrd-c6o1.5) — the OPERATOR allowlist from
        #      data/structures.yaml; this is the single operator-facing filter
        #      and subsumes the old _is_single_morpheme_structure ("<Bare>")
        #      special-case (the lone-dictionary-word structures now ship
        #      disabled in structures.yaml). Absent label → enabled.
        self.structs = {
            k: v
            for k, v in structs.items()
            if is_structurally_grammatical(k) and is_structure_enabled(k, culture)
        }
        # Loud-failure guard (generator-contract-reviewer P2, round 1):
        # if the filters empty an otherwise-non-empty structs dict, the
        # vector path would crash on a None struct from weighted_choice(rng,
        # []). Raise here with an operator-attributable message instead.
        if structs and not self.structs:
            raise ValueError(
                "NameGenerator: every structure was filtered out — by the "
                "wyrd-zzli grammaticality gate and/or the data/structures.yaml "
                "operator allowlist (wyrd-c6o1.5). Check that structures.yaml "
                "hasn't disabled every structure this culture has, and re-emit "
                "the bundle via `wyrd kenning lexicon rebuild-proportions` if the "
                "grammaticality filter dropped them all."
            )
        # Bit-stability drift warning (generator-contract-reviewer P2,
        # round 1): when the filter drops something, seeded callers
        # against the same shipped bundle get different output than they
        # did pre-fix. Emit one stderr line per construction so operators
        # see that their bundle is stale rather than discovering the
        # drift via diverging SPA explainer / share-link output. Silent
        # when the filter is a no-op (clean bundle).
        # wyrd-c6o1.5: only UNGRAMMATICAL drops are drift worth a "re-emit"
        # warning. Operator-intended allowlist disables (structures.yaml) are NOT
        # drift — they're a deliberate config choice — so they must not trip this
        # warning (else every clean load with a curated allowlist falsely tells the
        # operator to re-emit the bundle). Count the grammaticality drops directly.
        ungrammatical = [k for k in structs if not is_structurally_grammatical(k)]
        if ungrammatical:
            dropped_weight = sum(structs[k] for k in ungrammatical)
            total_weight = sum(structs.values()) or 1
            _logger.warning(
                "wyrd-zzli: NameGenerator: filtered %d ungrammatical structure "
                "templates (%.1f%% of structure weight); re-emit the bundle to "
                "clear the drift",
                len(ungrammatical),
                100 * dropped_weight / total_weight,
            )
        # wyrd-mj2 (D17 β-term per the ticket reframe): tag-level
        # bigram statistics from each culture's place-name corpus.
        # ``tag_cooccurrence`` keys are "left|right" tag pairs; values
        # are co-occurrence counts. Kept in raw-count shape so the
        # JSON↔SQLite bundle round-trip is byte-checkable (and
        # language_quality audits can rank it). Optional — legacy bundles
        # without it produce a no-op cohesion knob.
        self.tag_cooccurrence: dict[str, int] = tag_cooccurrence or {}
        self.tag_marginal: dict[str, int] = tag_marginal or {}
        # wyrd-e2b4: the cohesion multiplier consumes a NESTED conditional-
        # probability table (``{candidate_tag: {prior_tag: prob}}``), not the
        # flat counts above. Derive it once at construction from the raw
        # counts + marginals (see ``build_cohesion_table``). Empty when the
        # bundle lacks either input → cohesion degrades to a bit-stable
        # no-op (the multiplier also short-circuits on cohesion<=0 before
        # touching this table, so cohesion=0 output is unchanged regardless).
        self.cohesion_table: dict[str, dict[str, float]] = build_cohesion_table(
            self.tag_cooccurrence, self.tag_marginal
        )
        # wyrd-bol9: per-bucket empirical-frequency lookup for the
        # vector path. Pre-fix, vector scoring sampled each Meaning by
        # D36.2 score(lemma) regardless of how often the underlying
        # USAGE actually appeared in this culture's corpus. The pool
        # was uniformly distributed across culture-attested usages —
        # losing the empirical-frequency signal the per-bucket weight
        # tables carry. Welsh / Irish /
        # Breton names came out 25-35pp more OE-dominated than the
        # corpus proportions because the OE-heavy attested pool dominated
        # under uniform sampling.
        #
        # Frequency-weighted pool restoration: each Meaning's vector
        # score is composed with its USAGE's per-bucket frequency
        # from those same proportions tables. Within a usage's Meanings,
        # D36.2 score discriminates;
        # across usages, frequency drives pick-share — restoring the
        # corpus's per-usage weighting while preserving
        # vector's per-Meaning composition. Bucket-key shape and the
        # lookup contract live on ``_build_usage_frequency_by_bucket``.
        self.usage_frequency_by_bucket: dict[tuple, dict[str, float]] = (
            self._build_usage_frequency_by_bucket()
        )

    def _build_usage_frequency_by_bucket(self) -> dict[tuple, dict[str, float]]:
        """Snapshot ``meaning_gen.generators``'s per-bucket
        ``{usage: proportion}`` weight tables for the vector path's
        frequency-weighted pool sampling (wyrd-bol9).

        The vector dispatch threads each slot's full bucket-key tuple
        — ``(location, *flags)`` with optional ``name`` / ``saint`` /
        ``single`` — through to scoring; this snapshot is the
        constant lookup that maps bucket_key → per-usage frequency.

        Tests construct NameGenerator with ``meaning_gen=None`` to
        isolate struct-filter behavior
        (``test_kenning_zzli_slot_constraint``). Return an empty
        snapshot in that case so the vector path falls back to its
        unweighted-pool behavior (bit-stable with pre-bol9 for the
        legacy / non-NameGenerator callers that don't supply a real
        MeaningGenerator).
        """
        out: dict[tuple, dict[str, float]] = {}
        if self.meaning_gen is None:
            return out
        for bucket_key, gen in self.meaning_gen.generators.items():
            out[bucket_key] = dict(gen.elements)
        return out

    def select_via_vector(
        self,
        rng,
        *,
        request,
        priors,
        era_midpoint: int = 0,
        cohesion: float = 0.0,
        exclude_tags: tuple[str, ...] = (),
        pack_meaning_dbs: dict | None = None,
        include_unglossed: bool = True,
        spelling_variety: float = 0.0,
        inflection_density: float = 0.0,
        novelty: float = 0.0,
        era_render_language: str | None = None,
        era_requested: bool = False,
        pool_cache: dict | None = None,
    ):
        """Vector-scoring counterpart to :meth:`select` (wyrd-ecjp.5 PR C).

        Pick a structure from ``self.structs`` via the usual weighted
        choice, then dispatch into
        :func:`runtime.vector_name_select.select_via_vector_scoring`
        for the per-slot meaning pick. Returns a :class:`NewName`
        compatible with the same output surface as :meth:`select`
        (``__str__`` / ``description`` / ``components`` /
        ``inflection_labels``).

        Args:
            rng: seeded ``random.Random`` instance.
            request: pre-built :class:`RequestVector` (use
                :func:`vector_kenning_adapter.build_request_vector`
                to translate Kenning's per-call knobs).
            priors: loaded :class:`EmpiricalPriors` (the JSON sidecar
                via :func:`lexicon.empirical_priors.load_empirical_priors_from_json`).
            era_midpoint: int year for the baseline-axis lookup (the D36.7
                per-era frequency cells). 0 = wildcard (no era-fashion lean,
                the default). wyrd-3tvd: a bounded HISTORICAL request era now
                drives this with its midpoint (via
                kenning._request_era_midpoint); an era with no upper bound
                (present-day) stays 0. See vector_name_select.select_via_vector_scoring.
            cohesion: D17 cohesion knob in [0, 1]. Threads through
                to the vector primitive's slot-relative cohesion
                bias (:func:`_cohesion_raw` + :func:`_cohesion_multipliers`).
            exclude_tags: meanings carrying any of these tags are
                filtered out pre-score.
            pack_meaning_dbs: optional ``{pack_name: meaning_db}`` from
                the bundled pack catalog (wyrd-ecjp.10b / .11). When
                ``request.packs`` declares overlays, pack lemmas enter
                the eligible pool via this map. ``None`` (default) =
                no pack lemmas (pure native generation).
            spelling_variety: D18 per-morpheme probability of emitting an
                attested archaic spelling variant instead of the modern
                reflex. wyrd-nbpw: threaded through the shared
                ``_render_substitutions`` render path, so D8/D18 rendering
                is consistent for a given pick. 0 (default) = no variant
                substitution.
            inflection_density: D8 per-morpheme probability of emitting an
                inflected morphological form (with a grammatical-case
                label). 0 (default) = no inflection. Inflection wins ties
                over the variant axis.

        Returns:
            A :class:`NewName` or None when the vector path's gate or
            scoring filtered every candidate (caller decides whether
            to raise or fall back).

        Cohesion uses the slot-relative form from
        ``vector_name_select._cohesion_raw`` + ``_cohesion_multipliers``;
        the derived ``self.cohesion_table`` is threaded through to it.
        """
        # Lazy import: no actual cycle (vector_name_select imports
        # only from vectors.{schemas,scoring}, not runtime); the lazy
        # form also keeps module-load cost off callers that never
        # generate.
        from wyrd.generators.kenning.runtime.vector_name_select import (
            select_via_vector_scoring,
        )

        items = list(self.structs.items())
        # wyrd-i7uy: the eligibility pool + per-slot base-score map are built
        # fresh per call and live only for this dispatch (no cross-request state).
        # exclude_tags_fz is also threaded into the per-slot _score below.
        exclude_tags_fz = frozenset(exclude_tags)
        # wyrd-dag4: build the eligibility pool + base-score map ONCE per request
        # and reuse across the count loop's names. ``pool_cache`` lives in the
        # request's ``params`` (one gate per request, the same dict for every
        # name in the count loop), so a fixed key is correct AND it never
        # crosses requests — preserving the wyrd-i7uy no-cross-request-state
        # rule while recovering the per-name rebuild cost (~200ms/req at count=5
        # on the 1-vCPU Lambda). ``pool_cache is None`` → build fresh (single-
        # result / non-count callers, tests).
        cached_pools = pool_cache.get("vector_pools") if pool_cache is not None else None
        if cached_pools is not None:
            non_position_eligible, slot_base_scores = cached_pools
        else:
            non_position_eligible, slot_base_scores = self._build_vector_pools(
                request,
                exclude_tags_fz,
                pack_meaning_dbs,
                include_unglossed,
            )
            if pool_cache is not None:
                pool_cache["vector_pools"] = (non_position_eligible, slot_base_scores)

        # wyrd-izcr: pick a struct, flatten to positions + qualifier
        # flags, attempt the per-slot score+sample. On failure (empty
        # qualifier-restricted pool or all-zero scores in a qualifier
        # slot), retry with a DIFFERENT struct — failed candidates
        # are excluded from the next draw so retries can't converge
        # on the same un-satisfiable struct. Bounded so a
        # pathologically restrictive gate doesn't burn time.
        # Pre-fix the vector path silently filled qualifier slots
        # with non-qualifier morphemes; now correctly fails and
        # retries to find a struct whose qualifier slots are
        # satisfiable.
        # Flatten a candidate struct's word_keys into parallel position /
        # qualifier / bucket-key lists for the vector primitive, then score it.
        # Struct shape = tuple-of-words, each word tuple-of-keys, each key a
        # feature-tuple (("pre","single"), ("post","name"), …): element 0 is
        # the position label (per ``word_to_key``), the rest are flags
        # (``name`` / ``saint`` / ``single``). Preserving the full key (incl.
        # the "single" flag word_to_key adds for single-element words) makes
        # the per-slot frequency lookup hit the exact bucket proportions
        # samples from. ``permissive`` is threaded through for the wyrd-tbke
        # graceful-degradation fallback below.
        def _score(candidate_struct, *, permissive):
            positions, qualifiers, bucket_keys = _flatten_struct_slots(candidate_struct)
            return select_via_vector_scoring(
                rng,
                self.meaning_db,
                structure=positions,
                request=request,
                priors=priors,
                era_midpoint=era_midpoint,
                cohesion=cohesion,
                cohesion_table=self.cohesion_table or None,
                exclude_tags=exclude_tags_fz,
                pack_meaning_dbs=pack_meaning_dbs,
                non_position_eligible=non_position_eligible,
                slot_base_scores=slot_base_scores,
                slot_qualifiers=qualifiers,
                slot_bucket_keys=bucket_keys,
                usage_frequency_by_bucket=self.usage_frequency_by_bucket,
                novelty=novelty,
                permissive=permissive,
            )

        struct, picked = self._pick_struct_via_vector(rng, items, _score)
        if struct is None:
            # No struct is fully satisfiable AND even the permissive fallback
            # filled nothing (or the bundle is structure-less): genuinely no
            # eligible name. Return None so the caller raises its operator-visible
            # diagnostic — an all-empty name is useless, and proportions only ever
            # emits an empty name in this same degenerate case.
            return None
        words = _reconstruct_words(struct, picked)
        # wyrd-i4jd: carry the picked morpheme_ids alongside the surfaces so the
        # render + grid + breakdown resolve the etymon the generator ACTUALLY
        # selected, by id, instead of re-deriving it from the surface string.
        picked_ids = _reconstruct_picked_ids(struct, picked)

        new_name = NewName(struct, self.meaning_db, words, picked_ids=picked_ids)
        # wyrd-nbpw/6c8x: post-pick rendering — era-form (feature A) or the D8
        # inflection / D18 spelling-variant substitution — applied via
        # _apply_render. wyrd-24s6 (D41): default generation (all knobs 0, no era
        # requested) now renders the NATIVE source-language form into
        # new_name.rendered (the canonical result is native; the modern companion
        # is composed by modern_name()). Only an explicit era="modern-english"
        # leaves rendered None for the force-modern fallback.
        self._apply_render(
            rng, new_name, spelling_variety, inflection_density, era_render_language, era_requested
        )
        return new_name

    def _build_vector_pools(
        self,
        request,
        exclude_tags_fz: frozenset[str],
        pack_meaning_dbs: dict | None,
        include_unglossed: bool,
    ) -> tuple[list, dict]:
        """Build the non-position eligibility pool + an empty per-slot base-score
        map for ONE vector dispatch (a single generate() call → one name).

        wyrd-i7uy: these are scoped to this single call and are NEVER persisted on
        the instance. The pool is reused across this call's struct retries; the
        ``slot_base_scores`` dict is lazy-filled across this call's slots; both are
        discarded when the call returns. Previously they were memoized on the
        NameGenerator — which is shared (one per culture, via ``_load_culture``'s
        cache) and long-lived behind the Lambda — under a hand-built key that had
        drifted from the pool's actual filter inputs (it omitted
        ``gate.required_tags``), so one request's tag-filtered pool leaked into
        later different-tag requests from any user on the warm container. A
        webservice must NEVER cache request-derived state above a single request;
        the per-call rebuild is ~6 ms, negligible vs the correctness it buys."""
        from wyrd.generators.kenning.runtime.vector_name_select import (
            build_non_position_eligible,
        )

        non_position_eligible = build_non_position_eligible(
            self.meaning_db,
            gate=request.gate,
            exclude_tags=exclude_tags_fz,
            pack_meaning_dbs=pack_meaning_dbs,
            packs=request.packs,
            culture_attested_usages=self.culture_attested_usages,
            culture_attested_meanings=self.culture_attested_meanings,
            include_unglossed=include_unglossed,
        )
        # Lazy-filled within THIS call's slot loop (select_via_vector_scoring),
        # reused across the call's struct retries, discarded on return.
        slot_base_scores: dict = {}
        return non_position_eligible, slot_base_scores

    def _pick_struct_via_vector(self, rng, items, score_fn):
        """wyrd-izcr: pick a satisfiable struct + its per-slot picks. Draw a
        struct (weighted) and score it; on failure (empty qualifier-restricted
        pool, or all-zero scores in a qualifier slot) retry with a DIFFERENT
        struct — failed candidates are excluded from the next draw so retries
        can't converge on the same unsatisfiable struct, bounded so a
        pathologically restrictive gate doesn't burn time.

        Returns ``(struct, picked)`` on a full pick. If no struct is fully
        satisfiable, falls back to wyrd-tbke graceful degradation: pick a struct
        (weighted) and fill what it can, emitting ``None`` for slots whose gated
        pool is empty (NewName drops them → a shorter name). Returns
        ``(None, [])`` only when there is genuinely nothing to emit — a
        structure-less bundle, or a gate that excludes every morpheme at every
        slot — so the caller raises its operator-visible diagnostic."""
        tried: set = set()
        for _attempt in range(_VECTOR_STRUCT_RETRY_LIMIT):
            remaining = [it for it in items if it[0] not in tried]
            if not remaining:
                break  # exhausted the distinct structure pool → degrade below
            candidate_struct = weighted_choice(rng, remaining)
            if candidate_struct is None:
                break
            tried.add(candidate_struct)
            picked = score_fn(candidate_struct, permissive=False)
            if picked:
                return candidate_struct, picked

        candidate_struct = weighted_choice(rng, items)
        if candidate_struct is None:
            return None, []
        picked = score_fn(candidate_struct, permissive=True)
        if not any(p is not None for p in picked):
            return None, []
        return candidate_struct, picked

    def _render_substitutions(self, rng, name, spelling_variety, inflection_density):
        """Walk the picked usages and produce two parallel lists: surface
        forms (with D8 inflection or D18 variant substitution applied) and
        inflection labels (for the explainer; None where no inflection
        was picked).

        Per-morpheme rolling order: inflection first (more specific
        morphological data), then variant if no inflection fired. Both pass
        gates are independent draws; inflection wins ties because it
        carries grammatical case while the variant axis is purely
        spelling.
        """
        rendered: list[list[str | None]] = []
        labels: list[list[str | None]] = []
        for word in name:
            word_rendered: list[str | None] = []
            word_labels: list[str | None] = []
            for usage in word:
                if usage is None:
                    word_rendered.append(None)
                    word_labels.append(None)
                    continue
                # wyrd-eyjk/D40: usages are position-forms (`-giles`); resolve
                # the morpheme by its bare surface (the bundle may store only a
                # different dash-variant) so variant/inflection substitution
                # still finds the pools.
                meanings = (
                    self.meaning_gen._surface_index().get(usage.lower().replace("-", "")) or []
                )
                # When a usage maps to multiple Meanings (different senses
                # e.g. -y "district" vs "island"), each sense has its OWN
                # variant + inflection pools because the underlying etymons
                # are distinct. We pick meanings[0] as a deterministic
                # fallback; the picked substitution may not correspond to
                # the sense the structure's tags would have selected.
                # Acceptable trade-off: sampling at the usage level (not
                # the sense level) keeps the runtime simple and seed-stable.
                # A follow-up could thread the structure's tag context to
                # narrow to the matching sense before drawing.
                chosen = meanings[0] if meanings else None
                surface, label = self._pick_surface(
                    rng, usage, chosen, spelling_variety, inflection_density
                )
                word_rendered.append(surface)
                word_labels.append(label)
            rendered.append(word_rendered)
            labels.append(word_labels)
        return rendered, labels

    def _pick_surface(self, rng, usage, chosen, spelling_variety, inflection_density):
        """Decide one element's surface form and inflection label. Returns
        (surface_string, inflection_label_or_None) — the surface is always
        non-None given a non-None usage."""
        canonical = usage.replace("-", "")
        if chosen is None:
            return canonical, None
        if inflection_density > 0:
            picked = chosen.pick_inflection(rng, inflection_density)
            if picked is not None:
                form, label = picked
                return _mimic_case(usage, form), label
        if spelling_variety > 0:
            variant = chosen.pick_variant(rng, spelling_variety)
            if variant is not None:
                return _mimic_case(usage, variant), None
        return canonical, None

    def _apply_render(
        self,
        rng,
        new_name,
        spelling_variety,
        inflection_density,
        era_render_language,
        era_requested=False,
    ):
        """Post-pick render dispatch, used by select_via_vector().

        Precedence (wyrd-6c8x): when an era render-language is set, render each
        morpheme in its era-appropriate attested form — the era reflex IS the
        period spelling, so it supersedes the D18 spelling-variant axis (and D8
        inflection is not combined with era in v1). Otherwise fall back to the
        D8/D18 substitution. Either way the result is written to
        ``new_name.rendered`` (+ ``inflection_labels`` for D8). wyrd-24s6 (D41):
        when no era is set and both substitution knobs are 0 but no era was
        *requested* (the default ``era=""`` path), render each morpheme in its
        NATIVE source-language form (``_render_native_forms``) — the canonical
        result is native, with the modern companion composed by
        ``NewName.modern_name()``. Only an EXPLICIT ``era="modern-english"``
        request (which sets ``era_requested`` while resolving
        ``era_render_language`` to None) leaves ``new_name.rendered`` as None so
        __str__ falls back to the modern usage (the force-modern path)."""
        if era_render_language:
            new_name.rendered = self._render_era_forms(
                new_name.name, era_render_language, new_name.picked_ids
            )
            # wyrd-mf2u: record the era language so the breakdown can resolve +
            # label the era form's own pronunciation (the rendered surfaces are
            # this language's cluster reflexes, not modern-orthography variants).
            new_name.era_render_language = era_render_language
        elif spelling_variety > 0 or inflection_density > 0:
            rendered, labels = self._render_substitutions(
                rng, new_name.name, spelling_variety, inflection_density
            )
            new_name.rendered = rendered
            new_name.inflection_labels = labels
        elif not era_requested:
            # wyrd-24s6 (D41): NO era requested + no substitution → render each
            # morpheme in its NATIVE (source-language) form, so the default
            # (era="") name is "as-selected" rather than coerced to modern. The
            # MODERN companion is composed separately by NewName.modern_name().
            # (Supersedes the pre-D41 bit-stable "leave rendered None → modern".)
            # An EXPLICIT era="modern-english" request resolves era_render_language
            # to None (contemporary suppression) but DOES set era_requested, so it
            # falls through here → rendered stays None → modern usage (the explicit
            # force-modern path, distinct from the native default).
            new_name.rendered = self._render_native_forms(new_name.name, new_name.picked_ids)

    def _morpheme_for_slot(self, usage, mid):
        """wyrd-i4jd: the Meaning to render this slot from. When the generator's
        picked ``morpheme_id`` is known, resolve THAT exact morpheme (so the
        rendered form is one of its own reflexes — id-first); else fall back to
        the surface-derived sense list (legacy / None-id slots). Returns a list
        of Meanings for ``_*_for_meanings`` to consume (one merged morpheme in
        the id-first case)."""
        if mid:
            m = _resolve_morpheme(self.meaning_db, mid)
            if m is not None:
                return [m]
        return self.meaning_gen._surface_index().get(usage.lower().replace("-", "")) or []

    def _render_era_forms(self, name, era_render_language, picked_ids=None):
        """wyrd-6c8x (feature A): render each picked morpheme in its era-
        appropriate attested form for ``era_render_language`` (the canonical
        etymon-language of the requested era cell) instead of the modern
        canonical reflex, so ``era=oe-early`` produces period-LOOKING names.

        Returns a ``NewName.rendered``-shaped list (parallel to ``name``): the
        era form, case-projected onto the slot via ``_mimic_case``, where the
        morpheme carries a reflex for the target language; else None — which
        ``NewName.__str__`` renders as the canonical usage.

        wyrd-i4jd: resolves the morpheme the generator ACTUALLY picked (by
        ``picked_ids``), so the era form is one of that morpheme's OWN reflexes
        — not a first-with-data sibling from the surface bucket. Delegates form
        selection to ``_era_form_for_meanings`` (attested-over-reconstructed, no
        rng draw). None-id slots (legacy) fall back to the surface-derived
        senses; a morpheme with no reflex renders None → modern usage."""
        rendered: list[list[str | None]] = []
        for wi, word in enumerate(name):
            word_rendered: list[str | None] = []
            for ei, usage in enumerate(word):
                if usage is None:
                    word_rendered.append(None)
                    continue
                mid = picked_ids[wi][ei] if picked_ids else None
                meanings = self._morpheme_for_slot(usage, mid)
                form = _era_form_for_meanings(meanings, era_render_language)
                word_rendered.append(_mimic_case(usage, form) if form else None)
            rendered.append(word_rendered)
        return rendered

    def _render_native_forms(self, name, picked_ids=None) -> list[list[str | None]]:
        """wyrd-24s6 (D41): render each picked morpheme in its NATIVE
        (source-language) form so the default (era="") name is "as selected" —
        a genuinely mixed-era surface — rather than coerced to modern.

        wyrd-i4jd: the native form is the generator's PICKED morpheme's
        ``morpheme_id`` canonical (resolved by ``picked_ids``), so the rendered
        surface is that morpheme's own headword — not a first-with-data sibling.
        Case-projected onto the slot. A slot whose picked morpheme has no
        morpheme_id (or a None-id legacy slot with no surface match) renders None
        — ``NewName.__str__`` then falls back to the modern usage."""
        rendered: list[list[str | None]] = []
        for wi, word in enumerate(name):
            word_rendered: list[str | None] = []
            for ei, usage in enumerate(word):
                if usage is None:
                    word_rendered.append(None)
                    continue
                mid = picked_ids[wi][ei] if picked_ids else None
                form = _native_form_for_morpheme_id(mid) if mid else None
                if form is None and not mid:
                    # Legacy / None-id slot ONLY: derive from the surface bucket.
                    # When a slot HAS a picked id but it yields no canonical
                    # (malformed / dash-only morpheme_id) we deliberately leave
                    # form None → modern usage, rather than scanning the surface
                    # siblings — that fallback would render a DIFFERENT morpheme's
                    # canonical, the cross-sibling string-derivation wyrd-i4jd removes.
                    meanings = (
                        self.meaning_gen._surface_index().get(usage.lower().replace("-", "")) or []
                    )
                    form = _native_form_for_meanings(meanings)
                word_rendered.append(_mimic_case(usage, form) if form else None)
            rendered.append(word_rendered)
        return rendered


class NewName:
    def __init__(
        self, struct, meaning_db, name, rendered=None, inflection_labels=None, picked_ids=None
    ):
        self.struct = struct
        self.meaning_db = meaning_db
        self.name = name
        # ``rendered`` is a list-of-lists parallel to ``name`` carrying the
        # surface form to emit per element, with D18 variant substitution
        # or D8 inflection substitution already applied. None means "fall
        # back to dash-stripped usage" (today's bit-stable behavior).
        self.rendered = rendered
        # ``inflection_labels`` is a list-of-lists parallel to ``name`` carrying
        # the grammatical-case label per element when an inflected form was
        # picked at render time, else None. The explainer reads this to
        # surface 'cot@dative_or_pl' breakdowns (D8). None on the outer
        # list means no inflection rendering was performed at all.
        self.inflection_labels = inflection_labels
        # wyrd-mf2u: the canonical etymon-language ``rendered`` was rendered in
        # when an ERA render is active (e.g. 'old-english'), else None. Lets the
        # breakdown (components/to_dict) resolve the era form's OWN pronunciation
        # + label it, instead of mixing the modern surface with an arbitrary
        # cluster pronunciation. Set by _apply_render's era branch; None for the
        # D18/D8 variant/inflection renders (which are modern-orthography).
        self.era_render_language: str | None = None
        # wyrd-vd6y: per-element override Meaning for a slot whose surface
        # repeated earlier in the name and was diversified to a different-
        # language synonym ("Hill Hill" → "Hill Haeth"). Parallel to ``name``;
        # None where no diversification happened. Both __str__ (via rendered)
        # and to_dict honor it so the name + breakdown stay consistent.
        # Per-element override Meaning for a diversified slot (else None).
        self._lang_override = [[None] * len(w) for w in (name or [])]
        # wyrd-i4jd: per-element ``morpheme_id`` of the lemma the generator
        # actually selected, parallel to ``name``. The id-first identity that
        # render/grid/breakdown resolve through (via _resolve_morpheme) instead
        # of re-deriving the etymon from the surface string. None where unknown
        # (render-only / rewind-from-json builds, dropped slots, legacy bundles)
        # → consumers degrade to the surface path. Diversification updates this
        # in lockstep with _lang_override (see _resolve_repeat).
        self.picked_ids = (
            picked_ids if picked_ids is not None else [[None] * len(w) for w in (name or [])]
        )
        # Run lazily on first render (see _ensure_diversified): callers set
        # self.rendered (variant/inflection substitution) AFTER __init__, and
        # the diversification must layer on top of the FINAL rendered surfaces.
        self._diversified = False

    def _ensure_diversified(self) -> None:
        """Run the repeat-diversification once, on first render. Deferred from
        __init__ so it sees the final self.rendered (the caller applies
        variant/inflection substitution after construction)."""
        if self._diversified:
            return
        self._diversified = True
        self._diversify_repeats()

    def _diversify_repeats(self) -> None:
        """wyrd-vd6y + wyrd-72q9: when the SAME morpheme is selected for two+
        slots ("Hill Hill"), resolve the later occurrence(s):
          1. prefer a same-meaning synonym in a DIFFERENT language (cf. "Table
             Mesa"): "Hill Hill" → "Hill Haeth" (render override); else
          2. re-pick a DIFFERENT same-position morpheme: "Park ... Park" →
             "Park ... <other>" (replace the name key); else
          3. leave the duplicate (nothing else of that position available).
        Deterministic (no rng — synonym pick is ranked, re-pick is a stable
        md5 of the name) so it's seed-stable and only touches repeated names.

        D44 two-pass split: PASS 1 (identity) detects repeats by the
        era-INVARIANT modern usage fold — never the era render — so the
        skeleton mutations (synonym override / re-pick) are identical at
        every requested era and ``modern_name()`` is era-stable. PASS 2
        (render) then handles era-DEPENDENT visual collisions — distinct
        identities whose surfaces fold equal only at the requested era's
        render ('Biscop Biscop' from bishop + bishops natively) — by a
        render-only fallback that never touches the picked morphemes.
        """
        if not self.meaning_db or not self.name:
            return  # nothing to resolve siblings against (e.g. render-only tests)
        from wyrd.generators.kenning import _rank_siblings

        # wyrd-72q9: stable per-name signature for varied-but-deterministic
        # re-picks (hash the name, NOT the rng, so it stays seed-stable).
        name_sig = "|".join("/".join(c for c in w if c) for w in self.name)

        # PASS 1 — identity repeats (era-invariant; D44).
        seen: dict[str, set[str]] = {}  # folded usage -> languages already used
        for wi, word in enumerate(self.name):
            for ei, usage in enumerate(word):
                if usage is None:
                    continue
                fold = usage.replace("-", "").lower()
                siblings = _rank_siblings(_resolve_surface(self.meaning_db, usage))
                canon = siblings[0] if siblings else None
                if fold not in seen:
                    seen[fold] = self._meaning_langs([canon]) if canon else set()
                    continue
                # Repeat: resolve it (cross-language synonym → re-pick → leave).
                self._resolve_repeat(wi, ei, usage, fold, canon, siblings, seen, name_sig)

        # PASS 2 — render collisions (era-dependent, render-only; D44).
        self._break_render_collisions(seen)

    def _break_render_collisions(self, seen: dict[str, set[str]]) -> None:
        """D44 pass 2 of ``_diversify_repeats``: distinct identities whose
        surfaces fold equal only at the requested era's render ('Biscop
        Biscop' from bishop + bishops natively). Resolved by falling one
        colliding slot's render back to modern (_break_native_duplicate);
        the picked morphemes never change, so the skeleton — and
        modern_name() — stays identical at every era."""
        if self.rendered is None:
            return
        rseen: dict[str, tuple[int, int, str]] = {}  # rendered fold -> first slot
        for wi, word in enumerate(self.name):
            for ei, usage in enumerate(word):
                if usage is None:
                    continue
                surface = self.rendered[wi][ei] or usage
                fold = surface.replace("-", "").lower()
                if fold not in rseen:
                    rseen[fold] = (wi, ei, usage)
                    continue
                self._break_native_duplicate(wi, ei, usage, fold, seen, rseen)
                # Three-way collisions: a stale rseen entry (a slot that has
                # since been broken to modern) is safe by construction —
                # _break_native_duplicate skips rendered-None candidates, and
                # the first-slot-breaks case only happens when the CURRENT
                # slot is unbreakable (modern == fold), so re-pointing the
                # fold at it could never change a later collider's outcome.
                # (A re-point was briefly added on review and removed once a
                # coverage probe showed the branch is unobservable.)

    def _resolve_repeat(self, wi, ei, usage, fold, canon, siblings, seen, name_sig) -> None:
        """Resolve one repeated IDENTITY (wyrd-vd6y / wyrd-72q9 — pass 1 of
        ``_diversify_repeats``) by trying, in order: a same-meaning synonym in
        a DIFFERENT language ("Hill Hill" → "Hill Haeth", a render override);
        else re-pick a DIFFERENT same-position morpheme ("Park ... Park" →
        "Park ... <other>", replacing the name key); else leave the duplicate
        (render-level visual collisions are pass 2's job). Updates ``seen`` in
        place with whatever fold / language the resolution introduces.
        Deterministic — no rng."""
        canon_langs = self._meaning_langs([canon]) if canon else set()
        alt = self._cross_lang_synonym(canon, siblings, seen[fold] | canon_langs)
        if alt is not None:
            sib, lang, form = alt
            lead = usage[: len(usage) - len(usage.lstrip("-"))]
            trail = usage[len(usage.rstrip("-")) :]
            grafted = f"{lead}{form.strip('-')}{trail}"
            if self.rendered is None:
                self.rendered = [[None] * len(w) for w in self.name]
            self.rendered[wi][ei] = grafted
            self._lang_override[wi][ei] = sib
            # wyrd-i4jd: the slot's identity is now the synonym's — keep
            # picked_ids in lockstep so grid/breakdown resolve the override.
            self.picked_ids[wi][ei] = getattr(sib, "morpheme_id", None)
            seen.setdefault(form.replace("-", "").lower(), set()).add(lang)
            return
        self._repick_repeat(wi, ei, usage, seen, name_sig, canon)

    def _repick_repeat(self, wi, ei, usage, seen, name_sig, canon) -> None:
        """Re-pick a DIFFERENT same-position morpheme for a repeated surface
        ("Park ... Park" → "Park ... Dale"), replacing the name key + updating
        rendered / inflection_labels / ``seen``. When no non-duplicate
        replacement exists the duplicate is simply left (pass 2 may still
        break a render-level collision)."""
        from wyrd.generators.kenning import _rank_siblings

        repl = self._repick_nondup(usage, set(seen.keys()), name_sig, wi, ei, canon)
        if repl is None:
            return
        self.name[wi][ei] = repl
        repl_sibs = _rank_siblings(_resolve_surface(self.meaning_db, repl))
        # wyrd-i4jd: a re-pick chooses a BUCKET (not a scored Meaning), so the
        # identity is the ranked-first of that bucket — the one residual
        # string-derivation in generate, bounded to literal-repeat slots.
        self.picked_ids[wi][ei] = getattr(repl_sibs[0], "morpheme_id", None) if repl_sibs else None
        if self.rendered is not None:
            # wyrd-24s6 (D41): render the re-picked morpheme in its NATIVE
            # form so the native render stays consistent (pre-D41 set None →
            # the modern usage). modern_name() reads the replacement's modern
            # surface from self.name (e) directly.
            native = _native_form_for_meanings(repl_sibs)
            native_rendered = _mimic_case(repl, native) if native else None
            # If the re-pick's NATIVE form ALSO collides (a native homograph
            # of an earlier slot — re-pick excludes seen folds by BUCKET key,
            # but a distinct morpheme can share a native surface), render it
            # MODERN instead: repl's modern usage is collision-free since it
            # was chosen outside the seen folds.
            if native_rendered and native_rendered.replace("-", "").lower() in seen:
                native_rendered = None
            self.rendered[wi][ei] = native_rendered
        if self.inflection_labels is not None:
            self.inflection_labels[wi][ei] = None
        seen[repl.replace("-", "").lower()] = (
            self._meaning_langs([repl_sibs[0]]) if repl_sibs else set()
        )

    def _break_native_duplicate(self, wi, ei, usage, fold, seen, first_slot) -> None:
        """wyrd-24s6 (D41) / D44 pass 2: render-level visual collision. Era
        rendering can collide where the identities don't repeat — two morphemes
        sharing a source-era form but differing modern surfaces ('Biscop
        Biscop'). Break the visual duplicate by falling EITHER colliding slot
        back to its modern usage, preferring whichever one's modern surface
        differs from the collision fold (the OTHER may itself be an archaic
        morpheme whose modern == its native). Render-only by design (D44): the
        picked morphemes never change, so the skeleton stays era-invariant.
        Only acts when a render is active; else leaves the dupe."""
        if self.rendered is None:
            return
        candidates = [(wi, ei, usage)]
        if first_slot is not None and fold in first_slot:
            candidates.append(first_slot[fold])
        for slot_wi, slot_ei, slot_usage in candidates:
            if self.rendered[slot_wi][slot_ei] is None:
                continue
            modern_fold = (slot_usage or "").replace("-", "").lower()
            if modern_fold and modern_fold != fold:
                self.rendered[slot_wi][slot_ei] = None  # __str__ → modern for this slot
                seen.setdefault(modern_fold, set())
                return

    @staticmethod
    def _meaning_langs(meanings) -> set[str]:
        """Union of source-languages across ``meanings``. Meaning.sources is
        normally a ``{lang: [forms]}`` dict, but synthetic Meanings (some tests)
        pass a bare list — a non-dict contributes no languages, so diversification
        simply no-ops there."""
        return {
            lang
            for m in meanings
            if isinstance(getattr(m, "sources", None), dict)
            for lang in m.sources
        }

    @staticmethod
    def _cross_lang_synonym(canon, siblings, exclude_langs):
        """Pick a sibling Meaning that shares a gloss with ``canon`` but lives
        in a language not in ``exclude_langs``. Returns (sibling, lang, form)
        or None. Deterministic: first qualifying sibling in ranked order."""
        if canon is None:
            return None
        canon_glosses = {g.strip().lower() for g in canon.meanings}
        for sib in siblings:
            if not isinstance(getattr(sib, "sources", None), dict):
                continue
            other = set(sib.sources.keys()) - exclude_langs
            if not other:
                continue
            if not (canon_glosses & {g.strip().lower() for g in sib.meanings}):
                continue  # different meaning — not a synonym
            for lang in sorted(other):
                forms = sib.sources.get(lang) or []
                if forms:
                    return sib, lang, forms[0]
        return None

    def _repick_nondup(self, usage, seen_folds, name_sig, wi, ei, canon):
        """wyrd-72q9: pick a DIFFERENT morpheme of the SAME position (pre 'X-',
        post '-X', inner '-X-', bare 'X') to replace a repeat that has no
        cross-language synonym ("Park ... Park" → "Park ... Dale").

        Constrained to morphemes that share a SOURCE LANGUAGE with the original
        (so an English name can't pull a cross-script morpheme — a real bug:
        "Park" was once re-picked as Arabic "ياقوتي"), and PREFERRING those that
        also share a THEMATIC TAG. Deterministic but varied: a stable md5 of the
        name signature + slot indexes the (sorted) eligible pool — seed-stable
        (no rng draw) yet different across names. Returns a bucket key, or None
        when nothing appropriate is available (then the dupe is left as-is)."""
        import hashlib

        want = self._position_markers(usage)
        canon_langs = self._meaning_langs([canon]) if canon is not None else set()
        canon_tags = set(getattr(canon, "tags", []) or [])
        same_lang, tag_sharing = self._collect_repick_pools(
            want, seen_folds, canon_langs, canon_tags
        )
        pool = sorted(tag_sharing) or sorted(same_lang)
        if not pool:
            return None
        h = int(hashlib.md5(f"{name_sig}#{wi}.{ei}".encode()).hexdigest(), 16)
        return pool[h % len(pool)]

    @staticmethod
    def _position_markers(usage: str) -> tuple[bool, bool]:
        """Position signature of a bucket key: (has-leading-dash, has-trailing-
        dash) — distinguishes pre 'X-' / post '-X' / inner '-X-' / bare 'X'."""
        return (usage.startswith("-"), usage.endswith("-"))

    def _collect_repick_pools(self, want, seen_folds, canon_langs, canon_tags):
        """Scan meaning_db for same-position re-pick candidates (wyrd-72q9):
        same position as ``want``, not already used (``seen_folds``), and sharing
        a SOURCE LANGUAGE with the original (the language filter is skipped only
        when the original carries no language data, e.g. synthetic test
        Meanings). Returns ``(same_lang, tag_sharing)`` — ``tag_sharing`` is the
        subset additionally sharing a thematic tag with the original.

        wyrd-57d8: a candidate key is admitted only if at least one of its
        senses is base-pool-eligible (``not Meaning.is_base_pool_excluded()``) —
        the SAME class exclusion the scored native pool applies. Without it this
        re-pick read ``meaning_db`` raw and could break a ``corner corner``
        repeat by grafting in a synthesized proper-noun subject the base pool
        would never have picked (the manorial ``Malet`` leak: bare, old_french,
        so it passed the position + source-language filters)."""
        same_lang: list[str] = []
        tag_sharing: list[str] = []
        # Localize the staticmethod lookups — this loop scans the whole
        # meaning_db (thousands of buckets), so per-iteration self. attribute
        # resolution is measurable overhead.
        pos_markers = self._position_markers
        meaning_langs = self._meaning_langs
        for k, ms in self.meaning_db.items():
            if not isinstance(k, str) or pos_markers(k) != want:
                continue
            kf = k.replace("-", "").lower()
            if not kf or kf in seen_folds:
                continue
            # Drop keys whose every sense is a base-pool-excluded class
            # (synthesized saint / manorial subjects, connector particles): they
            # belong only in their dedicated slots, never a base re-pick. A key
            # keeps eligibility as long as one co-tagged place sense survives.
            # (Synthetic non-Meaning test entries carry no class, so an all-
            # synthetic key is left admissible.)
            senses = [m for m in ms if isinstance(m, Meaning)]
            if senses and all(m.is_base_pool_excluded() for m in senses):
                continue
            if canon_langs and not (meaning_langs(ms) & canon_langs):
                continue
            same_lang.append(k)
            if canon_tags and (
                {t for m in ms for t in getattr(m, "tags", None) or []} & canon_tags
            ):
                tag_sharing.append(k)
        return same_lang, tag_sharing

    def __str__(self):
        # wyrd-3xdb: title-case the first letter of each space-
        # separated word so generated names always read as proper
        # nouns. The bundle stores pre-modifiers cap'd ('Whit-',
        # 'Ash-') and post-modifiers / inner-modifiers lowercase
        # ('-ham', '-ton', '-ward'); a word constructed entirely
        # from lowercase morphemes used to land lowercase in the
        # output ('ward green', 'castle', 'holmrwood common'),
        # leaking bundle-implementation detail into operator
        # output.
        #
        # wyrd-ovsv: also collapse 3+-letter runs at the join
        # boundary down to 2 (e.g. 'Kill' + 'llen' → 'Killlen'
        # becomes 'Killen'). The doubling-then-tripling artifact
        # surfaces when one morpheme ends in 'XX' and the next
        # starts with 'X'; real English / Celtic place names
        # always elide this at the morpheme boundary (Killarney
        # NOT Killlarney; Llanelli NOT Llanellli). Conservative
        # regex normalization that doesn't touch single morphemes'
        # own legitimate double letters (Kill, Ball, Newcastle
        # stay intact when standalone).
        #
        # Algorithm: build each word by concatenating its morpheme
        # strings (dash-stripped) inside the word, collapse any 3+
        # run, then capitalize the first character of the joined
        # word. Words join with spaces. Already-cased openings
        # pass through unchanged because ``ch.upper() == ch`` for
        # already-uppercase letters.
        # wyrd-5z5j / DECISIONS D39: the render is the SINGLE owner of
        # positional case, keyed on the SLOT (word position), not the
        # morpheme's stored case. The word-INITIAL morpheme keeps its natural
        # case and is front-capped (so internal caps like `McLeod` / `O'Brien`
        # survive); every NON-initial morpheme (post/inner) is lowercased. So a
        # name used mid-word (`-buna-`) renders `buna`, never a stray mid-word
        # capital (`CornnamullacBunarath`), while a name at a word start stays
        # capitalized. Substituted spelling-variants / inflections
        # (``self.rendered``) obey the same slot rule — a variant dropped into
        # an inner slot lowercases like the base would.
        self._ensure_diversified()
        words: list[str] = []
        for wi, w in enumerate(self.name):
            chunks: list[str] = []
            first = True
            for ei, e in enumerate(w):
                if e is None:
                    continue
                if self.rendered is not None and self.rendered[wi][ei] is not None:
                    surface = self.rendered[wi][ei]
                else:
                    surface = e
                surface = surface.replace("-", "")
                chunks.append(surface if first else surface.lower())
                first = False
            joined = "".join(chunks)
            if joined:
                joined = _collapse_triple_letters(joined)
                joined = joined[0].upper() + joined[1:]
            words.append(joined)
        return " ".join(w for w in words if w)

    def modern_name(self) -> str:
        """wyrd-24s6 (D41): the MODERN rendering — every morpheme in its
        present-day surface (the ``Meaning.usage`` bucket key, sourced from the
        bundle's ``modern_usage`` field), the always-present secondary beside the
        native ``__str__`` canonical. Honors the later composition
        layers: a diversified cross-language synonym ("Hill Hill" → "Hill
        Haeth") shows ITS OWN modern surface (the override Meaning's ``usage``),
        and a re-picked morpheme uses its replacement's modern usage (already in
        ``self.name``). Same slot-case rules as ``__str__``."""
        self._ensure_diversified()
        words: list[str] = []
        for wi, w in enumerate(self.name):
            chunks: list[str] = []
            first = True
            for ei, e in enumerate(w):
                if e is None:
                    continue
                override = self._lang_override[wi][ei]
                surface = override.usage if override is not None else e
                surface = surface.replace("-", "")
                chunks.append(surface if first else surface.lower())
                first = False
            joined = "".join(chunks)
            if joined:
                joined = _collapse_triple_letters(joined)
                joined = joined[0].upper() + joined[1:]
            words.append(joined)
        return " ".join(w for w in words if w)

    def __repr__(self):
        words = []
        for w in self.name:
            for e in w:
                if e is None:
                    continue
                words.append(_resolve_surface(self.meaning_db, e))
        return repr(words)

    def description(self):
        # wyrd-c0xn: one entry per line, gloss block only — citations
        # ride in components() for the SPA's expandable citation box.
        # The CLI human-readable text channel stays scannable instead
        # of a dense citation-laden paragraph.
        #
        # wyrd-o53o + wyrd-ywm9: rank siblings and only render the
        # primary semantic glosses in the per-meaning gloss block.
        # Pre-fix this concatenated every sibling's glosses verbatim
        # so the SPA's result-card preview was dominated by
        # 'alternative form of X' pointers and modern_english homo-
        # graph noise. Post-fix the preview matches the morpheme
        # card: top-ranked etymon's semantic glosses only.
        from wyrd.generators.kenning import _partition_senses, _rank_siblings

        results = []
        for wi, word in enumerate(self.name):
            single = len(word) == 1
            for ei, e in enumerate(word):
                if e is None:
                    continue
                lemma = e.replace("-", "") if single else e
                label = self._inflection_label_for(wi, ei)
                head = f"{lemma}@{label}" if label else lemma
                ranked = _rank_siblings(_resolve_surface(self.meaning_db, e))
                # wyrd-o53o round 3 (Gemini MED): partition once per
                # sibling rather than twice (any-check + render loop).
                partitioned = [(_partition_senses(list(m.meanings)), m) for m in ranked]
                # If ANY sibling has semantic glosses, drop derivative-
                # only siblings from the preview. Otherwise (e.g.
                # -sing-) render derivatives so the preview isn't blank.
                any_semantic = any(semantic for (semantic, _), _ in partitioned)
                meanings = []
                for (semantic, derivative), m in partitioned:
                    chosen = semantic if any_semantic else derivative
                    if not chosen:
                        # Sibling contributed no glosses for the
                        # selected partition — drop entirely.
                        continue
                    roots = self._roots_str(m)
                    meanings.append(f"{roots} {', '.join(chosen)}")
                gloss_block = " or ".join(meanings) if meanings else "(unglossed)"
                results.append(f"{head} ({gloss_block})")
        return "\n".join(results)

    def _inflection_label_for(self, wi: int, ei: int) -> str | None:
        if self.inflection_labels is None:
            return None
        try:
            return self.inflection_labels[wi][ei]
        except IndexError:
            return None

    def _morpheme_id_for(self, wi: int, ei: int) -> str | None:
        """wyrd-i4jd: the morpheme_id the generator selected for this slot (after
        diversification override), or None when unknown → caller falls back to
        the surface-derived etymon. Single accessor so every consumer
        None-degrades identically."""
        try:
            return self.picked_ids[wi][ei]
        except (IndexError, TypeError):
            return None

    def to_dict(self) -> dict:
        """wyrd-cp2d: structured dump preserving per-word morpheme
        grouping + per-morpheme source-language picks. Consumed by
        downstream commands that want to skip re-decomposition
        (``kenning rewind --from-json``, ``kenning era-map
        --from-json``) and by the webapp's render → rewind
        continuity flow.

        Without this, downstream consumers receive only the rendered
        string and must re-run the trie matcher to decompose it,
        which can pick DIFFERENT morpheme lemmas than the generator
        did (the 'Cranwith → Cornlandian' situation observed in the
        2026-05-22 demo).

        Shape:
            {
              "name": str,                # rendered output (post-PR 299 title-case)
              "words": [                  # per-word, per-morpheme
                [
                  {"usage": str,          # bucket key (with dash markers)
                   "sources": {lang: [forms]},  # per-language lemmas
                   "tags": [tag],
                   "meanings": [gloss],
                   "rendered": str        # OPTIONAL: D18 variant / D8 inflection
                  },
                  ...
                ],
                ...
              ]
            }

        Multi-meaning ambiguity: a single bucket key can map to
        multiple Meaning objects (different etymological histories).
        Following ``components()``'s precedent we capture the FIRST
        Meaning for sources / tags / meanings. The full ambiguity
        stays available to downstream consumers that need it via
        their own meaning_db lookup using ``usage``.
        """
        self._ensure_diversified()
        words: list[list[dict]] = []
        for wi, word in enumerate(self.name):
            morphemes = [
                self._morpheme_to_dict(wi, ei, e) for ei, e in enumerate(word) if e is not None
            ]
            if morphemes:
                words.append(morphemes)
        return {"name": str(self), "words": words}

    def _morpheme_to_dict(self, wi: int, ei: int, e) -> dict:
        """Build one morpheme's breakdown dict for :meth:`to_dict`."""
        # wyrd-o53o + wyrd-ywm9 + wyrd-emlb: rank siblings (drop modern_english
        # homographs, prioritize older strata). Lazy-import to avoid a circular
        # dep with wyrd.generators.kenning.__init__ at module load — called per
        # morpheme, so the import is paid once after Python caches sys.modules.
        from wyrd.generators.kenning import _rank_siblings

        # wyrd-vd6y: a diversified repeat ("Hill Hill" → "Hill Haeth") uses its
        # override sibling — a different-language synonym — for BOTH the displayed
        # surface and the etymology, so the breakdown matches the rendered name
        # instead of showing the original repeated morpheme.
        override = self._lang_override[wi][ei] if self._lang_override else None
        if override is not None:
            ranked = [override]
            usage_str = (self.rendered[wi][ei] if self.rendered else None) or e
            grid_mid = getattr(override, "morpheme_id", None)
        else:
            ranked = _rank_siblings(_resolve_surface(self.meaning_db, e))
            usage_str = e
            # wyrd-i4jd: the reflex GRID (+ active_form_id highlight) is id-first
            # — built from the morpheme the generator actually PICKED, so the
            # grid + highlight match the rendered surface instead of a homograph
            # re-derived from the string. The display fields (tags/meanings/
            # sources) stay the surface-sibling union. Fall back to the
            # ranked-first's id for a None-id slot (legacy / rewind-from-json).
            grid_mid = self._morpheme_id_for(wi, ei) or (
                getattr(ranked[0], "morpheme_id", None) if ranked else None
            )
        first = ranked[0] if ranked else None
        morpheme: dict = {"usage": usage_str}
        if first is not None:
            self._add_etymology_fields(morpheme, ranked, first, grid_mid)
        self._add_rendered_fields(morpheme, wi, ei, first)
        self._set_active_form_id(morpheme, first, grid_mid)
        return morpheme

    def _union_with_picked(self, ranked: list, grid_mid):
        """wyrd-0wpd: the sibling set for the surface-UNION display fields (tags /
        meanings / roots), augmented with the PICKED etymon when ``_rank_siblings``
        dropped it. A surface shared by several etymons (e.g. a modern-english
        homograph the ranker drops, wyrd-o53o) could otherwise show tags/meanings
        for an etymon the generator did NOT pick — contradicting the id-first grid
        + active_form_id, which DO name the pick (e.g. seed=21 'Dūnworth': picked
        modern-english:worth [water], but '-worth' ranks to old-english:worþ
        [architecture]). Adding the picked morpheme to the union keeps the rich
        surface display while guaranteeing the pick's own tags/meanings are
        present. No-op when the pick is already among the siblings (the common
        case) or unknown (legacy / rewind-from-json / render-only)."""
        if (
            not grid_mid
            or not self.meaning_db  # render-only / direct calls with no db to resolve against
            or any(getattr(m, "morpheme_id", None) == grid_mid for m in ranked)
        ):
            return ranked
        picked = _resolve_morpheme(self.meaning_db, grid_mid)
        return [*ranked, picked] if picked is not None else ranked

    def _add_etymology_fields(self, morpheme: dict, ranked: list, first, grid_mid=None) -> None:
        """Add the etymology fields for the top-ranked sibling ``first``. Sources
        stay from ``first`` (the canonical etymon for this surface); tags union
        across all ranked siblings (wyrd-ywm9); meanings split into primary +
        derivative for the SPA card layout. meaning_groups / renderings /
        citations / era_grid are sparse — emitted only when non-empty, so
        Latin-script / rando-port-only morphemes don't carry noisy empties.

        wyrd-i4jd: the era_grid is resolved from ``grid_mid`` — the morpheme_id
        the generator PICKED — so the reflex grid (+ active_form_id highlight)
        matches the rendered surface, not a homograph re-derived from the string.
        Falls back to ``first.morpheme_id`` when ``grid_mid`` is None (legacy)."""
        # Lazy import (circular-dep avoidance — see _morpheme_to_dict).
        from wyrd.generators.kenning import (
            _all_senses,
            _meaning_groups,
            _split_senses_for_display,
        )

        # wyrd-0wpd: tags + meanings (+ groups) union over the siblings PLUS the
        # picked etymon, so a pick the surface ranker dropped still shows its own
        # tags/meanings (matching the id-first grid + active_form_id). sources stay
        # the surface-canonical `first`; renderings/citations stay the ranked
        # siblings (cross-script/scholarly, where a modern pick adds only noise).
        # NOTE: components() deliberately does NOT augment with the pick — it's the
        # realism gate's distributional basis and must mirror the corpus
        # reference's surface-union resolution (see the comment there). to_dict
        # (this) is the SPA inspector's display, so it shows what was picked.
        union = self._union_with_picked(ranked, grid_mid)
        morpheme["sources"] = {lang: list(forms) for lang, forms in first.sources.items()}
        morpheme["tags"] = list(dict.fromkeys(t for m in union for t in m.tags))
        primary, derivative = _split_senses_for_display(_all_senses(union))
        morpheme["meanings"] = primary
        morpheme["derivative_meanings"] = derivative
        # wyrd-0y3k: per-sibling deduped sense groups (the flat `meanings` above
        # stays for back-compat / other consumers).
        groups = _meaning_groups(union)
        if groups:
            morpheme["meaning_groups"] = groups
        # wyrd-cp2d r3 + wyrd-o53o: cross-script renderings (original_script /
        # transliteration / english_shaped / IPA) drawn from the ranked siblings,
        # so dropped modern_english entries don't add spurious data.
        renderings = _collect_renderings(ranked)
        if renderings:
            morpheme["renderings"] = renderings
        # wyrd-bvwu: scholarly citations (source_ids, wyrd-9kh.1) for the SPA
        # inspector's expandable view.
        citations = _collect_citations(ranked)
        if citations:
            morpheme["citations"] = citations
        # wyrd-lftl + wyrd-rogd.10 P3: per-morpheme family × era reflex grid,
        # resolved against the UNIFIED morpheme by its stable morpheme_id (no
        # dash-strip string fallback; an un-attributable surface → _era_grid(None)
        # → [] → no grid).
        morpheme_meaning = _resolve_morpheme(
            self.meaning_db, grid_mid or getattr(first, "morpheme_id", None)
        )
        # wyrd-phww: narrow the present-day stage to the surface this instance
        # used (morpheme['usage'] — the generated surface for generate, the
        # decomposed input surface for explain) so the era-grid shows what the
        # name actually used, not the genuine cluster reflex alone.
        grid = _era_grid(morpheme_meaning, renderings, own_surface=morpheme.get("usage"))
        if grid:
            morpheme["era_grid"] = grid

    def _add_rendered_fields(self, morpheme: dict, wi: int, ei: int, first) -> None:
        """Add the D18 variant / D8 inflection / era-substitute fields when this
        slot carries a rendered form. For an ERA render also carry the era
        language + the era form's OWN pronunciation (wyrd-mf2u), so the breakdown
        shows the period form alongside the modern anchor."""
        if self.rendered is None or self.rendered[wi][ei] is None:
            return
        era_form = self.rendered[wi][ei]
        morpheme["rendered"] = era_form
        if self.era_render_language:
            morpheme["rendered_language"] = self.era_render_language
            pron = _era_pronunciation(
                morpheme.get("renderings"), self.era_render_language, era_form, first
            )
            if pron:
                morpheme["rendered_pron"] = pron

    def _set_active_form_id(self, morpheme: dict, first, grid_mid=None) -> None:
        """wyrd-i4jd / wyrd-3vju.1: stamp ``active_form_id`` — the grid-cell id of
        the reflex this morpheme actually rendered — so the SPA highlights it BY
        ID. Call AFTER the grid + rendered fields are set.

        wyrd-3vju.1 (reflex-id contract): compute the id FORWARD from the exact
        reflex the render chose — its raw form + render language — and stamp it
        when that cell is present in the grid (exact-id check, no fold). This is
        the guarantee the operator specified: the rendered surface IS one of the
        listed reflexes, by construction, so 'a morpheme not in its own reflexes'
        and the macron Output/Inspect mismatch (the fold disagreeing on combining
        marks) can't happen for the common path.

        wyrd-3vju.3: the genuine data gap — the rendered form NOT being an
        enumerated reflex of the picked etymon (sparse era_reflexes / synthesized
        surface) — is closed by :func:`_ensure_cell`: the live form's cell is
        injected into its render stage so the highlight always has a target.
        (This replaced the legacy backwards cross-grid fold-search.)"""
        grid = morpheme.get("era_grid")
        if not grid:
            return
        mid = grid_mid or getattr(first, "morpheme_id", "")
        usage = morpheme.get("usage")
        # wyrd-3vju.3 ANCHOR: the morpheme's canonical usage is its IDENTITY (the
        # parent the name actually chose, e.g. '-ton'), independent of which era
        # reflex is live ('-tun' in OE). Always guarantee the identity is a present-
        # day cell (inject if no reflex carries it), then mark EVERY cell that folds
        # to it across eras — so the SPA can always show + swap back to the identity.
        anchor_id = None
        anchor_lang = _present_day_lang(grid, mid)
        if usage and anchor_lang:
            anchor_id = _ensure_cell(grid, anchor_lang, usage, source="usage")
        # ACTIVE: the live rendered reflex (highlight), distinct from the identity.
        # When the rendered reflex isn't an enumerated cell (sparse era_reflexes),
        # ensure it so the highlight has a target; force-modern's live form IS the
        # identity anchor (rendered is None → the usage), so reuse the anchor cell.
        rendered = morpheme.get("rendered")
        forward_id = self._forward_active_id(morpheme, mid)
        if forward_id and _grid_has_cell_id(grid, forward_id):
            cid = forward_id
        elif rendered is not None and morpheme.get("rendered_language"):
            cid = _ensure_cell(grid, morpheme["rendered_language"], rendered, source="self")
        elif rendered is not None:
            cid = _ensure_cell(grid, (mid or "").partition(":")[0] or None, rendered, source="self")
        else:
            cid = anchor_id
        # Mark the identity LAST, after every injection above, so a late-injected
        # cell that folds to the usage carries is_usage too — the SPA invariant is
        # "EVERY cell folding to the usage is marked".
        if usage:
            _mark_usage_cells(grid, usage)
        if cid:
            morpheme["active_form_id"] = cid

    def _forward_active_id(self, morpheme: dict, mid: str | None) -> str | None:
        """The grid-cell id of the reflex the render actually chose, computed FORWARD
        from its raw form + render language (era render) or the morpheme's native
        headword (native render). None for force-modern (rendered is None) or when
        the picked morpheme can't be resolved — the caller then falls back."""
        rendered = morpheme.get("rendered")
        if rendered is None:
            return None
        lang = morpheme.get("rendered_language")
        if lang:
            # Era render: the raw reflex chosen at this language, resolved against the
            # SAME morpheme the grid was built from (raw form keeps the '*' marker).
            grid_meaning = _resolve_morpheme(self.meaning_db, mid or None)
            raw = _era_reflex_raw_choice([grid_meaning] if grid_meaning else [], lang)
            return f"{lang}:{raw}" if raw is not None else None
        # Native render: the picked morpheme's own headword in its source language.
        lang = (mid or "").partition(":")[0] or None
        native = _native_form_for_morpheme_id(mid)
        return f"{lang}:{native}" if (lang and native) else None

    def components(self):
        """Structured component breakdown for the API envelope.

        wyrd-qhs0 Phase 2d: each component additionally carries a
        per-language ``renderings`` map that surfaces the four
        wyrd-ha9q rendering columns (original_script + transliteration
        + english_shaped + IPA) for non-Latin source-lang morphemes.
        Empty / absent for Latin-script langs and older bundles.
        Consumed by the SPA's etymological-provenance panel.

        wyrd-o53o + wyrd-ywm9: ranks siblings (drops modern_english
        homographs, prioritizes older strata) and splits semantic vs
        derivative glosses, matching to_dict()'s behavior. Pre-fix this
        method returned first-by-insertion-order; post-fix it returns
        the canonical etymon."""
        from wyrd.generators.kenning import (
            _all_roots,
            _all_senses,
            _rank_siblings,
            _split_senses_for_display,
        )

        self._ensure_diversified()
        out = []
        for wi, word in enumerate(self.name):
            for ei, e in enumerate(word):
                if e is None:
                    continue
                # wyrd-vd6y: a diversified repeat surfaces its override sibling
                # (the cross-language synonym) here too, so the provenance panel
                # matches the rendered name + to_dict breakdown.
                override = self._lang_override[wi][ei] if self._lang_override else None
                if override is not None:
                    ranked = [override]
                    usage_str = (self.rendered[wi][ei] if self.rendered else None) or e
                    grid_mid = getattr(override, "morpheme_id", None)
                else:
                    ranked = _rank_siblings(_resolve_surface(self.meaning_db, e))
                    usage_str = e
                    # wyrd-i4jd: id-first GRID — built from the picked morpheme so
                    # the reflex panel + active_form_id match the rendered surface;
                    # display fields stay the surface-sibling union (mirrors
                    # _morpheme_to_dict).
                    grid_mid = self._morpheme_id_for(wi, ei) or (
                        getattr(ranked[0], "morpheme_id", None) if ranked else None
                    )
                if not ranked:
                    continue
                first = ranked[0]
                # wyrd-0wpd: components() stays the SURFACE-UNION (no picked-etymon
                # augmentation, unlike to_dict/_add_etymology_fields). This field is
                # the realism gate's DISTRIBUTIONAL basis — run_realism_samples
                # measures sample tags from `components`, and the corpus reference
                # (realism_reference.compute_corpus_reference) resolves corpus tags
                # the SAME surface-union way (a corpus usage has no "pick"). The two
                # MUST use one resolution or the tag_kl_divergence gate breaks. The
                # picked-etymon fix lives in to_dict (morphemes_by_word), which is
                # what the SPA inspector renders; components is the API summary /
                # measurement basis.
                primary, derivative = _split_senses_for_display(_all_senses(ranked))
                renderings = _collect_renderings(ranked)
                morpheme = {
                    "usage": usage_str,
                    "location": first.location,
                    "meanings": primary,
                    "derivative_meanings": derivative,
                    "tags": list(dict.fromkeys(t for m in ranked for t in m.tags)),
                    # wyrd-o53o round 5 (Gemini MED): union roots
                    # across ranked siblings to match meanings + tags
                    # (and the explain endpoint's behavior). Pre-fix
                    # took first.sources only, so a usage with both
                    # OE + Celtic etymons surfaced 'EN' but not
                    # 'CL' in the API summary.
                    "roots": _all_roots(ranked),
                    "citations": _collect_citations(ranked),
                    "renderings": renderings,
                }
                # wyrd-lftl: family × era reflex grid (col-3 inspector) — same
                # structure to_dict emits, so the API envelope's `components`
                # and `morphemes_by_word` stay in lockstep. Sparse (wyrd-32t1).
                # wyrd-rogd.10 Phase 3: morpheme_id is authoritative for the grid
                # — no string-derivation fallback (see the to_dict site above).
                morpheme_meaning = _resolve_morpheme(
                    self.meaning_db, grid_mid or getattr(first, "morpheme_id", None)
                )
                grid = _era_grid(morpheme_meaning, renderings, own_surface=usage_str)
                if grid:
                    morpheme["era_grid"] = grid
                # wyrd-mf2u: carry the era form + its own pronunciation/language
                # so the API envelope's breakdown matches the era-rendered name
                # (the SPA shows era + modern). Was missing here — components()
                # dropped `rendered` entirely, so the breakdown showed only the
                # modern surface despite the name being era-rendered.
                if self.rendered is not None and self.rendered[wi][ei] is not None:
                    era_form = self.rendered[wi][ei]
                    morpheme["rendered"] = era_form
                    if self.era_render_language:
                        morpheme["rendered_language"] = self.era_render_language
                        pron = _era_pronunciation(
                            renderings, self.era_render_language, era_form, first
                        )
                        if pron:
                            morpheme["rendered_pron"] = pron
                self._set_active_form_id(morpheme, first, grid_mid)
                out.append(morpheme)
        return out

    def _roots(self, meaning) -> list[str]:
        keys = [
            ("old_english", "EN"),
            ("old_scandinavian", "SC"),
            ("old_french", "FR"),
            ("celtic_mix", "CL"),
            ("latin", "LA"),
            ("germanic", "GE"),
            ("greek", "GR"),
        ]
        return [code for src, code in keys if src in meaning.sources]

    def _roots_str(self, meaning) -> str:
        return "/".join(self._roots(meaning))


def _collect_citations(meanings):
    """Distinct sorted source_ids across every Meaning sharing one usage and
    every language they cover (wyrd-9kh.1). The runtime explainer surfaces
    this so a GM can see which scholars attest the morpheme. Empty list
    when no Meaning carries citations (rando-port-only entries with no
    scholarly witnesses yet, or older bundles that pre-date the
    wyrd-9kh.1 citation field)."""
    seen: set[str] = set()
    for m in meanings:
        for citations in m.citations.values():
            seen.update(citations)
    return sorted(seen)


def _collect_renderings(meanings):
    """wyrd-qhs0 Phase 2d: gather the four wyrd-ha9q rendering columns
    (original_script + transliteration + english_shaped + IPA) per
    language, indexed by canonical_form so the SPA can match them up
    against the language form arrays it already shows.

    Output shape — sparse `dict[lang_field, dict[canonical_form, slots]]`:

        {
          <lang_field>: {
            <canonical_form>: {
              # only the keys that have data appear; absent != None
              "original_script": str,    # optional
              "transliteration": str,    # optional
              "english_shaped":  str,    # optional
              "ipa":             str,    # optional
              "dialect":         str,    # optional, only when ipa is present
            },
            ...
          },
          ...
        }

    Each `slots` dict is sparse — only the keys for which the lexicon
    has data appear (consumers like the SPA panel use truthy / `.get()`
    checks rather than branching on None). Forms with no rendering
    data don't appear at all; lang_fields with no shaped forms don't
    appear at all. Returns an empty dict when no Meaning carries any
    rendering data (Latin-script source langs only, older bundles).

    Works across multiple Meanings sharing one usage. On collision
    (same lang_field + canonical_form across two Meanings), the slot
    for each rendering column is set to whichever value isn't None;
    a non-None earlier write is preserved when a later iteration tries
    to write None for the same key (matters most for the IPA/dialect
    pair where one Meaning could have ipa=None while another has the
    full pair).
    """
    by_lang_form: dict[str, dict[str, dict[str, str]]] = {}

    def _ensure(lang_field: str, form: str) -> dict[str, str]:
        return by_lang_form.setdefault(lang_field, {}).setdefault(form, {})

    # wyrd-nrxr: each rendering column is walked by its own per-attribute
    # helper so this function stays under the C901 ceiling and mirrors the
    # per-attribute structure of the schema. The three flat columns
    # (original_script / transliteration / english_shaped) share one helper;
    # pronunciation carries an ipa/dialect pair so it has its own.
    for m in meanings:
        _ingest_flat_rendering(m, "original_script", _ensure)
        _ingest_flat_rendering(m, "transliteration", _ensure)
        _ingest_flat_rendering(m, "english_shaped", _ensure)
        _ingest_pronunciation_rendering(m, _ensure)
    # wyrd-03cx: derive reader_pronunciation from ipa when no hand-
    # curated english_shaped value already filled the slot.
    _fill_reader_pronunciations(by_lang_form)
    return by_lang_form


def _set_rendering_slot(slot: dict[str, str], key: str, value: str | None) -> None:
    """Store a rendering value into ``slot[key]``, but never overwrite a
    previously-stored non-None value with None — matters when the same usage
    spans multiple Meanings and one carries richer data than another for the
    same canonical form (most importantly the ipa/dialect pair)."""
    if value is None:
        return
    slot[key] = value


def _ingest_flat_rendering(
    meaning,
    attr: str,
    ensure: Callable[[str, str], dict[str, str]],
) -> None:
    """Copy one flat ``dict[lang_field][form] = value`` rendering column
    (original_script / transliteration / english_shaped) into the slots.
    The attribute name doubles as the slot key."""
    for lang_field, forms in getattr(meaning, attr).items():
        for form, value in forms.items():
            _set_rendering_slot(ensure(lang_field, form), attr, value)


def _ingest_pronunciation_rendering(
    meaning,
    ensure: Callable[[str, str], dict[str, str]],
) -> None:
    """Copy the pronunciation column into the slots. Unlike the flat columns
    each entry is a dict carrying both ipa and dialect, written into two
    separate slot keys."""
    for lang_field, forms in meaning.pronunciation.items():
        for form, pron in forms.items():
            slot = ensure(lang_field, form)
            _set_rendering_slot(slot, "ipa", pron.get("ipa"))
            _set_rendering_slot(slot, "dialect", pron.get("dialect"))


def _fill_reader_pronunciations(
    by_lang_form: dict[str, dict[str, dict[str, str]]],
) -> None:
    """wyrd-03cx: walk the renderings dict in place, deriving
    reader_pronunciation from ipa via the deterministic anglicize
    mapper. english_shaped (when populated) wins because it's hand-
    curated; an existing reader_pronunciation (theoretically pre-
    populated by a future direct-from-bundle path) also wins.

    Best-effort: silent skip when anglicize_ipa returns None
    (input had no recognizable phonemes after stripping markers)."""
    from wyrd.generators.kenning.runtime.anglicize import anglicize_ipa

    for forms in by_lang_form.values():
        for slot in forms.values():
            if "reader_pronunciation" in slot or "english_shaped" in slot:
                continue
            ipa = slot.get("ipa")
            if not ipa:
                continue
            rendered = anglicize_ipa(ipa)
            if rendered:
                slot["reader_pronunciation"] = rendered


def word_to_key(word):
    elements = []
    for element in word:
        key = [element["location"]]
        if element.get("name", False):
            key.append("name")
        if element.get("saint", False):
            key.append("saint")
        elements.append(key)
    if len(word) == 1:
        elements[0].append("single")
    return tuple(tuple(e) for e in elements)


def load_proportions(data, meaning_db, tag_db, culture: str | None = None):
    usages = data["usages"]
    mg = MeaningGenerator(meaning_db, tag_db, usages)
    single_usages = data["single_usages"]
    mg.load_parts(single_usages, "single")
    # wyrd-rogd.13: per-word-position bare pools. Raw counts from the
    # bundle; the naked↔structured threshold resolves from the env here
    # (load time) so re-tuning needs an env flip + container recycle,
    # never a re-export. Absent key (legacy bundle) → no wp buckets →
    # the vector path's fallback keeps today's behavior.
    mg.load_bare_word_positions(
        data.get("bare_word_positions") or {},
        single_usages,
        bare_position_threshold(),
    )
    structures = data["structures"]
    struct = {}
    for element in structures:
        proportion = element["proportion"]
        words = tuple(word_to_key(w) for w in element["words"])
        # wyrd-c6o1.5: ALL structures load here; the operator allowlist
        # (data/structures.yaml, applied in NameGenerator) is now the single
        # structure filter. The old wyrd-g1hj single-morpheme exclusion ("<Bare>")
        # is migrated there — the lone-dictionary-word structures present in the
        # inventory ((bare), (bare[name])) ship disabled in structures.yaml. (Its
        # empty-structs guard moved too: NameGenerator raises if the allowlist +
        # grammaticality gate leave structs empty.)
        struct[words] = proportion
    # wyrd-mj2: tag-level co-occurrence — empirical bigram statistics over
    # the (left.tags × right.tags) cartesian product learned from each
    # culture's place-name corpus. Optional: legacy bundles without this
    # key produce a no-op cohesion knob. wyrd-e2b4: ``tag_marginal`` (also
    # shipped in the bundle) is the normalizing denominator for the nested
    # conditional-probability table NameGenerator derives for cohesion.
    cooccurrence = data.get("tag_cooccurrence", {})
    tag_marginal = data.get("tag_marginal", {})
    # Union the two attested-usage sources so the vector path filters
    # by "anything this culture's corpus attests" rather than
    # "compound usages only" or "standalone usages only".
    # ``or None`` collapses an empty union to ``None`` so a degenerate
    # bundle (a culture whose proportions are both empty) disables the
    # filter rather than silently emptying the entire native pool —
    # an empty frozenset is truthy under the downstream
    # ``is not None`` guard.
    # wyrd-eyjk/D40: proportions usages are POSITION-FORMS (`-giles`, `Stoke-`,
    # bare `giles`); the vector path compares this against each meaning_db
    # entry's STORED dash-variant key (`Giles-`). Collapse to bare SURFACE on
    # this side so the cross-mode culture filter matches by morpheme identity,
    # not by which dash-shape happened to be recorded vs stored (build_non_
    # position_eligible strips the meaning_db side to match).
    culture_attested_usages = (
        frozenset(u.lower().replace("-", "") for u in usages)
        | frozenset(u.lower().replace("-", "") for u in single_usages)
    ) or None
    # wyrd-pfoo: per-Meaning attestation per culture. Bundle ships
    # this as ``{usage_key: [primary_language, ...]}``; we frozenset
    # the per-usage language sets so the runtime filter does cheap
    # ``in`` lookups. Legacy bundles built pre-wyrd-pfoo carry no
    # ``attested_languages`` key → None falls back to the per-usage
    # filter only.
    # wyrd-c6o1.3: keyed by bare SURFACE, union-merged across dash-variants
    # — the same fold the per-usage filter above applies. The bundle records
    # attestation under the position-forms the corpus walk saw (`-ton`), but
    # meaning_db stores every dash-variant key (the bare `ton`, `Ton-`,
    # `-ton-`); an exact-key lookup left those variants un-narrowed, so
    # wrong-language homographs (welsh `ton` 'wave' in an english request)
    # slipped into the native pool through the variant keys.
    raw_attested = data.get("attested_languages") or {}
    folded_attested: dict[str, frozenset[str]] = {}
    for k, v in raw_attested.items():
        surface = k.lower().replace("-", "")
        folded_attested[surface] = folded_attested.get(surface, frozenset()).union(v)
    culture_attested_meanings: dict[str, frozenset[str]] | None = folded_attested or None
    return NameGenerator(
        meaning_db,
        mg,
        struct,
        cooccurrence,
        culture_attested_usages=culture_attested_usages,
        culture_attested_meanings=culture_attested_meanings,
        tag_marginal=tag_marginal,
        culture=culture,
    )


def weighted_choice(rng: random.Random, choices):
    """Like rng.choice, but each element can have a different weight.

    `choices` is an iterable of (item, weight) pairs. Items with weight <= 0
    are skipped. Returns None if no items have positive weight.
    """
    space = {}
    current = 0
    for choice, weight in choices:
        if weight > 0:
            space[current] = choice
            current += weight
    if current == 0:
        return None
    rand = rng.uniform(0, current)
    last_choice = None
    for key in sorted(list(space.keys()) + [current]):
        if rand < key:
            return last_choice
        if key in space:
            last_choice = space[key]
    return last_choice
