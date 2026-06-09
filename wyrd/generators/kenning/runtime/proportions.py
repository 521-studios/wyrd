"""Weighted random selection over learned morpheme proportions.

Ported from Rando (rando/proportions.py) — Python 2 → 3.

Key change beyond syntax: weighted_choice now takes an explicit Random instance
so generation is reproducible from the seed param.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Callable

from ..era.cells import family_stage_order, language_family
from .meaning import _mimic_case
from .word import _position_form

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


def _flatten_struct_slots(candidate_struct):
    """Flatten a struct (tuple-of-words, each word a tuple-of-keys) into parallel
    position / qualifier / bucket-key lists for the vector primitive. Each key's
    element 0 is the position label (per ``word_to_key``); a ``name`` / ``saint``
    flag becomes the qualifier; the FULL key is the bucket key — preserving the
    ``single`` flag ``word_to_key`` adds so the per-slot frequency lookup hits the
    exact bucket the proportions sample from."""
    positions: list[str] = []
    qualifiers: list[str | None] = []
    bucket_keys: list[tuple] = []
    for word in candidate_struct:
        for key in word:
            positions.append(key[0])
            if "name" in key:
                qualifiers.append("name")
            elif "saint" in key:
                qualifiers.append("saint")
            else:
                qualifiers.append(None)
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


def _merge_morpheme_meanings(meanings: list):
    """Collapse the Meanings sharing one morpheme_id (the connective-form
    fragments — ``-ing``/``-ing-``/``Ing-``) into a single Meaning whose
    ``era_reflexes`` + ``era_reflex_glosses`` are the union across the group, so
    the era-grid reflects the whole morpheme rather than one surface variant.

    **Genuine reflexes win; self-seeds are a fallback (wyrd-5olv).** A
    ``morpheme_id`` groups not only spelling/position variants of the morpheme
    (``-ing``/``Ing-``) but every usage CONSTRUCTED with it as a head: every
    ``-ton`` town (``-bolton``/``-newton``/...) carries
    ``morpheme_id=old-english:tūn`` because Wiktionary records the town's
    etymology as an inheritance edge FROM ``tūn`` — a word built *with* the
    etymon, recorded as a reflex *of* it (bad upstream data). Each such usage
    self-seeds its OWN surface (``source='self'``, the mook self-seed) into
    ``era_reflexes``, so a naive union floods tūn's modern era-grid with ~120
    town surfaces instead of its ~10 attested spellings.

    So per language: when the group carries ANY genuine — non-``self``:
    cluster / descent / period-form / phonology-rule — reflex, keep ONLY those
    (the attested spellings of the morpheme); fall back to the self-seeds for a
    language only when it has no attestation at all. The fallback is what keeps
    an all-self connective morpheme intact — ``-ing`` has no cluster/descent
    reflexes, so its self-seeded variants (including the legitimate ``-ling``)
    are retained rather than stranded. This deliberately avoids a
    compound-detector heuristic, which can't tell a real derivational suffix
    (``-ling``) from a constructed compound (``-bolton``) and would eat both.

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
    # Collect non-``self`` ("genuine", attestation-backed) reflexes separately
    # from self-seeds per language, so genuine can win over self-seeds below.
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

    # wyrd-phww: UNION self-seeds with genuine reflexes per language — genuine
    # listed first, source preserved; a self-seed is added only for a form no
    # genuine reflex already covers. The pre-fix genuine-WINS rule discarded ALL
    # self-seeds whenever any genuine reflex existed, which dropped the GENERATED
    # SURFACE (the attested modern toponym form — 'Park-'/'Paddock-' for pearroc)
    # from the grid -> "X not in its own reflexes". The flood risk the old rule
    # guarded against (every constructed -ton town self-seeds its whole surface,
    # ~120 for tūn) is handled downstream: _era_grid narrows the PRESENT-DAY
    # stage to the word's OWN surface (generate = the generated surface,
    # explain = the decomposed input surface), so only the one relevant
    # self-seed survives next to the genuine reflexes — never the ~120.
    def _union_lang(genuine: dict, selfs: dict) -> list:
        out = dict(genuine)  # genuine first, source preserved
        for form, entry in selfs.items():
            out.setdefault(form, entry)  # self-seed only if no genuine form collides
        return list(out.values())

    merged_reflexes = {
        lang: _union_lang(nonself.get(lang, {}), selfseed.get(lang, {}))
        for lang in sorted(nonself.keys() | selfseed.keys())
    }
    # Keep glosses only for forms that survived the genuine-wins filter; drop a
    # language whose every glossed form was a filtered-out self-seed (omit it
    # rather than leave an empty map).
    kept = {lang: {form for form, _src in entries} for lang, entries in merged_reflexes.items()}
    merged_glosses: dict[str, dict[str, str]] = {}
    for lang, gloss in glosses.items():
        surviving = {f: g for f, g in gloss.items() if f in kept.get(lang, ())}
        if surviving:
            merged_glosses[lang] = surviving
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
    """The position a position-form usage encodes, from its dashes — mirrors
    ``Meaning._set_location``: ``-x-`` inner, ``x-`` pre, ``-x`` post, bare."""
    lead = usage.startswith("-")
    trail = usage.endswith("-")
    if lead and trail:
        return "inner"
    if trail:
        return "pre"
    if lead:
        return "post"
    return "bare"


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
        # wyrd-eyjk/D40: proportions usages are POSITION-FORMS — a morpheme's
        # bare surface rendered at its DERIVED position (`-giles` post,
        # `Stoke-` pre, `giles` bare). The morpheme is resolved by its
        # dash-stripped SURFACE (the bundle may store it only under a different
        # dash-variant, e.g. saints have `Giles-`/`Giles` but never `-giles`),
        # and bucketed by the position the FORM encodes plus the morpheme's
        # name/saint flag. Position is NEVER read off the stored variant's
        # ``Meaning.location`` — it is carried by the recorded form. This
        # retires the old per-Meaning-location bucketing and the wyrd-5z5j
        # ``add_bare`` double-register (the lone pool's forms are already bare).
        #
        # wyrd-van9: a usage whose surface isn't in meaning_db at all is
        # skipped + counted (bundle/proportions drift) rather than crashing.
        index = self._surface_index()
        missing_count = 0
        for usage, proportion in proportions.items():
            meanings = index.get(usage.lower().replace("-", ""))
            if meanings is None:
                missing_count += 1
                continue
            position = _location_from_form(usage)
            # Flags (name/saint) come from the morpheme; position from the form.
            # wyrd-eyjk/D40 + wyrd-g1hj: a pure proper noun that is a SAINT
            # subject (`Mary`/`John`/`Giles` from the St-dedication list) or a
            # personal GIVEN name (`John`/`Edmund` male/female-name etymons)
            # must NOT enter the empirical base pool — both dangle as bare
            # personal names in plain generation (`Mary Margaret`, `By St Mary`,
            # `Leigh Alton Chapel John`). Saint subjects reach names only via the
            # param-gated St-dedication affix; given names need composition
            # (`Edmund`+`-ton`) which the matcher still mines. NOT excluded: a
            # place element merely co-tagged with a name (`stān`+`Stan` — not
            # pure), or a FAMILY-name etymon (`Smith`/Norman manorial families —
            # those form legitimate toponyms + are what the synthetic test
            # fixtures use), so neither realism nor the fixtures are disturbed.
            # wyrd-gwj3: also drop connector morphemes (cum / le / juxta …) —
            # they require a complement and must not dangle as a lone word.
            keys = {
                (position, *m.key()[1:])
                for m in meanings
                if not (m.is_pure_proper_noun() and (m.is_saint() or m.is_given_name()))
                and not m.is_connector_particle()
            }
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


def _is_single_morpheme_structure(struct_key: tuple) -> bool:
    """wyrd-g1hj: True when a structure is a SINGLE MORPHEME total — one word
    of one morpheme (``((("bare","single"),),)`` / ``((("bare","name","single"),),)``).

    Such a name renders as a lone dictionary word — `Old`, `In`, `St`, `Cum`,
    `Green`, `Bath`, `Village` — and even the legit single-etymon ones
    (`Chislehurst`, `Enfield`) read as flat, uninteresting place names. Real
    interest comes from COMPOSITION: a multi-morpheme word (`Higham` =
    `High-`+`-ham`) or multiple words (`Green Park`, 2 morphemes). So these are
    filtered out at LOAD time (``load_proportions``), before the NameGenerator
    is built — they're absent from ``structs``, so the vector generation path
    can't emit them. This is a generation-time exclusion, NOT a mining/bundle
    change: the on-disk proportions still RECORD these structures, so no
    re-export is needed. A single bare word standing INSIDE a larger structure
    (`Green` in `Green Park`) is unaffected — that's 2 morphemes total."""
    return sum(len(word_key) for word_key in struct_key) <= 1


def _era_form_for_meanings(meanings, era_render_language: str | None) -> str | None:
    """wyrd-6c8x: the era-appropriate form for a usage's resolved senses, or
    None when none of them carries a reflex at ``era_render_language``.

    Picks the first sense that has a reflex, and from it prefers an ATTESTED
    form: the reflex list is sorted and the ``*`` reconstruction marker sorts
    before letters, so a naive ``forms[0]`` is biased toward reconstructed
    (unattested) forms — we take the first non-starred form, falling back to the
    marker-stripped first form only when every reflex is reconstructed (still a
    plausible period spelling). Deterministic: no rng draw."""
    for meaning in meanings:
        forms = meaning.era_reflex_for(era_render_language)
        if not forms:
            continue
        attested = [f for f in forms if not f.startswith("*")]
        return attested[0] if attested else forms[0].lstrip("*")
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

    wyrd-rogd.7: each cell carries a stable ``id`` (``lang:index`` — a language
    is unique per stage within a morpheme's grid, so this uniquely identifies
    the cell). The SPA tracks the SELECTED variant by this id rather than by
    surface-fold, which collides for forms that fold equal (bǣre/bære, *ur/ur)
    and lit up two cells at once."""
    sources = meaning.era_reflex_sources_for(lang)
    glosses = meaning.era_reflex_gloss_for(lang)
    forms = []
    for i, form in enumerate(meaning.era_reflex_for(lang)):
        cell = _era_form_cell(meaning, renderings, lang, form, sources, glosses)
        cell["id"] = f"{lang}:{i}"
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
        sections = _narrow_present_day_self_seeds(sections, own_surface)
    return sections


def _narrow_present_day_self_seeds(sections: list[dict], own_surface: str) -> list[dict]:
    """wyrd-phww: in each family's PRESENT-DAY stage, drop ``source='self'`` cells
    that don't match ``own_surface`` (keep every genuine cell + the matching
    self-seed). Non-present-day stages are untouched, so an own-form self-seed in
    its own era (OE ``pearroc``) survives. Stages/sections emptied by the filter
    are dropped."""
    key = _grid_surface_key(own_surface)
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
                    if c.get("source") != "self" or _grid_surface_key(c["form"]) == key
                ]
                if not forms:
                    continue  # present-day stage emptied (only unmatched self-seeds)
                stage = {**stage, "forms": forms}
            stages.append(stage)
        if stages:
            out.append({"family": section["family"], "stages": stages})
    return out


class NameGenerator:
    def __init__(
        self,
        meaning_db,
        meaning_gen,
        structs,
        tag_cooccurrence: dict[str, int] | None = None,
        culture_attested_usages: frozenset[str] | None = None,
        culture_attested_meanings: dict[str, frozenset[str]] | None = None,
    ):
        self.meaning_db = meaning_db
        self.meaning_gen = meaning_gen
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
        # filter admitted as collateral damage. Keys are usage_keys;
        # values are frozensets of primary-language bundle-field
        # names. ``None`` (legacy bundles built pre-wyrd-pfoo) falls
        # back to the per-usage filter only.
        self.culture_attested_meanings: dict[str, frozenset[str]] | None = culture_attested_meanings
        # Cache the vector-path's non-position eligibility pool keyed
        # by (era_min, era_max, stratum, exclude_tags, packs_signature).
        # The dispatch loop calls ``select_via_vector`` once per
        # sub-seed; without this cache, the O(N=meaning_db_size) scan
        # rebuilds on every sub-seed and dominates count=N vector
        # latency (~300ms per sub-seed × N). With the cache, only the
        # first sub-seed pays the build; subsequent sub-seeds reuse.
        # Process-lifetime safe: meaning_db is immutable post-load.
        self._vector_eligible_cache: dict[tuple, list] = {}
        # Cache per-slot ``(meaning, base_score)`` lists keyed by
        # ``(eligible_cache_key, slot_position, slot_qualifier,
        # slot_bucket_key, request_signature, era_midpoint,
        # id(priors))``. Base scores depend only on request +
        # (slot_position, slot_qualifier, slot_bucket_key) +
        # era_midpoint + priors (all constant across sub-seeds in
        # one count=N dispatch); only the cohesion-multiplier +
        # weighted-sample steps vary per sub-seed. Caching the base
        # scores collapses the O(P) score loop per slot from "every
        # sub-seed" to "once per (request, slot) pair". Per-dispatch
        # the inner dict is fresh; across dispatches the outer cache
        # reuses by request signature so repeated identical requests
        # hit warm scores.
        #
        # Inner-key history: wyrd-izcr extended ``slot_position`` to
        # ``(slot_position, slot_qualifier)`` so one dispatch could
        # mix ``(pre, None)`` and ``(pre, "name")`` slots without
        # shadowing. wyrd-bol9 added ``slot_bucket_key`` so single-
        # element and multi-element bucket variants of the same
        # (position, qualifier) — which multiply by different per-
        # bucket empirical frequencies — get distinct cached lists.
        self._vector_slot_score_cache: dict[tuple, dict[tuple, list]] = {}
        # Bound the per-slot cache to avoid unbounded growth across a
        # warm Lambda's lifetime. Distinct request signatures grow
        # linearly with operator-knob diversity (mood / harshness /
        # tags / weights / era / stratum / packs). 16 distinct
        # (request, era_midpoint, priors) tuples covers typical SPA
        # exploration; older entries get FIFO-evicted on overflow.
        self._VECTOR_SLOT_CACHE_MAX = 16
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
        self.structs = {k: v for k, v in structs.items() if is_structurally_grammatical(k)}
        # Loud-failure guard (generator-contract-reviewer P2, round 1):
        # if the filter empties an otherwise-non-empty structs dict, the
        # vector path would crash on a None struct from weighted_choice(rng,
        # []). Raise here with an operator-attributable message instead.
        if structs and not self.structs:
            raise ValueError(
                "NameGenerator: every structure was filtered as ungrammatical "
                "(wyrd-zzli) — bundle has no shipped templates the runtime "
                "can use. Re-emit the bundle via `wyrd kenning lexicon "
                "rebuild-proportions` (which also applies the filter at "
                "emission time)."
            )
        # Bit-stability drift warning (generator-contract-reviewer P2,
        # round 1): when the filter drops something, seeded callers
        # against the same shipped bundle get different output than they
        # did pre-fix. Emit one stderr line per construction so operators
        # see that their bundle is stale rather than discovering the
        # drift via diverging SPA explainer / share-link output. Silent
        # when the filter is a no-op (clean bundle).
        dropped = len(structs) - len(self.structs)
        if dropped:
            dropped_weight = sum(structs[k] for k in structs if k not in self.structs)
            total_weight = sum(structs.values()) or 1
            _logger.warning(
                "wyrd-zzli: NameGenerator: filtered %d ungrammatical structure "
                "templates (%.1f%% of structure weight); re-emit the bundle to "
                "clear the drift",
                dropped,
                100 * dropped_weight / total_weight,
            )
        # wyrd-mj2 (D17 β-term per the ticket reframe): tag-level
        # bigram statistics from each culture's place-name corpus.
        # ``tag_cooccurrence`` keys are "left|right" tag pairs; values
        # are co-occurrence counts. Threaded to the vector path's
        # cohesion multiplier. Optional — legacy bundles without it
        # produce a no-op cohesion knob (cohesion=0 takes the bit-stable
        # path regardless).
        self.tag_cooccurrence = tag_cooccurrence or {}
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
            era_midpoint: int year for the baseline-axis lookup.
            cohesion: D17 cohesion knob in [0, 1]. Threads through
                to the vector primitive's :func:`_cohesion_multiplier`.
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

        Cohesion uses the simple-overlap form from
        ``vector_name_select._cohesion_multiplier``; the bundle's
        tag-cooccurrence data is threaded through to it.
        """
        # Lazy import: no actual cycle (vector_name_select imports
        # only from vectors.{schemas,scoring}, not runtime); the lazy
        # form also keeps module-load cost off callers that never
        # generate.
        from wyrd.generators.kenning.runtime.vector_name_select import (
            select_via_vector_scoring,
        )

        items = list(self.structs.items())
        # The eligibility pool + per-slot base-score map are cached (folding every
        # pool-membership filter into the key); see _vector_caches. exclude_tags_fz
        # is also threaded into the per-slot _score below.
        exclude_tags_fz = frozenset(exclude_tags)
        non_position_eligible, slot_base_scores = self._vector_caches(
            request,
            exclude_tags_fz,
            pack_meaning_dbs,
            include_unglossed,
            era_midpoint,
            priors,
        )

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
                tag_cooccurrence=self.tag_cooccurrence or None,
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

        new_name = NewName(struct, self.meaning_db, words)
        # wyrd-nbpw/6c8x: post-pick rendering — era-form (feature A) or the D8
        # inflection / D18 spelling-variant substitution — applied via
        # _apply_render. Default generation (all knobs 0, no era) stays
        # bit-stable: _apply_render is a no-op that leaves new_name.rendered None.
        self._apply_render(rng, new_name, spelling_variety, inflection_density, era_render_language)
        return new_name

    def _vector_caches(
        self,
        request,
        exclude_tags_fz: frozenset[str],
        pack_meaning_dbs: dict | None,
        include_unglossed: bool,
        era_midpoint: int,
        priors,
    ) -> tuple[list, dict]:
        """Get-or-build the cached non-position eligibility pool + per-slot
        base-score map for one vector dispatch.

        The eligibility cache key folds every filter that affects pool
        membership (era / stratum / exclude-tags / pack overlays via the
        ``request.packs`` tuple + ``id(pack_meaning_dbs)`` / include_unglossed).
        The slot-base-score cache adds the request signature, era_midpoint, and
        ``id(priors)`` — sub-seeds within one dispatch share it and lazy-fill the
        slot_position → base_scored map. Bounded FIFO eviction keeps the slot
        cache from growing unbounded over a warm-Lambda lifetime."""
        from wyrd.generators.kenning.runtime.vector_name_select import (
            build_non_position_eligible,
            request_signature,
        )

        gate = request.gate
        cache_key = (
            gate.era_min,
            gate.era_max,
            gate.stratum,
            exclude_tags_fz,
            request.packs,
            id(pack_meaning_dbs),
            include_unglossed,
        )
        non_position_eligible = self._vector_eligible_cache.get(cache_key)
        if non_position_eligible is None:
            non_position_eligible = build_non_position_eligible(
                self.meaning_db,
                gate=gate,
                exclude_tags=exclude_tags_fz,
                pack_meaning_dbs=pack_meaning_dbs,
                packs=request.packs,
                culture_attested_usages=self.culture_attested_usages,
                culture_attested_meanings=self.culture_attested_meanings,
                include_unglossed=include_unglossed,
            )
            self._vector_eligible_cache[cache_key] = non_position_eligible

        slot_cache_key = (cache_key, request_signature(request), era_midpoint, id(priors))
        slot_base_scores = self._vector_slot_score_cache.get(slot_cache_key)
        if slot_base_scores is None:
            slot_base_scores = {}
            if len(self._vector_slot_score_cache) >= self._VECTOR_SLOT_CACHE_MAX:
                # FIFO eviction: dict iteration order is insertion order in
                # Python 3.7+, so next(iter(...)) is the oldest key.
                oldest = next(iter(self._vector_slot_score_cache))
                del self._vector_slot_score_cache[oldest]
            self._vector_slot_score_cache[slot_cache_key] = slot_base_scores
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
        self, rng, new_name, spelling_variety, inflection_density, era_render_language
    ):
        """Post-pick render dispatch, used by select_via_vector().

        Precedence (wyrd-6c8x): when an era render-language is set, render each
        morpheme in its era-appropriate attested form — the era reflex IS the
        period spelling, so it supersedes the D18 spelling-variant axis (and D8
        inflection is not combined with era in v1). Otherwise fall back to the
        D8/D18 substitution. Either way the result is written to
        ``new_name.rendered`` (+ ``inflection_labels`` for D8). A no-op — leaving
        ``new_name.rendered`` as None so __str__ uses the canonical reflex — when
        no era is set and both substitution knobs are 0, keeping default
        generation bit-stable (no extra rng draws)."""
        if era_render_language:
            new_name.rendered = self._render_era_forms(new_name.name, era_render_language)
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

    def _render_era_forms(self, name, era_render_language):
        """wyrd-6c8x (feature A): render each picked morpheme in its era-
        appropriate attested form for ``era_render_language`` (the canonical
        etymon-language of the requested era cell) instead of the modern
        canonical reflex, so ``era=oe-early`` produces period-LOOKING names.

        Returns a ``NewName.rendered``-shaped list (parallel to ``name``): the
        era form, case-projected onto the slot via ``_mimic_case``, where the
        bundle carries a reflex for the target language; else None — which
        ``NewName.__str__`` renders as the canonical usage. Empty-reflex
        morphemes (the ~10% with no era data) thus fall back to the modern
        form, matching the rewind 'no era data → canonical' convention.

        Deterministic + bit-stable: resolves each usage to its senses by bare
        surface (mirroring ``_render_substitutions``, since usages are
        position-forms like ``-ton``) and delegates form selection to
        ``_era_form_for_meanings`` (attested-over-reconstructed, no rng draw).
        A usage with no reflex renders as None — ``NewName.__str__`` then falls
        back to the canonical usage (the ~10% with no era data)."""
        rendered: list[list[str | None]] = []
        for word in name:
            word_rendered: list[str | None] = []
            for usage in word:
                if usage is None:
                    word_rendered.append(None)
                    continue
                meanings = (
                    self.meaning_gen._surface_index().get(usage.lower().replace("-", "")) or []
                )
                form = _era_form_for_meanings(meanings, era_render_language)
                word_rendered.append(_mimic_case(usage, form) if form else None)
            rendered.append(word_rendered)
        return rendered


class NewName:
    def __init__(self, struct, meaning_db, name, rendered=None, inflection_labels=None):
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
        """wyrd-vd6y + wyrd-72q9: when the SAME surface is selected for two+
        slots ("Hill Hill"), resolve the later occurrence(s):
          1. prefer a same-meaning synonym in a DIFFERENT language (cf. "Table
             Mesa"): "Hill Hill" → "Hill Haeth" (render override); else
          2. re-pick a DIFFERENT same-position morpheme: "Park ... Park" →
             "Park ... <other>" (replace the name key); else
          3. leave the duplicate (nothing else of that position available).
        Deterministic (no rng — synonym pick is ranked, re-pick is a stable
        md5 of the name) so it's seed-stable and only touches repeated names.
        """
        if not self.meaning_db or not self.name:
            return  # nothing to resolve siblings against (e.g. render-only tests)
        from wyrd.generators.kenning import _rank_siblings

        # wyrd-72q9: stable per-name signature for varied-but-deterministic
        # re-picks (hash the name, NOT the rng, so it stays seed-stable).
        name_sig = "|".join("/".join(c for c in w if c) for w in self.name)

        seen: dict[str, set[str]] = {}  # folded surface -> languages already used
        for wi, word in enumerate(self.name):
            for ei, usage in enumerate(word):
                if usage is None:
                    continue
                surface = (self.rendered[wi][ei] if self.rendered else None) or usage
                fold = surface.replace("-", "").lower()
                siblings = _rank_siblings(_resolve_surface(self.meaning_db, usage))
                canon = siblings[0] if siblings else None
                if fold not in seen:
                    seen[fold] = self._meaning_langs([canon]) if canon else set()
                    continue
                # Repeat: resolve it (cross-language synonym → re-pick → leave).
                self._resolve_repeat(wi, ei, usage, fold, canon, siblings, seen, name_sig)

    def _resolve_repeat(self, wi, ei, usage, fold, canon, siblings, seen, name_sig) -> None:
        """Resolve one repeated surface (wyrd-vd6y / wyrd-72q9): prefer a
        same-meaning synonym in a DIFFERENT language ("Hill Hill" → "Hill Haeth",
        a render override); else re-pick a DIFFERENT same-position morpheme
        ("Park ... Park" → "Park ... <other>", replacing the name key); else
        leave the duplicate. Updates ``seen`` in place with whatever fold /
        language the resolution introduces. Deterministic — no rng."""
        from wyrd.generators.kenning import _rank_siblings

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
            seen.setdefault(form.strip("-").lower(), set()).add(lang)
            return
        repl = self._repick_nondup(usage, set(seen.keys()), name_sig, wi, ei, canon)
        if repl is not None:
            self.name[wi][ei] = repl
            if self.rendered is not None:
                self.rendered[wi][ei] = None
            if self.inflection_labels is not None:
                self.inflection_labels[wi][ei] = None
            repl_sibs = _rank_siblings(_resolve_surface(self.meaning_db, repl))
            seen[repl.strip("-").lower()] = (
                self._meaning_langs([repl_sibs[0]]) if repl_sibs else set()
            )
        # else: nothing else of this position available → leave the dupe.

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
        subset additionally sharing a thematic tag with the original."""
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
            kf = k.strip("-").lower()
            if not kf or kf in seen_folds:
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
        else:
            ranked = _rank_siblings(_resolve_surface(self.meaning_db, e))
            usage_str = e
        first = ranked[0] if ranked else None
        morpheme: dict = {"usage": usage_str}
        if first is not None:
            self._add_etymology_fields(morpheme, ranked, first)
        self._add_rendered_fields(morpheme, wi, ei, first)
        return morpheme

    def _add_etymology_fields(self, morpheme: dict, ranked: list, first) -> None:
        """Add the etymology fields for the top-ranked sibling ``first``. Sources
        stay from ``first`` (the canonical etymon for this surface); tags union
        across all ranked siblings (wyrd-ywm9); meanings split into primary +
        derivative for the SPA card layout. meaning_groups / renderings /
        citations / era_grid are sparse — emitted only when non-empty, so
        Latin-script / rando-port-only morphemes don't carry noisy empties."""
        # Lazy import (circular-dep avoidance — see _morpheme_to_dict).
        from wyrd.generators.kenning import (
            _all_senses,
            _meaning_groups,
            _split_senses_for_display,
        )

        morpheme["sources"] = {lang: list(forms) for lang, forms in first.sources.items()}
        morpheme["tags"] = list(dict.fromkeys(t for m in ranked for t in m.tags))
        primary, derivative = _split_senses_for_display(_all_senses(ranked))
        morpheme["meanings"] = primary
        morpheme["derivative_meanings"] = derivative
        # wyrd-0y3k: per-sibling deduped sense groups (the flat `meanings` above
        # stays for back-compat / other consumers).
        groups = _meaning_groups(ranked)
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
        morpheme_meaning = _resolve_morpheme(self.meaning_db, getattr(first, "morpheme_id", None))
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
                else:
                    # wyrd-o53o round 4 (Gemini MED): defensive access
                    # matching to_dict()'s pattern. e SHOULD exist (it
                    # came from the generator's pool) but the consistency
                    # win + no-IndexError-on-empty-siblings is worth the
                    # cost of one .get() + one truthy check.
                    ranked = _rank_siblings(_resolve_surface(self.meaning_db, e))
                    usage_str = e
                if not ranked:
                    continue
                first = ranked[0]
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
                    self.meaning_db, getattr(first, "morpheme_id", None)
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


def load_proportions(data, meaning_db, tag_db):
    usages = data["usages"]
    mg = MeaningGenerator(meaning_db, tag_db, usages)
    single_usages = data["single_usages"]
    mg.load_parts(single_usages, "single")
    structures = data["structures"]
    struct = {}
    for element in structures:
        proportion = element["proportion"]
        words = tuple(word_to_key(w) for w in element["words"])
        # wyrd-g1hj: exclude SINGLE-MORPHEME structures from the loaded
        # generator. A whole name that is one morpheme — `Old`, `In`, `St`,
        # `Cum`, `Green`, `Bath`, `Village` (even legit single-etymon ones like
        # `Chislehurst`) — renders as a flat lone dictionary word; real place
        # names compose (>=2 morphemes: `Higham` = `High-`+`-ham`, or multiple
        # words `Green Park`). The bundle/proportions STILL RECORD these (mining
        # keeps them — this is a generation-time exclusion applied at load, so
        # no re-export is needed); they're simply never loaded into a generator,
        # so the vector generation path can't produce them. A bare word standing
        # INSIDE a larger structure (`Green` in `Green Park`) is unaffected —
        # that structure has 2 morphemes total.
        if _is_single_morpheme_structure(words):
            continue
        struct[words] = proportion
    # wyrd-g1hj loud-failure guard: the single-morpheme filter above runs
    # BEFORE NameGenerator, so its empty-structs guard (which checks
    # ``if structs and not self.structs``) can't see this case — a culture
    # whose structures are ALL single-morpheme would yield an empty ``struct``,
    # and the vector path would later get a None struct from
    # weighted_choice([]). Fail loud + attributable here instead.
    if structures and not struct:
        raise ValueError(
            "load_proportions: every structure was a single morpheme (wyrd-g1hj) "
            "— this culture has no multi-morpheme templates to generate from. "
            "Re-emit the bundle (the proportions still record them) or relax the "
            "single-morpheme exclusion."
        )
    # wyrd-mj2: tag-level co-occurrence — empirical bigram statistics over
    # the (left.tags × right.tags) cartesian product learned from each
    # culture's place-name corpus. Optional: legacy bundles without this
    # key produce a no-op cohesion knob.
    cooccurrence = data.get("tag_cooccurrence", {})
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
    raw_attested = data.get("attested_languages") or {}
    culture_attested_meanings: dict[str, frozenset[str]] | None = {
        k: frozenset(v) for k, v in raw_attested.items()
    } or None
    return NameGenerator(
        meaning_db,
        mg,
        struct,
        cooccurrence,
        culture_attested_usages=culture_attested_usages,
        culture_attested_meanings=culture_attested_meanings,
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
