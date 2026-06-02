"""Weighted random selection over learned morpheme proportions.

Ported from Rando (rando/proportions.py) — Python 2 → 3.

Key change beyond syntax: weighted_choice now takes an explicit Random instance
so generation is reproducible from the seed param.
"""

from __future__ import annotations

import logging
import random
import re
from functools import lru_cache

from .meaning import _mimic_case
from .word import _position_form

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


class Generator:
    def __init__(self, tag_db, elements):
        self.tag_db = tag_db
        self.elements = elements

    def add_item(self, key, proportion):
        self.elements[key] = proportion

    def select(
        self,
        rng,
        *tags,
        novelty: float = 0.0,
        harshness: float = 0.0,
        exclude_tags: tuple[str, ...] = (),
        exclude_keys: frozenset[str] | None = None,
        keep_keys: frozenset[str] | None = None,
        key_boost: dict[str, float] | None = None,
    ):
        """Pick one element, optionally blending toward a uniform marginal
        and/or biasing toward phonologically-harsh keys.

        Per D17, the runtime samples from a mixture
        ``(1-novelty)·empirical + novelty·uniform``: at ``novelty=0`` (default)
        sampling is pure empirical-frequency (current behavior); at
        ``novelty=1`` every key in the bucket is equally likely. Intermediate
        values softly blend, allowing plausible-but-unattested combinations
        without abandoning the corpus. The D17 β-term (tag-class-prior) is
        realized via the multiplicative ``key_boost`` parameter — wyrd-mj2's
        cohesion knob threads neighbor-context through the structure walk
        in NameGenerator and computes the per-key prior from the tag
        co-occurrence model; that boost multiplies empirical weights here
        before novelty blends with the uniform marginal. See DECISIONS.md
        D17 for the realized-vs-textbook math.

        Per D6, ``harshness`` re-weights each candidate by its phonological
        harshness score (computed from the key string) — at ``harshness=0``
        every key keeps its empirical weight; at ``harshness=1`` soft-keyed
        items go to zero and harsh-keyed items keep 2× their original weight.
        Composition order: harshness is applied first to the empirical
        weights, then ``_blend_uniform`` blends the result with the uniform
        marginal. So ``--harsh 1 --novelty 1`` yields uniform sampling over
        the entire bucket (because ``_blend_uniform`` ignores the empirical
        side at novelty=1), while ``--harsh 1 --novelty 0`` is pure
        harsh-empirical.

        ``exclude_tags`` (wyrd-yan) drops keys whose usage carries any of
        the named tags. Applied AFTER the positive ``tags`` include-filter,
        so a fiction-tagged usage is removed even if the caller asked for
        a tag the usage also carries. The default ``()`` is a no-op for
        bit-stable historical behavior.

        ``keep_keys`` (D5-2 era filter, wyrd-lyp) restricts the bucket to
        only the named usages. None means 'no filter' (bit-stable). The
        caller (typically MeaningGenerator) precomputes the allowed-usage
        set from the meaning_db once per ``--era`` value and passes the
        same frozenset for every subsequent select call, so the per-bucket
        intersection here is a fast O(bucket-size) walk.

        ``key_boost`` (wyrd-mj2 cohesion) is a per-key multiplier applied
        to empirical weights before harshness/novelty composition. The
        caller (typically NameGenerator) precomputes it per slot from the
        tag co-occurrence model conditioned on previously-picked usages.
        Missing keys default to 1.0 (no boost). None means 'no boost'
        (bit-stable). Composes multiplicatively with harshness; novelty
        still blends with the uniform marginal LAST so the high-novelty
        path remains a clean tunable.
        """
        items = self.elements.items()
        if len(tags) > 0:
            items = self.filter_for_tag(*tags).items()
        if exclude_tags:
            items = self._apply_excludes(items, exclude_tags).items()
        if keep_keys is not None:
            # wyrd-eyjk/D40: keep_keys holds bare SURFACES; bucket items are
            # position-forms — compare by surface.
            items = [(k, v) for k, v in items if k.lower().replace("-", "") in keep_keys]
        # wyrd-gzvr: exclude_keys drops specific usages from sampling
        # (the previously-picked usage at an adjacent slot, to break
        # the 'North North' / 'Green Green' duplicate-word pattern).
        # Applied AFTER keep_keys + tag filters so the exclusion only
        # narrows the eligible pool — never widens it past the other
        # constraints. None means 'no exclusion' (bit-stable). Empty
        # frozenset is also a no-op (defensive).
        if exclude_keys:
            items = [(k, v) for k, v in items if k not in exclude_keys]
        if len(items) == 0:
            return None
        items_list = list(items)
        if key_boost is not None:
            items_list = [(k, v * key_boost.get(k, 1.0)) for k, v in items_list]
        if harshness <= 0 and novelty <= 0 and key_boost is None:
            return weighted_choice(rng, items_list)
        if harshness > 0:
            items_list = _blend_harsh(items_list, harshness)
        if novelty > 0:
            items_list = _blend_uniform(items_list, novelty)
        return weighted_choice(rng, items_list)

    def has_tag(self, *tags):
        return len(self.filter_for_tag(*tags)) > 0

    def filter_for_tag(self, *tags):
        # Dedupe via set, then sort: the resulting dict's iteration order
        # feeds weighted_choice's cumulative-threshold construction, which
        # is order-dependent. Set iteration is hash-randomized across
        # PYTHONHASHSEED, so without sort() the same (tags, seed) tuple
        # could produce different picks across processes. Same fix shape
        # as wyrd-mj2's _raw_class_score sort. wyrd-8ga.
        # wyrd-eyjk/D40: tag_db is keyed by the bundle's meaning_db usages
        # (dash-variants), but the bucket items (self.elements) are
        # position-forms — match by bare SURFACE so a morpheme tagged under
        # any variant is found at whatever position it was recorded. Iterate
        # sorted(self.elements) for the same order-determinism reason.
        tag_surfaces = {
            u.lower().replace("-", "") for tag in tags for u in self.tag_db.get(tag, [])
        }
        return {
            key: self.elements[key]
            for key in sorted(self.elements)
            if key.lower().replace("-", "") in tag_surfaces
        }

    def _apply_excludes(self, items, exclude_tags: tuple[str, ...]):
        """Drop keys whose usage appears in tag_db under any exclude tag.
        Returns a dict to keep the .items() call site consistent.

        Hot-path short-circuit: if no exclude tag is present in this
        bucket's tag_db, nothing can be filtered — return ``dict(items)``
        directly without the per-key membership check. Today's bundle has
        no fiction-tagged morphemes so this is the steady-state path."""
        # wyrd-eyjk/D40: match by bare SURFACE (tag_db holds dash-variants,
        # items are position-forms).
        excluded: set[str] = set()
        for tag in exclude_tags:
            for u in self.tag_db.get(tag, ()):
                excluded.add(u.lower().replace("-", ""))
        if not excluded:
            return dict(items)
        return {k: v for k, v in items if k.lower().replace("-", "") not in excluded}


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


def _bucket_keys_matching_surface(
    bucket_keys: tuple[str, ...],
    surface: str | None,
) -> frozenset[str] | None:
    """wyrd-gzvr: expand a surface form into the set of bucket keys
    whose dash-stripped lowercased form matches it. Used by
    ``_select_no_tag`` to dedup on surface form (not bucket key)
    so cross-bucket duplicates like '-bridge' (post-marker) /
    '-bridge-' (infix-marker) both get filtered when the previous
    slot rendered 'bridge'.

    Returns ``None`` when ``surface`` is None (no previous pick to
    dedup against) or when no bucket key matches — both paths
    short-circuit ``Generator.select``'s exclude_keys filter to its
    bit-stable no-filter behavior.
    """
    if surface is None:
        return None
    matches = {k for k in bucket_keys if k.replace("-", "").lower() == surface}
    return frozenset(matches) if matches else None


def _gloss_eligible(usage: str, has_gloss: bool, include_unglossed: bool) -> bool:
    """Gloss-policy predicate shared by the proportions keep-key filter
    (:meth:`MeaningGenerator.keep_keys_for_gloss`) and the vector pool
    filter (:func:`vector_name_select.build_non_position_eligible`) so
    both scoring modes apply identical generation-pool rules (wyrd-glos).

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


def _intersect_keep_keys(
    a: frozenset[str] | None,
    b: frozenset[str] | None,
) -> frozenset[str] | None:
    """Compose two keep-key filters (era + stratum, today; any future
    per-call USAGE-level gate would compose the same way). ``None``
    means 'no filter' for either side, so the result follows the
    standard intersection-with-identity rule:

      * both None → None (bit-stable no-filter fast path)
      * one None, other set → the set (single filter wins)
      * both set → frozenset intersection (must clear both gates)

    Returning ``None`` when both filters are absent is what lets
    ``Generator.select`` short-circuit to its bit-stable empirical
    path — see ``keep_keys_for_era`` for the contract.
    """
    if a is None:
        return b
    if b is None:
        return a
    return a & b


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
        # wyrd-cj6f: bucket keys a structure slot referenced but that have
        # no registered Generator (see `select`). Tracked so the drift
        # warning fires once per unique key, not once per generation.
        self._warned_missing_buckets: set[tuple] = set()
        # D5-2 era-filter cache: era_range tuple → frozenset of allowed
        # usages, OR None when the era covers every usage (the
        # bit-stable fast-path signal — see keep_keys_for_era). Keyed by
        # (start, end) so two callers passing the same range share the
        # precomputed value. Computed lazily on first lookup; the
        # meaning_db is immutable post-load so the cache is safe to
        # reuse for the process lifetime.
        self._era_keep_cache: dict[tuple[int | None, int | None], frozenset[str] | None] = {}
        # wyrd-lr4 Phase 3 stratum-filter cache: stratum string →
        # frozenset of allowed usages (or None for the full-coverage
        # / no-filter fast path). Same caching contract as
        # _era_keep_cache; meaning_db is immutable post-load so a
        # single computation per stratum value lasts the process
        # lifetime.
        self._stratum_keep_cache: dict[str, frozenset[str] | None] = {}
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

    def keep_keys_for_era(
        self, era_range: tuple[int | None, int | None] | None
    ) -> frozenset[str] | None:
        """Resolve an ``era_range`` to the set of usages that have at least
        one Meaning admissible under the half-open ``[start, end)`` window.

        Returns ``None`` when ``era_range`` is None — the caller's
        bit-stable 'no filter' signal. Also returns ``None`` when the
        computed keep-set covers EVERY usage in ``meaning_db`` — there's
        nothing to filter, so ``Generator.select`` should take its
        bit-stable fast path rather than walk every bucket through a
        membership check that admits everything. The full-coverage case
        is the steady state today (zero attested_years data → every
        morpheme passes the filter); collapsing it to None preserves
        bit-stability with the no-filter call until the bundle re-emit
        actually populates attestation data for some morphemes.

        The result is cached so a single ``--era`` value across many
        ``select`` calls only walks the meaning_db once.

        Granularity caveat: filtering happens at the USAGE level, not
        the SENSE level. A usage with two senses (one in-era, one out-
        of-era) stays in the pool; the downstream pick at
        ``NameGenerator._pick_surface`` falls back to ``meanings[0]`` for
        variant/inflection rendering, which may surface the wrong-era
        sense. See the existing comment block at ``_render_substitutions``
        for the deterministic-fallback rationale; tightening the filter
        to sense level would need that same call site reworked.
        """
        if era_range is None:
            return None
        if era_range in self._era_keep_cache:
            return self._era_keep_cache[era_range]
        # wyrd-eyjk/D40: keep-sets are keyed by bare SURFACE (the bucket items
        # are position-forms; Generator.select compares the item's surface).
        allowed: frozenset[str] | None = frozenset(
            usage.lower().replace("-", "")
            for usage, meanings in self.meaning_db.items()
            if any(m.attested_in_era_range(era_range) for m in meanings)
        )
        if len(allowed) == len(self._surface_index()):
            allowed = None
        self._era_keep_cache[era_range] = allowed
        return allowed

    def keep_keys_for_stratum(self, stratum: str | None) -> frozenset[str] | None:
        """wyrd-lr4 Phase 3: resolve a stratum tag to the set of usages
        with at least one Meaning admissible under that stratum.

        Mirrors ``keep_keys_for_era`` exactly — same None-on-no-filter,
        None-on-full-coverage signal, same per-process cache. Today
        only Welsh-family etymons carry stratum data, so most usages
        will pass via the 'no stratum data → admit' branch in
        ``Meaning.in_stratum``; the keep-set narrows only those usages
        whose Meaning HAS stratum data and the data doesn't include
        the requested tag. As Phase 4 lands and more languages get
        classified, the set tightens automatically.

        Granularity caveat (same as era): filtering happens at the
        USAGE level, not the SENSE level. A usage with two senses
        (one in-stratum, one out) stays in the pool; the downstream
        pick at ``NameGenerator._pick_surface`` falls back to
        ``meanings[0]`` for variant/inflection rendering, which may
        surface the wrong-stratum sense. Sense-level filtering would
        need ``_render_substitutions`` reworked.
        """
        if stratum is None:
            return None
        if stratum in self._stratum_keep_cache:
            return self._stratum_keep_cache[stratum]
        allowed: frozenset[str] | None = frozenset(
            usage.lower().replace("-", "")
            for usage, meanings in self.meaning_db.items()
            if any(m.in_stratum(stratum) for m in meanings)
        )
        if len(allowed) == len(self._surface_index()):
            allowed = None
        self._stratum_keep_cache[stratum] = allowed
        return allowed

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

        Mirrors ``keep_keys_for_era`` / ``keep_keys_for_stratum``: None
        when the keep-set covers every usage (no-filter fast path), else
        the frozenset; cached per include_unglossed value. Same
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

    def select(
        self,
        rng,
        key,
        *tags,
        novelty: float = 0.0,
        harshness: float = 0.0,
        exclude_tags: tuple[str, ...] = (),
        exclude_keys: frozenset[str] | None = None,
        keep_keys: frozenset[str] | None = None,
        key_boost: dict[str, float] | None = None,
    ):
        # wyrd-cj6f: a structure slot can reference a (position, tag,
        # count) bucket that has no registered Generator — structures are
        # kept in full while the usage(s) that would populate the bucket
        # can be absent (bundle/proportions drift) or trimmed out (the
        # --dev seed keeps top-N usages but all structures). Pre-fix this
        # KeyError'd mid-generation (welsh proportions, seed 806). Treat a
        # missing bucket like an empty one: return None, which the callers
        # (`_select_no_tag` / `_select_tag`) already handle by leaving that
        # element unfilled. Mirrors `load_parts`' wyrd-van9 skip-don't-
        # crash policy and `bucket_keys`' existing `.get()`.
        gen = self.generators.get(key)
        if gen is None:
            # Observability per DECISIONS.md D24: a returned-None here means
            # a structure slot rendered nothing (degraded / empty element).
            # Warn ONCE per unique key (bulk generation calls select per
            # slot per name). This is a DISTINCT drift dimension from
            # `load_parts`' wyrd-van9 warning — that one catches a
            # proportions usage with no Meaning; this catches a structure
            # slot whose bucket was never built — so it needs its own
            # message, not the constructor's.
            if key not in self._warned_missing_buckets:
                self._warned_missing_buckets.add(key)
                _logger.warning(
                    "wyrd-cj6f: MeaningGenerator: structure slot references "
                    "bucket %r with no registered Generator; rendering an "
                    "empty element. Re-emit the bundle to clear the drift.",
                    key,
                )
            return None
        return gen.select(
            rng,
            *tags,
            novelty=novelty,
            harshness=harshness,
            exclude_tags=exclude_tags,
            exclude_keys=exclude_keys,
            keep_keys=keep_keys,
            key_boost=key_boost,
        )

    def bucket_keys(self, key) -> tuple[str, ...]:
        """Return the tuple of usages registered under bucket ``key``.
        Used by NameGenerator to enumerate candidates when computing the
        per-slot cohesion boost. Empty tuple if the bucket doesn't exist
        — caller falls back to the no-boost path."""
        bucket = self.generators.get(key)
        if bucket is None:
            return ()
        return tuple(bucket.elements)


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
    is built — they're absent from BOTH ``structs`` and ``_all_structs``, so no
    generation path (select / vector / force_structure) can emit them. (Unlike
    the grammaticality filter, which runs inside ``NameGenerator.__init__`` and
    leaves its drops in ``_all_structs`` — still forceable.) This is a
    generation-time exclusion, NOT a mining/bundle change: the on-disk
    proportions still RECORD these structures, so no re-export is needed. A
    single bare word standing INSIDE a larger structure (`Green` in
    `Green Park`) is unaffected — that's 2 morphemes total."""
    return sum(len(word_key) for word_key in struct_key) <= 1


class NameGenerator:
    def __init__(
        self,
        meaning_db,
        meaning_gen,
        structs,
        tag_cooccurrence: dict[str, int] | None = None,
        tag_marginal: dict[str, int] | None = None,
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
        # wyrd-5z5j: retain the UNFILTERED structures so list_structures() /
        # force_structure can surface + use shapes the wyrd-zzli filter drops.
        self._all_structs = dict(structs)
        self.structs = {k: v for k, v in structs.items() if is_structurally_grammatical(k)}
        # Loud-failure guard (generator-contract-reviewer P2, round 1):
        # if the filter empties an otherwise-non-empty structs dict, the
        # legacy select() path would crash deep inside _select_no_tag on
        # a None struct from weighted_choice(rng, []). Raise here with an
        # operator-attributable message instead.
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
        # are co-occurrence counts. ``tag_marginal`` keys are tags;
        # values are how often each tag appeared as either side of any
        # pair. Both are optional — legacy bundles without them produce
        # a no-op cohesion knob (cohesion=0 takes the bit-stable path
        # regardless).
        self.tag_cooccurrence = tag_cooccurrence or {}
        self.tag_marginal = tag_marginal or {}
        # wyrd-bol9: per-bucket empirical-frequency lookup for the
        # vector path. Pre-fix, vector mode sampled each Meaning by
        # D36.2 score(lemma) regardless of how often the underlying
        # USAGE actually appeared in this culture's corpus. The pool
        # was uniformly distributed across culture-attested usages —
        # losing the empirical-frequency signal proportions mode
        # carries via its per-bucket weight tables. Welsh / Irish /
        # Breton names came out 25-35pp more OE-dominated than
        # proportions because the OE-heavy attested pool dominated
        # under uniform sampling.
        #
        # Frequency-weighted pool restoration: each Meaning's vector
        # score is composed with its USAGE's per-bucket frequency
        # from the same proportions tables proportions mode samples
        # from. Within a usage's Meanings, D36.2 score discriminates;
        # across usages, frequency drives pick-share — matching
        # proportions' per-usage weighted sampling while preserving
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

    def list_structures(self) -> list[dict]:
        """All structure templates (INCLUDING the wyrd-zzli-filtered ones),
        each as a readable label passable to ``force_structure``, sorted by
        weight. Intended to feed a future SPA Advanced 'force structure'
        dropdown (the CLI/API/SPA wiring is not landed yet — wyrd-5z5j
        force-structure backend; the consumer is a follow-up)."""
        total = sum(self._all_structs.values()) or 1
        return [
            {
                "label": structure_key_to_label(key),
                "weight": weight,
                "proportion": weight / total,
                "words": len(key),
                "grammatical": is_structurally_grammatical(key),
            }
            for key, weight in sorted(self._all_structs.items(), key=lambda kv: -kv[1])
        ]

    @staticmethod
    def _resolve_forced_structure(force_structure: str | tuple | list) -> tuple:
        """A force_structure arg (label string or key tuple) → a struct key.

        A nested-list input (e.g. decoded from a JSON request body) is
        recursively converted to tuples so the result is hashable for the
        dict-key lookups the structure key feeds (Gemini review)."""
        if isinstance(force_structure, str):
            return structure_label_to_key(force_structure)

        def _to_tuple(val):
            if isinstance(val, list):
                return tuple(_to_tuple(x) for x in val)
            return val

        return _to_tuple(force_structure)

    def select(
        self,
        rng,
        *tags,
        spelling_variety: float = 0.0,
        novelty: float = 0.0,
        inflection_density: float = 0.0,
        harshness: float = 0.0,
        exclude_tags: tuple[str, ...] = (),
        era_range: tuple[int | None, int | None] | None = None,
        stratum: str | None = None,
        cohesion: float = 0.0,
        include_unglossed: bool = True,
        force_structure: str | tuple | list | None = None,
    ):
        """Pick a structure, fill it with morpheme usages, optionally render
        each usage as an attested archaic spelling variant (D18) or an
        inflected form (D8), and blend sampling toward a uniform marginal
        (D17) or harsh-keyed empirical (D6).

        ``spelling_variety`` is the probability *per morpheme* that the
        rendered surface form is drawn from the etymon's variant pool
        rather than the canonical reflex.

        ``novelty`` (D17) blends each morpheme-bucket's empirical-frequency
        distribution with a uniform marginal over the same keys.

        ``inflection_density`` (D8) is the probability *per morpheme* that
        the rendered surface form is drawn from the lemma family's
        inflected-children pool ('cot' → 'cotum', 'cotan', 'cotes'). The
        inflection's grammatical-case label is preserved in
        ``new_name.inflection_labels`` for the explainer.

        ``harshness`` (D6) re-weights each morpheme-bucket toward
        phonologically-harsh keys (stop-final, cluster-heavy). Composes
        orthogonally with ``novelty``: the harsh skew applies to empirical
        weights first, then ``novelty`` blends the result with the uniform
        marginal.

        ``exclude_tags`` (wyrd-yan) drops morpheme usages tagged with any
        of the named tags. Threaded down to ``Generator.select`` and
        applied per-bucket. Used by the runtime's fiction gate: 'fiction'
        is excluded by default so realistic-mode generation never draws
        from constructed-etymology entries; the GM opts in via
        ``include_fiction``.

        ``era_range`` (D5-2 / wyrd-lyp) is a half-open ``[start, end)`` year
        window. Morphemes whose attested-year evidence falls outside the
        window are dropped from every bucket; morphemes with no evidence
        pass through (see ``Meaning.attested_in_era_range``). ``None``
        disables the filter — bit-stable behavior. The keep-set is computed
        once per call from ``meaning_gen.keep_keys_for_era`` and reused
        across every per-bucket pick within this name.

        ``stratum`` (wyrd-lr4 Phase 3) restricts the morpheme inventory
        to forms classified into a specific within-language register
        bucket — e.g. 'native-welsh' / 'brittonic-substrate' /
        'medieval-welsh' / 'latin-loan' / 'english-loan'. Morphemes
        with no stratum data pass through (Welsh is the only family
        classified today; routing every culture through a strict gate
        would gut bundles for Latin / OE / French / etc.). ``None``
        disables the filter — bit-stable. Composes with ``era_range``
        via frozenset intersection: a morpheme must clear BOTH gates
        when both are set. Same usage-level granularity caveat as era.

        ``cohesion`` (wyrd-mj2) biases each slot's pick toward usages whose
        tags co-occur with previously-picked slots' tags in the empirical
        corpus. At ``cohesion=0`` (default) every slot samples
        independently from its marginal — bit-stable with the pre-D17 path.
        At ``cohesion=1`` slot-N's empirical weights are scaled by the
        normalized class-conditional likelihood given the union of prior
        slots' tags. Intermediate values blend. Composes orthogonally with
        novelty (which still blends with the uniform marginal LAST) so a
        GM can dial 'attested-pair fidelity' (cohesion) and 'novelty'
        independently. No-op when the bundle carries no tag-cooccurrence
        data (legacy bundles or empty corpora) — ``_cohesion_boost``
        short-circuits to None and ``Generator.select`` takes its
        bit-stable fast path. Cohesion gating happens inside
        ``_cohesion_boost`` (not pre-filtered here) so a single source
        of truth for the no-op decision lives next to the boost
        computation.

        Dilution caveat (multi-tag selection): when ``tags`` carries more
        than one entry, ``_select_tags`` runs an independent pool walk per
        tag and merges via ``rng.choice``. Each pool maintains its own
        ``prior_tags`` accumulator, so cohesion biases within each pool
        but the cross-pool merge undoes some of that bias. Same dynamic
        applies to harshness and novelty — pre-existing pattern, not a
        new regression. A future refactor could share prior_tags across
        pool members, at the cost of breaking the existing per-tag
        independence invariant.

        When both inflection_density and spelling_variety would fire on the
        same morpheme, inflection wins — it carries grammatical meaning
        that the variant axis doesn't.

        ``include_unglossed`` (wyrd-glos) is the gloss policy. ``True``
        (this method's back-compat default) admits unglossed morphemes;
        the Kenning.generate operator surface defaults it to ``False``
        (glossed-only, the rando-era behavior) so the etymology line —
        the load-bearing feature — is always meaningful. Single-char
        UNGLOSSED usages are dropped regardless. Composes with era/stratum
        via the same keep-key intersection.
        """
        keep_keys = _intersect_keep_keys(
            _intersect_keep_keys(
                self.meaning_gen.keep_keys_for_era(era_range),
                self.meaning_gen.keep_keys_for_stratum(stratum),
            ),
            self.meaning_gen.keep_keys_for_gloss(include_unglossed),
        )
        if force_structure is not None:
            # wyrd-5z5j: bypass the weighted sample AND the wyrd-zzli filter so
            # the operator can force any template (incl. filter-dropped shapes).
            struct = self._resolve_forced_structure(force_structure)
            # Fail loud on an unknown/typo'd template rather than silently
            # filling every slot from missing buckets and emitting a blank
            # name (error-handling / type-design review). Validate against the
            # UNFILTERED set so filter-dropped shapes are still forceable.
            if struct not in self._all_structs:
                raise ValueError(
                    f"force_structure {force_structure!r} is not a known "
                    "template; call list_structures() for valid labels"
                )
        else:
            items = list(self.structs.items())
            struct = weighted_choice(rng, items)
        if len(tags) == 0:
            new_name = self._select_no_tag(
                rng,
                struct,
                novelty=novelty,
                harshness=harshness,
                exclude_tags=exclude_tags,
                keep_keys=keep_keys,
                cohesion=cohesion,
            )
        else:
            new_name = self._select_tags(
                rng,
                struct,
                *tags,
                novelty=novelty,
                harshness=harshness,
                exclude_tags=exclude_tags,
                keep_keys=keep_keys,
                cohesion=cohesion,
            )
        if spelling_variety > 0 or inflection_density > 0:
            rendered, labels = self._render_substitutions(
                rng, new_name.name, spelling_variety, inflection_density
            )
            new_name.rendered = rendered
            new_name.inflection_labels = labels
        return new_name

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
                reflex. wyrd-nbpw: threaded through the SAME
                ``_render_substitutions`` the legacy path uses, so both
                scoring modes produce identical D8/D18 rendering for a
                given pick. 0 (default) = no variant substitution.
            inflection_density: D8 per-morpheme probability of emitting an
                inflected morphological form (with a grammatical-case
                label). 0 (default) = no inflection. Inflection wins ties
                over the variant axis (same rule as the legacy path).

        Returns:
            A :class:`NewName` or None when the vector path's gate or
            scoring filtered every candidate (caller decides whether
            to raise or fall back).

        Out of scope for v1:
            * Cohesion uses the simple-overlap form from
              ``vector_name_select._cohesion_multiplier``, not the
              legacy ``_cohesion_boost`` / ``tag_marginal`` form.
              Tag-cooccurrence data from the bundle is threaded
              through; legacy and vector cohesion live in two
              independent code paths until ecjp.6/7 reconciles.
        """
        # Lazy import: no actual cycle (vector_name_select imports
        # only from vectors.{schemas,scoring}, not runtime), but the
        # lazy form keeps the legacy proportions path's cold-start
        # cost flat for callers that never reach scoring_mode='vector'.
        from wyrd.generators.kenning.runtime.vector_name_select import (
            build_non_position_eligible,
            request_signature,
            select_via_vector_scoring,
        )

        items = list(self.structs.items())

        # Build (or reuse) the non-position eligibility pool. The
        # cache key folds every filter that affects pool membership.
        # Pack-overlay signature uses the ``request.packs`` tuple +
        # id(pack_meaning_dbs) so the cache invalidates when overlays
        # change (operator passes a different pack set on a later
        # request); ``request.packs`` is itself a tuple of frozen
        # :class:`PackOverlay` dataclasses, hashable + comparable.
        gate = request.gate
        exclude_tags_fz = frozenset(exclude_tags)
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

        # Per-slot base-score cache: (eligible_cache_key, request,
        # era_midpoint, priors_id) — sub-seeds within one dispatch
        # all share the same key + lazy-fill the slot_position →
        # base_scored map. Bounded FIFO eviction prevents unbounded
        # growth across warm-Lambda lifetime.
        slot_cache_key = (
            cache_key,
            request_signature(request),
            era_midpoint,
            id(priors),
        )
        slot_base_scores = self._vector_slot_score_cache.get(slot_cache_key)
        if slot_base_scores is None:
            slot_base_scores = {}
            if len(self._vector_slot_score_cache) >= self._VECTOR_SLOT_CACHE_MAX:
                # FIFO eviction: drop the oldest entry. dict
                # iteration order is insertion order in Python 3.7+,
                # so next(iter(...)) is the oldest key.
                oldest = next(iter(self._vector_slot_score_cache))
                del self._vector_slot_score_cache[oldest]
            self._vector_slot_score_cache[slot_cache_key] = slot_base_scores

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
            candidate_positions: list[str] = []
            candidate_qualifiers: list[str | None] = []
            candidate_bucket_keys: list[tuple] = []
            for word in candidate_struct:
                for key in word:
                    candidate_positions.append(key[0])
                    if "name" in key:
                        candidate_qualifiers.append("name")
                    elif "saint" in key:
                        candidate_qualifiers.append("saint")
                    else:
                        candidate_qualifiers.append(None)
                    candidate_bucket_keys.append(key)
            return select_via_vector_scoring(
                rng,
                self.meaning_db,
                structure=candidate_positions,
                request=request,
                priors=priors,
                era_midpoint=era_midpoint,
                cohesion=cohesion,
                tag_cooccurrence=self.tag_cooccurrence or None,
                exclude_tags=exclude_tags_fz,
                pack_meaning_dbs=pack_meaning_dbs,
                non_position_eligible=non_position_eligible,
                slot_base_scores=slot_base_scores,
                slot_qualifiers=candidate_qualifiers,
                slot_bucket_keys=candidate_bucket_keys,
                usage_frequency_by_bucket=self.usage_frequency_by_bucket,
                permissive=permissive,
            )

        struct = None
        picked: list = []
        tried: set = set()
        for _attempt in range(_VECTOR_STRUCT_RETRY_LIMIT):
            remaining = [it for it in items if it[0] not in tried]
            if not remaining:
                # Exhausted the distinct structure pool — fall through to the
                # graceful-degradation fallback (don't raise).
                break
            candidate_struct = weighted_choice(rng, remaining)
            if candidate_struct is None:
                break
            tried.add(candidate_struct)
            picked = _score(candidate_struct, permissive=False)
            if picked:
                struct = candidate_struct
                break

        if struct is None:
            # wyrd-tbke empty-pick PARITY: no struct is FULLY satisfiable under
            # the gates (era / stratum / tag / qualifier). Rather than return
            # None (→ the caller's loud "no eligible name" raise), degrade
            # gracefully like the legacy proportions ``_select_no_tag`` path —
            # pick a struct (weighted, as proportions does) and fill what we
            # can, emitting ``None`` for slots whose gated pool is empty
            # (NewName drops them → a shorter name). Only a genuinely
            # structure-less bundle (nothing to draw) still returns None, which
            # preserves the legitimate bundle-broken raise.
            candidate_struct = weighted_choice(rng, items)
            if candidate_struct is None:
                return None
            picked = _score(candidate_struct, permissive=True)
            if not any(p is not None for p in picked):
                # Even permissive filled NOTHING — the gate excludes every
                # morpheme at every slot. Genuinely no eligible name: return
                # None so the caller raises its operator-visible diagnostic (an
                # all-empty name is useless, and proportions only ever emits an
                # empty name in this same degenerate case). The PARTIAL case —
                # some slots fillable, some empty — falls through to the
                # degraded NewName below, matching the proportions contract.
                return None
            struct = candidate_struct

        # Reconstruct the words list-of-lists from the flat picks,
        # matching the original struct's word-grouping shape so the
        # NewName surfaces with the same word-boundary semantics as
        # the legacy path.
        #
        # wyrd-g1hj: render each pick at its SLOT-derived position (the eyjk
        # position-form), NOT the stored dash-variant. Pre-fix the vector path
        # emitted ``picked.usage`` (e.g. ``gōs-`` / ``tōn-`` / ``Grange-`` /
        # ``-tre-``), so the morpheme breakdown showed nonsensical positions
        # (two ``pre`` in ``Gōstōn``, a lone ``pre`` ``Grange-``, ``-tre-``
        # inner-at-word-start) while the rendered NAME was fine. The proportions
        # path already feeds position-forms to NewName; this brings the vector
        # path into line — slot position is ``slot[0]``; NewName still strips
        # dashes + applies word-initial case for the rendered name.
        words: list[list[str | None]] = []
        idx = 0
        for word in struct:
            word_keys: list[str | None] = []
            for slot in word:
                # wyrd-tbke: a permissive-fallback pick may be None (slot whose
                # gated pool was empty) — pass it straight through as a dropped
                # slot, matching the proportions path's None slots; NewName
                # skips None elements when rendering.
                pick = picked[idx]
                word_keys.append(_position_form(pick, slot[0]) if pick is not None else None)
                idx += 1
            words.append(word_keys)

        new_name = NewName(struct, self.meaning_db, words)
        # wyrd-nbpw: D8 inflection + D18 spelling-variant rendering — the SAME
        # post-pick substitution the legacy proportions select() applies (it
        # resolves each position-form by bare surface, so it works identically
        # on the vector path's position-form words, and skips None slots). The
        # guard keeps default generation (both knobs 0) bit-stable — no extra
        # rng draws when neither axis is active.
        if spelling_variety > 0 or inflection_density > 0:
            rendered, labels = self._render_substitutions(
                rng, new_name.name, spelling_variety, inflection_density
            )
            new_name.rendered = rendered
            new_name.inflection_labels = labels
        return new_name

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

    def _select_no_tag(
        self,
        rng,
        struct,
        *,
        novelty: float = 0.0,
        harshness: float = 0.0,
        exclude_tags: tuple[str, ...] = (),
        keep_keys: frozenset[str] | None = None,
        cohesion: float = 0.0,
    ):
        words = []
        prior_tags: set[str] = set()
        # wyrd-a4p5 / wyrd-gzvr: track the most-recently-picked SURFACE
        # FORM (dash-stripped, lowercased) so adjacent slots don't
        # dupe — 'North North' / 'Bridge Bridge' / 'Portes Portes'
        # surface when independent buckets each contain a key that
        # renders to the same surface. Comparing bucket keys directly
        # misses cross-bucket dupes: '-bridge' (post-slot key) and
        # '-bridge-' (infix-slot key) are different keys with the same
        # render. The exclude_keys filter expands per-slot to all
        # bucket keys whose stripped-lowercased form matches the
        # previous surface. Fallback: if the dedup empties the
        # eligible bucket, sample without it and accept the dupe.
        prev_surface: str | None = None
        for w in struct:
            keys = []
            for key in w:
                key_boost = self._cohesion_boost(key, prior_tags, cohesion, keep_keys=keep_keys)
                exclude_for_dedup = (
                    _bucket_keys_matching_surface(self.meaning_gen.bucket_keys(key), prev_surface)
                    if prev_surface is not None
                    else None
                )
                picked = self.meaning_gen.select(
                    rng,
                    key,
                    novelty=novelty,
                    harshness=harshness,
                    exclude_tags=exclude_tags,
                    exclude_keys=exclude_for_dedup,
                    keep_keys=keep_keys,
                    key_boost=key_boost,
                )
                if picked is None and exclude_for_dedup:
                    # Dedup-filter emptied the eligible bucket — sample
                    # without it. Better to ship a dupe once than crash
                    # on a single-surface bucket post-dedup.
                    picked = self.meaning_gen.select(
                        rng,
                        key,
                        novelty=novelty,
                        harshness=harshness,
                        exclude_tags=exclude_tags,
                        keep_keys=keep_keys,
                        key_boost=key_boost,
                    )
                keys.append(picked)
                if picked is not None:
                    prior_tags.update(self._tags_for_usage(picked))
                    prev_surface = picked.replace("-", "").lower()
            words.append(keys)
        return NewName(struct, self.meaning_db, words)

    def _select_tags(
        self,
        rng,
        struct,
        *tags,
        novelty: float = 0.0,
        harshness: float = 0.0,
        exclude_tags: tuple[str, ...] = (),
        keep_keys: frozenset[str] | None = None,
        cohesion: float = 0.0,
    ):
        name_pool = [
            self._select_no_tag(
                rng,
                struct,
                novelty=novelty,
                harshness=harshness,
                exclude_tags=exclude_tags,
                keep_keys=keep_keys,
                cohesion=cohesion,
            )
        ]
        for tag in tags:
            name_pool.append(
                self._select_tag(
                    rng,
                    struct,
                    tag,
                    novelty=novelty,
                    harshness=harshness,
                    exclude_tags=exclude_tags,
                    keep_keys=keep_keys,
                    cohesion=cohesion,
                )
            )
        words = []
        # wyrd-a4p5: track the most-recently-picked usage across the
        # merged pools so the per-slot rng.choice doesn't produce
        # adjacent dupes ('port port' / 'les les'). Three re-roll
        # attempts on tied pools; accept the dupe if the pool only
        # contains the dupe.
        prev_picked: str | None = None
        for i in range(len(struct)):
            keys = []
            for j in range(len(struct[i])):
                pool = []
                for elem in name_pool:
                    e = elem.name[i][j]
                    if e:
                        pool.append(e)
                if not pool:
                    keys.append(None)
                    continue
                # rng.choice always returns a usage; the loop's job is
                # to retry up to 3 times if it's an adjacent dupe of
                # the previous slot's pick. After the loop the latest
                # picked stands either way (3rd-attempt dupe accepted
                # rather than infinite-loop on a single-element pool).
                for _ in range(3):
                    picked = rng.choice(pool)
                    if picked != prev_picked:
                        break
                keys.append(picked)
                prev_picked = picked
            words.append(keys)
        return NewName(struct, self.meaning_db, words)

    def _select_tag(
        self,
        rng,
        struct,
        tag,
        *,
        novelty: float = 0.0,
        harshness: float = 0.0,
        exclude_tags: tuple[str, ...] = (),
        keep_keys: frozenset[str] | None = None,
        cohesion: float = 0.0,
    ):
        words = []
        prior_tags: set[str] = set()
        # wyrd-gzvr: mirror the surface-form anti-dup from _select_no_tag
        # so the multi-tag generation path (--tag X) doesn't surface
        # adjacent-duplicate words either. Same algorithm: track
        # prev_surface, expand to all bucket keys matching that
        # surface, exclude them from the next pick; fall back when
        # the exclusion empties the eligible bucket.
        prev_surface: str | None = None
        for w in struct:
            keys = []
            for key in w:
                key_boost = self._cohesion_boost(key, prior_tags, cohesion, keep_keys=keep_keys)
                exclude_for_dedup = (
                    _bucket_keys_matching_surface(self.meaning_gen.bucket_keys(key), prev_surface)
                    if prev_surface is not None
                    else None
                )
                picked = self.meaning_gen.select(
                    rng,
                    key,
                    tag,
                    novelty=novelty,
                    harshness=harshness,
                    exclude_tags=exclude_tags,
                    exclude_keys=exclude_for_dedup,
                    keep_keys=keep_keys,
                    key_boost=key_boost,
                )
                if picked is None and exclude_for_dedup:
                    picked = self.meaning_gen.select(
                        rng,
                        key,
                        tag,
                        novelty=novelty,
                        harshness=harshness,
                        exclude_tags=exclude_tags,
                        keep_keys=keep_keys,
                        key_boost=key_boost,
                    )
                keys.append(picked)
                if picked is not None:
                    prior_tags.update(self._tags_for_usage(picked))
                    prev_surface = picked.replace("-", "").lower()
            words.append(keys)
        return NewName(struct, self.meaning_db, words)

    def _tags_for_usage(self, usage: str) -> set[str]:
        """Union of tags across every Meaning sharing one usage. Used to
        accumulate the prior-context tag set as the cohesion walk fills
        in slots."""
        out: set[str] = set()
        for m in _resolve_surface(self.meaning_db, usage):
            out.update(m.tags)
        return out

    def _cohesion_boost(
        self,
        bucket_key,
        prior_tags: set[str],
        cohesion: float,
        keep_keys: frozenset[str] | None = None,
    ) -> dict[str, float] | None:
        """Per-key multiplier for the bucket given the prior-context tag
        set. None means 'no boost' — taken on the bit-stable fast path
        (cohesion=0, no prior tags yet, no co-occurrence data, or no
        candidate carries any tag).

        ``keep_keys`` (D5-2 era filter) restricts the normalization
        denominator to the surviving subset of the bucket — without it,
        the mean would be computed over candidates that ``Generator.select``
        is about to drop, and the surviving subset's effective mean
        would drift from 1.0. None means 'no era filter applied' (use
        the full bucket).

        The boost is normalized so the surviving subset's mean
        multiplier at cohesion=1 is ~1.0 (preserves total mass): a
        candidate with average class-conditional likelihood gets ~1×,
        the strongest gets >1×, the weakest <1×. Composes orthogonally
        with novelty (uniform-marginal blend) and harshness
        (phonological re-weight) — both apply downstream in
        Generator.select.
        """
        if cohesion <= 0 or not prior_tags or not self.tag_cooccurrence:
            return None
        candidates = self.meaning_gen.bucket_keys(bucket_key)
        if keep_keys is not None:
            candidates = tuple(c for c in candidates if c in keep_keys)
        if not candidates:
            return None
        raw_scores: dict[str, float] = {}
        for usage in candidates:
            cand_tags = self._tags_for_usage(usage)
            raw_scores[usage] = self._raw_class_score(prior_tags, cand_tags)
        nonzero_count = sum(1 for s in raw_scores.values() if s > 0)
        if nonzero_count == 0:
            return None
        total_raw = sum(raw_scores.values())
        mean_raw = total_raw / len(raw_scores)
        # Multiplier composition: at cohesion=1, candidates with average
        # class likelihood get ×1; above-average get >1, below-average
        # get <1. At cohesion=0 we'd return None (handled above) so the
        # downstream path is bit-stable.
        return {
            usage: (1 - cohesion) + cohesion * (raw / mean_raw) for usage, raw in raw_scores.items()
        }

    def _raw_class_score(self, prior_tags: set[str], candidate_tags: set[str]) -> float:
        """Sum of P(tb | ta) over (ta in prior, tb in candidate) using
        the empirical bigram statistics. Zero when no candidate tag was
        ever observed following any prior tag in the corpus.

        Sum (rather than mean) is intentional: a candidate carrying many
        tags that each separately co-occur with the prior context is a
        stronger match than one carrying a single moderately-cooccurring
        tag. This biases toward semantically rich morphemes when prior
        slots have set strong context.

        Iteration is sorted on both axes — set iteration order is
        hash-randomized across PYTHONHASHSEED, and float += is
        non-associative. Without sorting, the same input data produces
        ULP-level different scores across processes, which can flip
        weighted_choice outcomes at boundaries. Locking iteration order
        keeps the cohesion path bit-stable across runs.
        """
        if not prior_tags or not candidate_tags:
            return 0.0
        score = 0.0
        for ta in sorted(prior_tags):
            ma = self.tag_marginal.get(ta, 0)
            if ma == 0:
                continue
            for tb in sorted(candidate_tags):
                count = self.tag_cooccurrence.get(f"{ta}|{tb}", 0)
                if count:
                    score += count / ma
        return score


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

        # Meaning.sources is normally a {lang: [forms]} dict, but synthetic
        # Meanings (some tests) pass a bare list — treat a non-dict as "no
        # language data" so diversification simply no-ops there.
        def _langs(m) -> set[str]:
            return set(m.sources.keys()) if isinstance(getattr(m, "sources", None), dict) else set()

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
                canon_langs = _langs(canon) if canon else set()
                if fold not in seen:
                    seen[fold] = set(canon_langs)
                    continue
                # Repeat. Prefer a same-meaning synonym in another language
                # ("Hill Hill" → "Hill Haeth"). If none exists, REJECT the
                # repeat and re-pick a DIFFERENT same-position morpheme
                # ("Park ... Park" → "Park ... <other>") — wyrd-72q9.
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
                    continue
                repl = self._repick_nondup(usage, set(seen.keys()), name_sig, wi, ei, canon)
                if repl is not None:
                    self.name[wi][ei] = repl
                    if self.rendered is not None:
                        self.rendered[wi][ei] = None
                    if self.inflection_labels is not None:
                        self.inflection_labels[wi][ei] = None
                    repl_sibs = _rank_siblings(_resolve_surface(self.meaning_db, repl))
                    seen[repl.strip("-").lower()] = _langs(repl_sibs[0]) if repl_sibs else set()
                # else: nothing else of this position available → leave the dupe.

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

        def _pos(u: str) -> tuple[bool, bool]:
            return (u.startswith("-"), u.endswith("-"))

        def _langs_of(ms) -> set[str]:
            out: set[str] = set()
            for m in ms:
                if isinstance(getattr(m, "sources", None), dict):
                    out |= set(m.sources.keys())
            return out

        want = _pos(usage)
        canon_langs = _langs_of([canon]) if canon is not None else set()
        canon_tags = set(getattr(canon, "tags", []) or [])

        same_lang: list[str] = []
        tag_sharing: list[str] = []
        for k, ms in self.meaning_db.items():
            if not isinstance(k, str) or _pos(k) != want:
                continue
            kf = k.strip("-").lower()
            if not kf or kf in seen_folds:
                continue
            # Keep the same language family (skip the filter only when the
            # original carries no language data, e.g. synthetic test Meanings).
            if canon_langs and not (_langs_of(ms) & canon_langs):
                continue
            same_lang.append(k)
            if canon_tags and ({t for m in ms for t in getattr(m, "tags", [])} & canon_tags):
                tag_sharing.append(k)
        pool = sorted(tag_sharing) or sorted(same_lang)
        if not pool:
            return None
        h = int(hashlib.md5(f"{name_sig}#{wi}.{ei}".encode()).hexdigest(), 16)
        return pool[h % len(pool)]

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
        # wyrd-o53o + wyrd-ywm9 + wyrd-emlb: rank siblings (drop
        # modern_english homographs, prioritize older strata) and
        # split semantic vs derivative glosses. Lazy-import to avoid
        # a circular dep with wyrd.generators.kenning.__init__ at
        # module load — to_dict is called per roll, so the import is
        # paid once after Python caches sys.modules.
        from wyrd.generators.kenning import (
            _all_senses,
            _rank_siblings,
            _split_senses_for_display,
        )

        self._ensure_diversified()
        words: list[list[dict]] = []
        for wi, word in enumerate(self.name):
            morphemes: list[dict] = []
            for ei, e in enumerate(word):
                if e is None:
                    continue
                # wyrd-vd6y: a diversified repeat ("Hill Hill" → "Hill Haeth")
                # uses its override sibling — a different-language synonym — for
                # BOTH the displayed surface and the etymology, so the breakdown
                # matches the rendered name instead of showing the original
                # repeated morpheme.
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
                    # Sources stay from the top-ranked sibling (the
                    # canonical etymon for THIS surface form). Tags
                    # union across all ranked siblings — same as the
                    # explainer (wyrd-ywm9). Meanings split into
                    # primary + derivative for the SPA card layout.
                    morpheme["sources"] = {
                        lang: list(forms) for lang, forms in first.sources.items()
                    }
                    morpheme["tags"] = list(dict.fromkeys(t for m in ranked for t in m.tags))
                    primary, derivative = _split_senses_for_display(_all_senses(ranked))
                    morpheme["meanings"] = primary
                    morpheme["derivative_meanings"] = derivative
                    # wyrd-cp2d round 3: pronunciation + cross-script
                    # renderings (wyrd-ha9q's original_script /
                    # transliteration / english_shaped / IPA) so the
                    # JSON continuity flow carries the same panel data
                    # ``components()`` exposes to the SPA. Sparse by
                    # design — only emit the field when the
                    # _collect_renderings dict isn't empty so Latin-
                    # script-only morphemes don't carry a noisy '{}'.
                    # wyrd-o53o: renderings drawn from the ranked
                    # siblings (so dropped modern_english entries
                    # don't contribute spurious pronunciation /
                    # transliteration data).
                    renderings = _collect_renderings(ranked)
                    if renderings:
                        morpheme["renderings"] = renderings
                    # wyrd-bvwu: carry the morpheme's scholarly citations
                    # (source_ids, wyrd-9kh.1) so the SPA inspector can
                    # surface them in an expandable view. Sparse — only
                    # when non-empty — matching renderings, so rando-port-
                    # only morphemes don't carry an empty list. components()
                    # already exposes this; to_dict (morphemes_by_word, what
                    # the cards iterate) didn't until now.
                    citations = _collect_citations(ranked)
                    if citations:
                        morpheme["citations"] = citations
                # D18 variant / D8 inflection substitute if present
                if self.rendered is not None and self.rendered[wi][ei] is not None:
                    morpheme["rendered"] = self.rendered[wi][ei]
                morphemes.append(morpheme)
            if morphemes:
                words.append(morphemes)
        return {"name": str(self), "words": words}

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
                out.append(
                    {
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
                        "renderings": _collect_renderings(ranked),
                    }
                )
        return out

    def _find_meaning(self, meaning):
        roots = self._roots_str(meaning)
        return f"{roots} {', '.join(meaning.meanings)}"

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

    def _set(slot: dict[str, str], key: str, value: str | None) -> None:
        # Don't overwrite a previously-stored non-None value with None
        # (matters when the same usage spans multiple Meanings and one
        # carries richer data than another for the same canonical form).
        if value is None:
            return
        slot[key] = value

    for m in meanings:
        for lang_field, forms in m.original_script.items():
            for form, value in forms.items():
                _set(_ensure(lang_field, form), "original_script", value)
        for lang_field, forms in m.transliteration.items():
            for form, value in forms.items():
                _set(_ensure(lang_field, form), "transliteration", value)
        for lang_field, forms in m.english_shaped.items():
            for form, value in forms.items():
                _set(_ensure(lang_field, form), "english_shaped", value)
        for lang_field, forms in m.pronunciation.items():
            for form, pron in forms.items():
                slot = _ensure(lang_field, form)
                _set(slot, "ipa", pron.get("ipa"))
                _set(slot, "dialect", pron.get("dialect"))
    # wyrd-03cx: derive reader_pronunciation from ipa when no hand-
    # curated english_shaped value already filled the slot. Extracted
    # to a helper to keep _collect_renderings under the C901 ceiling
    # (the per-Meaning loop above already pushed cyclomatic close to
    # threshold pre-PR; the fallback walk adds a nested loop + branch).
    _fill_reader_pronunciations(by_lang_form)
    return by_lang_form


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


# Compact display max for description()'s citation block. Above this, the
# explainer renders 'first, second, third (+N more)' rather than a wall of
# 18 source_ids. components() keeps the full list so the SPA can render
# its own disclosure UI.
_DESCRIPTION_CITATION_LIMIT = 3


def _format_citations_for_description(citations: list[str]) -> str:
    """Render a short scholar-ID list for the explainer breakdown line.
    Long lists collapse to 'first, second, third (+N more)'; lists at or
    under the limit display in full. source_ids stay as-is — the SPA
    layer (wyrd-9kh.6) substitutes prettier titles via the source table."""
    if len(citations) <= _DESCRIPTION_CITATION_LIMIT:
        return ", ".join(citations)
    head = ", ".join(citations[:_DESCRIPTION_CITATION_LIMIT])
    extra = len(citations) - _DESCRIPTION_CITATION_LIMIT
    return f"{head} (+{extra} more)"


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


# wyrd-5z5j force-structure: a structure key (tuple of word-keys) <-> a readable
# template label like 'A | B- -c- -d'. Words are separated by ' | ', slots
# within a word by spaces; each slot is a placeholder letter (word-INITIAL
# capitalized, else lowercase — mirroring D39's case rule) decorated with its
# position dashes (pre 'X-', post '-x', inner '-x-', bare 'X'); a name/saint
# flag is appended as '@name'/'@saint'. The letters are decorative — round-trip
# reads position from the dashes, flags from '@', word boundaries from ' | '.
_LOC_TO_DASHES = {"pre": ("", "-"), "post": ("-", ""), "inner": ("-", "-"), "bare": ("", "")}
# Exact inverse of _LOC_TO_DASHES, keyed by (has_leading_dash, has_trailing_dash).
# structure_label_to_key decodes through this table so the two directions can't
# drift (complexity review).
_DASHES_TO_LOC = {(lead != "", trail != ""): loc for loc, (lead, trail) in _LOC_TO_DASHES.items()}


def structure_key_to_label(key: tuple) -> str:
    """Render a structure key as a readable template string."""
    words_out: list[str] = []
    n = 0
    for word in key:
        slots: list[str] = []
        for i, elem in enumerate(word):
            loc = elem[0]
            flags = [f for f in elem[1:] if f in ("name", "saint")]
            ch = chr(65 + n % 26)
            ch = ch if i == 0 else ch.lower()  # word-initial cap, else lower
            lead, trail = _LOC_TO_DASHES.get(loc, ("", ""))
            tok = f"{lead}{ch}{trail}" + "".join(f"@{f}" for f in flags)
            slots.append(tok)
            n += 1
        words_out.append(" ".join(slots))
    return " | ".join(words_out)


def structure_label_to_key(label: str) -> tuple:
    """Parse a structure label back to a key (inverse of structure_key_to_label).

    The label format does not encode the ``single`` flag; it is re-added below
    for any single-element word, mirroring ``word_to_key``'s ``len(word) == 1``
    rule. (Both directions depend on that arity↔``single`` coupling.)"""
    words = []
    for word_str in label.split("|"):
        word_str = word_str.strip()
        if not word_str:
            continue
        elements = []
        for tok in word_str.split():
            parts = tok.split("@")
            base = parts[0]
            flags = [p for p in parts[1:] if p in ("name", "saint")]
            loc = _DASHES_TO_LOC[(base.startswith("-"), base.endswith("-"))]
            elements.append((loc, *flags))
        if len(elements) == 1:
            elements[0] = (*elements[0], "single")
        words.append(tuple(elements))
    return tuple(words)


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
        # so no path (select / vector / force_structure) can produce them. A
        # bare word standing INSIDE a larger structure (`Green` in `Green Park`)
        # is unaffected — that structure has 2 morphemes total.
        if _is_single_morpheme_structure(words):
            continue
        struct[words] = proportion
    # wyrd-g1hj loud-failure guard: the single-morpheme filter above runs
    # BEFORE NameGenerator, so its empty-structs guard (which checks
    # ``if structs and not self.structs``) can't see this case — a culture
    # whose structures are ALL single-morpheme would yield an empty ``struct``,
    # and the legacy select() path would later crash deep in _select_no_tag on
    # a None from weighted_choice([]). Fail loud + attributable here instead.
    if structures and not struct:
        raise ValueError(
            "load_proportions: every structure was a single morpheme (wyrd-g1hj) "
            "— this culture has no multi-morpheme templates to generate from. "
            "Re-emit the bundle (the proportions still record them) or relax the "
            "single-morpheme exclusion."
        )
    # wyrd-mj2: tag-level co-occurrence + marginal — empirical bigram
    # statistics over the (left.tags × right.tags) cartesian product
    # learned from each culture's place-name corpus. Optional: legacy
    # bundles without these keys produce a no-op cohesion knob.
    cooccurrence = data.get("tag_cooccurrence", {})
    marginal = data.get("tag_marginal", {})
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
        marginal,
        culture_attested_usages=culture_attested_usages,
        culture_attested_meanings=culture_attested_meanings,
    )


def _blend_uniform(items, novelty: float):
    """Blend each (item, weight) pair with a uniform marginal.

    Returns a new list of (item, blended_weight) where blended_weight is
    ``(1-novelty)·normalized_empirical + novelty·uniform_share``. The result
    is a normalized distribution (sums to ~1.0); weighted_choice handles
    fractional weights fine via its uniform draw over the cumulative range.

    novelty must be > 0 (callers fast-path the novelty=0 case to avoid
    re-allocating). Items with weight <= 0 are stripped from the empirical
    side but still get their uniform share — without that, novelty=1 would
    silently undercount keys that happened to have zero empirical weight.
    """
    if not items:
        return items
    n = len(items)
    total = sum(max(w, 0) for _, w in items)
    if total <= 0:
        # Pure uniform if every empirical weight is zero — there's nothing to
        # blend against, so the novelty knob has no meaningful axis. Returning
        # 1/n keeps the result a normalized probability distribution as the
        # docstring promises.
        return [(k, 1 / n) for k, _ in items]
    return [(k, (1 - novelty) * (max(w, 0) / total) + novelty / n) for k, w in items]


# D6 --harsh: phonological harshness scoring. Heuristic, not phonotactic.
# Goal is order-of-magnitude separation between vowel-soft morphemes
# ('ham', 'baron', 'borough') and stop-cluster-heavy ones ('shuck', 'crag',
# 'fork'). The exact ranking isn't load-bearing — only relative ordering
# matters for the per-item multiplier. Common OE/ON long-vowel diacritics
# count as vowels so 'tūn' / 'lēah' don't get falsely classified.
_VOWELS = frozenset("aeiouyæāēīōūȳáéíóúýǣǿǫâêîôûŵŷ")
_VOICED_STOPS = frozenset("bdg")
_VOICELESS_STOPS = frozenset("pktc")
_NASALS = frozenset("nm")


@lru_cache(maxsize=4096)
def _harshness_score(usage: str) -> float:
    """Phonological harshness in [0, 1]. Higher = more menacing-feeling.

    Cached: the bundle has ~1600 unique modern_usages and `_blend_harsh`
    runs the score over every key on every bucket on every sample, so
    the same string is hit thousands of times across one batch of name
    generations. lru_cache pins the per-string cost at one pass.

    Combines three signals on the dash-stripped lowercased usage:

    - Coda harshness: stops at the end (b/d/g, p/t/k, c) score 1.0; nasals
      0.4; vowels 0.0; everything else (l/r/s/h/f/etc.) 0.5.
    - Cluster density: fraction of adjacent-consonant positions.
    - Consonant density: non-vowel chars / total chars.

    Composition is a simple weighted sum (45/35/20) chosen so coda dominates
    perception ('cot' > 'co'), with cluster + density as tiebreakers. This
    is meant for a sampling reweight, not for serious phonological work.
    """
    s = "".join(c for c in usage.lower() if c.isalpha())
    if not s:
        return 0.5
    n = len(s)

    last = s[-1]
    if last in _VOICED_STOPS or last in _VOICELESS_STOPS:
        coda_score = 1.0
    elif last in _NASALS:
        coda_score = 0.4
    elif last in _VOWELS:
        coda_score = 0.0
    else:
        coda_score = 0.5

    cluster_count = 0
    consonant_count = 0
    for i, c in enumerate(s):
        if c not in _VOWELS:
            consonant_count += 1
            if i > 0 and s[i - 1] not in _VOWELS:
                cluster_count += 1
    cluster_score = cluster_count / max(1, n - 1)
    consonant_density = consonant_count / n

    return min(1.0, 0.45 * coda_score + 0.35 * cluster_score + 0.2 * consonant_density)


def _blend_harsh(items, harshness: float):
    """Re-weight items by phonological harshness score.

    Each (key, weight) becomes (key, weight * (1 - harshness + harshness *
    2 * score)) where score ∈ [0, 1] is computed from the key string. At
    ``harshness=0`` every multiplier is 1.0 (bit-stable); at ``harshness=1``
    soft-keyed items go to weight 0 and harsh-keyed items get 2× their
    empirical weight.

    The 2× scaling is deliberate: an absolute multiplier of 0..1 would
    mean even harsh=1 only halves the mean weight, leaving soft items
    competitive. 0..2 puts soft items at 0 at the harshness=1 limit.

    Callers should fast-path ``harshness <= 0`` to avoid re-allocating.
    """
    if not items:
        return items
    return [
        (k, max(w, 0) * (1 - harshness + harshness * 2 * _harshness_score(k))) for k, w in items
    ]


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
