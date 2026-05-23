"""Weighted random selection over learned morpheme proportions.

Ported from Rando (rando/proportions.py) — Python 2 → 3.

Key change beyond syntax: weighted_choice now takes an explicit Random instance
so generation is reproducible from the seed param.
"""

from __future__ import annotations

import random
import re
from functools import lru_cache

from .meaning import _mimic_case


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
            items = [(k, v) for k, v in items if k in keep_keys]
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
        keys: list[str] = []
        for tag in tags:
            keys.extend(self.tag_db.get(tag, []))
        return {key: self.elements[key] for key in sorted(set(keys)) if key in self.elements}

    def _apply_excludes(self, items, exclude_tags: tuple[str, ...]):
        """Drop keys whose usage appears in tag_db under any exclude tag.
        Returns a dict to keep the .items() call site consistent.

        Hot-path short-circuit: if no exclude tag is present in this
        bucket's tag_db, nothing can be filtered — return ``dict(items)``
        directly without the per-key membership check. Today's bundle has
        no fiction-tagged morphemes so this is the steady-state path."""
        excluded: set[str] = set()
        for tag in exclude_tags:
            excluded.update(self.tag_db.get(tag, ()))
        if not excluded:
            return dict(items)
        return {k: v for k, v in items if k not in excluded}


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


class MeaningGenerator:
    def __init__(self, meaning_db, tag_db, proportions):
        self.meaning_db = meaning_db
        self.tag_db = tag_db
        self.generators: dict[tuple, Generator] = {}
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
        self.load_parts(proportions)

    def load_parts(self, proportions, *addkeys):
        for usage, proportion in proportions.items():
            meanings = self.meaning_db[usage]
            keys = {m.key() for m in meanings}
            for key in keys:
                if addkeys:
                    key = tuple(list(key) + list(addkeys))
                gen = self.generators.setdefault(key, Generator(self.tag_db, {}))
                gen.add_item(usage, proportion)

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
        allowed: frozenset[str] | None = frozenset(
            usage
            for usage, meanings in self.meaning_db.items()
            if any(m.attested_in_era_range(era_range) for m in meanings)
        )
        if len(allowed) == len(self.meaning_db):
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
            usage
            for usage, meanings in self.meaning_db.items()
            if any(m.in_stratum(stratum) for m in meanings)
        )
        if len(allowed) == len(self.meaning_db):
            allowed = None
        self._stratum_keep_cache[stratum] = allowed
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
        return self.generators[key].select(
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


class NameGenerator:
    def __init__(
        self,
        meaning_db,
        meaning_gen,
        structs,
        tag_cooccurrence: dict[str, int] | None = None,
        tag_marginal: dict[str, int] | None = None,
    ):
        self.meaning_db = meaning_db
        self.meaning_gen = meaning_gen
        self.structs = structs
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
        """
        keep_keys = _intersect_keep_keys(
            self.meaning_gen.keep_keys_for_era(era_range),
            self.meaning_gen.keep_keys_for_stratum(stratum),
        )
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

        Returns:
            A :class:`NewName` or None when the vector path's gate or
            scoring filtered every candidate (caller decides whether
            to raise or fall back).

        Out of scope for v1:
            * D8 inflection / D18 variant substitution at render time
              (no ``rendered`` / ``inflection_labels`` produced; surface
              renders the dash-stripped usage). The legacy path's
              substitution helpers can plug in here as a separate
              follow-up.
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
            select_via_vector_scoring,
        )

        items = list(self.structs.items())
        struct = weighted_choice(rng, items)
        if struct is None:
            return None

        # Flatten the struct into a list of position labels for the
        # primitive. The struct shape is tuple-of-words, each word
        # is tuple-of-keys, each key is a feature-tuple (e.g.
        # ("pre", "single"), ("post", "name")) — the first element is
        # the position label per ``word_to_key`` in this module.
        # The primitive accepts bare position labels directly, so no
        # encode/decode round-trip required.
        flat_positions: list[str] = [key[0] for word in struct for key in word]

        picked = select_via_vector_scoring(
            rng,
            self.meaning_db,
            structure=flat_positions,
            request=request,
            priors=priors,
            era_midpoint=era_midpoint,
            cohesion=cohesion,
            tag_cooccurrence=self.tag_cooccurrence or None,
            exclude_tags=frozenset(exclude_tags),
            pack_meaning_dbs=pack_meaning_dbs,
        )
        # The primitive returns either [] (empty pool / no positive
        # scores) or a list with exactly one Meaning per requested
        # slot — never a partial. The empty check is sufficient.
        if not picked:
            return None

        # Reconstruct the words list-of-lists from the flat picks,
        # matching the original struct's word-grouping shape so the
        # NewName surfaces with the same word-boundary semantics as
        # the legacy path. Each picked meaning's usage string is what
        # NewName.__str__ renders (dash-stripped).
        words: list[list[str | None]] = []
        idx = 0
        for word in struct:
            word_keys: list[str | None] = []
            for _ in word:
                word_keys.append(picked[idx].usage)
                idx += 1
            words.append(word_keys)

        return NewName(struct, self.meaning_db, words)

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
                meanings = self.meaning_db.get(usage) or []
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
        for m in self.meaning_db.get(usage, ()):
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
        words: list[str] = []
        for wi, w in enumerate(self.name):
            chunks: list[str] = []
            for ei, e in enumerate(w):
                if e is None:
                    continue
                if self.rendered is not None and self.rendered[wi][ei] is not None:
                    chunks.append(self.rendered[wi][ei])
                else:
                    chunks.append(e.replace("-", ""))
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
                words.append(self.meaning_db[e])
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
                ranked = _rank_siblings(self.meaning_db.get(e, []))
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

        words: list[list[dict]] = []
        for wi, word in enumerate(self.name):
            morphemes: list[dict] = []
            for ei, e in enumerate(word):
                if e is None:
                    continue
                meanings_list = self.meaning_db.get(e, [])
                ranked = _rank_siblings(meanings_list)
                first = ranked[0] if ranked else None
                morpheme: dict = {"usage": e}
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

        out = []
        for word in self.name:
            for e in word:
                if e is None:
                    continue
                # wyrd-o53o round 4 (Gemini MED): defensive access
                # matching to_dict()'s pattern. e SHOULD exist (it
                # came from the generator's pool) but the consistency
                # win + no-IndexError-on-empty-siblings is worth the
                # cost of one .get() + one truthy check.
                meanings = self.meaning_db.get(e, [])
                ranked = _rank_siblings(meanings)
                if not ranked:
                    continue
                first = ranked[0]
                primary, derivative = _split_senses_for_display(_all_senses(ranked))
                out.append(
                    {
                        "usage": e,
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
        struct[words] = proportion
    # wyrd-mj2: tag-level co-occurrence + marginal — empirical bigram
    # statistics over the (left.tags × right.tags) cartesian product
    # learned from each culture's place-name corpus. Optional: legacy
    # bundles without these keys produce a no-op cohesion knob.
    cooccurrence = data.get("tag_cooccurrence", {})
    marginal = data.get("tag_marginal", {})
    return NameGenerator(meaning_db, mg, struct, cooccurrence, marginal)


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
