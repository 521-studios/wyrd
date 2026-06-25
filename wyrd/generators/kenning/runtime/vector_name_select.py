"""Gate→score→sample primitive for vector-driven name generation
(wyrd-ecjp.5).

The retired proportions pipeline did proportion-table sampling:
pre-baked per-(culture × tag × position) weight tables drove
``weighted_choice`` at each slot. This module — now the sole scoring
path — is the vector-driven approach: every slot's per-lemma weight is
computed at request time by the D36.2 composition rule
(:func:`wyrd.generators.kenning.vectors.scoring.score`):

    score = phon_w * phon_score(lemma_phon, request)
          + sem_w  * sem_score(lemma_tags, request)
          + pos_w  * pos_score(slot_position, request)
          + base_w * baseline_score(lemma_ref, request, priors)

Per-slot pipeline:

  1. **Gate** — filter the eligible meaning pool by the hard
     predicates in :class:`EligibilityGate` (culture, stratum, tags,
     and the D46 era record-entry cutoff). Lemmas that fail the gate
     never enter scoring. (Era's RENDER effect — which reflex — stays
     downstream per D44; the gate half is only "in the record by then".)
  2. **Score** — for each eligible meaning, compute the canonical
     composition score using its phonological vector + tags + the
     request's per-axis weights + the empirical priors.
  3. **Cohesion adapter (D17)** — when cohesion > 0, multiplicatively
     bias the per-meaning score by the tag co-occurrence overlap
     with previously-picked slots' tags. At cohesion=0 (default)
     scoring is independent per slot.
  4. **Sample** — :func:`weighted_choice` over the (meaning, score)
     pairs picks one. Zero / negative scores are filtered before
     sampling (``weighted_choice`` skips them anyway, but the
     explicit filter keeps the diagnostic cleaner).

This is the runtime side of ecjp.5, and the only scoring path: the
Kenning generator always dispatches into here. (The proportion-table
path it once coexisted with — opt-in via the old ``--scoring-mode``
flag — is retired; ecjp.6/7 drift measurement empirically validated
vector before the proportions path was removed.)

Out of scope for v1:
  * Multi-pack composition (wyrd-ecjp.8 / Phase 7).
  * Tolerance bands on the composed register (D8, deferred).
  * Per-slot inflection / variant resolution (handled downstream by
    the shared rendering path; this module just picks the lemma).
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import TYPE_CHECKING

# wyrd-glos: the gloss-policy predicate is defined in proportions (the
# canonical keep_keys_for_gloss home). Safe module-level import — proportions
# imports vector_name_select only lazily (inside select_via_vector), so there
# is no module-load cycle; this module is itself imported lazily, by which
# point proportions is fully loaded.
from wyrd.generators.kenning.runtime.proportions import _gloss_eligible
from wyrd.generators.kenning.runtime.word import WORD_POSITION_KEY_PREFIX
from wyrd.generators.kenning.vectors.schemas import (
    EmpiricalPriors,
    PhonologicalVector,
    RequestVector,
)
from wyrd.generators.kenning.vectors.scoring import score

if TYPE_CHECKING:
    from wyrd.generators.kenning.runtime.meaning import Meaning


def _default_empty_phon_vector() -> PhonologicalVector:
    """Per-meaning fallback when the bundle has no phonological_vector
    for that meaning (legacy bundles pre-kq7w.1 / pre-ecjp.8). The
    all-zero vector means phon_score contributes 0 to the composition
    and the score falls back to the other three axes."""
    return PhonologicalVector()


# wyrd-eyjk/D40: the `_matches_position` position-gate predicate was removed —
# position is no longer a match-time constraint. A morpheme may fill any slot;
# the DATA-driven restriction is the per-(position) bucket frequency
# (`_resolve_slot_usage_frequency`): a morpheme never observed at a position
# looks up to 0 there and is filtered. `_slot_position_label` survives below —
# it still labels the slot's position for the D36 position-axis SCORING.


def _matches_stratum(meaning: Meaning, stratum: str | None) -> bool:
    """Stratum-gate predicate. Delegates to ``Meaning.in_stratum`` —
    meanings with no stratum data pass through (only Welsh-family is
    classified today). ``None`` stratum disables the filter.

    Round-1 reviewer caught a bug here: an earlier defensive
    ``getattr(meaning, "has_stratum", None)`` looked up the wrong
    method name (Meaning's actual method is ``in_stratum``) and
    silently bypassed the stratum gate for every meaning. The fix
    is a direct call to the right method.
    """
    if stratum is None:
        return True
    return meaning.in_stratum(stratum)


def _cohesion_raw(
    meaning_tags: frozenset[str],
    prior_tags: frozenset[str],
    cohesion_table: dict[str, dict[str, float]] | None,
) -> float:
    """Per-candidate D17 *raw* coherence: the mean co-occurrence probability
    ``P(candidate_tag | prior_tag)`` over every (candidate_tag, prior_tag)
    pair that has data in ``cohesion_table``.

    Returns 0.0 when there's no prior (first slot), no table, or no
    overlapping pair — such a candidate sits at the bottom of the
    slot-relative ranking :func:`_cohesion_multipliers` builds.

    Tags are iterated in sorted order so the float sum is byte-identical
    across processes (the PYTHONHASHSEED / D17-refinement bit-stability
    concern, now load-bearing).
    """
    if not prior_tags or not cohesion_table:
        return 0.0
    overlap_sum = 0.0
    pair_count = 0
    for ltag in sorted(meaning_tags):
        ltag_co = cohesion_table.get(ltag)
        if not ltag_co:
            continue
        for ptag in sorted(prior_tags):
            p = ltag_co.get(ptag)
            if p is None:
                continue
            overlap_sum += p
            pair_count += 1
    if pair_count == 0:
        return 0.0
    return overlap_sum / pair_count


def _cohesion_multipliers(raws: list[float], cohesion: float) -> list[float]:
    """D17 *slot-relative* cohesion bias. Given the per-candidate raw
    coherence scores (:func:`_cohesion_raw`) for one slot's pool, return each
    candidate's multiplicative weight bias::

        multiplier_i = (1 - cohesion) + cohesion * (raw_i / mean_raw)

    so above-pool-average coherence gets >1x, below-average <1x, and the
    multipliers sum to ``len(raws)`` (algebraically exact; within float
    tolerance in practice) — mean-normalized, so cohesion *redistributes* a
    slot's weight toward attested-pair candidates rather than inflating it. At cohesion=1 a candidate with no co-occurrence to the
    priors (raw=0) is fully suppressed; the pool can't collapse because a
    positive ``mean_raw`` guarantees at least one candidate with raw>0.

    Returns all-1.0 — the bit-stable no-op — when cohesion<=0 or the pool has
    no coherence signal at all (every raw is 0 → mean_raw=0). At cohesion=0
    each multiplier is exactly 1.0, so default output is unchanged.

    This replaces the pre-fix per-candidate ``1 + cohesion*avg_p`` form, whose
    absolute probabilities (~0.07 mean) gave a near-uniform <=1.07x nudge that
    perturbed picks without concentrating mass on attested pairs (wyrd-e2b4).
    ``math.fsum`` keeps the pool mean order-independent and exact.
    """
    n = len(raws)
    if cohesion <= 0 or n == 0:
        return [1.0] * n
    total = math.fsum(raws)
    if total <= 0.0:
        return [1.0] * n
    mean = total / n
    return [(1.0 - cohesion) + cohesion * (r / mean) for r in raws]


def _slot_position_label(structural_element: str) -> str:
    """Resolve the slot position label from a structural element string.

    The input here is a render-side *structural* string (D39 position-form
    decoration applied to a slot), NOT a stored ``Meaning.usage`` — stored
    identity is bare now (``Meaning._set_location`` is always "bare", wyrd-aicu),
    so this dash-shape decode lives on the structural-string side only.
    wyrd-eyjk/D40: this label is no longer used to *gate* eligibility (the
    ``_matches_position`` gate is gone) — it feeds the D36 position-axis SCORING
    (``pos_score``) and the per-(position) bucket-key lookup. The mapping:

    * ``"-inner-"`` (both dashes) → ``"inner"``
    * ``"Place-"`` (trailing dash only) → ``"pre"``
    * ``"-shire"`` (leading dash only) → ``"post"``
    * ``"Bare"`` (no dashes) → ``"bare"``

    wyrd-vpri: bare (no-dashes) maps to its own ``"bare"`` label, NOT
    ``"post"`` — the historical conflation that let suffix-only keys
    ('-park' → post) be treated as bare single-word slots.
    """
    s = structural_element.strip()
    has_leading = s.startswith("-")
    has_trailing = s.endswith("-")
    if has_leading and has_trailing:
        return "inner"
    if has_trailing:
        return "pre"
    if has_leading:
        return "post"
    # no-dashes → "bare" (matches Meaning._set_location)
    return "bare"


def _passes_base_gates(
    m: Meaning,
    *,
    gate,
    exclude_tags: frozenset[str],
    include_unglossed: bool,
) -> bool:
    """Slot-independent eligibility gates shared by the native and pack pools,
    cheapest-first (matching ``eligibility.admits``): the tag gates (frozenset
    membership), the era record-entry gate (D46, a cached int compare), the
    stratum filter (per-form dict iteration), and the gloss policy. Returns
    False on the first failing gate.

    The record-entry gate is the ONLY way era touches eligibility (D46
    refining D44): a bounded era excludes morphemes whose earliest
    evidence is at or after the era's end ("in the record by then" — the
    Silicon rule); pools accrete, nothing expires, and an open cutoff
    (None — no era, or the present-day stage) gates nothing."""
    # frozenset.isdisjoint runs in C and skips the per-tag generator overhead
    # in this O(64k) hot path; equivalent to any(t in <fs> for t in m.tags).
    if exclude_tags and not exclude_tags.isdisjoint(m.tags):
        return False
    # wyrd-c6o1.4: required_tags is deliberately NOT a pool-wide gate. Gating
    # every slot to tagged lemmas (the wyrd-wv85 shape) collapsed variety into
    # thematically-saturated names and overshot the historical "one tagged
    # morpheme per name" semantics (the retired proportions path's _select_tags
    # blend). Like a mood (wyrd-4rp8), --tag now reserves a SINGLE slot in
    # select_via_vector_scoring and hard-restricts only that slot; the pool stays
    # tag-agnostic so the other slots sample freely. (--tag still adds a soft
    # semantic-axis bias via register.semantic_tags, set in the adapter.)
    if gate.era_record_cutoff is not None:
        start = m.record_start()
        if start is not None and start >= gate.era_record_cutoff:
            return False
    if not _matches_stratum(m, gate.stratum):
        return False
    # wyrd-glos: gloss policy — same predicate keep_keys_for_gloss applies, so
    # this filter and the keep-set share the rule.
    return _gloss_eligible(m.usage, bool(m.meanings), include_unglossed)


def _native_meaning_eligible(
    m: Meaning,
    *,
    admitted_langs: frozenset[str] | None,
    gate,
    exclude_tags: frozenset[str],
    include_unglossed: bool,
) -> bool:
    """Whether a native-pool Meaning is admitted: the per-Meaning attested-
    language narrowing (wyrd-pfoo), the shared base gates, and the native-only
    exclusions (pure-proper-noun saints / given names, connector particles).

    ``admitted_langs`` is ``None`` when the per-Meaning filter is inactive
    (legacy bundle, or this usage_key carried no per-Meaning attestation data);
    a frozenset (even empty) means the filter is active. The ``is not None``
    check is load-bearing — an empty frozenset ("attested, no eligible
    language") must not be conflated with the missing case.
    """
    if admitted_langs is not None:
        primary = m.primary_language()
        if primary is None or primary not in admitted_langs:
            return False
    if not _passes_base_gates(
        m, gate=gate, exclude_tags=exclude_tags, include_unglossed=include_unglossed
    ):
        return False
    # wyrd-57d8: the request-independent base-pool class exclusions (D40
    # pure-proper-noun saints / given names, wyrd-gwj3 connector particles, AND
    # synthesized Norman manorial subjects) live on Meaning so every base pool —
    # this scored pool, the proportions frequency-weight pool
    # (MeaningGenerator.load_parts), and the diversification re-pick
    # (proportions.NewName._collect_repick_pools) — applies them identically.
    # Family-name etymons with a real place sense stay in (manorial places).
    return not m.is_base_pool_excluded()


def _pack_meaning_eligible(
    m: Meaning,
    *,
    pack,
    gate,
    exclude_tags: frozenset[str],
    include_unglossed: bool,
) -> bool:
    """Whether a pack-overlay Meaning is admitted: the shared base gates plus
    the per-pack allowed/excluded tag filters (wyrd-ecjp.8). Pack overlays are
    intentionally NOT culture-restricted — they add other-culture flavor."""
    if not _passes_base_gates(
        m, gate=gate, exclude_tags=exclude_tags, include_unglossed=include_unglossed
    ):
        return False
    # isdisjoint (C-level) avoids the per-tag generator overhead.
    if pack.allowed_pack_tags and pack.allowed_pack_tags.isdisjoint(m.tags):
        return False
    # Eligible unless the morpheme carries an excluded tag (De Morgan of
    # `excluded and not isdisjoint`): no excluded set, or disjoint from it.
    return not pack.excluded_pack_tags or pack.excluded_pack_tags.isdisjoint(m.tags)


def _native_pool(
    meaning_db: dict[str, list[Meaning]],
    *,
    gate,
    exclude_tags: frozenset[str],
    culture_attested_usages: frozenset[str] | None,
    culture_attested_meanings: dict[str, frozenset[str]] | None,
    include_unglossed: bool,
) -> list[Meaning]:
    """Native (non-pack) eligibility pool, walked in ``meaning_db`` insertion
    order so the output list is reproducible across runs."""
    pool: list[Meaning] = []
    for usage_key, meanings_for_usage in meaning_db.items():
        # wyrd-eyjk/D40: culture_attested_usages holds bare SURFACES (the
        # proportions side records position-forms); compare this entry's stored
        # variant by surface so a morpheme stored as `Giles-` but recorded as
        # `-giles` isn't silently dropped from the pool.
        surface = usage_key.lower().replace("-", "")
        if culture_attested_usages is not None and surface not in culture_attested_usages:
            continue
        # Per-Meaning narrowing set for this surface (wyrd-pfoo); None when
        # the filter is inactive or this surface carried no per-Meaning data.
        # wyrd-c6o1.3: keyed by bare surface (fold-unioned in
        # ``load_proportions``), so every stored dash-variant key (the bare
        # `ton`, `Ton-`, `-ton-`) gets the same narrowing as the position-form
        # the corpus attested (`-ton`) instead of bypassing it and letting
        # wrong-language homographs into the native pool.
        admitted_langs: frozenset[str] | None = (
            culture_attested_meanings.get(surface)
            if culture_attested_meanings is not None
            else None
        )
        for m in meanings_for_usage:
            if _native_meaning_eligible(
                m,
                admitted_langs=admitted_langs,
                gate=gate,
                exclude_tags=exclude_tags,
                include_unglossed=include_unglossed,
            ):
                pool.append(m)
    return pool


def _pack_pool(
    pack_meaning_dbs: dict[str, dict[str, list[Meaning]]],
    *,
    packs,
    gate,
    exclude_tags: frozenset[str],
    include_unglossed: bool,
) -> list[Meaning]:
    """Pack-overlay eligibility pool (wyrd-ecjp.8), appended after the native
    pool. Per-pack tag filters gate on top of the shared base gates."""
    pool: list[Meaning] = []
    for pack in packs:
        pack_db = pack_meaning_dbs.get(pack.pack_name)
        if pack_db is None:
            continue
        for meanings_for_usage in pack_db.values():
            for m in meanings_for_usage:
                if _pack_meaning_eligible(
                    m,
                    pack=pack,
                    gate=gate,
                    exclude_tags=exclude_tags,
                    include_unglossed=include_unglossed,
                ):
                    pool.append(m)
    return pool


def build_non_position_eligible(
    meaning_db: dict[str, list[Meaning]],
    *,
    gate,
    exclude_tags: frozenset[str],
    pack_meaning_dbs: dict[str, dict[str, list[Meaning]]] | None,
    packs,
    culture_attested_usages: frozenset[str] | None = None,
    culture_attested_meanings: dict[str, frozenset[str]] | None = None,
    include_unglossed: bool = True,
) -> list[Meaning]:
    """Pre-compute the non-position-filtered eligibility pool.

    The gate (stratum / exclude_tags / era cutoff) doesn't depend on the slot — it
    applies uniformly across every slot in the structure. (wyrd-c6o1.4: ``--tag`` /
    ``required_tags`` is NOT applied here — it's a single reserved-slot restriction
    in ``select_via_vector_scoring``, so the pool is tag-agnostic.) Pulling this out
    into a free function lets one call build the pool once and reuse it across that
    dispatch's struct retries (the per-call caller is
    :meth:`NameGenerator._build_vector_pools`; wyrd-i7uy: the pool is built fresh
    per ``generate()`` call and discarded on return — never cached across requests,
    because it depends on the request's gate (culture / stratum / exclude_tags /
    era cutoff)).
    The list is per-meaning order-stable (we walk ``meaning_db.items()``, which is
    insertion-ordered in Python 3.7+), so the weighted-sampling loop can consume it
    directly without re-shuffling.

    ``culture_attested_usages`` is the per-usage culture-bleed filter
    (wyrd-361): only Meanings whose ``usage`` key appears in this
    culture's corpus are admitted to the native-pool. Without the
    filter, English-culture vector requests pull in Welsh / Persian /
    Hebrew morphemes that aren't attested in English place names.
    ``None`` disables the filter (back-compat for non-NameGenerator
    callers that don't carry per-culture data).

    ``culture_attested_meanings`` is the per-Meaning narrowing
    (wyrd-pfoo), keyed by bare SURFACE (dashes folded;
    ``load_proportions`` fold-unions the bundle's position-form keys
    before constructing the NameGenerator — wyrd-c6o1.3). For each
    surface, only Meanings
    whose primary source-language is in the attested set are admitted.
    Filters wrong-sense Meanings the per-usage filter let through:
    Celtic ``-ton``→'tone' (music), ``Saint-``→'greed', ``-bridge``
    →'card game'. ``None`` disables this narrower filter and falls
    back to per-usage only (back-compat for legacy bundles built
    before the per-Meaning attestation pipeline shipped).

    A consequence — intentional but worth flagging: synthetic plurals
    (``-X`` → ``-Xs`` for is_name() lemmas) and inflection shadows
    (``meaning.py:_register_inflection_shadows``) are added to
    ``meaning_db`` at non-canonical keys that don't appear in the
    proportions tables. They get filtered out here — matching the
    retired proportions path, which only sampled from its weight tables,
    so plurals + shadows were never sampled directly there either.
    The shadows still surface in the trie matcher for explainer
    requests; only the GENERATOR pool excludes them.

    The filter applies only to the NATIVE pool — pack overlays
    intentionally introduce other-culture flavor (per wyrd-ecjp.11)
    and are NOT culture-restricted.
    """
    # wyrd-8uvi: the per-guard predicates + pool builders are extracted to
    # module-level helpers to keep this function under the C901 ceiling; the
    # native pool is built first, then pack overlays are appended (preserving
    # the original native-then-pack output order for seed reproducibility).
    non_position_eligible = _native_pool(
        meaning_db,
        gate=gate,
        exclude_tags=exclude_tags,
        culture_attested_usages=culture_attested_usages,
        culture_attested_meanings=culture_attested_meanings,
        include_unglossed=include_unglossed,
    )
    # wyrd-ecjp.8: admit pack lemmas alongside native lemmas.
    if pack_meaning_dbs:
        non_position_eligible.extend(
            _pack_pool(
                pack_meaning_dbs,
                packs=packs,
                gate=gate,
                exclude_tags=exclude_tags,
                include_unglossed=include_unglossed,
            )
        )
    return non_position_eligible


def build_slot_base_scores(
    non_position_eligible: list[Meaning],
    *,
    slot_position: str,
    request: RequestVector,
    priors: EmpiricalPriors,
    era_midpoint: int,
    slot_qualifier: str | None = None,
    slot_usage_frequency: dict[str, float] | None = None,
) -> list[tuple[Meaning, float]]:
    """Score every position-eligible meaning against the request +
    return the ``(meaning, base_score)`` pairs whose score is > 0.

    "Base score" = phon + sem + pos + baseline. It excludes the
    cohesion-multiplier step (which depends on prior-slot picks +
    is per-sample). The caller memoizes this list within one dispatch
    (reused across that call's struct retries) so the O(P) score loop
    runs once per (slot, request) per call — wyrd-i7uy: per-call only,
    never cached across requests.

    ``slot_qualifier`` (wyrd-izcr) restricts the pool by qualifier
    flag when the structure's word_key carried one. The predicates
    mirror :meth:`Meaning.key`'s legacy bucket-assignment logic so
    vector mode admits the same meanings the proportions path would
    have admitted to the equivalent ``(location, "name")`` or
    ``(location, "saint")`` bucket:

    - ``"name"`` → :meth:`Meaning.is_name` (any morpheme tagged
      ``female name`` / ``male name`` / ``family name``)
    - ``"saint"`` → dash-stripped lowercased ``usage == "saint"``
      (the literal ``Saint-`` / ``-Saint`` qualifier morpheme;
      saint-tagged morphemes like ``Andrew-`` that are ALSO
      ``is_name()`` True are routed through the ``"name"`` bucket
      by :meth:`Meaning.key`'s ``if/elif``, not the ``"saint"`` one)

    ``None`` (default) = no qualifier restriction. Without this
    filter, the vector path silently drew bare-affix morphemes
    into qualifier-flagged slots (e.g. ``Port-`` + ``-all`` filling
    a ``[pre+saint, post+name]`` structure).

    ``slot_usage_frequency`` (wyrd-bol9) is the per-usage empirical
    frequency lookup for this slot's bucket — the same per-culture
    proportions data the retired proportions path once sampled from.
    When present,
    the final per-Meaning weight is composed via
    :func:`_apply_per_usage_frequency` (per-usage normalization, not
    a flat ``base_score * frequency`` multiplication — see that
    helper's docstring for the exact formula). Meanings whose usage
    is missing or carries frequency 0 in this bucket are filtered
    (they wouldn't have been sampled by the old proportions path either).
    ``None`` (legacy / non-NameGenerator callers) keeps the
    unweighted-pool behavior bit-stable.

    Hot path: the inner ``score()`` call is unchanged from the
    legacy inline form, so cached + uncached behavior is identical.
    """
    # Two-stage build (wyrd-bol9): first score every position+qualifier-
    # eligible Meaning, group by usage. Then in the apply-frequency
    # phase, normalize WITHIN each usage so its total pick weight is
    # the usage's per-bucket empirical frequency (per-usage parity
    # with proportions). Without the per-usage normalization, a
    # 2-Meaning usage would get 2× the pick weight of a 1-Meaning
    # usage purely because it has more terms to sum — over-counting
    # dual-Meaning usages and amplifying whichever language family
    # those tend to attest (in Welsh: OE-Celtic dual usages amplify
    # OE because both Meanings contribute, even though proportions
    # picks the USAGE only once per slot).
    #
    # Score zero handling — under the no-mood default request
    # (uniform-1.0 semantic_tags via vector_kenning_adapter, empty
    # phon, no priors), an untagged Meaning legitimately scores 0
    # on every axis. Pre-bol9 the ``bs > 0`` filter dropped these
    # entirely; under the frequency layer that's wrong, because
    # proportions ADMITS those usages via the per-bucket weight
    # table regardless of tag presence. We collect every non-
    # negative score into eligible_by_usage and let the per-usage
    # normalizer decide: usages with at least one positive-scored
    # Meaning distribute freq by score ratio; usages with only
    # zero-scored Meanings distribute freq uniformly so the usage
    # remains pickable. Strictly-negative scores still get filtered
    # (a register that actively opposes a Meaning shouldn't surface
    # it).
    # wyrd-eyjk/D40: no position gate. A morpheme is its string and may fill
    # any slot; the data-driven restriction on WHICH morphemes appear in a
    # given position is carried by the per-(position) bucket frequency
    # (slot_bucket_key → usage_frequency_by_bucket), built from the
    # build-time position-derived proportions — a morpheme never observed in
    # this position looks up to 0 and is filtered there, never here. The
    # name/saint QUALIFIER gates below stay: they're tag-based (which morpheme
    # may fill a name/saint slot), orthogonal to position (D39).
    eligible_by_usage: dict[str, list[tuple[Meaning, float]]] = {}
    for m in non_position_eligible:
        if slot_qualifier == "name" and not m.is_name():
            continue
        if slot_qualifier == "saint" and m.usage.replace("-", "").lower() != "saint":
            continue
        lemma_ref = _lemma_ref_for(m)
        lemma_tags = frozenset(m.tags)
        lemma_phon = m.phonological_vector or _default_empty_phon_vector()
        bs = score(
            lemma_ref=lemma_ref,
            lemma_tags=lemma_tags,
            lemma_phon=lemma_phon,
            request=request,
            slot_position=slot_position,
            era_midpoint=era_midpoint,
            priors=priors,
        )
        if bs < 0:
            continue
        eligible_by_usage.setdefault(m.usage, []).append((m, bs))

    if slot_usage_frequency is None:
        return _emit_unweighted(eligible_by_usage)
    return _apply_per_usage_frequency(eligible_by_usage, slot_usage_frequency)


def _resolve_slot_usage_frequency(
    usage_frequency_by_bucket: dict[tuple, dict[str, float]] | None,
    slot_bucket_key: tuple | None,
) -> dict[str, float] | None:
    """Three-state lookup for the wyrd-bol9 frequency layer:

    * ``usage_frequency_by_bucket is None`` — caller opted out of
      frequency weighting (legacy / non-NameGenerator callers).
      Returns ``None`` so :func:`build_slot_base_scores` falls
      back to its pre-bol9 unweighted-pool behavior.
    * map present + bucket key present — returns the bucket's
      per-usage frequency dict for this slot.
    * map present + bucket key MISSING (or no slot bucket key
      supplied) — returns ``{}`` so every Meaning looks up to 0
      and is filtered. Matches proportions: a bucket with no rows
      is unreachable. NOT a fall-through to the ``None`` branch.

    wyrd-rogd.13: a bare slot's key may carry a derived ``wp-<position>``
    word-position element (see ``_flatten_struct_slots``). When that
    extended key has no bucket — a legacy bundle without the
    per-word-position stats, or a bucket family (e.g. saint) the stats
    never observed — fall back to the un-extended key, i.e. the general
    bare distribution: exactly pre-rogd.13 behavior. When the extended
    bucket EXISTS it is authoritative (a structured word unattested at
    this word position is correctly absent from it).
    """
    if usage_frequency_by_bucket is None:
        return None
    if slot_bucket_key is None:
        return {}
    frequency = usage_frequency_by_bucket.get(slot_bucket_key)
    # Invariant: a non-None slot_bucket_key is never empty — every key is a
    # word_to_key element tuple carrying at least the location element. On a
    # malformed empty key, a loud IndexError here beats a truthiness guard
    # that would silently read as "bucket missing" → slot unreachable.
    if (
        frequency is None
        and isinstance(slot_bucket_key[-1], str)
        and slot_bucket_key[-1].startswith(WORD_POSITION_KEY_PREFIX)
    ):
        frequency = usage_frequency_by_bucket.get(slot_bucket_key[:-1])
    return frequency if frequency is not None else {}


def _emit_unweighted(
    eligible_by_usage: dict[str, list[tuple[Meaning, float]]],
) -> list[tuple[Meaning, float]]:
    """Flatten the per-usage-grouped pool back to a flat
    ``(meaning, score)`` list, dropping strictly-zero scores
    (matches pre-bol9 behavior for callers that opted out of the
    frequency layer). ``weighted_choice`` skips weight 0 anyway, but
    the explicit filter keeps the diagnostic cleaner."""
    out: list[tuple[Meaning, float]] = []
    for items in eligible_by_usage.values():
        for m, s in items:
            if s > 0:
                out.append((m, s))
    return out


def _apply_per_usage_frequency(
    eligible_by_usage: dict[str, list[tuple[Meaning, float]]],
    slot_usage_frequency: dict[str, float],
) -> list[tuple[Meaning, float]]:
    """Compose the bucket's per-usage empirical frequency with each
    Meaning's vector score (wyrd-bol9 frequency-weighted pool).

    Per-usage normalization: each Meaning's weight is
    ``freq[usage] × (score[meaning] / score_sum[usage])``. Cross-
    usage weights sum to ``freq[usage]`` (matches proportions'
    per-usage weighted sampling); within-usage relative weight
    follows the D36.2 score ratio (preserves per-Meaning
    discrimination).

    When all surviving Meanings of a usage tie at score 0 (untagged-
    only usages, common in Celtic corpora), distribute ``freq``
    uniformly within the usage so the usage stays pickable. The
    specific Meaning picked doesn't affect lang-mix output —
    ``NewName`` resolves the full Meaning list per usage downstream
    regardless of which Meaning was picked.
    """
    # wyrd-eyjk/D40: the bucket frequency is keyed by recorded POSITION-FORMS
    # (`-giles`, `Saint`), but the eligible pool is keyed by each Meaning's
    # stored usage variant (`Giles-`, `Saint-`). Collapse both to bare surface
    # so a morpheme's per-position frequency is found regardless of which
    # dash-variant the bundle stored it under.
    freq_by_surface: dict[str, float] = {}
    for form, f in slot_usage_frequency.items():
        surface = form.lower().replace("-", "")
        freq_by_surface[surface] = freq_by_surface.get(surface, 0.0) + f
    out: list[tuple[Meaning, float]] = []
    for usage, items in eligible_by_usage.items():
        freq = freq_by_surface.get(usage.lower().replace("-", ""), 0.0)
        if freq <= 0:
            continue
        score_sum = sum(s for _, s in items)
        if score_sum > 0:
            for m, s in items:
                if s > 0:
                    out.append((m, freq * s / score_sum))
        else:
            per_meaning = freq / len(items)
            for m, _ in items:
                out.append((m, per_meaning))
    return out


def _slot_position_for(element: str) -> str:
    """Resolve a structure element to a position label. Accepts either a literal
    position label (``pre``/``inner``/``post``/``bare`` — the caller's choice,
    skipping the encode/decode round-trip the legacy struct walker uses) or a
    structural-element string the heuristic decodes. (``bare`` added in
    wyrd-vpri for no-dash single-word slots.)"""
    if element in ("pre", "inner", "post", "bare"):
        return element
    return _slot_position_label(element)


def _slot_weighted_pool(
    non_position_eligible: list[Meaning],
    *,
    slot_position: str,
    slot_qualifier: str | None,
    slot_bucket_key: tuple | None,
    request: RequestVector,
    priors: EmpiricalPriors,
    cohesion: float,
    cohesion_table: dict[str, dict[str, float]] | None,
    usage_frequency_by_bucket: dict[tuple, dict[str, float]] | None,
    novelty: float,
    prior_tags: frozenset[str],
    slot_base_scores: dict[tuple, list[tuple[Meaning, float]]] | None,
    require_tags: frozenset[str] = frozenset(),
    # The baseline era axis (D36.7 priors cells). 0 = wildcard cell (the
    # default). wyrd-3tvd: a bounded HISTORICAL request era now drives this
    # with its midpoint; an era with no upper bound (present-day) stays 0.
    era_midpoint: int = 0,
) -> list[tuple[Meaning, float]]:
    """The weighted ``(meaning, score)`` pool for one slot: base scores ×
    per-sample cohesion bias, then the novelty blend. Returns an empty list when
    the slot collapses — no eligible meanings, or every score <= 0 after
    cohesion.

    ``require_tags`` (wyrd-c6o1.4): when non-empty, restrict the pool to morphemes
    carrying one of these tags BEFORE cohesion/novelty — used for the reserved
    --tag slot so the HARD guarantee survives cohesion. Applying it at membership
    level (not post-hoc on the final ``weighted``) is load-bearing: cohesion can
    drive a tagged candidate's ``final_score`` to 0 and drop it, so a post-hoc
    filter could empty out and silently fall back to an un-tagged pool. Filtering
    base_scored first means cohesion re-weights WITHIN the tagged subset, which a
    positive ``mean_raw`` (or the all-zero → all-1.0 no-op) keeps non-empty.

    Base scores are request-deterministic (no cohesion / no sampling), so they're
    memoized per ``(slot_position, slot_qualifier, slot_bucket_key)`` on the
    caller-supplied dict — reused across THIS call's struct retries (the dict is
    per-call and discarded on return; wyrd-i7uy: never across requests).
    The full bucket key keeps single- vs multi-element bucket variants of the same
    (position, qualifier) distinct — they multiply by different per-bucket
    frequencies, so sharing a cache entry would mis-score (wyrd-bol9)."""
    cache_key = (slot_position, slot_qualifier, slot_bucket_key)
    if slot_base_scores is not None and cache_key in slot_base_scores:
        base_scored = slot_base_scores[cache_key]
    else:
        base_scored = build_slot_base_scores(
            non_position_eligible,
            slot_position=slot_position,
            request=request,
            priors=priors,
            era_midpoint=era_midpoint,
            slot_qualifier=slot_qualifier,
            slot_usage_frequency=_resolve_slot_usage_frequency(
                usage_frequency_by_bucket, slot_bucket_key
            ),
        )
        if slot_base_scores is not None:
            slot_base_scores[cache_key] = base_scored
    if not base_scored:
        return []
    # wyrd-c6o1.4: reserved --tag slot — narrow to tagged morphemes on a LOCAL
    # copy (the cached base_scored stays tag-agnostic + reusable across structs).
    # An empty result is caught by the `if not weighted` guard below.
    if require_tags:
        base_scored = [(m, bs) for m, bs in base_scored if not require_tags.isdisjoint(m.tags)]

    # Cohesion depends on prior_tags, which evolves across slots within a single
    # name, so it's per-sample and NOT cacheable. Score each candidate's raw
    # tag-coherence against the already-picked slots, then reweight
    # slot-relative (mean-normalized) so above-average-coherence candidates
    # gain weight and below-average lose it, with the slot's total weight
    # preserved (D17). Skip the per-candidate raw walk entirely at the default
    # cohesion=0 (it would be discarded by _cohesion_multipliers anyway) so the
    # hot default path pays nothing.
    if cohesion > 0:
        raws = [
            _cohesion_raw(frozenset(m.tags), prior_tags, cohesion_table) for m, _ in base_scored
        ]
        multipliers = _cohesion_multipliers(raws, cohesion)
    else:
        multipliers = [1.0] * len(base_scored)
    weighted: list[tuple[Meaning, float]] = []
    for (m, bs), mult in zip(base_scored, multipliers, strict=True):
        final_score = bs * mult
        if final_score > 0:
            weighted.append((m, final_score))
    if not weighted:
        return []
    # wyrd-fcub: novelty blends the score distribution toward uniform over the
    # eligible pool, applied LAST just before the weighted draw (mirrors the
    # retired proportions path's _blend_uniform).
    if novelty > 0:
        weighted = _blend_uniform_by_novelty(weighted, novelty)
    return weighted


def _mood_morpheme_weight(meaning, mood_tags: dict[str, float]) -> float:
    """wyrd-4rp8: how strongly ``meaning`` fits the mood — the max mood-tag
    weight it carries (grim → a death morpheme scores 0.7, a military one 0.3).
    Multiplied onto the morpheme's normal freq×score weight inside the mood slot
    so the draw still favours frequent + position-natural morphemes but leans
    toward the more on-theme ones. Returns 0 for a morpheme with no mood tag (it
    is filtered out of the mood slot before this is applied)."""
    return max((mood_tags[t] for t in meaning.tags if t in mood_tags), default=0.0)


def _restrict_mood_slot(
    weighted: list[tuple[Meaning, float]],
    *,
    is_mood_slot: bool,
    mood_tags: dict[str, float],
) -> list[tuple[Meaning, float]]:
    """wyrd-4rp8 mood overlay (SOFT): on the reserved mood slot, restrict to
    mood-tagged morphemes, re-weighted by mood fit (× the strongest mood-tag
    weight) on top of the normal freq×score — so the mood morpheme is still picked
    by freq × mood-fit even at novelty=1 (the mood slot stays attestation-led while
    the rest of the name flattens). Graceful: an empty restriction falls back to
    the full pool (the mood is best-effort, unlike the hard --tag).

    The hard --tag overlay is NOT here: it restricts the slot's pool at the
    membership level inside ``_slot_weighted_pool`` (``require_tags``), so cohesion
    can't drop the only tagged candidate and silently un-guarantee the name."""
    if not is_mood_slot or not weighted:
        return weighted
    mood_tag_set = frozenset(mood_tags)
    mood_restricted = [
        (m, wt * _mood_morpheme_weight(m, mood_tags))
        for m, wt in weighted
        if not mood_tag_set.isdisjoint(m.tags)
    ]
    return mood_restricted or weighted


def _choose_tagged_slot(
    rng: random.Random,
    structure_list: list[str],
    *,
    target_tags: frozenset[str],
    non_position_eligible: list[Meaning],
    slot_qualifiers: list[str | None] | None,
    slot_bucket_keys: list[tuple] | None,
    request: RequestVector,
    priors: EmpiricalPriors,
    era_midpoint: int,
    cohesion: float,
    cohesion_table: dict[str, dict[str, float]] | None,
    usage_frequency_by_bucket: dict[tuple, dict[str, float]] | None,
    novelty: float,
    slot_base_scores: dict[tuple, list[tuple[Meaning, float]]] | None,
    require_tags: frozenset[str] = frozenset(),
    exclude_index: int | None = None,
) -> int | None:
    """Pick the slot that will carry a ``target_tags`` morpheme — the single-slot
    overlay shared by the thematic mood (wyrd-4rp8, soft) and the --tag guarantee
    (wyrd-c6o1.4, hard). A slot is *capable* when its ``require_tags``-restricted
    pool is non-empty; one capable slot is chosen at random. Returns ``None`` when
    no slot can carry the tag — the mood caller then comes out un-themed (soft, no
    failure); the --tag caller treats ``None`` as a hard struct failure and
    retries (see select_via_vector_scoring), so the >=1-tagged guarantee holds.

    The HARD --tag caller passes ``require_tags`` so capability is checked with the
    SAME membership restriction the fill loop uses — a slot reported capable cannot
    then collapse to empty (the cohesion-drop trap: a post-hoc filter on the
    un-restricted pool could find the tagged candidate already dropped by
    cohesion). The SOFT mood caller leaves ``require_tags`` empty and keeps the
    post-hoc ``target_tags`` membership probe (byte-identical to pre-c6o1.4 — mood
    degrades gracefully so it needn't survive the cohesion drop).
    ``prior_tags=frozenset()`` here is only the capability probe; the fill
    re-applies real priors.

    ``exclude_index`` (the already-reserved mood slot) is dropped from the
    candidate set so --tag and a mood land on DIFFERENT slots when possible; it is
    only re-admitted if it is the sole capable slot.

    Reuses the ``slot_base_scores`` cache the main pick loop reads, so the only
    added cost is the membership check + one ``rng`` draw."""
    capable: list[int] = []
    for slot_index, element in enumerate(structure_list):
        slot_qualifier = slot_qualifiers[slot_index] if slot_qualifiers is not None else None
        slot_bucket_key = slot_bucket_keys[slot_index] if slot_bucket_keys is not None else None
        weighted = _slot_weighted_pool(
            non_position_eligible,
            slot_position=_slot_position_for(element),
            slot_qualifier=slot_qualifier,
            slot_bucket_key=slot_bucket_key,
            request=request,
            priors=priors,
            era_midpoint=era_midpoint,
            cohesion=cohesion,
            cohesion_table=cohesion_table,
            usage_frequency_by_bucket=usage_frequency_by_bucket,
            novelty=novelty,
            prior_tags=frozenset(),
            slot_base_scores=slot_base_scores,
            require_tags=require_tags,
        )
        # Hard --tag: the pool is already tag-restricted, so non-empty = capable.
        # Soft mood: post-hoc membership probe on the un-restricted pool (the
        # byte-identical pre-c6o1.4 check).
        if require_tags:
            capable_here = bool(weighted)
        else:
            capable_here = bool(weighted) and any(
                not target_tags.isdisjoint(m.tags) for m, _ in weighted
            )
        if capable_here:
            capable.append(slot_index)
    # Prefer a slot distinct from the reserved mood slot, but fall back to it when
    # it is the only one that can carry the tag (the draw must stay non-empty).
    preferred = [i for i in capable if i != exclude_index] if exclude_index is not None else capable
    pool = preferred or capable
    if not pool:
        return None
    return pool[rng.randrange(len(pool))]


def select_via_vector_scoring(
    rng: random.Random,
    meaning_db: dict[str, list[Meaning]],
    *,
    structure: Iterable[str],
    request: RequestVector,
    priors: EmpiricalPriors,
    era_midpoint: int = 0,
    cohesion: float = 0.0,
    cohesion_table: dict[str, dict[str, float]] | None = None,
    exclude_tags: frozenset[str] = frozenset(),
    pack_meaning_dbs: dict[str, dict[str, list[Meaning]]] | None = None,
    non_position_eligible: list[Meaning] | None = None,
    slot_base_scores: dict[tuple, list[tuple[Meaning, float]]] | None = None,
    slot_qualifiers: list[str | None] | None = None,
    slot_bucket_keys: list[tuple] | None = None,
    usage_frequency_by_bucket: dict[tuple, dict[str, float]] | None = None,
    novelty: float = 0.0,
    permissive: bool = False,
) -> list[Meaning | None]:
    """Pick one Meaning per slot in the structure via the D36.2
    composition rule.

    Args:
        rng: seeded Random instance — caller's seed-stable contract.
        meaning_db: mapping ``usage → [Meaning, ...]`` (the
            ``meaning_db`` from :func:`load_meanings`).
        structure: iterable of structural element strings (e.g.
            ``["Place-", "-shire"]``). One slot per element.
        request: composed :class:`RequestVector` from the front-end.
            ``request.packs`` (a tuple of :class:`PackOverlay`)
            declares which scenario-pack overlays are in play; the
            per-pack lemmas come from ``pack_meaning_dbs`` (keyed by
            pack name).
        priors: loaded :class:`EmpiricalPriors` (the JSON sidecar).
        era_midpoint: int year used by ``baseline_score``'s era axis (the
            D36.7 per-era frequency cells). 0 = wildcard (no era-fashion lean,
            the default). wyrd-3tvd: a bounded HISTORICAL request era now
            drives this with its midpoint; an era with no upper bound
            (present-day) stays 0.
        cohesion: D17 cohesion knob in [0, 1]. 0 = no bias; 1 =
            strongest tag-overlap bias.
        cohesion_table: nested ``{candidate_tag → {prior_tag → prob}}``
            conditional co-occurrence table, derived by
            :func:`proportions.build_cohesion_table` from the bundle's
            flat counts + tag_marginal. None / empty = no cohesion bias
            even when cohesion > 0 (graceful degrade).
        exclude_tags: meanings carrying ANY tag in this set are
            filtered out pre-score. Used by the fiction / curated-
            exclusion paths.
        pack_meaning_dbs: per-pack lemma DBs keyed by pack name. Each
            value is a meaning_db (same shape as the native
            ``meaning_db`` arg) holding that pack's lemmas. Only
            packs declared in ``request.packs`` are walked; an
            unknown pack name in request.packs (no matching
            pack_meaning_dbs key) contributes no lemmas (pack still
            scores 0 via ``baseline_score_pack`` for any native
            lemma that happens to share lemma_ref). wyrd-ecjp.8.

        permissive: wyrd-tbke graceful-degradation switch. When False
            (default), a slot whose gated pool is empty aborts the whole
            struct (returns ``[]``) so the caller can retry a different
            struct. When True, such a slot instead contributes ``None`` and
            scoring continues — matching the retired proportions path's
            empty-bucket contract (an empty bucket → a ``None`` slot the
            NewName drops), so the vector path degrades to a shorter
            name instead of failing. The caller uses this as a last-resort
            fallback once no struct is fully satisfiable.

            EXCEPTION (wyrd-c6o1.4): a HARD ``--tag`` that no slot in this struct
            can carry returns ``[]`` EVEN under permissive — the tag is a hard
            guarantee, not a per-slot collapse to degrade around, so an
            un-placeable tag must drive a struct retry (and ultimately the
            caller's "no eligible name") rather than a silently un-tagged name.

    Returns:
        list[Meaning | None] — one entry per structural slot, in order.
        Non-permissive: empty list when the gate filters every meaning
        out for any slot (the caller decides whether to retry or fall
        back). Permissive: always full-length with ``None`` for unsatisfiable
        slots — EXCEPT an un-placeable ``required_tags`` (above) which returns
        ``[]`` even here.
    """
    picked: list[Meaning] = []
    prior_tags: set[str] = set()
    gate = request.gate

    # Build the non-position eligibility pool — the caller (select_via_vector)
    # passes a per-call pool so the struct-retry loop reuses one build within
    # this dispatch. wyrd-i7uy: per-call only, never cached across requests.
    if non_position_eligible is None:
        non_position_eligible = build_non_position_eligible(
            meaning_db,
            gate=gate,
            exclude_tags=exclude_tags,
            pack_meaning_dbs=pack_meaning_dbs,
            packs=request.packs,
        )

    structure_list = list(structure)
    # wyrd-4rp8: thematic-mood overlay. When a mood is requested, reserve ONE
    # slot (randomly, among those whose pool can carry a mood tag) to draw a
    # mood-appropriate morpheme; every other slot generates normally. No mood
    # (``request.mood_tags`` empty) → no draw → byte-identical to plain
    # generation, so the no-mood RNG stream + realism gate are untouched.
    mood_tag_set = frozenset(request.mood_tags)
    mood_slot_index = (
        _choose_tagged_slot(
            rng,
            structure_list,
            target_tags=mood_tag_set,
            non_position_eligible=non_position_eligible,
            slot_qualifiers=slot_qualifiers,
            slot_bucket_keys=slot_bucket_keys,
            request=request,
            priors=priors,
            era_midpoint=era_midpoint,
            cohesion=cohesion,
            cohesion_table=cohesion_table,
            usage_frequency_by_bucket=usage_frequency_by_bucket,
            novelty=novelty,
            slot_base_scores=slot_base_scores,
        )
        if mood_tag_set
        else None
    )
    # wyrd-c6o1.4: --tag overlay. Reserve ONE slot (distinct from the mood slot
    # when possible) to carry a required-tag morpheme; every other slot samples
    # freely. This restores the historical "names lean toward containing a tagged
    # morpheme, the rest is free" semantics — vs the wyrd-wv85 pool-wide gate that
    # forced EVERY slot tagged. No --tag (gate.required_tags empty) → no draw → the
    # RNG stream + realism gate are untouched (parity with the no-mood path).
    required_tags = gate.required_tags
    tag_slot_index = (
        _choose_tagged_slot(
            rng,
            structure_list,
            target_tags=required_tags,
            non_position_eligible=non_position_eligible,
            slot_qualifiers=slot_qualifiers,
            slot_bucket_keys=slot_bucket_keys,
            request=request,
            priors=priors,
            era_midpoint=era_midpoint,
            cohesion=cohesion,
            cohesion_table=cohesion_table,
            usage_frequency_by_bucket=usage_frequency_by_bucket,
            novelty=novelty,
            slot_base_scores=slot_base_scores,
            require_tags=required_tags,
            exclude_index=mood_slot_index,
        )
        if required_tags
        else None
    )
    # wyrd-c6o1.4: required_tags is a HARD guarantee, not the soft "maybe" a mood
    # is. If NO slot in THIS struct can carry the tag, signal a struct retry
    # (return []) exactly as an empty gated pool did under wv85 — the caller tries
    # other structs until one can place the tag. Exhausting every struct (a typo
    # like "religion" that no morpheme carries, or a tag valid only in positions
    # absent from all structs) surfaces as the caller's "no eligible name". This
    # is what keeps --tag a guarantee while the rest of the name samples freely.
    if required_tags and tag_slot_index is None:
        return []

    for slot_index, element in enumerate(structure_list):
        slot_position = _slot_position_for(element)
        # wyrd-izcr: per-slot qualifier ("name" / "saint" / None) and wyrd-bol9:
        # the full bucket key. Both ride lockstep with the structure — the sole
        # caller (select_via_vector) builds slot_qualifiers / slot_bucket_keys in
        # the same loop that builds the structure — so a length mismatch is a
        # programming error that should surface as IndexError, not silently
        # coerce to None. Legacy / non-NameGenerator callers pass neither, and
        # the cache key collapses to the prior (position, None, None) semantics.
        slot_qualifier: str | None = (
            slot_qualifiers[slot_index] if slot_qualifiers is not None else None
        )
        slot_bucket_key = slot_bucket_keys[slot_index] if slot_bucket_keys is not None else None
        weighted = _slot_weighted_pool(
            non_position_eligible,
            slot_position=slot_position,
            slot_qualifier=slot_qualifier,
            slot_bucket_key=slot_bucket_key,
            request=request,
            priors=priors,
            era_midpoint=era_midpoint,
            cohesion=cohesion,
            cohesion_table=cohesion_table,
            usage_frequency_by_bucket=usage_frequency_by_bucket,
            novelty=novelty,
            prior_tags=frozenset(prior_tags),
            slot_base_scores=slot_base_scores,
            # wyrd-c6o1.4: HARD --tag — restrict the reserved slot's pool to tagged
            # morphemes at the membership level (before cohesion), so the >=1
            # guarantee survives cohesion. Other slots stay tag-agnostic.
            require_tags=required_tags if slot_index == tag_slot_index else frozenset(),
        )
        # wyrd-4rp8: SOFT mood overlay on the reserved mood slot (post-hoc, falls
        # back to the full pool). The mood + tag slots can coincide only when the
        # tag's sole capable slot IS the mood slot; then the pool is already
        # tag-restricted (above) and the mood re-weights within it — hard tag wins.
        weighted = _restrict_mood_slot(
            weighted,
            is_mood_slot=slot_index == mood_slot_index,
            mood_tags=request.mood_tags,
        )
        # wyrd-tbke: permissive degrades an unsatisfiable slot to None and
        # continues (full-length result); non-permissive aborts the whole
        # struct. A collapsed pool (empty `weighted`) and a None weighted-draw
        # are the two ways a slot can fail to fill.
        if not weighted:
            if permissive:
                picked.append(None)
                continue
            return []
        picked_meaning = _weighted_choice(rng, weighted)
        if picked_meaning is None:
            if permissive:
                picked.append(None)
                continue
            return []
        picked.append(picked_meaning)
        # D17: accumulate picked tags for next slot's cohesion bias
        prior_tags.update(picked_meaning.tags)

    return picked


_BUNDLE_FIELD_TO_L3_LANG: dict[str, str] = {
    # Reverse of ``lexicon.bundle._emit._LANG_CODE_TO_JSON_FIELD``'s
    # primary L3 → bundle-field mapping. Used by ``_lemma_ref_for`` to
    # synthesize the ``language:canonical_form`` shape the priors tables
    # key on (per ``empirical_priors.extract_priors`` —
    # ``f"{row['language']}:{row['canonical_form']}"``).
    #
    # Lossy because multiple L3 langs collapse into one bundle field
    # (e.g. welsh + irish + old-irish + ... → celtic_mix). For each
    # bucket we pick the representative L3 code that surfaces in the
    # empirical-priors miner's output, so the baseline lookup hits the
    # priors data we ship today rather than missing on a code the
    # priors table doesn't carry.
    #
    # Wave-1 (English-corpus dominant):
    "old_english": "old-english",
    "old_scandinavian": "old-norse",
    "old_french": "norman-french",
    "modern_english": "modern-english",
    "celtic_mix": "celtic",
    "latin": "latin",
    "germanic": "germanic",
    "greek": "greek",
    "biblical": "biblical",
    # Wave-2 non-Latin buckets (wyrd-vsrn Phase 2c). Use the canonical
    # wikt language code that ingest routes into the bucket — same
    # values appear in the L3 ``etymon.language`` column for these
    # languages, so the priors lookup hits exactly.
    "hebrew": "he",
    "arabic": "ar",
    "persian": "fa",
    "sanskrit": "sa",
    "akkadian": "akk",
    "egyptian": "egy",
    "aramaic": "arc",
    "armenian": "axm",
}


def _lemma_ref_for(meaning: Meaning) -> str:
    """Resolve the ``lemma_ref`` string the priors-baseline lookup
    keys on (``f"{l3_language}:{canonical_form}"`` — same shape
    ``empirical_priors.extract_priors`` emits).

    Walks ``meaning.sources`` (dict[bundle_lang_field, list[form]]) in
    alphabetical order; for each language it scans ALL form entries and takes
    the FIRST non-empty one (so a leading-empty ``["", "wīc"]`` still attests
    via the later form — matching ``Meaning.primary_language`` /
    ``Meaning._forms_nonempty``, the documented shared "non-empty form"
    semantics). Maps bundle-field → L3 lang via ``_BUNDLE_FIELD_TO_L3_LANG`` and
    returns ``f"{l3_lang}:{form}"``. The alphabetical source walk is
    deterministic + stable across re-runs.

    Falls back to the bare-usage form (legacy ecjp.5 v1 behavior) when
    a meaning has no source-language form at all (synthesized /
    sidecar meanings without per-language data). The baseline lookup
    misses on the bare form; scoring degrades to phon/sem/pos per the
    graceful-degrade contract.
    """
    for lang_field in sorted(meaning.sources):
        forms = meaning.sources[lang_field]
        if not forms:
            continue
        l3_lang = _BUNDLE_FIELD_TO_L3_LANG.get(lang_field, lang_field.replace("_", "-"))
        # forms entries are usually plain strings; tolerate dict-shaped entries
        # (some bundle schemas wrap forms with metadata) by picking the
        # canonical-form-ish key. Scan ALL entries, not just forms[0]: a language
        # whose first form is empty (``["", "wīc"]``) still attests via a later
        # non-empty form. This matches ``Meaning.primary_language`` /
        # ``Meaning._forms_nonempty`` — the documented shared "non-empty form"
        # semantics. Checking only forms[0] diverged: ``primary_language`` (the
        # pool-eligibility key) scans all entries, so the same Meaning could be
        # keyed on one language for eligibility and a different one for this
        # priors-baseline lookup.
        for entry in forms:
            if isinstance(entry, dict):
                form_str = entry.get("form") or entry.get("canonical_form") or ""
            else:
                form_str = entry
            if form_str:
                return f"{l3_lang}:{form_str}"
    return meaning.usage.replace("-", "")


def _blend_uniform_by_novelty(
    weighted: list[tuple[Meaning, float]], novelty: float
) -> list[tuple[Meaning, float]]:
    """wyrd-fcub: interpolate a (meaning, score) distribution toward a uniform
    marginal by ``novelty`` in [0, 1].

    Each blended weight is ``(1 - novelty)·(score / total) + novelty·(1 / n)`` —
    a normalized distribution. novelty=0 returns the score-normalized
    distribution unchanged (up to scaling, which weighted_choice is invariant
    to); novelty=1 returns pure-uniform 1/n. Ported from the retired
    proportions ``_blend_uniform`` so the lexical-surprise knob behaves
    identically under vector generation. Callers fast-path novelty<=0.
    """
    n = len(weighted)
    if n == 0:
        return weighted
    total = sum(score for _, score in weighted)
    if total <= 0:
        # No score signal to blend against — uniform is the only meaningful axis.
        return [(m, 1.0 / n) for m, _ in weighted]
    # Hoist the per-element constants out of the comprehension: blended weight
    # is score * factor + term == (1-novelty)*(score/total) + novelty/n.
    factor = (1.0 - novelty) / total
    term = novelty / n
    return [(m, score * factor + term) for m, score in weighted]


def _weighted_choice(
    rng: random.Random,
    weighted: list[tuple[Meaning, float]],
) -> Meaning | None:
    """Weighted sampling over (meaning, score) pairs. Mirrors the
    semantics of :func:`runtime.proportions.weighted_choice` (skip
    non-positive weights, cumulative-threshold construction) so the
    bit-stability contract on a given rng + same weighted list
    matches the legacy path's behavior.

    Returns None when every weight is zero / negative — caller's
    contract is to treat that as "no pick possible" and either retry
    or fall back.
    """
    space: dict[float, Meaning] = {}
    current = 0.0
    for meaning, weight in weighted:
        if weight > 0:
            space[current] = meaning
            current += weight
    if current == 0:
        return None
    rand = rng.uniform(0, current)
    last_choice: Meaning | None = None
    for key in sorted([*space.keys(), current]):
        if rand < key:
            return last_choice
        if key in space:
            last_choice = space[key]
    return last_choice
