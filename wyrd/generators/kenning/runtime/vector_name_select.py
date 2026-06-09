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
     predicates in :class:`EligibilityGate` (culture, era, stratum,
     position). Lemmas that fail the gate never enter scoring.
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

import random
from collections.abc import Iterable
from typing import TYPE_CHECKING

# wyrd-glos: the gloss-policy predicate is defined in proportions (the
# canonical keep_keys_for_gloss home). Safe module-level import — proportions
# imports vector_name_select only lazily (inside select_via_vector), so there
# is no module-load cycle; this module is itself imported lazily, by which
# point proportions is fully loaded.
from wyrd.generators.kenning.runtime.proportions import _gloss_eligible
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


def _matches_era(
    meaning: Meaning,
    era_min: int | None,
    era_max: int | None,
) -> bool:
    """Era-gate predicate. Delegates to ``Meaning.attested_in_era_range``
    — open range (None on both sides) is pass-through; meanings with
    no attested-years evidence also pass through (the same convention
    as the legacy path).
    """
    if era_min is None and era_max is None:
        return True
    return meaning.attested_in_era_range((era_min, era_max))


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


def _cohesion_multiplier(
    meaning_tags: frozenset[str],
    prior_tags: frozenset[str],
    cohesion: float,
    tag_cooccurrence: dict[str, dict[str, float]] | None,
) -> float:
    """D17 cohesion adapter: returns a multiplicative score bias
    based on tag co-occurrence overlap with previously-picked slots'
    tags.

    Formula: ``1.0 + cohesion * avg_p`` where ``avg_p`` is the mean
    co-occurrence probability across all (lemma_tag, prior_tag)
    pairs that have data in ``tag_cooccurrence``.

    Edge values:
      * ``cohesion == 0`` (default) — returns 1.0 (no bias, every
        slot scores independently)
      * empty ``prior_tags`` (first slot in a name) — returns 1.0
      * ``tag_cooccurrence is None`` (legacy / empty bundle) — returns
        1.0 (same bit-stable degradation pattern as the legacy cohesion
        path)
      * ``cohesion == 1`` + ``avg_p == 1.0`` — returns 2.0 (the
        maximum bias under the formula; ``avg_p`` is bounded in
        [0, 1] since it's an average of co-occurrence probabilities).
      * Intermediate values linearly blend.

    The bias is per-slot multiplicative on top of the base score, so
    the composition stays consistent with the canonical formula:
    ``score'(lemma) = score(lemma) * cohesion_mult(lemma_tags, prior_tags)``.
    """
    if cohesion <= 0 or not prior_tags or not tag_cooccurrence:
        return 1.0
    # For each (lemma_tag, prior_tag) pair, look up the empirical
    # co-occurrence probability + sum. Higher overlap → higher bias.
    overlap_sum = 0.0
    pair_count = 0
    for ltag in meaning_tags:
        ltag_co = tag_cooccurrence.get(ltag)
        if not ltag_co:
            continue
        for ptag in prior_tags:
            p = ltag_co.get(ptag)
            if p is None:
                continue
            overlap_sum += p
            pair_count += 1
    if pair_count == 0:
        return 1.0
    # Average co-occurrence probability over the pairs we found.
    avg_p = overlap_sum / pair_count
    # Multiplicative bias: at avg_p=0 → 1.0; at avg_p=1 → 1.0 + cohesion.
    # cohesion=1 doubles the bias for perfectly-cooccurring tags; cohesion=0
    # is the no-bias identity. Linear in cohesion to keep the knob
    # legible to operators.
    return 1.0 + cohesion * avg_p


def _slot_position_label(structural_element: str) -> str:
    """Resolve the slot position label from a structural element string.

    Matches ``Meaning._set_location`` exactly. wyrd-eyjk/D40: this label is
    no longer used to *gate* eligibility (the ``_matches_position`` gate is
    gone) — it feeds the D36 position-axis SCORING (``pos_score``) and the
    per-(position) bucket-key lookup. The mapping (from
    ``runtime/meaning.py:_set_location``):

    * ``"-inner-"`` (both dashes) → ``"inner"``
    * ``"Place-"`` (trailing dash only) → ``"pre"``
    * ``"-shire"`` (leading dash only) → ``"post"``
    * ``"Bare"`` (no dashes) → ``"bare"``

    wyrd-vpri: bare (no-dashes) maps to its own ``"bare"`` label, NOT
    ``"post"`` — the historical conflation that let suffix-only keys
    ('-park' → post) be treated as bare single-word slots. Must stay in
    lockstep with ``Meaning._set_location``.
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
    """Slot-independent eligibility gates shared by the native and pack pools:
    era window, stratum, exclude-tags, the --tag HARD gate, and the gloss
    policy. Returns False on the first failing gate."""
    if not _matches_era(m, gate.era_min, gate.era_max):
        return False
    if not _matches_stratum(m, gate.stratum):
        return False
    # frozenset.isdisjoint runs in C and skips the per-tag generator overhead
    # in this O(64k) hot path; equivalent to any(t in <fs> for t in m.tags).
    if exclude_tags and not exclude_tags.isdisjoint(m.tags):
        return False
    # wyrd-wv85: --tag HARD gate (D36.6). Drop lemmas carrying none of the
    # requested tags — OR semantics, matching proportions' filter_for_tag.
    # Empty required_tags = no-op.
    if gate.required_tags and gate.required_tags.isdisjoint(m.tags):
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
    # wyrd-eyjk/D40 + wyrd-g1hj: exclude pure-proper-noun saint subjects AND
    # personal given names from the base pool — saints reach names only via the
    # param-gated St-dedication synthesis; given names would dangle as bare
    # personal names. Family-name etymons are NOT excluded (manorial places).
    if m.is_pure_proper_noun() and (m.is_saint() or m.is_given_name()):
        return False
    # wyrd-gwj3: connector morphemes (cum / le / juxta …) require a complement.
    return not m.is_connector_particle()


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
        if (
            culture_attested_usages is not None
            and usage_key.lower().replace("-", "") not in culture_attested_usages
        ):
            continue
        # Per-Meaning narrowing set for this usage_key (wyrd-pfoo); None when
        # the filter is inactive or this usage carried no per-Meaning data.
        admitted_langs: frozenset[str] | None = (
            culture_attested_meanings.get(usage_key)
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

    Era, stratum, and exclude_tags don't depend on the slot — they
    apply uniformly across every slot in the structure. Pulling this
    out into a free function lets the caller cache the result across
    sub-seeds (the dispatch loop calls
    :meth:`NameGenerator.select_via_vector` once per sub-seed; without
    caching, this O(N=64k) scan re-runs every time).

    Caller-side cache key for the result:
    ``(id(meaning_db), gate, exclude_tags, packs, id(pack_meaning_dbs),
       culture_attested_usages, id(culture_attested_meanings))`` —
    same meaning_db + same filters + same pack overlays + same
    culture sets → identical output. The list is per-meaning
    order-stable across re-runs (we walk ``meaning_db.items()`` which
    is insertion-ordered in Python 3.7+), so the caller can re-use
    the cached list directly in the weighted-sampling loop without
    re-shuffling.

    ``culture_attested_usages`` is the per-usage culture-bleed filter
    (wyrd-361): only Meanings whose ``usage`` key appears in this
    culture's corpus are admitted to the native-pool. Without the
    filter, English-culture vector requests pull in Welsh / Persian /
    Hebrew morphemes that aren't attested in English place names.
    ``None`` disables the filter (back-compat for non-NameGenerator
    callers that don't carry per-culture data).

    ``culture_attested_meanings`` is the per-Meaning narrowing
    (wyrd-pfoo). For each usage_key, only Meanings whose primary
    source-language is in the per-usage attested set are admitted.
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


def request_signature(request: RequestVector) -> tuple:
    """Return a hashable signature of the ``RequestVector`` fields
    that affect ``score()``'s output. Used as a cache key by the
    slot-scoring path so two distinct ``RequestVector`` instances
    with identical content hit the same cached weighted list.

    The signature folds:

    * The gate (culture / era / stratum / required_tags). Score
      doesn't read these directly, but they're already filtered into
      the eligibility pool, so different gates need different caches —
      two requests differing only in ``--tag`` (required_tags) have
      different pools and must not share cached scores.
    * The register's three dict axes (``semantic_tags``,
      ``phonological``, ``position_bias``) as frozensets so dict-
      iteration order doesn't change the key. All three are read by
      one of ``phon_score`` / ``sem_score`` / ``pos_score``;
      omitting any would let cached scores survive a register-axis
      change.
    * The four ``ScoringWeights`` fields.
    * The ``packs`` tuple (each :class:`PackOverlay` is already a
      hashable frozen dataclass).
    """
    return (
        request.gate.culture,
        request.gate.era_min,
        request.gate.era_max,
        request.gate.stratum,
        request.gate.required_tags,
        frozenset(request.register.semantic_tags.items()),
        frozenset(request.register.phonological.items()),
        frozenset(request.register.position_bias.items()),
        request.weights.phon_w,
        request.weights.sem_w,
        request.weights.pos_w,
        request.weights.base_w,
        request.packs,
    )


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
    is per-sample). Caching this list across sub-seeds in a
    count>1 dispatch saves the O(P) score loop per sub-seed —
    typically the dominant cost in count=N vector latency.

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
    """
    if usage_frequency_by_bucket is None:
        return None
    if slot_bucket_key is None:
        return {}
    return usage_frequency_by_bucket.get(slot_bucket_key, {})


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


def select_via_vector_scoring(
    rng: random.Random,
    meaning_db: dict[str, list[Meaning]],
    *,
    structure: Iterable[str],
    request: RequestVector,
    priors: EmpiricalPriors,
    era_midpoint: int = 0,
    cohesion: float = 0.0,
    tag_cooccurrence: dict[str, dict[str, float]] | None = None,
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
        era_midpoint: int year used by ``baseline_score``'s era axis
            (typically the midpoint of the requested era_min / era_max
            range, or 0 when no era is requested → hits the fallback
            tag-wildcard / all-wildcard cells).
        cohesion: D17 cohesion knob in [0, 1]. 0 = no bias; 1 =
            strongest tag-overlap bias.
        tag_cooccurrence: ``{tag → {tag → cooccurrence_prob}}`` from
            the bundle (legacy data). None = no cohesion bias even
            when cohesion > 0 (graceful degrade).
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

    Returns:
        list[Meaning | None] — one entry per structural slot, in order.
        Non-permissive: empty list when the gate filters every meaning
        out for any slot (the caller decides whether to retry or fall
        back). Permissive: always full-length; unsatisfiable slots are
        ``None``.
    """
    picked: list[Meaning] = []
    prior_tags: set[str] = set()
    gate = request.gate

    # Build the non-position eligibility pool — caller can supply a
    # pre-built one (cached across sub-seeds) to skip the O(N=64k)
    # scan on count>1 dispatches.
    if non_position_eligible is None:
        non_position_eligible = build_non_position_eligible(
            meaning_db,
            gate=gate,
            exclude_tags=exclude_tags,
            pack_meaning_dbs=pack_meaning_dbs,
            packs=request.packs,
        )

    structure_list = list(structure)
    for slot_index, element in enumerate(structure_list):
        # Accept either a structural-element string ("X-"/"-X-"/"-X")
        # the heuristic decodes, or a position label ("pre"/"inner"/
        # "post"/"bare") — caller's choice. Position labels skip the
        # encode/decode round-trip the legacy struct walker uses to
        # synthesize a string from its key tuples. ("bare" added in
        # wyrd-vpri for no-dash single-word slots.)
        if element in ("pre", "inner", "post", "bare"):
            slot_position = element
        else:
            slot_position = _slot_position_label(element)

        # wyrd-izcr: per-slot qualifier filter ("name" / "saint" /
        # None) — set when the caller's structure marked the slot as
        # a name/saint qualifier word. Pre-fix the vector path
        # silently flattened structures to bare position labels and
        # filled qualifier-flagged slots with any matching morpheme;
        # cache key now includes the qualifier so a single dispatch
        # can mix (pre, None) and (pre, "name") slots safely.
        slot_qualifier: str | None = (
            slot_qualifiers[slot_index] if slot_qualifiers is not None else None
        )
        # wyrd-bol9: cache key now includes the full bucket key so
        # single-element / multi-element bucket variants of the same
        # (position, qualifier) get distinct cached score lists —
        # they multiply by different per-bucket frequencies, so
        # caching one in place of the other would produce wrong
        # picks. When the caller doesn't supply slot_bucket_keys
        # (legacy / non-NameGenerator callers), bucket_key is None
        # and the 3-tuple collapses to the prior 2-tuple semantics
        # at the trailing None slot. Mirrors the slot_qualifiers
        # lockstep assumption above — sole caller (select_via_vector)
        # builds slot_bucket_keys in the same loop that builds
        # candidate_positions, so a length mismatch is a programming
        # error that should surface as IndexError rather than
        # silently coerce to None.
        slot_bucket_key = slot_bucket_keys[slot_index] if slot_bucket_keys is not None else None
        cache_key = (slot_position, slot_qualifier, slot_bucket_key)

        # Base scores: cached on caller-supplied dict if provided,
        # else built fresh. The base score is request-deterministic
        # (no cohesion / no sampling) so sub-seeds in a count>1
        # dispatch reuse the cached list.
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
            if permissive:
                picked.append(None)
                continue
            return []

        # Apply per-sample cohesion bias + weighted choice. Cohesion
        # depends on prior_tags which evolves across slots within a
        # single name, so it's per-sample and NOT cacheable.
        prior_tags_frozen = frozenset(prior_tags)
        weighted: list[tuple[Meaning, float]] = []
        for m, bs in base_scored:
            cohesion_mult = _cohesion_multiplier(
                frozenset(m.tags), prior_tags_frozen, cohesion, tag_cooccurrence
            )
            final_score = bs * cohesion_mult
            if final_score > 0:
                weighted.append((m, final_score))
        if not weighted:
            if permissive:
                picked.append(None)
                continue
            return []
        # wyrd-fcub: novelty blends the score distribution toward uniform over
        # the eligible pool — novelty=0 samples by score (typical morphemes
        # dominate), novelty=1 samples every eligible meaning equally (surfaces
        # rare morphemes as often as common). Applied LAST, just before the
        # weighted draw, mirroring the retired proportions path's _blend_uniform.
        if novelty > 0:
            weighted = _blend_uniform_by_novelty(weighted, novelty)
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

    Walks ``meaning.sources`` (dict[bundle_lang_field, list[form]])
    in alphabetical order, picks the first non-empty entry, maps
    bundle-field → L3 lang via ``_BUNDLE_FIELD_TO_L3_LANG``, and
    returns ``f"{l3_lang}:{form}"``. The alphabetical walk is
    deterministic + stable across re-runs; ``old_english`` lands
    first under it which matches the dominant-English-priors-data
    we have today.

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
        # forms entries are usually plain strings; tolerate dict-shaped
        # entries (some bundle schemas wrap forms with metadata) by
        # picking the canonical-form-ish key.
        first = forms[0]
        if isinstance(first, dict):
            first = first.get("form") or first.get("canonical_form") or ""
        if first:
            return f"{l3_lang}:{first}"
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
