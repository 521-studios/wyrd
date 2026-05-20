"""Gate→score→sample primitive for vector-driven name generation
(wyrd-ecjp.5).

The legacy ``Generator.select`` pipeline does proportion-table
sampling: pre-baked per-(culture × tag × position) weight tables
drive ``weighted_choice`` at each slot. This module is the
vector-driven alternative — every slot's per-lemma weight is
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

This is the runtime side of ecjp.5. The Kenning generator dispatches
into here via the ``--scoring-mode vector`` opt-in; bit-stable legacy
behavior stays on the proportion-table path. ecjp.6/7 (realism-
retention drift measurement) empirically compares the two.

Out of scope for v1:
  * Multi-pack composition (wyrd-ecjp.8 / Phase 7).
  * Tolerance bands on the composed register (D8, deferred).
  * Per-slot inflection / variant resolution (handled the same way
    as the legacy path; this module just picks the lemma).
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import TYPE_CHECKING

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


def _matches_position(meaning: Meaning, slot_position: str) -> bool:
    """Position-gate predicate. The slot's position label is one of
    ``pre`` / ``inner`` / ``post`` (matches Meaning.location). A
    meaning eligible for a slot must have a matching location.

    The legacy path filters at the per-structure / per-bucket level;
    here we do the same check explicitly on the meaning's location
    field. Joiners and other non-Meaning structural elements are
    handled by the caller before reaching this function.
    """
    return meaning.location == slot_position


def _matches_era(
    meaning: Meaning,
    era_min: int | None,
    era_max: int | None,
) -> bool:
    """Era-gate predicate. Delegates to Meaning.attested_in_era_range
    where available; defaults to pass-through when the era range is
    open (None on both sides) or when the meaning has no attested-
    years evidence (the same convention as the legacy path)."""
    if era_min is None and era_max is None:
        return True
    range_arg = (era_min, era_max)
    attested_in = getattr(meaning, "attested_in_era_range", None)
    if callable(attested_in):
        return attested_in(range_arg)
    # Meaning predates the era-range method; default open (don't drop).
    return True


def _matches_stratum(meaning: Meaning, stratum: str | None) -> bool:
    """Stratum-gate predicate. Mirrors the legacy path: meanings with
    no stratum data pass through (only Welsh-family is classified
    today). ``None`` stratum disables the filter."""
    if stratum is None:
        return True
    matches_stratum_method = getattr(meaning, "has_stratum", None)
    if callable(matches_stratum_method):
        return matches_stratum_method(stratum)
    return True


def _cohesion_multiplier(
    meaning_tags: frozenset[str],
    prior_tags: frozenset[str],
    cohesion: float,
    tag_cooccurrence: dict[str, dict[str, float]] | None,
) -> float:
    """D17 cohesion adapter: returns a multiplicative score bias
    based on tag co-occurrence overlap with previously-picked slots'
    tags.

    At ``cohesion == 0`` (default) returns 1.0 — no bias, every slot
    scores independently. At ``cohesion == 1`` and the strongest
    overlap, returns up to ``1.0 + max_cooccurrence_bias`` (currently
    capped at 2.0). Intermediate cohesion values linearly blend.

    No tag_cooccurrence data (legacy / empty bundle) short-circuits
    to 1.0 — same bit-stable degradation pattern as the legacy
    cohesion path.

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

    Structural elements in the legacy bundle are strings like
    ``"Place-suffix"`` / ``"prefix-Place"`` / ``"prefix-Place-suffix"``.
    The Meaning.location field convention is ``pre`` / ``inner`` /
    ``post`` matching ``usage.startswith('-')`` / both / neither.

    For the vector-scoring path, we treat each STRUCTURAL slot
    independently:
      * ``"prefix"`` / token at index 0 with trailing dash → ``"pre"``
      * ``"suffix"`` / token at index -1 with leading dash → ``"post"``
      * anything else → ``"inner"``

    This is a heuristic — the structure walk's actual slot mechanics
    in the legacy path is more nuanced. For ecjp.5 v1, the heuristic
    is correct for the common pre/post compound shapes the corpus
    exhibits; the few outlier inner-slot cases get scored against
    pos_score(inner) which is the fallback intent.
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
    # No dashes (bare nominal element). Treat as inner — it sits
    # between the pre + post slots conceptually.
    return "inner"


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
) -> list[Meaning]:
    """Pick one Meaning per slot in the structure via the D36.2
    composition rule.

    Args:
        rng: seeded Random instance — caller's seed-stable contract.
        meaning_db: mapping ``usage → [Meaning, ...]`` (the
            ``meaning_db`` from :func:`load_meanings`).
        structure: iterable of structural element strings (e.g.
            ``["Place-", "-shire"]``). One slot per element.
        request: composed :class:`RequestVector` from the front-end.
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

    Returns:
        list[Meaning] — one Meaning per structural slot, in order.
        Empty list when the gate filters every meaning out for any
        slot (the caller decides whether to retry or fall back).
    """
    picked: list[Meaning] = []
    prior_tags: set[str] = set()
    gate = request.gate

    for element in structure:
        slot_position = _slot_position_label(element)
        # Gate: walk every meaning, filter to eligible
        eligible: list[Meaning] = []
        for meanings_for_usage in meaning_db.values():
            for m in meanings_for_usage:
                if not _matches_position(m, slot_position):
                    continue
                if not _matches_era(m, gate.era_min, gate.era_max):
                    continue
                if not _matches_stratum(m, gate.stratum):
                    continue
                if exclude_tags and any(t in exclude_tags for t in m.tags):
                    continue
                eligible.append(m)
        if not eligible:
            return []
        # Score: per-meaning canonical composition + cohesion bias
        prior_tags_frozen = frozenset(prior_tags)
        weighted: list[tuple[Meaning, float]] = []
        for m in eligible:
            lemma_ref = _lemma_ref_for(m)
            lemma_tags = frozenset(m.tags)
            lemma_phon = m.phonological_vector or _default_empty_phon_vector()
            base_score = score(
                lemma_ref=lemma_ref,
                lemma_tags=lemma_tags,
                lemma_phon=lemma_phon,
                request=request,
                slot_position=slot_position,
                era_midpoint=era_midpoint,
                priors=priors,
            )
            cohesion_mult = _cohesion_multiplier(
                lemma_tags, prior_tags_frozen, cohesion, tag_cooccurrence
            )
            final_score = base_score * cohesion_mult
            if final_score > 0:
                weighted.append((m, final_score))
        if not weighted:
            return []
        # Sample: weighted choice
        picked_meaning = _weighted_choice(rng, weighted)
        if picked_meaning is None:
            return []
        picked.append(picked_meaning)
        # D17: accumulate picked tags for next slot's cohesion bias
        prior_tags.update(picked_meaning.tags)

    return picked


def _lemma_ref_for(meaning: Meaning) -> str:
    """Resolve the ``lemma_ref`` string the priors-baseline lookup
    keys on. The convention from the lexicon side
    (:mod:`lexicon.empirical_priors`) is ``"<language>:<canonical_form>"``.

    Meaning instances don't carry language directly — the language
    is implicit in the bundle subject's lang_field keys (old_english,
    celtic_mix, etc.). For ecjp.5 v1 we use the meaning's ``usage``
    string as the lemma_ref; this matches the empirical-priors
    extraction's lemma_ref shape on the English slice. Future
    expansion (per-language baselines) would derive a richer
    lemma_ref from the meaning's lang_field mapping; that's
    wyrd-ecjp.8 / Phase 7 territory.

    Returns the meaning's usage stripped of decoration dashes — so
    ``"Place-"`` → ``"Place"`` and ``"-shire"`` → ``"shire"``. The
    bare-usage form is what the priors tables key on.
    """
    return meaning.usage.replace("-", "")


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
