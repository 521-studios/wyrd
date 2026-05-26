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

    Matches ``Meaning._set_location`` exactly so a slot-position
    string returned here can be compared directly to ``meaning.location``
    by ``_matches_position``. The mapping (from
    ``runtime/meaning.py:_set_location``):

    * ``"-inner-"`` (both dashes) → ``"inner"``
    * ``"Place-"`` (trailing dash only) → ``"pre"``
    * ``"-shire"`` (leading dash only) → ``"post"``
    * ``"Bare"`` (no dashes) → ``"post"``

    Round-1 reviewer caught: an earlier draft treated bare elements
    as ``"inner"``, but Meaning maps no-dashes usage to ``post``, so
    bare structural elements would yield an empty eligible pool
    (no meaning matches an ``"inner"`` slot against a bare element).
    Fixed to mirror Meaning's exact convention.
    """
    s = structural_element.strip()
    has_leading = s.startswith("-")
    has_trailing = s.endswith("-")
    if has_leading and has_trailing:
        return "inner"
    if has_trailing:
        return "pre"
    # leading-only OR no-dashes → "post" (matches Meaning._set_location)
    return "post"


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

    Returns:
        list[Meaning] — one Meaning per structural slot, in order.
        Empty list when the gate filters every meaning out for any
        slot (the caller decides whether to retry or fall back).
    """
    picked: list[Meaning] = []
    prior_tags: set[str] = set()
    gate = request.gate

    # Pre-compute the non-position eligibility pool ONCE — era,
    # stratum, and exclude_tags don't depend on the slot. Position is
    # the only per-slot predicate. Round-1 reviewer caught the O(S*N)
    # redundancy and turned it into O(N + S*P) where P is the
    # per-slot position-filtered subset (typically ~N/3 for
    # pre/inner/post-balanced corpora).
    non_position_eligible: list[Meaning] = []
    for meanings_for_usage in meaning_db.values():
        for m in meanings_for_usage:
            if not _matches_era(m, gate.era_min, gate.era_max):
                continue
            if not _matches_stratum(m, gate.stratum):
                continue
            if exclude_tags and any(t in exclude_tags for t in m.tags):
                continue
            non_position_eligible.append(m)

    # wyrd-ecjp.8: admit pack lemmas alongside native lemmas. Each
    # declared pack contributes its meaning_db's lemmas, filtered by
    # the pack's allowed_pack_tags / excluded_pack_tags (gate-level
    # filter from PackOverlay) AND the gate's era / stratum / exclude
    # predicates (shared with native). Pack lemmas score via the
    # baseline_score_pack path inside score(), which uses the pack's
    # template_donor/template_recipient + pack.weight to compose the
    # multi-source baseline contribution per D36.4 Option B.
    if pack_meaning_dbs:
        for pack in request.packs:
            pack_db = pack_meaning_dbs.get(pack.pack_name)
            if pack_db is None:
                continue
            for meanings_for_usage in pack_db.values():
                for m in meanings_for_usage:
                    if not _matches_era(m, gate.era_min, gate.era_max):
                        continue
                    if not _matches_stratum(m, gate.stratum):
                        continue
                    if exclude_tags and any(t in exclude_tags for t in m.tags):
                        continue
                    # Pack-tag filter: PackOverlay.allowed_pack_tags
                    # narrows to a subset (e.g. only the 'war' slice
                    # of the Tatar pack); excluded_pack_tags drops a
                    # subset (e.g. drop the religious slice).
                    if pack.allowed_pack_tags and not any(
                        t in pack.allowed_pack_tags for t in m.tags
                    ):
                        continue
                    if pack.excluded_pack_tags and any(
                        t in pack.excluded_pack_tags for t in m.tags
                    ):
                        continue
                    non_position_eligible.append(m)

    for element in structure:
        # Accept either a structural-element string ("X-"/"-X-"/"-X")
        # the heuristic decodes, or a bare position label ("pre"/
        # "inner"/"post") — caller's choice. Bare labels skip the
        # encode/decode round-trip the legacy struct walker uses to
        # synthesize a string from its key tuples.
        if element in ("pre", "inner", "post"):
            slot_position = element
        else:
            slot_position = _slot_position_label(element)
        # Position-gate the pre-filtered pool
        eligible = [m for m in non_position_eligible if _matches_position(m, slot_position)]
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


_BUNDLE_FIELD_TO_L3_LANG: dict[str, str] = {
    # Reverse of ``lexicon.bundle._emit._LANG_CODE_TO_JSON_FIELD``'s
    # primary L3 → bundle-field mapping. Used by ``_lemma_ref_for`` to
    # synthesize the ``language:canonical_form`` shape the priors tables
    # key on (per ``empirical_priors.extract_priors`` —
    # ``f"{row['language']}:{row['canonical_form']}"``).
    #
    # Lossy because multiple L3 langs collapse into one bundle field
    # (e.g. welsh + irish + old-irish + ... → celtic_mix). We pick the
    # L3 code that the empirical-priors miner actually emits for the
    # majority of cells in that bucket:
    #   * old_english (401 native cells), old_scandinavian (127),
    #     norman-french (8), modern_english (151) — all dominant
    #     in the English-recipient priors slice.
    #   * celtic_mix → "celtic" — matches what the priors miner emits
    #     for celtic-bucket lemmas under the English-priors slice
    #     (cross-cultural priors are future Phase-2 work).
    # Bundle fields whose L3 mapping isn't 1:1-reversible (latin,
    # germanic, greek, biblical) map to the same name on both sides.
    "old_english": "old-english",
    "old_scandinavian": "old-norse",
    "old_french": "norman-french",
    "modern_english": "modern-english",
    "celtic_mix": "celtic",
    "latin": "latin",
    "germanic": "germanic",
    "greek": "greek",
    "biblical": "biblical",
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
