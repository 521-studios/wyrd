"""Vector-scoring runtime (wyrd-ecjp.4 Phase 4).

Per-lemma scoring under the D36.2 canonical composition rule:

    score = phon_w * phon_score
          + sem_w  * sem_score
          + pos_w  * pos_score
          + base_w * baseline_score

Each per-axis sub-scorer reads its own piece of the input vectors and
returns a real number; ``aggregate_score`` sums them per ScoringWeights
(D36.3 collapses 'realism' into ``base_w`` so the operator has a
single axis-weight knob per axis, not a separate realism dial).

The four sub-scorers run downstream of the eligibility gate (Phase 3,
``eligibility.py``) — by the time scoring sees a Meaning it has
already passed the culture / era / stratum / tag-required / tag-
excluded / pack hard gates. Scoring's job is the SOFT ranking that
turns the eligible pool into a weighted distribution Phase 5
(NameGenerator rewrite) samples from.

This module implements the v1 reference shape:

* ``phon_score`` uses the existing ``PhonologicalVector.dot`` from
  vector_schemas.py — component-wise weighted sum against the
  composed request's phonological-feature weights.
* ``sem_score`` sums the request's per-tag weights over the tags
  the lemma carries (lemma tag dot composed register tag-weight
  vector, no normalization).
* ``pos_score`` reads the composed register's ``position_bias`` dict
  for the slot's position label. v1 doesn't fold in the lemma's
  natural ``position_pref`` (etymon.position_pref) — that signal is
  carried by the eligibility-gate / structure-picker layer rather
  than the score; folding it in here would double-count.
* ``baseline_score_native`` aggregates the empirical-priors lookup
  over the lemma's tags, weighted by the request's per-tag interest
  (the "weighted-by-request-tag-weights" rule chosen for the
  cross-tag aggregator — D36.7 lists this as one of three candidates
  alongside sum / max and defers the call to the Phase 4 / Phase 2
  implementation). Hierarchical fallback per D36.7 walks the cells
  in most-specific-first order:
  ``(c, p, t, e) → (c, p, t, *) → (c, p, *, *) → (c, *, *, *) → 0``.

  IMPLICATION OF THE COUPLING: ``baseline_score`` reads the request's
  ``register.semantic_tags`` as the per-tag-weight vector for
  aggregation. A request with a phonology-only register (e.g.
  ``--register harsh`` alone, no ``--tag`` flags, no register-effect
  that brings semantic_tags) supplies an empty tag-weight vector,
  which collapses baseline to 0 even when ``base_w=1.0``. This is
  consistent with the chosen aggregation rule (no per-tag interest
  ⇒ no per-tag contribution to sum). Operators who want empirical
  baseline driving generation must also pass a register effect or
  --tag flag that supplies semantic_tags. Surfacing this trade-off
  in the CLI help is wyrd-ecjp.9's job.
* ``baseline_score_pack`` does the same against the
  ``loan_relationship`` priors keyed on the pack's
  (template_donor, template_recipient) — D36.4 Option B.

What's NOT here:

* Pre-indexed priors for fast cell-based lookups. The reference
  implementation iterates ``priors.native.items()`` per lookup; that's
  O(N_cells) per (lemma × slot × tag) triple and won't meet the
  Phase 4 acceptance criterion's 2x perf bound on a corpus-scale
  request. A flat ``dict[lemma_ref, dict[cell_key, weight]]``
  reverse index gives O(1) for the exact-match path and a small
  bounded walk for fallback levels; deferred to a follow-up perf
  ticket. The reference shape here is correctness-first.
* The D17 cohesion adapter (CohesionContext from vector_schemas)
  isn't called here — that's the wrapper layer between Phase 4 and
  Phase 5 (NameGenerator slot-walk), per D36.5. The cohesion
  multiplier applies to the composed score returned by this
  module, not inside it.
* Slot-walk integration — Phase 5 (wyrd-ecjp.5) consumes this
  module's ``score`` function in its per-slot loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from wyrd.generators.kenning.vector_schemas import (
    EmpiricalPriors,
    PackOverlay,
    PhonologicalVector,
    RegisterEffect,
    RequestVector,
    ScoringWeights,
)


def phon_score(
    lemma_vec: PhonologicalVector,
    request_register: RegisterEffect,
) -> float:
    """Phonological sub-score: lemma vector dotted with composed
    register's phonological-feature weights.

    Delegates to ``PhonologicalVector.dot`` which whitelists weights
    against the known dimension set (unknown-feature weights are
    ignored — additive evolution of the dimension set doesn't break
    old vectors).

    Positive score: lemma's phonological color is aligned with the
    register's preference (e.g. cluster-heavy lemma + harsh register).
    Negative: lemma is OPPOSED (e.g. vowel-final lemma + harsh
    register where harsh.phonological has ``vowel_final_bias: -0.4``).
    Zero: no overlap between lemma's non-default fields and register's
    requested features.
    """
    return lemma_vec.dot(request_register.phonological)


def sem_score(
    lemma_tags: Iterable[str],
    request_register: RegisterEffect,
) -> float:
    """Semantic sub-score: sum of request semantic-tag weights for
    tags the lemma carries.

    Equivalent to the dot product of (lemma_has_tag: 1/0 per tag) and
    (request.register.semantic_tags). A lemma carrying multiple
    relevant tags accumulates; a lemma with no relevant tags scores
    0.

    Tags the lemma carries but the request doesn't weight contribute
    0; tags the request weights but the lemma doesn't carry
    contribute 0. Only the intersection contributes.

    Iteration uses ``sorted(set(lemma_tags))`` so floating-point sum
    order is deterministic across processes — Python's set iteration
    order over string keys depends on per-process hash
    randomization, and FP addition is non-associative, so an
    unsorted walk over a set of strings would produce bit-level
    output differences for the same inputs. The seed-stable contract
    requires order-pinning.
    """
    tag_weights = request_register.semantic_tags
    if not tag_weights:
        return 0.0
    return sum(tag_weights.get(t, 0.0) for t in sorted(set(lemma_tags)))


def pos_score(
    slot_position: str,
    request_register: RegisterEffect,
) -> float:
    """Position sub-score: the request register's bias for the slot's
    position label.

    A register that pushes ``position_bias={'pre': 0.5, 'post': -0.3}``
    rewards lemmas in pre slots and penalizes lemmas in post slots,
    uniformly across all candidates for those slots. The lemma's own
    natural position_pref (etymon.position_pref) is NOT folded in
    here — that's an eligibility / structure concern (a 'post'-only
    morpheme would already have been filtered out of pre-slot
    candidates by the structure picker), and double-counting it here
    would compound with the gate's filtering.
    """
    return request_register.position_bias.get(slot_position, 0.0)


# ---- empirical-baseline lookup with hierarchical fallback (D36.7) ------


def _native_lookup_exact(
    priors: EmpiricalPriors,
    lemma_ref: str,
    culture: str,
    position: str,
    tag: str,
    era_midpoint: int,
) -> float | None:
    """Direct lookup at the most-specific (culture, position, tag, era)
    cell. Returns None when the cell doesn't exist OR exists but
    doesn't carry the lemma — both cases mean 'no exact-match data;
    caller should try the fallback'.

    Uses explicit ``in``-check rather than ``cell.get(lemma_ref)`` so a
    lemma stored at weight 0.0 (defensive: the L3-table schema CHECK
    excludes 0, but the in-memory EmpiricalPriors dataclass doesn't
    enforce that) is correctly reported as PRESENT (returning 0.0)
    rather than ABSENT (returning None and triggering fallback to a
    coarser cell). Matches the convention used in
    ``_loan_lookup_with_fallback``.
    """
    cell = priors.native.get((culture, position, tag, era_midpoint))
    if cell is None or lemma_ref not in cell:
        return None
    return cell[lemma_ref]


def _native_lookup_with_fallback(
    priors: EmpiricalPriors,
    lemma_ref: str,
    culture: str,
    position: str,
    tag: str,
    era_midpoint: int,
) -> float:
    """Hierarchical fallback for native priors per D36.7:

      1. exact          (culture, position, tag, era)
      2. era-wild       (culture, position, tag, *)
      3. tag+era-wild   (culture, position, *,   *)
      4. all-wild       (culture, *,        *,   *)
      5. 0.0            no cell at any specificity has this lemma

    At each level we take the MAX weight across matching cells — a
    lemma that's frequent in ONE era but rare in another shouldn't
    look rarer than its single best cell when we generalize across
    eras. (D36.7 leaves the aggregation rule for the fallback to
    Phase 4's call; max preserves the 'most generous evidence' read
    of the data which lines up with the empirical-baseline axis being
    a soft preference, not a hard filter.)

    Reference implementation: O(N_cells) per lookup. A flat
    lemma → cells reverse index is the planned follow-up perf
    optimization (see module docstring).
    """
    exact = _native_lookup_exact(priors, lemma_ref, culture, position, tag, era_midpoint)
    if exact is not None:
        return exact

    # Level 2: era wildcard. (c, p, t, *)
    best: float | None = None
    for (c, p, t, _e), cell in priors.native.items():
        if c == culture and p == position and t == tag and lemma_ref in cell:
            v = cell[lemma_ref]
            if best is None or v > best:
                best = v
    if best is not None:
        return best

    # Level 3: tag + era wildcard. (c, p, *, *)
    for (c, p, _t, _e), cell in priors.native.items():
        if c == culture and p == position and lemma_ref in cell:
            v = cell[lemma_ref]
            if best is None or v > best:
                best = v
    if best is not None:
        return best

    # Level 4: position + tag + era wildcard. (c, *, *, *)
    for (c, _p, _t, _e), cell in priors.native.items():
        if c == culture and lemma_ref in cell:
            v = cell[lemma_ref]
            if best is None or v > best:
                best = v
    return best if best is not None else 0.0


def _loan_lookup_with_fallback(
    priors: EmpiricalPriors,
    lemma_ref: str,
    donor: str,
    recipient: str,
    position: str,
    tag: str,
    era_midpoint: int,
) -> float:
    """Hierarchical fallback for loan-relationship priors per D36.4 +
    D36.7. Same shape as native but keyed on (donor, recipient, ...):

      1. exact          (donor, recipient, position, tag, era)
      2. era-wild       (donor, recipient, position, tag, *)
      3. tag+era-wild   (donor, recipient, position, *,   *)
      4. all-wild       (donor, recipient, *,        *,   *)
      5. 0.0
    """
    exact_cell = priors.loan_relationship.get((donor, recipient, position, tag, era_midpoint))
    if exact_cell is not None and lemma_ref in exact_cell:
        return exact_cell[lemma_ref]

    best: float | None = None
    # Level 2: era wildcard.
    for (d, r, p, t, _e), cell in priors.loan_relationship.items():
        if d == donor and r == recipient and p == position and t == tag and lemma_ref in cell:
            v = cell[lemma_ref]
            if best is None or v > best:
                best = v
    if best is not None:
        return best

    # Level 3: tag + era wildcard.
    for (d, r, p, _t, _e), cell in priors.loan_relationship.items():
        if d == donor and r == recipient and p == position and lemma_ref in cell:
            v = cell[lemma_ref]
            if best is None or v > best:
                best = v
    if best is not None:
        return best

    # Level 4: position + tag + era wildcard.
    for (d, r, _p, _t, _e), cell in priors.loan_relationship.items():
        if d == donor and r == recipient and lemma_ref in cell:
            v = cell[lemma_ref]
            if best is None or v > best:
                best = v
    return best if best is not None else 0.0


def baseline_score_native(
    lemma_ref: str,
    lemma_tags: Iterable[str],
    *,
    culture: str,
    slot_position: str,
    era_midpoint: int,
    priors: EmpiricalPriors,
    request_tag_weights: dict[str, float],
) -> float:
    """Native (non-pack) empirical-baseline aggregate for one lemma.

    The aggregation across the lemma's tags is weighted by the
    request's per-tag interest (D36.7 'weighted-by-request-tag-
    weights' rule):

        baseline = sum_over(tag in lemma.tags) of
                   request_tag_weights[tag] * lookup(lemma, culture,
                                                    slot_position, tag,
                                                    era)

    This couples the baseline axis to the request's semantic
    direction: a lemma that's empirically common in cells the
    request doesn't care about contributes 0 to baseline; a lemma
    that's empirically common in cells the request DOES care about
    contributes its full per-tag weight, summed across its tags.
    Tags the lemma carries that have weight 0 in the request short-
    circuit out of the per-tag lookup.

    Empty lemma_tags → 0.0 (lemma has no tags to look up against).
    Empty request_tag_weights → 0.0 (request expresses no semantic
    interest; no per-tag aggregation possible).
    """
    if not request_tag_weights:
        return 0.0
    total = 0.0
    # sorted(set(...)) pins the FP-sum iteration order against hash
    # randomization (see sem_score for the same rationale).
    for tag in sorted(set(lemma_tags)):
        rw = request_tag_weights.get(tag, 0.0)
        if rw == 0.0:
            continue
        per_tag = _native_lookup_with_fallback(
            priors, lemma_ref, culture, slot_position, tag, era_midpoint
        )
        total += rw * per_tag
    return total


def baseline_score_pack(
    lemma_ref: str,
    lemma_tags: Iterable[str],
    *,
    pack: PackOverlay,
    slot_position: str,
    era_midpoint: int,
    priors: EmpiricalPriors,
    request_tag_weights: dict[str, float],
) -> float:
    """Loan-relationship empirical-baseline aggregate for one lemma
    against ONE pack. D36.4 Option B: pack lemmas inherit their
    template's loan-relationship baseline, keyed on
    (pack.template_donor, pack.template_recipient).

    Returns the per-tag aggregate WITHOUT applying pack.weight — the
    caller composes via ``pack.weight * baseline_score_pack(...)`` so
    the multiplier can be observed / unit-tested independently.
    """
    if not request_tag_weights:
        return 0.0
    total = 0.0
    # sorted(set(...)) pins the FP-sum iteration order against hash
    # randomization (see sem_score for the same rationale).
    for tag in sorted(set(lemma_tags)):
        rw = request_tag_weights.get(tag, 0.0)
        if rw == 0.0:
            continue
        per_tag = _loan_lookup_with_fallback(
            priors,
            lemma_ref,
            pack.template_donor,
            pack.template_recipient,
            slot_position,
            tag,
            era_midpoint,
        )
        total += rw * per_tag
    return total


def baseline_score(
    lemma_ref: str,
    lemma_tags: Iterable[str],
    *,
    culture: str,
    slot_position: str,
    era_midpoint: int,
    priors: EmpiricalPriors,
    request_tag_weights: dict[str, float],
    packs: Sequence[PackOverlay] = (),
) -> float:
    """Empirical-baseline sub-score: native aggregate + sum of
    pack-weight-scaled pack aggregates (D36.2 canonical formula).

    The pack contributions are scaled by ``pack.weight`` here so the
    aggregator just dots ``base_w * baseline``. Pack-weight is
    INDEPENDENT from ``base_w`` per D36.3: the operator can request
    "high realism + low pack" (base_w=1.0, pack.weight=0.2) or "low
    realism + heavy pack" (base_w=0.2, pack.weight=2.0) as
    orthogonal knobs.

    Returns the composed baseline-axis value for ONE lemma; caller
    applies ``base_w`` from ScoringWeights.
    """
    native = baseline_score_native(
        lemma_ref,
        lemma_tags,
        culture=culture,
        slot_position=slot_position,
        era_midpoint=era_midpoint,
        priors=priors,
        request_tag_weights=request_tag_weights,
    )
    if not packs:
        return native
    pack_total = 0.0
    for pack in packs:
        per_pack = baseline_score_pack(
            lemma_ref,
            lemma_tags,
            pack=pack,
            slot_position=slot_position,
            era_midpoint=era_midpoint,
            priors=priors,
            request_tag_weights=request_tag_weights,
        )
        pack_total += pack.weight * per_pack
    return native + pack_total


# ---- top-level aggregator (canonical composition rule D36.2) -----------


def aggregate_score(
    *,
    phon: float = 0.0,
    sem: float = 0.0,
    pos: float = 0.0,
    baseline: float = 0.0,
    weights: ScoringWeights,
) -> float:
    """Canonical composition rule (D36.2):

        score = phon_w * phon + sem_w * sem + pos_w * pos + base_w * baseline

    All four axes are independently meaningful and independently
    weighted. Defaults on ScoringWeights are 1.0 per axis (every
    axis contributes equally); the user-facing knobs in the CLI map
    onto these (e.g. ``--realism 0.5`` halves ``base_w``).

    Per D36.3 ``base_w`` is the SOLE realism knob; there's no
    separate 'baseline-retention' or auto-blend on register effects.
    A request with ``base_w=0.0`` runs pure register-driven scoring;
    ``base_w=1.0`` runs the empirically-weighted form.
    """
    return (
        weights.phon_w * phon
        + weights.sem_w * sem
        + weights.pos_w * pos
        + weights.base_w * baseline
    )


def score(
    lemma_ref: str,
    lemma_tags: Iterable[str],
    lemma_phon: PhonologicalVector,
    *,
    request: RequestVector,
    slot_position: str,
    era_midpoint: int,
    priors: EmpiricalPriors,
) -> float:
    """Compose the full per-lemma score for a slot under one request.

    Drives the D36.2 canonical composition: each axis is computed
    independently, then ScoringWeights aggregates them.

    Caller is responsible for:

    * Passing a Meaning that already passed the eligibility gate
      (Phase 3) — this function does NOT re-filter.
    * Resolving the ``slot_position`` label from the slot's index in
      the structure (e.g. 'pre' / 'inner' / 'post' per
      ``Meaning.location``).
    * Resolving the ``era_midpoint`` from the request's era_min /
      era_max (typically the midpoint of the requested range, or 0
      when no era is set — then the baseline lookup hits the
      hierarchical fallback's tag-wildcard / all-wildcard levels).
    * Looking up ``lemma_phon`` from per-etymon phonological vectors
      (kq7w.1 corpus enrichment). For Meanings without phonological
      vectors yet, pass a default-constructed PhonologicalVector
      (all zeros) and ``phon_score`` returns 0.

    Seed-stable contract: same inputs → same float (no randomness,
    no hash-iteration order, no datetime).

    Note on ``lemma_tags``: the parameter is typed ``Iterable[str]``
    for caller convenience but the value is consumed by multiple
    sub-scorers (sem_score, baseline_score, plus one pass per
    admitted pack). To support iterator / generator callers safely,
    this function materializes the tags into a frozenset ONCE up
    front; sub-scorers see a multi-pass collection. A bare iterator
    passed in still works because ``frozenset(iter)`` drains it
    here; subsequent sub-scorers receive the frozen materialized
    view.
    """
    # Materialize once — supports generator callers + downstream
    # multi-pass safe.
    lemma_tag_set: frozenset[str] = frozenset(lemma_tags)

    phon = phon_score(lemma_phon, request.register)
    sem = sem_score(lemma_tag_set, request.register)
    pos = pos_score(slot_position, request.register)
    baseline = baseline_score(
        lemma_ref,
        lemma_tag_set,
        culture=request.gate.culture,
        slot_position=slot_position,
        era_midpoint=era_midpoint,
        priors=priors,
        request_tag_weights=request.register.semantic_tags,
        packs=request.packs,
    )
    return aggregate_score(phon=phon, sem=sem, pos=pos, baseline=baseline, weights=request.weights)
