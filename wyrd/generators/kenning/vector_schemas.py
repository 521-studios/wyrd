"""Schemas for the vector-driven generator architecture (D36).

This module defines the typed data structures for the vector-driven
generator that replaces the flat per-culture proportions JSON + scalar
mood/harshness knobs with composable vector scoring across four axes:

  * phonological — feature vector per lemma (cluster density, final
    fortition, vowel-final bias, etc.) calibrated against phonaesthetics
    literature.
  * semantic — tag-weight vector per lemma (death, plant, water, etc.)
    derived from existing meaning-database tags.
  * position — slot-prior vector per lemma (some morphemes prefer
    first-element, some second, some are equally productive).
  * empirical-baseline — frequency prior per (culture, position, tag,
    era) for native culture generation; per (donor, recipient, position,
    tag, era) for pack overlays.

The canonical composition rule is

    score(lemma) = phon_w * phon_score(lemma, request)
                 + sem_w  * sem_score(lemma, request)
                 + pos_w  * pos_score(lemma, request, slot)
                 + base_w * baseline_score(lemma, source, request)

with hard gates (culture / era / stratum / pack-allowlist / pack-tag-
filter) applied as boolean predicates OUTSIDE the vector space — they
shrink the eligible pool before any scoring happens.

These dataclasses are the contract between Phase 2 (priors extraction),
Phase 3 (eligibility-gate runtime), Phase 4 (vector-scoring runtime),
Phase 5 (NameGenerator rewrite), and Phase 6 (drift measurement). All
phases consume these types; none should redefine their own schemas
silently. See DECISIONS.md D36 for the full architectural narrative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---- phonological feature vector (kq7w.1 schema half) -------------------

# Each phonological-feature dimension is a scalar normalized to a stable
# range. Per-feature ranges:
#   * Rate / proportion features ([0, 1]):
#       cluster_density, final_fortition, final_cluster_rate,
#       vowel_final_bias, soft_consonants, polysyllabic_bias,
#       palatalization, sibilance, retroflexion, pharyngeal,
#       aspirated_voiceless
#   * Signed-direction features ([-1, 1] centered at the corpus mean):
#       vowel_height (low-to-high F1 mapped to [-1, +1])
#       vowel_backness (front-to-back F2 mapped to [-1, +1])
#       stop_vs_continuant (more-continuant negative, more-stop positive)
#
# The Phase A enrichment pass (kq7w.1) populates these from each
# etymon's canonical_form + pronunciation_ipa (preferring IPA where
# available) and persists alongside the etymon row. Re-running the
# enrichment is idempotent — same input form yields the same vector.
PhonologicalFeatureName = Literal[
    "cluster_density",
    "final_fortition",
    "final_cluster_rate",
    "vowel_final_bias",
    "soft_consonants",
    "polysyllabic_bias",
    "palatalization",
    "sibilance",
    "retroflexion",
    "pharyngeal",
    "vowel_height",
    "vowel_backness",
    "stop_vs_continuant",
    "aspirated_voiceless",
]


@dataclass(frozen=True)
class PhonologicalVector:
    """Per-lemma phonological feature vector (kq7w.1).

    Stored alongside each etymon (either inline columns or a sibling
    table — schema-side decision deferred to kq7w.1's data-modeling
    phase). Computed once per lemma from its canonical form + IPA,
    re-computable deterministically. The 14 named scalars are the v1
    dimension set; later dimensions append via ``extras`` so adding a
    feature doesn't require a schema migration on every consumer.

    Range conventions:
      * Rate features: [0, 1], where 1.0 means "every position has this".
      * Signed features (vowel_height, vowel_backness,
        stop_vs_continuant): [-1, +1] centered on the corpus mean.

    Composition with a register effect's phonological-weight vector is
    component-wise dot product: see RegisterEffect.compose.
    """

    cluster_density: float = 0.0
    final_fortition: float = 0.0
    final_cluster_rate: float = 0.0
    vowel_final_bias: float = 0.0
    soft_consonants: float = 0.0
    polysyllabic_bias: float = 0.0
    palatalization: float = 0.0
    sibilance: float = 0.0
    retroflexion: float = 0.0
    pharyngeal: float = 0.0
    vowel_height: float = 0.0
    vowel_backness: float = 0.0
    stop_vs_continuant: float = 0.0
    aspirated_voiceless: float = 0.0
    # Forward-compat slot for future-added dimensions. Consumers that
    # don't know a name in extras must ignore it (additive only).
    extras: dict[str, float] = field(default_factory=dict)

    def dot(self, weights: dict[str, float]) -> float:
        """Component-wise weighted sum against a feature-weight dict.

        Unknown weight keys (not in the named fields and not in extras)
        are ignored — additive evolution doesn't break old vectors.
        Missing weights default to 0 (no contribution). The phonological
        sub-score of the canonical composition rule is this dot product.
        """
        s = 0.0
        for name, w in weights.items():
            if w == 0:
                continue
            if name in self.__dataclass_fields__ and name != "extras":
                s += w * getattr(self, name)
            elif name in self.extras:
                s += w * self.extras[name]
        return s


# ---- register-effect catalog (kq7w.2 catalog format) --------------------


@dataclass(frozen=True)
class RegisterEffect:
    """A named register effect (mood/aesthetic primitive) — kq7w.2.

    A register effect is a triple of (phonological-feature weights,
    semantic-tag weights, position bias). Multiple effects compose
    component-wise: phonological vectors add, semantic-tag dicts add per
    tag, position biases add per slot. The composed result is clamped
    to [-1, +1] per dimension after summation.

    Per-effect graduation: callers pass an optional weight multiplier
    (the colon-suffix syntax: ``harsh:0.5`` means apply harsh at 50%%
    strength). The multiplier scales every component of the effect
    uniformly before composition with other effects.

    Catalog migration (from MOODS in kenning/__init__.py):
      * grim → semantic-tag weights only ({death, monster, undead, magic, military})
      * harsh → phonological-feature weights only ({cluster_density: +0.6,
        final_fortition: +0.5, vowel_final_bias: -0.4})
      * pastoral → mixed ({plant, water, agriculture, tree, bird} +
        soft phonology)
      * devotional → semantic-tag weights ({saint, religious})
      * mortuary → semantic-tag weights ({death, undead})
    Plus the new entries the 2026-05-16 design discussion surfaced:
    noble, mystical, melodic, sinister, ancient, exotic, martial.

    The full catalog lives in
    ``wyrd/generators/kenning/data/register_effects.yaml`` as the
    authoring source; this dataclass is the in-memory representation
    after YAML load.
    """

    name: str
    phonological: dict[str, float] = field(default_factory=dict)
    semantic_tags: dict[str, float] = field(default_factory=dict)
    position_bias: dict[str, float] = field(default_factory=dict)

    def scaled(self, weight: float) -> RegisterEffect:
        """Return a copy of this effect with all component weights
        scaled by ``weight``. Used for graduation (``harsh:0.5``).
        """
        return RegisterEffect(
            name=self.name,
            phonological={k: v * weight for k, v in self.phonological.items()},
            semantic_tags={k: v * weight for k, v in self.semantic_tags.items()},
            position_bias={k: v * weight for k, v in self.position_bias.items()},
        )


def compose_register_effects(effects: list[RegisterEffect]) -> RegisterEffect:
    """Sum a list of register effects component-wise, clamping the
    result to [-1, +1] per dimension. Returns a synthetic 'composed'
    effect with name reflecting the inputs.

    Component-wise sum semantics:
      * phonological weights: sum same-named keys, leave others alone
      * semantic_tags: sum same-named keys
      * position_bias: sum same-named keys

    Final clamp to [-1, +1]: a caller that requests
    ``--register harsh:1.0,grim:1.0`` shouldn't be able to compose
    weights past the bounds the catalog uses for individual effects.
    Clamping preserves the "each effect is one unit of pull on each
    axis" semantics.
    """
    phon: dict[str, float] = {}
    sem: dict[str, float] = {}
    pos: dict[str, float] = {}
    for e in effects:
        for k, v in e.phonological.items():
            phon[k] = phon.get(k, 0.0) + v
        for k, v in e.semantic_tags.items():
            sem[k] = sem.get(k, 0.0) + v
        for k, v in e.position_bias.items():
            pos[k] = pos.get(k, 0.0) + v
    _clamp_in_place(phon)
    _clamp_in_place(sem)
    _clamp_in_place(pos)
    return RegisterEffect(
        name="+".join(e.name for e in effects) or "<empty>",
        phonological=phon,
        semantic_tags=sem,
        position_bias=pos,
    )


def _clamp_in_place(d: dict[str, float]) -> None:
    for k, v in d.items():
        if v > 1.0:
            d[k] = 1.0
        elif v < -1.0:
            d[k] = -1.0


# ---- eligibility gates (hard filters) -----------------------------------


@dataclass(frozen=True)
class EligibilityGate:
    """Boolean predicates applied BEFORE vector scoring (D1, D5).

    Hard gates shrink the eligible-lemma pool to those that pass every
    predicate. They are NOT continuous — a lemma either matches the
    requested culture or it doesn't. Soft preferences (a lemma that
    LEANS toward a register) belong in the vector axes, not in the
    gate.

    ``culture`` is the hard-gate axis from D5. A request for ``--culture
    english`` excludes everything not in the english stratum, including
    pack lemmas (the pack overlay re-introduces them as a separate
    eligibility set, see PackOverlay below).

    ``era`` filters to lemmas attested in the requested era (D33's
    era-reflex infrastructure, D5-2 era filter wyrd-lyp). Open-ended
    bounds (``None``) mean no era constraint on that side.

    ``stratum`` (D32, wyrd-lr4) filters within-language strata
    (Brittonic in Welsh, Norse-substrate in Northern English, etc.).
    None means no stratum constraint.

    ``allowed_pack_tags`` and ``excluded_pack_tags`` further refine
    pack-overlay eligibility — narrative input that wants only the
    ``war`` slice of the Tatar pack passes ``allowed_pack_tags={'war'}``;
    same shape for excluding.
    """

    culture: str
    era_min: int | None = None
    era_max: int | None = None
    stratum: str | None = None
    allowed_pack_tags: frozenset[str] = frozenset()
    excluded_pack_tags: frozenset[str] = frozenset()


# ---- pack overlay (scenario-pack composition) ---------------------------


@dataclass(frozen=True)
class PackOverlay:
    """A scenario-pack overlay on top of a native culture (v2gm-foundation).

    A pack overlay introduces pack-specific lemmas into the eligible
    pool alongside the native culture's lemmas. The pack's lemmas
    inherit their template's empirical-baseline (D3 Option B): a
    Khuzdul pack templated on ON→OE uses the ON→OE empirical baseline
    mining as its baseline-axis reference. Pack-weight is a multiplier
    on the pack's baseline contribution to the canonical composition
    rule.

    ``pack_name`` identifies the pack ('neo-khuzdul', 'tatar-9c', etc.)
    and indexes into the bundled pack-catalog where the pack's lemma
    set + template descriptor live.

    ``template_donor`` and ``template_recipient`` name the empirical
    loan-relationship the pack is templated against (e.g. donor='old-
    norse', recipient='old-english' for the canonical Norse-loans-into-
    English baseline). Phase 2's priors extractor reads
    ``loan_relationship_priors[(donor, recipient, position, tag, era)]``
    to get the baseline-score contribution.

    ``weight`` (default 1.0) is the multiplier on the pack's baseline
    contribution. The 0.0 weight produces "native only" generation
    even when the pack is declared; 1.0 produces pack lemmas at the
    empirical loan rate; >1.0 over-weights the pack beyond historical
    realism. Pack-weight is INDEPENDENT from baseline-axis weight
    (D2): the operator can ask for "high realism + low pack" or
    "low realism + heavy pack" as orthogonal knobs.
    """

    pack_name: str
    template_donor: str
    template_recipient: str
    weight: float = 1.0


# ---- request vector (ecjp.1 four-axis composed request) -----------------


@dataclass(frozen=True)
class ScoringWeights:
    """Per-axis weight scalars in the canonical composition rule (D1).

    The canonical formula is

        score = phon_w * phon_score
              + sem_w  * sem_score
              + pos_w  * pos_score
              + base_w * baseline_score

    Each weight scales the contribution of one axis. Defaults
    (1.0/1.0/1.0/1.0) treat all axes equally. The user-facing
    CLI maps the named knobs onto these:

      * ``--register harsh`` adds the harsh effect's phonological
        contribution to the request, but doesn't move ``phon_w`` —
        the AMOUNT of weight on the phonological axis stays at 1.0;
        the DIRECTION (what kind of phonology is preferred) shifts
        toward harsh.
      * ``--realism 0.5`` halves ``base_w`` (the empirical-baseline
        axis weight), letting non-baseline-favoured lemmas score
        higher relative to the baseline.

    Per D2: empirical baseline is just another axis with continuous
    weight. There's no separate 'realism' concept — it's literally
    just ``base_w``.
    """

    phon_w: float = 1.0
    sem_w: float = 1.0
    pos_w: float = 1.0
    base_w: float = 1.0


@dataclass(frozen=True)
class RequestVector:
    """The composed scoring input for a single generator invocation
    (ecjp.1 deliverable).

    A request vector bundles every piece of input that the per-lemma
    scoring engine needs to score a candidate lemma in a slot:

      1. ``gate`` — the hard-filter predicates that shrink the eligible
         pool (D1, D5). Lemmas that fail the gate are removed BEFORE
         scoring; they never appear in the score-then-sample pipeline.
      2. ``register`` — the composed register-effect (sum of caller-
         requested effects, scaled by graduation, clamped). Drives the
         direction of the phonological + semantic + position axes.
      3. ``weights`` — per-axis multipliers on the canonical formula.
         User knobs like ``--realism`` map here.
      4. ``packs`` — list of pack overlays in play. Empty list means
         native-only generation. Multiple packs compose additively on
         the baseline axis (each pack's baseline contribution sums
         into a multi-source baseline; see Phase 7 for the
         multi-pack-composition semantics).

    Building a RequestVector is the responsibility of the CLI / API
    front-end layer (Phase 8): it parses ``--culture english --era
    1066-1300 --register harsh:0.8,grim:0.5 --pack neo-khuzdul:0.6``
    and emits a RequestVector that the scoring runtime (Phase 4)
    consumes.

    Per D8, tolerance bands on the composed register are deferred to
    Phase 6a — the request vector itself doesn't carry them; the
    drift-measurement layer compares scoring outputs across priors
    versions independently of the request.
    """

    gate: EligibilityGate
    register: RegisterEffect
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    packs: tuple[PackOverlay, ...] = ()


# ---- empirical-baseline priors (ecjp.1 priors schema) -------------------

# Priors-key tuples are the lookup key for the empirical-baseline axis.
# A native lookup is keyed by (culture, position, tag, era); a loan-
# relationship lookup is keyed by (donor, recipient, position, tag, era).
# Both are normalized to lowercase strings (position) and integer-year
# era buckets. ``tag`` is a single semantic-tag string; lemmas that
# carry multiple tags appear under each one independently — the
# baseline_score combinator decides how to aggregate across tags
# (Phase 2 design decision).
NativePriorsKey = tuple[str, str, str, int]
LoanPriorsKey = tuple[str, str, str, str, int]


@dataclass(frozen=True)
class EmpiricalPriors:
    """Per-(culture × position × tag × era) and per-(donor × recipient
    × position × tag × era) frequency priors for the empirical-baseline
    axis (D3, D7).

    The priors are a derived artifact — Phase 2 reads the lexicon DB
    and materializes this structure as a versioned file. Re-running
    Phase 2 on the same lexicon DB state produces a byte-identical
    artifact; re-running after new mining lands produces a new version,
    which Phase 6a compares against the previous version for drift
    measurement.

    ``native[(culture, position, tag, era)]`` returns a dict mapping
    each eligible lemma form to its empirical-frequency weight in that
    cell. The weights are NOT normalized across cells — they're raw
    counts (or smoothed counts; smoothing-spec lives in Phase 2).
    Phase 4's scoring runtime normalizes-per-slot at scoring time.

    ``loan_relationship[(donor, recipient, position, tag, era)]``
    returns the analogous mapping for pack-template baselines. Pack
    overlays look up here, weighted by ``pack.weight``.

    Versioning: the ``version`` field (a content-hash or sequential
    integer) identifies the priors snapshot. Downstream caches and
    bundle builds key on this so a regenerated priors artifact
    invalidates the right downstream artifacts (D8 Phase 6 drift
    plumbing).
    """

    native: dict[NativePriorsKey, dict[str, float]] = field(default_factory=dict)
    loan_relationship: dict[LoanPriorsKey, dict[str, float]] = field(default_factory=dict)
    version: str = "unversioned"


# ---- cohesion adapter (D17 integration) ---------------------------------


@dataclass(frozen=True)
class CohesionContext:
    """Slot-by-slot D17 cohesion context, threaded through generation
    (ecjp.1 adapter design).

    The vector-driven generator preserves D17's tag-class-prior
    mechanism via a wrapper around per-slot scoring. For each slot
    after the first, the cohesion adapter computes a per-lemma
    multiplier (the ``key_boost`` of the existing Generator.select API)
    from the tag co-occurrence model conditioned on previously-picked
    lemmas, then applies it to the per-lemma vector score BEFORE the
    novelty blend.

    ``picked_tags`` is the accumulated tag set from earlier slots in
    the current name being assembled. ``novelty`` is the D17 novelty
    knob — at 0.0 the result is pure cohesion-boosted empirical; at 1.0
    the result is uniform-over-eligible (full marginal). The blend
    semantics match the existing Generator.select implementation; see
    DECISIONS.md D17 for the realized-vs-textbook math.

    The cohesion adapter sits between Phase 4 (vector scoring) and
    Phase 5 (NameGenerator rewrite): Phase 4 produces per-lemma vector
    scores, the cohesion adapter wraps them with the context-conditional
    boost, Phase 5 calls the wrapped scorer in its slot walk.

    No new D-entry for cohesion — D17 stays the canonical decision;
    this struct is the API hook that lets the vector-driven generator
    plug into it without re-implementing the mixture math.
    """

    picked_tags: frozenset[str] = frozenset()
    novelty: float = 0.0
