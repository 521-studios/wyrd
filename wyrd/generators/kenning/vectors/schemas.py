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

with hard gates (culture / stratum / pack-allowlist / pack-tag-filter)
applied as boolean predicates OUTSIDE the vector space — they shrink
the eligible pool before any scoring happens. (Era is not a gate —
D44: it selects the rendered reflex, never the eligible morphemes.)

These dataclasses are the contract between Phase 2 (priors extraction),
Phase 3 (eligibility-gate runtime), Phase 4 (vector-scoring runtime),
Phase 5 (NameGenerator rewrite), and Phase 6 (drift measurement). All
phases consume these types; none should redefine their own schemas
silently. See DECISIONS.md D36 for the full architectural narrative.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, get_args

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
#
# ``PhonologicalFeatureName`` is the canonical Literal type of the v1
# dimension names — published for downstream typed-dict consumers (the
# kq7w.2 register-effects YAML loader uses it for catalog validation;
# external callers writing typed weight dicts can annotate against it).
# Its runtime mirror, ``_DIMENSION_NAMES``, is used by
# :meth:`PhonologicalVector.dot` for the dimension-recognition step —
# it's the single source of truth for "what counts as a named
# dimension" and is automatically derived from the Literal via
# ``get_args`` so the Literal and the frozenset can't drift.
PhonologicalFeatureName = Literal[
    "cluster_density",
    "final_fortition",
    "final_cluster_rate",
    "vowel_final_bias",
    # ``soft_consonants`` lumps /l m n/ (Gentle per Whissell) with /r/
    # (Harsh per Whissell) — kept for back-compat with already-stored
    # vectors but new catalog entries should prefer ``liquid_l_m_n`` /
    # ``rhotic_r`` instead. wyrd-119p.
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
    # wyrd-119p: Whissell 2000 splits the sonorant-consonant cluster.
    # Lumping them under soft_consonants made /r/-heavy lemmas score
    # high on softness when they should read the opposite. These two
    # dimensions disjointly cover what soft_consonants used to roll up.
    "liquid_l_m_n",
    "rhotic_r",
    # wyrd-mkry: Whissell 2000's vowel ratings are non-monotonic in
    # height (/iː/ Gentle, /ɪ/ Harsh; /ɔ/ Gentle, /uː/ Harsh) — tense
    # / lax matters more than front / back, but the mapping does NOT
    # cleanly partition into Gentle vs Harsh on its own (/uː/ is
    # tense AND Harsh; /ɔ/ is lax AND Gentle). Signed dimension in
    # [-1, +1] like vowel_height / vowel_backness: +1 = every
    # monophthong is in _TENSE_VOWELS (13-glyph set including all
    # close + close-mid + the open back vowels), -1 = every
    # monophthong is in _LAX_VOWELS (10-glyph set: near-close +
    # open-mid + near-open centralized vowels). Catalog entries
    # decide per-effect direction.
    "vowel_tenseness",
]
_DIMENSION_NAMES: frozenset[str] = frozenset(get_args(PhonologicalFeatureName))


@dataclass(frozen=True)
class PhonologicalVector:
    """Per-lemma phonological feature vector (kq7w.1).

    Stored alongside each etymon (either inline columns or a sibling
    table — schema-side decision deferred to kq7w.1's data-modeling
    phase). Computed once per lemma from its canonical form + IPA,
    re-computable deterministically. The named scalars are the v1
    dimension set; later dimensions append via ``extras`` so adding a
    feature doesn't require a schema migration on every consumer.

    Range conventions:
      * Rate features: [0, 1], where 1.0 means "every position has this".
      * Signed features (vowel_height, vowel_backness,
        stop_vs_continuant): [-1, +1] centered on the corpus mean.

    Composition with a register effect's phonological-weight vector is
    component-wise dot product: see RegisterEffect.compose.

    Note on immutability: ``frozen=True`` blocks field rebinding but
    does NOT make the ``extras`` dict immutable, and does NOT make
    instances hashable (mutable dict fields preclude that regardless
    of ``frozen``). Callers must treat ``extras`` as read-only after
    construction by convention — silent in-place dict mutation would
    invalidate any per-instance derived state (e.g. memoized scores
    keyed on ``id(vector)``) downstream. The type-system guarantee
    is "no field reassignment"; the conventional guarantee is "treat
    dicts as read-only too".
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
    # wyrd-119p: laterals + nasals (the Gentle half of the historical
    # soft_consonants compound). Rate feature in [0, 1].
    liquid_l_m_n: float = 0.0
    # wyrd-119p: rhotics (the Harsh half). Rate feature in [0, 1].
    rhotic_r: float = 0.0
    # wyrd-mkry: signed +1 (all-tense-monophthong) to -1 (all-lax)
    # vowel-tenseness score, corpus-centered at 0. The mapping to
    # Whissell Gentle / Harsh isn't monotonic — see the Literal
    # comment above + _TENSE_VOWELS / _LAX_VOWELS in compute.
    # Mirrors the shape of vowel_height / vowel_backness.
    vowel_tenseness: float = 0.0
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
            # Explicit dimension whitelist via ``_DIMENSION_NAMES``
            # rather than ``self.__dataclass_fields__``: future
            # non-dimension metadata fields added to this dataclass
            # (e.g. a version stamp or provenance field) won't be
            # mistakenly treated as scoring dimensions if their
            # names happen to collide with a weight key.
            if name in _DIMENSION_NAMES:
                s += w * getattr(self, name)
            elif name in self.extras:
                s += w * self.extras[name]
        return s


# ---- register-effect catalog (kq7w.2 catalog format) --------------------


# wyrd-we1u: tier vocabulary for per-weight scholarly-grounding
# metadata. Three rated categories + an "untagged" default so the
# parser can accept old-shape bare-float weights without forcing
# the migration to be atomic. Definitions:
#   * universal — cross-linguistic sound-symbolic primitive with
#     experimental + cross-cultural replication (Blasi 2016,
#     Ćwiek 2022, Fort 2015, Whissell 1999, Ohala 1994). Applies
#     across speech communities.
#   * ie-conventional — perceptual association documented for
#     English-speaking operators (e.g. cluster_density,
#     polysyllabic_bias, final_fortition as "harsh / grand" per
#     Mooshammer 2024) but without cross-cultural support.
#   * identity-marking — features that mark a specific phonological
#     color without carrying crossmodal-perception payload (the
#     exotic register's role; see REGISTERS.md §5 ('exotic')
#     + §1c for the 'identity-marking' framing).
#   * untagged — no scholarly grounding assigned yet. Default for
#     weights written in the bare-float (pre-wyrd-we1u) shape.
UNIVERSAL_TIER = "universal"
IE_CONVENTIONAL_TIER = "ie-conventional"
IDENTITY_MARKING_TIER = "identity-marking"
UNTAGGED_TIER = "untagged"
WEIGHT_TIERS: frozenset[str] = frozenset(
    {UNIVERSAL_TIER, IE_CONVENTIONAL_TIER, IDENTITY_MARKING_TIER, UNTAGGED_TIER}
)


@dataclass(frozen=True)
class RegisterEffect:
    """A named register effect (mood/aesthetic primitive) — kq7w.2.

    A register effect is a triple of (phonological-feature weights,
    semantic-tag weights, position bias). Multiple effects compose
    component-wise: phonological vectors add, semantic-tag dicts add per
    tag, position biases add per slot. The composed result is clamped
    to [-1, +1] per dimension after summation.

    Note on immutability: ``frozen=True`` blocks field rebinding but
    does NOT make the three dict fields immutable, and does NOT make
    instances hashable (mutable dict fields preclude that regardless
    of ``frozen``). Callers must treat a constructed RegisterEffect's
    dicts as read-only by convention — silent in-place dict mutation
    would invalidate any per-instance derived state (e.g. memoized
    composed-effect scores keyed on ``id(effect)``) downstream. The
    type-system guarantee is "no field reassignment"; the conventional
    guarantee on the dicts is "do not mutate after construction".

    Per-effect graduation: callers pass an optional weight multiplier
    (the colon-suffix syntax: ``harsh:0.5`` means apply harsh at 50%
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
    # wyrd-we1u: per-weight tier metadata. Each tier dict mirrors
    # its sibling weight dict (same keys). Values are one of
    # WEIGHT_TIERS. Keys absent from the tier dict are conventionally
    # 'untagged' — consumers that filter by tier treat untagged as
    # "no scholarly grounding yet" rather than as one of the three
    # rated categories. Parallel-dict shape (rather than per-weight
    # wrapping) keeps the existing scoring callers — dot(), composed
    # weights, the legacy mood-translation helper — unchanged; tier
    # is a sidecar that filters / audits consult but the scoring
    # math itself ignores. See D37.4 + REGISTERS.md for the scholarly
    # citations behind each tier assignment.
    phonological_tiers: dict[str, str] = field(default_factory=dict)
    semantic_tag_tiers: dict[str, str] = field(default_factory=dict)
    position_bias_tiers: dict[str, str] = field(default_factory=dict)

    def scaled(self, weight: float) -> RegisterEffect:
        """Return a copy of this effect with all component weights
        scaled by ``weight``. Used for graduation (``harsh:0.5``).
        Tier metadata is preserved unchanged — scaling a weight
        doesn't change the scholarly grounding it carries.
        """
        return RegisterEffect(
            name=self.name,
            phonological={k: v * weight for k, v in self.phonological.items()},
            semantic_tags={k: v * weight for k, v in self.semantic_tags.items()},
            position_bias={k: v * weight for k, v in self.position_bias.items()},
            phonological_tiers=dict(self.phonological_tiers),
            semantic_tag_tiers=dict(self.semantic_tag_tiers),
            position_bias_tiers=dict(self.position_bias_tiers),
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
    phon: defaultdict[str, float] = defaultdict(float)
    sem: defaultdict[str, float] = defaultdict(float)
    pos: defaultdict[str, float] = defaultdict(float)
    for e in effects:
        for k, v in e.phonological.items():
            phon[k] += v
        for k, v in e.semantic_tags.items():
            sem[k] += v
        for k, v in e.position_bias.items():
            pos[k] += v
    # Convert defaultdicts back to plain dicts at the return boundary so
    # downstream callers can't accidentally create zero entries via
    # missing-key access on the composed effect. _clamp_in_place operates
    # on plain dicts (it iterates keys, doesn't touch missing-key
    # semantics) so it's safe to call before or after the conversion;
    # called after to match the dict-typed RegisterEffect field types.
    out_phon, out_sem, out_pos = dict(phon), dict(sem), dict(pos)
    _clamp_in_place(out_phon)
    _clamp_in_place(out_sem)
    _clamp_in_place(out_pos)
    # wyrd-we1u: composed effects intentionally drop per-weight tier
    # metadata. A dim that accumulated from a UNIVERSAL grim weight
    # plus an IE-CONVENTIONAL harsh weight has no single tier to
    # report; consumers that filter by tier (wyrd-kz2b.33.1) apply
    # the filter on the per-effect inputs BEFORE composition rather
    # than trying to attribute composed values back to source tiers.
    return RegisterEffect(
        name="+".join(e.name for e in effects) or "<empty>",
        phonological=out_phon,
        semantic_tags=out_sem,
        position_bias=out_pos,
    )


def _clamp_in_place(d: dict[str, float]) -> None:
    """Clamp every value in ``d`` to [-1.0, +1.0] in place.

    Rejects NaN and Inf — those would otherwise pass through every
    comparison (`v > 1.0` is False for NaN, `v < -1.0` is False for
    NaN) and propagate downstream through `dot()` to produce NaN
    scores. NaN scores break ranking comparisons non-deterministically
    (Python's `max` over NaN-containing iterables returns the first
    NaN it sees, depending on iteration order), so the runtime would
    sample garbage instead of crashing. Raise loudly here instead.
    """
    for k, v in d.items():
        if math.isnan(v) or math.isinf(v):
            raise ValueError(
                f"register-effect weight {k!r}={v!r} is not finite; "
                "NaN/Inf inputs would silently corrupt downstream scoring"
            )
        if v > 1.0:
            d[k] = 1.0
        elif v < -1.0:
            d[k] = -1.0


# ---- eligibility gates (hard filters) -----------------------------------


@dataclass(frozen=True)
class EligibilityGate:
    """Boolean predicates applied BEFORE vector scoring (D36.1, D36.6).

    Hard gates shrink the eligible-lemma pool to those that pass every
    predicate. They are NOT continuous — a lemma either matches the
    requested culture or it doesn't. Soft preferences (a lemma that
    LEANS toward a register) belong in the vector axes, not in the
    gate.

    ``culture`` is the hard-gate axis from D5. A request for ``--culture
    english`` excludes everything not in the english stratum, including
    pack lemmas (the pack overlay re-introduces them as a separate
    eligibility set, see PackOverlay below).

    Era gates only by RECORD ENTRY (D46 refining D44): a bounded
    historical era excludes morphemes whose earliest evidence
    (``Meaning.record_start``) is at or after the era's end — "it has to
    have appeared in the record by then" — and nothing else. Pools
    accrete monotonically; a morpheme never expires (D44's fix stands),
    and open-ended eras (the present-day stage, ``era_record_cutoff is
    None``) gate nothing. Era's other effect — which reflex renders —
    stays outside the gate (D44). The retired D5-2 attested-inside-
    window filter is NOT this gate; see D46 for the distinction.

    ``stratum`` (D32, wyrd-lr4) filters within-language strata
    (Brittonic in Welsh, Norse-substrate in Northern English, etc.).
    None means no stratum constraint.

    ``allowed_pack_tags`` and ``excluded_pack_tags`` further refine
    pack-overlay eligibility — narrative input that wants only the
    ``war`` slice of the Tatar pack passes ``allowed_pack_tags={'war'}``;
    same shape for excluding.
    """

    culture: str
    stratum: str | None = None
    # D46: the requested era's END year; a morpheme is eligible iff its
    # record_start() is None or < this. None (no era / open-ended era,
    # incl. the deployed present-day default) disables the gate.
    era_record_cutoff: int | None = None
    # wyrd-wv85: the operator's ``--tag`` semantic filter as a HARD gate
    # (D36.6). A lemma must carry AT LEAST ONE of these tags to be
    # eligible (OR semantics, matching proportions' ``filter_for_tag``
    # which unions the keys for each requested tag). Empty = no tag
    # constraint. Distinct from the SOFT ``register.semantic_tags`` axis
    # (which biases among already-eligible lemmas) and from the pack-tag
    # filters (which gate scenario-pack overlays only).
    required_tags: frozenset[str] = frozenset()
    allowed_pack_tags: frozenset[str] = frozenset()
    excluded_pack_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        # D46 guard: cutoffs are era-cell END years (the earliest defined
        # end is latin/classical at 200); zero or negative would silently
        # exclude every dated morpheme, presenting as "0 names generated"
        # with no diagnostic. Raise loudly at construction instead.
        if self.era_record_cutoff is not None and self.era_record_cutoff <= 0:
            raise ValueError(
                f"era_record_cutoff ({self.era_record_cutoff}) must be a "
                "positive year; None disables the record gate"
            )


# ---- pack overlay (scenario-pack composition) ---------------------------


@dataclass(frozen=True)
class PackOverlay:
    """A scenario-pack overlay on top of a native culture (v2gm-foundation).

    A pack overlay introduces pack-specific lemmas into the eligible
    pool alongside the native culture's lemmas. The pack's lemmas
    inherit their template's empirical-baseline (D36.4 Option B): a
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
    (D36.3): the operator can ask for "high realism + low pack" or
    "low realism + heavy pack" as orthogonal knobs.
    """

    pack_name: str
    template_donor: str
    template_recipient: str
    weight: float = 1.0
    # wyrd-ecjp.8: pack-tag narrowing. ``allowed_pack_tags`` admits
    # only pack lemmas carrying at least one matching tag; empty set
    # (default) admits all pack lemmas. ``excluded_pack_tags`` drops
    # pack lemmas carrying any matching tag (applied after the
    # allowed filter). Both predicates run pre-score so the gate
    # cost stays O(N) in the pack's lemma count rather than full
    # scoring. Used by '--pack-tag-filter <pack>:<tag1,tag2>' (the
    # operator-facing narrowing for narrative subset selection per
    # wyrd-sreb).
    allowed_pack_tags: frozenset[str] = frozenset()
    excluded_pack_tags: frozenset[str] = frozenset()


# ---- pack manifest (bundle authoring shape) -----------------------------


@dataclass(frozen=True)
class PackManifest:
    """The authoring-side descriptor for a scenario pack (wyrd-ecjp.10b).

    Each pack lives in its own ``packs/<pack_name>/`` directory
    alongside the main bundle's meanings.json (operator chose per-pack
    directories for max flexibility — see ecjp.10 design notes). The
    directory contains:

    * ``manifest.json`` — this dataclass serialized.
    * ``meanings.json`` — the pack's lemma set, in the same shape as
      the main bundle's meanings.json (list of subjects with words).

    The runtime's pack-overlay support (wyrd-ecjp.8
    ``select_via_vector_scoring(pack_meaning_dbs=...)``) consumes the
    loaded packs to admit pack lemmas into the eligible pool. The
    PackOverlay request-side dataclass references a manifest by
    ``pack_name``; load_packs returns ``{pack_name: PackBundle}`` so
    the dispatch layer can resolve overlay → meaning_db at request
    time.

    ``default_weight`` is the pack's default contribution scalar when
    the operator doesn't override via ``--pack <name>:<weight>``. 1.0
    matches the empirical loan rate; 0.3 (a reasonable v2gm starting
    point) under-weights so packs feel like minority influence rather
    than overwhelming the native register.

    ``version`` is a versioning string the operator manages — content
    hash, semver, or date string. Used by the loader to surface drift
    when a deploy includes packs with different versions than
    expected.
    """

    pack_name: str
    template_donor: str
    template_recipient: str
    default_weight: float = 1.0
    version: str = "unversioned"


# ---- request vector (ecjp.1 four-axis composed request) -----------------


@dataclass(frozen=True)
class ScoringWeights:
    """Per-axis weight scalars in the canonical composition rule (D36.2).

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

    Per D36.3: empirical baseline is just another axis with continuous
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
         pool (D36.1, D36.6). Lemmas that fail the gate are removed BEFORE
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
    # wyrd-4rp8: a thematic mood's tag preference ({tag: weight}, e.g. grim →
    # {death: 0.7, military: 0.3, …}). NOT a hard gate (that's the broken design
    # this replaces — it gated the whole pool and fell back) and NOT the soft
    # score axis (no cross-usage teeth post-bol9). It drives a "one mood morpheme
    # per name" overlay in select_via_vector_scoring: one slot whose pool can
    # carry a mood tag is restricted to mood-tagged morphemes; every other slot
    # generates normally. Empty (default / no mood) → no overlay → byte-identical
    # to plain generation. Distinct from gate.required_tags (the HARD --tag
    # filter, D36.6).
    mood_tags: dict[str, float] = field(default_factory=dict)


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
    axis (D36.4, D36.7).

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
    invalidates the right downstream artifacts (D36.8 Phase 6 drift
    plumbing).
    """

    native: dict[NativePriorsKey, dict[str, float]] = field(default_factory=dict)
    loan_relationship: dict[LoanPriorsKey, dict[str, float]] = field(default_factory=dict)
    version: str = "unversioned"

    @cached_property
    def native_fallback_index(
        self,
    ) -> tuple[
        dict[tuple[str, str, str], dict[str, float]],
        dict[tuple[str, str], dict[str, float]],
        dict[str, dict[str, float]],
    ]:
        """Reverse indices for the era/tag-wildcard native-priors fallback
        levels, built once from ``native`` so the per-lemma lookups are O(1)
        instead of an O(N_cells) scan per call (the empirical-baseline scorer
        calls them tens of thousands of times per generate; on a full-corpus
        bundle the scan dominated generation latency — wyrd-akov-adjacent perf
        fix).

        Three maps, each ``key -> {lemma_ref: max weight over the collapsed
        dimensions}`` — exactly the maximum the scans computed:

        * ``l2[(culture, position, tag)]`` — level 2, era wildcard
          ``(c, p, t, *)`` (level 1 exact stays a direct ``native.get``).
        * ``l3[(culture, position)]`` — level 3, ``(c, p, *, *)``.
        * ``l4[culture]`` — level 4, ``(c, *, *, *)``.

        Max-aggregation preserves the scans' ``best is None or v > best``
        semantics exactly, via a ``not in`` first-seen check (correct for ANY
        float weight — a ``.get(lemma, -inf)`` seed would drop a weight of
        exactly ``-inf``, since ``-inf > -inf`` is False). This includes a
        lemma stored at weight 0.0 (present, not absent): it lands in the index
        and the lookup returns 0.0, not None, matching ``_native_lookup_*``'s
        ``in``-check convention.
        """
        l2: dict[tuple[str, str, str], dict[str, float]] = {}
        l3: dict[tuple[str, str], dict[str, float]] = {}
        l4: dict[str, dict[str, float]] = {}
        for (culture, position, tag, _era), cell in self.native.items():
            d2 = l2.setdefault((culture, position, tag), {})
            d3 = l3.setdefault((culture, position), {})
            d4 = l4.setdefault(culture, {})
            for lemma_ref, weight in cell.items():
                if lemma_ref not in d2 or weight > d2[lemma_ref]:
                    d2[lemma_ref] = weight
                if lemma_ref not in d3 or weight > d3[lemma_ref]:
                    d3[lemma_ref] = weight
                if lemma_ref not in d4 or weight > d4[lemma_ref]:
                    d4[lemma_ref] = weight
        return l2, l3, l4
