"""Knob → RequestVector translator for the Kenning vector-scoring
dispatch (wyrd-ecjp.5 PR C).

The legacy ``Kenning.generate`` path consumes per-call knobs
(``culture``, ``tags``, ``harshness``, ``mood``, ``cohesion``,
``era``, ``stratum``, ``include_fiction``) that translate into the
proportion-table sampler's call signature. The vector-scoring path
needs the same knobs translated into a :class:`RequestVector` so
:func:`select_via_vector_scoring` can drive the D36.2 composition
rule.

This adapter is the bridge. It accepts the same knob set the legacy
path uses and emits a fully-built RequestVector. The translation is
deliberately conservative for v1:

* ``culture`` + ``era`` + ``stratum`` map to the EligibilityGate's
  hard predicates.
* ``tags`` becomes ``register.semantic_tags = {tag: 1.0, ...}`` so
  the sem-axis scores lemmas carrying any of those tags.
* ``harshness`` (D6) becomes ``register.phonological`` weights —
  positive on cluster_density / final_fortition, negative on
  vowel_final_bias / soft_consonants. This mirrors the legacy
  proportion-table sampler's harshness bias direction.
* ``mood`` (D6) expands through ``MOODS`` into tag + harshness
  contributions, summed alongside any explicit ``tags`` /
  ``harshness`` knobs.
* ``cohesion`` and ``exclude_tags`` are NOT in the RequestVector —
  they pass through separately to ``select_via_vector_scoring`` as
  per-call args (cohesion drives the D17 adapter; exclude_tags is
  a pre-score gate).

This is intentionally MINIMAL — the operator-facing CLI knobs that
land in ecjp.9 (``--register harsh:0.8,grim:0.5`` direct register-
catalog access, ``--realism 0.5`` direct scoring-weights access,
``--pack neo-khuzdul`` pack overlays) bypass this adapter and build
the RequestVector directly. The adapter exists so the EXISTING
operator surface (the knobs Kenning already accepts) round-trips
through the vector path without regression.
"""

from __future__ import annotations

from collections.abc import Iterable

from wyrd.generators.kenning.registers.moods import MOODS
from wyrd.generators.kenning.vectors.schemas import (
    EligibilityGate,
    RegisterEffect,
    RequestVector,
    ScoringWeights,
)


def _harshness_to_phonological(harshness: float) -> dict[str, float]:
    """Translate the D6 harshness scalar (0..1) into a register's
    phonological-weight dict.

    The legacy proportion-table sampler's harshness bias prefers
    stop-final / cluster-heavy morphemes and de-prefers vowel-final /
    soft ones. The vector-side equivalent dials the relevant
    PhonologicalVector dimensions:

    * ``cluster_density``: +harshness (more clusters)
    * ``final_fortition``: +harshness * 0.8 (more final stops)
    * ``vowel_final_bias``: -harshness * 0.6 (penalize vowel-final)
    * ``soft_consonants``: -harshness * 0.5 (penalize sonorant-heavy)

    Coefficients are calibrated such that ``harshness=1.0`` produces
    a register roughly comparable to the legacy harsh-mood's bias
    direction. Empirical drift comparison lands in ecjp.6/7.

    Returns ``{}`` when harshness <= 0 (no contribution; saves a
    dot-product term in the per-lemma scoring loop).
    """
    if harshness <= 0:
        return {}
    return {
        "cluster_density": harshness,
        "final_fortition": harshness * 0.8,
        "vowel_final_bias": -harshness * 0.6,
        "soft_consonants": -harshness * 0.5,
    }


def _mood_to_tags_and_harshness(
    mood_specs: Iterable[str],
    base_tags: list[str],
    base_harshness: float,
) -> tuple[list[str], float]:
    """Expand ``--mood`` flag values through the MOODS catalog into
    additional tags + harshness. Matches the legacy ``_apply_mood``
    semantics so the same operator input produces equivalent
    behavior on the vector path.

    Each mood spec is either a bare name (``grim``) or a graduated
    form (``harsh:0.5``). For tag-bearing moods, the mood's tags
    union into ``base_tags``. For harshness-bearing moods, the
    spec's value (or the MOODS default) is max'd into
    ``base_harshness`` — matching the legacy max-harshness
    composition rule.

    Returns a (tags, harshness) tuple; the caller threads these into
    the RequestVector construction.
    """
    tags = list(base_tags)
    harshness = base_harshness
    for spec in mood_specs:
        # Spec may be "grim" or "harsh:0.5"
        if ":" in spec:
            name, _, value_str = spec.partition(":")
            override_value = float(value_str)
        else:
            name = spec
            override_value = None
        mood = MOODS.get(name)
        if mood is None:
            continue
        if "tags" in mood:
            for t in mood["tags"]:
                if t not in tags:
                    tags.append(t)
        if "harshness" in mood:
            mood_harshness = override_value if override_value is not None else mood["harshness"]
            harshness = max(harshness, mood_harshness)
    return tags, harshness


def build_request_vector(
    *,
    culture: str,
    tags: Iterable[str] = (),
    harshness: float = 0.0,
    mood: Iterable[str] = (),
    era_min: int | None = None,
    era_max: int | None = None,
    stratum: str | None = None,
    weights: ScoringWeights | None = None,
) -> RequestVector:
    """Translate Kenning's existing per-call knobs into a fully-built
    :class:`RequestVector` for the vector-scoring runtime.

    Args:
        culture: hard-gate culture (e.g. ``english``, ``welsh``).
        tags: optional list of semantic tags the operator wants the
            request to favor. Each contributes weight 1.0 to the
            register's ``semantic_tags`` dict.
        harshness: D6 harshness scalar in [0, 1]. Translates to the
            register's ``phonological`` weights (see
            :func:`_harshness_to_phonological`).
        mood: D6 mood specs (``grim``, ``harsh:0.5``, ...). Expands
            through MOODS to additional tags + harshness.
        era_min / era_max: era-gate bounds. ``None`` disables either
            bound; both ``None`` disables the era gate.
        stratum: within-language stratum filter (``native-welsh``,
            ``latin-loan``, etc.). ``None`` disables the filter.
        weights: per-axis scoring weights. Defaults to
            ``ScoringWeights()`` (1.0 across all four axes).

    Returns:
        A :class:`RequestVector` ready to feed
        :func:`select_via_vector_scoring`.

    Per-axis breakdown:
        * **phon** — driven by ``harshness`` (positive cluster_density
          + final_fortition; negative vowel_final_bias +
          soft_consonants).
        * **sem** — driven by ``tags`` + mood-expanded tags.
        * **pos** — left empty in v1 (no operator-facing knob for it;
          packs / scenario-translator can populate when they land).
        * **base** — driven by the loaded priors at request time; the
          adapter doesn't touch it here.
    """
    # Expand mood into tag + harshness contributions
    expanded_tags, expanded_harshness = _mood_to_tags_and_harshness(mood, list(tags), harshness)

    register = RegisterEffect(
        name="adapter",
        phonological=_harshness_to_phonological(expanded_harshness),
        semantic_tags=dict.fromkeys(expanded_tags, 1.0),
    )

    gate = EligibilityGate(
        culture=culture,
        era_min=era_min,
        era_max=era_max,
        stratum=stratum,
    )

    return RequestVector(
        gate=gate,
        register=register,
        weights=weights or ScoringWeights(),
    )


def era_midpoint_from_range(
    era_min: int | None,
    era_max: int | None,
) -> int:
    """Resolve an era_midpoint integer from the era-range bounds for
    the baseline-axis lookup (D36.7 baseline_score_native).

    * Both bounds present → midpoint
    * One bound present → use that bound directly (single-cell
      request, baseline lookup walks the tag-wildcard fallback)
    * Neither bound (open request) → 0 (matches the priors-table
      lookup convention for "no era specified" → hits the
      hierarchical fallback's tag-wildcard / all-wildcard cells)
    """
    if era_min is not None and era_max is not None:
        return (era_min + era_max) // 2
    if era_min is not None:
        return era_min
    if era_max is not None:
        return era_max
    return 0
