"""Knob → RequestVector translator for the Kenning vector-scoring
dispatch (wyrd-ecjp.5 PR C).

``Kenning.generate`` consumes per-call knobs (``culture``, ``tags``,
``harshness``, ``mood``, ``cohesion``, ``era``, ``stratum``,
``include_fiction``). The vector-scoring path (now the only scoring
path) needs them translated into a :class:`RequestVector` so
:func:`select_via_vector_scoring` can drive the D36.2 composition
rule.

This adapter is the bridge. It accepts that knob set and emits a
fully-built RequestVector. The translation is
deliberately conservative for v1:

* ``culture`` + ``era`` + ``stratum`` + ``tags`` + a mood's SEMANTIC
  tags map to the EligibilityGate's hard predicates (``required_tags``,
  the D36.6 hard filter, wyrd-wv85). Both explicit ``--tag`` and the
  tags a mood expands to (``grim`` → death/military/monster/...) gate
  the pool, mirroring proportions' ``_apply_mood`` → ``filter_for_tag``.
* ``tags`` ALSO becomes ``register.semantic_tags = {tag: 1.0, ...}``
  so the sem-axis biases among the (now tag-gated) lemmas. The hard
  gate restricts the pool; the soft axis discriminates within it.
* ``harshness`` (D6) becomes ``register.phonological`` weights —
  positive on cluster_density / final_fortition, negative on
  vowel_final_bias / soft_consonants. This mirrors the harshness bias
  direction the retired proportion-table sampler used.
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

from wyrd.generators.kenning.registers.effects import (
    available_register_effects,
    parse_mood_spec,
)
from wyrd.generators.kenning.vectors.schemas import (
    EligibilityGate,
    PackOverlay,
    RegisterEffect,
    RequestVector,
    ScoringWeights,
    compose_register_effects,
)


def _uniform_tag_weights() -> dict[str, float]:
    """Return a ``{tag: 1.0}`` dict covering every tag the bundled
    meaning_db carries. Used by :func:`build_request_vector` as the
    "no-opinion default" register when the operator doesn't supply
    explicit tags, a mood, or harshness — without it, vector scoring
    would only produce output when the request expressed semantic
    interest, which collapses the plain unfiltered-generation use case.

    Lazy + cached: the tag universe is bundle-stable for the
    container's lifetime, so we compute it once and reuse. Pulled via
    ``_load_meanings`` (not a freestanding registry) because the
    bundled ``tag_db`` IS the source of truth for the tag universe."""
    global _UNIFORM_TAG_WEIGHTS_CACHE
    if _UNIFORM_TAG_WEIGHTS_CACHE is None:
        from wyrd.generators.kenning import _load_meanings

        _, tag_db = _load_meanings()
        _UNIFORM_TAG_WEIGHTS_CACHE = dict.fromkeys(tag_db, 1.0)
    return dict(_UNIFORM_TAG_WEIGHTS_CACHE)


_UNIFORM_TAG_WEIGHTS_CACHE: dict[str, float] | None = None


def _harshness_to_phonological(harshness: float) -> dict[str, float]:
    """Translate the D6 harshness scalar (0..1) into a register's
    phonological-weight dict.

    The retired proportion-table sampler's harshness bias preferred
    stop-final / cluster-heavy morphemes and de-preferred vowel-final /
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


def _mood_specs_to_register_effects(
    mood_specs: Iterable[str],
) -> list[RegisterEffect]:
    """wyrd-kq7w.3: resolve each mood spec to a graduated catalog
    register effect. The returned list composes via
    :func:`compose_register_effects` (sum-then-clamp) into the
    request's register, alongside the adapter's harshness-driven
    phonological effect + the explicit-tags effect.

    Each spec is either a bare name (``grim``) or graduated form
    (``harsh:0.5``) per the catalog's :func:`parse_mood_spec`
    contract. Replaces the pre-rip MOODS-dict + harshness-scalar
    extraction that lossily compressed catalog dims through a
    single legacy harshness float; the catalog now flows through
    end-to-end on the vector path.

    Raises:
        ValueError: when ``name`` (the part before any colon) is
            unknown — the catalog's ``KeyError`` gets translated
            here so callers grepping ``"unknown mood"`` stay bug-
            compatible with the legacy ``_apply_mood`` error shape.
            Also raised by :func:`parse_mood_spec` itself when the
            graduation suffix is unparseable as a float
            (``"harsh:x"``); that path bubbles through unchanged
            with the per-spec error message intact.
    """
    out: list[RegisterEffect] = []
    for spec in mood_specs:
        try:
            out.append(parse_mood_spec(spec))
        except KeyError as exc:
            # Re-raise as ValueError with the operator-facing
            # available-names list, matching the legacy
            # ``_apply_mood`` error shape so existing callers'
            # diagnostics don't regress.
            raise ValueError(
                f"unknown mood; expected one of {available_register_effects()}"
            ) from exc
    return out


def build_request_vector(
    *,
    culture: str,
    tags: Iterable[str] = (),
    harshness: float = 0.0,
    mood: Iterable[str] = (),
    era_record_cutoff: int | None = None,
    stratum: str | None = None,
    weights: ScoringWeights | None = None,
    packs: tuple[PackOverlay, ...] = (),
) -> RequestVector:
    """Translate Kenning's existing per-call knobs into a fully-built
    :class:`RequestVector` for the vector-scoring runtime.

    Args:
        culture: hard-gate culture (e.g. ``english``, ``welsh``).
        tags: the operator's ``--tag`` filter. Becomes a HARD
            eligibility gate (``gate.required_tags``, wyrd-wv85): only
            lemmas carrying at least one requested tag survive. Each
            also contributes weight 1.0 to the soft ``semantic_tags``
            axis for within-pool discrimination.
        harshness: D6 harshness scalar in [0, 1]. Translates to the
            register's ``phonological`` weights (see
            :func:`_harshness_to_phonological`).
        mood: D6 mood specs (``grim``, ``harsh:0.5``, ...). Expands
            through MOODS to additional tags + harshness.
        era_record_cutoff: the requested era's END year (D46) — a
            bounded era excludes morphemes whose earliest evidence
            postdates it ("in the record by then"). ``None`` (no era,
            or an open-ended era like the present-day stage) disables
            the gate; pools accrete and nothing expires.
        stratum: within-language stratum filter (``native-welsh``,
            ``latin-loan``, etc.). ``None`` disables the filter.
        weights: per-axis scoring weights. Defaults to
            ``ScoringWeights()`` (1.0 across all four axes).
        packs: tuple of :class:`PackOverlay` declaring scenario-pack
            overlays for this request (wyrd-ecjp.11). Empty tuple
            (default) means no packs — pure native generation. The
            runtime's :func:`select_via_vector_scoring` consumes
            ``request.packs`` for the baseline-axis pack composition
            (D36.4 Option B) and a sibling ``pack_meaning_dbs`` arg
            for admitting pack lemmas into the eligible pool.

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
    # Materialize Iterable inputs UP FRONT — both ``tags`` and ``mood``
    # are declared ``Iterable[str]``, so generator-typed callers would
    # otherwise be exhausted by the first walk
    # (``_mood_specs_to_register_effects`` for ``mood``;
    # ``dict.fromkeys`` for ``tags``) and the later emptiness check
    # below would incorrectly fire the uniform-tag fallback.
    tags = tuple(tags)
    mood = tuple(mood)

    # wyrd-kq7w.3: mood specs resolve to graduated catalog effects
    # that compose component-wise (sum + clamp) into the request
    # register. The adapter's explicit-tags + harshness-scalar knobs
    # produce a separate adapter-shape RegisterEffect; the final
    # request register is the composition of [adapter, *catalog].
    mood_effects = _mood_specs_to_register_effects(mood)

    # Default-fill the semantic_tags dict when the operator hasn't
    # expressed any preference (no explicit tags + no mood + no
    # harshness). Without this, ``baseline_score_native`` early-returns
    # 0 on the empty-register branch and the vector path collapses to
    # an empty pool — vector mode would only ever fire with a mood.
    # The default register surfaces empirical popularity as the
    # composition signal (every tag in the lemma's set gets uniform
    # 1.0 weight), which IS the sensible "no opinion → match the
    # corpus" default.
    if not tags and not mood and harshness == 0.0:
        semantic_tags = _uniform_tag_weights()
    else:
        semantic_tags = dict.fromkeys(tags, 1.0)

    adapter_effect = RegisterEffect(
        name="adapter",
        phonological=_harshness_to_phonological(harshness),
        semantic_tags=semantic_tags,
    )

    register = compose_register_effects([adapter_effect, *mood_effects])

    # wyrd-4rp8: a thematic mood's tags are a SOFT preference, NOT a hard gate.
    # The pre-4rp8 design (wyrd-wv85) unioned the mood's positive semantic tags
    # into gate.required_tags to mirror proportions' _apply_mood → filter_for_tag.
    # In the vector path that gate is too aggressive: it shrinks the pool to
    # mood-only morphemes, which can't fill most structures, so generation
    # silently FELL BACK to the ungated pool — making moods a no-op. Per D36.6 a
    # mood's semantic component is the SOFT preference; only --tag is the hard
    # gate. So required_tags carries ONLY the explicit --tag values; the mood's
    # per-tag weights ride on RequestVector.mood_tags, which drives the
    # "one mood morpheme per name" overlay in select_via_vector_scoring. A mood's
    # PHONOLOGICAL component (harsh) stays on the register's phon axis as before.
    mood_tags: dict[str, float] = {}
    for effect in mood_effects:
        for tag, weight in effect.semantic_tags.items():
            if weight > 0:
                mood_tags[tag] = mood_tags.get(tag, 0.0) + weight
    # Era reaches the gate ONLY as the D46 record-entry cutoff ("in the
    # record by then"); its render effect stays downstream
    # (era_render_language / the D33 era-reflex machinery, D44).
    gate = EligibilityGate(
        culture=culture,
        era_record_cutoff=era_record_cutoff,
        stratum=stratum,
        required_tags=frozenset(tags),
    )

    return RequestVector(
        gate=gate,
        register=register,
        weights=weights or ScoringWeights(),
        packs=tuple(packs),
        mood_tags=mood_tags,
    )
