"""Eligibility-gate runtime (wyrd-ecjp.3 Phase 3).

The vector-driven generator's hard-gate primitive (D36.1, D36.6): a
predicate function that, given a Meaning and a request, decides
whether the Meaning is eligible to be considered for SCORING in the
first place. Lemmas that fail any gate are removed BEFORE the four
scoring axes run; lemmas that pass all gates flow into the per-axis
sub-scoring (Phase 4, wyrd-ecjp.4).

Per D36 the gates are HARD: a lemma either matches or it doesn't.
Soft preferences belong on the vector axes (phonological / semantic
/ position / empirical-baseline). The split is load-bearing:

* The gate runs once per request and shrinks the pool from ~8500
  bundle morphemes to ~hundreds per (culture, era, stratum) cell.
* The per-axis scoring runs N (number of slots) × M (filtered pool
  size) times per generated name. Doing all the work in the scorer
  would multiply ~hundreds by ~tens for every slot, every name.

The gate predicates implemented here:

* **culture**: validation only — Meanings in the bundle's culture-
  specific slice already come pre-filtered by culture (upstream
  pick uses ``<culture>_place_names.json``). The gate raises on an
  unknown culture string so a misconfigured request fails loudly
  rather than silently selecting from the union of all cultures.
* **era**: ``Meaning.attested_in_era_range`` — wraps the existing
  D5-2 / wyrd-lyp era filter under the (era_min, era_max) tuple.
  Meanings with no attested-year data pass any era filter (the
  "no data ≠ evidence of absence" rule from D5-2).
* **stratum**: ``Meaning.in_stratum`` — wraps the existing wyrd-lr4
  Phase 3 stratum filter. Same "no data → pass" convention.
* **tag-required**: every requested tag must appear in
  ``meaning.tags``. Empty required-set means no constraint.
* **tag-excluded**: no excluded tag may appear in ``meaning.tags``.
* **pack-allowlist + pack-tag-filter**: today a no-op (scenario
  packs aren't in the bundle yet). Stub functions are provided
  with a stable signature so the Phase 5 NameGenerator rewrite
  + Phase 7 bundle / pack-overlay work (wyrd-ecjp.5 / wyrd-ecjp.8)
  can wire packs in without disturbing the gate API. See
  wyrd-v2gm for the scenario-pack epic.

Per the D36.7 design (per (culture × position × tag × era) cells),
the gate produces an output set that the scoring runtime consumes
directly. Pre-computed bitmap indexes for fast cell-based lookups
(wyrd-ecjp.3 ticket's "pre-computed indexes" half) belong in the
bundle-build layer; that work touches lexicon.py / export_meanings
and is deferred to a follow-up to keep this PR scoped to runtime
predicate logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from wyrd.generators.kenning import CULTURES
from wyrd.generators.kenning.meaning import Meaning
from wyrd.generators.kenning.vector_schemas import (
    EligibilityGate,
    PackOverlay,
)


class UnknownCultureError(ValueError):
    """Raised when a request references a culture name that isn't in
    the v1 ``CULTURES`` set. Distinct subclass so the eligibility
    gate's culture-validation failure surfaces clearly in operator
    output."""


def passes_culture_gate(culture: str) -> bool:
    """Validate the culture string. Returns True for any culture in
    the v1 ``CULTURES`` set; raises ``UnknownCultureError`` otherwise.

    The gate does NOT filter per-Meaning on culture — Meanings in the
    bundle's per-culture slice already come from
    ``<culture>_place_names.json`` (the per-culture vocabulary
    selection happens upstream of the eligibility gate). This
    predicate's job is to fail loudly on a misconfigured request
    rather than silently selecting from the union of all cultures.
    """
    if culture not in CULTURES:
        raise UnknownCultureError(f"unknown culture {culture!r}; expected one of {CULTURES}")
    return True


def passes_era_gate(meaning: Meaning, era_min: int | None, era_max: int | None) -> bool:
    """True if the Meaning has at least one attested form in
    ``[era_min, era_max)``, OR has no attested-year data at all (the
    D5-2 'no data → pass' rule). Both bounds None → no filter.
    """
    if era_min is None and era_max is None:
        return True
    return meaning.attested_in_era_range((era_min, era_max))


def passes_stratum_gate(meaning: Meaning, stratum: str | None) -> bool:
    """True if the Meaning has at least one form classified into
    ``stratum``, OR has no stratum data at all (wyrd-lr4 Phase 3
    'no data → pass' rule). ``stratum=None`` → no filter.
    """
    return meaning.in_stratum(stratum)


def passes_tag_required_gate(meaning: Meaning, required: frozenset[str]) -> bool:
    """True if every requested tag appears in ``meaning.tags``.
    Empty required-set → no filter.

    This is the AND semantics of ``--tag death --tag military``: a
    Meaning must carry both tags. For OR semantics use multiple
    separate generation requests + merge.
    """
    if not required:
        return True
    return all(t in meaning.tags for t in required)


def passes_tag_excluded_gate(meaning: Meaning, excluded: frozenset[str]) -> bool:
    """True if NO excluded tag appears in ``meaning.tags``. Empty
    excluded-set → no filter.

    Pairs with the existing ``--exclude-tags`` knob (wyrd-yan).
    """
    if not excluded:
        return True
    return not any(t in meaning.tags for t in excluded)


def passes_pack_gate(
    meaning: Meaning,
    gate: EligibilityGate,
    packs: Sequence[PackOverlay],  # noqa: ARG001 — load-bearing for future pack metadata
) -> bool:
    """Pack admission + pack-tag filtering. Today a no-op because
    Meanings in the v1 bundle don't carry scenario-pack metadata.

    The signature is stable for when scenario packs land in the
    bundle (wyrd-v2gm epic) and Meanings gain a ``pack_name``
    attribute. At that point this predicate will check:

      1. If ``meaning.pack_name`` is set, the pack must appear in
         ``packs`` (one of the admitted PackOverlay.pack_name values).
      2. If ``gate.allowed_pack_tags`` is non-empty, the meaning's
         tags must intersect.
      3. If ``gate.excluded_pack_tags`` is non-empty, the meaning's
         tags must NOT contain any.

    For now the gate returns True for every Meaning, which means
    pack admission flows through the v1 culture-bundle path
    (Meanings are either in the bundle or not; there's no third
    "pack source" alternative yet). Even when
    ``gate.allowed_pack_tags`` / ``excluded_pack_tags`` are set,
    they have nothing to discriminate against — Meanings don't
    carry pack-specific tags yet. The wyrd-sreb narrative-translator
    epic will need the pack-tag-filter half of this predicate to
    admit narrative-subset of pack lemmas.
    """
    # gate.allowed_pack_tags / .excluded_pack_tags are read in the
    # docstring above; the conditional that would consume them lives
    # in the future scenario-pack rewrite. Keeping the signature
    # stable so the call sites in admits/ filter_meanings don't move.
    _ = gate.allowed_pack_tags, gate.excluded_pack_tags
    return True


def admits(
    meaning: Meaning,
    gate: EligibilityGate,
    *,
    packs: Sequence[PackOverlay] = (),
    tag_required: frozenset[str] = frozenset(),
    tag_excluded: frozenset[str] = frozenset(),
) -> bool:
    """True if ``meaning`` passes EVERY gate predicate.

    Gate predicates are checked in cheapest-first order so a Meaning
    that fails an early gate doesn't pay the cost of the later ones.
    Culture comes first because it's pure validation (no per-Meaning
    work); tag gates are cheap dict/set lookups; era and stratum
    iterate the per-form dicts.

    Pack gates are last because they're stubs today; ordering will
    matter once scenario packs land.
    """
    # Validation gate — raises on unknown culture.
    passes_culture_gate(gate.culture)
    if not passes_tag_required_gate(meaning, tag_required):
        return False
    if not passes_tag_excluded_gate(meaning, tag_excluded):
        return False
    if not passes_era_gate(meaning, gate.era_min, gate.era_max):
        return False
    if not passes_stratum_gate(meaning, gate.stratum):
        return False
    return passes_pack_gate(meaning, gate, packs)


def filter_meanings(
    meanings: Iterable[Meaning],
    gate: EligibilityGate,
    *,
    packs: Sequence[PackOverlay] = (),
    tag_required: frozenset[str] = frozenset(),
    tag_excluded: frozenset[str] = frozenset(),
) -> list[Meaning]:
    """Apply ``admits`` to every Meaning, return the passing subset.

    Caller-side concerns this function does NOT handle:

    * The per-culture vocabulary selection (the bundle's
      ``<culture>_place_names.json`` keys the meaning pool to begin
      with). Callers must supply ``meanings`` already restricted to
      the culture's vocabulary.
    * Pre-computed indexes for fast cell-based lookup — that's the
      bundle-build layer's job (deferred wyrd-ecjp.3 follow-up).
    * Empirical scoring — that's Phase 4 (wyrd-ecjp.4).
    """
    return [
        m
        for m in meanings
        if admits(
            m,
            gate,
            packs=packs,
            tag_required=tag_required,
            tag_excluded=tag_excluded,
        )
    ]
