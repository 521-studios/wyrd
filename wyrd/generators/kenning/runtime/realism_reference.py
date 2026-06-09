"""Absolute corpus-realism reference (wyrd-jfaz).

The realism gate originally measured vector-mode DRIFT against the
now-removed proportions scoring path (a drift report over two sample
sets). Once the proportions *scoring* mode was retired (epic
wyrd-ej28), that baseline disappeared — there are no proportions
samples to compare against.

This module re-anchors the gate to an ABSOLUTE corpus reference derived
purely from the bundle DATA (which survives the scoring-mode deletion
under fork A and is rebuilt from the real toponym corpus on every
``rebuild-proportions`` / re-export):

* tags / positions / morpheme frequencies come from the per-bucket
  empirical frequency tables (``usage_frequency_by_bucket``) and the
  structure proportions (``structs``) — exactly the data the generator
  samples from.

So as the corpus grows (more toponyms / morphemes / variants mined), the
reference updates automatically with the next export — no frozen snapshot
to go stale, and no dependency on the removed proportions scoring path.

Correctness is provable by construction: the reference is the EXACT
expected distribution of a corpus-faithful sampler (one that draws
structs by proportion and usages by frequency), so such a sampler would
land at ~0 divergence against it. (The proportions sampler that once
empirically confirmed this is gone with the retired scoring path.) The
vector path's divergence against this reference is the absolute realism
signal.

The per-usage features (morpheme surface, position label, tags) are
computed the SAME way ``NewName.components`` computes them — resolve the
usage by bare surface, ``_rank_siblings``, take the top sibling's
``location`` and the union of the ranked siblings' tags — so the
reference distributions are directly comparable to the ``NameSample``
distributions the drift metrics build from generated names.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CorpusReference:
    """The corpus-empirical distributions a corpus-faithful generator
    would produce, derived analytically from a culture's bundle data.

    * ``tag_distribution`` — normalized tag-occurrence shares ({tag: p}),
      comparable to ``drift_measurement._tag_distribution``.
    * ``position_distribution`` — normalized slot-position shares
      ({position: p}), comparable to
      ``drift_measurement.position_distribution``.
    * ``morpheme_counts`` — expected per-morpheme pick weights
      ({bare_surface: weight}); only the RANKING is used (vs a sample's
      morpheme counts) by ``morpheme_rank_correlation``, so the absolute
      scale is irrelevant.
    """

    culture: str
    tag_distribution: dict[str, float] = field(default_factory=dict)
    position_distribution: dict[str, float] = field(default_factory=dict)
    morpheme_counts: dict[str, float] = field(default_factory=dict)


@dataclass
class _WeightAccumulator:
    """Mutable per-feature weight tallies accumulated across slots.

    Bundles the three weight dicts so the per-slot helper takes one
    accumulator (not three out-params) with type-checked attribute
    access rather than stringly-typed keys."""

    tag: dict[str, float] = field(default_factory=dict)
    position: dict[str, float] = field(default_factory=dict)
    morpheme: dict[str, float] = field(default_factory=dict)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize a weight dict to shares summing to 1.0; empty when the
    total is non-positive."""
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in weights.items()}


def _accumulate_slot_weights(
    bucket: dict[str, float],
    p_struct: float,
    gloss_keep: frozenset[str] | None,
    features_fn: Callable[[str], tuple[str | None, list[str], str | None]],
    acc: _WeightAccumulator,
) -> None:
    """Fold one slot's usage-frequency ``bucket`` into the ``acc`` tallies.

    Applies the gloss keep-set BEFORE normalizing, so ``P(usage|slot)`` is
    over the eligible pool the generator actually samples (matches
    select's keep_keys filter + re-normalization). Each usage's weight is
    ``p_struct * (f / total_f)``, distributed to its morpheme, position
    (when present), and every tag — resolved via ``features_fn``. A
    usage whose features resolve to ``morpheme is None`` is skipped.
    """
    if gloss_keep is not None:
        bucket = {e: f for e, f in bucket.items() if e.lower().replace("-", "") in gloss_keep}
    total_f = sum(bucket.values())
    if total_f <= 0:
        return
    for usage, f in bucket.items():
        weight = p_struct * (f / total_f)
        morpheme, tags, location = features_fn(usage)
        if morpheme is None:
            continue
        acc.morpheme[morpheme] = acc.morpheme.get(morpheme, 0.0) + weight
        if location:
            acc.position[location] = acc.position.get(location, 0.0) + weight
        for tag in tags:
            acc.tag[tag] = acc.tag.get(tag, 0.0) + weight


def compute_corpus_reference(
    culture: str, name_gen, *, include_unglossed: bool = False
) -> CorpusReference:
    """Build the :class:`CorpusReference` for ``culture`` from a
    ``NameGenerator``'s bundle data.

    Analytical expected distribution of a corpus-faithful sampler:

        P(struct S)              = structs[S] / sum(structs)
        P(usage e | slot)        = freq[e] / sum(freq[ELIGIBLE bucket(slot)])
        weight of a slot's pick  = P(S) * P(e | slot)

    accumulated over every (struct, slot, usage) into per-tag,
    per-position, and per-morpheme weights. Per-usage features mirror
    ``NewName.components`` exactly (resolve by bare surface →
    ``_rank_siblings`` → top sibling ``location`` + union of ranked
    siblings' tags; morpheme = dash-stripped usage, case preserved, per
    ``drift_runner._sample_from_generation_result``).

    ``include_unglossed`` MUST match the generation request the samples
    are drawn under (the gate uses the ``False`` default — glossing is a
    project pillar, so the generator only draws gloss-eligible usages
    unless the operator opts in). The reference applies the SAME gloss
    keep-set the generator applies (``keep_keys_for_gloss``,
    a frozenset of bare surfaces), filtering each bucket BEFORE
    normalizing P(usage|slot) — so the reference is the exact convergence
    target of the actual generator, not of a hypothetical unglossed one.
    (era / stratum are not modeled: the gate runs at the open default, so
    their keep-sets are None / no-op — add them here if the gate ever
    requests a window.)
    """
    # Lazy import: ``_rank_siblings`` lives on the package ``__init__``;
    # importing it at module load would create a runtime→package cycle
    # (the same reason ``NewName.components`` imports it lazily).
    from wyrd.generators.kenning import _rank_siblings
    from wyrd.generators.kenning.runtime.proportions import _resolve_surface

    structs: dict = name_gen.structs
    freq_by_bucket: dict = name_gen.usage_frequency_by_bucket
    meaning_db: dict = name_gen.meaning_db
    total_struct = sum(structs.values()) or 1.0

    # Gloss keep-set: frozenset of bare surfaces, or None when every usage
    # is eligible (no-filter fast path). Mirrors the generator's bucket
    # filter (``k.lower().replace("-","") in keep_keys``).
    gloss_keep = name_gen.meaning_gen.keep_keys_for_gloss(include_unglossed)

    # Accumulator bundles the three weight tallies so the per-slot helper
    # takes one accumulator rather than three out-params.
    acc = _WeightAccumulator()

    # Resolve each usage's (morpheme, tags, location) once — the same
    # _resolve_surface + _rank_siblings work NewName.components does.
    feature_cache: dict[str, tuple[str | None, list[str], str | None]] = {}

    def _usage_features(usage: str) -> tuple[str | None, list[str], str | None]:
        cached = feature_cache.get(usage)
        if cached is not None:
            return cached
        ranked = _rank_siblings(_resolve_surface(meaning_db, usage))
        if not ranked:
            feat: tuple[str | None, list[str], str | None] = (None, [], None)
        else:
            tags = list(dict.fromkeys(t for m in ranked for t in m.tags))
            feat = (usage.replace("-", ""), tags, ranked[0].location)
        feature_cache[usage] = feat
        return feat

    for struct_key, proportion in structs.items():
        p_struct = proportion / total_struct
        for word in struct_key:
            for slot in word:
                # ``slot`` IS the per-bucket key the frequency table is
                # keyed by (select_via_vector threads the slot key straight
                # into _resolve_slot_usage_frequency).
                bucket = freq_by_bucket.get(slot, {})
                _accumulate_slot_weights(bucket, p_struct, gloss_keep, _usage_features, acc)

    return CorpusReference(
        culture=culture,
        tag_distribution=_normalize_weights(acc.tag),
        position_distribution=_normalize_weights(acc.position),
        morpheme_counts=acc.morpheme,
    )
