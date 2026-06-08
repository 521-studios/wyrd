"""Realism-sample runner for the absolute corpus-realism gate (wyrd-jfaz).

Generates N names from the Kenning generator under the vector scoring
path, captures the :class:`NameSample` features the realism metrics
consume, and pairs them with a corpus reference derived from the same
bundle — ready to feed :func:`compute_realism_report`.

This module is the bridge between the high-level realism-measurement
primitives (in `drift_measurement.py`) and the actual Kenning
generator. Splitting keeps the metric code pure-functional and
testable in isolation; the runner here is the side-effecting wrapper
that actually drives generation.

Operator workflow:
    >>> from wyrd.generators.kenning.runtime.drift_runner import (
    ...     run_realism_samples,
    ... )
    >>> from wyrd.generators.kenning.runtime.drift_measurement import (
    ...     compute_realism_report,
    ... )
    >>> samples, reference = run_realism_samples(
    ...     culture='english', count=1000, base_seed=42,
    ...     priors_path='priors.json',  # optional; enables baseline axis
    ... )
    >>> report = compute_realism_report('english', samples, reference)
"""

from __future__ import annotations

from typing import Any

from wyrd.generators.kenning.runtime.drift_measurement import NameSample


def _sample_from_generation_result(result: Any) -> NameSample:
    """Translate a Kenning :class:`GenerationResult` into the
    :class:`NameSample` shape the drift metrics consume.

    The result surface comes from ``result.result``. Tags + positions
    + morphemes derive from ``result.components`` — each component
    is a dict with ``usage`` / ``location`` / ``tags`` keys (per the
    GenerationResult contract surfaced in cli/generate.py).

    ``decomposes`` is True when every component represents a
    full-decomposition leaf (no unaccounted residue). The current
    Kenning generator always returns fully-decomposed names by
    construction, so this is True for now; the drift-measurement
    framework is set up to surface noise-shifted drift if a future
    path produces unaccounted-bearing names.
    """
    # GenerationResult is a dataclass with .result + .components — no
    # need for defensive getattr. Drop in if the registry contract
    # ever changes.
    components = result.components or []
    tags: list[str] = []
    positions: list[str] = []
    morphemes: list[str] = []
    for c in components:
        tags.extend(c.get("tags") or [])
        loc = c.get("location")
        if loc:
            positions.append(loc)
        usage = c.get("usage")
        if usage:
            # Strip decoration dashes per the priors-table lemma_ref
            # convention (matches _lemma_ref_for in vector_name_select)
            morphemes.append(str(usage).replace("-", ""))
    return NameSample(
        # TODO(wyrd-ecjp.6 follow-up): the current Kenning generator
        # always returns fully-decomposed names by construction, so
        # decomposes=True is constant across both scoring modes and
        # the decomposition_rate metric is currently degenerate (both
        # sides 1.0, delta=0). The metric is in place for a future
        # path that could produce unaccounted-bearing names (e.g. a
        # vector scoring that picks lemmas whose surface doesn't
        # trie-decompose). Until then, treat decomposition_rate as
        # confirmatory rather than discriminating.
        surface=result.result or "",
        tags=tuple(tags),
        positions=tuple(positions),
        decomposes=True,
        morphemes=tuple(morphemes),
    )


def run_realism_samples(  # noqa: V103 — engine of the absolute corpus-realism CI gate (test is the consumer)
    *,
    culture: str,
    count: int,
    base_seed: int = 0,
    priors_path: str | None = None,
    tags: list[str] | None = None,
    harshness: float = 0.0,
    cohesion: float = 0.0,
    include_unglossed: bool = False,
):
    """Generate ``count`` VECTOR-mode samples for ``culture`` + compute the
    absolute corpus reference from the same bundle (wyrd-jfaz).

    Drives only the vector path (no proportions baseline) and pairs the samples
    with a :class:`realism_reference.CorpusReference` derived from the same
    ``NameGenerator`` data — so the gate has no dependency on the
    proportions scoring path and survives its deletion (epic wyrd-ej28).

    Returns ``(samples_vector, reference)`` ready to feed
    :func:`drift_measurement.compute_realism_report`. The reference is
    computed from the same ``_load_culture`` instance ``Kenning`` generates
    from (both cached), so reference + samples reflect the same bundle.
    """
    from wyrd.generators.kenning import _load_culture
    from wyrd.generators.kenning.generators.kenning import Kenning
    from wyrd.generators.kenning.runtime.realism_reference import compute_corpus_reference

    k = Kenning()
    name_gen, _ = _load_culture(culture)
    # The reference's gloss policy MUST match the generation request below
    # (same include_unglossed), so it's the exact convergence target.
    reference = compute_corpus_reference(culture, name_gen, include_unglossed=include_unglossed)

    base_params: dict[str, Any] = {
        "culture": culture,
        "tags": tags or [],
        "harshness": harshness,
        "cohesion": cohesion,
        "scoring_mode": "vector",
        "include_unglossed": include_unglossed,
    }
    if priors_path:
        base_params["priors_path"] = priors_path

    samples_vector: list[NameSample] = []
    for i in range(count):
        try:
            r_vec = k.generate(base_params, seed=base_seed + i)
            samples_vector.append(_sample_from_generation_result(r_vec))
        except ValueError:
            # Post-wyrd-tbke the vector path degrades rather than raising,
            # so this is now only a defensive skip for a genuinely empty
            # (gate-excludes-everything) request.
            pass

    return samples_vector, reference
