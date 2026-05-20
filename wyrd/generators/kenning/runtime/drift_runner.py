"""Drift-measurement sample runner (wyrd-ecjp.6).

Generates N names from the Kenning generator under both scoring modes
(legacy `proportions` and the new `vector` path), captures the
:class:`NameSample` features the drift metrics consume, and returns
the populated sample lists ready to feed :func:`compute_drift_report`.

This module is the bridge between the high-level drift-measurement
primitives (in `drift_measurement.py`) and the actual Kenning
generator. Splitting keeps the metric code pure-functional and
testable in isolation; the runner here is the side-effecting wrapper
that actually drives generation.

Operator workflow:
    >>> from wyrd.generators.kenning.runtime.drift_runner import (
    ...     run_drift_samples,
    ... )
    >>> from wyrd.generators.kenning.runtime.drift_measurement import (
    ...     compute_drift_report, format_drift_report_markdown,
    ... )
    >>> samples_a, samples_b = run_drift_samples(
    ...     culture='english', count=1000, base_seed=42,
    ...     priors_path='priors.json',  # optional; enables baseline axis
    ... )
    >>> report = compute_drift_report('english', samples_a, samples_b)
    >>> print(format_drift_report_markdown(report))
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
    components = getattr(result, "components", None) or []
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
        surface=getattr(result, "result", "") or "",
        tags=tuple(tags),
        positions=tuple(positions),
        decomposes=True,
        morphemes=tuple(morphemes),
    )


def run_drift_samples(
    *,
    culture: str,
    count: int,
    base_seed: int = 0,
    priors_path: str | None = None,
    tags: list[str] | None = None,
    harshness: float = 0.0,
    cohesion: float = 0.0,
) -> tuple[list[NameSample], list[NameSample]]:
    """Generate ``count`` names per scoring mode for ``culture`` and
    return the two sample lists ready for drift measurement.

    Args:
        culture: target culture (english / scottish / welsh / irish / breton).
        count: number of names to generate per side. The drift metrics
            stabilize around N=1000+; smaller samples produce noisier
            reports.
        base_seed: starting seed; each generated name uses ``base_seed + i``.
            Same base_seed → same sample sets (bit-stable for re-runs).
        priors_path: optional JSON sidecar path for the vector path's
            baseline axis. When absent, vector scoring falls back to
            phon + sem + pos axes (and may produce empty results,
            which surface as a smaller sample_size_b in the report).
        tags / harshness / cohesion: per-call knobs threaded through
            to both modes for fair comparison (the request shape is
            identical; only scoring_mode changes).

    Returns:
        ``(samples_proportions, samples_vector)`` — two lists ready
        to feed ``compute_drift_report``. Either list may be shorter
        than ``count`` if the corresponding mode raised on individual
        seeds (vector path raises ValueError on empty pick; the runner
        swallows + skips so the drift measurement can proceed).
    """
    # Lazy import to keep this module's cold-start cost from leaking
    # into callers that only import the drift-measurement primitives.
    from wyrd.generators.kenning.generators.kenning import Kenning

    k = Kenning()
    base_params: dict[str, Any] = {
        "culture": culture,
        "tags": tags or [],
        "harshness": harshness,
        "cohesion": cohesion,
    }

    samples_proportions: list[NameSample] = []
    samples_vector: list[NameSample] = []

    for i in range(count):
        seed = base_seed + i
        # Proportions path — bit-stable legacy
        try:
            r_prop = k.generate({**base_params, "scoring_mode": "proportions"}, seed=seed)
            samples_proportions.append(_sample_from_generation_result(r_prop))
        except Exception:  # noqa: BLE001 — per-sample isolation
            # Legacy path shouldn't normally raise, but a seed that
            # hits a bundle edge case shouldn't abort the whole run.
            pass

        # Vector path — opt-in, may raise on empty
        vector_params: dict[str, Any] = {**base_params, "scoring_mode": "vector"}
        if priors_path:
            vector_params["priors_path"] = priors_path
        try:
            r_vec = k.generate(vector_params, seed=seed)
            samples_vector.append(_sample_from_generation_result(r_vec))
        except Exception:  # noqa: BLE001
            # Vector raises on empty pick; that's expected when the
            # bundle / priors / register combo can't produce a non-
            # zero score for any slot. Skip this seed.
            pass

    return samples_proportions, samples_vector
