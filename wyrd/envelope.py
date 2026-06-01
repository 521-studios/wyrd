"""Uniform response envelope for every generator."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from wyrd.registry import GenerationResult


def envelope(
    *,
    generator: str,
    parameters: dict[str, Any],
    seed: int,
    results: list[GenerationResult],
    bundle_version: dict[str, str] | None = None,
) -> dict[str, Any]:
    # wyrd-dsl5: stamp the data/bundle version the generator ran against so
    # the SPA can echo it back on a defect report (reproducibility). Omitted
    # when the generator has no versioned data layer (runtime_version() is
    # None), keeping the shape uniform across generators.
    out: dict[str, Any] = {
        "generator": generator,
        "parameters": parameters,
        "seed": seed,
        "results": [
            {
                "result": r.result,
                "explanation": r.explanation,
                "components": [
                    asdict(c) if hasattr(c, "__dataclass_fields__") else c for c in r.components
                ],
                "canonical": r.canonical,
                "canonical_source": r.canonical_source,
                # wyrd-pr9g: forward the morpheme-pick breakdown so the
                # SPA can plumb the same continuity flow CLI exposes
                # via 'generate --json | rewind --from-json'. Generators
                # that don't surface morpheme structure leave the field
                # None; the API echoes None rather than omitting so the
                # response shape stays uniform across generators.
                "morphemes_by_word": r.morphemes_by_word,
            }
            for r in results
        ],
    }
    if bundle_version is not None:
        out["bundle_version"] = bundle_version
    return out
