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
) -> dict[str, Any]:
    return {
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
            }
            for r in results
        ],
    }
