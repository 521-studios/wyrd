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
    result: GenerationResult,
) -> dict[str, Any]:
    return {
        "generator": generator,
        "parameters": parameters,
        "seed": seed,
        "result": result.result,
        "explanation": result.explanation,
        "components": [
            asdict(c) if hasattr(c, "__dataclass_fields__") else c for c in result.components
        ],
    }
