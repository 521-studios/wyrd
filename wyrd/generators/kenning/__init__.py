"""Kenning — town-name generator. Compounds Old English / Norse / Celtic morphemes."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

from wyrd.generators.kenning.meaning import load_meanings
from wyrd.generators.kenning.proportions import load_proportions
from wyrd.registry import GenerationResult, Generator, register

CULTURES = ["english", "scottish", "welsh", "irish"]

# Tags that are filtering primitives, not meaningful selections to expose.
_INTERNAL_TAGS = {"male name", "female name", "saint"}


def _data_path(filename: str):
    return resources.files("wyrd.generators.kenning.data").joinpath(filename)


@lru_cache(maxsize=1)
def _load_meanings():
    with _data_path("meanings.json").open() as f:
        return load_meanings(json.load(f))


@lru_cache(maxsize=4)
def _load_culture(culture: str):
    if culture not in CULTURES:
        raise ValueError(f"unknown culture: {culture}; expected one of {CULTURES}")
    meaning_db, tag_db = _load_meanings()
    with _data_path(f"{culture}_proportions.json").open() as f:
        proportions = json.load(f)
    return load_proportions(proportions, meaning_db, tag_db), tag_db


def available_tags() -> list[str]:
    """User-visible tags from the meaning DB (excludes internal filtering tags)."""
    _, tag_db = _load_meanings()
    return sorted(t for t in tag_db if t not in _INTERNAL_TAGS)


class Kenning(Generator):
    name = "kenning"
    display_name = "Kenning — Town Names"
    description = (
        "Generates British Isles–style town names by composing Old English, Old Norse, "
        "Old French, and Celtic morphemes. Pick a culture; optionally filter morphemes "
        "by tag (e.g. 'tree', 'water', 'religion')."
    )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "culture": {
                    "type": "string",
                    "enum": CULTURES,
                    "default": "english",
                    "description": "Linguistic culture to draw morphemes and structures from.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "enum": available_tags()},
                    "default": [],
                    "description": (
                        "Optional tag filters. Each tag biases the name toward morphemes "
                        "with that meaning category."
                    ),
                },
                "count": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "How many names to generate (1–10).",
                },
                "seed": {
                    "type": "integer",
                    "description": "Optional 64-bit seed for reproducible output.",
                },
            },
            "required": ["culture"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        from wyrd.seed import rng_for

        culture = params.get("culture", "english")
        raw_tags = params.get("tags", []) or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        tags = tuple(raw_tags)

        name_gen, _ = _load_culture(culture)
        rng = rng_for(seed)
        new_name = name_gen.select(rng, *tags)
        return GenerationResult(
            result=str(new_name),
            explanation=new_name.description(),
            components=new_name.components(),
        )


register(Kenning())
