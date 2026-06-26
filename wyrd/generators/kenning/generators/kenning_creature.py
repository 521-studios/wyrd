"""The `kenning-creature` Generator — fantasy creature etymology (wyrd-ami / D30)."""

from __future__ import annotations

from typing import Any

from wyrd.generators.kenning import (
    _LEGEND,
    _format_creature_explanation,
    _load_fantasy_morphemes,
    _required_name,
)
from wyrd.registry import GenerationResult, Generator


class KenningCreature(Generator):
    """wyrd-vz7f: surface fantasy-creature etymology from the wyrd-ami
    pipeline data. Input is a creature name (Harpy, Daemon, Dwarf,
    Drake, ...); output is the etymology line + descent context + era
    reflexes when available.

    The wyrd-ami pipeline (D30) mines fantasy_morpheme rows linking
    creature input_names to real etymons in the lexicon; this is the
    runtime tap. ``_load_fantasy_morphemes`` reads the bundle's
    ``fantasy_morphemes`` field (populated by
    ``lexicon.collect_fantasy_morphemes`` at export time). 234 usable
    creatures shipped at wyrd-vz7f time; the catalog grows as the
    wyrd-ami pipeline ingests more pfsrd2-monsters input.
    """

    name = "kenning-creature"
    display_name = "Kenning — Creature etymology"
    description = (
        "Look up the historical etymology of a fantasy / mythological "
        "creature name. Returns the linked attested ancestor (language, "
        "canonical form, glosses) and any era reflexes mined for that "
        "etymon's cluster."
    )
    details = (
        "<p>"
        "Type a creature name like <em>Harpy</em>, <em>Daemon</em>, or "
        "<em>Drake</em> and Kenning surfaces the historical etymon the "
        "wyrd-ami research pipeline linked it to: the source language, "
        "the attested ancestral form, and any descendant forms across "
        "later eras."
        "</p>"
        "<p>"
        "Unknown names (modern coinages, non-corpus mythologies, or "
        "names the LLM-research pipeline couldn't resolve) return a "
        "polite 'no etymology found' result rather than an error."
        "</p>"
    )
    legend = _LEGEND
    multi_result = False

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Creature name (e.g. 'Harpy', 'Daemon').",
                },
            },
            "required": ["name"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        text = _required_name(params)
        creatures = _load_fantasy_morphemes()
        entry = creatures.get(text.lower())
        if entry is None:
            return GenerationResult(
                result=text,
                explanation=(
                    f"No etymology found for {text!r} in the wyrd-ami "
                    "fantasy-morpheme corpus. The pipeline may classify "
                    "it as a modern coinage, an outside-language-family "
                    "name, or simply not yet mined."
                ),
            )
        return GenerationResult(
            result=entry["input_name"],
            explanation=_format_creature_explanation(entry),
        )
