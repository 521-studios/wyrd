"""The `kenning-render` Generator — alt-script transliteration (D34 / wyrd-y10)."""

from __future__ import annotations

from typing import Any

from wyrd.generators.kenning import (
    _LEGEND,
    _required_name,
)
from wyrd.registry import GenerationResult, Generator


class KenningRender(Generator):
    """wyrd-y10 — render an English (or English-rendered) name in
    an alternate phonemic script.

    A phonemic script written for English IS English, just visually
    disguised. Killer GM-handout demo: produce signage / inscriptions
    that look foreign but are decodable for committed players.

    Initial target: Shavian (~48 glyphs, plane-1 codepoints
    U+10450-U+1047F). Future scripts (Tengwar / Cirth / Elder Futhark
    / Ogham) drop in via additional ``transliterate`` dispatch arms
    in ``wyrd.generators.kenning.runtime.scripts``.
    """

    name = "kenning-render"
    display_name = "Kenning — Render in an Alternate Script"
    description = (
        "Render an English (or English-rendered) name in an alternate "
        "phonemic script — Shavian today, Tengwar / Cirth / Elder "
        "Futhark on follow-up. Atmospheric for tabletop handouts; the "
        "result is still English, just visually disguised."
    )
    legend = _LEGEND
    multi_result = False

    def input_schema(self) -> dict[str, Any]:
        from wyrd.generators.kenning.runtime.scripts import SUPPORTED_SCRIPTS

        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "English-rendered name to transliterate.",
                },
                "script": {
                    "type": "string",
                    "enum": list(SUPPORTED_SCRIPTS),
                    "default": "shavian",
                    "description": "Target script. Currently only Shavian.",
                },
            },
            "required": ["name"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        from wyrd.generators.kenning.runtime.scripts import transliterate

        text = _required_name(params)
        script = params.get("script") or "shavian"
        rendered = transliterate(text, script)
        return GenerationResult(
            result=rendered,
            explanation=f"{text} → {rendered} ({script})",
            components=[
                {
                    "input": text,
                    "rendered": rendered,
                    "script": script,
                }
            ],
        )
