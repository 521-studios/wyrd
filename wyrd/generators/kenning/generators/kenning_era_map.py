"""The `kenning-era-map` Generator — bulk-roll names + render across eras (D34 / wyrd-381)."""

from __future__ import annotations

from typing import Any

from wyrd.generators.kenning import (
    _LEGEND,
    CULTURES,
    available_tags,
)
from wyrd.registry import GenerationResult, Generator

_DEFAULT_COUNT = 6
_MAX_COUNT = 50


def _region_count(params: dict[str, Any]) -> int:
    """The requested region size (name count) for the era map: schema 1-50,
    default 6.

    A non-coercible value (list, dict, non-numeric string) or an out-of-range one
    raises ``ValueError`` — which the dispatcher maps to 400 bad_params — rather
    than a bare ``int()`` raising ``TypeError`` on a list/dict, which escapes
    uncaught as a 500. A blank/missing count defaults to 6; a CLI/POST int and a
    GET numeric-string coerce identically. (The API dispatcher used to POP
    ``count`` for multi_result generators, pinning this to 6 and ignoring the
    user — the CLI calls generate_all directly and always honored it.)"""
    raw = params.get("count")
    try:
        count = _DEFAULT_COUNT if raw is None or raw == "" else int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"count must be an integer, got {raw!r}") from e
    if not 1 <= count <= _MAX_COUNT:
        raise ValueError(f"count must be between 1 and {_MAX_COUNT}, got {count}")
    return count


class KenningEraMap(Generator):
    """wyrd-381 — bulk-generate N names AND render each at multiple
    historical strata in one shot.

    The Domesday-vs-modern map demo: roll a region of N invented
    toponyms, then surface the SAME underlying names rendered at
    e.g. {oe-late, me, modern}. Same Kenning roll, three eras of
    paper. The killer GM-handout pattern is the "ancient map +
    modern map" pair — both are period-consistent because they
    share the same generated morpheme stack.

    Implementation: composes the existing ``Kenning`` (name
    generator) with ``KenningRewind`` (era-stop renderer) — each
    result carries one generated name plus its era-stop table as
    components, so the SPA / CLI can render the table directly.
    """

    name = "kenning-era-map"
    display_name = "Kenning — Stratified Era Map"
    description = (
        "Bulk-generate N invented toponyms AND render each at "
        "multiple historical strata in one shot. Pairs naturally "
        "with a Domesday-vs-modern handout: same map, three eras "
        "of paper. Each result is one generated name with its "
        "era-stop table attached."
    )
    legend = _LEGEND
    multi_result = True

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "culture": {
                    "type": "string",
                    "enum": CULTURES,
                    "default": "english",
                    "description": (
                        "Linguistic culture. Forwarded to the underlying Kenning generator."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "enum": available_tags()},
                    "default": [],
                    "description": "Optional tag filters (forwarded to Kenning).",
                },
                "count": {
                    "type": "integer",
                    "default": 6,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "How many names to generate (region size).",
                },
            },
            "required": ["culture"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        # multi_result generators surface one-result-per-name; the
        # single-result fallback returns the first.
        results = self.generate_all(params, seed)
        if not results:
            raise ValueError("kenning-era-map produced no results")
        return results[0]

    def generate_all(self, params: dict[str, Any], seed: int) -> list[GenerationResult]:
        culture = params.get("culture") or "english"
        tags = params.get("tags") or []
        count = _region_count(params)

        # Deferred imports — Kenning + KenningRewind live in sibling
        # modules under generators/, and importing them at module-load time
        # would create a partial-init cycle through the generators/ package's
        # alphabetical import order. Resolving at method-call time sidesteps
        # the issue entirely.
        from wyrd.generators.kenning.generators.kenning import Kenning
        from wyrd.generators.kenning.generators.kenning_rewind import KenningRewind

        # 1. Generate N names with the existing Kenning generator.
        # Kenning.generate returns ONE name per call (the framework
        # normally loops via the dispatcher); since we're inside a
        # multi_result=True generator we drive the loop directly,
        # offsetting the seed per roll so successive names are
        # distinct.
        kenning = Kenning()
        kenning_params = {"culture": culture, "tags": tags}
        names: list[str] = []
        seen: set[str] = set()
        for i in range(count):
            roll = kenning.generate(kenning_params, seed + i)
            name = (roll.result or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        if not names:
            return []

        # 2. For each name, run KenningRewind to capture the
        # era-stop table. Each generated-name → one
        # GenerationResult whose components hold the era table.
        rewind = KenningRewind()
        outputs: list[GenerationResult] = []
        for name in names:
            try:
                era_results = rewind.generate_all({"name": name}, seed)
            except Exception:
                # A name that doesn't decompose cleanly still
                # belongs in the map — fall back to a one-cell
                # 'modern' result so the table stays uniform.
                era_results = []
            era_cells: list[dict[str, Any]] = []
            for er in era_results:
                # KenningRewind components are a single dict per
                # era stop; flatten into the era-map's table.
                if er.components:
                    era_cells.append(er.components[0])
            if not era_cells:
                era_cells = [
                    {
                        "era": "modern",
                        "family": "english",
                        "rendered": name,
                        "morphemes": [],
                        "unaccounted": [name],
                    }
                ]
            modern_form = next(
                (c["rendered"] for c in era_cells if c.get("era") == "modern"),
                name,
            )
            outputs.append(
                GenerationResult(
                    result=modern_form,
                    explanation=" → ".join(
                        f"{c.get('era', '?')}: {c.get('rendered', '?')}" for c in era_cells
                    ),
                    components=[
                        {
                            "name": name,
                            "era_cells": era_cells,
                        }
                    ],
                )
            )
        return outputs
