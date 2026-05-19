"""``wyrd kenning era-map`` — bulk-roll N names + render each at era stops."""

from __future__ import annotations

import json

import click

from wyrd.generators.kenning import CULTURES, KenningEraMap


@click.command("era-map")
@click.argument("culture", type=click.Choice(CULTURES))
@click.option("--count", type=int, default=6, help="How many names to roll.")
@click.option("--seed", type=int, default=0, help="RNG seed; successive names use seed+i.")
@click.option("--as-json", is_flag=True, default=False, help="Emit JSON.")
def era_map(culture: str, count: int, seed: int, as_json: bool) -> None:
    """wyrd-381: bulk-generate N names AND render each at era stops.

    The Domesday-vs-modern map demo. Composes the existing Kenning
    name-generator with the wyrd-skm era-reflex rewinder: same roll,
    same morpheme stack, multiple eras of paper.

    Default ladder is the three English stops (oe-late / me / modern).
    Names whose morphemes haven't been mined for era_reflexes render
    uniformly across all eras (the bundle-only renderer falls back
    to the modern usage); that's a coverage limit, not a bug.
    """
    gen = KenningEraMap()
    results = gen.generate_all({"culture": culture, "count": count}, seed)

    if as_json:
        out = []
        for r in results:
            comp = r.components[0] if r.components else {}
            out.append(
                {
                    "name": comp.get("name"),
                    "era_cells": comp.get("era_cells", []),
                    "rendered_modern": r.result,
                }
            )
        click.echo(json.dumps(out, indent=2))
        return

    click.echo(f"culture={culture} count={len(results)} (requested {count})", err=True)
    for r in results:
        comp = r.components[0] if r.components else {}
        name = comp.get("name", "?")
        cells = comp.get("era_cells", [])
        click.echo(f"\n{name}")
        for cell in cells:
            label = f"{cell.get('family', '?')}/{cell.get('era', '?')}"
            click.echo(f"  {label:<20} {cell.get('rendered', '?')}")


def add_to(parent: click.Group) -> None:
    """Register ``era-map`` on the top-level ``@cli`` group."""
    parent.add_command(era_map)
