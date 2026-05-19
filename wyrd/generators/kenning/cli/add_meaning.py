"""``wyrd kenning add-meaning`` — emit meanings.json entries from names on stdin."""

from __future__ import annotations

import json
import sys

import click


@click.command("add-meaning")
@click.option("--tag", "tags", multiple=True, help="Modifier tag (repeatable).")
def add_meaning(tags: tuple[str, ...]) -> None:
    """Read names from stdin, emit JSON entries suitable for meanings.json.

    Replaces Rando's `bin/generate_names`. One name per line; '/'-separated
    forms become alternate `modern_usage` entries.
    """
    if not tags:
        click.echo("At least one --tag is required.", err=True)
        sys.exit(1)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        forms = line.split("/")
        entry = {
            "meaning": [f"{n} (Name)" for n in forms],
            "modifier_tags": list(tags),
            "modifier_type": "Habitative",
            "words": [{"modern_usage": f"{n}-"} for n in forms],
        }
        click.echo(json.dumps(entry) + ",")


def add_to(parent: click.Group) -> None:
    """Register ``add-meaning`` on the top-level ``@cli`` group."""
    parent.add_command(add_meaning)
