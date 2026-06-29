"""``wyrd kenning validate-meanings`` — verify every modern_usage carries a hyphen."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.command("validate-meanings")
@click.argument(
    "meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
)
def validate_meanings(meanings: Path | None) -> None:
    """Check meanings.json: every modern_usage must contain a hyphen.

    Replaces Rando's `bin/validate_names`. Reads the path argument or stdin.
    Exits non-zero if any entries are malformed.
    """
    raw = meanings.read_text(encoding="utf-8") if meanings else sys.stdin.read()
    data = json.loads(raw)

    bad = []
    for subject in data:
        for word in subject["words"]:
            if "-" not in word["modern_usage"]:
                bad.append(word)

    for word in bad:
        click.echo(json.dumps(word), err=True)
    if bad:
        click.echo(f"{len(bad)} malformed entries.", err=True)
        sys.exit(1)
    click.echo(f"OK: {sum(len(s['words']) for s in data)} entries valid.", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``validate-meanings`` on the top-level ``@cli`` group."""
    parent.add_command(validate_meanings)
