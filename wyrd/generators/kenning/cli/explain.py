"""``wyrd kenning explain`` — decompose a name into every matching morpheme reading."""

from __future__ import annotations

import click

from wyrd.generators.kenning import KenningExplain


@click.command("explain")
@click.argument("name")
def explain(name: str) -> None:
    """Decompose a name into morphemes; print every matching reading."""
    explainer = KenningExplain()
    results = explainer.generate_all({"name": name}, 0)
    for r in results:
        click.echo(r.explanation)


def add_to(parent: click.Group) -> None:
    """Register ``explain`` on the top-level ``@cli`` group."""
    parent.add_command(explain)
