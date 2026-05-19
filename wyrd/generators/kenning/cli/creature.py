"""``wyrd kenning creature`` — surface fantasy-creature etymology (wyrd-ami)."""

from __future__ import annotations

import click


@click.command("creature")
@click.argument("name")
def creature(name: str) -> None:
    """wyrd-vz7f: surface fantasy-creature etymology from the
    wyrd-ami pipeline.

    Looks up the creature name in the bundled ``fantasy_morphemes``
    map (populated by ``lexicon export-meanings`` from the
    ``fantasy_morpheme`` table). Prints the linked attested ancestor
    (language, canonical form, glosses, citation) and any era
    reflexes mined for that etymon's cluster.

    Unknown names return a polite 'not found' rather than erroring,
    so chaining ``wyrd kenning creature <random>`` against a list
    is safe.
    """
    from wyrd.generators.kenning import KenningCreature

    gen = KenningCreature()
    result = gen.generate({"name": name}, seed=0)
    click.echo(result.result)
    if result.explanation:
        click.echo(f"  {result.explanation}")


def add_to(parent: click.Group) -> None:
    """Register ``creature`` on the top-level ``@cli`` group."""
    parent.add_command(creature)
