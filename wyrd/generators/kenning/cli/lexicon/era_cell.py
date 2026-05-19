"""``wyrd kenning lexicon era-cell`` — show the canonical era cell for an input year/label."""

from __future__ import annotations

import click

from wyrd.generators.kenning.era import era_cell, language_family


@click.command("era-cell")
@click.argument("language")
@click.argument("year", type=int)
def lexicon_era_cell(language: str, year: int) -> None:
    """Resolve a (language, year) pair to its D5-2 era-cell label.

    Prints ``family/label`` (e.g. ``english/oe-late``) on stdout. Exits
    non-zero with a stderr message if the language has no era family
    (proto-languages, untracked classical languages) or if the year
    falls outside any defined cell. Useful for ad-hoc cell lookup
    when sanity-checking attestation data.
    """
    family = language_family(language)
    if family is None:
        click.echo(
            f"language {language!r} has no era family defined "
            "(proto-languages and untracked classical languages "
            "intentionally don't get cell labels)",
            err=True,
        )
        raise click.exceptions.Exit(1)
    label = era_cell(language, year)
    if label is None:
        click.echo(
            f"year {year} is outside the defined cells for family {family!r}",
            err=True,
        )
        raise click.exceptions.Exit(1)
    click.echo(f"{family}/{label}")


def add_to(parent: click.Group) -> None:
    """Register ``era-cell`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_era_cell)
