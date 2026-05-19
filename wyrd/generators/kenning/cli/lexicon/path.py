"""``wyrd kenning lexicon path`` — print the resolved authoring-DB path."""

from __future__ import annotations

import click

from wyrd.generators.kenning.paths import default_lexicon_path


@click.command("path")
def lexicon_path() -> None:
    """Print the resolved lexicon DB path and exit.

    Resolution order:

    \b
    1. WYRD_LEXICON_DB env var
    2. ~/.wyrd/lexicon.db (default)

    Triggers the one-time legacy → ~/.wyrd migration if a pre-wyrd-366
    DB still lives in the repo's data/ dir. Useful for confirming what
    DB a CLI run would hit.
    """
    click.echo(default_lexicon_path())


def add_to(parent: click.Group) -> None:
    """Register ``path`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_path)
