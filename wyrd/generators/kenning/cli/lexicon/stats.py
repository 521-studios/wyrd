"""``wyrd kenning lexicon stats`` — terse per-table row counts for the lexicon DB."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("stats")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_stats(db_path: Path) -> None:
    """Show row counts per table in the lexicon DB."""
    with LexiconDB(db_path) as db:
        stats = db.stats()
    for table, n in stats.items():
        click.echo(f"{table:<30} {n:>8}")


def add_to(parent: click.Group) -> None:
    """Register ``stats`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_stats)
