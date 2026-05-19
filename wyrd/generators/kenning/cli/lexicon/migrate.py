"""``wyrd kenning lexicon migrate`` — run pending schema migrations against the lexicon DB."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, migrate_schema
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("migrate")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_migrate(db_path: Path) -> None:
    """Apply pending schema migrations to an existing lexicon DB.

    Idempotent. Currently brings older DBs up to the lemma_id /
    inflection schema and rebuilds the etymon_consensus view.
    """
    with LexiconDB(db_path) as db:
        applied = migrate_schema(db)
    click.echo("Migrations:", err=True)
    for name, did in applied.items():
        click.echo(f"  {name:32} {'applied' if did else 'no-op'}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``migrate`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_migrate)
