"""``wyrd kenning lexicon synsets show`` — show meaning_synsets an etymon belongs to."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, get_meaning_synsets_for_etymon
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("show")
@click.argument("etymon_id", type=int)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_synsets_show(etymon_id: int, db_path: Path) -> None:
    """Show the meaning_synsets an etymon belongs to."""
    with LexiconDB(db_path) as db:
        synsets = get_meaning_synsets_for_etymon(db, etymon_id)
    if not synsets:
        click.echo(f"(etymon {etymon_id} has no meaning_synset memberships)")
        return
    for s in synsets:
        click.echo(f"  {s['canonical_label']:<32} ({s['fit']})")


def add_to(parent: click.Group) -> None:
    """Register ``show`` on the @lexicon_synsets group."""
    parent.add_command(lexicon_synsets_show)
