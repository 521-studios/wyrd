"""``wyrd kenning lexicon synsets seed`` — seed meaning_synset from the bundled catalog."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, seed_meaning_synsets
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("seed")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_synsets_seed(db_path: Path) -> None:
    """Idempotently populate meaning_synset from the bundled catalog
    (data/seed/meaning_synsets.json). Reports inserts / updates / unchanged."""
    with LexiconDB(db_path) as db:
        result = seed_meaning_synsets(db)
    click.echo(
        f"meaning_synset seed: inserted={result['inserted']} "
        f"updated={result['updated']} unchanged={result['unchanged']}"
    )


def add_to(parent: click.Group) -> None:
    """Register ``seed`` on the @lexicon_synsets group."""
    parent.add_command(lexicon_synsets_seed)
