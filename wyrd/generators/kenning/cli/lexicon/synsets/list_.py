"""``wyrd kenning lexicon synsets list`` — list meaning_synset rows."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, list_meaning_synsets
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("list")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--with-counts",
    "with_counts",
    is_flag=True,
    default=False,
    help="Include member-count column (etymons assigned per synset).",
)
def lexicon_synsets_list(db_path: Path, with_counts: bool) -> None:
    """List all meaning_synset rows, sorted by canonical_label."""
    with LexiconDB(db_path) as db:
        synsets = list_meaning_synsets(db, with_member_counts=with_counts)
    if not synsets:
        click.echo("(no meaning_synset rows — run `wyrd kenning lexicon synsets seed`)")
        return
    for s in synsets:
        if with_counts:
            click.echo(f"  {s['canonical_label']:<32} ({s['member_count']:>4} members)")
        else:
            click.echo(f"  {s['canonical_label']}")


def add_to(parent: click.Group) -> None:
    """Register ``list`` on the @lexicon_synsets group."""
    parent.add_command(lexicon_synsets_list)
