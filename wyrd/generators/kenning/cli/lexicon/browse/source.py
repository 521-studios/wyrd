"""``wyrd kenning lexicon browse source`` — one source: bibliographic record + counts."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.bundle.browse import fetch_source, format_source
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("source")
@click.argument("source_id")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--list-toponyms",
    is_flag=True,
    default=False,
    help="Also print the full list of toponyms covered by this source.",
)
def lexicon_browse_source(source_id: str, db_path: Path, list_toponyms: bool) -> None:
    """Show one source: bibliographic record + contribution counts.

    SOURCE_ID is the source row's id (e.g. 'skeat_1901_cambridgeshire').
    """
    with _readonly_lexicon(db_path) as conn:
        data = fetch_source(conn, source_id)

    if data is None:
        click.echo(f"No source found for id={source_id!r}", err=True)
        raise SystemExit(1)
    click.echo(format_source(data, list_toponyms=list_toponyms))


def add_to(parent: click.Group) -> None:
    """Register ``source`` on the @lexicon_browse group."""
    parent.add_command(lexicon_browse_source)
