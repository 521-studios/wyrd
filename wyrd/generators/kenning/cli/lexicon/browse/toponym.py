"""``wyrd kenning lexicon browse toponym`` — one toponym: attestations + etymologies."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.browse import (
    fetch_toponym_detail,
    fetch_toponyms_matching,
    format_toponym,
    format_toponym_list,
    parse_toponym_ref,
)
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("toponym")
@click.argument("query")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_browse_toponym(query: str, db_path: Path) -> None:
    """Show one toponym: attestations + etymologies + decompositions.

    QUERY is either a bare modern name ('Cotton') or 'name@region'
    ('Cotton@Norfolk'). A bare name that matches multiple toponyms
    prints the disambiguation list instead.
    """
    name, region = parse_toponym_ref(query)
    with _readonly_lexicon(db_path) as conn:
        matches = fetch_toponyms_matching(conn, name, region)
        if not matches:
            click.echo(f"No toponym found for query={query!r}", err=True)
            raise SystemExit(1)
        if len(matches) > 1:
            click.echo(format_toponym_list(matches))
            return
        data = fetch_toponym_detail(conn, matches[0]["id"])
    click.echo(format_toponym(data))


def add_to(parent: click.Group) -> None:
    """Register ``toponym`` on the @lexicon_browse group."""
    parent.add_command(lexicon_browse_toponym)
