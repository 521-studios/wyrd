"""``wyrd kenning lexicon browse decomposition`` — matcher decompositions for one toponym."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.browse import (
    fetch_decompositions,
    fetch_toponyms_matching,
    format_decompositions,
    format_toponym_list,
    parse_toponym_ref,
)
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("decomposition")
@click.argument("query")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_browse_decomposition(query: str, db_path: Path) -> None:
    """Show matcher decompositions for one toponym.

    QUERY is 'name' or 'name@region'. Surfaces the canonical pick +
    every alternative the matcher emitted.
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
        decompositions = fetch_decompositions(conn, matches[0]["id"])
        match = matches[0]
    click.echo(format_decompositions(match["modern_name"], match["region"], decompositions))


def add_to(parent: click.Group) -> None:
    """Register ``decomposition`` on the @lexicon_browse group."""
    parent.add_command(lexicon_browse_decomposition)
