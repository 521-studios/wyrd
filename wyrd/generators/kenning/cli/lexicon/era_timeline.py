"""``wyrd kenning lexicon era-timeline`` — bucket one toponym's attestations by era."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("era-timeline")
@click.argument("query")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_era_timeline(query: str, db_path: Path) -> None:
    """Show one toponym's attestations bucketed by era (wyrd-ub76).

    QUERY is either a bare modern name ('Acton') or 'name@region'
    ('Acton@Cheshire'). For ambiguous bare names the first region in
    ASCII order is shown — disambiguate with @region.

    Era buckets: Anglo-Saxon (<1100), Middle English (1100-1500),
    Early Modern (1500-1800), Modern (>=1800), Undated.
    """
    from wyrd.generators.kenning.era_timeline import (
        fetch_era_timeline,
        format_era_timeline,
    )

    with _readonly_lexicon(db_path) as conn:
        timeline = fetch_era_timeline(conn, query)

    if timeline is None:
        click.echo(f"No toponym found for query={query!r}", err=True)
        raise SystemExit(1)
    click.echo(format_era_timeline(timeline))


def add_to(parent: click.Group) -> None:
    """Register ``era-timeline`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_era_timeline)
