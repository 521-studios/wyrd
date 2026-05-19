"""``wyrd kenning lexicon browse etymon`` — one etymon: glosses, tags, citations."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.bundle.browse import fetch_etymon, format_etymon
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("etymon")
@click.argument("ref")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_browse_etymon(ref: str, db_path: Path) -> None:
    """Show one etymon: glosses, tags, citations, descent, lemma family.

    REF is the cross-file etymon ref, e.g. 'old-english:cot'.
    """
    with _readonly_lexicon(db_path) as conn:
        data = fetch_etymon(conn, ref)

    if data is None:
        click.echo(f"No etymon found for ref={ref!r}", err=True)
        raise SystemExit(1)
    click.echo(format_etymon(data))


def add_to(parent: click.Group) -> None:
    """Register ``etymon`` on the @lexicon_browse group."""
    parent.add_command(lexicon_browse_etymon)
