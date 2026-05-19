"""``wyrd kenning lexicon era-coverage`` — aggregate cross-era coverage report."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("era-coverage")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_era_coverage(db_path: Path) -> None:
    """Aggregate cross-era coverage across the toponym corpus (wyrd-ub76).

    Reports per-era toponym counts, the histogram of how many eras
    each toponym is attested in, and the percent with >=3 dated eras
    — the headline 'is the cross-era story landing' number.
    """
    from wyrd.generators.kenning.era.timeline import (
        fetch_era_coverage,
        format_era_coverage,
    )

    with _readonly_lexicon(db_path) as conn:
        report = fetch_era_coverage(conn)
    click.echo(format_era_coverage(report))


def add_to(parent: click.Group) -> None:
    """Register ``era-coverage`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_era_coverage)
