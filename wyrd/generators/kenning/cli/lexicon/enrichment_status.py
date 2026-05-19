"""``wyrd kenning lexicon enrichment-status`` — per-column coverage of the L3 enrichment passes."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.enrichment import (
    enrichment_status,
    format_enrichment_status,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("enrichment-status")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_enrichment_status(db_path: Path) -> None:
    """Report L3 enrichment coverage on the current DB (wyrd-ilam).

    Read-only: shows per-column populated/total counts and method-
    version distributions where available. Useful BEFORE running
    `lexicon enrich --apply` to see what's already done, and AFTER to
    verify the passes populated what was expected.
    """
    with _readonly_lexicon(db_path) as conn:
        status = enrichment_status(conn)
    click.echo(format_enrichment_status(status))


def add_to(parent: click.Group) -> None:
    """Register ``enrichment-status`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_enrichment_status)
