"""``wyrd kenning lexicon recompute-reflex-productivity`` — populate
reflex.productivity from corpus observations (wyrd-14p).

Per-reflex productivity = count of distinct toponyms whose breakdown
contains a matching (etymon, position) element. The reflex layer is
populated once by ``seed_from_meanings`` with productivity=0 across
the board; this command derives the real counts from
``toponym_etymology_element`` joins.

Idempotent: re-running on an unchanged DB writes byte-identical rows.
Safe to run as part of a regular re-index after corpus updates.
"""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.reflex_productivity import (
    recompute_reflex_productivity,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("recompute-reflex-productivity")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_recompute_reflex_productivity(db_path: Path) -> None:
    """Recompute reflex.productivity counts from corpus observations.

    Reads ``toponym_etymology_element`` joined against ``reflex_etymon``
    + ``reflex`` (matching on etymon + the (ordinal, element_count)-
    derived position), counts distinct toponyms per reflex, and writes
    the result back to ``reflex.productivity``.

    Atomic + idempotent. Safe to re-run after corpus updates — the
    full table is zeroed before the recompute so reflexes that lost
    all corpus support read 0 again rather than carrying a stale count.
    """
    db = LexiconDB(db_path)
    try:
        stats = recompute_reflex_productivity(db)
        click.echo("Reflex productivity recompute:", err=True)
        click.echo(f"  total reflexes:   {stats['total_reflexes']:,}", err=True)
        click.echo(f"  updated (>0):     {stats['updated']:,}", err=True)
        click.echo(f"  max productivity: {stats['max_productivity']:,}", err=True)
        click.echo(f"  sum productivity: {stats['sum_productivity']:,}", err=True)
    finally:
        db.close()


def add_to(parent: click.Group) -> None:
    """Register ``recompute-reflex-productivity`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_recompute_reflex_productivity)
