"""``wyrd kenning lexicon backfill-toponym-country`` — derive country from region.

One-shot data fix for the empirical-priors pipeline. The parser ingest
path historically populated ``toponym.region`` but left
``toponym.country`` NULL, which made ``mine-empirical-baselines`` skip
every row on ``country_unknown`` and zero out the English baseline.

This CLI runs the in-place region → country derivation
(``backfill_toponym_country``) against the L3 lexicon, then re-runs
the priors miner so the next ``export-runtime-db`` picks up populated
priors.

Idempotent: re-running against an already-backfilled DB does nothing.
"""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.backfill_country import backfill_toponym_country
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("backfill-toponym-country")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_backfill_toponym_country(db_path: Path) -> None:
    """Derive ``toponym.country`` from ``toponym.region`` for rows where
    country is currently NULL. Merges with any existing same-name +
    same-country row when present (dependent ``toponym_etymology`` FKs
    move to the merged row)."""
    with LexiconDB(db_path) as db:
        result = backfill_toponym_country(db)
    click.echo("backfill-toponym-country:", err=True)
    click.echo(
        f"  inspected={result.inspected}  "
        f"updated={result.updated}  "
        f"merged={result.merged}  "
        f"region_unknown={result.region_unknown}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``backfill-toponym-country`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_backfill_toponym_country)
