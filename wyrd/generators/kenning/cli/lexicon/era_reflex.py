"""``wyrd kenning lexicon era-reflex`` — show etymon era reflexes (D33 wyrd-skm picker)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.era.cells import era_cell_for_input, language_family
from wyrd.generators.kenning.lexicon import LexiconDB, etymon_era_reflexes
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("era-reflex")
@click.argument("etymon_id", type=int)
@click.option(
    "--era",
    "era_input",
    required=True,
    help=(
        "Target era: a year (1086), a cell label (oe-late), or a "
        "family/label pair (english/me). Resolved against the "
        "etymon's source-language family."
    ),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_era_reflex(etymon_id: int, era_input: str, db_path: Path) -> None:
    """wyrd-skm Phase 3.0b: print cognate-cluster mates of ETYMON_ID
    that match the target era. Used by wyrd-rni / wyrd-381 demos to
    render etymons across historical strata.

    One row per matching cluster mate on stdout, tab-separated:
    ``<reflex_id>\\t<canonical_form>\\t<language>``. The metadata
    header (etymon canonical / language / target cell) goes to
    stderr.

    Empty output when the etymon has no cognate_id (~76% of OE
    toponym etymons today — cluster_cognates coverage gap) or when
    the target cell has no canonical language tag (most Norse +
    Latin late cells; see ``CANONICAL_LANGUAGE_FOR_CELL`` in era/cells.py).
    Extending coverage means adding entries to that dict; the picker
    surfaces any new entry the next call.
    """
    with LexiconDB(db_path) as db:
        row = db.conn.execute(
            "SELECT canonical_form, language FROM etymon WHERE id = ?",
            (etymon_id,),
        ).fetchone()
        if row is None:
            click.echo(f"no etymon with id={etymon_id}", err=True)
            raise click.exceptions.Exit(1)

        family = language_family(row["language"])
        if family is None:
            click.echo(
                f"etymon {etymon_id} ({row['canonical_form']!r}, "
                f"language={row['language']!r}) has no era family — "
                "proto-languages and untracked classical languages "
                "don't define era cells",
                err=True,
            )
            raise click.exceptions.Exit(1)

        # Resolve --era directly to (family, cell) — the era-reflex
        # picker is keyed on cell labels, so round-tripping through
        # year ranges (resolve_era_input → era_cell) loses information
        # for open-low cells like 'oe-early' where start=None.
        try:
            family, cell = era_cell_for_input(era_input, default_family=family)
        except ValueError as exc:
            click.echo(str(exc), err=True)
            raise click.exceptions.Exit(1) from exc

        reflexes = etymon_era_reflexes(db, etymon_id, target_family_cell=(family, cell))

    click.echo(
        f"# era-reflex etymon={etymon_id} "
        f"({row['canonical_form']!r}, {row['language']}) "
        f"target={family}/{cell}",
        err=True,
    )
    if not reflexes:
        click.echo("(no cluster mates at this era)", err=True)
        return
    for reflex in reflexes:
        click.echo(f"{reflex.etymon_id}\t{reflex.form}\t{reflex.language}")


def add_to(parent: click.Group) -> None:
    """Register ``era-reflex`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_era_reflex)
