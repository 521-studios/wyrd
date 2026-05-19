"""``wyrd kenning lexicon mine-empirical-baselines`` — D36.9 priors-artifact regeneration."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.empirical_priors import mine_empirical_baselines
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("mine-empirical-baselines")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help=(
        "Replace empirical_priors_native + empirical_priors_loan with "
        "freshly-extracted counts. Without this, dry-run only."
    ),
)
@click.option(
    "--progress-every",
    type=int,
    default=5000,
    show_default=True,
    help="Emit a stderr progress line every N rows scanned. 0 = silent.",
)
def lexicon_mine_empirical_baselines(
    db_path: Path, apply_changes: bool, progress_every: int
) -> None:
    """wyrd-ecjp.2: extract empirical-baseline priors per (culture x
    position x tag x era) from toponym_etymology + elements + tags.

    Two parallel views (D36.4 Option B):
      * empirical_priors_native — all elements in <culture>-recipient
        toponyms.
      * empirical_priors_loan — same observations partitioned by donor
        language; scenario packs read this for their template's
        loan-relationship baseline.

    LLM-free, deterministic, idempotent. Replace-not-merge each apply
    run — stale cells from removed observations don't persist.

    Pair with ``lexicon dump-empirical-priors --output ...`` to emit a
    review-friendly JSON sidecar after writing the tables.
    """
    with LexiconDB(db_path) as db:
        result = mine_empirical_baselines(db, apply=apply_changes, progress_every=progress_every)
    click.echo("mine-empirical-baselines:", err=True)
    click.echo(
        "  scanned={scanned:>6}  emitted={emitted:>6}  "
        "native_cells={nc:>5}  loan_cells={lc:>5}".format(
            scanned=result["rows_scanned"],
            emitted=result["rows_emitted"],
            nc=result["native_cells_written"],
            lc=result["loan_cells_written"],
        ),
        err=True,
    )
    click.echo(
        "  skipped: country_unknown={cu:>5}  country_unmapped={cm:>5}  "
        "year_unknown={yu:>5}  year_out_of_range={yo:>5}  "
        "no_tag={nt:>5}".format(
            cu=result["skipped_country_unknown"],
            cm=result["skipped_country_unmapped"],
            yu=result["skipped_year_unknown"],
            yo=result["skipped_year_out_of_range"],
            nt=result["skipped_no_tag"],
        ),
        err=True,
    )
    if not apply_changes:
        click.echo(
            "(dry-run; pass --apply to write empirical_priors_* rows)",
            err=True,
        )


def add_to(parent: click.Group) -> None:
    """Register ``mine-empirical-baselines`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_empirical_baselines)
