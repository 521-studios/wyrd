"""``wyrd kenning lexicon lookup-attested-years`` — show per-etymon attestation-year coverage (D5-1)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, lookup_attested_years
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("lookup-attested-years")
@click.argument(
    "sources_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
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
    help="Populate etymon_text_match.attested_year. Without this, dry-run only.",
)
def lexicon_lookup_attested_years(
    sources_dir: Path,
    db_path: Path,
    apply_changes: bool,
) -> None:
    """Post-mining stage: scan source bodies for date citations near
    matched forms and populate etymon_text_match.attested_year (D5-1 /
    wyrd-3ux). Foundation for D5-2 era-cell sampling.

    LLM-free, idempotent, reversible. Per D21/D22 enrichment-only — no
    mining evidence is touched. Re-runs against unchanged data are
    no-ops; only rows where attested_year IS NULL are scanned.

    Reverse via:

        wyrd kenning lexicon clear-enrichment --stage=attested-years --apply
    """
    with LexiconDB(db_path) as db:
        result = lookup_attested_years(db, sources_dir, apply=apply_changes)

    etm = result["etymon_text_match"]
    te = result["toponym_etymology"]
    click.echo("lookup-attested-years:", err=True)
    click.echo(
        f"  etymon_text_match  scanned={etm['rows_scanned']:>5}  "
        f"candidates={etm['candidates']:>5}  rows_written={etm['rows_written']:>5}",
        err=True,
    )
    click.echo(
        f"  toponym_etymology  scanned={te['rows_scanned']:>5}  "
        f"candidates={te['candidates']:>5}  rows_written={te['rows_written']:>5}",
        err=True,
    )
    if result["sources_missing"]:
        click.echo(
            f"  warn: {result['sources_missing']} source_id(s) referenced in text-match "
            "rows have no .txt file under sources_dir; their rows were skipped.",
            err=True,
        )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write attested_year values)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``lookup-attested-years`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_lookup_attested_years)
