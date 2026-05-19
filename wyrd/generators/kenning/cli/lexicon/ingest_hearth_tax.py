"""``wyrd kenning lexicon ingest-hearth-tax`` — ingest Hearth Tax operator CSV (wyrd-myh1)."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("ingest-hearth-tax")
@click.argument(
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/hearth_tax_1660s.jsonl"),
    show_default=True,
    help="Where to write the JSONL events.",
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write the JSONL. Without this the parse runs but writes nothing.",
)
def lexicon_ingest_hearth_tax(csv_path: Path, out_path: Path, apply_changes: bool) -> None:
    """Ingest a Hearth-Tax-returns CSV → JSONL events (wyrd-myh1).

    Expected CSV columns (named header row):
      place_name (required), parish, county, year_specific, country, modern_name

    year_specific captures the actual collection year (1662-1674);
    blank/unparseable falls back to 1665. Out-of-range integers warn.
    """
    from wyrd.generators.kenning.ingesters.hearth_tax import ingest

    counts = ingest(csv_path, out_path, apply=apply_changes)
    click.echo(f"CSV rows scanned:    {counts['csv_rows_scanned']:>10}", err=True)
    click.echo(f"Toponym events:      {counts['toponym_events']:>10}", err=True)
    click.echo(f"Attestation events:  {counts['attestation_events']:>10}", err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to write)", err=True)
    else:
        click.echo(f"\nWrote {out_path}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``ingest-hearth-tax`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_ingest_hearth_tax)
