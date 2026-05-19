"""``wyrd kenning lexicon ingest-speed-1611`` — ingest Speed 1611 operator CSV (wyrd-myh1)."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("ingest-speed-1611")
@click.argument(
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/speed_1611.jsonl"),
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
def lexicon_ingest_speed_1611(csv_path: Path, out_path: Path, apply_changes: bool) -> None:
    """Ingest a Speed-1611 transcription CSV → JSONL events (wyrd-myh1).

    Expected CSV columns (named header row):
      place_name (required), county, parish, country, modern_name
    """
    from wyrd.generators.kenning.speed_1611_ingester import ingest

    counts = ingest(csv_path, out_path, apply=apply_changes)
    click.echo(f"CSV rows scanned:    {counts['csv_rows_scanned']:>10}", err=True)
    click.echo(f"Toponym events:      {counts['toponym_events']:>10}", err=True)
    click.echo(f"Attestation events:  {counts['attestation_events']:>10}", err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to write)", err=True)
    else:
        click.echo(f"\nWrote {out_path}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``ingest-speed-1611`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_ingest_speed_1611)
