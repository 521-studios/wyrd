"""``wyrd kenning lexicon ingest-hundred-rolls`` — ingest Hundred Rolls operator CSV (wyrd-3atv)."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("ingest-hundred-rolls")
@click.argument(
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/rotuli_hundredorum.jsonl"),
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
def lexicon_ingest_hundred_rolls(csv_path: Path, out_path: Path, apply_changes: bool) -> None:
    """Ingest Hundred Rolls (1279-80) CSV → JSONL events (wyrd-3atv).

    Reads a transcribed Rotuli Hundredorum CSV (operator-supplied) and
    writes toponym + attestation events with date_year=1279.

    Expected CSV columns (named in header row):
      vill (required), hundred, county, country, modern_name

    Cross-era cross-linking is automatic — when a Hundred Rolls
    modern_name matches a Domesday or OS Open Names toponym ref, all
    attestations attach to the same toponym in the rebuilt DB.
    """
    from wyrd.generators.kenning.ingesters.hundred_rolls import ingest

    counts = ingest(csv_path, out_path, apply=apply_changes)
    click.echo(f"CSV rows scanned:    {counts['csv_rows_scanned']:>10}", err=True)
    click.echo(f"Toponym events:      {counts['toponym_events']:>10}", err=True)
    click.echo(f"Attestation events:  {counts['attestation_events']:>10}", err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to write)", err=True)
    else:
        click.echo(f"\nWrote {out_path}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``ingest-hundred-rolls`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_ingest_hundred_rolls)
