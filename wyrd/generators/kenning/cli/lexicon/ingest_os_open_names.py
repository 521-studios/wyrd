"""``wyrd kenning lexicon ingest-os-open-names`` — ingest the Ordnance Survey OpenNames CSV corpus."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("ingest-os-open-names")
@click.argument(
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/os_open_names.jsonl"),
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
def lexicon_ingest_os_open_names(csv_path: Path, out_path: Path, apply_changes: bool) -> None:
    """Ingest OS Open Names CSV → JSONL events (wyrd-3ypp).

    Reads the OS Open Names data product, filters to populated places
    (City / Town / Village / Hamlet / Suburban Area / Other Settlement),
    and writes toponym + attestation events to the named JSONL.

    Operator workflow:
        1. Download OS Open Names CSV to sources/os_open_names.csv
        2. lexicon ingest-os-open-names sources/os_open_names.csv --apply
        3. lexicon rebuild-from-jsonl

    The JSONL file is gitignored (30-50MB) — it's regenerated from
    the L1 CSV on demand. Source attribution: synthetic 'os_open_names'
    source, recorded with OGL v3 license notes.
    """
    from wyrd.generators.kenning.ingesters.os_open_names import ingest

    counts = ingest(csv_path, out_path, apply=apply_changes)
    click.echo(f"CSV rows scanned:       {counts['csv_rows_scanned']:>10}", err=True)
    click.echo(f"Populated places kept:  {counts['populated_places_kept']:>10}", err=True)
    click.echo(f"Toponym events:         {counts['toponym_events']:>10}", err=True)
    click.echo(f"Attestation events:     {counts['attestation_events']:>10}", err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to write)", err=True)
    else:
        click.echo(f"\nWrote {out_path}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``ingest-os-open-names`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_ingest_os_open_names)
