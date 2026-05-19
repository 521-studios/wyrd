"""``wyrd kenning lexicon fetch-bulk-sources`` — populate ~/.wyrd/sources/ from S3 (wyrd-0vj3)."""

from __future__ import annotations

import click


@click.command("fetch-bulk-sources")
@click.option(
    "--slice",
    "slice_names",
    multiple=True,
    help="Restrict to named slices (repeatable). Default: all slices in manifest.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-download even when local cache sha256 matches the manifest.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report what would happen; don't touch S3 or the local cache.",
)
def lexicon_fetch_bulk_sources(slice_names: tuple[str, ...], force: bool, dry_run: bool) -> None:
    """Populate ~/.wyrd/sources/ from S3 (wyrd-0vj3).

    Reads data/mining/_bulk_manifest.json, downloads any slice whose
    local cache file is missing or whose sha256 doesn't match.
    """
    from wyrd.generators.kenning.bulk_sources import (
        fetch_missing_slices,
        load_config,
        load_manifest,
    )

    manifest = load_manifest()
    config = load_config(manifest)
    result = fetch_missing_slices(
        manifest,
        config,
        slice_names=list(slice_names) if slice_names else None,
        force=force,
        dry_run=dry_run,
    )
    if dry_run:
        click.echo("(dry-run; nothing written)", err=True)
    click.echo(f"Fetched: {len(result.fetched)}", err=True)
    for name in result.fetched:
        click.echo(f"  + {name}", err=True)
    click.echo(f"Skipped: {len(result.skipped)}  (already current)", err=True)
    if result.failed:
        click.echo(f"FAILED:  {len(result.failed)}", err=True)
        for name, reason in result.failed:
            click.echo(f"  ! {name}: {reason}", err=True)
        raise SystemExit(1)


def add_to(parent: click.Group) -> None:
    """Register ``fetch-bulk-sources`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_fetch_bulk_sources)
