"""``wyrd kenning lexicon push-bulk-sources`` — upload local bulk slices to S3 + update manifest."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("push-bulk-sources")
@click.option(
    "--slice",
    "slice_names",
    multiple=True,
    help="Restrict to named slices (repeatable). Default: every slice in manifest.",
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_bulk_manifest.json"),
    show_default=True,
)
def lexicon_push_bulk_sources(slice_names: tuple[str, ...], manifest_path: Path) -> None:
    """Upload ~/.wyrd/sources/ slices to S3 and rewrite the manifest
    (wyrd-0vj3).

    For each manifest slice: if a <slice>.jsonl.zst exists locally,
    upload it; else if <slice>.jsonl exists, compress + upload;
    else skip (slice not mined locally). The manifest file is
    overwritten with new sha256s + sizes; operator commits the
    manifest change to git.
    """
    from wyrd.generators.kenning.bulk_sources import (
        load_config,
        load_manifest,
        manifest_to_json,
        upload_slices,
    )

    manifest = load_manifest(manifest_path)
    config = load_config(manifest)
    result = upload_slices(
        manifest,
        config,
        slice_names=list(slice_names) if slice_names else None,
    )

    click.echo(f"Uploaded: {len(result.uploaded)}", err=True)
    for name in result.uploaded:
        click.echo(f"  + {name}", err=True)
    click.echo(f"Skipped:  {len(result.skipped)}  (no local file)", err=True)
    if result.failed:
        click.echo(f"FAILED:   {len(result.failed)}", err=True)
        for name, reason in result.failed:
            click.echo(f"  ! {name}: {reason}", err=True)
        raise SystemExit(1)

    manifest_path.write_text(manifest_to_json(result.new_manifest), encoding="utf-8")
    click.echo(f"\nWrote {manifest_path}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``push-bulk-sources`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_push_bulk_sources)
