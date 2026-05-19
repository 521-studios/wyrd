"""``wyrd kenning lexicon verify-bulk-sources`` — verify local cache matches manifest sha256s."""

from __future__ import annotations

import click


@click.command("verify-bulk-sources")
def lexicon_verify_bulk_sources() -> None:
    """Verify ~/.wyrd/sources/ matches the manifest (wyrd-0vj3).

    Walks each manifest slice, checks the local cache for presence
    + sha256 match, and exits 1 if any are missing or mismatched.
    Operator-on-demand only — not gated in CI.
    """
    from wyrd.generators.kenning.bulk_sources import (
        load_config,
        load_manifest,
        verify_local_cache,
    )

    manifest = load_manifest()
    config = load_config(manifest)
    statuses = verify_local_cache(manifest, config)

    ok = 0
    missing = 0
    mismatch = 0
    for status in statuses:
        if not status.present:
            click.echo(f"  ! {status.slice_name}: MISSING ({status.cache_path})", err=True)
            missing += 1
        elif not status.sha256_matches:
            click.echo(
                f"  ! {status.slice_name}: SHA256 MISMATCH ({status.cache_path})",
                err=True,
            )
            mismatch += 1
        else:
            ok += 1

    click.echo(f"\nOK: {ok}   Missing: {missing}   Mismatch: {mismatch}", err=True)
    if missing or mismatch:
        click.echo(
            "\nRun `lexicon fetch-bulk-sources` to repair the local cache.",
            err=True,
        )
        raise SystemExit(1)


def add_to(parent: click.Group) -> None:
    """Register ``verify-bulk-sources`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_verify_bulk_sources)
