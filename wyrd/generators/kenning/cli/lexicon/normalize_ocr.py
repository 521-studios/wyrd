"""``wyrd kenning lexicon normalize-ocr`` — OCR-cluster etymon spelling variants via merged_into_id redirect (D22)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, cluster_ocr_variants
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("normalize-ocr")
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
    help="Actually perform the merge. Without this flag the command runs as a dry-run.",
)
def lexicon_normalize_ocr(db_path: Path, apply_changes: bool) -> None:
    """Cluster etymons by OCR-normalized form and merge variants.

    Without --apply this is a dry-run that just reports what would happen.
    Pass --apply to actually merge redundant etymon rows. Each merge group
    keeps the lowest-id row and repoints citations, glosses, tags, reflex
    links, and breakdown elements onto it.
    """
    with LexiconDB(db_path) as db:
        result = cluster_ocr_variants(db, apply=apply_changes)

    click.echo(
        f"OCR-variant groups: {result['groups']}, "
        f"etymons {'merged' if apply_changes else 'mergeable'}: "
        f"{result['etymons_merged']}",
        err=True,
    )
    if result["sample_groups"]:
        click.echo("Sample merges (first 20):", err=True)
        for g in result["sample_groups"]:
            members = ", ".join(g["members"])
            click.echo(
                f"  [{g['language']:12}] {g['normalized']:20} <- {members}",
                err=True,
            )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``normalize-ocr`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_normalize_ocr)
