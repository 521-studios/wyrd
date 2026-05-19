"""``wyrd kenning lexicon bridge-celtic-forms`` — bridge Celtic forms to OE / ON cognates."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, bridge_celtic_forms
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("bridge-celtic-forms")
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
    help="Actually write merged_into_id. Without this, dry-run reporting only.",
)
def lexicon_bridge_celtic_forms(db_path: Path, apply_changes: bool) -> None:
    """Bridge celtic place-name etymons to Wiktionary lemmas via a
    hand-curated form→lemma table.

    Where bridge-language only handles same-form lookups (celtic/dun →
    old-irish/dun), this pass uses a curated table to map inflected /
    Anglicized celtic forms (choill, drum, lough, kin, gcorr) to their
    lemma equivalents (coill, druim, loch, ceann, corr) and then searches
    across the celtic candidate languages, preferring clustered targets.

    Iterates ALL celtic rows (including pre-existing tombstones from
    bridge-language) so a stub-bridge to an unclustered old-irish entry
    can be re-routed to a clustered modern Irish / Welsh / Scottish-Gaelic
    counterpart in the same pass.

    Run AFTER bridge-language and AFTER wiktextract ingest.
    Re-run cluster-cognates afterward to refresh cognate_id assignments
    via the merged_into_id rollup.

    Reverse via `clear-enrichment --stage=ocr --apply` (the bridge uses
    the same merged_into_id mechanism).
    """
    with LexiconDB(db_path) as db:
        result = bridge_celtic_forms(db, apply=apply_changes)

    verb = "bridged" if apply_changes else "would bridge"
    click.echo(
        f"bridge-celtic-forms: {result['examined']} celtic etymon(s) examined; "
        f"{verb} {result['bridged']}, {result['unmatched']} unmatched, "
        f"{result['missing_target']} missing-target.",
        err=True,
    )
    if result["missing_target"]:
        click.echo(
            f"  warn: {result['missing_target']} table entry/entries name a "
            f"lemma that doesn't exist as a canonical etymon in any candidate "
            f"language (operator may need to ingest more or fix the table)",
            err=True,
        )
    if apply_changes:
        click.echo(f"  rows_written = {result['rows_written']}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``bridge-celtic-forms`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_bridge_celtic_forms)
