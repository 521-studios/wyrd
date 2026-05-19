"""``wyrd kenning lexicon bridge-phonological-oe`` — bridge Old English forms via phonological-shape heuristics."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, bridge_phonological_oe
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("bridge-phonological-oe")
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
def lexicon_bridge_phonological_oe(db_path: Path, apply_changes: bool) -> None:
    """Bridge OE place-name forms to Wiktionary canonicals.

    Place-name dictionaries write modernized OE forms (`ton`, `lea`,
    `burgh`, `dale`); Wiktionary uses scholarly orthography (`tūn`,
    `lēah`, `burh`, `dæl`). normalize-ocr handles macron-strip OCR
    variants but NOT vowel-weakening / silent-e / gh-spelling shifts.
    This pass uses a hand-curated mapping table to merge known pairs
    via merged_into_id (D22 non-destructive).

    Run AFTER wiktextract ingest. Re-run cluster-cognates afterward
    to refresh cognate_id assignments via the merged_into_id rollup.

    Reverse via `clear-enrichment --stage=ocr --apply` (the bridge
    uses the same merged_into_id mechanism).
    """
    with LexiconDB(db_path) as db:
        result = bridge_phonological_oe(db, apply=apply_changes)

    verb = "bridged" if apply_changes else "would bridge"
    click.echo(
        f"bridge-phonological-oe: {result['examined']} canonical OE etymon(s) examined; "
        f"{verb} {result['bridged']}, {result['unmatched']} unmatched.",
        err=True,
    )
    if result["missing_target"]:
        click.echo(
            f"  warn: {result['missing_target']} table entry/entries name a "
            f"target that doesn't exist as a canonical OE etymon "
            f"(operator may need to ingest more or extend the bridge table)",
            err=True,
        )
    if apply_changes:
        click.echo(f"  rows_written = {result['rows_written']}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``bridge-phonological-oe`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_bridge_phonological_oe)
