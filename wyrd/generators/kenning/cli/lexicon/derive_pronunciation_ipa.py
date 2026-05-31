"""``wyrd kenning lexicon derive-pronunciation-ipa`` — G2P IPA backfill (wyrd-vm8t).

Standalone runner for the enrichment pass: fills ``etymon.pronunciation_ipa``
from the grapheme→IPA converter for OE/ON/welsh and fixes OE onset-h (/x/→/h/).
Also run automatically by ``rebuild-from-jsonl --with-enrichment`` /
``lexicon enrich``; this command is for a targeted re-run.
"""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("derive-pronunciation-ipa")
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
    help="Write pronunciation_ipa. Without this, dry-run + report counts.",
)
def lexicon_derive_pronunciation_ipa(db_path: Path, apply_changes: bool) -> None:
    """Backfill etymon pronunciation_ipa from the G2P (wyrd-vm8t)."""
    from wyrd.generators.kenning.lexicon.pronunciation_backfill import (
        derive_pronunciation_ipa,
        format_pronunciation_run,
    )

    with LexiconDB(db_path) as db:
        result = derive_pronunciation_ipa(db, apply=apply_changes)
        if apply_changes:
            db.commit()
    click.echo(format_pronunciation_run(result), err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to write)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``derive-pronunciation-ipa`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_derive_pronunciation_ipa)
