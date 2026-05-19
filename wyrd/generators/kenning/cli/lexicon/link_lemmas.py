"""``wyrd kenning lexicon link-lemmas`` — cluster inflected etymons under their lemma (D8 / wyrd-7fn enrichment pass)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, link_lemmas
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("link-lemmas")
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
    help="Actually link lemmas. Without this, dry-run reporting only.",
)
def lexicon_link_lemmas(db_path: Path, apply_changes: bool) -> None:
    """Link inflected etymons to their lemma rows.

    For each etymon with lemma_id IS NULL, tries common inflection-strip
    rules per language. If the stripped stem matches an EXISTING etymon
    row in the same language, sets lemma_id + inflection on the inflected
    etymon. Conservative — never fabricates a lemma.

    Run after `migrate` (and after any major mining/OCR-cluster work).
    """
    with LexiconDB(db_path) as db:
        result = link_lemmas(db, apply=apply_changes)

    click.echo(
        f"Lemma linkage candidates: {result['candidates']}  "
        f"({'applied' if result['applied'] else 'dry-run'})",
        err=True,
    )
    if result["sample"]:
        click.echo("First 25 proposals:", err=True)
        for p in result["sample"]:
            click.echo(
                f"  [{p['language']:14}] {p['inflected_form']:18} → lemma={p['lemma_form']:14} ({p['inflection']})",
                err=True,
            )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``link-lemmas`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_link_lemmas)
