"""``wyrd kenning lexicon synsets candidates`` — show meaning-preserving substitution candidates."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, get_meaning_preserving_candidates
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("candidates")
@click.argument("etymon_id", type=int)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--target-language",
    type=str,
    default=None,
    help="Restrict candidates to one language (e.g. 'old-english').",
)
@click.option(
    "--fit",
    type=click.Choice(["core", "peripheral"]),
    default=None,
    help="Restrict to high-confidence ('core') or peripheral memberships only.",
)
def lexicon_synsets_candidates(
    etymon_id: int,
    db_path: Path,
    target_language: str | None,
    fit: str | None,
) -> None:
    """Show meaning-preserving substitution candidates for an etymon —
    other etymons that share at least one meaning_synset.

    Used by upcoming Lab transforms (replace-root, calque, anglicize)
    to look up same-meaning morphemes across languages and registers.
    """
    with LexiconDB(db_path) as db:
        rows = get_meaning_preserving_candidates(
            db, etymon_id, target_language=target_language, fit=fit
        )
    if not rows:
        click.echo(f"(no meaning-preserving candidates for etymon {etymon_id})")
        return
    for r in rows:
        click.echo(
            f"  {r['canonical_form']:<24} {r['language']:<14} "
            f"{r['synset_label']:<28} (fit: {r['candidate_fit']})"
        )


def add_to(parent: click.Group) -> None:
    """Register ``candidates`` on the @lexicon_synsets group."""
    parent.add_command(lexicon_synsets_candidates)
