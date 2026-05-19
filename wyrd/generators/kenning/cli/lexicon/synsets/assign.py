"""``wyrd kenning lexicon synsets assign`` — add an etymon to a meaning_synset."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from wyrd.generators.kenning.cli.lexicon.synsets._helpers import _resolve_etymon_ident
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, assign_etymon_to_meaning_synset
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("assign")
@click.argument("etymon_ident", type=str)
@click.argument("synset_label", type=str)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--language",
    type=str,
    default=None,
    help=(
        "When set, ETYMON_IDENT is treated as a canonical_form looked up "
        "via (form, language) — more ergonomic for human curators than "
        "looking up the integer id first. Without --language, "
        "ETYMON_IDENT must be a numeric etymon.id."
    ),
)
@click.option(
    "--fit",
    type=click.Choice(["core", "peripheral"]),
    default="core",
    show_default=True,
    help="'core' = primary sense; 'peripheral' = secondary sense.",
)
def lexicon_synsets_assign(
    etymon_ident: str,
    synset_label: str,
    db_path: Path,
    language: str | None,
    fit: str,
) -> None:
    """Add an etymon as a member of a meaning_synset. Idempotent: re-running
    with the same args is a no-op; re-running with a different --fit
    updates the existing row.

    ETYMON_IDENT is either a numeric etymon.id (default) or a
    canonical_form when --language is given (e.g. ``assign wæter
    water/flowing --language old-english``).
    """
    with LexiconDB(db_path) as db:
        try:
            etymon_id = _resolve_etymon_ident(db, etymon_ident, language)
            inserted = assign_etymon_to_meaning_synset(db, etymon_id, synset_label, fit=fit)
        except ValueError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
    click.echo(f"{'Added' if inserted else 'Updated'} etymon {etymon_id} → {synset_label} ({fit})")


def add_to(parent: click.Group) -> None:
    """Register ``assign`` on the @lexicon_synsets group."""
    parent.add_command(lexicon_synsets_assign)
