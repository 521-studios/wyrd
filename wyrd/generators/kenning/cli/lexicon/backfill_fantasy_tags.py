"""``wyrd kenning lexicon backfill-fantasy-tags`` — backfill fantasy_morpheme tags from descent walk (D30)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import click

from wyrd.generators.kenning import fantasy_pipeline
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("backfill-fantasy-tags")
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
    help="Actually write the new etymon_tag rows. Without this, dry-run.",
)
def lexicon_backfill_fantasy_tags(db_path: Path, apply_changes: bool) -> None:
    """Tag existing seed monster etymons as fantasy (wyrd-ami).

    The seed entries from meanings.json that already carry `monster`
    (wyrm, elf, thyrs, grīma, sceocca, phúca, ...) don't know about
    the `fantasy` register tag. This adds it. Idempotent — safe to
    re-run.
    """
    fp = fantasy_pipeline

    if apply_changes:
        with LexiconDB(db_path) as db:
            n_etymons, n_tags = fp.backfill_fantasy_tag_from_monster_tag(db.conn)
            db.commit()
        click.echo(
            f"Backfilled `fantasy` tag onto {n_etymons} monster-tagged etymon(s).",
            err=True,
        )
    else:
        # Dry-run: count without writing.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            n = conn.execute("""SELECT COUNT(DISTINCT et.etymon_id)
                   FROM etymon_tag et
                   WHERE et.tag = 'monster'
                     AND NOT EXISTS (
                       SELECT 1 FROM etymon_tag et2
                       WHERE et2.etymon_id = et.etymon_id AND et2.tag = 'fantasy'
                     )""").fetchone()[0]
        finally:
            conn.close()
        click.echo(
            f"Would backfill `fantasy` tag onto {n} monster-tagged etymon(s). "
            f"(dry-run; pass --apply to write)",
            err=True,
        )


def add_to(parent: click.Group) -> None:
    """Register ``backfill-fantasy-tags`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_backfill_fantasy_tags)
