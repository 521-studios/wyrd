"""``wyrd kenning lexicon mine-wiktextract-forms`` — ingest wiktextract per-language form variants."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY
from wyrd.generators.kenning.wiktextract_corpus_miner import mine_corpus_forms


@click.command("mine-wiktextract-forms")
@click.argument("slice_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop after N entries (for smoke testing). Default: walk the full slice.",
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Write etymon_text_match rows. Without --apply this is a dry run that "
    "reports counts only.",
)
def lexicon_mine_wiktextract_forms(
    slice_path: Path,
    db_path: Path,
    limit: int | None,
    apply_changes: bool,
) -> None:
    """wyrd-vx09: ingest a wiktextract slice's ``forms`` arrays as
    etymon_text_match rows so the bundle's per-language _variants
    pool gets non-OE/ON variant data (audit follow-up wyrd-f6v4).

    Walks the slice file and, for each entry whose canonical
    headword is already an etymon in the lexicon, writes each
    meaningful form (plural, comparative, soft mutation, etc.) as a
    text-match row tagged ``source='wiktionary-forms'``. Verbal
    conjugation forms and Wiktionary table-rendering scaffolding
    (``table-tags`` / ``inflection-template`` / ``error-unrecognized-
    form``) are filtered out — they're not useful for place-name
    variant pools.

    Pass ``--apply`` to write; without it the run is a dry-count
    pass for size-checking before committing to a full slice ingest.

    Idempotent: re-runs increment match_count on existing
    (etymon_id, source_id, matched_form) rows rather than duplicating.
    """
    with LexiconDB(db_path) as db:
        counts = mine_corpus_forms(
            db,
            slice_path,
            apply=apply_changes,
            limit=limit,
        )
    suffix = "--apply" if apply_changes else "(dry-run)"
    click.echo(
        f"slice={slice_path.name} {suffix} "
        f"entries_walked={counts['entries_walked']} "
        f"forms_processed={counts['forms_processed']} "
        f"forms_written={counts['forms_written']} "
        f"etymons_missing={counts['etymons_missing']}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``mine-wiktextract-forms`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_wiktextract_forms)
