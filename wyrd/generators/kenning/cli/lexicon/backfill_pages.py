"""``wyrd kenning lexicon backfill-pages`` — backfill citation page anchors from parsed page boundaries (wyrd-azv)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, backfill_citation_pages
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("backfill-pages")
@click.argument(
    "sources_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
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
    help="Actually write page numbers. Without this, dry-run reporting only.",
)
@click.option(
    "--source",
    "source_filter",
    type=str,
    default=None,
    help="Process only the source matching this id (filename stem). Default: every .txt in sources_dir.",
)
def lexicon_backfill_pages(
    sources_dir: Path,
    db_path: Path,
    apply_changes: bool,
    source_filter: str | None,
) -> None:
    """Backfill etymon_citation.page + toponym_etymology.page (wyrd-azv).

    Walks each .txt in <sources_dir>, tries both Mawer-style and
    Skeat-§ running header conventions, and for rows where page IS
    NULL locates the row's quoted excerpt in body text to resolve a
    page number. Books matching neither convention are skipped
    (reported as no_headers).

    Idempotent: re-runs only touch rows where page is still NULL.
    """
    totals = dict.fromkeys(_BACKFILL_TOTALS_KEYS, 0)

    with LexiconDB(db_path) as db:
        for path in sorted(sources_dir.glob("*.txt")):
            source_id = path.stem
            if source_filter and source_id != source_filter:
                continue
            text = path.read_text(errors="replace", encoding="utf-8")
            counts = backfill_citation_pages(db, source_id, text, apply=apply_changes)
            totals["sources_processed"] += 1
            if counts["no_headers"]:
                totals["sources_no_headers"] += 1
                click.echo(f"  [{source_id:50}]  no recognized headers; skipped", err=True)
                continue
            click.echo(
                f"  [{source_id:50}]  cit={counts['citations_updated']:>4}  "
                f"ety={counts['etymologies_updated']:>4}  "
                f"miss={counts['quote_not_in_text']:>3}",
                err=True,
            )
            for key in _BACKFILL_PER_SOURCE_KEYS:
                totals[key] += counts[key]

    if source_filter and totals["sources_processed"] == 0:
        click.echo(
            f"warn: --source {source_filter!r} matched no .txt file in {sources_dir}",
            err=True,
        )
    _print_backfill_totals(totals, apply_changes)


_BACKFILL_PER_SOURCE_KEYS = (
    "citations_updated",
    "etymologies_updated",
    "quote_not_in_text",
    "no_quote",
    "before_first_page",
)
_BACKFILL_TOTALS_KEYS = (*_BACKFILL_PER_SOURCE_KEYS, "sources_processed", "sources_no_headers")


def _print_backfill_totals(totals: dict[str, int], apply_changes: bool) -> None:
    """Render the backfill summary. Header counts come from sources_*
    keys; per-source counters listed below come from _BACKFILL_PER_SOURCE_KEYS
    in display order."""
    click.echo(
        f"\nTotals: {totals['sources_processed']} sources processed, "
        f"{totals['sources_no_headers']} skipped (no headers).",
        err=True,
    )
    for key in _BACKFILL_PER_SOURCE_KEYS:
        click.echo(f"  {key:<22}= {totals[key]}", err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to write)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``backfill-pages`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_backfill_pages)
