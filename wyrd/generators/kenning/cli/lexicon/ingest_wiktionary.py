"""``wyrd kenning lexicon ingest-wiktionary`` — ingest a wiktextract slice into the etymon table (wyrd-4rt)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY
from wyrd.generators.kenning.wiktextract_ingester import ingest_wiktextract_path


@click.command("ingest-wiktionary")
@click.argument(
    "source_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
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
    help="Actually write etymons + descent edges. Without this, dry-run reporting only.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop after processing N entries (smoke / partial run).",
)
@click.option(
    "--since-line",
    type=int,
    default=0,
    help="Skip the first N lines of the JSONL (resume across multi-hour ingest sessions).",
)
def lexicon_ingest_wiktionary(
    source_path: Path,
    db_path: Path,
    apply_changes: bool,
    limit: int | None,
    since_line: int,
) -> None:
    """Ingest a wiktextract JSONL dump into etymon + etymon_descent
    (wyrd-4rt / wyrd-hun).

    Walks each entry's Etymology + Descendants sections, upserts the
    referenced etymons (across languages), and inserts descent edges
    with edge_type per the D27 taxonomy: inh→inheritance, bor→borrowing,
    der→derivation, cal→calque, desc→inheritance (downward).

    Pure structured-data ingest — no LLM, no form-in-body validation.
    Source attribution is the synthetic 'wiktionary' source row.

    Run `wyrd kenning lexicon cluster-cognates --apply` afterward to
    populate cognate_id from the new descent edges.

    Both .jsonl and .jsonl.gz are accepted; the suffix selects the
    open mode.
    """
    with LexiconDB(db_path) as db:
        result = ingest_wiktextract_path(
            db, source_path, apply=apply_changes, limit=limit, since_line=since_line
        )

    click.echo(
        f"ingest-wiktionary: {result['lines_read']} line(s) read, "
        f"{result['entries_parsed']} entry/entries parsed",
        err=True,
    )
    if result["entries_skipped_malformed"]:
        click.echo(
            f"  skipped {result['entries_skipped_malformed']} malformed line(s)",
            err=True,
        )
    click.echo(f"  upward_edges    = {result['upward_edges']}", err=True)
    click.echo(f"  downward_edges  = {result['downward_edges']}", err=True)
    if result["unsupported_templates"]:
        click.echo(
            f"  unsupported_templates = {result['unsupported_templates']} "
            f"(operator may want to extend the maps in wiktextract_ingester.py)",
            err=True,
        )
    # wyrd-ha9q Phase 2a: pronunciation + multi-script capture stats.
    # Only show non-zero counters so the slim Latin-script slices
    # (Old English, Old Norse, etc. that wiktextract usually leaves
    # without `sounds` arrays) don't print three zero lines.
    pron = result.get("pronunciation_captured", 0)
    orig = result.get("original_script_captured", 0)
    trans = result.get("transliteration_captured", 0)
    if pron or orig or trans:
        click.echo(
            f"  pronunciation captured: ipa={pron} original_script={orig} transliteration={trans}",
            err=True,
        )
    # wyrd-vsvi: tag-extraction stats from sense categories.
    tags_added = result.get("tags_added", 0)
    entries_with_tags = result.get("entries_with_tags", 0)
    if tags_added:
        click.echo(
            f"  tags captured: {tags_added} tag-writes across {entries_with_tags} entries",
            err=True,
        )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``ingest-wiktionary`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_ingest_wiktionary)
