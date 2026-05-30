"""``wyrd kenning lexicon dump-jsonl`` — round-trip the lexicon DB out to per-source JSONL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("dump-jsonl")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Source SQLite DB to read from.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Directory to write <source_id>.jsonl files into.",
)
@click.option(
    "--source-id",
    default=None,
    help=(
        "Dump just one source's file. Without this, every source row is "
        "dumped to its own JSONL file."
    ),
)
@click.option(
    "--include-bulk",
    is_flag=True,
    default=False,
    help=(
        "Also dump bulk wiktextract-derived sources (wiktionary, "
        "wiktionary-empirical, wiktionary-forms). Default skips them — "
        "their rows are re-derivable from L1 raw inputs and the "
        "wiktionary dump file alone is ~200MB+."
    ),
)
def lexicon_dump_jsonl(
    db_path: Path,
    out_dir: Path,
    source_id: str | None,
    include_bulk: bool,
) -> None:
    """Dump per-source L2 facts from the lexicon DB to JSONL (wyrd-f295).

    Reads the current SQLite lexicon and emits one
    ``<source_id>.jsonl`` per ``source`` row, containing canonical-state
    rows for the source itself + every etymon it cites, plus list rows
    for citations, descent edges, mining-run audits, and the source's
    toponym etymologies (with element lists inline).

    Output is the "first compaction" — pure canonical-state rows that
    replay back to the same DB state via ``lexicon rebuild`` (when that
    lands). Once committed to git, ``data/mining/<source>.jsonl`` becomes
    the source of truth and the SQLite DB becomes a rebuildable build
    artifact.
    """
    from urllib.parse import quote

    from wyrd.generators.kenning.jsonl.dump import (
        DEFAULT_BULK_EXCLUDED_SOURCES,
        dump_all_sources,
        dump_fantasy_morphemes_to_file,
        dump_reflexes_to_file,
        dump_source_to_file,
    )

    # Read-only DB access — dump never writes.
    db_uri = f"file:{quote(str(db_path.absolute()))}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row

    exclude = () if include_bulk else DEFAULT_BULK_EXCLUDED_SOURCES

    # Pre-initialize so the post-try summary doesn't UnboundLocalError
    # if dump_all_sources raises (the finally still runs; control then
    # jumps past the summary lines, but a future maintainer reorganizing
    # this block won't trip on a phantom name binding).
    counts: dict[str, int] = {}
    fm_count = 0
    reflex_count = 0
    try:
        if source_id is not None:
            path, count = dump_source_to_file(conn, source_id, out_dir)
            click.echo(f"Wrote {count} rows → {path}", err=True)
            return
        counts = dump_all_sources(conn, out_dir, exclude=exclude)
        # wyrd-2thc: fantasy_morpheme has no source attribution — emit
        # to the synthetic ``_fantasy_morphemes.jsonl`` file.
        _, fm_count = dump_fantasy_morphemes_to_file(conn, out_dir)
        # wyrd-ned5: the seed reflex layer has no source attribution —
        # emit to the synthetic ``_reflexes.jsonl`` file.
        _, reflex_count = dump_reflexes_to_file(conn, out_dir)
    finally:
        conn.close()

    total_rows = sum(counts.values()) + fm_count + reflex_count
    sources_dumped = len(counts) + (1 if fm_count else 0) + (1 if reflex_count else 0)
    click.echo(f"Dumped {sources_dumped} sources, {total_rows} rows → {out_dir}", err=True)
    for sid, n in sorted(counts.items()):
        click.echo(f"  {sid:<40} {n:>6}", err=True)
    if fm_count:
        click.echo(f"  {'fantasy-mining':<40} {fm_count:>6}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``dump-jsonl`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_dump_jsonl)
