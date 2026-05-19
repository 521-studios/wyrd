"""``wyrd kenning lexicon rebuild-from-jsonl`` — rebuild the lexicon DB from JSONL slices (wyrd-f295 L2 → L3)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("rebuild-from-jsonl")
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Target SQLite path. WIPED if it exists.",
)
@click.option(
    "--jsonl-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Directory of <source_id>.jsonl files to replay into the DB.",
)
@click.option(
    "--with-enrichment",
    is_flag=True,
    default=False,
    help=(
        "After replay, run the L3 enrichment passes (normalize-ocr + "
        "link-lemmas) in canonical order. Without this flag the rebuild "
        "stops at L2 and operators run the enrichment commands manually."
    ),
)
@click.option(
    "--skip-bulk",
    is_flag=True,
    default=False,
    help=(
        "Skip L1 wiktextract bulk ingest. By default rebuild-from-jsonl "
        "ingests every slice listed in data/mining/_bulk_manifest.json "
        "from ~/.wyrd/sources/. --skip-bulk produces an L2-only DB — "
        "useful for fast iteration on a single curated source file or "
        "for debugging without paying the multi-minute wiktextract cost."
    ),
)
@click.option(
    "--fetch-bulk",
    is_flag=True,
    default=False,
    help=(
        "Download missing/mismatched bulk slices from S3 to "
        "~/.wyrd/sources/ before ingesting. Without this flag, missing "
        "slices fail loud with a hint to run `lexicon fetch-bulk-sources` "
        "separately. Opt-in to keep CI / offline operators from silently "
        "fetching."
    ),
)
def lexicon_rebuild_from_jsonl(
    db_path: Path,
    jsonl_dir: Path,
    with_enrichment: bool,
    skip_bulk: bool,
    fetch_bulk: bool,
) -> None:
    """Rebuild the lexicon SQLite from per-source JSONL + L1 bulk
    wiktextract slices — wyrd-f295 + wyrd-hidb.

    Initializes a fresh schema at ``--db`` (wiping any existing DB at
    that path), then:

    1. (default) Ingests each L1 wiktextract slice listed in
       ``data/mining/_bulk_manifest.json`` from ``~/.wyrd/sources/``.
       Pass ``--skip-bulk`` for an L2-only debug rebuild; pass
       ``--fetch-bulk`` to auto-download missing slices from S3.
    2. Replays every ``<source_id>.jsonl`` under ``--jsonl-dir`` into
       the SQLite. Each L2 file must contain exactly one ``source``
       canonical-state row — its ``ref`` is the source_id for every
       list-type row in the file. Etymons appearing in multiple files
       are merged: glosses + tags are unioned; scalar fields follow
       last-write-wins by file order.
    3. (with ``--with-enrichment``) Runs the full L3 enrichment chain
       via ``run_full_enrichment`` (wyrd-hidb Phase 2): normalize-ocr →
       link-lemmas → apply-curation → decompose → cluster-cognates →
       classify-stratum → derive-english-shaped → project-period-forms.
       All eight passes run in one invocation; standalone CLI commands
       remain for targeted reruns (e.g. with ``--force``).
    """
    from wyrd.generators.kenning.jsonl_build import (
        build_from_jsonl,
        jsonl_paths_in,
    )
    from wyrd.generators.kenning.lexicon import init_schema

    init_schema(db_path)

    # L1 bulk ingest first — its rows are inputs for the L2 replay
    # (citations + descent edges referencing wiktionary-empirical
    # etymons would orphan-skip otherwise).
    if not skip_bulk:
        from wyrd.generators.kenning.bulk_sources import ingest_all_slices

        click.echo("Bulk ingest (L1):", err=True)
        with LexiconDB(db_path) as db:
            bulk_result = ingest_all_slices(db, apply=True, fetch=fetch_bulk)
        if bulk_result.fetched:
            click.echo(f"  fetched from S3:   {len(bulk_result.fetched)}", err=True)
        click.echo(f"  slices ingested:   {len(bulk_result.per_slice)}", err=True)
        for k, v in bulk_result.totals.items():
            click.echo(f"  {k:<35} {v:>10,}", err=True)
        if bulk_result.failed:
            click.echo(f"  FAILED: {len(bulk_result.failed)}", err=True)
            for name, reason in bulk_result.failed:
                click.echo(f"    ! {name}: {reason}", err=True)
            raise SystemExit(1)
        click.echo("", err=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        paths = jsonl_paths_in(jsonl_dir)
        counts = build_from_jsonl(conn, paths)
    finally:
        conn.close()

    click.echo(f"Rebuilt {db_path} from {len(paths)} JSONL files", err=True)
    click.echo("Inserted:", err=True)
    for table, n in counts.items():
        # Some count fields are sample-lists (orphan_refs from wyrd-lene)
        # rather than ints. Format the scalar counts as right-aligned
        # numbers; render list fields as just their length so the line
        # stays readable.
        if isinstance(n, list):
            click.echo(f"  {table:<35} ({len(n)} samples)", err=True)
        else:
            click.echo(f"  {table:<35} {n:>8}", err=True)

    if with_enrichment:
        from wyrd.generators.kenning.enrichment import (
            format_enrichment_run,
            run_full_enrichment,
        )
        from wyrd.generators.kenning.jsonl_build import collect_curation_overrides

        curation_state = collect_curation_overrides(paths)
        click.echo("", err=True)
        with LexiconDB(db_path) as db:
            result = run_full_enrichment(db, apply=True, curation_state=curation_state or None)
        click.echo(format_enrichment_run(result), err=True)


def add_to(parent: click.Group) -> None:
    """Register ``rebuild-from-jsonl`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_rebuild_from_jsonl)
