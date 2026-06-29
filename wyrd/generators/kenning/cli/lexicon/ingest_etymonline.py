"""``wyrd kenning lexicon ingest-etymonline`` — ingest etymonline.com etymology data."""

from __future__ import annotations

import time
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, etymonline_ingester
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("ingest-etymonline")
@click.argument(
    "source_dir",
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
    help=(
        "Actually upsert etymons + descent edges. Without this, the "
        "command parses each file and reports counts without writing."
    ),
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop after processing N files (smoke testing).",
)
def lexicon_ingest_etymonline(
    source_dir: Path,
    db_path: Path,
    apply_changes: bool,
    limit: int | None,
) -> None:
    """Ingest pre-scraped Etymonline prose into the etymology graph (wyrd-zqkp).

    Each file in `source_dir` should be one Etymonline word page's
    prose, captured separately by the operator (e.g. via `rodney text
    'section.prose-lg' > harpy.txt`). The CLI doesn't fetch — it
    consumes saved text — so the scrape rate, identity, and policy
    decisions stay with the operator.

    Per-file processing:
    1. Parse the prose into one or more Senses (multi-sense pages
       like 'troll' produce v./n.1/n.2 blocks).
    2. For each Sense's etymological chain, upsert (language, word)
       pairs and connect them with `etymon_descent` edges
       source_id='etymonline'.
    3. Wire chain[0] to the headword's existing modern-english
       etymon row when present.

    Idempotent: re-running over the same dir is a no-op via the
    etymon_descent UNIQUE on (parent, child, edge_type, source).
    """
    files = sorted(source_dir.glob("*.txt"))
    if limit is not None:
        files = files[:limit]
    click.echo(
        f"Ingesting {len(files)} Etymonline file(s) from {source_dir}. "
        f"{'Applying' if apply_changes else 'Dry-run'}.",
        err=True,
    )
    totals = {
        "files": 0,
        "senses_parsed": 0,
        "senses_with_chain": 0,
        "senses_without_chain": 0,
        "chain_links": 0,
        "etymons_added_or_existing": 0,
        "edges_added": 0,
        "edges_skipped_dupe": 0,
        "leaf_edge_skipped_no_headword": 0,
        "glosses_added": 0,
    }
    progress_start = time.time()
    total_files = len(files)
    with LexiconDB(db_path) as db:
        if apply_changes:
            etymonline_ingester.ensure_source(db)
        for completed, f in enumerate(files, start=1):
            text = f.read_text(encoding="utf-8")
            counts = etymonline_ingester.ingest_text(db, text, apply=apply_changes)
            totals["files"] += 1
            for k, v in counts.items():
                if k in totals:
                    totals[k] += v
            edges_field = f" edges_added={counts['edges_added']}" if apply_changes else ""
            elapsed = time.time() - progress_start
            rate = elapsed / completed
            click.echo(
                f"  [{completed}/{total_files}] {f.name:<32} "
                f"senses={counts['senses_parsed']:<2} "
                f"links={counts['chain_links']:<3}{edges_field} "
                f"({rate:.2f}s/file)",
                err=True,
            )
        if apply_changes:
            db.commit()

    click.echo("", err=True)
    click.echo(f"Summary: {totals}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``ingest-etymonline`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_ingest_etymonline)
