"""``wyrd kenning lexicon diff-rebuild`` — diff the current DB against a rebuild from JSONL (wyrd-w3x0)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("diff-rebuild")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Current SQLite DB to compare against.",
)
@click.option(
    "--jsonl-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Source JSONL directory to rebuild from.",
)
@click.option(
    "--with-enrichment",
    is_flag=True,
    default=False,
    help=(
        "Run the L3 enrichment orchestrator after rebuild. Without "
        "this, L3 columns (lemma_id, merged_into_id, etc.) on the "
        "rebuilt DB are NULL — only L2 row counts are compared."
    ),
)
def lexicon_diff_rebuild(db_path: Path, jsonl_dir: Path, with_enrichment: bool) -> None:
    """Rebuild from JSONL into a temp DB; compare row counts to live (wyrd-f4nl).

    Snapshots the current DB's L2 table counts, rebuilds a fresh DB from
    ``--jsonl-dir`` into a temp file, and reports per-table deltas.
    Useful for catching regressions in the JSONL pipeline as ingesters
    get converted — CI can run this and gate on the exit code.

    Exit status: 0 when no table changed row count; 1 when any did.
    """
    # Deferred imports: jsonl.build + init_schema pull alembic-driven
    # schema setup that other lexicon commands don't touch. Deferring
    # keeps `wyrd kenning --help` snappy. tempfile + urllib.parse are
    # deferred for symmetry with the other heavy imports in this body.
    import tempfile
    from urllib.parse import quote

    from wyrd.generators.kenning.jsonl.build import (
        build_from_jsonl,
        diff_table_counts,
        format_diff_rebuild,
        has_any_delta,
        jsonl_paths_in,
        table_counts,
    )
    from wyrd.generators.kenning.lexicon import init_schema

    # Read current DB counts before touching anything else.
    db_uri = f"file:{quote(str(db_path.absolute()))}?mode=ro"
    current_conn = sqlite3.connect(db_uri, uri=True)
    current_conn.row_factory = sqlite3.Row
    try:
        before = table_counts(current_conn)
    finally:
        current_conn.close()

    # Rebuild into a temp directory so the WAL sidecars (-wal, -shm
    # that init_schema creates because of journal_mode=WAL) get
    # cleaned up alongside the main DB file. NamedTemporaryFile
    # would only track the main file and leak the sidecars in /tmp.
    # Directory also dodges the init_schema-unlinks-the-tempfile
    # ownership confusion.
    with tempfile.TemporaryDirectory(prefix="wyrd-diff-rebuild-") as tmpdir:
        rebuilt_path = Path(tmpdir) / "rebuilt.db"
        init_schema(rebuilt_path)
        rebuilt_conn = sqlite3.connect(rebuilt_path)
        rebuilt_conn.row_factory = sqlite3.Row
        try:
            paths = jsonl_paths_in(jsonl_dir)
            build_from_jsonl(rebuilt_conn, paths)
        finally:
            rebuilt_conn.close()

        if with_enrichment:
            from wyrd.generators.kenning.enrichment import run_full_enrichment
            from wyrd.generators.kenning.jsonl.build import (
                collect_curation_overrides,
                collect_etymon_splits,
                collect_gloss_suppressions,
            )

            curation = collect_curation_overrides(paths) or None
            suppressions = collect_gloss_suppressions(paths) or None
            splits = collect_etymon_splits(paths) or None
            with LexiconDB(rebuilt_path) as db:
                run_full_enrichment(
                    db,
                    apply=True,
                    curation_state=curation,
                    suppression_state=suppressions,
                    split_state=splits,
                )

        rebuilt_conn = sqlite3.connect(rebuilt_path)
        rebuilt_conn.row_factory = sqlite3.Row
        try:
            after = table_counts(rebuilt_conn)
        finally:
            rebuilt_conn.close()

    rows = diff_table_counts(before, after)
    click.echo(f"## diff-rebuild: {db_path} ↔ rebuild({jsonl_dir})")
    click.echo("")
    click.echo(format_diff_rebuild(rows))

    if has_any_delta(rows):
        click.echo("\n⚠️  Row counts differ between current and rebuild.", err=True)
        raise SystemExit(1)
    click.echo("\n✓ No row-count deltas.", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``diff-rebuild`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_diff_rebuild)
