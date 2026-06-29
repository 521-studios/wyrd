"""``wyrd kenning lexicon import-mining-log`` — back-fill mining_run from a JSONL log file (D23)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.lexicon.review import _import_mining_log_record
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("import-mining-log")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
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
    help="Actually insert mining_run rows. Without this flag the command runs as a dry-run.",
)
def lexicon_import_mining_log(path: Path, db_path: Path, apply_changes: bool) -> None:
    """Back-fill the mining_run table from a JSONL log of historical runs.

    The input file is line-delimited JSON; each line describes one
    mining or review run with the fields written by record_mining_run.
    Recovered from session transcripts, hand-curated logs, etc. Use this
    once after wyrd-ej4 lands to seed the audit table with everything
    that already happened pre-table.

    Required fields per JSON line:
      source_id, provider, model, mode, accepted, declined, rejected
    Optional:
      parsed_count (defaults to accepted+declined+rejected),
      by_failure (object), started_at, completed_at, notes

    Idempotent on (source_id, provider, model, mode, completed_at) so
    re-running with the same JSONL is safe.
    """
    inserted = 0
    skipped = 0
    errors: list[str] = []

    with LexiconDB(db_path) as db:
        # Check that mining_run exists; if not, run migrate first.
        tables = {
            r["name"] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "mining_run" not in tables:
            raise click.ClickException(
                "mining_run table missing — run `wyrd kenning lexicon migrate` first."
            )

        # Pre-load known source ids so we can flag rows that reference a
        # source that doesn't exist yet (rather than FK-failing on insert).
        known = {r["id"] for r in db.conn.execute("SELECT id FROM source")}

        with path.open(encoding="utf-8") as f:
            for ln, raw in enumerate(f, 1):
                outcome = _import_mining_log_record(db, raw, ln, known, apply_changes)
                if outcome == "inserted":
                    inserted += 1
                elif outcome == "skipped":
                    skipped += 1
                elif outcome is not None:
                    errors.append(outcome)

    _echo_mining_log_summary(inserted, skipped, errors, apply_changes)


def _echo_mining_log_summary(
    inserted: int, skipped: int, errors: list[str], apply_changes: bool
) -> None:
    """Print the import counts to stderr: inserted/would-insert, skipped
    duplicates (apply only), and up to 20 errors; plus the dry-run hint."""
    verb = "inserted" if apply_changes else "would insert"
    click.echo(f"{verb}: {inserted}", err=True)
    if apply_changes:
        click.echo(f"skipped (duplicate): {skipped}", err=True)
    if errors:
        click.echo(f"errors: {len(errors)}", err=True)
        for e in errors[:20]:
            click.echo(f"  {e}", err=True)
        if len(errors) > 20:
            click.echo(f"  ... +{len(errors) - 20} more", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``import-mining-log`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_import_mining_log)
