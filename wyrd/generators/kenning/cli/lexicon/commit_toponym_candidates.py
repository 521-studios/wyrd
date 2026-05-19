"""``wyrd kenning lexicon commit-toponym-candidates`` — commit reviewed toponym candidates to the lexicon."""

from __future__ import annotations

import json
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("commit-toponym-candidates")
@click.option(
    "--jsonl",
    "triage_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Operator-edited triage JSONL (from prepare-toponym-candidates).",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Lexicon SQLite DB.",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Write to DB. Default: dry-run reports predicted counts only.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print per-row decisions to stderr (one line per non-defer row).",
)
def lexicon_commit_toponym_candidates(
    triage_path: Path,
    db_path: Path,
    apply: bool,
    verbose: bool,
) -> None:
    """Apply the operator's triage decisions (wyrd-x82p Phase 2b.3).

    Reads the edited triage JSONL. For each row:

    * ``action: map`` — writes a ``toponym_attestation`` row pointing
      at ``toponym_id``.
    * ``action: create`` — creates a new ``toponym`` row from
      ``create_modern_name`` / ``create_country`` / ``create_region``
      and a matching attestation. If the (name, country, region)
      tuple already exists (UNIQUE-index collision), the decision is
      treated as a MAP to the existing id — prevents accidental
      duplicate toponym rows.
    * ``action: skip`` — no DB write.
    * ``action: defer`` — no DB write; left for a future pass.

    Idempotent for the ``map`` path via the UNIQUE index on
    ``toponym_attestation``. Re-running with the same triage JSONL
    is a no-op for already-applied rows.
    """
    from wyrd.generators.kenning.toponym_candidate_review import (
        commit_triage_decisions,
    )

    click.echo(f"Using DB {db_path}", err=True)
    # Read the whole JSONL into memory — pilot scale is hundreds to
    # thousands of rows; streaming isn't worth the complexity here.
    rows: list[dict] = []
    with triage_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise click.ClickException(f"{triage_path}:{line_no}: invalid JSON: {e}") from e
    click.echo(f"Loaded {len(rows):,} triage rows", err=True)

    with LexiconDB(db_path) as db:
        report = commit_triage_decisions(db.conn, rows, apply=apply)
        if apply:
            db.conn.commit()

    if verbose:
        for idx, msg in report.error_records:
            click.echo(f"  row {idx}: ERROR — {msg}", err=True)
        # Surface CREATE → MAP demotions per row so the operator
        # sees which of their CREATEs collided with an existing
        # toponym (silently demoting was a load-bearing UX gap
        # otherwise).
        for idx, tid, name in report.demoted_records:
            click.echo(
                f"  row {idx}: CREATE→MAP — {name!r} collides with existing toponym {tid}",
                err=True,
            )

    # Warn when most rows are still at the default-defer placeholder:
    # operator may have run commit on an unedited triage file. Gated
    # on processed >= 5 so a single-row "I deferred this on purpose"
    # invocation doesn't get a noisy 100% warning.
    if report.deferred > 0 and report.processed >= 5:
        defer_ratio = report.deferred / report.processed
        if defer_ratio >= 0.8:
            click.echo(
                f"warning: {report.deferred}/{report.processed} "
                f"({defer_ratio:.0%}) rows are still at action=defer — "
                f"if this is unintended, the triage file may not have been edited yet",
                err=True,
            )

    click.echo("", err=True)
    click.echo(
        f"TOTAL processed={report.processed} "
        f"mapped={report.mapped} "
        f"created={report.created} "
        f"demoted={report.demoted_count} "
        f"skipped={report.skipped} "
        f"deferred={report.deferred} "
        f"errors={report.errors} "
        f"({'APPLIED' if apply else 'dry-run'})",
        err=True,
    )
    if report.errors and not verbose:
        click.echo(
            "  (re-run with --verbose to see per-row error messages)",
            err=True,
        )


def add_to(parent: click.Group) -> None:
    """Register ``commit-toponym-candidates`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_commit_toponym_candidates)
