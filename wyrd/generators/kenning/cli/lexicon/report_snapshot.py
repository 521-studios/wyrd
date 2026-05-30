"""``wyrd kenning lexicon ingest-report-snapshot`` / ``report-snapshot-diff``
— port file-based dashboard reports into the queryable report tables (wyrd-d24).
"""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.lexicon.report_snapshot import (
    diff_snapshots,
    ingest_snapshot_to_db,
    parse_report_dir,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("ingest-report-snapshot")
@click.argument(
    "report_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--label", required=True, help="Snapshot label, e.g. 'before-rebuild-20260529'.")
@click.option(
    "--captured-at",
    default=None,
    help="ISO timestamp for the snapshot. Defaults to the report dir's mtime.",
)
@click.option("--notes", default=None, help="Free-text note stored on the snapshot row.")
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
    help="Write rows. Without this, dry-run: print the parsed metric count per category.",
)
def lexicon_ingest_report_snapshot(
    report_dir: Path,
    label: str,
    captured_at: str | None,
    notes: str | None,
    db_path: Path,
    apply_changes: bool,
) -> None:
    """Parse a dashboard report directory into the ``report_snapshot`` /
    ``report_metric`` tables (wyrd-d24).

    Reads the structured artifacts (stats.txt, enrichment_status.txt,
    era_coverage.txt, rando_port_readiness.txt, empirical_priors.json,
    drift_*.json) and stores each as a ``(category, metric, value_num,
    value_text)`` row keyed by ``--label``, so before/after comparisons are
    a SQL JOIN. Idempotent: re-ingesting a label replaces its metric rows.
    """
    metrics = parse_report_dir(report_dir)
    by_cat: dict[str, int] = {}
    for m in metrics:
        by_cat[m.category] = by_cat.get(m.category, 0) + 1

    click.echo(f"parsed {len(metrics)} metric(s) from {report_dir}:", err=True)
    for cat, n in sorted(by_cat.items()):
        click.echo(f"  {cat:14} {n}", err=True)

    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)
        return

    if captured_at is None:
        # Resolve a deterministic timestamp from the report dir's mtime so
        # re-ingesting the same committed dir is byte-stable.
        from datetime import UTC, datetime

        captured_at = datetime.fromtimestamp(report_dir.stat().st_mtime, tz=UTC).isoformat()

    n = ingest_snapshot_to_db(
        db_path,
        label=label,
        captured_at=captured_at,
        metrics=metrics,
        source_dir=str(report_dir),
        notes=notes,
    )
    click.echo(f"wrote snapshot {label!r}: {n} metric row(s)", err=True)


@click.command("report-snapshot-diff")
@click.argument("label_a")
@click.argument("label_b")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_report_snapshot_diff(label_a: str, label_b: str, db_path: Path) -> None:
    """Diff two stored snapshots' metrics (wyrd-d24). Prints rows that
    changed between ``LABEL_A`` and ``LABEL_B`` (or exist in only one),
    with the numeric delta where both sides are numeric."""
    with _readonly_lexicon(db_path) as conn:
        deltas = diff_snapshots(conn, label_a, label_b)
    if not deltas:
        click.echo(f"no metric differences between {label_a!r} and {label_b!r}")
        return
    click.echo(f"{'category':<12} {'metric':<32} {label_a:>16} {label_b:>16} {'Δ':>12}")
    for d in deltas:
        delta = f"{d.delta_num:+.4g}" if d.delta_num is not None else ""
        click.echo(
            f"{d.category:<12} {d.metric:<32} {str(d.a_text or '—'):>16} "
            f"{str(d.b_text or '—'):>16} {delta:>12}"
        )


def add_to(parent: click.Group) -> None:
    """Register the report-snapshot commands on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_ingest_report_snapshot)
    parent.add_command(lexicon_report_snapshot_diff)
