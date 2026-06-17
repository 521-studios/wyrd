"""``wyrd kenning lexicon retag-celtic-branch`` — one-time re-tag of flat ``celtic``
L2 etymons to their branch (brittonic/goidelic) — wyrd-xdam.2."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command("retag-celtic-branch")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Directory of L2 JSONL files to re-tag in place.",
)
@click.option("--dry-run", is_flag=True, help="Report what would change without writing files.")
def lexicon_retag_celtic_branch(data_dir: Path, dry_run: bool) -> None:
    """Re-tag clean single-branch ``celtic`` etymons → brittonic/goidelic (wyrd-xdam.2).

    Branch is derived from the countries of the toponyms that use each etymon
    (Goidelic nations → goidelic; Wales/England-substrate/Breton-France →
    brittonic). Mixed-branch, unknown-country, and unattached celtic etymons stay
    ``celtic``. Two passes: build the global branch map across all files, then
    rewrite each file (etymon language + ref, and every cascading celtic:<form>
    ref). Idempotent; only files with a change are rewritten.
    """
    from wyrd.generators.kenning.jsonl.log import write_jsonl
    from wyrd.generators.kenning.lexicon.celtic_retag import (
        RetagReport,
        build_branch_map,
        retag_rows,
        summarize,
    )

    files = sorted(data_dir.glob("*.jsonl"))
    per_file: dict[Path, list[dict]] = {}
    all_rows: list[dict] = []
    for path in files:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        per_file[path] = rows
        all_rows.extend(rows)

    distinct_celtic = len(
        {
            r.get("ref")
            for r in all_rows
            if r.get("_type") == "etymon" and r.get("language") == "celtic"
        }
    )
    branch_map = build_branch_map(all_rows)
    report = RetagReport()
    summarize(branch_map, distinct_celtic, report)
    click.echo(
        f"Branch map: {report.to_goidelic} → goidelic, {report.to_brittonic} → brittonic, "
        f"{report.stayed_celtic}/{report.distinct_celtic} stay celtic "
        "(mixed / unknown-country / unattached).",
        err=True,
    )

    changed_files = 0
    for i, (path, rows) in enumerate(per_file.items(), start=1):
        if retag_rows(rows, branch_map, report):
            changed_files += 1
            if not dry_run:
                write_jsonl(path, rows)
        if i % 10 == 0 or i == len(files):
            click.echo(
                f"  [{i}/{len(files)}]  records rewritten={report.records_rewritten}", err=True
            )

    verb = "would change" if dry_run else "changed"
    click.echo(
        f"\nDone. {verb} {changed_files}/{len(files)} files; "
        f"{report.records_rewritten} records rewritten "
        f"({report.to_goidelic} etymons→goidelic, {report.to_brittonic}→brittonic).",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``retag-celtic-branch`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_retag_celtic_branch)
