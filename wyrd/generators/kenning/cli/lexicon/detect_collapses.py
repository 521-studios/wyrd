"""``wyrd kenning lexicon detect-collapses`` — emit deterministic form-of /
variant collapse rows to the ``_collapses.jsonl`` ledger (wyrd-y651 P2b)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click


@click.command("detect-collapses")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Lexicon DB to scan (default: $WYRD_LEXICON_DB or ~/.wyrd/lexicon.db).",
)
@click.option(
    "--collapse-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_collapses.jsonl"),
    show_default=True,
    help="Ledger to append collapse rows to. Created if absent.",
)
@click.option("--dry-run", is_flag=True, default=False, help="Preview; write nothing.")
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Cap the number of new rows emitted (safety valve).",
)
def lexicon_detect_collapses(
    db_path: Path | None, collapse_file: Path, dry_run: bool, limit: int | None
) -> None:
    """Detect the DETERMINISTIC form-of / variant collapses and append them
    to the L2 collapse ledger (wyrd-y651, the consolidation epic).

    Two high-confidence methods (see ``lexicon.collapse_detect``):
    (A) variant-doublings that share an exact gloss with their lemma, and
    (B) pure form-of etymons whose pointer gloss names a resolvable lemma.
    The ambiguous bulk is left for the LLM merge pass (P3).

    Rows already present in the ledger are skipped (last-write-wins
    semantics mean a re-run only adds newly-detected refs). The emitted
    rows are replayed for free by ``run_full_enrichment`` on the next
    rebuild — this command itself never runs at rebuild time.
    """
    from wyrd.generators.kenning.cli.utils import _readonly_lexicon
    from wyrd.generators.kenning.lexicon.collapse_detect import (
        detect_deterministic_collapses,
    )

    if db_path is None:
        env_path = os.environ.get("WYRD_LEXICON_DB")
        db_path = Path(env_path) if env_path else Path.home() / ".wyrd" / "lexicon.db"
    if not db_path.exists():
        raise click.ClickException(f"Lexicon DB not found: {db_path}")

    click.echo(f"Scanning {db_path} for deterministic collapses...", err=True)
    with _readonly_lexicon(db_path) as conn:
        detected = detect_deterministic_collapses(conn)

    # Skip refs already recorded (idempotent re-runs).
    existing: set[str] = set()
    if collapse_file.exists():
        for line in collapse_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_type") == "collapse" and row.get("ref"):
                existing.add(row["ref"])

    new_rows = [r for r in detected if r["ref"] not in existing]
    if limit is not None:
        new_rows = new_rows[:limit]

    by_method: dict[str, int] = {}
    for r in new_rows:
        by_method[r["method"]] = by_method.get(r["method"], 0) + 1
    click.echo(
        f"  detected={len(detected)} already-recorded={len(detected) - len([r for r in detected if r['ref'] not in existing])} "
        f"new={len(new_rows)}",
        err=True,
    )
    for method, n in sorted(by_method.items()):
        click.echo(f"    {method}: {n}", err=True)

    if not new_rows:
        click.echo("Nothing new to emit.", err=True)
        return
    if dry_run:
        for r in new_rows[:20]:
            click.echo(f"  [dry-run] {r['ref']} -> {r['into']} ({r['method']})", err=True)
        if len(new_rows) > 20:
            click.echo(f"  ... +{len(new_rows) - 20} more", err=True)
        click.echo(f"DRY-RUN: would append {len(new_rows)} rows to {collapse_file}", err=True)
        return

    # Each L2 file needs exactly one source row (id matches the synthetic
    # source apply_collapses uses for variant rows).
    if not collapse_file.exists():
        collapse_file.parent.mkdir(parents=True, exist_ok=True)
        with collapse_file.open("w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "_type": "source",
                        "ref": "collapse",
                        "title": "Etymon collapse ledger (wyrd-y651)",
                        "notes": (
                            "Form-of / variant etymons folded into their lemma. "
                            "Replayed by run_full_enrichment's curation slot. "
                            "Deterministic rows from detect-collapses; LLM rows "
                            "from the merge pass. Revert with into: ''."
                        ),
                    },
                    ensure_ascii=False,
                )
            )
            fh.write("\n")

    with collapse_file.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(new_rows, 1):
            fh.write(
                json.dumps(
                    {
                        "_type": "collapse",
                        "ref": r["ref"],
                        "into": r["into"],
                        "variant_class": r["variant_class"],
                        "method": r["method"],
                    },
                    ensure_ascii=False,
                )
            )
            fh.write("\n")
            if i % 25 == 0:
                click.echo(f"  appended [{i}/{len(new_rows)}]", err=True)

    click.echo(f"Appended {len(new_rows)} collapse rows → {collapse_file}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``detect-collapses`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_detect_collapses)
