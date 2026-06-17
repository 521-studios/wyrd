"""``wyrd kenning lexicon normalize-l2-vocab`` — one-time Class-A region/country
normalization of the committed L2 JSONL (wyrd-3q6m.4)."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command("normalize-l2-vocab")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Directory of L2 JSONL files to normalize in place.",
)
@click.option(
    "--kepn-source",
    type=click.Path(path_type=Path),
    default=Path.home() / ".wyrd" / "sources" / "kepn",
    show_default=True,
    help="kepn source dir (for yorkshire.csv → Riding recode). Skipped if absent.",
)
@click.option("--dry-run", is_flag=True, help="Report changes without writing files.")
def lexicon_normalize_l2_vocab(data_dir: Path, kepn_source: Path, dry_run: bool) -> None:
    """Rewrite region + country to canonical across ``data-dir`` (wyrd-3q6m.4, D49).

    Idempotent and safe in place (L2 is git-tracked; region/country are
    provenance, not D21 evidence). Applies the Class-A region folds, the
    Yorkshire→Riding provenance recode (needs ``yorkshire.csv``), and country
    normalization (Republic of Ireland→Ireland, The Isle of Man→Isle of Man,
    Brittany→France, null→derive-from-region). Unfoldable region values
    (zones/quarantine, no-Riding Yorkshire) are left unchanged and reported.
    Only files with a change are rewritten.
    """
    from wyrd.generators.kenning.jsonl.log import write_jsonl
    from wyrd.generators.kenning.lexicon.region_normalize import (
        NormReport,
        build_riding_map,
        normalize_rows,
    )

    riding_map: dict[str, str] = {}
    yorkshire_csv = kepn_source / "yorkshire.csv"
    if yorkshire_csv.exists():
        riding_map = build_riding_map(yorkshire_csv)
        click.echo(f"Loaded {len(riding_map)} Yorkshire place→Riding mappings.", err=True)
    else:
        click.echo(
            f"No {yorkshire_csv} — skipping Yorkshire→Riding recode (mechanical folds only).",
            err=True,
        )

    report = NormReport()
    files = sorted(data_dir.glob("*.jsonl"))
    changed_files = 0
    for i, path in enumerate(files, start=1):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        before = [json.dumps(r, ensure_ascii=False, sort_keys=False) for r in rows]
        normalize_rows(rows, riding_map=riding_map, report=report)
        after = [json.dumps(r, ensure_ascii=False, sort_keys=False) for r in rows]
        if before != after:
            changed_files += 1
            if not dry_run:
                write_jsonl(path, rows)
        if i % 25 == 0 or i == len(files):
            click.echo(
                f"  [{i}/{len(files)}]  toponyms={report.toponyms_seen} "
                f"folded={report.region_folded} ridings={report.yorkshire_recoded} "
                f"country={report.country_aliased + report.country_derived} "
                f"refs={report.refs_rewritten}",
                err=True,
            )

    verb = "would change" if dry_run else "changed"
    click.echo(
        f"\nDone. {verb} {changed_files}/{len(files)} files.\n"
        f"  region folds: {report.region_folded}\n"
        f"  Yorkshire→Riding: {report.yorkshire_recoded}  "
        f"(left at county, no Riding provenance: {report.yorkshire_left_county})\n"
        f"  country aliased: {report.country_aliased}  derived-from-region: {report.country_derived}\n"
        f"  refs rewritten: {report.refs_rewritten}",
        err=True,
    )
    if report.unfoldable:
        click.echo("  left unchanged (deferred — not a node/alias):", err=True)
        for value, count in sorted(report.unfoldable.items(), key=lambda kv: -kv[1]):
            click.echo(f"    {count:6}  {value!r}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``normalize-l2-vocab`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_normalize_l2_vocab)
