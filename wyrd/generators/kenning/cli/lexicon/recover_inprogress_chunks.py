"""``wyrd kenning lexicon recover-inprogress-chunks`` — promote
mid-source crash artifacts left behind by ``mine-toponym-mentions-staged``
into canonical per-source JSONLs (wyrd-w7i3)."""

from __future__ import annotations

import json
from pathlib import Path

import click

# Local-import the in-progress conventions to keep them defined exactly
# once. If the directory/suffix ever changes, the staged miner and the
# recovery tool stay in lockstep.
from wyrd.generators.kenning.cli.lexicon.mine_toponym_mentions_staged import (
    _INPROGRESS_SUBDIR,
    _INPROGRESS_SUFFIX,
)


def _dedup_key(row: dict) -> tuple[str, int | None, str | None, str]:
    """Dedup key matches ``_run_resume_from_failures``'s logic exactly
    so a fresh-mining→crash→recover and a resume→crash→recover both
    converge on the same canonical file shape."""
    return (
        row.get("form", ""),
        row.get("date_year"),
        row.get("region_hint"),
        row.get("context", ""),
    )


def _read_jsonl_rows(path: Path) -> tuple[list[str], int]:
    """Read a JSONL file as ``(verbatim_lines, malformed_count)``.

    Preserves the original line text so stage-specific fields like
    ``extractor`` survive the round-trip unchanged. Malformed (non-JSON
    or non-dict) rows are dropped from the output — the operator sees
    a count on stderr.
    """
    lines: list[str] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(row, dict):
                malformed += 1
                continue
            lines.append(stripped)
    return lines, malformed


def _atomic_write_lines(out_path: Path, lines: list[str]) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as sink:
        for line in lines:
            sink.write(line + "\n")
    tmp_path.replace(out_path)


@click.command("recover-inprogress-chunks")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/mining/phase2"),
    show_default=True,
    help="Directory containing canonical <source>.jsonl outputs and the "
    "_inprogress/ subdir produced by mine-toponym-mentions-staged.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be promoted/merged without writing.",
)
def lexicon_recover_inprogress_chunks(output_dir: Path, dry_run: bool) -> None:
    """Promote in-progress chunk logs from a crashed/killed mining run
    into the canonical per-source JSONLs.

    The ``mine-toponym-mentions-staged`` CLI streams each successful
    chunk's mentions to ``<output_dir>/_inprogress/<source>.chunks.jsonl``
    and fsyncs after every chunk. SIGTERM or crash mid-source leaves
    that log behind. This command:

    1. PROMOTE: if the canonical ``<source>.jsonl`` does NOT exist,
       atomic-rename the in-progress log into place. Fresh-mining
       crash → no data loss.

    2. MERGE: if the canonical ``<source>.jsonl`` DOES exist (resume-
       from-failures crash), build the union of canonical rows + new
       rows from the in-progress log (deduped on
       form/date_year/region_hint/context) and atomic-write back.

    Idempotent: re-running after success is a no-op.
    """
    inprogress_dir = output_dir / _INPROGRESS_SUBDIR
    if not inprogress_dir.exists():
        click.echo(f"No in-progress logs under {inprogress_dir}", err=True)
        return

    candidates = sorted(inprogress_dir.glob(f"*{_INPROGRESS_SUFFIX}"))
    if not candidates:
        click.echo(f"No in-progress logs under {inprogress_dir}", err=True)
        return

    promoted = 0
    merged = 0
    empty = 0
    for ip_path in candidates:
        source_id = ip_path.name[: -len(_INPROGRESS_SUFFIX)]
        out_path = output_dir / f"{source_id}.jsonl"

        new_lines, new_malformed = _read_jsonl_rows(ip_path)
        if new_malformed:
            click.echo(
                f"  warning: {ip_path.name}: dropped {new_malformed} malformed row(s)",
                err=True,
            )

        if not new_lines:
            click.echo(
                f"  [{source_id}] in-progress log empty — removing",
                err=True,
            )
            if not dry_run:
                ip_path.unlink(missing_ok=True)
            empty += 1
            continue

        if not out_path.exists():
            # PROMOTE: no canonical file → in-progress log IS the
            # canonical file. Atomic rename preserves on-disk content
            # bit-for-bit.
            click.echo(
                f"  [{source_id}] PROMOTE → {out_path} ({len(new_lines)} mention(s))",
                err=True,
            )
            if not dry_run:
                ip_path.rename(out_path)
            promoted += 1
            continue

        # MERGE: build the union, deduped against existing canonical
        # rows. Preserve existing rows verbatim (stage-specific fields
        # like ``extractor`` must keep their original value).
        existing_lines, existing_malformed = _read_jsonl_rows(out_path)
        if existing_malformed:
            click.echo(
                f"  warning: {out_path.name}: dropped {existing_malformed} "
                f"malformed row(s) from canonical",
                err=True,
            )

        existing_keys: set[tuple[str, int | None, str | None, str]] = set()
        for line in existing_lines:
            try:
                existing_keys.add(_dedup_key(json.loads(line)))
            except json.JSONDecodeError:
                continue

        union: list[str] = list(existing_lines)
        dup_count = 0
        added_count = 0
        for line in new_lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _dedup_key(row)
            if key in existing_keys:
                dup_count += 1
                continue
            existing_keys.add(key)
            union.append(line)
            added_count += 1

        click.echo(
            f"  [{source_id}] MERGE → {out_path} "
            f"(added {added_count}, deduped {dup_count}, "
            f"existing {len(existing_lines)})",
            err=True,
        )
        if not dry_run:
            _atomic_write_lines(out_path, union)
            ip_path.unlink(missing_ok=True)
        merged += 1

    suffix = " (dry-run, no changes written)" if dry_run else ""
    click.echo(
        f"recover-inprogress-chunks: promoted={promoted} merged={merged} empty={empty}{suffix}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``recover-inprogress-chunks`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_recover_inprogress_chunks)
