"""``wyrd kenning lexicon ingest-toponym-mentions`` — ingest mined toponym-mention candidates into the DB."""

from __future__ import annotations

import json
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("ingest-toponym-mentions")
@click.option(
    "--jsonl",
    "jsonl_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file of mentions (output of mine-toponym-mentions).",
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
    "--source",
    "source_id_override",
    default=None,
    help="Override the source_id used for attestation rows. Default: read "
    "the source_id field from each JSONL row.",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Write to DB. Without this flag, runs in dry-run mode and prints "
    "the predicted insert/dup counts.",
)
@click.option(
    "--candidates-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write unresolved mentions to this JSONL for Phase 2b.3 operator "
    "review. Default: omit (counted but not persisted).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite --candidates-out if it exists.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print per-source breakdown to stderr.",
)
def lexicon_ingest_toponym_mentions(
    jsonl_path: Path,
    db_path: Path,
    source_id_override: str | None,
    apply: bool,
    candidates_out: Path | None,
    force: bool,
    verbose: bool,
) -> None:
    """Ingest LLM-extracted mentions (wyrd-x82p Phase 2b.1).

    Consumes the JSONL emitted by ``mine-toponym-mentions`` and writes
    ``toponym_attestation`` rows for mentions that resolve to a known
    toponym. Unresolved mentions are tallied (and optionally written
    to ``--candidates-out``) for the Phase 2b.3 review tool to
    triage.

    Resolver: matches form to toponym via Phase 1's form-to-toponym
    lookup, using ``region_hint`` to disambiguate homonyms when the
    bare form maps to multiple toponyms. Mentions that don't resolve
    are NOT inserted — Phase 2b.1 only emits attestations for
    toponyms that already exist; new-toponym creation is Phase 2b.3.

    Idempotent: re-running against the same JSONL is a no-op.
    """
    from wyrd.generators.kenning.toponym_mention_ingest import (
        IngestReport,
        build_resolver_indexes,
        ingest_mentions,
    )

    click.echo(f"Using DB {db_path}", err=True)

    if candidates_out is not None and candidates_out.exists() and not force:
        raise click.ClickException(
            f"--candidates-out {candidates_out} already exists; pass --force to overwrite"
        )
    if candidates_out is not None and candidates_out.exists() and force:
        click.echo(
            f"warning: --force overwriting existing {candidates_out}",
            err=True,
        )

    # Parse JSONL once and group by source. Each row carries its own
    # source_id (the JSONL format permits mixed sources), so we have
    # to read the whole file before dispatching per-source. The
    # groupings stay in memory; for the pilot scope (~thousands of
    # mentions) that's bounded, and the per-source SELECT bulk-load
    # downstream amortizes the parse cost. A future streaming
    # rewrite (Phase 2b.2 full-scope) would emit per-source
    # sub-batches and avoid the upfront groupby.
    mentions_by_source: dict[str, list[dict]] = {}
    total_lines = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"{jsonl_path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(row, dict):
                raise click.ClickException(
                    f"{jsonl_path}:{line_no}: row is not a JSON object ({type(row).__name__})"
                )
            sid = source_id_override or row.get("source_id")
            if not sid:
                raise click.ClickException(
                    f"{jsonl_path}:{line_no}: mention missing source_id "
                    f"and no --source override given"
                )
            mentions_by_source.setdefault(sid, []).append(row)
            total_lines += 1
            if total_lines % 10000 == 0:
                # Project mining-progress convention from CLAUDE.md —
                # surface load progress for large files.
                click.echo(
                    f"  loaded {total_lines:,} mentions...",
                    err=True,
                )

    click.echo(
        f"Loaded {total_lines:,} mentions across {len(mentions_by_source)} source(s)",
        err=True,
    )

    totals = IngestReport(source_id="TOTAL")
    all_unresolved: list[dict] = []

    with LexiconDB(db_path) as db:
        indexes = build_resolver_indexes(db.conn)
        click.echo(
            f"Built resolver indexes: {len(indexes.form_to_ids):,} forms, "
            f"{sum(1 for r in indexes.toponym_regions.values() if r):,} "
            f"toponyms with region",
            err=True,
        )
        source_count = len(mentions_by_source)
        for src_i, (sid, mentions) in enumerate(sorted(mentions_by_source.items()), start=1):
            report = ingest_mentions(db.conn, sid, mentions, indexes, apply=apply)
            totals.mentions_processed += report.mentions_processed
            totals.resolved += report.resolved
            totals.unresolved += report.unresolved
            totals.inserted += report.inserted
            totals.already_present += report.already_present
            all_unresolved.extend(report.unresolved_records)
            # Project mining-progress convention: per-source line to
            # stderr (always, not only --verbose) so a multi-source
            # run shows progress.
            click.echo(
                f"  [{src_i}/{source_count}] {sid}: "
                f"processed={report.mentions_processed} "
                f"resolved={report.resolved} unresolved={report.unresolved} "
                f"inserted={report.inserted} dup={report.already_present}",
                err=True,
            )
            if verbose and report.unresolved_records:
                # In verbose mode, show a few unresolved samples per
                # source so the operator can sense-check the failure
                # mode without opening the candidates file.
                for rec in report.unresolved_records[:5]:
                    click.echo(
                        f"      unresolved: {rec['form']!r} region_hint={rec['region_hint']!r}",
                        err=True,
                    )
        if apply:
            db.conn.commit()

    click.echo("", err=True)
    click.echo(
        f"TOTAL processed={totals.mentions_processed} "
        f"resolved={totals.resolved} "
        f"unresolved={totals.unresolved} "
        f"inserted={totals.inserted} "
        f"dup={totals.already_present} "
        f"({'APPLIED' if apply else 'dry-run'})",
        err=True,
    )

    if candidates_out is not None:
        # Atomic write: write to a sibling tempfile then rename, so a
        # mid-write crash leaves the prior file intact rather than
        # truncated. silent-failure round-1 finding (the JSONL is the
        # only artifact of the unresolved set; an interrupted write
        # under --force was a real foot-gun).
        tmp_path = candidates_out.with_suffix(candidates_out.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as sink:
            for rec in all_unresolved:
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_path.replace(candidates_out)
        click.echo(
            f"Wrote {len(all_unresolved)} unresolved → {candidates_out}",
            err=True,
        )


def add_to(parent: click.Group) -> None:
    """Register ``ingest-toponym-mentions`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_ingest_toponym_mentions)
