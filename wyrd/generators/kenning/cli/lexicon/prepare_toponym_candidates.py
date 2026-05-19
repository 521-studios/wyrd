"""``wyrd kenning lexicon prepare-toponym-candidates`` — stage toponym mention candidates for operator review."""

from __future__ import annotations

import json
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("prepare-toponym-candidates")
@click.option(
    "--jsonl",
    "candidates_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL of unresolved candidates (output of ingest-toponym-mentions --candidates-out).",
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
    "--out",
    "triage_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Write triage JSONL to this path. Operator hand-edits this file, "
    "then feeds it to commit-toponym-candidates.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite --out if it exists.",
)
def lexicon_prepare_toponym_candidates(
    candidates_path: Path,
    db_path: Path,
    triage_path: Path,
    force: bool,
) -> None:
    """Prepare a triage JSONL from Phase 2b.1's unresolved-candidates
    output (wyrd-x82p Phase 2b.3).

    For each candidate, attaches the top-3 fuzzy-matched existing
    toponyms (by SequenceMatcher.ratio over normalized forms), with
    a region-hint boost when the candidate carries one. Emits a
    JSONL where each row pre-fills ``action: "defer"`` —
    operator edits the action to ``"map"`` (with ``toponym_id``),
    ``"create"`` (with ``create_modern_name`` / ``create_country``
    / ``create_region``), or ``"skip"``, then feeds the file to
    ``commit-toponym-candidates``.
    """
    from wyrd.generators.kenning.toponym_candidate_review import (
        _toponym_rows,
        prepare_candidate,
    )

    if triage_path.exists() and not force:
        raise click.ClickException(f"--out {triage_path} already exists; pass --force to overwrite")
    if triage_path.exists() and force:
        click.echo(
            f"warning: --force overwriting existing {triage_path}",
            err=True,
        )

    click.echo(f"Using DB {db_path}", err=True)
    with LexiconDB(db_path) as db:
        toponyms = _toponym_rows(db.conn)
    click.echo(f"Loaded {len(toponyms):,} toponyms for fuzzy match", err=True)

    # Stream the candidates: read input one line at a time, write
    # triage output as we go. Atomic-rename pattern (temp + replace)
    # so a mid-write crash leaves any prior triage file intact.
    tmp_path = triage_path.with_suffix(triage_path.suffix + ".tmp")
    n_prepared = 0
    n_with_suggestions = 0
    with (
        candidates_path.open("r", encoding="utf-8") as src,
        tmp_path.open("w", encoding="utf-8") as sink,
    ):
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"{candidates_path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(raw, dict):
                raise click.ClickException(f"{candidates_path}:{line_no}: row is not a JSON object")
            prepared = prepare_candidate(raw, toponyms)
            row_out = {
                "source_id": prepared.source_id,
                "form": prepared.form,
                "date_year": prepared.date_year,
                "region_hint": prepared.region_hint,
                "context": prepared.context,
                "suggestions": [
                    {
                        "toponym_id": s.toponym_id,
                        "modern_name": s.modern_name,
                        "region": s.region,
                        "country": s.country,
                        "fuzzy_score": s.fuzzy_score,
                    }
                    for s in prepared.suggestions
                ],
                "action": prepared.action,
                "toponym_id": prepared.toponym_id,
                "create_modern_name": prepared.create_modern_name,
                "create_country": prepared.create_country,
                "create_region": prepared.create_region,
            }
            sink.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            n_prepared += 1
            if prepared.suggestions:
                n_with_suggestions += 1
            if n_prepared % 1000 == 0:
                click.echo(f"  prepared {n_prepared:,} candidates...", err=True)
    tmp_path.replace(triage_path)
    click.echo(f"Wrote {n_prepared:,} triage rows → {triage_path}", err=True)
    click.echo(
        f"  with fuzzy suggestions: {n_with_suggestions:,} "
        f"({n_prepared - n_with_suggestions:,} likely new-toponym candidates)",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``prepare-toponym-candidates`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_prepare_toponym_candidates)
