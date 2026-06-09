"""``wyrd kenning lexicon merge-collapses`` — LLM same-meaning judgment over
variant-doubling candidates, emitting verdicts to the _collapses.jsonl ledger
(wyrd-6xzs, consolidation epic P3)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click


@dataclass
class _MergeRun:
    """Shared state for the per-candidate judge loop: the LLM client, the
    output handle (``None`` in ``--dry-run``), the dry-run flag, the fold
    confidence threshold, and the mutable folds/rejects/skipped counters."""

    client: Any
    fh: Any
    dry_run: bool
    min_confidence: str
    counts: dict[str, int]


def _load_judged(collapse_file: Path) -> set[str]:
    """Refs already in the ledger (a positive collapse OR a prior no-op
    rejection) — skipped on re-run so each pair is judged once."""
    judged: set[str] = set()
    if collapse_file.exists():
        for line in collapse_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_type") == "collapse" and row.get("ref"):
                judged.add(row["ref"])
    return judged


def _select_todo(candidates: list, judged: set[str], limit: int | None) -> list:
    """Candidates not yet in the ledger, capped at ``limit`` when set."""
    todo = [c for c in candidates if c.from_ref not in judged]
    if limit is not None:
        todo = todo[:limit]
    return todo


def _judge_and_record(ctx: _MergeRun, c: Any) -> None:
    """Judge one candidate with the LLM (one retry on transport/parse failure)
    and either print (``--dry-run``) or append its verdict row to the ledger.
    A transient failure is skipped WITHOUT recording, so a later run retries
    rather than burning the pair on noise."""
    from wyrd.generators.kenning.lexicon.collapse_merge import (
        build_judge_prompt,
        parse_verdict,
        verdict_to_row,
    )

    system, user = build_judge_prompt(c)
    verdict = None
    for _attempt in range(2):  # one retry on transport/parse failure
        try:
            verdict = parse_verdict(ctx.client.chat_json(system, user, {}))
        except Exception:
            verdict = None
        if verdict is not None:
            break
    if verdict is None:
        ctx.counts["skipped"] += 1
        return
    row = verdict_to_row(c, verdict, ctx.min_confidence)
    is_fold = row["into"] != ""
    ctx.counts["folds"] += is_fold
    ctx.counts["rejects"] += not is_fold
    if ctx.dry_run:
        tag = "FOLD" if is_fold else "skip"
        click.echo(
            f"  [{tag}] {c.from_form}->{c.into_form} {verdict.confidence} ({verdict.reason})",
            err=True,
        )
    else:
        ctx.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        ctx.fh.flush()  # crash-safety: each verdict is durable as we go


@click.command("merge-collapses")
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
    help="Ledger to append verdicts to (positives + no-op rejections).",
)
@click.option(
    "--model",
    default="gemma4:26b",
    show_default=True,
    help="Ollama model. gemma4:26b judges this task best; qwen2.5:7b is a structured-JSON fallback.",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Ollama base URL (default: $WYRD_OLLAMA_URL or http://localhost:11434).",
)
@click.option(
    "--min-confidence",
    type=click.Choice(["low", "medium", "high"]),
    default="medium",
    show_default=True,
    help="Minimum LLM confidence to actually FOLD a 'same meaning' verdict.",
)
@click.option("--limit", type=int, default=None, help="Cap candidates judged this run.")
@click.option("--dry-run", is_flag=True, default=False, help="Judge + print; write nothing.")
def lexicon_merge_collapses(
    db_path: Path | None,
    collapse_file: Path,
    model: str,
    ollama_url: str | None,
    min_confidence: str,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Judge the no-shared-gloss variant-doublings with an LLM and record the
    verdicts in the _collapses.jsonl ledger (wyrd-6xzs).

    The deterministic detector (detect-collapses) handles doublings that
    share an exact gloss; this handles the rest, where deciding "same lexical
    item?" needs reading the two gloss sets. A confident same-meaning verdict
    becomes a real collapse; anything else is a no-op ``into: ""`` row that
    records the judgment so a re-run skips the pair. The LLM is non-
    deterministic, so its verdicts round-trip through L2 and are replayed
    deterministically — this command is NEVER re-run at rebuild time (that
    would make rebuilds non-reproducible).
    """
    from wyrd.generators.kenning.cli.utils import _readonly_lexicon
    from wyrd.generators.kenning.extractors.llm import OllamaClient
    from wyrd.generators.kenning.lexicon.collapse_merge import detect_merge_candidates

    if db_path is None:
        env_path = os.environ.get("WYRD_LEXICON_DB")
        db_path = Path(env_path) if env_path else Path.home() / ".wyrd" / "lexicon.db"
    if not db_path.exists():
        raise click.ClickException(f"Lexicon DB not found: {db_path}")
    base_url = ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")

    click.echo(f"Detecting merge candidates in {db_path}...", err=True)
    with _readonly_lexicon(db_path) as conn:
        candidates = detect_merge_candidates(conn)

    # Skip pairs already judged (any ref already in the ledger — a positive
    # collapse OR a prior no-op rejection).
    judged = _load_judged(collapse_file)
    todo = _select_todo(candidates, judged, limit)
    already_judged = len(candidates) - len([c for c in candidates if c.from_ref not in judged])
    click.echo(
        f"  candidates={len(candidates)} already-judged={already_judged} "
        f"to-judge={len(todo)} (model={model}, min-confidence={min_confidence})",
        err=True,
    )
    if not todo:
        click.echo("Nothing to judge.", err=True)
        return

    ctx = _MergeRun(
        client=OllamaClient(base_url=base_url, model=model, timeout_s=90.0),
        fh=None if dry_run else collapse_file.open("a", encoding="utf-8"),
        dry_run=dry_run,
        min_confidence=min_confidence,
        counts={"folds": 0, "rejects": 0, "skipped": 0},
    )
    start = time.perf_counter()
    try:
        for i, c in enumerate(todo, 1):
            _judge_and_record(ctx, c)
            if i % 10 == 0 or i == len(todo):
                rate = (time.perf_counter() - start) / i
                click.echo(
                    f"  [{i}/{len(todo)}]  folds={ctx.counts['folds']} rejects={ctx.counts['rejects']} "
                    f"skipped={ctx.counts['skipped']} ({rate:.1f}s/entry)",
                    err=True,
                )
    finally:
        if ctx.fh is not None:
            ctx.fh.close()

    verb = "would record" if dry_run else "recorded"
    click.echo(
        f"{verb}: {ctx.counts['folds']} folds + {ctx.counts['rejects']} rejections "
        f"({ctx.counts['skipped']} skipped) → {collapse_file}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``merge-collapses`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_merge_collapses)
