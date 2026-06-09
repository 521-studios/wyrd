"""``wyrd kenning lexicon merge-lemmas`` — LLM same-morpheme judgment over
same-language cross-cluster minor↔major candidates (v3ow P3 cross-cluster
same-meaning merge; sibling of ``fold-variants`` which folds barren duplicates).
A confident same-morpheme verdict becomes a real ``into`` fold (tombstoning the
minor lineage into the major); anything else is a no-op recording the judgment
so a re-run skips the pair. Verdicts round-trip ``_collapses.jsonl`` and replay
deterministically at rebuild — the LLM is NEVER re-run at rebuild.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click


@dataclass
class _MergeLemmaRun:
    """Shared state for the per-candidate merge-judge loop: the LLM client, the
    output handle (``None`` in ``--dry-run``), the dry-run flag, the merge
    confidence threshold, and the mutable merges/rejects/skipped counters."""

    client: Any
    fh: Any
    dry_run: bool
    min_confidence: str
    counts: dict[str, int]


def _load_judged(collapse_file: Path) -> tuple[set[tuple[str, str]], set[str]]:
    """Parse the ledger for already-judged meaning-merge state. Returns
    ``(judged, merged_refs)``: ``judged`` is the set of (minor, major) PAIRS
    judged (method-tagged ``llm-meaning-``); ``merged_refs`` is the set of
    minor refs that were actually merged (a merge is TERMINAL — a merged minor
    is never re-judged, so no later reject cancels it at replay)."""
    judged: set[tuple[str, str]] = set()
    merged_refs: set[str] = set()
    if collapse_file.exists():
        with collapse_file.open(encoding="utf-8") as fled:  # stream — the ledger grows
            for line in fled:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    row.get("_type") == "collapse"
                    and row.get("ref")
                    and str(row.get("method", "")).startswith("llm-meaning-")
                ):
                    judged.add((row["ref"], row.get("candidate_lemma") or row.get("into") or ""))
                    if row.get("into"):
                        merged_refs.add(row["ref"])
    return judged, merged_refs


def _select_todo(
    candidates: list, merged_refs: set[str], judged: set[tuple[str, str]], limit: int | None
) -> list:
    """Candidates whose minor isn't already merged and whose (minor, major)
    pair isn't already judged, capped at ``limit`` when set."""
    todo = [
        c
        for c in candidates
        if c.minor_ref not in merged_refs and (c.minor_ref, c.major_ref) not in judged
    ]
    if limit is not None:
        todo = todo[:limit]
    return todo


def _judge_and_record(ctx: _MergeLemmaRun, c: Any) -> None:
    """Judge one merge candidate with the LLM (one retry with a brief backoff on
    transport/parse failure) and either print (``--dry-run``) or append its
    verdict row to the ledger. A transient failure is skipped WITHOUT recording
    (logging why), so a later run retries rather than burning the pair."""
    from wyrd.generators.kenning.lexicon.meaning_merge_detect import (
        build_merge_judge_prompt,
        merge_verdict_to_row,
        parse_merge_verdict,
    )

    system, user = build_merge_judge_prompt(c)
    verdict = None
    last_err: Exception | None = None
    for attempt in range(2):
        if attempt:  # brief backoff before the single retry
            time.sleep(1.0)
        try:
            verdict = parse_merge_verdict(ctx.client.chat_json(system, user, {}))
        except Exception as e:
            last_err = e
            verdict = None
        if verdict is not None:
            break
    if verdict is None:
        ctx.counts["skipped"] += 1
        why = last_err if last_err is not None else "unparseable verdict"
        click.echo(f"  [skip] {c.minor_form}->{c.major_form}: {why}", err=True)
        return
    row = merge_verdict_to_row(c, verdict, ctx.min_confidence)
    is_merge = row["into"] != ""
    ctx.counts["merges"] += is_merge
    ctx.counts["rejects"] += not is_merge
    if ctx.dry_run:
        tag = "MERGE" if is_merge else "skip"
        click.echo(
            f"  [{tag}] {c.minor_form}({c.minor_cluster_size})->{c.major_form}"
            f"({c.major_cluster_size}) {verdict.confidence} ({verdict.reason})",
            err=True,
        )
    else:
        ctx.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        ctx.fh.flush()


@click.command("merge-lemmas")
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
    help="Ledger to append verdicts to (merges + no-op rejections).",
)
@click.option("--model", default="gemma4:26b", show_default=True, help="Ollama model.")
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
    help="Minimum LLM confidence to actually MERGE a same-morpheme verdict.",
)
@click.option("--min-similarity", type=float, default=0.7, show_default=True)
@click.option("--minor-max", type=int, default=10, show_default=True)
@click.option("--major-min", type=int, default=40, show_default=True)
@click.option("--ratio", type=int, default=5, show_default=True)
@click.option("--max-per-gloss", type=int, default=600, show_default=True)
@click.option("--limit", type=int, default=None, help="Cap candidates judged this run.")
@click.option("--dry-run", is_flag=True, default=False, help="Judge + print; write nothing.")
def lexicon_merge_lemmas(
    db_path: Path | None,
    collapse_file: Path,
    model: str,
    ollama_url: str | None,
    min_confidence: str,
    min_similarity: float,
    minor_max: int,
    major_min: int,
    ratio: int,
    max_per_gloss: int,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Judge same-language cross-cluster merge candidates with an LLM."""
    from wyrd.generators.kenning.cli.utils import _readonly_lexicon
    from wyrd.generators.kenning.extractors.llm import OllamaClient
    from wyrd.generators.kenning.lexicon.meaning_merge_detect import (
        detect_meaning_merge_candidates,
    )

    if db_path is None:
        env_path = os.environ.get("WYRD_LEXICON_DB")
        db_path = Path(env_path) if env_path else Path.home() / ".wyrd" / "lexicon.db"
    if not db_path.exists():
        raise click.ClickException(f"Lexicon DB not found: {db_path}")
    base_url = ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")

    click.echo(f"Detecting meaning-merge candidates in {db_path}...", err=True)
    with _readonly_lexicon(db_path) as conn:
        candidates = detect_meaning_merge_candidates(
            conn,
            min_similarity=min_similarity,
            minor_max=minor_max,
            major_min=major_min,
            ratio=ratio,
            max_per_gloss=max_per_gloss,
        )

    judged, merged_refs = _load_judged(collapse_file)
    todo = _select_todo(candidates, merged_refs, judged, limit)
    click.echo(
        f"  candidates={len(candidates)} already-judged={len(candidates) - len(todo)} "
        f"to-judge={len(todo)} (model={model}, min-confidence={min_confidence})",
        err=True,
    )
    if not todo:
        click.echo("Nothing to judge.", err=True)
        return

    if not dry_run:
        collapse_file.parent.mkdir(parents=True, exist_ok=True)
    ctx = _MergeLemmaRun(
        client=OllamaClient(base_url=base_url, model=model, timeout_s=90.0),
        fh=None if dry_run else collapse_file.open("a", encoding="utf-8"),
        dry_run=dry_run,
        min_confidence=min_confidence,
        counts={"merges": 0, "rejects": 0, "skipped": 0},
    )
    start = time.perf_counter()
    try:
        for i, c in enumerate(todo, 1):
            _judge_and_record(ctx, c)
            if i % 10 == 0 or i == len(todo):
                rate = (time.perf_counter() - start) / i
                click.echo(
                    f"  [{i}/{len(todo)}]  merges={ctx.counts['merges']} rejects={ctx.counts['rejects']} "
                    f"skipped={ctx.counts['skipped']} ({rate:.1f}s/entry)",
                    err=True,
                )
    finally:
        if ctx.fh is not None:
            ctx.fh.close()
    click.echo(
        f"{'(dry-run) ' if dry_run else ''}recorded: {ctx.counts['merges']} merges + "
        f"{ctx.counts['rejects']} rejections ({ctx.counts['skipped']} skipped) → {collapse_file}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    parent.add_command(lexicon_merge_lemmas)
