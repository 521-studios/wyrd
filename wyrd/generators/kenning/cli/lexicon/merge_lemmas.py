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
from pathlib import Path

import click


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
        build_merge_judge_prompt,
        detect_meaning_merge_candidates,
        merge_verdict_to_row,
        parse_merge_verdict,
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
                # Only meaning-merge rows count as already-judged (method tag), at
                # the (minor, major) PAIR level; a merge is TERMINAL (a merged
                # minor is never re-judged, so no later reject cancels it at replay).
                if (
                    row.get("_type") == "collapse"
                    and row.get("ref")
                    and str(row.get("method", "")).startswith("llm-meaning-")
                ):
                    judged.add((row["ref"], row.get("candidate_lemma") or row.get("into") or ""))
                    if row.get("into"):
                        merged_refs.add(row["ref"])

    todo = [
        c
        for c in candidates
        if c.minor_ref not in merged_refs and (c.minor_ref, c.major_ref) not in judged
    ]
    if limit is not None:
        todo = todo[:limit]
    click.echo(
        f"  candidates={len(candidates)} already-judged={len(candidates) - len(todo)} "
        f"to-judge={len(todo)} (model={model}, min-confidence={min_confidence})",
        err=True,
    )
    if not todo:
        click.echo("Nothing to judge.", err=True)
        return

    client = OllamaClient(base_url=base_url, model=model, timeout_s=90.0)
    if not dry_run:
        collapse_file.parent.mkdir(parents=True, exist_ok=True)
    merges = rejects = skipped = 0
    start = time.perf_counter()
    fh = None
    try:
        if not dry_run:
            fh = collapse_file.open("a", encoding="utf-8")
        for i, c in enumerate(todo, 1):
            system, user = build_merge_judge_prompt(c)
            verdict = None
            last_err: Exception | None = None
            for attempt in range(2):
                if attempt:  # brief backoff before the single retry
                    time.sleep(1.0)
                try:
                    verdict = parse_merge_verdict(client.chat_json(system, user, {}))
                except Exception as e:
                    last_err = e
                    verdict = None
                if verdict is not None:
                    break
            if verdict is None:
                skipped += 1
                why = last_err if last_err is not None else "unparseable verdict"
                click.echo(f"  [skip] {c.minor_form}->{c.major_form}: {why}", err=True)
                continue
            row = merge_verdict_to_row(c, verdict, min_confidence)
            is_merge = row["into"] != ""
            merges += is_merge
            rejects += not is_merge
            if dry_run:
                tag = "MERGE" if is_merge else "skip"
                click.echo(
                    f"  [{tag}] {c.minor_form}({c.minor_cluster_size})->{c.major_form}"
                    f"({c.major_cluster_size}) {verdict.confidence} ({verdict.reason})",
                    err=True,
                )
            else:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            if i % 10 == 0 or i == len(todo):
                rate = (time.perf_counter() - start) / i
                click.echo(
                    f"  [{i}/{len(todo)}]  merges={merges} rejects={rejects} "
                    f"skipped={skipped} ({rate:.1f}s/entry)",
                    err=True,
                )
    finally:
        if fh is not None:
            fh.close()
    click.echo(
        f"{'(dry-run) ' if dry_run else ''}recorded: {merges} merges + {rejects} rejections "
        f"({skipped} skipped) → {collapse_file}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    parent.add_command(lexicon_merge_lemmas)
