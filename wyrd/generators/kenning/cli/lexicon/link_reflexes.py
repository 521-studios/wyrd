"""``wyrd kenning lexicon link-reflexes`` — LLM reflex-link judgment over
cross-era candidates, emitting ``inherits`` verdicts to the _collapses.jsonl
ledger (wyrd-rogd.15).

Sibling of ``merge-collapses``: that folds same-language variants; this LINKS a
later-era reflex to its earlier-era ancestor via an inheritance edge (kept
distinct, not folded). A confident reflex verdict becomes an ``inherits`` row;
anything else is a no-op ``inherits: ""`` row that records the judgment so a
re-run skips the pair. The LLM is non-deterministic, so its verdicts round-trip
through the ledger and are replayed deterministically at rebuild — this command
is NEVER re-run at rebuild time.
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
class _LinkRun:
    """Shared state for the per-candidate reflex-judge loop: the LLM client,
    the output handle (``None`` in ``--dry-run``), the dry-run flag, the link
    confidence threshold, and the mutable links/rejects/skipped counters."""

    client: Any
    fh: Any
    dry_run: bool
    min_confidence: str
    counts: dict[str, int]


def _load_judged_links(collapse_file: Path) -> set[str]:
    """Child refs already judged as reflex links in the ledger (rows carrying
    ``inherits``). A fold row sharing a ref must NOT block a reflex candidate
    for the same child surface, so only ``inherits``-bearing rows count."""
    judged: set[str] = set()
    if collapse_file.exists():
        for line in collapse_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("{"):  # tolerate blank lines / operator comments
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("_type") == "collapse" and row.get("ref") and "inherits" in row:
                judged.add(row["ref"])
    return judged


def _select_todo(candidates: list, judged: set[str], limit: int | None) -> list:
    """Candidates whose child ref isn't yet judged, capped at ``limit``."""
    todo = [c for c in candidates if c.child_ref not in judged]
    if limit is not None:
        todo = todo[:limit]
    return todo


def _judge_and_record(ctx: _LinkRun, c: Any) -> None:
    """Judge one reflex-link candidate with the LLM (one retry on transport/
    parse failure) and either print (``--dry-run``) or append its verdict row
    to the ledger. A transient failure is skipped WITHOUT recording, so a later
    run retries rather than burning the pair on noise."""
    from wyrd.generators.kenning.lexicon.reflex_link_detect import (
        build_reflex_judge_prompt,
        parse_reflex_verdict,
        reflex_verdict_to_row,
    )

    system, user = build_reflex_judge_prompt(c)
    verdict = None
    for _attempt in range(2):  # one retry on transport/parse failure
        try:
            verdict = parse_reflex_verdict(ctx.client.chat_json(system, user, {}))
        except Exception:
            verdict = None
        if verdict is not None:
            break
    if verdict is None:
        ctx.counts["skipped"] += 1  # transient/unparseable — don't burn the pair
        return
    row = reflex_verdict_to_row(c, verdict, ctx.min_confidence)
    is_link = row["inherits"] != ""
    ctx.counts["links"] += is_link
    ctx.counts["rejects"] += not is_link
    if ctx.dry_run:
        tag = "LINK" if is_link else "skip"
        click.echo(
            f"  [{tag}] {c.parent_form}->{c.child_form} {verdict.confidence} ({verdict.reason})",
            err=True,
        )
    else:
        ctx.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        ctx.fh.flush()  # crash-safety: each verdict durable as we go


@click.command("link-reflexes")
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
    help="Ledger to append verdicts to (links + no-op rejections).",
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
    help="Minimum LLM confidence to actually LINK a reflex verdict.",
)
@click.option(
    "--min-similarity",
    type=float,
    default=0.7,
    show_default=True,
    help="Form-similarity floor for candidate detection.",
)
@click.option("--max-per-gloss", type=int, default=20, show_default=True)
@click.option("--limit", type=int, default=None, help="Cap candidates judged this run.")
@click.option("--dry-run", is_flag=True, default=False, help="Judge + print; write nothing.")
def lexicon_link_reflexes(
    db_path: Path | None,
    collapse_file: Path,
    model: str,
    ollama_url: str | None,
    min_confidence: str,
    min_similarity: float,
    max_per_gloss: int,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Judge cross-era reflex-link candidates with an LLM, recording verdicts."""
    from wyrd.generators.kenning.cli.utils import _readonly_lexicon
    from wyrd.generators.kenning.extractors.llm import OllamaClient
    from wyrd.generators.kenning.lexicon.reflex_link_detect import detect_reflex_link_candidates

    if db_path is None:
        env_path = os.environ.get("WYRD_LEXICON_DB")
        db_path = Path(env_path) if env_path else Path.home() / ".wyrd" / "lexicon.db"
    if not db_path.exists():
        raise click.ClickException(f"Lexicon DB not found: {db_path}")
    base_url = ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")

    click.echo(f"Detecting reflex-link candidates in {db_path}...", err=True)
    with _readonly_lexicon(db_path) as conn:
        candidates = detect_reflex_link_candidates(
            conn, min_similarity=min_similarity, max_per_gloss=max_per_gloss
        )

    judged = _load_judged_links(collapse_file)
    todo = _select_todo(candidates, judged, limit)
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
    ctx = _LinkRun(
        client=OllamaClient(base_url=base_url, model=model, timeout_s=90.0),
        fh=None if dry_run else collapse_file.open("a", encoding="utf-8"),
        dry_run=dry_run,
        min_confidence=min_confidence,
        counts={"links": 0, "rejects": 0, "skipped": 0},
    )
    start = time.perf_counter()
    try:
        for i, c in enumerate(todo, 1):
            _judge_and_record(ctx, c)
            if i % 10 == 0 or i == len(todo):
                rate = (time.perf_counter() - start) / i
                click.echo(
                    f"  [{i}/{len(todo)}]  links={ctx.counts['links']} rejects={ctx.counts['rejects']} "
                    f"skipped={ctx.counts['skipped']} ({rate:.1f}s/entry)",
                    err=True,
                )
    finally:
        if ctx.fh is not None:
            ctx.fh.close()

    verb = "would record" if dry_run else "recorded"
    click.echo(
        f"{verb}: {ctx.counts['links']} links + {ctx.counts['rejects']} rejections "
        f"({ctx.counts['skipped']} skipped) → {collapse_file}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``link-reflexes`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_link_reflexes)
