"""``wyrd kenning lexicon fold-variants`` — LLM same-morpheme judgment over
same-language barren↔rich duplicate candidates, emitting FOLD (``into``)
verdicts to the _collapses.jsonl ledger (same-language variant fold, the
empty-grid root fix; sibling of ``link-reflexes`` which LINKS cross-era
reflexes). A confident same-morpheme verdict becomes a real ``into`` fold
(tombstoning the barren into the rich lemma); anything else is a no-op
``into: ""`` row recording the judgment so a re-run skips the pair. The LLM is
non-deterministic, so verdicts round-trip the ledger + replay deterministically
at rebuild — this command is NEVER re-run at rebuild time.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click


@click.command("fold-variants")
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
    help="Ledger to append verdicts to (folds + no-op rejections).",
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
    help="Minimum LLM confidence to actually FOLD a same-morpheme verdict.",
)
@click.option(
    "--min-similarity",
    type=float,
    default=0.65,
    show_default=True,
    help="Form-similarity floor for candidate detection.",
)
@click.option("--max-per-gloss", type=int, default=600, show_default=True)
@click.option("--limit", type=int, default=None, help="Cap candidates judged this run.")
@click.option("--dry-run", is_flag=True, default=False, help="Judge + print; write nothing.")
def lexicon_fold_variants(
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
    """Judge same-language variant-fold candidates with an LLM, recording verdicts."""
    from wyrd.generators.kenning.cli.utils import _readonly_lexicon
    from wyrd.generators.kenning.extractors.llm import OllamaClient
    from wyrd.generators.kenning.lexicon.variant_fold_detect import (
        build_fold_judge_prompt,
        detect_variant_fold_candidates,
        fold_verdict_to_row,
        parse_fold_verdict,
    )

    if db_path is None:
        env_path = os.environ.get("WYRD_LEXICON_DB")
        db_path = Path(env_path) if env_path else Path.home() / ".wyrd" / "lexicon.db"
    if not db_path.exists():
        raise click.ClickException(f"Lexicon DB not found: {db_path}")
    base_url = ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")

    click.echo(f"Detecting variant-fold candidates in {db_path}...", err=True)
    with _readonly_lexicon(db_path) as conn:
        candidates = detect_variant_fold_candidates(
            conn, min_similarity=min_similarity, max_per_gloss=max_per_gloss
        )

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
            # Dedup at the (barren, lemma) PAIR level so a barren rejected
            # against a weak lemma can still be judged against a better one. Only
            # variant-fold rows count (method tag). ``candidate_lemma`` is the
            # lemma judged against; legacy fold rows fall back to ``into`` (still
            # pair-correct), legacy rejects (no lemma recorded) fall back to ""
            # and so do NOT shadow a new better-lemma pair — they get re-judged.
            if (
                row.get("_type") == "collapse"
                and row.get("ref")
                and str(row.get("method", "")).startswith("llm-variant-")
            ):
                lemma = row.get("candidate_lemma") or row.get("into") or ""
                judged.add((row["ref"], lemma))

    todo = [c for c in candidates if (c.barren_ref, c.lemma_ref) not in judged]
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
    fh = None if dry_run else collapse_file.open("a", encoding="utf-8")
    folds = rejects = skipped = 0
    start = time.perf_counter()
    try:
        for i, c in enumerate(todo, 1):
            system, user = build_fold_judge_prompt(c)
            verdict = None
            for _attempt in range(2):  # one retry on transport/parse failure
                try:
                    verdict = parse_fold_verdict(client.chat_json(system, user, {}))
                except Exception:
                    verdict = None
                if verdict is not None:
                    break
            if verdict is None:
                skipped += 1
                continue
            row = fold_verdict_to_row(c, verdict, min_confidence)
            is_fold = row["into"] != ""
            folds += is_fold
            rejects += not is_fold
            if dry_run:
                tag = "FOLD" if is_fold else "skip"
                click.echo(
                    f"  [{tag}] {c.barren_form}->{c.lemma_form} {verdict.confidence} ({verdict.reason})",
                    err=True,
                )
            else:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            if i % 10 == 0 or i == len(todo):
                rate = (time.perf_counter() - start) / i
                click.echo(
                    f"  [{i}/{len(todo)}]  folds={folds} rejects={rejects} skipped={skipped} ({rate:.1f}s/entry)",
                    err=True,
                )
    finally:
        if fh is not None:
            fh.close()
    click.echo(
        f"{'(dry-run) ' if dry_run else ''}recorded: {folds} folds + {rejects} rejections "
        f"({skipped} skipped) → {collapse_file}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    parent.add_command(lexicon_fold_variants)
