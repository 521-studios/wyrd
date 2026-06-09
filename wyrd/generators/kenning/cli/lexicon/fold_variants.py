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
from typing import TYPE_CHECKING, TextIO

import click

if TYPE_CHECKING:
    from wyrd.generators.kenning.extractors.llm import OllamaClient
    from wyrd.generators.kenning.lexicon.variant_fold_detect import (
        VariantFoldCandidate,
        VariantFoldVerdict,
    )


def _resolve_db_path(db_path: Path | None) -> Path:
    """Resolve the lexicon DB path (explicit flag, else $WYRD_LEXICON_DB, else
    ~/.wyrd/lexicon.db) and confirm it exists."""
    if db_path is None:
        env_path = os.environ.get("WYRD_LEXICON_DB")
        db_path = Path(env_path) if env_path else Path.home() / ".wyrd" / "lexicon.db"
    if not db_path.exists():
        raise click.ClickException(f"Lexicon DB not found: {db_path}")
    return db_path


def _load_prior_judgments(collapse_file: Path) -> tuple[set[tuple[str, str]], set[str]]:
    """Read the collapse ledger and return (judged, folded).

    ``judged`` is the set of (barren ref, lemma ref) PAIRs already adjudicated —
    pair-level so a barren rejected against a weak lemma can still be judged
    against a better one. ``candidate_lemma`` is the lemma judged against; legacy
    fold rows fall back to ``into`` (still pair-correct), legacy rejects (no lemma
    recorded) fall back to "" and so do NOT shadow a new better-lemma pair — they
    get re-judged.

    ``folded`` is the set of barren refs that have ALREADY folded into some lemma.
    A FOLD is TERMINAL: once a barren folds, never judge it again, else a later
    reject against a DIFFERENT lemma would land in the ledger and — since the
    collapse replay keys by ``ref`` (last-write-wins) — silently cancel the fold
    at rebuild. Keep the ledger one-fold-per-barren so replay is sound."""
    judged: set[tuple[str, str]] = set()
    folded: set[str] = set()
    if not collapse_file.exists():
        return judged, folded
    for line in collapse_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):  # tolerate blank lines / operator comments
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Only variant-fold rows count (method tag).
        if (
            row.get("_type") == "collapse"
            and row.get("ref")
            and str(row.get("method", "")).startswith("llm-variant-")
        ):
            lemma = row.get("candidate_lemma") or row.get("into") or ""
            judged.add((row["ref"], lemma))
            if row.get("into"):
                folded.add(row["ref"])
    return judged, folded


def _judge_with_retry(
    client: OllamaClient, c: VariantFoldCandidate
) -> tuple[VariantFoldVerdict | None, Exception | None]:
    """Judge one candidate with the LLM, retrying once on transport/parse
    flakiness. Returns (verdict, None) on success or (None, last_error) when both
    attempts fail. Imports are deferred to keep CLI registration cheap, mirroring
    the command body's pattern."""
    from wyrd.generators.kenning.lexicon.variant_fold_detect import (
        build_fold_judge_prompt,
        parse_fold_verdict,
    )

    system, user = build_fold_judge_prompt(c)
    last_err: Exception | None = None
    for _attempt in range(2):  # one retry on transport/parse failure
        try:
            verdict = parse_fold_verdict(client.chat_json(system, user, {}))
        except Exception as e:  # transient LLM transport/parse flakiness
            last_err = e
            verdict = None
        if verdict is not None:
            return verdict, None
    return None, last_err


def _run_judgments(
    todo: list[VariantFoldCandidate],
    client: OllamaClient,
    min_confidence: str,
    fh: TextIO | None,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Judge each candidate and either print (dry-run) or append its verdict row
    to the open ledger handle ``fh``. Returns (folds, rejects, skipped)."""
    from wyrd.generators.kenning.lexicon.variant_fold_detect import fold_verdict_to_row

    folds = rejects = skipped = 0
    start = time.perf_counter()
    for i, c in enumerate(todo, 1):
        verdict, last_err = _judge_with_retry(client, c)
        if verdict is None:
            skipped += 1
            # Don't skip silently — a wrong model/URL or a systematically
            # unparseable response would otherwise skip every pair and the run
            # would "succeed" with 0 folds, indistinguishable from nothing-to-do
            # (CLAUDE.md debuggability convention).
            why = last_err if last_err is not None else "unparseable verdict"
            click.echo(f"  [skip] {c.barren_form}->{c.lemma_form}: {why}", err=True)
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
            # fh is non-None whenever dry_run is False (see _judge_and_record);
            # narrow it for type checkers.
            assert fh is not None
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
        if i % 10 == 0 or i == len(todo):
            rate = (time.perf_counter() - start) / i
            click.echo(
                f"  [{i}/{len(todo)}]  folds={folds} rejects={rejects} skipped={skipped} ({rate:.1f}s/entry)",
                err=True,
            )
    return folds, rejects, skipped


def _judge_and_record(
    todo: list[VariantFoldCandidate],
    client: OllamaClient,
    collapse_file: Path,
    min_confidence: str,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Open the ledger for append (unless dry-run), run the judgments, and close
    it. Returns (folds, rejects, skipped)."""
    if not dry_run:
        collapse_file.parent.mkdir(parents=True, exist_ok=True)
    fh = None if dry_run else collapse_file.open("a", encoding="utf-8")
    try:
        return _run_judgments(todo, client, min_confidence, fh, dry_run)
    finally:
        if fh is not None:
            fh.close()


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
    from wyrd.generators.kenning.lexicon.variant_fold_detect import detect_variant_fold_candidates

    db_path = _resolve_db_path(db_path)
    base_url = ollama_url or os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")

    click.echo(f"Detecting variant-fold candidates in {db_path}...", err=True)
    with _readonly_lexicon(db_path) as conn:
        candidates = detect_variant_fold_candidates(
            conn, min_similarity=min_similarity, max_per_gloss=max_per_gloss
        )

    judged, folded = _load_prior_judgments(collapse_file)
    todo = [
        c
        for c in candidates
        if c.barren_ref not in folded and (c.barren_ref, c.lemma_ref) not in judged
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
    folds, rejects, skipped = _judge_and_record(
        todo, client, collapse_file, min_confidence, dry_run
    )
    click.echo(
        f"{'(dry-run) ' if dry_run else ''}recorded: {folds} folds + {rejects} rejections "
        f"({skipped} skipped) → {collapse_file}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    parent.add_command(lexicon_fold_variants)
