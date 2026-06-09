"""``wyrd kenning lexicon disambiguate-fuzzy`` — LLM-disambiguate fuzzy text-match candidates (D25)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("disambiguate-fuzzy")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Apply the disambiguator's verdicts to etymon_text_match. Without this, dry-run.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process at most N ambiguous rows (smoke testing).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Override the Gemini model. Default: $WYRD_GEMINI_MODEL or gemini-2.5-flash.",
)
@click.option(
    "--timeout",
    type=float,
    default=120.0,
    show_default=True,
)
@click.option(
    "--agentic",
    is_flag=True,
    default=False,
    help=(
        "wyrd-uct: enable the agentic expand-context loop. The model can "
        "request a wider snippet when 200 chars isn't enough. Requires "
        "--sources-dir so wider snippets can be drawn from the source body."
    ),
)
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory of normalized source *.txt files. Required when "
        "--agentic is set. Same shape as the directory used by "
        "reverse-search and fuzzy-search."
    ),
)
@click.option(
    "--max-expansions",
    # >=0 — zero means "no expansions, force commit on round 0",
    # which is a valid degenerate config equivalent to single-shot
    # but with the agentic schema's defensive parsing.
    type=click.IntRange(min=0),
    default=3,
    show_default=True,
    help="Max expand-context rounds per case (only used with --agentic).",
)
@click.option(
    "--total-char-cap",
    type=click.IntRange(min=1),
    default=2000,
    show_default=True,
    help="Hard cap on total snippet chars per case (only used with --agentic).",
)
def lexicon_disambiguate_fuzzy(
    db_path: Path,
    apply_changes: bool,
    limit: int | None,
    model: str | None,
    timeout: float,
    agentic: bool,
    sources_dir: Path | None,
    max_expansions: int,
    total_char_cap: int,
) -> None:
    """Re-resolve fuzzy text-matches that have multiple plausible etymons.

    Per wyrd-6z7: when a body word X is within edit distance ≤ 1 of more
    than one canonical etymon, gloss anchoring alone isn't reliable. This
    command finds those ambiguous rows and asks Gemini to pick the right
    etymon based on the surrounding passage. The chosen row's `method`
    becomes 'llm-disambiguated-v1' and the model's reasoning is stored in
    `disambiguator_reason`. Rows where the model says "none" are deleted.

    Cost gate: rows with only one candidate etymon (no ambiguity) are
    skipped entirely — no LLM call.

    With --agentic (wyrd-uct), the disambiguator can request a wider
    snippet when the initial 200-char window isn't enough. Caps at
    --max-expansions rounds or --total-char-cap chars.
    """
    # Local imports keep the disambiguator deps off the cold-start path
    # for unrelated CLI commands.
    from wyrd.generators.kenning.extractors.gemini import GeminiClient
    from wyrd.generators.kenning.lexicon.disambiguator import (
        SnippetExpander,
        find_ambiguous_rows,
    )

    if agentic and sources_dir is None:
        raise click.UsageError("--agentic requires --sources-dir")

    with LexiconDB(db_path) as db:
        cases = find_ambiguous_rows(db, max_distance=1, limit=limit)

    if not cases:
        click.echo("No ambiguous fuzzy rows found.", err=True)
        return

    mode_label = "Agentic" if agentic else "Single-shot"
    click.echo(
        f"{len(cases)} ambiguous rows found. {mode_label} "
        f"{'apply' if apply_changes else 'dry-run'}.",
        err=True,
    )

    ctx = _DisambiguationRun(
        client=GeminiClient(timeout_s=timeout, **({"model": model} if model else {})),
        expander=SnippetExpander(sources_dir=sources_dir) if agentic else None,
        agentic=agentic,
        max_expansions=max_expansions,
        total_char_cap=total_char_cap,
        apply_changes=apply_changes,
        write_db=LexiconDB(db_path) if apply_changes else None,
        counts={"kept": 0, "reassigned": 0, "deleted": 0, "errors": 0},
        # Aggregate agentic stats so the operator can see how often the loop
        # had to widen and how often the cap forced a commit. Not surfaced
        # in the non-agentic path.
        agentic_totals={"expansions": 0, "forced_commits": 0},
    )
    try:
        for case in cases:
            _process_one_case(ctx, case)
    finally:
        if ctx.write_db is not None:
            ctx.write_db.close()

    click.echo("", err=True)
    click.echo(f"Disambiguator summary: {ctx.counts}", err=True)
    if agentic:
        click.echo(
            f"Agentic stats: expansions={ctx.agentic_totals['expansions']} "
            f"forced_commits={ctx.agentic_totals['forced_commits']}",
            err=True,
        )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to update etymon_text_match)", err=True)


def _classify_dry_run_row_counts(
    result: Any,
    case: Any,
) -> dict[str, int]:
    """Mirror ``apply_disambiguator_result``'s per-row classification
    without writing. Used by the CLI's dry-run path so the summary
    tells the operator what WOULD have happened on --apply."""
    n_rows = len(case.text_match_ids)
    if result.chosen_etymon_id is None:
        return {"kept": 0, "reassigned": 0, "deleted": n_rows}
    counts = {"kept": 0, "reassigned": 0, "deleted": 0}
    for current in case.current_etymon_ids:
        key = "kept" if result.chosen_etymon_id == current else "reassigned"
        counts[key] += 1
    return counts


def _disambiguate_dispatch(
    client: Any,
    case: Any,
    expander: Any,
    *,
    agentic: bool,
    max_expansions: int,
    total_char_cap: int,
) -> tuple[Any, Any]:
    """Single-shot vs agentic dispatch. Returns ``(result, stats)``;
    ``stats`` is None on the single-shot path. Raises RuntimeError on
    transport failure — caller increments the error counter and
    continues."""
    from wyrd.generators.kenning.lexicon.disambiguator import (
        disambiguate_one,
        disambiguate_one_agentic,
    )

    if agentic:
        result, stats = disambiguate_one_agentic(
            client,
            case,
            expander,
            max_expansions=max_expansions,
            total_char_cap=total_char_cap,
        )
        return result, stats
    return disambiguate_one(client, case), None


@dataclass
class _DisambiguationRun:
    """Shared state for the per-case disambiguation loop: the LLM client +
    optional snippet expander, the agentic config, the apply flag + write
    handle, and the mutable ``counts`` / ``agentic_totals`` accumulators."""

    client: Any
    expander: Any
    agentic: bool
    max_expansions: int
    total_char_cap: int
    apply_changes: bool
    write_db: Any
    counts: dict[str, int]
    agentic_totals: dict[str, int]


def _process_one_case(ctx: _DisambiguationRun, case: Any) -> None:
    """Disambiguate one ambiguous case: dispatch to the (LLM) single-shot or
    agentic path, accumulate agentic stats, echo the verdict, then apply it
    (``--apply``, committing per-case) or classify it for the dry-run summary.
    A transport ``RuntimeError`` is counted + skipped so the batch continues."""
    # Lazy import (matches _disambiguate_dispatch): keeps the disambiguator
    # deps off the cold-start path for unrelated CLI commands.
    from wyrd.generators.kenning.lexicon.disambiguator import apply_disambiguator_result

    try:
        result, stats = _disambiguate_dispatch(
            ctx.client,
            case,
            ctx.expander,
            agentic=ctx.agentic,
            max_expansions=ctx.max_expansions,
            total_char_cap=ctx.total_char_cap,
        )
    except RuntimeError as e:
        ctx.counts["errors"] += 1
        click.echo(f"  ERROR  {case.matched_form:20} {e}", err=True)
        return
    if stats is not None:
        ctx.agentic_totals["expansions"] += stats.expansions
        if stats.forced_commit:
            ctx.agentic_totals["forced_commits"] += 1
    choice_str = f"id={result.chosen_etymon_id}" if result.chosen_etymon_id is not None else "none"
    n_rows = len(case.text_match_ids)
    row_label = f" ({n_rows} rows)" if n_rows > 1 else ""
    click.echo(
        f"  {case.matched_form:20}{row_label} → {choice_str} "
        f"(conf={result.confidence}) — {result.reason[:80]}",
        err=True,
    )
    if ctx.apply_changes and ctx.write_db is not None:
        row_counts = apply_disambiguator_result(ctx.write_db, case, result)
        # Commit per-case so a crash mid-run doesn't lose previously
        # disambiguated rows (the LLM calls have already been paid for).
        ctx.write_db.commit()
    else:
        row_counts = _classify_dry_run_row_counts(result, case)
    for k, v in row_counts.items():
        ctx.counts[k] += v


def add_to(parent: click.Group) -> None:
    """Register ``disambiguate-fuzzy`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_disambiguate_fuzzy)
