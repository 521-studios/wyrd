"""``wyrd kenning lexicon audit-short-quotes`` — flag LLM-truncated citation short_quotes (wyrd-bd68)."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("audit-short-quotes")
@click.option(
    "--dir",
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Directory of <source>.jsonl files to scan.",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Restrict to named source_ids (repeat for multiple).",
)
@click.option(
    "--samples",
    "sample_limit",
    type=int,
    default=3,
    show_default=True,
    help="Max sample short_quotes to print per source.",
)
@click.option(
    "--top",
    "top_n",
    type=int,
    default=20,
    show_default=True,
    help="Max sources to include in the report.",
)
def lexicon_audit_short_quotes(
    directory: Path, sources: tuple[str, ...], sample_limit: int, top_n: int
) -> None:
    """Audit citation short_quotes for LLM truncation (wyrd-bd68).

    Walks every <source>.jsonl in DIRECTORY, flags short_quotes that
    look cut off mid-sentence, and prints a markdown report. Operator
    decides whether to re-mine the flagged sources or context-snippet
    refill via the page-number infrastructure.
    """
    from wyrd.generators.kenning.short_quote_audit import (
        audit_jsonl_dir,
        format_audit_report,
    )

    report = audit_jsonl_dir(
        directory,
        sample_limit=sample_limit,
        sources=sources or None,
    )
    click.echo(format_audit_report(report, top_n=top_n))


def add_to(parent: click.Group) -> None:
    """Register ``audit-short-quotes`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_audit_short_quotes)
