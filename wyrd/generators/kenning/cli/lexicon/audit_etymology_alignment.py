"""``wyrd kenning lexicon audit-etymology-alignment`` — flag toponym/etymology cross-misalignment (wyrd-8upf)."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("audit-etymology-alignment")
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
    help="Max sample findings to print per source.",
)
@click.option(
    "--top",
    "top_n",
    type=int,
    default=20,
    show_default=True,
    help="Max sources to include in the report.",
)
def lexicon_audit_etymology_alignment(
    directory: Path, sources: tuple[str, ...], sample_limit: int, top_n: int
) -> None:
    """Audit toponym/etymology cross-misalignment (wyrd-8upf).

    Walks every <source>.jsonl in DIRECTORY, flags etymology_element
    rows whose element list doesn't appear supported by the
    historical_form reconstruction. Markdown report on stdout.
    """
    from wyrd.generators.kenning.etymology_alignment_audit import (
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
    """Register ``audit-etymology-alignment`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_audit_etymology_alignment)
