"""``wyrd kenning lexicon parse-pages`` — parse page anchors out of a source body (wyrd-azv prelude)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.lexicon import detect_running_headers


@click.command("parse-pages")
@click.argument("source_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--limit",
    type=int,
    default=10,
    show_default=True,
    help="Number of detected (page, headword-fragment) pairs to print.",
)
def lexicon_parse_pages(source_path: Path, limit: int) -> None:
    """Audit running-header coverage on a source book (wyrd-9kh.5, wyrd-8st).

    Tries both header conventions — Mawer-style `<HEADWORD> <number>`
    and Skeat-§ `§ N. NAMES IN -X. <page>` — and reports which matched
    along with a sample. Use this to decide whether a book is amenable
    to page-anchored citations.
    """
    text = source_path.read_text(errors="replace", encoding="utf-8")
    headers, parser = detect_running_headers(text)
    click.echo(f"{source_path.name}: {len(headers)} running header(s) detected (parser: {parser})")
    if not headers:
        click.echo(
            "  No headers matched either Mawer-style or Skeat-§ patterns. The "
            "book may use a third convention entirely.",
            err=True,
        )
        return
    click.echo(f"  Page range: {headers[0][1]} → {headers[-1][1]}")
    click.echo(f"  Sample (first {min(limit, len(headers))}):")
    for offset, page in headers[:limit]:
        # Snippet of the matched line for visual inspection.
        line_end = text.find("\n", offset)
        line = text[offset : line_end if line_end > 0 else offset + 80].strip()
        click.echo(f"    p.{page:>4}  @offset={offset:>7}  {line}")


def add_to(parent: click.Group) -> None:
    """Register ``parse-pages`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_parse_pages)
