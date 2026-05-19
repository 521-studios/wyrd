"""``wyrd kenning lexicon reverse-search`` — mine etymon text matches by scanning source bodies (D20)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, reverse_search_attestations
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("reverse-search")
@click.argument(
    "sources_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
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
    help="Insert search-attested citation rows. Without this, dry-run only.",
)
@click.option(
    "--min-form-length",
    type=int,
    default=4,
    show_default=True,
    help="Skip rando etymons with canonical_form shorter than this (avoids "
    "false-positive substring matches).",
)
def lexicon_reverse_search(
    sources_dir: Path,
    db_path: Path,
    apply_changes: bool,
    min_form_length: int,
) -> None:
    """Reverse-direction verification + parser-bug diagnostic.

    For each rando-port etymon with no scholarly citation, search the
    bundled source-book texts (sources_dir/*.txt) for the form as a
    word-boundary match. Two outputs:

    1. PROMOTION: forms that appear in scholarly texts but were never
       formally extracted get a 'search-attested:<book>' citation,
       moving them from 'unverified rando' to 'mentioned in published
       philology'.

    2. PARSER-BUG DIAGNOSTIC: forms that appear MANY times in source
       text but were never produced as an etymon by any extractor are
       likely systemic pipeline misses. Worth investigating the prompt,
       the parser segmentation, or both.

    Conservative: minimum form length defaults to 4 to avoid false-
    positive substring matches.
    """
    with LexiconDB(db_path) as db:
        result = reverse_search_attestations(
            db, sources_dir, apply=apply_changes, min_form_length=min_form_length
        )

    click.echo(
        f"Rando-only candidates: {result['rando_only_candidates']}",
        err=True,
    )
    click.echo(
        f"Etymons with text-match: {result['etymons_with_match']} "
        f"({100 * result['etymons_with_match'] / max(result['rando_only_candidates'], 1):.1f}%)",
        err=True,
    )
    click.echo(
        f"Total citation records: {result['total_match_records']}",
        err=True,
    )

    click.echo("", err=True)
    click.echo("=== Sample promotions (first 25) ===", err=True)
    for s in result["sample"]:
        books = ", ".join(f"{b}({c})" for b, c, _ in s["matches"])
        click.echo(f"  {s['form']:24} {books}", err=True)
        # Show one sample snippet per etymon so the user can see context
        if s["matches"]:
            _, _, snippet = s["matches"][0]
            click.echo(f"      → {snippet[:140]}", err=True)

    if result["parser_bug_suspects"]:
        click.echo("", err=True)
        click.echo(
            "=== Parser-bug suspects: high text-count, zero extraction-count ===",
            err=True,
        )
        click.echo(
            "(These forms appear often in scholarly texts but our LLM "
            "extractor never emitted them as etymons. Likely pipeline gaps.)",
            err=True,
        )
        for s in result["parser_bug_suspects"][:20]:
            click.echo(
                f"  {s['form']:20} text-count={s['text_count']:>5}  "
                f"extracted=0  in: {', '.join(s['books'])}",
                err=True,
            )

    if not apply_changes:
        click.echo("", err=True)
        click.echo("(dry-run; pass --apply to write search-attested citations)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``reverse-search`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_reverse_search)
