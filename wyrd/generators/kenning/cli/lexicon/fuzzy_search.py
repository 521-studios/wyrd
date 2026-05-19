"""``wyrd kenning lexicon fuzzy-search`` — find body-form variants of canonical etymons within edit distance ≤ 1 (D15)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, fuzzy_search_attestations
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("fuzzy-search")
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
    help="Write fuzzy matches to etymon_text_match. Without this, dry-run.",
)
@click.option(
    "--max-distance",
    type=int,
    default=1,
    show_default=True,
    help="Max Levenshtein distance for fuzzy match. 1 catches spelling "
    "variants; 2 produces many more false positives.",
)
@click.option(
    "--min-form-length",
    type=int,
    default=5,
    show_default=True,
    help="Skip etymons shorter than this (short forms have too many fuzzy neighbors).",
)
def lexicon_fuzzy_search(
    sources_dir: Path,
    db_path: Path,
    apply_changes: bool,
    max_distance: int,
    min_form_length: int,
) -> None:
    """Fuzzy-match rando-only etymons against scholarly source text.

    For each rando-only etymon NOT already exact-matched, find tokens in
    source text within Levenshtein distance N. Filter by gloss-anchor:
    only count a fuzzy candidate if one of the etymon's glosses appears
    within ±100 chars of the candidate's first occurrence.

    The gloss anchor is the safety mechanism. Without it, edit-distance-1
    would match 'bere' (barley) to 'bera' (bear) — completely different
    morphemes. With it, we only count matches where the meaning aligns.

    19th-c scholarly OE spelling wasn't standardized — denu/dene/denū are
    all the same word. This command finds those variants automatically.

    Run AFTER `reverse-search` (which handles exact matches).
    """
    with LexiconDB(db_path) as db:
        result = fuzzy_search_attestations(
            db,
            sources_dir,
            apply=apply_changes,
            max_distance=max_distance,
            min_form_length=min_form_length,
        )

    click.echo(
        f"Candidates with at least one gloss: {result['candidates_with_gloss']}",
        err=True,
    )
    click.echo(
        f"Etymons with fuzzy match (gloss-confirmed): {result['etymons_with_fuzzy_match']}",
        err=True,
    )
    click.echo(f"Total fuzzy match records: {result['total_match_records']}", err=True)
    click.echo("", err=True)
    click.echo("=== Sample fuzzy matches (first 25) ===", err=True)
    for s in result["sample"]:
        ms = ", ".join(f"{src}:{mf}(d={d},×{c})" for src, mf, d, c in s["matches"])
        click.echo(f"  {s['form']:20} → {ms}", err=True)
    if not apply_changes:
        click.echo("", err=True)
        click.echo("(dry-run; pass --apply to write to etymon_text_match)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``fuzzy-search`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_fuzzy_search)
