"""``wyrd kenning lexicon cluster-cognates`` — walk etymon_descent edges and materialize cognate_id (D27)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, cluster_cognates
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("cluster-cognates")
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
    help="Actually write cognate_id assignments. Without this flag the command runs as a dry-run.",
)
def lexicon_cluster_cognates(db_path: Path, apply_changes: bool) -> None:
    """Walk the etymon_descent graph from each root and assign cognate_id
    to every reachable descendant (D27 / wyrd-81n).

    Roots are etymons that participate in inheritance/borrowing edges as
    parent but never as child — typically Proto-* forms. Every descendant
    reachable via inheritance + borrowing edges gets cognate_id =
    root.id, so cross-language cognates cluster behind a single canonical
    pointer. The 'cognate' edge type is a peer relation, NOT a chain,
    and does NOT bridge synsets.

    Run after Wiktionary mining (wyrd-4rt) populates etymon_descent.
    Reversible via `clear-enrichment --stage=cognates --apply`.
    """
    with LexiconDB(db_path) as db:
        result = cluster_cognates(db, apply=apply_changes)

    verb = "assigned" if apply_changes else "would assign"
    click.echo(
        f"cluster-cognates: {result['roots']} root(s) walked, "
        f"{verb} cognate_id on {result['candidates']} etymon(s)",
        err=True,
    )
    if apply_changes:
        click.echo(f"  rows_written = {result['rows_written']}", err=True)
    if result.get("tombstones_cleared"):
        verb_t = "cleared" if apply_changes else "would clear"
        click.echo(
            f"  {verb_t} stale cognate_id on {result['tombstones_cleared']} "
            f"tombstone(s) (wyrd-hn03 invariant)",
            err=True,
        )
    if result["cycle_orphans"]:
        click.echo(
            f"  warn: {result['cycle_orphans']} etymon(s) sit in a bridging-edge "
            f"cycle with no external root and were left unassigned",
            err=True,
        )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


# Place-name dictionaries write 'celtic' as a generic catch-all for
# morphemes whose specific Celtic-family origin isn't pinned. Wiktextract
# uses specific codes. The default --candidates resolves 'celtic' onto
# Proto-Celtic (most ancestral) first, then Old-* and Middle-* forms,
# then modern reflexes.
_CELTIC_CANDIDATES_DEFAULT = (
    "proto-celtic",
    "old-irish",
    "old-welsh",
    "old-breton",
    "middle-irish",
    "middle-welsh",
    "middle-breton",
    "irish",
    "welsh",
    "scottish-gaelic",
    "manx",
    "breton",
    "cornish",
)


def add_to(parent: click.Group) -> None:
    """Register ``cluster-cognates`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_cluster_cognates)
