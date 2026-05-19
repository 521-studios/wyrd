"""``wyrd kenning lexicon prune-toponym`` — append a remove event for a toponym row in a source JSONL (wyrd-lene)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _append_remove_event


@click.command("prune-toponym")
@click.argument("toponym_ref")
@click.argument("source_id")
@click.option(
    "--reason",
    default=None,
    help=("Operator note. Doesn't affect the DB; recorded in the JSONL event for git-blame audit."),
)
@click.option(
    "--jsonl-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
)
def lexicon_prune_toponym(
    toponym_ref: str, source_id: str, reason: str | None, jsonl_dir: Path
) -> None:
    """Append a 'remove' event for a toponym to its source's L2 file (wyrd-lene).

    TOPONYM_REF is the toponym to prune ('Cart@Herefordshire'). SOURCE_ID
    is the source whose JSONL file owns the toponym row (must already
    exist as <jsonl-dir>/<source_id>.jsonl).

    On the next `lexicon rebuild-from-jsonl`, the toponym row is dropped
    from the DB and any etymology_element rows referencing it are
    counted as orphans + skipped. Operator commits the JSONL change in
    git; reverting is appending an 'add' event with the original data.
    """
    target_file = _append_remove_event(
        jsonl_dir,
        source_id,
        "toponym",
        toponym_ref,
        reason,
        ref_format_hint="`name@region` format; `@-` is null region",
    )
    click.echo(f"Appended remove event for toponym `{toponym_ref}` → {target_file}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``prune-toponym`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_prune_toponym)
