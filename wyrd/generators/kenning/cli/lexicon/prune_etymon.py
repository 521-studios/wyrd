"""``wyrd kenning lexicon prune-etymon`` — append a remove event for an etymon row in a source JSONL (wyrd-8wgr)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _append_remove_event


@click.command("prune-etymon")
@click.argument("etymon_ref")
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
def lexicon_prune_etymon(
    etymon_ref: str, source_id: str, reason: str | None, jsonl_dir: Path
) -> None:
    """Append a 'remove' event for an etymon to its source's L2 file (wyrd-8wgr).

    ETYMON_REF is the etymon to prune ('old-english:pīe'). SOURCE_ID
    is the source whose JSONL file owns the etymon row (must already
    exist as <jsonl-dir>/<source_id>.jsonl).

    On the next `lexicon rebuild-from-jsonl`, the etymon row is dropped
    from the DB and any referencing fact-rows (citations, descent edges
    in this source, etymology elements that name it) are counted as
    orphans and skipped. Operator commits the JSONL change in git;
    reverting is appending an 'add' event with the original data.

    Typical use: dead-rando audit prunes (spurious morphemes the LLM
    hallucinated or scribal variants the operator wants removed).
    """
    target_file = _append_remove_event(
        jsonl_dir,
        source_id,
        "etymon",
        etymon_ref,
        reason,
        ref_format_hint="`<language>:<canonical_form>` format",
    )
    click.echo(f"Appended remove event for etymon `{etymon_ref}` → {target_file}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``prune-etymon`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_prune_etymon)
