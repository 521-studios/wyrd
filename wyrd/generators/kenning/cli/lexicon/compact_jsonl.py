"""``wyrd kenning lexicon compact-jsonl`` — collapse a JSONL log to its canonical-state head rows."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("compact-jsonl")
@click.argument(
    "jsonl_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def lexicon_compact_jsonl(jsonl_path: Path) -> None:
    """Compact a JSONL event-log into canonical-state rows in place (wyrd-f295).

    Reads ``JSONL_PATH``, replays every row (canonical seed + add/set/
    patch/remove events) into final state, and writes back as canonical-
    state rows only. Idempotent — running on an already-compacted file
    is a no-op.

    Useful after a curation pass appends a batch of events (e.g.
    dead-rando prune, lemma normalization corrections) — once review
    is complete, compaction folds the events back into a clean canonical
    state for diff readability.
    """
    from wyrd.generators.kenning.jsonl_log import compact_file

    n = compact_file(jsonl_path)
    click.echo(f"Compacted {jsonl_path} → {n} canonical rows", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``compact-jsonl`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_compact_jsonl)
