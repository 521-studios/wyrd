"""``wyrd kenning lexicon tag-phonological-vectors`` — compute + persist
the 14-dim PhonologicalVector for every etymon (wyrd-kq7w.1).

Standalone CLI command. Also runs as part of the canonical L3
enrichment chain (``lexicon enrich``); operators reach for this
command directly to:

  * Force-recompute the corpus after a classifier change
    (``--force``).
  * Smoke-test the enrichment pass on a single language
    (``--language``).
  * Preview without writing (default; ``--apply`` for the real write).
"""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.phonological_vector_enrichment import (
    phonological_vector_coverage,
    tag_phonological_vectors_all,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("tag-phonological-vectors")
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
    help=(
        "Write computed phonological_vector blobs to the etymon table. "
        "Without this flag, the command runs as a dry-run that prints "
        "the candidate count + a 5-row sample of what WOULD be written."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Recompute vectors for every etymon (including rows that already "
        "have a non-NULL phonological_vector). Use after a classifier-rule "
        "change or dimension addition to refresh the corpus. Default "
        "behavior is incremental — only NULL rows are processed."
    ),
)
def lexicon_tag_phonological_vectors(
    db_path: Path,
    apply_changes: bool,
    force: bool,
) -> None:
    """Compute + persist the 14-dim PhonologicalVector for every etymon.

    Reads ``canonical_form`` + ``pronunciation_ipa`` per etymon, runs
    ``compute_phonological_vector``, and stores the result as a JSON
    blob in ``etymon.phonological_vector``. Deterministic + idempotent —
    re-running the pass on an unchanged DB writes byte-identical blobs.

    Dry-run prints candidate count + 5-row sample so an operator can
    spot-check the vector shape before committing.
    """
    db = LexiconDB(db_path)
    try:
        result = tag_phonological_vectors_all(db, apply=apply_changes, force=force)
        coverage = phonological_vector_coverage(db.conn)

        mode = "force" if force else "incremental"
        click.echo(
            f"PhonologicalVector enrichment ({mode}, {'apply' if apply_changes else 'dry-run'}):",
            err=True,
        )
        click.echo(
            f"  candidates: {result['candidates']}",
            err=True,
        )
        if apply_changes:
            click.echo(f"  written:    {result['written']}", err=True)
        click.echo(
            f"  coverage:   {coverage['tagged']:,} / {coverage['total']:,} "
            f"({coverage['ratio'] * 100:.1f}%)",
            err=True,
        )
        if result["sample"]:
            click.echo("  sample:", err=True)
            for etymon_id, canonical_form, blob in result["sample"]:
                click.echo(f"    [{etymon_id}] {canonical_form}: {blob}", err=True)
    finally:
        db.close()


def add_to(parent: click.Group) -> None:
    """Register ``tag-phonological-vectors`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_tag_phonological_vectors)
