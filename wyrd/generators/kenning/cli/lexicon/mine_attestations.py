"""``wyrd kenning lexicon mine-attestations`` — extract dated historical spellings from toponym_etymology.notes (D33 Phase 3.0a)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, mine_toponym_attestations
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("mine-attestations")
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
    help="Insert into toponym_attestation. Without this, dry-run only.",
)
@click.option(
    "--progress-every",
    type=int,
    default=500,
    show_default=True,
    help="Emit a progress line every N rows scanned.",
)
def lexicon_mine_attestations(
    db_path: Path,
    apply_changes: bool,
    progress_every: int,
) -> None:
    """wyrd-skm Phase 3.0a: extract (form, year) attestation pairs from
    toponym_etymology.notes and populate toponym_attestation.

    Pattern set: ``FORM in YEAR`` / ``FORM, YEAR`` / ``FORM in Domesday
    Book`` / ``Domesday has FORM`` / ``FORM, D.B.``. Year range is
    700-1700 (post-Roman through pre-modern), matching the
    lookup-attested-years filter. Page-reference false positives
    (``"Bedinga feld, p. 59"``) are rejected via the same page-marker
    guard that ``_earliest_year_in_notes`` uses.

    LLM-free, idempotent (unique-index on
    ``toponym_id, form, date_year, source_doc``), reversible:

        wyrd kenning lexicon clear-enrichment --stage=attestations --apply

    The output rows are the raw input that wyrd-skm Phase 3.0b will
    derive per-etymon period-keyed surface forms from.
    """
    with LexiconDB(db_path) as db:
        click.echo("mine-attestations:", err=True)
        result = mine_toponym_attestations(db, apply=apply_changes, progress_every=progress_every)

    click.echo(
        f"  scanned={result['rows_scanned']:>5}  "
        f"rows_with_pairs={result['rows_with_pairs']:>5}  "
        f"candidates={result['candidates']:>5}  "
        f"rows_written={result['rows_written']:>5}",
        err=True,
    )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write toponym_attestation rows)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``mine-attestations`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_attestations)
