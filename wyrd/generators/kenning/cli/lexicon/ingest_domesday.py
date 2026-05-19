"""``wyrd kenning lexicon ingest-domesday`` — ingest the Open Domesday folio data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("ingest-domesday")
@click.argument(
    "mdb_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
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
    help=(
        "Actually insert toponym + attestation rows. Without this the "
        "command parses the MDB and reports counts without writing."
    ),
)
@click.option(
    "--include-uncertain",
    is_flag=True,
    default=False,
    help=(
        "Include PlaceForm rows tagged OScodes='speculative' / "
        "'unknown' (lost vills, uncertain modern identifications). "
        "Default skips these (~600 rows)."
    ),
)
def lexicon_ingest_domesday(
    mdb_path: Path,
    db_path: Path,
    apply_changes: bool,
    include_uncertain: bool,
) -> None:
    """Ingest the Open Domesday (Hull) place-name corpus (wyrd-el93).

    Downloads not handled here — point ``MDB_PATH`` at a local copy
    of the Hull Places database (.mdb file from
    digitalcollections.hull.ac.uk/downloads/th83kz829). Walks the
    PlaceForm table (~20K rows across ~14K places) and writes one
    toponym row per (modern_name, county) + one toponym_attestation
    row per Domesday entry with date_year=1086.

    Idempotent: re-runs detect existing toponyms by (modern_name,
    region) and existing attestations by (toponym_id, form, year,
    source_doc).

    License: Hull's data is CC-BY-NC-SA 4.0 (J.J.N. Palmer team).
    See ``domesday_ingester.DOMESDAY_SOURCE_NOTES`` for the full
    attribution recorded in the ``source`` table.
    """
    from urllib.parse import quote

    from wyrd.generators.kenning.domesday_ingester import ingest_domesday

    # URL-quote the path so spaces / reserved chars (?, #) in
    # operator-supplied paths don't get parsed as URI components.
    db_uri = f"file:{quote(str(db_path.absolute()))}?mode={'rw' if apply_changes else 'ro'}"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row

    last_pct = -1

    def _progress(done: int, total: int) -> None:
        nonlocal last_pct
        pct = int(done / total * 100.0) if total else 100
        # Throttle stderr to once per percent-tick.
        if pct != last_pct:
            last_pct = pct
            click.echo(
                f"  domesday-ingest: {done}/{total} ({pct}%)",
                err=True,
            )

    try:
        counts = ingest_domesday(
            conn,
            mdb_path,
            apply=apply_changes,
            include_uncertain=include_uncertain,
            progress_callback=_progress,
        )
    finally:
        conn.close()

    click.echo("", err=True)
    click.echo(f"Summary: {counts}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``ingest-domesday`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_ingest_domesday)
