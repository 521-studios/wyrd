"""``wyrd kenning lexicon ingest-briggs-personal-names`` — Briggs EPNS index ingest (wyrd-uzoh)."""

from __future__ import annotations

import time
from pathlib import Path

import click

from wyrd.generators.kenning.briggs_personal_names_ingester import (
    SOURCE_DOC,
    IngestStats,
    ingest_briggs_index,
)
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.schema import migrate_schema
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("ingest-briggs-personal-names")
@click.argument(
    "txt_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_ingest_briggs_personal_names(
    txt_path: Path,
    db_path: Path,
) -> None:
    """Ingest Keith Briggs's *Index to Personal Names in English Place-
    Names* (EPNS Supplementary Series 2, 3rd revised pdf edn 2024)
    into the lexicon DB.

    The CLI takes a pdftotext-extracted .txt of the EPNS-distributed
    PDF and lands two artifacts in the lexicon DB:

    1. ``personal_name`` rows — one per unique PN headform with the
       full Briggs metadata (PASE count, DLV presence, ASCh refs,
       language hints, feminine marker). Diacritics preserved in the
       ``headform`` column; the ``normalized_form`` column holds the
       diacritic-folded ASCII spelling for lookup against the lexicon's
       existing toponym data.

    2. ``personal_name_toponym_attestation`` rows — one per (PN,
       toponym, county) occurrence. ``attested_variant`` captures the
       PN's surface form as it appears IN the toponym when Briggs's
       "X in Y" syntax exposes it.

    Both tables are keyed on ``source_doc = "briggs_2024_personal_names_index"``
    so this ingest can coexist with future PN-index ingests from
    other scholars.

    The ingest is idempotent — UNIQUE indexes on (headform, source_doc)
    and the COALESCE-padded attestation dedup tuple absorb duplicates.
    Re-run on an already-populated DB does no net writes.

    Source format: pdftotext output of the EPNS-distributed PDF (see
    the ticket wyrd-uzoh for staging location). Get the pdf from
    https://www.nottingham.ac.uk/research/groups/epns/documents/briggs-an-index-to-personal-names-in-english-place-names-3rd-edn.pdf
    and run ``pdftotext briggs.pdf briggs.txt`` to produce the input.
    """
    start_ts = time.monotonic()
    click.echo(f"Ingesting Briggs personal-names index from {txt_path}", err=True)
    click.echo(f"  → {db_path}  (source_doc = {SOURCE_DOC})", err=True)

    db = LexiconDB(db_path)
    try:
        # Defensive migrate: ensures the two target tables exist even
        # on legacy DBs that haven't been alembic'd up. No-op on
        # fresh-install / current schema.
        migrate_schema(db)

        def _report(stats: IngestStats) -> None:
            elapsed = time.monotonic() - start_ts
            # CLAUDE.md mining-progress convention: emit s/entry, not
            # entries/s — operators extrapolate ETA from the per-record
            # cost.
            s_per_entry = elapsed / stats.entries_seen if stats.entries_seen > 0 else 0.0
            click.echo(
                f"  [{stats.entries_seen}] "
                f"pn_inserted={stats.personal_names_inserted} "
                f"pn_skipped={stats.personal_names_skipped} "
                f"att_inserted={stats.attestations_inserted} "
                f"att_skipped={stats.attestations_skipped} "
                f"att_unknown_county={stats.attestations_unknown_county} "
                f"({s_per_entry:.3f}s/entry)",
                err=True,
            )

        stats = ingest_briggs_index(db, txt_path, on_progress=_report)
    finally:
        db.close()

    elapsed = time.monotonic() - start_ts
    click.echo("", err=True)
    click.echo(
        f"Done in {elapsed:.1f}s. "
        f"entries={stats.entries_seen} "
        f"personal_names={stats.personal_names_inserted}+{stats.personal_names_skipped}skip "
        f"attestations={stats.attestations_inserted}+{stats.attestations_skipped}skip "
        f"unknown_county_dropped={stats.attestations_unknown_county}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``ingest-briggs-personal-names`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_ingest_briggs_personal_names)
