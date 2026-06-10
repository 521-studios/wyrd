"""``wyrd kenning lexicon ingest-briggs-personal-names`` — Briggs EPNS index ingest (wyrd-uzoh, wyrd-11zh)."""

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
@click.option(
    "--jsonl-out",
    "jsonl_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        f"Output path for the L2 JSONL artifact. Defaults to "
        f"data/mining/{SOURCE_DOC}.jsonl. This file is the canonical "
        "reconstruction source — blowing away the DB and running "
        "rebuild-from-jsonl restores the Briggs rows from this file alone."
    ),
)
def lexicon_ingest_briggs_personal_names(
    txt_path: Path,
    db_path: Path,
    jsonl_path: Path | None,
) -> None:
    """Ingest Keith Briggs's *Index to Personal Names in English Place-
    Names* (EPNS Supplementary Series 2, 3rd revised pdf edn 2024)
    into the lexicon DB via the JSONL-first pattern (wyrd-11zh).

    Architectural contract (db-reconstructibility-reviewer / wyrd-fe1v):
    the JSONL file at ``--jsonl-out`` (default
    ``data/mining/briggs_2024_personal_names_index.jsonl``) is the
    canonical artifact. The lexicon DB is a queryable index over it.
    After this command runs, the recipe

        rm ~/.wyrd/lexicon.db
        wyrd kenning lexicon init
        wyrd kenning lexicon rebuild-from-jsonl

    restores the Briggs rows from the JSONL alone — no .txt re-parse,
    no PDF re-fetch.

    What lands in the DB on a fresh ingest (wyrd-2b50 — Briggs names are
    standard etymons, not bespoke tables):

    1. one ``etymon`` per unique PN headform (``canonical_form`` =
       headform with diacritics preserved; ``language`` resolved from
       the Briggs hint, defaulting to old-english), tagged ``male name``
       / ``female name``.

    2. one ``etymon_citation`` per source attesting the name: Briggs
       (the file's primary source) plus, where Briggs records them,
       secondary citations to PASE / DLV (Durham Liber Vitae) /
       Anglo-Saxon charters — registered as ``cited_source`` rows so
       one file can cite on their behalf. A name attested in ≥2 of
       these promotes through the normal witness gate.

    Name→toponym attestations are not stored as rows: the decompose-all
    enrichment re-derives name elements in toponyms for proportions.

    The ingest is idempotent — etymon UNIQUE(canonical_form, language)
    merge-on-conflict and etymon_citation UNIQUE(etymon_id, source_id,
    page) absorb duplicates. Re-running this command, or re-running
    ``rebuild-from-jsonl``, is a no-op on a populated DB.

    Source format: pdftotext output of the EPNS-distributed PDF. Get
    the pdf from
    https://www.nottingham.ac.uk/research/groups/epns/documents/briggs-an-index-to-personal-names-in-english-place-names-3rd-edn.pdf
    and run ``pdftotext briggs.pdf briggs.txt`` to produce the input.
    """
    start_ts = time.monotonic()
    resolved_jsonl = jsonl_path or (Path("data/mining") / f"{SOURCE_DOC}.jsonl")
    click.echo(f"Ingesting Briggs personal-names index from {txt_path}", err=True)
    click.echo(f"  → JSONL artifact: {resolved_jsonl}", err=True)
    click.echo(f"  → DB:             {db_path}  (source_doc = {SOURCE_DOC})", err=True)

    db = LexiconDB(db_path)
    try:
        # Defensive migrate: bring a legacy DB up to the current etymon
        # schema the ingest writes into (etymon columns/views + citation
        # table, wyrd-2b50). No-op on fresh-install / current schema.
        migrate_schema(db)

        def _report(stats: IngestStats) -> None:
            elapsed = time.monotonic() - start_ts
            # CLAUDE.md mining-progress convention: emit s/entry, not
            # entries/s — operators extrapolate ETA from the per-record
            # cost.
            s_per_entry = elapsed / stats.entries_seen if stats.entries_seen > 0 else 0.0
            click.echo(
                f"  [{stats.entries_seen}] "
                f"etymons={stats.pn_rows_emitted} "
                f"citations={stats.citations_emitted} "
                f"att_dropped={stats.attestations_dropped} "
                f"({s_per_entry:.3f}s/entry)",
                err=True,
            )

        stats = ingest_briggs_index(db, txt_path, jsonl_path=resolved_jsonl, on_progress=_report)
    finally:
        db.close()

    elapsed = time.monotonic() - start_ts
    click.echo("", err=True)
    click.echo(
        f"Done in {elapsed:.1f}s. "
        f"entries={stats.entries_seen} "
        f"etymons_emitted={stats.pn_rows_emitted} etymons_inserted_in_db={stats.etymons_inserted} "
        f"citations_emitted={stats.citations_emitted} "
        f"citations_inserted_in_db={stats.citations_inserted} "
        f"citation_orphans={stats.citation_orphans} "
        f"attestations_dropped={stats.attestations_dropped}",
        err=True,
    )
    # Silent-skip visibility line (wyrd-jac1): each non-zero value here
    # is a parse-time drop the operator should know about.
    click.echo(
        f"Silent-skip counters: "
        f"entries_unparsed={stats.entries_unparsed} "
        f"att_groups_no_county={stats.attestation_groups_skipped_no_county} "
        f"att_groups_lang_only={stats.attestation_groups_skipped_lang_only} "
        f"entries_no_atts={stats.entries_with_zero_attestations} "
        f"entries_citation_only={stats.entries_citation_only}",
        err=True,
    )
    click.echo(f"  JSONL artifact: {stats.jsonl_path}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``ingest-briggs-personal-names`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_ingest_briggs_personal_names)
