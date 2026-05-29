"""``wyrd kenning lexicon ingest-kepn`` — KEPN county-CSV ingest (wyrd-bc5z)."""

from __future__ import annotations

import time
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.kepn_csv_ingester import (
    KepnStats,
    county_from_filename,
    ingest_kepn_csv,
)
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.schema import migrate_schema
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

_DEFAULT_WHITELIST = Path.home() / ".wyrd" / "sources" / "kepn" / "elements_all.txt"


@click.command("ingest-kepn")
@click.argument(
    "csv_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--county",
    default=None,
    help="County name (region). Defaults to the CSV filename stem, title-cased. "
    "Override for stems that don't title-case cleanly. Ignored in directory mode.",
)
@click.option(
    "--jsonl-out",
    "jsonl_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output L2 JSONL path (single-file mode). Defaults to "
    "data/mining/kepn_<stem>.jsonl. This file is the canonical artifact; "
    "rebuild-from-jsonl restores the KEPN rows from it alone.",
)
@click.option(
    "--briggs-jsonl",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/briggs_2024_personal_names_index.jsonl"),
    show_default=True,
    help="Briggs L2 JSONL — source of the personal-name match index.",
)
@click.option(
    "--whitelist",
    "whitelist_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_WHITELIST,
    show_default=True,
    help="Canonical element-form whitelist (KEPN's 5,831-element list). "
    "Lexical element forms not in it are quarantined (counted, not ingested).",
)
def lexicon_ingest_kepn(
    csv_path: Path,
    db_path: Path,
    county: str | None,
    jsonl_path: Path | None,
    briggs_jsonl: Path,
    whitelist_path: Path,
) -> None:
    """Ingest a KEPN advanced-search county CSV (or a directory of them).

    KEPN's CSV has no county column, so the county (toponym region) comes
    from the filename — name each file ``<county>.csv`` (e.g.
    ``rutland.csv``, ``north-riding-yorkshire.csv``). In directory mode
    every ``*.csv`` is ingested with its own county.

    Each place yields a ``toponym`` (region=county, country=England), an
    ``etymology_element`` breakdown, and per lexical element an ``etymon``
    + ``kepn`` citation (+ a ``kepn_external`` citation when the row cites
    external dictionaries). Personal-name slots are fold-matched against
    the Briggs name index — a match corroborates the Briggs etymon; a miss
    is recorded as a note on a partial breakdown.

    The JSONL artifact(s) are canonical; the DB is a queryable index
    rebuildable from them via ``rebuild-from-jsonl``.
    """
    csvs = sorted(csv_path.glob("*.csv")) if csv_path.is_dir() else [csv_path]
    if not csvs:
        raise click.ClickException(f"no CSV files found in {csv_path}")
    wl = whitelist_path if whitelist_path.exists() else None
    if wl is None:
        click.echo(f"  (no element whitelist at {whitelist_path}; quarantine disabled)", err=True)

    db = LexiconDB(db_path)
    try:
        migrate_schema(db)
        for cp in csvs:
            this_county = county if (county and not csv_path.is_dir()) else county_from_filename(cp)
            jout = (
                jsonl_path
                if (jsonl_path and not csv_path.is_dir())
                else (Path("data/mining") / f"kepn_{cp.stem}.jsonl")
            )
            start = time.monotonic()
            click.echo(f"KEPN ingest: {cp.name}  (county={this_county})", err=True)

            def _report(s: KepnStats, _start: float = start) -> None:
                el = time.monotonic() - _start
                rate = el / s.places if s.places else 0.0
                click.echo(
                    f"  [{s.places}] etymons={s.etymons_emitted} "
                    f"pn={s.pn_match_t1 + s.pn_match_t2}/{s.pn_slots}match "
                    f"miss={s.pn_miss} quarantine={s.whitelist_quarantined} "
                    f"({rate:.3f}s/place)",
                    err=True,
                )

            stats = ingest_kepn_csv(
                db,
                cp,
                county=this_county,
                jsonl_path=jout,
                briggs_jsonl=briggs_jsonl,
                element_whitelist_path=wl,
                on_progress=_report,
            )
            click.echo(
                f"  done {cp.name}: places={stats.places} "
                f"etymons_in_db={stats.etymons_inserted} "
                f"citations_in_db={stats.citations_inserted} "
                f"(kepn+external) orphans={stats.citation_orphans} | "
                f"names: extracted={stats.pn_extracted}/{stats.pn_slots} "
                f"T1={stats.pn_match_t1} T2={stats.pn_match_t2} miss={stats.pn_miss} | "
                f"quarantined={stats.whitelist_quarantined} "
                f"unmapped_lang={stats.unmapped_lang_slots}",
                err=True,
            )
            if stats.unmapped_lang_slots:
                click.echo(
                    f"    UNMAPPED language slots (investigate): {stats.unmapped_lang_sample}",
                    err=True,
                )
            if stats.whitelist_quarantined:
                click.echo(f"    quarantined forms (sample): {stats.quarantine_sample}", err=True)
    finally:
        db.close()


def add_to(parent: click.Group) -> None:
    """Register this subcommand on the shared ``lexicon`` group."""
    parent.add_command(lexicon_ingest_kepn)
