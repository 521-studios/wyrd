"""``wyrd kenning lexicon report-wikipedia-backfill`` — Wikipedia-seed retirement progress report (wyrd-4453)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY, seed_data_dir


@click.command("report-wikipedia-backfill")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Lexicon SQLite DB (read-only — this command never writes).",
)
@click.option(
    "--data-dir",
    "data_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=seed_data_dir(),
    show_default="<repo>/data/seed",
    help="Directory containing the *_place_names.json files.",
)
@click.option(
    "--language",
    "languages",
    multiple=True,
    help="Restrict to one or more cultures (repeatable). Tokens are matched "
    "case-insensitively against the file prefix: --language english picks "
    "english_place_names.json. Default: all 5 (english, scottish, welsh, "
    "irish, breton).",
)
@click.option(
    "--as-json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit machine-readable JSON on stdout instead of the human table on "
    "stderr. Useful for piping into jq / downstream tooling.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Expand each region to per-county/subregion rows in the table "
    "output. Default rolls up to per-country only.",
)
def lexicon_report_wikipedia_backfill(
    db_path: Path,
    data_dir: Path,
    languages: tuple[str, ...],
    as_json: bool,
    verbose: bool,
) -> None:
    """Report Wikipedia-place-name backfill progress (wyrd-4453).

    The 5 ``*_place_names.json`` files in
    ``wyrd/generators/kenning/data`` are Wikipedia-sourced seed lists
    used to populate the toponym table. The epic-terminal goal is to
    retire them once scholar mining has independently attested enough
    entries that dropping the Wikipedia provenance is safe.

    Per file → region → (optionally county), this command tallies:

      * total     — Wikipedia entries the file lists
      * in_db     — entries that exist as a toponym row (exact-match
        on ``modern_name``)
      * attested  — entries with ≥1 ``toponym_attestation`` row
        (scholar attestation — the retirement criterion)
      * gap       — total - attested

    Strictly read-only. Name lookup is NFC-normalized exact match
    against ``toponym.modern_name`` (case- and punctuation-sensitive
    otherwise); fuzzy-matching is a follow-up.
    """
    from wyrd.generators.kenning.bundle.wikipedia_backfill_report import (
        UnknownLanguageError,
        compute_backfill_report,
        format_report,
        report_to_dict,
    )

    with LexiconDB(db_path) as db:
        click.echo(f"Using DB {db_path}", err=True)

        try:
            reports = compute_backfill_report(
                db,
                data_dir=data_dir,
                languages=languages or None,
            )
        except UnknownLanguageError as e:
            # Re-raise as click.BadParameter so the operator sees a
            # clean CLI error message rather than a traceback. ONLY
            # catches the unknown-language case — data-shape errors
            # from _load_place_names (ValueError) and invariant
            # violations from CountyReport.__post_init__ (ValueError)
            # represent file corruption / producer bugs and propagate
            # as full tracebacks. R2 silent-failure-hunter MEDIUM
            # flagged the previous bare-ValueError catch as conflating
            # three distinct error classes.
            raise click.BadParameter(str(e)) from e

        if as_json:
            # JSON to stdout; the "Using DB ..." line still on stderr
            # so the pipe stays clean.
            click.echo(json.dumps(report_to_dict(reports), ensure_ascii=False, indent=2))
            return

        # Human-readable table to stderr per project mining-progress
        # convention (data on stdout, status on stderr).
        click.echo(format_report(reports, verbose=verbose), err=True)


def add_to(parent: click.Group) -> None:
    """Register ``report-wikipedia-backfill`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_report_wikipedia_backfill)
