"""``wyrd kenning lexicon bridge-language`` — open-ended language-bridging enrichment (wyrd-bridge-* family)."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from wyrd.generators.kenning.cli.lexicon.cluster_cognates import _CELTIC_CANDIDATES_DEFAULT
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, bridge_generic_language
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("bridge-language")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--generic",
    "generic_lang",
    type=str,
    required=True,
    help=(
        "Generic language tag to bridge (e.g. 'celtic'). Each etymon with "
        "this language gets merged_into_id set to the highest-priority "
        "specific-language match with the same canonical_form."
    ),
)
@click.option(
    "--candidates",
    "candidates_csv",
    type=str,
    default=None,
    help=(
        "Comma-separated specific languages to consider, in priority order. "
        "If omitted and --generic is 'celtic', uses the built-in Celtic "
        "candidate list (Proto-Celtic > Old-* > Middle-* > modern reflexes)."
    ),
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write merged_into_id. Without this, dry-run reporting only.",
)
def lexicon_bridge_language(
    db_path: Path,
    generic_lang: str,
    candidates_csv: str | None,
    apply_changes: bool,
) -> None:
    """Bridge a generic-language etymon family onto specific-language
    canonicals via merged_into_id.

    Place-name dictionaries write generic tags like 'celtic' for
    morphemes whose specific Celtic-family origin isn't pinned.
    Wiktextract entries are language-specific. This pass canonicalizes
    each generic-language etymon onto a specific-language match with
    the same canonical_form, so cluster-cognates' redirect-aware walk
    will fold the generic etymons into the cross-source cognate clusters.

    Run AFTER wiktextract ingest + AFTER normalize-ocr. Re-running
    cluster-cognates after this pass is recommended to refresh
    cognate_id assignments.

    Reverse via `clear-enrichment --stage=ocr --apply` (the bridge uses
    the same merged_into_id mechanism).
    """
    if candidates_csv is not None:
        candidates = tuple(s.strip() for s in candidates_csv.split(",") if s.strip())
        if not candidates:
            click.echo(
                "error: --candidates resolved to an empty list (need at least "
                "one specific-language code)",
                err=True,
            )
            sys.exit(1)
    elif generic_lang == "celtic":
        candidates = _CELTIC_CANDIDATES_DEFAULT
    else:
        click.echo(
            f"error: --candidates required when --generic is not 'celtic' "
            f"(no built-in default for {generic_lang!r})",
            err=True,
        )
        sys.exit(1)

    with LexiconDB(db_path) as db:
        result = bridge_generic_language(
            db,
            generic_lang=generic_lang,
            candidate_langs=candidates,
            apply=apply_changes,
        )

    verb = "bridged" if apply_changes else "would bridge"
    click.echo(
        f"bridge-language: {result['generic_etymons']} generic '{generic_lang}' etymon(s) examined; "
        f"{verb} {result['bridged']}, {result['unmatched']} unmatched.",
        err=True,
    )
    if apply_changes:
        click.echo(f"  rows_written = {result['rows_written']}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``bridge-language`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_bridge_language)
