"""``wyrd kenning lexicon clear-enrichment`` — revert one enrichment stage without touching mining evidence (D21/D22)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, clear_enrichment
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

# Per clearable enrichment artifact: (result key, human noun for the count line).
# Printed in this order; a count of 0 (or an absent key for a stage that doesn't
# produce it) prints nothing.
_CLEAR_COUNT_LINES: tuple[tuple[str, str], ...] = (
    ("ocr_merges_to_clear", "merged_into_id rows"),
    ("lemma_links_to_clear", "lemma links"),
    ("text_match_rows_to_clear", "text-match rows"),
    ("cognate_assignments_to_clear", "cognate_id assignments"),
    ("attested_years_to_clear", "attested_year values"),
    ("attestation_rows_to_clear", "toponym_attestation rows"),
    ("period_form_rows_to_clear", "etymon_period_form rows"),
    ("canonical_nodes_to_clear", "canonical nodes (+ binds/labels)"),
)


@click.command("clear-enrichment")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--stage",
    type=click.Choice(
        [
            "ocr",
            "lemmas",
            "text-match",
            "cognates",
            "attested-years",
            "attestations",
            "period-forms",
            "canonical",
            "all-derived",
        ]
    ),
    required=True,
    help=(
        "Which enrichment stage to clear. 'ocr' un-marks merged_into_id, "
        "'lemmas' resets lemma_id/inflection/lemma_method, 'text-match' "
        "drops the etymon_text_match table, 'cognates' resets "
        "cognate_id/cognate_method (D27/wyrd-81n), 'attested-years' resets "
        "etymon_text_match.attested_year (D5-1/wyrd-3ux), 'attestations' "
        "drops every toponym_attestation row (wyrd-skm Phase 3.0a), "
        "'period-forms' drops every etymon_period_form row "
        "(wyrd-unuo Phase 3.3), 'canonical' drops the project-canonical "
        "L2->L3 projection (canonical_* tables + binding columns; "
        "wyrd-u6fn.3/D50.6), 'all-derived' does all eight."
    ),
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually perform the clear. Without this flag the command runs as a dry-run.",
)
def lexicon_clear_enrichment(db_path: Path, stage: str, apply_changes: bool) -> None:
    """Reset one or more enrichment stages so they can be re-run.

    Mining evidence (etymons, citations, glosses, tags, etymon_descent,
    toponyms, toponym_etymology rows) is never touched. Per D21, only
    enrichment inferences are reversible.

    Run before re-running the relevant stage with new heuristics:

        wyrd kenning lexicon clear-enrichment --stage=ocr --apply
        wyrd kenning lexicon normalize-ocr --apply
    """
    with LexiconDB(db_path) as db:
        result = clear_enrichment(db, stage=stage, apply=apply_changes)

    verb = "cleared" if apply_changes else "would clear"
    click.echo(f"Stage: {result['stage']}", err=True)
    for key, noun in _CLEAR_COUNT_LINES:
        count = result.get(key)
        if count:
            click.echo(f"  {verb} {count} {noun}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``clear-enrichment`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_clear_enrichment)
