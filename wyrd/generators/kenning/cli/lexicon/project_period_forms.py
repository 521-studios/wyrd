"""``wyrd kenning lexicon project-period-forms`` — segment historical compounds via suffix-anchoring (D33 Phase 3.0b)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, project_period_forms
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("project-period-forms")
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
    help="Insert into etymon_period_form. Without this, dry-run only.",
)
@click.option(
    "--progress-every",
    type=int,
    default=200,
    show_default=True,
    help="Emit a progress line every N rows scanned.",
)
def lexicon_project_period_forms(
    db_path: Path,
    apply_changes: bool,
    progress_every: int,
) -> None:
    """wyrd-unuo Phase 3.3: project per-etymon period forms from
    toponym_attestation rows.

    For each binary toponym breakdown, segment the attested historical
    form's suffix against the last morpheme's known reflexes (canonical
    form + cognate-cluster mates + etymon variants); the remaining
    prefix is the first morpheme's projected period form. Bradford
    (1377) "Bradeford" projects to "Brade" (OE 'brad') + "ford" (OE
    'ford'); Chesterton (1210) "Cestretone" projects to "Cestre" (OE
    'ceaster') + "tone" (OE 'tūn').

    Output is the Tier 3 fallback for ``etymon_era_reflexes`` —
    closes the coverage gap on isolated OE etymons (no cognate_id, no
    descent edges) when the toponym had a binary attested form.

    LLM-free, idempotent (unique-index on
    ``etymon_id, form, date_year, source_doc``), reversible:

        wyrd kenning lexicon clear-enrichment --stage=period-forms --apply

    Run after ``mine-attestations`` populates the toponym_attestation
    table; re-run any time the cluster mates expand to recover more
    suffix matches.
    """
    with LexiconDB(db_path) as db:
        click.echo("project-period-forms:", err=True)
        result = project_period_forms(db, apply=apply_changes, progress_every=progress_every)

    click.echo(
        f"  scanned={result['rows_scanned']:>5}  "
        f"rows_projected={result['rows_projected']:>5}  "
        f"candidates={result['candidates']:>5}  "
        f"rows_written={result['rows_written']:>5}",
        err=True,
    )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write etymon_period_form rows)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``project-period-forms`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_project_period_forms)
