"""``wyrd kenning lexicon enrich`` — run the full L3 enrichment chain in canonical order."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("enrich")
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
        "Actually write merged_into_id + lemma_id. Without this flag the "
        "command runs both passes as a dry-run and reports counts."
    ),
)
@click.option(
    "--jsonl-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help=(
        "Directory holding the per-source JSONL files. Scanned for "
        "etymon_curation events (wyrd-2jhs) that apply after the auto-"
        "clustering passes. Skip via --no-curation."
    ),
)
@click.option(
    "--with-curation/--no-curation",
    default=True,
    show_default=True,
    help=(
        "Apply operator curation overrides from the JSONL dir as the "
        "final enrichment step. Default on; disable for a pure auto-"
        "clustering run."
    ),
)
def lexicon_enrich(
    db_path: Path,
    apply_changes: bool,
    jsonl_dir: Path,
    with_curation: bool,
) -> None:
    """Run L3 enrichment passes in canonical order (wyrd-ilam).

    Today this is normalize-ocr → link-lemmas — the FIRST L3 enrichment
    migration per the wyrd-eni4 plan. Follow-on PRs extend the
    orchestrator to cluster-cognates, classify-stratum,
    derive-english-shaped, project-period-forms, etc.

    Order matters: normalize-ocr writes merged_into_id (OCR tombstones)
    before link-lemmas's lemma_id targets are picked, so inflected
    forms can only link to canonical etymons, never to tombstones.

    Standalone-pass commands (`lexicon normalize-ocr`,
    `lexicon link-lemmas`) stay available for fine-grained operator
    control; this orchestrator is the canonical one-shot.
    """
    from wyrd.generators.kenning.enrichment import (
        format_enrichment_run,
        run_full_enrichment,
    )
    from wyrd.generators.kenning.jsonl.build import (
        collect_curation_overrides,
        jsonl_paths_in,
    )

    curation_state = None
    if with_curation:
        curation_state = collect_curation_overrides(jsonl_paths_in(jsonl_dir)) or None
    with LexiconDB(db_path) as db:
        result = run_full_enrichment(db, apply=apply_changes, curation_state=curation_state)
    click.echo(format_enrichment_run(result), err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to commit)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``enrich`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_enrich)
