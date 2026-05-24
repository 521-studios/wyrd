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
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help=(
        "wyrd-van9: exit non-zero when any curation / suppression / split "
        "event has unresolved refs or other operator-typo signals (missing "
        "gloss, missing parent, invalid suffix, self-reference). Without "
        "this flag those signals only surface in the markdown summary, so "
        "an operator running ``enrich --apply`` in CI / a deploy script "
        "won't see them fail. Use --strict in those contexts; leave off "
        "for interactive review."
    ),
)
def lexicon_enrich(
    db_path: Path,
    apply_changes: bool,
    jsonl_dir: Path,
    with_curation: bool,
    strict: bool,
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
        format_unresolved_warnings,
        run_full_enrichment,
    )
    from wyrd.generators.kenning.jsonl.build import (
        collect_curation_overrides,
        collect_etymon_splits,
        collect_gloss_additions,
        collect_gloss_suppressions,
        jsonl_paths_in,
    )

    curation_state = None
    suppression_state = None
    addition_state = None
    split_state = None
    if with_curation:
        paths = list(jsonl_paths_in(jsonl_dir))
        curation_state = collect_curation_overrides(paths) or None
        suppression_state = collect_gloss_suppressions(paths) or None
        addition_state = collect_gloss_additions(paths) or None
        split_state = collect_etymon_splits(paths) or None
    with LexiconDB(db_path) as db:
        result = run_full_enrichment(
            db,
            apply=apply_changes,
            curation_state=curation_state,
            suppression_state=suppression_state,
            addition_state=addition_state,
            split_state=split_state,
        )
    click.echo(format_enrichment_run(result), err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to commit)", err=True)

    # wyrd-van9: surface unresolved-ref / typo signals on stderr; exit
    # non-zero under --strict so CI / deploy scripts fail loudly on
    # operator typos.
    warnings = format_unresolved_warnings(result)
    if warnings:
        click.echo("\n" + warnings, err=True)
        if strict:
            raise click.ClickException(
                "wyrd-van9 --strict: unresolved-ref signals present. "
                "Review the warnings above; fix the offending events "
                "(typos in etymon refs, missing gloss text, invalid "
                "suffix chars) and re-run."
            )


def add_to(parent: click.Group) -> None:
    """Register ``enrich`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_enrich)
