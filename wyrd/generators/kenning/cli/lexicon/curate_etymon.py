"""``wyrd kenning lexicon curate-etymon`` — append an _curation event recording an operator lemma/merge override."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command("curate-etymon")
@click.argument("etymon_ref")
@click.option(
    "--lemma-ref",
    default=None,
    help=(
        "Set the etymon's lemma_id to this ref ('<language>:<canonical_form>'). "
        "Pass an empty string '' to explicitly CLEAR the lemma assignment."
    ),
)
@click.option(
    "--inflection",
    default=None,
    help=(
        "Inflection label written alongside --lemma-ref (e.g. 'spelling_variant', 'weak_oblique')."
    ),
)
@click.option(
    "--merged-into-ref",
    default=None,
    help=(
        "Set the etymon's merged_into_id (OCR-cluster tombstone target) "
        "to this ref. Pass '' to CLEAR. Mutually informative with "
        "--lemma-ref but applied independently."
    ),
)
@click.option(
    "--reason",
    default=None,
    help="Operator note. Doesn't affect the DB; recorded for audit.",
)
@click.option(
    "--curation-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_curation.jsonl"),
    show_default=True,
    help="JSONL file to append the curation event to. Created if it doesn't exist.",
)
def lexicon_curate_etymon(
    etymon_ref: str,
    lemma_ref: str | None,
    inflection: str | None,
    merged_into_ref: str | None,
    reason: str | None,
    curation_file: Path,
) -> None:
    """Append a curation event to the L2 curation file (wyrd-2jhs).

    Records an operator override that's applied AFTER the auto-clustering
    enrichment passes when `lexicon enrich --apply` (or
    `lexicon rebuild-from-jsonl --with-enrichment`) runs.

    ETYMON_REF is the etymon to curate (e.g. 'old-english:caelf').
    Pass --lemma-ref / --merged-into-ref / --inflection / --reason for
    the override fields; absent flags mean "no opinion on that column",
    and the literal empty string '' explicitly CLEARS the column.
    """
    if lemma_ref is None and merged_into_ref is None:
        raise click.UsageError(
            "Need at least one of --lemma-ref / --merged-into-ref to record a curation."
        )
    # SET both → the applier would produce a contradictory net state
    # (lemma_ref handler runs first then merged_into_ref clears it, or
    # vice versa). Surface the conflict at the CLI.
    # CLEAR both (both passed as '') → legitimate revert-to-standalone
    # operation; the applier correctly nulls out both columns.
    if lemma_ref and merged_into_ref:
        raise click.UsageError(
            "--lemma-ref and --merged-into-ref are mutually exclusive when "
            "SETTING values — an etymon is either a canonical inflection "
            "(lemma) or an OCR-cluster tombstone, not both. "
            "Clearing both with '' is allowed."
        )

    payload: dict[str, str | None] = {"_type": "etymon_curation", "ref": etymon_ref}
    # CLI '' → explicit null (clear column). Absent flag → key omitted.
    if lemma_ref is not None:
        payload["lemma_ref"] = None if lemma_ref == "" else lemma_ref
    if inflection is not None:
        payload["inflection"] = inflection
    if merged_into_ref is not None:
        payload["merged_into_ref"] = None if merged_into_ref == "" else merged_into_ref
    if reason is not None:
        payload["reason"] = reason

    # Create the file with its synthetic source row if it doesn't exist
    # yet — every L2 JSONL file needs exactly one source row.
    if not curation_file.exists():
        curation_file.parent.mkdir(parents=True, exist_ok=True)
        with curation_file.open("w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "_type": "source",
                        "ref": "manual-curation",
                        "title": "Operator curation overrides (wyrd-2jhs)",
                        "notes": (
                            "L3 enrichment overrides applied after the auto-"
                            "clustering passes. Each event records a "
                            "deliberate operator decision; reverting is "
                            "appending another event with the cleared "
                            "value (lemma_ref: '')."
                        ),
                    },
                    ensure_ascii=False,
                )
            )
            fh.write("\n")

    with curation_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")

    click.echo(f"Appended curation event for `{etymon_ref}` → {curation_file}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``curate-etymon`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_curate_etymon)
