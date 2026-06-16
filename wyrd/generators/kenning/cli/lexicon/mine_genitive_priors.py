"""``wyrd kenning lexicon mine-genitive-priors`` — wyrd-aicu.9 prior extraction."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.genitive_priors import mine_genitive_priors
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("mine-genitive-priors")
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
        "Replace genitive_split_prior with freshly-extracted counts. Without this, dry-run only."
    ),
)
@click.option(
    "--progress-every",
    type=int,
    default=5000,
    show_default=True,
    help="Emit a stderr progress line every N toponyms scanned. 0 = silent.",
)
def lexicon_mine_genitive_priors(db_path: Path, apply_changes: bool, progress_every: int) -> None:
    """wyrd-aicu.9: extract per-suffix genitive-``s`` split priors.

    For each auto-discovered genitive-overlap pair (``ston`` / ``ton`` …), count
    scholarly toponym breakdowns whose final element is the genitive-split
    reading (the short suffix — tūn → town) vs the literal long form (stān →
    stone), classified by the breakdown's cognate-cluster. The historical
    ``-es-``/``-s-`` genitive marker (wyrd-aicu.9.1) resolves the residue the
    cluster left ambiguous/unclassified — subordinate, never overriding a
    decisive cluster verdict.

    LLM-free, deterministic, idempotent. Replace-not-merge each apply run. Raw
    counts only — smoothing + backoff live at matcher lookup time. Pair with
    ``lexicon dump-genitive-priors --output ...`` for a review-friendly sidecar.
    """
    with LexiconDB(db_path) as db:
        result = mine_genitive_priors(db, apply=apply_changes, progress_every=progress_every)
    click.echo("mine-genitive-priors:", err=True)
    click.echo(
        "  scanned={ts:>6}  candidate_pairs={cp:>4}  active_pairs={ap:>4}".format(
            ts=result["toponyms_scanned"],
            cp=result["candidate_pairs"],
            ap=result["active_pairs"],
        ),
        err=True,
    )
    click.echo(
        "  classified: split={cs:>5}  literal={cl:>5}  (attestation-resolved={ar:>5})".format(
            cs=result["classified_split"],
            cl=result["classified_literal"],
            ar=result["resolved_by_attestation"],
        ),
        err=True,
    )
    click.echo(
        "  skipped: both_classes={sb:>5}  unclassified={su:>5}".format(
            sb=result["skipped_both"],
            su=result["skipped_unclassified"],
        ),
        err=True,
    )
    if not apply_changes:
        click.echo(
            "(dry-run; pass --apply to write genitive_split_prior rows)",
            err=True,
        )


def add_to(parent: click.Group) -> None:
    """Register ``mine-genitive-priors`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_genitive_priors)
