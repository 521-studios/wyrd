"""``wyrd kenning lexicon derive-surface-in-modern`` — per-element modern-surface slices via suffix-anchoring (wyrd-ujyo)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, derive_surface_in_modern
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("derive-surface-in-modern")
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
    help="Write surface_in_modern. Without this, dry-run only.",
)
@click.option(
    "--progress-every",
    type=int,
    default=200,
    show_default=True,
    help="Emit a progress line every N breakdowns scanned.",
)
def lexicon_derive_surface_in_modern(
    db_path: Path,
    apply_changes: bool,
    progress_every: int,
) -> None:
    """wyrd-ujyo: derive ``toponym_etymology_element.surface_in_modern`` by
    suffix-anchoring each binary breakdown against the toponym's MODERN name.

    For each binary breakdown, segment the modern name's suffix against the last
    morpheme's known reflexes (canonical form + cognate-cluster mates + etymon
    variants); the matched suffix is the last morpheme's surface and the
    remaining prefix is the first morpheme's surface. 'Ardeley' = OE ``earda`` +
    OE ``lēah`` records 'Arde' (ordinal 0) + 'ley' (ordinal 1) — capturing that
    ``earda`` surfaces as 'Arde' in this place-name.

    Shares the suffix-anchoring machinery with ``project-period-forms`` (binary
    breakdowns only, ≥2-char segments, skips OCR-cluster losers). Deterministic,
    LLM-free, idempotent (re-derives the same surface).

    Runs automatically inside ``lexicon enrich`` after ``project-period-forms``;
    this command exposes it standalone for a focused re-run.
    """
    with LexiconDB(db_path) as db:
        click.echo("derive-surface-in-modern:", err=True)
        result = derive_surface_in_modern(db, apply=apply_changes, progress_every=progress_every)

    click.echo(
        f"  scanned={result['rows_scanned']:>5}  "
        f"rows_projected={result['rows_projected']:>5}  "
        f"elements_updated={result['elements_updated']:>5}  "
        f"rows_written={result['rows_written']:>5}",
        err=True,
    )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write surface_in_modern)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``derive-surface-in-modern`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_derive_surface_in_modern)
