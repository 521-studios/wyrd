"""``wyrd kenning lexicon vocab-health`` — controlled-vocabulary canary (wyrd-cgl4).

Counts non-canonical country / region / language values against the Class-A
controlled vocabularies (D49). Read-only. ``--strict`` exits non-zero when any
``stale`` or ``dirty`` value is present, so the report doubles as a CI / pre-build
gate. Logic lives in ``lexicon/vocab_health.py``; this is just presentation.
"""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.lexicon.vocab_health import scan, violations
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

_SEVERITY_ORDER = {"dirty": 0, "stale": 1, "info": 2, "ok": 3}


@click.command("vocab-health")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--strict",
    is_flag=True,
    help="Exit non-zero if any stale/dirty value is found (CI / pre-build gate).",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Also list the canonical (ok) and info values, not just violations.",
)
@click.option(
    "--top",
    type=int,
    default=25,
    show_default=True,
    help="Cap the per-dimension value list (0 = no cap).",
)
def lexicon_vocab_health(db_path: Path, strict: bool, show_all: bool, top: int) -> None:
    """Controlled-vocab health: non-canonical country/region/language values."""
    with _readonly_lexicon(db_path) as conn:
        findings = scan(conn)
    _emit(findings, show_all=show_all, top=top)
    bad = violations(findings)
    if strict and bad:
        raise SystemExit(1)


def _emit(findings, *, show_all: bool, top: int) -> None:
    total_bad = 0
    for dimension, items in findings.items():
        shown = sorted(
            (f for f in items if show_all or f.severity in ("stale", "dirty")),
            key=lambda f: (_SEVERITY_ORDER[f.severity], -f.count),
        )
        bad = sum(f.count for f in items if f.severity in ("stale", "dirty"))
        distinct_bad = sum(1 for f in items if f.severity in ("stale", "dirty"))
        total_bad += distinct_bad
        click.echo(f"=== {dimension} === ({distinct_bad} non-canonical value(s), {bad} row(s))")
        if not shown:
            click.echo("  clean ✓")
        capped = shown[:top] if top > 0 else shown
        for f in capped:
            label = f.value if f.value is not None else "<NULL>"
            click.echo(f"  [{f.severity:5}] {label:32} {f.count:>7}  ({f.status})")
        if len(capped) < len(shown):
            click.echo(f"  … and {len(shown) - len(capped)} more (--top 0 for all)")
        click.echo("")
    if total_bad == 0:
        click.echo("All controlled vocabularies are canonical ✓")
    else:
        click.echo(
            f"{total_bad} non-canonical value(s) found — "
            "stale=rebuild from clean L2; dirty=needs a vocab decision or data fix."
        )


def add_to(parent: click.Group) -> None:
    """Register ``vocab-health`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_vocab_health)
