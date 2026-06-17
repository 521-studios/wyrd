"""``wyrd kenning lexicon project-canonical`` — project the L2 canonicalization
assertions into the L3 collapse graph (wyrd-u6fn.3, D50.6).

Deterministic, idempotent, reversible (``clear-enrichment --stage=canonical``).
ADDITIVE: populates the ``canonical_*`` tables + ``canonical_*_id`` columns; it
does NOT migrate the existing ``merged_into_id`` / ``cognate_id`` / ``lemma_id``
readers — that cutover is wyrd-u6fn.4."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.canonicalization_projection import (
    assess_identity_fidelity,
    project_canonical,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("project-canonical")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--mining-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="L2 mining root; reads the canonicalization assertion streams under <dir>/canonicalization/.",
)
@click.option(
    "--confidence-gate",
    default="high",
    show_default=True,
    help="Minimum bind confidence to apply (D46 leave-separate). A label or a float in [0,1].",
)
@click.option(
    "--fold-legacy/--no-fold-legacy",
    default=True,
    show_default=True,
    help="Fold today's deterministic merged_into_id/lemma_id clustering into the "
    "canonical graph (so it reproduces today's identity clustering, no snapshot).",
)
@click.option(
    "--assess-fidelity",
    is_flag=True,
    default=False,
    help="After projecting, report whether canonical_morpheme_id reproduces the "
    "legacy COALESCE(merged_into_id, lemma_id, id) partition (every legacy family "
    "maps to one canonical group).",
)
@click.option(
    "--apply/--dry-run",
    default=False,
    show_default=True,
    help="--apply clears + rebuilds the canonical graph; --dry-run (default) reports only.",
)
def lexicon_project_canonical(
    db_path: Path,
    mining_dir: Path,
    confidence_gate: str,
    fold_legacy: bool,
    assess_fidelity: bool,
    apply: bool,
) -> None:
    """Project the canonicalization assertion streams into the L3 collapse graph
    (mint canonical hubs, bind observations, merge-canonical, per-stratum labels).
    Idempotent + reversible; the same L2 yields a byte-identical L3 (D36.9)."""
    click.echo("project-canonical: loading assertions + projecting…", err=True)
    with LexiconDB(db_path) as db:
        res = project_canonical(
            db,
            mining_dir=mining_dir,
            apply=apply,
            confidence_gate=confidence_gate,
            fold_legacy=fold_legacy,
        )
        fidelity = assess_identity_fidelity(db) if (assess_fidelity and apply) else None

    verb = "wrote" if apply else "would write"
    minted = ", ".join(f"{t}={n}" for t, n in sorted(res.minted.items())) or "none"
    bound = ", ".join(f"{t}={n}" for t, n in sorted(res.bound.items())) or "none"
    click.echo(
        f"  {verb}: minted [{minted}]  bound [{bound}]  merged={res.merged}  labels={res.labels}",
        err=True,
    )
    if res.conflicts:
        click.echo(f"  CONFLICTS (left unbound, D46): {len(res.conflicts)}", err=True)
        for subj_type, ref, nodes in res.conflicts[:20]:
            click.echo(f"    {subj_type}:{ref} -> {nodes}", err=True)
    if res.skipped_unsupported:
        click.echo(f"  skipped (no ref convention yet): {dict(res.skipped_unsupported)}", err=True)
    if res.warnings:
        click.echo(f"  warnings: {len(res.warnings)} (e.g. {res.warnings[0]})", err=True)
    if fidelity is not None:
        v = fidelity["violations"]
        marker = "OK" if v == 0 else "MISMATCH"
        click.echo(
            f"  fidelity [{marker}]: {fidelity['faithful']}/{fidelity['legacy_families']} "
            f"legacy families map to one canonical group; {v} violations",
            err=True,
        )
        for root, members, roots in fidelity["sample_violations"]:
            click.echo(f"    legacy root {root}: members {members} -> canonical {roots}", err=True)
    if not apply:
        click.echo("  dry-run — nothing written (pass --apply to project).", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``project-canonical`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_project_canonical)
