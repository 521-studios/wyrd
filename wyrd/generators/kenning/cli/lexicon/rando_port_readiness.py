"""``wyrd kenning lexicon rando-port-readiness`` — report whether the rando-port retirement gate is open (wyrd-j2bv)."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("rando-port-readiness")
@click.option(
    "--bundle",
    "bundle_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional frozen meanings.json bundle to score against. Default: "
    "rehydrate the bundle from the L4 runtime DB (bundled seed-runtime.db, or "
    "whatever WYRD_RUNTIME_DB / WYRD_RUNTIME_DB_BUCKET resolves to). The on-disk "
    "meanings.json was retired when bundle storage moved to SQLite (d90t).",
)
@click.option(
    "--coverage-threshold",
    type=float,
    default=0.80,
    show_default=True,
    help="Per-language scholar+empirical attestation threshold (criterion 1).",
)
@click.option(
    "--language",
    "languages",
    multiple=True,
    help="Restrict to named target bundle siblings (repeat for multiple). "
    "Default: old_english, old_french, old_scandinavian, celtic_mix, latin. "
    "Names use the bundle's underscore form (Welsh/Irish are conflated under "
    "celtic_mix in the current bundle).",
)
def lexicon_rando_port_readiness(
    bundle_path: Path | None,
    coverage_threshold: float,
    languages: tuple[str, ...],
) -> None:
    """Report whether the rando-port retirement gate is open (wyrd-j2bv).

    Three criteria per target language:
      1. Scholar + empirical coverage ≥ threshold (default 80%)
      2. Zero rando-only bundle subjects
      3. Rando-only count ≤ scholar-attested count

    Exits 0 if all pass (gate OPEN, retirement can proceed); exit 1
    if any fail (gate CLOSED).

    Reads the bundle from the L4 runtime DB by default (wyrd-52ha) — the on-disk
    meanings.json was retired when bundle storage moved to SQLite (d90t); pass
    ``--bundle`` to score a frozen JSON bundle instead.
    """
    from wyrd.generators.kenning.bundle.rando_port_readiness import (
        DEFAULT_TARGET_LANGUAGES,
        compute_readiness,
        format_readiness,
    )
    from wyrd.generators.kenning.cli.utils import _load_meanings_data

    target = languages or DEFAULT_TARGET_LANGUAGES
    bundle = _load_meanings_data(bundle_path)
    report = compute_readiness(
        bundle, target_languages=target, coverage_threshold=coverage_threshold
    )
    click.echo(format_readiness(report))
    if not report.overall_passes:
        raise SystemExit(1)


def add_to(parent: click.Group) -> None:
    """Register ``rando-port-readiness`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_rando_port_readiness)
