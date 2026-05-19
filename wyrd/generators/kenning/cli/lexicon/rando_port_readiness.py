"""``wyrd kenning lexicon rando-port-readiness`` — report whether the rando-port retirement gate is open (wyrd-j2bv)."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("rando-port-readiness")
@click.option(
    "--bundle",
    "bundle_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("wyrd/generators/kenning/data/meanings.json"),
    show_default=True,
    help="Path to meanings.json (the runtime bundle).",
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
    bundle_path: Path,
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
    """
    from wyrd.generators.kenning.rando_port_readiness import (
        DEFAULT_TARGET_LANGUAGES,
        compute_readiness,
        format_readiness,
        load_bundle,
    )

    target = languages or DEFAULT_TARGET_LANGUAGES
    bundle = load_bundle(bundle_path)
    report = compute_readiness(
        bundle, target_languages=target, coverage_threshold=coverage_threshold
    )
    click.echo(format_readiness(report))
    if not report.overall_passes:
        raise SystemExit(1)


def add_to(parent: click.Group) -> None:
    """Register ``rando-port-readiness`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_rando_port_readiness)
