"""``wyrd kenning lexicon drift-report`` — realism-retention drift
measurement between the legacy proportions path and the vector path.

wyrd-ecjp.6 Phase 6a: empirical drift measurement BEFORE committing
to tolerance bands. Generates N names per culture from each scoring
mode + emits a per-culture drift report (markdown or JSON).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from wyrd.generators.kenning.runtime.drift_measurement import (
    compute_drift_report,
    format_drift_report_json,
    format_drift_report_markdown,
)
from wyrd.generators.kenning.runtime.drift_runner import run_drift_samples


@click.command("drift-report")
@click.option(
    "--culture",
    type=str,
    required=True,
    help=(
        "Target culture (english / scottish / welsh / irish / breton). "
        "Each culture produces a separate report; run per-culture for "
        "comparison."
    ),
)
@click.option(
    "--count",
    type=int,
    default=1000,
    show_default=True,
    help=(
        "Number of names per scoring mode. Drift metrics stabilize "
        "around N=1000+; smaller samples produce noisier reports."
    ),
)
@click.option(
    "--seed",
    "base_seed",
    type=int,
    default=0,
    show_default=True,
    help=(
        "Starting seed; each generated name uses base_seed + i. Same "
        "seed → byte-identical drift report on re-run."
    ),
)
@click.option(
    "--priors-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional empirical-priors JSON sidecar (from 'lexicon "
        "dump-empirical-priors'). Enables the vector path's baseline "
        "axis. Without it, baseline contributes 0 and the score falls "
        "back to phon + sem + pos."
    ),
)
@click.option(
    "--tags",
    type=str,
    default="",
    help=(
        "Comma-separated tag filter (e.g. 'water,settlement'). "
        "Threaded through to BOTH scoring modes for fair comparison."
    ),
)
@click.option(
    "--harshness",
    type=float,
    default=0.0,
    show_default=True,
    help="D6 harshness scalar [0..1]. Threaded through to both modes.",
)
@click.option(
    "--cohesion",
    type=float,
    default=0.0,
    show_default=True,
    help="D17 cohesion scalar [0..1]. Threaded through to both modes.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    show_default=True,
    help="Output shape. 'json' is machine-readable for trend tracking.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write to file instead of stdout (useful for batch runs).",
)
def lexicon_drift_report(
    culture: str,
    count: int,
    base_seed: int,
    priors_path: Path | None,
    tags: str,
    harshness: float,
    cohesion: float,
    fmt: str,
    output_path: Path | None,
) -> None:
    """Generate N names per scoring mode + emit a drift report.

    Pure measurement — no DB writes, no bundle modifications. Safe to
    run repeatedly; same seed produces byte-identical output.
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    click.echo(
        f"drift-report (culture={culture}, count={count}, "
        f"seed={base_seed}, priors={'yes' if priors_path else 'no'})...",
        err=True,
    )

    samples_a, samples_b = run_drift_samples(
        culture=culture,
        count=count,
        base_seed=base_seed,
        priors_path=str(priors_path) if priors_path else None,
        tags=tag_list,
        harshness=harshness,
        cohesion=cohesion,
    )

    click.echo(
        f"  sample sizes: proportions={len(samples_a)}, vector={len(samples_b)}",
        err=True,
    )

    report = compute_drift_report(culture, samples_a, samples_b)

    if fmt == "json":
        payload = format_drift_report_json(report)
        out = json.dumps(payload, indent=2, sort_keys=True)
    else:
        out = format_drift_report_markdown(report)

    if output_path:
        output_path.write_text(out)
        click.echo(f"  wrote: {output_path}", err=True)
    else:
        # stdout for the report; stderr for progress (operator-friendly
        # for piping to a file via shell redirect)
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")


def add_to(parent: click.Group) -> None:
    """Register ``drift-report`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_drift_report)
