"""``wyrd kenning lexicon dump-empirical-priors`` — emit empirical_priors_* tables as a versioned JSON sidecar."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.empirical_priors import dump_empirical_priors_to_json
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("dump-empirical-priors")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to write the JSON priors artifact.",
)
@click.option(
    "--version",
    type=str,
    default="unversioned",
    show_default=True,
    help=(
        "Version string written verbatim into the JSON. Use a content "
        "hash, sequential int, or date per your versioning policy."
    ),
)
def lexicon_dump_empirical_priors(db_path: Path, output_path: Path, version: str) -> None:
    """wyrd-ecjp.2: dump empirical_priors_* tables to a sorted, byte-
    stable JSON artifact. Run after ``mine-empirical-baselines --apply``.

    Output shape: a top-level dict with ``version`` (verbatim from
    --version) and two lists of cell records (``native``, ``loan``).
    Each cell record carries its key fields + a sorted ``lemmas``
    dict mapping lemma_ref to count. Sort order is deterministic so
    re-running on the same DB state produces a byte-identical file.

    The committed JSON is the operator-visible diff surface for
    priors evolution.
    """
    with LexiconDB(db_path) as db:
        result = dump_empirical_priors_to_json(db, output_path, version=version)
    click.echo("dump-empirical-priors:", err=True)
    click.echo(
        f"  wrote {result['native_cells']} native cells + "
        f"{result['loan_cells']} loan cells to {output_path}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``dump-empirical-priors`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_dump_empirical_priors)
