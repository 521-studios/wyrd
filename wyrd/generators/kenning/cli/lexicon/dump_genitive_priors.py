"""``wyrd kenning lexicon dump-genitive-priors`` — emit genitive_split_prior as a versioned JSON sidecar."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.genitive_priors import dump_genitive_priors_to_json
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("dump-genitive-priors")
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
    help="Path to write the JSON genitive-priors artifact.",
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
def lexicon_dump_genitive_priors(db_path: Path, output_path: Path, version: str) -> None:
    """wyrd-aicu.9: dump genitive_split_prior to a sorted, byte-stable JSON
    artifact. Run after ``mine-genitive-priors --apply``.

    Output shape: a top-level dict with ``version`` (verbatim from --version)
    and a ``pairs`` list of ``{long_form, short_form, split_count,
    literal_count}`` records, ordered by total evidence. Raw counts only — the
    matcher applies smoothing + backoff at lookup. The committed JSON is the
    operator-visible diff surface for the prior's evolution as breakdowns accrue.
    """
    with LexiconDB(db_path) as db:
        result = dump_genitive_priors_to_json(db, output_path, version=version)
    click.echo("dump-genitive-priors:", err=True)
    click.echo(f"  wrote {result['pairs']} pairs to {output_path}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``dump-genitive-priors`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_dump_genitive_priors)
