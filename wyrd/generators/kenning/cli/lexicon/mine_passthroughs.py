"""``wyrd kenning lexicon mine-passthroughs`` — author D50 composed-of edges for
composite morphemes from cross-scholar coarse-vs-fine breakdown evidence (wyrd-h5u1).

A composite morpheme (``ington``) that a scholar elsewhere split finely (``ing``+``tūn``)
is recorded as ``composed-of`` its ordered constituents (Family B, D50.3). The matcher
keeps matching the composite, but downstream it is ATTRIBUTED to its constituents
(surface != attribution, D51) — which is what lets the grader credit a coarse match's
constituent clusters (the wyrd-h5u1 net-win that gates wyrd-oth3 admission)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.canonicalization import append_assertion, load_assertions
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.passthrough_mining import (
    extract_cross_scholar_passthroughs,
    passthrough_assertions,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("mine-passthroughs")
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
    help="L2 mining root; composed-of edges append under <dir>/canonicalization/.",
)
@click.option("--source", default="passthrough-mining", show_default=True)
@click.option(
    "--min-support",
    type=int,
    default=1,
    show_default=True,
    help="Min distinct toponyms attesting a composite=constituents pair (>=2 → high confidence).",
)
@click.option("--apply/--dry-run", default=False, show_default=True)
def lexicon_mine_passthroughs(
    db_path: Path, mining_dir: Path, source: str, min_support: int, apply: bool
) -> None:
    """wyrd-h5u1: mine composite→constituent passthroughs from cross-scholar
    coarse-vs-fine breakdowns and author them as D50 composed-of assertions to the
    L2 canonicalization stream (idempotent — already-authored ids skipped)."""
    click.echo("mine-passthroughs: scanning cross-scholar coarse-vs-fine breakdowns…", err=True)
    with LexiconDB(db_path) as db:
        passthroughs = extract_cross_scholar_passthroughs(db, min_support=min_support)
        assertions = passthrough_assertions(passthroughs, source=source)

    high = sum(1 for p in passthroughs if p.support >= 2)
    click.echo(
        f"  {len(passthroughs)} passthroughs (high={high} medium={len(passthroughs) - high}), "
        f"{len(assertions)} composed-of assertions",
        err=True,
    )
    if not apply:
        click.echo("  dry-run — nothing written (pass --apply to author).", err=True)
        return

    existing = {a.id for a in load_assertions(mining_dir)}
    written = skipped = 0
    for assertion in assertions:
        if assertion.id in existing:
            skipped += 1
            continue
        append_assertion(mining_dir, assertion)
        written += 1
    click.echo(
        f"  wrote {written} composed-of assertions to {mining_dir}/canonicalization/ "
        f"(skipped {skipped} already-authored)",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``mine-passthroughs`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_passthroughs)
