"""``wyrd kenning lexicon mine-same-morpheme-binds`` — author D50 same-morpheme
binds for admitted breakdown-morpheme variants (wyrd-c7iz, uplift 1a)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.canonicalization import append_assertion, load_assertions
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.etymon_refs import etymon_refs_for
from wyrd.generators.kenning.lexicon.same_morpheme_mining import (
    bind_assertions,
    mine_same_morpheme_binds,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("mine-same-morpheme-binds")
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
    help="L2 mining root; binds append under <dir>/canonicalization/.",
)
@click.option(
    "--source",
    default="same-morpheme-uplift",
    show_default=True,
    help="Provenance source id stamped on the authored assertions.",
)
@click.option(
    "--apply/--dry-run",
    default=False,
    show_default=True,
    help="--apply appends to the L2 streams; --dry-run (default) reports only.",
)
def lexicon_mine_same_morpheme_binds(
    db_path: Path, mining_dir: Path, source: str, apply: bool
) -> None:
    """wyrd-c7iz (uplift 1a): bind admitted breakdown-morpheme variants that are
    the SAME morpheme as a shipped one to a shared canonical_morpheme node (D50
    Family A identity). Authors mint-canonical + same-morpheme bind assertions to
    the L2 canonicalization streams (idempotent — already-authored ids are
    skipped).

    DORMANT until wyrd-u6fn.3 projects the L2 binds into the L3 collapse graph.
    Reads the lexicon DB directly (etymon / etymon_gloss / toponym_etymology_element
    + the export promotion gate); the family rollup + cognate clustering should be
    current for the shipped/cluster signals to be accurate.
    """
    click.echo("mine-same-morpheme-binds: building family rollup + scanning…", err=True)
    with LexiconDB(db_path) as db:
        groups = mine_same_morpheme_binds(db)
        # wyrd-c6wu: resolve member ids -> stable natural keys at WRITE time.
        refs = etymon_refs_for(db.conn, {m for g in groups for m in g.member_etymons})
        assertions = bind_assertions(groups, refs, source=source)

    high = sum(1 for g in groups if g.confidence == "high")
    variants = sum(len(g.breakdown_etymons) for g in groups)
    click.echo(
        f"  {len(groups)} bind groups (high={high} medium={len(groups) - high}), "
        f"{variants} variants, {len(assertions)} assertions",
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
        f"  wrote {written} assertions to {mining_dir}/canonicalization/ "
        f"(skipped {skipped} already-authored)",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``mine-same-morpheme-binds`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_same_morpheme_binds)
