"""``wyrd kenning lexicon mine-cognate-descents`` — author D50 Family-B descends-from
edges placing unclustered admitted breakdown morphemes into existing cognate
clusters (wyrd-zrce.1, uplift 1b)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.canonicalization import append_assertion, load_assertions
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.cognate_descent_mining import (
    descent_assertions,
    mine_cognate_descents,
)
from wyrd.generators.kenning.lexicon.etymon_refs import etymon_refs_for
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("mine-cognate-descents")
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
    help="L2 mining root; descends-from edges append under <dir>/canonicalization/.",
)
@click.option(
    "--source",
    default="cognate-descent-uplift",
    show_default=True,
    help="Provenance source id stamped on the authored assertions.",
)
@click.option(
    "--apply/--dry-run",
    default=False,
    show_default=True,
    help="--apply appends to the L2 streams; --dry-run (default) reports only.",
)
def lexicon_mine_cognate_descents(
    db_path: Path, mining_dir: Path, source: str, apply: bool
) -> None:
    """wyrd-zrce.1 (uplift 1b): place unclustered admitted breakdown morphemes into
    an existing cognate cluster via a RELATIONAL descends-from edge (D50 Family B),
    NOT an identity bind. A cohort morpheme that folds to a clustered etymon descends
    from that cluster's root, so cluster_cognates later assigns it the cluster's
    cognate_id. Tiered confidence: gloss-corroborated -> medium, surface-only -> low.

    DORMANT until the descends-from->etymon_descent projection applies it (run as a
    pipeline stage before cluster_cognates, or via project-canonical's relational
    sibling). Reads the lexicon DB directly; cognate clustering should be current for
    the cluster targets to be accurate.
    """
    click.echo("mine-cognate-descents: scanning unclustered breakdown morphemes…", err=True)
    with LexiconDB(db_path) as db:
        edges = mine_cognate_descents(db)
        # wyrd-c6wu: resolve endpoint ids -> stable natural keys at WRITE time so
        # the committed L2 carries refs that survive a rebuild's id reassignment.
        refs = etymon_refs_for(
            db.conn, {e.child_etymon for e in edges} | {e.cluster_root for e in edges}
        )
        assertions = descent_assertions(edges, refs, source=source)

    medium = sum(1 for e in edges if e.confidence == "medium")
    click.echo(
        f"  {len(edges)} cognate-descent edges (medium={medium} low={len(edges) - medium})",
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
    """Register ``mine-cognate-descents`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_cognate_descents)
