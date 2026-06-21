"""``wyrd kenning lexicon mine-implied-reflexes`` — mine IMPLIED reflexes by
residual attribution of a toponym's modern name (wyrd-65jh).

Anchors the spans a breakdown's KNOWN-reflex elements cover, attributes the one
contiguous residual span to the remaining element, and (with ``--apply``) authors
BOTH a D50 ``canonical-label@modern-english`` assertion (source of truth) AND a
``reflex`` / ``reflex_etymon`` projection + ``surface_in_modern`` fill, re-dumping
``_reflexes.jsonl`` and the affected toponym_etymology L2 so they round-trip.
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from wyrd.generators.kenning.canonicalization import append_assertion, load_assertions
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.jsonl.dump import dump_all_sources, dump_reflexes_to_file
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.implied_reflex_mining import (
    apply_implied_reflexes,
    extract_implied_reflexes,
    implied_reflex_assertions,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("mine-implied-reflexes")
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
    help="L2 mining root; canonical-label assertions append under "
    "<dir>/canonicalization/, and _reflexes.jsonl + per-source toponym_etymology "
    "are re-dumped here on --apply.",
)
@click.option(
    "--source",
    default="implied-reflex-residual",
    show_default=True,
    help="Provenance source id stamped on the authored assertions.",
)
@click.option(
    "--min-support",
    type=int,
    default=None,
    help="Minimum distinct attesting toponyms to emit an implied reflex. Defaults "
    "to 1 for the dry-run report (vet everything) but 2 for --apply (support=1 is "
    "weak evidence for the effectful reflex/surface_in_modern/canonical-label "
    "projection — ~3656 of 4009 mined are singletons). Pass to override either.",
)
@click.option(
    "--apply/--dry-run",
    default=False,
    show_default=True,
    help="--apply authors assertions + projects reflexes/surface_in_modern and "
    "re-dumps the affected L2; --dry-run (default) reports only.",
)
def lexicon_mine_implied_reflexes(
    db_path: Path, mining_dir: Path, source: str, min_support: int | None, apply: bool
) -> None:
    """wyrd-65jh: mine implied reflexes by aligning a toponym's modern name to its
    breakdown elements — anchor the KNOWN-reflex spans, attribute the one
    contiguous residual span to the remaining element.

    DETERMINISTIC, LLM-free. ``--apply`` writes two targets: the
    canonical-label@modern-english assertion (source of truth, under
    <mining-dir>/canonicalization/) AND the reflex/reflex_etymon projection +
    surface_in_modern fill (re-dumped to _reflexes.jsonl + per-source
    toponym_etymology so both survive a rebuild).

    Default support floors: the dry-run report shows EVERYTHING (≥1) for vetting;
    --apply writes only the high-support clean reflexes (≥2), since support=1 is weak
    evidence for the effectful projection. Override either with --min-support.
    """
    # The report vets everything (≥1); the write path defaults to ≥2 (weak singletons).
    report_floor = 1 if min_support is None else min_support
    apply_floor = 2 if min_support is None else min_support
    click.echo("mine-implied-reflexes: aligning modern names to breakdowns…", err=True)
    start = time.monotonic()
    with LexiconDB(db_path) as db:
        reflexes = extract_implied_reflexes(db, min_support=report_floor)
        total = len(reflexes)
        surfaces = len({(r.residual, r.position) for r in reflexes})
        high = sum(1 for r in reflexes if r.support >= 2)
        elapsed = time.monotonic() - start
        rate = elapsed / total if total else 0.0
        click.echo(
            f"  [{total}/{total}] mined={total} surfaces={surfaces} "
            f"(high={high} medium={total - high}) ({rate:.3f}s/entry)",
            err=True,
        )
        if not apply:
            click.echo("  dry-run — nothing written (pass --apply to author).", err=True)
            return

        # Author + project only the high-support reflexes (apply_floor, default ≥2).
        apply_reflexes = extract_implied_reflexes(db, min_support=apply_floor)
        assertions = implied_reflex_assertions(apply_reflexes, source=source)
        existing = {a.id for a in load_assertions(mining_dir)}
        written = skipped = 0
        for assertion in assertions:
            if assertion.id in existing:
                skipped += 1
                continue
            append_assertion(mining_dir, assertion)
            written += 1
        click.echo(
            f"  wrote {written} canonical-label assertions to "
            f"{mining_dir}/canonicalization/ (skipped {skipped} already-authored)",
            err=True,
        )

        counts = apply_implied_reflexes(db, min_support=apply_floor)
        click.echo(
            f"  projected reflexes={counts['reflexes']} "
            f"links={counts['reflex_links']} "
            f"surface_in_modern={counts['surface_in_modern']}",
            err=True,
        )
        # Re-dump the durable L2 so the reflex + surface_in_modern round-trip on a
        # from-scratch rebuild (the ned5/br5o gap): _reflexes.jsonl via the
        # dedicated dumper, and the per-source toponym_etymology via dump_all_sources.
        _, n = dump_reflexes_to_file(db.conn, mining_dir)
        click.echo(f"  re-dumped {n} rows to {mining_dir}/_reflexes.jsonl", err=True)
        dump_all_sources(db.conn, mining_dir)
        click.echo(
            f"  re-dumped per-source L2 (toponym_etymology surface_in_modern) to {mining_dir}/",
            err=True,
        )


def add_to(parent: click.Group) -> None:
    """Register ``mine-implied-reflexes`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_implied_reflexes)
