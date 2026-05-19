"""``wyrd kenning lexicon reverse-search-toponyms`` — scan source bodies for toponym mentions (wyrd-9wuq)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("reverse-search-toponyms")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Lexicon SQLite DB.",
)
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("sources"),
    show_default=True,
    help="Directory of <source_id>.txt source bodies to scan.",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Restrict to named source_ids (repeat for multiple).",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="INSERT OR IGNORE matched attestations. Without this, dry-run.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print samples of unmatched forms per source (Phase 2 LLM-mining candidates).",
)
def lexicon_reverse_search_toponyms(
    db_path: Path,
    sources_dir: Path,
    sources: tuple[str, ...],
    apply: bool,
    verbose: bool,
) -> None:
    """Toponym reverse-search: extract (form, year) attestations from scholar
    source bodies (wyrd-x82p Phase 1).

    Walks each scholar source's ``.txt`` body, runs the
    ``mine-attestations`` regex set against it (rather than only against
    ``toponym_etymology.notes``), maps each extracted form back to a known
    toponym, and emits new ``toponym_attestation`` rows via the existing
    UNIQUE-index-respecting INSERT OR IGNORE pattern.

    Forms that don't map to a known toponym are tallied with sample
    output (--verbose) — those are Phase 2 LLM-mining candidates for
    new-toponym discovery.

    After --apply, run ``lexicon dump-jsonl`` to propagate to the
    committed ``data/mining/<source>.jsonl`` files.
    """
    # Deferred for CLI help-snappiness (per wyrd-xz6l pattern).
    from wyrd.generators.kenning.toponym_reverse_search import (
        _build_form_to_toponym_lookup,
        reverse_search_source,
    )

    totals = {
        "pairs": 0,
        "matched": 0,
        "unmatched": 0,
        "inserted": 0,
        "already_present": 0,
    }

    with LexiconDB(db_path) as db:
        if sources:
            known = {r["id"] for r in db.conn.execute("SELECT id FROM source")}
            unknown = sorted(set(sources) - known)
            if unknown:
                raise click.ClickException(
                    f"unknown source_id(s): {', '.join(unknown)}. "
                    f"Check spelling against `lexicon report --sources` output."
                )
            target_sources = list(sources)
        else:
            # Iterate the source table — every registered scholar
            # source is a candidate for body-text reverse-search
            # regardless of whether it has citations yet. Earlier
            # version used `DISTINCT source_id FROM etymon_citation`
            # (mirroring wyrd-1hpc); Gemini PR #212 round-2 finding
            # corrected: a source can have toponym attestations
            # without etymons (operator-uploaded source body before
            # the etymon mining pass runs, or a source that's
            # toponym-only by design). Sources without a committed
            # .txt body skip via _load_source_body returning None;
            # no harm in iterating them.
            target_sources = [r["id"] for r in db.conn.execute("SELECT id FROM source ORDER BY id")]

        # Build the form→toponym lookup ONCE up front. 21,969+ toponyms
        # in the live DB; rebuilding per-source would re-execute two
        # full-table scans per source.
        click.echo("Building form→toponym lookup...", err=True)
        form_to_toponym = _build_form_to_toponym_lookup(db.conn)
        click.echo(f"  {len(form_to_toponym):,} normalized forms indexed", err=True)

        for source_id in target_sources:
            report = reverse_search_source(
                db.conn, source_id, sources_dir, form_to_toponym, apply=apply
            )
            if report.pairs_extracted == 0:
                continue
            totals["pairs"] += report.pairs_extracted
            totals["matched"] += report.matched
            totals["unmatched"] += report.unmatched
            totals["inserted"] += report.inserted
            totals["already_present"] += report.already_present
            # Per-source progress goes to stderr — diagnostic output;
            # operators redirecting to a report file want the TOTAL
            # tally on stdout (set after the loop).
            click.echo(
                f"{source_id:<55} pairs={report.pairs_extracted:>5} "
                f"matched={report.matched:>5} unmatched={report.unmatched:>5} "
                f"inserted={report.inserted:>5} dup={report.already_present:>5}",
                err=True,
            )
            if verbose and report.unmatched_samples:
                # Unmatched forms are the prime candidates for Phase 2's
                # LLM-driven new-toponym discovery. Display ALL samples
                # the module collected (default cap is 10); operators
                # shouldn't see a smaller slice than the data the report
                # exposes.
                for form, year in report.unmatched_samples:
                    click.echo(f"    unmatched: {form!r} ({year})", err=True)
        # Commit once after walking all sources — the function-level
        # commit was removed for caller transaction control; this is
        # the natural commit point with partial-failure rollback
        # semantics if a later source raises.
        if apply:
            db.conn.commit()
    click.echo("")
    click.echo(
        f"TOTAL: pairs={totals['pairs']} matched={totals['matched']} "
        f"unmatched={totals['unmatched']} inserted={totals['inserted']} "
        f"dup={totals['already_present']} "
        f"({'APPLIED' if apply else 'dry-run'})"
    )


def add_to(parent: click.Group) -> None:
    """Register ``reverse-search-toponyms`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_reverse_search_toponyms)
