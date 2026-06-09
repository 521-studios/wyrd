"""``wyrd kenning lexicon refill-short-quotes`` — context-snippet refill for LLM-truncated citation short_quotes (wyrd-bd68)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


def _resolve_target_sources(conn, sources: tuple[str, ...]) -> list[str]:
    """Resolve which source_ids to refill.

    With explicit ``--source`` values, validate them up front (a
    misspelled id used to silently produce zero output) and raise a
    ``ClickException`` listing any unknown ids. Without ``--source``,
    only return sources that actually have citations (Gemini PR #211
    round-7 efficiency) — many `source` rows are placeholders with zero
    citations whose refill_source call would return empty anyway.
    """
    if sources:
        known = {r["id"] for r in conn.execute("SELECT id FROM source")}
        unknown = sorted(set(sources) - known)
        if unknown:
            raise click.ClickException(
                f"unknown source_id(s): {', '.join(unknown)}. "
                f"Check spelling against `lexicon report --sources` output."
            )
        return list(sources)
    return [
        r[0]
        for r in conn.execute("SELECT DISTINCT source_id FROM etymon_citation ORDER BY source_id")
    ]


def _warn_if_body_missing(report, sources_dir: Path, source_id: str) -> None:
    """Warn when refill_source never consulted the source body.

    Fires only when the report has truncated rows but neither refilled
    nor flagged-hallucinated any — which happens exactly when
    ``_load_source_body`` returned None (missing file, decode error, or
    empty/whitespace-only body). The narrow ``refilled==0 AND
    hallucinated==0`` condition distinguishes "source body unavailable"
    from "source body fine but every row hallucinated" (Gemini PR #211
    rounds 2 / 5b). Uses ``source_body_path`` so the reported path
    matches what ``_load_source_body`` looked at (round-9 DRY fix).
    """
    if report.refilled != 0 or report.hallucinated != 0:
        return
    from wyrd.generators.kenning.short_quote_refill import source_body_path

    txt = source_body_path(sources_dir, source_id)
    if not txt.exists():
        click.echo(f"    warning: {txt} not found — refill needs the source body")
    else:
        click.echo(
            f"    warning: {txt} exists but read returned empty/unreadable "
            f"content (non-UTF8 bytes? whitespace-only?) — refill needs a "
            f"readable source body"
        )


def _echo_refill_samples(report) -> None:
    """Print before/after RefillReport.samples so operators can
    spot-check refill quality without manual DB inspection."""
    for sample in report.samples:
        click.echo(f"    [{sample.status}] {sample.etymon_ref}")
        if sample.old_short_quote:
            click.echo(f"      old (...{sample.old_short_quote[-60:]!r})")
        if sample.new_short_quote:
            click.echo(
                f"      new (+{sample.recovered_chars}c: ...{sample.new_short_quote[-100:]!r})"
            )


@click.command("refill-short-quotes")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Lexicon SQLite DB to read + (with --apply) write.",
)
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("sources"),
    show_default=True,
    help="Directory of <source_id>.txt source bodies for refill lookup.",
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
    help="Actually UPDATE etymon_citation.short_quote rows. Without this, dry-run.",
)
@click.option(
    "--window",
    type=int,
    default=200,
    show_default=True,
    help="Forward chars of recovered context past the matched tail.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print before/after samples per source so operators can spot-check the refill quality.",
)
def lexicon_refill_short_quotes(
    db_path: Path,
    sources_dir: Path,
    sources: tuple[str, ...],
    apply: bool,
    window: int,
    verbose: bool,
) -> None:
    """Refill truncated short_quotes from source body text (wyrd-1hpc).

    For each citation whose short_quote is flagged by
    `audit-short-quotes`, locates the tail in `sources/<source_id>.txt`
    and extends forward by --window chars to recover the
    LLM-truncated continuation. Non-locatable cases (LLM-hallucinated
    commentary that was never in the source body) are reported as
    `hallucinated`; those need an LLM re-mine, not refill.

    After --apply, run `lexicon dump-jsonl` to propagate the refilled
    short_quotes into the committed `data/mining/*.jsonl` files.
    """
    # Deferred for CLI help-snappiness — the refill module imports
    # short_quote_audit (re + regex compilation), and we only want
    # to pay that cost when the command actually runs. Pattern
    # documented in wyrd-xz6l.
    from wyrd.generators.kenning.short_quote_refill import refill_source

    total_truncated = 0
    total_refilled = 0
    total_hallucinated = 0

    # Use LexiconDB context manager for consistency with other lexicon
    # commands (Gemini PR #211 round-2): the wrapper sets PRAGMA
    # foreign_keys=ON + synchronous=NORMAL on every open, which the
    # bare sqlite3.connect skipped.
    with LexiconDB(db_path) as db:
        target_sources = _resolve_target_sources(db.conn, sources)

        for source_id in target_sources:
            report = refill_source(db.conn, source_id, sources_dir, apply=apply, window=window)
            if report.total_truncated == 0:
                continue
            total_truncated += report.total_truncated
            total_refilled += report.refilled
            total_hallucinated += report.hallucinated
            # Report content goes to stdout so operators can redirect
            # cleanly (`refill-short-quotes > report.txt`) while genuine
            # errors stay on stderr. Matches sibling report-generating
            # CLIs (audit-short-quotes, audit-etymology-alignment).
            # Gemini PR #211 round-4.
            click.echo(
                f"{source_id:<55} truncated={report.total_truncated:>4} "
                f"refilled={report.refilled:>4} "
                f"hallucinated={report.hallucinated:>4}"
            )
            _warn_if_body_missing(report, sources_dir, source_id)
            if verbose:
                _echo_refill_samples(report)
    click.echo("")
    click.echo(
        f"TOTAL: truncated={total_truncated} refilled={total_refilled} "
        f"hallucinated={total_hallucinated} "
        f"({'APPLIED' if apply else 'dry-run'})"
    )


def add_to(parent: click.Group) -> None:
    """Register ``refill-short-quotes`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_refill_short_quotes)
