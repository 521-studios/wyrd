"""``wyrd kenning lexicon refill-short-quotes`` — context-snippet refill for LLM-truncated citation short_quotes (wyrd-bd68)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


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
        # Validate --source values up front (silent-failure-hunter
        # round-1 finding): a misspelled source id used to silently
        # produce zero output, masking operator intent. Surface as a
        # ClickException with the list of unknown ids.
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
            # Only iterate sources that actually have citations
            # (Gemini PR #211 round-7 efficiency). Many sources in
            # the source table have zero citations (placeholder rows
            # for not-yet-mined books, retired sources, etc.) —
            # skipping them avoids per-source refill_source calls
            # that would return empty reports anyway.
            target_sources = [
                r[0]
                for r in db.conn.execute(
                    "SELECT DISTINCT source_id FROM etymon_citation ORDER BY source_id"
                )
            ]

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
            # Gemini PR #211 round-2 visibility fix (kept narrow per
            # round-5 reconsideration on hallucinated false-positives,
            # broadened scope per Gemini round-5b on default-mode
            # visibility): warning fires when refill_source returned
            # without ever consulting the source body — which only
            # happens when _load_source_body returned None (missing
            # file, UnicodeDecodeError, OSError, or empty/whitespace-
            # only body). The narrow condition (refilled==0 AND
            # hallucinated==0) correctly distinguishes 'source body
            # unavailable' from 'source body fine but LLM
            # hallucinated everything'.
            #
            # No longer gated on `sources` (explicit --source) per
            # Gemini round-5b: default-mode runs across all sources
            # should ALSO surface missing-body gaps. The narrow
            # condition keeps the noise floor low (only fires for
            # sources that have truncated rows but no consulted
            # source body), so default-mode runs only warn for
            # sources where the operator actually has a refill
            # surface to fix.
            if report.refilled == 0 and report.hallucinated == 0:
                # Use the centralized helper so this warning's reported
                # path is guaranteed to match what _load_source_body
                # looked at (Gemini PR #211 round-9 DRY fix; the
                # round-7 path-traversal hardening + round-8
                # consistency fix now live in one place).
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
            if verbose:
                # Surface RefillReport.samples so operators can
                # spot-check before/after quality without manual DB
                # inspection .
                for sample in report.samples:
                    click.echo(f"    [{sample.status}] {sample.etymon_ref}")
                    if sample.old_short_quote:
                        click.echo(f"      old (...{sample.old_short_quote[-60:]!r})")
                    if sample.new_short_quote:
                        click.echo(
                            f"      new (+{sample.recovered_chars}c: "
                            f"...{sample.new_short_quote[-100:]!r})"
                        )
    click.echo("")
    click.echo(
        f"TOTAL: truncated={total_truncated} refilled={total_refilled} "
        f"hallucinated={total_hallucinated} "
        f"({'APPLIED' if apply else 'dry-run'})"
    )


def add_to(parent: click.Group) -> None:
    """Register ``refill-short-quotes`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_refill_short_quotes)
