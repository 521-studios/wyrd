"""``wyrd kenning lexicon report`` — corpus snapshot: top morphemes + per-source contribution."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.lexicon.vocab_health import (
    VIOLATION_SEVERITIES,
    Severity,
    scan_countries,
    scan_languages,
    scan_regions,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("report")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--top",
    type=int,
    default=15,
    show_default=True,
    help="Show top-N items in each ranking section.",
)
def lexicon_report(db_path: Path, top: int) -> None:
    """Read-only summary of the lexicon: who said what, where it agrees."""
    with _readonly_lexicon(db_path) as conn:
        _emit_report(conn, top)


def _emit_report(conn, top: int) -> None:
    """Body of the report command. Thin sequencer over the per-section
    helpers below — each section is its own function so adding a new
    one is a one-line append here + a new ``_section_*`` helper, and
    the overall function stays under the C901 cyclomatic threshold.
    wyrd-h0u1 + wyrd-h6x7 round-2 complexity fix."""
    _section_sources(conn)
    _section_totals(conn)
    _section_consensus(conn)
    _section_top_etymons(conn, top)
    _section_languages(conn, top)
    _section_confidence(conn)
    _section_countries(conn)
    _section_regions(conn, top)
    _section_toponyms_per_source(conn)
    _section_disagreements(conn, top)


def _section_sources(conn) -> None:
    click.echo("=== sources ===")
    for r in conn.execute("SELECT id, year, region FROM source ORDER BY year, id").fetchall():
        year = r["year"] or "?"
        region = r["region"] or "?"
        click.echo(f"  {r['id']:50}  {year}  {region}")
    click.echo("")


def _section_totals(conn) -> None:
    click.echo("=== totals ===")
    for tbl in (
        "etymon",
        "etymon_citation",
        "etymon_gloss",
        "etymon_tag",
        "reflex",
        "reflex_etymon",
        "toponym",
        "toponym_etymology",
        "toponym_etymology_element",
    ):
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]  # noqa: S608
        click.echo(f"  {tbl:32} {n:>8}")
    click.echo("")


def _section_consensus(conn) -> None:
    click.echo("=== consensus ===")
    rows = conn.execute(
        "SELECT witnesses, COUNT(*) AS n FROM etymon_consensus "
        "GROUP BY witnesses ORDER BY witnesses DESC"
    ).fetchall()
    for r in rows:
        click.echo(f"  etymons cited by {r['witnesses']:>2} sources: {r['n']:>5}")
    click.echo("")


def _section_top_etymons(conn, top: int) -> None:
    click.echo(f"=== top {top} etymons by witnesses ===")
    rows = conn.execute(
        "SELECT canonical_form, language, witnesses FROM etymon_consensus "
        "WHERE witnesses > 1 ORDER BY witnesses DESC, canonical_form LIMIT ?",
        (top,),
    ).fetchall()
    for r in rows:
        click.echo(f"  {r['canonical_form']:24} ({r['language']:14})  {r['witnesses']:>2} sources")
    click.echo("")


def _mark(severity: Severity) -> str:
    """Inline marker for a stale/dirty value; ``ok`` and ``info`` read clean. The
    ``info`` bucket is everything expected/out-of-scope — NULLs, deferred cultural
    zones (e.g. ``England (Danelaw)``), and non-England regions with no model yet
    (the full mapping is ``vocab_health._REGION_SEVERITY`` + the _classify_* funcs)."""
    return f"  [{severity}]" if severity in VIOLATION_SEVERITIES else ""


def _section_languages(conn, top: int) -> None:
    top = max(0, top)  # a negative --top would otherwise slice from the end
    findings = scan_languages(conn)
    noncanon = sum(1 for f in findings if f.severity in VIOLATION_SEVERITIES)
    click.echo(
        f"=== top {top} languages of mined etymons "
        f"({len(findings)} distinct, {noncanon} non-canonical) ==="
    )
    for f in findings[:top]:
        click.echo(f"  {(f.value or '(null)'):20} {f.count:>6}{_mark(f.severity)}")
    click.echo("")


def _section_countries(conn) -> None:
    # No top cap (unlike regions/languages): the canonical country set is tiny
    # (~10), so the full distribution always fits.
    click.echo("=== toponym countries ===")
    for f in scan_countries(conn):
        click.echo(f"  {(f.value or '(null)'):28} {f.count:>7}{_mark(f.severity)}")
    click.echo("")


def _section_regions(conn, top: int) -> None:
    top = max(0, top)  # a negative --top would otherwise slice from the end
    findings = scan_regions(conn)
    noncanon = sum(1 for f in findings if f.severity in VIOLATION_SEVERITIES)
    click.echo(
        f"=== top {top} toponym regions ({len(findings)} distinct, {noncanon} non-canonical) ==="
    )
    for f in findings[:top]:
        label = f"{f.value or '(null)'} @ {f.context}" if f.context else (f.value or "(null)")
        click.echo(f"  {label:34} {f.count:>7}{_mark(f.severity)}")
    click.echo("")


def _section_confidence(conn) -> None:
    click.echo("=== confidence distribution (toponym_etymology) ===")
    rows = conn.execute(
        "SELECT confidence, COUNT(*) AS n FROM toponym_etymology "
        "GROUP BY confidence ORDER BY n DESC"
    ).fetchall()
    for r in rows:
        click.echo(f"  {r['confidence'] or '(null)':12} {r['n']:>6}")
    click.echo("")


def _section_toponyms_per_source(conn) -> None:
    click.echo("=== toponyms per source ===")
    rows = conn.execute(
        "SELECT source_id, COUNT(*) AS n FROM toponym_etymology GROUP BY source_id ORDER BY n DESC"
    ).fetchall()
    for r in rows:
        click.echo(f"  {r['source_id']:50} {r['n']:>5}")
    click.echo("")


def _section_disagreements(conn, top: int) -> None:
    click.echo("=== disagreements ===")
    # A toponym with >1 distinct breakdown signature has scholars disagreeing.
    disagreement_count = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT toponym_id
            FROM toponym_breakdown_signature
            GROUP BY toponym_id
            HAVING COUNT(DISTINCT signature) > 1
        )
        """
    ).fetchone()[0]
    click.echo(f"  toponyms with conflicting breakdowns: {disagreement_count}")
    if disagreement_count > 0 and top > 0:
        click.echo(f"  examples (first {min(top, disagreement_count)}):")
        rows = conn.execute(
            """
            SELECT t.modern_name,
                   GROUP_CONCAT(DISTINCT tbs.signature) AS sigs,
                   GROUP_CONCAT(DISTINCT tbs.source_id) AS srcs
            FROM toponym_breakdown_signature tbs
            JOIN toponym t ON t.id = tbs.toponym_id
            GROUP BY tbs.toponym_id
            HAVING COUNT(DISTINCT tbs.signature) > 1
            LIMIT ?
            """,
            (top,),
        ).fetchall()
        for r in rows:
            click.echo(f"    {r['modern_name']:24} sources={r['srcs']}")


def add_to(parent: click.Group) -> None:
    """Register ``report`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_report)
