"""``wyrd kenning lexicon report`` — corpus snapshot: top morphemes + per-source contribution."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
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
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    def q(sql: str, *args) -> list:
        return conn.execute(sql, args).fetchall()

    def count(sql: str, *args) -> int:
        return conn.execute(sql, args).fetchone()[0]

    click.echo("=== sources ===")
    for r in q("SELECT id, year, region FROM source ORDER BY year, id"):
        year = r["year"] or "?"
        region = r["region"] or "?"
        click.echo(f"  {r['id']:50}  {year}  {region}")

    click.echo("")
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
        click.echo(f"  {tbl:32} {count(f'SELECT COUNT(*) FROM {tbl}'):>8}")

    click.echo("")
    click.echo("=== consensus ===")
    consensus_dist = q(
        "SELECT witnesses, COUNT(*) AS n FROM etymon_consensus "
        "GROUP BY witnesses ORDER BY witnesses DESC"
    )
    for r in consensus_dist:
        click.echo(f"  etymons cited by {r['witnesses']:>2} sources: {r['n']:>5}")

    click.echo("")
    click.echo(f"=== top {top} etymons by witnesses ===")
    rows = q(
        "SELECT canonical_form, language, witnesses FROM etymon_consensus "
        "WHERE witnesses > 1 ORDER BY witnesses DESC, canonical_form LIMIT ?",
        top,
    )
    for r in rows:
        click.echo(f"  {r['canonical_form']:24} ({r['language']:14})  {r['witnesses']:>2} sources")

    click.echo("")
    click.echo("=== languages of mined etymons ===")
    rows = q("SELECT language, COUNT(*) AS n FROM etymon GROUP BY language ORDER BY n DESC")
    for r in rows:
        click.echo(f"  {r['language']:20} {r['n']:>6}")

    click.echo("")
    click.echo("=== confidence distribution (toponym_etymology) ===")
    rows = q(
        "SELECT confidence, COUNT(*) AS n FROM toponym_etymology "
        "GROUP BY confidence ORDER BY n DESC"
    )
    for r in rows:
        click.echo(f"  {r['confidence'] or '(null)':12} {r['n']:>6}")

    click.echo("")
    click.echo("=== toponyms per source ===")
    rows = q(
        "SELECT source_id, COUNT(*) AS n FROM toponym_etymology GROUP BY source_id ORDER BY n DESC"
    )
    for r in rows:
        click.echo(f"  {r['source_id']:50} {r['n']:>5}")

    click.echo("")
    click.echo("=== disagreements ===")
    # A toponym with >1 distinct breakdown signature has scholars disagreeing.
    disagreement_count = count("""
        SELECT COUNT(*) FROM (
            SELECT toponym_id
            FROM toponym_breakdown_signature
            GROUP BY toponym_id
            HAVING COUNT(DISTINCT signature) > 1
        )
        """)
    click.echo(f"  toponyms with conflicting breakdowns: {disagreement_count}")

    if disagreement_count > 0 and top > 0:
        click.echo(f"  examples (first {min(top, disagreement_count)}):")
        rows = q(
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
            top,
        )
        for r in rows:
            click.echo(f"    {r['modern_name']:24} sources={r['srcs']}")

    conn.close()


def add_to(parent: click.Group) -> None:
    """Register ``report`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_report)
