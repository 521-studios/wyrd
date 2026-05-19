"""``wyrd kenning lexicon decompose`` — apply the trie matcher to a toponym corpus and persist canonical picks (wyrd-08m)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _load_meanings_data
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.meaning import load_meanings
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("decompose")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to meanings.json (defaults to bundled).",
)
@click.option(
    "--toponym-id",
    type=int,
    default=None,
    help=(
        "Run only this toponym (by id). Useful for spot-checking the "
        "picker on a single name. Without this, processes every "
        "toponym row up to --limit."
    ),
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process at most N toponyms. Without this, processes all.",
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Write the decomposition rows + canonical picks. Without this, dry-run.",
)
def lexicon_decompose(
    db_path: Path,
    meanings: Path | None,
    toponym_id: int | None,
    limit: int | None,
    apply_changes: bool,
) -> None:
    """Compute and persist matcher-derived decompositions per toponym
    (wyrd-08m Phase 1).

    For each toponym:
      1. Run ``Name.find_meaning(reduce=False)`` to enumerate every
         plausible breakdown (morphemes + unaccounted fragments).
      2. Persist each unique decomposition into
         ``toponym_decomposition`` (idempotent on signature).
      3. Apply the canonical picker (rules a + b — Phase 1).

    Rules (a) [scholar match] and (b) [unique-zero-unaccounted] are
    implemented. Multi-zero ties fall through with no canonical pick
    (rule (c) tiebreaker is a future enhancement). Consumer integration
    for ``rebuild-proportions`` and ``unaccounted`` lives in Phase 3a
    (wyrd-70s0); ``KenningExplain`` (Lambda / SPA path) has no DB access
    and remains heuristic-only until a future bundle-side projection
    lands.
    """
    meanings_data = _load_meanings_data(meanings)
    word_db, _ = load_meanings(meanings_data)

    from wyrd.generators.kenning.decomposition import populate_and_pick

    with LexiconDB(db_path) as db:
        if toponym_id is not None:
            rows = db.conn.execute(
                "SELECT id, modern_name FROM toponym WHERE id = ?",
                (toponym_id,),
            ).fetchall()
        else:
            sql = "SELECT id, modern_name FROM toponym ORDER BY id"
            params: tuple = ()
            if limit is not None:
                sql += " LIMIT ?"
                params = (limit,)
            rows = db.conn.execute(sql, params).fetchall()

        if not rows:
            click.echo("No toponym rows to process.", err=True)
            return

        rule_counts: dict[str, int] = {
            "scholar": 0,
            "scholar-disagreement": 0,
            "unique-zero-unaccounted": 0,
            "tiebreaker": 0,
            "no-canonical": 0,
        }
        decomposition_total = 0
        for idx, row in enumerate(rows, start=1):
            summary = populate_and_pick(db, row["id"], row["modern_name"], word_db)
            decomposition_total += int(summary["decomposition_count"] or 0)
            rule_key = summary["rule"] or "no-canonical"
            rule_counts[rule_key] = rule_counts.get(rule_key, 0) + 1
            if idx % 10 == 0 or idx == len(rows):
                click.echo(
                    f"  [{idx}/{len(rows)}] decompositions={decomposition_total} "
                    f"scholar={rule_counts['scholar']} "
                    f"scholar-disagreement={rule_counts['scholar-disagreement']} "
                    f"unique-zero={rule_counts['unique-zero-unaccounted']} "
                    f"tiebreaker={rule_counts['tiebreaker']} "
                    f"no-canonical={rule_counts['no-canonical']}",
                    err=True,
                )

        if not apply_changes:
            db.conn.rollback()
            click.echo("(dry-run; pass --apply to persist)", err=True)
            return
        db.commit()
        click.echo(
            f"applied={apply_changes} toponyms={len(rows)} "
            f"decompositions={decomposition_total} "
            f"scholar={rule_counts['scholar']} "
            f"scholar-disagreement={rule_counts['scholar-disagreement']} "
            f"unique-zero={rule_counts['unique-zero-unaccounted']} "
            f"tiebreaker={rule_counts['tiebreaker']} "
            f"no-canonical={rule_counts['no-canonical']}",
            err=True,
        )


def add_to(parent: click.Group) -> None:
    """Register ``decompose`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_decompose)
