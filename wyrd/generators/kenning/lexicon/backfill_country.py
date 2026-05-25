"""One-shot backfill: derive ``toponym.country`` from ``toponym.region``
for rows where ``country`` is NULL.

The empirical-priors miner depends on ``toponym.country`` to group
baseline counts per culture (``country=England`` → ``culture=english``,
etc.). Pre-fix, the parser ingest path (Skeat / LLM mining / review CLI)
populated ``region`` but never ``country``, so every toponym with a
decomposition landed with ``country=NULL`` and the miner skipped them
all on ``country_unknown``.

This module:

1. Updates the in-place rows where the region maps unambiguously to a
   country via :func:`country_for_region`.
2. Handles the rare case where the country-backfilled row would collide
   with an existing ``(modern_name, country, region)`` triple — moves
   the dependent ``toponym_etymology`` FKs to the existing row, then
   drops the orphan-country row.

Idempotent: re-running against an already-backfilled DB does nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.regions import country_for_region


@dataclass
class BackfillResult:
    """Operator-readable summary of one ``backfill_toponym_country`` run."""

    inspected: int
    """Toponym rows with ``country IS NULL`` at the start of the run."""

    updated: int
    """Rows whose country was set in place from ``region``."""

    merged: int
    """Rows whose country-backfill would have collided with an existing
    ``(name, country, region)`` triple — the dependent ``toponym_etymology``
    FKs were moved to the existing row and the orphan was dropped."""

    region_unknown: int
    """Rows whose ``region`` doesn't map to any country (multi-country
    catch-alls like 'British Isles' / 'Europe', or NULL region with no
    other signal). Left untouched."""


def backfill_toponym_country(db: LexiconDB) -> BackfillResult:
    """Populate ``toponym.country`` from ``region`` for every row where
    ``country`` is currently NULL."""
    candidates = list(
        db.conn.execute(
            "SELECT id, modern_name, region FROM toponym WHERE country IS NULL OR country = ''"
        )
    )
    updated = 0
    merged = 0
    region_unknown = 0

    for row in candidates:
        country = country_for_region(row["region"])
        if country is None:
            region_unknown += 1
            continue

        # Does an existing row already hold (modern_name, country, region)?
        # If so we must merge rather than create a duplicate (the
        # idx_toponym_unique would reject the UPDATE).
        existing = db.conn.execute(
            """
            SELECT id FROM toponym
            WHERE modern_name = ?
              AND COALESCE(country, '') = ?
              AND COALESCE(region, '') = COALESCE(?, '')
              AND id != ?
            """,
            (row["modern_name"], country, row["region"], row["id"]),
        ).fetchone()

        if existing is None:
            db.conn.execute(
                "UPDATE toponym SET country = ? WHERE id = ?",
                (country, row["id"]),
            )
            updated += 1
        else:
            # Move dependent toponym_etymology rows to the existing
            # row, then drop the orphan. The unique index on
            # toponym_etymology is (toponym_id, source_id) — if both
            # rows have an etymology from the same source we let
            # SQLite's INSERT OR IGNORE-equivalent semantics drop the
            # duplicate (UPDATE will fail with UNIQUE constraint;
            # handled by retrying with INSERT OR REPLACE semantics
            # at the row level).
            _merge_toponym_into(db, source_id=row["id"], target_id=existing["id"])
            db.conn.execute("DELETE FROM toponym WHERE id = ?", (row["id"],))
            merged += 1

    db.conn.commit()
    return BackfillResult(
        inspected=len(candidates),
        updated=updated,
        merged=merged,
        region_unknown=region_unknown,
    )


def _merge_toponym_into(db: LexiconDB, *, source_id: int, target_id: int) -> None:
    """Move every ``toponym_etymology`` row from ``source_id`` to
    ``target_id``. When a (target_id, source_id) duplicate already
    exists on the target side, the source-side row is dropped (the
    target row's etymology is the merge winner — older row wins by id
    ordering, which is deterministic across re-runs)."""
    rows = list(
        db.conn.execute(
            "SELECT id, source_id FROM toponym_etymology WHERE toponym_id = ?",
            (source_id,),
        )
    )
    for row in rows:
        target_has = db.conn.execute(
            """
            SELECT 1 FROM toponym_etymology
            WHERE toponym_id = ? AND source_id = ?
            """,
            (target_id, row["source_id"]),
        ).fetchone()
        if target_has is None:
            db.conn.execute(
                "UPDATE toponym_etymology SET toponym_id = ? WHERE id = ?",
                (target_id, row["id"]),
            )
        else:
            # Cascade-drop the source-side etymology + its elements
            # so the FK constraint is satisfied when we delete the
            # source toponym row.
            db.conn.execute(
                "DELETE FROM toponym_etymology_element WHERE toponym_etymology_id = ?",
                (row["id"],),
            )
            db.conn.execute(
                "DELETE FROM toponym_etymology WHERE id = ?",
                (row["id"],),
            )
