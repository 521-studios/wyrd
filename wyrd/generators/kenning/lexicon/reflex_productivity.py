"""Compute + persist reflex.productivity counts (wyrd-14p).

The reflex.productivity column was seeded as 0 by ``seed_from_meanings``
and never recomputed. This module derives the per-reflex productivity
count from the corpus's toponym_etymology_element observations: how
many distinct toponyms use each (surface_form, position) reflex.

The count feeds downstream generator weighting — a high-productivity
reflex like ``-ton`` should bias sampling more strongly than a one-off
hapax. Until this pass runs, every reflex reads productivity=0 and the
weighting collapses to flat-uniform.

Idempotent: the recompute zeros every row first, then writes the fresh
counts. Re-running on an unchanged corpus produces byte-identical row
values. Reflexes with no observed toponym usage (their etymons aren't
referenced by any toponym_etymology_element) read 0 — this is the
correct "no corpus support" value, not a stale leftover.
"""

from __future__ import annotations

from typing import Any

from wyrd.generators.kenning.lexicon.sql.queries.reflex import (
    RESET_REFLEX_PRODUCTIVITY,
    SELECT_REFLEX_PRODUCTIVITY_COUNTS,
    UPDATE_REFLEX_PRODUCTIVITY,
)


def recompute_reflex_productivity(db: Any) -> dict[str, int]:
    """Recompute productivity counts for every reflex from the corpus.

    Atomic: the reset + per-row updates run in a single transaction
    so concurrent readers see either the pre-recompute or post-recompute
    state, never a partial mix.

    Returns a stats dict:
      * ``total_reflexes`` — count of reflex rows in the table.
      * ``updated`` — count of reflexes with a non-zero productivity
        after the recompute (== count of reflexes with corpus support).
      * ``max_productivity`` — highest productivity count seen.
      * ``sum_productivity`` — sum across all reflexes (== total
        (toponym × element-position) hits across the corpus, modulo
        the distinct-toponym semantic).
    """
    conn = db.conn
    with conn:
        conn.execute(RESET_REFLEX_PRODUCTIVITY)
        rows = list(conn.execute(SELECT_REFLEX_PRODUCTIVITY_COUNTS))
        for reflex_id, productivity in rows:
            conn.execute(UPDATE_REFLEX_PRODUCTIVITY, (productivity, reflex_id))
    total = conn.execute("SELECT COUNT(*) FROM reflex").fetchone()[0]
    max_prod = max((p for _, p in rows), default=0)
    sum_prod = sum(p for _, p in rows)
    return {
        "total_reflexes": int(total),
        "updated": len(rows),
        "max_productivity": int(max_prod),
        "sum_productivity": int(sum_prod),
    }
