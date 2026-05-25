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

# Priority hierarchy for ``toponym_decomposition.canonical_source`` per
# ``data/lexicon.sql:418-429`` (which mirrors the rules in
# ``runtime/decomposition.pick_canonical_decomposition``). Lower index
# = higher priority. Used by ``_merge_toponym_into`` to pick the
# winning canonical_source when both sides of a signature collision
# carry ``is_canonical=1`` with different sources.
_CANONICAL_SOURCE_PRIORITY = (
    "scholar",
    "scholar-disagreement",
    "unique-zero-unaccounted",
    "tiebreaker",
)


def _canonical_source_rank(source: str | None) -> int:
    """Return the priority rank (0 = highest) for a ``canonical_source``
    value. Unknown / NULL sources rank lowest (after every named
    source) so a known source always wins against an unknown."""
    if source is None:
        return len(_CANONICAL_SOURCE_PRIORITY)
    try:
        return _CANONICAL_SOURCE_PRIORITY.index(source)
    except ValueError:
        return len(_CANONICAL_SOURCE_PRIORITY)


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
            # Re-parent every FK-dependent row from the from-side to
            # the target row before dropping the from-side. Three
            # tables FK to toponym(id) with ON DELETE CASCADE
            # (toponym_etymology, toponym_attestation,
            # toponym_decomposition); ``_merge_toponym_into`` handles
            # all three. If we DROPPED first, CASCADE would silently
            # destroy the attestation + decomposition rows — the
            # decomposition history would just disappear.
            _merge_toponym_into(
                db,
                from_toponym_id=row["id"],
                into_toponym_id=existing["id"],
            )
            db.conn.execute("DELETE FROM toponym WHERE id = ?", (row["id"],))
            merged += 1

    db.commit()
    return BackfillResult(
        inspected=len(candidates),
        updated=updated,
        merged=merged,
        region_unknown=region_unknown,
    )


def _merge_toponym_into(db: LexiconDB, *, from_toponym_id: int, into_toponym_id: int) -> None:
    """Re-parent every FK-dependent row from ``from_toponym_id`` to
    ``into_toponym_id`` before the from-side ``toponym`` row is dropped.

    Three tables FK to ``toponym(id)`` with ``ON DELETE CASCADE``:

    * ``toponym_etymology`` — no unique constraint on ``toponym_id``;
      bulk UPDATE re-parents all rows. The dependent
      ``toponym_etymology_element`` rows ride along automatically
      because they FK to ``toponym_etymology(id)``, not to
      ``toponym(id)``.
    * ``toponym_attestation`` — no unique constraint on ``toponym_id``;
      bulk UPDATE.
    * ``toponym_decomposition`` — UNIQUE on
      ``(toponym_id, decomposition_signature)``. Per-row UPDATE,
      skipping (and DELETE-ing) any from-side row whose signature is
      already present on the target side (duplicate decompositions
      collapse — the target's pick wins, deterministic across re-runs
      because target rows are by definition older / pre-existing).

    Parameter names avoid the literary ``source_id`` column on
    ``toponym_etymology`` to keep merge-direction unambiguous —
    callers always read "merge FROM x INTO y"."""
    db.conn.execute(
        "UPDATE toponym_etymology SET toponym_id = ? WHERE toponym_id = ?",
        (into_toponym_id, from_toponym_id),
    )
    db.conn.execute(
        "UPDATE toponym_attestation SET toponym_id = ? WHERE toponym_id = ?",
        (into_toponym_id, from_toponym_id),
    )

    # toponym_decomposition has UNIQUE(toponym_id, decomposition_signature);
    # collapse from-side rows whose signature is already on the target.
    # When BOTH rows exist with the same signature, promote the
    # canonical flag (``is_canonical=1`` + ``canonical_source``) from
    # the from-side to the target row before dropping the from-side.
    # Otherwise the scholar pick would silently disappear in the merge.
    target_by_sig: dict[str, dict] = {
        row["decomposition_signature"]: dict(row)
        for row in db.conn.execute(
            "SELECT id, decomposition_signature, is_canonical, canonical_source "
            "FROM toponym_decomposition WHERE toponym_id = ?",
            (into_toponym_id,),
        )
    }
    for row in list(
        db.conn.execute(
            "SELECT id, decomposition_signature, is_canonical, canonical_source "
            "FROM toponym_decomposition WHERE toponym_id = ?",
            (from_toponym_id,),
        )
    ):
        sig = row["decomposition_signature"]
        if sig in target_by_sig:
            target_row = target_by_sig[sig]
            # Canonical-flag merge: pick the row whose canonical_source
            # has the higher priority per the schema's hierarchy
            # (scholar > scholar-disagreement > unique-zero-unaccounted
            # > tiebreaker; NULL ranks lowest). The surviving row is
            # the target (id stable, FKs unchanged); when the from-
            # side wins on priority we UPDATE the target's flags to
            # match before dropping the from-side. This preserves the
            # provenance hierarchy through merges — a from-side
            # ``scholar`` pick never silently demotes to a target's
            # ``unique-zero-unaccounted`` source.
            from_canonical = row["is_canonical"] or 0
            target_canonical = target_row["is_canonical"] or 0
            from_rank = _canonical_source_rank(row["canonical_source"])
            target_rank = _canonical_source_rank(target_row["canonical_source"])
            from_wins = (from_canonical, -from_rank) > (target_canonical, -target_rank)
            if from_wins:
                db.conn.execute(
                    "UPDATE toponym_decomposition "
                    "SET is_canonical = ?, canonical_source = ? WHERE id = ?",
                    (from_canonical, row["canonical_source"], target_row["id"]),
                )
            db.conn.execute("DELETE FROM toponym_decomposition WHERE id = ?", (row["id"],))
        else:
            db.conn.execute(
                "UPDATE toponym_decomposition SET toponym_id = ? WHERE id = ?",
                (into_toponym_id, row["id"]),
            )
