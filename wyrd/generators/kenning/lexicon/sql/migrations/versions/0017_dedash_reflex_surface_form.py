"""de-dash reflex.surface_form — wyrd-aicu.3 (D45)

A morpheme's stored identity is its BARE surface; position is an explicit
axis, never dash-encoded (D45). ``reflex.surface_form`` historically stored a
dash-DECORATED surface that merely re-encoded the value already in the
``reflex.position`` column: ``pre`` → ``Great-``, ``inner`` → ``-great-``,
``post`` → ``-great``. The dash is redundant with ``position`` and every
reader already SELECTs the ``position`` column, so the dash carries no
information — it just leaks position decoration into the identity.

This migration strips the dashes with ``TRIM(surface_form, '-')`` — the
position markers are always LEADING and/or TRAILING (``pre`` trailing, ``post``
leading, ``inner`` both), so ``TRIM`` removes them while preserving any
legitimate INTERNAL hyphen in a surface (e.g. a hyphenated form like
``al-adha``); a blanket ``REPLACE(... '-', '')`` would corrupt those. Because
de-dashing collapses the up-to-3 dash-variants of a surface onto one
``(bare_surface, position)`` key, it would
violate ``UniqueConstraint(surface_form, position)`` — so colliding reflexes
are MERGED first: per ``(bare_surface, position)`` group the lowest ``id``
survives, the losers' ``reflex_etymon`` links are repointed onto the survivor
(``INSERT OR IGNORE`` dedups the ``(reflex_id, etymon_id)`` PK when two losers
shared an etymon), the loser links + loser reflexes are deleted, and ONLY THEN
are the survivors' surfaces de-dashed — so the unique constraint is never
violated mid-flight.

``reflex.productivity`` is recomputed from corpus observations at the rebuild
tail (wyrd-14p), so the survivor's stale value is corrected downstream; we
don't sum it here.

NB the dash is recoverable from the ``position`` column, but the MERGE is
destructive (loser ids + any productivity divergence are gone), so the
downgrade re-dashes from ``position`` without un-merging — it cannot restore
the pre-merge row identities.
"""

from __future__ import annotations

from alembic import op

revision = "0017_dedash_reflex_surface_form"
down_revision = "0016_etymon_citation_attested_form"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # All work below runs on the STILL-DASHED surface_form; the surface is not
    # changed until the final UPDATE, so UniqueConstraint(surface_form,
    # position) is never violated mid-migration.

    # Survivor per (bare_surface, position) group = MIN(id) — deterministic,
    # stable, lowest id wins.
    op.execute(
        """
        CREATE TEMP TABLE _reflex_survivor AS
        SELECT TRIM(surface_form, '-') AS bare, position, MIN(id) AS survivor_id
        FROM reflex
        GROUP BY TRIM(surface_form, '-'), position
        """
    )
    # Map every reflex to its survivor (a survivor maps to itself).
    op.execute(
        """
        CREATE TEMP TABLE _reflex_merge AS
        SELECT r.id AS loser_id, s.survivor_id
        FROM reflex r
        JOIN _reflex_survivor s
          ON s.bare = TRIM(r.surface_form, '-') AND s.position = r.position
        """
    )

    # Repoint loser links onto the survivor. INSERT OR IGNORE dedups against the
    # (reflex_id, etymon_id) PK when the survivor already links the etymon, or
    # when two losers shared one etymon.
    op.execute(
        """
        INSERT OR IGNORE INTO reflex_etymon (reflex_id, etymon_id)
        SELECT m.survivor_id, re.etymon_id
        FROM reflex_etymon re
        JOIN _reflex_merge m ON m.loser_id = re.reflex_id
        WHERE m.loser_id <> m.survivor_id
        """
    )
    # Drop the loser links, then the now-orphaned loser reflex rows. (Explicit,
    # not via ON DELETE CASCADE — FK enforcement is off by default under SQLite
    # so the cascade can't be relied on inside the migration.)
    op.execute(
        """
        DELETE FROM reflex_etymon
        WHERE reflex_id IN (
            SELECT loser_id FROM _reflex_merge WHERE loser_id <> survivor_id
        )
        """
    )
    op.execute(
        """
        DELETE FROM reflex
        WHERE id IN (
            SELECT loser_id FROM _reflex_merge WHERE loser_id <> survivor_id
        )
        """
    )

    # Now every (bare, position) group has exactly one row — de-dash in place
    # without colliding.
    op.execute("UPDATE reflex SET surface_form = TRIM(surface_form, '-')")

    op.execute("DROP TABLE _reflex_merge")
    op.execute("DROP TABLE _reflex_survivor")


def downgrade() -> None:
    # Best-effort: the dash is recoverable from the position column, but the
    # MERGE is irreversible (loser rows are gone). Re-decorate the surviving
    # surfaces so a downgraded DB at least matches the old dash convention;
    # the pre-merge row identities cannot be restored.
    op.execute(
        """
        UPDATE reflex SET surface_form = CASE position
            WHEN 'pre'   THEN upper(substr(surface_form, 1, 1)) || substr(surface_form, 2) || '-'
            WHEN 'inner' THEN '-' || lower(surface_form) || '-'
            WHEN 'post'  THEN '-' || lower(surface_form)
            ELSE surface_form
        END
        """
    )
