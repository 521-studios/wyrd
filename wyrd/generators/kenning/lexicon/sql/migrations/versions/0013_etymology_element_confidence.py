"""toponym_etymology_element.confidence — wyrd-2n1.

Revision ID: 0013_etymology_element_confidence
Revises: 0012_attestation_source_doc_index
Create Date: 2026-05-21

wyrd-2n1: scholars hedge per-element ('the prefix is clearly X; the
suffix may be Y'). The pre-PR schema records one confidence per
toponym_etymology breakdown, so the per-element hedge is silently
flattened down to "what's the weakest element across the row?" at
recording time. Move confidence to toponym_etymology_element so the
per-element granularity survives ingest + dump + rebuild.

Back-compat: the parent ``toponym_etymology.confidence`` column stays
populated as the aggregate signal for legacy queries (review.py +
the dump's parent-level field both already consume it). The
migration backfills the new per-element column from the parent's
value so every existing element inherits its parent's confidence —
a lossless starting point that future mining can then refine
per-element.

Constraint mirrors the parent: ``confidence IN ('high', 'medium',
'low')``. Nullable matches the parent (an element with NULL
confidence reads as "scholars didn't hedge this element
explicitly," distinct from the three rated levels).
"""

from __future__ import annotations

from alembic import op

revision = "0013_etymology_element_confidence"
down_revision = "0012_attestation_source_doc_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE toponym_etymology_element ADD COLUMN confidence TEXT "
        "CHECK (confidence IN ('high', 'medium', 'low'))"
    )
    op.execute(
        """
        UPDATE toponym_etymology_element
           SET confidence = (
                SELECT te.confidence
                  FROM toponym_etymology AS te
                 WHERE te.id = toponym_etymology_element.toponym_etymology_id
           )
         WHERE confidence IS NULL
        """
    )


def downgrade() -> None:
    # SQLite ALTER TABLE DROP COLUMN landed in 3.35 (2021-03); the
    # bundled Python sqlite3 build supports it. Down-migration is
    # rarely run in practice but kept for symmetry.
    op.execute("ALTER TABLE toponym_etymology_element DROP COLUMN confidence")
