"""etymon_id indexes on reflex_etymon + toponym_etymology_element — wyrd-53ep.

Revision ID: 0020_select_targets_indexes
Revises: 0019_canonicalization_graph
Create Date: 2026-06-20

wyrd-53ep / PR #681: ``tag_mining.select_targets`` correlates two reachability
subqueries on ``etymon_id`` — an ``EXISTS`` on ``reflex_etymon`` and a
``count(DISTINCT toponym_id)`` on ``toponym_etymology_element`` — but neither
table had an index on ``etymon_id``: ``reflex_etymon``'s PK leads with
``reflex_id`` and the element table's PK is ``(toponym_etymology_id, ordinal)``,
so ``WHERE etymon_id = ?`` fell back to a full table SCAN *inside* the
per-etymon correlated subquery. Over the 2.4M-row ``etymon`` outer scan on the
4.8GB authoring DB that made the query run for minutes (EXPLAIN QUERY PLAN
showed ``SCAN re`` / ``SCAN el``). These indexes turn those into SEARCHes.

They also back the ``ON DELETE CASCADE`` from ``etymon``: SQLite scans the
referencing child rows on a parent delete, and an index on the referencing
column avoids that scan.
"""

from __future__ import annotations

from alembic import op

revision = "0020_select_targets_indexes"
down_revision = "0019_canonicalization_graph"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE INDEX idx_reflex_etymon_etymon ON reflex_etymon(etymon_id)")
    op.execute(
        "CREATE INDEX idx_toponym_etymology_element_etymon ON toponym_etymology_element(etymon_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_toponym_etymology_element_etymon")
    op.execute("DROP INDEX IF EXISTS idx_reflex_etymon_etymon")
