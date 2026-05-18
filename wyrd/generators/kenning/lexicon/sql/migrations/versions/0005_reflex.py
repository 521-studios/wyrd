"""reflex layer — modern surface form ↔ etymon mapping

Revision ID: 0005_reflex
Revises: 0004_descent
Create Date: 2026-05-18

A modern surface fragment can descend from multiple etymons across
languages (ON býr / OE byrh both show as ``-bury`` in some toponym
endings). The reflex table is keyed by (surface_form, position); the
many-to-many ``reflex_etymon`` join carries the actual mapping.
Populated during seed (``lexicon build`` from meanings.json) and
otherwise read-only.
"""

from __future__ import annotations

from alembic import op

revision = "0005_reflex"
down_revision = "0004_descent"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE reflex (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          surface_form  TEXT NOT NULL,
          position      TEXT NOT NULL CHECK (position IN ('pre', 'post', 'inner')),
          productivity  INTEGER NOT NULL DEFAULT 0,
          UNIQUE (surface_form, position)
        )
        """
    )
    op.execute("CREATE INDEX idx_reflex_position ON reflex(position)")

    op.execute(
        """
        CREATE TABLE reflex_etymon (
          reflex_id INTEGER NOT NULL REFERENCES reflex(id) ON DELETE CASCADE,
          etymon_id INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          PRIMARY KEY (reflex_id, etymon_id)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE reflex_etymon")
    op.execute("DROP TABLE reflex")
