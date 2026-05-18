"""sources — bibliographic record table

Revision ID: 0001_sources
Revises:
Create Date: 2026-05-18

The bibliographic source row that every etymon citation, descent edge,
text match, mining run, etymology element, and variant points at.
First migration in the layered baseline because every other table has
a FK back into ``source.id``.
"""

from __future__ import annotations

from alembic import op

revision = "0001_sources"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE source (
          id              TEXT PRIMARY KEY,
          author          TEXT,
          title           TEXT NOT NULL,
          year            INTEGER,
          region          TEXT,
          language_focus  TEXT,
          notes           TEXT
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE source")
