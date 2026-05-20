"""etymon.phonological_vector — per-etymon 14-dim phonological feature vector

Revision ID: 0009_phonological_vector
Revises: 0008_views
Create Date: 2026-05-20

wyrd-kq7w.1 Phase A. Adds a single TEXT column on the ``etymon`` table
that holds the JSON-encoded :class:`PhonologicalVector` (14 named
dimensions + optional ``extras`` forward-compat slot). NULL on rows
that haven't been processed by the ``lexicon tag-phonological-vectors``
enrichment pass.

JSON over inline-column-per-dimension because the vector schema is
designed to evolve additively (new dimensions land in ``extras`` without
schema migration on every consumer). Single-column UPDATE per row,
single-column SELECT for the scoring runtime — read/write cost stays
flat as the dimension set grows.

The accompanying :mod:`wyrd.generators.kenning.lexicon.schema` legacy
migration helper carries the equivalent ``ALTER TABLE`` so legacy DBs
that don't run alembic still get the column.
"""

from __future__ import annotations

from alembic import op

revision = "0009_phonological_vector"
down_revision = "0008_views"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE etymon ADD COLUMN phonological_vector TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE etymon DROP COLUMN phonological_vector")
