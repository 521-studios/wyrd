"""toponym_attestation source_doc index — wyrd-rhnw / wyrd-x82p Phase 2b.1

Revision ID: 0012_attestation_source_doc_index
Revises: 0011_fantasy_morpheme
Create Date: 2026-05-20

wyrd-rhnw: ``idx_attestation_unique`` has ``source_doc`` as the LAST
composite column, so the bulk preflight ``SELECT WHERE source_doc=?``
in attestation mining falls back to a full-table scan. At current scale
(~5-50K attestations/source × 32 sources, ~10ms full scan) the
N+1→bulk-SELECT fix in PR #214 already dominates. But Phase 2b.2 brings
multi-million-row corpus extraction online, where the missing dedicated
index will start to matter.

Adds ``idx_attestation_source_doc`` directly on ``source_doc`` so SQLite
can use it for the preflight without a leading-column-of-the-composite
contortion.
"""

from __future__ import annotations

from alembic import op

revision = "0012_attestation_source_doc_index"
down_revision = "0011_fantasy_morpheme"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE INDEX idx_attestation_source_doc ON toponym_attestation(source_doc)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_attestation_source_doc")
