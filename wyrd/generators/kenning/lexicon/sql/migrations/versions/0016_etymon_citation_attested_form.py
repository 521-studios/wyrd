"""etymon_citation.attested_form — wyrd-jhdw hybrid cite migration

Revision ID: 0016_etymon_citation_attested_form
Revises: 0015_report_snapshot
Create Date: 2026-05-30

Adds a nullable ``attested_form`` column to ``etymon_citation``, for the
consolidation epic's HYBRID citation handling (wyrd-v3ow / wyrd-jhdw, P1).

When a form-of / variant etymon is folded into its parent lemma (e.g. the
scholar-cited ``burh`` collapses into ``burg``), its scholarly citations
MIGRATE to the parent so coverage / scholar-witness queries keep counting
them off the surviving lemma. But the source actually attested the *form*
(Ekwall wrote "burh", not "burg"), so the migrating citation stamps the
original surface form here. Result: the cite reads off the parent
(``etymon_id`` → ``burg``) for metrics, while ``attested_form`` preserves
"Ekwall attests the burh form of burg" for display + audit.

``NULL`` (the default, and every pre-existing row) means "attested under
the etymon's own canonical form" — no migration happened. Only the collapse
applier (run_full_enrichment step 3, replaying ``data/mining/_collapses.jsonl``)
writes a non-NULL value.
"""

from __future__ import annotations

from alembic import op

revision = "0016_etymon_citation_attested_form"
down_revision = "0015_report_snapshot"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE etymon_citation ADD COLUMN attested_form TEXT")


def downgrade() -> None:
    # SQLite ALTER TABLE DROP COLUMN (3.35+, 2021-03) — the bundled
    # sqlite3 supports it. Kept for symmetry; rarely run.
    op.execute("ALTER TABLE etymon_citation DROP COLUMN attested_form")
