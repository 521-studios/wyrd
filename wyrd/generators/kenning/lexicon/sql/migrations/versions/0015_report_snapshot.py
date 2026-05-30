"""report_snapshot + report_metric — wyrd-d24 dashboard observability

Revision ID: 0015_report_snapshot
Revises: 0014_drop_personal_name_tables
Create Date: 2026-05-30

Ports the file-based dashboard report dirs (``data/mining/reports/<date>/``:
stats.txt, enrichment_status.txt, rando_port_readiness.txt, era_coverage.txt,
empirical_priors.json, drift_*.json) into a queryable pair of tables so
before/after comparisons are SQL-diffable instead of file-diffable — the D24
"add an observability path" rule applied to the snapshots themselves.

- ``report_snapshot`` is one row per captured snapshot, keyed by an operator
  ``label`` (e.g. ``before-rebuild-20260529`` / ``after-f1sstail-20260530``).
- ``report_metric`` is the long-form (category, metric) → value store. Numeric
  values land in ``value_num`` for SQL diffing; the verbatim string is kept in
  ``value_text`` so non-numeric cells (gate verdicts, etc.) round-trip too.
  UNIQUE(snapshot_label, category, metric) keeps each (category, metric) once
  per snapshot; the writer re-ingests idempotently by DELETE-then-INSERT of
  the snapshot's metric rows.
"""

from __future__ import annotations

from alembic import op

revision = "0015_report_snapshot"
down_revision = "0014_drop_personal_name_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE report_snapshot (
          label       TEXT PRIMARY KEY,
          captured_at TEXT NOT NULL,
          source_dir  TEXT,
          notes       TEXT
        )
        """
    )
    op.execute(
        """
        CREATE TABLE report_metric (
          id             INTEGER PRIMARY KEY AUTOINCREMENT,
          snapshot_label TEXT NOT NULL REFERENCES report_snapshot(label) ON DELETE CASCADE,
          category       TEXT NOT NULL,
          metric         TEXT NOT NULL,
          value_num      REAL,
          value_text     TEXT,
          UNIQUE (snapshot_label, category, metric)
        )
        """
    )
    op.execute("CREATE INDEX idx_report_metric_snapshot ON report_metric (snapshot_label)")
    op.execute("CREATE INDEX idx_report_metric_cat ON report_metric (category, metric)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS report_metric")
    op.execute("DROP TABLE IF EXISTS report_snapshot")
