"""descent graph, variants, mining-run audit (D27, wyrd-fqil, D23)

Revision ID: 0004_descent
Revises: 0003_citations
Create Date: 2026-05-18

Three audit / evidence layers that ride alongside the
citation / text-match tables in 0003:

* ``etymon_descent`` — D27 directed etymological-descent graph
  (parent ←→ child with edge_type). Populated by Wiktionary mining
  (wyrd-4rt) and by LLM-extracted chain assertions ("from OE tūn").
* ``etymon_variant`` — wyrd-fqil per-form alternative / inflectional /
  romanized variants harvested from wiktextract ``forms`` arrays.
* ``mining_run`` — D23 per-(book, provider, model, mode) audit log.
  Was lost-to-stdout pre-D23; now persisted for "what's the state of
  the harvest" queries.
"""

from __future__ import annotations

from alembic import op

revision = "0004_descent"
down_revision = "0003_citations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE etymon_descent (
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          parent_id   INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          child_id    INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          edge_type   TEXT NOT NULL CHECK (edge_type IN (
                        'inheritance', 'borrowing', 'calque',
                        'compound', 'derivation', 'cognate', 'unknown'
                      )),
          source_id   TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
          confidence  TEXT CHECK (confidence IN ('high', 'medium', 'low')),
          notes       TEXT,
          UNIQUE (parent_id, child_id, edge_type, source_id)
        )
        """
    )
    op.execute("CREATE INDEX idx_etymon_descent_parent ON etymon_descent(parent_id)")
    op.execute("CREATE INDEX idx_etymon_descent_child  ON etymon_descent(child_id)")

    op.execute(
        """
        CREATE TABLE etymon_variant (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          etymon_id     INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          form          TEXT NOT NULL COLLATE NOCASE,
          variant_class TEXT NOT NULL CHECK (variant_class IN (
                          'alternative', 'inflection', 'romanization',
                          'canonical', 'other'
                        )),
          tags          TEXT,
          source_id     TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
          UNIQUE (etymon_id, form, variant_class)
        )
        """
    )
    op.execute("CREATE INDEX idx_etymon_variant_etymon ON etymon_variant(etymon_id)")
    op.execute("CREATE INDEX idx_etymon_variant_form   ON etymon_variant(form)")
    op.execute("CREATE INDEX idx_etymon_variant_class  ON etymon_variant(variant_class)")

    op.execute(
        """
        CREATE TABLE mining_run (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          source_id     TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
          provider      TEXT NOT NULL,
          model         TEXT NOT NULL,
          mode          TEXT NOT NULL CHECK (mode IN ('mine', 'review')),
          started_at    TEXT,
          completed_at  TEXT,
          parsed_count  INTEGER NOT NULL DEFAULT 0,
          accepted      INTEGER NOT NULL DEFAULT 0,
          declined      INTEGER NOT NULL DEFAULT 0,
          rejected      INTEGER NOT NULL DEFAULT 0,
          by_failure    TEXT,
          notes         TEXT,
          UNIQUE (source_id, provider, model, mode, completed_at)
        )
        """
    )
    op.execute("CREATE INDEX idx_mining_run_source    ON mining_run(source_id)")
    op.execute("CREATE INDEX idx_mining_run_completed ON mining_run(completed_at)")


def downgrade() -> None:
    op.execute("DROP TABLE mining_run")
    op.execute("DROP TABLE etymon_variant")
    op.execute("DROP TABLE etymon_descent")
