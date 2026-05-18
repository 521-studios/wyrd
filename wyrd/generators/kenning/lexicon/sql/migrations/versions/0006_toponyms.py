"""toponym + attestation + etymology + decomposition + period_form

Revision ID: 0006_toponyms
Revises: 0005_reflex
Create Date: 2026-05-18

The toponym layer carries place-name evidence from raw modern names
through historical attestations and scholarly etymology breakdowns to
matcher-derived decompositions and the wyrd-unuo period-form
projections.

Six tables in dependency order:

* ``toponym`` — modern place names. ``(modern_name, country, region)``
  unique with NULL-tolerant COALESCE.
* ``toponym_attestation`` — wyrd-skm Phase 3.0a historical spellings.
* ``toponym_etymology`` — one scholar's breakdown per row;
  disagreement is information.
* ``toponym_etymology_element`` — ordered morpheme list per breakdown.
* ``toponym_decomposition`` — wyrd-08m matcher enumeration of every
  plausible breakdown, with at most one canonical pick.
* ``etymon_period_form`` — wyrd-unuo Phase 3.3 per-etymon
  period-keyed surface forms, projected from attestations.
"""

from __future__ import annotations

from alembic import op

revision = "0006_toponyms"
down_revision = "0005_reflex"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE toponym (
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          modern_name TEXT NOT NULL,
          country     TEXT,
          region      TEXT
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX idx_toponym_unique
          ON toponym(modern_name, COALESCE(country, ''), COALESCE(region, ''))
        """
    )
    op.execute("CREATE INDEX idx_toponym_name ON toponym(modern_name)")

    op.execute(
        """
        CREATE TABLE toponym_attestation (
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          toponym_id  INTEGER NOT NULL REFERENCES toponym(id) ON DELETE CASCADE,
          form        TEXT NOT NULL,
          date_year   INTEGER,
          source_doc  TEXT
        )
        """
    )
    op.execute("CREATE INDEX idx_attestation_topo ON toponym_attestation(toponym_id)")
    op.execute(
        """
        CREATE UNIQUE INDEX idx_attestation_unique
          ON toponym_attestation(toponym_id, form, date_year, source_doc)
        """
    )

    op.execute(
        """
        CREATE TABLE toponym_etymology (
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          toponym_id      INTEGER NOT NULL REFERENCES toponym(id) ON DELETE CASCADE,
          source_id       TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
          page            TEXT,
          historical_form TEXT,
          confidence      TEXT CHECK (confidence IN ('high', 'medium', 'low')),
          notes           TEXT,
          attested_year   INTEGER,
          is_canonical    INTEGER NOT NULL DEFAULT 0 CHECK (is_canonical IN (0, 1)),
          consensus_size  INTEGER NOT NULL DEFAULT 1,
          cluster_key     TEXT
        )
        """
    )
    op.execute("CREATE INDEX idx_etymology_toponym ON toponym_etymology(toponym_id)")
    op.execute("CREATE INDEX idx_etymology_source  ON toponym_etymology(source_id)")
    op.execute("CREATE INDEX idx_toponym_etymology_year ON toponym_etymology(attested_year)")
    op.execute(
        """
        CREATE INDEX idx_toponym_etymology_canonical
          ON toponym_etymology(toponym_id) WHERE is_canonical = 1
        """
    )

    op.execute(
        """
        CREATE TABLE toponym_etymology_element (
          toponym_etymology_id INTEGER NOT NULL REFERENCES toponym_etymology(id) ON DELETE CASCADE,
          ordinal              INTEGER NOT NULL,
          etymon_id            INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          inflection           TEXT,
          surface_in_modern    TEXT,
          PRIMARY KEY (toponym_etymology_id, ordinal)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE toponym_decomposition (
          id                       INTEGER PRIMARY KEY AUTOINCREMENT,
          toponym_id               INTEGER NOT NULL REFERENCES toponym(id) ON DELETE CASCADE,
          decomposition_signature  TEXT NOT NULL,
          morpheme_ids             TEXT NOT NULL,
          unaccounted_fragments    TEXT NOT NULL,
          unaccounted_count        INTEGER NOT NULL DEFAULT 0,
          morpheme_count           INTEGER NOT NULL DEFAULT 0,
          is_canonical             INTEGER NOT NULL DEFAULT 0 CHECK (is_canonical IN (0, 1)),
          canonical_source         TEXT CHECK (
            canonical_source IN (
              'scholar', 'scholar-disagreement', 'unique-zero-unaccounted', 'tiebreaker'
            ) OR canonical_source IS NULL
          ),
          created_at               TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX idx_toponym_decomposition_unique
          ON toponym_decomposition(toponym_id, decomposition_signature)
        """
    )
    op.execute(
        """
        CREATE INDEX idx_toponym_decomposition_topo
          ON toponym_decomposition(toponym_id)
        """
    )
    op.execute(
        """
        CREATE INDEX idx_toponym_decomposition_canonical
          ON toponym_decomposition(toponym_id) WHERE is_canonical = 1
        """
    )

    op.execute(
        """
        CREATE TABLE etymon_period_form (
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          etymon_id       INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          form            TEXT NOT NULL,
          date_year       INTEGER NOT NULL,
          source_doc      TEXT,
          attestation_id  INTEGER REFERENCES toponym_attestation(id) ON DELETE SET NULL
        )
        """
    )
    op.execute("CREATE INDEX idx_period_form_etymon ON etymon_period_form(etymon_id)")
    op.execute("CREATE INDEX idx_period_form_year ON etymon_period_form(date_year)")
    op.execute(
        """
        CREATE UNIQUE INDEX idx_period_form_unique
          ON etymon_period_form(etymon_id, form, date_year, source_doc)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE etymon_period_form")
    op.execute("DROP TABLE toponym_decomposition")
    op.execute("DROP TABLE toponym_etymology_element")
    op.execute("DROP TABLE toponym_etymology")
    op.execute("DROP TABLE toponym_attestation")
    op.execute("DROP TABLE toponym")
