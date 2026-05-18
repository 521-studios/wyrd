"""citations + text-match evidence (D4, D12, D21)

Revision ID: 0003_citations
Revises: 0002_etymons
Create Date: 2026-05-18

Two evidence layers attached to etymon + source:

* ``etymon_citation`` — formal extraction evidence (a scholar's
  identification of the morpheme in a toponym breakdown). Counted by
  ``etymon_consensus``; the load-bearing promotion-threshold input.
* ``etymon_text_match`` — looser body-text presence + Levenshtein
  fuzzy matches + LLM-disambiguated rows. NOT counted by
  ``etymon_consensus`` per D12 (would dilute the witness count).
"""

from __future__ import annotations

from alembic import op

revision = "0003_citations"
down_revision = "0002_etymons"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE etymon_citation (
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          etymon_id       INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          source_id       TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
          page            TEXT,
          short_quote     TEXT,
          context_snippet TEXT
        )
        """
    )
    # SQLite forbids expressions in inline UNIQUE constraints, so the
    # "treat NULL page as ''" uniqueness goes here as a partial index.
    op.execute(
        """
        CREATE UNIQUE INDEX idx_etymon_citation_unique
          ON etymon_citation(etymon_id, source_id, COALESCE(page, ''))
        """
    )

    op.execute(
        """
        CREATE TABLE etymon_text_match (
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          etymon_id   INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          source_id   TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
          matched_form  TEXT NOT NULL,
          match_count   INTEGER NOT NULL,
          edit_distance INTEGER NOT NULL DEFAULT 0,
          snippet       TEXT,
          method        TEXT NOT NULL DEFAULT 'reverse-search-v1',
          disambiguator_reason TEXT,
          attested_year INTEGER,
          UNIQUE (etymon_id, source_id, matched_form)
        )
        """
    )
    op.execute("CREATE INDEX idx_etymon_text_match_etymon ON etymon_text_match(etymon_id)")
    op.execute("CREATE INDEX idx_etymon_text_match_source ON etymon_text_match(source_id)")
    op.execute("CREATE INDEX idx_etymon_text_match_year ON etymon_text_match(attested_year)")


def downgrade() -> None:
    op.execute("DROP TABLE etymon_text_match")
    op.execute("DROP TABLE etymon_citation")
