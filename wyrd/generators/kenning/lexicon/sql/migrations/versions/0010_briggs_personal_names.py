"""personal_name + personal_name_toponym_attestation — Briggs index ingest

Revision ID: 0010_briggs_personal_names
Revises: 0009_phonological_vector
Create Date: 2026-05-20

wyrd-uzoh. Captures Keith Briggs's *Index to Personal Names in English
Place-Names* (EPNS Supplementary Series 2, 3rd revised pdf edn 2024)
as two normalized tables:

* ``personal_name`` — one row per unique PN headform (with diacritics
  preserved verbatim plus a diacritic-folded ``normalized_form`` for
  ASCII lookups). Carries citation-count metadata (PASE count, DLV
  presence, ASCh charter refs) and Briggs-supplied language hints.

* ``personal_name_toponym_attestation`` — one row per (PN, toponym,
  county) occurrence. ``attested_variant`` captures the PN's surface
  form as it appears IN the toponym when Briggs's "X in Y" syntax
  exposes it (so a single table supports both PN→toponym and
  toponym→PN consumer queries via different indexes).

Both tables key on a ``source_doc`` value so future ingest paths
(e.g. a different scholar's PN-index) can coexist without collision.
"""

from __future__ import annotations

from alembic import op

revision = "0010_briggs_personal_names"
down_revision = "0009_phonological_vector"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE personal_name (
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          headform        TEXT NOT NULL,
          normalized_form TEXT NOT NULL,
          language_hints  TEXT,
          is_feminine     INTEGER NOT NULL DEFAULT 0,
          pase_count      INTEGER,
          has_dlv         INTEGER NOT NULL DEFAULT 0,
          ascharter_refs  TEXT,
          source_doc      TEXT NOT NULL,
          raw_entry       TEXT
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX idx_personal_name_headform_source
          ON personal_name(headform, source_doc)
        """
    )
    op.execute(
        """
        CREATE INDEX idx_personal_name_normalized
          ON personal_name(normalized_form)
        """
    )

    op.execute(
        """
        CREATE TABLE personal_name_toponym_attestation (
          id                 INTEGER PRIMARY KEY AUTOINCREMENT,
          personal_name_id   INTEGER NOT NULL
                             REFERENCES personal_name(id) ON DELETE CASCADE,
          toponym_form       TEXT NOT NULL,
          attested_variant   TEXT,
          county_code        TEXT NOT NULL,
          county_canonical   TEXT NOT NULL,
          date_qualifier     TEXT,
          is_uncertain       INTEGER NOT NULL DEFAULT 0,
          is_serious_doubt   INTEGER NOT NULL DEFAULT 0,
          source_doc         TEXT NOT NULL,
          raw_text           TEXT
        )
        """
    )
    # Idempotency: re-ingest of the same Briggs slice produces zero
    # net inserts. The COALESCE-padded uniqueness key matches the
    # natural identity (PN × toponym × county × variant × date).
    op.execute(
        """
        CREATE UNIQUE INDEX idx_pn_toponym_dedup
          ON personal_name_toponym_attestation(
            personal_name_id,
            toponym_form,
            county_code,
            COALESCE(attested_variant, ''),
            COALESCE(date_qualifier, '')
          )
        """
    )
    # Forward query: "given a toponym in a county, what PN attestations
    # exist?" — covered by (toponym_form, county_canonical).
    op.execute(
        """
        CREATE INDEX idx_pn_toponym_form
          ON personal_name_toponym_attestation(toponym_form, county_canonical)
        """
    )
    # Reverse query: "given a PN, list all its toponym occurrences."
    op.execute(
        """
        CREATE INDEX idx_pn_toponym_personal_name
          ON personal_name_toponym_attestation(personal_name_id)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS personal_name_toponym_attestation")
    op.execute("DROP TABLE IF EXISTS personal_name")
