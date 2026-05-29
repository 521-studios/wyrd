"""drop personal_name + personal_name_toponym_attestation

Revision ID: 0014_drop_personal_name_tables
Revises: 0013_etymology_element_confidence
Create Date: 2026-05-29

wyrd-2b50. The bespoke Briggs personal-name tables (wyrd-uzoh /
0010_briggs_personal_names) are retired: Briggs names are now ingested
into the standard ``etymon`` schema — one ``etymon`` per name tagged
``male name`` / ``female name``, with multi-source citations (Briggs
primary plus PASE / DLV / Anglo-Saxon-charter ``cited_source``
attestations). That makes personal names first-class etymons that flow
through the normal witness gate, proportions, and vectors, instead of
sitting in a parallel pipeline no consumer read.

``downgrade`` recreates the tables (empty) so the chain is reversible;
the data is not restored — re-run ``ingest-briggs-personal-names`` to
repopulate the etymon-shaped rows.
"""

from __future__ import annotations

from alembic import op

revision = "0014_drop_personal_name_tables"
down_revision = "0013_etymology_element_confidence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # personal_name_toponym_attestation first — it FKs personal_name.
    op.execute("DROP TABLE IF EXISTS personal_name_toponym_attestation")
    op.execute("DROP TABLE IF EXISTS personal_name")


def downgrade() -> None:
    # Mirror of 0010_briggs_personal_names upgrade (empty tables).
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
    op.execute(
        """
        CREATE INDEX idx_pn_toponym_form
          ON personal_name_toponym_attestation(toponym_form, county_canonical)
        """
    )
    op.execute(
        """
        CREATE INDEX idx_pn_toponym_personal_name
          ON personal_name_toponym_attestation(personal_name_id)
        """
    )
