"""etymons — central morpheme table + per-row gloss / tag children

Revision ID: 0002_etymons
Revises: 0001_sources
Create Date: 2026-05-18

The etymon table is the lexicon's spine. Three self-FKs wire its
behavioural axes:

* ``lemma_id`` — D8 inflection clustering (cotan / cotum → cot)
* ``merged_into_id`` — D22 non-destructive OCR clustering
* ``cognate_id`` — D27 cross-language cognate clusters

Plus the wyrd-ha9q Phase 2a / 2b columns (pronunciation_ipa,
pronunciation_dialect, original_script, transliteration,
english_shaped) and the wyrd-lr4 stratum column. Every column shape
in lexicon.sql is preserved verbatim — this migration captures the
schema as of wyrd-67fv.
"""

from __future__ import annotations

from alembic import op

revision = "0002_etymons"
down_revision = "0001_sources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE etymon (
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          canonical_form  TEXT NOT NULL,
          language        TEXT NOT NULL,
          modifier_type   TEXT,
          position_pref   TEXT CHECK (position_pref IN ('pre', 'post', 'inner', 'free')),
          notes           TEXT,
          lemma_id        INTEGER REFERENCES etymon(id) ON DELETE SET NULL,
          inflection      TEXT,
          lemma_method    TEXT,
          merged_into_id  INTEGER REFERENCES etymon(id) ON DELETE SET NULL,
          cognate_id      INTEGER REFERENCES etymon(id) ON DELETE SET NULL,
          cognate_method  TEXT,
          pronunciation_ipa     TEXT,
          pronunciation_dialect TEXT,
          original_script       TEXT,
          transliteration       TEXT,
          english_shaped        TEXT,
          stratum               TEXT,
          UNIQUE (canonical_form, language)
        )
        """
    )
    op.execute("CREATE INDEX idx_etymon_lemma       ON etymon(lemma_id)")
    op.execute("CREATE INDEX idx_etymon_merged_into ON etymon(merged_into_id)")
    op.execute("CREATE INDEX idx_etymon_cognate     ON etymon(cognate_id)")
    op.execute("CREATE INDEX idx_etymon_stratum     ON etymon(stratum)")
    op.execute("CREATE INDEX idx_etymon_lang        ON etymon(language)")

    op.execute(
        """
        CREATE TABLE etymon_gloss (
          etymon_id INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          gloss     TEXT NOT NULL,
          PRIMARY KEY (etymon_id, gloss)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE etymon_tag (
          etymon_id INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          tag       TEXT NOT NULL,
          PRIMARY KEY (etymon_id, tag)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE etymon_tag")
    op.execute("DROP TABLE etymon_gloss")
    op.execute("DROP TABLE etymon")
