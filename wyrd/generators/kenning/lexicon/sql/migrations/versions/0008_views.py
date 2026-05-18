"""views — consensus, canonical, breakdown-signature

Revision ID: 0008_views
Revises: 0007_synsets_priors
Create Date: 2026-05-18

All views live in a single migration so the consensus / canonical
rollup logic — the load-bearing piece of how mining evidence collapses
across the merged_into_id and lemma_id chains (D22) — stays
co-located. SA Core has no first-class view object; views land via
``op.execute("CREATE VIEW ...")``.

Seven views:

* ``toponym_etymology_canonical`` — convenience over the
  ``is_canonical=1`` slice. wyrd-08qv.
* ``etymon_canonical`` — etymons that aren't OCR-merge losers.
* ``etymon_consensus`` — per-canonical witness count rolled up via
  merged_into_id → lemma_id chain. D4 promotion-threshold input.
* ``etymon_gloss_canonical`` / ``etymon_tag_canonical`` /
  ``etymon_text_match_canonical`` — wyrd-7lo per-canonical
  child-data rollups for the gloss / tag / text-match tables.
* ``toponym_breakdown_signature`` — per-toponym disagreement
  detector. >1 distinct signature → scholars disagree on the
  breakdown.
"""

from __future__ import annotations

from alembic import op

revision = "0008_views"
down_revision = "0007_synsets_priors"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE VIEW toponym_etymology_canonical AS
          SELECT * FROM toponym_etymology WHERE is_canonical = 1
        """
    )

    op.execute(
        """
        CREATE VIEW etymon_canonical AS
          SELECT * FROM etymon WHERE merged_into_id IS NULL
        """
    )

    # GROUP BY includes every non-aggregated SELECT column so the
    # result is well-defined regardless of SQLite's "bare column"
    # tolerance. ``canonical_form`` and ``language`` are functionally
    # dependent on ``lemma_id`` via the rollup chain, so the extra
    # GROUP BY columns don't change the row set — only the
    # determinism guarantee.
    op.execute(
        """
        CREATE VIEW etymon_consensus AS
          SELECT lemma_id,
                 canonical_form,
                 language,
                 COUNT(DISTINCT source_id) AS witnesses
          FROM (
            SELECT
              COALESCE(le.id, target.id, e.id) AS lemma_id,
              COALESCE(le.canonical_form, target.canonical_form, e.canonical_form)
                AS canonical_form,
              e.language,
              c.source_id
            FROM etymon e
            LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
            LEFT JOIN etymon le ON le.id = target.lemma_id
            LEFT JOIN etymon_citation c ON c.etymon_id = e.id
          )
          GROUP BY lemma_id, canonical_form, language
        """
    )

    op.execute(
        """
        CREATE VIEW etymon_gloss_canonical AS
          SELECT DISTINCT
                 COALESCE(le.id, target.id, e.id) AS canonical_etymon_id,
                 g.gloss
          FROM etymon e
          JOIN etymon_gloss g ON g.etymon_id = e.id
          LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
          LEFT JOIN etymon le ON le.id = target.lemma_id
        """
    )

    op.execute(
        """
        CREATE VIEW etymon_tag_canonical AS
          SELECT DISTINCT
                 COALESCE(le.id, target.id, e.id) AS canonical_etymon_id,
                 t.tag
          FROM etymon e
          JOIN etymon_tag t ON t.etymon_id = e.id
          LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
          LEFT JOIN etymon le ON le.id = target.lemma_id
        """
    )

    op.execute(
        """
        CREATE VIEW etymon_text_match_canonical AS
          SELECT COALESCE(le.id, target.id, e.id) AS canonical_etymon_id,
                 m.source_id,
                 m.matched_form,
                 SUM(m.match_count) AS total_match_count,
                 MIN(m.edit_distance) AS edit_distance,
                 MIN(m.attested_year) AS attested_year
          FROM etymon e
          JOIN etymon_text_match m ON m.etymon_id = e.id
          LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
          LEFT JOIN etymon le ON le.id = target.lemma_id
          GROUP BY canonical_etymon_id, m.source_id, m.matched_form
        """
    )

    # GROUP_CONCAT(..., ',' ORDER BY ...) is a SQLite 3.44+ feature
    # (released late 2023). The Lambda runtime ships SQLite 3.40,
    # and AL2023-based dev machines may also be older. To keep the
    # ordinal-stable signature without requiring 3.44, pre-sort the
    # rows in a subquery and then aggregate without ORDER BY —
    # GROUP_CONCAT preserves the input row order.
    op.execute(
        """
        CREATE VIEW toponym_breakdown_signature AS
          SELECT toponym_id,
                 toponym_etymology_id,
                 source_id,
                 GROUP_CONCAT(etymon_id, ',') AS signature
          FROM (
            SELECT te.toponym_id,
                   te.id AS toponym_etymology_id,
                   te.source_id,
                   tee.etymon_id,
                   tee.ordinal
            FROM toponym_etymology te
            LEFT JOIN toponym_etymology_element tee
              ON tee.toponym_etymology_id = te.id
            ORDER BY te.id, tee.ordinal
          )
          GROUP BY toponym_etymology_id, toponym_id, source_id
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW toponym_breakdown_signature")
    op.execute("DROP VIEW etymon_text_match_canonical")
    op.execute("DROP VIEW etymon_tag_canonical")
    op.execute("DROP VIEW etymon_gloss_canonical")
    op.execute("DROP VIEW etymon_consensus")
    op.execute("DROP VIEW etymon_canonical")
    op.execute("DROP VIEW toponym_etymology_canonical")
