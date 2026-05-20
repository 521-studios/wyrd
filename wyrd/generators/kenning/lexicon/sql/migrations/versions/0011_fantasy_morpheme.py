"""fantasy_morpheme — wyrd-ami / wyrd-rrse

Revision ID: 0011_fantasy_morpheme
Revises: 0010_briggs_personal_names
Create Date: 2026-05-20

wyrd-rrse: belated alembic-ification of the ``fantasy_morpheme`` table.
Previously created only by the legacy ``schema.migrate_schema`` helper
(``_create_fantasy_morpheme_table``), which ``init_schema`` doesn't
call — so fresh DBs spawned via ``init_schema`` (rebuild-from-jsonl,
diff-rebuild, tests) were missing this table entirely. ``build_from_jsonl``
crashed with ``no such table: fantasy_morpheme`` when replaying any
``data/mining/_fantasy_morphemes.jsonl`` events.

Carrying the same shape the legacy helper produced:

- ``input_name`` is COLLATE NOCASE so ``Harpy`` and ``harpy`` dedup
  under ``UNIQUE(input_name, approach_version)``. Saves repeat LLM
  research calls when callers happen to vary casing.
- ``etymon_id`` is nullable + ON DELETE SET NULL — barred morphemes
  (``usable = 0``) record ``bar_reason`` without a real etymon link;
  cascading delete of an etymon should not orphan-cascade the row.
- ``approach_version`` is the wyrd-ami APPROACH_VERSION constant so
  later pipelines can re-process older-version rows.
- ``unapproved_language`` / ``unapproved_form`` capture
  ``bar_reason = 'outside_language_family'`` cases for prioritizing
  language additions.

The legacy helper stays in place for legacy DBs created before this
migration (it's idempotent — no-op when the table already exists).
"""

from __future__ import annotations

from alembic import op

revision = "0011_fantasy_morpheme"
down_revision = "0010_briggs_personal_names"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE fantasy_morpheme (
          id                  INTEGER PRIMARY KEY AUTOINCREMENT,
          input_name          TEXT NOT NULL COLLATE NOCASE,
          input_description   TEXT,
          usable              INTEGER NOT NULL CHECK (usable IN (0, 1)),
          etymon_id           INTEGER REFERENCES etymon(id) ON DELETE SET NULL,
          bar_reason          TEXT,
          resolution_method   TEXT NOT NULL,
          approach_version    TEXT NOT NULL,
          confidence          TEXT,
          citation            TEXT,
          reasoning           TEXT,
          unapproved_language TEXT,
          unapproved_form     TEXT,
          processed_at        TEXT NOT NULL,
          UNIQUE (input_name, approach_version)
        )
        """
    )
    op.execute("CREATE INDEX idx_fantasy_morpheme_approach ON fantasy_morpheme(approach_version)")
    op.execute("CREATE INDEX idx_fantasy_morpheme_etymon ON fantasy_morpheme(etymon_id)")
    op.execute("CREATE INDEX idx_fantasy_morpheme_usable ON fantasy_morpheme(usable)")
    op.execute(
        "CREATE INDEX idx_fantasy_morpheme_unapproved ON fantasy_morpheme(unapproved_language)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS fantasy_morpheme")
