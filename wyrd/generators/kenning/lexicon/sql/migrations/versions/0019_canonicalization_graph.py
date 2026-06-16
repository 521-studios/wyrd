"""canonicalization graph — synthetic canonical nodes + binding columns (wyrd-u6fn.2, D50).

The L3 half of the Class-B canonicalization layer (D49/D50). Synthetic identity
hubs (``canonical_morpheme`` / ``canonical_place`` / ``canonical_sense``, D50.1)
with a ``merged_into`` self-FK for ``merge-canonical`` reconciliation;
``canonical_label`` for per-stratum canonical surfaces (D50.6); and a
``canonical_*_id`` binding column on each observation table — the generalization
of ``merged_into_id`` / ``cognate_id`` (D22) from per-table redirect columns to a
uniform synthetic-node model.

L3 derived artifact: the nodes / binds / labels are PROJECTED from the
``data/mining/canonicalization/`` L2 assertion streams by the projection pass
(wyrd-u6fn.3, pending), so a rebuild creates the schema here but leaves these
tables / columns empty until the projection runs — same posture as
``genitive_split_prior`` / ``empirical_priors``. Place binds at
``toponym``-row granularity by default, with ``toponym_etymology``-row overrides
(D50.6), hence the column on both.
"""

from __future__ import annotations

from alembic import op

revision = "0019_canonicalization_graph"
down_revision = "0018_genitive_split_prior"
branch_labels = None
depends_on = None


_UPGRADE = [
    # synthetic identity hubs (D50.1); ids are L2-declared (mint-canonical).
    """
    CREATE TABLE canonical_morpheme (
      id          TEXT PRIMARY KEY,
      merged_into TEXT REFERENCES canonical_morpheme(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE canonical_place (
      id          TEXT PRIMARY KEY,
      merged_into TEXT REFERENCES canonical_place(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE canonical_sense (
      id          TEXT PRIMARY KEY,
      merged_into TEXT REFERENCES canonical_sense(id) ON DELETE SET NULL
    )
    """,
    "CREATE INDEX idx_canonical_morpheme_merged ON canonical_morpheme(merged_into)",
    "CREATE INDEX idx_canonical_place_merged ON canonical_place(merged_into)",
    "CREATE INDEX idx_canonical_sense_merged ON canonical_sense(merged_into)",
    # per-stratum canonical labels (D50.6). stratum '' = era-invariant (NOT
    # NULL keeps the composite PK clean, matching the COALESCE-to-'' idiom used
    # for nullable unique-index columns elsewhere). canonical_id is a
    # polymorphic ref (discriminated by canonical_type), so no single FK.
    """
    CREATE TABLE canonical_label (
      canonical_type   TEXT NOT NULL,
      canonical_id     TEXT NOT NULL,
      stratum          TEXT NOT NULL DEFAULT '',
      value            TEXT NOT NULL,
      source_assertion TEXT,
      PRIMARY KEY (canonical_type, canonical_id, stratum)
    )
    """,
    # binding columns: which canonical node each observation row binds to
    # (generalizes merged_into_id / cognate_id). Nullable, default NULL, so the
    # ADD COLUMN is FK-safe under PRAGMA foreign_keys=ON. Populated by u6fn.3.
    "ALTER TABLE etymon ADD COLUMN canonical_morpheme_id TEXT "
    "REFERENCES canonical_morpheme(id) ON DELETE SET NULL",
    "ALTER TABLE etymon_gloss ADD COLUMN canonical_sense_id TEXT "
    "REFERENCES canonical_sense(id) ON DELETE SET NULL",
    "ALTER TABLE toponym ADD COLUMN canonical_place_id TEXT "
    "REFERENCES canonical_place(id) ON DELETE SET NULL",
    "ALTER TABLE toponym_etymology ADD COLUMN canonical_place_id TEXT "
    "REFERENCES canonical_place(id) ON DELETE SET NULL",
    "CREATE INDEX idx_etymon_canonical_morpheme ON etymon(canonical_morpheme_id)",
    "CREATE INDEX idx_etymon_gloss_canonical_sense ON etymon_gloss(canonical_sense_id)",
    "CREATE INDEX idx_toponym_canonical_place ON toponym(canonical_place_id)",
    "CREATE INDEX idx_toponym_etymology_canonical_place ON toponym_etymology(canonical_place_id)",
]

# Reverse order: drop the binding-column indexes, then the columns (SQLite
# DROP COLUMN refuses an indexed column), then the canonical_* tables (their
# own indexes drop with them).
_DOWNGRADE = [
    "DROP INDEX idx_toponym_etymology_canonical_place",
    "DROP INDEX idx_toponym_canonical_place",
    "DROP INDEX idx_etymon_gloss_canonical_sense",
    "DROP INDEX idx_etymon_canonical_morpheme",
    "ALTER TABLE toponym_etymology DROP COLUMN canonical_place_id",
    "ALTER TABLE toponym DROP COLUMN canonical_place_id",
    "ALTER TABLE etymon_gloss DROP COLUMN canonical_sense_id",
    "ALTER TABLE etymon DROP COLUMN canonical_morpheme_id",
    "DROP TABLE canonical_label",
    "DROP TABLE canonical_sense",
    "DROP TABLE canonical_place",
    "DROP TABLE canonical_morpheme",
]


def upgrade() -> None:
    for stmt in _UPGRADE:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _DOWNGRADE:
        op.execute(stmt)
