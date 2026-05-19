"""Schema bootstrap + legacy migrations for the lexicon DB.

Two entry points operate on a ``LexiconDB``:

* ``init_schema(db_path)`` — fresh-install path. Wipes any existing
  file, opens a transient connection long enough to set the
  file-persistent WAL pragma, then hands off to alembic's
  ``upgrade_head`` (wyrd-67fv) to lay down every table / view /
  index. Subsequent ``LexiconDB(path)`` opens inherit WAL and run
  with synchronous=NORMAL via the SA connect-event listener on the
  engine.
* ``migrate_schema(db)`` — legacy uplift. Bring an existing
  lexicon.db forward column-by-column with the 18 idempotent
  ``_create_*_table`` / ``_migrate_*`` helpers in this module. Per
  the wyrd-67fv direction, new schema changes go through new
  alembic migrations, not new helpers here — the legacy path stays
  for operators with DBs that pre-date alembic.

``record_mining_run`` (D23 audit writer) lives here too because it
shares the schema-management mindset: every persistent mining
operation must land in the ``mining_run`` table; this is the writer.

The bulk of this file is the 18 helpers ``migrate_schema``
orchestrates. Each helper:
* Inspects ``sqlite_master`` / ``PRAGMA table_info`` to decide
  whether its target column / index / table is missing.
* Issues the appropriate ALTER / CREATE.
* Flips its key in the ``applied`` dict so callers can report what
  changed.

They run in a fixed order on every legacy DB and the ordering
matters (renames before adds; column-adds before index creates;
table-creates before downstream constraints). See ``migrate_schema``
for the canonical sequence.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from wyrd.generators.kenning.lexicon.db import LexiconDB, _apply_persistent_pragmas


def init_schema(db_path: Path | str) -> None:
    """Create a fresh lexicon DB at db_path with the schema applied.

    Wipes any existing file at the path. Sets file-persistent
    journal_mode=WAL (so subsequent opens inherit the mode) and then
    runs the layered alembic migrations via ``upgrade_head`` to create
    the schema and stamp the ``alembic_version`` table at the current
    revision (wyrd-67fv: 0008_views).

    The WAL pragma is applied on a transient connection BEFORE alembic
    runs so the mode is recorded under the same journal_mode every
    subsequent open sees — alembic's transaction would otherwise see
    the default ``delete`` mode for the first transaction.
    """
    from wyrd.generators.kenning.lexicon.sql import upgrade_head

    path = Path(db_path)
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        _apply_persistent_pragmas(conn)
        conn.commit()
    finally:
        conn.close()
    upgrade_head(path)


# --- legacy column / table / view migrations ------------------------------
#
# The functions below are the per-column / per-table / per-view helpers
# that ``migrate_schema`` orchestrates in a fixed sequence. They predate
# the wyrd-67fv alembic introduction; the alembic baseline already
# carries every shape they produce on a fresh DB, so the helpers only
# ever do anything on legacy DBs that pre-date a given column or table.
# Each helper is idempotent — running on an already-migrated DB is a
# no-op. New schema changes go through new alembic migrations, NOT new
# helpers here.


def _add_etymon_columns(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Add the etymon-table column migrations: lemma_id / inflection /
    merged_into_id / lemma_method (D8 + D22), cognate_id / cognate_method
    (D27), and the wyrd-ha9q Phase 2a pronunciation + multi-script set
    (pronunciation_ipa, pronunciation_dialect, original_script,
    transliteration, english_shaped). Each column is independent and
    idempotent — re-running migrate_schema on an already-migrated DB is
    a no-op for all of them.
    """
    cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
    if "lemma_id" not in cols:
        # ON DELETE SET NULL matches lexicon.sql's base schema and keeps
        # the migration / fresh-install paths in sync. Without it, deleting
        # a lemma row would FK-fail when foreign_keys=ON is enabled.
        db.conn.execute(
            "ALTER TABLE etymon ADD COLUMN lemma_id INTEGER "
            "REFERENCES etymon(id) ON DELETE SET NULL"
        )
        applied["etymon.lemma_id"] = True
    if "inflection" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN inflection TEXT")
        applied["etymon.inflection"] = True
    # merged_into_id: non-destructive OCR clustering. When two etymons are
    # merged by cluster_ocr_variants, the loser gets merged_into_id pointing
    # at the canonical winner instead of being DELETE'd. Per D22, this lets
    # callers blow away clustering (UPDATE etymon SET merged_into_id = NULL)
    # and re-run with new heuristics without re-mining.
    if "merged_into_id" not in cols:
        db.conn.execute(
            "ALTER TABLE etymon ADD COLUMN merged_into_id INTEGER "
            "REFERENCES etymon(id) ON DELETE SET NULL"
        )
        applied["etymon.merged_into_id"] = True
    # lemma_method: which version of the lemma-linker produced the
    # lemma_id/inflection attribution on this row. Lets future link-lemmas
    # versions identify their predecessors' work for selective rebuild.
    if "lemma_method" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN lemma_method TEXT")
        applied["etymon.lemma_method"] = True
    # cognate_id (D27 / wyrd-0ug): cognate-cluster rollup. Points at the
    # most-ancestral known etymon in this cognate set; populated by the
    # cluster-cognates pass (wyrd-81n) walking etymon_descent inheritance
    # edges. Cross-language by design — distinct from merged_into_id
    # (within-language OCR cluster) and lemma_id (within-language
    # inflection family).
    if "cognate_id" not in cols:
        db.conn.execute(
            "ALTER TABLE etymon ADD COLUMN cognate_id INTEGER "
            "REFERENCES etymon(id) ON DELETE SET NULL"
        )
        applied["etymon.cognate_id"] = True
    # cognate_method: which version of cluster-cognates produced this
    # cognate_id assignment. Mirrors lemma_method.
    if "cognate_method" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN cognate_method TEXT")
        applied["etymon.cognate_method"] = True
    # wyrd-ha9q Phase 2a: pronunciation + multi-script renderings.
    # All TEXT NULL. Populated by the wiktextract ingester from
    # `entry.sounds[*]` (IPA + dialect) and `entry.head_templates[0].args`
    # (vocalized native form + academic transliteration). english_shaped
    # stays NULL until Phase 2b (rule-derived from IPA + transliteration).
    # Existing rows that pre-date this migration keep NULLs until the
    # next ingest pass picks them up via the ingester's upsert logic.
    if "pronunciation_ipa" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN pronunciation_ipa TEXT")
        applied["etymon.pronunciation_ipa"] = True
    if "pronunciation_dialect" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN pronunciation_dialect TEXT")
        applied["etymon.pronunciation_dialect"] = True
    if "original_script" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN original_script TEXT")
        applied["etymon.original_script"] = True
    if "transliteration" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN transliteration TEXT")
        applied["etymon.transliteration"] = True
    if "english_shaped" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN english_shaped TEXT")
        applied["etymon.english_shaped"] = True
    # wyrd-lr4: within-language stratum tag. NULL on rows that haven't
    # been classified yet (stays NULL until ``lexicon classify-stratum``
    # runs for that language family). Strata are language-specific — no
    # global CHECK constraint.
    if "stratum" not in cols:
        db.conn.execute("ALTER TABLE etymon ADD COLUMN stratum TEXT")
        applied["etymon.stratum"] = True


def _create_etymon_indexes(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Idempotent index creation for the etymon table's self-FK columns."""
    existing_indexes = {
        row["name"] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    if "idx_etymon_lemma" not in existing_indexes:
        db.conn.execute("CREATE INDEX idx_etymon_lemma ON etymon(lemma_id)")
        applied["idx_etymon_lemma"] = True
    if "idx_etymon_merged_into" not in existing_indexes:
        db.conn.execute("CREATE INDEX idx_etymon_merged_into ON etymon(merged_into_id)")
        applied["idx_etymon_merged_into"] = True
    if "idx_etymon_cognate" not in existing_indexes:
        db.conn.execute("CREATE INDEX idx_etymon_cognate ON etymon(cognate_id)")
        applied["idx_etymon_cognate"] = True
    if "idx_etymon_stratum" not in existing_indexes:
        db.conn.execute("CREATE INDEX idx_etymon_stratum ON etymon(stratum)")
        applied["idx_etymon_stratum"] = True


def _recreate_view(db: LexiconDB, applied: dict[str, bool], name: str, body: str) -> None:
    """Drop-and-recreate a single view, marking the applied-flag dict.

    Views are always rebuilt (DROP IF EXISTS + CREATE) because the
    definitions change when new columns are introduced and SQLite has
    no CREATE OR REPLACE VIEW. The applied-flag key is ``f"{name}_view"``
    by convention; callers that need a different key can update
    ``applied`` directly after this call.
    """
    db.conn.execute(f"DROP VIEW IF EXISTS {name}")
    db.conn.execute(body)
    applied[f"{name}_view"] = True


def _rebuild_etymon_views(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Recreate the etymon view layer (canonical + consensus + the
    wyrd-7lo per-child rollups).

    Each view body lives in this function rather than a constant so
    the migration path stays self-contained. The full rationale for
    each view (why MIN(attested_year), why SUM(match_count), how the
    two-step chain handles merged_into_id → lemma_id) lives next to
    the matching CREATE VIEW blocks in data/lexicon.sql; this function
    just mirrors them for legacy-DB migration.
    """
    _recreate_view(
        db,
        applied,
        "etymon_canonical",
        """
        CREATE VIEW etymon_canonical AS
          SELECT * FROM etymon WHERE merged_into_id IS NULL
        """,
    )
    _recreate_view(
        db,
        applied,
        "etymon_consensus",
        """
        CREATE VIEW etymon_consensus AS
          SELECT lemma_id,
                 canonical_form,
                 language,
                 COUNT(DISTINCT source_id) AS witnesses
          FROM (
            SELECT
              -- Two-step rollup so merged_into_id → lemma_id chains
              -- collapse correctly: target = e's redirect (merged or lemma);
              -- le = target's lemma if target is itself inflected.
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
          GROUP BY lemma_id
        """,
    )
    _recreate_view(
        db,
        applied,
        "etymon_gloss_canonical",
        """
        CREATE VIEW etymon_gloss_canonical AS
          SELECT DISTINCT
                 COALESCE(le.id, target.id, e.id) AS canonical_etymon_id,
                 g.gloss
          FROM etymon e
          JOIN etymon_gloss g ON g.etymon_id = e.id
          LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
          LEFT JOIN etymon le ON le.id = target.lemma_id
        """,
    )
    _recreate_view(
        db,
        applied,
        "etymon_tag_canonical",
        """
        CREATE VIEW etymon_tag_canonical AS
          SELECT DISTINCT
                 COALESCE(le.id, target.id, e.id) AS canonical_etymon_id,
                 t.tag
          FROM etymon e
          JOIN etymon_tag t ON t.etymon_id = e.id
          LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
          LEFT JOIN etymon le ON le.id = target.lemma_id
        """,
    )
    _recreate_view(
        db,
        applied,
        "etymon_text_match_canonical",
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
        """,
    )


def _create_etymon_descent_table(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Create etymon_descent if missing (D27 / wyrd-0ug).

    Holds directed edges of the etymological descent graph populated by
    Wiktionary mining (wyrd-4rt) and any future scholar-cited chain
    assertions. The cognate cluster column etymon.cognate_id is derived
    from this graph by the cluster-cognates pass (wyrd-81n).

    Schema mirrors data/lexicon.sql so fresh-install and migration paths
    stay in lockstep. Idempotent — checks for the table before creating.
    """
    existing = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "etymon_descent" in existing:
        return
    db.conn.executescript(
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
        );
        CREATE INDEX idx_etymon_descent_parent ON etymon_descent(parent_id);
        CREATE INDEX idx_etymon_descent_child  ON etymon_descent(child_id);
        """
    )
    applied["etymon_descent_table"] = True


def _create_etymon_variant_table(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Create etymon_variant if missing (wyrd-fqil).

    Holds the per-form rows from wiktextract entries' `forms` arrays —
    inflected forms (Greek genitive ἁρπυίᾱς, OE dative-or-pl
    `worde`/`hwone`), alternative spellings (ME 'harpe' alt 'harp'),
    romanizations, and other lemma variants Wiktionary records.

    The wiktextract corpus has 2M+ such rows that the original ingester
    dropped. Surfacing them lets the wyrd-ami fantasy pre-filter resolve
    historical-spelling inputs (Chaucer's `harpie` → modern lemma
    `harpy` → ancient-greek ἅρπυια) and generally widens the lookup
    surface for any "is this attested form in our corpus?" question.

    `form` is COLLATE NOCASE so case-insensitive lookups work without
    LOWER() in queries; `tags` carries the original wiktextract tag
    list as JSON for callers that want finer-grained filtering than
    the variant_class enum provides.
    """
    existing = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "etymon_variant" in existing:
        return
    db.conn.executescript(
        """
        CREATE TABLE etymon_variant (
          id            INTEGER PRIMARY KEY AUTOINCREMENT,
          etymon_id     INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
          form          TEXT NOT NULL COLLATE NOCASE,
          variant_class TEXT NOT NULL CHECK (variant_class IN (
                          'alternative', 'inflection', 'romanization',
                          'canonical', 'other'
                        )),
          tags          TEXT,  -- JSON array of original wiktextract tags
          source_id     TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
          UNIQUE (etymon_id, form, variant_class)
        );
        CREATE INDEX idx_etymon_variant_etymon ON etymon_variant(etymon_id);
        CREATE INDEX idx_etymon_variant_form   ON etymon_variant(form);
        CREATE INDEX idx_etymon_variant_class  ON etymon_variant(variant_class);
        """
    )
    applied["etymon_variant_table"] = True


def _migrate_text_match_table(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Create etymon_text_match if missing; otherwise add the method column
    if it doesn't exist. Tracks which heuristic produced each row
    ('reverse-search-v1', 'fuzzy-search-v1', 'llm-disambiguator-v1', ...)
    so a future rebuild can selectively clear-and-regenerate by method.
    """
    existing_tables = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "etymon_text_match" not in existing_tables:
        db.conn.executescript(
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
            );
            CREATE INDEX idx_etymon_text_match_etymon ON etymon_text_match(etymon_id);
            CREATE INDEX idx_etymon_text_match_source ON etymon_text_match(source_id);
            CREATE INDEX idx_etymon_text_match_year   ON etymon_text_match(attested_year);
            """
        )
        applied["etymon_text_match_table"] = True
        return
    text_match_cols = {
        row["name"] for row in db.conn.execute("PRAGMA table_info(etymon_text_match)")
    }
    if "method" not in text_match_cols:
        db.conn.execute(
            "ALTER TABLE etymon_text_match ADD COLUMN method TEXT "
            "NOT NULL DEFAULT 'reverse-search-v1'"
        )
        applied["etymon_text_match.method"] = True
    if "disambiguator_reason" not in text_match_cols:
        db.conn.execute("ALTER TABLE etymon_text_match ADD COLUMN disambiguator_reason TEXT")
        applied["etymon_text_match.disambiguator_reason"] = True
    if "attested_year" not in text_match_cols:
        db.conn.execute("ALTER TABLE etymon_text_match ADD COLUMN attested_year INTEGER")
        # The index is only valuable once the column exists; create alongside.
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_etymon_text_match_year "
            "ON etymon_text_match(attested_year)"
        )
        applied["etymon_text_match.attested_year"] = True


def _migrate_citation_context_snippet(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Add etymon_citation.context_snippet to existing DBs (wyrd-9kh.3).

    Idempotent: PRAGMA-checks the column before adding. Fresh installs
    pick the column up from data/lexicon.sql so this is a migration-only
    path.
    """
    cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon_citation)")}
    if "context_snippet" not in cols:
        db.conn.execute("ALTER TABLE etymon_citation ADD COLUMN context_snippet TEXT")
        applied["etymon_citation.context_snippet"] = True


def _migrate_toponym_etymology_attested_year(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Add toponym_etymology.attested_year + index to existing DBs
    (wyrd-bag — D5-1 expansion).

    Idempotent: PRAGMA-checks the column before adding. Fresh installs
    pick the column up from data/lexicon.sql so this is a migration-only
    path.
    """
    cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(toponym_etymology)")}
    if "attested_year" not in cols:
        db.conn.execute("ALTER TABLE toponym_etymology ADD COLUMN attested_year INTEGER")
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_toponym_etymology_year "
            "ON toponym_etymology(attested_year)"
        )
        applied["toponym_etymology.attested_year"] = True


def _migrate_toponym_etymology_canonical(db: LexiconDB, applied: dict[str, bool]) -> None:
    """wyrd-08qv: add is_canonical / consensus_size / cluster_key columns
    + partial index + canonical view to toponym_etymology on legacy DBs.

    Idempotent: PRAGMA-checks each column before adding. Fresh installs
    pick everything up from data/lexicon.sql; this is migration-only.

    Why three columns rather than a sibling table: keeps the canonical
    decision colocated with the hypothesis it applies to (no JOIN to
    answer 'is this row canonical?'), and the partial index on
    is_canonical=1 keeps canonical-only queries fast. The cluster_key
    lives on the row so an operator can grep the canonicalize-run
    output and SEE what cluster a row belongs to without re-running
    the clustering logic.
    """
    cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(toponym_etymology)")}
    if "is_canonical" not in cols:
        db.conn.execute(
            "ALTER TABLE toponym_etymology ADD COLUMN is_canonical INTEGER NOT NULL DEFAULT 0"
        )
        applied["toponym_etymology.is_canonical"] = True
    if "consensus_size" not in cols:
        db.conn.execute(
            "ALTER TABLE toponym_etymology ADD COLUMN consensus_size INTEGER NOT NULL DEFAULT 1"
        )
        applied["toponym_etymology.consensus_size"] = True
    if "cluster_key" not in cols:
        db.conn.execute("ALTER TABLE toponym_etymology ADD COLUMN cluster_key TEXT")
        applied["toponym_etymology.cluster_key"] = True
    db.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_toponym_etymology_canonical "
        "ON toponym_etymology(toponym_id) WHERE is_canonical = 1"
    )
    # Recreate view idempotently (DROP + CREATE — same pattern the
    # _rebuild_etymon_views helper uses).
    db.conn.execute("DROP VIEW IF EXISTS toponym_etymology_canonical")
    db.conn.execute(
        "CREATE VIEW toponym_etymology_canonical AS "
        "SELECT * FROM toponym_etymology WHERE is_canonical = 1"
    )
    applied["toponym_etymology_canonical_view"] = True
    applied["idx_toponym_etymology_canonical"] = True


def _create_etymon_period_form_table(db: LexiconDB, applied: dict[str, bool]) -> None:
    """wyrd-unuo Phase 3.3: ensure ``etymon_period_form`` table +
    indexes exist on legacy DBs.

    Tier 3 fallback for the era-reflex picker (after cognate-cluster +
    descent-edge paths). Stores per-etymon period-keyed surface forms
    projected from toponym_attestation rows. The unique index makes
    re-projection idempotent.

    Idempotent: PRAGMA-checks for the table first; runs as no-op on
    fresh installs (table ships in data/lexicon.sql).
    """
    cur = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='etymon_period_form'"
    )
    if cur.fetchone() is not None:
        return
    db.conn.execute(
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
    db.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_period_form_etymon ON etymon_period_form(etymon_id)"
    )
    db.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_period_form_year ON etymon_period_form(date_year)"
    )
    db.conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_period_form_unique "
        "ON etymon_period_form(etymon_id, form, date_year, source_doc)"
    )
    applied["etymon_period_form_table"] = True


def _create_empirical_priors_tables(db: LexiconDB, applied: dict[str, bool]) -> None:
    """wyrd-ecjp.2: ensure the empirical-baseline priors tables exist
    on legacy DBs.

    Two tables (native + loan-relationship) per D36.4 / D36.7. Both
    are derived-artifact L3 tables; ``lexicon mine-empirical-baselines
    --apply`` is the only writer. Re-runnable + idempotent (replace-
    not-merge each run).

    Idempotent: PRAGMA-checks for each table first; runs as no-op on
    fresh installs (tables ship in data/lexicon.sql).
    """
    cur = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='empirical_priors_native'"
    )
    if cur.fetchone() is None:
        db.conn.execute(
            """
            CREATE TABLE empirical_priors_native (
              culture       TEXT NOT NULL,
              position      TEXT NOT NULL,
              tag           TEXT NOT NULL,
              era_midpoint  INTEGER NOT NULL,
              lemma_ref     TEXT NOT NULL,
              count         INTEGER NOT NULL CHECK (count > 0),
              PRIMARY KEY (culture, position, tag, era_midpoint, lemma_ref)
            )
            """
        )
        # Only the lemma_ref index is non-redundant — PK-prefix lookups
        # are already covered by the auto-generated PK B-tree.
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_priors_native_lemma "
            "ON empirical_priors_native(lemma_ref)"
        )
        applied["empirical_priors_native_table"] = True

    cur = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='empirical_priors_loan'"
    )
    if cur.fetchone() is None:
        db.conn.execute(
            """
            CREATE TABLE empirical_priors_loan (
              donor         TEXT NOT NULL,
              recipient     TEXT NOT NULL,
              position      TEXT NOT NULL,
              tag           TEXT NOT NULL,
              era_midpoint  INTEGER NOT NULL,
              lemma_ref     TEXT NOT NULL,
              count         INTEGER NOT NULL CHECK (count > 0),
              PRIMARY KEY (donor, recipient, position, tag, era_midpoint, lemma_ref)
            )
            """
        )
        # Only the lemma_ref index is non-redundant — PK-prefix lookups
        # are already covered by the auto-generated PK B-tree.
        db.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_priors_loan_lemma ON empirical_priors_loan(lemma_ref)"
        )
        applied["empirical_priors_loan_table"] = True


def _create_toponym_decomposition_table(db: LexiconDB, applied: dict[str, bool]) -> None:
    """wyrd-08m Phase 1: ensure ``toponym_decomposition`` table + indexes
    exist on legacy DBs.

    Stores matcher-derived alternative breakdowns per toponym (one row
    per signature), with a single canonical pick per toponym (rule-driven
    in ``decomposition.pick_canonical_decomposition``). The unique index
    on (toponym_id, decomposition_signature) makes re-running the
    populator idempotent.

    Idempotent: checks for the table first; runs as no-op on fresh
    installs (table ships in data/lexicon.sql).
    """
    cur = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='toponym_decomposition'"
    )
    if cur.fetchone() is not None:
        return
    db.conn.execute(
        """
        CREATE TABLE toponym_decomposition (
          id                       INTEGER PRIMARY KEY AUTOINCREMENT,
          toponym_id               INTEGER NOT NULL REFERENCES toponym(id) ON DELETE CASCADE,
          decomposition_signature  TEXT NOT NULL,
          morpheme_ids             TEXT NOT NULL,
          unaccounted_fragments    TEXT NOT NULL,
          unaccounted_count        INTEGER NOT NULL DEFAULT 0,
          morpheme_count           INTEGER NOT NULL DEFAULT 0,
          is_canonical             INTEGER NOT NULL DEFAULT 0
                                   CHECK (is_canonical IN (0, 1)),
          canonical_source         TEXT CHECK (
            canonical_source IN (
              'scholar', 'scholar-disagreement',
              'unique-zero-unaccounted', 'tiebreaker'
            ) OR canonical_source IS NULL
          ),
          created_at               TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    db.conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_toponym_decomposition_unique "
        "ON toponym_decomposition(toponym_id, decomposition_signature)"
    )
    db.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_toponym_decomposition_topo "
        "ON toponym_decomposition(toponym_id)"
    )
    db.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_toponym_decomposition_canonical "
        "ON toponym_decomposition(toponym_id) WHERE is_canonical = 1"
    )
    applied["toponym_decomposition_table"] = True


def _create_toponym_attestation_unique_index(db: LexiconDB, applied: dict[str, bool]) -> None:
    """wyrd-skm Phase 3.0a: ensure ``toponym_attestation`` has a unique
    index on ``(toponym_id, form, COALESCE(date_year, 0),
    COALESCE(source_doc, ''))`` so the ``mine-attestations`` ingest
    is idempotent.

    The base table ships in ``data/lexicon.sql`` but pre-existed without
    a unique constraint. Adding one via CREATE UNIQUE INDEX rather than
    a table-recreate keeps existing rows intact and runs as a no-op on
    fresh schemas (the index ships in lexicon.sql alongside the table).

    The COALESCE wraps match idx_toponym_unique + idx_etymon_citation_unique.
    Without them SQLite treats every NULL as distinct under UNIQUE, so two
    attestations of the same (toponym, form, source_doc) with NULL year
    would both insert — the wrap closes that idempotency hole. wyrd-67fv
    round-5 review (Gemini) caught the pre-PR shape; legacy DBs whose
    index was created BEFORE this fix still carry the bare-NULL shape
    and need manual DROP + CREATE to update.

    Existing rows are NOT deduped here — the table is empty in
    production today (mining hasn't run) so there's nothing to clean
    up. If a deployment somehow has duplicate rows, the unique-index
    creation will fail loudly rather than silently dropping data; the
    operator should investigate before re-running migrate.
    """
    rows = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_attestation_unique'"
    ).fetchall()
    if not rows:
        db.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_attestation_unique "
            "ON toponym_attestation("
            "  toponym_id, form, COALESCE(date_year, 0), COALESCE(source_doc, '')"
            ")"
        )
        applied["idx_attestation_unique"] = True


def _create_mining_run_table(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Create mining_run if missing. Per D23, this audit table closes the
    'stop losing accept/decline/reject counts to stdout' gap. One row per
    (book, provider, model, started_at) tuple; the writer is invoked at
    the end of every mine-llm and review run.

    Schema:
      id           - autoincrement
      source_id    - FK to source.id (the book that was mined)
      provider     - 'ollama' | 'gemini' | 'anthropic' | 'unknown'
      model        - free-text model id (e.g. 'qwen3.5:9b', 'gemini-2.5-flash')
      mode         - 'mine' | 'review' — distinguishes Tier 1 from Tier 2
      started_at   - ISO timestamp; null if not recorded
      completed_at - ISO timestamp; null if not recorded
      parsed_count - how many entries were fed to the model
      accepted     - extracted etymology with elements
      declined     - model returned found:false
      rejected     - extraction failed validation (form-in-body, etc.)
      by_failure   - JSON object of {reason: count}; null if not recorded
      notes        - free-text context, e.g. 're-run after parser fix'
    """
    existing = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "mining_run" in existing:
        return
    db.conn.executescript(
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
        );
        CREATE INDEX idx_mining_run_source ON mining_run(source_id);
        CREATE INDEX idx_mining_run_completed ON mining_run(completed_at);
        """
    )
    applied["mining_run_table"] = True


def _create_fantasy_morpheme_table(db: LexiconDB, applied: dict[str, bool]) -> None:
    """wyrd-ami: create fantasy_morpheme to record (name, description) → resolution.

    Each input fed to `lexicon mine-fantasy-name` produces ONE row here,
    regardless of whether the morpheme was usable. Usable rows link to
    a real etymon (already in the etymon table from corpus mining); barred
    rows record `bar_reason` so a later pass can revisit (e.g. wyrd-0ab's
    constructed-etymology Phase 2 may rescue rows barred as
    `no_etymology_found` or `outside_language_family`).

    `approach_version` carries the pipeline-version marker the user asked
    for: when we ship a stronger pipeline (e.g. after wyrd-gpif lands
    alt-form ingestion), bump the version constant in
    `extractors.fantasy.APPROACH_VERSION` and re-process any rows from
    earlier versions.
    """
    existing = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "fantasy_morpheme" in existing:
        # Table exists from an earlier migration; add columns if missing.
        # Track BOTH presence and notnull-ness so we can verify required
        # columns aren't silently nullable.
        col_info = {
            row["name"]: bool(row["notnull"])
            for row in db.conn.execute("PRAGMA table_info(fantasy_morpheme)")
        }
        cols = set(col_info)
        if "unapproved_language" not in cols:
            db.conn.execute("ALTER TABLE fantasy_morpheme ADD COLUMN unapproved_language TEXT")
            db.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fantasy_morpheme_unapproved "
                "ON fantasy_morpheme(unapproved_language)"
            )
            applied["fantasy_morpheme.unapproved_language"] = True
        if "unapproved_form" not in cols:
            db.conn.execute("ALTER TABLE fantasy_morpheme ADD COLUMN unapproved_form TEXT")
            applied["fantasy_morpheme.unapproved_form"] = True
        # Required NOT NULL columns from the current schema. If any are
        # missing the table is from a partial/corrupted state — we can't
        # silently ALTER NOT NULL without defaults, so raise instead.
        required_not_null = {
            "input_name",
            "usable",
            "resolution_method",
            "approach_version",
            "processed_at",
        }
        missing_required = required_not_null - cols
        if missing_required:
            raise RuntimeError(
                f"fantasy_morpheme table missing required columns "
                f"{sorted(missing_required)}; manual intervention needed "
                f"(drop the table or add columns with defaults)."
            )
        nullable_required = {c for c in required_not_null if not col_info[c]}
        if nullable_required:
            raise RuntimeError(
                f"fantasy_morpheme table has columns {sorted(nullable_required)} "
                f"declared NULL where the current schema requires NOT NULL; "
                f"data integrity issues will follow. Manual intervention needed."
            )
        # SQLite ALTER TABLE can't change a column's collation; the
        # original CREATE statement is the source of truth. Read it
        # from sqlite_master and look for the COLLATE NOCASE clause
        # on input_name. If it's missing, recreate the table — but
        # only when there are no existing rows (drop+recreate is
        # destructive). If rows exist, raise so the operator can
        # decide how to migrate.
        ddl_row = db.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='fantasy_morpheme'"
        ).fetchone()
        ddl = (ddl_row["sql"] if ddl_row else "") or ""
        ddl_lower = ddl.lower()
        if "input_name" in ddl_lower and "collate nocase" not in ddl_lower:
            row_count = db.conn.execute("SELECT COUNT(*) FROM fantasy_morpheme").fetchone()[0]
            if row_count == 0:
                db.conn.executescript(
                    "DROP TABLE fantasy_morpheme;"
                    "DROP INDEX IF EXISTS idx_fantasy_morpheme_approach;"
                    "DROP INDEX IF EXISTS idx_fantasy_morpheme_etymon;"
                    "DROP INDEX IF EXISTS idx_fantasy_morpheme_usable;"
                    "DROP INDEX IF EXISTS idx_fantasy_morpheme_unapproved;"
                )
                applied["fantasy_morpheme_recreated_for_collation"] = True
                # Fall through to the fresh CREATE block below.
            else:
                raise RuntimeError(
                    "fantasy_morpheme.input_name is missing COLLATE NOCASE; "
                    "case-sensitivity will produce duplicate rows for "
                    "different casings of the same name. The table has "
                    f"{row_count} rows so we won't auto-recreate; "
                    "rebuild manually after exporting any data you need to keep."
                )
        else:
            return
    db.conn.executescript(
        """
        CREATE TABLE fantasy_morpheme (
          id                  INTEGER PRIMARY KEY AUTOINCREMENT,
          -- COLLATE NOCASE so 'Harpy' and 'harpy' are the same row
          -- under the (input_name, approach_version) UNIQUE — avoids
          -- redundant LLM research calls when callers happen to vary
          -- the casing of the same name.
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
          -- When bar_reason='outside_language_family', record what
          -- language the LLM thinks the etymology lives in and what
          -- form. Lets us aggregate `SELECT unapproved_language,
          -- COUNT(*) ... GROUP BY unapproved_language` to prioritize
          -- new languages to approve.
          unapproved_language TEXT,
          unapproved_form     TEXT,
          processed_at        TEXT NOT NULL,
          UNIQUE (input_name, approach_version)
        );
        CREATE INDEX idx_fantasy_morpheme_approach    ON fantasy_morpheme(approach_version);
        CREATE INDEX idx_fantasy_morpheme_etymon      ON fantasy_morpheme(etymon_id);
        CREATE INDEX idx_fantasy_morpheme_usable      ON fantasy_morpheme(usable);
        CREATE INDEX idx_fantasy_morpheme_unapproved  ON fantasy_morpheme(unapproved_language);
        """
    )
    applied["fantasy_morpheme_table"] = True


def record_mining_run(
    db: LexiconDB,
    *,
    source_id: str,
    provider: str,
    model: str,
    mode: str,
    parsed_count: int,
    accepted: int,
    declined: int,
    rejected: int,
    by_failure: dict[str, int] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    notes: str | None = None,
) -> int:
    """Persist one mining-run summary row. Idempotent on
    (source_id, provider, model, mode, completed_at) — a re-run with
    the same completed_at gets ignored. completed_at IS NULL means
    "not yet finished" and won't dedupe (each call gets a new row).
    """
    # sort_keys keeps the serialized form stable across runs so log-style
    # diffs / human inspection of the audit table behave deterministically
    # regardless of dict insertion order.
    by_failure_json = json.dumps(by_failure, sort_keys=True) if by_failure else None
    # ON CONFLICT(...) DO NOTHING (vs INSERT OR IGNORE) limits the silence
    # to UNIQUE conflicts on the dedupe key. CHECK / NOT NULL / FK
    # violations still raise — bad mode values must fail loudly rather
    # than silently dropping the row.
    cur = db.conn.execute(
        """
        INSERT INTO mining_run
          (source_id, provider, model, mode, started_at, completed_at,
           parsed_count, accepted, declined, rejected, by_failure, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id, provider, model, mode, completed_at) DO NOTHING
        """,
        (
            source_id,
            provider,
            model,
            mode,
            started_at,
            completed_at,
            parsed_count,
            accepted,
            declined,
            rejected,
            by_failure_json,
            notes,
        ),
    )
    db.commit()
    # rowcount is 1 on insert, 0 on UNIQUE-conflict skip. Returning 0 lets
    # callers (e.g. import-mining-log) distinguish "inserted" from "duplicate".
    return cur.lastrowid if cur.rowcount else 0


def migrate_schema(db: LexiconDB) -> dict[str, bool]:
    """Bring an existing lexicon DB up to the current schema.

    Idempotent — running on an already-migrated DB is a no-op. Returns a
    dict of {migration_name: applied?} so callers can report.

    Each helper operates on its own slice of the schema, but the
    call order matters where there's a dependency: renames must
    precede column-adds (so a rename isn't followed by an ADD of
    the now-orphaned legacy column name), column-adds precede
    index-creates against those columns, and table-creates precede
    helpers that ALTER columns onto those tables. The
    ``_rename_synset_to_cognate`` → ``_add_etymon_columns`` ordering
    just below is the explicit example. Reading the helpers in the
    call order below shows the full migration surface.
    """
    applied = {
        "etymon.lemma_id": False,
        "etymon.inflection": False,
        "etymon.merged_into_id": False,
        "etymon.lemma_method": False,
        "etymon.cognate_id": False,
        "etymon.cognate_method": False,
        "idx_etymon_lemma": False,
        "idx_etymon_merged_into": False,
        "idx_etymon_cognate": False,
        "etymon_text_match.method": False,
        "etymon_text_match.attested_year": False,
        "etymon_consensus_view": False,
        "etymon_canonical_view": False,
        "etymon_gloss_canonical_view": False,
        "etymon_tag_canonical_view": False,
        "etymon_text_match_canonical_view": False,
        "etymon_text_match_table": False,
        "etymon_descent_table": False,
        "etymon_variant_table": False,
        "mining_run_table": False,
        "etymon_citation.context_snippet": False,
        "toponym_etymology.attested_year": False,
        # wyrd-08qv: multi-extractor consensus canonicalization.
        "toponym_etymology.is_canonical": False,
        "toponym_etymology.consensus_size": False,
        "toponym_etymology.cluster_key": False,
        "idx_toponym_etymology_canonical": False,
        "toponym_etymology_canonical_view": False,
        "wal_mode": False,
        "meaning_synset_table": False,
        "etymon_meaning_synset_table": False,
        "etymon.synset_id_renamed_to_cognate_id": False,
        "etymon.synset_method_renamed_to_cognate_method": False,
        "idx_etymon_synset_renamed_to_idx_etymon_cognate": False,
        "fantasy_morpheme_table": False,
        "fantasy_morpheme.unapproved_language": False,
        "fantasy_morpheme.unapproved_form": False,
        # wyrd-ha9q Phase 2a: pronunciation + multi-script renderings
        "etymon.pronunciation_ipa": False,
        "etymon.pronunciation_dialect": False,
        "etymon.original_script": False,
        "etymon.transliteration": False,
        "etymon.english_shaped": False,
        # wyrd-lr4: within-language stratum tag.
        "etymon.stratum": False,
        "idx_etymon_stratum": False,
        # wyrd-skm Phase 3.0a: idempotent ingest of toponym_attestation.
        "idx_attestation_unique": False,
        # wyrd-unuo Phase 3.3: per-etymon period-form projection table.
        "etymon_period_form_table": False,
        # wyrd-08m Phase 1: matcher-derived decomposition multiplicity.
        "toponym_decomposition_table": False,
        # wyrd-ecjp.2: vector-driven empirical-baseline priors.
        "empirical_priors_native_table": False,
        "empirical_priors_loan_table": False,
    }
    # wyrd-44a: rename the legacy cognate-cluster column from synset_id
    # to cognate_id BEFORE the add-columns helper runs — otherwise
    # _add_etymon_columns would see no `cognate_id` and try to ADD it,
    # leaving the legacy `synset_id` column orphaned.
    _rename_synset_to_cognate(db, applied)
    _add_etymon_columns(db, applied)
    _create_etymon_indexes(db, applied)
    _rebuild_etymon_views(db, applied)
    _migrate_text_match_table(db, applied)
    _create_etymon_descent_table(db, applied)
    _create_etymon_variant_table(db, applied)
    _create_mining_run_table(db, applied)
    _migrate_citation_context_snippet(db, applied)
    _migrate_toponym_etymology_attested_year(db, applied)
    _migrate_toponym_etymology_canonical(db, applied)
    _create_meaning_synset_tables(db, applied)
    _create_fantasy_morpheme_table(db, applied)
    _migrate_wal_mode(db, applied)
    # wyrd-skm Phase 3.0a: runs last so an IntegrityError from a
    # legacy DB that somehow has duplicate (toponym_id, form, date_year,
    # source_doc) rows doesn't abort an earlier migration's commit.
    # The mine-attestations ingest is a fresh population in production,
    # but defending against unknown legacy state is cheap.
    _create_toponym_attestation_unique_index(db, applied)
    _create_etymon_period_form_table(db, applied)
    _create_toponym_decomposition_table(db, applied)
    _create_empirical_priors_tables(db, applied)
    db.commit()
    return applied


def _rename_synset_to_cognate(db: LexiconDB, applied: dict[str, bool]) -> None:
    """wyrd-44a: rename legacy etymon.synset_id → etymon.cognate_id (and
    .synset_method → .cognate_method, idx_etymon_synset →
    idx_etymon_cognate). Idempotent — runs only on legacy DBs that
    still carry the old names; fresh-install DBs already have the
    cognate-prefixed names from data/lexicon.sql.

    Renames the COLUMN in place via SQLite's ALTER TABLE RENAME COLUMN
    (3.25+) so existing data carries over without a copy. Drops + re-
    creates the index by the new name (SQLite has no RENAME INDEX).
    Order matters: rename the column FIRST so the index recreate
    references the new column name.
    """
    cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
    if "synset_id" in cols and "cognate_id" not in cols:
        db.conn.execute("ALTER TABLE etymon RENAME COLUMN synset_id TO cognate_id")
        applied["etymon.synset_id_renamed_to_cognate_id"] = True
    if "synset_method" in cols and "cognate_method" not in cols:
        db.conn.execute("ALTER TABLE etymon RENAME COLUMN synset_method TO cognate_method")
        applied["etymon.synset_method_renamed_to_cognate_method"] = True
    existing_indexes = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    if "idx_etymon_synset" in existing_indexes:
        db.conn.execute("DROP INDEX idx_etymon_synset")
        # The post-drop CREATE happens in _create_etymon_indexes so the
        # rename helper stays focused on the rename itself; that helper
        # is idempotent and will see the missing index next.
        applied["idx_etymon_synset_renamed_to_idx_etymon_cognate"] = True


def _create_meaning_synset_tables(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Create meaning_synset + etymon_meaning_synset if missing (wyrd-7tz).

    Schema mirrors data/lexicon.sql so fresh-install and migration paths
    stay in lockstep. Idempotent — checks for the tables before creating.

    Distinct from etymon.cognate_id (cognate-cluster ID from
    cluster-cognates enrichment); these are MEANING synsets — fine-
    grained semantic equivalence used by upcoming generator transforms.
    See the comment block in data/lexicon.sql for why the naming
    overlaps and how to disambiguate.
    """
    existing = {
        row["name"]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "meaning_synset" not in existing:
        db.conn.executescript(
            """
            CREATE TABLE meaning_synset (
              id              INTEGER PRIMARY KEY AUTOINCREMENT,
              canonical_label TEXT NOT NULL UNIQUE,
              hypernym_id     INTEGER REFERENCES meaning_synset(id) ON DELETE SET NULL,
              notes           TEXT
            );
            CREATE INDEX idx_meaning_synset_hypernym ON meaning_synset(hypernym_id);
            """
        )
        applied["meaning_synset_table"] = True
    if "etymon_meaning_synset" not in existing:
        db.conn.executescript(
            """
            CREATE TABLE etymon_meaning_synset (
              etymon_id         INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
              meaning_synset_id INTEGER NOT NULL REFERENCES meaning_synset(id) ON DELETE CASCADE,
              fit               TEXT NOT NULL CHECK (fit IN ('core', 'peripheral')),
              PRIMARY KEY (etymon_id, meaning_synset_id)
            );
            CREATE INDEX idx_etymon_meaning_synset_synset
              ON etymon_meaning_synset(meaning_synset_id);
            """
        )
        applied["etymon_meaning_synset_table"] = True


def _migrate_wal_mode(db: LexiconDB, applied: dict[str, bool]) -> None:
    """Idempotently bring legacy DBs (created before WAL was the
    default) onto journal_mode=WAL + synchronous=NORMAL. Both are
    file-persistent: once set on a file, every subsequent open
    inherits them — so this is a one-shot migration, not a
    per-connection setup. Reports applied=True only when the actual
    on-disk mode changed."""
    current_mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    if current_mode.lower() != "wal":
        _apply_persistent_pragmas(db.conn)
        applied["wal_mode"] = True
