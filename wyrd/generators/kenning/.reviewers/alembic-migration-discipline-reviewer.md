# alembic-migration-discipline-reviewer

Schema changes must go through Alembic. Per wyrd-67fv, the layered
migrations under `lexicon/sql/migrations/versions/` (currently
0001_sources … 0008_views) are the source of truth for the lexicon
schema. The committed `data/seed/lexicon.sql` file is regenerated from
those migrations and is documentation, not authoritative.

**FLAG when:**

1. **`data/seed/lexicon.sql` is edited directly** — the file is a
   regenerated artifact. Schema changes belong in a new alembic
   migration under `lexicon/sql/migrations/versions/`; regenerate
   `data/seed/lexicon.sql` as the last step.

2. **A new `_create_*_table` or `_migrate_*` helper function lands
   in `lexicon/__init__.py`.** The historical helpers (~1,300
   lines) exist because the schema evolved before Alembic was
   adopted. New schema changes go in Alembic migrations, not in
   another `_migrate_*` helper. Adding to the legacy chain locks in
   the old pattern and grows the file the wyrd-67fv split is
   shrinking.

3. **A new migration is added but the SA Core MetaData in
   `lexicon/sql/tables.py` isn't updated.** The MetaData is the
   target of `alembic revision --autogenerate` and should mirror
   what the migrations produce on a fresh DB. Drift between them
   silently breaks future autogenerate runs.

4. **A migration uses `op.create_table()` with constraint or
   default formatting that doesn't survive a round-trip through
   `sqlite_master.sql`.** SQLite stores DDL verbatim, so the
   side-by-side equality check (init_schema-vs-alembic) the
   wyrd-67fv design relies on can be broken by SA's auto-quoting.
   Prefer `op.execute("""CREATE TABLE ...""")` with the SQL written
   to match the existing `data/seed/lexicon.sql` shape when capturing
   the current schema. Use `op.create_table()` etc. for genuinely
   new tables added after the wyrd-67fv baseline.

**Acceptable** (don't flag):

* Edits to `data/seed/lexicon.sql` that are the regenerated output of a
  new migration (the migration is in the same PR).
* Removing `_migrate_*` helpers — that's the deprecation path
  we're on.

**Review approach:**
1. Check that any change to `data/seed/lexicon.sql` has a corresponding
   new file under `lexicon/sql/migrations/versions/`.
2. Check that any new `_create_*_table` / `_migrate_*` function
   isn't a duplicate of an Alembic migration that should have
   landed instead.
3. Confirm that the next migration's `revision` ID is
   monotonically ordered and its `down_revision` chains to the
   previous head.

