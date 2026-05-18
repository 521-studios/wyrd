# Kenning-specific agent reviewers

These reviewers compose with the universal `AGENT-REVIEWERS.md` at
repo root — they don't replace it. Scope: files under
`wyrd/generators/kenning/` and `tests/test_kenning_*`.

## lexicon-package-structure-reviewer

Enforce that lexicon-DB concerns live under
`wyrd/generators/kenning/lexicon/`. Per wyrd-67fv, the lexicon was
split out of a 7,272-line `lexicon.py` into a package. Every file
that touches the lexicon DB belongs in the package; the rest of
`wyrd/generators/kenning/` stays runtime-only (`__init__.py`,
`meaning.py`, `name.py`, `proportions.py`, `trie_matcher.py`, etc.).

**FLAG when a file outside `lexicon/` contains:**

1. **A SQL `CREATE TABLE`, `CREATE INDEX`, or `CREATE VIEW`.** These
   are schema concerns and belong in an Alembic migration under
   `lexicon/sql/migrations/versions/`.

2. **An `import sqlite3` + opening `wyrd.generators.kenning` data
   from disk.** That code is a candidate lexicon caller and should
   either use `LexiconDB` (which already encapsulates the connection
   + pragmas) or move into the package next to its peer enrichment
   modules.

3. **A new module that does "scan the lexicon DB and write back to
   it"** (enrichment / ingest / audit / report pass). New modules of
   this shape belong inside `lexicon/` — at minimum under
   `lexicon/<descriptive_name>.py`, and ideally inside the
   subpackage that wyrd-67fv follow-ups will create
   (`lexicon/enrichment/`, `lexicon/ingest/`, etc.). Existing
   modules outside the package (`disambiguator.py`, `strata.py`,
   `english_shaping.py`, `wiktextract_ingester.py`, etc.) are
   pre-split files that will move in follow-up tickets — DON'T
   re-flag those; only flag NEW files of the same shape.

**Acceptable** (don't flag):

* Runtime-layer files (`__init__.py`, `meaning.py`, `name.py`,
  `proportions.py`, `era.py`, `respelling.py`, `scripts.py`,
  `trie_matcher.py`, `decomposition.py`, `rewind.py`,
  `phonology.py`, `vector_schemas.py`) — these read the bundled
  `meanings.json` and have no DB access.
* `cli.py` — CLI wiring lives here per project convention; will be
  split separately under wyrd-g143.
* Pre-existing files in `wyrd/generators/kenning/` that touch the
  lexicon DB (above list). Their move is a known follow-up.

**Review approach:**
1. Grep changed files for `CREATE TABLE` / `CREATE INDEX` /
   `CREATE VIEW`.
2. For each new `.py` file in `wyrd/generators/kenning/` (not under
   `lexicon/`), open it and check whether it does lexicon-side I/O.
3. If a NEW file is misplaced: **block the PR.** Request the file
   move before merge. Placement is a structural decision; landing
   new code in the wrong place and ticketing the cleanup recreates
   the exact debt wyrd-67fv was filed to repay. There is no
   "defer with a ticket" path for new files.
4. If pre-existing out-of-place code is touched (the
   disambiguator / strata / english_shaping / ingester family),
   that's not the same — those moves are explicit follow-up
   tickets. Don't flag those re-edits.

## alembic-migration-discipline-reviewer

Schema changes must go through Alembic. Per wyrd-67fv, the layered
migrations under `lexicon/sql/migrations/versions/` (currently
0001_sources … 0008_views) are the source of truth for the lexicon
schema. The committed `data/lexicon.sql` file is regenerated from
those migrations and is documentation, not authoritative.

**FLAG when:**

1. **`data/lexicon.sql` is edited directly** — the file is a
   regenerated artifact. Schema changes belong in a new alembic
   migration under `lexicon/sql/migrations/versions/`; regenerate
   `data/lexicon.sql` as the last step.

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
   to match the existing `data/lexicon.sql` shape when capturing
   the current schema. Use `op.create_table()` etc. for genuinely
   new tables added after the wyrd-67fv baseline.

**Acceptable** (don't flag):

* Edits to `data/lexicon.sql` that are the regenerated output of a
  new migration (the migration is in the same PR).
* Removing `_migrate_*` helpers — that's the deprecation path
  we're on.

**Review approach:**
1. Check that any change to `data/lexicon.sql` has a corresponding
   new file under `lexicon/sql/migrations/versions/`.
2. Check that any new `_create_*_table` / `_migrate_*` function
   isn't a duplicate of an Alembic migration that should have
   landed instead.
3. Confirm that the next migration's `revision` ID is
   monotonically ordered and its `down_revision` chains to the
   previous head.

## importlib-resources-reviewer

Package data files (JSON sidecars, SQL schema, fixture text) must
be loaded via `importlib.resources`, not `Path(__file__).parent`.
The `__file__.parent` pattern silently broke when `lexicon.py` was
renamed to `lexicon/__init__.py` because the parent directory
shifted by one level (caught in wyrd-67fv, fix at
`lexicon/__init__.py:_load_norman_manorial_family_tokens`). The
importlib.resources pattern is robust to package moves and works
identically in dev (editable install) and Lambda (frozen package).

**FLAG when a file under `wyrd/generators/kenning/` contains:**

* `Path(__file__).parent / "data"` (or any `__file__.parent` /
  `Path(__file__).parents[N]` to navigate to a sibling data file).

**Acceptable pattern:**

```python
from importlib import resources

data = resources.files("wyrd.generators.kenning.data").joinpath(
    "norman_manorial_families.json"
)
families = json.loads(data.read_text())
```

**Acceptable** (don't flag):

* `Path(__file__).parent` used for write paths (e.g. test fixtures
  writing to a tmp dir relative to the test file). Resources are
  read-only by definition.
* `__file__` references in `tests/` (tests have a stable layout
  and aren't packaged).

**Review approach:**
1. Grep changed files under `wyrd/generators/kenning/` for
   `Path(__file__).parent` and `__file__.parents`.
2. For each hit, check whether the path resolves to a bundled
   data file. If so, recommend the `importlib.resources` form.

This rule is kenning-specific only because the package's layered
structure makes path drift more likely than in flat packages. The
underlying principle (use importlib.resources for package data) is
universal; if other generators grow data sidecars, promote this
reviewer to the repo-root `AGENT-REVIEWERS.md`.
