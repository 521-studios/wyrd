# lexicon-package-structure-reviewer

Enforce that lexicon-DB concerns live under
`wyrd/generators/kenning/lexicon/`. Per wyrd-67fv, the lexicon was
split out of a 7,272-line `lexicon.py` into a package. Every file
that touches the lexicon DB belongs in the package; the rest of
`wyrd/generators/kenning/` stays runtime-only (`__init__.py`,
`runtime/meaning.py`, `runtime/name.py`, `runtime/proportions.py`, `runtime/trie_matcher.py`, etc.).

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

* Runtime-layer files — the `runtime/` subpackage
  (`runtime/__init__.py`, `runtime/meaning.py`, `runtime/name.py`,
  `runtime/word.py`, `runtime/proportions.py`, `runtime/respelling.py`,
  `runtime/scripts.py`, `runtime/trie_matcher.py`,
  `runtime/decomposition.py`) plus `era/cells.py`, `era/rewind.py`,
  `registers/phonology.py`, `vectors/schemas.py`) — these read the bundled
  `meanings.json` and have no DB access.
* `cli.py` — CLI wiring lives here per project convention; will be
  split separately under wyrd-g143.
* The `jsonl/` subpackage (`jsonl/build.py`, `jsonl/dump.py`,
  `jsonl/log.py`) — these own the L2 side of the L2/L3 boundary
  per `L2_L3_BOUNDARY.md`. They DO touch the lexicon DB (build.py
  replays events into it; dump.py reads from it) but the
  jsonl/-as-peer-package-to-lexicon/ shape is the intentional
  architectural split: lexicon/ owns the SQLite + enrichment
  surface; jsonl/ owns the canonical-state event-log surface;
  together they implement the contract that JSONL is the source of
  truth and the DB is a rebuildable build artifact.
* The `ingesters/` subpackage (`ingesters/hundred_rolls.py`,
  `ingesters/speed_1611.py`, `ingesters/hearth_tax.py`,
  `ingesters/os_open_names.py`, `ingesters/domesday.py`) — these
  own the operator-side data-ingest surface. The 4 CSV ingesters
  route through `jsonl/log.py` (no direct DB writes); `domesday.py`
  writes directly to the lexicon DB via sqlite3 (Open Domesday's
  Hull-team data is bulk-imported into per-source tables for
  attestation mining). The ingesters/-as-peer-package-to-lexicon/
  shape is the intentional architectural split: lexicon/ owns
  DB-side enrichment passes (wiktextract / etymonline / mining-LLM
  output); ingesters/ owns operator-imports-CSV-or-gazetteer-data
  paths. The split is by INTENT (who runs it: enrichment pipeline
  vs operator manual ingest) not by I/O target. Note: future
  wyrd-29mn shared-base extraction will land in
  `ingesters/_csv_base.py`.
* The `bundle/` subpackage (`bundle/browse.py`,
  `bundle/rando_port_readiness.py`, `bundle/wikipedia_backfill_report.py`)
  — these READ the lexicon DB to produce developer-facing reports and
  browses, but do not OWN any DB-writing surface. The
  bundle/-as-peer-package-to-lexicon/ shape is the intentional
  read-vs-write split: lexicon/ owns DB-side authoring (build, mining,
  enrichment, dump); bundle/ owns DB-side reading (browse helpers,
  rando-server readiness, Wikipedia-seed retirement reports). Note
  the parallel-name distinction with `kenning/lexicon/bundle/`
  (which lives one level deeper and owns BUILDING the deploy
  bundle from the lexicon DB) — both are correctly named for their
  context but distinct subpackages.
* The `language_quality/` subpackage (`language_quality/models.py`,
  `language_quality/audits.py`, `language_quality/reporting.py`)
  — these READ the lexicon DB to compute per-language scorecard
  metrics and produce the markdown / JSON dashboard report (wyrd-wzwa,
  parent epic wyrd-eni4). Like `bundle/`, the audits surface opens
  sqlite3 connections extensively but does not OWN any DB-writing
  surface. Same read-vs-write split rationale: lexicon/ owns DB-side
  authoring; language_quality/ owns the dashboard READING surface.

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

