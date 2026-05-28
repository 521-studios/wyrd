# db-reconstructibility-reviewer

Review changes to the lexicon DB schema, ingesters, and mining pipelines for **reconstructibility from JSONL artifacts**. **Default severity: P1** — this is a data-loss class of failure: when reconstructibility is broken, blowing away the DB means re-running expensive LLM calls, days-long Ollama runs, or irreversible OCR effort.

The architectural rule this reviewer enforces:

1. **The lexicon DB at `~/.wyrd/lexicon.db` can be blown away at any time.** Operators must be able to `rm ~/.wyrd/lexicon.db && wyrd kenning lexicon rebuild-from-jsonl` and end up with the same DB state.
2. **Reconstruction MUST require no mining.** Mining is expensive — LLM calls cost API credits, Ollama runs take days, OCR is irreversible operator effort. None of this should be re-run to rebuild the DB.
3. **All mining MUST leave durable JSONL artifacts** under `data/mining/` (or a sibling per-source directory). The JSONL files are the canonical source of truth; the DB is a queryable index over them.
4. **`rebuild-from-jsonl` is the canonical reconstruction path.** Every event type any ingester writes must be replayable by it.

The failure mode this guards against: an ingester that writes DB rows but emits no JSONL, so blowing away the DB means re-running the (expensive) ingest from the source materials — which may themselves be lost, paywalled, or behind a flaky network the operator no longer wants to depend on.

**Scope rule (touch-it-you-own-it):** New ingesters, schema additions, mining CLIs, and any module that does `INSERT INTO` against the lexicon DB are in scope. Read-only paths (queries, reports, generation code) are out of scope.

**What to check:**

1. Walk the PR diff for any of:
   - `CREATE TABLE` in `wyrd/generators/kenning/data/lexicon.sql`, `wyrd/generators/kenning/lexicon/schema.py`, or `wyrd/generators/kenning/lexicon/sql/migrations/versions/`
   - `INSERT INTO` / `INSERT OR IGNORE INTO` / `INSERT OR REPLACE INTO` in any new ingester or CLI module
   - New `mine-*` / `ingest-*` CLI subcommands
   - LLM calls (`chat_json`, `provider="ollama"|"gemini"|"anthropic"`, etc.) or network fetches whose outputs land in the DB

2. For each new write path, confirm:
   - **JSONL emission exists.** Either the ingester writes to `data/mining/<source_id>.jsonl` as it ingests (per-event), OR a `dump-jsonl` subcommand at `wyrd/generators/kenning/jsonl/dump.py` knows how to round-trip the new rows back out.
   - **`rebuild-from-jsonl` handles the new event types.** Look at `wyrd/generators/kenning/jsonl/build.py` (or wherever `rebuild-from-jsonl` dispatches event `_type` values) and confirm the new `_type` strings are wired in.
   - **The reconstruction does not require the source materials.** The JSONL must be self-contained — no `pdftotext`, no re-LLM-calling, no S3 fetch is acceptable as part of `rebuild-from-jsonl`.

3. For each new mining CLI, confirm:
   - The CLI writes JSONL **before** (or in parallel with) DB writes, so a SIGTERM/crash leaves a recoverable trail.
   - The JSONL path is documented in the docstring + `data/mining/` is the conventional location.

**Canonical pattern** (the one to point new ingesters at):

```
mine-llm <source.txt>
    → DB writes
    → operator runs `lexicon dump-jsonl`
    → committed to `data/mining/<source>.jsonl`

rebuild-from-jsonl
    → reads every `data/mining/*.jsonl`
    → replays into a fresh DB
```

A future-better pattern (event-log-first):

```
ingest-X <source>
    → writes JSONL events DIRECTLY to data/mining/<source>.jsonl
    → loads JSONL into DB via the same code path rebuild-from-jsonl uses

rebuild-from-jsonl
    → reads every data/mining/*.jsonl (including the freshly-ingested one)
    → no separate dump-jsonl step needed
```

**Flag issues if:**

- A new ingester writes to the lexicon DB but emits no JSONL artifact AND is not handled by `dump-jsonl`. Concretely: blowing away the DB and running `rebuild-from-jsonl` would leave those rows missing.
- A new table is added to the schema (alembic migration / `data/lexicon.sql` / `schema.py` helper) but `wyrd/generators/kenning/jsonl/dump.py` is not updated to round-trip it AND no per-ingester JSONL emission lands the rows.
- A new `mine-*` / `ingest-*` CLI writes DB rows but the only path back is "re-run the LLM on the source materials" — i.e., reconstruction depends on either expensive compute or staged source files.
- A migration adds a column whose value is mined (not derived) and the mining step has no JSONL trail.
- An ingester takes an external source (HTTP URL, S3 fetch, PDF) and writes to the DB without first capturing the raw response / parsed events to JSONL.
- The PR description claims "idempotent re-run on the same .txt reproduces the DB" — but the .txt itself isn't part of the in-repo reconstruction surface. The JSONL is the boundary; staged source materials are not.

**Do NOT flag:**

- Schema migrations with no data (pure `CREATE TABLE` / `ALTER TABLE` with no INSERT).
- L3 enrichment columns whose values are deterministically derivable from existing DB rows via a re-runnable `lexicon enrich --apply` pass. These are computed from DB content, not from external sources.
- Test fixtures, smoke-test code, or paths gated by `if TESTING` / `pytest`-only construction. Tests build throwaway DBs.
- Read-only paths (queries, reports, browse, audit) that don't write to the DB.
- The bundle export path (`bundle.json` / `meanings.json`) — those are downstream artifacts derived from DB queries, not ingest sources.
- One-shot operational scripts (`tools/*` ad-hoc fixups) that are tracked in beads and intended to run once. Document them as such.

**Acceptable resolutions** (when the gap is real but the fix is out of scope):

- File a beads ticket explicitly scoped to add the JSONL-emit + rebuild-from-jsonl support, AND mark the PR's docstring / merge-readiness summary with "ingester is DB-only; reconstructibility tracked in <ticket>". The ticket must be P3 or higher, not buried at P4.
- The reviewer does NOT accept "we'll add JSONL later" without a ticket; that's how this rule keeps getting violated.

**Review approach:**

1. Grep the diff for `INSERT INTO` / `INSERT OR IGNORE INTO` and for new `*_ingester.py` / `mine_*.py` / `ingest_*.py` files.
2. For each new write site, walk the call graph backward: is there a corresponding emit-to-JSONL path? If yes, follow forward: does `rebuild-from-jsonl` consume the JSONL?
3. Grep `wyrd/generators/kenning/jsonl/dump.py` and `wyrd/generators/kenning/jsonl/build.py` for the new table names / event `_type` strings.
4. Try to articulate the reconstruction recipe in one sentence. If it requires more than "`rm db && rebuild-from-jsonl`" (e.g., "and also re-fetch the source PDF, then re-run `ingest-X`"), that's a reconstruction-gap finding.

**Recent violations this reviewer would have caught:**

- wyrd-uzoh (PR #276): Briggs personal-names ingester writes `personal_name` + `personal_name_toponym_attestation` rows directly to the DB with no JSONL emission. `rebuild-from-jsonl` doesn't know about either table. Blowing away the DB requires re-running `ingest-briggs-personal-names ~/.wyrd/source-staging/briggs_2024_personal_names_index.txt` against the staged PDF→txt extract — which means staging the PDF, running pdftotext, and re-paying the parse cost. The JSONL gap was caught only by a post-merge operator question, after merge.

