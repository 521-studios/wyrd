# `seed-runtime.db`

A committed snapshot of the kenning L4 runtime DB at ~24MB (grows with the
corpus as mining campaigns enrich existing meanings, and with the
`DEV_TOP_N_PER_CULTURE` per-culture cap — raised to 350 in the wyrd-x5y4.5
case-fold reseed to keep the --dev subset in its 1000–1500 meaning range),
large enough for unit tests + local dev to exercise the
generator surface against, small enough to ship in git. Built from the live L3
lexicon via `lexicon export-runtime-db --dev` (see D38 in `DECISIONS.md` for the
L4 architecture).

## What's in it

A subset of the production L4 DB:

* **Meanings** — only the words whose `modern_usage` appears in the per-culture
  top-N (default N=200) usage / single_usage keep-set. Roughly 1,000-1,500 rows.
* **Fantasy morphemes** — the full set (~240 rows; the live corpus is small
  enough that subsetting doesn't help).
* **Canonical decompositions** — the full set (~11,500 rows; the lookup is
  per-toponym-name and we don't want the seed to lose coverage of common names).
* **Proportions** — top-N by weight per culture for `usages` /
  `single_usages`; `structures` / `tag_marginal` / `tag_cooccurrence` kept in
  full. 5 cultures (english / scottish / welsh / irish / breton).
* **`bundle_metadata`** — `built_at` pinned to `1970-01-01T00:00:00Z`,
  `source_lexicon_db` pinned to `DEV_SOURCE_LEXICON` so the file is byte-equal
  across operators / CI / machine moves.

## Regenerating

```bash
wyrd kenning lexicon export-runtime-db --dev \
  --output wyrd/generators/kenning/data/seed-runtime.db
```

The `--dev` flag forces every upstream filter knob to its canonical default
(operator-supplied `--min-witnesses` / `--lang-threshold` / `--include-rando` /
etc. are rejected with a `UsageError`). This is intentional — the committed
seed has to be reproducible across operators, and silently honoring a custom
filter would produce a seed that differs from the one in the repo without the
operator noticing.

**When to regenerate:**

* `SCHEMA_VERSION` bump in `runtime_db_export.py` (incompatible L4 schema
  change). The `test_committed_seed_carries_current_schema_version` regression
  catches this — when it red-lights in CI, regenerate locally and commit.
* `DEV_TOP_N_PER_CULTURE` change in `runtime_db_export.py`.
* New mining data lands and you want the seed to reflect a fresher slice of
  the live corpus. Not strictly required — the seed is a dev convenience, not
  a contract.

## What it is NOT

* **Not the production L4 DB.** The production DB is built without `--dev`
  (full corpus, real timestamp, real source path) and pushed to S3 — see
  `bin/publish-runtime-db.sh` (PR 3 of wyrd-d90t) and the loader (PR 4).
* **Not a fixture lexicon.** It's an L4 (runtime artifact), not an L3
  (lexicon DB). The L3 source lives at `~/.wyrd/lexicon.db` and is never
  committed.
