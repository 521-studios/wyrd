# Rebuilding the lexicon from scratch

> **When you need this:** you want to wipe `~/.wyrd/lexicon.db` (L3) and
> replay it from the committed `data/mining/` JSONL (L2), then re-export
> the runtime bundle (L4). Reasons: a taxonomy/schema change that the
> committed seed predates, suspected enrichment corruption, or a
> "rebuild-the-world" after a batch of new sources merged.
>
> If you only added one source and want it folded in, you do NOT need a
> full wipe — just `dump-jsonl` it and `rebuild-from-jsonl` is overkill.
> See `L2_L3_BOUNDARY.md` for the incremental path.

This doc is the procedure manual for a **full** rebuild. It exists
because the rebuild done 2026-05-29 → 2026-05-30 was painful: the
canonical `rebuild-from-jsonl --with-enrichment` command does **not**
restore the whole DB, and the missing pieces were discovered one failing
test at a time across three sessions. Everything below is the recovered,
verified sequence plus the traps that cost the most time.

---

## The one thing to internalise

**`rebuild-from-jsonl` only restores what is carried in L2 JSONL.**
A full wipe silently drops every *L3-only* enrichment layer — no error,
no warning, just a bundle that's ~37% smaller and tests that fail on
morphemes you "know" are there.

The 12-pass enrichment chain (`--with-enrichment`) rebuilds the
*derived columns* (OCR clusters, lemma links, cognates, stratum,
english-shaped, phonological vectors, decompositions, period-forms).
It does **not** run the *mining* passes that populate the empirical,
attestation, forms-variant, and baseline layers. Those live only in the
DB and must be re-run by hand after the rebuild.

| Layer | Carried by L2 → restored by `rebuild-from-jsonl`? | If not, restore with |
|---|---|---|
| `etymon` / citations / descent / glosses / tags | ✅ yes (per-source JSONL) | — |
| `toponym` / `toponym_etymology` / elements | ✅ yes | — |
| `mining_run` audit | ✅ yes | — |
| reflexes (`reflex` / `reflex_etymon`) | ✅ **now** — via synthetic `data/mining/_reflexes.jsonl` (wyrd-ned5, PR #387) | — |
| fantasy morphemes | ✅ **now** — via synthetic `data/mining/_fantasy_morphemes.jsonl` (PR #388) | — |
| curation overrides | ✅ yes — `data/mining/_curation.jsonl` | — |
| **`toponym.country`** | ❌ dropped | `backfill-toponym-country` |
| **phase-2 attestations** (`toponym_attestation`) | ❌ L3-only (boundary doc "deferred") | `ingest-toponym-mentions` over `data/mining/phase2/*.jsonl` |
| **empirical layer** (`wiktionary-empirical` citations) | ❌ L3-only | `mine-wiktextract-corpus` + `cleanup-wiktionary-empirical` |
| **forms-variants** (`etymon_text_match`, D18 spelling pools) | ❌ L3-only ("deferred") | `mine-wiktextract-forms` per slice |
| **empirical baselines / priors** | ❌ L3-only (derived from the above) | `mine-empirical-baselines` + `dump-empirical-priors` |

The reflex and fantasy layers were **L3-only at the time of the May
rebuild** and were the cause of the worst surprises (16 canonical-
morpheme test failures). They now round-trip through L2 synthetic files,
so a fresh rebuild restores them automatically. The remaining four rows
above are still manual — that's Phase 2 below.

---

## Prerequisites

1. **Code is current.** Merge `origin/main` into your working tree first.
   The export reads `runtime_db_export.py` *from the working tree*, not
   from origin — a stale tree silently exports with the old proportions
   logic. (This bit us: a merged Phase-2a export change wasn't in the
   working tree, so proportions came from the static gazetteer instead
   of DB toponyms until the file was `git show origin/main:… >`'d in.)

2. **Back up the live DB.** It is ~3.3 GB. The wipe is destructive.
   ```bash
   cp ~/.wyrd/lexicon.db ~/.wyrd/lexicon.db.bak-prerebuild-$(date +%Y%m%d)
   df -h ~/.wyrd      # confirm headroom; you want several GB free
   ```

3. **Bulk L1 sources cached.** The wiktextract slices (~76 files) live
   **outside** git at `~/.wyrd/sources/` (`.jsonl` / `.jsonl.zst`). The
   rebuild ingests them as L1. On a machine that already has them you're
   fine. On a fresh checkout:
   ```bash
   wyrd kenning lexicon fetch-bulk-sources     # pulls from S3 (needs staging AWS access)
   # or add --fetch-bulk to the rebuild command below to auto-download
   ```
   Slices must be **uncompressed `.jsonl`** for the forms/empirical miners
   (`mine-wiktextract-*` read `.jsonl`, not `.zst`).

4. **Env vars.**
   - `WYRD_LEXICON_DB` — overrides the L3 path (default `~/.wyrd/lexicon.db`).
     Confirm with `wyrd kenning lexicon path`.
   - `WYRD_RUNTIME_DB` — points read-side commands/tests at a specific L4
     bundle `.db`. **Required** for the empirical mine and several
     dashboards (see Phase 2).
   - `~/521Studios/pfsrd2-data` checkout present (only if you ever need to
     re-extract fantasy inputs; normally the L2 `_fantasy_morphemes.jsonl`
     covers it).

5. **From a git worktree the `wyrd` console script does not exist** — it's
   installed only in the main checkout's venv. Invoke as
   `/home/devon/521Studios/wyrd/.venv/bin/python -m wyrd.cli kenning lexicon …`
   with `PYTHONPATH=<worktree>` so worktree code shadows the installed
   package. From the main checkout, plain `.venv/bin/wyrd …` works.

6. **Capture before-reports** so you can diff the rebuild's effect:
   ```bash
   D=data/mining/reports/before-rebuild-$(date +%Y%m%d); mkdir -p "$D"
   for r in report stats enrichment-status era-coverage; do
     wyrd kenning lexicon $r > "$D/$r.txt"; done
   ```

---

## Phase 1 — wipe + replay + enrichment (L2 → L3)

```bash
mkdir -p scratch
wyrd kenning lexicon rebuild-from-jsonl --with-enrichment \
  2>&1 | tee scratch/rebuild-$(date +%Y%m%d).log
# fresh checkout with no ~/.wyrd/sources cache: add --fetch-bulk
```

What it does, in order:

1. **Wipes** the target DB (`--db`, default `~/.wyrd/lexicon.db`).
2. Ingests the L1 wiktextract bulk (skip with `--skip-bulk`; download
   with `--fetch-bulk`).
3. Replays every `data/mining/*.jsonl` L2 file — including the synthetic
   `_reflexes.jsonl`, `_fantasy_morphemes.jsonl`, `_curation.jsonl`.
   Later file order wins on scalar conflicts; glosses/tags union.
4. Runs the 12-pass `run_full_enrichment` chain (because
   `--with-enrichment`): `normalize-ocr → link-lemmas → [curation /
   gloss-suppress / gloss-add / etymon-splits] → decompose →
   cluster-cognates → classify-stratum → derive-english-shaped →
   tag-phonological-vectors → project-period-forms`.

This is the slow part (hours — it's L1 bulk over ~2.4M etymons plus the
enrichment passes). Background it / `tee` it and watch the log.

**Orphan counts are expected, not errors.** Fact-rows that reference a
pruned entity skip their insert and increment `*_orphans` /
`*_orphan_refs` in the build summary. A *surprising* nonzero count means
a typo'd ref; expected post-prune orphans are fine.

> Step-by-step alternative (if you want to run enrichment separately):
> ```bash
> wyrd kenning lexicon rebuild-from-jsonl --jsonl-dir data/mining
> wyrd kenning lexicon enrich --apply         # the full 12-pass chain
> wyrd kenning lexicon enrichment-status       # verify per-pass coverage
> # re-run one pass with --force, e.g.:
> wyrd kenning lexicon classify-stratum --apply --force
> ```
>
> Fresh **empty** DB only: `rebuild-from-jsonl` handles schema creation,
> but if you ever script a build directly, call `init_schema` (creates
> tables) — **not** `migrate_schema`, which only ALTERs existing tables
> and dies with `no such table: etymon` on an empty file.

---

## Phase 2 — restore the L3-only layers (the part that isn't automated)

These are NOT in L2 and NOT run by `--with-enrichment`. Run them in this
order — the dependencies are real (priors read country + attestations;
the empirical mine reads a bundle to know what's unaccounted).

```bash
# 1. Country — dropped by the wipe, prerequisite for empirical baselines.
#    (Skipping this makes mine-empirical-baselines emit 0 cells with
#     skip_country_unknown == every row.)  No --apply flag; writes directly.
wyrd kenning lexicon backfill-toponym-country

# 2. Phase-2 toponym attestations (56 files). Loop, --apply each.
for f in data/mining/phase2/*.jsonl; do
  wyrd kenning lexicon ingest-toponym-mentions --jsonl "$f" --apply
done

# 3. Forms-variants (D18 spelling pools → etymon_text_match). One pass per
#    slice. Read .jsonl (uncompressed) from the bulk cache.
SLICES=(old_english old_norse english welsh irish scottish_gaelic \
        middle_irish old_irish breton cornish manx middle_english old_french)
for s in "${SLICES[@]}"; do
  wyrd kenning lexicon mine-wiktextract-forms ~/.wyrd/sources/wiktextract_${s}.jsonl --apply
done

# 4. Interim bundle export — the empirical miner needs a current bundle to
#    compute "unaccounted" place-name fragments (it calls load_meanings()).
wyrd kenning lexicon export-runtime-db --db ~/.wyrd/lexicon.db \
  --output /tmp/rebuild-interim.db

# 5. Empirical layer. MUST pass --sources-dir ~/.wyrd/sources (see Traps).
#    Point WYRD_RUNTIME_DB at the interim bundle from step 4.
WYRD_RUNTIME_DB=/tmp/rebuild-interim.db \
  wyrd kenning lexicon mine-wiktextract-corpus \
    --culture all --apply --sources-dir ~/.wyrd/sources

# 6. De-pollute the empirical layer (drops modern-english content-word
#    homographs with no historical cluster mate — ~10–12k rows). Dry-run
#    first, then --apply.
wyrd kenning lexicon cleanup-wiktionary-empirical            # dry-run
wyrd kenning lexicon cleanup-wiktionary-empirical --apply

# 7. Empirical baselines + the git-tracked priors sidecar.
wyrd kenning lexicon mine-empirical-baselines --apply
D=data/mining/reports/after-rebuild-$(date +%Y%m%d); mkdir -p "$D"
wyrd kenning lexicon dump-empirical-priors \
  --output "$D/empirical_priors.json" --version after-rebuild-$(date +%Y%m%d)
```

---

## Phase 3 — export L4 (the shipped artifacts)

`export-runtime-db` rebuilds proportions **inline** from L3 + the bundled
`<culture>_place_names.json` corpora, so there is no separate
`rebuild-proportions` step in the L4 path. It is slow (~15–20 min: family
collection over ~61k roots, then the inline proportions rebuild over ~85k
toponyms). `Cross-product cap (256) reached for toponym '…'` lines are
normal progress, not errors. Background it.

```bash
# Full bundle (inspection / S3 publish target):
wyrd kenning lexicon export-runtime-db --db ~/.wyrd/lexicon.db \
  --output /tmp/rebuild-final.db

# The committed dev seed (this is the artifact that ships in git, ~6–7 MB):
wyrd kenning lexicon export-runtime-db --db ~/.wyrd/lexicon.db --dev \
  --output wyrd/generators/kenning/data/seed-runtime.db
```

`--dev` forces canonical filter defaults and **rejects** operator filter
flags (`--min-witnesses`, etc.) so the committed seed is reproducible
across operators. If the seed is much larger than ~6–7 MB you exported
the full bundle by mistake — see `data/seed-runtime.README.md`. (A 79 MB
seed got committed during the May rebuild precisely because a full export
landed where the `--dev` seed belonged.)

> **Do NOT commit the big JSON exports.** `meanings_*.json` is 100 MB+
> (over GitHub's push limit — the very reason D38 moved runtime to
> SQLite-on-S3) and `phase2_candidates/unresolved.jsonl` is ~128 MB. Both
> are gitignored. Only `seed-runtime.db` and the report artifacts under
> `data/mining/reports/` get committed.

---

## Phase 4 — verify

```bash
# 1. Canonical-morpheme gate — the test that catches a dropped reflex layer.
#    (-ton must resolve to OE tūn, not "tone"; -bridge not "bridge (card game)".)
WYRD_RUNTIME_DB=/tmp/rebuild-final.db \
  .venv/bin/python -m pytest tests/test_bundle_canonical_morphemes.py -q

# 2. Full suite against the committed seed.
.venv/bin/python -m pytest tests/ -q -p no:cacheprovider

# 3. Generation smoke, both scoring modes:
wyrd kenning generate english --seed 42
wyrd kenning generate english --scoring-mode vector --tag water --seed 42

# 4. Per-culture drift report (also confirms all 5 cultures generate):
WYRD_RUNTIME_DB=/tmp/rebuild-final.db \
  wyrd kenning lexicon drift-report --culture english --count 500 --format markdown

# 5. Export-side drift vs the committed bundle:
wyrd kenning lexicon diff-bundle      # exit 0 = byte-identical; exit 1 = drift summary

# 6. After-reports + diff vs the before-snapshot from prereqs:
for r in report stats enrichment-status era-coverage; do
  wyrd kenning lexicon $r > "$D/$r.txt"; done
```

Sanity numbers from the 2026-05-30 converged rebuild (for scale, not as
hard assertions — they move as the corpus grows): live DB ~85k toponyms,
empirical citations ~3.7k post-cleanup, attestations ~86k, forms-variant
matches ~1.09M, 333 usable fantasy morphemes; full bundle ~78.7k
meanings (~87 MB); committed dev seed ~1.2k meanings / ~6.7 MB; full
suite 4584 passed / 6 skipped.

---

## Traps that cost the most time (error → cause → fix)

| Symptom | Cause | Fix |
|---|---|---|
| Empirical mine reports `hits=0` for every culture, `etymons_examined=0`, no error | `--sources-dir` defaulted to the repo `sources/` (OCR books — zero wiktextract slices). The miner **silently `continue`s** on a missing slice. | Pass `--sources-dir ~/.wyrd/sources`. PR #385 changed the default to the bulk cache and added a `missing_slices` WARNING (surfaced into `counts['missing_slices']`) so this is loud now — but still check the warning block. |
| `mine-empirical-baselines` emits 0 cells; `skip_country_unknown` == every row | `toponym.country` dropped by the wipe; it is **not** an enrichment pass | Run `backfill-toponym-country` *before* `mine-empirical-baselines`. |
| 16 failures in `test_bundle_canonical_morphemes` (`-ton`→"tone", `-bridge`→"bridge (card game)") | Reflex layer was L3-only and the wipe dropped it; only ~65 of ~1,600 rando reflexes survived | Fixed durably: reflexes now round-trip via `data/mining/_reflexes.jsonl` (wyrd-ned5). A current rebuild restores them in build Pass 3. If they're ever missing again, re-seed via `seed_from_meanings` and re-dump `_reflexes.jsonl`. |
| `test_high_spelling_variety_changes_output_when_pool_present` fails (0 variant pools) | `etymon_text_match` empty — forms-variant layer dropped by the wipe | Run `mine-wiktextract-forms` across all 13 slices (Phase 2 step 3). |
| Proportions look wrong / come from the static gazetteer | Working tree had stale `runtime_db_export.py` (pre-Phase-2a) even though origin was merged | `git show origin/main:wyrd/generators/kenning/lexicon/runtime_db_export.py > <same path>` before exporting; confirm with `grep -c "_load_culture_toponyms"`. |
| `ingest-toponym-mentions` aborts: `--candidates-out … already exists; pass --force` | Leftover candidates file from a prior run; the command refuses to overwrite (no partial writes) | Re-run with `--force` (or remove the file). |
| `language-report` / `rando-port-readiness` raise `UnboundLocalError` when given only `--bundle` | Read-side commands rehydrate from the runtime DB | Set `WYRD_RUNTIME_DB=/tmp/rebuild-final.db` and drop `--bundle` (some don't accept it). |
| Building directly on an empty DB: `sqlite3.OperationalError: no such table: etymon` | Called `migrate_schema` (ALTER-only) on a file with no tables | Call `init_schema` first. `rebuild-from-jsonl` already does this. |
| `seed-runtime.db` committed at ~79 MB | A full (non-`--dev`) export was written to the seed path before the L3-only layers were restored | Always export the committed seed with `--dev`, and only after Phase 2 is complete. |

---

## Other gotchas

- **Two exports, by design.** The empirical miner reads a bundle to find
  unaccounted fragments, so the sequence is: interim export → empirical
  mine + cleanup + baselines → final export. Don't try to mine empirical
  before any export exists.
- **`cleanup-wiktionary-empirical` is mandatory after the mine.** The POS
  filter only drops function words; modern-english content-word
  homographs still come in and must be cleaned.
- **`tag-phonological-vectors` is incremental (NULL-only) by default.**
  The enrichment chain runs it that way. To recompute the whole corpus
  after a vector-algorithm change, run the standalone command with
  `--force`.
- **Idempotency is what makes recovery safe.** Seed/upsert helpers use
  `INSERT OR IGNORE` / `ON CONFLICT`, so re-running a Phase-2 step only
  adds missing rows. You can interrupt and resume the long miners; they
  commit per-resolution.
- **The shell working directory resets between Bash calls** in agent
  sessions — use absolute paths or `cd` every command.
- **Ollama runs one inference at a time** (`10.5.2.31`); parallel mining
  batches against it don't speed up. Not relevant to a pure replay, only
  if you re-mine LLM sources.

---

## What changed since the painful rebuild (history)

The 2026-05-29/30 rebuild surfaced these and they're now fixed in `main`:

- **PR #385 (wyrd-34kt)** — `mine-wiktextract-corpus` default
  `--sources-dir` → `~/.wyrd/sources` + loud `missing_slices` warning.
- **PR #386 (wyrd-bo01.3)** — proportions corpus from DB toponyms
  (Phase 2a); the export reads the DB instead of the static gazetteer.
- **PR #387 (wyrd-ned5)** — reflex layer round-trips through L2 via
  `data/mining/_reflexes.jsonl`, so rebuilds no longer drop it.
- **PR #388** — fantasy morphemes round-trip via
  `data/mining/_fantasy_morphemes.jsonl`; converged seed re-emit.
- **PR #389** — `ingest-report-snapshot` / `report-snapshot-diff` so the
  before/after report deltas are SQL-queryable, not just files.

The four remaining manual layers in Phase 2 (country, attestations,
forms-variants, empirical+baselines) are still L3-only — folding
`toponym_attestation` and `etymon_text_match` into L2 is the open
follow-up tracked in `L2_L3_BOUNDARY.md` ("Deferred (planned L2)").

---

## See also

- `L2_L3_BOUNDARY.md` — the authoritative L2/L3 column-by-column map and
  the bulk-sources / S3 workflow.
- `INGESTION.md` — adding a *new* source (the incremental path).
- `DECISIONS.md` D38 — why L4 is SQLite-on-S3; D22/D21 — why enrichment
  is reversible and never destroys mining evidence.
- `data/seed-runtime.README.md` — what the committed dev seed contains
  and when to regenerate it.
