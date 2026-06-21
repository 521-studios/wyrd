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

The enrichment chain (`--with-enrichment`) rebuilds the
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
| `etymon_variant` (~5.8M form rows from the wiktextract slices) | ✅ yes — via the **bulk L1 ingest** rebuild-from-jsonl runs by default (skipped only under `--skip-bulk`) | — |
| reflexes (`reflex` / `reflex_etymon`) | ✅ **now** — via synthetic `data/mining/_reflexes.jsonl` (wyrd-ned5, PR #387). **Orphan reflexes too** (generation surfaces with no etymon link) since wyrd-br5o: the dump LEFT-JOINs so the *full* reflex table round-trips, not just the linked subset | — |
| fantasy morphemes | ✅ **now** — via synthetic `data/mining/_fantasy_morphemes.jsonl` (PR #388). **Their referenced (uncited) etymons round-trip too** since wyrd-ruvk: the dump emits those etymons as canonical-state rows in the same file, else a rebuild resolves only ~8 of ~333 fantasy etymon FKs | — |
| curation overrides | ✅ yes — `data/mining/_curation.jsonl` | — |
| collapse ledger (form-of/variant folds, wyrd-y651) | ✅ yes — `data/mining/_collapses.jsonl`, replayed by `run_full_enrichment`'s curation slot (`apply_collapses`) | — |
| element-gloss backfill (`reflex_etymon` links for unglossed generation surfaces, wyrd-u9k6) | ✅ yes — `data/mining/_element_glosses.jsonl` (deterministic consensus) + `_element_gloss_adjudications.jsonl` (LLM picks), replayed by `run_full_enrichment`'s element-gloss pass (`apply_element_glosses`) | — |
| LLM tag backfill (`etymon_tag` rows for untagged-but-glossed generation morphemes, wyrd-xz3g) | ✅ yes — `data/mining/_tags.jsonl` (gemma4:26b decisions classifying a gloss into the 44-existing + `kinship` controlled vocab; PAID), replayed by `run_full_enrichment`'s curation slot (`collect_tags` → `apply_tag_additions`, INSERT OR IGNORE) so the LLM pass never re-runs | — |
| pronunciation IPA backfill (`etymon.pronunciation_ipa`, wyrd-vm8t) | ✅ yes — **two tiers**: (1) the deterministic G2P fill for OE/ON/welsh/celtic is re-derived for free by `run_full_enrichment`'s `derive_pronunciation_ipa` pass (no jsonl — same as english-shaped/stratum); (2) the LLM tier for the no-G2P-table languages (Goidelic/Romance/Middle English/Breton) replays the high+medium-confidence rows of `data/mining/_pronunciation.jsonl` through the same pass (`collect_pronunciation` → `llm_state`, gaps only) | — |
| **`toponym.country`** | ❌ dropped | `backfill-toponym-country` |
| **phase-2 attestations** (`toponym_attestation`) | ❌ L3-only (boundary doc "deferred") | `ingest-toponym-mentions` over `data/mining/phase2/*.jsonl` |
| **empirical layer** (`wiktionary-empirical` citations) | ✅ **now** — L2-replay via `data/mining/wiktionary-empirical.jsonl` (wyrd-x33t). Was an L3-only re-mine, but `mine-wiktextract-corpus` is non-convergent (~1,870–2,676 vs accumulated ~3,682), and the shortfall regressed the worth gate + breton realism (wyrd-ruvk); so the citations now round-trip | — |
| **forms-variants** (`etymon_text_match`, D18 spelling pools) | ❌ L3-only ("deferred") | `mine-wiktextract-forms` per slice |
| **empirical baselines / priors** | ❌ L3-only (derived from the above) | `mine-empirical-baselines` + `dump-empirical-priors` |
| **genitive split prior** (`genitive_split_prior`, wyrd-aicu.9 + .9.1) | ❌ L3-only (derived from toponym_etymology + historical_form + reflex cognate clusters + toponym_attestation) | `mine-genitive-priors` + `dump-genitive-priors` |

The reflex and fantasy layers were **L3-only at the time of the May
rebuild** and were the cause of the worst surprises (16 canonical-
morpheme test failures). They now round-trip through L2 synthetic files,
so a fresh rebuild restores them automatically. The remaining four rows
above are still manual — that's Phase 2 below.

> **wyrd-br5o (2026-06-18):** wyrd-ned5's `_reflexes.jsonl` dump carried
> *linked* reflexes only (INNER JOIN). The **orphan** reflexes — generation
> surfaces with no etymon link, originally seeded by `seed_from_meanings`
> from `data/meanings.json` — were never in L2. When d90t (#357) deleted
> `meanings.json` (it was both the runtime artifact *and* the authoring
> seed), the orphan layer (~16k surfaces) had no L2 source: a clean rebuild
> re-derived only ~12k via enrichment (`derive_positions`), silently losing
> ~4k — which cost ~8k promoted meanings (worth → modern "worth", fantasy
> 333→8, english realism). Fixed by making the dump `LEFT JOIN` so the *full*
> reflex table round-trips; `seed_from_meanings` stays retired. If reflexes
> ever look thin again, re-dump `_reflexes.jsonl` from a known-good DB
> (`dump_reflexes_to_file`) — do **not** resurrect `seed_from_meanings`.

### Deferred / empty layers (no data to lose today, but know the rule)

A code audit (the same pass that confirmed the table above is complete)
found three more tables a wipe touches. None holds data on the current
DB, so they aren't active rebuild steps — but if you ever populate them,
here's the obligation:

- **`meaning_synset`** (semantic-equivalence catalog, D28) — seeded FREE
  from the committed `data/seed/meaning_synsets.json` via `lexicon synsets
  seed`. `rebuild-from-jsonl` does **not** run it. If you populate it,
  run `synsets seed` after the rebuild and promote it to a documented
  rebuild step.
- **`etymon_meaning_synset`** (LLM Phase-2 classification, D28) — empty /
  deferred. If activated it is **paid** (LLM), so per this doc's
  principle it must round-trip through L2 (a synthetic JSONL like
  `_reflexes.jsonl`), **not** become a re-mine step.
- **`personal_name` / `personal_name_toponym_attestation`** (Briggs) —
  the tables no longer exist on the live DB (re-routed into the `etymon`
  schema by PR #380); nothing to restore.
- **canonicalization graph** (`canonical_morpheme` / `canonical_place` /
  `canonical_sense` / `canonical_label` + the `canonical_*_id` binds on
  `etymon` / `etymon_gloss` / `toponym` / `toponym_etymology`, D49/D50,
  wyrd-u6fn) — the SCHEMA ships in alembic migration `0019` (auto, like every
  table). The DATA is **projected** from the `data/mining/canonicalization/`
  L2 assertion streams by the **`project-canonical`** pass (wyrd-u6fn.3,
  D50.6), which now runs as the terminal L3 derivation inside
  `run_full_enrichment` — so `rebuild-from-jsonl --with-enrichment` populates
  it **for free** (deterministic + idempotent, same L2 → byte-identical L3;
  never a paid re-mine). (The pass runs only when a `canonicalization_dir` is
  supplied, which `rebuild-from-jsonl` passes; a bare `lexicon enrich` leaves
  the canonical graph untouched — the wipe-and-rebuild path is what matters
  here.) By default it **folds today's deterministic identity clustering**
  (`merged_into_id` OCR variants + `lemma_id` inflections) into the graph
  (wyrd-u6fn.4), so the rebuilt graph **reproduces today's identity clustering**
  from the columns the deterministic passes already produce — no committed
  snapshot. (`cognate_id` is NOT folded — it's relational, D50.2, re-derived
  from `etymon_descent`.) **Additive**: it populates the canonical graph but does
  **not** yet migrate the legacy `merged_into_id` / `cognate_id` / `lemma_id`
  **readers** onto the canonical FK join — that reader cutover is **wyrd-b2mf**.
  Reverse with `clear-enrichment --stage=canonical`.
- **cognate-descent edges** (`etymon_descent` rows under
  `source_id='cognate-descent-uplift'`, D50 Family B, wyrd-zrce.1) — mined cognate
  descents that place unclustered breakdown morphemes into existing clusters live
  ONLY in the `data/mining/canonicalization/_assert_descends_from.jsonl` L2 stream;
  they are **projected** into `etymon_descent` by the **`project-descent`** pass,
  which runs EARLY in `run_full_enrichment` (before `cluster-cognates`, so the new
  edges feed the `cognate_id` rollup in the same run). Deterministic + idempotent
  (clears its own `source_id` then re-inserts from the live assertion set; Wiktionary
  edges untouched). Like `project-canonical`, runs only when a `canonicalization_dir`
  is supplied (which `rebuild-from-jsonl` passes). Reverse by appending a `retract`
  (append-only, D21). The LLM half of these edges (wyrd-zrce.2) carries a verdict
  cache, `data/mining/_cognate_descent_audit.jsonl` — like the other `_*_audit.jsonl`
  ledgers it is in `build.REPLAY_EXCLUDED_LEDGERS` (NOT replayed as a source file at
  rebuild; it only resumes the `mine-cognate-descents-llm` pass and re-derives edges
  threshold-independently without re-calling the LLM). Its endpoints are natural keys
  (wyrd-s964), so it survives a rebuild's id reassignment.
- **implied reflexes** (`mine-implied-reflexes`, wyrd-65jh) — residual attribution
  of a toponym's modern name (anchor the KNOWN-reflex spans, attribute the one
  contiguous residual span to the remaining element; `Houghton − ton(tūn) ⇒ hough`
  is an implied reflex of `hōh`). NOT a wipe-restore step: `--apply` writes BOTH the
  D50 `canonical-label@modern-english` assertion + its `mint-canonical` hub
  (`data/mining/canonicalization/_assert_canonical_label.jsonl` +
  `_canonical_nodes.jsonl`, covered by the `canonicalization-graph` layer — the mint
  is what lets a singleton residual morpheme's label project) AND a `reflex` /
  `reflex_etymon` projection, then RE-DUMPS `_reflexes.jsonl` (the `reflexes` layer)
  so the reflex round-trips on rebuild. `surface_in_modern` is NOT written here
  (wyrd-ujyo owns its derivation). Its outputs all ride existing layers; the miner
  itself is never re-run on rebuild.

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
   installed only in the main checkout's venv. Invoke the MAIN checkout's
   console script with `PYTHONPATH=<worktree>` so worktree code shadows the
   installed package:
   `PYTHONPATH=<worktree> /home/devon/521Studios/wyrd/.venv/bin/wyrd kenning lexicon …`.
   From the main checkout, plain `.venv/bin/wyrd …` works.
   Do NOT use `python -m wyrd.cli …` — `wyrd/cli.py` has no
   `if __name__ == "__main__"` guard, so `-m` imports the module, runs
   nothing, and exits 0: a silent no-op that looks like success.

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
3. Replays the conforming `data/mining/*.jsonl` L2 files — the curated
   sources plus the synthetic `_reflexes.jsonl`, `_fantasy_morphemes.jsonl`,
   `_curation.jsonl`, `_collapses.jsonl`, `_tags.jsonl`, `_merge_audit.jsonl`.
   The replay-excluded ledgers in `build.REPLAY_EXCLUDED_LEDGERS` (the audit
   verdict logs + `_element_glosses` / `_element_gloss_adjudications` /
   `_pronunciation` / `_modern_reflexes`) are skipped — they don't conform to
   the replay schema; their effects round-trip through the conforming ledgers
   above or are re-applied by their own enrichment/import pass (wyrd-5qg7).
   Later file order wins on scalar conflicts; glosses/tags union.
4. Runs the `run_full_enrichment` chain (because
   `--with-enrichment`): `normalize-ocr → link-lemmas → [curation /
   gloss-suppress / gloss-add / etymon-splits / collapses / element-glosses /
   tag-additions] → flatten-merge-chains → decompose →
   cluster-cognates → classify-stratum → derive-english-shaped →
   derive-pronunciation-ipa → tag-phonological-vectors → project-period-forms →
   derive-surface-in-modern`. `derive-surface-in-modern` (wyrd-ujyo) suffix-anchors
   each binary breakdown against the toponym's modern name to fill
   `toponym_etymology_element.surface_in_modern`.
   `flatten-merge-chains` (wyrd-lpxq) runs when a curation slot ran — it collapses
   any multi-hop `merged_into_id` chain a curated merge built to a terminal winner,
   before the L3 derivations consume the graph. When the canonicalization streams
   are supplied (which `rebuild-from-jsonl` does), two more **conditional** passes
   run: **`project-descent`** right before `cluster-cognates` (so mined
   cognate-descent edges feed the `cognate_id` rollup in the same run) and the
   terminal **`project-canonical`**.

This is the slow part (hours — it's L1 bulk over ~2.4M etymons plus the
enrichment passes). Background it / `tee` it and watch the log.

**Orphan counts are expected, not errors.** Fact-rows that reference a
pruned entity skip their insert and increment `*_orphans` /
`*_orphan_refs` in the build summary. A *surprising* nonzero count means
a typo'd ref; expected post-prune orphans are fine.

> Step-by-step alternative (if you want to run enrichment separately):
> ```bash
> wyrd kenning lexicon rebuild-from-jsonl --jsonl-dir data/mining
> wyrd kenning lexicon enrich --apply         # the base enrichment chain (no canonicalization projections)
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

# 4–6. Empirical layer — NO LONGER a rebuild step (wyrd-x33t). The
#    wiktionary-empirical citations round-trip via L2
#    (data/mining/wiktionary-empirical.jsonl) and are replayed by
#    rebuild-from-jsonl in Phase 1. Do NOT re-run mine-wiktextract-corpus /
#    cleanup-wiktionary-empirical here: the mine is non-convergent and would
#    add citations ON TOP of the replayed set, re-breaking reproducibility
#    (worth gate + breton realism, wyrd-ruvk). Its reflexes already ride
#    _reflexes.jsonl (wyrd-ned5/br5o).
#    [Operators minting NEW empirical: run mine-wiktextract-corpus (needs an
#     interim bundle export + --sources-dir ~/.wyrd/sources) then
#     cleanup-wiktionary-empirical, then RE-DUMP the jsonl via dump-jsonl.]

# 7. Empirical baselines + the git-tracked priors sidecar. Derives from the
#    now-replayed empirical citations + country + attestations.
wyrd kenning lexicon mine-empirical-baselines --apply
D=data/mining/reports/after-rebuild-$(date +%Y%m%d); mkdir -p "$D"
wyrd kenning lexicon dump-empirical-priors \
  --output "$D/empirical_priors.json" --version after-rebuild-$(date +%Y%m%d)

# 8. Modern-reflex curation (wyrd-vewk) — land the curated modern-English
#    reflexes so OE/ON/OF/ME morphemes' era-grid modern stage populates
#    (-ham -> home, holmr -> holm/holme, ...). Free + deterministic; replays
#    data/mining/_modern_reflexes.jsonl. Run AFTER enrich/cluster-cognates so
#    each new reflex inherits its morpheme's cognate cluster (the importer sets
#    the reflex's cognate_id from the already-clustered morpheme).
wyrd kenning lexicon import-modern-reflexes --apply

# 9. Genitive-s split prior (wyrd-aicu.9 + .9.1) — per-suffix town/stone split
#    counts for the decomposition matcher's homograph disambiguation. Free +
#    deterministic; reads toponym_etymology (+ historical_form) + reflex cognate
#    clusters + toponym_attestation (the subordinate -es- marker, step 2). Run
#    AFTER cluster-cognates (needs the cognate_id classes), step 2 (attestations)
#    AND step 8 (so curated modern reflexes carry their cognate cluster). dump
#    writes the git-tracked sidecar (the operator-visible diff surface).
wyrd kenning lexicon mine-genitive-priors --apply
wyrd kenning lexicon dump-genitive-priors \
    --output data/mining/_genitive_priors.json
```

---

## Phase 3 — export L4 (the shipped artifacts)

`export-runtime-db` rebuilds proportions **inline** from L3 + the bundled
`<culture>_place_names.json` corpora, so there is no separate
`rebuild-proportions` step in the L4 path. ("Proportions" here is the
**vector** scorer's empirical input — slot structures + tag co-occurrence —
baked into the L4 DB; the proportions *scoring* mode is retired, D36.) Takes
~10 min wall (measured 9m37s / ~2 GB peak RSS on 2026-06-11: family
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

# 3. Generation smoke (vector is the only scoring path):
wyrd kenning generate english --seed 42
wyrd kenning generate english --tag water --seed 42

# 4. Realism gate (also confirms all 5 cultures generate):
WYRD_RUNTIME_DB=/tmp/rebuild-final.db \
  .venv/bin/python -m pytest tests/test_kenning_realism_absolute.py -q

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
| 16 failures in `test_bundle_canonical_morphemes` (`-ton`→"tone", `-bridge`→"bridge (card game)") | Reflex layer was L3-only and the wipe dropped it; only ~65 of ~1,600 rando reflexes survived | Fixed durably: reflexes now round-trip via `data/mining/_reflexes.jsonl` (wyrd-ned5 for linked, wyrd-br5o for orphan/generation-surface). A current rebuild restores them in build Pass 3. If they're ever missing again, re-dump `_reflexes.jsonl` from a known-good DB via `dump_reflexes_to_file` — do **not** resurrect `seed_from_meanings` (retired; its `meanings.json` seed was deleted by d90t). |
| `test_high_spelling_variety_changes_output_when_pool_present` fails (0 variant pools) | `etymon_text_match` empty — forms-variant layer dropped by the wipe | Run `mine-wiktextract-forms` across all 13 slices (Phase 2 step 3). |
| Proportions look wrong / come from the static gazetteer | Working tree had stale `runtime_db_export.py` (pre-Phase-2a) even though origin was merged | `git show origin/main:wyrd/generators/kenning/lexicon/runtime_db_export.py > <same path>` before exporting; confirm with `grep -c "_load_culture_toponyms"`. |
| `ingest-toponym-mentions` aborts: `--candidates-out … already exists; pass --force` | Leftover candidates file from a prior run; the command refuses to overwrite (no partial writes) | Re-run with `--force` (or remove the file). |
| `language-report` / `rando-port-readiness` raise `UnboundLocalError` when given only `--bundle` | Read-side commands rehydrate from the runtime DB | Set `WYRD_RUNTIME_DB=/tmp/rebuild-final.db` and drop `--bundle` (some don't accept it). |
| Building directly on an empty DB: `sqlite3.OperationalError: no such table: etymon` | Called `migrate_schema` (ALTER-only) on a file with no tables | Call `init_schema` first. `rebuild-from-jsonl` already does this. |
| `seed-runtime.db` committed at ~79 MB | A full (non-`--dev`) export was written to the seed path before the L3-only layers were restored | Always export the committed seed with `--dev`, and only after Phase 2 is complete. |

---

## Other gotchas

- **Empirical mine/cleanup are OPERATOR-ONLY now, not rebuild steps
  (wyrd-x33t).** The `wiktionary-empirical` citations round-trip via
  `data/mining/wiktionary-empirical.jsonl` (replayed in Phase 1), so a rebuild
  does NOT run `mine-wiktextract-corpus` / `cleanup-wiktionary-empirical`. The
  two notes below apply only when an **operator mints NEW empirical** (then
  re-dumps the jsonl), never during a rebuild:
  - *Two exports, by design.* The empirical miner reads a bundle to find
    unaccounted fragments, so the mint sequence is: interim export → empirical
    mine + cleanup + baselines → re-dump. Don't mine empirical before any
    export exists.
  - *`cleanup-wiktionary-empirical` is mandatory after the mine.* The POS
    filter only drops function words; modern-english content-word homographs
    still come in and must be cleaned before re-dumping.
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
- **PR #673 (wyrd-br5o)** — the `_reflexes.jsonl` dump round-trips the
  ORPHAN reflexes too (was linked-only), recovering ~16k generation
  surfaces a clean rebuild had silently dropped after d90t deleted the
  `meanings.json` seed.
- **PR #674 (wyrd-ruvk)** — `_fantasy_morphemes.jsonl` now also carries the
  morphemes' referenced (uncited) etymons; without them a rebuild resolved
  only ~8 of ~333 fantasy etymon FKs. Loser refs follow `merged_into_id` to
  the winner so no tombstone resurfaces.

The four remaining manual layers in Phase 2 (country, attestations,
forms-variants, empirical+baselines) are still L3-only — folding
`toponym_attestation` and `etymon_text_match` into L2 is the open
follow-up tracked in `L2_L3_BOUNDARY.md` ("Deferred (planned L2)").

---

## Keeping this runbook honest (enforcement)

This runbook drifts out of date the instant someone adds a new mined
layer and forgets to document its restore. Three things keep it complete:

1. **`data/mining/_rebuild_layers.json`** — the structured source of
   truth. Every data-population CLI (`mine-*` / `ingest-*` / `backfill-*`
   / `cleanup-*`) is categorized as a `rebuild_step_commands` entry (a
   free step you must run after a wipe) or a `non_rebuild_commands` entry
   (with a one-line reason — paid mining, source-specific ingester whose
   output round-trips through L2, etc.). The `layers` array records each
   layer a wipe affects and how it's restored.

2. **`tests/test_kenning_rebuild_runbook.py`** — a CI gate (no DB, runs in
   ms). It walks the live Click group and **fails** if a new
   data-population command isn't categorized in the manifest, if the
   manifest references a command that no longer exists, or if a
   rebuild-step command / layer token isn't written into this file. So a
   newly-added free-mining step can't merge until it's documented here.

3. **`rebuild-runbook-currency-reviewer`** (`AGENT-REVIEWERS.md`) — judges
   what the test can't: whether a command landed in the *right* bucket,
   whether its reason is accurate (paid mining must never be a rebuild
   step), and whether it's placed in the correct Phase-2 order.

The division of labour: the repo-root **`db-reconstructibility-reviewer`**
asks *"can this data be recovered at all without re-paying for mining?"*
(→ L2 round-trip or free-rebuildable). This runbook + its reviewer ask the
follow-on: *"is that recovery automatic, or at least written down here in
order?"* When you add a layer, satisfy both.

**So: adding a new mined/ingested layer is a four-file change** — the
ingester, a `_rebuild_layers.json` entry, a REBUILD.md restore step (if
it's a free rebuild step), and (if it's a new table moving the L2/L3 line)
the `L2_L3_BOUNDARY.md` map.

## See also

- `L2_L3_BOUNDARY.md` — the authoritative L2/L3 column-by-column map and
  the bulk-sources / S3 workflow.
- `INGESTION.md` — adding a *new* source (the incremental path).
- `DECISIONS.md` D38 — why L4 is SQLite-on-S3; D22/D21 — why enrichment
  is reversible and never destroys mining evidence.
- `data/seed-runtime.README.md` — what the committed dev seed contains
  and when to regenerate it.
