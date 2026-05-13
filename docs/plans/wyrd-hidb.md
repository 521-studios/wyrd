# wyrd-hidb — One-command rebuild: L1 wiktextract + full L3 in `rebuild-from-jsonl`

PR B of the wyrd-0vj3-then-hidb chain. See `docs/plans/wyrd-0vj3.md`
for the storage strategy this depends on.

## Goal

`lexicon rebuild-from-jsonl --with-enrichment` becomes the **one
command** that goes from L2 JSONL + L1 bulk cache to a fully-derived
DB that `lexicon export-meanings` can serialize to a byte-identical
bundle. No operator-driven multi-step pipelines, no "operators
still run those manually post-rebuild" caveats.

## What's missing today

`rebuild-from-jsonl --with-enrichment` runs:

1. `init_schema` (wipe + fresh tables)
2. `build_from_jsonl` (replay every `data/mining/*.jsonl`)
3. `run_ocr_lemma_enrichment`:
   - `cluster_ocr_variants` (normalize-ocr)
   - `link_lemmas`
   - `apply_curation_overrides` (when curation state present)

It does NOT run:

- **L1 wiktextract ingest** — `ingest_wiktextract_path` per slice
  (~50 slices, ~4.6 GB raw, ~340-460 MB zstd in `~/.wyrd/sources/`
  per the wyrd-0vj3 manifest). Without this, the rebuilt DB has 0
  wiktionary-empirical citations; the bundle shrinks ~76%.
- **L3 derivations**:
  - `decompose` — matcher-derived toponym decompositions
  - `cluster-cognates` — etymon_descent graph → cognate_id
  - `classify-stratum` — within-language stratum tags
  - `derive-english-shaped` — non-Latin-script → english_shaped
  - `project-period-forms` — wyrd-unuo Phase 3.3

Discovered the hard way in the wyrd-j2bv Latin lift: rebuild-from-
jsonl from L2-only collapsed the bundle from 6,173 subjects to
1,452 — visible immediately in `lexicon rando-port-readiness`
showing 0 empirical attestation across all languages.

## Architecture

```
lexicon rebuild-from-jsonl --with-enrichment
│
├── init_schema                     (current)
├── ingest_l1_bulk                  (NEW — Phase 1)
│     └── fetch + verify slices via bulk_sources manifest
│         then ingest_wiktextract_path per slice
├── build_from_jsonl                (current — replays L2)
└── run_full_enrichment             (RENAMED + EXPANDED)
      ├── cluster_ocr_variants      (current — was run_ocr_lemma)
      ├── link_lemmas               (current)
      ├── apply_curation_overrides  (current)
      ├── decompose                 (NEW — L3 wire-up)
      ├── cluster_cognates          (NEW)
      ├── classify_stratum          (NEW)
      ├── derive_english_shaped     (NEW)
      └── project_period_forms      (NEW)
```

## Canonical order (and why)

The L3 chain has real ordering dependencies. The order below comes
from the ticket + tracing today's CLI commands' implicit
prerequisites:

1. **`cluster_ocr_variants`** — tombstones OCR-variant duplicates so
   later passes don't waste work on dead rows.
2. **`link_lemmas`** — inflection → canonical-lemma linkage. Must
   follow OCR (lemma candidates can't be tombstoned variants).
3. **`apply_curation_overrides`** — operator decisions overlay both
   prior passes (`etymon_curation` rows in `_curation.jsonl`).
4. **`decompose`** — toponym element matching against the etymon
   inventory. Needs OCR + lemma + curation to be settled (else
   matches against tombstoned/wrong rows).
5. **`cluster_cognates`** — etymon_descent graph traversal. Needs
   lemma assignment (cognates flow through canonical lemmas, not
   inflections).
6. **`classify_stratum`** — needs etymon inventory + descent edges.
7. **`derive_english_shaped`** — non-Latin-script transliteration;
   independent of stratum but logically after cognates (stable
   canonical_form).
8. **`project_period_forms`** — wyrd-unuo Phase 3.3; depends on
   stratum + cognate assignments for period bucketing.

If any of these turn out to be order-independent during testing
(e.g., english-shaped could move earlier), tighten the plan based
on byte-diff results.

## Components

### 1. `bulk_sources.ingest_all_slices(db, *, fetch=False, apply=True)`

New helper in `wyrd/generators/kenning/bulk_sources.py`. For each
slice in the manifest:

1. If `fetch=True` and the local cache is missing/mismatched,
   download from S3 (via `fetch_missing_slices`).
2. Else if local cache missing → raise a clear error pointing at
   `lexicon fetch-bulk-sources`.
3. Call `ingest_wiktextract_path(db, local_path, apply=apply)`.

Returns a counts dict: per-slice `lines_read`, `entries_parsed`,
totals.

### 2. `enrichment.run_full_enrichment(db, *, apply=True, curation_state=None) -> dict`

Extends `run_ocr_lemma_enrichment` with the five new passes. New
function name (keep the old one around as a deprecated alias for
one release cycle? or just call the new function from the existing
name? — recommend: rename + redirect with deprecation no-op).

Each new pass:

- Takes `db: LexiconDB` (or `conn`, depending on the existing
  signature)
- Runs idempotently (re-running on a populated DB is a no-op)
- Returns counts dict for the consolidated report

Counts dict shape:

```python
{
    "order": [...],
    "applied": True,
    "ocr": {...},          # existing
    "lemmas": {...},       # existing
    "curation": {...},     # existing
    "decompose": {...},
    "cognates": {...},
    "stratum": {...},
    "english_shaped": {...},
    "period_forms": {...},
}
```

`format_enrichment_run` extends to render the new sections.

### 3. `rebuild-from-jsonl` CLI changes

New flags on `lexicon rebuild-from-jsonl`:

```
--fetch-bulk   Download missing/mismatched bulk slices from S3
               before ingest. Without this, missing slices fail
               loud with a hint to run lexicon fetch-bulk-sources
               separately.
--skip-bulk    Skip L1 wiktextract ingest entirely. Useful for
               L2-only operator workflows (debugging a single
               source file, fast iteration on curation).
```

`--with-enrichment` semantics: now runs the FULL L3 chain. To get
the old "OCR + lemmas only" behavior, add `--enrichment-stop-at
<pass>` (or similar; bikeshed during impl).

### 4. Round-trip integration test

New `tests/test_kenning_rebuild_round_trip.py`:

1. Read the canonical bundle at `wyrd/generators/kenning/data/meanings.json`.
2. Dump current DB to a tmp manifest.
3. Wipe DB.
4. Run `rebuild-from-jsonl --with-enrichment --fetch-bulk` against
   the tmp manifest.
5. `export-meanings` to a tmp file.
6. Byte-diff against the canonical bundle.

Expected: zero diff (or documented diff if any pass is
non-deterministic — we'll discover that in the first run).

This test gates the round-trip invariant going forward. Without it,
any pass that drifts the bundle silently (e.g., new randomness in a
derivation) is invisible until the next operator full-rebuild.

### 5. Update `wyrd/generators/kenning/L2_L3_BOUNDARY.md`

Drop the "operators still run those manually post-rebuild" line.
Replace with: "One command — `lexicon rebuild-from-jsonl
--with-enrichment` — produces a byte-identical bundle."

## Determinism risks (open during impl)

Each new L3 pass needs verification that it's deterministic. Likely
hot spots:

- **`cluster_cognates`** — graph traversal order may depend on dict
  iteration; should be sorted by etymon_id.
- **`decompose`** — matcher emits multiple breakdown candidates;
  one is canonical. Selection criterion must be deterministic.
- **`derive_english_shaped`** — transliteration is rule-based,
  should be fine.
- **`classify_stratum`** — rule-based, should be fine.
- **`project_period_forms`** — bucketing depends on attestation
  dates; deterministic if attestation order is deterministic
  (already guaranteed by `ORDER BY toponym_id, id` in
  `build_from_jsonl`).

If the round-trip diff is non-empty, fix the offending pass before
calling wyrd-hidb done.

## Phasing

Single PR. The pieces are tightly coupled — splitting the L3 wire-up
across multiple PRs would leave the rebuild chain in a broken
intermediate state.

Order of work within the PR:

1. `ingest_all_slices` helper + tests
2. Wire L1 ingest into `rebuild-from-jsonl`
3. Run the round-trip test as a smoke (expect failure — L3
   passes not yet wired)
4. Add `run_full_enrichment` with the 5 new passes
5. Run the round-trip test again — expect zero diff
6. If non-zero diff, diagnose + fix the offending pass before
   declaring done
7. Update docs

## Effort estimate

- **PR**: 2-4 days. The L3 wire-up is mostly mechanical (call
  existing functions in order), but the round-trip determinism
  hunt has variable cost depending on what we discover.

## Locked-in decisions

| | Decision |
|---|---|
| Old `run_ocr_lemma_enrichment` signature | **Rename outright** to `run_full_enrichment`. Two in-tree callers (`rebuild-from-jsonl`, `diff-rebuild`); easy to update both. No deprecation alias. |
| `--fetch-bulk` default | **Opt-in**. CI / offline operators shouldn't silently fetch from S3; explicit flag makes the network call deliberate. Without the flag, missing slices fail loudly with a hint. |
| Per-pass `--enrichment-stop-at` flag | **YAGNI**. The individual `lexicon decompose` / `cluster-cognates` / `classify-stratum` / `derive-english-shaped` / `project-period-forms` commands already exist for one-off iteration; no need to duplicate that surface inside `--with-enrichment`. |

## Test plan

- [ ] `ingest_all_slices` unit tests (with moto-mocked S3 + a
  tiny fixture .jsonl.zst)
- [ ] `run_full_enrichment` happy-path: each pass invoked once,
  counts dict shaped correctly
- [ ] `run_full_enrichment` failure path: one pass raises → counts
  dict captures partial results, error surfaces
- [ ] Round-trip integration test against the committed bundle
- [ ] `rebuild-from-jsonl --skip-bulk` path: produces L2-only DB
  (existing behavior, regressioning the new flag)

## Dependencies

Blocks-on: **wyrd-0vj3** (storage strategy must be implemented so
`ingest_all_slices` has a manifest + bucket to call against).
Both wyrd-0vj3 PRs landed; bucket exists; manifest in flight via
the seed.

Companion of: nothing else right now. Once this lands, the
bundle-deploy chain (`wyrd-j43l` + `wyrd-b7fo`) becomes much
shorter because the rebuild step is one command.
