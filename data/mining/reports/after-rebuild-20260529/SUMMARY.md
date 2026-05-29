# AFTER-rebuild snapshot — 2026-05-29 (FINAL)

**Captured on:** post-vpri code (PR #376) + post-POS-filter code (PR #378
wyrd-33cv) + freshly rebuilt lexicon DB with the **de-polluted empirical
layer restored** + final `seed-runtime.db` (74,501 meaning rows, 31,817
proportion rows, 5 cultures, 76 MB).

Compare against `../before-rebuild-20260529/` (pre-vpri, pre-rebuild). This
AFTER state = rebuild + vpri + clean wiktextract re-mine. A later briggs-only
rebuild diffs against THIS snapshot to isolate briggs.

## Full pipeline that produced this state
rebuild-from-jsonl --with-enrichment (wipe + L1/L2 replay + 8 enrichment
passes) → re-ingest phase2 (+33,318 attestations) → backfill-toponym-country
→ mine-empirical-baselines → dump-priors → tag-phonological-vectors (100%,
2.37M) → **export #1 (empirical-less)** → **mine-wiktextract-corpus (POS-
filtered)** → **cleanup-wiktionary-empirical (−12,410 polluted)** → **export
#2 (final)** → after-reports.

## ✅ vpri validated — `bare` is a first-class, balanced position

Before-rebuild drift positions: `inner/post/pre` only, NO bare (bare keys
misclassified as post). FINAL after (full bundle):

| culture | bare | inner | post | pre |
|---|--:|--:|--:|--:|
| english | 0.281 | 0.069 | 0.312 | 0.337 |
| scottish | 0.389 | 0.118 | 0.242 | 0.252 |
| welsh | 0.425 | 0.117 | 0.224 | 0.234 |
| irish | 0.474 | 0.191 | 0.166 | 0.170 |
| breton | 0.344 | 0.126 | 0.242 | 0.288 |

`bare` is now ~28–47% (real single-word morphemes), distinct from `post`
suffixes. All 5 cultures generate (english/welsh/irish no longer crash on the
`('pre','name')` stale-proportions KeyError). Suffix-only keys no longer fill
single-word slots.

## ✅ Empirical layer restored, de-polluted (wyrd-smtc fix)

The rebuild dropped the DB-only wiktionary-empirical layer; the POS filter
(wyrd-33cv) + cleanup re-mined it clean:

| lang | total b→f | scholar b→f | empirical b→f | coverage b→f |
|---|---|---|---|---|
| old_english | 1396→1385 | 555→540 | 483→455 | 74.4%→71.8% |
| old_french | 548→498 | 16→10 | 505→458 | 95.1%→94.0% |
| old_scandinavian | 453→413 | 151→146 | 198→161 | 77.0%→74.3% |
| celtic_mix | 2332→1987 | 521→469 | 1538→1279 | 88.3%→88.0% |
| latin | 22→20 | 7→5 | 0→0 | 31.8%→25.0% |

Empirical recovered to ~94% of pre-rebuild; the delta is the pollution removed
(re-mine dropped function words via the POS filter; cleanup dropped 12,410
modern-english-homograph / derivative etymons). Coverage essentially preserved
minus noise. Rando gate still FAIL (C1 < 80% on OE/ON/latin) — unchanged
posture, expected; C2/C3 pass everywhere.

Intermediate empirical-less export (for the record): OE 886 subjects / 52.6%
— fully restored by the re-mine to 1385 / 71.8%.

## ⚠ Pending decision — commit the 76 MB seed-runtime.db?

The final `seed-runtime.db` (76 MB, was 6.7 MB committed) is the deliverable
runtime bundle but is currently uncommitted (working-tree modified). The size
grew because the bundle now carries the empirical layer + per-meaning
phonological_vector blobs + era data. Operator to decide: commit the binary to
the repo (established pattern, but large), publish to S3, or handle
separately.

## Files
report.txt · stats.txt · enrichment_status.txt · era_coverage.txt ·
language_report.{md,json} · empirical_priors.json · rando_port_readiness.txt ·
meanings_after.json (101 MB) ·
drift_{english,scottish,welsh,irish,breton}.{md,json} · rebuild.log.

## ✅ wyrd-36ez semantic audit — de-pollution VERIFIED (2026-05-29)

Re-ran `audit-semantic-coherence` on the cleaned bundle (mxbai-embed-large @
localhost, 24,557 subjects). Output: `audit/cross-sibling-suspects.csv` +
`audit/intra-entry-suspects.csv` (top 200 each).

**Result: every top suspect is a LEGITIMATE homonym, not pollution** — the
exact wyrd-smtc success condition ("once clean, the audit surfaces only real
homonyms hiding as polysemy"):

- cross-sibling: `sea` (modern-english "salt water" vs celtic "yes"-copula),
  `ham` (modern-english "pasture" vs old_french "Homs/village").
- intra-entry: `ric` ("stream" vs "king"), `wer` ("weir" vs "man"), `gear`
  ("weir" vs "year"), `mag` ("maiden" vs "kinsman"), `Gip` ("given name" vs
  "broad"), `rigg` ("stormy wind" vs "ridge").

ZERO "alternative form of X" / "plural of Y" / function-word entries in the
top suspects — the POS filter (#378 wyrd-33cv) + cleanup (−12,410) + the
redirect/derivative filter (a106) removed the noise the epic measured. The
remaining suspects are real homonyms for hand-resolution under wyrd-sfdj
(semantic-coherence split tool), NOT pollution. **wyrd-smtc closed.**
