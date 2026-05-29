# BEFORE-rebuild snapshot — 2026-05-29

**Captured on:** pre-vpri code (local `main` @ `9688916`, i.e. WITHOUT PR #376
wyrd-vpri `bare`-location change) + the current pre-rebuild lexicon DB
(`~/.wyrd/lexicon.db`, 2.9 GB, phase-2 toponym attestations already ingested:
toponym_attestation = 58,345, +34,386 from phase2).

This is the baseline for the nuclear-rebuild before/after diff. The AFTER
snapshot (`after-rebuild-20260529/`) is captured with **post-vpri code +
rebuilt data**, so the diff reflects rebuild + vpri combined. A later
briggs-only rebuild diffs against the AFTER snapshot to isolate briggs.

## Rando-port retirement readiness — FAIL (gate closed)

| lang | total | scholar | empirical | rando-only | uncited | scholar-only | coverage | C1 |
|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| old_english | 1396 | 555 | 483 | 0 | 358 | 39.8% | 74.4% | ✗ |
| old_french | 548 | 16 | 505 | 0 | 27 | 2.9% | 95.1% | ✓ |
| old_scandinavian | 453 | 151 | 198 | 0 | 104 | 33.3% | 77.0% | ✗ |
| celtic_mix | 2332 | 521 | 1538 | 0 | 273 | 22.3% | 88.3% | ✓ |
| latin | 22 | 7 | 0 | 0 | 15 | 31.8% | 31.8% | ✗ |

C1 fails on OE / ON / latin (combined coverage < 80%). C2/C3 pass everywhere
(zero rando-only). The scholar-only column (wyrd-w1ak) shows how much rests on
the empirical pipeline: OF is 95.1% combined but only 2.9% scholar.

## Drift reports — 2/5 cultures captured, 3/5 CRASH (before-finding)

- **scottish, breton**: captured. Vector path works (A=1000, B=1000,
  decomposition rate 1.0 both modes — confirms #359–369 vector commits are
  live; the earlier "vectors broken" alarm was a stale-import artifact).
- **english, welsh, irish**: CRASH — `KeyError: ('pre','name')` at
  `proportions.py` select(). Root cause: the bundled `*_proportions.json` are
  STALE vs the current bundle. This is precisely what the rebuild's
  `rebuild-proportions` / `export-runtime-db` (inline proportions rebuild)
  fixes. Expect all 5 cultures to generate in the AFTER snapshot.

## Key vpri before/after signal

Pre-vpri, the drift position distribution is **`inner / post / pre`** only —
no `bare` (bare keys were misclassified as `post`). Example (scottish):
`inner=0.153, post=0.422, pre=0.425`. After rebuild + vpri, expect a distinct
**`bare`** position to appear, and single-word slots to stop drawing
suffix-only (`post`) morphemes (the `-park` → "Park" standalone). Grep AFTER
drift samples for suffix/prefix morphemes standing alone — should be gone.

## Files
report.txt · stats.txt · enrichment_status.txt · era_coverage.txt ·
language_report.{md,json} · empirical_priors.json · rando_port_readiness.txt ·
meanings_before.json (113 MB) · drift_{scottish,breton}.{md,json} ·
drift_errors.log (records the 3 crashes).
