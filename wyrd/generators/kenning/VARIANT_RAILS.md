# Variant Rails — autonomous reflex-authoring loop (wyrd-eni4.3.1 + wyrd-eni4.2.2)

Durable state for the variant-gap **reflex** loop. The 5-minute cron prompt
carries the load-bearing invariants; this file is the source of truth for the
model + run state.

## PIVOT (owner, 2026-06-23): one reflex rail, NOT `etymon_variant`

A toponymic worn-form **is a reflex**, so it belongs in `reflex`/`reflex_etymon`
(which already feeds BOTH the matcher and generation) — **not** `etymon_variant`:

- reflexes already deliver 4.2.2's stated goal ("recall + worn-form generation");
- `reflex.position` (pre/post/inner) already encodes *"compositional, never a
  standalone name"* — the exact property the dropped B4 gate was hand-rolling on a
  position-less table;
- `etymon_variant` is 5.78M Wiktionary lexicographic forms + 501 collapse folds —
  generation correctly ignores it; it was the wrong home.

**Consequence:** wyrd-eni4.2.2 (the "barthos spread" — alt-spellings of accepted
morphemes) **folds into the variant-gap selector**. Both are the same lever:
*a morpheme whose reflexes don't substring-match a toponym it appears in → author
the worn surface as a reflex.* The `etymon_variant` machinery (former B1–B6: the
`_variants.jsonl` ledger, `toponymic-surface` tag, `variants-*` CLI, replay +
rebuild-discipline wiring) and the B4 / wyrd-740t generation-integration are
**REMOVED**. One rail, one selector, reflexes only.

## Locked decisions
1. **Span** = the agent proposes the worn surface from the toponym evidence,
   grounding-gated (must substring-match an attested toponym form, else PARK).
   Authored as **reflex `surface_form` rows** in `_reflexes.jsonl`.
2–4. **SUPERSEDED by the pivot** — no `etymon.variants`, no `toponymic-surface`
   tag, no separate variant rail, no build-both-then-run. One reflex rail.

## Guardrails (unchanged)
- **Grounding gate:** surface must substring-match an attested toponym form of a
  referenced etymon, else PARK.
- **NEVER touch parse selection / scorer / tiebreak** — reflexes are the CAN-IT /
  generation side; never change which parse is selected.
- **NEVER author cognate-binds or lemma-wiring.**
- **Dashes are never stored identity (D45)** — reject affix-position dashes.
- **`ruff format .` + `ruff check .` + `pytest` the touched suites before every push.**

## Build checklist — COMPLETE
- [x] A1. Diagnostic census — `enrich-campaign variant-gap-status` /
      `variant_gap_census`.
- [x] A2. Selector — `enrich-campaign variant-gap-next-slice` /
      `variant_gap_next_slice` (per-etymon reflexes = DB ∪ committed
      `_reflexes.jsonl`; flip-CAN-IT-first then by gap frequency; parked-aware via
      `--parked-path`).
- [x] A3/A4. Authoring path (no new code — `variant-gap-next-slice` → author reflex
      rows → `enrich-campaign validate` (`validate_candidates`) → append
      `_reflexes.jsonl`) + tests. No new ledger/table.
- ~~B1–B6~~ **removed in the pivot** (was the `etymon_variant` rail).

## Run phase — author reflexes via the one selector
Each fire: `enrich-campaign variant-gap-next-slice --n N` → propose grounded worn
surfaces (a clean, generalizable worn form per morpheme; D45-clean) →
`enrich-campaign validate --candidates …` → append `data/mining/_reflexes.jsonl`.
**PARK** (`enrich-campaign park`) anything ungroundable, homographic, a too-short
form, a misattribution, or a well-covered morpheme whose only remaining gaps are
risky-tail — rather than pollute the matcher. Commit + push every fire.

## Loop procedure (each 5-min fire)
0. `date '+%F %T %Z'`; cd the worktree; `git pull --rebase`; read this file.
1. Run a variant-gap authoring slice (above).
2. `ruff format`/`check` + `pytest` the touched suites; commit + push.
3. ITERATION FINISH timestamp; update the run log + wyrd-eni4.3.1 bd notes.
- Single branch `enrichment/variant-rails`, single PR #740 — commit+push every fire.

## Run log
- Fires #8–#29 (2026-06-23): **87 reflexes authored, ~163 parked** across 20
  authoring slices (highlights: `ceaster→chester/caster/cester/castle`,
  `prēost→pres/priest`, `lēah→ley`, `dūn→den`, `bōðl→bottle`, `clǣg→cley/clee`,
  `lēac-tūn→latton/layton/letton`, ON `skógr→scoe/sceugh`, `brekka→brick`). See
  the git log for per-fire detail.
- 2026-06-23: **pivot landed** — removed the `etymon_variant` rail (former B1–B6)
  and the B4/740t generation-integration; folded 4.2.2 into this one reflex rail.
- Fire #30 (2026-06-23, post-pivot): authored 3 — eglēs→`eggle`/`eglys` (Eggleston/Eglysham), Cuda→`cud` (Cudham). Parked 10 (melr, thwaite, canto, cnoc, sub, Acca, Alhmund, Banna, Boll, Pica). Cumulative: 90 authored, ~173 parked. Deep tail — mostly PN/short/translation forms now.
- Fire #31 (2026-06-23): authored 7 — amore→`amber` (Amberley), biscop→`bisp` (Bispham), brant→`brent` (Brentor), bred→`bret` (Bretford), gelād→`lade`/`lode` (Cricklade/Evenlode), grēne→`gren` (Grendon). Parked 6 (calc, croft, cyning, elfitu, flēot, hldw). Cumulative: 97 authored, ~179 parked.
- Fire #32 (2026-06-23): authored 7 — hrīs→`rus` (Ruston), næss→`naze`/`nass` (Nazeing/Nassington), scēp→`skip`/`shif` (Skipton/Shifford), ON berg→`barrow`/`borrow` (Barrowby/Borrowby). Parked 8 (hlinc, hrycg, hār, lea, stoc, wice, wīd, þrop). Cumulative: 104 authored, ~187 parked (crossed 100).
- Fire #33 (2026-06-23): authored 3 — Babba→`bab` (Babington), warde→`ward` (Westward), Affa→`uff` (Uffington). Parked 9 (lundr, þorp, þveit, creig, aber, druim, celtic:dun, odhar, Aldo). Cumulative: 107 authored, ~196 parked. Tail now heavy with well-covered Celtic/ON clusters + PNs + OCR-garbled refs.
