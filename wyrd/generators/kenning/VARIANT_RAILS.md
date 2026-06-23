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
- Fire #34 (2026-06-23): authored 5 — brēr→`brier` (Brierley), burh-tūn→`barton` (Barton), Hrōc→`rox`/`ruck` (Roxwell/Ruckinge), Cifel→`wivel` (Wiveliscombe). Parked 8 (Bild, Bitta, Cocc, Deor, Ēata, Otta, Scott, camp). Cumulative: 112 authored, ~204 parked. NB: build candidates by diacritic-folding the slice canonical_form to look up the exact ref (hand-typing macron forms KeyErrors).
- Fire #35 (2026-06-23): authored 6 — cetel→`chettle`/`cheddle` (Chettle/Cheddleton), cærse→`kears` (Kearsley), dēop→`dept`/`dib` (Deptford/Dibden), dēor→`dere` (Dereham). Parked 8 (cweorn, edisc, ende, eorl, fearn, gāra, heafod, healh). Cumulative: 118 authored, ~212 parked. (Used index-based slice selection to dodge diacritic matching.)
- Fire #36 (2026-06-23): authored 9 — hæcc→`heck`/`hac` (Heck/Haccombe), lēac→`lack`/`leek` (Lackford/Leek), myrge→`mir` (Mirfield), stapol→`stable` (Dunstable), steort→`stort`/`stert` (Stortford/Stert), ticcen→`tick` (Tickhill). Parked 6 (hege, hwīt, ryge, seofon, stōw, swīn). Cumulative: 127 authored, ~218 parked.
- Fire #37 (2026-06-23): authored 8 — twisla→`twizel`/`twis` (Twizel/Twiston), winn→`wen` (Wenham), æcer→`aker` (Fazakerley), kjarr→`car` (Redcar/Altcar), klakkr→`claugh`/`clax`/`cleck` (Claughton/Claxton/Cleckheaton). Parked 7 (ware, wisc, īeg, Bruni, eng, hlíð, hryggr). Cumulative: 135 authored, ~225 parked.
- Fire #38 (2026-06-23): authored 3 — kunung→`coney`/`coning` (Coney Weston/Conington), vatn→`was` (Wasdale). Parked 10 (rauðr, skáli, breac, caer, carn, cill, din, fionn, gwaun, liath). Cumulative: 138 authored, ~235 parked. Slice now Celtic-dominated (low-yield tail).
- Fire #39 (2026-06-23): authored 1 — Billa→`billing` (Billingham). Parked 11 (all Celtic/Latin-translation/PN, no clean form). Cumulative: 139 authored, ~246 parked. FLOOR REACHED: slice now ~1/12 authorable (gaps=4, Celtic/PN/translation tail). The OE/ON topographic inventory is mined out. Recommend merge PR #740 + pause the cron.
- Fire #40 (2026-06-23): authored 2 — Cula→`cul` (Culford), Tibba→`tib` (Tibshelf). Parked 12 (all personal-name tail). Cumulative: 141 authored, ~258 parked. Slice now 100% PN tail; topographic inventory exhausted. Strongly recommend merge #740 + pause cron.
- Fire #41 (2026-06-23): authored 3 — Wōden→`woodnes` (Woodnesborough), bere-tūn→`berton` (Pemberton), copp→`cop` (Warcop). Parked 13 (mostly conflations: camb/cumb, carr/kjarr, cild/cill + well-covered + PN). Cumulative: 144 authored, ~271 parked. Still floor (~3/16); remaining gaps now largely conflation-risk. Recommend merge #740 + pause cron.
- Fire #42 (2026-06-23): authored 6 — fāg→`vow`/`vau`/`fown` (Vowchurch/Vauchurch/Fownhope), grund→`ground` (Stanground), græf→`grove`/`graf` (Chalgrove/Grafton). Parked 13 (conflations + well-covered). Cumulative: 150 authored, ~284 parked. Occasional real element still surfaces amid the conflation tail (fāg/græf were under-covered).
- Fire #43 (2026-06-23): authored 6 — pull→`pill` (Huntspill), pīc→`pitch` (Pitchcott), ribbe→`rib` (Ribston), sceaft→`skeff` (Skeffington), scīr→`skir` (Skircoat), setl→`sedl` (Sedlescombe). Parked 10. Cumulative: 156 authored, ~294 parked. Norse-influenced/palatalized forms (skir/skeff/pitch/pill) still surface as real gaps on otherwise well-covered elements.
- Fire #44 (2026-06-23): authored 5 — wīc-hām→`wykeham`/`witcham` (Wykeham/Witchampton), æðeling→`adling` (Adlingfleet), ON borg→`borough`/`borrow` (Scarborough/Borrowdale — correct ON-vs-burh etymon). Parked 13. Cumulative: 161 authored, ~307 parked.
- Fire #45 (2026-06-23): authored 2 — episcopus→`episcopi` (Huish Episcopi — correct Latin etymon for the manorial affix), Baldhere→`balder` (Baldersby). Parked 14 (Celtic/goidelic/Latin-translation/PN tail). Cumulative: 163 authored, ~321 parked. Celtic-heavy low-yield slice.
- Fire #46 (2026-06-23): authored 5 — Cippa→`kip` (Kiplin), Cumbre→`comber`/`cummers` (Comberton/Cummersdale), Frīsa→`frys` (Fryston), Huna→`hon` (Honiton). Parked 12 (PN/conflating tail). Cumulative: 168 authored, ~343 parked.
