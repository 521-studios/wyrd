# Variant Rails — autonomous build+author loop (wyrd-eni4.3.1 + wyrd-eni4.2.2)

This is the durable state/checklist for the **variant rails** loop. The 5-minute
cron prompt carries the load-bearing invariants; this file carries the **ordered
build checklist + completion state** the loop advances one coherent piece per
fire. Update the checkboxes here as pieces land.

## Locked decisions (owner, 2026-06-23)

1. **4.3.1 span extraction = LLM-assisted, grounding-gated.** The cron agent reads
   the not-yet-CAN-IT toponym + unmatched scholar morpheme + evidence and proposes
   the surface span; the existing `validate_candidates` grounding guard still gates
   it (span MUST substring-match an attested toponym form). Authored as **reflex
   `surface_form` rows** in `_reflexes.jsonl` — within loop remit, NOT the
   human-gated `etymon.variants` field.
2. **4.2.2 toponymic-surface tag = flag + generation-gate.** Variants live in the
   D18 spelling-variant pool tagged `toponymic-surface`; **standalone-name
   generation skips tagged variants**, the composition / worn-form path uses them.
   One pool, one flag.
3. **4.2.2 authority = autonomous, grounded gate.** Author `etymon.variants`
   autonomously; every variant MUST appear in an attested toponym form (the
   "barthos" rule) or it is PARKED. Same trust model as the reflex campaign.
4. **Sequencing = build BOTH rails, then run** both authoring loops.

## Guardrails (NEVER violate)

- **Grounding gate is identical to reflexes** for both rails: a surface/variant
  must substring-match an attested toponym form of a referenced etymon, else PARK.
- **NEVER touch parse selection / scorer / tiebreak.** 4.3.1 authors reflexes
  (CAN-IT, generation side) only; it never changes which parse is selected.
- **NEVER author cognate-binds or lemma-wiring.** Variants (4.2.2) and reflexes
  (4.3.1) only.
- **Dashes are never stored identity (D45):** reject affix-position dashes in any
  authored surface/variant; position is its own field.
- **Rebuild discipline:** any new mining CLI / L2 ledger / table is a 4-file change
  — code + `data/mining/_rebuild_layers.json` + `REBUILD.md` restore step +
  (new L2 file) `L2_L3_BOUNDARY.md` map — and must satisfy
  `tests/test_kenning_rebuild_runbook.py`, the rebuild-runbook-currency-reviewer,
  and the repo-root db-reconstructibility-reviewer.
- **Always `ruff format .` + `ruff check .` + `pytest` the touched suites before
  every push.** CI gates on ruff; the local pre-commit hook does NOT run it.

## Build checklist (one coherent, tested piece per fire)

### Rail A — 4.3.1 variant-gap reflex selector (reflexes; within remit)
- [x] A1. Diagnostic census: reflex-less vs variant-gap vs matched pairs. DONE —
      `enrich-campaign variant-gap-status` + `variant_gap_census()` (pure folded-
      substring test, per-etymon reflexes; no matcher). Live DB census 2026-06-23:
      43,361 pairs — matched 37.8% / variant-gap 19.8% (8,581) / reflex-less 42.4%.
      (Differs from the ticket's cluster-level launch estimate; reflex-less is high
      because the live DB is NOT yet rebuilt with PR #727's merged reflexes. A2 must
      decide per-etymon vs cluster-level coverage — ticket wording says per-etymon.)
- [x] A2. Selector `enrich-campaign variant-gap-next-slice --n N`. DONE —
      `variant_gap_next_slice()` + `VariantGapTask`. Per-etymon reflex surfaces =
      DB ∪ committed `_reflexes.jsonl` ledger (so it drains between fires without a
      rebuild). Prioritized flip-CAN-IT-first (sole unmatched morpheme) then by
      gap-toponym frequency. Each task carries evidence toponyms with a residual-
      span HINT (best-effort, not authoritative — author proposes, gate decides).
      Live smoke: lēah (697 gaps), dūn (133, Battlesden→den), hundred (125). Tested.
- [x] A3. Authoring path. DONE — confirmed NO new production code needed: the run
      phase is `variant-gap-next-slice` → author reflex rows → `enrich-campaign
      validate` (= `validate_candidates`) → append `_reflexes.jsonl`. Grounding
      holds because `_attested_folds` includes the toponym `modern_name`, and the
      worn span comes from that name (e.g. `den` ⊂ `battlesden`).
- [x] A4. Tests + ruff. DONE — `test_variant_gap_authored_reflex_validates`
      (worn span accepted, off-name span rejected by the grounding gate). 24/24
      enrichment tests pass, ruff clean. **Rail A complete.** (No new ledger/table.)

### Rail B — 4.2.2 variant rail + tag convention (etymon.variants; authorized, grounded)
- [x] B1. Tag convention (740t). DONE. Storage = `etymon_variant` (the D18 pool).
      Marker = the string `"toponymic-surface"` in the row's `tags` JSON array
      (`schema.TOPONYMIC_SURFACE_TAG` + `is_toponymic_surface_variant()`), NOT a new
      `variant_class` — the variant_class CHECK enum is closed
      (alternative/inflection/romanization/canonical/other), and the locked
      decision says "tagged", so the tags column avoids a schema migration.
      Toponymic-surface rows use `variant_class='alternative'` + the tag.
      **B2 writes** rows as (etymon_id, form, 'alternative', tags=["toponymic-surface"]).
      **B4 gates** the standalone-name path with `is_toponymic_surface_variant`;
      `period_form`/composition already reads all forms, so it needs no change. Tested.
- [x] B2. Variant L2 ledger `data/mining/_variants.jsonl` + replay. DONE. Row
      shape: `{_type:variant, ref, form, tags:["toponymic-surface"]}` (+ a one
      `source` row, ref `variant-uplift`, per the L2 contract). Replayed by
      `build._insert_variant_rows` → `etymon_variant` (INSERT OR IGNORE on the
      (etymon_id, form, variant_class) UNIQUE key; orphan refs skipped+counted;
      `variant_class='alternative'`). Rebuild-discipline wiring: `log.LIST_TYPES`
      adds `variant`; `_rebuild_layers.json` layer `toponymic-surface-variants`;
      `REBUILD.md` step-3 list; `L2_L3_BOUNDARY.md` map. Replay test +
      rebuild-runbook green (77 pass), ruff clean. NB authoring uses `form` (the
      variant spelling), not `surface_form`.
- [x] B3. Rail CLI. DONE — built as per-dimension commands (mirroring the SHIPPED
      `tags-next-slice` style, NOT a `--phase` flag; the prompt's "--phase variants"
      wording predates that convention). `variants-next-slice` (cohort missing a
      committed toponymic-surface variant, remainder from `_variants.jsonl` +
      `_variant_parked.jsonl`, reuses `etymon_evidence` grounding) +
      `variants-validate` (`validate_variant_candidates`: grounding guard identical
      to reflexes, D45 dash-reject, requires the toponymic-surface tag, scope, dedup
      on (ref,folded-form)) + `variants-park`. Lexicon fns: `committed_variant_keys`
      / `committed_variant_refs` / `variant_next_slice` / `validate_variant_candidates`.
      RUN PHASE uses `variants-next-slice` / `variants-validate` (not `next-slice
      --phase`). Live smoke OK (tūn, lēah). Tested (26 pass), ruff clean.
      MINOR DEBT for B6: `variants-park` ~dups reflex `cmd_park` — dedup into a
      shared `_park_ref` helper.
- [⛔ PARKED] B4. Generation gate. **BLOCKED — needs a user decision.** Discovery:
      the runtime bundle export does NOT read `etymon_variant` at all (its `_variants`
      field is reflex-surface dash-grouping, unrelated). So generation cannot emit
      toponymic-surface variants today — there is **no standalone read path to gate**
      (hence no leak risk), and the variants would be **inert until generation is
      wired to consume them**. Wiring `etymon_variant` (incl. the toponymic-surface
      pool) into the runtime bundle + name generation IS the substance of wyrd-740t:
      a separate design effort (bundle schema for the spelling pool; runtime
      composition-vs-standalone usage; respelling/name.py integration) the locked
      decisions don't cover. The gate PREDICATE is already built and waiting
      (`schema.is_toponymic_surface_variant`, B1). DO NOT guess the integration
      architecture. Resolution options for the user: (a) authorize the 740t
      generation-integration as new scoped work; (b) defer Rail B authoring and run
      Rail A only; (c) other.
- [x] B5. Authoring path. DONE — no new code: `variants-next-slice` →
      `variants-validate` (grounding) → append `_variants.jsonl` → replay
      (`_insert_variant_rows`, B2). Covered end-to-end by the B3 validate test +
      B2 replay test + the `variants-park` CLI test.
- [x] B6. DONE — park dedup landed (`_park_ref` shared by reflex + variant park,
      tested both rails); enrichment (27) + build/replay + rebuild-runbook suites
      green; ruff clean; db-reconstructibility 4-file wiring in place (B2).

### Run phase — STATUS: Rail A runnable; Rail B HELD on B4 (user decision)
Rail A (variant-gap reflexes) is fully built + EFFECTIVE (reflexes feed the
matcher/CAN-IT) — the unblocked CAN-IT lever. Rail B authoring is build-complete
but **inert until B4/740t**, so authoring it now would write ledger data that
changes no output.
- [ ] **Next fire (judgment call, surfaced to user): RUN RAIL A authoring** —
      `variant-gap-next-slice --n 20`, propose grounded worn spans, `validate`
      (reflex gate), append `_reflexes.jsonl`, park the ungroundable. This is
      unblocked, effective, and serves the CAN-IT north star.
- [ ] **HOLD Rail B authoring** (`variants-next-slice`) until the user resolves B4.
      Do NOT author inert variant rows; do NOT build the 740t integration unprompted.
- NB run command is `variants-next-slice`, NOT `next-slice --phase variants`.

### Run log
- Fire #8 (2026-06-23): first Rail A authoring slice. Authored 5 grounded
  variant-gap reflexes — lēah→`ley`, dūn→`den`, æsc→`ash`, burh→`borough`,
  ēg→`ey` (all validated). Parked 5 — `hundred` (admin annotation, no surface),
  `latin:parva`/`latin:magna` ("Little"/"Great" are English translations not Latin
  reflexes), `hyll` (well-covered), `ing` (rare hill-homograph). Also FIXED a gap:
  `variant_gap_next_slice` now takes `parked_path` (shared reflex park ledger) so
  parked morphemes don't re-surface — `variant-gap-next-slice` CLI gained
  `--parked-path`. High-impact morphemes (lēah 697, dūn 133 gaps) correctly
  re-surface with fewer gaps until enough surfaces are authored.
- Fire #9 (2026-06-23): authored 4 — stoc→`stoke`, stān→`ston`, hēah→`hea`,
  berg→`barrow` (validated). Parked 8 well-covered/long-tail morphemes (dūn, lēah,
  ingtūn, tūn, halh, ingas, ēg, ford) whose remaining gaps are idiosyncratic,
  homographic (-low=hlāw, -sea=sǣ), or junk toponyms ("Not in Dom"). Emerging
  policy: author the clean generalizable worn form; park a well-covered morpheme
  once its tail is only homographic/noise rather than pollute the matcher.
- Fire #10 (2026-06-23): authored 2 — burna→`borne` (Enborne/Golborne), hōh→`hoo`
  (Hoo/Hooton). Parked 10 (berg, hēah, tun, by, burh, mere, haugr, celtic:lann,
  denu, ōra). YIELD DROPPING (5→4→2 clean authors/fire): the top variant-gap
  morphemes are draining — their clean generalizable worn forms are authored, and
  the long tail is mostly misattributions, junk toponyms, or homographs being
  parked. Cumulative: 11 reflexes authored, 23 parked.
- Fire #11 (2026-06-23): authored 4 — bearu→`beare` (Aylesbeare/Loxbeare),
  fenn→`fen` (Fen Ditton), mersc→`mars` (Marston), wudu→`wode` (Chetwode). Parked
  8 (hōh, ham, hām, āc, hamm, hlāw, leah, ēa). Yield recovered to 4 (distinctive
  worn forms). Cumulative: 15 reflexes authored, 31 parked.

## Loop procedure (each 5-min fire)
0. `date '+%F %T %Z'` (ITERATION START). `git pull --rebase`.
1. Read this checklist. If a build piece is unchecked → do the NEXT one (one
   coherent, tested unit), check it here, `ruff` + `pytest`, commit+push.
2. If ALL build pieces are checked → run an authoring slice (A then B), validate
   (or park), append the ledger, commit+push.
3. ITERATION FINISH timestamp; update wyrd-eni4.3.1 / wyrd-eni4.2.2 notes;
   commit+push. Keep the prompt-cache idle gap < 5 min.
- Single branch `enrichment/variant-rails`, single PR — commit+push every fire.
  Do NOT merge to main mid-build; merge only when BOTH rails are built, authoring
  has drained (or a clean milestone), and CI is green on HEAD.
- Fire #12 (2026-06-23): authored 2 — cald→`cold` (Cold Brayfield), cumb→`coombe` (Coombe). Parked 10 (cēd, feld, ēast, hǣð, trēow, wella, west, wudu, bearu, saint — incl. `St.`-abbrev & homographs -will/-beer). Cumulative: 17 authored, 41 parked. CLEARLY into diminishing returns: slices now dominated by well-covered morphemes + homographs + misattributions; ~2 clean authors/fire.
- Fire #13 (2026-06-23): authored 4 — hæg→`hey` (Heydon), clif→`clive` (Radclive), brōc→`brough` (Broughton), ōfer→`sor` (Edensor). Parked 8 (geat, wīc, hēafod, ofer, stān, þorn, ærn, haga). Cumulative: 21 authored, 49 parked.
- Fire #14 (2026-06-23): authored 4 — hæsel→`hasel` (Haselbury), hām-stede→`hamstead` (Hamstead), grāf→`grave` (Hargrave/Bygrave), clif→`clyffe`. Parked 8 (weg, worð, dubh, hop, mǣd, wīðig, hol, cumb). Cumulative: 25 authored, 57 parked.
- Fire #15 (2026-06-23): authored 6 — hwǣte→`wheat`, hīd→`hyde`, scēp→`shep`, bǣce→`bach` (Sandbach), cirice→`cheri` (Cheriton), hyrst→`hirst`. Parked 6 (ōfer, hæg, lacu, middel, sūð, burna). Cumulative: 31 authored, 63 parked.
- Fire #16 (2026-06-23): authored 3 — bōðl→`bottle` (Newbottle), Catta→`chat` (Chatburn), hām-stede→`hempstead` (Hempstead). Parked 9 (clif, land, stede, wall, dīc, fenn, hrēod, penn, an) — incl. wealh/Briton & Celtic-pen conflations. Cumulative: 34 authored, 72 parked.
- Fire #17 (2026-06-23): authored 8 — ceaster→`chester`/`caster`/`cester`/`castle` (was only ceaster/ster — big under-coverage fix), askr→`ash` (Ashby, ON cognate), ald→`aud` (Audlem), dæl→`dal` (Dalton), hæsel→`hesle` (Hesleden). Parked 7 (strēt, brycg, brōm, cald, calf, lang, mersc). Cumulative: 42 authored, 79 parked.
- Fire #18 (2026-06-23): authored 4 — micel→`much` (Much Birch), þyrne→`thurn` (Thurnham), næss→`nes` (Totnes), breg→`bray` (High Bray). Parked 8 (myln, ald, bǣce, cirice, dun, hūs, lane, scelf). Cumulative: 46 authored, 87 parked.
- Fire #19 (2026-06-23): authored 3 — pull→`poul` (Poulton), bōc→`buck` (Buckland/Buckholt), Huna→`hun` (Hunwick). Parked 9 (æsc, breg, crūg, Ella, ceole, hæsel, hām-stede, lȳtel, mǣre) — many are now-mature morphemes re-surfacing with only their tails left. Cumulative: 49 authored, 96 parked.
- Fire #20 (2026-06-23): authored 2 — fearn→`fern` (Fernham), castel→`castle` (Barnard Castle, ME). Parked 10 (wic, worth, na, Bota, Cana, cot, cyne, hyrst, hēg, hīd). Cumulative: 51 authored, 106 parked. RAIL MATURE — park:author now ~5:1; clean worn-form yield down to ~2/fire. The high-value inventory is captured.
- Fire #21 (2026-06-23): authored 6 — sand→`samp` (Sampford), scēp→`shef`/`shap` (Shefford/Shapwick), wincel→`wink`/`winch` (Winkleigh/Winchfield), rauðr→`raw` (Rawcliffe). Parked 8 (rīð, ufan, þyrne, Abba, rath, acus, super, Ali). Cumulative: 57 authored, 114 parked.
- Fire #22 (2026-06-23): authored 5 — brocc→`brock` (Brockley), bōc→`bough` (Boughton), ceaster→`castor`/`caistor` (Castor/Caistor), gāra→`gore` (Ferngore). Parked 8 (Catta, cocc, es, hall, hecg, hlid, holt, hwǣte). Cumulative: 62 authored, 122 parked.
- Fire #23 (2026-06-23): authored 4 — scīr→`shir`/`sheer` (Shirley/Sheerness), worðign→`worthy` (Bradworthy), kirkja→`kir` (Kirby). Parked 9 (hǣme, nīwe, set, stan, styfic, sǣ, Bada, Hutta, ac). Cumulative: 66 authored, 131 parked.
- Fire #24 (2026-06-23): authored 3 — hangra→`anger` (Shelfanger), hlinc→`lynch` (Lynch), hrycg→`rudge` (Rudge). Parked 9 (belle, blæc, bī, bōðl, bǣr, cū, earn, micel, mōr). Cumulative: 69 authored, 140 parked.
- Fire #25 (2026-06-23): authored 8 — prēost→`pres`/`priest` (Preston/Prescott), rēad→`rat` (Ratcliffe), winn→`wim` (Wimborne), brekka→`brick`, hlíð→`lyth`, skógr→`scoe`/`sceugh` (Haddiscoe/Loscoe). Parked 6 (pirige, worðign, byr, bȳ, toft, Amma). Cumulative: 77 authored, 146 parked. (prēost→pres is high-frequency — Preston.)
- Fire #26 (2026-06-23): authored 5 — clǣg→`cley`/`clee` (Cley/Clee), lēac-tūn→`latton`/`layton`/`letton` (Latton/Layton/Letton). Parked 10 (Bula, Coppa, Dene, beorg, brōc, ceaster, crāwe, crōc, leak, neoðera). Cumulative: 82 authored, 156 parked.
