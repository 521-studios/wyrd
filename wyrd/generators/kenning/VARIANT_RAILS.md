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
- [ ] A1. Diagnostic census: reflex-less vs variant-gap vs matched pairs
      (launch baseline was reflex-less≈21.3%, variant-gap≈22.4%, matched≈56.2% of
      ~43,361 pairs). Add as `enrich-campaign variant-gap-status`.
- [ ] A2. Selector `enrich-campaign variant-gap-next-slice --n N`: per not-yet-
      CAN-IT toponym, surface the scholar morpheme(s) whose reflexes DON'T
      substring-match this toponym + the residual span(s) to author. Prioritize
      toponyms where it's the LAST unmatched morpheme (flips CAN-IT), then
      morphemes appearing in the most such toponyms. JSON on stdout + progress.
- [ ] A3. Authoring path: reuse `validate_candidates` (grounding holds: span ∈
      toponym). Append accepted rows to `data/mining/_reflexes.jsonl`.
- [ ] A4. Tests + ruff. (No new ledger/table → no rebuild-discipline wiring.)

### Rail B — 4.2.2 variant rail + tag convention (etymon.variants; authorized, grounded)
- [ ] B1. Tag convention (740t): add the `toponymic-surface` marker to variant
      storage (D18 pool — confirm `etymon_text_match` vs `etymon_variant`).
- [ ] B2. Variant L2 ledger `data/mining/_variants.jsonl`
      (`{_type:variant, ref, surface_form, tag:"toponymic-surface", ...}`) + a
      replay pass that applies it. **Rebuild-discipline 4-file change.**
- [ ] B3. Rail CLI: `enrich-campaign next-slice --phase variants` (admit-cohort
      etymons missing a variant; remainder from committed `_variants.jsonl`; with
      grounding evidence) + `validate --phase variants` (grounding guard identical
      to reflexes) + park. (Tag rail half already shipped in #727 — mirror it.)
- [ ] B4. Generation gate: standalone-name generation excludes
      `toponymic-surface`-tagged variants; composition/worn-form path includes
      them. Test it both ways.
- [ ] B5. Authoring path: grounded alt-spellings → `validate` → append
      `_variants.jsonl`; park un-groundable.
- [ ] B6. Tests + ruff + rebuild-runbook + db-reconstructibility wiring green.

### Run phase (after BOTH rails are built + green)
- [ ] Drain `variant-gap-next-slice` (Rail A) and `next-slice --phase variants`
      (Rail B), authoring grounded rows / parking, committing each slice.

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
