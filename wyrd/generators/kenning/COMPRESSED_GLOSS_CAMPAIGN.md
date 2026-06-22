# Compressed-gloss campaign — loop runbook (wyrd-u6fn.10)

Sibling of `ENRICHMENT_CAMPAIGN.md` (the eni4 reflex/tag/IPA loop). This loop
compresses each morpheme's **gloss WALL** into its distinct **senses**, each
carrying one short **canonical label** — the readable "what does this toponym
mean" token (`'New'`+`'Town'`). It realizes the D50-deferred compressed gloss
(DECISIONS D50 "Bounded by"; KENNING_DOCS invariant on `canonical_sense`).

## North star

Give every generation-eligible morpheme a **canonical_sense** layer: cluster its
glosses into senses (`same-sense` identity binds) and label each with one short
token. Monosemous walls (`tūn`: 'enclosure','farmstead','manor','town' → ONE
sense 'town') collapse to a single label; genuine polysemy ('wall' AND 'frog') →
a small array, one label per distinct sense. The presentation surface
(`NewName.description()`, wyrd-u6fn.10.3) is a SEPARATE human-gated ticket — the
loop only authors the data.

## Mechanism

- **A fixed 5-minute `/loop`** (`/loop 5m <prompt>`), NOT self-paced. The eni4
  loop showed chunks essentially never leave a <5-min gap, so we stop chasing the
  prompt-cache window and just fix the interval.
- **Worktree** `/home/devon/521Studios/wyrd-gloss`, branch
  `enrichment/compressed-gloss-campaign`, **one long-lived PR that never merges**
  (a human reviews + merges on return).
- `wyrd` = `python -c "from wyrd.cli import main; main()"` (or the installed
  `wyrd`); the live authoring DB is `~/.wyrd/lexicon.db` (`WYRD_LEXICON_DB`).

## Per-iteration timing report

Each fire reports its timing (matching the eni4 loop) via a helper kept OUT of
the repo (no CI/PR churn) at `~/.wyrd/gloss_loop_timing.py`, state in
`~/.wyrd/gloss_loop_timing.json`:

- `python3 ~/.wyrd/gloss_loop_timing.py start` — the **first** action of the fire;
  prints `ITERATION START` + the **idle gap from the prior finish** (flags `✓` if
  `<5m`, `⚠ >5m (cache cold)` otherwise — the prompt-cache-warmth target).
- `python3 ~/.wyrd/gloss_loop_timing.py finish` — the **last** action; prints
  `ITERATION FINISH` + elapsed, and records this finish as the next iteration's
  prior. The cron prompt calls both and the fire summary echoes the blocks.

If the worktree/machine is ever re-provisioned, recreate the helper from this
runbook (it's a ~50-line stdlib-only script; the campaign carries no other copy).

## The loop prompt

This is the self-contained prompt to hand `/loop 5m`. Everything load-bearing is
in here because `/loop` refreshes it into context every fire.

```
You are running the compressed-gloss campaign (wyrd-u6fn.10) — a 5-minute /loop.
Worktree: /home/devon/521Studios/wyrd-gloss (branch enrichment/compressed-gloss-campaign).
Runbook: wyrd/generators/kenning/COMPRESSED_GLOSS_CAMPAIGN.md. Progress -> wyrd-u6fn.10 notes.

DON'T PIVOT MID-WORK. A chunk is ATOMIC: slice -> cluster -> validate -> author -> commit+push.
If this fire lands while a previous chunk is unfinished (a /tmp candidate file with un-authored
rows, or a dirty/un-pushed git state), FINISH and push that FIRST, before pulling a new slice.
Never abandon a half-authored batch.

PER FIRE:
1. cd /home/devon/521Studios/wyrd-gloss && git pull --rebase. Resolve any unfinished chunk (above).
2. PHASE 1 (existing glosses -> senses). Get work:
     wyrd kenning lexicon gloss-campaign next-slice --n 12 > /tmp/gloss_slice.json
   For EACH morpheme, partition its gloss wall into senses. BIAS TO MONOSEMY: default to ONE sense
   (most morphemes are one meaning expressed many ways); only split when meanings are genuinely
   UNRELATED (a true homograph). Give each sense a SHORT label: 1-2 lowercase words, the single
   clearest word. EVERY listed gloss must go in exactly one sense, VERBATIM (never invent/alter
   gloss text — the validator rejects it). Write one row per morpheme to /tmp/gloss_cand.jsonl:
     {"_type":"gloss_sense","ref":"<ref from slice>","language":"...","canonical_form":"...",
      "senses":[{"label":"town","glosses":["enclosure","farmstead","town"]}],
      "confidence":"high","method":"opus-gloss-sense-v1","model":"opus","rationale":"<one clause>"}
   You may instead shell to an Ollama model on the Mac (gemma4:26b default; qwen3.5:9b / qwen3:8b
   ok; do NOT use qwen3.5:35b). A/B different models/prompts and keep what reads best; stamp method
   + model honestly. Then:
     wyrd kenning lexicon gloss-campaign validate --candidates /tmp/gloss_cand.jsonl   # exit 0 or FIX/DROP
     wyrd kenning lexicon gloss-campaign author   --candidates /tmp/gloss_cand.jsonl
     git add data/mining/canonicalization && git commit -m "gloss-sense: <n> morphemes" && git pull --rebase && git push
   PARK anything you can't confidently compress (a junk/over-segmentation fragment, an opaque
   foreign gloss): wyrd kenning lexicon gloss-campaign park --ref "<ref>" --reason "<one line>".
   Run more chunks to fill the interval, but NEVER start a chunk you can't finish+push before the
   next 5-min fire.
3. PHASE 2 (when phase-1 next-slice returns []): gloss the UNGLOSSED, scholar-grounded only.
     wyrd kenning lexicon gloss-campaign next-slice --n 8 --unglossed > /tmp/gloss_unglossed.json
   For each, derive a gloss ONLY from scholar-attributed toponym-etymology evidence / a confident
   etymological reading; PARK over-segmentation fragments and no-consensus morphemes — NEVER invent
   (D51.2 gloss-blind rule; wyrd-dxtn showed naive glossing of this cohort is garbage). [Phase-2
   gloss authoring verb lands in a follow-up; until then, PARK the unglossed cohort and stay on
   phase 1 / phase 3.] Then canonicalize the freshly-glossed via the phase-1 path.
4. PHASE 3 (when 1+2 empty): same as phase 1 with --full-inventory (drops the impact floor to the
   whole glossed morpheme inventory, biggest walls first).
5. SCOREBOARD every fire: wyrd kenning lexicon gloss-campaign status ; append a one-line progress
   note to wyrd-u6fn.10 (run bd from this worktree is fine here — it's the campaign branch).

INVARIANTS (don't relax): remainder is computed from the COMMITTED ledger, never the live DB
(wyrd-1rw4) — next-slice already does this. Grounding: every gloss in a sense must be a REAL row of
that morpheme's wall (validate enforces). Bias to monosemy. Park, don't guess. ONE PR, never merge.
Stay on data/mining/canonicalization/** + data/mining/_sense_parked.jsonl + .beads/issues.jsonl —
do NOT touch _curation.jsonl / _reflexes.jsonl / _tags.jsonl (the eni4 loop owns those; collision-free).
```

## Verbs (the rail, wyrd-u6fn.10.1)

| Verb | Does |
|---|---|
| `gloss-campaign next-slice --n N [--min-impact K] [--unglossed] [--full-inventory]` | Impact-ordered morphemes still needing senses (JSON). Remainder from the committed ledger. |
| `gloss-campaign validate --candidates F` | Gate: ref resolves; every gloss is a REAL wall row; complete partition; short labels; dedup. Exit 0/1. |
| `gloss-campaign author --candidates F` | Validate, then append `mint-canonical` + `same-sense` binds + `canonical-label` to `canonicalization/`. |
| `gloss-campaign status [--min-impact K]` | Committed sense-coverage of the cohort (the per-fire number). |
| `gloss-campaign park --ref R --reason S` | Drop a fragment / no-consensus / ambiguous morpheme from future slices. |

## What lands where (collision-free with the eni4 loop)

- **Sense assertions** → `data/mining/canonicalization/_canonical_nodes.jsonl`,
  `_assert_bind.jsonl` (kind `same-sense`), `_assert_canonical_label.jsonl`.
  The eni4 loop never writes `canonicalization/`, so the two long-lived branches
  don't conflict.
- **Parked** → `data/mining/_sense_parked.jsonl` (campaign-local, net-new file).
- **Phase-2 raw glosses** (when that verb lands) → a campaign-local ledger with a
  DISTINCT `_type`, NOT `etymon_gloss_add` in `_curation.jsonl` (which the eni4
  `gloss-next-slice` owns).

## Why the invariants

- **Remainder from the committed ledger, not the live DB** (wyrd-1rw4): the live
  `~/.wyrd/lexicon.db` carries locally-applied-but-uncommitted state; only the
  committed `_assert_bind.jsonl` is reproducible. `next-slice` reads the ledger.
- **Grounding**: a sense may only group glosses that actually exist on the
  morpheme — the compression can't invent or reword meaning. `validate` enforces.
- **Monosemy bias**: over-splitting a monosemous wall into spurious "shades" is
  the failure the `tūn`→'town' example calls out — the mirror of the
  duplicate-finder's leave-separate default.
- **One PR, no merge**: everything lands on the campaign branch; a human reviews
  and merges on return. The loop never merges.

## Periodic (not every fire)

The readable meaning only materializes after a rebuild runs the projection
(`project_canonical` populates `etymon_gloss.canonical_sense_id` + the labels) —
expensive, so run it at milestones, not per fire. The per-fire `status` number
(committed sense coverage) is the cheap proxy.
