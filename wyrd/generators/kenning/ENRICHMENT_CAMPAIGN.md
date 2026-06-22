# Scholar-morpheme enrichment campaign — loop runbook

## CURRENT MODEL — push CAN-IT, reflex-first (wyrd-eni4.3)

**North star: CAN-IT → ~100%.** CAN-IT is the fraction of scholar toponyms whose
breakdown the decomposer can GENERATE (recall=1 over the candidate list, before
selection — `grade-decomposition --can-it`). It is the inventory ceiling, and it
is the loop's job to raise it. A toponym is CAN-IT only when **every** one of its
scholar morphemes has a matchable reflex, so the lever is: **give reflex-less
scholar morphemes a grounded reflex.** (DID-WE — whether the matcher *selects*
that parse — is the human's separate post-trip problem; the loop NEVER touches
the scorer/tiebreak.)

Baseline at launch (after-bundle): CAN-IT 28.8% / DID-WE 12.3% (recall=1). Of the
43,361 scholar (toponym,morpheme) pairs: 56% matched, **21% no-reflex** (this
loop's lever), 22% variant-gap (a separate lever — see wyrd-eni4.3 follow-up).
6,316 of 8,939 distinct scholar morphemes still have **no reflex** → ample runway.

**Priority order (per the campaign owner):**
1. **Reflex first** — the CAN-IT lever. Author grounded reflexes for reflex-less
   scholar morphemes, **impact-ordered: the morphemes in the MOST toponyms first,
   working down.** This is exactly what `next-slice` emits (impact-descending).
2. **Then the quality metrics** — tag → IPA → gloss — on the same impact-ordered
   morphemes, raising the covered morphemes' quality.

**Loop-owned dimensions** (all author to existing L2 ledgers, all gated):
- **reflex** → `_reflexes.jsonl` (grounding guard) — **PRIMARY (CAN-IT)**
- **tag** → `_tags.jsonl` (controlled `TAG_VOCAB`, "none" allowed)
- **IPA** → `_pronunciation.jsonl` (`/.../`-delimited)
- **gloss** → `_curation.jsonl` (`etymon_gloss_add` rows; only the ~5% missing)

**Human-gated, NEVER author in the loop:** lemma-wiring, cognate clustering,
variant pool, AND parse selection (the DID-WE scorer/tiebreak). Structural /
decision-gated (wyrd-740t, over-merge bugs).

### Per-fire procedure (cron fires every 5 min)

**Prompt-cache window is controlled by the INTERVAL, not by padding work.**
Measured reality (git timestamps): a fire's wall-clock is only ~2–3 min even
when it touches all four dimensions — the work is fast LLM+bash. So you CANNOT
keep the conversation in Anthropic's prompt cache (5-min TTL) by "filling time";
the idle gap = `interval − fire_duration`, and the lever is a short interval.
The cron is therefore every **5 min**: with ~2.5-min fires the gap is ~2.5 min
(< 5), and if a fire ever runs long the next just fires on idle (~0 gap). Idling
past ~5 min evicts the cache and re-bills the full context, so keeping the gap
small is a **cost** rule. Do a normal-sized fire (one reflex slice + a quality
batch or two) — only backfill extra dimensions when a slice comes back empty/tiny.

**Emit timestamps every fire** (so the < 5-min idle goal is verifiable): run
`date '+%F %T %Z'` as the FIRST and LAST action of the fire, append
`START<TAB>FINISH<TAB>summary` to `data/mining/campaign_timing.log` (committed),
and report START/FINISH/elapsed. The idle gap = this fire's FINISH → the next
fire's START.

0. `date '+%F %T %Z'` → ITERATION START (report it).
1. `git pull --rebase`.
2. **Reflex (primary):** `enrich-campaign next-slice --n N` (impact-ordered, most
   toponyms first) → author grounded reflexes to a temp file, park un-groundable
   ones → `enrich-campaign validate` (exit 0 or fix/drop) → append
   `data/mining/_reflexes.jsonl` → commit + push. Repeat sets while groundable.
3. **Quality (secondary):** once the reflex set for this fire is in, spend
   remaining time on quality, impact-ordered: `tags-next-slice` → classify
   (empty=none) → `tags-validate` → `_tags.jsonl`; then `ipa-next-slice` →
   `/IPA/` → `ipa-validate` → `_pronunciation.jsonl`; then `gloss-next-slice`
   (only the ~5% truly missing a gloss) → `gloss-validate` → `_curation.jsonl`.
   Commit + push each.
4. **Track progress (cheap, every fire):** `enrich-campaign status` (reflex
   ledger coverage) — the per-iteration scoreboard. The REAL CAN-IT number
   (`grade-decomposition --can-it`) reads the runtime bundle, which only reflects
   authored reflexes **after a rebuild**, so run it only at **milestones** (after
   a `pf2_build`/rebuild replays the ledger), not every fire.
5. Update wyrd-eni4.3 notes; keep everything on PR #727.

6. `date '+%F %T %Z'` → ITERATION FINISH; append the timing line; report
   START/FINISH/elapsed.

Stop the whole campaign only when `next-slice` AND all three quality selectors
(tags/IPA/gloss) return empty.

The old reflex-only "Phase ladder" below is **superseded**; its validation
gates, ledger invariants, and "never invent / park instead" rules still apply.

---

# (superseded) reflex-only runbook (wyrd-eni4.1)

A `/loop` ride that raises toponym-decomposition **coverage** by giving the
~8,900 etymons the scholarly breakdowns reference a **reflex** the matcher can
emit/recognize (only 14.6% had one at baseline — the matcher can recover a
morpheme *only* if it has a reflex, which is why recall≈0.205 / coverage≈0.286).

Each iteration authors **grounded** reflexes for the next, highest-impact
batch of reflex-less scholar etymons, validates them, and commits to one PR.

## The one rule that keeps this safe unattended

**Never invent a reflex surface.** A reflex `surface_form` must be a span that
actually appears in an attested toponym for that etymon (e.g. `old-english:tūn`
→ `ton` because of New**ton**/Thorn**ton**; `Bartholomew` → `barthol` only if a
real place-name shows it). The `validate` gate enforces this (the *grounding
guard*) — but author to it, don't fight it. If you can't find a confident
grounded surface, **park** the etymon; don't guess.

## Per-iteration steps

Run from the campaign worktree (`/home/devon/521Studios/wyrd-campaign`), branch
`enrichment/scholar-morpheme-campaign`. `wyrd` below = `python -c "from wyrd.cli import main; main()"`.

1. **Sync**: `git pull --rebase` (the branch only ever moves forward via this loop).
2. **Get work**: `wyrd kenning lexicon enrich-campaign next-slice --n 40 > /tmp/slice.json`
   - **Slice size (`--n`) self-tuning:** the cron fires every **15 min** (900s);
     size each slice to fill **~2/3 of the interval (~10 min)** of useful work.
     `--n 40` is the phase-1 (reflex) default. When a phase changes — or an
     iteration finishes well under/over ~10 min — re-evaluate and adjust `--n`
     (bump up if fast, ease off if long). Tag-classification work per etymon is
     lighter than reflex authoring, so phase 2 will likely want a larger `--n`.
   - Empty array ⇒ **phase 1 (reflexes) is exhausted** (every remaining etymon
     done or parked). Note phase-1 complete on **wyrd-eni4.1.2**, then
     **advance to phase 2 (tags)** — see "Phase ladder" below. Do NOT stop.
3. **Author**: for each etymon in the slice, write reflex rows to
   `/tmp/cand.jsonl`, one JSON object per line:
   ```json
   {"_type":"reflex","surface_form":"<span>","position":"pre|post|inner","etymon_refs":["<ref copied verbatim from the slice>"]}
   ```
   - **Grounding source, in order:** (a) `surface_in_modern` on the etymon's
     toponyms (the scholar already recorded the realized span — use it directly);
     (b) otherwise align the morpheme to the `modern_name` / `historical_form`
     spans yourself and take the observed span.
   - **Position:** `pre` if the span sits at the name's start (Stainton→`stain`),
     `post` if at the end (New**ton**→`ton`), `inner` if medial.
   - Author every *distinct attested* surface (a morpheme often has several:
     `ton`/`tone`/`don`). One row per (surface, position).
   - **Park the rest**: for any etymon with no confident grounded surface
     (foreign gloss-heads, abstract elements), record it so it leaves the queue:
     `wyrd kenning lexicon enrich-campaign park --ref "<ref>" --reason "<one line>"`
4. **Validate**: `wyrd kenning lexicon enrich-campaign validate --candidates /tmp/cand.jsonl`
   - Must exit 0. On failure, FIX or drop the offending rows and re-validate —
     never bypass the gate. (A failure is the guard doing its job.)
5. **Append**: concatenate the validated `/tmp/cand.jsonl` onto
   `data/mining/_reflexes.jsonl` (append only — never rewrite existing rows).
6. **Commit + push** on the campaign branch (only `data/mining/_reflexes.jsonl`
   and `data/mining/_reflex_parked.jsonl` change in a normal iteration):
   `git add data/mining/_reflexes.jsonl data/mining/_reflex_parked.jsonl && git commit && git pull --rebase && git push`
7. **Update PR + ticket**: keep the single PR's body current; append progress to
   **wyrd-eni4.1.2** notes from
   `wyrd kenning lexicon enrich-campaign status`.

## Phase ladder (what runs, in order)

The loop works **phase 1 to completion first, then advances** (wyrd-eni4.2 is
the staged-second track):

1. **Phase 1 — reflexes** (steps above). Run until `next-slice` is empty.
2. **Phase 2 — tags** (wyrd-eni4.2.3). Same loop shape, different verbs:
   - `enrich-campaign tags-next-slice --n 20` → each etymon carries its glosses
     + the controlled `vocab`.
   - Classify each gloss into that vocab. Write rows
     `{"_type":"tags","ref":"<ref>","tags":["..."],"confidence":"high","method":"claude-tags-v1","model":"opus"}`
     to `/tmp/tags.jsonl`. **Use ONLY vocab tags; an empty `tags: []` is the
     valid "none" outcome for genuinely-abstract glosses — no tag sprawl, no bad
     matches.**
   - `enrich-campaign tags-validate --candidates /tmp/tags.jsonl` (must exit 0),
     then append to `data/mining/_tags.jsonl`, commit + push, update
     **wyrd-eni4.2.3** from `enrich-campaign tags-status`.
   - When `tags-next-slice` is empty, phase 2 is done → **stop the loop.**
3. **Human-gated, NOT auto-entered:** variants/D18 spelling pool (wyrd-eni4.2.2,
   blocked on the wyrd-740t tag-convention decision) and cognate binds
   (wyrd-eni4.2.4, active over-merge bugs). The loop must **not** author these
   unattended — leave them for a human session.

## Why these invariants (don't relax them)

- **Remainder from the committed ledger, not the live DB.** `next-slice`
  computes "still needs a reflex" from committed `_reflexes.jsonl`, because the
  working `~/.wyrd/lexicon.db` carries locally-applied-but-uncommitted reflexes
  — a live-DB remainder is wrong and non-reproducible (wyrd-1rw4).
- **Safe scoped path only.** Author reflexes as `_reflexes.jsonl` rows. NEVER run
  `mine-implied-reflexes --apply` / `dump_all_sources` here — it's blocked by the
  wyrd-tg0f durability bug (drops 7,327 attestation rows on rebuild).
- **One PR, no merge.** Everything lands on the campaign branch / single PR; a
  human reviews and merges on return. The loop never merges.

## Periodic (not every tick)

A full rebuild + `grade-decomposition` is the only way to see coverage actually
move; it's expensive (multi-GB DB), so run it ~nightly (or let the operator run
it), not per iteration, and record the recall/coverage delta on wyrd-eni4.1.
The per-tick `status` number (committed reflex coverage of the scholar set) is
the cheap proxy to watch in between.
