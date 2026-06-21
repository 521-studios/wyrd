# Scholar-morpheme enrichment campaign — loop runbook (wyrd-eni4.1)

A `/loop` ride that raises toponym-decomposition **coverage** by giving the
~8,900 etymons the scholarly breakdowns reference a **reflex** the matcher can
emit/recognize (only 14.6% had one at baseline — the matcher can recover a
morpheme *only* if it has a reflex, which is why recall≈0.205 / coverage≈0.286).

Each 30-minute iteration authors **grounded** reflexes for the next, highest-impact
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
2. **Get work**: `wyrd kenning lexicon enrich-campaign next-slice --n 20 > /tmp/slice.json`
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
