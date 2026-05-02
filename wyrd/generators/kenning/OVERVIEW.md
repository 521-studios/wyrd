# Kenning — overview

This document is the entry point for the kenning sub-system. Read it first;
the others (`DECISIONS.md`, `INGESTION.md`, `README.md`) are leaves attached
to the trunk this doc describes.

If you're a fresh Claude session or a new collaborator, the four-step
context recovery is:

1. **Read this doc.** ~10 minutes.
2. `bd prime && bd ready` — see what's queued.
3. `wyrd kenning lexicon report` — current corpus snapshot.
4. Skim `DECISIONS.md` D-entries that touch whatever you're about to
   change.

`INGESTION.md` is the procedure manual; you only need it when actually
mining a new source.

---

## What this is

Wyrd is a name generator that produces plausible historical-feeling names
on demand. Kenning is the place-name sub-generator (towns, villages,
rivers, hamlets). The end-state user experience is a GM typing something
like

> 5 menacing Welsh village names

and getting names back with the etymology shown:

> **Bryntir** — *bryn* (Welsh, hill — 3 witnesses) + *tir* (Welsh, land — 5 witnesses)

The etymology line isn't decoration. It's the load-bearing thing: the
generator knows which morphemes are well-attested, which language they
belong to, which inflections are valid, and how scholars typically
combine them. Without that, we'd be a Markov chain on syllables.

## The two layers

Kenning has two strict layers, and they're easy to confuse.

```
┌──────────────────────────────────────────┐
│            AUTHORING LAYER               │
│  (where work happens; this DB & code)    │
│                                          │
│   sources/*.txt  ──┐                     │
│                    ▼                     │
│            mining + post-processing      │
│                    │                     │
│                    ▼                     │
│            data/lexicon.db (SQLite)      │
│                    │                     │
└────────────────────┼─────────────────────┘
                     │
              export-meanings
                     │
                     ▼
┌──────────────────────────────────────────┐
│             RUNTIME LAYER                │
│       (what users hit; Lambda)           │
│                                          │
│      data/meanings.json (bundled)        │
│      + per-culture proportions JSONs     │
│      + Generator subclass                │
│                                          │
└──────────────────────────────────────────┘
```

**Authoring layer** is where this project's day-to-day work lives:
mining etymology dictionaries, tracking citations, recording scholarly
disagreement, normalizing OCR. The SQLite DB at
`wyrd/generators/kenning/data/lexicon.db` is the source of truth for
authoring. The CLI commands under `wyrd kenning lexicon ...` operate on
it. The DB is gitignored — it's a build artifact regenerable from
sources + mining.

**Runtime layer** is what users hit. The Lambda generator imports a
bundled `meanings.json` (and per-culture proportions) via
`importlib.resources`, never touches a database, and runs in
milliseconds. Adding a generator should require zero schema changes; the
schema lives in the authoring layer only.

The two are connected by `wyrd kenning lexicon export-meanings`, which
regenerates `meanings.json` from the lexicon. You run it only when you
want a new bundle to ship to production.

This split means: **the authoring DB can be aggressively re-shaped
without touching production.** Most of the schema gymnastics in this
project (D8 lemma layer, D12 search-evidence table, D22 non-destructive
clustering, D23 mining_run audit) are authoring-layer work. The runtime
sees nothing of them until export.

## Why we're mining

The base data is `data/meanings.json` inherited from the Rando port (a
Wikipedia-derived seed). It gives broad lemma coverage — about 1600
morphemes — but unverified per entry. There's no record of which scholar
proposed which etymology, no count of independent witnesses, no
distinction between high-confidence forms and "maybe-from-Old-English-X."

Mining scholarly etymology dictionaries (Mawer, Skeat, Ekwall, Joyce,
Watson, Gillies, etc.) gives us:

- **Independent witnesses per morpheme.** If three different
  19th/20th-century scholars cite *cot* (Old English, "cottage") in
  their place-name dictionaries, we trust *cot* — D4. The "consensus
  view" rolls citations up across all sources sharing a morpheme.
- **Per-source attribution.** Every citation has a source row; you can
  ask "which etymons did Skeat 1901 contribute?" and get a clean answer.
- **Inflectional variants.** Mining surfaces *cotum*, *cotan*, *cotes*
  alongside the lemma *cot*; we link them via `lemma_id` (D8) so the
  generator can compose `<lemma>@<inflection>` for archaic feel.
- **Spelling variants.** Scholars wrote *denu* / *dene* / *denū* / *dená*
  for the same morpheme; we record all four (D18) and the generator
  samples from the variant pool to add archaic-feel variety.
- **Multi-language coverage.** The seed is heavily English. Mining adds
  Celtic (Welsh, Gaelic, Irish, Manx, Pictish), Old Norse, Scottish
  Gaelic — different (language × era) cells per D5.

The promotion threshold is **≥3 independent witnesses**. That's when a
morpheme moves from "trust the legacy data" to "promotion-eligible —
include in the live inventory." Currently ~114 morphemes are
promotion-eligible (out of ~1600 rando lemmas + new lemmas surfaced by
mining). Long-term goal: most of them. Every new mining run nudges that
number up.

## Mental model for mining

Three tiers, three providers, escalating cost-per-call (D2):

1. **Tier 1 (bulk extraction)** — Qwen 3.5 on Ollama for English (free,
   our GPU host); Anthropic Haiku 4.5 for Celtic / Welsh / Irish / Manx
   (~$0.60/book, paid because Qwen underperforms on those — D13). Tier
   1 is what processes thousands of entries per book.
2. **Tier 2 (review)** — Gemini 2.5 Flash on Qwen-mined low/medium-
   confidence rows. Doesn't replace the Qwen row; writes a parallel
   row tagged `extracted_by:gemini:` so the consensus view sees both.
   This is the standard second pass on English mining, not a contingency.
3. **Tier 3 (rare review)** — Anthropic on whatever Gemini still
   declines. Per D19 the Sonnet uplift over Gemini Flash for mining is
   empirically ~zero, so this tier mostly stays unused. Reserved for
   user-facing runtime features (explainer, register-shift) where
   quality genuinely lifts.

Then a separate **post-mining chain** (LLM-free, idempotent, cheap):
`link-lemmas → normalize-ocr → reverse-search → fuzzy-search → report`.
This consolidates inflectional variants, clusters OCR-mangled spellings,
finds where canonical forms appear in source body text, and fuzzy-matches
gloss-anchored variants.

The cost asymmetry between mining and post-processing is the project's
core scaling principle: **mining is expensive (hours, dollars);
enrichment is cheap (seconds).** Optimize the design so iterating on
enrichment heuristics costs almost nothing while mining stays the only
slow / costly operation.

## Why the schema looks the way it does

Three guiding principles, all subordinate to *"don't lose mining
evidence":*

- **Mining evidence is sacred (D21).** Citations, glosses, tags, and
  text-match rows live on the etymon they were originally attached to
  and never get moved or destroyed without an explicit
  `clear-enrichment` call. Even when two etymons get OCR-cluster-merged,
  the loser keeps its evidence; the consensus view rolls it up via the
  redirect column.
- **Enrichment is reversible (D22).** OCR clustering uses a
  `merged_into_id` self-FK redirect rather than DELETEing rows. To
  undo a clustering pass: `wyrd kenning lexicon clear-enrichment
  --stage=ocr`. Re-running with new heuristics is a one-liner. Same
  shape for `lemma_id` linkage and `etymon_text_match` rows; each has
  a clear-enrichment stage.
- **Operations are audited (D23).** Every mining run writes a row to
  `mining_run` with accept/decline/reject counts, provider/model, and
  timestamps. Every enrichment row carries a method/version stamp
  (`lemma_method = 'link-lemmas-v1'`,
  `etymon_text_match.method = 'reverse-search-v1'`, etc.) so a future
  v2 heuristic can selectively rebuild only its predecessor's work.

If you find yourself wondering "should I add a feature that mutates
mining data?", the answer is almost always: **add a redirect column or
a method stamp; never overwrite the original row.** The schema's
quirks all flow from that one rule.

## What we're NOT doing

A few non-goals to keep in mind:

- **No Native American, Aboriginal Australian, Maori, Hawaiian, or
  Khoisan corpora (D7).** PD status is a legal property; what we honor
  here is the ethical one. We provide the language-pack format; living
  communities supply the data on their own terms.
- **No bundled commercial pantheons or IP-restricted material (D11).**
  Tolkien languages, D&D pantheons, Pathfinder pantheons, homebrew
  worlds: pluggable via YAML packs the user supplies. We don't bundle.
- **No "ethnic = bad" / "ethnic = good" coding (D6).** Slavic isn't
  orcish. Norman-French isn't villainous. Old Norse isn't dwarven.
  Languages are morally-neutral palette options; the dark/menacing
  effect comes from `--mood harsh` (phonological) and `--mood grim`
  (semantic-tag) filters that apply to *any* language.
- **No Sonnet for mining (D19).** Tested empirically; lifts ~zero over
  Gemini Flash. Reserve Anthropic budget for runtime user features.

## Where we are right now

For the live snapshot, run:

```bash
wyrd kenning lexicon report --top 25
wyrd kenning lexicon stats
```

Plus the mining_run audit table for "what's been mined and how it went":

```sql
SELECT source_id, provider || '/' || model AS tier, mode,
       SUM(parsed_count) AS parsed,
       SUM(accepted) AS acc, SUM(declined) AS dec, SUM(rejected) AS rej
FROM mining_run
GROUP BY source_id, provider, model, mode
ORDER BY source_id;
```

The shape today (project age: ~3 days) is roughly: ~40+ sources mined,
~5400+ etymon rows, ~4500+ etymology extractions, **389+ morphemes
promotion-eligible at ≥3 witnesses** (more at the per-language preset).
Several methodology / name-list books contribute few or zero rows by
design (Quilgars 1906 was the latest such case — see INGESTION.md
"low-yield book class" note for the topographical-vs-toponymic
distinction).

The bundled `meanings.json` is now exported from the lexicon (D1
follow-through), so all the mining + per-reflex narrowing + per-language
thresholds work reaches the runtime. Current bundle: **1616 subjects**,
**~2900 unique modern_usages**, with **444 morphemes carrying spelling
variant pools (D18)** and **301 inflected etymons (D8)** across 9
case labels (Bannister 1916 Herefordshire mining surfaced ~130 new OE
inflections).

The runtime exposes four GM-facing generation knobs, all defaulting
to off / 0 (bit-stable historical behavior):

- `--novelty` (D17): blend empirical-frequency sampling with a uniform
  marginal — high values let plausible-but-unattested combinations
  through.
- `--spelling-variety` (D18): per-morpheme probability of substituting
  an attested archaic spelling for the canonical reflex.
- `--inflection-density` (D8): per-morpheme probability of substituting
  an inflected form (genitive/dative/plural) for the lemma.
- `--mood` (D6, repeatable): stylistic-mood preset. `grim` applies a
  menacing semantic-tag union (death, military, monster, undead,
  magic); `harsh` biases sampling toward stop-final / cluster-heavy
  morphemes; `harsh:0.5` graduates the skew via colon-suffix. Multiple
  flags compose by tag-union and max-harshness.

Five cultures: `english`, `scottish`, `welsh`, `irish`, **`breton`**.
The breton register was added with a 1214-commune corpus pulled from
Wikidata (CC0); morpheme corpus expansion still pending — wyrd-fmg.

## Pointers to the other docs

- **`DECISIONS.md`** — D1–D26, the architectural decisions and their
  rationale. Read individual entries when you're about to change
  something they touch. Don't try to read all of it linearly.
- **`INGESTION.md`** — the procedure manual: how to add a new source,
  smoke-test the parser, pick a tier, run mining, run the post-mining
  chain, verify, ship. You only need it when actually mining.
- **`README.md`** (in this directory) — short generator-side overview
  for the runtime perspective: how the Lambda generator works, what
  `meanings.json` contains, how proportions are computed. Lighter than
  this doc and oriented at the runtime layer rather than authoring.
- **`bd`** (beads) — issue tracker. `bd ready` shows what's queued;
  `bd memories` carries persistent knowledge across sessions. Use bd
  for ALL task tracking; never use TodoWrite or markdown TODO lists.

## Common questions, fast answers

**"What's the state of book X?"** —
`SELECT * FROM mining_run WHERE source_id = 'X'` plus
`SELECT COUNT(*) FROM toponym_etymology WHERE source_id = 'X'`.

**"Which books need re-mining via Haiku?"** — see INGESTION.md
"worst-yield-on-Qwen probe" query. Anything under ~30% accept on
≥50 parsed entries is a candidate.

**"Has Tier 2 review run on book X?"** —
`SELECT * FROM mining_run WHERE source_id = 'X' AND mode = 'review'`.

**"What's promotion-eligible?"** — `etymon_consensus` view; rows with
`witnesses >= 3` are promotion-eligible.

**"Where's the original FK fix the wyrd-go5 ticket talks about?"** —
superseded by wyrd-et0; the fix lives at lexicon.py
`cluster_ocr_variants` as the `merged_into_id` redirect path.

**"How do I undo a clustering pass?"** —
`wyrd kenning lexicon clear-enrichment --stage=ocr --apply`.

## Process notes for future-Claude

A few hard-earned lessons specific to operating this project:

- **Trust the audit table over agent summaries.** When an agent
  summarizes mining stats from transcripts or other indirect sources,
  verify the numbers against `mining_run` before propagating them into
  docs. (Concrete example: 2026-05-01 the Watson 1904 "Haiku 6× yield"
  claim from a transcript-mining agent was wrong; actual data showed
  3% Haiku vs 7% Qwen. INGESTION.md briefly carried the wrong
  empirical exception before being corrected.)
- **Date-anchor before narrating.** This project started 2026-04-30.
  "Months of mining history" is wrong; it's all 2 days. Check
  timestamps in `mining_run.completed_at` or git log before writing
  anything historical.
- **When you add data, add an observability path (D24-pending).** Every
  new persistent operation should be queryable from the DB. If you
  catch yourself printing accept/decline/reject counts to stdout
  without a corresponding write — stop, you're recreating wyrd-ej4.
- **The handoff path is the DB, not a /tmp file.** The 4-step recovery
  at the top of this doc plus `wyrd kenning lexicon report` should give
  any new session enough context to act. Transient handoff docs should
  be a last resort, not a default.
