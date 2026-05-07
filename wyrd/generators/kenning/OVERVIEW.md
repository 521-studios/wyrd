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
│            ~/.wyrd/lexicon.db (SQLite)   │
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
disagreement, normalizing OCR. The SQLite DB at `~/.wyrd/lexicon.db`
(override with `WYRD_LEXICON_DB`) is the source of truth for authoring
— one per user, repo-/worktree-independent. The CLI commands under
`wyrd kenning lexicon ...` operate on it; `wyrd kenning lexicon path`
prints the resolved location. Never committed; regenerable from
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

## Sibling pipeline: wyrd-ami fantasy-name research

Kenning hosts a second mining pipeline alongside the place-name one:
**wyrd-ami** researches the etymology of fantasy / gaming creature
names so the same lemma layer that drives town generation can also
generate fantasy-flavored content with verifiable etymological
provenance.

Where the place-name pipeline mines scholarly etymology dictionaries
to populate the lexicon, wyrd-ami works the other direction: it takes
a creature name (Harpy, Bugbear, Djinni, Tiamat) and routes it through
two stages to find the underlying historical morpheme:

1. **Descent-walking pre-filter.** Look up the input in the existing
   `etymon` table; if it resolves cleanly to an attested etymon in an
   approved language, mark the row usable and skip the LLM. Common
   inputs (Harpy → ancient-greek ἅρπυια, Troll → ON trǫll) take this
   fast path.
2. **Gemini Flash full-research.** When the pre-filter misses, the
   LLM is asked to identify the historical attested form, language,
   and citation. A semantic-check pass on borderline pre-filter hits
   uses the same LLM. Results land in `fantasy_morpheme` with
   `bar_reason` distinguishing the failure modes (modern_coinage,
   outside_language_family, attested_but_not_in_corpus,
   uncertain_attestation, no_etymology_found, homograph_collision,
   proper_noun_only).

The canonical input source is **pfsrd2-data** (~/521Studios/pfsrd2-data),
the Pathfinder 2 SRD bestiary as structured JSON. The
`extract-pfsrd2-monsters` CLI walks that corpus and emits a JSONL of
`{name, description}` records. Each monster contributes up to two
records: the family root (Genie, Bugbear, Demon — deduped across
variants) and the monster's own single-word name when distinct from
the family root (Djinni, Efreeti, Marid under Genie). Multi-word
variant names are dropped; "Ancient Black Dragon" with no family
field is the canonical "no clean morpheme" case.

Approved languages live in `fantasy_pipeline.APPROVED_LANGUAGES` —
keep it dashed-lowercase for descriptive names ('old-english',
'ancient-greek') and ISO-code form for languages where the etymon
table uses ISO codes ('he', 'ar', 'fa', 'sa', 'akk', 'egy', 'arc',
'pal'). The `_LANGUAGE_ALIAS_MAP` normalizes LLM output ('sanskrit' →
'sa', 'arabic' → 'ar', 'ancient-egyptian' → 'egy') so descriptive
names from the LLM resolve against ISO etymon rows.

`approach_version` (in `fantasy_pipeline.APPROACH_VERSION`) is a
pipeline-version stamp: the (input_name, approach_version) UNIQUE
on `fantasy_morpheme` lets a stronger pipeline re-process all rows
by bumping the version. The `--skip-resolved` flag on
`mine-fantasy-name` lets you grow the input corpus and re-run without
paying for already-resolved entries (case-insensitive match — the
column uses COLLATE NOCASE).

**Output:** every usable row links to an `etymon_id` and inherits its
descent chain. So Harpy resolves to ancient-greek ἅρπυια and the chain
ancient-greek → latin → middle-french → middle-english → modern-
english is available, letting town-name generation pick a register
(Harpyia for ancient-mythological, Harpie for medieval, Harpy for
modern). The same temporal-axis trick works for any usable wyrd-ami
morpheme.

Procedure for running a fresh mine — `INGESTION.md` "Mining the
wyrd-ami fantasy-name corpus".

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

The shape today (project age: ~4 days) is roughly: **46+ sources mined,
~77K etymon rows, ~5,200 toponym etymology extractions, ~7,800 citations,
403 morphemes promotion-eligible at ≥3 witnesses** (more at the per-
language preset). Several methodology / name-list books contribute few
or zero rows by design (Quilgars 1906 and Joret 1881 were the latest
such cases — see INGESTION.md "low-yield book class" note for the
topographical-vs-toponymic and dialectical-vs-toponymic distinctions).

**Recent corpus expansion (2026-05-03)**: three new Romance / French-
substrate sources added — Longnon 1920 vol 1 (16 acc, 27 etymons),
d'Arbois de Jubainville 1890 'Recherches sur l'origine de la propriété
foncière (période celtique et période romaine)' (127 acc, 219 etymons
post-wyrd-z56-remine), and Mawer 1920 Northumberland-Durham re-mined
via Gemini Flash 2.5 as Tier-1 (501 acc, 714 etymons, independent 3rd-
model witness layer). Net: +644 acc rows / +960 etymons touched / +3
new lemmas crossing the D4 ≥3-witness promotion threshold.

The bundled `meanings.json` is now exported from the lexicon (D1
follow-through), so all the mining + per-reflex narrowing + per-language
thresholds work reaches the runtime. Current bundle (post-2026-05-04
re-emit, PR #58): **1697 subjects**, **3807 words**, with **298 morphemes
carrying spelling variant pools (D18)**, **306 inflected etymons (D8)**
across 9 case labels, **1669 morphemes with scholarly citation
metadata (wyrd-9kh.1)**, and **1215 morphemes (31.9%) carrying
attested_year data (D5-1)** that activates the runtime `--era`
filter (D5-3).

The runtime exposes six GM-facing generation knobs, all defaulting
to off / 0 (bit-stable historical behavior):

- `--novelty` (D17): blend empirical-frequency sampling with a uniform
  marginal — high values let plausible-but-unattested combinations
  through.
- `--cohesion` (D17 refinement, wyrd-mj2): the OPPOSITE direction
  from novelty — bias each slot toward usages whose tags co-occur
  with previously-picked slots' tags in the empirical corpus. Composes
  orthogonally with novelty so a GM can dial 'attested-pair fidelity'
  and 'novelty' independently. No-op when the bundle carries no
  tag_cooccurrence data.
- `--era` (D5-3, wyrd-lyp): restrict morpheme inventory to forms
  attested in a particular period. Accepts year (`1086`), cell label
  (`oe-late`), or `family/label` (`english/oe-late`). Active in
  production as of the 2026-05-04 bundle re-emit (PR #58 / wyrd-j5v):
  31.9% of bundle words carry `_attested_years` data, and the filter
  empirically narrows the English keep-set from 2901 → 2327 (oe-early)
  / 2433 (oe-late) / 2690 (me) / 2323 (early-modern) / 2246 (modern).
  The keep-set collapses to None when the era covers every usage, so
  the filter is bit-stable with no-filter on coverage gaps and
  short-circuits the per-bucket walk back to the historic fast path.
- `--stratum` (D32, wyrd-lr4): restrict morpheme inventory to forms
  classified into a within-language register bucket. Per-family
  vocabularies — Welsh: `native-welsh / latin-loan / english-loan /
  brittonic-substrate / medieval-welsh`; French: `native-french /
  frankish-substrate / gaulish-substrate / gallo-roman /
  medieval-french`; Old English: `native-old-english / latin-loan /
  norse-loan / celtic-substrate`; Old Norse: `native-old-norse /
  east-norse / latin-loan / low-german-loan / english-loan /
  gaelic-substrate`. Validation is per-culture: SPA dropdown and
  CLI both reject culturally-incoherent values
  (`--culture welsh --stratum east-norse` 4xxs at request time).
  Composes with `--era` via frozenset intersection on the
  per-bucket keep-set. Bit-stable with no-filter until a bundle
  re-emit populates the per-language `_stratum` siblings;
  classifier output captured in the wyrd-lr4 ticket notes (Welsh
  31,067 etymons / French 21,782 / OE 70,151 / ON 18,977). Plus
  `lexicon set-stratum` for operator hand-corrections that survive
  subsequent `classify-stratum --apply` runs (D32 idempotency
  contract).
- `--spelling-variety` (D18): per-morpheme probability of substituting
  an attested archaic spelling for the canonical reflex.
- `--inflection-density` (D8): per-morpheme probability of substituting
  an inflected form (genitive/dative/plural) for the lemma.
- `--mood` (D6, repeatable): stylistic-mood preset. Five entries today —
  `grim` (death/military/monster/undead/magic), `harsh` (phonological
  stop-final/cluster-heavy bias), `pastoral` (plant/animal/water/
  agriculture/tree/bird), `devotional` (saint/religious), `mortuary`
  (death/undead — narrower subset of grim). `harsh:0.5` graduates the
  phonological skew via colon-suffix. Multiple flags compose by
  tag-union and max-harshness. Lives in `__init__.MOODS`.

**Recent infrastructure (2026-05-04 / 2026-05-05)**: meaning_synset
layer (wyrd-7tz Phase 1, PR #61) — semantic-equivalence catalog for
upcoming generator transforms (calque, anglicize, drift-toward-X);
53 seed synsets covering high-frequency place-name semantic classes;
Phase 2 (LLM-assisted etymon classification of the production
lexicon) and Phase 3 (transform integration) deferred. Plus the
trie-indexed segmentation DAG matcher (wyrd-k8e Phase 1, PR #64) —
foundation for wyrd-08m (decomposition multiplicity) and wyrd-cv3
(corpus DB ingest); standalone in `trie_matcher.py` for now, Phase 2
wires it into `Name.find_meaning`. And the etymon.synset_id →
etymon.cognate_id rename (wyrd-44a, PR #63) cleared the naming
collision between cognate-cluster IDs and meaning_synset IDs. The
numbered-list parser (wyrd-5af partial, PR #65) handles ordinal-
prefixed treatise sections (Longnon vol 2 saint-names: 7× lift on
parsed-entry count over the alphabetical parser); the prompt-side
sub-feature (custom schema for "Latin → French Saint-X" terse
derivations) is deferred — empirical 30-entry smoke against the
default English-etymology prompt yielded 3.3% accept. The
`lexicon review --include-haiku` flag (wyrd-eca, PR #66) widens the
Tier-2 candidate query to also pick up Haiku-Tier-1-mined rows;
unblocks Gemini Tier-2 review on non-Celtic Haiku-mined books
(Bannister, Moorman, Duignan, etc.). Plus the parallelized
`mine-llm --concurrency N` (wyrd-l0r, PR #62) and the
filter_for_tag PYTHONHASHSEED bit-stability fix (wyrd-8ga, PR #60).

Five cultures: `english`, `scottish`, `welsh`, `irish`, **`breton`**.
The breton register was added with a 1214-commune corpus pulled from
Wikidata (CC0); morpheme corpus expansion still pending — wyrd-fmg.

## Pointers to the other docs

- **`DECISIONS.md`** — D1–D30, the architectural decisions and their
  rationale. Read individual entries when you're about to change
  something they touch. Don't try to read all of it linearly. Latest
  additions: D27 (etymological descent graph), D28 (cognate vs
  meaning_synset axes), D29 (trie-indexed segmentation DAG matcher),
  D30 (wyrd-ami fantasy-name pipeline as sibling); D5-3 and D17 have
  refinements covering the era runtime filter (wyrd-lyp) and cohesion
  knob (wyrd-mj2).
- **`INGESTION.md`** — the procedure manual: how to add a new source,
  smoke-test the parser, pick a tier, run mining, run the post-mining
  chain, verify, ship. You only need it when actually mining.
- **`README.md`** (in this directory) — short generator-side overview
  for the runtime perspective: how the Lambda generator works, what
  `meanings.json` contains, how proportions are computed. Lighter than
  this doc and oriented at the runtime layer rather than authoring.
- **`COVERAGE.md`** — rolling log of place-name decomposition rate
  per culture. Append a snapshot after every bundle re-emit so the
  trend stays visible. Coarse "surface morphemes recognized" gauge,
  not the north star.
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
