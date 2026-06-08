# Glossary

A reference for the vocabulary that runs through the wyrd codebase — the
kenning place-name generator in particular. Linguistics terms, pipeline
and architecture terms, the LLM-provider tiers, and the runtime
generation knobs.

Pointers like *(D17)* refer to entries in
[`wyrd/generators/kenning/DECISIONS.md`](../wyrd/generators/kenning/DECISIONS.md);
*OVERVIEW* / *INGESTION* / *README* / *L2_L3_BOUNDARY* / *REBUILD* refer
to the sibling docs in that same directory.

---

## Linguistics terms

### Morpheme
The smallest meaningful unit of a name — a root or affix that carries a
sense and a language. `tūn` ("enclosure/farmstead"), `brycg` ("bridge"),
`-ham` ("homestead"), `llan-` ("church") are morphemes. A place name is a
**compound** of morphemes; decomposition is the act of splitting a
surface name back into them. In this codebase a morpheme corresponds to
an `etymon` row (or a [lemma](#lemma) when inflected variants are rolled
up). Note: morpheme **position** (which slot it fills — pre/inner/post)
is *derived* from where it matched in the surface string, never a match
criterion — the matcher matches by string only.

### Lemma
The canonical, uninflected dictionary form of a morpheme — the headword
you'd look up. `cot` is the lemma; `cotan`, `cotum`, `cotes` are its
**inflected forms** (see [inflection](#inflection)). Inflected `etymon`
rows point at their lemma via a nullable `lemma_id` self-FK, so the
[consensus](#consensus) view can roll up all witnesses to the one
morpheme instead of undercounting it as 3–5 separate rows. Set by the
`link-lemmas` enrichment pass (D8).

### Toponym
A place name — the thing being generated or decomposed (town, village,
river, hamlet). "Cambridge", "Llanfair", "Athlone" are toponyms. The
authoring DB stores real attested toponyms (`toponym` table) with their
scholarly etymological breakdowns (`toponym_etymology` →
`toponym_etymology_element`); the generator invents new plausible ones
from the morpheme inventory.

### Etymon
The historical source-form a morpheme descends from, tagged with its
language (`old-english:tūn`, `welsh:bryn`). The central unit of the
authoring lexicon — the `etymon` table is where mining evidence
accumulates. One etymon can collect [citations](#citation) from many
scholars; that's what builds [consensus](#consensus). Plural: *etyma* or
*etymons* (the code uses "etymons").

### Reflex
The modern (or later-era) surface form that an older etymon shows up as.
OE `tūn` has the modern-English reflex `town`; the same etymon renders
differently at each historical [stratum](#stratum-register) /
[era](#era). Reflexes are what let the generator render one morpheme
across time (Harpyia → Harpie → Harpy). Stored in `reflex` /
`reflex_etymon`; era-specific reflexes are resolved by the
`etymon_era_reflexes()` primitive (D33).

### Phonological / phonology
Pertaining to the *sounds* of a name (as opposed to its meaning). The
`--mood harsh` effect is phonological — it reweights morphemes by
consonant-cluster density, stop-finality, and the like, independent of
what they mean. Contrast with **semantic** (meaning-based) effects like
`--mood grim`, which filter by tags (death, military, …). The scholarly
grounding for each phonological dimension lives in
[`REGISTERS.md`](../wyrd/generators/kenning/REGISTERS.md). See also
[phonaesthetics](#phonaesthetics).

### Phonaesthetics
Sound symbolism — the (sometimes universal, sometimes
culture-conventional) tendency for certain sounds to *feel* a certain
way: plosives read sharp/threatening, sonorants read soft/pleasant, high
front vowels read small/ethereal (Ohala's frequency code). The
register-effect catalog encodes this into per-dimension weights;
`REGISTERS.md` annotates each weight as *universal*, *IE-conventional*,
or *identity-marking*.

### Inflection
A grammatical variant of a lemma — genitive, dative, plural, etc. OE
`cot` → `cotes` (genitive), `cotum` (dative/plural). Mining surfaces
inflected forms alongside the lemma; the generator can substitute one
for archaic feel via `--inflection-density` (D8). When a substitution
contest arises with a [spelling variant](#spelling-variant), inflection
wins (it carries grammatical case).

### Spelling variant
An attested *spelling* of the same morpheme — `denu` / `dene` / `denū` /
`dená` — from the era before orthography was standardized. Not a
grammatical change (that's [inflection](#inflection)), just a different
written form. The generator samples from the variant pool via
`--spelling-variety` (D18). Pool comes from `etymon_text_match` rows.

### Cognate
Two morphemes that descend from a common ancestor — etymologically
related, even if their meanings have since diverged. OE `tūn`, ON `tún`,
Icelandic `tún`, modern English `town` are cognates (all from
Proto-Germanic `*tūnaz`). Materialized as a shared `etymon.cognate_id`.
Distinct from a [meaning synset](#meaning-synset): cognate = *same
origin*, synset = *same meaning*. (D27, D28)

### Meaning synset
A set of morphemes that share a *sense*, regardless of etymology. OE
`wæter`, OE `strēam`, ON `bekkr` all belong to the `water/flowing`
synset even though they aren't [cognates](#cognate). Drives meaning-based
transforms (anglicize, calque, drift-toward-X). The two axes — cognate
(etymological) and synset (semantic) — are orthogonal (D28).

### Descent / descent graph
The directed graph of *which morpheme came from which* —
`etymon → parent_etymon → grandparent`, with edge types (`inheritance`,
`borrowing`, `calque`, `derivation`, `cognate`, `compound`, `unknown`).
A separate axis from toponym decomposition; lives in `etymon_descent`.
Populated mainly from Wiktionary/wiktextract. The transitive closure of
inheritance + borrowing edges defines a [cognate](#cognate) cluster.
(D27)

### Era
The time axis of the register model — the historical period a form is
attested in: `oe-early`, `oe-late`, `me` (Middle English), `early-modern`,
`modern`, plus Celtic equivalents (`middle-irish`, …). The `--era` knob
filters the morpheme inventory to forms attested in a period; combined
with [language](#culture--register), any (language × era) cell produces
a different surface form (D5, D5-3).

### Stratum (register)
A *within-language* layer — where a morpheme sits in its own language's
borrowing history. Welsh splits into `native-welsh` / `latin-loan` /
`english-loan` / `brittonic-substrate` / `medieval-welsh`; Old English
into `native-old-english` / `latin-loan` / `norse-loan` /
`celtic-substrate`; etc. The `--stratum` knob restricts to one bucket;
validation is per-culture (D32). Don't confuse with [era](#era) (time)
or [culture](#culture--register) (language).

### Attestation
A dated historical record that a form existed — a `(form, year)` pair
extracted from scholarly notes or a gazetteer ("Domesday Book, 1086").
The backbone of the [era](#era) machinery. Stored in
`toponym_attestation`; `attested_year` on an etymon activates the
`--era` filter (D5-1).

### Gloss
A short paraphrase of what a morpheme means ("cottage", "untilled
land"). Informational, **not** validated against the source — unlike the
[form](#form-in-body-validation), which is the load-bearing truth claim.
Stored in `etymon_gloss`.

---

## Pipeline & architecture terms

### Authoring layer
Where day-to-day work happens: mining etymology dictionaries, tracking
citations, normalizing OCR, enrichment. Backed by the SQLite DB at
`~/.wyrd/lexicon.db` (override with `WYRD_LEXICON_DB`). Can be reshaped
aggressively without touching production — the runtime sees nothing of
it until export. (OVERVIEW)

### Runtime layer
What users actually hit: the Lambda generator imports a bundled,
pre-baked artifact (`meanings.json` / `seed-runtime.db` + per-culture
proportions) via `importlib.resources`, never opens a database, and runs
in milliseconds. Connected to the authoring layer only by the
`export-meanings` / `export-runtime-db` step you run when you want to
ship. (OVERVIEW)

### Lexicon
The authoring-side SQLite database (`~/.wyrd/lexicon.db`) and, loosely,
the whole corpus of mined morpheme data it holds. The
`wyrd kenning lexicon …` CLI operates on it.

### Bundle / meanings.json
The shipped runtime data file — a list of "subjects" (meaning + its
candidate morphemes across languages), exported from the lexicon. Per-form
metadata rides on `<lang>_<feature>` sibling fields (`old_english_variants`,
`old_english_inflections`, …) so legacy loaders that ignore unknown
fields keep working (D26). Increasingly shipped as SQLite-on-S3
(`seed-runtime.db`) rather than JSON (D38).

### L1 / L2 / L3 / L4
The four pipeline layers (see
[`L2_L3_BOUNDARY.md`](../wyrd/generators/kenning/L2_L3_BOUNDARY.md),
[`REBUILD.md`](../wyrd/generators/kenning/REBUILD.md)):
- **L1** — raw inputs: OCR'd books, wiktextract slices, gazetteer CSVs.
  Large, mostly not committed.
- **L2** — curated per-source JSONL in `data/mining/<id>.jsonl`. The
  **source of truth**, committed to git.
- **L3** — the SQLite query index (`~/.wyrd/lexicon.db`). A rebuildable
  build artifact; never the source of truth.
- **L4** — the shipped runtime bundle (`seed-runtime.db` / proportions).

### Mining
Running an LLM (or a structured ingester) over a source to *extract*
morpheme etymologies into the lexicon. Expensive — hours and dollars.
The project's core scaling principle: **mining is expensive, enrichment
is cheap**, so optimize so iterating on enrichment costs almost nothing
and mining stays the only slow/costly operation. (OVERVIEW, INGESTION)

### Enrichment
The cheap, idempotent, LLM-free post-mining passes that *derive* data
from mining evidence: `link-lemmas`, `normalize-ocr`, `reverse-search`,
`fuzzy-search`, `cluster-cognates`, `classify-stratum`, … Reversible by
design (D22): every derived row carries a method stamp and can be cleared
and rebuilt. Never destroys mining evidence (D21).

### Consensus
The trust mechanism: count *distinct sources* that independently cite an
etymon (`etymon_consensus` view). The promotion threshold is **≥3
independent witnesses** — at which a morpheme moves from "trust the
legacy seed" to "promotion-eligible, include in the live inventory" (D4).
Per-language thresholds relax this where a corpus is genuinely thin (D4
refinement). [Descent](#descent--descent-graph) edges deliberately do
**not** count toward consensus.

### Witness
One independent source that cites a morpheme. "3+ witnesses" = three
distinct scholarly sources proposed it. The unit [consensus](#consensus)
counts.

### Citation
An *extraction* witness — a record that a specific source formally
proposed a morpheme as part of a toponym's etymology (`etymon_citation`).
Distinct from a [text match](#text-match): a citation is "Mawer derives
Dean from OE `denu`"; a text match is merely "the string `denu` appears
in Mawer's body text". Source attribution is mandatory — no etymon enters
without one (D12, OVERVIEW).

### Text match
Looser evidence than a [citation](#citation): an occurrence of an
etymon's form (or a fuzzy variant) somewhere in a source's body text
(`etymon_text_match`). Does **not** count toward [consensus](#consensus).
Doubles as the [spelling-variant](#spelling-variant) pool. (D12, D18)

### Form-in-body validation
The load-bearing anti-hallucination guard: **every form a model emits
must appear in the source paragraph** (case-insensitive, OCR-tolerant
substring match). Forms that don't match get the whole row *rejected*,
not ingested. [Glosses](#gloss) are deliberately exempt — they're
paraphrases. Don't loosen it without thought. (D3)

### rando-port
The original seed data — ~1600 morphemes ported from Rando's
Wikipedia-derived name maker. Broad lemma coverage but unverified per
entry (no witnesses, no attribution). Mining adds scholar witnesses on
top; the goal is to push rando-port lemmas across the ≥3-witness
[promotion](#consensus) threshold. Appears as the `rando-port` source_id.

### OCR clustering / normalize-ocr
Collapsing OCR-mangled spellings of one form into a single etymon row
(`Hædan` / `Hcsdan` / `Haedan` → one row). Non-destructive: uses a
`merged_into_id` redirect rather than DELETE, so the loser keeps its
evidence and the merge is reversible (D9, D22).

### Decomposition
Splitting a surface name into its constituent [morphemes](#morpheme) —
the inverse of generation. The trie-indexed segmentation DAG matcher
(`runtime/trie_matcher.py`) enumerates *every* valid parse, since one
surface can have multiple readings (D29). "Perfect" decomposition =
every character maps to a known morpheme (the rate tracked in
[`COVERAGE.md`](../wyrd/generators/kenning/COVERAGE.md)).

### Proportions
Per-culture pre-baked statistics (`<culture>_proportions.json`) the
runtime samples from in the default `proportions` scoring mode — joint
frequencies at the **tag** level (semantic classes), not the morpheme
level, because a K×K tag matrix is learnable where a morpheme matrix
would be hopelessly sparse (D16). Rebuilt by `rebuild-proportions`.

### mining_run / audit
Every mining and review run writes one `mining_run` row capturing
provider/model/mode and accept/decline/reject counts. The principle
(D23/D24): **when you add data, add an observability path** — if a CLI
prints a number, that number must also land in a queryable row, never
stdout-only. "Trust the audit table over agent summaries."

### Sibling pipeline (wyrd-ami)
The fantasy-name research pipeline that runs *alongside* place-name
mining on the same lexicon. It works in the inverse direction: take a
creature name (Harpy, Djinni) and walk it *back* to an existing etymon,
recording the result in `fantasy_morpheme`. The etymon table is
read-only from its perspective. (D30, OVERVIEW)

---

## Providers & tiers (D2, D13, D19)

### Tier 1 (bulk extraction)
First-pass extraction across thousands of entries. **Qwen 3.5 on Ollama**
(free, local GPU host) for English; **Anthropic Haiku 4.5** (~$0.60/book)
for Celtic/Welsh/Irish/Manx, where Qwen underperforms (D13). The tier
choice is per-book, not per-language.

### Tier 2 (review)
A second pass with **Gemini 2.5 Flash** over low/medium-confidence Tier-1
rows. Writes a *parallel* row tagged by provider rather than replacing
the original, so [consensus](#consensus) sees both. The standard second
pass on English mining (and, empirically, productive over Haiku on Celtic
too — INGESTION).

### Tier 3 (rare review)
**Anthropic** on whatever Gemini still declines. Mostly unused for mining
— the Sonnet uplift over Gemini Flash is empirically ~zero (D19) — so
Anthropic budget is reserved for user-facing runtime features (explainer,
register-shift).

### Ollama
The local inference server hosting Qwen for free bulk mining, run on the
operator's macbook at `http://10.5.2.31:11434` (`WYRD_OLLAMA_URL`). Runs
one inference at a time, so parallel batches against it don't speed up.

---

## Generation knobs

All default to off / 0.0 and preserve historical seed-stable behavior
bit-for-bit. Full descriptions in
[`README.md`](../wyrd/generators/kenning/README.md) and
[`OVERVIEW.md`](../wyrd/generators/kenning/OVERVIEW.md).

### --culture / register
The language palette: `english`, `scottish`, `welsh`, `irish`, `breton`.
Morally neutral — no language is pre-coded "good" or "evil"; the
dark/menacing feel comes from [mood](#--mood) filters that apply to any
language (D6).

### --novelty
Blends each bucket's empirical-frequency distribution toward a uniform
marginal. At 1.0 every in-bucket morpheme is equally likely — plausible
-but-unattested combinations become possible (D17).

### --cohesion
The opposite of novelty: biases each slot toward usages whose tags
co-occur with already-picked slots' tags in the corpus — "attested-pair
fidelity". Composes orthogonally with novelty (D17 / wyrd-mj2).

### --mood
A stylistic-register preset (repeatable; e.g. `grim`, `harsh`, `noble`,
`mystical`). Each is a per-dimension vector of phonological + semantic-tag
+ position weights. `harsh:0.5` scales an effect; multiple flags compose.
Catalog lives in `data/register_effects.yaml`; grounding in
[`REGISTERS.md`](../wyrd/generators/kenning/REGISTERS.md). (D6, D37)

### --era / --stratum
Filter the morpheme inventory by historical period ([era](#era)) or
within-language layer ([stratum](#stratum-register)). They compose via
frozenset intersection.

### --spelling-variety / --inflection-density
Per-morpheme probability of substituting an archaic
[spelling variant](#spelling-variant) (D18) or an
[inflected](#inflection) form (D8) for the canonical reflex. Inflection
wins when both fire on the same morpheme.

### --scoring-mode {proportions, vector}
The per-slot sampling pipeline. `proportions` (default) samples the
pre-baked [proportions](#proportions) tables — bit-stable with the legacy
path. `vector` computes each lemma's score at request time from four
weighted axes — phonological, semantic, position, and empirical-baseline
— via the D36.2 canonical composition
`score = phon_w·phon + sem_w·sem + pos_w·pos + base_w·baseline`. (D36 /
the ecjp epic)

### --priors-path / --*-weight
`--priors-path` points at the empirical-priors sidecar JSON (from
`dump-empirical-priors`) that feeds vector mode's baseline axis.
`--baseline-weight` / `--phonological-weight` / `--semantic-weight` /
`--position-weight` scale each axis (default 1.0; 0 disables). Only
meaningful with `--scoring-mode=vector`.

---

## See also

- [`OVERVIEW.md`](../wyrd/generators/kenning/OVERVIEW.md) — start here for
  the kenning sub-system as a whole.
- [`DECISIONS.md`](../wyrd/generators/kenning/DECISIONS.md) — the D-entries
  these definitions reference.
- [`INGESTION.md`](../wyrd/generators/kenning/INGESTION.md) — the mining
  procedure manual.
- [`L2_L3_BOUNDARY.md`](../wyrd/generators/kenning/L2_L3_BOUNDARY.md) /
  [`REBUILD.md`](../wyrd/generators/kenning/REBUILD.md) — the layer model
  and rebuild runbook.
- [`REGISTERS.md`](../wyrd/generators/kenning/REGISTERS.md) — phonaesthetic
  grounding for the mood/register weights.
