# Architecture decisions for the kenning lexicon

This file records the policy and design decisions we've landed on so they
don't live only in conversation. Each entry has a short rationale.

> **Superseded decisions are archived, not kept here.** When a decision is
> retired, its body moves to `archive/superseded-decisions.md` and the entry
> below shrinks to a one-line tombstone (`— SUPERSEDED <date> by <DN>`). The
> ledger of what moved, when, and what replaced it is `archive/DEPRECATED.md`.
> The `archive/` folder is intentionally **outside** the read-all-kenning-docs
> working set — skip it unless you're doing archaeology.

## D1. The lexicon is the authoring layer; meanings.json stays the runtime.

The runtime generator continues to read `data/meanings.json`. The lexicon
SQLite DB is an authoring-side store: mining, citation tracking, consensus,
lemma linkage. A future export step (filed) will regenerate `meanings.json`
from the lexicon when we want changes to ship.

Why: keeps the runtime path stable while we churn on the data layer.

## D2. LLM-provider tiers.

Three roles, three providers:

- **Bulk mining**: Ollama on the operator's MacBook (Qwen 3.5 9B) at
  `http://10.5.2.31:11434` (set `WYRD_OLLAMA_URL`). Free, fast enough, no
  rate limits. Used for first-pass extraction across thousands of entries.
  (Hades is *not* used for mining — its GPU is too weak, capped at
  ~qwen2.5:7b — per the parent-workspace `521Studios/CLAUDE.md`, one level
  above this repo.)
- **Quality review**: Gemini 2.5 Flash. Cheap per-call, native JSON-schema
  enforcement, stronger on hedge-recognition and OE-form OCR. Used for the
  `lexicon review` second-pass on questionable Ollama rows.
- **User-facing runtime**: AWS Bedrock (Claude Sonnet 4.6+). Best quality,
  intra-AWS latency, IAM-scoped. Used for live "explain this name in
  detail," "render across registers," "generate variations" from the lambda.

Why: cost gradient (free → cheap → premium) matched to quality requirement
(volume → review → user-facing).

## D3. Form-in-body validation is the load-bearing safety guard.

Every claim a model makes about a morpheme — the `form` field of an
extracted etymon — must appear (case-insensitive, OCR-tolerant) in the
source paragraph. Forms that don't match get the entire row rejected, not
ingested. Glosses are intentionally *not* validated; they're paraphrases.

Why: this is the one thing standing between an LLM hallucinating an etymon
and that etymon being citation-ed into the lexicon. Don't loosen it
without thought.

## D4. Cross-source consensus is the trust mechanism.

Each etymon accumulates citations from multiple sources via
`etymon_citation`. The `etymon_consensus` view counts distinct witnesses
per etymon (rolled up to the lemma when `lemma_id` is set). Promotion
threshold is informal — N≥3 witnesses with no breakdown disagreement
treated as "live inventory"-eligible.

Why: rando-port (the meanings.json seed, derived from Wikipedia) provides
~1600 unverified etymons. Mining adds scholar witnesses. Anything cited by
≥3 independent scholars is unlikely to be wrong; anything still standing
on rando-port alone is "trust the legacy data" until corroborated.

**Refinement (wyrd-yz7, 2026-05-02): per-language thresholds.** A uniform
≥3-witness gate filters out morphemes for corpus-thinness reasons rather
than quality. Old-norse has effectively one focused dictionary
(Lindkvist 1912) and 93% of its lemmas don't reach 3 witnesses; spot-
checking the w=2 ON pool finds it ~95% clean. Old-english is the
opposite — well-mined across 32 sources with measurable Tier-1 prose-
extraction noise at w=2.

`export-meanings` now ships `RECOMMENDED_LANG_THRESHOLDS`: OE at 3
(strict, well-mined corpus), Celtic / ON / modern-English / norman-
french / latin / biblical at 2 (corpus-thinness limited; quality
holds). Languages absent from the map fall back to global
`--min-witnesses`. Net effect on the bundle: ~+340 modern_usages over
uniform w=3, mostly Celtic (+278) and ON (+46).

## D5. Two-axis register model.

Generation registers come from two orthogonal axes:

- **Language tag** (the data axis): `old-english`, `old-norse`, `celtic`,
  `latin`, `greek`, `norman-french`, `nahuatl`, `mayan`, `slavic-east`,
  etc. Each has its own etymon pool, phonology, composition patterns.
- **Era** (the time axis): pre-1066 archaic forms, Domesday-period,
  late-medieval, early-modern, modern. Each language has its surface
  forms across these.

Combined, you get any (language × era) cell: "Old English at ~900 AD"
produces `Hædan-tūn`; "Old English modernized" produces `Hadenham`;
"Latin at empire" produces `Pons Aquae`.

Why: keeps the model honest about what's data vs. what's stylistic.

**Refinement (wyrd-z56, 2026-05-03): attested_year validator range
widened.** The D5-1 prompt-side capture (PR #27, wyrd-3ux foundation)
shipped with `_ATTESTED_YEAR_MIN=800` to filter publication-year noise
(1880-1928 in the scholarly-source pool) and folio/page numbers. That
floor was sized for British/Norman-period sources where Domesday Book
(1086) is the typical earliest dated attestation. On Romance/Celtic-
substrate sources like d'Arbois 1890 *Recherches sur l'origine de la
propriété foncière (période celtique et période romaine)*, the 800
floor cut legitimate Roman-empire and early-Merovingian charter dates
— concretely, 39 entries on the d'Arbois full pass (wyrd-cmz) were
rejected as `attested_year_out_of_range` for years like 580 (Aria
monasterio) and 200 (Lugudunum) that are valid Roman/Gallo-Roman
attestations.

PR #42 widened the floor to 100 (Roman 1st-c. AD onward through
Restoration) and tightened the SYSTEM_PROMPT with explicit
RIGHT/WRONG examples for JSON-integer-not-string year format
(addressing 32 `attested_year_not_int` rejections from models
emitting `"1333 (?)"` strings instead of bare integers). Re-mining
d'Arbois recovered 27 toponym etymology rows (100 → 127 accepted)
and 51 etymon citations (168 → 219 etymons touched).

**Pattern**: validator default ranges should be reviewed when a
new source class enters the corpus. A guard tuned for one
substrate's attestation period may exclude legitimate rows on a
different substrate. When a single rejection_reason dominates the
breakdown (>30% of rejections), check it against the source's
content-period before re-mining.

## D6. Languages are morally-neutral palette options.

No language is pre-coded as the "good" or "bad" register. Slavic isn't
"orcish"; Norman-French isn't "noble villainy"; Celtic isn't "elven";
Old Norse isn't "dwarven." Those are convention-mappings the GM can
choose, not engine-imposed labels.

The dark/menacing effect is provided by **two orthogonal filters** that
apply to *any* language:

- `--harsh` — phonological skew (short syllables, harsh consonant
  clusters, voiced-stop endings)
- `--grim` — semantic-tag filter (etymons tagged with grim / mortuary /
  monstrous / battle / wilderness)

A GM gets `slavic-east --harsh --grim` for menacing Russian-coded
fortresses, or `slavic-east --noble --pastoral` for Russian-coded heroic
villages. The data doesn't decide.

Why: avoids reinforcing the "Slavic = orc" / "Norman = villain" patterns
that fantasy genre has often relied on without examining.

**Runtime status (wyrd-9yf + wyrd-tbd, 2026-05-02): wired under a
single `--mood` flag.** Two axes ride under one GM-facing surface:

- `--mood grim` applies a tag-union over the menacing tag set
  `(death, military, monster, undead, magic)` — the closest extant tags
  in the bundle (D6 originally specified
  `(grim, mortuary, monstrous, battle, wilderness)`, none of which exist
  yet; the spec-named tags remain a future-mining target).
- `--mood harsh` applies a per-bucket re-weight by phonological harshness
  score: stop-final / cluster-heavy keys get up to 2× their empirical
  weight at full harshness, soft keys go to 0. The score combines coda
  harshness (45%), cluster density (35%), and consonant density (20%) on
  the dash-stripped lowercased usage. (This describes the retired
  proportion-table sampler's per-bucket re-weighting; under the vector
  path harshness enters as a phonological register dimension. The
  `--novelty` knob it once composed with was removed with proportions
  scoring — re-wired onto vector, wyrd-fcub.)
- `--mood harsh:0.5` graduates the harshness skew via colon-suffix.
- `--mood pastoral` (plant / animal / water / agriculture / tree / bird
  tag union) for rural / agricultural feel.
- `--mood devotional` (saint / religious) for monastery / pilgrim feel.
- `--mood mortuary` (death / undead) for funerary feel — a strict subset
  of `grim` for cases where the GM wants death-themed without the
  military / monster axes.
- Multiple `--mood` flags compose by tag-union and max-harshness.

The mood vocabulary lives in `wyrd/generators/kenning/data/register_effects.yaml`
(catalog-driven since wyrd-kq7w.3 — the legacy `registers/moods.MOODS`
dict was ripped and replaced). New presets are picked from a tag-coverage
audit (≥5 subjects per candidate tag, distinct semantic identity, minimal
overlap with existing moods); `noble` was considered in wyrd-aky and
deferred until mining surfaced a `royalty` tag — landed in the kq7w.2
catalog migration. Lookup goes through `registers/effects.parse_mood_spec`
(the live vector path). The catalog is the single source of truth, so a
mood added to YAML is immediately operator-visible without
code changes.

Power-user JSON API still exposes `harshness` (number, 0..1) for
graduated control without the colon syntax; `harshness` and
mood-derived harshness resolve via `max(...)`. The original `--grim`
boolean / `--harsh` float flags were superseded by `--mood` because
two parallel flags muddled axes that conceptually belong under one
preset.

**Superseded by D37 (2026-05-21, wyrd-kq7w epic):** the 5-entry
preset list above is historical. The post-kq7w catalog has 12
entries (five legacy + noble, mystical, melodic, sinister, ancient,
exotic, martial) with per-dimension phonological + semantic +
position weights instead of the legacy `{tags, harshness}` recipe
shape. See D37 for the phonaesthetic-vector composition framework
that replaced the MOODS-dict + harshness-scalar approach.

Defaults to off (bit-stable historical behavior). The harshness score
is a heuristic — meant for relative ranking, not phonotactic fidelity.
Per-language phonology integration (using the IPA tables in
`registers/phonology.py` for a more principled score) is a future refinement.

## D7. Sensitivity heuristic for non-European corpora.

Active living communities with sovereignty, land, or revitalization
stakes get **framework-only** treatment for **user-facing culture
options**: we provide the language-pack format and let community
members supply the data. The PD-1900s literature doesn't override
this just because the books are out of copyright.

- **Mine and bundle as a user-facing culture** (ancient-civilization
  frame, no living-community veto): Mayan, Nahuatl, Egyptian, Sumerian,
  Greek, Roman, all European.
- **Framework-only as a user-facing culture** (active sovereignty /
  revitalization): Native American, Aboriginal Australian, Maori,
  Hawaiian, Khoisan.
- **Edge cases**: Hebrew (biblical-strata = bundle; modern political =
  caution), Bantu (varies by community size and politics).

Why: PD status is a legal property, not an ethical one. The line is
about whether a living community has a stake in how their language is
used as **a feature of our shipped product**.

### Refinement (2026-05-16): inspiration vs. bundling

The framework-only rule applies to **shipping a real-world language as
a culture the user selects**. It does NOT apply to **using a
real-world language's empirical loan-adaptation patterns as inspiration
research for a constructed/fictional pack** that is itself the shipped
artifact.

Example: a Na'vi pack (Paul Frommer's constructed language, shipped as
the user-facing artifact) can use Hawaiian / Polynesian → English
toponym adaptation patterns as the empirical template for how its
phonology lands in English. The Hawaiian corpus lives internally as
research input (mining-time, not bundle-time); the Na'vi pack is what
ships. Distinct from "bundle Hawaiian as a culture option," which would
still be framework-only.

The test for whether the refinement applies:
1. Is the shipped artifact a real-world language pack the user
   selects? → D7 framework-only rules apply.
2. Is the shipped artifact a constructed/fictional pack that used
   real-world data only as research input? → fair game; cite
   research sources in pack metadata for transparency.

Composes with D11 (no bundled IP): both rules block bundling — D7
on ethical grounds, D11 on legal. The inspiration carve-out applies
to both: real-world *patterns* are not IP, and using them to inform
new constructed work is research, not redistribution.

Operational guardrails for inspiration-research work:
- Cite the source-language scholarly references (e.g., Pukui & Elbert
  for Hawaiian, Bright 1979 for Algonquian) in pack metadata so the
  research lineage is transparent.
- Maintain phonological dignity in output. Don't generate
  comic-book-mockery output — adaptation rules should preserve the
  source language's coherent phoneme inventory rather than caricature
  it.
- The internal-research corpus stays on the authoring side
  (`~/.wyrd/lexicon.db`), never in the shipped `meanings.json` bundle.

## D8. Lemma + inflection schema.

Each `etymon` row has a nullable self-FK `lemma_id` and a nullable
`inflection` text column. Inflected variants point at the lemma; the
consensus view rolls citations up to the lemma. Generation can compose
`<lemma>@<inflection>` and look up or derive the surface form.

Why: rando-port (Wikipedia-sourced) gives lemma forms; mined scholarly
sources give inflected forms. Without this linkage, the same morpheme
shows up as 3-5 separate etymon rows and consensus undercounts. With it,
`cot` / `cotan` / `cotum` / `cotes` are one lemma with witnesses unioned.

**Runtime status (wyrd-7fn, 2026-05-02): wired.** `export-meanings`
emits per-language `<lang>_inflections` arrays carrying `(form, label)`
tuples. `Meaning.pick_inflection(rng, density)` does a uniform draw
(mining tracks labels not counts; inflections aren't weighted) with
lazy-cached pool flatten. `NameGenerator.select_via_vector(inflection_density=…)`
threads through and pre-renders surface substitutions; `NewName` grows
an `inflection_labels` parallel list for the explainer. CLI:
`--inflection-density` (0..1). When both `--inflection-density` and
`--spelling-variety` would fire on the same morpheme, inflection wins
— it carries grammatical case while the variant axis is purely
spelling. Bundle has 168 inflected etymons across 9 case labels (OE
dominates: dative_or_pl 80, weak_oblique 23, genitive_strong 25,
nominative 23). Per-position rules ('genitive-strong more common
pre-, dative-or-pl more common post-') are NOT yet implemented —
current picker is uniform across positions; deferred until corpus
diversity supports the position bias. Explainer surfacing of
`<lemma>@<label>` in `description()` is also deferred (the labels
ride on NewName but description() still emits canonical meanings).

## D9. OCR ligature normalization is upstream of lemma linkage.

Before lemma-linking runs, an OCR-cluster pass merges variants of the
same form across spelling drift: `Hædan` / `Hcsdan` / `Haedan` collapse
to one row with citations unioned. Then the lemma linker runs against
the cleaned forms.

Why: OE characters (æ, ð, þ) get mangled by OCR engines in predictable
ways (æ → "cs" / "ce", ð → "§"). Without normalization, the lemma linker
sees four different "Hædan"s instead of one. With it, the linker can
focus on actual inflection variation.

## D10. Don't mock the data layer in tests.

Lexicon tests use real SQLite (a tmp file per test) rather than mocking
the DB. The schema and the queries are too coupled to the data shape;
mocked tests would pass while real ones would fail.

Why: We've already had bugs (e.g., `UNIQUE (..., COALESCE(...))` is
forbidden in inline constraints) that would have shipped silently behind
mocks. The real-DB approach catches them at test time.

## D11. Language-pack and pantheon-pack are user-pluggable.

Tolkien languages, D&D pantheons, Pathfinder pantheons, homebrew
worlds — all delivered via a YAML format the user drops in. We never
bundle commercial or sensitivity-restricted data; we provide the
machinery and let users supply their own packs.

Why: legal cleanliness (no IP we don't own), ethical cleanliness (no
overriding community vetoes), and infinite extensibility (the GM's
homebrew world gets the same treatment as Middle-earth).

## D12. Search-evidence lives in a parallel table, not in citations.

`etymon_text_match` records body-text occurrences of an etymon's
canonical form (or a fuzzy variant) inside a source. Separate from
`etymon_citation`, which records *extraction* witnesses
(a scholar formally proposing this morpheme as part of a toponym's
etymology). The `etymon_consensus` view does not include text matches.

Why: text-presence and extraction-presence are different evidence
weights. "The string `denu` appears 14 times in Mawer's body text" is
weaker than "Mawer formally derives Dean from OE `denu`." Mixing the two
into one citation count would inflate consensus and break the
≥3-witnesses promotion threshold. Considered the simpler "just add a
column" path and rejected it as architectural debt.

## D13. Haiku 4.5 is Tier 1 for Celtic content; Qwen stays Tier 1 for English.

Empirically, Qwen 3.5 9B on the operator's MacBook (the Ollama mining
host — D2) produces good extraction yields on English-etymology books
(Skeat, Mawer, Ekwall) but underperforms on
Celtic and Welsh sources where headword recognition gets confused by
mutations and digraphs. Haiku 4.5 (~$0.60/book on the Anthropic API)
roughly doubles the yield on those books with the same SYSTEM_PROMPT.

Why: cost-per-marginal-citation. For high-yield English-etymology
sources, Qwen's free-and-fast wins. For Celtic content where Qwen
declines half the entries, paying $0.60 to recover them is cheap. The
tier choice is per-book, not per-language-family in general.

## D14. Hedge-aware extraction; `confidence='low'` instead of decline.

The SYSTEM_PROMPT now instructs models to extract hedged etymologies
("possibly from", "perhaps a corruption of", "obscure but compare with")
as `confidence='low'` rows with the hedge captured in `notes`, rather
than declining the entry. The lexicon stores the confidence on the
`toponym_etymology` row; consumers can filter.

Why: scholarly etymology is full of hedging — declining hedged entries
threw away a substantial chunk of the corpus. The information is still
useful (especially for novel-name generation, where low-confidence rare
patterns are exactly what we want), as long as the confidence is
preserved end-to-end.

## D15. Fuzzy match requires a gloss anchor.

`fuzzy_search_attestations` accepts edit-distance ≤ 1 only when a
gloss-string from `etymon_gloss` appears within ~100 characters of the
matched form in the source body. No gloss confirmation, no match.

Why: edit-distance alone is too noisy at the single-character scale
that matters for OE/ON variants. `bere` (barley) and `bera` (bear) are
one edit apart and mean entirely different things. The gloss anchor
filters coincidental near-matches without losing the legitimate
spelling drift case (`denu` → `dene`, `denū`, `dená`).

**Refinement (wyrd-c3x, 2026-05-02)**: gloss anchoring on its own is
too permissive on generic gloss words. ON `herath` ("district",
glossed "land/area/district") was claiming OE `heath` as a fuzzy
variant 47 times across 5 sources because "land" co-occurs with
"heath" routinely in scholarly toponym discussion. Two morphemes,
unrelated etymologies, but the gloss anchor fired anyway.

The fix is structural rather than statistical: if the matched body
token is itself a canonical etymon in the lexicon, suppress the fuzzy
claim entirely. The body word is its own thing — not a fuzzy variant
of someone else. OCR variants between two etymons (`bōthl` ≈ `botl`)
are `normalize-ocr`'s responsibility upstream, not fuzzy-search's.
This refinement removed 131 false-positive rows from
`etymon_text_match` and prevents the shape from reappearing.

## D16. Co-occurrence lives at the tag level, not the morpheme level.

The corpus statistics emitted in `_proportions_from` track joint
frequencies of *tags* (semantic classes like `descriptive`,
`topography`, `architecture`, `tree`, `water`) rather than individual
morpheme IDs. A K×K tag matrix is small and learnable; a K×K morpheme
matrix would be sparse and useless past a few hundred rows.

Why: the goal is selectional preference ("descriptive + topography is
attested all the time, even when the specific Brait+combe pair never
co-occurred"), not memorizing the corpus. Tag-level lets us condition
sampling on patterns we can actually estimate, and lets us generalize
across languages (descriptive+topography is a thing in Welsh too).

## D17. Bayesian mixture is the novelty knob, not just better fit.

— **SUPERSEDED 2026-05-18 by D36 (vector-driven generator architecture); body archived → `archive/superseded-decisions.md`.** Live residue: `--cohesion` is now a multiplicative boost inside the vector scorer; `--novelty` was removed and re-wired onto the vector path (wyrd-fcub).

## D18. Spelling variants are generative.

`etymon_text_match.matched_form` stores the actual spelling found in
the source — not just the canonical etymon form. This is a
randomization target at generation time: 19th-century scholars wrote
`denu` / `dene` / `denū` / `dená` for the same morpheme, and the
generator can sample from the variant pool to add archaic-feel variety
without inventing surface forms.

Why: spelling wasn't standardized. The variant pool is essentially
free archaic-spelling data we already collected; throwing it away by
collapsing to canonical forms loses generation flexibility.

**Runtime status (wyrd-q13, 2026-05-02): wired.** `export-meanings`
emits per-language `<lang>_variants` arrays carrying `(form, weight)`
tuples; weights are the per-attestation `match_count` totals from the
lexicon, summed across the family during the descendant walk.
`Meaning.pick_variant(rng, variety)` does a probability-gated weighted
draw across all language pools. `NameGenerator.select_via_vector(spelling_variety=…)`
pre-renders surface substitutions at pick time so `__str__` stays pure.
Module helper `_mimic_case` projects the canonical's casing onto the
picked variant (title-case template → first-letter cap with internal
capitals preserved; lowercase template → forced lowercase). CLI:
`--spelling-variety` (0..1). Bundle has 416 morphemes with non-empty
variant pools (mostly Celtic + ON; macron-stripped reverse-search hits
+ fuzzy-search winners + post-disambiguator survivors).

## D19. Sonnet review doesn't lift over Gemini Flash for this task.

Tested Sonnet 4.6 on Mawer disagreement cases and found it
over-extrapolates (proposing `+tūn` for Angerton, `+lēah` for Ardley)
where the body text doesn't support it. The form-in-body
validator catches the over-reach, but the net useful-extraction lift
over Gemini Flash was ~zero. Reserve Anthropic API budget for runtime
user features (explainer, register-conversion, MCP) rather than mining.

Why: pay for the tier where it actually helps. Cheaper Tier 2 already
hits the ceiling on what extraction can confidently recover from the
body text alone.

## D20. Modern-English etymons are excluded from reverse-search.

`reverse_search_attestations` filters out etymons whose `language` is
modern English (or unmarked). Otherwise, scanning for the etymon `with`
matches 12,473 occurrences across the corpus — pure noise.

Why: reverse-search assumes the etymon is a historical morpheme being
hunted in scholarly text. Modern-English seed entries are real English
words, and the assumption breaks. Filtering them out is the simplest
correct fix.

## D21. Enrichment must not destroy mining evidence.

The lexicon distinguishes two kinds of data:

- **Mining evidence** — expensive, irreplaceable. Each `etymon_citation`
  row is a real LLM call against a real OCR'd book that passed
  form-in-body validation. Each `etymon_text_match` row is a
  regex/Levenshtein scan across the ~1.8GB source corpus. Each
  `toponym_etymology` row is a structured extraction that passed the
  same validation. This data takes hours of LLM time and thousands of
  API calls to produce.
- **Enrichment inferences** — cheap, rebuildable. `etymon.lemma_id`
  linkage, OCR-cluster merges, reverse-search hits, fuzzy-search
  attestations. All of this is derivable from the mining evidence by
  re-running the post-mining chain in seconds.

**Implication for any operation that mutates `etymon`:** mining
evidence attached to a row being modified must be preserved (repointed
to the survivor) — not silently dropped via `ON DELETE CASCADE`, not
broken via a missed `lemma_id` self-FK. The FK gap fixed in `wyrd-go5`
was exactly this class of bug: `cluster_ocr_variants` was repointing
some child tables but missing two, so OCR clustering would have
silently destroyed text-match rows and crashed on lemma children.

**Better long-term shape (`wyrd-et0`):** stop deleting at all. Use a
`merged_into_id` self-FK on `etymon` to mark losers as merged-into
their canonical winners, and have downstream consumers read through an
`etymon_canonical` view. This makes enrichment iteration cheap forever
— there is no destructive step that can lose data, and re-running the
chain after a tweak is a no-op rather than a partial-state hazard.

Why: a re-mining pass costs hours and dollars; a re-enrichment pass
should cost seconds. The design optimizes for "redo enrichment is
free" so we can iterate on linkage/clustering/search heuristics
without fear, and reserve the expensive operation for the one thing
that genuinely needs it (extraction from new sources).

## D22. OCR clustering is non-destructive via `merged_into_id`.

`cluster_ocr_variants` does NOT delete loser etymons. Instead, the
loser's `merged_into_id` self-FK column is set to the canonical
winner. Citations, glosses, tags, text-match rows, and reflex links
stay attached to the loser exactly where they were originally
written. The `etymon_consensus` view rolls them up to the canonical
group via the rollup chain `COALESCE(merged_into_id, lemma_id, id)`;
the `etymon_canonical` view exposes the un-merged set.

Reverting a clustering pass is a one-liner:

```sql
UPDATE etymon SET merged_into_id = NULL;
-- or via CLI:
wyrd kenning lexicon clear-enrichment --stage=ocr --apply
```

After which `cluster_ocr_variants` can be re-run with new heuristics
on the full canonical set. Re-runs filter `WHERE merged_into_id IS
NULL` so previously-merged tombstones don't re-enter clustering.

**`--stage=ocr` is partially reversible.** It reverts the
`merged_into_id` redirects but does not restore the `lemma_id`
re-parenting that happened at merge time (see flatten rules below).
For a fully clean reset, use `--stage=all-derived`, which clears
both the merged_into_id redirects AND the lemma_id /
inflection / lemma_method linkage. Re-running `link-lemmas` after
that produces a fresh lemma assignment from scratch. The mining
evidence layer (citations, glosses, tags, text-match rows) is never
touched by either stage.

Two flatten-at-write-time rules keep the consensus rollup correct
without recursive CTEs:

- Lemma children of a loser (`child.lemma_id = loser`) are
  re-parented to the canonical destination at merge time.
- Existing redirects pointing at the loser
  (`X.merged_into_id = loser`) are also re-routed to canonical, so
  no `X → loser → canonical` chain forms.

Both are mining-evidence-preserving (no citation/gloss data moves)
but they are NOT cosmetic: a 2-deep chain through the consensus view
would split witnesses across two GROUP BY buckets, undercounting the
canonical morpheme's witness total and incorrectly gating the D4
≥3-witness promotion threshold.

The view itself also does a two-step rollup (`merged_into_id` →
`lemma_id`) so the combination
`OCR-loser-of-an-inflected-variant-of-a-lemma` rolls all the way to
the lemma in a single GROUP BY pass, no matter what order the
clustering and lemma-linking stages were run in.

Why: this is the schema-level expression of D21. As long as the only
mutation made to the etymon table is a redirect column, mining
evidence cannot be lost by an enrichment pass — the worst case is
"some merges are recorded that the user then wants to undo," and
undoing is free. Supersedes the wyrd-go5 interim FK-repoint fix:
once nothing is being deleted, there are no FK gaps to miss.

## D23. Mining stats are persisted, not stdout-only.

Every `mine-llm` run and every `lexicon review` run writes one row
to the `mining_run` audit table at end-of-run, capturing
`(source_id, provider, model, mode, started_at, completed_at,
parsed_count, accepted, declined, rejected, by_failure_json,
notes)`. Idempotent on
`(source_id, provider, model, mode, completed_at)`.

Why this exists: pre-D23 the mining pipeline computed
accept/decline/reject counts in-memory and only printed them to
stdout. Once the process exited, they were gone. Asking the DB
"how many declines did Joyce 1875 produce?" would always answer
zero — only accepts (extracted morpheme breakdowns) became
`toponym_etymology` rows. This made it impossible to audit harvest
quality, identify books worth re-mining with a different model, or
reconstruct a per-book history without re-running the full LLM
pipeline.

The `wyrd kenning lexicon import-mining-log <path>` CLI back-fills
the table from a JSONL file of historical runs (one record per
run). It's how prior batches that pre-date the writer get folded
in: agents can extract them from session transcripts, hand-curated
logs, or wherever the stdout was captured.

## D24. When you add data, add an observability path.

Every persistent operation must be queryable from the DB. If the
mining pipeline computes a number, that number must land somewhere
queryable — a row, a column, a view — not just in stdout. If a CLI
emits `accepted=X declined=Y rejected=Z`, the same numbers must
also be in a `mining_run` row (or analogous audit row).

Why: D23 was filed because the mining pipeline had been printing
accept/decline/reject counts to stdout for two days without anyone
noticing the data was unrecoverable. The user asked "what's the
state of the harvest?" and the DB couldn't answer; we had to mine
session transcripts to recover the numbers. The fix was small; the
discovery was what cost us.

How to apply: when designing or reviewing any feature that mutates
the lexicon, ask three questions in order:

1. **What does this write?** Lemma links, OCR clusters, text-match
   rows, mining-batch results, future enrichment outputs.
2. **What stage stamps it?** A `method` / `mode` / `provider` column
   so future-you can ask "which version of this heuristic produced
   this row?" (D22 lemma_method, D23 mining_run.mode, etc.)
3. **What query answers 'how is X doing?'** If the answer is "grep
   stdout" or "rerun the pipeline," there's a missing column or
   row. Add it.

Counter-example to watch for: a "results summary" that exists only in
the closing log line of a CLI command. That's the wyrd-ej4 shape.
Catch it at design time, not after two days of data loss.

## D25. LLM disambiguator gates on candidate-set size, not gloss anchor.

When fuzzy-search finds a body word X within edit distance ≤ 1 of more
than one canonical etymon, the gloss anchor (D15) is too coarse to
choose between them — generic glosses overlap, scholarly prose tends to
gloss multiple morphemes near the same word. The disambiguator
(`wyrd kenning lexicon disambiguate-fuzzy`) takes over here.

Cost gate: only ambiguous rows reach the LLM. A fuzzy row with exactly
one candidate etymon (after the within-distance scan) is left alone —
the gloss-anchored heuristic was sufficient.

Verdict storage: `etymon_text_match.method` becomes
`'llm-disambiguated-v1'`, the model's one-sentence reason lands in a
new `disambiguator_reason` column, and the row's `etymon_id` may be
reassigned if the LLM picked a candidate other than the original.
"None" answers delete the row.

Why: doing this in code would require encoding the linguist's
intuition about which morpheme fits a passage. The LLM already has
that intuition. Spending ~$0.0001 per ambiguous row is cheaper than
the cumulative wrong-attribution noise of leaving it heuristic. The
audit trail (method + reason) lets future-us re-run with a stronger
model when one exists.

How to apply: any future "we found multiple plausible answers" code
path should follow the same shape — find_ambiguous_X → cost-gate →
disambiguate → record verdict + reason — rather than hand-coding
disambiguation rules. wyrd-uct extends this with an agentic
expand_context loop for cases where the initial snippet isn't
sufficient.

## D26. Schema metadata adds via per-language sibling fields.

When an export feature needs to carry per-form metadata that the
existing language form arrays can't express, the bundle uses a parallel
sibling field keyed by `<lang>_<feature>`. Examples currently in the
bundle:

- `<lang>_variants`: list of `{form, weight}` entries for D18 spelling-
  variant sampling (`old_english_variants`, `celtic_mix_variants`, …).
- `<lang>_inflections`: list of `{form, inflection}` entries for D8
  inflection metadata.
- `<lang>_citations`: list of source_id strings for wyrd-9kh.1
  scholarly attribution (rando-port filtered out so only real scholars
  surface).

The runtime's `load_meanings` strips off these suffixes (using
`_VARIANT_SUFFIX`, `_INFLECTION_SUFFIX`, and `_CITATIONS_SUFFIX`
constants) before populating `Meaning.sources` so canonical language
form arrays stay clean.

Why this shape over alternatives:

- **Backward-compatible by construction.** Old code that ignores
  unknown fields keeps working; a generator that hasn't upgraded its
  `load_meanings` still loads the bundle.
- **Avoids replacing the language array** with a richer structured type.
  The base shape `"old_english": ["aecern"]` stays a flat list of strings
  that existing bundle-loading code understands.
- **Easy to extend.** A new feature (e.g. `<lang>_phonology` for D6
  `--harsh`) drops in alongside without disturbing existing fields.

How to apply: any future bundle metadata that's per-form-per-language
should follow this `<lang>_<feature>` pattern, with the runtime's
loader stripping the suffix and routing into a dedicated `Meaning`
attribute. Don't pile metadata into the language array as nested
objects (would break the legacy load path); don't introduce a top-level
metadata-only entry (would orphan the link to the canonical forms).

## D27. Etymological descent is a separate graph, not a citation axis.

Wiktionary etymology + Descendants sections produce a directed graph
that doesn't fit the toponym-decomposition shape:

- **Decomposition** (existing pipeline): `toponym → [etymon, etymon, ...]`.
  Mawer / Skeat / Joyce dictionaries break a place name into morphemes.
  Each etymon row collects extraction citations from multiple scholars.
- **Descent** (new — this entry): `etymon → parent_etymon → grandparent`.
  Wiktionary asserts that OE *tūn* descends from Proto-Germanic *\*tūnaz*,
  which has Descendants in OE, ON, modern English, modern Icelandic,
  modern German, etc.

The shape is fundamentally different and lives in a new `etymon_descent`
table:

```sql
CREATE TABLE etymon_descent (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_id   INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  child_id    INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  edge_type   TEXT NOT NULL CHECK (edge_type IN (
                'inheritance', 'borrowing', 'calque',
                'compound', 'derivation', 'cognate', 'unknown'
              )),
  source_id   TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
  confidence  TEXT CHECK (confidence IN ('high', 'medium', 'low')),
  notes       TEXT,
  UNIQUE (parent_id, child_id, edge_type, source_id)
);
```

Edge types map to Wiktextract template kinds: `{{inh}}` →
`inheritance` (high conf), `{{bor}}` → `borrowing` (high), `{{cog}}` →
`cognate` (medium, peer not chain), `{{der}}` → `derivation`
(medium), `{{cal}}` → `calque`, `{{compound}}` / `{{affix}}` →
`compound`, free-text "compare with X" → `unknown`. Documented in
`INGESTION.md` once wyrd-4rt actually populates them.

**Source attribution** is per-edge (not per-node). Wiktionary fits as
a single `'wiktionary'` source row for v1 — per-edit attribution lives
in the wiki history and per-language-section slicing is overkill until
needed.

### Why descent does NOT contribute to `etymon_consensus`

The `witnesses` count measures *extraction* witnesses (D4: "N
scholars formally identify this morpheme as part of a toponym
breakdown"). Descent is a different axis — it relates morphemes
across language and era. Counting descent edges as extraction
witnesses would inflate the count and break the ≥3-witnesses
promotion threshold; a morpheme with one real scholar citation but
ten Wiktionary descent edges would auto-promote despite being
under-attested for actual place-name use. Cognate clustering is the
correct rollup for descent.

### Cognate clustering: materialized via `etymon.cognate_id`

(Historically named `synset_id`; renamed in wyrd-44a to remove a
naming collision with the wyrd-7tz `meaning_synset` table — that
table carries SEMANTIC equivalence; this column carries
ETYMOLOGICAL descent.)

The cognate set = transitive closure of inheritance + borrowing
edges from a common root. A separate `cluster-cognates` enrichment
pass (filed as wyrd-81n) walks the graph and writes
`etymon.cognate_id` pointing at the most-ancestral known etymon.
All etymons reachable from that root via inheritance/borrowing
share the same `cognate_id`.

Materialized (column) rather than derived (recursive CTE view)
because cross-language cognate-cluster queries would walk unbounded
depth on every call; materialization makes the query a single JOIN.

The `cognate` edge type does NOT bridge cognate clusters — it's a
peer relationship Wiktionary uses when two languages have lexically
similar forms but the chain isn't pinned. Bridging on cognate would
over-unify; we want cognate-cluster assignments to require an explicit
ancestor.

### Reference queries

The descendants / ancestors recursive CTEs and the post-wyrd-81n
cognate-cluster query live in `INGESTION.md` under "Etymological
descent graph (D27)" — that's the operational manual; this entry
holds rationale.

### Why this matters beyond Wiktionary mining

Three downstream consumers blocked on this schema:

- **wyrd-4rt** — Wiktionary mining via wiktextract. Without
  `etymon_descent`, ingestion has nowhere to put the chain data.
- **wyrd-7tz** — synset / cross-language equivalence layer. The
  Descendants tree IS the cognate cluster; once descent edges land,
  cluster-cognates produces the cognate-cluster assignments.
- **Cross-cultural rendering** (no ticket yet) — "render this English
  place name in Welsh-coded form" needs the equivalence graph to swap
  each English morpheme for its Welsh cognate.

### Generality

The schema accepts descent edges from ANY source, not just
Wiktionary. Existing dictionary mining that says "from OE tūn" could
populate `etymon_descent` rows too — Mawer / Skeat / Ekwall already
make these chain assertions in passing. Extracting them is a
follow-on; the schema is general enough.

## D28. Two equivalence axes: cognate (etymological) vs meaning_synset (semantic).

After wyrd-44a + wyrd-7tz Phase 1 (PRs #61, #63), the lexicon
distinguishes two distinct axes of "these morphemes are related":

- **Cognate cluster** — `etymon.cognate_id` (renamed from
  `synset_id` in wyrd-44a). Etymology-based: every etymon reachable
  via inheritance + borrowing edges from a common Proto-* root
  shares one cognate_id. OE `tūn` / ON `tún` / Icelandic `tún` /
  modern English `town` all point at PGmc *tūnaz. Populated by the
  `cluster-cognates` enrichment pass against `etymon_descent`.

- **Meaning synset** — `meaning_synset` table + `etymon_meaning_synset`
  join. Semantic equivalence: every etymon assigned to the
  `water/flowing` synset shares "flowing water" as a sense, regardless
  of etymological relation. OE `wæter`, OE `strēam`, ON `bekkr` are
  all in `water/flowing`; OE `mere` is in `water/body`. Populated by
  manual seed (Phase 1) + LLM-assisted classification (Phase 2,
  upcoming).

The two axes are orthogonal:

- A pair sharing cognate_id is etymologically related but may diverge
  in sense (English `silly` and German `selig` are cognates but mean
  different things now).
- A pair sharing a meaning_synset is semantically equivalent but may
  be etymologically unrelated (OE `wæter` and ON `bekkr` are not
  cognates but both mean "flowing water" in place-name use).

Generator transforms use one axis or the other based on intent:

- **Anglicize / drift-toward-X**: meaning_synset (find a same-meaning
  morpheme in the target language).
- **Calque** (structural translation): meaning_synset (White Hill →
  Albus Mons → Bryn Gwyn).
- **Cognate-substitute** (etymological reverence): cognate_id (replace
  modern `town` with the historical reflex `tūn`).
- **Drift toward archaic register**: cognate_id (walk back to a more
  ancestral reflex of the same etymon).

The naming collision before wyrd-44a ("synset_id" was used for both
the cognate-cluster column AND the new semantic-equivalence concept)
was technical debt; the rename clears it.

## D29. Trie-indexed segmentation DAG matcher.

The matcher (wyrd-k8e Phase 1, PR #64) is a trie-indexed segmentation
DAG with multi-path enumeration — NOT a pure trie matcher. The trie
is the character-level prefix index; the matcher walks it from every
input position collecting EVERY accepting state along the path
(including intermediate ones), then DFS-enumerates every full path
through the resulting segmentation DAG. Multi-parse is the load-
bearing invariant: a word with two senses for one surface (-y =
'island' OR 'district') OR competing breakdowns (`-ham-` AND
`-hamlet-` both in trie → 'hamlet' surfaces as both `ham + let` and
`hamlet`) surfaces ALL parses.

Public API (`wyrd/generators/kenning/runtime/trie_matcher.py`):

- `all_decompositions(word, trie)` — every parse, including
  partial-match alternatives that fall through the skip-into-
  unaccounted branch.
- `canonical_decompositions(word, trie)` (plural) — every parse tied
  for 'best' under the `(unaccounted_chars, morpheme_count)` score.
  Plural for callers that need every reading the explainer would
  surface.
- `canonical_decomposition(word, trie)` (singular) — ONE
  deterministic pick. Tiebreaker is list-index of the first
  matched element; ties beyond that fall back to meaning_db
  insertion order.

Composition with the legacy matcher (Phase 2): `Name.find_meaning`
gets a flag to choose trie or iterator; Phase 3 makes trie the
default and removes the iterator. Phase 1 ships standalone.

Why this matters:

- ~200x speedup vs the legacy O(M)-per-recursion-level iterator on
  routine corpus passes (10K toponyms × 1500 morphemes).
- Multi-decomposition becomes a graph walk instead of a heuristic
  reduce + dedupe-by-signature.
- Inflection resolution at match time is the natural Phase 2
  extension if inflected variants are inserted directly into the
  trie source data.

## D5-3. Era runtime filter (wyrd-lyp, PR #57).

Refines D5-2: the era-cell mapping (D5-2 design) now has a runtime
counterpart. The kenning generator's `--era` knob (CLI) /
`era` input_schema property accepts a year (`1086`), a cell label
(`oe-late`, `me`, `middle-irish`), or a `family/label` pair
(`english/oe-late`).

Implementation:

- `Meaning.attested_in_era_range(era_range)` — predicate on the
  per-language `attested_years` data (D5-1 mining output).
- The vector path's eligibility gate applies this per-lemma
  (`eligibility.passes_era_gate` / `vector_name_select._matches_era`,
  both delegating to `attested_in_era_range`), shrinking the eligible
  pool BEFORE scoring. `era=None` is a no-op gate.

  (The original proportions implementation precomputed an allowed-usage
  frozenset via `MeaningGenerator.keep_keys_for_era` and intersected it
  at the bucket level inside the proportions sampler; that path was
  retired with proportions scoring.)

`era=None` is bit-stable with the pre-PR sampler. The 2026-05-04
bundle re-emit (PR #58 / wyrd-j5v) populated `_attested_years` on
1215 / 3807 words (31.9% density), activating the filter end-to-end.
Empirical narrowing on the English keep-set: 2901 → 2327 (oe-early)
/ 2433 (oe-late) / 2690 (me) / 2323 (early-modern) / 2246 (modern).

Bare-label resolution is strict: a label not in the culture's default
era family raises ValueError listing the families that DO define it,
prompting the user toward the explicit `family/label` form. No silent
cross-family fallback — it's too easy to accidentally route an
'english' culture's `--era middle-irish` to the goidelic range.

> **Filter half SUPERSEDED 2026-06-12 by D44 (same day as the refinement
> below).** The era INVENTORY filter described in this entry — including the
> keep-set narrowing numbers and the wyrd-c6o1.3 open-ended-window refinement
> below — is retired: era never gates (or weights) the morpheme draw at all.
> What survives from D5-3: the era-input grammar (year / cell label /
> family-label), the strict bare-label resolution, and the resolver as the
> request VALIDATOR + render-language anchor. See D44.

**Refinement (wyrd-c6o1.3, 2026-06-12): open-ended windows pass every
morpheme.** An era window with `end=None` — the present-day / `modern`
stage of every living family, and what the culture-agnostic
`present-day` token (wyrd-kqyf, the deployed `WYRD_DEFAULT_ERA` in both
envs) resolves to — no longer requires an attestation year inside the
window. Place names accrete: the present contains every historical
stratum, and the bundle's scholarly attestations are inherently
medieval (etymology dictionaries cite Domesday-era forms), so the
year-inside-window rule was unsatisfiable-by-construction for the core
OE corpus. The observable failure: once `present-day` became the
deployed default era, `tūn`/`-ton` (≈20% of real British place names)
and every other well-documented OE morpheme silently vanished from
default generation, while thinly-documented homographs (welsh `ton`
'wave') passed via the no-data rule and absorbed the surface's picks.
The perverse incentive was structural — the better a morpheme's
scholarship coverage, the more reliably it was excluded. Bounded
historical windows (`oe-late`, `me`, …) keep the year-inside-window
semantics as a deliberate period-flavor knob. Regression gate:
`tests/test_kenning_present_day_core_morphemes.py`.

## D17 refinement: cohesion knob (wyrd-mj2, PR #59).

— **SUPERSEDED 2026-05-18 by D36 (vector-driven generator architecture); body archived → `archive/superseded-decisions.md`.** The cohesion math/rationale carry over to the vector scorer's multiplier; the named proportions-era helpers (`Generator.select`, `_cohesion_boost`, `_raw_class_score`) are gone. The tag-co-occurrence data it consumed is still defined by D16.

## D30. wyrd-ami fantasy-name research is a sibling pipeline, not a new generator.

The fantasy-name pipeline (introduced 2026-05-06; see OVERVIEW.md
"Sibling pipeline") researches creature names like Harpy, Djinni,
Tiamat against the existing etymon corpus. It runs alongside the
place-name mining pipeline but uses the *same* lexicon: same
`etymon` table, same `etymon_descent` graph (D27), same descent-
walking logic. What's new is a single result table `fantasy_morpheme`
that records the (input → resolution) mapping per pipeline version.

The shape is **inverse** to place-name mining:

- **Place-name mining** (D1, D4): scrape scholarly dictionaries to
  *populate* the etymon table. Rows accumulate citations from
  multiple scholars; consensus rolls up via `etymon_consensus`.
- **Fantasy-name research**: take a *single* creature name and walk
  it back to an existing etymon. The etymon table is read-only from
  this pipeline's perspective. `fantasy_morpheme` rows record where
  the input landed, not what got contributed to the corpus.

### Schema

```sql
CREATE TABLE fantasy_morpheme (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  -- COLLATE NOCASE so 'Harpy' and 'harpy' are the same row.
  input_name          TEXT NOT NULL COLLATE NOCASE,
  input_description   TEXT,
  usable              INTEGER NOT NULL CHECK (usable IN (0, 1)),
  etymon_id           INTEGER REFERENCES etymon(id) ON DELETE SET NULL,
  bar_reason          TEXT,                    -- when usable=0
  resolution_method   TEXT NOT NULL,           -- 'descent_lookup' | 'llm_full_research'
  approach_version    TEXT NOT NULL,           -- 'fantasy-v1' today
  confidence          TEXT,
  citation            TEXT,
  reasoning           TEXT,
  unapproved_language TEXT,                    -- LLM said this lang; we don't approve it
  unapproved_form     TEXT,                    -- the historical form the LLM identified
  processed_at        TEXT NOT NULL,
  UNIQUE (input_name, approach_version)
);
```

### Bar-reason taxonomy

The pipeline's failure modes are distinct on purpose so each can be
fixed by a targeted intervention:

- `modern_coinage` — game designer invented (Vrock, Werebear); not
  fixable.
- `outside_language_family` — real etymon in a language not in
  `APPROVED_LANGUAGES`. Fix by approving more languages.
- `attested_but_not_in_corpus` — LLM identified a real etymon in an
  approved language, but our `etymon` table doesn't have a row for
  that `(canonical_form, language)` pair. Fix by backfilling the
  etymon corpus (epic wyrd-ialp).
- `uncertain_attestation` — low-confidence LLM response (often a
  network timeout that degraded gracefully). Fix by re-running.
- `no_etymology_found`, `homograph_collision`, `proper_noun_only` —
  edge cases.

The `unapproved_language` + `unapproved_form` columns capture the
LLM's finding even on a barred row, so a future "languages to
consider approving" report can rank candidates empirically.

### Why a separate pipeline, not a new generator

Town-name generation reads the lexicon's morpheme inventory and the
descent graph. Fantasy-name research uses the same data — it just
*identifies* which morphemes a creature name maps onto, so the
generator's existing temporal-axis machinery (descent chain → era
filter → register-shifted form) Just Works for fantasy content. No
new generator class, no new bundle schema, no runtime change.

### Family-root + variant emission

The pfsrd2 monster extractor emits both the family root and each
single-word variant name when distinct (Genie + Djinni + Efreeti +
Marid + Shaitan + Janni from the genie family). Earlier collapsing
all variants to the family root silently lost etymologically distinct
morphemes (Djinni from Arabic *jinn* is not the same morpheme as
Genie from Latin *genius*; both are real, both worth researching).
The single-word strict filter still rejects multi-word names
("Bugbear Thug", "Ancient Black Dragon").

### Approved-language stratification (precursors + postcursors)

`APPROVED_LANGUAGES` is the union of every code along the descent
chain of an approved canonical language, not just the canonical
language itself. (The container is a flat `frozenset`; "stratified"
here means the *contents* span multiple eras of the same chain, not
that the data structure is ordered.) Each canonical language pulls
in its precursors and postcursors so `descent_walking_lookup` can
walk the FULL chain from a modern reflex to its deepest attested
ancestor without bailing at a language gate. The Hebrew chain, for
example, isn't just `he` — it's `{hbo, sem-wes-pro, sem-pro,
afa-pro, he}` (Biblical Hebrew → West-Semitic → Semitic → Afroasiatic
+ Modern Hebrew), and the Iranian chain mirrors the pattern
(`{peo, fa-cls, xpr, ira-pro, fa, pal, iir-pro}`). Without the
chain form, a Modern Hebrew word's descent edge to Biblical Hebrew
terminates the walk at the language boundary even though the
relationship is genuine; with it, the BFS continues all the way to
Proto-Afroasiatic when the data carries that depth.

Practical consequence: when a new "canonical" language is approved,
its precursors / postcursors should be added at the same time so
the descent walker doesn't accumulate dead ends at intermediate
nodes. The cost is small (each code is one line) and the upside is
that downstream temporal-axis demos (wyrd-rni / wyrd-381) can
render the same morpheme at any era stop the corpus carries data
for. User policy 2026-05-06: precursors / postcursors of an
approved language can be added without separate sign-off; new
*canonical* languages still require explicit approval (the wave-2
canonical set was negotiated carefully).

Two related implementation knobs:

- `CANONICAL_LANGUAGES` vs `APPROVED_LANGUAGES`. The LLM prompt
  template uses the smaller `CANONICAL_LANGUAGES` set (the user-
  facing fantasy register) rather than the full `APPROVED_LANGUAGES`
  superset; the precursor/postcursor stack is pipeline machinery the
  LLM has weak training signal on. `_classify_llm_result` still
  accepts any `attested_in` value that lands in `APPROVED_LANGUAGES`
  after alias normalization, so an LLM that happens to identify a
  Pali/Coptic attestation directly still resolves — we just don't
  *advertise* the obscure codes in the prompt.
- `_dedupe_by_form_lang` sorts reconstructed canonical forms (those
  starting with the linguistic-convention `*` prefix) AFTER attested
  forms. With the proto-* stack now reachable via BFS, both an
  attested ancestor and a reconstructed root can surface at the same
  layer; the sort guarantees `_skip_llm_resolution` and
  `_semantic_check_candidates` see the attested form first.

### approach_version

`extractors.fantasy.APPROACH_VERSION` is a pipeline-version stamp.
The (input_name, approach_version) UNIQUE means re-running with the
same version updates rows in place; bumping the version (e.g. when a
stronger pipeline ships) writes parallel rows so the old results stay
queryable. The `--skip-resolved` flag scopes to the current version
— callers can't accidentally skip rows from an older pipeline run.

## D31. wyrd-ha9q Phase 2a — multi-script renderings sit alongside canonical_form.

When wyrd-ha9q (Phase 2 of the wyrd-ami sibling pipeline) needed to add
pronunciation + multi-script renderings to the etymon table, the
choice was between two options. Option A (chosen): keep `canonical_form`
as wiktextract gave it (often native script for Hebrew / Arabic / etc.)
and add the new renderings as parallel columns. Option B (rejected):
migrate `canonical_form` for non-Latin-script rows so it holds academic
transliteration, with native script moving to a new column.

Reasons:
- `(canonical_form, language)` is the corpus-wide UNIQUE key. A
  migration that re-shapes canonical_form for ~85k Hebrew/Arabic/
  Persian rows changes the meaning of every existing reference: every
  descent edge that resolves a parent by canonical_form, every
  extractors.fantasy LLM-anchor lookup, every fuzzy-search index.
- The wiktextract ingester writes `canonical_form = entry.word`. A
  migration that diverges from this would need both a one-time bulk
  UPDATE and an ongoing translation layer in the ingester.
- The SPA's etymological-provenance panel only needs the four
  renderings (original_script, transliteration, english_shaped, IPA)
  to exist and be retrievable; it doesn't require canonical_form to
  be any specific kind of value.

Concretely the schema adds five nullable TEXT columns to `etymon`:
`pronunciation_ipa`, `pronunciation_dialect`, `original_script`,
`transliteration`, `english_shaped`. Phase 2a populates the first
four from wiktextract's `sounds[*]` and `head_templates[0].args`
(via the upsert COALESCE-on-conflict pattern, so a re-ingest backfills
NULLs without clobbering existing values). `english_shaped` stays
NULL until Phase 2b's IPA-driven derivation lands.

The `head` head-template arg is an Arabic / Egyptian fallback when
`wv` (the cleaner Hebrew-style key) is absent. `head` values are run
through `_is_clean_native_form` to filter out multi-form display
strings (`'و / و'`), affix markers (leading/trailing `ـ` or `-`), and
reconstructed `*-prefixed` proto-forms before storage.

For non-Latin-script source rows the SPA panel reads original_script
when present, falling back to canonical_form (which IS the native
script for those rows) — so display callers don't need to know the
partition.

### Phase 2b: english_shaped derivation rules

`english_shaped` is populated by `english_shaping.derive_english_shaped`
in priority order:

1. **`KNOWN_FORM_OVERRIDES`** — case-insensitive lookup against the
   pre-strip transliteration. Cultural / literary precedent: rakṣasa
   → rakshasa, ʿifrīt → ifrit, gōlem → golem, ǧinn → jinn, šayṭān →
   shaitan, etc. The dict is hardcoded in the module (~70 entries
   spanning Sanskrit creature canon, Arabic / Persian creature names,
   Hebrew biblical creatures, Akkadian Mesopotamian, Egyptian
   deities, Aramaic). Overrides win even when the rule pipeline
   would produce a serviceable output — the goal is to match what
   English readers expect, not what's mechanically derivable.
2. **Rule-based diacritic stripping** of the transliteration. Two
   passes: digraph replacements (run before single-char strip; ṯ→th,
   ḫ→kh, ǧ→j, š→sh, ġ→gh, ḏ→dh, ś/ṣ→sh, ʒ→zh, θ→th, χ→kh) then
   single-char replacements (emphatics ḍ/ḥ/ṭ/ẓ/ḳ→base, retroflex
   ṛ/ṇ/ḷ→base, Sanskrit palatal/velar nasals ñ/ṅ→n single-char so
   the following consonant supplies the palatal/velar quality
   without double-counting (liṅga→linga, añjali→anjali), long
   vowels ā/ī/ū/ē/ō→base, stress-acute á/é/í/ó/ú→base, IPA
   ɪ/ʊ/ɛ/ɔ/ɐ/ə→nearest English vowel, Hebrew rafe ḇ→v, Arabic ḥā
   ħ→h, IPA pharyngeals ʿ/ʾ/ʔ/ʕ silenced). Per-language overrides
   on top: Arabic ṣ→s instead of the default sh (ṣabr→sabr,
   ṣalāt→salat).
   Final `_looks_english_readable` check rejects outputs that still
   carry non-ASCII residuals (the maps don't cover everything;
   better NULL than half-stripped).
3. **IPA fallback** when transliteration is absent. Strips IPA
   delimiters (slashes, brackets, length marks, stress marks,
   tie-bar) and runs the same digraph + single-char pipeline.
   Coarser than (2); intended for the small wave-2 tail where we
   have IPA but no transliteration.

Returns NULL when the source language is in `_LATIN_SCRIPT_LANGS`
(canonical_form is already English-readable for those — no
derivation needed).

CLI: `wyrd kenning lexicon derive-english-shaped --apply` runs the
backfill across every NULL row in the wave-2 non-Latin set; takes a
`--language he` filter for incremental smoke runs and `--reshape` to
re-derive non-NULL rows after editing the override table or rules.

Empirical first-pass coverage (post-Phase-2b backfill, 2026-05-07):
fa 13,777 / he 12,889 / ar 11,548 / sa 5,028 / egy 2,618 / arc 1,245 /
akk 965 = 48,070 rows populated. Skipped 85,034 rows without
sufficient transliteration / IPA input.

### Phase 2c: bundle plumbing for english_shaped + wave-2 lang fields

Phase 2c wires the `english_shaped` column through to the runtime in
two cuts:

1. **Wave-2 bundle field expansion.** Pre-Phase-2c, the lexicon's
   `_LANG_CODE_TO_JSON_FIELD` map dropped non-Latin source-lang codes
   (`he` / `ar` / `fa` / `sa` / `akk` / `egy` / `arc` / `pal`) on
   the floor — `_emit_word_languages`'s `if not json_field: continue`
   silently skipped them. Phase 2c adds eight new bundle field
   names (`hebrew`, `arabic`, `persian`, `sanskrit`, `akkadian`,
   `egyptian`, `aramaic`, `armenian`) and routes the canonical
   wave-2 codes plus their precursor / postcursor stack codes
   (`hbo` → hebrew, `peo` / `fa-cls` / `xpr` / `pal` / `ira-pro`
   → persian, `iir-pro` / `inc-pro` / `pra` / `pi` → sanskrit,
   `cop` → egyptian, `syc` → aramaic, `sem-pro` / `sem-wes-pro` /
   `afa-pro` → hebrew, `sux` → akkadian, `axm` → armenian) into
   them. Same pattern as `celtic_mix` already bundles welsh /
   old-welsh / middle-welsh / scottish-gaelic / etc.
2. **`<lang>_english_shaped` sibling arrays.** Per the D26
   sibling-suffix pattern (variants / inflections / citations /
   attested_years already use it), Phase 2c adds
   `<lang>_english_shaped` arrays of `{form, english_shaped}`
   entries to each wave-2 word in the bundle. Sparse — only forms
   whose `english_shaped` column is non-NULL emit; rows that
   lacked transliteration / IPA at ingest time stay in the language
   form array but don't pollute the shaping sibling.

Runtime:
- `meaning._ENGLISH_SHAPED_SUFFIX = "_english_shaped"` lets
  `load_meanings` strip the suffix and route the data into
  `Meaning.english_shaped: dict[lang_field, dict[canonical_form,
  english_shaped]]`.
- `Meaning.english_shaped_for(lang_field, canonical_form)` is the
  accessor; returns the English-friendly rendering or `None` when
  no shaping is available (Latin-script source lang, or the form
  lacked sufficient input at derive time).

Out of scope for Phase 2c: the actual generator-side preference
for `english_shaped` over `canonical_form` when rendering surface
forms. The current generator's `Meaning.__str__` returns
`modern_usage` (the English-side key), not per-language forms, so
the existing town-name render path doesn't yet consume the new data.
Per-language rendering is the wyrd-rni / wyrd-381 era-rewind demo's
concern — those need to pick non-English source-lang renders, which
is exactly when `english_shaped_for` becomes the right surface-form
source. Phase 2c is the plumbing those demos consume.

> **SUPERSEDED by D41 (wyrd-24s6).** Per-language / native rendering is no longer
> "the demos' concern" — it is the canonical render. Every name now renders BOTH
> native (source-era, canonical) AND modern. See D41.

### Phase 2d: SPA etymological-provenance panel + the other 3 renderings

Phase 2d completes wyrd-ha9q's runtime story by surfacing the four
renderings (per-form for non-Latin source-lang morphemes) in the
SPA's etymology disclosure UI:

  1. **Native script (`original_script`)** — vocalized form from
     wiktextract's head_templates `wv` / `head` arg.
  2. **Academic transliteration** — Latin-script form with diacritics
     from head_templates `tr`.
  3. **English-shaped** — the wyrd-ha9q derive_english_shaped output
     (Phase 2b shipped, Phase 2c plumbed through bundle).
  4. **IPA pronunciation** — from `sounds[*].ipa` paired with
     `sounds[*].tags[0]` as the dialect tag.

Three sibling-suffix arrays added to the bundle (mirroring Phase 2c's
english_shaped pattern): `<lang>_original_script`,
`<lang>_transliteration`, `<lang>_pronunciation`. The pronunciation
sibling carries an extra `dialect` key alongside `ipa` (NULL when the
IPA was the untagged-canonical form per wiktextract's sounds array).

Runtime layer adds three more Meaning attributes
(`original_script`, `transliteration`, `pronunciation`) with
matching accessors (`original_script_for`, `transliteration_for`,
`pronunciation_for`). All three default to empty dicts on legacy
bundles; accessors return None when no data is available.

`NewName.components()` (the API envelope shape consumed by the SPA)
gains a `renderings` field per component. Shape:

    "renderings": {
      <lang_field>: {
        <canonical_form>: {
          "original_script": str | None,
          "transliteration": str | None,
          "english_shaped":  str | None,
          "ipa":             str | None,
          "dialect":         str | None,
        },
        ...
      },
      ...
    }

Sparse — only forms that have at least ONE non-None rendering land in
the dict. Forms with no rendering data don't appear; lang_fields with
no shaped forms don't appear; consumers can iterate uniformly without
branching on emptiness.

SPA layer:
- `_renderProvenancePanel(components)` builds a `<details>` disclosure
  ("Etymological provenance") when ANY component carries renderings.
  Skipped entirely for English / Celtic / Romance generation, so the
  UI stays compact for the common case.
- `_renderProvenanceRow(langField, canonicalForm, slots)` renders one
  row per (lang_field, canonical_form), surfacing whichever of the
  four renderings the lexicon has — labeled `native`, `translit`,
  `english`, `ipa`. The IPA cell appends `(dialect)` when the dialect
  tag is non-null.
- CSS in `style.css` follows the existing `.citations` panel
  conventions (accent label, monospace value cells).

Out of scope deferred from Phase 2d: town-name generator preference
for english_shaped over modern_usage at render time. That stays the
wyrd-rni / wyrd-381 era-rewind demos' concern — Phase 2d delivers
the EDUCATIONAL view (panel) but the GENERATION default still uses
modern_usage everywhere (bit-stable historical behavior).

> **SUPERSEDED by D41 (wyrd-24s6).** The generation default is NO LONGER
> modern_usage everywhere. `era=""` now renders native (as-selected); modern is
> the always-present secondary, not the default. See D41.


## D32. Within-language stratum tagging (wyrd-lr4, PRs #105 / #107 / #109 / #111 / #112 / #113 / #115 / #120 / #121).

The per-etymon `language` column is too coarse for some palettes —
`language='welsh'` blends Brittonic substrate, Latin loans, medieval
Welsh, and English borrowings into one bucket, and a Welsh-flavored
generator pulling indiscriminately from all of them produces
stylistic mush. D32 partitions each language family's etymons into
named within-language register buckets so the runtime `--stratum`
filter can target a specific layer.

### Schema

`etymon.stratum TEXT` plus an index on the column. Sparse — only
languages with a Phase 1 classifier populate it; legacy / mining-
backfill rows stay NULL.

### Per-family vocabularies

Each language family ships a `STRATA` tuple (priority-ordered) and
two maps (`*_ANCESTOR_TO_STRATUM` for the ancestor-walk pass on
the modern variety, `*_SELF_LANGUAGE_TO_STRATUM` for ancestor /
period varieties). The shared `_classify_family` engine in
`strata.py` runs both passes; per-family `classify_<lang>` wrappers
supply the constants.

  * Welsh (5 buckets): native-welsh / latin-loan / english-loan /
    brittonic-substrate / medieval-welsh.
  * French (5 buckets): native-french / frankish-substrate /
    gaulish-substrate / gallo-roman / medieval-french. **Latin and
    vulgar-latin are deliberately ABSENT from the French ancestor
    map** — every French word descends from Latin via the standard
    Romance path, so including them would collapse the bundle into
    one bucket and erase the substrate distinctions. Same descent-
    absence rationale for Old English (gmw-pro / proto-germanic /
    proto-indo-european excluded) and Old Norse (gmq-pro /
    proto-germanic / etc. excluded).
  * Old English (4 buckets): native-old-english / latin-loan /
    norse-loan / celtic-substrate. The umbrella ticket scoped
    West Saxon / Mercian / Anglian dialect strata, but the live
    DB has only 33 dialect-coded etymons (orphans without descent
    edges) — too sparse for own buckets. Self-language map left
    empty so wave-2 dialect-corpus mining can populate later.
  * Old Norse (6 buckets): native-old-norse / east-norse /
    latin-loan / low-german-loan / english-loan / gaelic-substrate.
    East Norse fires only via self-language (`gmq-osw` Old Swedish
    + `gmq-oda` Old Danish) — the ancestor walk doesn't apply
    because old-norse is the PARENT of those varieties, not the
    other way around.
  * Brittonic (2 buckets, wyrd-de77): native-brittonic /
    proto-celtic-substrate.
  * Goidelic (4 buckets, wyrd-de77): native-goidelic /
    proto-celtic-substrate / old-irish / middle-irish (priority
    order: substrate first, then the attested Gaelic stages, default
    last — the Welsh convention).
  * Celtic (2 buckets, wyrd-de77): native-celtic /
    proto-celtic-substrate. The coarse / unbranched tag, used when
    the source didn't distinguish P- vs Q-Celtic.

The three Celtic-family classifiers (wyrd-de77) cover the coarse
`brittonic` / `goidelic` / `celtic` language tags — 476 of the
>=2-toponym admit cohort were unclassified before this phase. The
data drives the lean design:

  * **87% of Celtic admits are orphans** (only ~60 of 476 have any
    `etymon_descent` parent edge), so the overwhelming majority land
    in the native-* default bucket.
  * **No loan buckets.** The data carries ZERO Norse / Anglo-Norman /
    French / English loan ancestries on these rows, so — unlike
    Welsh / OE / ON — these families ship no loan strata.
  * **proto-germanic parents are mining NOISE** (mis-traced edges)
    and are deliberately NOT mapped, mirroring every existing
    classifier (none map proto-germanic).
  * **All three are LEAF tags** (`brittonic` / `goidelic` / `celtic`
    are never the PARENT of another etymon in the data), so each has
    an EMPTY self-language map — there's no self-language pass, the
    ancestor walk over modern_lang rows does all the work.

Cross-family stratum names overlap (`latin-loan` exists in Welsh /
OE / ON; `english-loan` in Welsh / ON). The value alone is
ambiguous about which family it belongs to; family-aware validation
must consult the per-family `STRATA_BY_FAMILY` tuples.

### Priority order convention

Iteration order on `STRATA` is the priority order: when an etymon's
parents include languages mapped to multiple strata, the first
match wins. Each family's order encodes a scholarly-historical
convention (institutional / clerical signals like latin-loan
displace contact loans like norse-loan, which displace residual
substrates, which displace dialect axes, which fall through to
default). Welsh's "Latin > Brittonic" encodes the convention that
explicit loans displace inherited forms.

### No-data passes through

Etymons with `stratum IS NULL` (every Meaning's
`stratum_for(lang_field, form)` returning None) ALWAYS pass any
`--stratum` filter. Without this rule the filter would gut
unclassified-language bundles (post-Phase-4-shipped, only Welsh /
French / OE / ON have classifier output; Latin / OS / IE / etc.
all admit). As classifier coverage grows the rule tightens
naturally.

### Per-culture validation

`_resolve_stratum_param(stratum, culture)` validates request-side
input against `_CULTURE_TO_VALID_STRATA[culture]` (the union of
STRATA tuples for language families that culture's place-name
corpus draws from). Catches both typos AND culturally-incoherent
values (`--culture welsh --stratum east-norse` 4xxs because
east-norse isn't in any classified Welsh-bundle language family).
Since wyrd-de77 every culture has a non-empty per-culture
restriction (the Celtic-family classifiers gave irish / breton
their allowed-sets, and widened welsh / scottish to include their
Celtic-family substrate). The culture↔family mappings are a
curation call flagged for review: irish = Goidelic + Celtic;
breton = Brittonic + Celtic + French; welsh = Welsh + French +
Brittonic + Celtic; scottish = the British spread + Goidelic +
Brittonic + Celtic. The `LANGUAGE_TO_FAMILY` map is built
programmatically from each classifier's `*_SELF_LANGUAGE_TO_STRATUM`
keys + the modern_lang strings, so adding a new self-language
entry (e.g. an Old French dialect variety) auto-extends the
family-aware validation without a separate manual list.

### Bundle export shape

Per-form `<bucket>_stratum` siblings on each subject word,
mirroring the Phase 2c/2d wyrd-ha9q rendering siblings:

```
{
  "modern_usage": "Caer-",
  "celtic_mix": ["caer", "din"],
  "celtic_mix_stratum": [
    {"form": "caer", "stratum": "latin-loan"},
    {"form": "din",  "stratum": "native-welsh"}
  ],
  ...
}
```

Bucket-level union: when multiple lexicon language codes funnel
into one bundle bucket (`welsh + middle-welsh + old-welsh + cel-bry-pro`
all → `celtic_mix`), per-form stratum values accumulate via
`bucket.stratum.update(...)` in lang-sort order — last-lang-wins
on collision (matches the `english_shaped` / `pronunciation`
plumbing's contract). Each Meaning then carries `stratum: dict[
lang_field, dict[canonical_form, stratum_tag]]`; the `stratum_for(
lang_field, form)` accessor returns `tag | None`.

### Idempotency contract for hand-corrections (Phase 4d)

`lexicon set-stratum` lets an operator manually correct individual
etymon stratum values. `classify-stratum --apply` (without
`--force`) skips rows that already have a non-NULL stratum — so
hand-corrections survive subsequent bulk-classifier runs. The
load-bearing `AND stratum IS NULL` clause on the UPDATE WHERE
preserves this; removing it would silently blow away every
operator override on the next classify pass.

Pinned end-to-end by
`test_cli_set_stratum_survives_classify_stratum_apply_without_force`.
Documented at the SQL site (`cli.py:_build_case_update`) AND in
`set-stratum`'s docstring AND in the test name itself, so a future
editor of either side sees the contract.

### Granularity caveat

Filtering happens at the USAGE level, not the SENSE level. A usage
with two Meaning instances (one `native-welsh`, one `latin-loan`)
stays in the pool when `--stratum native-welsh` is requested; the
downstream pick at `NameGenerator._pick_surface` falls back to
`meanings[0]` for variant/inflection rendering, which may surface
the wrong-stratum sense. Same caveat applies to `--era`. Sense-
level filtering would need `_render_substitutions` reworked.

### Bundle re-emit activation

Until the bundle is re-emitted post-classifier-apply, the bundled
`meanings.json` carries no `_stratum` siblings — every Meaning
hits the no-data passthrough and the filter is bit-stable with
no-flag. The classifier output is captured in the wyrd-lr4 ticket
notes (per-language smoke counts) and ready for the operator's
re-emit pass.

## D33. Time-aware era-reflex infrastructure (wyrd-skm + wyrd-rni + Phase 3.3, PRs #108 / #110 / #114 / #116-118 / #122).

Renders the same etymon at multiple historical strata. The story
spans authoring (mining + projection) and runtime (CLI rewinder +
SPA Generator). Three lookup tiers + bundle plumbing.

### Three-tier era-reflex picker

`lexicon.etymon_era_reflexes(etymon_id, target_language=...)` is
the single primitive every consumer reads. It tries three lookup
paths in order and returns the first non-empty result:

1. **Cognate cluster** — when `etymon.cognate_id` is set
   (D27 + D28 cluster_cognates output), select cluster mates of the
   target language. ~24% of OE toponym etymons today; this is the
   high-quality path because cluster_cognates output is
   human-vetted via the merge pass.
2. **Direct descent fallback** — when `cognate_id` is NULL but the
   etymon has `etymon_descent` rows, walk the immediate children
   via `inheritance` + `borrowing` edges (peer `cognate` edges
   excluded — too loose for v1). Recovers ~4% additional coverage
   on the OE toponym etymons that have descent edges but never
   reached the cluster pass.
3. **Period-form projection (`etymon_period_form`)** — when both
   cluster + descent return empty AND the caller passed
   `target_family_cell`, query `etymon_period_form` rows whose
   `date_year` falls in the cell's year range. Closes the ~72%
   coverage gap on isolated OE etymons. Tier 3 results carry
   `etymon_id == etymon_id` (the queried etymon) and the cell's
   canonical language tag (since `etymon_period_form` rows don't
   carry an explicit language).

All three tiers filter `merged_into_id IS NULL` so OCR-cluster
losers (D22) don't surface as period forms; the merge winner is
the canonical voice.

`era.canonical_language_for_cell((family, cell))` is the
era→language adapter: `('english', 'me')` resolves to
`'middle-english'`. Cells missing from the dict produce `[]`
from the picker — there is NO family-level fallback.
`era.era_cell_for_input(input, default_family)` parses CLI/API
era values directly to `(family, cell)` without round-tripping
through year ranges (the round-trip loses information for
open-low cells like `oe-early` whose `start=None`).

### Schema additions

- `toponym_attestation` (table existed, was empty pre-Phase-3.0a):
  `(toponym_id, form, date_year, source_doc)` with idempotent
  unique index. Populated by `lexicon mine-attestations` from
  `toponym_etymology.notes` body text.
- `etymon_period_form` (new in Phase 3.3): per-etymon period-keyed
  surface forms, FK to `etymon` + soft FK to `toponym_attestation`.
  Unique index on `(etymon_id, form, date_year, source_doc)` for
  idempotent re-projection.

### Mining + projection commands

- `lexicon mine-attestations [--apply]` — extracts `(form, year)`
  pairs from `toponym_etymology.notes` via 5 high-precision regex
  patterns:
  * `FORM in YEAR` / `FORM, YEAR` / `FORM, in YEAR`
  * `FORM in Domesday[ Book]` / `Domesday has FORM` / `D.B. has FORM` / `FORM, D.B.`
  * `;FORM YEAR` (chain-anchored bare connector for trailing chain elements)
  Three precision gates:
  * Connector requirement (comma, semicolon, or `in`) between form
    and year — drops `"After 1066"` / `"Source 1234"` flow-text FPs.
  * Lowercase-required form filter — drops scholarly source
    abbreviations (LPR / LI / LF / DB / MS) without an explicit
    per-token list.
  * Source-attribution-chain detector (PR #117) — suppresses
    `<year> <source>, <year>` shapes like
    `"Chevington 1535 VE, 1539 Wills, 1544 LP"` where Wills is
    the source name in a multi-source chain, not a place form.
  * Page-marker rejection both pre-year (`p. 1086` shape via
    `_TOPONYM_NOTE_PAGE_MARKER_RE`) and post-year (`1086 (p. 59)`
    via `(?!\s*\(p+\.)` lookahead).
  Year range 700-1700 (post-Roman through pre-modern). Welsh
  (ŵâêôûŷ + ē) + Norman (çéè) + OE (æðþœǣĒĀĪŌŪ) diacritics covered
  in the form-character class.
  Live: 1,476 rows / 669 toponyms / 29 sources.
- `lexicon project-period-forms [--apply]` — segments the
  historical compound against the toponym's binary breakdown via
  suffix-anchoring against the last morpheme's known reflexes
  (canonical_form + cognate-cluster mates + etymon_variants). The
  remaining prefix is the first morpheme's projected period form.
  V1 limits: binary breakdowns only (ternary alignment isn't
  reliable without phonetic distance); suffix-anchoring only;
  ≥2-char projected segments. Skips breakdowns whose components
  point at OCR-cluster losers (`merged_into_id IS NOT NULL`).
  Live: 755 rows; ~85-90% precision on 40-row stratified
  spot-check. The few suspect projections trace to upstream
  scholarly extraction noise, not the projector's algorithm.
  Eager-loads breakdowns in one query (`_preload_binary_breakdowns`)
  to avoid N+1.
- `lexicon clear-enrichment --stage=attestations|period-forms` —
  reverse paths; `all-derived` rolls them in.

### CLI rewinder (`wyrd-rni`)

`wyrd kenning rewind <name>` decomposes via the existing
`Name + meaning_db` matcher, anchors each morpheme to its
source-language etymon, and renders the compound at multiple era
stops. Default ladder: 3 English-family stops (oe-late / me /
modern); `--era` repeated for custom ladders.

Anchor-resolver design notes:
- `_ANCHOR_LANG_PREFERENCE` walks OE → ON → OldFrench → Celtic → Latin
  → ModernEnglish in priority order.
- **Hyphen-variant sibling lookup**: the bundle keys morphemes by
  hyphen-marked usage (`-ton` vs `ton` vs `Ton-`). When the trie
  matches the no-hyphen Celtic Meaning at `ton`, the OE
  post-modifier Meaning at `-ton` carries the right anchor —
  resolver collects siblings across all keys whose stripped-hyphen
  form matches. Cached via `_get_stripped_index(meaning_db)` keyed
  by `id(meaning_db)` for O(1) lookup.
- **Three-tier picker preference** (matches Tier 4 of Claude review
  on PR #114):
  * Tier 1: case-insensitive match for `morpheme.canonical` (modern
    usage). Handles mining-artifact cases where archaic forms get
    co-tagged as modern-english (Wiktionary cross-reference
    entries) — alphabetical-first would surface `cyning` as the
    "modern reflex" of OE `king`, breaking the era progression.
  * Tier 2: case-insensitive match for `anchor.canonical_form`
    (source form). Handles morphemes whose orthography didn't
    shift much across eras (`mynster → mynster → minster`) —
    alphabetical-first would otherwise return a noise mate
    (`amounten` for OE `mynster` at ME).
  * Tier 3: alphabetical first.
- **Fallback rule**: when no era reflex found, render the
  morpheme's modern canonical (NOT the anchor's OE source). The
  <!-- D41 (wyrd-24s6): this is a FALLBACK for a missing era reflex, not a
  statement that modern is THE canonical surface. Per D41 native is canonical;
  modern is the parallel/secondary rendering. -->
  asterisk `*` flag in the CLI output is the truth-marker. Falling
  back to the source would make the era ladder look reversed
  (`oe-late: king → me: chinge → modern: cyning`).

### Bundle plumbing for SPA-side rewinder (wyrd-obpw)

The Lambda runs on bundled data (the lexicon DB is 673MB —
too big to ship). To enable a SPA `KenningRewind`, era-reflex
data is precomputed at bundle-build time:

- `lexicon._fetch_family_era_reflexes(db, member_ids, root_language)`
  computes `{target_language: [forms]}` for the whole family by
  UNIONing the same picker across every member (wyrd-rogd.16) — not
  just the root, so a folded reflex's cluster isn't orphaned. Wired
  into `_gather_family` so each family carries `era_reflexes` data.
- `_emit_era_reflexes(word, link_pairs)` stamps the family root's
  reflexes onto the word entry's top-level `era_reflexes` field.
  Multiple linked families merge by set-union per target language.
- Runtime: `Meaning.era_reflexes` (dict[target_lang, list[form]])
  parsed at `load_meanings`. `Meaning.era_reflex_for(target)`
  returns sorted-list copy. Empty for legacy bundles.

`era_reflexes` is a TOP-LEVEL field on each word entry, NOT a
per-language sibling. It represents the family root's cluster
reflexes — one set per family, not per source language. The
`load_meanings` pipeline excludes `era_reflexes` from the
per-language `sources` dict.

### KenningRewind generator class

Registered alongside `KenningExplain`. Input schema `{name}`.
Renders the input across 3 default English-family era stops
(oe-late / me / modern). Per-Meaning era reflex picked via
canonical-match preference (matches CLI rewinder's tier-1);
falls back to alphabetical first, then `morpheme.canonical` when
no era data. **No DB access** — reads exclusively from
`Meaning.era_reflex_for`.

The CLI rewinder (`era/rewind.py:rewind_name`) and `KenningRewind`
have parallel small picker logic. The full `EraReflexProvider`
protocol abstraction is **deliberately deferred** until wyrd-381
(stratified era-map) lands as a second consumer — designing the
protocol with one consumer would bake in assumptions the second
might want to change.

### Coverage limits + extension paths

The remaining ~72% of OE toponym etymons that have neither
cognate_id, descent edges, nor period-form projection (because
their toponym wasn't binary-decomposable or the suffix-anchor
algorithm couldn't align the historical form) still return `[]`
from the picker. Two orthogonal closure paths:

- **Phonological-rule fallback** (wyrd-4i6 — sound-change rules
  library). Forward-derives surface forms from canonical via
  per-language phonetic transforms. NOT data-driven; rules-driven.
- **Mining + re-projection**. Each new English-corpus mining run
  expands `toponym_attestation`; re-running `project-period-forms`
  picks up newly-recoverable suffix anchors.

### Bundle re-emit dependency

Until the bundled `meanings.json` is re-emitted post-Phase-3.3,
`KenningRewind` reads empty `era_reflexes` and falls back to
canonical (still functional, just no era progression visible in
the SPA). The deploy ticket is `wyrd-j43l` (P1).


## D34. SPA-shortlist Phase A: respelling, alt-scripts, corpus evidence, era-map (PRs #125 / #127 / #128 / #129).

Four user-visible features layered on top of the D33 era-reflex
infrastructure. Each ships as a Generator class (so the SPA
exposes it via `/api/generators`) and/or a CLI subcommand. None
introduces new data; all compose existing primitives.

### wyrd-17t: pronunciation respelling (PR #125)

`Meaning.respelling_for(form, language)` delegates to
`runtime/respelling.py`, a per-language SAMPA-lite rule table
(Old English, Welsh, Old Norse, Latin, Greek, Norman-French).
`KenningRewind` components surface respellings inline so users
who can't sound out a non-modern-English morpheme see a
reading hint next to the rendered form. Modern-English passes
through with `respelling=None`. Atomic alternation guards
against rule-chaining bugs (`ff → f → v` → `ffynon` becoming
`vuhnon`).

### wyrd-y10: alternate-script transliteration (PR #127)

`runtime/scripts.py` exposes `transliterate(text, script)` and
`SUPPORTED_SCRIPTS = ("shavian",)`. `KenningRender` Generator
wraps it. v1 ships Shavian (~48 glyphs, U+10450-U+1047F);
Tengwar / Cirth / Elder Futhark / Ogham drop in as additional
dispatch arms. Lossy grapheme-based heuristic — Read Lex
(~30K-word phoneme-precise dictionary) is filed as a future
refinement. Hyphens / spaces / digits pass through verbatim
so compound names retain structure.

### wyrd-bvp: corpus-evidence annotations (PR #128)

`annotate_fragments_with_corpus_evidence` extends
`wyrd kenning unaccounted` with a `--sources-dir` flag. Each
top-N fragment is annotated with `corpus_hits` (distinct
source files where the fragment word-boundary-matches),
`snippets` (~60-char context), `in_etym_body` (heuristic flag
when the snippet's left context contains a year-citation OR a
source marker like A.S. / O.E. / M.E. / cf. / from), and
`strong_hits` (count of in_etym_body snippets). Reuses
`_load_normalized_source_texts` from the existing
reverse-search machinery. Future SPA integration would surface
the same evidence next to KenningExplain output, but Lambda
has no DB — needs either bundle-time precomputation or a
separate API endpoint.

### wyrd-381: stratified era-map (PR #129)

`KenningEraMap` Generator + `wyrd kenning era-map` CLI.
Bulk-rolls N invented toponyms (drives the underlying
`Kenning.generate` loop with `seed+i` since Kenning is
`multi_result=False`) and renders each at the three English-
family era stops (oe-late / me / modern). Each result carries
`{name, era_cells: [{era, family, rendered, morphemes,
unaccounted}, ...]}`. The killer GM-handout pattern is the
"Domesday-vs-modern map" pair — same morpheme stack, different
eras of paper.

**Coverage caveat (unchanged from D33):** names whose
morphemes lack mined `era_reflexes` render uniformly across
all eras. Not a bug — a coverage limit. Expanding
`era_reflexes` mining is the natural follow-on for the
kenning-data-mining track.


## D35. Bundle is dict-shape going forward (wyrd-c1vq, PR #145).

`lexicon export-meanings` always emits dict-shape
`{"subjects": [...], "canonical_decompositions": {...},
"joiners": {...}}` — empty optional keys are omitted to keep
the rendered JSON tight, but the top-level shape is always a
dict, never the legacy list of subjects.

Why: future bundle additions (joiners pool, fantasy morpheme
runtime data, per-language tag projections) need named keys.
Re-deciding the shape each time a field lands fragments the
forward-compat policy. The runtime loaders (`load_meanings`,
`load_canonical_decompositions`, `load_joiners`) already route
through `meaning._bundle_subjects`, which tolerates BOTH list-
shape (legacy bundles checked into git pre-2026-05-08) and
dict-shape, so flipping the export default does not break any
consumer.

Joiners sidecar path: `--joiners-from PATH` on
`lexicon export-meanings` reads
`{lang_field: [{"form": str, "weight": int}, ...], ...}` and
folds it into the bundle's `joiners` key. Phase 2 of
wyrd-q0g6 / wyrd-semi (manual audit + pool population) will
produce the sidecar; the export wiring is in place ahead of
the data.

Defensive `collect_canonical_decompositions`: returns empty
dict when the `toponym_decomposition` table is missing
(older DBs predate wyrd-08m Phase 1 migration). Prevents
the bundle export from crashing on unmigrated DBs.


## D36. Vector-driven generator architecture (wyrd-ecjp, wyrd-kq7w.1/.2, 2026-05-18).

The flat per-culture proportions JSON + scalar `--mood` /
`--harshness` / `--cohesion` / `--novelty` knobs are replaced
by composable vector scoring across four axes, hard
eligibility gates outside vector space, and pack-overlay
composition for scenario packs. Foundation for the fantasy-
language epic (wyrd-v2gm), the narrative-translator epic
(wyrd-sreb), and the rip-and-replace mood-system work
(wyrd-kq7w).

Nine sub-decisions locked 2026-05-16, consolidated here as
the spec entry for Phase 1 (wyrd-ecjp.1). The in-memory
representation lives in
`wyrd/generators/kenning/vectors/schemas.py`; the catalog
authoring source is
`wyrd/generators/kenning/data/register_effects.yaml` (Phase B,
wyrd-kq7w.2). Both file paths are part of the API contract
for downstream phases.

### D36.1. Four scoring axes.

Generation scores each candidate lemma against four
independent axes:

* **Phonological** — feature vector per lemma. v1 has 14
  named dimensions (cluster_density, final_fortition,
  final_cluster_rate, vowel_final_bias, soft_consonants,
  polysyllabic_bias, palatalization, sibilance, retroflexion,
  pharyngeal, vowel_height, vowel_backness,
  stop_vs_continuant, aspirated_voiceless) plus an `extras`
  forward-compat slot. Computed once per etymon from
  canonical_form + IPA; persisted alongside the etymon row
  (`PhonologicalVector`, schema half of wyrd-kq7w.1).
* **Semantic** — tag-weight dict per lemma. Reuses the
  existing meaning-database tag set (death, plant, water,
  saint, religious, military, magic, etc.). No new schema
  on the lemma side — the semantic-score is a dot product
  between the request's semantic-tag weight vector and the
  lemma's tag membership (1.0 if tagged, 0.0 if not).
* **Position** — slot-prior dict per lemma. Position labels
  are free-form strings, not a closed enum — the data shape
  supports arbitrary positions (first-element / second-element
  / folk-connector like `-inga-` / manorial-affix like Mandeville
  / locative-phrase like upon-Tyne / hundred-prefix / bishopric-
  prefix / etc.) as the corpus surfaces them. The v1 catalog
  migrates the existing proportions-shape position labels
  unchanged; later positions append as new string keys without
  a schema migration. The "first vs second" framing in informal
  prose is illustrative shorthand only — the scoring runtime,
  the priors-keyed lookups, and the register-effect
  `position_bias` dict all treat position as an opaque string.
* **Empirical-baseline** — frequency prior per (culture,
  position, tag, era) for native generation; per (donor,
  recipient, position, tag, era) for pack overlays. Derived
  artifact: Phase 2 (wyrd-ecjp.2) materializes this from the
  lexicon DB. Re-runnable / versioned (see D36.9).

Hard gates (culture, era, stratum, pack allowlist/tag-filter)
live OUTSIDE the vector space as boolean predicates — they
shrink the eligible-lemma pool BEFORE any scoring happens.
A lemma either matches or it doesn't; soft preferences
belong on the vector axes.

### D36.2. Canonical composition rule.

For each candidate lemma in each slot, the total score is

```
score(lemma) = phon_w * phon_score(lemma, request)
             + sem_w  * sem_score(lemma, request)
             + pos_w  * pos_score(lemma, request, slot)
             + base_w * baseline_score(lemma, source, request)
```

where `phon_w` / `sem_w` / `pos_w` / `base_w` are the per-axis
scalar weights from `ScoringWeights`. Default weights are
all 1.0 (every axis contributes equally). The user-facing
knobs map onto these:

* `--register harsh:0.8,grim:0.5` composes the harsh and grim
  register effects (scaled by 0.8 and 0.5), sums them, clamps
  to [-1, +1] per dimension, and uses the result as the
  request's `register` vector. The axis weights stay at
  default; the DIRECTION of preference changes.
* `--realism 0.5` halves `base_w`. The empirical-baseline
  axis contributes half as much; non-baseline-favoured
  lemmas score relatively higher. Per D36.3 below, this is
  the ONE realism knob — there's no separate concept.
* `--novelty` was the D17 uniform-marginal blend (not a per-axis
  weight; see D36.5). It was removed with proportions scoring and is
  re-wired onto the vector path (wyrd-fcub); the `--cohesion`
  half of D17 survives as a multiplicative boost in the vector scorer.

`baseline_score(lemma, source, request)` reads from the
empirical-priors artifact: `source=native` looks up
`priors.native[(culture, position, tag, era)][lemma]`;
`source=pack P` looks up
`priors.loan_relationship[(P.template_donor, P.template_recipient,
position, tag, era)][lemma]`, weighted by `P.weight`.

### D36.3. Empirical baseline is an axis, not a separate concept.

Pre-2026-05-16 thinking included a "realism" or "baseline-
retention" axis distinct from the per-axis weights. That
created an N²-knob problem (every register-axis weight could
in principle be modulated by realism independently) and
confused the auto-blend semantics when register effects
engaged. D36.3 collapses realism into a single axis weight
(`base_w`) in the canonical formula — no separate `realism`
flag, no "baseline-retention" property on register effects.
The auto-blend semantics fall out of the formula itself; no
hidden re-weighting needed.

Pack weight is INDEPENDENT from base_w (D36.4). The operator
can ask for "high realism + low pack" (base_w=1.0,
pack.weight=0.2) or "low realism + heavy pack" (base_w=0.2,
pack.weight=2.0) as orthogonal knobs.

### D36.4. Pack lemmas inherit their template's empirical baseline (Option B).

A scenario pack (Khuzdul, Tatar, Polynesian, etc.) declares a
TEMPLATE relationship — `(template_donor, template_recipient)`
— that maps onto a real-world loan relationship the pack is
modeled on. Khuzdul-templated-on-ON→OE uses the ON→OE
loan-relationship empirical baseline as its
baseline-axis reference. A bare invocation of `--culture
english --pack neo-khuzdul` (no other knobs) produces output
where Khuzdul lemmas appear at the rate REAL Old Norse loans
empirically appeared in Old English place-naming.

Pack weight (`pack.weight`, default 1.0) is a multiplier on
the pack's baseline contribution. weight=0 produces native-
only generation even with the pack declared; weight=1 is the
historical loan rate; weight>1 over-weights the pack. The
multiplier is purely on the baseline axis; the pack's lemmas
still go through the same phonological/semantic/position
axes as native lemmas, so a request for `--register harsh`
biases pack lemmas toward harsh ones too.

Two packs simultaneously declared compose additively on the
baseline axis. Multi-pack composition semantics live in
Phase 7 (wyrd-ecjp.8) — for the spec lock we just say "their
baseline contributions sum" without locking the exact
normalization-across-packs rule (which Phase 7 calibrates
against empirical data).

### D36.5. D17 cohesion preserved via adapter, not re-derived.

> **Outcome note:** as specified, D36.5 kept the D17 novelty + cohesion
> blend on the proportions sampling layer behind an adapter. When
> proportions scoring was retired, the **`--novelty`** blend was removed
> (re-wired onto the vector path, wyrd-fcub) while **`--cohesion`**
> was carried over as a multiplicative boost inside the vector scorer.
> The original design below is kept for rationale.

The existing D17 Bayesian-mixture novelty + cohesion model
stays the runtime sampling layer. The vector-driven
generator produces per-lemma vector scores; a
`CohesionContext`-wrapped scorer applies the existing
tag-class-prior multiplier (`key_boost` in the then-current
`Generator.select` API) BEFORE the novelty blend
(`(1-novelty)·boosted + novelty·uniform`).

In other words: D36 replaces the WEIGHT-COMPUTATION machinery
(harshness scoring, tag-include filter, etc.) but plugs into
the existing SAMPLING machinery unchanged. `--novelty` and
`--cohesion` semantics stay byte-stable; the Braitham Gate
regression test from D17 must still pass after the
NameGenerator rewrite (wyrd-ecjp.5).

The adapter glue lives between Phase 4 (vector scoring) and
Phase 5 (NameGenerator rewrite). Phase 4 emits per-slot
vector scores; the cohesion adapter wraps them with the
context-conditional boost from picked-tags-so-far; Phase 5's
slot walk calls the wrapped scorer.

### D36.6. Culture is a hard gate in v1.

`--culture english` excludes everything not in the english
stratum, period. Soft culture blends ("Welsh-50%-English") are
future expressivity, not v1. The hard gate makes runtime
behavior predictable and indexes cleanly (Phase 3,
wyrd-ecjp.3, pre-computes per-culture eligibility sets at
bundle-build time).

The `--tag` knob also stays (per Q6 user direction). A
positive-tag filter is a hard gate; lemmas not carrying the
requested tag are excluded BEFORE scoring. Exclude-tags
(wyrd-yan) likewise stay as a hard exclusion filter.

> **SUPERSEDED (D47, wyrd-c6o1.4):** the positive `--tag` filter is NO
> LONGER a pool-wide pre-score gate. It reserves ONE slot per name; the
> rest sample freely. See D47. (Culture hard-gate + exclude-tags here
> still stand.)

The vector model's semantic-axis weights provide the SOFT
version of tag preference (this is the "lean toward death-
themed names" knob); the existing `--tag` is the HARD version
("only names with the death tag"). Both coexist.

### D36.7. Per (culture × position × tag × era) priors granularity.

The empirical-priors artifact is keyed at (culture, position,
tag, era) for native cells and (donor, recipient, position,
tag, era) for loan-relationship cells. Era buckets are
integer-year mid-points of D5's era model (pre-Conquest,
Domesday, late-medieval, early-modern, modern). Tag is a
single semantic-tag string — lemmas carrying multiple tags
appear under each independently, and the baseline-score
combinator aggregates across tags at scoring time (the
aggregation rule is Phase 2's call: sum / max / weighted-by-
request-tag-weights).

Per D7 in the parent epic locking: this granularity matches
the existing meaning-db's tag/era surface, so no new lemma-
side schema is needed beyond what wyrd-kq7w.1 adds. Sparsity
in low-coverage cells (e.g. Old Norse × stratum-marginal ×
era-marginal) is handled by Phase 2's smoothing rule, also a
Phase 2 decision.

### D36.8. Tolerance bands deferred to Phase 6a.

The drift-measurement work (Phase 6a, wyrd-ecjp.6) measures
how priors shift between regenerations. Tolerance bands —
the thresholds at which a priors shift is judged
"significant" — are calibrated against empirical drift, not
declared a priori at the spec lock. Phase 6a runs a series
of priors regenerations across mining-checkpoint snapshots,
measures per-cell distributions of the inter-version delta,
and proposes bands sized to bracket typical drift. Phase 6b
(wyrd-ecjp.7) locks those bands into the regression test
suite.

### D36.9. The priors artifact is a re-runnable deterministic derived artifact.

Priors extraction (Phase 2) is a derived artifact, not a
one-shot import. Re-running on the same lexicon DB state
produces a byte-identical priors file; re-running after new
mining lands produces a new versioned file. Downstream caches
+ bundle builds key on the priors version so a regenerated
artifact invalidates only what depends on it.

Operational implication: the framework code (Phases 1, 3, 4,
5, 7, 8) is stable across data evolution. The priors and
the bundles built from them re-roll as the corpus grows.
Tests parametrize on priors rather than asserting against
specific values — bad pattern: `assert score("Edwarston")
== 0.847` (breaks every regen); good pattern: `assert
score(name) > score(distractor)` (relative ordering,
robust to priors evolution). Frozen test-priors snapshots
are used where a specific value is needed; they regenerate
from a fixed-seed corpus subset.

### Why this shape.

Three load-bearing properties of the architecture:

1. **Composability.** Multiple register effects compose
   additively (`--register harsh:0.5,grim:0.7,ancient:0.3`).
   Multiple packs compose additively on the baseline axis.
   The four scoring axes compose additively per the
   canonical formula. The CLI / API surface stays small
   because each axis is independently meaningful.

2. **Data evolution without code churn.** New mining data
   regenerates priors; the framework code stays put. The
   register catalog YAML lets the operator add a new
   register effect (e.g. `melodic`) without a code change.
   Phase 6's drift bands quantify "how much priors evolution
   is benign."

3. **Pack overlays without bespoke pack-rule code.** A new
   scenario pack just declares its (template_donor,
   template_recipient) and ships lemmas; the same scoring
   runtime + priors lookup handles every pack identically.

### Cross-references.

`wyrd-ecjp.1` (this spec). Depends on `wyrd-kq7w.1` (Phase A:
per-lemma phonological-vector schema + corpus enrichment
pass — schema half locked here, enrichment pass is the
follow-on work) and `wyrd-kq7w.2` (Phase B:
register_effects.yaml catalog format + composition rules —
format locked here, catalog content-population is the
follow-on work).

Blocks: `wyrd-ecjp.2` (Phase 2 priors extraction),
`wyrd-ecjp.3` (Phase 3 eligibility-gate runtime),
`wyrd-ecjp.4` (Phase 4 vector-scoring runtime),
`wyrd-ecjp.5` (Phase 5 NameGenerator rewrite),
`wyrd-ecjp.6/7` (Phase 6 drift measurement + tolerance),
`wyrd-ecjp.8` (Phase 7 bundle + pack overlay),
`wyrd-ecjp.9` (Phase 8 CLI + integration + docs). Downstream
consumers (wyrd-v2gm, wyrd-sreb, wyrd-kq7w) listed in the
introduction above.

### Implementation status (2026-05-19).

Nine phase tickets shipped via PRs #237 / #240 / #245 / #260 /
#264 / #265 / #268 / #269 / #270 / #271 / #272 / #273.
Module locations as built:

* **Schemas** — `vectors/schemas.py` carries
  `PhonologicalVector`, `EligibilityGate`, `RegisterEffect`,
  `ScoringWeights`, `PackOverlay`, `RequestVector`, and
  `EmpiricalPriors` dataclasses. `PackOverlay` grew
  `allowed_pack_tags` + `excluded_pack_tags` in ecjp.8 for
  the pack-tag-filter operator knob.
* **Scoring primitives** — `vectors/scoring.py` ships
  `phon_score`, `sem_score`, `pos_score`, `baseline_score`
  (native + multi-pack composition via `baseline_score_pack`),
  `aggregate_score`, and the orchestrating `score(...)`
  function. Pack-baseline composition follows Option B
  (template-donor inheritance).
* **Empirical priors** — `lexicon/empirical_priors.py` carries
  the JSON-sidecar dump format + loader; CLI exposed as
  `wyrd kenning lexicon dump-empirical-priors`. Priors are
  versioned by emission timestamp; the loader validates
  schema on parse.
* **Vector selection primitive** —
  `runtime/vector_name_select.py:select_via_vector_scoring`
  implements gate → score → weighted-sample with optional
  `pack_meaning_dbs` for scenario-pack overlay support.
* **Adapter** — `runtime/vector_kenning_adapter.py:
  build_request_vector` translates Kenning's per-call knobs
  (culture / tags / mood / harshness / era / stratum /
  weights) into a `RequestVector`.
* **Dispatch** — `runtime/proportions.py:
  NameGenerator.select_via_vector` is the vector-scoring entry
  point; `generators/kenning.py:_generate_via_vector` is the
  Kenning-level dispatcher. `Kenning.generate` always routes
  through vector scoring (the original `scoring_mode` param /
  proportions branch is retired).
* **Drift measurement** — `runtime/drift_measurement.py` ships
  pure-Python metric primitives (KL divergence,
  total-variation distance, decomposition-rate delta,
  position-distribution delta, Spearman rank correlation).
  `runtime/drift_runner.py` bridges metrics to the live
  `Kenning.generate` for per-seed isolation. (The original
  relative vector-vs-proportions comparison — its top-N
  name-overlap metric and the `drift-report` CLI — was retired
  with the proportions scoring path; what survives is the
  absolute corpus-realism gate.)
* **Tolerance bands** — `runtime/realism_tolerance.py` ships
  the `AbsoluteToleranceBand` dataclass +
  `check_realism_against_tolerance` primitive, with per-culture
  bands resolved via `absolute_tolerance_for` (`ABSOLUTE_DEFAULT`
  is the wide-open fallback). (This replaced the earlier relative
  `ToleranceBand` / `check_drift_against_tolerance` /
  `PER_CULTURE_TOLERANCES` API, removed with proportions scoring.)
* **Regression suite** —
  `tests/test_kenning_realism_absolute.py` parametrizes
  per-culture (english / scottish / welsh / irish / breton) name
  generation against the absolute corpus-realism bands. A 0-sample
  result FAILs as a regression guard.
* **CLI surface** — `cli/generate.py` carries
  `--priors-path`, `--baseline-weight`,
  `--phonological-weight`, `--semantic-weight`,
  `--position-weight`. (The original `--scoring-mode` selector
  was removed once vector became the only scoring path.)

Deferred work (filed as follow-up tickets in the ecjp epic):

* `wyrd-ecjp.10` — Phase 7 bundle export changes (per-lemma
  `phonological_vector` emission from `etymon_consensus`,
  embedded `empirical_baselines` section, pack manifest
  format). Required for ecjp.11 / .12 to wire packs from
  real bundles rather than synthetic fixtures.
* `wyrd-ecjp.11` — `--pack <name>[:<weight>]` /
  `--pack-tag-filter <pack>:<tag1,tag2>` CLI flags. Blocked
  on ecjp.10 (needs pack manifest to resolve template
  donor/recipient).
* `wyrd-ecjp.12` — SPA / Lambda integration (Lambda handler
  schema updates, KenningRewind / KenningRender / KenningEraMap
  bundle-shape propagation, SPA generator class updates).
  Blocked on ecjp.10.

UPDATE (proportions-scoring retirement): vector is now the only
scoring path — the `--scoring-mode` flag and the proportions
*scoring* path were retired once the absolute corpus-realism gate
superseded the relative drift comparison. The
`<culture>_proportions.json` DATA tables survive (they feed the
vector path's frequency weighting and the corpus-realism
reference); only the sampler that scored directly off them is gone.

## D37. Phonaesthetic-vector framework supersedes the legacy MOODS dict (wyrd-kq7w, 2026-05-21).

D6 originally specified moods as a code-defined `MOODS` dict of
`{name: {tags, harshness}}` recipes living in
`registers/moods.py`. The wyrd-kq7w epic ripped that approach and
replaced it with a catalog-driven composition framework where each
named effect is a per-dimension vector triple. The MOODS dict is
gone (deleted in wyrd-kq7w.3); the catalog at
`wyrd/generators/kenning/data/register_effects.yaml` is the single
source of truth for mood-name resolution.

Why the rip:

* **Composition was lossy.** Legacy `--mood harsh:0.5,grim:0.8`
  computed a max-harshness scalar + tag union — the harsh effect
  collapsed to a single 0..1 scalar even though it conceptually
  pulls on cluster density AND final fortition AND vowel-final bias
  AND soft-consonant share AND stop-vs-continuant AND aspirated-
  voiceless dims. Per-dimension composition (sum + clamp) preserves
  the multi-axis pull intact.
* **Mood vs register were conflated.** Calling a phonological-only
  effect ('harsh') and a tag-only effect ('grim') by the same word
  ('mood') made GMs treat them as if they were on a single axis.
  The catalog explicitly separates `phonological` / `semantic_tags`
  / `position_bias` per effect, so 'harsh' is phonological-only
  and 'grim' is tag-only by construction. Operators compose them
  by listing multiple effects; the result is a single composed
  vector with the contributions explicitly visible in the catalog
  source.
* **New register names needed to ship without code changes.** The
  2026-05-16 design discussion surfaced 7 new names (noble,
  mystical, melodic, sinister, ancient, exotic, martial) that
  would have required hand-editing the MOODS dict. The YAML catalog
  + dynamic JSON-schema description (wyrd-3uzp) means a YAML edit
  is the entire addition surface — operator help / SPA UI /
  Lambda input docs all pick up the new name automatically on the
  next process restart.

### D37.1. Composition rule.

A request's register is the sum-then-clamp composition of every
effect in the request:

```
register.phonological[k]    = clamp(sum over effects: effect.phonological.get(k, 0), -1, +1)
register.semantic_tags[k]   = clamp(sum over effects: effect.semantic_tags.get(k, 0), -1, +1)
register.position_bias[k]   = clamp(sum over effects: effect.position_bias.get(k, 0), -1, +1)
```

Each effect can be graduated (`harsh:0.5` scales every dimension
uniformly by 0.5) before composition via `RegisterEffect.scaled`.
Negative weights are first-class — a register with
`vowel_final_bias: -0.4` actively penalizes vowel-final morphemes
rather than just declining to boost them.

### D37.2. Catalog format.

Each entry in `register_effects.yaml`:

```yaml
<name>:
  phonological:   {<feature>: <weight in [-1, +1]>, ...}
  semantic_tags:  {<tag>: <weight in [-1, +1]>, ...}
  position_bias:  {<position>: <weight in [-1, +1]>, ...}
```

Phonological feature names MUST be in the `PhonologicalFeatureName`
Literal in `vectors/schemas.py`. The loader validates at parse
time; typos raise loudly rather than silently no-op'ing. Semantic
tag names are open-set against the meaning-database tag set
(loader does not validate — a tag the meaning DB doesn't carry
simply contributes nothing to the semantic axis). Position bias
keys are free-form strings matching D36.1's position vocabulary.

### D37.3. Two-path resolution.

Mood-spec resolution flows through two helpers in
`registers/effects.py`:

* **Vector path** (the live scoring path): `parse_mood_spec(spec)`
  returns a graduated `RegisterEffect`. The dispatch composes
  `[adapter_effect, *catalog_effects]` via
  `compose_register_effects` into the request's register, and the
  per-lemma scoring loop dot-products this register against each
  lemma's stored `PhonologicalVector`.
### D37.4. Sourcing.

Per-effect weight directions are documented against the
phonaesthetics literature in `REGISTERS.md` (wyrd-2166 grounding
pass). Anchor sources: Whissell 1999/2000/2017 (Dictionary of
Affect in Language phonemic-emotion scores), Fort/Martin/Peperkamp
2015 (bouba/kiki replication), Ohala 1994 (frequency-code:
high-F2 = small/light, low-F2 = large/heavy), Sidhu & Pexman
2024 (meta-analysis of cross-linguistic sound symbolism),
Mooshammer 2024 (English-perception conventions for cluster /
polysyllabic dims). Citation key in `register_effects.yaml`
header: UNIVERSAL (cross-linguistic primitive), IE-CONVENTIONAL
(English-speaking operator perception), IDENTITY-MARKING
(features that mark a specific phonological color).

### D37.5. Schema dimensions.

The `PhonologicalVector` carries 17 named dimensions after the
wyrd-119p + wyrd-mkry tightenings (was 14):

* Rate features [0, 1]: `cluster_density`, `final_fortition`,
  `final_cluster_rate`, `vowel_final_bias`, `soft_consonants`,
  `polysyllabic_bias`, `palatalization`, `sibilance`,
  `retroflexion`, `pharyngeal`, `aspirated_voiceless`,
  `liquid_l_m_n` (Whissell-gentle laterals + nasals),
  `rhotic_r` (Whissell-harsh rhotics).
* Signed features [-1, +1] centered on the corpus mean:
  `vowel_height`, `vowel_backness`, `stop_vs_continuant`,
  `vowel_tenseness`.

`soft_consonants` is kept for back-compat with stored vectors;
its set is `_FRICATIVES ∪ _LIQUIDS ∪ _NASALS ∪ _APPROXIMANTS`
(where `_LIQUIDS = _LATERALS ∪ _RHOTICS` post-wyrd-119p). The
fricatives + rhotics inclusions are the misalignment Whissell
flagged — new catalog entries should prefer `liquid_l_m_n` /
`rhotic_r` for the Gentle / Harsh distinction rather than
weighting `soft_consonants` directly.
`vowel_tenseness` captures the non-monotonic tense / lax signal
Whissell 2000 documented (/iː/ Gentle, /ɪ/ Harsh; /ɔ/ Gentle,
/uː/ Harsh — catalog entries decide per-effect direction
rather than baking in a global tense → Gentle assumption).

### D37.6. What did NOT change.

D36 (vector-driven generator architecture) remains the umbrella;
D37 specifies the register-effect catalog half of D36 in
detail. The eligibility-gate predicates (culture / era / stratum
/ pack-allowlist / pack-tag-filter) are unaffected — they apply
OUTSIDE vector composition. The `harshness` scalar knob on
`Kenning.generate` still works (power-user back door); it's
translated to `_harshness_to_phonological` weights via the same
catalog-composition seam. (At the time D37 shipped the
proportion-table sampler still ran behind the default
`scoring_mode='proportions'`; the rip swapped its mood-resolution
source from MOODS to catalog, not the sampler itself. That sampler
and `scoring_mode` have since been retired — vector is the only
scoring path.)

### D37.7. Calibration owed.

The wyrd-kq7w.4 acceptance gate calls for two operator-judgment
calibration passes that this PR doesn't satisfy:

* **1000-name side-by-side**: generate 1000 names per legacy mood
  pre/post rip; spot-check operator-graded equivalence.
* **100-etymon vector audit**: 100-row sample of `etymon.phonological_vector`
  manually graded for "does this lemma's vector match its
  intuitive register?" Iterate the feature-extraction algorithm
  if precision is <80%.

Filed as a separate operator-driven follow-up under the kq7w epic
since neither can run autonomously.

## D38. L4 runtime DB: SQLite-on-S3 replaces meanings.json + proportions JSONs (wyrd-d90t, 2026-05-24 / cutover 2026-05-25).

The 2026-05-20 post-wyrd-wz82 bundle re-emit grew `meanings.json`
from 54MB to 113MB — over GitHub's 100MB push limit. Growth was
driven by the `*_phonological_vector` fields landing on every
form (wyrd-kq7w.1 enrichment). The JSON-bundle era for kenning
runtime data is over; this entry records the L4 architecture that
replaces it.

**Cutover state (2026-05-25):** the runtime reads from L4 only.
`meanings.json` + the five `<culture>_proportions.json` bundles are
deleted from the repo. The previous `WYRD_USE_RUNTIME_DB` feature
flag (PR 5) has been removed — the SQLite path is the only path.
Loaders resolve the L4 DB via, in order:
``WYRD_RUNTIME_DB`` env-var (local path override) →
``WYRD_RUNTIME_DB_BUCKET`` S3 download with ETag-keyed `/tmp` cache
→ the bundled ``seed-runtime.db`` (offline / CI fallback,
top-200-per-culture subset). The terraform now provisions the
per-env runtime DB bucket + Lambda IAM read policy + sets
``WYRD_RUNTIME_DB_BUCKET`` on the function. The build-time
``export-runtime-db`` CLI rebuilds per-culture proportions inline
from L3 + the bundled `<culture>_place_names.json` corpora, so no
intermediate proportions-JSON artifact exists in the pipeline.

### D38.1. Why SQLite-on-S3.

Five options considered, four rejected:

| option | why not |
|---|---|
| Keep JSON, use git-lfs | Doesn't fix cold-start parse cost, growing memory footprint, or queryability. Pure size workaround. |
| SQLite bundled in Lambda container image | Couples data updates to code deploys — the exact problem we're solving. |
| DynamoDB / NoSQL | Network hop per request, schema gymnastics, $/read; wrong shape for static read-only data. |
| Aurora / RDS | VPC cold-start penalty, ENI management, ops overhead; overkill for read-only generator data. |
| EFS mount | Lambda cold-start mount cost, throughput limits, more infra. |

SQLite-on-S3 is the **pfsrd2-data-api pattern** already in
production at 521 (see top-level CLAUDE.md). Same shop, same
problem class, working solution. Decouples data lifecycle from
code lifecycle.

### D38.2. Schema split: blob vs normalized.

Two design principles divide the L4 tables:

* **Blob columns where the row IS the unit of consumption** —
  `meaning`, `fantasy_morpheme`, `canonical_decomposition`. The
  runtime always fetches the whole row (one Meaning, one fantasy
  morpheme, one canonical pick) so there's no payoff to
  normalizing internal structure. Bonus: schema-stable across
  future per-form field additions; the blob just gets larger.
* **Normalized columns where SQL operates on the values** —
  `proportions_usage`, `proportions_single_usage`,
  `proportions_structure`, `proportions_tag_marginal`,
  `proportions_tag_cooccurrence`. The runtime samples weighted
  random over the proportions (~21K rows across 5 cultures on the
  current corpus) and point-looks-up tag statistics. SQL is the
  right tool there.

### D38.3. Cumulative precomputed at emit time.

Each `proportions_*` table that the runtime samples weighted-random
from carries a `cumulative INTEGER` column with
`PRIMARY KEY (culture, cumulative)`. Sampling becomes an O(log n)
index seek:

```python
total = conn.execute(
    f"SELECT MAX(cumulative) FROM {table} WHERE culture = ?",
    (culture,),
).fetchone()[0]
roll = rng.randint(1, total)
row = conn.execute(
    f"SELECT usage_key FROM {table} "
    f"WHERE culture = ? AND cumulative >= ? "
    f"ORDER BY cumulative LIMIT 1",
    (culture, roll),
).fetchone()
```

No data loaded into Python beyond the sampled row. Memory
footprint is the SQLite page cache for hot pages.

Tradeoff: cumulative columns are fragile to live updates (insert
in the middle of the distribution breaks the monotonic sequence).
Acceptable because the L4 DB is read-only at runtime and
re-emitted whole on each build — no live mutation.

### D38.4. Deferred culture column on `canonical_decomposition`.

The d90t design ticket's schema had
`canonical_decomposition (toponym_name, culture)` with a culture
PK column. The current `collect_canonical_decompositions`
(decomposition_export.py) returns a flat
`{modern_name: {signature, source}}` map with no culture data —
and the runtime's `_load_canonical_decompositions()` consumes
the same shape, culture-agnostic. There's no upstream source of
"which culture does this canonical belong to" in the L3 DB; the
`toponym` table has `region` / `country`, not `culture`.

PR 1 drops the culture column rather than emit empty strings:
`canonical_decomposition (toponym_name PRIMARY KEY, data BLOB)`.

When would adding it back be motivated:

* A toponym with two legitimate canonicals depending on which
  culture's generator is decomposing — e.g. a Welsh-flavored
  generator should pick a different parse for an ambiguous name
  than an English-flavored one. Today the matcher's culture-fed
  meaning_db handles this by ranking, so canonical-front-loading
  only needs the signature, not a per-culture branch.
* A region→culture mapping that lets the build-time projection
  attribute each canonical to its origin culture and let the
  runtime restrict its lookups. Would require designing the
  mapping (regions like Cornwall don't cleanly map to one
  culture).

Migration path: add culture as a nullable column, then a unique
`(toponym_name, COALESCE(culture, ''))` index. Existing rows stay
null (treated as "applies to all cultures"); new culture-tagged
rows take precedence on conflict via runtime lookup order.

## D39. Morpheme position slots: four forms, and post/inner are always lowercased (wyrd-5z5j, 2026-06-01).

> **Corrected by D40 (wyrd-eyjk).** The RENDER half of this entry stands: there
> are four slots, post/inner are always lowercased, only word-initial carries a
> capital, and the render is the single owner of positional case + dashes. What
> D40 corrects is the *matching* framing below — position is **derived from where
> a morpheme lands in the split**, NOT decoded from stored dash markers and used
> as a match-time constraint. Stored dashes (and `Meaning._set_location`) are a
> position *rendering hint*, never a gate on what a morpheme may match. Read D40
> before touching the matcher.

There are exactly **four** position slots a morpheme can occupy, encoded as dash
markers on its `usage` and decoded by `Meaning._set_location`
(`runtime/meaning.py`):

- `Word`   — bare / standalone word (no dash)
- `Word-`  — prefix, word-initial (trailing dash)
- `-word`  — suffix, word-final (leading dash)
- `-word-` — inner, mid-word (dashes both sides)

**The invariant: a morpheme used in `-word` or `-word-` is ALWAYS lowercased.**
Only the word-initial slots (`Word`, `Word-`) carry a capital. This is what lets
a single base form *switch* between slots — you derive the positional surface
(add the dashes; lowercase unless word-initial) from the SLOT, instead of mining
and storing four separate variants. A proper name `Buna` used mid-word renders
`-buna-`; used at a word start it stays `Buna`.

This is the original Rando model, and it is deliberately tiny — it is the entire
grammar of how morphemes compose into words and words into names. Decomposition:
break a toponym into words (on spaces), break each word into morphemes, tag each
with the slot it occupied, and tally +1 for the structure (the tuple of slots)
the toponym decomposed into. Generation: word boundaries become spaces (`Name`
joins words with `" "` via `NewName.__str__`), morphemes inside a word join with `""` (no space), and
only the word-initial morpheme is capitalized.

**The bug this guards against (wyrd-5z5j):** when the position+case model
degrades, names collapse to bare-only. `etymon_tag` showed **8932 bare** name
morphemes vs **~0** in pre/post/inner. With no positioned variant to slot,
generation drops the bare *capitalized* `Buna` into a mid-word slot, producing
capital-mid-word run-togethers (`CornnamullacBunarath` instead of
`Cornnamullac Bunarath`) and collapsing two-word structure weight.

Why: the render must derive a morpheme's surface from the SLOT it fills (dashes +
lowercase for post/inner; capital only word-initial), NOT from the morpheme's
stored case. Do NOT "fix" this by mining four stored variants per morpheme —
that is the very thing the four-slot + lowercase rule exists to avoid. Any base
form can fill any slot correctly if the render honours the slot.

**Spelling variants + inflections (added since rando) ride the same rule.** The
render is the SINGLE owner of positional case + dash markers. Every surface form
— the base usage, a substituted spelling variant (`pick_variant` /
`_pick_surface`), an inflected form — stays raw/lowercase until the render
applies the slot's markers. A substitution path must NEVER copy case from the
stored usage (the `_mimic_case` bug): a variant dropped into a `-word-` slot
lowercases exactly like the base would. If you touch the render's slot-casing,
you MUST keep variants + inflections flowing through that same single owner.


## D40. Position is a DERIVED label and a SOFT statistic — never a match-time enforcer (wyrd-eyjk, 2026-06-01).

> **READ THIS FIRST — the recurring confusion (corrected 5+ times across sessions).**
> The storage-layer half of this rule is D45 (no dashes in stored morpheme
> identity, ever — enforced by `morpheme-surface-identity-reviewer`); read both.
> Position (`bare` / `pre-` / `-inner-` / `-post`) is an **OUTPUT of decomposition,
> not an input to it.** A morpheme's identity is its **bare surface** (`giles`);
> `Giles-` / `-giles` / `giles` are the *same morpheme rendered at a derived
> position*, never separate things to match against or select between. Do NOT make
> `Meaning.location` (the stored dash-shape) gate or filter what *can* match. If you
> find yourself forcing location into the town-name deconstruction, STOP — that is
> the exact, repeated error.
>
> **The three-layer pipeline:**
> 1. **Scholarly prior** — the thousands of `is_canonical` / scholar-attributed
>    splits give the per-(morpheme, position) frequency distribution. This is the
>    ONLY role position statistics play: a credibility prior.
> 2. **Decompose the real-town corpus** — puzzle-piece each town into morphemes by
>    **string match only**; when several breakdowns are viable, a **heuristic
>    grounded in the Layer-1 prior picks the most credible**; then **record that
>    breakdown as `(morpheme, derived-position)` increments**. `Stokegiles` →
>    `Stoke-` (pre) +1, `-giles` (post) +1; `Gileston` → `Giles-` (pre) +1, `-ton`
>    (post) +1. A word-final morpheme records as `-post` regardless of which
>    dash-variant is stored. Those increments ARE the proportions.
> 3. **Generation** — samples per-morpheme position likelihoods from those
>    proportions; structure slots are keyed by position.

D39 described the four slots and (correctly) made the render derive surface from
the slot. But the surrounding machinery had the dependency **backwards**: it used
a morpheme's stored dash-position (`Meaning.location`, decoded from `-x` / `x-` /
`-x-`) as a **hard constraint on matching** — `trie_matcher._location_allows` and
its vector-path twin `vector_name_select._matches_position` would *reject* a
string-match whose stored dash didn't fit the span. That is wrong in kind and
produced clearly-bad results (lone `Andrew` matching the suffix `-andrew` instead
of bare `Andrew`; two-word toponyms like `Mount Pleasant` dropped; saint /
personal names leaking into base generation; one morpheme stored as three
position-variant entries `-andrew` / `Andrew-` / `andrew`).

### The model (what was always intended)

**Two classes of toponym:**

- **Known / pre-split** — we have the scholar-attributed breakdown (the
  `is_canonical` decompositions). Ground truth, AND the only legitimate source of
  position *evidence*.
- **Unknown** — we do not know the composition. Every split is **prospective**:
  could be two morphemes, could be fifteen. We enumerate candidates and rank them.

**Matching is string-first.** A morpheme is its base *string* (dash-stripped). To
decompose a toponym you credibly segment the *string* into morpheme strings. The
trie already does exactly this — `build_morpheme_trie` keys every entry by
`usage.lower().replace("-","")` and funnels all senses of a surface onto one node.
The trie is **not** the bug; it is position-agnostic and correct.

**Position is a derived label, computed AFTER the split** from where each morpheme
landed in the word: sole piece → `bare`, first → `pre`, last → `post`, interior →
`inner` (`Word.get_structure`, by index). The stored dashes never gate a match.

**Position rarity is SOFT evidence, never grounds for rejection.** A morpheme that
has only ever been recorded inner is "inner-only" merely because that is all the
evidence we have so far — not a property of the morpheme. History is weird and
words morph; `don` at the front of a word (`donhole`) is a *legitimate candidate*.
If a morpheme is attested 1× front / 50× middle and the toponym has less
statistically-weird parses, that weirdness should make the matcher *not select*
that parse — but the parse stays on the candidate list. Selection is by
credibility score (fewest unaccounted chars, then fewest morphemes, then a soft
per-morpheme position-plausibility term learned from the known/pre-split
breakdowns), **never** by a position gate that discards candidates outright.

### Consequences (LANDED — wyrd-eyjk)

- `_location_allows` and `_matches_position` (the dash-as-constraint gates) **were
  removed**: `_location_allows` (+ `_position_for_span` / `_location_for`) is gone
  from `trie_matcher.py`, and the `_matches_position` predicate + its call site are
  gone from `vector_name_select.py`. Every string-match now stands; position is
  derived from the span (`Word.get_structure` / `Word.get_samples`, which re-dash
  the surface to its derived position).
- The recording is keyed by bare SURFACE (the redundant per-position entries
  `-andrew` / `Andrew-` / `andrew` resolve to one morpheme via `_surface_index_for`
  + `_resolve_surface`). The derived position flows into structures + proportion
  buckets, retiring dash-based `Meaning.location` for matching/bucketing (it
  survives only as a render/scoring hint) and the interim `load_parts`
  double-register + vector bare-permissive workarounds added under wyrd-5z5j.
  Synthesized saint subjects (pure proper nouns that are saint-tagged) are kept out
  of the base pool.
- `wyrd-zewx`'s strict-inner gate (which blocked `-don-` at boundaries to avoid
  `donhole`) is removed: credibility scoring + the per-position bucket frequency,
  not a position constraint, is what ranks `donhole` below better parses.

Landed under wyrd-eyjk (P0), folding in its steps wyrd-ffut (remove the gates) /
wyrd-g6u9 (position-plausibility via bucket frequency) / wyrd-fbdb (derived position
into structures/buckets via bare-surface resolution). Residual vector-mode
realism alignment is tracked in the wyrd-vidi follow-up.


## D41. Generated names render BOTH native and modern; native is canonical (wyrd-24s6, supersedes D31's "modern_usage everywhere").

**The decision: every generated name carries TWO renderings, and we surface
both.** A **native** rendering (each morpheme in its source-era attested form —
"as selected", e.g. an Old-English-sourced morpheme renders `Tūn`, `Pearroc`)
is the **canonical/primary** render (`result`); a **modern** rendering
(`modern_usage`, the present-day surface) is the **always-present secondary**.
Neither is "the" rendering — the product decision is that we render *both*.

This SUPERSEDES the D31 (wyrd-ha9q Phase 2c/2d) position that "the GENERATION
default uses `modern_usage` everywhere (bit-stable historical behavior)" and that
native / per-language rendering is "the era-rewind demos' concern / out of
scope." That framing made `Meaning.__str__` → `modern_usage` the one true
surface and kept re-seeding a recurring bug: users select from all eras with
`era=""` but the output silently coerced every morpheme to modern, with no way
to "display what was selected."

### Behavior

- **`era=""` means "render as-selected" (native), NOT "coerce to modern."** Each
  morpheme renders in its own source-era form. `era="modern-english"` is the
  explicit way to force the all-modern rendering. `era="old-english"` etc. render
  at that requested era (unchanged). A morpheme whose source IS the present day
  stays modern (native == modern for it).
- **Both renderings are exposed** on every result: `result` (native) +
  `result_modern`, and per-morpheme native + modern surfaces in the API envelope
  (`components` / `morphemes_by_word`).
- **SPA surfaces both:** the Output column shows native primary + modern in the
  darker secondary lettering to the right; Inspect & Transform shows native on
  the left and modern on the right (the darker "MODERN" card).

### Deliberate bit-stability break

Changing the canonical `result` from modern to native **breaks the
`(generator, params, seed) → result` string contract** by design. The
`test_rogd10_parity` snapshot is regenerated once (`WYRD_REGEN_PARITY=1`) and the
break is recorded here. The bit-stability contract serves the product, not the
reverse — pinning output to a behavior the product owner calls a bug is not a
reason to keep it.

### Guard against recurrence (audit)

The "force modern" assumption was a *decision* (D31) plus a bit-stability guard,
not a stray code path — which is why code-only fixes kept regressing. As part of
wyrd-24s6 the codebase / tests / docs were audited for the assumption:
`Meaning.__str__`/`NewName.__str__` (canonical render), `_resolve_era_render_language`
returning None→modern, `_contemporary_language_for_family` suppression, the
era-reflex modern-canonical fallback (D33), and the parity snapshot. Each was
flipped to "native is the default; modern is a parallel rendering," and this
entry is the single source of truth so it cannot re-seed.

### Relationship to D33

D33's era-reflex picker is unchanged as machinery. Its **"fallback: when no era
reflex found, render the morpheme's modern canonical"** rule now means "fall back
to the morpheme's own form" — modern is no longer privileged as *the* canonical,
it is the fallback surface for a morpheme with no source-era reflex. See the
annotation on D33.

## D42. Graph DB / Cypher (graphqlite) evaluated and NOT adopted for the descent layer (spike, 2026-06-11).

**The decision: keep relational SQLite as the spine for the etymon /
`etymon_descent` graph; do NOT route the cognate / descent passes through a graph
engine.** Evaluated [graphqlite](https://github.com/colliery-io/graphqlite) — a
SQLite extension that adds Cypher + graph algorithms (WCC, SCC, Louvain,
PageRank, Dijkstra) in-process, same `.db` file, MIT, ~98% openCypher TCK. The
question was whether the genuinely graph-shaped data here (`etymon_descent`
inheritance/borrowing edges, cognate clustering, the wyrd-ami descent walk) would
be better modeled with native graph tooling. Spike conclusion: **no.**

### What the spike found

- **Performance is a non-issue.** 784,697 inheritance+borrowing edges loaded in
  7.7s; weakly-connected-components ran in 0.9s. The engine handles corpus scale
  trivially.
- **Generic graph primitives ERASE the domain semantics the passes encode.**
  The headline test — use connected-components to validate `cluster-cognates` —
  failed by design: only ~33k of ~68k components matched 1:1, with a single
  **272,905-node WCC mega-component (41.9% of the graph)**. Cause (per
  `lexicon/cognate_cluster.py`): `cluster-cognates-v2` is *not* pure WCC. It
  (1) **drops every edge touching `proto-indo-european`** (`_NON_BRIDGING_LANGUAGES`
  — PIE fans out across all IE branches; "143 of 155 clusters >200 members were
  PIE-rooted" before the fix), (2) **resolves parent/child through `merged_into_id`**
  (canonical space, D22), and (3) does a **directional, smallest-root-wins BFS**,
  not undirected closure. Re-running WCC with the PIE filter + canonical
  resolution shrank the hairball only to 150,960 (24%) — the residual is
  Latin/Greek borrowing hubs that undirected closure fuses but root-anchored
  walking correctly keeps apart. A generic WCC cannot reach the pass's partition
  without re-implementing its exact rules, at which point the engine adds nothing
  over the existing ~35-line BFS.
- **Operational cost:** graphqlite needs a Python built with loadable sqlite
  extensions. pyenv's default build lacks `enable_load_extension`; uv's
  standalone CPython works. A real friction point if it were ever a pipeline dep.

### Where the residual value is (narrow — a disposable debugging sidecar, not architecture)

If reached for at all, only as an ad-hoc analysis lens, never on the committed
pipeline: ergonomic one-off traversal queries (Cypher beats recursive CTEs for
"show the path between X and Y"); `strongly_connected_components` to characterize
the `cycle_orphans` the cognate pass only *counts* today; centrality to surface
the Latin/Greek borrowing hubs that might warrant PIE-style non-bridging
treatment. Nothing for the **runtime** (flat key-value lookups, zero traversal —
D38) or the **provenance/audit model** (relational by nature — D21/D23/D24).

Why record a rejection: so "should we use a graph DB / Cypher for the descent
graph?" doesn't get re-litigated. The graph-shaped subset here is *already*
modeled correctly by purpose-built passes whose domain rules (PIE non-bridging,
canonical resolution, root-anchoring, deterministic tiebreaks) a generic graph
engine can't see. Fast and pleasant ≠ a better model.

## D43. Bare word-sequence placement: per-WORD-position stats for bare words (wyrd-rogd.13, 2026-06-11).

Some bare standalone words have a strong **word-sequence** position bias the
model didn't capture — `Saint` is essentially always the FIRST word
(Saint Albans), the Latin postpositives `Parva` / `Magna` the LAST (Wigston
Parva; the live corpus tally is `parva` 37/37 post), `Great` overwhelmingly
first, `End` / `Hall` / `Green` overwhelmingly last. Pre-D43 any bare slot
could pull any bare word: the generator could emit "Parva Wigston".

**Vocabulary guard — two distinct position axes.** This is the WITHIN-NAME
word position (which word slot of a multi-word name a bare word occupies),
NOT the within-word morpheme position the D39/D40 dash markers encode. Bare
words have morpheme position `bare` and *additionally* get a word position.
The label reuses the pre/inner/post names; bucket keys carry it with a
`wp-` prefix (`wp-pre` / `wp-inner` / `wp-post`) so the two axes can't be
confused in code.

### The model

* **Data** — derived at proportions-build time from the place-name corpus
  walk the builder already does (`Name.get_bare_word_positions`): for each
  MULTI-word name, each single-morpheme (bare) word records one
  `(word_position, surface)` observation — first word → `pre`, interior →
  `inner`, last → `post`. Single-word names contribute **nothing** (no solo
  case: a 1-word name carries no word-sequence evidence; its observation
  still feeds the general `single_usages` pool). No new mining;
  rebuildable-from-JSONL by construction.

  The tally key is the morpheme's **identity** — the bare lowercase
  surface (D40) — NOT the matched variant's stored form. The bundle
  routinely stores case twins of one surface (`Ghyll` / `ghyll`) that
  both parse the same word; recording stored forms would double-count
  every sighting AND split it across two keys, diluting the threshold
  (each twin at 2 < 3 while the word genuinely has 2 sightings... and
  the sampler sums them back together anyway —
  `_apply_per_usage_frequency` already aggregates bucket frequency by
  surface). One name → one observation per (word_position, surface),
  and the load-side threshold also aggregates by surface defensively
  for operator-supplied form-keyed JSON.
* **Naked vs structured** — a bare word with ≥ threshold total positional
  observations is **structured**: sampled per its per-position weights and
  eligible only at positions it's attested in. Below the threshold it is
  **naked**: eligible at ANY bare slot at its general `single_usages`
  weight (which includes solo-name sightings). The generation pool for a
  bare slot at word-position P = structured-words-attested-at-P (at their
  P-weight) ∪ all naked words (at their general weight).
* **Threshold** — resolved at LOAD time from `WYRD_BARE_POSITION_THRESHOLD`
  (default 3, clamped ≥ 1). The L4 table carries RAW counts, so re-tuning
  is an env flip + container recycle — never a re-export. Deliberately NOT
  a request param and NOT an SPA advanced-panel option (user 2026-06-11);
  empirically, threshold 3 admits ~1,244 structured English bare words
  (~52% of which have ≥80% single-position skew), threshold 5 ~724.

### Plumbing (where each piece lives)

* Build: `Name.get_bare_word_positions` → `proportions_from`'s
  `bare_word_positions` key (`{position: {usage: count}}`, raw).
* L4: `proportions_bare_word_position (culture, position, usage_key,
  weight)` — point-lookup table, no cumulative column (the vector path
  samples in Python). **No schema_version bump**: the adapter reads it
  defensively (missing table → `{}`), following the
  `proportions_attested_language` precedent, so deploy ordering between
  runtime code and re-exported DBs doesn't matter.
* Load: `MeaningGenerator.load_bare_word_positions` registers per-word-
  position buckets keyed `('bare', *flags, 'single', 'wp-<position>')` —
  the same `load_parts` machinery, so name/saint bare bucket families get
  the dimension uniformly.
* Vector path: `_flatten_struct_slots` derives each bare slot's word
  position from its word index in the struct (the struct already encodes
  word order — no template change, the label is derived, never stored) and
  extends the slot's bucket key; `_resolve_slot_usage_frequency` falls back
  to the un-extended key when the wp bucket is absent (legacy bundle, or a
  bucket family with no positional stats) — bit-stable pre-D43 behavior.
  When the wp bucket EXISTS it is authoritative: a structured word
  unattested at P is correctly absent from P's pool.

### Scope + bit-stability

Bare words only (the spec's scope; general word-sequence position for
compound words is future work). Once a re-exported L4 ships, seeded output
changes for any structure containing bare slots — accepted deliberately
(pre-launch posture; same contract stance as D41): the placement fix IS the
product improvement, and legacy DBs keep byte-identical output via the
fallback.

## D44. Era selects the REFLEX, never the MORPHEME (wyrd-c6o1.3 follow-up, 2026-06-12).

> **Refined same-day by D46 (wyrd-6ah2).** The "never the morpheme" absolute
> is narrowed: a bounded historical era DOES gate by RECORD ENTRY ("in the
> record by then" — no Silicon on a 1086 map), and the seed-level
> same-skeleton-at-every-era invariant narrows to the ungated pair
> (era="" ↔ present-day) plus the town-level statement (a generated town's
> structure persists across renderings). Everything else here stands:
> nothing ever expires, render is era's primary effect, and the diversify
> split below remains load-bearing. (Later refinement, wyrd-3tvd: a bounded
> HISTORICAL era now ALSO leans the priors-baseline toward its midpoint — a
> mild fashion re-weight — while present-day/open-ended stays neutral; see
> D46 "What this changes vs D44".)

**The decision: the morpheme inventory is time-invariant. A request's era
changes which SURFACE each morpheme renders as (its era reflex, via the D33
machinery) — it never changes which morphemes can be drawn, and it never
re-weights the draw. The guaranteed invariant: the same ``(culture, params,
seed)`` produces the same name SKELETON (the same picked morphemes, hence the
same ``modern_name()``) at every era; only the rendering tracks the requested
period.**

The product model behind it: town names change over time, but their morphemes
don't. ``Tūn`` → ``-ton`` is one morpheme whose surface evolved; a town
"founded" by the generator carries its morpheme stack through history and we
render that stack at whatever period the user asks for. This is what makes
``kenning-rewind`` / the era-map coherent with plain generation — they were
already "generate once, render at stops"; D44 makes the main generator agree.

**Recorded assumption (deliberate simplification): a name's STRUCTURE is
constant over time.** Real toponyms occasionally restructured (folk
re-etymologization, partial translation, affix accretion like the Norman
manorial layer); we accept the simplification that the morpheme stack and
word structure persist, and only surfaces evolve. If a future feature wants
historically-mutating structure, it composes ON TOP of this invariant (a
transform), rather than weakening it.

### What was removed

* The D5-2/D5-3 era ELIGIBILITY gate, end to end: ``EligibilityGate.era_min``
  / ``era_max`` (and the inverted-range validation), ``passes_era_gate``
  (eligibility.py), ``_matches_era`` (vector_name_select.py), and
  ``Meaning.attested_in_era_range``. The attested-years DATA stays in the
  bundle (display / explainer / future analytics); only the gate predicate is
  gone.
* The request-era → ``era_midpoint`` coupling into the empirical-priors
  baseline axis (``era_midpoint_from_range``). Era-weighting the draw would
  break the same-skeleton-at-every-era invariant just as surely as gating.
  The scoring layer's ``era_midpoint`` parameter survives at its no-era
  default (0 → the priors tables' wildcard-cell convention); the priors DATA
  keeps its era cells (D36.7) for pack-template lookups and future
  non-request-driven uses.

### What era still does

1. **Render language** (wyrd-6c8x / D41): ``era`` resolves to the stage's
   canonical language; each morpheme renders its reflex at that stage
   (``era="present-day"`` → force-modern; ``era=""`` → native per-morpheme
   mix; historical stages → period forms).
2. **Request validation**: ``_resolve_era_param`` still parses + validates
   the era input (year / cell label / family-label, strict bare-label
   resolution) so a typo'd era is a clean 4xx — its resolved range is
   otherwise unused.

### Repeat-diversification made era-stable (the subtle half)

The skeleton invariant exposed a hidden era-coupling: ``_diversify_repeats``
(wyrd-vd6y) detected repeats on the RENDERED surface, so a native render could
trigger a skeleton mutation (synonym override / re-pick) that the force-modern
render didn't ('Biscop'×2 collides natively; 'bishop'/'bishops' don't collide
modern). Split into two passes:

* **Pass 1 — identity repeats (era-invariant)**: detection folds on the
  modern usage key. Skeleton mutations (cross-language synonym, re-pick)
  happen identically at every era.
* **Pass 2 — render collisions (era-dependent, render-only)**: distinct
  identities whose surfaces collide only at the requested era's render fall
  one slot back to its modern surface (``_break_native_duplicate``). Never
  mutates the picked morphemes.

### Bit-stability

``era=None`` requests are unchanged (the gate was already a no-op and the
midpoint already 0). Era-set requests change output where the old gate or
midpoint used to bite — accepted deliberately (pre-launch posture, same
stance as D41/D43): the invariant IS the product improvement. (Originally
pinned by ``tests/test_kenning_era_renders_not_gates.py``; superseded by
``tests/test_kenning_era_accretion.py`` when D46 narrowed the invariant.)

### Supersessions

Supersedes D5-3's inventory-filter half (annotated there), including the
same-day wyrd-c6o1.3 open-ended-window refinement — D44 is that refinement
generalized to every window: not just "the present contains all strata" but
"the inventory is era-independent, period". The wyrd-c6o1.3 homograph fix
(surface-keyed pfoo narrowing) is unaffected and still load-bearing.
## D45. Dashes are NEVER morpheme identity — position is display-time decoration (2026-06-12).

**The hard rule, verbatim from the product owner (recorded because it has now
been re-explained 5+ times — D39, D40, wyrd-eyjk, wyrd-zewx, wyrd-c6o1.3):
we do NOT put `-` in the morpheme name, EVER, and we NEVER key off of it.
`pre-` / `-inner-` / `-post` is ONLY a function of position, and position is
ONLY an output of breaking down toponyms (or of using pre-broken-down ones).
The dash is purely something we decorate the morpheme with AT DISPLAY TIME.**

A morpheme's identity is its bare surface (D40: `ton`, `giles`, `stoke`).
Position (`bare` / `pre` / `inner` / `post`) is:

1. **derived** at decomposition time from where the morpheme landed in the
   split (`Word.get_structure`, by index) — an OUTPUT, never a match input;
2. **carried** as an explicit position FIELD wherever a table genuinely
   needs the axis — a `(surface, position)` composite key, never a position
   encoded INTO the surface string;
3. **applied** as dash decoration (plus positional case) only by the render
   layer, which is the decoration's single owner (D39).

### Why this keeps coming back (the root cause)

D40 enforced the rule at the MATCHING layer (string-first matching, position
gates removed) — but the STORAGE layers still carried dash-marked identities:
L3 `reflex.surface_form`, L4 `meaning.usage_key` / `modern_usage` /
`proportions_*` keys, and the runtime `meaning_db` keys, with
`Meaning._set_location` DECODING dashes back out of the stored string. So one
surface existed as up to four separate stored identities (`ton` / `Ton-` /
`-ton` / `-ton-`), every consumer needed fold-the-dash compat code (~80
production sites across the runtime/bundle/lexicon-export layers as of the
2026-06-12 audit — inventory in epic wyrd-aicu; the raw repo-wide grep is
~100, the remainder being exempt raw-source handling in parsers/extractors),
and each new consumer that forgot to fold re-introduced the bug class.

**L4 landed (wyrd-aicu.1, 2026-06-13, schema v3).** The L4 runtime DB is now
fully de-dashed: `meaning.usage_key` is the bare surface (one row per surface,
the up-to-4 dash-variants merged with entries unioned + primary_language /
stratum re-picked over the union); the `proportions_usage` /
`proportions_single_usage` tables carry an explicit `position` column with
bare-surface keys; `proportions_attested_language` is bare-surface-keyed — fixing the
wyrd-c6o1.3 homograph leak at the source (the c6o1.3 load-time fold-union
becomes redundant on v3 bundles but stays as the legacy-v2 compat path until
the wyrd-aicu.4 fold-site sweep removes it). The build tally
(`Word.get_samples` → `(surface, position)`) and the
proportions transport (`{surface: {position: weight}}`) carry position as an
explicit axis end to end; the render (D39) still owns dash decoration + case.
Deliberate output drift (the weight-merge shifts the draw — parity regen,
pre-launch posture). Remaining: L3/L2 (wyrd-aicu.3), the runtime
`Meaning.location` retirement + fold-site sweep (wyrd-aicu.2/.4), and the CI
data-gate (wyrd-aicu.5). The
latest instance was wyrd-c6o1.3: the per-Meaning attested-language narrowing
keyed by exact stored usage_key, so the welsh `ton` homograph walked into
english generation through the un-narrowed dash-variant keys.

### Enforcement

* **Reviewer**: `morpheme-surface-identity-reviewer` (kenning
  `AGENT-REVIEWERS.md`, spec under `.reviewers/`) runs on every kenning PR
  and flags new code that stores a dash-marked surface as identity, keys a
  lookup by dash-shape, or branches on dash markers outside the render
  layer / documented legacy fold sites.
* **Refactor epic**: wyrd-aicu de-dashes the storage layers end-to-end
  (L4 bare-surface keys + explicit position columns → runtime → L3/L2 →
  delete the ~80 fold sites), finishing with a CI data-gate asserting no
  stored identity contains decoration. Until it lands, fold-at-the-boundary
  (`surface.replace("-", "").lower()`) is the REQUIRED compat idiom for any
  code that must read the legacy dash-marked storage — and is itself
  deleted by the epic's final sweep.

### Exemptions (the only legitimate dash-handling)

* The RENDER layer applying positional decoration + case (D39's single
  owner: `Word` / `NewName` / the explainer surface builders).
* Parsers / extractors / miners reading dashes in RAW SOURCE TEXT
  (scholarly books hyphenate; that's document syntax, not identity).
* Genuinely hyphenated lexical forms inside language data (e.g. the OE
  compound headword `lēac-tūn` as a FORM, or a word that genuinely contains a
  hyphen — `al-Quadim`, `al-Adha`, `Bēl-šarra-uṣur`) — a form's spelling may
  contain a hyphen; the STORED IDENTITY of a morpheme may not carry positional
  dash markers.

### The morpheme STORE is in scope too — no "content key" exception (wyrd-aicu.8, 2026-06-21)

D45 governs **every** place a morpheme's identity is stored or keyed — including
`etymon.canonical_form` and the derived `morpheme_id` (`language:canonical_form`,
the L4 `morpheme` table's content key). The tempting rationalization —
"`morpheme_id` is a scholarly CONTENT key, a different scheme, so D45 doesn't
apply, close as not-applicable" — is **REJECTED**. That is precisely the
half-dashed-corpus state the rule exists to prevent: when *some* morphemes are
stored `-ach` and others `ach`, one morpheme forks into phantom duplicates, edges
multiply (an `-ach`→x edge AND an `ach`→x edge), and you are forced to mint a
separate `-ton-` the moment a name happens to carry `ton` medially. It is noise in
the corpus. **Dashes go; the morpheme is its bare surface; position is the separate
pre/inner/post axis** — uniformly, with no "this layer is special" carve-outs.

So the affix-POSITION dashes in the store are stripped too (`-ach`→`ach`, the
boundary recorded as the position axis), with content-key **collision merges**
(`celtic:-ach` + `celtic:ach` → one `celtic:ach`) propagated across every L2 ref
that uses the `language:canonical_form` natural key. The ONLY dashes that survive
are the genuine in-WORD hyphens of the Exemptions above (the *word itself* contains
the hyphen — `al-Quadim` — which is form spelling, not position decoration). The
classifier is structural: a leading/trailing boundary dash on a bound morpheme is
position decoration → **strip**; a hyphen between real word-characters is part of
the word → **keep**.

Composes with D39 (render owns decoration), D40 (position derived, never a
match gate), D44 (identity is also era-invariant). Together: a morpheme's
stored identity is a bare surface, timeless and position-free; position and
era are lenses applied on the way out.

## D46. Era pools ACCRETE — "in the record by then" (wyrd-6ah2, 2026-06-12, refines D44).

**The decision, in the product owner's framing: a historical era may gate a
morpheme out if the morpheme has no evidence as old as the period — but never
if it is as old or older. It has to have appeared in the record by then.**
The motivating example: *Silicon* (a modern coinage; the element is 1817, the
toponym pattern 1971+) must not appear on a 1086 map — while ``tūn`` appears
on every map from OE onward, because old things persist (D44's accretion
stands; nothing ever expires, so the wyrd-c6o1.3 ``-ton`` starvation remains
impossible by construction).

### The eligibility rule (most-generous-evidence-of-age)

``Meaning.record_start()`` = the minimum over two signals, either of which may
vouch the morpheme older:

1. **Source-language record entry** (``_LANG_RECORD_ENTRY``,
   runtime/meaning.py): the year each language LAYER enters the *British
   place-name record* — deliberately not the language's own birth (Old Norse
   and Norman French existed earlier elsewhere; what's gated is when they
   could plausibly appear in a name on this island). Founding strata
   (old_english, celtic_mix, latin, germanic, the ancient pack languages) are
   ``None`` — in the record before every era we model. Contact layers:
   old_scandinavian → 800 (Danelaw), old_french / norman_french → 1066
   (Conquest), middle_english → 1100, modern_english → 1500. Unmapped
   languages default to ``None``.
2. **Earliest attested year** across the morpheme's ``attested_years`` — a
   date can vouch a morpheme OLDER than its stage suggests (a
   ``modern_english``-bucket morpheme with a 1086 Domesday date is in the
   record by oe-late).

Eligible at era E iff ``record_start`` is ``None``, OR E is open-ended
(``end is None`` — the present-day stage, the deployed default, and the
Mixed-Era empty value), OR ``record_start < E.end``. No evidence on either
axis → pass (the same coverage posture as stratum and the legacy era rule).

**The asymmetric failure mode is the point**: thin data errs toward
INCLUSION (harmless — an old-looking thing on a period map); the only hard
exclusions are positively-evidenced late arrivals, the confident case.
Contrast with the retired D5-2/D5-3 attested-INSIDE-window rule, whose
failure mode punished exactly the best-documented morphemes (wyrd-c6o1.3).

### What this changes vs D44

* ``EligibilityGate`` regains one era field — ``era_record_cutoff`` (the
  resolved era range's END) — consumed by ``passes_record_gate`` /
  ``_passes_base_gates``. As of D44 the request's era did not re-weight the
  draw (the priors ``era_midpoint`` stayed at the wildcard convention). The
  optional fashion-weighting follow-up **wyrd-3tvd** has since landed: a
  bounded HISTORICAL era now also leans the priors-baseline toward its
  midpoint (kenning ``_request_era_midpoint`` → ``select_via_vector``), while
  an era with no upper bound (present-day, the deployed default) stays at the
  wildcard 0 — so default generation is unchanged.
* Pools are MONOTONE in era: pool(oe-early) ⊆ pool(me) ⊆ pool(present).
* The D44 seed-level invariant narrows: per-request determinism is untouched
  and the UNGATED pair (era="" ↔ present-day) stays skeleton-identical per
  seed — but the same seed at a bounded era may differ from present-day (a
  smaller pool shifts the weighted draw). The durable, product-level
  invariant restates at TOWN level: a generated town's structure persists
  across renderings — rewind / era-map / regenerate carry the morpheme stack
  and re-render it; they never re-draw from the seed.
* Render is unchanged (D44's reflex lens).

### Known corner (documented, not solved)

Stage-level vouching is coarse: an 1817 coinage in the ``modern_english``
stage passes the early-modern (1500–1700) window via the stage's 1500 start.
Every medieval era excludes it correctly — the corner only opens at the
youngest bounded window. Per-coinage dating (or finer stage cells) would
tighten this later; not worth the data work now.

Pinned by ``tests/test_kenning_era_accretion.py`` (evidence rule, gate
predicate, live-bundle monotonicity, the end-to-end Silicon-rule property on
era=me picks, the ungated-pair skeleton invariance, and the render check).
Composes with D40/D45 (the gate keys on Meaning identity and attestation
data, never on dash-shapes) and D33/D41 (render machinery untouched).

## D47. `--tag` reserves ONE slot, it does NOT gate every slot (wyrd-c6o1.4 / wyrd-ah53, 2026-06-13).

**Supersedes the "hard gate excludes before scoring / only names with the tag"
half of D36.6.** (D36.6's culture hard-gate + exclude-tags stand; only the
positive `--tag` semantics change.)

### The model (the tagged-slot rule)

When one or more `--tag`s are requested, generation:

1. builds each structural slot's candidate pool (the normal per-slot membership);
2. finds which slots have ≥1 candidate carrying a requested tag (OR across
   multiple `--tag`s);
3. randomly **reserves ONE** such slot and HARD-restricts it to tagged candidates;
4. generates **every other slot normally** — the pool stays tag-agnostic, so the
   rest of the name samples freely (the soft semantic-axis weight, D36.6's "lean
   toward death-themed names" knob, still nudges the free slots).

Net effect: a name leans toward *containing* a tagged morpheme, the rest is free —
the historical proportions-path intent. This is the SAME single-reserved-slot
overlay a thematic mood uses (wyrd-4rp8), but **hard** (a mood is soft: no capable
slot → un-themed, no failure).

### Why (the bug it fixes)

wyrd-wv85 implemented D36.6 literally — `--tag` as a POOL-WIDE gate restricting
EVERY slot to tagged lemmas. That collapsed variety into thematically-saturated
names (operator-reported: `--tag animal` → every morpheme animal, ~40 distinct
morphemes vs ~70 free). The one-slot model restores variety while still
guaranteeing the theme is present.

### Hard guarantee, cohesion-safe

The reserved slot's restriction is applied at the POOL MEMBERSHIP level (filter
the slot's candidates to the tagged subset BEFORE cohesion re-weighting), NOT
post-hoc on the final weighted list — because cohesion can drive a tagged
candidate's score to 0 and drop it, which a post-hoc filter would silently fall
back around (un-tagged name). Restricting first means cohesion re-weights *within*
the tagged subset, which stays non-empty. So every name is guaranteed ≥1 tagged
morpheme even at cohesion=1.

- If no slot in the chosen struct can carry the tag → struct retry (return `[]`).
- If NO morpheme anywhere carries the tag (a typo, or a tag absent from the
  culture) → retries exhaust → the caller's "no eligible name" (the typo signal).

Implementation: `select_via_vector_scoring` + `_choose_tagged_slot` +
`_slot_weighted_pool(require_tags=…)` in `runtime/vector_name_select.py`. Re-rolls
mirror it: `kenning_regenerate` restricts the re-rolled slot to tagged only when
it is the name's SOLE tag-carrier (else free).

### Satisfiability is per-culture (the wyrd-ah53 corollary)

A tag is offered for a culture **iff ≥1 morpheme in that culture's pool
carries it** — a conservative SUPERSET of the strictly-placeable set (it ignores
per-position frequency, so it never omits a usable tag but may over-offer one the
runtime then degrades; see `_tags_options_by_culture`). The tag vocabulary is
culture-agnostic (`available_tags` is the union over all cultures), so tags exist
that no morpheme in a given culture carries — e.g. `monster` (a fantasy/pfsrd2 tag)
has zero English morphemes.
Offering such a tag for that culture is a UX bug: bare it raises "no eligible
name", combined it silently no-ops. So the SPA's tag composer is **culture-scoped**
via `x-options-by-culture` (`_tags_options_by_culture`, the same dependent-select
mechanism as era / stratum); the `items.enum` stays the union for CLI/API
validation. Pinned by `tests/test_kenning.py` (per-culture options) +
`tests/test_kenning_vector_name_select.py` (the slot mechanics + cohesion-safety).

## D48. Structure allowlist is the single operator filter for which structures generate (wyrd-c6o1.5, 2026-06-15; per-language + refresh-merge wyrd-hzqs, 2026-06-21).

Which name STRUCTURES the generator may use is an operator choice, expressed in a
bundled YAML (`data/structures.yaml`). The file is **per-language** (top-level
culture sections — see the wyrd-hzqs subsection below); within a section each
structure is keyed by a canonical label (`struct_key_to_label`: `(bare)`,
`(pre+post)`, `(bare) (bare)`, `(bare[name])`, …). Each defaults to
`enabled: true`; the operator opts a structure OUT for that culture with
`enabled: false`. A structure ABSENT from its section — or an absent section — is
enabled, so a bundle rebuild that surfaces a brand-new structure generates by
default (operator opts out, never in). `wyrd kenning dump-structures` emits the
per-language inventory (with frequencies) to curate from.

### One filtering path

This is the ONLY operator-facing structure filter, applied once at
`NameGenerator.__init__` (`runtime.structure_allowlist.is_structure_enabled`,
AND-ed with the grammaticality gate). The old wyrd-g1hj "`<Bare>`" special-case
(`_is_single_morpheme_structure`, a hard-coded lone-dictionary-word exclusion in
`load_proportions`) is DELETED and migrated here: the lone-word structures
(`(bare)`, `(bare[name])`) ship `enabled: false`. wyrd-zzli
(`is_structurally_grammatical`) stays a SEPARATE hard grammaticality gate — an
operator can't `enabled: true` a structure into ungrammaticality.

### Why a sibling YAML (not a runtime-DB table)

Mirrors `register_effects.yaml`: a package-data file shipped via the `data/*.yaml`
glob, loaded lazily through `importlib.resources`. The structures themselves stay
in the runtime DB; only the enable/disable overlay is YAML, so curating the
allowlist needs no DB migration or re-emit.

### Per-language sections + refresh-merge dump (wyrd-hzqs, 2026-06-21)

The file is **per-language**, not global: a top-level mapping of CULTURE → section,
each section the `{label: {enabled}}` map for that culture. The same structure has
different frequencies (and different "is this trash") per language — English run-on
compounds aren't Irish patterns — so `is_structure_enabled(struct_key, culture)`
keys on the culture. The culture NAME is threaded `_load_culture → load_proportions
→ NameGenerator(culture=…)`; a `NameGenerator` built without a culture (legacy/test)
fails open (no section to key on → unfiltered). Absent label → enabled; absent
culture section → enabled. This SUPERSEDES the original "global, not per-culture"
choice (the label alone couldn't express that a shape is fine in one language and
junk in another).

Each dumped row carries an **advisory** `# <weight> (<pct>%)` frequency comment (a
dump-time snapshot; the loader ignores it) so the operator can curate by commonness
— most "trash" is rare-tail. Rows are **alphabetically** sorted (stable) so a
rebuild that only nudges frequencies doesn't reorder the file and churn the git
diff; new structures land as clean added lines.

`dump-structures` is a **refresh-merge**, not a regen: it reads the existing
committed file, the operator's `enabled` value wins for any label already present
(every `false` is sticky), a NEW label gets the default (`true`, except
single-morpheme `<Bare>` → `false`), and a structure no longer in the corpus drops
out. It prints a per-culture `+new / -dropped` summary (with the new labels' freq)
to **stderr** — an explicit "go filter these" signal on top of the git diff. The
flat (pre-hzqs) format is hard-cut: an old flat file fails the loader loudly (its
`{enabled: bool}` values aren't section mappings) — re-dump to migrate. Auto-regen
on rebuild + a CI drift-guard is the follow-up (wyrd-x7w4).

### Posture

Disabling a structure changes seeded output across CLI / Lambda / SPA (they all
generate via NameGenerator) — acceptable pre-launch (bit-stability subordinate to
product, D41/D43). Empty-after-filter raises an operator-attributable error.
Pinned by `tests/test_kenning_structure_allowlist.py` + the migrated
`tests/test_kenning_position_forms.py`.

## D49. Corpus uncleanness splits into two classes; each gets its own mechanism (2026-06-16).

**The decision: stop treating "the corpus is a mess" as one problem. It is two
problems with two different shapes, two different correct mechanisms, and two
different failure modes. Conflating them is why cleanup has felt intractable.**

| | **Class A — controlled vocabulary** | **Class B — scholarly identity** |
|---|---|---|
| what | a dimension drawn from a *closed external authority* | "are these two things the same thing?" |
| examples | `region`, `country`, language code, `source_id`, era cell | morpheme≈morpheme, place≈place, gloss≈gloss, "is this breakdown spurious?" |
| nature | a **coding** problem — same value, many spellings | an **entity-resolution** problem — real evidence, real disagreement |
| evidence to preserve? | none | yes — every claim has witnesses, sources, a rationale |
| mechanism | canonical **enum + alias map**, normalized + validated **fail-closed at the ingest / JSONL boundary** | append-only **assertion log** (L2) → projected **collapse graph** (L3) |
| where the truth lives | the JSONL (L2, git-tracked → the diff IS the provenance) | the assertion stream (L2) + raw observations (L2) |
| failure mode | unknown value → **hard error** (can't silently accumulate) | low-confidence merge → **leave separate** (asymmetric, per D46) |
| rationale on each fix? | no — the alias map is the whole audit | yes — confidence + method + source + reason, per edge |

Class A makes Class B *tractable* (clean blocking keys), but never substitutes
for it. They are complementary, not alternatives.

### The partition is over OPERATIONS, not fields — one dimension is usually both

A field like `region` is not wholly Class A or wholly B. Decompose it into
operations:

- **Label-at-a-stratum (Class A):** for a given (place, period), what is the
  canonical token for it / its containing region? `SUR` → `Surrey`,
  `County Durham` → `Durham`. Same value, different coding, picked from a closed
  vocabulary. Mechanical, no evidence.
- **Identity-across-change (Class B):** the *same place* persists while its name
  and jurisdiction change over time. `Cumberland` (historic) and `Cumbria`
  (modern) are NOT alias variants to fold — they are different strata of a
  containing-jurisdiction succession, and tying a place's strata together is an
  evidence-bearing identity assertion.

This is the SAME split the morpheme axis already runs (D44/D46): a morpheme is
one time-invariant *identity* (`tūn` ≡ `-ton`, Class B) whose canonical *surface*
is selected per era (the reflex, Class A). D49 generalizes that
reflex-vs-identity discipline from morphemes to **places and jurisdictions** —
and, by the same logic, to a toponym's own name-path (`Neutune` → `Neuton` →
`Newton`): the place is one Class-B identity; the canonical label per stratum is
Class A. (`modern_name` is therefore just the present-day-stratum label, and the
historical attested forms are other-stratum labels of the same identity — today
the schema privileges `modern_name` and strands the rest in `notes`/attestation;
the principled fix is this same place-identity + per-stratum-label model,
mirroring D41's native-vs-modern split on the generation side.)

### Region is a stratified, multi-axis containment HIERARCHY, not a flat enum

Two structures hide inside the one `region` string:

1. **Containment depth (one axis, many levels):** `England → Yorkshire →
   North Riding`. The flat field crams different *levels* of one tree into one
   slot — which is why `Yorkshire` and `West Riding of Yorkshire` sit as if they
   were peers when they are parent and child.
2. **Overlapping axes:** `England (Danelaw)` is NOT a node in the administrative
   tree — the Danelaw is a historical-linguistic / legal *zone* that cuts across
   many shires. A place is at once "England → Yorkshire → North Riding" (admin
   axis) *and* "in the Danelaw" (zone axis). Cramming both into one field
   conflates two classification systems.

So the model is a **forest of stratified containment trees plus cross-cutting
zone axes**, with places linked to nodes by membership edges. The work splits:

| thing | class |
|---|---|
| canonical **label** of each node per stratum (`SUR`→`Surrey`) | A |
| the **containment skeleton** within a stratum (North Riding ⊂ Yorkshire ⊂ England) | A |
| **which axis** a value belongs to (Danelaw = zone, not county) | A |
| **assigning an ambiguous place to a node** (bare "Yorkshire" → which Riding?) | B |
| **cross-axis / zone membership** (does this vill fall in the Danelaw?) | deferred — cultural surface, not dedup (see next section) |
| **succession across strata** (Cumberland + Westmorland → Cumbria; hundreds reorganized) | B |

The Class-A alias map therefore folds **within-stratum, same-axis, same-level
coding variants only**. Cross-level (`Yorkshire`↔`North Riding`), cross-axis
(`Danelaw`), cross-stratum (`Cumbria`↔`Cumberland`) are NOT alias entries — they
are containment/membership/succession edges on the Class-B side.

### Region serves two purposes — DEDUP (now) and CULTURAL COLOR (deferred)

Region's *live* purpose is **deduplication**: deciding whether `Newton` and
`Niweton` in the same area are the same place, so they collapse to one record.
For that, region must be **tight and canonical** — precise enough to
disambiguate places. It is the blocking key for the Class-B place-identity pass,
and its granularity is driven by *what separates places*, not by what is
culturally meaningful.

A *separate, deferred* purpose is **cultural color on generated names**:
linguistic / settlement zones carry different morpheme mixes (the Danelaw is
ON-heavy; Wales Brittonic; Roman England Latin-flavored), so a future generator
surface could offer "Danelaw town names" the way `--culture` / `--stratum` (D32)
/ `--era` (D44) compose today. This axis is **coarse and
linguistically-defined**, cuts across administrative boundaries, and is a
generator surface we are not building yet.

The two do NOT share a granularity or an axis:

- Administrative subdivisions (Ridings, hundreds) serve **dedup precision** — no
  one wants "North Riding of Yorkshire" as a name flavor.
- Linguistic zones (Danelaw / Celtic / Roman) serve **cultural color** — and are
  useless as a dedup key (too coarse to tell two Newtons apart).

Consequence: `England (Danelaw)` in the `region` field today is a category error
on both counts — it can't dedup (the place has a real shire) and it isn't wired
as cultural color. Dedup wants the actual shire; the Danelaw membership, if
retained, is a tag on a *future* cultural-zone axis (an enrichment
classification like D32 stratum), not part of the canonical dedup region. The
Class-A region work below is scoped to the **dedup hierarchy**; the cultural-zone
axis is explicitly deferred.

### Class A: normalize controlled vocabulary at the source

Region is the flagship offender. The live corpus carries `SUF`/`Suffolk`,
`LEC`/`Leicestershire`, `BUK`/`Buckinghamshire`, `SUR`/`Surrey`, `SUS`/`Sussex`,
`County Durham`/`Durham` all coexisting (~2,700 England toponyms hidden under
code-form duplicates), plus country-level `Ireland`/`Republic of Ireland`,
`Isle of Man`/`The Isle of Man`, `Brittany`-as-country, and 1,078 null
countries. These are the **same value differently coded** — there is nothing to
resolve, only to normalize.

The fix:
1. A canonical **enum** per dimension (`data/regions_england.yaml` first — the
   historic-county list the sources actually use).
2. An **alias map** (code → canonical); the map is the audit trail.
3. Normalize at the **parser / ingest boundary** going forward, and a one-time
   pass over the committed L2 JSONL. This is safe to do *in place* **because L2
   is git-tracked** — the original coding is recoverable from history, so
   normalization is a reviewable commit, not data loss. (Region is provenance
   metadata, not a D21 etymological *claim*, so D21's "evidence sacred" does not
   sanctify it. Contrast: the dated historical forms in `toponym_etymology.notes`
   ARE evidence — you extract them additively, never normalize them away.)
4. **Validate fail-closed at ingest**: a value neither in the enum nor the alias
   map is a hard error / quarantine. This is the load-bearing guard — without it
   the mess re-accumulates on the next source.

**Not everything that looks like a code is Class A.** The enum carries a real
scholarly choice (historic counties; keep Yorkshire at Riding level since EPNS
does — bare `Yorkshire` collapses the W/S-W Ridings that exist as separate
values), and the *lossy* cases are genuinely Class B: `Cumbria` = historic
`Cumberland` + `Westmorland` cannot be aliased to one value without per-toponym
evidence. The straddle set (Ridings, Sussex E/W, `Northumberland and Durham`
merged-volume strings, modern metropolitan non-counties, `England (Danelaw)`)
falls to the Class-B toponym-ER pass.

### Class B: the universal canonicalization assertion layer

The collapse-with-provenance machinery **already exists and already works** —
`merged_into_id` (D22) and `cognate_id` (D27/D28) correctly unify `tun`→`tūn`
and `nīwe`→`niwe`. The problem is it is **partial, inconsistent, and the *why*
is not on the edge**. From the live Newton trace (2026-06-16):

- OE "new" exists as **two canonical nodes both glossed "new"** — `niwe`
  (671826, cognate cluster 330032) and `ne` (369498, cluster 369500, an
  OCR-truncated "new" spelling cluster) — never merged into each other, so
  "Newton" decomposes differently depending on which dictionary the row came
  from.
- Ekwall's `new` (4631595) was merged into `ne` with **2 witnesses and no
  recorded rationale** — a distinction-collapsing automated merge that nobody
  can audit because the edge carries no *why*. Had `ne` been the negative
  particle (its usual sense), this would have silently fused "new" with "not".
- A high-confidence Ekwall row (toponym 1725, row 780) carries a **hallucinated
  `ford+botl` breakdown** on a "new tun" entry; it passed form-in-body
  validation (D3) because those strings appear *elsewhere* in the paragraph
  (D3's neighbor-contamination blind spot).
- Toponym 1725 also collapses **three distinct Lancashire Newtons** (a village,
  an old manor, one S. of Dalton) into one row, and the dated name-path
  (Neutune → Neuton 1242 …) is stranded in `notes` prose rather than structured.

The fix generalizes `merged_into_id`/`cognate_id` from per-table redirect
columns into one uniform model: **reified assertion edges** — `(subject,
predicate, object, confidence, method, source, actor, rationale, timestamp)` —
living **append-only in L2 JSONL**, projected into the L3 collapse graph. The
cleaned graph is a **rebuildable L3 projection** of (raw observations L2) +
(assertions L2); this preserves D21 (evidence additive, never destroyed), D24
(observable), the four-layer model, and the mining-expensive / enrichment-cheap
asymmetry. The *assertions* are the treasured new artifact; the graph is their
queryable shadow.

Three load-bearing constraints:

1. **The edge-type taxonomy is the first and most dangerous design step.**
   Identity-bearing / collapsible predicates (`same-morpheme-as`,
   `same-place-as`, `variant-of`, `inflection-of`) are a *different class* from
   relational / never-collapsible ones (`descends-from`, `glosses-as`,
   `co-occurs-with`). Conflating axes recreates the D28 `synset_id` collision
   and the D42 PIE mega-component. `tun ≈ tūn` is an identity collapse;
   `newton descends-from tun` is emphatically not.
2. **Merge asymmetry (per D46).** A missed merge is harmless duplication; a wrong
   identity collapse is corruption. Identity-collapse assertions need high
   confidence + rationale; the **default failure mode is leave-separate**.
   Agent-proposed merges face an adversarial skeptic before promotion.
3. **No graph database engine — D42 stands.** The graph is an L3 SQLite
   projection. D42 already proved generic graph primitives erase the domain
   rules (PIE non-bridging, canonical resolution, root-anchoring); a property
   graph encoded in SQLite keeps those rules explicit. This decision is about a
   **data model + curation workflow**, not a storage engine.

The agentic cleanup loop the product owner wants — processes that research
candidate merges/corrections on the internet and emit *proposed* assertions that
a gate or human promotes — is exactly what this model enables: agent assertions
are a quarantined tier (cf. D2/D19), never auto-trusted for identity collapse,
that grind the corpus cleaner over time.

### Relationship to prior decisions

Generalizes D22 (`merged_into_id` is the proto-pattern) and D27/D28 (the cognate
vs meaning_synset axis split is the taxonomy fork, writ large). Bounded by D21
(evidence sacred — assertions are additive; Class-A region is not evidence),
D24 (observability), D38 (runtime / L4 untouched — this is authoring-layer
work), D42 (no graph engine), D46 (asymmetric failure). Surfaces a D3 gap
(form-in-body passes neighbor-contaminated breakdowns) that a Class-B cleanup
pass addresses without weakening D3.

### Deliberately deferred to the epics (not decided here)

The edge-type taxonomy itself; synthetic canonical nodes vs winner-redirect
(the D31 UNIQUE-key lesson argues for synthetic, at an indirection cost);
one assertion stream vs per-predicate JSONL; the exact promotion gate. These are
the opening tickets of the two epics, not settled by this entry. **The Class-B
half of these — the edge taxonomy, the synthetic-node decision, the assertion
record schema, and the stream layout — is now settled by D50.**

Epics: **wyrd-3q6m** (Class A — controlled-vocabulary normalization at source)
and **wyrd-u6fn** (Class B — universal canonicalization assertion layer). The
worked examples above double as regression checks: Class-B cleanup must catch
the `ne`≈`niwe` duplicate and the `ford+botl` hallucination; Class-A must fold
the region codes and produce the blocking keys the toponym-ER pass needs.

## D50. Canonicalization assertion layer: synthetic identity nodes + a typed edge taxonomy (wyrd-u6fn, 2026-06-16).

D49 split corpus uncleanness into Class A (controlled-vocabulary coding →
enum + alias map at the ingest boundary) and Class B (scholarly entity
resolution → an append-only assertion log projected to a collapse graph). D50
specifies the **Class-B mechanism** in detail — the node model, the edge-type
taxonomy, and the assertion record schema. (D49 is the umbrella; D50 is to D49's
Class-B half what D37 is to D36.)

### D50.1. Synthetic canonical nodes, not winner-redirect.

The existing collapse mechanism (`merged_into_id`, D22) is winner-redirect: a
loser observation points at a *winner observation*. D50 replaces this with
**synthetic canonical nodes**: per identity, mint an abstract node
(`canonical_morpheme` / `canonical_place` / `canonical_sense`) with a **stable id
declared in L2**, and **bind observations to it**. The canonical node carries no
source of its own — it is a derived hub.

Why synthetic wins (winner-redirect's three faults, all visible in the live
Newton data):

1. **It conflates identity with evidence.** The winner row (`niwe` etymon
   671826, `tūn` 377353) is itself a mined observation with its own source +
   citations. Making an observation *be* the identity privileges it — exactly
   what D49 separates.
2. **The canonical label drifts by stratum, so "the winner" is ill-defined.** A
   place's canonical surface changes by era (`Neutune`@1086 → `Newton`@modern);
   no observation is "the place." Winner-redirect would re-point forever (the
   D31 UNIQUE-key trap).
3. **Per-stratum labels have no home** on any single observation.

Synthetic nodes fix all three: identity is separate from evidence (observations
stay as mined, D21); per-stratum labels are asserted *on* the canonical node;
re-canonicalization is a one-assertion change with no reference rewrites; and the
model is uniform across entity types (morphemes degenerate to "node + one label"
— the indirection is paid only where the label actually drifts, i.e. places).
Cost: an extra indirection layer and a migration of existing `merged_into_id`
clusters — all L3-rebuildable authoring work; runtime/L4 is the flattened
projection (D38), so zero runtime cost.

### D50.2. Two tests sort every edge.

The taxonomy is not a list; it is two tests applied to each candidate edge:

1. **Collapsibility:** *"if these two nodes merged into one, would we lose
   information we still need?"* No → **identity** (collapsible). Yes →
   **relational** (never collapse). (`ne`≈`niwe` → merge, one morpheme.
   `town`–descends-from–`tūn` → never; the chain across eras/languages is the
   point. This is also why `cognate_id` is NOT identity — a cognate cluster is
   the *closure* of descent edges; `silly`/German `selig` are cognate but mean
   different things now, D28.)
2. **Identity is bare and timeless** (D40/D45/D44/D46/D8): identity edges connect
   *stripped, position-free, era-free, inflection-free* identities. Surface,
   dash-position, era-reflex, and inflection are per-stratum *lenses* on one
   identity — never separate identities, never identity keys. (This is what
   stops the assertion layer from re-spawning the D45 dash-identity bug class.)

Two guardrails: relational axes are **orthogonal** (D28 — descent ≠
semantic-equivalence ≠ cognate-peer ≠ containment; never a generic `related-to`),
and everything **projects to SQLite** (no graph engine, D42).

### D50.3. Three edge families.

- **Family A — identity (collapsible).** Realized as `mint-canonical` + `bind`
  (observation → canonical) + `merge-canonical` (reconcile two minted nodes) +
  `canonical-label` (per-stratum label on a canonical node). `bind.kind` ∈
  {same-morpheme, same-place, same-sense, inflection-of}. Generalizes
  `merged_into_id` (same-morpheme) and `lemma_id` (inflection-of). Region
  within-stratum coding folds (`SUR`≈`Surrey`) are NOT here — they are Class-A
  alias-map (D49).
- **Family B — relational (never collapse; orthogonal axes).** `descends-from`
  (etymon→etymon, edge_type inh/borrow/cog; generalizes `etymon_descent`) ·
  `means-same-as` (etymon↔etymon semantic; `meaning_synset`, orthogonal to
  identity AND cognate, D28) · `glosses-as` (etymon→sense; `etymon_gloss`) ·
  `decomposes-into` (toponym→ordered etymon; `toponym_etymology_element` — **the
  only birthplace of position**, derived by index per D40/D43; identity edges
  never carry position) · `composed-of` (etymon→ordered etymon — the
  morpheme-level analog of `decomposes-into`: a composite morpheme like
  `ington` is composed-of `ing`+`tūn`; part-whole, orthogonal to descent and
  identity; mined gloss-correctly from cross-scholar coarse-vs-fine breakdowns,
  wyrd-h5u1) · `contains` / `succeeds` / `located-in`
  (region/jurisdiction structure per stratum). `cognate_id` is a *projection* of
  the `descends-from` closure, not stored identity.
- **Family C — canonicalization choice (select/flag among Family-B
  alternatives).** `canonical-decomposition` (which `decomposes-into` parse is
  canonical, per place[+culture]; `toponym_decomposition.is_canonical`) ·
  `decomposition-spurious` (flag a contaminated breakdown — the Class-B answer to
  D3's neighbor-contamination blind spot, e.g. the `ford+botl` row) ·
  `canonical-label@stratum` (which observed form is canonical at a stratum — the
  bridge to Class-A labeling).

### D50.4. The assertion record (one uniform shape).

```
{ id, predicate, subject:{type,ref}, object:{type,ref}|null,
  polarity: affirm|refute|retract, retracts:<id>|null,
  qualifiers:{ordinal,stratum,edge_type,kind,value,…},
  confidence, method, source, actor, rationale, timestamp }
```

- **Typed refs** keep the orthogonal axes clean (an edge knows its endpoints'
  types).
- **Qualifiers** carry the per-predicate axes (ordinal, stratum, edge_type) so
  position/era stay derived lens axes, never folded into a ref (D40/D45).
- **Append-only** (D21/D22): `refute` (assert NOT-same / spurious) and `retract`
  (withdraw a prior assertion by id) are new records, never edits. The Newton
  trace needs all three polarities — *affirm* `ne`≈`niwe`, *refute* the
  `ford+botl` breakdown, *retract* the bad `new→ne` legacy merge.
- Every record carries the full provenance quad (confidence/method/source/actor +
  rationale, D24). Identity-collapse defaults to **leave-separate** (D46
  asymmetry): a missed merge is harmless duplication, a wrong merge is corruption
  — so binds need a high confidence bar + adversarial verification before the
  projection applies them. A heuristic's pairwise "looks-same" is just a
  low-confidence `bind`; confidence does the gating, so there is no separate
  "proposal" type.

### D50.5. L2 streams + L3 projection.

Assertions live in **per-predicate JSONL** under `canonicalization/`
(`_canonical_nodes.jsonl` for the minted ids; `_assert_<predicate>.jsonl` for
each predicate), mirroring the existing `_reflexes.jsonl` /
`_element_glosses.jsonl` sidecars so diffs stay scoped and the
**rebuild-runbook-currency-reviewer** tracks each as a discrete rebuild input.
The projection (the rebuildable L3 collapse graph) reads observations (L2) +
canonical-node declarations + assertions and materializes: canonical entity
tables with per-stratum labels; `observation.canonical_id` (generalizing
`merged_into_id`); relational edge tables; rollup views (generalizing the
`*_canonical` views). **Relational edges as mined point at observations**
(preserve what was mined, D21); rollup to canonicals flows through the binds.
Determinism per D36.9: same L2 → byte-identical L3.

### D50.6. The projection: L2 assertions → L3 collapse graph (wyrd-u6fn.3).

The assertion streams are projected into the rebuildable L3 graph by a
deterministic stage that runs *after* the existing `rebuild-from-jsonl`
materializes the observation tables.

**Bind granularity** (the one open parameter D50 left): bind at observation-row
level — morpheme = `etymon` row, sense = `etymon_gloss` row, **place = `toponym`
row by default, with per-`toponym_etymology`-row overrides that take
precedence.** The override-beats-default rule yields both directions with no
special predicate: merge (Archdeacon Newton's two `toponym` rows → one
`canonical_place`) and split (toponym 1725's three readings, rows 778/779/780 →
three places).

**Schema** generalizes the D22 redirect columns: synthetic `canonical_morpheme`
/ `canonical_place` / `canonical_sense` tables (stable, L2-declared ids; a
`merged_into` self-FK for `merge-canonical`, smallest-id-wins + chain-flatten per
D22's two-step lesson); `canonical_*_id` binding columns on the observation
tables (same style as `merged_into_id`/`cognate_id`); a
`canonical_label(type, id, stratum, value)` table for per-stratum labels.
Family-B relational edges mostly already exist (`etymon_descent`,
`etymon_meaning_synset`, `etymon_gloss`, `toponym_etymology_element`) — the
projection adds rollup-through-canonical views (replacing the
`COALESCE(merged_into_id, lemma_id, id)` chains with a clean FK join); new tables
only for region structure.

**Algorithm** (deterministic, D36.9): mint → bind (keep affirmed, non-retracted,
non-refuted, `confidence ≥ gate`; place overrides win) → `merge-canonical` →
per-stratum labels → relational + Family-C flags. **Conflict rule** (D46-faithful):
an observation bound above-gate to two different canonicals is left **unbound +
flagged** (observability, D24), never silently resolved — the contradiction is
the signal a `merge-canonical` is owed. Reversible via
`clear-enrichment --stage=canonical` (D22); the streams join REBUILD.md's L2
replay registry (enforced by the rebuild-runbook-currency-reviewer).

Implementation order: `wyrd-u6fn.2` (streams + `canonical_*` DDL) → `.3` (this
projection) → `.4` (migrate `merged_into_id` / `cognate_id` / `lemma_id` into
legacy-import binds so the first projection reproduces today's clustering, then
repairs adjust).

### Why this shape.

It is the minimal generalization of patterns already proven in the codebase:
non-destructive merge (D22), method-stamped reversible enrichment (D22/D24), the
cognate-vs-synset axis split (D28), and the timeless-identity / per-stratum-lens
discipline the morpheme axis already runs (D44/D46). It makes the
partial-but-correct collapse machinery **uniform** (one model for every entity
type), **reviewable** (every canonicalization is an attributed, dated, reversible
claim with a rationale — strictly more rigorous than today's scattered
method-stamp columns), and **agent-augmentable** (an agent's web-researched merge
is a low-confidence assertion in a quarantined tier, cf. D2/D19, that a gate or
human promotes). The graph is the working surface; the assertion log is the
canonical artifact.

### Bounded by / deferred.

Authoring-layer only; runtime/L4 untouched (D38). Bind granularity and the
concrete L3 DDL are settled in D50.6. Still open: gloss cleanup is itself
recursively A-vs-B (string-normalization vs `same-sense-as`), and the *compressed
gloss* (a short canonical label per sense — the readable "what does this toponym
mean" output) rides on the `canonical_sense` nodes here (wyrd-u6fn, separate
ticket). The migration of `merged_into_id` / `cognate_id` / `lemma_id` into
legacy-import binds is wyrd-u6fn.4. The full edge-taxonomy + schema + projection
first-cut detail live in those tickets' design fields.

Epic: wyrd-u6fn (Class B). Specifies D49's Class-B half.

## D51. Scholarly toponym breakdowns are decomposition substrate — admission evidence, decomposer test corpus, gloss-correct mining source (wyrd-oth3 / wyrd-h5u1 / wyrd-7hbp, 2026-06-16).

The `toponym_etymology` breakdown corpus (~17K toponyms of scholar-attributed
morpheme decompositions) is not merely reference data — it is the substrate
three distinct pieces of decomposition machinery stand on. Naming the three
roles, plus the one rule that governs mining from it, is the through-line of the
wyrd-aicu.9 / wyrd-oth3 work.

### D51.1. Three roles.

1. **Admission evidence** (wyrd-oth3). A morpheme a scholar used as an element in
   a real place-name breakdown is admissible to the runtime inventory on that
   evidence alone — a fifth `export_meanings` admission path
   (`include_toponym_breakdown`), parallel to rando-port / wiktionary-empirical /
   wave-2-enriched and bypassing the D4 dictionary-witness gate. It is a
   place-name-specific evidence channel, distinct from (and for place-name
   generation arguably stronger than) dictionary witnesses: it recovers the
   specifiers and heads scholars actually used that the witness gate misses — the
   absence of which made the matcher junk-tile `Aldermaston` → `Al+Der+Ma+Ston`
   and dump spurious counts into the proportions.

2. **Labeled test corpus** (the decomposition grader). The breakdowns are ground
   truth for the decomposer: grade matcher versions against them by cognate-
   cluster recall / precision / coverage / head agreement — order-robust, because
   `ordinal` is not reliably surface-order (wyrd-z3me) — and diff configurations
   head-to-head. Two findings are load-bearing. The **regression list** (parses
   the new config gets wrong that the old got right) is the **overfitting
   tripwire**: a decomposer change is acceptable only if net agreement rises AND
   the regression list is empty or each entry is individually defensible — this
   keeps the matcher to general rules and out of corpus-fitted special cases. And
   **coverage-up-while-agreement-down is a real failure mode**: the unfiltered
   breakdown admit raised coverage but dropped cluster recall/head — a regression
   coverage alone would have scored a win. Grade against the production inventory
   (what the proportions train on), not the committed dev seed.

3. **Gloss-correct mining source** (wyrd-h5u1 passthroughs, wyrd-65jh implied
   reflexes). Relational facts — a composite morpheme's constituents (`ington` →
   `ing`+`tūn`), a morpheme's modern reflex (`hough` ← `hōh`) — are mined from the
   breakdowns and modeled as D50 assertions (`composed-of`; `canonical-label`
   per stratum / reflex rows).

### D51.2. Mine from scholarly evidence, never surface segmentation (the gloss-blind rule).

A composite's expansion or a reflex pairing must come from what scholars actually
attributed, never from segmenting the surface against the inventory. Surface
segmentation is **gloss-blind**: it would expand `barton` → `bar`+`ton`, but the
scholar's breakdown is `bere`+`tūn` (barley-farm). Evidence streams, in order of
cleanliness: (1) cross-scholar coarse-vs-fine on the same toponym — one breakdown
coarsens a 2+-element span of another, the rest aligning; intra-toponym, both
gloss-bearing, zero inference (the "whole-vs-split" disagreements the aicu.9
cross-region investigation treated as *noise* — `burhtun` == `burh`+`tūn` — are
reclaimed here as *signal*); (2) matcher-vs-scholar miss (matcher emits the
composite, scholar is finer; the residual scholar elements are the expansion);
(3) pooling across toponyms sharing the composite ending. This rule sits
alongside D40/D45 (identity is bare and derived) as a "don't reintroduce this bug
class" invariant — extended to the mining boundary.

### D51.3. Composites are passthroughs: surface ≠ attribution.

A composite toponymic morpheme (`ington`) is an acceptable thing for the matcher
to MATCH — it parses cleanly, survives Occam, carries no junk-tiling risk — but it
is RECORDED as a **passthrough** that expands to its constituents for attribution
and rendering: the same surface-vs-attribution divergence as the aicu.9
connective. Realized as a `composed-of` relational edge (D50.3 Family B). Without
it, the matcher's Occam tiebreaker (fewer morphemes wins) prefers the coarse
single composite over the finer correct split (`Ald+ington` beats `Ald+ing+tūn`),
dropping agreement — so **breakdown admission (D51.1.1) is gated OFF by default**
until the passthrough lands (wyrd-h5u1) and the grader (D51.1.2) shows a net win.
The originally-considered fix — a composite-penalty / granularity-weight
tiebreaker fighting the segmentation-DAG pruning — was rejected in favor of the
passthrough: the passthrough lets the composite win the parse and expands at
attribution, needing no scoring surgery and working even when the finer split is
not independently the best parse.

### D51.4. Genitive-`s` split prior — homograph disambiguation from scholarly breakdowns (wyrd-aicu.9 / wyrd-aicu.9.1).

The matcher mis-parses the genitive `s` of `X's-tūn` (a *town*) as `X + ston` (a
*stone*): the greedy `(unaccounted, morphemes)` score prefers the longer `ston`
(0 unaccounted) over `X + s + ton` (the genitive `s` left unaccounted), so Occam
picks stone. `Bishopston` = Bishop's-`tūn` decomposes as `bishop + ston`,
inflating the `stone` morpheme's weight and starving `ton` in generation. The fix
is a **homograph-aware, per-suffix probabilistic split** seeded by the breakdown
corpus — NOT a hardcoded `if ston` rule (which `-ster`/minster and the ~6%
genuine `-stone` cases would break). This is a fourth use of the D51 substrate,
sitting under the same provenance discipline:

- **Candidate suffix pairs are auto-discovered**, never hardcoded. Every surface
  `L = 's' + S` where both `L` and `S` are real reflexes (`ston`/`ton`,
  `sley`/`ley`, …) is a candidate; the set grows as the inventory grows. Most are
  coincidental noise (`sage` ⊃ `age`) and only pairs with real scholarly split
  evidence become active.
- **Classification is by COGNATE CLUSTER, not surface membership** — the scholarly
  etymon (`stān`) and the surface reflex (`ston`) are different rows; clustering
  (cluster 358859 unifies stān/ston/stone/stan) recovers both the genitive-split
  (`tūn → town`) and the genuine-`stone` classes, where surface membership is
  lossy. A breakdown touching both classes or neither is skipped and counted, not
  guessed. This is **D51.2's gloss-blind rule applied** to suffix homographs: the
  reading comes from what scholars attributed, never from segmenting the surface.
- **The evidence hierarchy is deliberate and ordered**: the cognate-cluster
  verdict is **decisive**; the historical `-es-`/`-s-` genitive marker
  (`Kingston ← cyninge·s·tun`, wyrd-aicu.9.1) is a **subordinate** tiebreaker that
  only resolves the cluster-ambiguous `both`/`unclassified` residue and NEVER
  overrides a decisive cluster verdict. The marker fires only on genuinely
  historical forms (folded form ≠ folded modern name) — `toponym_attestation` is
  ~97% modern-name echo, and running the marker on a bare modern `ston`
  (`= s + ton`) is what mis-split the protected genuine-stone set in the first
  cut; restricting it to real historical spellings keeps that set safe by
  construction.
- **Raw counts in storage, smoothing + hierarchical backoff at lookup**
  (`genitive_split_prior` table; `split_probability` at matcher-lookup time) — the
  D43 pattern: the table stays a faithful data mirror and the prior sharpens
  automatically as breakdowns accrue. LLM-free, deterministic, idempotent,
  replace-not-merge (mirrors `empirical_priors`).

**Why this is its own sub-entry and not just application of D43 + D51.2:** the two
load-bearing choices are not mechanical. (1) The *evidence hierarchy* — cluster
decisive, `-es-` marker strictly subordinate and historical-only — is a design
decision with a costly rejected alternative (the first cut ran the marker on
modern echoes and broke the genuine-stone set). (2) Disambiguation is **per-suffix
homograph**, a different axis from D8's deferred per-position *inflection*-density
case selection (genitive_strong/dative_or_pl by pre-/post-); this subsystem does
not implement D8's rules and D8's "not yet implemented" note still holds. CLI:
`lexicon mine-genitive-priors` / `dump-genitive-priors`;
`lexicon/genitive_priors.py`; migration `0018_genitive_split_prior`.

### Why this shape.

It unifies three previously-separate concerns onto one corpus under one
provenance discipline, and composes with the existing architecture: admission
extends the D4 + bypass-path gate; the grader operationalizes "general defensible
rules only" (D40); the mining lands in D50's assertion layer (`composed-of`); and
the gloss-blind rule extends the D40/D45 identity discipline to the mining
boundary. The breakdowns were always ground truth — D51 is the decision to use
them as such across admission, validation, and enrichment.

Tickets: wyrd-7hbp (uplift epic), wyrd-oth3 (admission), wyrd-myv4
(occurrence/junk filter), wyrd-h5u1 (`composed-of` passthrough mining), wyrd-65jh
(implied reflexes), wyrd-aicu.9 (genitive-`s` split prior, D51.4), wyrd-aicu.9.1
(subordinate historical `-es-` marker).

## D52. Cultural/linguistic-zone axis — rule-based, distinct from the dedup region hierarchy (2026-06-17).

**The decision: model "cultural zones" (Danelaw / Celtic / Roman) as a separate,
coarse, linguistically-defined generator axis — NOT a level of the dedup region
hierarchy — and derive zone membership rule-based from existing signals rather
than mining a classifier.** This is the home D49 promised for the
`England (Danelaw)` region rows (a category error in the dedup field, migrated
out by wyrd-3q6m.1/.4) and the backing for a future "Danelaw town names"
generator surface. Filed/built as the wyrd-hytz design + data-model spike; the
generator knob, runtime bundle, and SPA are deferred (below).

### Why a separate axis (not a region level)

Per D49's "two purposes" split: the region hierarchy serves **dedup** (tight,
admin, place-disambiguating); zones serve **cultural color** (coarse,
linguistic, cuts across shires). A place is at once "England → Yorkshire" (admin)
*and* "in the Danelaw" (zone). Cramming both into `region` conflated two
classification systems (the `England (Danelaw)` mess). Zones are their own axis,
composing with `--culture` / `--stratum` (D32) / `--era` (D44) the way those
already compose — never contaminating the dedup blocking key.

### Two classifications

- **morpheme → zone (by etymon `language`) — generation-relevant.** A zone's
  flavor IS its characteristic morpheme mix. `old-norse/east-norse → danelaw`,
  `celtic/brittonic/welsh/cornish/… → celtic`, `latin/medieval-latin → roman`;
  everything else (Old/Middle/Modern English, `unknown`) is the un-zoned
  `anglo-saxon` default.
- **place → zone (by `region` / `country`) — secondary.** For analysis /
  empirical priors; multi-membership allowed. Danelaw = the historic Danelaw
  shires; Celtic = Wales/Cornwall/Ireland/Scotland. Roman place-membership
  (scattered Roman sites) is name-pattern work (`ceaster`), deferred.

### Rule-based, not an LLM/enrichment classifier (the spike's finding)

The ticket gestured at a D32-stratum-style mined classifier. The spike measured
the existing signals and found they map to zones cleanly enough that mining is
unnecessary for a first axis:

- morpheme→zone (by `language`): danelaw 2,061 / celtic 3,269 / roman 461
  etymons, vs ~22,342 native old-english — a clean, deterministic partition.
- place→zone (by `region`/`country`): the Danelaw shires cover 20,158 / 53,647
  England toponyms (37%); Wales 2,164.

So the recommendation is **rule-based** (deterministic, no cost, reversible),
captured in `data/zones.yaml` + `lexicon/zone_classify.py`. A mined classifier
remains an option later for the genuinely-ambiguous tail (a substrate Celtic name
in an English shire, Roman-by-name-pattern), but is not the first cut.

### Deliberately NOT in this slice

The generator `--zone` knob, its vector-scoring integration (weight morphemes by
zone, like `--stratum`), the bundle export, and the SPA. Those are the actual
feature; this slice is the design + the rule-based data model that de-risks it.
Zone definitions are a `zones.yaml` edit (no code change) the way register
effects (D37) and structures are.

### Relationship to prior decisions

Realizes D49's deferred zone axis. Mirrors D32 (within-language stratum) as a
sibling generator axis and D44 (`--era`) / `--culture` in how it will compose.
Bounded by wyrd-3q6m.1 (must not contaminate the dedup region hierarchy) and the
Class-A guards (zones are an enrichment classification, not an ingest-gated
controlled vocabulary). Ticket: wyrd-hytz.

## D53. Case is NEVER morpheme identity — the canonical surface-encoding standard (2026-06-30, wyrd-x5y4).

**The decision: a morpheme's stored identity is a BARE, LOWER-CASE surface
string. Case — like dashes (D45) and position (D40) — is render-time decoration
the generator owns by position (D39), never part of identity.** This generalizes
D45 (dashes-never-identity) into ONE normative encoding standard for every stored
surface key (`meaning.usage_key`, the `proportions_*` `usage_key` columns,
`fantasy_morpheme.usage_key`, reflex surfaces).

### The canonical surface-encoding standard (the normative spec the gate enforces)

A stored morpheme surface key MUST be:

1. **NFC-normalized** (Unicode canonical composition — one code-point sequence
   per grapheme).
2. **Lower-cased**, with a documented internal-capital allowlist for genuine
   medial capitals (`McMansion`, `al-Quadim`). The allowlist is **currently
   empty** — the data has 0 internal-capital morphemes (every capital is a
   *leading* capital inherited from a source toponym), so the present rule is a
   blanket lower-case. A real medial-capital morpheme is added to the allowlist,
   never granted a fold bypass.
3. **Dash-free** in identity (D45 — boundary position dashes stripped; a genuine
   interior hyphen like `al-Quadim` is part of the surface, not a position mark).
4. **Whitespace-clean** (surrounding whitespace stripped, wyrd-an8u; interior
   spaces in a multi-word surface kept).

Everything else about how a morpheme *appears* — capitalization, position
dashes, stratum/era forms — is DERIVED at render (D39) from the bare surface +
the slot it lands in.

### Why this was a bug (the same root cause as D45)

Morpheme surfaces are minted from source-toponym text. A morpheme attested
word-initially inherits the toponym's LEADING CAPITAL (`Abbey`); the same
morpheme attested internally stays lower-case (`abbey`). Dashes were folded out
of identity (D45) but **case never was** — so `Abbey` and `abbey` were stored as
two identities. Committed-seed audit: 657 / 1383 meaning keys (48%) carried an
uppercase letter, ALL leading (0 internal), producing **566 case-variant
duplicate groups (~41% of the pool)** — confirmed true duplicates, not
homographs (`Abbey`/`abbey` both `modern_english`), some DEGRADED (a capitalized
`Acre` with no language vs the `old_english` `acre`). Duplicates fragment the
proportions/scoring pool (D36) and silently collide on the case-folding runtime
load (last-wins data loss).

### Lookup was already case-insensitive — the fix is at the MINT

The match path already folds case: `build_morpheme_trie` lowercases every key
(`trie_matcher.py:186`), `Meaning.__str__` folds on access, the SPA `normKey`
lowercases, `fantasy_morpheme` collates NOCASE. Lookups always *tolerated* case
variation — the bug is purely that STORED keys weren't folded, leaving
duplicates at rest. The fix lowercases the surface at MINT (the export
normalizers), exactly as D45 de-dashed at mint; the de-dash union machinery
(`_write_meanings` regroup-then-repick) collapses the case duplicates
mechanically — no LLM merge.

### The inconsistency this standard closes

Case-folding was ad hoc across the export writers: `meaning`
(`_bare_modern_usage`), `proportions_usage`, and `proportions_single_usage` did
NOT fold; `proportions_attested_language`, `proportions_bare_word_position`, and
`fantasy_morpheme` DID. Three folded, three didn't — exactly the drift a single
normative standard + an enforcing data-gate prevents.

### Enforcement

A data-gate (parallel to `test_kenning_dedash_data_gate.py`) asserts no stored
surface key contains an uppercase letter (outside the allowlist), across the seed
+ any built bundle — flipped to CI-gating once the reseed makes the committed
data conform (wyrd-x5y4.4/.5). Stops case reaccreting the way it (and dashes) did.

### Relationship to prior decisions

Generalizes D45 (dashes-never-identity) — same principle, same playbook. Composes
with D39 (render owns decoration), D40 (position derived, never a match gate),
and D44 (identity is era-invariant): a morpheme's stored identity is a bare,
lower-case, timeless, position-free surface; case, dashes, position, and era are
lenses applied on the way out. Ticket: wyrd-x5y4 — D53 standard (.1),
meaning-table fold (.2), proportions fold (.3), enforcement gate (.4), reseed (.5).
