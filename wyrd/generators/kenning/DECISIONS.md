# Architecture decisions for the kenning lexicon

This file records the policy and design decisions we've landed on so they
don't live only in conversation. Each entry has a short rationale.

## D1. The lexicon is the authoring layer; meanings.json stays the runtime.

The runtime generator continues to read `data/meanings.json`. The lexicon
SQLite DB is an authoring-side store: mining, citation tracking, consensus,
lemma linkage. A future export step (filed) will regenerate `meanings.json`
from the lexicon when we want changes to ship.

Why: keeps the runtime path stable while we churn on the data layer.

## D2. LLM-provider tiers.

Three roles, three providers:

- **Bulk mining**: Ollama on Hades (Qwen 3.5 9B). Free, fast enough, no rate
  limits. Used for first-pass extraction across thousands of entries.
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
  the dash-stripped lowercased usage. Composes orthogonally with
  `--novelty` (D17): harsh skew applies to empirical first, then novelty
  blends with uniform.
- `--mood harsh:0.5` graduates the harshness skew via colon-suffix.
- `--mood pastoral` (plant / animal / water / agriculture / tree / bird
  tag union) for rural / agricultural feel.
- `--mood devotional` (saint / religious) for monastery / pilgrim feel.
- `--mood mortuary` (death / undead) for funerary feel — a strict subset
  of `grim` for cases where the GM wants death-themed without the
  military / monster axes.
- Multiple `--mood` flags compose by tag-union and max-harshness.

The mood vocabulary lives in `__init__.MOODS` as `{name: recipe}`. New
presets are picked from a tag-coverage audit (≥5 subjects per
candidate tag, distinct semantic identity, minimal overlap with
existing moods); `noble` was considered in wyrd-aky and deferred until
mining surfaces a `royalty` tag.

Power-user JSON API still exposes `harshness` (number, 0..1) for
graduated control without the colon syntax; `harshness` and
mood-derived harshness resolve via `max(...)`. The original `--grim`
boolean / `--harsh` float flags were superseded by `--mood` because
two parallel flags muddled axes that conceptually belong under one
preset.

Defaults to off (bit-stable historical behavior). The harshness score
is a heuristic — meant for relative ranking, not phonotactic fidelity.
Per-language phonology integration (using the IPA tables in
`phonology.py` for a more principled score) is a future refinement.

## D7. Sensitivity heuristic for non-European corpora.

Active living communities with sovereignty, land, or revitalization
stakes get **framework-only** treatment: we provide the language-pack
format and let community members supply the data. The PD-1900s
literature doesn't override this just because the books are out of
copyright.

- **Mine and bundle** (ancient-civilization frame, no living-community
  veto): Mayan, Nahuatl, Egyptian, Sumerian, Greek, Roman, all European.
- **Framework only** (active sovereignty / revitalization): Native
  American, Aboriginal Australian, Maori, Hawaiian, Khoisan.
- **Edge cases**: Hebrew (biblical-strata = bundle; modern political =
  caution), Bantu (varies by community size and politics).

Why: PD status is a legal property, not an ethical one. The line is
about whether a living community has a stake in how their language is
used.

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
lazy-cached pool flatten. `NameGenerator.select(inflection_density=…)`
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

Empirically, Qwen 3.5 9B on Hades produces good extraction yields on
English-etymology books (Skeat, Mawer, Ekwall) but underperforms on
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

Once the generator wires tag-level co-occurrence in, it samples from a
mixture: α·empirical-pair-frequency + β·tag-class-prior + γ·marginal.
The `--novelty` knob (or equivalent) shifts weight toward γ to allow
plausible-but-unattested combinations. The Braitham Gate regression
test pins this: the generator must still be capable of producing names
of that shape (descriptive+topography compound first-word) under
default settings.

Why: pure empirical conditioning produces a generator that can only
remix existing combinations. Pure marginal produces noise. The knob
gives the GM a continuum, with "do not regress on Braitham Gate" as the
fixed point on the novel side.

**Runtime status (wyrd-gfa, 2026-05-02): partial — two-term mixture
shipped.** `Generator.select(novelty)` blends each bucket's empirical-
frequency distribution with a uniform marginal:
`(1-novelty)·empirical + novelty·uniform`. New `_blend_uniform` helper
handles the all-zero-empirical-weights edge case (returns 1/n so the
result is a normalized probability distribution per the contract). CLI:
`--novelty` (0..1). The `β·tag-class-prior` term is **not** yet wired —
the D16 `tag_cooccurrence` + `tag_marginal` data is in the proportions
JSONs but threading neighbor-context through the structure walk is a
follow-up. Current implementation is the (empirical, uniform-marginal)
two-term mixture; the third term would refine novel combinations toward
attested *patterns* (descriptive+topography) even when the specific
morpheme pair is novel.

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
draw across all language pools. `NameGenerator.select(spelling_variety=…)`
pre-renders surface substitutions at select time so `__str__` stays pure.
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
  the legacy proportions code understands.
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

Public API (`wyrd/generators/kenning/trie_matcher.py`):

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
- `MeaningGenerator.keep_keys_for_era(era_range)` — precomputes the
  allowed-usage frozenset; cached per range; collapses to None on
  full-coverage so the no-filter fast path stays bit-stable.
- `Generator.select(keep_keys=...)` — bucket-level intersection.
- `NameGenerator.select(era_range=...)` — threads through every
  per-bucket pick.

`era=None` is bit-stable with the pre-PR sampler. Today's bundled
`meanings.json` carries no attested_years data — the filter is a
documented no-op end-to-end until the next bundle re-emit. Wiring
is in place so a single export cycle activates the feature.

Bare-label resolution is strict: a label not in the culture's default
era family raises ValueError listing the families that DO define it,
prompting the user toward the explicit `family/label` form. No silent
cross-family fallback — it's too easy to accidentally route an
'english' culture's `--era middle-irish` to the goidelic range.

## D17 refinement: cohesion knob (wyrd-mj2, PR #59).

D17 originally specified the novelty knob: blend each morpheme
bucket's empirical-frequency distribution with a uniform marginal,
allowing plausible-but-unattested combinations. wyrd-mj2 adds the
companion **cohesion** knob, which biases the OPPOSITE direction —
toward attested tag-class pairings.

As the structure walk fills slots, the union of previously-picked
usages' tags becomes the prior-context set. Each subsequent slot's
bucket gets a per-key multiplier:

```
raw_score(usage)  = Σ over (ta in prior, tb in usage.tags)
                       of tag_cooccurrence[ta|tb] / tag_marginal[ta]
multiplier(usage) = (1 - cohesion) + cohesion * (raw_score / mean_raw_in_bucket)
```

Mean-normalization preserves total mass — at cohesion=1 average-
likelihood candidates get ~1×, above-average >1×, below-average <1×.

Composes orthogonally with novelty (uniform-marginal blend, applied
LAST in `Generator.select`) and harshness (D6 phonological re-weight).
GMs can dial 'attested-pair fidelity' (cohesion) and 'novelty'
independently. No-op when the bundle carries no `tag_cooccurrence`
data (legacy bundles ride the bit-stable path even at cohesion=1).

Bit-stable at cohesion=0 — `_cohesion_boost` short-circuits to None
and `Generator.select` takes its harshness=0/novelty=0/key_boost=None
fast path.

Bit-stability across PYTHONHASHSEED: `_raw_class_score` sorts
`prior_tags` and `candidate_tags` before iteration. Without the
sort, set-iteration order varies across processes and float-
summation accumulates ULP-level different scores that could flip
weighted_choice outcomes at boundaries.
