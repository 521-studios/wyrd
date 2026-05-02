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
