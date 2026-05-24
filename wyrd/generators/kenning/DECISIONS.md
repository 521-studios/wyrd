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

The mood vocabulary lives in `wyrd/generators/kenning/data/register_effects.yaml`
(catalog-driven since wyrd-kq7w.3 — the legacy `registers/moods.MOODS`
dict was ripped and replaced). New presets are picked from a tag-coverage
audit (≥5 subjects per candidate tag, distinct semantic identity, minimal
overlap with existing moods); `noble` was considered in wyrd-aky and
deferred until mining surfaced a `royalty` tag — landed in the kq7w.2
catalog migration. Lookup goes through `registers/effects.parse_mood_spec`
(vector path) or `registers/effects.mood_spec_to_legacy_form`
(proportion-table path); both consult the same catalog so a mood added
to YAML is immediately operator-visible on both scoring modes without
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

**Runtime status (wyrd-gfa, 2026-05-02; β-term wyrd-mj2, 2026-05-04):
shipped.** `Generator.select(novelty)` blends each bucket's empirical-
frequency distribution with a uniform marginal:
`(1-novelty)·empirical + novelty·uniform`. New `_blend_uniform` helper
handles the all-zero-empirical-weights edge case (returns 1/n so the
result is a normalized probability distribution per the contract). CLI:
`--novelty` (0..1). The `β·tag-class-prior` term landed via the
`--cohesion` knob (wyrd-mj2): `NameGenerator` threads `prior_tags`
through the structure walk, `_cohesion_boost` computes the per-key
class-conditional likelihood from the tag co-occurrence model, and the
result is passed to `Generator.select` as a `key_boost` multiplier
applied to empirical weights before the novelty blend. The realized
math is multiplicative-then-blend (`weights = empirical * boost; result
= (1-novelty)·weights + novelty·uniform`) rather than the strict
additive `α·empirical + β·tag-class-prior + γ·marginal` of the textbook
formulation, but the operational effect is the same — novel
combinations refine toward attested *patterns* (descriptive+topography)
when cohesion>0, and the GM tunes the strength independently of
novelty. wyrd-9gt closed as superseded by this realized form.

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
- `MeaningGenerator.keep_keys_for_era(era_range)` — precomputes the
  allowed-usage frozenset; cached per range; collapses to None on
  full-coverage so the no-filter fast path stays bit-stable.
- `Generator.select(keep_keys=...)` — bucket-level intersection.
- `NameGenerator.select(era_range=...)` — threads through every
  per-bucket pick.

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
Cultures without a per-culture restriction (irish / breton today
— no classifier yet for those families) fall back to the broader
`ALL_STRATA` typo-check. The `LANGUAGE_TO_FAMILY` map is built
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
  asterisk `*` flag in the CLI output is the truth-marker. Falling
  back to the source would make the era ladder look reversed
  (`oe-late: king → me: chinge → modern: cyning`).

### Bundle plumbing for SPA-side rewinder (wyrd-obpw)

The Lambda runs on bundled data (the lexicon DB is 673MB —
too big to ship). To enable a SPA `KenningRewind`, era-reflex
data is precomputed at bundle-build time:

- `lexicon._fetch_root_era_reflexes(db, root_id, root_language)`
  computes `{target_language: [forms]}` for the family root via
  the same three-tier picker. Wired into `_gather_family` so each
  family carries `era_reflexes` data.
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
* `--novelty 0.5` is handled by the D17 cohesion-adapter
  layer, not the per-axis weights. See D36.5.

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

The existing D17 Bayesian-mixture novelty + cohesion model
stays the runtime sampling layer. The vector-driven
generator produces per-lemma vector scores; a
`CohesionContext`-wrapped scorer applies the existing
tag-class-prior multiplier (`key_boost` in the current
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
  NameGenerator.select_via_vector` is the vector-mode entry
  point; `generators/kenning.py:_generate_via_vector` is the
  Kenning-level dispatcher. `Kenning.generate` reads
  `scoring_mode` from params and routes accordingly.
* **Drift measurement** — `runtime/drift_measurement.py` ships
  pure-Python metric primitives (KL divergence,
  total-variation distance, top-N name overlap,
  decomposition-rate delta, position-distribution delta,
  Spearman rank correlation). `runtime/drift_runner.py`
  bridges metrics to the live `Kenning.generate` for per-seed
  isolation. CLI: `wyrd kenning lexicon drift-report`.
* **Tolerance bands** — `runtime/realism_tolerance.py` ships
  the `ToleranceBand` dataclass + `check_drift_against_tolerance`
  primitive. Default bands are wide-open today (regression
  suite is INACTIVE as a drift gate per the explicit Phase 6b
  review-then-codify cycle); operator tightens via
  `PER_CULTURE_TOLERANCES`.
* **Regression suite** —
  `tests/test_kenning_realism_regression.py` parametrizes
  per-culture (english / welsh / irish / breton) +
  register-composition smoke tests. Per-culture tests
  `pytest.skip` when one side returns 0 samples (today's
  expected state for cultures without operator-supplied
  `--priors-path` — meaningless drift comparison shouldn't
  bogusly fail).
* **CLI surface** — `cli/generate.py` adds `--scoring-mode`,
  `--priors-path`, `--baseline-weight`,
  `--phonological-weight`, `--semantic-weight`,
  `--position-weight`. Vector-only flags in proportions mode
  are a silent no-op (deliberate operator-friendly contract:
  'I forgot --scoring-mode=vector' produces a normal
  proportions name).

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

Default `--scoring-mode` stays `proportions` until tolerance
bands are tightened from the review-then-codify cycle. The
legacy proportions path remains bit-stable throughout the
ecjp epic; no `<culture>_proportions.json` deprecation has
been triggered yet (planned as part of the ecjp.10 bundle
work — one release period of warning-not-error during
transition).

## D37. Phonaesthetic-vector framework supersedes the legacy MOODS dict (wyrd-kq7w, 2026-05-21).

D6 originally specified moods as a code-defined `MOODS` dict of
`{name: {tags, harshness}}` recipes living in
`registers/moods.py`. The wyrd-kq7w epic ripped that approach and
replaced it with a catalog-driven composition framework where each
named effect is a per-dimension vector triple. The MOODS dict is
gone (deleted in wyrd-kq7w.3); the catalog at
`wyrd/generators/kenning/data/register_effects.yaml` is the single
source of truth for mood-name resolution on both scoring modes.

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

* **Vector path** (`scoring_mode='vector'`): `parse_mood_spec(spec)`
  returns a graduated `RegisterEffect`. The dispatch composes
  `[adapter_effect, *catalog_effects]` via
  `compose_register_effects` into the request's register, and the
  per-lemma scoring loop dot-products this register against each
  lemma's stored `PhonologicalVector`.
* **Legacy proportion-table path** (`scoring_mode='proportions'`,
  default): `mood_spec_to_legacy_form(spec)` returns the legacy
  `(tags_list, harshness_scalar)` tuple by extracting
  `effect.semantic_tags.keys()` for tags + the catalog's
  `cluster_density` dim for the harshness scalar. Bit-stability
  drift is bounded — the catalog's `harsh: cluster_density=0.6`
  reads slightly softer than the legacy `MOODS["harsh"]=1.0`
  (acceptable per the kq7w.3 "distribution match within tolerance"
  gate; not byte-identical).

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
catalog-composition seam. The legacy proportion-table sampler
still runs (default `scoring_mode='proportions'`); the rip
swapped its mood-resolution source from MOODS to catalog, not
the sampler itself.

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

## D38. L4 runtime DB: SQLite-on-S3 replaces meanings.json + proportions JSONs (wyrd-d90t, 2026-05-24).

The 2026-05-20 post-wyrd-wz82 bundle re-emit grew `meanings.json`
from 54MB to 113MB — over GitHub's 100MB push limit. Growth was
driven by the `*_phonological_vector` fields landing on every
form (wyrd-kq7w.1 enrichment). The JSON-bundle era for kenning
runtime data is over; this entry records the L4 architecture that
replaces it.

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
  random over the proportions (millions of rows summed) and
  point-looks-up tag statistics. SQL is the right tool there.

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

