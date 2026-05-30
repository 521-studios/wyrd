# Census — form-of / duplicate etymon consolidation (wyrd-p4nb, P0)

Measured 2026-05-30 against the live DB (`~/.wyrd/lexicon.db`, the converged
post-rebuild corpus). Answers the epic's headline question — **"how many
lemmas would disappear if we folded form-of forms + merged same-meaning
lemmas?"** — and sizes each phase. No code; measurement only.

## Denominators

| set | count |
|---|--:|
| total etymons (unmerged) | 2,316,563 |
| glossed etymons | 19,350 |
| **reflex-linked etymons (reach generation)** | **4,968** |
| lemma_id-linked (`link-lemmas-v1`) | 94,666 |

`4,968` is the number that matters for what a user sees — the lemmas that
actually reach generation. Percentages below are against it.

## The headline: variant-doublings

A **variant-doubling** is an etymon whose `(canonical_form, language)` *also*
appears in `etymon_variant` pointing to a **different same-language lemma** —
i.e. it lives as both a lemma AND a registered variant of another lemma. Per
the target model these should not be their own lemma.

| | count | of reflex-linked |
|---|--:|--:|
| **variant-doublings, reflex-linked** | **1,303** | **26.2%** |
| └ share an exact gloss with their parent → **deterministic fold (P2)** | 70 | 1.4% |
| └ no shared gloss → **need LLM meaning-judgment (P3)** | ~1,233 | 24.8% |
| └ carry scholarly cites → **hybrid cite migration applies** | 301 | 6.1% |
| └ already `lemma_id`-linked | 48 | — |
| variant-doublings, **whole corpus** | 750,486 | — |

**So up to ~1,303 of 4,968 generation-reachable lemmas (26%) are variants
masquerading as lemmas.** Only 70 are deterministically foldable today (they
share an exact gloss with their parent); the other ~1,233 are exactly the
case the user described — **same-form-as-a-variant but the meaning match
needs an LLM to confirm** (`burh`'s gloss is literally "alternative form of
burg", which doesn't string-match `burg`'s "fort"). That validates the P3
LLM-with-JSONL design: the bulk can't be decided by string equality.

Corpus-wide, **750,486** forms double as variants — the full cleanup scope,
though the vast majority never reach generation.

## The visible junk (what surfaced this epic)

Pointer-gloss etymons — gloss is a cross-reference, not a meaning
("h-prothesized form of ea", "alternative form of burg"):

| | count |
|---|--:|
| only-gloss-is-a-pointer | 22 |
| └ **reach generation** (the `Hea-` junk users see) | **9** |
| └ scholar-cited | 13 |
| └ already `lemma_id`-linked | 2 |
| pointer **plus** a real gloss | 60 |

The visible problem is tiny (**9** morphemes) — but it's the tip of the
1,303-lemma variant-doubling iceberg. Fixing only the 9 is the
`[P7]`-smallest option; the real win is the doubling fold.

## Why "merge same-meaning lemmas" can't be string-equality

A naive exact-gloss-duplicate pass reports `800 groups / 2,024 mergeable`
among reflex-linked lemmas — but that number is **not usable**:

- **Generic glosses dominate.** Top shared glosses are `a male given name`
  (39 etymons), `hill` (38), `wood` (30), `valley` (29), `moor` (27),
  `marsh` (26), `grove` (25), `forest` (25). Thirty-eight *different* OE
  words all gloss "hill" — they are **not** the same lemma and must not
  merge.
- **Multi-gloss double-counting** inflates the excess (an etymon with 5
  glosses lands in 5 groups).

So same-meaning merging must be a **judgment on a specific candidate pair**
(a variant against *its* parent lemma), not free clustering on gloss strings
— which is why P3 routes the doubling/`lemma_id` candidates through an LLM
and persists the verdict to `_merges.jsonl`, rather than auto-merging on
gloss text.

## Estimated lemma reduction (generation-reachable)

- **Floor (deterministic, P2 only):** ~70 lemmas fold away (share parent
  gloss) + the 9 pure-pointer-reaching → on the order of **~80**.
- **Ceiling (if every doubling folds):** **~1,303 (26%)**.
- **Realistic:** between those — the LLM pass (P3) decides how many of the
  ~1,233 no-shared-gloss doublings are genuine variants vs homographs that
  stay. A back-of-envelope split won't be honest; P3's per-pair verdict is
  the real number, and it lands in `_merges.jsonl` so it's auditable.

## Implications for the phases

- **P1 (schema `attested_form`)** is load-bearing: **301** reflex-linked
  doublings carry scholarly cites that must migrate to the parent with the
  surface form stamped.
- **P2 (deterministic fold)** is small and safe: ~70 share-gloss pairs +
  the pointer-glosses parseable to an existing lemma.
- **P3 (LLM merge → `_merges.jsonl`)** is where the bulk (~1,233) is
  decided — confirms the user's read that this needs an LLM and must
  round-trip through L2.
- **P4 (export resolution)** is mandatory regardless: the runtime follows
  neither `merged_into_id` nor `lemma_id`, so even the 70 deterministic
  folds won't reach generation without it.

## Queries

All counts reproducible from `~/.wyrd/lexicon.db`; the variant-doubling join
is `etymon e ⋈ etymon_variant v (v.form = e.canonical_form) ⋈ etymon parent
(parent.id = v.etymon_id AND parent.language = e.language AND parent.id !=
e.id)`, filtered `e.merged_into_id IS NULL`. The whole-corpus count (750,486)
runs ~minutes; the reflex-linked subset is sub-second.
