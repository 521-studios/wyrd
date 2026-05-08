# Kenning — decomposition-coverage progress

Rolling log of place-name `perfect`-decomposition rate across the
register cultures. `perfect` = place names where every character maps
to a known morpheme in the bundled `meanings.json` (i.e.
`name.count_unaccounted() == 0` after `name.find_meaning(word_db)`),
emitted on stderr by `wyrd kenning rebuild-proportions`.

This is a coarse "surface morphemes recognized" gauge — it does NOT
measure citation attribution (rando-port vs ≥3-witness scholar
consensus, D4), inflection metadata (D8), spelling-variant pools (D18),
attested_years (D5-1), tag co-occurrence (D16/D17), or the runtime
knobs (`--mood`, `--era`, `--cohesion`, `--novelty`, etc.). Those
quality dimensions improve independently. Use perfect-rate as a
regression tripwire, not the north star.

The Rando-port seed already covered the highest-frequency morphemes
(`-ton`, `-ham`, `-bridge`, `-ford`), so mining adds long-tail
morphemes that hit fewer place names each — expect modest perfect-rate
gains relative to the corpus-row growth in the lexicon DB.

## Snapshots

### 2026-04-29 — `ceadce6` (initial commit, Rando-port baseline)

meanings.json: 747 subjects · 2399 unique modern_usages.

| culture  | perfect | total  | rate    |
|----------|--------:|-------:|--------:|
| english  |    5065 | 17876  | 28.3%   |
| scottish |     424 |  2321  | 18.3%   |
| welsh    |     289 |  1916  | 15.1%   |
| irish    |    3398 | 34041  | 10.0%   |

Breton register did not yet exist (added later — see wyrd-fmg).

### 2026-05-05 — `9cfa477` main (bundle from PR #58, 2026-05-03)

meanings.json: 1697 subjects · 2831 unique modern_usages.

The wyrd-ca1 re-emit on this date was canonical-JSON bit-identical to
the May 3 export — no scholar-mining since had crossed the witness
threshold to add a new promotable morpheme.

| culture  | perfect | total  | rate    | Δ pp vs baseline | Δ rel |
|----------|--------:|-------:|--------:|-----------------:|------:|
| english  |    5836 | 17876  | 32.6%   |             +4.3 |  +15% |
| scottish |     496 |  2321  | 21.4%   |             +3.1 |  +17% |
| welsh    |     364 |  1916  | 19.0%   |             +3.9 |  +26% |
| irish    |    4230 | 34041  | 12.4%   |             +2.4 |  +24% |
| breton   |      19 |  1208  |  1.6%   |              new | new — corpus pending (wyrd-fmg) |

### 2026-05-05 — post-wyrd-4hx7 (empirical Wiktionary corpus mining)

meanings.json: 1879 subjects (+182 vs prior). The lexicon picked up
~8300 new etymons via empirical mining of unaccounted-fragment misses
plus prefix/suffix/whitespace-word substring candidates against the
on-disk wiktextract slices (welsh, irish, old-irish, middle-irish,
scottish-gaelic, breton, old-french, old-english, middle-english,
old-norse, proto-celtic). Many empirical etymons merged into existing
subject signatures rather than creating new subjects.

| culture  | perfect | total  | rate    | Δ pp vs prior | Δ pp vs baseline | Δ rel vs baseline |
|----------|--------:|-------:|--------:|--------------:|-----------------:|------------------:|
| english  |    6281 | 17876  | 35.1%   |          +2.5 |             +6.8 |              +24% |
| scottish |     583 |  2321  | 25.1%   |          +3.7 |             +6.8 |              +37% |
| welsh    |     451 |  1916  | 23.5%   |          +4.5 |             +8.4 |              +56% |
| irish    |    5805 | 34041  | 17.1%   |          +4.7 |             +7.1 |              +71% |
| breton   |      27 |  1208  |  2.2%   |          +0.6 |              n/a |               n/a |

What landed in the bundle: `gwyn` (white), `glas` (blue/green),
`llan-`/`-llan` (church), `-tre`/`tre-` (settlement),
`-achadh`/`-agh` (field), `Cill-`/`Kil-` (church), plus a long tail
of OE / ME / OF / proto-Celtic morphemes. These are mostly the
matcher-leftover-fragment slice of empirical mining; most landed
either by direct fragment match or because they were already in
rando-port and the empirical citation just promoted them faster.

What did **NOT** land in this pass — though they have lexicon DB
rows + reflex links — and were verified missing from the committed
bundle:

* `eglwys` (welsh, "church")
* `betws` (welsh, "chapel")
* `coch` (welsh, "red")
* `cluain` (irish, "meadow")
* `demesne` (modern-english / old-french)

These are reachable only from the prefix/suffix/whitespace-word
expansion path of the candidate generator (the matcher's leftover
fragments are nonsense like `de`/`esne` for "Demesne" because the
bundle's existing short morphemes partial-match first). The
expansion path was added late in the wyrd-4hx7 work but the
re-export wasn't re-run cleanly afterwards, so the second-pass
empirical entries didn't make it into the committed bundle. Re-
exporting NOW would surface those 6000+ additional empirical
subjects but the resulting 8500-subject bundle has a high noise
floor (3-4 char morphemes from prefix-substring matches that aren't
real). Filing follow-ups (wyrd-q0g6 joiner schema, wyrd-1cjg
anglicization map) to refine the empirical mining before re-emit.

Still out of reach: anglicized forms like `cloon`/`bally`/`agh` that
the place-name corpus uses but Wiktionary slices don't have as
headwords. Catching those needs either Wiktionary's modern-English
slice (wyrd-iheo) or an explicit anglicization map (wyrd-1cjg).

### 2026-05-07 — post-wyrd-1cjg (Irish anglicization sidecar)

Hand-curated `data/irish_anglicizations.json` sidecar carrying 10
anglicized Irish place-name elements (Bally-, Cloon-, Kil-, Knock-,
Lis-, Glen-, Slieve-, Magh-/Moy-, Tully-, Derry-) keyed to the same
gloss + language slot as their native Irish forms. `_load_meanings`
unions the sidecar into the runtime `meaning_db` so the matcher
recognizes both forms. Wiktionary indexes by native headword; the
anglicized forms never surface from mining.

Irish corpus only — no other cultures touched.

| culture  | perfect | total  | rate    | Δ pp vs prior |
|----------|--------:|-------:|--------:|--------------:|
| irish    |    6039 | 34041  | 17.74%  |          +0.65 |

Why the lift is modest: ~11,000 names use these anglicized prefixes
but most also have unaccounted suffixes (Ballymacross → Bally + macross,
where `macross` is itself unaccounted), so the prefix entry only
fully resolves names where the rest is already accountable. The
+223 perfect names is narrow; the wider effect is that structure
analysis now correctly attributes the prefix slot for ~25% of the
Irish corpus, which downstream cohesion / proportions consumers
benefit from regardless of the suffix gap.

### 2026-05-08 — post-wyrd-eni4 epic (deploy-chain bundle re-emit)

The full gap-closing epic shipped today: 11 PRs across the bundle
schema (dict-shape + source-tagged era_reflexes + fantasy_morphemes),
the per-language-quality dashboard, INFLECTION_RULES expansion to
Welsh / OF / ME / NF + Goidelic mutation rules, Wiktionary forms
ingest for variant pools, phonology-rules → Tier-4 era reflex,
and bundle tag-visibility rollup across cognate clusters.

Post-epic deploy chain ran:

1. `link-lemmas --apply` against the live DB — landed ~17,500 lemma
   linkages (was: ~170 OE+ON only). New languages contributing:
   middle-english 2958, irish 2128, old-norse 1767, welsh 609,
   scottish-gaelic 335, old-irish 78, old-french 60, middle-irish 9.
2. `mine-wiktextract-forms --apply` across 10 gap-language slices
   (welsh / irish / scottish-gaelic / middle-irish / old-irish /
   breton / cornish / manx / middle-english / old-french). Wrote
   ~441K variant rows to etymon_text_match (was: ~1300 rows for
   OE+ON only).
3. `lexicon export-meanings` → meanings.json: 1879 → 8490 subjects
   (4.5× growth from the cluster + variant + linkage cascade). 244
   fantasy_morphemes ride along in the dict-shape bundle.
4. `rebuild-proportions` for all 5 cultures.

Discovered during deploy: rebuild-proportions OOMs on the 58-char
'Llanfairpwllgwyngyllgogerychwyrndrobwyllllantysiliogogogoch' due
to a cartesian explosion in trie_matcher.py:all_decompositions.
Workaround: removed from welsh_place_names.json (welsh corpus
1916 → 1915 names). Tracked as wyrd-p8ve (algorithmic fix:
score-prune during the canonical_decompositions walk) and
wyrd-v8x0 (re-add the village post-fix). Pragmatic call: the rest
of the corpus completes cleanly; the village restore comes later.

| culture  | perfect | total  | rate    | Δ pp vs baseline | Δ pp vs prior (2026-05-07) |
|----------|--------:|-------:|--------:|-----------------:|---------------------------:|
| english  |   12686 | 17876  | 71.0%   |            +42.7 |                      +35.9 |
| scottish |    1526 | 2321   | 65.7%   |            +47.4 |                      +40.6 |
| welsh    |    1312 | 1915   | 68.5%   |            +53.4 |                      +45.0 |
| irish    |   17565 | 34041  | 51.6%   |            +41.6 |                      +33.9 |
| breton   |     221 | 1208   | 18.3%   |              new |                      +16.1 |

What landed in the lift: the bundle's morpheme inventory grew 4.5×,
so far more place-name fragments have a registered match. The
inflection-link path in particular surfaces forms like Welsh
`dyddiau` / `siroedd` / `cantorion` (plural) and OF `cisoires`
(feminine plural) that pre-this-epic were unmatched. Forms-mining
expanded the variants pool for runtime sampling under
``--spelling-variety``, but doesn't directly affect decomposition
rate (the matcher decomposes against canonical forms; variants
sample at generation time).

Audit follow-ups remaining: wyrd-j43l (deploy bundle + Lambda +
SPA — operator-driven), wyrd-p8ve (matcher score-pruning fix),
wyrd-v8x0 (restore Llanfairpwll... post-fix). Every code-side
ticket on the gap-closing epic is closed; the bundle re-emit
captured here is the artifact the epic produced.

## How to record a new snapshot

After a bundle re-emit (`wyrd kenning lexicon export-meanings` →
commit), run `rebuild-proportions` for each culture and capture the
stderr line `culture=… perfect=… names=… saints=… total=…`:

```bash
for c in english scottish welsh irish breton; do
  .venv/bin/wyrd kenning rebuild-proportions "$c" \
    "wyrd/generators/kenning/data/${c}_place_names.json" \
    > /dev/null
done
```

Append a new `### <date> — <commit> (<context>)` section under
**Snapshots** with the resulting table. Don't rewrite older snapshots
— the trend across them is the point.
