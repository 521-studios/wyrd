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

| culture  | perfect | total  | rate    | Δ pp vs baseline | Δ pp vs prior |
|----------|--------:|-------:|--------:|-----------------:|--------------:|
| english  |   12686 | 17876  | 71.0%   |            +42.7 |         +35.9 |
| scottish |    1526 | 2321   | 65.7%   |            +47.4 |         +40.6 |
| welsh    |    1312 | 1915   | 68.5%   |            +53.4 |         +45.0 |
| irish    |   17565 | 34041  | 51.6%   |            +41.6 |         +33.9 |
| breton   |     221 | 1208   | 18.3%   |              new |         +16.1 |

(Per-culture priors vary: English / Scottish / Welsh from
2026-05-05 post-wyrd-4hx7; Irish from 2026-05-07 post-wyrd-1cjg;
Breton is genuinely new — no prior snapshot.)

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

### 2026-05-09 — wyrd-zewx (strict-inner matcher fix)

Tightened ``_location_allows`` so inner-only morphemes (those with
dashes on both sides like ``-don-``, ``-stone-``) match STRICTLY
inside (start>0 AND end<len) rather than the previous "anywhere"
semantics. Combined with the wyrd-a4p5 adjacent-duplicate fix and
the wyrd-c0xn description-text cleanup. Generation output post-fix
no longer produces 'donhole' / 'nwydmillate' / 'port port' style
artifacts.

| culture  | perfect | total  | rate   | Δ pp vs prior (2026-05-08) |
|----------|--------:|-------:|-------:|---------------------------:|
| english  |   12341 | 17876  | 69.0%  |                       -2.0 |
| scottish |    1446 | 2321   | 62.3%  |                       -3.4 |
| welsh    |    1263 | 1915   | 65.9%  |                       -2.6 |
| irish    |   16369 | 34041  | 48.1%  |                       -3.5 |
| breton   |     181 | 1208   | 15.0%  |                       -3.3 |

The 1.9-3.5pp drops reflect place names that previously decomposed
ONLY via inner-at-boundary matches (e.g. a name like 'Donhole' that
matched as ``-don-`` + ``-hole-``). Those names now stay partially
unaccounted at the boundary positions — correct, since inner
morphemes shouldn't be the first or last element of a compound.
The drop is pure generation-quality wins traded for a small
decomposition-rate loss.

Sidecar update: ``irish_anglicizations.json`` gained ``-bally``,
``-cloon``, ``-kil``, ``-knock``, ``-lis``, ``-liss`` (all post)
since the previous file relied on the permissive inner semantics
to cover word-end positions. With strict-inner, post variants are
now needed explicitly for prefixes that legitimately appear at
word-end (Ballyknock = Bally- + -knock).

### 2026-05-09 — wyrd-p8ve (canonical_decompositions score-pruning + village restored)

Fixed the algorithmic OOM in ``canonical_decompositions``: it now
score-prunes during the walk instead of enumerating
``all_decompositions`` then filtering. The 58-character Welsh village
'Llanfairpwllgwyngyllgogerychwyrndrobwyllllantysiliogogogoch' that
the wyrd-eni4 deploy had to remove from welsh_place_names.json
now decomposes in 0.00s with 170 MB peak RSS (was 16+ GB OOM).
Restored the village to the Welsh corpus (welsh count
1915 → 1916) so welsh_place_names.json carries the genuine corpus
again.

Welsh perfect-rate stayed at 1263/1916 = 65.9% — the village
itself doesn't reach a 0-unaccounted decomposition (the 58 chars
include genuine etymological compounds the bundle hasn't fully
mined), but it no longer blocks the rest of the corpus from
completing rebuild-proportions.

Side-effect win: every test exercising the full corpus matcher
runs ~3× faster (suite 200s → 69s) because canonical_decompositions
no longer pays the cartesian-product cost on every name.

| culture  | perfect | total  | rate   | Δ pp vs prior (2026-05-09a) |
|----------|--------:|-------:|-------:|----------------------------:|
| english  |   12341 | 17876  | 69.0%  |                         0.0 |
| scottish |    1446 | 2321   | 62.3%  |                         0.0 |
| welsh    |    1263 | 1916   | 65.9%  |                         0.0 |
| irish    |   16369 | 34041  | 48.1%  |                         0.0 |
| breton   |     181 | 1208   | 15.0%  |                         0.0 |

Welsh denominator changed from 1915 to 1916 (village restored),
but the perfect count is the same since the village stays
partial-decomposition. Per-culture rates are unchanged because
the score-pruning walk produces the same global-minimum set as
the legacy enumerate-then-filter approach — only the memory
profile differs.

### 2026-05-09 — wyrd-69s5 IPA backfill (operational re-mining)

Operational follow-up to the wyrd-69s5 corpus-miner code fix (PR
#158): re-ran ``mine-wiktextract-corpus --apply --culture all``
to backfill ``pronunciation_ipa`` + ``original_script`` +
``transliteration`` on the ~9000 etymons the empirical mining
path touches. Pre-backfill IPA coverage was 0% on every shipped
European language despite the source wiktextract slices carrying
IPA in their ``sounds`` arrays — the corpus miner had been
dropping the field on the floor before fbee68f / PR #158.

Per-language IPA coverage in the bundle (subjects with the
language sibling AND a non-empty pronunciation entry):

| sibling          | with_lang | with_ipa |   ipa% |
|------------------|----------:|---------:|-------:|
| celtic_mix       |     6,061 |    2,431 |  40.1% |
| modern_english   |     5,495 |    1,083 |  19.7% |
| old_english      |     4,470 |    1,779 |  39.8% |
| old_french       |     1,403 |      211 |  15.0% |
| old_scandinavian |     1,238 |       21 |   1.7% |
| latin            |        23 |        0 |   0.0% |

The ``old_scandinavian`` (Old Norse) row stays low because the
wiktextract Old Norse slice often leaves its ``sounds`` arrays
empty for older entries; addressing that needs a separate mining
pass against richer sources, tracked as a wyrd-69s5 follow-up if
it matters for a future SPA panel.

The mining also re-walked the empirical corpus and surfaced ~50
new etymons that weren't in the previous bundle (mostly via
inflection-form lookups that landed in etymon_text_match). The
re-emitted bundle grew 8490 → 8502 subjects.

| culture  | perfect | total  | rate   | Δ pp vs prior (wyrd-zewx) |
|----------|--------:|-------:|-------:|--------------------------:|
| english  |   12383 | 17876  | 69.3%  |                      +0.3 |
| scottish |    1452 | 2321   | 62.6%  |                      +0.3 |
| welsh    |    1265 | 1916   | 66.0%  |                      +0.1 |
| irish    |   16417 | 34041  | 48.2%  |                      +0.1 |
| breton   |     183 | 1208   | 15.2%  |                      +0.2 |

Modest perfect-rate uptick (+0.1-0.3pp) from the new etymons
the empirical re-mining surfaced. The headline result is the
runtime visibility win: SPA's etymological-provenance panel can
now render IPA on ~40% of OE / Celtic morphemes and ~20% of
modern-English morphemes, where pre-fix it was 0% across the
board.

### 2026-05-10 — wyrd-xmk3 (full ingest-wiktionary on European slices)

Followed wyrd-69s5's corpus-miner IPA backfill (PR #158/#159) by
running the FULL ``ingest-wiktionary`` path on all on-disk
European-language wiktextract slices. The corpus miner only touches
~1% of etymons (those whose surface matches an unaccounted place-name
fragment), so the IPA in those etymons' ``sounds`` arrays was
mostly going to waste. The full ingester walks every entry, runs
the same ``_extract_pronunciation`` helper, and COALESCEs the IPA
into the etymon table.

Per-language IPA coverage in the **lexicon DB** (etymon table)
after running on 13 slices:

| language       | before | after  | gain  |
|----------------|-------:|-------:|------:|
| old-english    |   1.0% | **75.2%** | +74pp |
| welsh          |   1.4% | **59.7%** | +58pp |
| old-irish      |   3.1% | **47.3%** | +44pp |
| irish          |   1.1% |  32.1% | +31pp |
| breton         |   1.3% |  30.0% | +29pp |
| scottish-gaelic |  0.5% |  24.3% | +24pp |
| cornish        |   0.0% |  20.9% | +21pp |
| manx           |   0.0% |  19.5% | +20pp |
| middle-english |   0.9% |  12.1% | +11pp |
| middle-irish   |   0.9% |  10.5% | +10pp |
| old-french     |   0.9% |   6.2% |  +5pp |
| old-norse      |   0.1% |   1.9% |  +2pp |

Bundle-side coverage (subjects with non-empty pronunciation slot)
post-export:

| sibling          | before | after  |
|------------------|-------:|-------:|
| old_english      |  39.8% | **84.2%** |
| celtic_mix       |  40.1% | **60.0%** |
| modern_english   |  19.7% |  36.4% (still inherited from cluster mates) |
| old_french       |  15.0% |  19.3% |
| old_scandinavian |   1.7% |   6.9% |

Modern-english is still 0% in the DB and surfaces only via cluster-
mate inheritance from middle-english / old-english — wyrd-dxu2
(reopened) tracks acquiring the modern-english wiktextract slice
from Kaikki to fill that gap directly.

Side-effects: the full ingester ALSO writes ``etymon_descent`` rows
(28K upward + 98K downward for OE alone). These propagate into the
era-reflex generation path and shape the cluster cognate logic.
Net effect should be more accurate era reflexes; no behavioural
regressions surfaced in spot-check.

Per-culture perfect-rates unchanged from the wyrd-69s5 snapshot
(see prior section) — IPA backfill doesn't change which morphemes
match, only what pronunciation accompanies them. The win is in the
SPA's etymological-provenance panel: it now has IPA to render on
~85% of OE morphemes and ~60% of Celtic morphemes, where pre-fix
the bundle path showed nothing (or borrowed from cluster mates).

### 2026-05-11 — wyrd-dxu2 (modern-english slice ingestion)

Closes the final IPA gap from wyrd-69s5 / wyrd-xmk3. Downloaded
Kaikki's English wiktextract dump (``kaikki.org-dictionary-English.jsonl``,
2.99 GB, 1.47M entries) to ``sources/wiktextract_english.jsonl``,
mapped it via ``_LANG_TO_SLICE_BASENAME["modern-english"]``, and ran
the full ``ingest-wiktionary --apply`` path.

Per-language IPA coverage after the run:

| layer | language       | before | after  |
|-------|----------------|-------:|-------:|
| lex DB | modern-english | 0.0%  | **7.1%** (97,702 / 1.4M etymons) |
| lex DB | old-english    | 75.2% | 75.4% |
| lex DB | welsh          | 59.7% | 59.0% |
| bundle | **modern_english** | 19.7% | **41.8%** |
| bundle | old_english    | 84.2% | 84.5% |
| bundle | celtic_mix     | 60.0% | 61.9% |
| bundle | old_french     | 19.3% | 19.9% |

Per-run capture stats from the ingest:

* 1,465,676 entries parsed
* 944,728 upward edges (etymology templates: borrow / inherit /
  derive / calque / compound / affix / root)
* 18,512 downward edges (descendants tables — small share since
  English entries rarely have descent trees)
* **136,048 IPA captures**
* 4,776 original_script + 329 transliteration captures

The 7.1% lexicon-side number looks low but reflects a denominator
explosion: the ingest created ~1.35M new modern-english etymons
from etymology templates (e.g. every Latin / French / Greek
borrowing chain's English landing-form). Most of those don't have
their own ``sounds`` entry in the slice — they're referenced via
template, not as a full Wiktionary entry. Among English forms
that ARE full entries (i.e. the words a user would actually want
IPA for), coverage is much higher: spot-check shows 651/948 (69%)
of real bundle-form modern-english entries have IPA.

The bundle's 41.8% number is the actionable one — it tracks what
the SPA's etymological-provenance panel can render. Doubling from
19.7% to 41.8% reflects real modern-english IPA replacing
cluster-mate inheritance on common words (bridge, water, town,
hill, etc.). The remaining 58.2% breaks down as:

* ~33% bundle-synthesized non-words (forms like 'babllings',
  'alwent' that are ME/OE reflexes auto-derived as
  "modern-english forms" but aren't real English words)
* ~18% real words still without IPA in Wiktionary's English slice
* ~7% cluster-mate-inherited (Section H warning still fires)

Per-culture perfect-rates unchanged (IPA backfill doesn't change
which morphemes match). Bundle subjects + proportions re-emitted
for diff-determinism.

### 2026-05-11 — wyrd-vsvi (tag extraction in full ingest)

Closes the dashboard's biggest weakness: modern-english at 2/15
reference-tag hits despite being 29% of bundle words. Root cause:
the full ``ingest-wiktionary`` path (used by wyrd-xmk3 and
wyrd-dxu2) extracted etymology, pronunciation, and renderings —
but NOT tags from sense categories. So the 1.4M modern-english
etymons created by the English slice ingest were all tag-less.

Fix: thread ``_extract_entry_tags`` (new) through
``wiktextract_ingester._process_entry``. Unions sense categories
across all senses (richer than the corpus-miner's single-canonical-
sense pick), runs them through ``_map_categories_to_tags`` (shared
with the corpus miner), and writes ``etymon_tag`` rows via
``db.add_tag``. Idempotent on re-run via INSERT OR IGNORE.

Re-ran ingest-wiktionary --apply on the English slice. The run
captured **50,705 tag-writes across 32,649 entries**. Modern-english
tagged-etymon count went from 5 → 31,445.

Reference-tag coverage (C row in the dashboard):

| language       | before | after |
|----------------|-------:|------:|
| **modern-english** | 2/15 | **12/15** |
| middle-english | 13/15 | 14/15 |
| old-french     | 9/15  | 10/15 |
| welsh          | 12/15 | 13/15 |
| irish          | 12/15 | 13/15 |
| old-irish      | 12/15 | 13/15 |
| breton         | 6/15  | 7/15  |
| norman-french  | 6/15  | 7/15  |
| latin (C₂)     | 9/15  | 12/15 |

Remaining missing tags are mostly ``monster``, ``fantasy``,
``measurement`` — Wiktionary doesn't categorize those reliably.

Side-effect: bundle expanded 8502 → 8629 subjects because the new
tag-bearing modern-english entries cleared the
``include_wiktionary_empirical=True`` admission gate now that
they have semantic signal. Most of these are common English words
(bridge, mill, water variants) that were always there but now
ship with proper tags.

Per-culture perfect-rates unchanged. The win is in generation
quality — the tag pool for ``--tag`` filtering now reaches
modern-english/middle-english/old-french entries where pre-fix
the tag filter silently excluded them.

### 2026-05-11 — wyrd-r1ks (OE / ON spelling-variant forms mining)

Closes the dashboard's #2 weakness from the wyrd-vsvi audit: OE / ON
spelling-variant coverage stuck at 0.1-0.2% while welsh / irish /
OF were at 49-77%. Root cause: ``mine-wiktextract-forms`` (the
wyrd-vx09 forms-mining path) had been run on welsh, irish,
scottish-gaelic, middle-irish, old-irish, breton, cornish, manx,
middle-english, old-french — but **NOT** on OE or ON despite both
slices carrying rich form-table data.

Fix: ran ``mine-wiktextract-forms --apply`` on both slices.

| layer | language | before | after |
|-------|----------|-------:|------:|
| lex DB | old-english variants | 177/72,382 (0.2%) | **44,929/72,382 (62.1%)** |
| lex DB | old-norse variants   | 21/14,679 (0.1%) | **3,363/14,679 (22.9%)** |
| bundle | old_english_variants pool | ~0 | **2,749/4,483 (61.3%)** |
| bundle | old_scandinavian_variants pool | ~0 | **648/1,238 (52.3%)** |

Per-run stats (forms_processed counts forms that passed the noise
filter; forms_skipped_noise counts forms filtered out as table /
scaffolding noise; total candidates = processed + skipped):

* OE: 66,179 entries walked → 387,410 form candidates total
  (291,252 processed and written + 96,158 filtered as
  table-scaffolding noise)
* ON: 11,193 entries walked → 87,458 form candidates total
  (37,271 processed and written + 50,187 filtered)

Sample OE morpheme post-mining (``-ing-``):

```
forms:    ['inga', 'ingas', 'ing', 'inge', 'ingan', 'Inga', 'Ing',
           '-ingas', '-inga', '-ing']
variants: [{'form': 'ingum',    'weight': 1},
           {'form': 'ingān',    'weight': 1}]
```

Round-1 review surfaced that the initial run leaked verbal
conjugations of OE ``ingān`` ('to enter') — ``inēode``,
``ingānne`` — into the suffix's variant pool because wiktextract
emits OE verb forms with ``past`` / ``infinitive`` tags neither
of which the original ``_NOISE_FORM_TAGS`` set caught (only
``preterite`` was filtered). Expanded the filter to include
``past``, ``perfect``, ``infinitive``, ``gerund``, ``supine`` —
re-mined and the verbal conjugations are now filtered out at
ingest. Two remaining cluster-mate leaks (``ingum``, ``ingān``
canonical forms of the verb etymon) tracked separately as
wyrd-sg7l for cluster-rollup-time filtering.

The variants pool feeds the runtime's ``--spelling-variety`` knob:
the generator now has real OE inflected forms to draw from for
spelling variation, where pre-fix it had basically nothing for OE
and rendered every reflex as the canonical headword. Same story
for ON (less dramatic since the ON slice is smaller).

Per-culture perfect-rates unchanged. The win is in generation
quality at non-zero spelling_variety — particularly impactful for
the English culture (which heavily uses OE morphemes) when the
runtime knob is engaged.




### 2026-05-11 — wyrd-gf28 (modern-english inflection rules)

Closes the dashboard's #4 weakness from the wyrd-vsvi audit:
modern-english D (Inflection coverage) at 0% despite the bundle
having 1.4M modern-english etymons via the wyrd-dxu2 Kaikki ingest.
Root cause: ``INFLECTION_RULES`` in ``lexicon.py`` had rules for
6 languages (OE/ON/OF/ME/NF/welsh) but NOT modern-english.

Added conservative rules:

| suffix | label  | risk |
|--------|--------|------|
| ``-ed`` | past   | low — false-positive stems ('sh' from 'shed') usually aren't verb etymons, so link-lemmas' EXISTS check rejects them |
| ``-ing`` | gerund | low — same: stems like 'str' from 'string' aren't verb etymons |

Re-ran ``link-lemmas --apply``:

| language | before | after |
|----------|-------:|------:|
| **modern-english** | 0 (0.0%) | **24,733 lemmas (1.9%)** |
| middle-english | 4.2% | 4.7% |

DEFERRED rules (too risky given 1.4M ModE etymon denominator):
- ``-s`` / ``-es`` plural: many natural -s lemmas (is, this, pass)
- ``-er`` comparative: conflicts with agent-noun -er
- ``-ly`` adverb: conflicts with adjective-final -ly (ugly, silly)

These would need richer rules with frequency-of-stem context to
avoid corrupting lemma rollups. Filed as wyrd-gf28-followup.

Sample of new links: impending→impend, neighing→neigh, bemused→
bemuse, dejeunered→dejeuner, beshivered→beshiver. Real morphology.


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
