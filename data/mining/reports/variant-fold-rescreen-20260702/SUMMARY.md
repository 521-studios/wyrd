# variant-fold rescreen — 2026-07-02 (wyrd-21p8)

Re-screen of all **487** `llm-variant-fold-v1` fold rows in `data/mining/_collapses.jsonl`, judged individually by the loop agent (Opus 4.8) per the wyrd-21p8 owner directive — no external LLM, no deterministic-only shortcut. Revert-when-uncertain (D46: a wrong revert is harmless duplication; a wrong keep leaves a distinct morpheme tombstoned).

## Verdict counts

- **KEEP: 465** / 487
- **REVERT: 22** / 487 (uncertain-treated-as-revert included)

Reverts are appended to `_collapses.jsonl` as `into: ""` rows (method `llm-variant-fold-rescreen-v1`, last-write-wins per ref); the live-DB `merged_into_id` heals on the next **full rebuild-from-jsonl (fresh DB)** (derived — the L2 ledger is the fix). Note: an *incremental* enrichment re-apply does NOT un-tombstone an already-folded row — `apply_collapses` no-ops on an `into: ""` row (`empty_into_skipped`) before touching `merged_into_id`; only a fresh rebuild re-derives it from the corrected ledger. Existing rows are never edited (append-only, D50).

## Method

For each `(ref, into)` pair, both etymons' glosses / language / cognate cluster were pulled from the live `~/.wyrd/lexicon.db` (READ-ONLY). Triage split the set into 257 clear-keep (identical meaning-bearing gloss or shared cognate cluster) and 230 that needed close judgment (the exact-match triage over-flags because legit folds are often worded differently); all 230 were judged individually.

**REVERT when** the fold rested only on: entry-type framing glosses ('a feminine personal name' etc., via `_is_placeholder_gloss`) + distinct names; a lemma whose *dominant* sense differs from the folded form (secondary-sense/homograph conflation); etymologically distinct words on a surface/broad-category overlap; or an unverifiable/garbled gloss. **KEEP when** a meaning-bearing sense is shared as the *primary* sense, the surface is a clear spelling/OCR/inflectional variant, or it is the same specific name/cognate cluster.

## The 22 reverts

| ref | into | rationale |
|-----|------|-----------|
| `old-english:Ceatwa` | `old-english:Ceatta` | framing-only glosses ('masculine personal name') + surface-similar but distinct OE names; no meaning signal (Ede/Eve class) |
| `old-english:erg` | `old-english:beorg` | etymologically distinct: erg = ON-derived hill-PASTURE/shieling vs beorg = OE hill/barrow/tumulus; folded on 'hill' surface overlap (medium conf) |
| `old-english:cnocc` | `old-english:cocc` | cocc's dominant sense is 'cock (rooster)'; cnocc='hillock' matches only cocc's secondary 'heap/hillock' — distinct-morpheme conflation risk (D46) |
| `old-english:ellern` | `old-english:ellen` | ellen's dominant OE sense is 'strength/courage'; ellern='elder-tree' matches only ellen's secondary tree sense — homograph conflation risk (D46) |
| `old-english:spald` | `old-english:sceald` | distinct meaning: spald = ditch/trench/moat vs sceald = 'shallow'; surface-similar fold, no shared sense (medium conf) |
| `old-english:scot` | `old-english:sceot` | ref gloss EMPTY (unverifiable) + scot has a distinct common 'tax/payment' homograph vs sceot='shooting/projection' (D46) |
| `old-english:spearr` | `old-english:spere` | spearr = spar/rafter/beam vs spere = spear (weapon); distinct lexemes, only a thin 'shaft' overlap (D46) |
| `old-english:warde` | `old-english:weard` | warde = territorial 'district/ward/area' vs weard = 'guardian/watch/warden'; related root but semantically distinct senses conflated (D46) |
| `old-norse:ask` | `old-norse:aska` | ask = ash-TREE (acc. of askr) vs aska = ashes/burnt residue; distinct referents folded on surface |
| `old-norse:hain` | `old-norse:hagi` | hain = 'hedge / to enclose with a fence' (verb, likely hegna) vs hagi = pasture/grazing-enclosure (noun); POS+sense mismatch, medium conf (D46) |
| `old-norse:hjalla` | `old-norse:hjallr` | hjalli = ledge/terrace/shelf vs hjallr = a hut/shed (primary); only secondary ledge overlap (D46) |
| `middle-english:Ede` | `middle-english:Eve` | CONFIRMED WRONG (seed): Ede = medieval pet form of Edith/OE Eadgifu (framing-only gloss) vs Eve = Hebrew Eva (biblical); etymologically distinct names folded on framing + surface |
| `old-english:clacc` | `old-english:canc` | clacc and canc are distinct OE hill-words (far surface: clacc vs canc); a synonym-merge, not a spelling variant |
| `old-english:how` | `old-english:hol` | how is ambiguous — 'a hollow, a low hill'; likely haugr 'hill/mound' (ON), the OPPOSITE of hol 'hollow' it folds into (D46) |
| `old-english:sparr` | `old-english:spearca` | sparr = spar/stick/pole vs spearca = spark/brushwood (messy 'uncertain' gloss); distinct words, divergent surface |
| `old-english:weorj)` | `old-english:wer` | garbled OCR ref; if weorþ ('worth', settlement-enclosure) it is distinct from wer ('weir', river fishing-dam) — unverifiable surface (D46) |
| `old-english:clenc` | `old-english:canc` | 'possibly a hill' (uncertain gloss) folded into canc; distinct surface (clenc/canc), medium conf (D46) |
| `old-english:cylfe` | `old-english:clif` | 'probably a prow/eminence' (uncertain gloss) vs clif = cliff; distinct surface, medium conf (D46) |
| `old-english:coppede` | `old-english:cropp` | coppede = 'peaked' (adj. from copp = summit/top) vs cropp = crop/sprout/craw; distinct words, medium conf (D46) |
| `old-norse:eski` | `old-norse:aska` | eski = ash-tree place / ash-wood vs aska = ashes/burnt residue; distinct referents (same class as ask->aska) |
| `old-english:byr` | `old-english:byre` | byr = birch (birce) vs byre = byre/cow-shelter; distinct OE etymologies folded on a contaminated 'birch tree' gloss + near-surface (D46) |
| `old-english:helde` | `old-english:hlid` | helde = hielde (slope/incline) vs hlid = lid/swing-gate (+slope); distinct words, divergent surface, medium conf (D46) |

Full per-row verdicts (all 487, keep + revert) in `verdicts.jsonl`.

