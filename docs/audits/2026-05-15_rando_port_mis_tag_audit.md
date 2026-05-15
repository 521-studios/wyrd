# Rando-port mis-tag audit (wyrd-e784)

Generated 2026-05-15 during wyrd-eni4 epic closeout. Catalogs candidate
language mis-tags in `rando-port`-only etymons by cross-referencing against
scholar-attributed siblings in the same canonical form but a different
language.

## Methodology

Filter: etymons whose only L2 attribution is `rando-port` (excluding
`wiktionary-empirical` + `wiktionary` bulk ingests, which are not
operator-curated scholar sources), AND a same-canonical-form sibling
exists under a DIFFERENT language tag with ≥1 scholar source.

Query at `data/mining/recipes/README.md` reproduces the candidate list
against the live DB.

## Snapshot (2026-05-15)

- **1,203** rando-only etymons exist in the DB (cited only by rando-port +
  empirical Wiktionary bulk mining).
- **37 of those** have a same-form sibling under a different language tag
  with one or more SCHOLAR source citations — strongly suggesting the
  rando-port LLM mis-tagged the language.

## Applied this pass

Two cases the original wyrd-e784 ticket explicitly named, and which have
unambiguous scholar attribution under the corrected language. Pruned from
`rando-port.jsonl` via `lexicon prune-etymon`:

| ref pruned | scholar sibling | scholar source(s) | rationale |
|---|---|---|---|
| `celtic:micel` | `old-english:micel` | mawer_stenton_1927 | micel is well-attested OE "great/large" |
| `old-norse:draeg` | `old-english:draeg` | skeat_1901 | draeg is OE place-name element "drag-place" (towing place) |

## Deferred — needs domain review or new scholar entries

The remaining 35 candidates fall into three buckets:

### Bucket A: scholar sibling exists but has NO citations (orphan scholar row)

Pruning the rando entry would make the morpheme vanish from the bundle.
Need to add a real scholar citation first.

| rando ref | empty scholar sibling | from original ticket? |
|---|---|---|
| `celtic:hearg` + `old-norse:hearg` | `old-english:hearg` | yes |
| `old-english:llumon` | `welsh:llumon` | yes |
| `old-norse:mapel` | `middle-english:mapel` | yes |

### Bucket B: ticket called this an Irish mis-tag but no Irish row exists

| rando ref | ticket says correct lang | action |
|---|---|---|
| `old-norse:guth` | irish (Old Irish "guth" = "voice") | needs `old-irish:guth` entry created before pruning |

### Bucket C: scholar sibling exists with citations, but the call requires domain expertise

These are candidates where one direction (rando → scholar lang) is plausible
but the rando tag isn't obviously wrong — could be a real borrowing or
parallel form across language families. Listed for an operator with relevant
scholarship at hand to review case-by-case.

| rando ref | scholar candidate | scholar source(s) | notes |
|---|---|---|---|
| `old-english:ald` | `old-norse:ald` | lindkvist 1912 | both languages have "ald" / "eald"; rando call probably wrong but scholar attestation needs verification |
| `old-norse:beinn` | `celtic:beinn` | macbain 1922 | beinn is Gaelic; ON has its own beinn-form |
| `celtic:boc` | `old-english:boc` (4 sources), `old-norse:boc` | skeat 1901, bannister 1916, mawer 1920, roberts 1914, skeat 1911 | strong OE case |
| `old-english:by` | `old-norse:by` | ekwall 1922, harrison 1898, johnston 1915, lindkvist 1912, mawer 1920, mawer-stenton 1924, moorman 1910, smith 1928 (8 sources) | -by is the iconic ON village suffix |
| `old-english:col` | `modern-english:col` | mawer 1920 | OE col = "coal/charcoal" is legitimate; modern-english may be the wrong direction |
| `norman-french:denis` | `modern-english:denis` | morgan 1912 | personal name; both attestations may be valid |
| `celtic:dew` | `old-english:dew`, `modern-english:dew` | mawer 1920 | OE dew = "moisture"; rando call wrong |
| `celtic:dūn` | `old-english:dūn` | mawer 1920, skeat 1906 | dūn is Celtic-origin but borrowed into OE place-naming; ambiguous |
| `celtic:elri` | `old-norse:elri` | ekwall 1922 | elri is ON "alder"; rando call wrong |
| `celtic:gall` | `old-norse:gall` | johnston 1904, joyce 1898, joyce 1913 | both ON and Celtic "gall" exist (foreigner) |
| `old-english:gardin` | `norman-french:gardin` | mawer 1920 | gardin is NF; rando call wrong |
| `celtic:glind` | `old-english:glind` | mawer-stenton 1930 | OE glind = "enclosure" |
| `celtic:gof` | `modern-english:gof`, `germanic:gof` | mawer 1920 | gof = Cornish "smith"; Celtic may be right and scholar wrong |
| `old-english:grange` | `modern-english:grange` | johnston 1904 | grange came via NF from Latin; OE tag is wrong |
| `old-english:gres` | `old-norse:gres` | ekwall 1922 | gres is ON "grass" |
| `celtic:gro` | `old-norse:gro` | watson 1904 x2 | gró is ON "grew"; ambiguous |
| `old-english:hafri` | `old-norse:hafri` | ekwall 1922 | hafri is ON "oats" |
| `old-english:hals` | `old-norse:hals` | ekwall 1922 | hals is ON "neck of land" |
| `celtic:haut` | `norman-french:haut` | mawer 1920 | haut is NF "high"; rando call wrong |
| `old-norse:helm` | `old-english:helm` | mawer 1920 | helm = OE "helmet/covering" |
| `old-english:hross` | `old-norse:hross` | ekwall 1922, watson 1904 | hross is ON "horse" |
| `celtic:hrycg` | `old-english:hrycg` | mawer 1920, mawer-stenton 1925, mawer-stenton 1927, moorman 1910, roberts 1914 (5 sources) | hrycg is OE "ridge" — strong case |
| `celtic:long` | `old-english:long` | duignan 1902, ekwall 1922 | long is OE/Germanic |
| `old-english:mael` | `celtic:mael` | gillies 1906 | Old Irish "mael" = "chieftain" / "bald" |
| `norman-french:pont` | `celtic:pont` | morgan 1887, watson 1926 | pont is BOTH NF "bridge" AND Welsh; legitimate ambiguity |
| `old-english:ric` | `germanic:ric` | arbois 1890, ekwall 1928 | rīc is OE/Germanic |
| `norman-french:roche` | `latin:roche` | longnon 1920 v2 | roche is from Latin via NF; rando call may be the more useful one |
| `old-norse:star` | `old-english:star` | ekwall 1928 | star = OE "rush/sedge" |
| `celtic:stob` | `modern-english:stob` | mawer 1920 | stob is Scots from Gaelic; Celtic call may be right |
| `celtic:tor` | `old-norse:tor` | harrison 1898 | tor is Brythonic Celtic; ambiguous |
| `old-english:torr` | `celtic:torr` | gillies 1906, johnston 1904, watson 1904 (3 sources) | torr is Celtic, borrowed into OE; both forms legitimate |
| `norman-french:val` | `celtic:val` | gillies 1906 | val is NF "valley"; rando call may be right |
| `celtic:whin` | `modern-english:whin` | ekwall 1922 | whin is Scots/ME from ON hvin |
| `celtic:wig` | `old-english:wig` | mawer-stenton 1927 | wīc = OE "dwelling"; rando call wrong |

## How to extend this audit

1. Run the recipe at `data/mining/recipes/README.md` to refresh the candidate
   set against current DB state.
2. For each Bucket-A case, file a wiktextract / scholar ingest task to
   populate the empty scholar entry. Then re-audit.
3. For Bucket-B cases, decide whether to author a new scholar entry directly
   (CLI add-event-style commit) or wait for a relevant mining pass.
4. For Bucket-C cases, work case-by-case with reference to a place-name
   dictionary. The strongest cases (5+ scholar sources for the alternative
   language) are the next safest to prune; the borderline cases (single
   scholar source, or both languages plausible) should stay open until a
   subject-matter expert weighs in.

## Why this isn't bulk-fixable

The mis-tag retags via `prune-etymon` are append-only `_op: remove` events
in `data/mining/rando-port.jsonl`. Each prune loses whatever gloss /
modifier_type data the rando entry carried — fine when the scholar sibling
has equally rich coverage, lossy when it doesn't.

Bridge-language (the closest CLI cousin) only works for `celtic → specific
Celtic-family` canonicalization via `merged_into_id`; it doesn't help with
the OE/ON/NF cases that dominate this audit.

The right shape for a future bulk pass would be: extract the rando entry's
gloss list before pruning, then patch them onto the surviving scholar entry
via a `_op: patch` event in the scholar source's JSONL. This isn't in scope
for wyrd-e784 (per 2026-05-15 user direction; "just the 2 safe ticket
cases + audit doc"). File as a follow-up if the audit identifies a
gloss-transfer pattern worth productizing.
