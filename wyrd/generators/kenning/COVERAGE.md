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
