# Kenning — architecture

This generator started as a port of Rando's morpheme-statistics name maker
and has grown into a small etymology pipeline: scrape scholarly place-name
dictionaries, normalize them, store them, and use the result to drive
generation. The runtime hasn't changed shape — it still reads
`data/meanings.json` — but the authoring pipeline behind it is now richer.

## Data model

The lexicon (SQLite, `data/lexicon.db`) is normalized so multiple sources
can disagree without losing each other's claims. Schema in `data/lexicon.sql`.

```
                ┌────────────┐
                │  source    │   bibliographic record (Skeat 1901, etc.)
                └─────┬──────┘
                      │
        ┌─────────────┼──────────────────┐
        │             │                  │
        ▼             ▼                  ▼
  ┌────────────┐  ┌──────────────────┐  ┌──────────────────────┐
  │ etymon_    │  │ toponym_         │  │ toponym_etymology_   │
  │ citation   │  │ etymology        │  │ element              │
  └─────┬──────┘  └─────────┬────────┘  │  (ordinal + inflect) │
        │                   │           └──────────┬───────────┘
        ▼                   ▼                      │
  ┌────────────┐  ┌──────────────────┐             │
  │   etymon   │←─┤   toponym        │             │
  │  (form,    │  │  (modern_name,   │             │
  │  language) │  │   region)        │             │
  └─────┬──────┘  └──────────────────┘             │
        │                                          │
        │           ┌─────────────────┐            │
        ├──────────►│ etymon_gloss    │            │
        ├──────────►│ etymon_tag      │            │
        │           └─────────────────┘            │
        ▼                                          │
  ┌────────────────┐  ┌─────────────┐              │
  │ reflex_etymon  │──│   reflex    │              │
  │  (M:N)         │  │ (modern     │              │
  └────────────────┘  │  surface)   │              │
                      └─────────────┘              │
                            ▲                      │
                            └──────────────────────┘
                            (etymon_id ties through)
```

Two views drive the consensus model:

- `etymon_consensus` — how many distinct sources cite each etymon. Drives
  promotion confidence ("3+ witnesses, no disagreement → live inventory").
- `toponym_breakdown_signature` — distinct ordered etymon-id sequences per
  toponym. >1 signature = scholars disagree on the breakdown.

## Authoring flow

```
   sources/<book>.txt   (public-domain OCR text)
            │
            ▼
   ┌──────────────────────┐
   │ parser               │   skeat_parser.py        (TOC + suffix sections)
   │  → ParsedEntry list  │   dictionary_parser.py   (alphabetical headwords)
   └──────────┬───────────┘
              │  (auto-detect by yield)
              ▼
   ┌──────────────────────┐
   │ LLM extractor        │   llm_extractor.py       (Ollama, local, free)
   │  + form-in-body      │   gemini_extractor.py    (paid, more accurate)
   │  validation guard    │   shared validate_response, USER_TEMPLATE,
   └──────────┬───────────┘   SYSTEM_PROMPT
              │
              ▼
   ┌──────────────────────┐
   │ ingester             │   lexicon.ingest_parsed_entries
   │  → lexicon.db        │
   └──────────────────────┘
```

Key invariants:

- **Every form a model emits must appear in the source paragraph.**
  `validate_response` substring-matches with OCR-tolerant normalization;
  forms that don't match get the row rejected, not ingested. This is the
  load-bearing anti-hallucination guard. Don't loosen it without thinking.
- **Glosses are paraphrased, not quoted.** They're informational, not
  validated. The form is the truth claim; the gloss is annotation.
- **Source attribution is mandatory.** No etymon goes in without an
  `etymon_citation` row pointing at a `source` row.

## Modules

| Module | Purpose |
|---|---|
| `lexicon.py` | DB schema init, ingest, OCR-variant clustering. The data layer. |
| `skeat_parser.py` | Skeat-style books: TOC + "§ N. The suffix -X" body sections. Cambridgeshire/Bedfordshire/Suffolk. |
| `dictionary_parser.py` | Alphabetical-headword books (Mawer, Ekwall, Watson, Johnston, Joyce, Morgan). Auto-detected fallback. |
| `llm_extractor.py` | Ollama client + shared SYSTEM_PROMPT, USER_TEMPLATE, RESPONSE_SCHEMA, `validate_response`. |
| `gemini_extractor.py` | Gemini-API parallel implementation with the same `extract_one` shape. Slightly stronger on hedge-recognition and OE-form OCR garble; pay-per-call. |
| `fantasy_pipeline.py` | wyrd-ami: fantasy-name etymology research (Harpy → ancient-greek ἅρπυια, Djinni → arabic jinn). Resolves creature names against the etymon corpus via descent_walking_lookup pre-filter + Gemini Flash full-research fallback. Output rows in `fantasy_morpheme`. See OVERVIEW.md "Sibling pipeline" + INGESTION.md "Mining the wyrd-ami fantasy-name corpus". |
| `pfsrd2_monster_extractor.py` | Walks the Pathfinder 2 SRD bestiary JSON corpus (`pfsrd2-data`) and emits `{name, description}` JSONL records for `mine-fantasy-name --batch`. Emits both family root + single-word variants. |
| `cli.py` | All `wyrd kenning lexicon …` commands. |

Generator runtime (`__init__.py`, `name.py`, `word.py`, `meaning.py`,
`proportions.py`) reads the bundled `meanings.json`, which is now exported
from the lexicon DB by `wyrd kenning lexicon export-meanings` (closes
the D1 loop). The runtime keeps the historical "load JSON, sample" shape
— the lexicon DB is invisible to the runtime.

## Cultures

The generator currently supports five register cultures:

| culture | corpus | notes |
|---|---|---|
| `english` | EPNS county dicts (Mawer, Ekwall, Skeat) | Densest morpheme set; well-mined. |
| `scottish` | Watson, Macbain, Johnston | Celtic + Old Norse blend. |
| `welsh` | Morgan, Johnston | Brythonic Celtic; dense Welsh morphology. |
| `irish` | Joyce vols 1–3 | Goidelic Celtic; large place-name list. |
| `breton` | Wikidata commune list (1214 names); morpheme corpus pending | Witcher 'remote cultured empire' register: French-Celtic with Norman-French overlay. Decomp rate currently 1.7% (lexicon doesn't yet have Plou-/Ker-/Tre- morphemes); see wyrd-fmg. |

## Generation knobs

The runtime exposes four orthogonal knobs alongside the existing
`--culture` and `--tag` filters. All default to off / 0.0 — historical
seed-stable behavior preserved bit-for-bit.

| knob | spec | description |
|---|---|---|
| `--novelty` | D17 | Per-bucket mixture between empirical-frequency sampling and a uniform marginal. At 1.0, every in-bucket morpheme is equally likely — plausible-but-unattested combinations become possible without abandoning the corpus. The D17 β-term (tag-class-prior) shipped as `--cohesion` (see below); composed with novelty, the runtime supports both refining novel combinations toward attested patterns AND blending toward the uniform marginal. |
| `--cohesion` | D17 β | Bias each slot's pick toward usages whose tags co-occur with previously-picked slots' tags in the empirical corpus. At 0 (default) every slot samples independently from its marginal — bit-stable. At 1, slot-N's empirical weights are scaled by the normalized class-conditional likelihood given the union of prior slots' tags. Implements D17's tag-class-prior term in multiplicative-then-blend form (`weights = empirical * boost; result = (1-novelty)·weights + novelty·uniform`). Composes orthogonally with `--novelty` so a GM can dial 'attested-pair fidelity' and 'novelty' independently. No-op when the bundle carries no tag-cooccurrence data. |
| `--spelling-variety` | D18 | Per-morpheme probability of substituting an attested archaic spelling variant for the canonical reflex. Pool comes from `etymon_text_match.matched_form` rows surviving the post-mining chain + LLM disambiguator. 416 morphemes have non-empty pools today. |
| `--inflection-density` | D8 | Per-morpheme probability of substituting an inflected form (genitive/dative/plural) for the lemma. Bundle has 168 inflected etymons across 9 grammatical-case labels. When both this and `--spelling-variety` would fire on the same morpheme, inflection wins. |
| `--mood` | D6 | Stylistic-mood preset (repeatable). Five entries today — `grim` (death/military/monster/undead/magic), `harsh` (phonological stop-final/cluster-heavy bias), `pastoral` (plant/animal/water/agriculture/tree/bird), `devotional` (saint/religious), `mortuary` (death/undead, narrower than grim). `harsh:0.5` graduates the phonological skew via colon-suffix. Multiple `--mood` flags compose by tag-union and max-harshness. Vocabulary lives in `__init__.MOODS`. See `DECISIONS.md` D6 for the spec-named-vs-extant-tags backstory. |

## Bundle schema (meanings.json)

The runtime reads a JSON file whose top-level shape is a list of subjects:

```json
{
  "meaning": ["Bridge"],
  "modifier_tags": ["water", "architecture"],
  "modifier_type": "Topographical",
  "words": [
    {
      "modern_usage": "Bridg-",
      "old_english": ["brycg"],
      "old_english_variants": [{"form": "brigge", "weight": 8}],
      "old_english_inflections": [{"form": "brycgan", "inflection": "weak_oblique"}]
    }
  ]
}
```

Per D26, the per-language metadata fields (`<lang>_variants`,
`<lang>_inflections`) are sibling to the canonical language array, so
legacy code that ignores unknown fields keeps loading. The runtime's
`load_meanings` strips the `_variants` / `_inflections` suffixes and
routes them into `Meaning.variants` / `Meaning.inflections` attributes.

## CLI cheatsheet

```bash
# Build the authoring DB from meanings.json, attributed to 'rando-port'.
wyrd kenning lexicon build

# Mine a public-domain etymology book via local Ollama (default).
wyrd kenning lexicon mine-llm sources/skeat_1901_cambridgeshire.txt

# Mine via Gemini for higher quality (uses GEMINI_API_KEY env var).
wyrd kenning lexicon mine-llm sources/mawer_1920_northumberland_durham.txt --provider gemini

# Cluster OCR-variant etymons (Hædan / Hcsdan / Haedan → one row).
wyrd kenning lexicon normalize-ocr --apply

# Export the lexicon as a runtime-shaped meanings.json (D1 / D26).
wyrd kenning lexicon export-meanings --output wyrd/generators/kenning/data/meanings.json

# Read-only summary of what's in the lexicon.
wyrd kenning lexicon report

# Generate names with the new knobs.
wyrd kenning generate english --novelty 0.5 --spelling-variety 0.3 --inflection-density 0.2
wyrd kenning generate english --mood grim --mood harsh --seed 42  # menacing English-Saxon
wyrd kenning generate breton --seed 42

# wyrd-ami fantasy-name research (creature etymologies; Pathfinder bestiary).
wyrd kenning lexicon extract-pfsrd2-monsters ~/521Studios/pfsrd2-data \
    --output /tmp/pfsrd2-monsters.jsonl
wyrd kenning lexicon mine-fantasy-name --batch /tmp/pfsrd2-monsters.jsonl \
    --skip-resolved --apply
```

## Extending

- **New format of etymology book**: subclass parser style. If it's
  alphabetical-headword-shaped, extend `dictionary_parser._ENTRY_HEADWORD`
  patterns. If it's outline-shaped, write a third parser.
- **New language family**: add to `LANGUAGE_FIELDS` in `lexicon.py` and the
  enum in both `RESPONSE_SCHEMA` and `GEMINI_RESPONSE_SCHEMA`.
- **New culture register**: add the string to `CULTURES` in `__init__.py`,
  bundle a `<culture>_place_names.json` (real names list) and a
  `<culture>_proportions.json` built via `wyrd kenning rebuild-proportions`.
- **New per-language bundle metadata** (e.g. for D6 `--mood harsh` phonological
  scoring): follow the D26 sibling-field pattern (`<lang>_<feature>`)
  and route through `load_meanings` into a dedicated `Meaning` attribute.
- **New LLM provider**: implement `chat_json(system, user, schema)` and
  `extract_one(client, toponym, body, suffix_hint)`. Reuse
  `validate_response` from `llm_extractor.py`. Wire as a `--provider`
  option in `cli.py`.
- **Time-aware generation**, **cross-cultural rendering**, **name
  families**, **stratified maps**: see the open tickets in `bd list`.
