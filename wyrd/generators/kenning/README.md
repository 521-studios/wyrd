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
| `cli.py` | All `wyrd kenning lexicon …` commands. |

Generator runtime (`__init__.py`, `name.py`, `word.py`, `meaning.py`,
`proportions.py`) is unchanged from the original Rando port. The lexicon
is currently an authoring-side store — the generator still reads the
denormalized `meanings.json`. Closing that loop (lexicon → meanings.json
export) is filed as a future ticket.

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

# Read-only summary of what's in the lexicon.
wyrd kenning lexicon report
```

## Extending

- **New format of etymology book**: subclass parser style. If it's
  alphabetical-headword-shaped, extend `dictionary_parser._ENTRY_HEADWORD`
  patterns. If it's outline-shaped, write a third parser.
- **New language family**: add to `LANGUAGE_FIELDS` in `lexicon.py` and the
  enum in both `RESPONSE_SCHEMA` and `GEMINI_RESPONSE_SCHEMA`.
- **New LLM provider**: implement `chat_json(system, user, schema)` and
  `extract_one(client, toponym, body, suffix_hint)`. Reuse
  `validate_response` from `llm_extractor.py`. Wire as a `--provider`
  option in `cli.py`.
- **Time-aware generation**, **cross-cultural rendering**, **name
  families**, **stratified maps**: see the open tickets in `bd list`.
