# wyrd

Random-generation service for tabletop RPGs.

Each random generator is a plugin under `wyrd/generators/`. The Flask app discovers
plugins automatically and exposes them under a uniform API. The static SPA
renders forms from the manifest, so adding a new generator on the server
requires no client changes.

## Generators (v1)

- **Kenning** — town-name generator. Compounds Old English / Old Norse / Old
  French / Celtic morphemes statistically derived from real British Isles
  place names. Picks a culture (`english`, `scottish`, `welsh`, `irish`) and
  optional tag filters (`tree`, `water`, `religion`, ...).

## Local development

```bash
# Install
pip install -e '.[dev]'

# Run the dev server
wyrd serve --debug
# → http://127.0.0.1:5000

# CLI generation
wyrd kenning generate english
wyrd kenning generate english --tag tree --tag water --seed 42

# Rebuild proportions for a culture (after editing meanings or place names)
wyrd kenning rebuild-proportions english \
    wyrd/generators/kenning/data/english_place_names.json \
    > wyrd/generators/kenning/data/english_proportions.json

# Append new name-based meanings (e.g. saints' names)
cat saints.txt | wyrd kenning add-meaning --tag saint --tag religion

# Validate meanings.json
wyrd kenning validate-meanings wyrd/generators/kenning/data/meanings.json

# Tests
pytest
```

## API

- `GET  /api/manifest` — list of registered generators with their input schemas.
- `GET  /api/<generator>?<query>&seed=N` — simple invocation; query params coerced.
- `POST /api/<generator>` — JSON body, canonical form for complex inputs.

Every successful response has a uniform envelope:

```json
{
    "generator": "kenning",
    "parameters": { "culture": "english", "tags": ["tree"] },
    "seed": 12345,
    "result": "Brecombe",
    "explanation": "Bre- (EN hill) -combe (EN valley)",
    "components": [...]
}
```

## Adding a new generator

1. Create a package under `wyrd/generators/<name>/` with an `__init__.py` that
   defines a `Generator` subclass and calls `register(YourGen())`.
2. Implement `input_schema()` (JSON Schema) and `generate(params, seed)` returning
   a `GenerationResult`.
3. Optionally add `wyrd/generators/<name>/cli.py` with a click `cli` group; it
   will be auto-mounted as `wyrd <name> ...`.
4. The SPA picks it up from `/api/manifest` automatically.

## Layout

```
wyrd/
├── app.py              Flask app, routes, dispatcher
├── lambda_handler.py   Lambda entry (apig-wsgi)
├── registry.py         Generator base class + plugin registry
├── envelope.py         Uniform response shape
├── seed.py             Seed handling + deterministic RNG
├── cli.py              `wyrd` entry point
└── generators/
    └── kenning/        Town name generator (ported from Rando)
spa/                    Static SPA — manifest-driven form renderer
tests/                  pytest suite
```
