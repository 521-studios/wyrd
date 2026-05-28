# generator-contract-reviewer

Each generator in `wyrd/generators/<name>/` must implement the `Generator` ABC (see `wyrd/registry.py`) and self-register via `register()`. Review new/modified generators against the contract. **Default severity: P2** — contract violations break the registry surface (the SPA's generator dropdown, CLI subcommands, the dispatcher).

**Required class attributes:**

- `name` — short identifier, lowercase, matches the package directory name and the URL path under `/api/`.
- `display_name` — human-readable title shown in the SPA dropdown.
- `description` — one or two sentences shown under the dropdown.

**`input_schema()` method must return a JSON Schema object with:**

- `type: "object"`.
- `properties` covering all params, each with `type`, `description`, and a sensible `default` where applicable.
- `required` listing only the parameters the generator depends on.
- A `seed` property if the generator accepts one — but `seed` should NOT be in `required` (the dispatcher resolves it before calling `generate()`).
- A `count` property if the generator supports batching, with `minimum` and `maximum` matching `wyrd/app.py:MAX_COUNT`.

**`generate(params, seed)` method:**

- Pure function of `(params, seed)` — same inputs, same `GenerationResult`. (Cross-references `seed-reproducibility-reviewer` and `test-coverage-reviewer`'s deterministic-output rule.)
- Returns a `GenerationResult` with `result`, `explanation`, and (optionally) `components`.
- Must use `rng_for(seed)` for randomness — see `seed-reproducibility-reviewer`.
- Must not perform unbounded I/O. Lambda cold starts amortize bundled data only — no S3, no HTTP, no disk writes at request time.

**Registration:** the bottom of the generator's `__init__.py` must call `register(MyGenerator())`. Without it the generator never appears in the registry, the dispatcher can't route to it, and the SPA dropdown is missing the entry.

**CLI subcommand:** new generators should expose a `cli.py` with a click command group mounted under `wyrd <name> ...` via `wyrd/cli.py`. CLI defaults must match the input_schema defaults — drift between the two surfaces is a bug because operators expect `wyrd <name>` from the command line to behave the same as the SPA.

**Acceptable patterns:**

- A generator that intentionally has no CLI subcommand (rare — usually for generators that only make sense behind the web UI). The reason should be in a comment near `register()`.
- A generator that uses additional bundled data via `importlib.resources` — see `importlib-resources-reviewer`.

**Review approach:**

1. For new generator modules, verify the ABC contract is fully implemented (`name`, `display_name`, `description`, `input_schema()`, `generate()`).
2. Confirm `input_schema` defaults match CLI defaults.
3. Spot-check that `generate()` is deterministic for fixed `(params, seed)` — typically by reading the code and confirming the routing through `rng_for`.
4. Verify `register()` is called exactly once at module import (bottom of `__init__.py`).
5. Verify the CLI subcommand is mounted in `wyrd/cli.py`.

