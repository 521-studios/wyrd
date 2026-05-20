# Configuration

```json
{
  "defaults_version_checked": "1.3.0",
  "overlap_acknowledged": {
    "complexity-reviewer": {
      "overlaps_with": "code-simplifier",
      "reason": "Different lenses: complexity-reviewer applies hard metric rules (the and/or test, the ~50-line one-screen rule, extractable inner structures), while the default code-simplifier focuses on genuinely confusing patterns (nested ternaries, redundant abstractions, dense one-liners). Both contributed independent signal during the wyrd-67fv review loop (PR #234)."
    },
    "test-coverage-reviewer": {
      "overlaps_with": "pr-test-analyzer",
      "reason": "Different lenses: test-coverage-reviewer enforces the absolute rule that every touched Python function needs a unit test, while the default pr-test-analyzer scores behavioral-coverage gaps on a criticality axis (HIGH/MEDIUM/LOW). Both contributed independent findings during PR #234 — pr-test-analyzer flagged unverified new abstractions (engine surface, alembic stamp, close lifecycle) that test-coverage-reviewer's binary rule would have missed."
    }
  },
  "independent_validator": {
    "enabled": true,
    "skip_for": [],
    "uncertain_action": "post_with_annotation"
  }
}
```

# Agents

## complexity-reviewer

Review **production code only** for function complexity. **Skip all files in `tests/` directory** — test files often have long fixtures and assertion blocks that don't need the same complexity constraints.

Apply these heuristics:

1. **"And/Or" test**: Minimize the number of "and" or "or" needed to describe what a function does. If you need multiple conjunctions, the function is doing too much.
   - Good: "This function generates a town name from a culture's morpheme proportions"
   - Bad: "This function generates a name AND parses tags AND seeds the RNG AND formats the explanation"

2. **One-screen rule**: Functions should fit on one screen (~50–60 lines) so they can be reviewed at a glance. Longer functions are harder to reason about.
   - **Internal functions don't count**: When a function contains internal helper functions (defined with `def` inside the parent), those lines do NOT count against the parent's line limit. Only the "main body" lines count.
   - **Internal functions at the top**: Internal private functions should be declared at the top of their parent function, before the main logic begins.

3. **Extractable inner structures**: If a block of code has a clear purpose, consider extraction:
   - **First choice**: Extract to module level (prefixed with `_`) if reusable within the file.
   - **Second choice**: Extract to a sibling helper module under the same generator package.
   - **Last choice**: Keep as an internal function if truly tied to the parent's context.

**Review approach:**
1. Skip test files.
2. For each new/modified function, try to describe it in one sentence without "and"/"or".
3. If the description requires conjunctions, identify which parts should be separate functions.
4. Flag functions over ~50 lines (excluding internal function definitions) and suggest logical split points.
5. Look for nested loops, long conditionals, or repeated patterns that could be extracted.

**Note:** It is acceptable to acknowledge complexity and defer refactoring by creating a beads ticket, rather than fixing it in the current PR.

## import-reviewer

Review code for PEP8-compliant import practices. Imports belong at the top of the file, not inside functions.

**Rules to enforce:**

1. **Imports at top of file**: All imports should be at module level, after any module docstring and before any code.

2. **No function-level imports**: Flag any `import` or `from ... import` statements inside functions. These should be moved to the top of the file.

3. **Import ordering** (PEP8):
   - Standard library imports first
   - Blank line
   - Related third-party imports
   - Blank line
   - Local application/library imports

**Exceptions:**
- Conditional imports for optional dependencies (wrapped in try/except).
- Imports that genuinely break a circular dependency (must be documented with a comment explaining why).
- Imports deferred to minimize startup cost in CLI tools or other entry points (must be documented with a comment explaining why).

**Review approach:**
1. Search for `import` statements inside function bodies (`def` blocks).
2. Flag each with the specific function name and line.
3. Suggest moving to the appropriate import section at the top of the file.

**Note:** It is acceptable to acknowledge import issues and defer cleanup by creating a beads ticket, rather than fixing it in the current PR.

## test-coverage-reviewer

Review code changes to **require unit tests for all new or modified functions**. Tests live in `tests/` and run under pytest.

**Core rule: Every PR must include unit tests for the code it touches.**

This is NOT optional. PRs without tests for new/modified Python code should be blocked.

**Rules to enforce:**

1. **Unit tests REQUIRED for all touched Python code**: Any modified or new function MUST have corresponding unit tests in `tests/`. If tests don't exist, the PR author must add them. No exceptions for "refactoring" or "simple changes" — tests prove the code works.

2. **Test file naming**: Tests should be in `tests/test_<module>.py` matching the module being tested (e.g. `wyrd/seed.py` → `tests/test_seed.py`, `wyrd/generators/kenning/__init__.py` → `tests/test_kenning.py`).

3. **Bug fix documentation**: If the code change fixes a bug:
   - Require a comment explaining what was broken and why the fix works.
   - Require a regression test that would have caught the bug.

4. **SPA changes** (in `spa/`) currently have no test harness. Note this as a coverage gap in PRs that touch SPA code, but do not block on it — the SPA is small enough to verify by hand.

**Review approach:**
1. Identify ALL Python functions/methods that were added or modified in the PR.
2. For EACH function, verify `tests/` contains tests that exercise it.
3. If tests are missing, post a comment listing the specific functions that need tests.
4. Suggest specific test cases based on the function's logic and edge cases.
5. Do NOT accept "manual smoke test" or "I verified it via the CLI" as a substitute for unit tests.

**Do NOT allow:**
- Accepting "pure refactoring" as an excuse — refactoring PRs especially need tests to prove behavior is preserved.
- Marking test coverage as "out of scope" — test coverage is NEVER out of scope for Python code. Tests must be included in the same PR as the code they cover.

**Acceptable:**
- Creating a beads ticket to track adding tests, as long as the ticket is created before merging.

## seed-reproducibility-reviewer

Wyrd's contract is that the same `(generator, params, seed)` tuple always yields the same output. Any randomness must flow through `wyrd.seed.rng_for(seed)` so seeds are reproducible.

**Patterns to FLAG:**

1. **Direct use of the global `random` module:**
   ```python
   # BAD — not reproducible
   import random
   random.choice(words)
   random.random()
   random.randrange(2**63)
   ```
   These pull from a process-wide RNG with no relationship to the request's seed.

2. **`secrets` / `os.urandom` in generator code:**
   ```python
   # BAD — non-deterministic, can't reproduce from a seed
   secrets.randbits(64)
   ```
   Acceptable only in `wyrd/seed.py:resolve_seed()` (the one place that's allowed to mint a seed) and tests that exercise `resolve_seed` itself.

3. **Multiple results without sub-seed derivation:**
   When a generator (or the dispatcher loop) produces multiple results, sub-seeds must be derived deterministically from the parent seed via a `random.Random` instance — see `wyrd/app.py:_dispatch()` for the canonical pattern.

**Acceptable patterns:**
- `rng = rng_for(seed); rng.choice(...)`
- A `random.Random` instance threaded through the call stack as a parameter
- `secrets.randbits` in `resolve_seed()` only

**Review approach:**
1. Grep for `import random`, `random.`, `secrets.`, `os.urandom` in `wyrd/`.
2. For each hit, confirm it routes through `rng_for(seed)` or is in the explicitly-allowed location.
3. For loops that produce N results, confirm sub-seeds are derived from the parent seed deterministically.

## generator-contract-reviewer

Each generator in `wyrd/generators/<name>/` must implement the `Generator` ABC (see `wyrd/registry.py`) and self-register via `register()`. Review new/modified generators against the contract.

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
- Pure function of `(params, seed)` — same inputs, same `GenerationResult`.
- Returns a `GenerationResult` with `result`, `explanation`, and (optionally) `components`.
- Must use `rng_for(seed)` for randomness — see `seed-reproducibility-reviewer`.
- Must not perform unbounded I/O. Lambda cold starts amortize bundled data only — no S3, no HTTP, no disk writes.

**Registration:** the bottom of the generator's `__init__.py` must call `register(MyGenerator())`.

**CLI subcommand:** new generators should expose a `cli.py` with a click command group mounted under `wyrd <name> ...` via `wyrd/cli.py`. CLI defaults must match the input_schema defaults — drift between the two surfaces is a bug.

**Review approach:**
1. For new generator modules, verify the ABC contract is fully implemented.
2. Confirm `input_schema` defaults match CLI defaults.
3. Spot-check that `generate()` is deterministic for fixed `(params, seed)` — typically by reading the code, not running it.
4. Verify `register()` is called exactly once at module import.

## terraform-reviewer

Review terraform changes in `terraform/` to ensure this app stays in its lane within the **three-layer state stack**:

```
infra (baseline) → apps (this repo) → infra-frontend
```

**This repo's layer: apps.** Its terraform owns app-specific resources only. It MUST NOT own baseline platform resources or public-facing edge resources.

For full context, read `infra/CLAUDE.md` and `infra-frontend/CLAUDE.md` in the workspace before reviewing.

### What this app's terraform SHOULD own
- App compute: Lambda function + Function URL, ECS tasks, ALBs.
- App storage: S3 buckets the app reads or writes (including the SPA bucket).
- App IAM: roles, policies, and inline permissions the app's compute consumes.
- Internal origin DNS records the app owns (e.g. `<app>-origin.521studios.com` pointing at an ALB).
- CloudWatch log groups and alarms scoped to the app.

### What this app's terraform MUST NOT own
- **CloudFront distributions** — owned by `infra-frontend`.
- **ACM certificates** for public domains — owned by `infra-frontend` (these must live in `us-east-1` for CloudFront).
- **Public DNS records** (apex, www, custom subdomains the public hits directly) — owned by `infra-frontend`.
- **CloudFront Functions** — owned by `infra-frontend`.
- **Foundational shared resources**: VPCs, subnets, security groups, Aurora clusters, ECS clusters — owned by `infra`.

### What this app's terraform MUST NOT do
- **Read from `infra-frontend` remote state.** Apps deploy *before* infra-frontend, so this is a circular dependency. If a CloudFront-owned value is needed, the value belongs in the app's outputs and infra-frontend should consume it, not the other way around.
- **Embed AWS account IDs as literals** outside of remote-state backend configs. Use `data "aws_caller_identity"` or variables.
- **Reach across into another app's resources.** Apps consume shared values from `infra` via remote state and shared primitives via AWS data sources — they do not poke into peer apps' state.

### What this app's terraform SHOULD do
- **Export the outputs that `infra-frontend` consumes**: `lambda_function_url`, `lambda_function_name`, `s3_bucket_name`, `s3_bucket_arn`, `s3_bucket_regional_domain`. Naming should match what `infra-frontend` already reads — see the `terraform_remote_state` blocks in `infra-frontend/terraform/environments/<env>/main.tf`.
- **Read from `infra` remote state** when consuming shared platform values (VPC IDs, ECS cluster ARN, DB endpoints).
- **Use AWS data sources** (e.g. `data "aws_route53_zone"`) instead of hardcoding values that already exist in the account.
- **Keep environments separate**: `terraform/environments/staging/` and `terraform/environments/production/` are distinct root modules with distinct state.

### Cost discipline
- A new CloudFront distribution + ACM cert costs ~$0.60/month minimum to exist, before any data transfer. Before suggesting "this app should own its own distribution," ask whether a path behavior on an existing distribution would work — and remember that even when the answer is "yes, a new distribution is justified," the distribution still belongs in `infra-frontend`, not here.

### Review approach
1. For each `resource "aws_*"` and `module ".*"` in the diff, ask: does this belong in the app layer, or is it overreach into `infra` or `infra-frontend`?
2. Flag any `terraform_remote_state` block reading from `infra-frontend/<env>/terraform.tfstate` — that's the smoking gun for a layer violation.
3. Flag any `aws_cloudfront_distribution`, `aws_acm_certificate`, `aws_cloudfront_function`, public-facing `aws_route53_record` (anything not under `*-origin.521studios.com` or similar internal aliases), or VPC/subnet resources.
4. Flag hardcoded account IDs, region literals that mismatch the rest of the repo, or duplicated provider blocks.
5. For new outputs, confirm they have a clear consumer in `infra-frontend` (or another known reader) — orphan outputs accumulate over time.
6. For new variables, confirm sensible defaults and that the staging/production root modules both wire them through.

**Note:** It is acceptable to acknowledge a layering violation and defer the fix by creating a beads ticket — but mark it P1, not P3. Layer violations create deploy-order coupling that gets harder to untangle the longer it sits.

## db-reconstructibility-reviewer

Review changes to the lexicon DB schema, ingesters, and mining pipelines for **reconstructibility from JSONL artifacts**. The architectural rule this reviewer enforces:

1. **The lexicon DB at `~/.wyrd/lexicon.db` can be blown away at any time.** Operators must be able to `rm ~/.wyrd/lexicon.db && wyrd kenning lexicon rebuild-from-jsonl` and end up with the same DB state.
2. **Reconstruction MUST require no mining.** Mining is expensive — LLM calls cost API credits, Ollama runs take days, OCR is irreversible operator effort. None of this should be re-run to rebuild the DB.
3. **All mining MUST leave durable JSONL artifacts** under `data/mining/` (or a sibling per-source directory). The JSONL files are the canonical source of truth; the DB is a queryable index over them.
4. **`rebuild-from-jsonl` is the canonical reconstruction path.** Every event type any ingester writes must be replayable by it.

The failure mode this guards against: an ingester that writes DB rows but emits no JSONL, so blowing away the DB means re-running the (expensive) ingest from the source materials — which may themselves be lost, paywalled, or behind a flaky network the operator no longer wants to depend on.

**Scope rule (touch-it-you-own-it):** New ingesters, schema additions, mining CLIs, and any module that does `INSERT INTO` against the lexicon DB are in scope. Read-only paths (queries, reports, generation code) are out of scope.

**What to check:**

1. Walk the PR diff for any of:
   - `CREATE TABLE` in `wyrd/generators/kenning/data/lexicon.sql`, `wyrd/generators/kenning/lexicon/schema.py`, or `wyrd/generators/kenning/lexicon/sql/migrations/versions/`
   - `INSERT INTO` / `INSERT OR IGNORE INTO` / `INSERT OR REPLACE INTO` in any new ingester or CLI module
   - New `mine-*` / `ingest-*` CLI subcommands
   - LLM calls (`chat_json`, `provider="ollama"|"gemini"|"anthropic"`, etc.) or network fetches whose outputs land in the DB

2. For each new write path, confirm:
   - **JSONL emission exists.** Either the ingester writes to `data/mining/<source_id>.jsonl` as it ingests (per-event), OR a `dump-jsonl` subcommand at `wyrd/generators/kenning/jsonl/dump.py` knows how to round-trip the new rows back out.
   - **`rebuild-from-jsonl` handles the new event types.** Look at `wyrd/generators/kenning/jsonl/build.py` (or wherever `rebuild-from-jsonl` dispatches event `_type` values) and confirm the new `_type` strings are wired in.
   - **The reconstruction does not require the source materials.** The JSONL must be self-contained — no `pdftotext`, no re-LLM-calling, no S3 fetch is acceptable as part of `rebuild-from-jsonl`.

3. For each new mining CLI, confirm:
   - The CLI writes JSONL **before** (or in parallel with) DB writes, so a SIGTERM/crash leaves a recoverable trail.
   - The JSONL path is documented in the docstring + `data/mining/` is the conventional location.

**Canonical pattern** (the one to point new ingesters at):

```
mine-llm <source.txt>
    → DB writes
    → operator runs `lexicon dump-jsonl`
    → committed to `data/mining/<source>.jsonl`

rebuild-from-jsonl
    → reads every `data/mining/*.jsonl`
    → replays into a fresh DB
```

A future-better pattern (event-log-first):

```
ingest-X <source>
    → writes JSONL events DIRECTLY to data/mining/<source>.jsonl
    → loads JSONL into DB via the same code path rebuild-from-jsonl uses

rebuild-from-jsonl
    → reads every data/mining/*.jsonl (including the freshly-ingested one)
    → no separate dump-jsonl step needed
```

**Flag issues if:**

- A new ingester writes to the lexicon DB but emits no JSONL artifact AND is not handled by `dump-jsonl`. Concretely: blowing away the DB and running `rebuild-from-jsonl` would leave those rows missing.
- A new table is added to the schema (alembic migration / `data/lexicon.sql` / `schema.py` helper) but `wyrd/generators/kenning/jsonl/dump.py` is not updated to round-trip it AND no per-ingester JSONL emission lands the rows.
- A new `mine-*` / `ingest-*` CLI writes DB rows but the only path back is "re-run the LLM on the source materials" — i.e., reconstruction depends on either expensive compute or staged source files.
- A migration adds a column whose value is mined (not derived) and the mining step has no JSONL trail.
- An ingester takes an external source (HTTP URL, S3 fetch, PDF) and writes to the DB without first capturing the raw response / parsed events to JSONL.
- The PR description claims "idempotent re-run on the same .txt reproduces the DB" — but the .txt itself isn't part of the in-repo reconstruction surface. The JSONL is the boundary; staged source materials are not.

**Do NOT flag:**

- Schema migrations with no data (pure `CREATE TABLE` / `ALTER TABLE` with no INSERT).
- L3 enrichment columns whose values are deterministically derivable from existing DB rows via a re-runnable `lexicon enrich --apply` pass. These are computed from DB content, not from external sources.
- Test fixtures, smoke-test code, or paths gated by `if TESTING` / `pytest`-only construction. Tests build throwaway DBs.
- Read-only paths (queries, reports, browse, audit) that don't write to the DB.
- The bundle export path (`bundle.json` / `meanings.json`) — those are downstream artifacts derived from DB queries, not ingest sources.
- One-shot operational scripts (`tools/*` ad-hoc fixups) that are tracked in beads and intended to run once. Document them as such.

**Review approach:**

1. Grep the diff for `INSERT INTO` / `INSERT OR IGNORE INTO` and for new `*_ingester.py` / `mine_*.py` / `ingest_*.py` files.
2. For each new write site, walk the call graph backward: is there a corresponding emit-to-JSONL path? If yes, follow forward: does `rebuild-from-jsonl` consume the JSONL?
3. Grep `wyrd/generators/kenning/jsonl/dump.py` and `wyrd/generators/kenning/jsonl/build.py` for the new table names / event `_type` strings.
4. Try to articulate the reconstruction recipe in one sentence. If it requires more than "`rm db && rebuild-from-jsonl`" (e.g., "and also re-fetch the source PDF, then re-run `ingest-X`"), that's a reconstruction-gap finding.

**Acceptable resolutions** (when the gap is real but the fix is out of scope):

- File a beads ticket explicitly scoped to add the JSONL-emit + rebuild-from-jsonl support, AND mark the PR's docstring / merge-readiness summary with "ingester is DB-only; reconstructibility tracked in <ticket>". The ticket must be P2 or higher, not buried at P4.
- The reviewer does NOT accept "we'll add JSONL later" without a ticket; that's how this rule keeps getting violated.

**Recent violations this reviewer would have caught:**

- wyrd-uzoh (PR #276): Briggs personal-names ingester writes `personal_name` + `personal_name_toponym_attestation` rows directly to the DB with no JSONL emission. `rebuild-from-jsonl` doesn't know about either table. Blowing away the DB requires re-running `ingest-briggs-personal-names ~/.wyrd/source-staging/briggs_2024_personal_names_index.txt` against the staged PDF→txt extract — which means staging the PDF, running pdftotext, and re-paying the parse cost. The JSONL gap was caught only by a post-merge operator question, after merge.

## clarity-reviewer

Review markdown documentation for terseness. Every token costs money and attention — cut the fat.

**What to check:**

1. Look at the PR diff for changes to `.md` files.
2. **Read the full file, not just the diff** — context is needed to spot redundancy with existing content.
3. Examine new or modified text for:
   - Redundant phrasing ("in order to" → "to").
   - Filler words ("actually", "basically", "simply", "really").
   - Stating the obvious or repeating context already established.
   - Overly long explanations where a short one suffices.

**Common patterns to flag:**

| Verbose | Terse |
|---------|-------|
| "in order to" | "to" |
| "for the purpose of" | "to" / "for" |
| "in the event that" | "if" |
| "at this point in time" | "now" |
| "due to the fact that" | "because" |
| "it is important to note that" | (delete, just state the thing) |
| "as mentioned above/previously" | (delete or use a link) |
| "This section describes how to..." | (delete, describe it directly) |

**Flag issues if:**
- A sentence can be cut in half without losing meaning.
- The same information is stated twice in different words.
- Explanatory text explains something already obvious from context.
- New text restates something already covered in unchanged parts of the file.

**Do NOT flag:**
- Necessary detail that aids understanding.
- Examples and code blocks (these should be complete).
- Repetition that serves as a deliberate reminder (e.g. "NEVER push directly to main" repeated for emphasis).
- Technical precision that requires specific wording.

## fixture-data-reviewer

Review test changes for **isolation from live operator data**. Tests must run against fixtures, not the operator's live lexicon DB or live bundle data, for two reasons:

1. **CI doesn't have the live data.** `~/.wyrd/lexicon.db` doesn't exist on CI runners. A test that depends on it works locally and fails on CI — false-green that surfaces only at deploy time.
2. **The live DB is huge (2.4M+ etymon rows, 5GB+ wiktextract slices in `~/.wyrd/sources/`).** A test path that walks it is minutes-long instead of milliseconds. Slow tests discourage running pytest before commit, which discourages catching regressions early.

**Rules to enforce:**

1. **No live-DB dependency.** Tests must NOT assume `~/.wyrd/lexicon.db` exists. The conftest autouse `_isolate_default_lexicon_db` fixture redirects `_DEFAULT_LEXICON_PATH` to `tmp_path/test.db` — fine. Tests that need a populated DB build one inside the test (`sqlite3.connect(":memory:")` + `executescript(SCHEMA_SQL)` + a few INSERT rows, OR open `tmp_path/test.db` and populate).

2. **No live-bundle dependency.** Tests that exercise generator behaviour use the `swap_bundle` / `bundle_swapper` fixtures from `tests/conftest.py` rather than the runtime `meanings.json`. Broad-stroke smoke tests against the live bundle are explicitly named (`test_find_meaning_runs_full_bundled_corpus_without_crashing`, `test_sidecar_lifts_irish_corpus_perfect_rate`) and are the exception, not the rule.

3. **No live-bulk-manifest dependency.** Tests must NOT trigger the L1 wiktextract ingest against the operator's `~/.wyrd/sources/`. The conftest autouse `_isolate_bulk_manifest` fixture handles this for `rebuild-from-jsonl` tests; tests that exercise bulk-source paths directly should monkeypatch `bulk_sources.MANIFEST_PATH` to a tmp manifest with synthetic slices.

4. **Fixture data must stay representative.** When the production data shape changes (schema migration, new column, manifest schema_version bump, new row type), the fixture data has to evolve in lockstep. A test that pins a 2-row fixture against an obsolete shape passes locally but is no longer testing what its name claims. Watch for:
   - Schema changes in `wyrd/generators/kenning/data/lexicon.sql` not reflected in test fixtures that `executescript` partial schemas
   - Manifest `schema_version` bumps in `bulk_sources.py` not reflected in test-fixture manifests
   - New required columns / fields on JSONL row types that fixtures still elide
   - New L3 derivations or enrichment columns that fixtures don't seed

5. **CLI tests must override `--db` (or rely on the autouse default redirect).** A test that invokes `lexicon <command>` without explicit `--db` defaults to whatever `_DEFAULT_LEXICON_PATH` resolves to — locally the live DB, on CI nonexistent. The conftest autouse fixture redirects this to `tmp_path/test.db`, but tests should still prefer explicit `--db <tmp>` for clarity.

**Review approach:**

1. Walk the diff for test files (any `tests/test_*.py` change).
2. Grep new tests for `~/.wyrd/`, `_DEFAULT_LEXICON_PATH`, `meanings.json`, `MANIFEST_PATH`, `data/mining/_bulk_manifest.json` — any reference is a flag-worthy review point. Confirm the test either:
   * passes an explicit `tmp_path` override, OR
   * uses an autouse fixture from `tests/conftest.py`, OR
   * is one of the documented live-data smoke tests (then ask whether it should be).
3. For any non-test source diff that changes a data shape (schema, manifest, row type), grep `tests/` for fixtures of the same shape and flag whether the fixtures need updates.
4. Note any test that takes >1s in the `--durations` output as a possible live-data leak.

**Acceptable:**
- A test deliberately opting out via `@pytest.mark.no_lexicon_isolation` (with comment explaining why fixture data won't work)
- Creating a beads ticket to track a fixture update if the production shape changed and updating fixtures is out of scope

**Do NOT allow:**
- "Local run works, CI will figure it out" — CI doesn't have the local data.
- "It's only a smoke test" — smoke tests against the live bundle are documented exceptions; new ones need explicit justification.
- Skipping fixture updates after a schema/manifest change with "we'll catch it later" — the fixture stops being representative the moment the production shape diverges.

# Guidelines

## SPA must remain manifest-driven

The SPA at `spa/app.js` builds its form from each generator's `input_schema`. **Do NOT hardcode per-generator UI in the SPA** — adding a new generator on the server should require zero SPA changes. If a generator needs a control the SPA can't render from JSON Schema, extend the schema-to-form renderer generically rather than special-casing the generator.

## Hashed asset names

`spa/index.html` references `app.__SHA__.js` and `style.__SHA__.css`. The `__SHA__` placeholder is replaced with the commit SHA at deploy time by `bin/deploy-spa.sh`. The dev server (`wyrd/app.py:spa_static`) substitutes "dev" and resolves hashed asset URLs back to source. Don't change the placeholder format without updating both sides.

## CloudFront + Lambda Function URL body hash

For POST/PUT requests, the SPA computes `x-amz-content-sha256` of the body and sends it as a header — see `spa/app.js:sha256Hex()`. CloudFront does NOT compute this for OAC-signed Lambda Function URL origins; without the header Lambda rejects with `InvalidSignatureException`. Any new client (CLI, other SPA, third-party) calling the API directly behind CloudFront must do the same.

## Bundled data, not S3

Generator data (proportions, meanings, etc.) is bundled inside the Lambda package via `importlib.resources`. Don't fetch data from S3 in generator code — Lambda cold start times depend on a small package, and the deploy story is simpler with everything bundled.

## Always do PRs

All work goes through pull requests — never push directly to `main`. Open a feature branch, commit there, push, and open a PR. Merge only after CI passes and review is complete.

# Context

Wyrd is a Flask + Click app deployed as a Lambda + CloudFront SPA. The generator registry pattern (`wyrd/registry.py`) means each new generator is a self-contained subpackage under `wyrd/generators/<name>/`. Tests live in `tests/test_<generator>.py` and run under pytest.

CI: pytest + ruff format/check on PR/push (see `.github/workflows/ci.yml`).
Deploy: `.github/workflows/deploy.yml` builds the Lambda zip, applies terraform, and uploads hashed SPA assets behind CloudFront.
