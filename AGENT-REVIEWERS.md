# Configuration

```json
{
  "defaults_version_checked": "1.6.0",
  "disabled": [
    "silent-failure-hunter",
    "code-simplifier"
  ],
  "overlap_acknowledged": {
    "test-coverage-reviewer": {
      "overlaps_with": "pr-test-analyzer",
      "reason": "Different lenses: test-coverage-reviewer enforces hard rules (touched code must be covered; mocking discipline; missing assertions; skip-without-reason). pr-test-analyzer scores behavioral gaps on a criticality 1-10 axis. Both contribute independent signal during review loops."
    }
  }
}
```

---

# Agents

Each H2 below names a reviewer. The one-line summary tells the main loop **what
the reviewer checks and when to spawn it** — use it to decide whether the PR
diff is in scope. The body points at `.reviewers/<name>.md`, which the spawned
Task reads as its complete specification.

## test-coverage-reviewer

**What it checks:** every PR-touched function has a test that exercises it; flags missing tests for new branches/exceptions, weak mocking (patch point-of-definition, mocking third parties), tests that don't assert, `@pytest.mark.skip` without `reason=`.
**When to spawn:** PR modifies any `*.py` outside `tests/` (and ideally also when `tests/**/*.py` changes — to lint the new tests themselves).

Read `.reviewers/test-coverage-reviewer.md` and follow it as your complete review specification.

---

## error-handling-reviewer

**What it checks:** silent exception swallows, `return` inside `finally`, generic `except Exception` without re-raise, bare `except:`, missing `raise ... from`, `assert` used for runtime validation.
**When to spawn:** PR touches production `*.py` (skip if changes are docs / configs / tests only).

Read `.reviewers/error-handling-reviewer.md` and follow it as your complete review specification.

---

## logging-reviewer

**What it checks:** sensitive data interpolated into log calls, `logger.error` inside `except` without `exc_info`, `logging.basicConfig` from library code, eager debug formatting in hot paths; for CLIs also default level / verbosity flag / stdout-vs-stderr discipline.
**When to spawn:** PR touches `*.py` that uses `logger.`, `logging.`, or `print(` for diagnostic output. Skip if the diff is data files / tests only.

Read `.reviewers/logging-reviewer.md` and follow it as your complete review specification.

---

## complexity-reviewer

**What it checks:** McCabe complexity > 10 (hard floor), one-screen rule, nesting depth, parameter count > 5, class size > 20 public methods, nested ternaries, redundant single-call wrappers, generic identifiers in long functions.
**When to spawn:** PR touches production `*.py` (skip `tests/`). Skip if the diff is data / docs / SQL / config only.
**Post only what this PR owns and what decisively breaks a threshold.** Two posting gates (firing data showed ~half of valid findings get dispositioned out-of-scope or won't-fix): (1) **introduced/worsened only** — do NOT post complexity in a function that was already over a threshold before this PR unless the PR pushes it *further* over; a pre-existing long function the PR merely edits is out of scope. (2) **no borderline soft flags** — the "And/Or" test and one-screen rule are advisory; do not post them as standalone findings for a function that passes the objective `C901`/length floor (e.g. ~50–60 lines). Post the hard objective violations always; raise the soft heuristics only when they compound a hard one.

Read `.reviewers/complexity-reviewer.md` and follow it as your complete review specification.

---

## concurrency-reviewer

**What it checks:** any new `threading.Lock`/`RLock`/shared mutable global (P1 by default), CPU-bound work in `ThreadPoolExecutor`, fire-and-forget `asyncio.create_task`, `asyncio.gather` exception handling, blocking calls in `async def`, missing async timeouts, daemon threads that write data.
**When to spawn:** PR diff imports or modifies code using `threading`, `multiprocessing`, `asyncio`, `concurrent.futures`, or `queue`. Skip otherwise — this reviewer has nothing to say about purely synchronous code.

Read `.reviewers/concurrency-reviewer.md` and follow it as your complete review specification.

---

## seed-reproducibility-reviewer

**What it checks:** wyrd's `(generator, params, seed) → same output` contract — direct `random.*` use, `secrets`/`os.urandom` outside `resolve_seed`, multi-result generators that don't derive sub-seeds deterministically.
**When to spawn:** PR touches `wyrd/**/*.py` that calls into RNG or generator code. Skip for terraform, docs, tests-only, or non-`wyrd/` changes.

Read `.reviewers/seed-reproducibility-reviewer.md` and follow it as your complete review specification.

---

## generator-contract-reviewer

**What it checks:** new/modified `Generator` subclasses implement the ABC (`name`/`display_name`/`description`/`input_schema()`/`generate()`), self-register, expose a CLI subcommand with defaults matching `input_schema`, and don't perform unbounded I/O.
**When to spawn:** PR touches `wyrd/generators/**/*.py` — especially under a new or recently-added generator subpackage. Skip otherwise.

Read `.reviewers/generator-contract-reviewer.md` and follow it as your complete review specification.

---

## resource-leak-reviewer

**What it checks:** `open()` / `subprocess.Popen` / DB connections / HTTP responses without context managers, unbounded `lru_cache` on external key spaces, long-lived module-level caches without eviction, `aiohttp.ClientSession` per request, manual `try/finally` where `ExitStack` would do.
**When to spawn:** PR touches `*.py` that allocates fds, sockets, connections, subprocesses, or long-lived caches. Skip if the diff is pure logic / tests / docs.

Read `.reviewers/resource-leak-reviewer.md` and follow it as your complete review specification.

---

## external-process-reviewer

**What it checks:** missing `timeout=` on `subprocess.run`, `shell=True` with non-literal input (P1 command injection), `Popen` without context manager, no stderr capture on failure, missing `cwd=`, no differentiation between timeout / exit code / missing binary.
**When to spawn:** PR diff imports or uses `subprocess`, `os.system`, or `shutil.which`. Skip otherwise.

Read `.reviewers/external-process-reviewer.md` and follow it as your complete review specification.

---

## db-reconstructibility-reviewer

**What it checks:** the lexicon DB at `~/.wyrd/lexicon.db` must be reconstructible from JSONL artifacts under `data/mining/` without re-running expensive mining (LLM/OCR/Ollama). Flags new `INSERT INTO` paths, new ingester/mining CLIs, or new tables that don't emit JSONL or aren't replayable by `rebuild-from-jsonl`.
**When to spawn:** PR touches `wyrd/generators/kenning/**/*.py` and the diff includes new schema, new ingester, new `mine-*`/`ingest-*` CLI, or `INSERT INTO` against the lexicon DB. Skip for read-only paths (queries, reports, browse, generation).

Read `.reviewers/db-reconstructibility-reviewer.md` and follow it as your complete review specification.

---

## dead-code-reviewer

**What it checks:** unused module-level functions/classes/constants, unused `__init__.py` re-exports, unused conftest fixtures, commented-out code, partial refactors leaving the old name, stale `__all__` entries, `if False:` / `if True:` branches, `pass`-only stubs.
**When to spawn:** PR touches `*.py`, especially when it removes call sites, renames functions, or extracts/moves code. Skip for pure additive changes with no refactor surface.

Read `.reviewers/dead-code-reviewer.md` and follow it as your complete review specification.

---

## fixture-data-reviewer

**What it checks:** tests must not depend on `~/.wyrd/lexicon.db`, the live `meanings.json` bundle, or `~/.wyrd/sources/` wiktextract slices — CI doesn't have any of these. Flags references to `_DEFAULT_LEXICON_PATH`, `MANIFEST_PATH`, etc., that don't go through the conftest autouse isolation fixtures.
**When to spawn:** PR touches `tests/**/*.py` OR modifies data shapes (schema, manifest, JSONL row types) that test fixtures might mirror. Skip if no test files changed and no data shape moved.

Read `.reviewers/fixture-data-reviewer.md` and follow it as your complete review specification.

---

## import-reviewer

**What it checks:** function-level imports without a documented exception, `from X import *`, import-order violations (only when ruff `I` isn't running in CI), relative imports past package boundaries.
**When to spawn:** PR touches `*.py`. Skip for non-Python diffs.

Read `.reviewers/import-reviewer.md` and follow it as your complete review specification.

---

## importlib-resources-reviewer

**What it checks:** package data files loaded via `Path(__file__).parent` or `__file__.parents[N]` instead of `importlib.resources` — fragile under module moves and broken in Lambda zips / frozen builds.
**When to spawn:** PR touches `*.py` inside a packaged module (under `wyrd/`) that loads bundled data files. Skip for tests, ad-hoc scripts, and CLI shims.

Read `.reviewers/importlib-resources-reviewer.md` and follow it as your complete review specification.

---

## dataclass-decorator-reviewer

**What it checks:** classes that use `dataclasses.field(default_factory=...)` or `field(default=...)` without a `@dataclass` decorator — a P1 correctness bug that passes type checkers and surfaces at first instance use.
**When to spawn:** PR **moves or extracts a class between files**, OR a diff hunk adds `field(default=...)`/`field(default_factory=...)` to a class body **whose `@dataclass` line is outside the hunk** (so the decorator can't be confirmed next to the `field()` call) — the cases where a missing `@dataclass` actually hides. Skip PRs that merely touch `dataclasses` with the decorator visibly intact in-hunk, and in-place edits to already-decorated classes: a brand-new undecorated `field()` class fails fast at first instantiation and tests catch it, so it doesn't need this reviewer. (Firing data: the any-`dataclasses`-touch trigger fired 27× for 1 finding; the move/out-of-hunk case is the subtle one worth the spawn.)

Read `.reviewers/dataclass-decorator-reviewer.md` and follow it as your complete review specification.

---

## typing-consistency-reviewer

**What it checks:** files that have opted into type annotations — flags `# type: ignore` without code+reason, half-typed signatures, `Any` as an escape hatch, `cast` used to silence errors, mixed `Optional[X]` and `X | None` within a file, completely-unannotated functions in an otherwise-typed file.
**When to spawn:** PR touches `*.py` files that already use any annotations. Skip for files with zero annotations (untyped is a valid choice — out of scope).

Read `.reviewers/typing-consistency-reviewer.md` and follow it as your complete review specification.

---

## terraform-reviewer

**What it checks:** this repo's terraform stays in the `apps` layer — must NOT own CloudFront distributions, public ACM certs, public DNS records, CloudFront Functions, or foundational shared resources (VPC, subnets, ECS cluster). Must NOT read from `infra-frontend` remote state or hardcode account IDs.
**When to spawn:** PR adds or modifies a `resource`, `data`, `module`, `provider`, or `terraform`/backend block under `terraform/**/*.tf` (i.e. a *structural* change to what infra is owned or where state is read). Skip value-only edits (variable defaults, `locals`, tags, counts) and all non-`terraform/` diffs — every check here is about resource ownership / remote-state boundaries, so a value-only change can't violate them. (Firing data: fired 46× → 8 findings, half low-value, because it ran on every `.tf` touch.)

Read `.reviewers/terraform-reviewer.md` and follow it as your complete review specification.

---

## clarity-reviewer

**What it checks:** terseness/structure of markdown docs (verbose phrasings, walls of text, missing TL;DR, missing examples, heading inflation) AND Python docstrings/comments (restate-the-signature docstrings, WHAT-not-WHY comments, stale comments, docstring claims that don't match the code's tables/symbols/regex counts).
**When to spawn:** PR touches `*.md`, OR modifies docstrings/comments in `*.py`. Skip for pure-code diffs that don't add or change comments.

Read `.reviewers/clarity-reviewer.md` and follow it as your complete review specification.

---

# Guidelines

## SPA must remain manifest-driven

The SPA's `ConfigureColumn` / `Field` (in `spa-next/src/`) builds the params form from each generator's `input_schema`. **Do NOT hardcode per-generator UI in the SPA** — adding a new generator on the server should require zero SPA changes. If a generator needs a control the SPA can't render from JSON Schema, extend `Field.svelte` generically rather than special-casing the generator. Per-generator headline-knob curation lives in `lib/headlineFields.js` and is the one allowed place to mention generator names.

## Vite-built SPA (post-cutover, wyrd-20pz)

`spa-next/` is a Svelte 5 + Vite project. Vite handles its own asset hashing — `npm run build` produces `dist/index.html` referencing `dist/assets/index-<hash>.{js,css}`. `bin/deploy-spa.sh` runs `npm ci && npm run build` and syncs `dist/` to S3 (immutable cache on assets, no-cache on index.html). Dev workflow: `flask --app wyrd.app run` for the API on :5000 + `cd spa-next && npm run dev` for the Vite dev server on :5173 (proxies `/api/*` back to Flask).

## CloudFront + Lambda Function URL body hash

For POST/PUT requests, the SPA computes `x-amz-content-sha256` of the body and sends it as a header — see `spa-next/src/lib/api.js:postSignedJson()`. CloudFront does NOT compute this for OAC-signed Lambda Function URL origins; without the header Lambda rejects with `InvalidSignatureException`. Any new client (CLI, other SPA, third-party) calling the API directly behind CloudFront must do the same.

## Bundled data, not S3

Generator data (proportions, meanings, etc.) is bundled inside the Lambda package via `importlib.resources`. Don't fetch data from S3 in generator code — Lambda cold start times depend on a small package, and the deploy story is simpler with everything bundled.

## Always do PRs

All work goes through pull requests — never push directly to `main`. Open a feature branch, commit there, push, and open a PR. Merge only after CI passes and review is complete.

## How the pack runs

**This file is consumed BY the `pr-review-loop` skill — do not run these reviewers directly.** If you've read this file and are about to spawn the agents yourself (outside the skill), stop and invoke `pr-review-loop` instead: the skill owns the parts this file doesn't define — agents posting findings as PR line comments, independent validation of posted findings, per-thread replies, per-agent retirement, CI gating, and exit conditions. Hand-spawning from this file skips all of that and breaks the PR audit trail.

Each reviewer runs independently and reports findings without coordination. A reviewer's silence on something is not an endorsement — it just means that reviewer didn't see anything in its scope.

**Per-reviewer file scope:**

| Reviewer | Files in scope |
|----------|----------------|
| `test-coverage-reviewer` | `*.py` |
| `error-handling-reviewer` | `*.py` (production code) |
| `logging-reviewer` | `*.py` (rules differ for CLI entry points vs libraries/services) |
| `complexity-reviewer` | `*.py` (skips `tests/`) |
| `concurrency-reviewer` | `*.py` touching `threading`, `multiprocessing`, `asyncio`, `concurrent.futures`, or `queue` |
| `seed-reproducibility-reviewer` | `wyrd/**/*.py` (production code that uses RNG) |
| `generator-contract-reviewer` | `wyrd/generators/**/*.py` |
| `resource-leak-reviewer` | `*.py` |
| `external-process-reviewer` | `*.py` touching `subprocess`, `os.system`, `shutil.which` |
| `db-reconstructibility-reviewer` | `wyrd/generators/kenning/**/*.py` (and any new ingester/mining modules) |
| `dead-code-reviewer` | `*.py` |
| `fixture-data-reviewer` | `tests/**/*.py` |
| `import-reviewer` | `*.py` |
| `importlib-resources-reviewer` | `*.py` in packaged modules (skips `tests/` and ad-hoc scripts) |
| `dataclass-decorator-reviewer` | `*.py` using `dataclasses.field` |
| `typing-consistency-reviewer` | `*.py` files that already use type annotations |
| `terraform-reviewer` | `terraform/**/*.tf` |
| `clarity-reviewer` | `*.md`, `*.py` (docstrings and comments) |

Skip reviewers whose file scope doesn't match the PR diff.

## Tooling assumed in CI

Reviewers do not duplicate work already done by CI. Each Python project running this pack should have these in CI:

- `ruff check .` — lints, import sorting (`I`), unused imports (`F401`), unused vars (`F841`), and more
- `ruff format --check .` — formatter (Black-compatible)
- `mypy .` or `pyright .` — type checker, strict mode preferred
- `pytest -v` with `--cov` — tests with coverage tracking
- `bandit -r .` — security linter (catches `shell=True`, weak crypto, etc.)
- `vulture .` — dead-code detector
- `ruff check --select C901 --max-complexity 10 .` — mccabe complexity ceiling (see `complexity-reviewer`)

A missing tool is itself a **P1** finding on the first PR that brushes against its scope. For example, if CI doesn't run `bandit` and the PR touches `subprocess`, `external-process-reviewer` flags the CI gap as well as the code.

## Severity convention

Every finding must be tagged with a beads-style priority:

| Priority | Disposition | Examples |
|----------|-------------|----------|
| **P1** | Blocking — must fix before merge | Silent exception swallow, new `threading.Lock` without justification, missing `@dataclass` on a class using `field()`, leaked DB connection, command injection via `shell=True`, direct `random` use in a generator, ingester that breaks lexicon DB reconstructibility, terraform layering violation |
| **P2** | Should fix in this PR | Missing `raise ... from`, missing `subprocess` timeout, missing tests for new branches, mccabe complexity floor violation, function-level import, `Generator` ABC violation, fixture-data leak in a test |
| **P3** | Advisory — deferrable with a beads ticket | Complexity heuristic findings, dead code, clarity nits, markdown structural suggestions |

**Default severity per reviewer:**

- `concurrency-reviewer`, `resource-leak-reviewer`: **P1** by default.
- `error-handling-reviewer`: **P2** by default; **P1** for silent exception swallows and `return` in `finally`.
- `logging-reviewer`: **P2** by default; **P1** for sensitive data in log messages.
- `dataclass-decorator-reviewer`: **P1** by default (it's a correctness bug that escapes type checkers).
- `test-coverage-reviewer`, `external-process-reviewer`, `importlib-resources-reviewer`, `typing-consistency-reviewer`: **P2** by default; specific flags inside each reviewer may be P1.
- `complexity-reviewer`: **P2** for mccabe floor violations; **P3** for heuristic findings.
- `import-reviewer`, `dead-code-reviewer`, `clarity-reviewer`: **P3** by default.
- `seed-reproducibility-reviewer`: **P1** by default (reproducibility contract is foundational to wyrd).
- `generator-contract-reviewer`: **P2** by default (ABC violations break the registry).
- `terraform-reviewer`: **P1** for layering violations / forbidden resource types; **P2** for hardcoded values / output-shape changes without coordinated infra-frontend changes.
- `db-reconstructibility-reviewer`: **P1** by default (data-loss class — losing reconstructibility means re-running expensive LLM/OCR/mining).
- `fixture-data-reviewer`: **P2** by default (CI failures from live-data leaks).

A reviewer may promote or demote a specific finding from its default, but must state why.

## Output format

Findings must be structured. Use this template:

```
[<reviewer>] [<severity>] <one-line title>

File: path/to/file.py:LINE
Quote:
    <1-5 lines of code or text being flagged>

Issue: <one or two sentences on what's wrong>
Suggested fix:
    <concrete diff or rewritten code/text>
Reason (optional): <only if not obvious>
```

Example:

```
[concurrency-reviewer] [P1] New threading.Lock without justification

File: cache/store.py:18
Quote:
    class Cache:
        def __init__(self):
            self._lock = threading.Lock()
            self._data: dict[str, Any] = {}

Issue: New threading.Lock protects mutable shared state. Per the pack's design
philosophy, threading locks are flagged by default — move the data into a worker
process behind a queue, or into an asyncio task that owns the state, unless this
falls into "Tolerated uses" in concurrency-reviewer.
Suggested fix: Use multiprocessing.Process + queue.Queue for cross-process shared
state, or an asyncio task that owns the dict and exposes async methods.
```

Structured findings are diff-able, easy to triage, and easy to deduplicate when multiple reviewers flag the same line.

## Deferring findings with beads

To defer a P2 or P3 finding to a follow-up:

1. Create a beads ticket capturing reviewer name, severity, file, and quote.
2. Link the ticket in the PR description or as a reply to the reviewer's comment.
3. The reviewer accepts the deferral only when the beads ticket exists.

**P1 findings are not deferrable** — they must be fixed in-PR.

# Context

Wyrd is a Flask + Click app deployed as a Lambda + CloudFront SPA. The generator registry pattern (`wyrd/registry.py`) means each new generator is a self-contained subpackage under `wyrd/generators/<name>/`. Tests live in `tests/test_<generator>.py` and run under pytest.

CI: pytest + ruff format/check on PR/push (see `.github/workflows/ci.yml`).
Deploy: `.github/workflows/deploy.yml` builds the Lambda zip, applies terraform, and uploads hashed SPA assets behind CloudFront.
