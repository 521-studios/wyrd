# Kenning-specific agent reviewers

These reviewers compose with the universal `AGENT-REVIEWERS.md` at
repo root — they don't replace it. Scope: files under
`wyrd/generators/kenning/` and `tests/test_kenning_*`.

**This file is consumed BY the `pr-review-loop` skill — do not run these
reviewers directly.** If you've read this file and are about to spawn the
agents yourself (outside the skill), stop and invoke `pr-review-loop`
instead: the skill owns agents posting findings as PR line comments,
independent validation of posted findings, per-thread replies, per-agent
retirement, CI gating, and exit conditions. Hand-spawning from this file
skips all of that and breaks the PR audit trail.

# Agents

Each H2 below names a kenning-scoped reviewer. The summary tells the main loop **what the reviewer checks and when to spawn it** — use it to decide whether the PR diff is in scope. The pointer line names the complete spec under `.reviewers/`.

## morpheme-surface-identity-reviewer

**What it checks:** DECISIONS.md D45 — a morpheme's stored identity is its bare surface; dashes are NEVER part of a morpheme name and code NEVER keys off them. Flags new dash-marked identity storage, dash-shaped lookups (the wyrd-c6o1.3 homograph-leak shape), matching/eligibility logic branching on dash markers or `Meaning.location`, and inconsistent folds. Render decoration (D39), raw-source parsing, hyphenated lexical FORMS, and documented legacy-boundary folds are exempt.
**When to spawn:** PR touches `*.py` under `wyrd/generators/kenning/` that handles morpheme surfaces, usage keys, meaning_db-style maps, bundle/L4 emission, or proportions keys — in practice almost every runtime/bundle/lexicon-export diff. Skip pure-docs diffs, parser/extractor-only diffs (raw-source dash handling is exempt), and terraform/SPA-only changes.

Read `wyrd/generators/kenning/.reviewers/morpheme-surface-identity-reviewer.md` and follow it as your complete review specification.

---

## lexicon-package-structure-reviewer

**What it checks:** lexicon-DB concerns (SQL DDL, `INSERT INTO`, schema helpers, enrichment/ingest/audit passes) live under `wyrd/generators/kenning/lexicon/`; runtime files outside the package stay DB-free. Standing placement invariant (established by wyrd-67fv, now closed — the rule outlives the refactor).
**When to spawn:** PR adds new `.py` files under `wyrd/generators/kenning/` (outside `lexicon/`), or any file in the diff contains `CREATE TABLE/INDEX/VIEW` or new `sqlite3` connections. Skip pure runtime/`registers/`/`vectors/`/`bundle/` edits.

Read `wyrd/generators/kenning/.reviewers/lexicon-package-structure-reviewer.md` and follow it as your complete review specification.

---

## alembic-migration-discipline-reviewer

**What it checks:** schema changes go through new Alembic migrations under `lexicon/sql/migrations/versions/` — not direct edits to `data/seed/lexicon.sql`, not new `_create_*_table` / `_migrate_*` helpers in `lexicon/__init__.py`. SA Core MetaData in `lexicon/sql/tables.py` must stay synced with migrations.
**When to spawn:** PR touches `data/seed/lexicon.sql`, any file under `lexicon/sql/migrations/`, `lexicon/sql/tables.py`, or schema helpers in `lexicon/__init__.py`. Skip otherwise.

Read `wyrd/generators/kenning/.reviewers/alembic-migration-discipline-reviewer.md` and follow it as your complete review specification.

---

## importlib-resources-reviewer

**What it checks:** kenning-scoped variant of the universal importlib-resources rule — the package's layered structure makes `Path(__file__).parent` drift more likely than in flat packages. Flags any `__file__.parent` / `__file__.parents[N]` used to navigate to bundled data inside `wyrd/generators/kenning/`.
**When to spawn:** PR touches `*.py` under `wyrd/generators/kenning/` that loads JSON sidecars, SQL schema, or fixture text. Skip tests, ad-hoc scripts, and pure logic edits.

Read `wyrd/generators/kenning/.reviewers/importlib-resources-reviewer.md` and follow it as your complete review specification.

---

## docstring-grep-verify-reviewer

**What it checks:** new/modified module + function docstrings that name specific tables, columns, function names, regex counts, output shapes, or field counts must match the code body (grep the SQL string literal, the dict construction, the dataclass field count). Extraction PRs repeatedly land docstrings whose factual claims have drifted from the code.
**When to spawn:** PR adds or modifies docstrings (module or function) in `wyrd/generators/kenning/**/*.py`. Especially load-bearing for extraction PRs and new modules. Skip for pure-code diffs that don't touch docstrings.

Read `wyrd/generators/kenning/.reviewers/docstring-grep-verify-reviewer.md` and follow it as your complete review specification.

---

## rebuild-runbook-currency-reviewer

**What it checks:** new/renamed data-population CLIs (`mine-*` / `ingest-*` / `backfill-*` / `cleanup-*`) and new mined L3-only layers are registered in `data/mining/_rebuild_layers.json` and documented in `REBUILD.md` so a from-scratch rebuild never silently drops a layer. Free re-runnable mining is an acceptable *documented* rebuild step; paid mining must round-trip through L2 (defers to the repo-root `db-reconstructibility-reviewer`). The doc-currency complement to that reviewer: it asks "is the recovery automatic OR written down in the runbook?", not just "is it recoverable at all?".
**When to spawn:** PR adds/renames a `mine-*`/`ingest-*`/`backfill-*`/`cleanup-*` subcommand under `cli/lexicon/` (or a hand-enumerated data-writing command like `build` / `synsets seed`/`assign`), adds a new mined/ingested table or column, or edits `REBUILD.md` / `_rebuild_layers.json` / `tests/test_kenning_rebuild_runbook.py`. Skip read-only/enrichment-only diffs. (Note: `tests/test_kenning_rebuild_runbook.py` already gates *categorization* in CI — the reviewer judges whether the bucket + reason + runbook placement are **correct**.)

Read `wyrd/generators/kenning/.reviewers/rebuild-runbook-currency-reviewer.md` and follow it as your complete review specification.

---

## dataclass-extraction-decorator-reviewer

**What it checks:** classes moved between files via line-range copy that use `field(default_factory=...)` or `field(default=...)` without a `@dataclass` decorator on the line above. The bug surfaces at instance-construction time, not import time, so type checkers and smoke imports won't catch it.
**When to spawn:** PR moves or extracts classes that use `dataclasses.field` within `wyrd/generators/kenning/`. Skip for non-extraction diffs (in-place edits to existing dataclasses, non-dataclass work).

Read `wyrd/generators/kenning/.reviewers/dataclass-extraction-decorator-reviewer.md` and follow it as your complete review specification.

---

## cli-extraction-test-monkeypatch-reviewer

**What it checks:** when a CLI helper moves from `cli/__init__.py` into a per-command module, existing `monkeypatch.setattr(cli_mod, "_helper", ...)` test sites become stale — they patch the shim, not the consumer's local-bound reference, and the test silently runs against the original.
**When to spawn:** PR moves helpers out of `cli/__init__.py` into modules under `cli/` AND `tests/test_kenning_*` exist that monkeypatch those helpers. Skip when no CLI helpers moved.

Read `wyrd/generators/kenning/.reviewers/cli-extraction-test-monkeypatch-reviewer.md` and follow it as your complete review specification.

---

## Retired reviewers

- **`cli-extraction-placement-reviewer`** and **`cli-extraction-cross-module-imports-reviewer`** — retired 2026-06-21. They guarded the one-time `cli.py` → `cli/` subpackage extraction (wyrd-g143, now complete). Firing data showed them tailed off with ~0 acted-on findings. The durable lessons live in the `pre-push-extraction-sweep` practice + its wyrd-g143 extension below; the per-round reviewers are spent. Restore from git history if a comparable large extraction starts. `cli-extraction-test-monkeypatch-reviewer` stays — it generalizes to any helper move, not just the g143 extraction.

---

# Practices

Documentation for manual practices around kenning PRs. Unlike `# Agents` entries, these are not spawned as reviewer Tasks by `pr-review-loop` — they describe one-shot operator workflows.

## pre-push-extraction-sweep

For PRs that extract >500 lines from one file into multiple new
modules, run a comprehensive sweep agent BEFORE pushing. The
wyrd-67fv slice D introduced this pattern and caught 5 things up
front (a dead-code drag-in, a stranded constant block, a docstring
inaccuracy, a missing production re-export that would have hit
`ImportError` at collection time, a broken test monkey-patch).

Slices A and B each landed in 1 review round; slice C needed 4
rounds (the "rule/transform" terminology drift kept recurring
across 6 bridges files because no up-front sweep). Slice D, with
the sweep, still needed 5 docstring-fix rounds — the sweep didn't
catch every claim — but the rounds were shorter and avoided the
production-import break the sweep DID catch. The lesson isn't
"sweep replaces review rounds"; it's "sweep catches the cheap
class of problems (missing re-exports, broken monkey-patches,
orphan comments) before they cost a round, leaving the review
loop for the harder docstring-accuracy class".

**When to use:**

* Extraction PRs that touch >500 LOC across >3 new files.
* Subpackage creation (the new `subpackage/__init__.py` is a public
  surface that needs auditing before any external caller sees it).
* PRs that move underscore-prefixed helpers callers depend on.

**Don't bother:**

* Bug-fix PRs, single-file refactors, or anything that fits in a
  ~200-line diff.

**Reusable sweep prompt template:**

```
Pre-push sweep for [refactor description] in [worktree path].

CONTEXT: [what was extracted, why]
NEW FILES: [list each new file + 1-line purpose]
PACKAGE __init__: [path to the back-compat re-export shim, e.g.
                   wyrd/generators/kenning/lexicon/__init__.py]
PRE-EXISTING LESSON: [most recent slice's caught patterns, so
                     the sweep doesn't waste time re-discovering them]

YOUR TASK — comprehensive sweep, no fixes:

1. ORPHAN COMMENTS: scan each new file for comment lines that
   start mid-sentence (lowercase first word, no preceding
   continuation). Extraction artifacts.
2. DEAD CODE / DRAG-INS: identify any function or constant in the
   new files that has NO call site anywhere in the repo (grep
   wyrd/ tests/ for the name).
3. DATACLASS DECORATORS: confirm @dataclass on every class that
   uses field(default_factory=...) or field(default=...).
4. DOCSTRING ACCURACY: for each NEW module docstring, verify every
   factual claim against the code body (named tables/functions/
   output shape match the SQL/dict construction).
5. CROSS-MODULE IMPORTS: verify deferred imports and shared
   helpers resolve through the back-compat re-export.
6. RE-EXPORT-SURFACE: re-exports through the PACKAGE __init__ path
   above must have at least one external caller. Grep production
   code AND tests for each re-exported name; a missing re-export
   is a worse failure than a dead one (the production caller
   ImportErrors at test-collection time, not at import-time).

REPORT: numbered list of concrete findings with file:line refs.
"No issues found" if nothing of substance.
```

This is documented as a practice (not a reviewer) because it runs
once per PR, before review starts, rather than as part of the
review loop's per-round agents.

### Extension from wyrd-g143 (cli.py 10k → 110-line shim, 5 slices)

The cli.py refactor surfaced two new failure modes the slice-D
sweep template above didn't cover. Add these checks to step 1 +
step 2:

7. **AUTO-RANGE HELPER OVERFLOW**: when extraction is driven by
   "decorator → next anchor" line-range detection, the trailing
   `def _helper` definitions between two commands get pulled into
   the PRECEDING command's range — but they may belong to the
   FOLLOWING command (or be shared). Scan each new module's last
   ~50 lines for `def _` defs and decide if they belong with that
   command, with the next one, or in a shared utils.
   Concrete examples from wyrd-g143:
   * slice 3 diff-bundle range auto-grabbed `_append_remove_event`
     (actually a prune-toponym / prune-etymon helper). Range
     hand-overridden to stop at the real end of `lexicon_diff_bundle`.
   * slice 4 fuzzy-search range auto-grabbed
     `_classify_dry_run_row_counts` + `_disambiguate_dispatch`
     (actually disambiguate-fuzzy helpers). Moved post-hoc.

8. **LAMBDA-BIND IN NEW TEST FIXTURES**: when an extraction
   requires updating a test's monkeypatch site, do not introduce
   `_stub = lambda x: ...` style assignments. ruff E731 rejects
   them and CI will fail. Use `def _stub(x): return ...` instead.
   wyrd-g143 slice 5 burned a CI cycle on this — 5 sites in
   `tests/test_kenning_lexicon.py` had the lambda-bind shape after
   the monkeypatch retargeting; the round-1 fix was mechanical.
