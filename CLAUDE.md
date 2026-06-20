# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Git Worktrees

**Do non-trivial work in a git worktree, not the main checkout.** Create one
with `git worktree add` and work there. The only exceptions — fine to do in
place on `main` — are small, self-contained changes: beads
(`.beads/issues.jsonl`) syncs and small docs edits (a typo, a README/CLAUDE.md
tweak like this one).

Why: the commit-and-push flow stages with `git add -A`, so working in the main
checkout risks sweeping unrelated untracked files into a PR. An isolated
worktree keeps each PR scoped to exactly its own change.

**Never run `commit-and-push.sh` (or any `git add -A` commit) from the main
checkout, and never sit in the main checkout while running `bd` mutations
(`create`/`close`/`update`/`dep`).** `bd` rewrites `.beads/issues.jsonl` on
every mutation, so a later `git add -A` from the main checkout sweeps those
edits — plus anything else dirty in the tree — onto `main` without a deliberate
decision (it "ships" as you type). Keep the main checkout **idle**: do real work
in a worktree, and run `bd` ticket ops as standalone steps where you then
explicitly commit *only* `.beads/issues.jsonl` (`git commit -- .beads/issues.jsonl`).
Always check `git status` before any `git add -A`.

## Build & Test

**Run ruff before every push.** CI's "Test & Lint" job gates on
`ruff format --check .` and `ruff check .`, but `commit-and-push.sh`'s
pre-commit hook does NOT run ruff in this environment (ruff/pre-commit aren't
installed locally), so a format/lint slip passes locally and fails CI — and
has merged red to `main` that way. Always run, and confirm clean, before
pushing:

```bash
ruff format .          # auto-format (or `ruff format --check .` to verify only)
ruff check .           # lint
pytest -n auto         # full suite (pytest-xdist; -n auto ≈ CPU count)
```

A green from `check-ci.sh` immediately after a push can be stale (it races the
new commit's check-runs and may report the previous commit's result). Before
merging, verify the **actual HEAD SHA's** checks:

```bash
HEAD=$(gh pr view <pr> --json headRefOid --jq .headRefOid)
gh api /repos/521-studios/wyrd/commits/$HEAD/check-runs --jq '.check_runs[]|{name,conclusion}'
```

## Architecture Overview

_Add a brief overview of your project architecture_

### Working on kenning?

**If the task touches kenning (the place-name generator under
`wyrd/generators/kenning/`), read [`KENNING_DOCS.md`](KENNING_DOCS.md) first.**
It's a router that tells you which of the kenning docs are actually worth
reading for your specific task — don't blindly read all of
`wyrd/generators/kenning/*.md` (that's ~80KB of prose; `DECISIONS.md` alone is
~63k tokens). The router also lists the load-bearing invariants (dashes are
never morpheme identity, the two-layer split, vector-only scoring, era
accretion, …) that get re-explained every few sessions.

## SPA Feature Flags

The `spa-next/` config UI gates each advanced option behind a feature flag so
untested options stay hidden in production and can be turned on one-by-one as
validated (wyrd-0gou). This is **UI-only** — the API stays permissive; flags
only decide what the SPA renders. Resolution is env-driven and resolved
**per request** onto `/api/manifest` (`config` block), so flipping a Lambda env
var takes effect with **no rebuild/redeploy**. Server logic:
`wyrd/feature_flags.py`; SPA mapping: `spa-next/src/lib/featureFlags.js`.

### Env vars

| Var | Effect |
|-----|--------|
| `WYRD_FF_ALL=true` | Master override — every flag on. **Staging sets this** so a forgotten flag can't hide an option. |
| `WYRD_FF_<NAME>=true\|false` | Per-flag toggle. **Default off.** `NAME` = the flag name upper-cased with `.`/`-` → `_`. |
| `WYRD_DEFAULT_<OPTION>=<value>` | Override an option's **default value** (not just visibility). `OPTION` = the option name upper-cased with `.`/`-` → `_`; the server lower-cases it back to the snake_case field key. |

Resolution is fail-closed: `flagOn = WYRD_FF_ALL OR WYRD_FF_<name>`; an absent or
unknown flag is **off**. The canonical flag-name list lives on the SPA side
(`featureFlags.js`); the server just resolves whatever `WYRD_FF_*` /
`WYRD_DEFAULT_*` it sees, so adding a flag = a new env var + a new SPA mapping
entry (no server change).

### Flag names

- **Advanced knobs** — 1:1 with the schema field key: `WYRD_FF_NOVELTY`,
  `WYRD_FF_ERA`, `WYRD_FF_STRATUM`, `WYRD_FF_PACKS`, `WYRD_FF_PRIORS_PATH`,
  `WYRD_FF_COHESION`, … When all advanced flags are off the SPA's Advanced
  panel doesn't render at all.
- **`WYRD_FF_SCORING_MODE`** — gates the scoring-mode selector **and** its
  vector axis-weight fields (`phonological_weight`, `semantic_weight`,
  `position_weight`, `baseline_weight`) as a unit.
- **Cultures** — `WYRD_FF_CULTURE_<NAME>` for non-english cultures
  (`WYRD_FF_CULTURE_WELSH`, `…_SCOTTISH`, `…_IRISH`, `…_BRETON`). **English is
  always available** and can't be gated off.
- **Composer** — `WYRD_FF_MOODS` and `WYRD_FF_TAGS` gate the two halves of the
  "Customize moods & tags" composer independently.
- **Transforms** — `WYRD_FF_REWIND` gates the **Rewind** pipeline transform
  (wyrd-nwpa). Off by default, so prod hides Rewind while its era-rendering
  bugs are fixed; the palette won't offer it AND the pipeline skips any
  saved/shared Rewind step. A transform tags itself with `flag: '<name>'` in
  `spa-next/src/lib/transforms/*.js`; `TransformPalette` + `pipeline.run`
  filter on it. Re-enable in prod via `enabled_feature_flags = ["rewind"]`.

Default-value examples: `WYRD_DEFAULT_CULTURE=english`, `WYRD_DEFAULT_COUNT=5`,
`WYRD_DEFAULT_ERA=present-day` (the culture-agnostic token for each culture's
present-day stage — the deployed default in both envs, wyrd-kqyf),
`WYRD_DEFAULT_SCORING_MODE=vector` (`vector` is the only live scorer — the
`proportions` mode is retired, D36).

### Terraform (`terraform/`)

Staging gets `WYRD_FF_ALL=true` automatically via `var.env == "staging"`;
production defaults all-off. To enable validated options + override defaults
per environment:

```hcl
enabled_feature_flags = ["novelty", "culture.welsh", "moods"]  # → WYRD_FF_<NAME>=true
feature_flag_defaults = { culture = "english", count = "5" }   # → WYRD_DEFAULT_<OPTION>
```

`var.env` is validated to `staging` | `production` (a typo would otherwise
silently deploy all-flags-off).

## Conventions & Patterns

### Mining progress reporting

**Every long-running mining/ingestion CLI MUST emit a periodic
progress line.** The canonical shape (set by `lexicon mine-llm`):

```
  [completed/total]  <key>=N <key>=N (<rate>s/entry)
```

- Echo every ~10 records (or every file for file-walking ingesters).
- Always echo a final line at completion so the last partial chunk
  shows up.
- Print to stderr (`click.echo(..., err=True)`) so JSONL/data output
  on stdout stays clean.
- When wall-clock budget matters, include the `s/entry` rate so
  operators can extrapolate ETA.

Following the pattern: `lexicon mine-llm`,
`lexicon mine-fantasy-name`, `lexicon ingest-etymonline`. Lacking
per-record `[N/total]` progress (retrofit candidates):
`lexicon mine-wiktextract-corpus`, `lexicon mine-skeat`,
`lexicon review`. Silently waiting 20+ minutes with no signs of life
is a debuggability hole.
