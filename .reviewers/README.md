# .reviewers/

Per-agent review specifications spawned by the `pr-review-loop` skill.
Each `<name>.md` is the complete review spec for the agent of that name in
`../AGENT-REVIEWERS.md`. The root file's `# Agents` H2 bodies are intentionally
one-line pointers — the main `pr-review-loop` process never needs to read the
full spec, because each reviewer spawns as its own Task that reads its file directly.

## Why each default is disabled

The root `# Configuration` block in `../AGENT-REVIEWERS.md` lists three baked-in
defaults under `disabled`. The rationale lives here so the active configuration
stays terse:

- **`silent-failure-hunter`** — covered by `error-handling-reviewer`. Our reviewer flags silent exception swallows (P1), generic `except Exception` without re-raise (P2), bare `except:` (P2), and missing `raise ... from` (P3) with detailed BAD/GOOD examples. The default's patterns are a strict subset.
- **`comment-analyzer`** — covered by `clarity-reviewer`. The clarity-reviewer's docstring grep-verify section explicitly checks factual accuracy of every named symbol/table/regex count/output key against the function body, plus comment-rot and value-free comment patterns.
- **`code-simplifier`** — covered by `complexity-reviewer` (nesting depth, parameter count, redundant wrappers, generic naming, nested ternaries) and `dead-code-reviewer` (commented-out code, `if False:`, `pass`-only stubs) combined. Patterns from code-simplifier that weren't in our reviewers (nested ternaries, redundant single-call wrappers, generic naming) have been added to `complexity-reviewer`.

The defaults `code-reviewer` (CLAUDE.md compliance + significant bugs) and `type-design-analyzer` (encapsulation, invariant design) are kept as-is — they cover scopes our pack does not.

