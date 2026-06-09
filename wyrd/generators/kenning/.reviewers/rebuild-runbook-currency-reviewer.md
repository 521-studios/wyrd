# rebuild-runbook-currency-reviewer

Review changes that add or rename a **data-population** path in the
kenning lexicon for **rebuild-runbook currency**. **Default severity:
P2** (P1 if the new layer is mined and has no documented restore path —
that's a silent-data-loss-on-wipe class of failure).

## The rule this enforces

A from-scratch rebuild is `rm ~/.wyrd/lexicon.db && lexicon
rebuild-from-jsonl --with-enrichment` followed by a sequence of manual
restore steps for the layers a wipe drops. That sequence lives in
`wyrd/generators/kenning/REBUILD.md`, with the structured truth in
`data/mining/_rebuild_layers.json`. The May-2026 rebuild was painful
because new L3-only layers (empirical, forms-variants, attestations,
`toponym.country`) had accreted over months with **no single complete
restore sequence** — each was rediscovered by a failing test.

This reviewer keeps that from recurring. It is the doc-currency
complement to `db-reconstructibility-reviewer` (repo root):

- **db-reconstructibility-reviewer** asks *"can this data be recovered at
  all, without paying for mining again?"* (→ must round-trip through L2,
  or be free-rebuildable).
- **this reviewer** asks *"is the recovery either automatic (L2 replay /
  enrichment) or written down as a step in the rebuild runbook?"*

The distinction that motivated this reviewer: **free** re-runnable mining
(e.g. `mine-wiktextract-corpus` / `mine-wiktextract-forms` against the
local bulk slices — no API/Ollama cost) is an *acceptable manual rebuild
step* — but only if it is registered in `_rebuild_layers.json` and
documented in REBUILD.md. **Paid** mining (LLM / Gemini / Ollama API
cost) must never be a rebuild step; its output must round-trip through L2
(that part is db-reconstructibility-reviewer's job).

## Scope (touch-it-you-own-it)

In scope when the PR does any of:

- Adds or renames a `mine-*` / `ingest-*` / `backfill-*` / `cleanup-*`
  subcommand under `wyrd/generators/kenning/cli/lexicon/` — or a
  data-writing whole-word / subgroup command the test enumerates by hand
  (`build`, `synsets seed`, `synsets assign`; add new such commands to
  `_EXTRA_DATA_POPULATION_COMMANDS` in the test).
- Adds a new table or layer (schema migration / `data/seed/lexicon.sql` /
  `schema.py`) whose rows are **mined or ingested** (not deterministically
  derivable by `lexicon enrich`), i.e. something a full wipe would drop
  unless restored.
- Edits `data/mining/_rebuild_layers.json`, `REBUILD.md`, or
  `tests/test_kenning_rebuild_runbook.py`.

Out of scope: read-only paths (report / browse / export / drift), pure
enrichment columns derivable from existing DB rows, and tests/fixtures.

## What to check

1. **Every new data-population command is categorized** in
   `data/mining/_rebuild_layers.json` — either in `rebuild_step_commands`
   (and documented in REBUILD.md) or in `non_rebuild_commands` with a
   one-line reason. `tests/test_kenning_rebuild_runbook.py` enforces that
   it's categorized *somewhere*; your job is to confirm it landed in the
   **right** bucket.

2. **The reason is accurate, not a rubber stamp.** The test only checks
   the reason string is non-empty. Read it. Common misclassifications to
   flag:
   - A **paid** command parked in `non_rebuild_commands` with a reason
     implying it's a rebuild step, or worse, added to
     `rebuild_step_commands`. Paid mining is never a rebuild step.
   - A command whose output is **not** actually round-tripped through L2
     allowlisted as "output lands in L2, replayed by rebuild-from-jsonl"
     — verify the L2 emit / `dump-jsonl` path actually exists (this is
     the seam db-reconstructibility-reviewer also guards).

3. **Free rebuild steps are documented in order.** If the command is in
   `rebuild_step_commands`, confirm REBUILD.md's Phase 2 names it AND
   places it correctly relative to its dependencies (e.g.
   `backfill-toponym-country` must precede `mine-empirical-baselines`;
   the empirical mine needs an interim export first).

4. **A new mined L3-only layer has a `layers` entry** in the manifest
   (with `restore_command` / `runbook_token`) AND a corresponding restore
   step in REBUILD.md. A new table that a wipe drops with no manifest
   layer + no runbook step is the headline failure this reviewer exists
   for.

5. **The L2_L3_BOUNDARY.md table is updated** when a layer moves between
   L2 and L3, or a new L3-only layer is added (its "Deferred (planned L2)"
   list is the canonical map).

## Flag issues if

- A new mined/ingested layer (new table, or a column whose value is mined)
  has no restore path in REBUILD.md and no `layers` entry in the manifest
  — i.e. a full wipe would silently drop it.
- A **paid** mining command is listed in `rebuild_step_commands`, or is
  allowlisted with a reason that misrepresents it as free / L2-backed.
- A free rebuild step is in `rebuild_step_commands` but absent from (or
  mis-ordered in) REBUILD.md.
- A `non_rebuild_commands` reason claims "round-trips through L2" but the
  ingester has no JSONL emit / `dump-jsonl` coverage (cross-check with
  db-reconstructibility-reviewer).
- A manifest layer's `runbook_token` doesn't actually appear in REBUILD.md
  (the test catches the literal-token case, but flag near-misses where the
  token was renamed in prose but not in the manifest).

## Do NOT flag

- Read-only / structural commands (`report`, `export-runtime-db`,
  `enrich`, `decompose`, `diff-*`) — they're always part of the runbook
  prose and not in the auto-discovery prefix set.
- Pure enrichment columns derivable via `lexicon enrich --apply` — those
  are restored by `--with-enrichment`, no manual step needed.
- Source-specific one-off ingesters that genuinely round-trip through L2
  and are correctly allowlisted with an accurate reason.
- Renames where both the manifest entry and REBUILD.md were updated in the
  same PR.

## Review approach

1. Grep the diff for new/renamed Click commands under `cli/lexicon/`
   (`add_to(`, `@click.command(`) matching the four prefixes.
2. For each, open `data/mining/_rebuild_layers.json` and confirm it's
   categorized; judge whether the bucket + reason are correct.
3. If it's a rebuild step, grep REBUILD.md for the command name and check
   placement/order in Phase 2.
4. For new tables, confirm a `layers` entry + a REBUILD.md restore step
   exist, and that L2_L3_BOUNDARY.md reflects the L2/L3 placement.
5. State the one-sentence restore recipe for the new layer. If it's "rerun
   this free command, documented in REBUILD.md step N" or "replayed from
   L2", it's fine. If it's "nobody wrote it down," that's the finding.

## Recent context this reviewer would have caught

The empirical / forms-variants / attestations / country layers were all
L3-only and silently dropped by `rebuild-from-jsonl`; the gap surfaced
only mid-rebuild as failing tests (`test_bundle_canonical_morphemes`
16-fail on the dropped reflex layer; spelling-variety test on the dropped
forms layer). Reflexes + fantasy morphemes were subsequently moved to L2
round-trip (`_reflexes.jsonl` / `_fantasy_morphemes.jsonl`); the four
free-mining layers above are documented rebuild steps. This reviewer
keeps the next such layer from accreting undocumented.
