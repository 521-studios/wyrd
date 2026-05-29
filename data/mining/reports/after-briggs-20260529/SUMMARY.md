# AFTER-briggs snapshot — 2026-05-29

Briggs personal-names index ingested ADDITIVELY on top of the after-rebuild
DB (no wipe; after-rebuild DB backed up at
`~/.wyrd/lexicon.db.bak-afterrebuild-20260529`).

## Ingest result
- 8,881 personal names → `personal_name` table
- 15,021 attestations → `personal_name_toponym_attestation` table
- source: `briggs_2024_personal_names_index`; canonical JSONL committed
  (`data/mining/briggs_2024_personal_names_index.jsonl`, replayable).

## KEY FINDING — briggs has NO bundle/generation effect yet

The after-briggs `export-runtime-db` produced a bundle **functionally
identical** to after-rebuild: 74,501 meaning rows / 31,817 proportion rows
(same as after-rebuild). The byte-level diff was only SQLite page churn from
the DB writes; the committed after-rebuild bundle was restored.

**Why:** briggs writes to dedicated `personal_name` /
`personal_name_toponym_attestation` tables — a SEPARATE layer from `etymon`.
The bundle export (export-meanings / export-runtime-db) reads `etymon`, NOT
`personal_name`. So the briggs layer is correctly ingested + durable but has
**no consumer** wiring it into the bundle or generator.

**"Briggs on its own" = no generation change** until a consumer surfaces the
personal_name layer. Filed as a follow-up ticket (how to surface — promotion
path vs saint-style sidecar vs explainer-only — is a design decision for the
operator, not auto-decided overnight).

The after-rebuild bundle (committed) remains the current correct runtime
bundle. No after-briggs report batch was run (it would be identical to
after-rebuild).
