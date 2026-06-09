# Deprecated / superseded kenning docs — ledger

This `archive/` folder holds kenning documentation that is **no longer valid**
(superseded decisions, retired approaches). It exists so that "read all the
docs in kenning" does **not** load dead ideas into context — when you point an
agent at the kenning docs, point it at the files in
`wyrd/generators/kenning/*.md` and **skip this `archive/` folder**.

This ledger logs every superseded decision/doc: **what** it was, **when** it
was superseded, **what replaced it**, and **where the body now lives**.

## Convention — how to retire a decision

When a `DECISIONS.md` entry (or any kenning doc) becomes invalid:

1. **Move** its full body verbatim into `archive/superseded-decisions.md`
   (keep the original `## DN.` heading so search cross-references still land),
   prefixed with a `> **SUPERSEDED <date> by <DN / replacement>**` note.
2. **Stub** the entry in `DECISIONS.md` down to one line: the heading + a
   `— SUPERSEDED <date> by <DN>; body archived → archive/superseded-decisions.md`
   pointer (plus a half-line on any live residue so the reader isn't left
   wondering what survived).
3. **Log** it in the table below.

The point is that `DECISIONS.md` stays a list of *current* decisions (each dead
one shrinks to a one-line tombstone), and the archaeology lives here, out of the
default reading path.

## Ledger

| Decision / doc | Superseded | Superseded by | Moved to | Live residue (still current) |
|---|---|---|---|---|
| **D17.** Bayesian mixture is the novelty knob | 2026-05-18 | D36 (vector-driven generator) | `archive/superseded-decisions.md` | `--cohesion` → vector-scorer multiplier; `--novelty` re-wired onto the vector path (wyrd-fcub) |
| **D17 refinement.** cohesion knob (wyrd-mj2) | 2026-05-18 | D36 (vector-driven generator) | `archive/superseded-decisions.md` | cohesion math carries over to the vector scorer's multiplier; the tag-co-occurrence *data* (D16) still feeds it |

## Reviewed and KEPT (not archived)

Decisions that mention proportions but remain **load-bearing** — left in
`DECISIONS.md` deliberately:

- **D16.** Co-occurrence lives at the tag level — the `tag_cooccurrence` /
  `tag_marginal` **data** it defines still feeds the vector scorer's cohesion
  multiplier (`runtime/vector_name_select.py`). Only the proportions *sampler*
  that originally consumed it is gone.
- **D36.** Vector-driven generator architecture — this **is** the current
  scoring path; it documents the proportions retirement (do not archive).

## Candidates for archival (audited, not yet moved)

Identified during the 2026-06-09 doc audit; queued for the same treatment once
the convention above is confirmed:

- **D31 Phase 2a / Phase 2d** sub-sections (`DECISIONS.md`) — already carry
  inline `> **SUPERSEDED by D38 (wyrd-24s6)**` banners; their bodies still sit
  in `DECISIONS.md` and should move here under the convention.
- **Operational-doc proportions framing** — `INGESTION.md`, `REBUILD.md`,
  `README.md` ("Extending" §), and the repo-root `README.md` present
  `rebuild-proportions` / `<culture>_proportions.json` as the *scoring*
  mechanism. The proportions **data** pipeline (structures + tag_cooccurrence)
  is still a live vector-scorer input, so these need a *reframe* ("proportions
  is a data feed for the vector scorer, not a scorer") rather than archival —
  tracked separately from this decision-archive work.
