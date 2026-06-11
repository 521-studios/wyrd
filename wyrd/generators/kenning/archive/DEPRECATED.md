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

Decisions that carry a supersession marker but remain **load-bearing** — left in
`DECISIONS.md` deliberately. The distinction that matters: a whole-entry
supersession (the implementation it describes is entirely gone, like D17) is
archived; a **partial** supersession (one specific claim inside an otherwise-live
decision, already corrected by an inline `> **SUPERSEDED by …**` banner) is
**kept in place** — the banner is the right tool, and moving the body would
strip live documentation.

- **D16.** Co-occurrence lives at the tag level — the `tag_cooccurrence` /
  `tag_marginal` **data** it defines still feeds the vector scorer's cohesion
  multiplier (`runtime/vector_name_select.py`). Only the proportions *sampler*
  that originally consumed it is gone.
- **D36.** Vector-driven generator architecture — this **is** the current
  scoring path; it documents the proportions retirement (do not archive).
- **D6.** Languages are morally-neutral palette options — LIVE (the
  morally-neutral-palette principle + the `--mood` system ship today). Only the
  5-entry preset *list* is superseded (D37's 12-entry phonaesthetic-vector
  catalog), already marked by an inline banner. Keep.
- **D31.** Multi-script renderings (Phase 2a–2d) — LIVE (the renderings schema,
  bundle siblings, and SPA provenance panel all ship). Only two specific claims
  ("per-language rendering is the demos' concern"; "generation default is
  modern_usage everywhere") are superseded by D41 (wyrd-24s6), already marked by
  inline banners. Keep — NOT an archival candidate despite the markers.

## Follow-ups (not archival)

- **Operational-doc proportions framing** — `INGESTION.md`, `REBUILD.md`,
  `README.md` ("Extending" §), and the repo-root `README.md` present
  `rebuild-proportions` / `<culture>_proportions.json` as the *scoring*
  mechanism. The proportions **data** pipeline (structures + tag_cooccurrence)
  is still a live vector-scorer input, so these need a *reframe* ("proportions
  is a data feed for the vector scorer, not a scorer") rather than archival —
  tracked separately from this decision-archive work.
