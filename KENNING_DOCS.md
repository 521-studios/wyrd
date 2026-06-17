# Reading the kenning docs — a guide for future Claude

You (a fresh session) were probably told "read KENNING_DOCS.md." Good. The
point of this file is to stop you from blindly reading every `.md` under
`wyrd/generators/kenning/` — that's ~80KB of prose, and `DECISIONS.md` alone
is 3,100+ lines / ~63k tokens. You almost never need all of it. Read what the
task needs.

All kenning docs live in **`wyrd/generators/kenning/`**. This file is just the
router.

---

## Always read first (cheap, ~10 min, frames everything)

- **`OVERVIEW.md`** — the trunk. The two-layer split (authoring vs runtime),
  why we mine, the three-tier provider model, the mental model. Every other
  doc is a leaf hanging off this. If you read exactly one doc, read this one.

Then orient on **live state**, not doc snapshots (the docs carry dated
snapshots that go stale — treat them as history, not current truth):

```bash
bd prime && bd ready                  # what's queued
wyrd kenning lexicon report           # current corpus snapshot
```

---

## Then read by task

| You're about to… | Read |
|---|---|
| Mine a new etymology source | `INGESTION.md` (the procedure manual — only needed when actually mining) |
| Touch the matcher / decomposition / morpheme keys / bundle export | **`DECISIONS.md` D40 + D45** (non-negotiable — see below), then the specific D-entry you're changing |
| Touch era / time-axis / rewind / era-map | `DECISIONS.md` D44 + D46 (+ D33 for the reflex machinery) |
| Touch scoring / generation knobs / moods | `DECISIONS.md` D36 (vector architecture) + D37 (register catalog); `REGISTERS.md` only if you're tuning per-effect weights against the phonaesthetics literature |
| Touch the cultural-zone axis (Danelaw / Celtic / Roman) | `DECISIONS.md` D52 (rule-based zone axis, distinct from the dedup region hierarchy) + D32 (stratum, the sibling axis) + D49 (two-purposes split). Data model: `data/zones.yaml` + `lexicon/zone_classify.py` (generator knob / runtime / SPA still deferred) |
| Touch rendering / native-vs-modern surface | `DECISIONS.md` D41 + D39 |
| Clean up / canonicalize the corpus, merge duplicates, normalize region/source dims | **`DECISIONS.md` D49** (the two-class partition) + **D50** (the Class-B assertion layer: synthetic nodes + edge taxonomy), then D22 (the proto-merge pattern) + D21 (evidence sacred) |
| Do a full DB wipe + rebuild | `REBUILD.md` (read it BEFORE any `rebuild-from-jsonl --with-enrichment` — it lists the L3-only layers a wipe silently drops) |
| Move a table/column between layers, or wonder "is this rebuildable?" | `L2_L3_BOUNDARY.md` |
| Wonder where runtime data lives / why it's SQLite-on-S3 | `DECISIONS.md` D38 |
| Open a kenning PR | `AGENT-REVIEWERS.md` (but `pr-review-loop` consumes it — don't hand-spawn) |
| Just curious about decomposition coverage trends | `COVERAGE.md` (a tripwire gauge, NOT the north star) |

`README.md` is the runtime-perspective sibling of OVERVIEW — module map, knob
list, CLI cheatsheet, bundle schema. Skim it if you need the CLI surface or the
module layout; skip it if OVERVIEW already answered your question.

---

## The invariants worth knowing cold (these get re-explained every few sessions)

If you internalize nothing else, internalize these. Most of the repeated pain
in this subsystem is one of these getting violated:

1. **Dashes are NEVER morpheme identity. Position is DERIVED, never matched**
   (D40 + D45, enforced by `morpheme-surface-identity-reviewer`). A morpheme's
   identity is its bare surface (`ton`, `giles`). `pre-`/`-inner-`/`-post` is an
   *output* of decomposition and a *display-time* decoration — never an input to
   matching, never stored in the name, never a lookup key. This has been
   corrected 5+ times across sessions. If you find yourself gating a match on a
   stored dash-shape, STOP — that's the recurring bug.

2. **Two strict layers.** Authoring (`~/.wyrd/lexicon.db`, SQLite — mine, cite,
   enrich) is invisible to runtime. `export-runtime-db` ships the L4 bundle
   (SQLite-on-S3, D38). The authoring DB can be re-shaped freely; the runtime
   sees nothing until export.

3. **Four data layers.** L1 raw wiktextract → L2 curated JSONL (git, source of
   truth) → L3 SQLite (rebuildable build artifact) → L4 runtime bundle. Mining
   evidence is sacred (D21) — enrichment is reversible (redirect columns + method
   stamps), never destructive.

4. **Vector is the only scorer** (D36). The old proportions *scoring* mode is
   retired; the proportions *data* survives as the vector path's empirical input.
   Don't reach for `--scoring-mode` — it's gone.

5. **Era selects the reflex, never the morpheme** (D44) — but pools *accrete*:
   "in the record by then" (D46; no Silicon on a 1086 map, but `tūn` on every
   map). Era changes the rendered surface and gates positively-dated late
   arrivals; it never re-weights the draw.

6. **Native is canonical, modern is the always-present secondary** (D41).
   `era=""` renders as-selected (native), NOT coerced-to-modern.

7. **Pre-launch posture** (also in auto-memory): prod has no users yet, so
   bit-stability is subordinate to product correctness (D41/D43/D46 all
   deliberately break seed→name parity). Don't gate a correctness fix on
   "but it changes seeded output."

8. **Corpus uncleanness is TWO problems, not one** (D49). *Class A* —
   controlled-vocabulary coding (region/country/lang/source): same value, many
   spellings → canonical enum + alias map, normalized + validated fail-closed at
   the JSONL boundary (git = provenance), no rationale needed. *Class B* —
   scholarly entity resolution (morpheme≈morpheme, place≈place, spurious
   breakdown): an append-only assertion log (L2) projected to the L3 collapse
   graph, generalizing `merged_into_id`/`cognate_id` — every edge carries
   confidence + rationale, identity-collapse defaults to leave-separate. Don't
   solve a Class-A coding mess with the Class-B machinery, or vice-versa. Still
   no graph DB engine (D42). The Class-B mechanism is specified by **D50**:
   **synthetic canonical nodes** (mint + bind, NOT winner-redirect — identity is
   separate from evidence and survives label-drift-by-era), a typed edge taxonomy
   (identity / relational / canonicalization-choice, sorted by the collapsibility
   test), and an append-only assertion record with affirm/refute/retract polarity.

9. **Mine morpheme facts from scholarly breakdowns, NEVER surface segmentation**
   (D51, the gloss-blind rule). A composite's expansion or a reflex pairing must
   come from what scholars actually attributed, because segmentation is
   gloss-blind: `barton` segments to `bar`+`ton` but the scholar's breakdown is
   `bere`+`tūn` (barley-farm). The `toponym_etymology` corpus is triple-duty —
   morpheme-admission evidence (wyrd-oth3), the decomposer's labeled test corpus
   (the grader; the regression list is the overfitting tripwire, and
   coverage-up-while-agreement-down is a real failure mode), and the gloss-correct
   mining source for passthroughs (`composed-of`, D50.3) + implied reflexes. A
   composite morpheme is matched but ATTRIBUTED to its constituents
   (surface ≠ attribution, like the connective) — never recorded opaque.

---

## Don't bother reading (unless you have a specific reason)

- **`archive/`** (`DEPRECATED.md`, `superseded-decisions.md`) — intentionally
  *outside* the working set. Only for archaeology ("what did D17 used to say?").
  Superseded DECISIONS entries are one-line tombstones pointing here.
- **`.reviewers/*.md`** — full reviewer specs. `AGENT-REVIEWERS.md` summarizes
  them; you only open an individual spec when running that reviewer.
- **`data/seed-runtime.README.md`** — only if you're regenerating the committed
  dev seed.

---

## How to read DECISIONS.md without drowning

Don't read it linearly. It's a reference, not a narrative. Each entry is a
self-contained decision + rationale with a stable `D<N>` anchor. Workflow:

```bash
grep -n "^## D" wyrd/generators/kenning/DECISIONS.md   # the table of contents
```

Then `Read` with an offset to the entry you need. Watch for
`— SUPERSEDED <date> by D<N>` tombstones and `> **Corrected by D<N>**`
banners — newer entries override older ones, and the cross-references are
reliable. When two entries disagree, the higher number wins.

---

## Meta: keep this file honest

When you add a new top-level kenning doc, or a DECISIONS entry that changes one
of the invariants above, update this router. It earns its keep only if telling
a fresh session "read KENNING_DOCS.md" actually saves them from reading
everything — which means it has to stay accurate about what's worth reading.
