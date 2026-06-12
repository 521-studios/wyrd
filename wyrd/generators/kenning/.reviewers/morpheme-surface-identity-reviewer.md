# morpheme-surface-identity-reviewer

Enforces DECISIONS.md **D45**: a morpheme's stored identity is its bare
surface — dashes are NEVER part of a morpheme name, and code NEVER keys
off them. Position (`bare`/`pre`/`inner`/`post`) is derived from
decomposition (D40), carried as an explicit field where a table needs
the axis, and applied as dash decoration only at display/render time
(D39). This rule has been re-explained 5+ times (D39, D40, wyrd-eyjk,
wyrd-zewx, wyrd-c6o1.3) because legacy storage still carries dash-marked
identities; this reviewer exists so NEW code never extends the disease
while epic wyrd-aicu removes it.

**FLAG when the diff introduces:**

1. **A dash-marked surface written into storage as identity** — a new
   DB column value, bundle field, JSONL row, or fixture where a
   morpheme key/name carries positional dashes (`"-ton"`, `"Ton-"`,
   `"-ton-"`). New tables/fields that need the position axis must use
   a `(surface, position)` composite — an explicit position column —
   never position encoded into the surface string.
2. **A lookup keyed by dash-shape** — `.get(usage_key)` /
   `dict[usage_key]` against a meaning_db-style map, or any new map
   built with dash-marked keys, where the dash-shape determines which
   entry matches. During the wyrd-aicu transition, code reading the
   LEGACY dash-marked storage must fold at the boundary
   (`surface.replace("-", "").lower()`) — an exact-key lookup against
   legacy keys is the wyrd-c6o1.3 bug shape (the welsh `ton` homograph
   entered english generation through un-narrowed dash-variant keys).
3. **Matching/eligibility/identity logic branching on dash markers** —
   `startswith("-")` / `endswith("-")` / `strip("-")` used to decide
   what a morpheme IS, what it can match, or whether two things are
   the same morpheme. Position is an OUTPUT of the split (D40), never
   an input to matching.
4. **`Meaning.location` (the dash-decoded hint) used as a gate** —
   it survives only as a render/scoring hint; any new use that filters
   or matches on it re-introduces the wyrd-eyjk P0.
5. **Inconsistent folds** — mixing `strip("-")` with
   `replace("-", "")` for the same fold domain (they differ on
   internal dashes; PR #625 normalized the diversify region's FOLDS
   to `replace` — render-layer `strip`/`lstrip`/`rstrip` for dash
   DECORATION remains and is exempt. New fold code must use
   `replace`).

**Acceptable (do NOT flag):**

* The render layer applying positional decoration + case — D39's
  single owner (`Word`, `NewName.__str__`, explainer surface
  builders, the SPA-facing `usage` display strings).
* Parsers / extractors / miners handling dashes in RAW SOURCE TEXT
  (`parsers/`, `extractors/`, mining CLIs) — document syntax, not
  identity.
* Genuinely hyphenated lexical FORMS in language data (OE compound
  headwords like `lēac-tūn`) — a form's spelling may contain a
  hyphen; a stored morpheme IDENTITY may not carry positional
  markers.
* Boundary folds (`replace("-", "").lower()`) on reads of the
  LEGACY dash-marked storage — the required compat idiom until
  wyrd-aicu deletes the storage dashes and then the folds.
* Tests asserting against the CURRENT legacy storage shape (they
  pin today's behavior; the epic migrates them).

**Review approach:**

1. Grep the diff for `-` inside string literals that flow into keys,
   identity comparisons, or storage writes; for `startswith("-")` /
   `endswith("-")` / `strip("-")` / `replace("-", "")` on surfaces,
   plus dash probes like `"-" in` / `split("-")` / `count("-")`; and
   for `.location` reads.
2. For each hit, classify: render decoration (OK), raw-source parsing
   (OK), legacy-boundary fold (OK if `replace`-style), form-spelling
   data (OK) — or identity storage / keying / matching (FLAG).
3. For new tables, bundle fields, or fixture shapes: check the
   position axis is an explicit field, not a surface decoration.
4. Cite D45 (and D39/D40 where relevant) in every finding; suggest
   the bare-surface + explicit-position alternative concretely.

**Severity convention:** identity storage or matching gates → P1/P2
(this is the recurring bug class); inconsistent folds → P3 with the
concrete rewrite.
