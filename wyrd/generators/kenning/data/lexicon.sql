-- Kenning lexicon: a normalized store for place-name etymology mined from
-- multiple scholarly sources. Authoritative for authoring; the runtime still
-- reads the derived meanings.json view exported by `wyrd kenning lexicon
-- export-meanings`.

PRAGMA foreign_keys = ON;

-- A bibliographic record for an etymological claim.
CREATE TABLE source (
  id              TEXT PRIMARY KEY,
  author          TEXT,
  title           TEXT NOT NULL,
  year            INTEGER,
  region          TEXT,
  language_focus  TEXT,
  notes           TEXT
);

-- A historical morpheme. (canonical_form, language) is the natural key, so
-- when two scholars cite the same Old English "ham" they hit the same row.
--
-- lemma_id (nullable, self-FK) links inflected or spelling-variant etymons
-- to their canonical lemma. Example: "cotan" (genitive), "cotes" (genitive),
-- and "cotum" (dative pl) all point lemma_id → "cot". The consensus view
-- aggregates citations across all etymons sharing a lemma. This matters
-- because the rando-port seed (Wikipedia-derived) gives uninflected lemma
-- forms while scholarly mining gives inflected real-world forms — so the
-- same morpheme often arrives as multiple etymon rows.
CREATE TABLE etymon (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  canonical_form  TEXT NOT NULL,
  language        TEXT NOT NULL,
  modifier_type   TEXT,
  position_pref   TEXT CHECK (position_pref IN ('pre', 'post', 'inner', 'free')),
  notes           TEXT,
  lemma_id        INTEGER REFERENCES etymon(id) ON DELETE SET NULL,
  -- When this etymon is an inflected variant of a lemma, this column says
  -- WHICH inflection. Examples: 'genitive', 'dative_pl', 'plural',
  -- 'weak_oblique'. Null on lemmas themselves and on uninflected etymons.
  -- Lets generation compose <lemma>@<inflection> → look up or derive the
  -- surface form.
  inflection      TEXT,
  -- Which version of link-lemmas produced lemma_id/inflection on this row.
  -- Lets a future link-lemmas-v2 selectively clear and rebuild only its
  -- predecessor's work.
  lemma_method    TEXT,
  -- Non-destructive OCR clustering (D22). When cluster_ocr_variants
  -- merges this row into a canonical winner, this points at the winner
  -- (and the row is otherwise left intact). The etymon_consensus view
  -- rolls citations/glosses up via this column. Reset to NULL via
  -- `wyrd kenning lexicon clear-enrichment --stage=ocr --apply` to
  -- re-run clustering with new heuristics.
  merged_into_id  INTEGER REFERENCES etymon(id) ON DELETE SET NULL,
  UNIQUE (canonical_form, language)
);
CREATE INDEX idx_etymon_lemma       ON etymon(lemma_id);
CREATE INDEX idx_etymon_merged_into ON etymon(merged_into_id);

CREATE TABLE etymon_gloss (
  etymon_id INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  gloss     TEXT NOT NULL,
  PRIMARY KEY (etymon_id, gloss)
);

CREATE TABLE etymon_tag (
  etymon_id INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  tag       TEXT NOT NULL,
  PRIMARY KEY (etymon_id, tag)
);

CREATE TABLE etymon_citation (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  etymon_id   INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  source_id   TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
  page        TEXT,
  short_quote TEXT
);
-- SQLite forbids expressions in inline UNIQUE constraints, so the
-- "treat NULL page as ''" uniqueness goes here as a partial index instead.
CREATE UNIQUE INDEX idx_etymon_citation_unique
  ON etymon_citation(etymon_id, source_id, COALESCE(page, ''));

-- Looser-evidence text attestations: a morpheme's canonical form appears as
-- a word-boundary substring in a source book's body text. Distinct from
-- etymon_citation (which represents EXTRACTION evidence — a scholar
-- formally identifying the morpheme as part of a toponym's etymology).
-- Both are useful but carry different evidence weight, so they live in
-- separate tables. The etymon_consensus view does NOT include text matches,
-- so consensus stays a measure of extraction-witness count.
CREATE TABLE etymon_text_match (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  etymon_id   INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  source_id   TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
  -- The form actually found in the text. For exact matches this equals the
  -- normalized etymon.canonical_form. For fuzzy matches this is the variant
  -- spelling found in the source — e.g., rando has 'denu', text has 'dene',
  -- edit_distance=1, matched_form='dene'.
  matched_form  TEXT NOT NULL,
  match_count   INTEGER NOT NULL,
  edit_distance INTEGER NOT NULL DEFAULT 0,  -- 0 = exact, 1-2 = fuzzy
  snippet       TEXT,
  -- Which heuristic produced this row: 'reverse-search-v1',
  -- 'fuzzy-search-v1', 'llm-disambiguator-v1', etc. Lets a future rebuild
  -- selectively clear-and-regenerate by method.
  method        TEXT NOT NULL DEFAULT 'reverse-search-v1',
  UNIQUE (etymon_id, source_id, matched_form)
);
CREATE INDEX idx_etymon_text_match_etymon ON etymon_text_match(etymon_id);
CREATE INDEX idx_etymon_text_match_source ON etymon_text_match(source_id);

-- Per-(book, provider, model, run) audit log. The mine-llm and review CLI
-- flows compute accept/decline/reject counts in-memory and used to print
-- them only to stdout — once the process exits the numbers are gone. Per
-- D23 this table closes that gap. Rows are written at end-of-run by the
-- mining writers and via `wyrd kenning lexicon import-mining-log` for
-- back-filling historical batches recovered from session transcripts.
CREATE TABLE mining_run (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id     TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
  -- Provider tier identity. Free-text 'unknown' is allowed for back-fill
  -- rows where the transcript context didn't pin it down.
  provider      TEXT NOT NULL,
  model         TEXT NOT NULL,
  -- 'mine' = Tier-1 mine-llm; 'review' = Tier-2 lexicon review pass.
  mode          TEXT NOT NULL CHECK (mode IN ('mine', 'review')),
  started_at    TEXT,
  completed_at  TEXT,
  parsed_count  INTEGER NOT NULL DEFAULT 0,
  accepted      INTEGER NOT NULL DEFAULT 0,
  declined      INTEGER NOT NULL DEFAULT 0,
  rejected      INTEGER NOT NULL DEFAULT 0,
  -- JSON object: {form_not_in_body: 5, bad_language: 2, ...}
  by_failure    TEXT,
  notes         TEXT,
  -- Idempotent on (book, provider, model, mode, completed_at) so the
  -- same run can be back-filled twice without duplicating.
  UNIQUE (source_id, provider, model, mode, completed_at)
);
CREATE INDEX idx_mining_run_source    ON mining_run(source_id);
CREATE INDEX idx_mining_run_completed ON mining_run(completed_at);


-- A modern surface fragment that descends from one or more etymons.
-- (surface_form, position) is the natural key; multiple etymons may share
-- a single reflex (cross-language convergence, e.g. ON býr / OE byrh both
-- showing as -bury in some names).
CREATE TABLE reflex (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  surface_form  TEXT NOT NULL,
  position      TEXT NOT NULL CHECK (position IN ('pre', 'post', 'inner')),
  productivity  INTEGER NOT NULL DEFAULT 0,
  UNIQUE (surface_form, position)
);

CREATE TABLE reflex_etymon (
  reflex_id INTEGER NOT NULL REFERENCES reflex(id) ON DELETE CASCADE,
  etymon_id INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  PRIMARY KEY (reflex_id, etymon_id)
);

-- A real or attested place name.
CREATE TABLE toponym (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  modern_name TEXT NOT NULL,
  country     TEXT,
  region      TEXT
);
CREATE UNIQUE INDEX idx_toponym_unique
  ON toponym(modern_name, COALESCE(country, ''), COALESCE(region, ''));

-- Historical spellings of a toponym (Domesday "Abbendune", Piers Plowman
-- "Abyndoun", etc.). Multiple rows expected per toponym.
CREATE TABLE toponym_attestation (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  toponym_id  INTEGER NOT NULL REFERENCES toponym(id) ON DELETE CASCADE,
  form        TEXT NOT NULL,
  date_year   INTEGER,
  source_doc  TEXT
);

-- One scholar's proposed breakdown of a toponym. Multiple rows per toponym
-- are expected and welcomed: they encode disagreement, which is information.
CREATE TABLE toponym_etymology (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  toponym_id      INTEGER NOT NULL REFERENCES toponym(id) ON DELETE CASCADE,
  source_id       TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
  page            TEXT,
  historical_form TEXT,
  confidence      TEXT CHECK (confidence IN ('high', 'medium', 'low')),
  notes           TEXT
);

CREATE TABLE toponym_etymology_element (
  toponym_etymology_id INTEGER NOT NULL REFERENCES toponym_etymology(id) ON DELETE CASCADE,
  ordinal              INTEGER NOT NULL,
  etymon_id            INTEGER NOT NULL REFERENCES etymon(id),
  inflection           TEXT,
  surface_in_modern    TEXT,
  PRIMARY KEY (toponym_etymology_id, ordinal)
);

-- The set of canonical etymons — i.e. the ones that aren't OCR-merge
-- losers. Most queries that want "the real etymons" should read through
-- this view instead of the raw `etymon` table. Tombstoned merge-losers
-- still exist in `etymon` so their citations / glosses can roll up to
-- canonical via etymon_consensus, and so clear-enrichment --stage=ocr
-- can revive them.
CREATE VIEW etymon_canonical AS
  SELECT * FROM etymon WHERE merged_into_id IS NULL;

-- Per-etymon consensus: how many independent sources cite this morpheme?
-- Drives auto-promotion thresholds (≥3 witnesses → live inventory).
--
-- Aggregates by canonical group, with rollup priority:
--   merged_into_id → lemma_id → self
-- An OCR-cluster loser's citations roll up to the winner; an inflected
-- variant's citations roll up to the lemma; everyone else is their own
-- group. Per D22 this lets non-destructive OCR clustering work without
-- moving citation data.
CREATE VIEW etymon_consensus AS
  SELECT lemma_id,
         canonical_form,
         language,
         COUNT(DISTINCT source_id) AS witnesses
  FROM (
    SELECT
      -- Two-step rollup so merged_into_id → lemma_id chains collapse
      -- correctly. target = e's first-hop redirect; le = target's lemma
      -- if target is itself inflected. Without the second JOIN, an OCR
      -- variant of an inflected form would split witnesses across the
      -- target group and the lemma group.
      COALESCE(le.id, target.id, e.id) AS lemma_id,
      COALESCE(le.canonical_form, target.canonical_form, e.canonical_form)
        AS canonical_form,
      e.language,
      c.source_id
    FROM etymon e
    LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
    LEFT JOIN etymon le ON le.id = target.lemma_id
    LEFT JOIN etymon_citation c ON c.etymon_id = e.id
  )
  GROUP BY lemma_id;

-- Per-toponym disagreement: distinct breakdown signatures per toponym.
-- A signature is the ordered list of etymon_ids; if a toponym has >1
-- distinct signature, scholars disagree on the breakdown.
-- Note: relies on GROUP_CONCAT honoring the ORDER BY clause, which it does
-- in SQLite 3.44+.
CREATE VIEW toponym_breakdown_signature AS
  SELECT te.toponym_id,
         te.id        AS toponym_etymology_id,
         te.source_id,
         GROUP_CONCAT(tee.etymon_id, ',' ORDER BY tee.ordinal) AS signature
  FROM toponym_etymology te
  LEFT JOIN toponym_etymology_element tee
    ON tee.toponym_etymology_id = te.id
  GROUP BY te.id;

CREATE INDEX idx_etymon_lang        ON etymon(language);
CREATE INDEX idx_reflex_position    ON reflex(position);
CREATE INDEX idx_toponym_name       ON toponym(modern_name);
CREATE INDEX idx_etymology_toponym  ON toponym_etymology(toponym_id);
CREATE INDEX idx_etymology_source   ON toponym_etymology(source_id);
CREATE INDEX idx_attestation_topo   ON toponym_attestation(toponym_id);
