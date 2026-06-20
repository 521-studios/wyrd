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
  -- D27 / wyrd-0ug cognate clustering. Populated by cluster-cognates
  -- enrichment (wyrd-81n), points at the most-ancestral known etymon in
  -- this cognate set (the Proto-* root, or the earliest surfaceable form).
  -- All etymons reachable from that root via etymon_descent inheritance
  -- edges share the same cognate_id. Cross-language by design: OE tūn,
  -- ON tún, modern Icelandic tún, modern English town would all carry
  -- the same cognate_id pointing at Proto-Germanic *tūnaz. Distinct from
  -- merged_into_id (within-language OCR cluster) and lemma_id (within-
  -- language inflection family). Also distinct from the meaning_synset
  -- table — that one carries SEMANTIC equivalence ('water/flowing'),
  -- while cognate_id carries ETYMOLOGICAL descent. The historical name
  -- of this column was synset_id; renamed in wyrd-44a to remove the
  -- naming collision once the meaning_synset layer landed.
  cognate_id      INTEGER REFERENCES etymon(id) ON DELETE SET NULL,
  -- Which version of cluster-cognates assigned cognate_id. Lets a future
  -- cluster-cognates-v2 selectively clear and rebuild only its
  -- predecessor's work, mirroring the lemma_method shape.
  cognate_method  TEXT,
  -- wyrd-ha9q Phase 2a: pronunciation + multi-script renderings used by
  -- the SPA's etymological-provenance panel and (in Phase 2b) by the
  -- english_shaped derivation that drives non-Latin-script town-name
  -- generation. Decision A (PR after #94): we DON'T migrate canonical_form
  -- for non-Latin-source rows; canonical_form continues to hold whatever
  -- wiktextract used as `entry.word` (often the native script for
  -- Hebrew/Arabic/etc.), and these columns sit alongside.
  --
  -- pronunciation_ipa     — most authoritative IPA from wiktextract's
  --                         `sounds[*].ipa`; e.g. /d͡ʒɪnː/ for Arabic jinn.
  -- pronunciation_dialect — first dialect tag on the chosen sound entry
  --                         when present (e.g. 'Biblical-Hebrew',
  --                         'Yemenite-Hebrew', 'Modern Standard Arabic').
  -- original_script       — vocalized native-script form when wiktextract
  --                         provides one via `head_templates[0].args.wv`
  --                         (Hebrew niqqud, Arabic harakat, etc.). NULL
  --                         when canonical_form already carries the
  --                         intended display string.
  -- transliteration       — academic Latin-script form from
  --                         `head_templates[0].args.tr` (e.g. 'kɛ́lɛḇ',
  --                         'ʿifrīt', 'rakṣasa'). Carries diacritics that
  --                         the english_shaped derivation strips.
  -- english_shaped        — Phase 2b: rule-derived from IPA + transliteration,
  --                         sanitized for English readers (no diacritics,
  --                         no exotic characters). Town-name generation
  --                         picks this when set.
  pronunciation_ipa     TEXT,
  pronunciation_dialect TEXT,
  original_script       TEXT,
  transliteration       TEXT,
  english_shaped        TEXT,
  -- wyrd-lr4: within-language stratum tag. Partitions the etymons
  -- inside one language into named buckets (e.g. for 'welsh':
  -- 'latin-loan', 'english-loan', 'brittonic-substrate',
  -- 'medieval-welsh', 'native-welsh') so generation can filter to a
  -- specific register. Strata are language-specific — there is no
  -- global enum. Populated by ``lexicon classify-stratum`` from
  -- the per-language heuristics in
  -- wyrd/generators/kenning/strata.py. NULL on rows that haven't
  -- been classified yet.
  stratum               TEXT,
  UNIQUE (canonical_form, language)
);
CREATE INDEX idx_etymon_lemma       ON etymon(lemma_id);
CREATE INDEX idx_etymon_merged_into ON etymon(merged_into_id);
CREATE INDEX idx_etymon_cognate     ON etymon(cognate_id);
CREATE INDEX idx_etymon_stratum     ON etymon(stratum);

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
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  etymon_id       INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  source_id       TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
  page            TEXT,
  short_quote     TEXT,
  -- wyrd-9kh.3: scholarly-prose context window per citation. Populated by
  -- siblings (.4 reverse-search snippet capture, .5 page-number parser);
  -- the runtime SPA (.6) renders it in a citation disclosure UI. NULL for
  -- legacy citations and the rando-port seed (no source body to quote).
  context_snippet TEXT
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
  -- 'fuzzy-search-v1', 'llm-disambiguated-v1', etc. Lets a future rebuild
  -- selectively clear-and-regenerate by method.
  method        TEXT NOT NULL DEFAULT 'reverse-search-v1',
  -- One-sentence reason from the LLM disambiguator (wyrd-6z7), set when
  -- method='llm-disambiguated-v1'. Null for purely heuristic rows.
  disambiguator_reason TEXT,
  -- D5-1 / wyrd-3ux: earliest plausibly-attested year for the matched form
  -- in this source's body, populated by the lookup-attested-years post-
  -- mining stage. Pattern: a year-citation (3-4 digits, 100-1700) within a
  -- short window of the matched form, with date-context filters. Idempotent
  -- + reversible via clear-enrichment --stage=attested-years. Foundation
  -- for D5-2 era-cell sampling.
  attested_year INTEGER,
  UNIQUE (etymon_id, source_id, matched_form)
);
CREATE INDEX idx_etymon_text_match_etymon ON etymon_text_match(etymon_id);
CREATE INDEX idx_etymon_text_match_source ON etymon_text_match(source_id);
CREATE INDEX idx_etymon_text_match_year ON etymon_text_match(attested_year);

-- D27 / wyrd-0ug: directed etymological descent graph. A row asserts that
-- `parent` is an ancestor of `child` under one of the documented edge
-- types. Wiktionary's Etymology + Descendants sections are the bulk
-- populator (wyrd-4rt); existing dictionary mining can also write rows
-- when the LLM identifies an explicit chain ("from OE tūn"). The cognate
-- cluster column etymon.cognate_id is derived from this graph by walking
-- inheritance edges (wyrd-81n).
--
-- Edge types (mapped from Wiktextract template kinds):
--   inheritance — {{inh}}: child is the direct descendant of parent in
--                 the same lineage. High confidence; bridges cognate_id.
--   borrowing   — {{bor}}: child borrowed from parent across language
--                 lines. High confidence; also bridges cognate_id (a
--                 borrowed word is part of the borrowing language's
--                 cognate set).
--   calque      — {{cal}}: structural translation; child has parent's
--                 form-meaning shape but distinct lexical material.
--   compound    — {{compound}} / {{affix}}: child is a derivational
--                 product of parent + other elements.
--   derivation  — {{der}}: parent influenced child but the direct
--                 chain is unproven. Medium confidence.
--   cognate     — {{cog}}: peer relationship — both descend from a
--                 common ancestor without specifying which is parent.
--                 Does NOT bridge cognate_id (clustering would over-
--                 unify; Wiktionary cognate links cross probable
--                 boundaries that aren't always real).
--   unknown     — fallback for free-text "compare with X" assertions.
--
-- Source attribution: the source_id points at the bibliographic record
-- that asserted this edge. For Wiktionary, a single 'wiktionary' source
-- row is sufficient for v1 (per-edit attribution lives in wiki history).
-- Same shape as etymon_citation — one piece of evidence per row, dedupe
-- via the UNIQUE constraint.
--
-- Why descent does NOT contribute to etymon_consensus:
-- The witnesses count in etymon_consensus measures EXTRACTION witnesses
-- (D4: "N scholars formally identify this morpheme as part of a toponym
-- breakdown"). Descent is a different axis — it relates morphemes
-- across language and era. Counting descent edges as extraction witnesses
-- would inflate consensus and break the ≥3-witness promotion threshold.
-- Synset clustering is the correct rollup for descent.
CREATE TABLE etymon_descent (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_id   INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  child_id    INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  edge_type   TEXT NOT NULL CHECK (edge_type IN (
                'inheritance', 'borrowing', 'calque',
                'compound', 'derivation', 'cognate', 'unknown'
              )),
  source_id   TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
  confidence  TEXT CHECK (confidence IN ('high', 'medium', 'low')),
  notes       TEXT,
  UNIQUE (parent_id, child_id, edge_type, source_id)
);
CREATE INDEX idx_etymon_descent_parent ON etymon_descent(parent_id);
CREATE INDEX idx_etymon_descent_child  ON etymon_descent(child_id);

-- wyrd-fqil: per-form variant rows from wiktextract entries' `forms`
-- arrays. Each row records one inflected, alternative-spelling,
-- romanized, or otherwise lemma-variant surface form Wiktionary
-- attests for the parent etymon. Lets surface-form lookups (the
-- wyrd-ami fantasy pre-filter, etc.) match historical / dialectal /
-- Chaucer-era spellings that aren't separate lemmas in wiktextract.
-- Tags column carries the original wiktextract tag list as JSON for
-- callers that want finer-grained filtering than variant_class.
CREATE TABLE etymon_variant (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  etymon_id     INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  form          TEXT NOT NULL COLLATE NOCASE,
  variant_class TEXT NOT NULL CHECK (variant_class IN (
                  'alternative', 'inflection', 'romanization',
                  'canonical', 'other'
                )),
  tags          TEXT,
  source_id     TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
  UNIQUE (etymon_id, form, variant_class)
);
CREATE INDEX idx_etymon_variant_etymon ON etymon_variant(etymon_id);
CREATE INDEX idx_etymon_variant_form   ON etymon_variant(form);
CREATE INDEX idx_etymon_variant_class  ON etymon_variant(variant_class);

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
-- wyrd-53ep: PK leads with reflex_id, so reflex-reachability lookups by
-- etymon_id (tag_mining.select_targets; the etymon FK's ON DELETE CASCADE)
-- can't use it.
CREATE INDEX idx_reflex_etymon_etymon ON reflex_etymon(etymon_id);

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
-- "Abyndoun", etc.). Multiple rows expected per toponym. Populated by
-- the ``mine-attestations`` post-mining stage (wyrd-skm Phase 3.0a),
-- which extracts (form, date_year) pairs from
-- ``toponym_etymology.notes`` body text.
--
-- Design note (multi-source): the unique index includes ``source_doc``,
-- so the same (toponym_id, form, date_year) cited by Skeat AND Mawer
-- AND Ekwall produces three rows. The ingest treats each scholarly
-- citation as independent evidence; aggregation (COUNT vs
-- COUNT DISTINCT) is deferred to consumers (wyrd-skm Phase 3.0b's
-- per-etymon period-form derivation will choose explicitly).
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
  notes           TEXT,
  -- D5-1 / wyrd-bag: earliest plausibly-attested year for this
  -- toponym's historical form, populated by the lookup-attested-years
  -- post-mining stage. Pattern: a year-citation (3-4 digits, 700-1700)
  -- in the LLM-extracted notes, which typically carry dense scholarly
  -- date citations like "Tune, 1086 (DB); Tunes, 1242". Scoped to
  -- post-Roman attestations (≥700) to filter page / volume / footnote
  -- numbers that cluster in the low hundreds. Idempotent + reversible
  -- via clear-enrichment --stage=attested-years.
  attested_year   INTEGER,
  -- wyrd-08qv: multi-extractor consensus canonicalization. When >=2
  -- different LLM extractors propose the same element-tuple for a
  -- toponym, ONE row in the agreeing cluster is marked is_canonical=1
  -- and carries the witness count in consensus_size. Downstream
  -- consumers (proportions, generation, validation) read canonical-
  -- only by default; future LLM re-extraction passes can skip
  -- toponyms with a canonical etymology to save cost on settled
  -- answers. Single-witness rows stay is_canonical=0 + consensus_size=1.
  -- Recomputed in bulk by `canonicalize-toponym-etymology --apply` —
  -- the operation is idempotent and automatically demotes rows that
  -- lose consensus on a re-run, so there's no separate
  -- clear-enrichment stage; re-run the canonicalize command after the
  -- corpus changes to refresh.
  is_canonical    INTEGER NOT NULL DEFAULT 0 CHECK (is_canonical IN (0, 1)),
  consensus_size  INTEGER NOT NULL DEFAULT 1,
  -- The cluster key used to group agreeing rows. Persisted (rather
  -- than recomputed on read) so operators can grep/filter the
  -- canonicalize-run output and understand WHY a cluster formed.
  -- Shape: JSON-encoded list of [norm-form, lang] pairs in ordinal
  -- order, e.g. '[["bedelinga","old-english"],["tun","old-english"]]'.
  -- Null until canonicalize-toponym-etymology has run. JSON-encoded
  -- (vs. pipe-delimited) so a normalized form containing `|` or `:`
  -- can't ambiguously collide with a different element-tuple.
  cluster_key     TEXT
);
CREATE INDEX idx_toponym_etymology_year ON toponym_etymology(attested_year);
CREATE INDEX idx_toponym_etymology_canonical
  ON toponym_etymology(toponym_id) WHERE is_canonical = 1;

-- wyrd-08qv: convenience view for canonical-only reads. Downstream
-- consumers (proportions, generation) join this to bypass disagreement
-- rows + single-witness fringe hypotheses.
CREATE VIEW toponym_etymology_canonical AS
  SELECT * FROM toponym_etymology WHERE is_canonical = 1;

CREATE TABLE toponym_etymology_element (
  toponym_etymology_id INTEGER NOT NULL REFERENCES toponym_etymology(id) ON DELETE CASCADE,
  ordinal              INTEGER NOT NULL,
  etymon_id            INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  inflection           TEXT,
  surface_in_modern    TEXT,
  PRIMARY KEY (toponym_etymology_id, ordinal)
);
-- wyrd-53ep: PK is (toponym_etymology_id, ordinal), so breakdown-admit lookups
-- by etymon_id (tag_mining.select_targets; the etymon FK's ON DELETE CASCADE)
-- have no usable index.
CREATE INDEX idx_toponym_etymology_element_etymon ON toponym_etymology_element(etymon_id);

-- wyrd-08m: matcher-derived alternative breakdowns per toponym. One row
-- per (toponym_id, signature), where signature is a stable hash of the
-- decomposition's morpheme + unaccounted-fragment shape. Many rows per
-- toponym are expected (the matcher routinely emits 2-5 alternatives for
-- compound names). At most one row per toponym is canonical (is_canonical=1).
--
-- The canonical pick follows priority rules in
-- ``decomposition.pick_canonical_decomposition``:
--   1. scholar match — a decomposition whose (lang_field, canonical_form)
--      slots align with an existing toponym_etymology row's element list.
--      ``canonical_source='scholar'``. Multiple scholarly breakdowns ->
--      every matching decomposition is canonical, tagged
--      ``'scholar-disagreement'`` (the scholarship genuinely disagrees;
--      the schema preserves the disagreement).
--   2. unique-zero-unaccounted — exactly one decomposition has zero
--      unaccounted_fragments. ``canonical_source='unique-zero-unaccounted'``.
--   3. tiebreaker — multi-zero-unaccounted disambiguation (Phase 2; not
--      yet implemented).
--
-- Distinct from ``toponym_etymology``: that's the SCHOLAR's claim
-- (mining evidence), this is the MATCHER's enumeration of every plausible
-- breakdown derivable from the bundled meanings.json. Scholar evidence
-- INFORMS the canonical pick but doesn't generate the rows.
--
-- ``decomposition_signature`` is the canonical-shape hash; reused as the
-- UNIQUE key so re-running the populator is idempotent. Collisions on the
-- same (toponym, shape) update existing rows rather than duplicating.
--
-- ``morpheme_ids`` is a JSON list whose entries are either a string
-- ``modern_usage`` (matched morpheme — e.g. ``"Bridg-"``) or a JSON
-- object ``{"unaccounted": "<chars>"}`` for unaccounted fragments. The
-- mixed shape preserves slot ordering — the morpheme/unaccounted
-- interleaving is part of the decomposition's identity. Plus
-- ``unaccounted_fragments`` for fast querying of pure-fragment lists.
CREATE TABLE toponym_decomposition (
  id                       INTEGER PRIMARY KEY AUTOINCREMENT,
  toponym_id               INTEGER NOT NULL REFERENCES toponym(id) ON DELETE CASCADE,
  decomposition_signature  TEXT NOT NULL,
  morpheme_ids             TEXT NOT NULL,
  unaccounted_fragments    TEXT NOT NULL,
  unaccounted_count        INTEGER NOT NULL DEFAULT 0,
  morpheme_count           INTEGER NOT NULL DEFAULT 0,
  is_canonical             INTEGER NOT NULL DEFAULT 0 CHECK (is_canonical IN (0, 1)),
  canonical_source         TEXT CHECK (
    canonical_source IN (
      'scholar', 'scholar-disagreement', 'unique-zero-unaccounted', 'tiebreaker'
    ) OR canonical_source IS NULL
  ),
  created_at               TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX idx_toponym_decomposition_unique
  ON toponym_decomposition(toponym_id, decomposition_signature);
CREATE INDEX idx_toponym_decomposition_topo
  ON toponym_decomposition(toponym_id);
CREATE INDEX idx_toponym_decomposition_canonical
  ON toponym_decomposition(toponym_id) WHERE is_canonical = 1;

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
  -- canonical_form and language are functionally dependent on
  -- lemma_id via the rollup chain, but explicit GROUP BY keeps the
  -- result well-defined without relying on SQLite's "bare column"
  -- tolerance.
  GROUP BY lemma_id, canonical_form, language;

-- wyrd-7lo: per-canonical rollup views for the gloss / tag / text-match
-- child tables. After D22's non-destructive OCR clustering, child rows
-- stay attached to the original (loser) etymons; a SELECT against the
-- raw tables scoped to a canonical etymon's id misses everything on its
-- merged tombstones. These views supply the "all data for this
-- canonical group, including from merged variants" read path that
-- etymon_consensus already provides for citations.
--
-- Same two-step rollup chain as etymon_consensus: target = first-hop
-- redirect (merged_into_id → lemma_id); le = target's lemma. The
-- canonical_etymon_id surfaced is the deepest non-NULL row in the
-- chain.
CREATE VIEW etymon_gloss_canonical AS
  SELECT DISTINCT
         COALESCE(le.id, target.id, e.id) AS canonical_etymon_id,
         g.gloss
  FROM etymon e
  JOIN etymon_gloss g ON g.etymon_id = e.id
  LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
  LEFT JOIN etymon le ON le.id = target.lemma_id;

CREATE VIEW etymon_tag_canonical AS
  SELECT DISTINCT
         COALESCE(le.id, target.id, e.id) AS canonical_etymon_id,
         t.tag
  FROM etymon e
  JOIN etymon_tag t ON t.etymon_id = e.id
  LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
  LEFT JOIN etymon le ON le.id = target.lemma_id;

-- text_match aggregates — when a canonical group has multiple member
-- etymons that each saw the same matched_form in the same source,
-- SUM(match_count) collapses them into one row per (canonical, source,
-- matched_form). Mirrors the UPSERT semantics D21 documents for
-- writes against the raw table. attested_year is MIN'd because the
-- field stores the EARLIEST plausibly-attested year per row, and the
-- earliest across a canonical group remains the earliest.
CREATE VIEW etymon_text_match_canonical AS
  SELECT COALESCE(le.id, target.id, e.id) AS canonical_etymon_id,
         m.source_id,
         m.matched_form,
         SUM(m.match_count) AS total_match_count,
         MIN(m.edit_distance) AS edit_distance,
         MIN(m.attested_year) AS attested_year
  FROM etymon e
  JOIN etymon_text_match m ON m.etymon_id = e.id
  LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id, e.lemma_id)
  LEFT JOIN etymon le ON le.id = target.lemma_id
  GROUP BY canonical_etymon_id, m.source_id, m.matched_form;

-- Per-toponym disagreement: distinct breakdown signatures per toponym.
-- A signature is the ordered list of etymon_ids; if a toponym has >1
-- distinct signature, scholars disagree on the breakdown.
-- Pre-sorts in a subquery instead of using GROUP_CONCAT(... ORDER BY ...)
-- so the view works on SQLite 3.40 (Lambda runtime) — that syntax is
-- 3.44+ only. GROUP_CONCAT preserves the input row order, so the
-- subquery's ORDER BY te.id, tee.ordinal is what makes the signature
-- ordinal-stable.
CREATE VIEW toponym_breakdown_signature AS
  SELECT toponym_id,
         toponym_etymology_id,
         source_id,
         GROUP_CONCAT(etymon_id, ',') AS signature
  FROM (
    SELECT te.toponym_id,
           te.id AS toponym_etymology_id,
           te.source_id,
           tee.etymon_id,
           tee.ordinal
    FROM toponym_etymology te
    LEFT JOIN toponym_etymology_element tee
      ON tee.toponym_etymology_id = te.id
    ORDER BY te.id, tee.ordinal
  )
  GROUP BY toponym_etymology_id, toponym_id, source_id;

-- D? / wyrd-7tz: meaning-synset layer. A meaning_synset is a fine-grained
-- semantic equivalence class — 'water/flowing' (members: OE wæter,
-- OE strēam, ON bekkr, ...) versus 'water/body' (members: OE mere,
-- OE sǣ, ON sær, ...). Distinct from etymon.cognate_id (renamed from
-- the historical 'synset_id' in wyrd-44a), which is a COGNATE-cluster
-- ID populated by cluster-cognates enrichment from the etymon_descent
-- graph (etymology-based: OE tūn / ON tún / modern English town all
-- share one cognate cluster pointing at PGmc *tūnaz). After the
-- rename, the two concepts are unambiguously distinct: cognate_id =
-- ETYMOLOGICAL descent, meaning_synset = SEMANTIC equivalence.
--
-- Used by upcoming generator transforms (Lab replace-root, calque,
-- anglicize/foreignize, drift-toward-X) that need same-meaning
-- candidate lookups across languages and registers.
--
-- canonical_label: a human-curated string under 'category/subcategory'
-- convention ('water/flowing', 'hill/rounded'). Free-text but the
-- seed catalog uses the convention; future LLM-assisted assignment
-- (Phase 2 of wyrd-7tz) will validate against it.
--
-- hypernym_id: optional self-FK for hierarchy walks. 'water/flowing'
-- can have hypernym 'water'. Lets a query 'find candidates broadly
-- meaning water' walk the hierarchy without exhaustively enumerating
-- subcategories.
CREATE TABLE meaning_synset (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  canonical_label TEXT NOT NULL UNIQUE,
  hypernym_id     INTEGER REFERENCES meaning_synset(id) ON DELETE SET NULL,
  notes           TEXT
);
CREATE INDEX idx_meaning_synset_hypernym ON meaning_synset(hypernym_id);

-- Many-to-many: an etymon can be in multiple synsets ('OE bourne' may
-- be both water/flowing and water/crossable). 'fit' marks whether the
-- synset is the etymon's primary sense ('core') or a secondary one
-- ('peripheral') so candidate queries can rank.
CREATE TABLE etymon_meaning_synset (
  etymon_id         INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  meaning_synset_id INTEGER NOT NULL REFERENCES meaning_synset(id) ON DELETE CASCADE,
  fit               TEXT NOT NULL CHECK (fit IN ('core', 'peripheral')),
  PRIMARY KEY (etymon_id, meaning_synset_id)
);
CREATE INDEX idx_etymon_meaning_synset_synset ON etymon_meaning_synset(meaning_synset_id);

CREATE INDEX idx_etymon_lang        ON etymon(language);
CREATE INDEX idx_reflex_position    ON reflex(position);
CREATE INDEX idx_toponym_name       ON toponym(modern_name);
CREATE INDEX idx_etymology_toponym  ON toponym_etymology(toponym_id);
CREATE INDEX idx_etymology_source   ON toponym_etymology(source_id);
CREATE INDEX idx_attestation_topo   ON toponym_attestation(toponym_id);
-- wyrd-rhnw: dedicated source_doc index. idx_attestation_unique has
-- source_doc as the LAST composite column, so a `SELECT WHERE
-- source_doc=?` falls back to a full-table scan without this. Phase
-- 2b.2 corpus extraction needs the dedicated index for the bulk
-- preflight to stay sub-linear.
CREATE INDEX idx_attestation_source_doc ON toponym_attestation(source_doc);
-- wyrd-skm Phase 3.0a: keeps mine-attestations re-runs idempotent.
-- Allows multiple attestations per toponym (different form, year, or
-- scholarly source) without duplicating identical (toponym, form, year,
-- source) rows. COALESCE on the nullable columns matches the pattern
-- used by idx_toponym_unique (country / region) and
-- idx_etymon_citation_unique (page) — without the wrap, SQLite treats
-- every NULL as distinct under UNIQUE so an idempotent re-mine could
-- silently double-insert NULL-year rows.
CREATE UNIQUE INDEX idx_attestation_unique
  ON toponym_attestation(
    toponym_id, form, COALESCE(date_year, 0), COALESCE(source_doc, '')
  );

-- wyrd-unuo Phase 3.3: per-etymon period-keyed surface forms,
-- projected from toponym_attestation rows by segmenting historical
-- compound forms onto the toponym's canonical morpheme breakdown.
-- Used by etymon_era_reflexes as a Tier 3 fallback when neither
-- cognate-cluster (Tier 1) nor direct descent (Tier 2) produces a
-- reflex. Closes the ~72% coverage gap on isolated OE etymons
-- that have no cognate_id and no descent edges.
--
-- attestation_id is a soft FK back to the source toponym_attestation
-- row so a reverse-stage clear can drop projection-derived rows
-- without losing the underlying attestations.
CREATE TABLE etymon_period_form (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  etymon_id       INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
  form            TEXT NOT NULL,
  date_year       INTEGER NOT NULL,
  source_doc      TEXT,
  attestation_id  INTEGER REFERENCES toponym_attestation(id) ON DELETE SET NULL
);
CREATE INDEX idx_period_form_etymon
  ON etymon_period_form(etymon_id);
CREATE INDEX idx_period_form_year
  ON etymon_period_form(date_year);
-- Idempotent re-projection: identical (etymon, form, year, source)
-- pairs from the same attestation should not duplicate.
CREATE UNIQUE INDEX idx_period_form_unique
  ON etymon_period_form(etymon_id, form, date_year, source_doc);

-- wyrd-ecjp.2: empirical-baseline priors for the vector-driven
-- generator (D36.4, D36.7). Two parallel views of the same
-- observation set:
--
--   * empirical_priors_native  — keyed by (culture, position, tag,
--     era, lemma_ref). The "all elements that appeared in a
--     <culture>-recipient toponym" count, regardless of source
--     language.
--   * empirical_priors_loan    — keyed by (donor_language, recipient
--     culture, position, tag, era, lemma_ref). The same observations
--     partitioned by the lemma's source language; scenario packs
--     (D36.4 Option B) read this slice for their template's
--     loan-relationship baseline.
--
-- ``lemma_ref`` is the canonical "<language>:<canonical_form>"
-- string (matches the L2 cross-file etymon ref shape). The
-- lemma is the donor-language form even in the native view, so a
-- Latin loan in an English toponym surfaces as
-- ``empirical_priors_native(culture='english', lemma_ref='latin:castrum')``
-- — native aggregates across loan languages.
--
-- ``era_midpoint`` is the integer-year midpoint of the era cell
-- (per ``era.ERA_CELLS``; open-ended cells use a fixed-offset
-- convention defined in the extractor). The era LABEL is recoverable
-- via ``era.era_cell(language, midpoint)`` so the schema doesn't
-- store it.
--
-- For ad-hoc SQL: enumerate valid era_midpoint values via
-- ``empirical_priors.known_era_midpoints()`` or the
-- ``empirical_priors.json`` sidecar — e.g. english/oe-late = 950,
-- english/me = 1300, english/early-modern = 1600, english/modern
-- (open-high) = 1800.
--
-- ``count`` is the raw occurrence count in this cell. Smoothing
-- (hierarchical fallback per D36.7) happens at lookup time, not
-- at bake time — the table holds raw observations so we can choose
-- different smoothing rules without re-mining.
--
-- LLM-free, deterministic, idempotent. Re-generable via
-- ``lexicon mine-empirical-baselines --apply [--force]``; dump to
-- the committed JSON sidecar via ``--dump-json <path>`` so diffs
-- are reviewable.
-- The PRIMARY KEY ordering on both tables intentionally matches the
-- lookup pattern (culture/donor + recipient → position → tag → era →
-- lemma). SQLite covers PK-prefix lookups via the auto-generated
-- B-tree on the PK; only the secondary `_lemma` index is non-redundant
-- (it indexes lemma_ref, which is NOT a PK prefix and IS needed for
-- "find all cells containing this lemma" queries).
CREATE TABLE empirical_priors_native (
  culture       TEXT NOT NULL,
  position      TEXT NOT NULL,
  tag           TEXT NOT NULL,
  era_midpoint  INTEGER NOT NULL,
  lemma_ref     TEXT NOT NULL,
  count         INTEGER NOT NULL CHECK (count > 0),
  PRIMARY KEY (culture, position, tag, era_midpoint, lemma_ref)
);
CREATE INDEX idx_priors_native_lemma
  ON empirical_priors_native(lemma_ref);

CREATE TABLE empirical_priors_loan (
  donor         TEXT NOT NULL,
  recipient     TEXT NOT NULL,
  position      TEXT NOT NULL,
  tag           TEXT NOT NULL,
  era_midpoint  INTEGER NOT NULL,
  lemma_ref     TEXT NOT NULL,
  count         INTEGER NOT NULL CHECK (count > 0),
  PRIMARY KEY (donor, recipient, position, tag, era_midpoint, lemma_ref)
);
CREATE INDEX idx_priors_loan_lemma
  ON empirical_priors_loan(lemma_ref);


-- wyrd-2b50: Briggs personal names are now ingested into the standard
-- ``etymon`` schema (etymon + male/female-name tag + multi-source
-- citations via the cited_source path), so the bespoke ``personal_name``
-- and ``personal_name_toponym_attestation`` tables (wyrd-uzoh) were
-- dropped. See migration 0014_drop_personal_name_tables.


-- wyrd-ami / wyrd-rrse: fantasy_morpheme records (name, description) →
-- LLM-research-derived resolution. One row per input fed to
-- ``lexicon mine-fantasy-name`` regardless of whether the morpheme
-- was usable; barred rows carry ``bar_reason`` so later pipelines can
-- revisit them.
--
-- ``input_name COLLATE NOCASE`` dedupes "Harpy" / "harpy" under
-- UNIQUE(input_name, approach_version) so repeat LLM calls don't fire
-- for the same name in different casings. ``approach_version`` is the
-- wyrd-ami APPROACH_VERSION constant; bumping it lets us re-process
-- older-version rows when the pipeline gets stronger. ``etymon_id``
-- is nullable + ON DELETE SET NULL so barred morphemes can record
-- ``bar_reason`` without a real etymon link.
CREATE TABLE fantasy_morpheme (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  input_name          TEXT NOT NULL COLLATE NOCASE,
  input_description   TEXT,
  usable              INTEGER NOT NULL CHECK (usable IN (0, 1)),
  etymon_id           INTEGER REFERENCES etymon(id) ON DELETE SET NULL,
  bar_reason          TEXT,
  resolution_method   TEXT NOT NULL,
  approach_version    TEXT NOT NULL,
  confidence          TEXT,
  citation            TEXT,
  reasoning           TEXT,
  unapproved_language TEXT,
  unapproved_form     TEXT,
  processed_at        TEXT NOT NULL,
  UNIQUE (input_name, approach_version)
);
CREATE INDEX idx_fantasy_morpheme_approach    ON fantasy_morpheme(approach_version);
CREATE INDEX idx_fantasy_morpheme_etymon      ON fantasy_morpheme(etymon_id);
CREATE INDEX idx_fantasy_morpheme_usable      ON fantasy_morpheme(usable);
CREATE INDEX idx_fantasy_morpheme_unapproved  ON fantasy_morpheme(unapproved_language);
