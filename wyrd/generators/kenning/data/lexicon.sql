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
  UNIQUE (canonical_form, language)
);
CREATE INDEX idx_etymon_lemma       ON etymon(lemma_id);
CREATE INDEX idx_etymon_merged_into ON etymon(merged_into_id);
CREATE INDEX idx_etymon_cognate     ON etymon(cognate_id);

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
  notes           TEXT,
  -- D5-1 / wyrd-bag: earliest plausibly-attested year for this
  -- toponym's historical form, populated by the lookup-attested-years
  -- post-mining stage. Pattern: a year-citation (3-4 digits, 700-1700)
  -- in the LLM-extracted notes, which typically carry dense scholarly
  -- date citations like "Tune, 1086 (DB); Tunes, 1242". Scoped to
  -- post-Roman attestations (≥700) to filter page / volume / footnote
  -- numbers that cluster in the low hundreds. Idempotent + reversible
  -- via clear-enrichment --stage=attested-years.
  attested_year   INTEGER
);
CREATE INDEX idx_toponym_etymology_year ON toponym_etymology(attested_year);

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
