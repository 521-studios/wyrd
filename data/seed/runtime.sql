-- L4 kenning runtime DB schema (wyrd-d90t).
--
-- Replaces meanings.json + 5 per-culture proportions JSON files. Lambda
-- downloads this DB on cold start; warm invocations reuse the SQLite
-- connection. See L2_L3_BOUNDARY.md for the L2/L3 layers — this is L4,
-- the runtime artifact, and D38 in DECISIONS.md for the architecture.
--
-- Two design principles:
--   * Blob columns where the row IS the unit of consumption (meaning,
--     fantasy_morpheme, canonical_decomposition). No payoff to normalizing
--     internal structure when the runtime always fetches the whole row.
--   * Normalized columns where SQL operates on the values (proportions
--     sampling, tag statistics). Cumulative sums precomputed at emit time
--     so weighted-random sampling is an O(log n) index seek:
--         total = SELECT MAX(cumulative) FROM tbl WHERE culture = ?
--         roll  = rng.randint(1, total)
--         row   = SELECT key FROM tbl WHERE culture = ? AND cumulative >= ?
--                 ORDER BY cumulative LIMIT 1
--
-- The DB is treated as read-only at runtime (file:...?mode=ro URI).
-- Re-emit and ship a new versioned key on S3 to update.

-- ===== Meanings (denormalized blob) =====

CREATE TABLE meaning (
    usage_key        TEXT PRIMARY KEY,        -- modern_usage, e.g. '-ham-'
    primary_language TEXT,                    -- 'old-english', 'old-norse', ...; NULL for orphan-reflex entries
    stratum          TEXT,                    -- nullable; some morphemes lack one
    data             BLOB NOT NULL            -- JSON: full Meaning payload (forms, IPA, vectors)
);
CREATE INDEX idx_meaning_language ON meaning(primary_language);
CREATE INDEX idx_meaning_stratum  ON meaning(stratum);

-- wyrd-rogd.10: the morpheme entity. Same {meaning, modifier_*, word} entries
-- as `meaning`, but regrouped by the family root id (morpheme_id) instead of
-- the connective surface string, so ONE morpheme owns all its surface variants
-- (-ing- / -ing / Ing- collapse to one record). Additive + dormant in Phase 1
-- (the runtime still resolves via the dash-strip surface index); Phase 2
-- follows this table by id for the era-grid, and Phase 3 makes the era-grid
-- morpheme_id-authoritative (drops the string-derivation fallback).
-- _resolve_surface is NOT retired — it stays the generation matcher
-- (usage → meaning/tags/siblings); only the era-grid stopped using it. The id
-- is a CONTENT
-- key "{language}:{canonical_form}" (NOT the autoincrement root_id, which
-- shifts across a rebuild) so the export stays byte-identical across
-- rebuild-from-jsonl — the reconstructibility guarantee.
CREATE TABLE morpheme (
    morpheme_id      TEXT PRIMARY KEY,
    primary_language TEXT,
    stratum          TEXT,
    data             BLOB NOT NULL            -- JSON {"entries":[...]} regrouped by morpheme_id
);
CREATE INDEX idx_morpheme_language ON morpheme(primary_language);
CREATE INDEX idx_morpheme_stratum  ON morpheme(stratum);

CREATE TABLE fantasy_morpheme (
    -- COLLATE NOCASE matches the L3 ``fantasy_morpheme.input_name`` contract.
    -- The emitter lowercases the key, so write-side collisions are already
    -- caught by a plain case-sensitive PK; the NOCASE collation is for the
    -- read side — the runtime can look up ``WHERE usage_key = 'Harpy'`` and
    -- still hit the stored lowercase row, mirroring the L3 COLLATE NOCASE
    -- semantics that callers already expect.
    usage_key TEXT PRIMARY KEY COLLATE NOCASE,
    data      BLOB NOT NULL                   -- JSON: full fantasy_morpheme payload
);

-- canonical_decomposition: culture-agnostic — matches the existing bundle
-- field's flat {toponym_name -> {signature, source}} shape. See D38.4 for
-- the deferred culture column + what would motivate adding it back.
CREATE TABLE canonical_decomposition (
    toponym_name TEXT PRIMARY KEY,
    data         BLOB NOT NULL                -- JSON: {"signature": str, "source": str}
);

-- ===== Proportions (normalized, SQL-sampled) =====

CREATE TABLE proportions_usage (
    culture    TEXT NOT NULL,
    usage_key  TEXT NOT NULL,
    weight     INTEGER NOT NULL,
    cumulative INTEGER NOT NULL,
    PRIMARY KEY (culture, cumulative)
);
CREATE INDEX idx_proportions_usage_by_key ON proportions_usage(culture, usage_key);

CREATE TABLE proportions_single_usage (
    culture    TEXT NOT NULL,
    usage_key  TEXT NOT NULL,
    weight     INTEGER NOT NULL,
    cumulative INTEGER NOT NULL,
    PRIMARY KEY (culture, cumulative)
);
CREATE INDEX idx_proportions_single_usage_by_key ON proportions_single_usage(culture, usage_key);

CREATE TABLE proportions_structure (
    culture    TEXT NOT NULL,
    template   BLOB NOT NULL,                 -- JSON: 'words' template (nested list of element dicts)
    weight     INTEGER NOT NULL,
    cumulative INTEGER NOT NULL,
    PRIMARY KEY (culture, cumulative)
);

-- Two statistics tables: point lookups, no sampling.

CREATE TABLE proportions_tag_marginal (
    culture TEXT NOT NULL,
    tag     TEXT NOT NULL,
    weight  INTEGER NOT NULL,
    PRIMARY KEY (culture, tag)
);

CREATE TABLE proportions_tag_cooccurrence (
    culture TEXT NOT NULL,
    tag1    TEXT NOT NULL,
    tag2    TEXT NOT NULL,
    weight  INTEGER NOT NULL,
    PRIMARY KEY (culture, tag1, tag2)
);

-- wyrd-rogd.13: per-WORD-position counts for bare words of multi-word
-- toponyms. ``position`` is the word's slot in the NAME's word sequence
-- (pre = first word, inner = interior, post = last word) — distinct from
-- the within-word morpheme position the dash markers encode (D39/D40).
-- ``Saint`` is essentially always pre ("Saint Albans"); Latin
-- postpositives like ``Parva`` are post ("Wigston Parva").
--
-- Counts are RAW (no threshold applied at emit). The runtime applies the
-- naked↔structured threshold at load time (WYRD_BARE_POSITION_THRESHOLD,
-- default 3): a bare word at/above the threshold of total positional
-- observations samples per its per-position weights and is eligible only
-- at positions it's attested in; below it the word is "naked" and stays
-- eligible at ANY bare slot via the general single_usage distribution.
-- Point lookups only (the vector path samples in Python) — no cumulative
-- column. Absent on pre-rogd.13 DBs; the adapter degrades gracefully.
CREATE TABLE proportions_bare_word_position (
    culture   TEXT NOT NULL,
    position  TEXT NOT NULL,                  -- 'pre' | 'inner' | 'post'
    usage_key TEXT NOT NULL,
    weight    INTEGER NOT NULL,
    PRIMARY KEY (culture, position, usage_key)
);

-- wyrd-pfoo: per-Meaning attestation per culture. Records which
-- (usage_key, primary_language) Meanings the matcher actually used
-- when decomposing this culture's place names. The runtime vector
-- path consumes this to filter its eligibility pool at the Meaning
-- level — preventing wrong-sense picks (Celtic ``-ton``→"tone",
-- ``Saint-``→"greed", ``-bridge``→"card game") that the prior
-- per-usage filter admitted as collateral damage.
--
-- A Meaning passes the runtime filter iff
-- ``(culture, meaning.usage, meaning.primary_language) ∈`` this
-- table. Multiple rows per (culture, usage_key) when the matcher
-- attested the usage via multiple language senses in the same
-- culture's corpus.
CREATE TABLE proportions_attested_language (
    culture           TEXT NOT NULL,
    usage_key         TEXT NOT NULL,
    primary_language  TEXT NOT NULL,
    PRIMARY KEY (culture, usage_key, primary_language)
);
-- No separate index on culture — the composite PK is a B-tree whose
-- leading column already serves WHERE culture = ? scans (same pattern
-- as proportions_tag_marginal / proportions_tag_cooccurrence above).

-- ===== Empirical priors (vector-scoring baseline) =====
--
-- Singleton row: the full priors payload (same shape ``dump_empirical_priors_to_json``
-- emits, what ``load_empirical_priors_from_payload`` consumes). Stored as one blob
-- because the runtime always loads the entire EmpiricalPriors object up front
-- (vector scoring needs the whole cell space available for any sample).
--
-- Replaces the previous ``data/priors.json`` sidecar — without this row in the
-- bundle, ``scoring_mode='vector'`` only produces output when the operator
-- supplies a non-trivial mood (register weights take over from the baseline).
-- Folding priors into the L4 lets vector mode work out of the box.
CREATE TABLE empirical_priors (
    id   INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton
    data BLOB NOT NULL                         -- JSON: {version, native: [...], loan: [...]}
);

-- ===== Operational =====

CREATE TABLE bundle_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);
-- Conventional keys (all written by export-runtime-db):
--   schema_version    -- L4 schema revision (bumped on table-shape changes)
--   built_at          -- UTC ISO 8601 timestamp of emission
--   emitter_version   -- identifier of the lexicon export-runtime-db build
--   source_lexicon_db -- absolute path of the L3 DB the emit read from
