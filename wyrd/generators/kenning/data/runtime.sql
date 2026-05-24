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

CREATE TABLE fantasy_morpheme (
    -- COLLATE NOCASE matches the L3 ``fantasy_morpheme.input_name`` contract.
    -- The emitter lowercases the key but the COLLATE keeps the PK aligned
    -- semantically so a future case-distinct input is rejected at write time
    -- rather than silently shadowed.
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
