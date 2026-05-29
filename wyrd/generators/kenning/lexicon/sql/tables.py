"""SQLAlchemy Core MetaData mirroring the lexicon's SQLite schema.

This is the single source of truth for tooling that needs to introspect
the schema (Alembic autogenerate, query builders, future ORM mappers).
The layered Alembic migrations under ``migrations/versions/`` create
exactly this shape on a fresh DB; the two must stay in sync.

Design choices:

* **Core, not ORM.** The lexicon code base is built around raw SQL
  patterns (cursor.execute, executemany, executescript). Wrapping it in
  ORM models would require a sweeping behavioural change well outside
  the wyrd-67fv split. Core gives us Table metadata for query builders
  and Alembic autogenerate without forcing the rewrite.
* **SQLite-typed.** SA Core types map to SQLite affinities; the few
  SQLite-only features in use (``COLLATE NOCASE`` on
  ``etymon_variant.form``, partial unique indexes via
  ``sqlite_where``, ``COALESCE`` expressions in unique indexes) are
  expressed in the Core API where supported and as ``text()``
  expressions where not.
* **Views live in migrations, not here.** SA Core has no first-class
  view object. ``etymon_canonical``, ``etymon_consensus``, and the
  ``*_canonical`` family are created in the ``0008_views`` migration
  via ``op.execute("CREATE VIEW ...")``. Tests that need to read from
  them go through raw SQL the way the existing code does.

Cross-reference: D21 (mining evidence is sacred), D22 (non-destructive
OCR clustering via merged_into_id), D27 (etymological descent graph),
D32 (within-language stratum tagging) for the rationale behind the
schema's quirks.
"""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)

metadata = MetaData()

# ---------------------------------------------------------------------------
# Bibliographic + corpus tables
# ---------------------------------------------------------------------------

source = Table(
    "source",
    metadata,
    Column("id", Text, primary_key=True),
    Column("author", Text),
    Column("title", Text, nullable=False),
    Column("year", Integer),
    Column("region", Text),
    Column("language_focus", Text),
    Column("notes", Text),
)

# Etymon — the central morpheme table. Three self-FKs (lemma_id,
# merged_into_id, cognate_id) wire the inflection / OCR-cluster /
# cognate-cluster axes. See D8, D22, D27 for the meanings.
etymon = Table(
    "etymon",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("canonical_form", Text, nullable=False),
    Column("language", Text, nullable=False),
    Column("modifier_type", Text),
    Column(
        "position_pref",
        Text,
        CheckConstraint("position_pref IN ('pre', 'post', 'inner', 'free')"),
    ),
    Column("notes", Text),
    Column("lemma_id", Integer, ForeignKey("etymon.id", ondelete="SET NULL")),
    Column("inflection", Text),
    Column("lemma_method", Text),
    Column("merged_into_id", Integer, ForeignKey("etymon.id", ondelete="SET NULL")),
    Column("cognate_id", Integer, ForeignKey("etymon.id", ondelete="SET NULL")),
    Column("cognate_method", Text),
    Column("pronunciation_ipa", Text),
    Column("pronunciation_dialect", Text),
    Column("original_script", Text),
    Column("transliteration", Text),
    Column("english_shaped", Text),
    Column("stratum", Text),
    Column("phonological_vector", Text),
    UniqueConstraint("canonical_form", "language"),
)
Index("idx_etymon_lemma", etymon.c.lemma_id)
Index("idx_etymon_merged_into", etymon.c.merged_into_id)
Index("idx_etymon_cognate", etymon.c.cognate_id)
Index("idx_etymon_stratum", etymon.c.stratum)
Index("idx_etymon_lang", etymon.c.language)

etymon_gloss = Table(
    "etymon_gloss",
    metadata,
    Column("etymon_id", Integer, ForeignKey("etymon.id", ondelete="CASCADE"), nullable=False),
    Column("gloss", Text, nullable=False),
    PrimaryKeyConstraint("etymon_id", "gloss"),
)

etymon_tag = Table(
    "etymon_tag",
    metadata,
    Column("etymon_id", Integer, ForeignKey("etymon.id", ondelete="CASCADE"), nullable=False),
    Column("tag", Text, nullable=False),
    PrimaryKeyConstraint("etymon_id", "tag"),
)

# ---------------------------------------------------------------------------
# Citation + evidence tables (D4, D12, D21)
# ---------------------------------------------------------------------------

etymon_citation = Table(
    "etymon_citation",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("etymon_id", Integer, ForeignKey("etymon.id", ondelete="CASCADE"), nullable=False),
    Column("source_id", Text, ForeignKey("source.id", ondelete="CASCADE"), nullable=False),
    Column("page", Text),
    Column("short_quote", Text),
    Column("context_snippet", Text),
)
# SQLite forbids expressions in inline UNIQUE constraints, so the
# "treat NULL page as ''" uniqueness goes here as a unique index using
# COALESCE in the expression.
Index(
    "idx_etymon_citation_unique",
    etymon_citation.c.etymon_id,
    etymon_citation.c.source_id,
    text("COALESCE(page, '')"),
    unique=True,
)

etymon_text_match = Table(
    "etymon_text_match",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("etymon_id", Integer, ForeignKey("etymon.id", ondelete="CASCADE"), nullable=False),
    Column("source_id", Text, ForeignKey("source.id", ondelete="CASCADE"), nullable=False),
    Column("matched_form", Text, nullable=False),
    Column("match_count", Integer, nullable=False),
    Column("edit_distance", Integer, nullable=False, server_default=text("0")),
    Column("snippet", Text),
    Column("method", Text, nullable=False, server_default=text("'reverse-search-v1'")),
    Column("disambiguator_reason", Text),
    Column("attested_year", Integer),
    UniqueConstraint("etymon_id", "source_id", "matched_form"),
)
Index("idx_etymon_text_match_etymon", etymon_text_match.c.etymon_id)
Index("idx_etymon_text_match_source", etymon_text_match.c.source_id)
Index("idx_etymon_text_match_year", etymon_text_match.c.attested_year)

# ---------------------------------------------------------------------------
# Etymological descent graph + variants (D27, wyrd-fqil)
# ---------------------------------------------------------------------------

etymon_descent = Table(
    "etymon_descent",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("parent_id", Integer, ForeignKey("etymon.id", ondelete="CASCADE"), nullable=False),
    Column("child_id", Integer, ForeignKey("etymon.id", ondelete="CASCADE"), nullable=False),
    Column(
        "edge_type",
        Text,
        CheckConstraint(
            "edge_type IN ('inheritance', 'borrowing', 'calque', "
            "'compound', 'derivation', 'cognate', 'unknown')"
        ),
        nullable=False,
    ),
    Column("source_id", Text, ForeignKey("source.id", ondelete="CASCADE"), nullable=False),
    Column(
        "confidence",
        Text,
        CheckConstraint("confidence IN ('high', 'medium', 'low')"),
    ),
    Column("notes", Text),
    UniqueConstraint("parent_id", "child_id", "edge_type", "source_id"),
)
Index("idx_etymon_descent_parent", etymon_descent.c.parent_id)
Index("idx_etymon_descent_child", etymon_descent.c.child_id)

etymon_variant = Table(
    "etymon_variant",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("etymon_id", Integer, ForeignKey("etymon.id", ondelete="CASCADE"), nullable=False),
    Column("form", String(collation="NOCASE"), nullable=False),
    Column(
        "variant_class",
        Text,
        CheckConstraint(
            "variant_class IN ('alternative', 'inflection', 'romanization', 'canonical', 'other')"
        ),
        nullable=False,
    ),
    Column("tags", Text),
    Column("source_id", Text, ForeignKey("source.id", ondelete="CASCADE"), nullable=False),
    UniqueConstraint("etymon_id", "form", "variant_class"),
)
Index("idx_etymon_variant_etymon", etymon_variant.c.etymon_id)
Index("idx_etymon_variant_form", etymon_variant.c.form)
Index("idx_etymon_variant_class", etymon_variant.c.variant_class)

# ---------------------------------------------------------------------------
# Mining-run audit (D23)
# ---------------------------------------------------------------------------

mining_run = Table(
    "mining_run",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source_id", Text, ForeignKey("source.id", ondelete="CASCADE"), nullable=False),
    Column("provider", Text, nullable=False),
    Column("model", Text, nullable=False),
    Column("mode", Text, CheckConstraint("mode IN ('mine', 'review')"), nullable=False),
    Column("started_at", Text),
    Column("completed_at", Text),
    Column("parsed_count", Integer, nullable=False, server_default=text("0")),
    Column("accepted", Integer, nullable=False, server_default=text("0")),
    Column("declined", Integer, nullable=False, server_default=text("0")),
    Column("rejected", Integer, nullable=False, server_default=text("0")),
    Column("by_failure", Text),
    Column("notes", Text),
    UniqueConstraint("source_id", "provider", "model", "mode", "completed_at"),
)
Index("idx_mining_run_source", mining_run.c.source_id)
Index("idx_mining_run_completed", mining_run.c.completed_at)

# ---------------------------------------------------------------------------
# Reflex layer (modern surface form ↔ etymon)
# ---------------------------------------------------------------------------

reflex = Table(
    "reflex",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("surface_form", Text, nullable=False),
    Column(
        "position",
        Text,
        CheckConstraint("position IN ('pre', 'post', 'inner')"),
        nullable=False,
    ),
    Column("productivity", Integer, nullable=False, server_default=text("0")),
    UniqueConstraint("surface_form", "position"),
)
Index("idx_reflex_position", reflex.c.position)

reflex_etymon = Table(
    "reflex_etymon",
    metadata,
    Column("reflex_id", Integer, ForeignKey("reflex.id", ondelete="CASCADE"), nullable=False),
    Column("etymon_id", Integer, ForeignKey("etymon.id", ondelete="CASCADE"), nullable=False),
    PrimaryKeyConstraint("reflex_id", "etymon_id"),
)

# ---------------------------------------------------------------------------
# Toponym layer
# ---------------------------------------------------------------------------

toponym = Table(
    "toponym",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("modern_name", Text, nullable=False),
    Column("country", Text),
    Column("region", Text),
)
Index(
    "idx_toponym_unique",
    toponym.c.modern_name,
    text("COALESCE(country, '')"),
    text("COALESCE(region, '')"),
    unique=True,
)
Index("idx_toponym_name", toponym.c.modern_name)

toponym_attestation = Table(
    "toponym_attestation",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("toponym_id", Integer, ForeignKey("toponym.id", ondelete="CASCADE"), nullable=False),
    Column("form", Text, nullable=False),
    Column("date_year", Integer),
    Column("source_doc", Text),
)
Index("idx_attestation_topo", toponym_attestation.c.toponym_id)
# wyrd-rhnw: dedicated source_doc index. idx_attestation_unique has
# source_doc as the LAST composite column, so a `SELECT WHERE
# source_doc=?` falls back to a full-table scan without this. Phase
# 2b.2 corpus extraction needs the dedicated index for the bulk
# preflight to stay sub-linear.
Index("idx_attestation_source_doc", toponym_attestation.c.source_doc)
# COALESCE on the nullable columns matches idx_toponym_unique
# (country / region) and idx_etymon_citation_unique (page). Without
# the wrap, SQLite treats every NULL as distinct under UNIQUE, so
# an idempotent re-mine could silently double-insert two NULL-year
# attestations for the same (toponym, form, source_doc). Gemini
# round-5 review caught the pre-PR inconsistency.
Index(
    "idx_attestation_unique",
    toponym_attestation.c.toponym_id,
    toponym_attestation.c.form,
    text("COALESCE(date_year, 0)"),
    text("COALESCE(source_doc, '')"),
    unique=True,
)

toponym_etymology = Table(
    "toponym_etymology",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("toponym_id", Integer, ForeignKey("toponym.id", ondelete="CASCADE"), nullable=False),
    Column("source_id", Text, ForeignKey("source.id", ondelete="CASCADE"), nullable=False),
    Column("page", Text),
    Column("historical_form", Text),
    Column(
        "confidence",
        Text,
        CheckConstraint("confidence IN ('high', 'medium', 'low')"),
    ),
    Column("notes", Text),
    Column("attested_year", Integer),
    Column(
        "is_canonical",
        Integer,
        CheckConstraint("is_canonical IN (0, 1)"),
        nullable=False,
        server_default=text("0"),
    ),
    Column("consensus_size", Integer, nullable=False, server_default=text("1")),
    Column("cluster_key", Text),
)
Index("idx_etymology_toponym", toponym_etymology.c.toponym_id)
Index("idx_etymology_source", toponym_etymology.c.source_id)
Index("idx_toponym_etymology_year", toponym_etymology.c.attested_year)
Index(
    "idx_toponym_etymology_canonical",
    toponym_etymology.c.toponym_id,
    sqlite_where=text("is_canonical = 1"),
)

toponym_etymology_element = Table(
    "toponym_etymology_element",
    metadata,
    Column(
        "toponym_etymology_id",
        Integer,
        ForeignKey("toponym_etymology.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("ordinal", Integer, nullable=False),
    # CASCADE matches the sibling toponym_etymology_id FK and every
    # other NON-NULLABLE etymon-targeting FK (etymon_gloss,
    # etymon_tag, etymon_citation, etymon_text_match, etymon_descent
    # parent/child, etymon_variant, reflex_etymon,
    # etymon_period_form, etymon_meaning_synset). The pattern across
    # the schema is: nullable etymon FKs use SET NULL (lemma_id,
    # merged_into_id, cognate_id); non-nullable use CASCADE. The
    # pre-PR schema left this column un-annotated (SQLite default is
    # NO ACTION ≈ RESTRICT), which would block etymon deletion and
    # is the one outlier across the pattern. Gemini +
    # type-design-analyzer round-1 review (wyrd-67fv) caught it.
    Column(
        "etymon_id",
        Integer,
        ForeignKey("etymon.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("inflection", Text),
    Column("surface_in_modern", Text),
    # wyrd-2n1: per-element confidence (the parent-row
    # ``toponym_etymology.confidence`` aggregate is kept for back-
    # compat queries). NULL = scholars didn't hedge this element
    # explicitly; the three rated levels match the parent's
    # constraint.
    Column(
        "confidence",
        Text,
        CheckConstraint("confidence IN ('high', 'medium', 'low')"),
    ),
    PrimaryKeyConstraint("toponym_etymology_id", "ordinal"),
)

toponym_decomposition = Table(
    "toponym_decomposition",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("toponym_id", Integer, ForeignKey("toponym.id", ondelete="CASCADE"), nullable=False),
    Column("decomposition_signature", Text, nullable=False),
    Column("morpheme_ids", Text, nullable=False),
    Column("unaccounted_fragments", Text, nullable=False),
    Column("unaccounted_count", Integer, nullable=False, server_default=text("0")),
    Column("morpheme_count", Integer, nullable=False, server_default=text("0")),
    Column(
        "is_canonical",
        Integer,
        CheckConstraint("is_canonical IN (0, 1)"),
        nullable=False,
        server_default=text("0"),
    ),
    Column(
        "canonical_source",
        Text,
        CheckConstraint(
            "canonical_source IN ("
            "'scholar', 'scholar-disagreement', 'unique-zero-unaccounted', 'tiebreaker'"
            ") OR canonical_source IS NULL"
        ),
    ),
    Column("created_at", Text, nullable=False, server_default=text("(datetime('now'))")),
)
Index(
    "idx_toponym_decomposition_unique",
    toponym_decomposition.c.toponym_id,
    toponym_decomposition.c.decomposition_signature,
    unique=True,
)
Index("idx_toponym_decomposition_topo", toponym_decomposition.c.toponym_id)
Index(
    "idx_toponym_decomposition_canonical",
    toponym_decomposition.c.toponym_id,
    sqlite_where=text("is_canonical = 1"),
)

etymon_period_form = Table(
    "etymon_period_form",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("etymon_id", Integer, ForeignKey("etymon.id", ondelete="CASCADE"), nullable=False),
    Column("form", Text, nullable=False),
    Column("date_year", Integer, nullable=False),
    Column("source_doc", Text),
    Column(
        "attestation_id",
        Integer,
        ForeignKey("toponym_attestation.id", ondelete="SET NULL"),
    ),
)
Index("idx_period_form_etymon", etymon_period_form.c.etymon_id)
Index("idx_period_form_year", etymon_period_form.c.date_year)
Index(
    "idx_period_form_unique",
    etymon_period_form.c.etymon_id,
    etymon_period_form.c.form,
    etymon_period_form.c.date_year,
    etymon_period_form.c.source_doc,
    unique=True,
)

# ---------------------------------------------------------------------------
# Meaning-synset layer (D28)
# ---------------------------------------------------------------------------

meaning_synset = Table(
    "meaning_synset",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("canonical_label", Text, nullable=False, unique=True),
    Column("hypernym_id", Integer, ForeignKey("meaning_synset.id", ondelete="SET NULL")),
    Column("notes", Text),
)
Index("idx_meaning_synset_hypernym", meaning_synset.c.hypernym_id)

etymon_meaning_synset = Table(
    "etymon_meaning_synset",
    metadata,
    Column("etymon_id", Integer, ForeignKey("etymon.id", ondelete="CASCADE"), nullable=False),
    Column(
        "meaning_synset_id",
        Integer,
        ForeignKey("meaning_synset.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "fit",
        Text,
        CheckConstraint("fit IN ('core', 'peripheral')"),
        nullable=False,
    ),
    PrimaryKeyConstraint("etymon_id", "meaning_synset_id"),
)
Index("idx_etymon_meaning_synset_synset", etymon_meaning_synset.c.meaning_synset_id)

# ---------------------------------------------------------------------------
# Empirical-baseline priors (wyrd-ecjp.2, D36.4 / D36.7)
# ---------------------------------------------------------------------------

empirical_priors_native = Table(
    "empirical_priors_native",
    metadata,
    Column("culture", Text, nullable=False),
    Column("position", Text, nullable=False),
    Column("tag", Text, nullable=False),
    Column("era_midpoint", Integer, nullable=False),
    Column("lemma_ref", Text, nullable=False),
    Column("count", Integer, CheckConstraint("count > 0"), nullable=False),
    PrimaryKeyConstraint("culture", "position", "tag", "era_midpoint", "lemma_ref"),
)
Index(
    "idx_priors_native_cell",
    empirical_priors_native.c.culture,
    empirical_priors_native.c.position,
    empirical_priors_native.c.tag,
    empirical_priors_native.c.era_midpoint,
)
Index("idx_priors_native_lemma", empirical_priors_native.c.lemma_ref)

empirical_priors_loan = Table(
    "empirical_priors_loan",
    metadata,
    Column("donor", Text, nullable=False),
    Column("recipient", Text, nullable=False),
    Column("position", Text, nullable=False),
    Column("tag", Text, nullable=False),
    Column("era_midpoint", Integer, nullable=False),
    Column("lemma_ref", Text, nullable=False),
    Column("count", Integer, CheckConstraint("count > 0"), nullable=False),
    PrimaryKeyConstraint("donor", "recipient", "position", "tag", "era_midpoint", "lemma_ref"),
)
Index(
    "idx_priors_loan_cell",
    empirical_priors_loan.c.donor,
    empirical_priors_loan.c.recipient,
    empirical_priors_loan.c.position,
    empirical_priors_loan.c.tag,
    empirical_priors_loan.c.era_midpoint,
)
Index("idx_priors_loan_lemma", empirical_priors_loan.c.lemma_ref)

# ---------------------------------------------------------------------------
# Fantasy-morpheme research log (wyrd-ami / wyrd-rrse 0011)
# ---------------------------------------------------------------------------

fantasy_morpheme = Table(
    "fantasy_morpheme",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # COLLATE NOCASE so "Harpy" / "harpy" dedup under
    # UNIQUE(input_name, approach_version) — see migration 0011.
    # Same idiom as etymon_variant.form (line 201). Without the
    # collation here, ``alembic revision --autogenerate`` would
    # produce a spurious "drop COLLATE" diff against a live DB.
    Column("input_name", String(collation="NOCASE"), nullable=False),
    Column("input_description", Text),
    Column("usable", Integer, CheckConstraint("usable IN (0, 1)"), nullable=False),
    Column("etymon_id", Integer, ForeignKey("etymon.id", ondelete="SET NULL")),
    Column("bar_reason", Text),
    Column("resolution_method", Text, nullable=False),
    Column("approach_version", Text, nullable=False),
    Column("confidence", Text),
    Column("citation", Text),
    Column("reasoning", Text),
    Column("unapproved_language", Text),
    Column("unapproved_form", Text),
    Column("processed_at", Text, nullable=False),
    UniqueConstraint("input_name", "approach_version"),
)
Index("idx_fantasy_morpheme_approach", fantasy_morpheme.c.approach_version)
Index("idx_fantasy_morpheme_etymon", fantasy_morpheme.c.etymon_id)
Index("idx_fantasy_morpheme_usable", fantasy_morpheme.c.usable)
Index("idx_fantasy_morpheme_unapproved", fantasy_morpheme.c.unapproved_language)
