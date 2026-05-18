"""Row-count statistics across the lexicon's authoritative tables.

``LexiconDB.stats`` returns ``{table_name: count}`` for the tables
that mining + ingestion actively write to. The list is intentionally
narrower than ``MetaData.tables`` — derived tables (toponym_decomposition,
empirical_priors_*) are bake artifacts, not authoritative state, and
including them in a generic ``stats()`` output would conflate the two.
"""

from __future__ import annotations

# Tables ``LexiconDB.stats`` reports counts for. Authoritative-only;
# derived tables (toponym_decomposition, etymon_period_form,
# etymon_meaning_synset, mining_run, empirical_priors_*) are
# rebuildable artifacts and don't belong in the "what's in the
# lexicon" headline number.
STATS_TABLES: tuple[str, ...] = (
    "source",
    "etymon",
    "etymon_gloss",
    "etymon_tag",
    "etymon_citation",
    "reflex",
    "reflex_etymon",
    "toponym",
    "toponym_attestation",
    "toponym_etymology",
    "toponym_etymology_element",
)


# Template for the per-table count query. The table name is
# substituted into the SQL string by ``LexiconDB.stats`` — safe
# because the values come exclusively from the hard-coded
# ``STATS_TABLES`` tuple above, never from user input.
STATS_COUNT_TEMPLATE = "SELECT COUNT(*) AS n FROM {table}"
