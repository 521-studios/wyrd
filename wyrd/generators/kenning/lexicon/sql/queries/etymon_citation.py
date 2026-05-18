"""Queries against the ``etymon_citation`` table.

The citation table holds the load-bearing extraction-witness evidence
counted by ``etymon_consensus`` for D4's ≥3-witness promotion
threshold. Dedupe semantics are subtle (see the long block comment
on the query below).
"""

from __future__ import annotations

# Insert an etymon_citation row only when one doesn't already exist
# for (etymon_id, source_id). Dedupe is wyrd-2pd: the unique index
# uses ``COALESCE(page, '')`` so a (etymon, source, NULL) row and a
# (etymon, source, '15') row are technically distinct keys. That
# means after ``backfill_citation_pages`` fills the page on an
# existing row, a re-mine that calls ``add_citation(page=None)``
# would slip past ``INSERT OR IGNORE`` and split the same evidence
# into two rows.
#
# The ``INSERT ... WHERE NOT EXISTS`` form below collapses the
# pre-check and write into a single atomic statement so two parallel
# writers can't both pass a pre-check and both insert. Matches the
# intended one-row-per-evidence-pair semantic the unique index meant
# to enforce.
INSERT_CITATION_IF_ABSENT = """
    INSERT INTO etymon_citation
        (etymon_id, source_id, page, short_quote, context_snippet)
    SELECT ?, ?, ?, ?, ?
    WHERE NOT EXISTS (
        SELECT 1 FROM etymon_citation
        WHERE etymon_id = ? AND source_id = ?
    )
"""
