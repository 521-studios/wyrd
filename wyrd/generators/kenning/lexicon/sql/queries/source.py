"""Queries against the ``source`` table.

One bibliographic row per scholarly source the lexicon mines. The
upsert below is the only write path; reads are inline at call sites
(``SELECT * FROM source WHERE id = ?`` etc.) and aren't worth a
named constant.
"""

from __future__ import annotations

# Insert a new bibliographic source row, or refresh every metadata
# field of an existing row. Used by ``LexiconDB.upsert_source`` and
# by the L2 JSONL replay path (the canonical-state ``source`` row at
# the top of each ``data/mining/<id>.jsonl`` file).
UPSERT_SOURCE = """
    INSERT INTO source (id, author, title, year, region, language_focus, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        author = excluded.author,
        title = excluded.title,
        year = excluded.year,
        region = excluded.region,
        language_focus = excluded.language_focus,
        notes = excluded.notes
"""
