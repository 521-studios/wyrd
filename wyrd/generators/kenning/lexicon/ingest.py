"""Persist parsed corpus entries into the lexicon (D21).

Single public function ``ingest_parsed_entries`` writes
``ParsedEntry`` rows from any of the dictionary parsers
(Mawer / Skeat / Ekwall / etc.) into ``toponym`` +
``toponym_etymology`` + ``etymon`` + supporting tables.

Per-toponym dedupe is on ``(modern_name, country='', region)`` via
``_upsert_toponym`` — re-running against the same source for the
same place reuses the existing toponym row. The
``toponym_etymology`` write itself is NOT idempotent: there is no
UNIQUE constraint on ``(toponym_id, source_id)``, so re-ingesting
the same source will produce duplicate etymology rows. Re-mining
workflows are expected to clear the prior etymology rows for that
source first.

Returns a counts dict (``toponyms`` / ``etymologies`` /
``elements`` / ``etymons_touched``) for mining-progress callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wyrd.generators.kenning.lexicon.db import LexiconDB

if TYPE_CHECKING:
    from wyrd.generators.kenning.skeat_parser import ParsedEntry


def _upsert_toponym(db: LexiconDB, modern_name: str, region: str | None) -> int:
    cur = db.conn.execute(
        """
        SELECT id FROM toponym
        WHERE modern_name = ?
          AND COALESCE(country, '') = ''
          AND COALESCE(region, '') = COALESCE(?, '')
        """,
        (modern_name, region),
    )
    row = cur.fetchone()
    if row is not None:
        return row["id"]
    cur = db.conn.execute(
        "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
        (modern_name, region),
    )
    return cur.lastrowid


def ingest_parsed_entries(
    db: LexiconDB,
    parsed_entries: list[ParsedEntry],
    source_id: str,
    *,
    region: str | None = None,
) -> dict[str, int]:
    """Persist ParsedEntry rows from a Skeat-style parser into the DB.

    For each entry:
      - Upsert toponym row (region carries the source's coverage area).
      - For HIGH/MEDIUM confidence: insert a toponym_etymology row pointing
        at <source_id>, plus one toponym_etymology_element per parsed
        element. Each element's etymon is upserted, glosses/tags attached,
        and a citation row added.
      - For LOW confidence: only the toponym row is written, so we still
        record that the source covers this place even when we couldn't
        recover a breakdown.

    Returns a counts dict for sanity-checking.
    """
    counts = {
        "toponyms": 0,
        "etymologies": 0,
        "elements": 0,
        "etymons_touched": 0,
    }
    for entry in parsed_entries:
        toponym_id = _upsert_toponym(db, entry.toponym, region)
        counts["toponyms"] += 1

        if entry.confidence == "low" or not entry.elements:
            continue

        confidence = entry.confidence if entry.confidence in ("high", "medium", "low") else "low"
        cur = db.conn.execute(
            """
            INSERT INTO toponym_etymology
                (toponym_id, source_id, historical_form, confidence, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                toponym_id,
                source_id,
                entry.historical_form,
                confidence,
                entry.source_quote or None,
            ),
        )
        etymology_id = cur.lastrowid
        counts["etymologies"] += 1

        for ordinal, elem in enumerate(entry.elements):
            etymon_id = db.upsert_etymon(elem.form, elem.language)
            counts["etymons_touched"] += 1
            if elem.gloss:
                db.add_gloss(etymon_id, elem.gloss)
            db.add_citation(etymon_id, source_id, short_quote=entry.source_quote or None)
            db.conn.execute(
                """
                INSERT INTO toponym_etymology_element
                    (toponym_etymology_id, ordinal, etymon_id, inflection, surface_in_modern)
                VALUES (?, ?, ?, ?, ?)
                """,
                (etymology_id, ordinal, etymon_id, elem.inflection, None),
            )
            counts["elements"] += 1

    db.commit()
    return counts
