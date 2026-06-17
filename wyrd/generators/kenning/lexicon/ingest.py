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

from wyrd.generators.kenning.lexicon.controlled_vocab import (
    canonicalize_country,
    canonicalize_language,
)
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.region_model import canonicalize_region
from wyrd.generators.kenning.lexicon.regions import country_for_region

if TYPE_CHECKING:
    from wyrd.generators.kenning.parsers.skeat import ParsedEntry


def _upsert_toponym(
    db: LexiconDB,
    modern_name: str,
    region: str | None,
    country: str | None = None,
) -> int:
    """Upsert a ``toponym`` row keyed on the (name, country, region)
    unique index. ``country`` defaults to the region-derived value
    from :func:`country_for_region` — callers only need to pass it
    explicitly to override the derivation (or to set country when no
    region is known).

    Migration-window edge case: when a legacy NULL-country row already
    exists for (name, region) and the caller now derives a non-null
    country, the exact SELECT misses (NULL vs derived-value). Instead
    of letting that create a twin row, opportunistically UPDATE the
    legacy row in place — same effect as running
    ``backfill_toponym_country`` against just that row. Idempotent +
    self-repairing across the migration window.

    Region is validated + canonicalized fail-closed at this boundary
    (wyrd-3q6m.3): an alias variant is folded to its canonical node, and an
    England-scoped value that is neither a node nor an alias raises
    ``RegionValidationError`` rather than silently creating a new bucket.
    """
    # Fail-closed Class-A guards (wyrd-3q6m.3/.5). COUNTRY FIRST (wyrd-61p9): the
    # canonical country scopes the region check (canonicalize_region keys its
    # England-scoping on `country`, so a country alias must be folded before that).
    # No alias folds to England today, but this keeps the order robust + uniform
    # across the three ingest guards. An explicit country is folded+validated; when
    # absent it is derived from the canonical region (regions.py only yields
    # canonical countries — pinned by test_every_region_derived_country_is_canonical).
    country = canonicalize_country(country)
    region = canonicalize_region(region, country=country)
    if country is None:
        country = country_for_region(region)
    cur = db.conn.execute(
        """
        SELECT id FROM toponym
        WHERE modern_name = ?
          AND COALESCE(country, '') = COALESCE(?, '')
          AND COALESCE(region, '') = COALESCE(?, '')
        """,
        (modern_name, country, region),
    )
    row = cur.fetchone()
    if row is not None:
        return row["id"]
    # Migration-window self-repair: if a legacy NULL-country row exists
    # for (name, region), and the new caller has derived a non-null
    # country, backfill the row in place rather than creating a twin.
    if country is not None:
        legacy = db.conn.execute(
            """
            SELECT id FROM toponym
            WHERE modern_name = ?
              AND country IS NULL
              AND COALESCE(region, '') = COALESCE(?, '')
            """,
            (modern_name, region),
        ).fetchone()
        if legacy is not None:
            db.conn.execute(
                "UPDATE toponym SET country = ? WHERE id = ?",
                (country, legacy["id"]),
            )
            return legacy["id"]
    cur = db.conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
        (modern_name, country, region),
    )
    return cur.lastrowid


def ingest_parsed_entries(
    db: LexiconDB,
    parsed_entries: list[ParsedEntry],
    source_id: str,
    *,
    region: str | None = None,
    country: str | None = None,
) -> dict[str, int]:
    """Persist ParsedEntry rows from a Skeat-style parser into the DB.

    For each entry:
      - Upsert toponym row (region carries the source's coverage area;
        country defaults to the region-derived value, so historical
        callers that only pass ``region`` automatically populate country
        too — fixes the empirical-priors miner's ``country_unknown``
        gap that was zeroing out the English baseline).
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
        has_breakdown = entry.confidence != "low" and bool(entry.elements)
        # Fail-closed language guard (wyrd-3q6m.5): validate every element's
        # language BEFORE any write for this entry — including the toponym upsert
        # below — so a bad language rejects the whole entry without persisting a
        # partial toponym/etymology. Curated path only; bulk wiktextract uses ISO
        # codes and never reaches here.
        elem_languages = (
            [canonicalize_language(elem.language) for elem in entry.elements]
            if has_breakdown
            else []
        )

        toponym_id = _upsert_toponym(db, entry.toponym, region, country)
        counts["toponyms"] += 1
        if not has_breakdown:
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
            etymon_id = db.upsert_etymon(elem.form, elem_languages[ordinal])
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
