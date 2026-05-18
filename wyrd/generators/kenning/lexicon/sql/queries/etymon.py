"""Queries against the ``etymon`` table + its per-row gloss / tag children.

The ``etymon`` upsert is the trickiest write in the lexicon because
the wyrd-ha9q ``pronunciation_ipa`` / ``pronunciation_dialect`` pair
has to update atomically (you can't keep an existing dialect tag
describing an IPA you never stored). The CASE-on-conflict expression
encodes that contract; don't simplify without re-reading the
docstring on ``LexiconDB.upsert_etymon``.

The ``etymon_gloss`` and ``etymon_tag`` writes are plain
``INSERT OR IGNORE`` against composite primary keys.
"""

from __future__ import annotations

# Insert an etymon row if (canonical_form, language) is new, else
# fill any NULL fields without overwriting existing non-null values.
#
# pronunciation_ipa and pronunciation_dialect update as a unit so a
# dialect tag never ends up describing an IPA the row never stored.
# When the existing row HAS an IPA, BOTH the IPA and dialect stay
# at the existing values (incoming pair is dropped, even if the
# incoming dialect is non-NULL and the existing one is NULL). When
# the existing row's IPA is NULL, BOTH come from the incoming row
# via the COALESCE on IPA + the CASE on dialect.
#
# Returns the row id via SQLite's RETURNING clause so the caller
# doesn't pay for a follow-up SELECT.
UPSERT_ETYMON = """
    INSERT INTO etymon (
        canonical_form, language, modifier_type, position_pref, notes,
        pronunciation_ipa, pronunciation_dialect,
        original_script, transliteration
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(canonical_form, language) DO UPDATE SET
        modifier_type         = COALESCE(etymon.modifier_type, excluded.modifier_type),
        position_pref         = COALESCE(etymon.position_pref, excluded.position_pref),
        notes                 = COALESCE(etymon.notes, excluded.notes),
        pronunciation_ipa     = COALESCE(etymon.pronunciation_ipa, excluded.pronunciation_ipa),
        pronunciation_dialect = CASE
                                    WHEN etymon.pronunciation_ipa IS NULL
                                    THEN excluded.pronunciation_dialect
                                    ELSE etymon.pronunciation_dialect
                                END,
        original_script       = COALESCE(etymon.original_script, excluded.original_script),
        transliteration       = COALESCE(etymon.transliteration, excluded.transliteration)
    RETURNING id
"""


# Composite-PK insert; duplicates are silently ignored. Gloss strings
# are intentionally NOT normalized at write time (mining preserves
# scholar wording) so the same morpheme can carry several glosses
# from several sources.
INSERT_GLOSS_OR_IGNORE = "INSERT OR IGNORE INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)"


# Composite-PK insert; same idempotency contract as the gloss write.
INSERT_TAG_OR_IGNORE = "INSERT OR IGNORE INTO etymon_tag (etymon_id, tag) VALUES (?, ?)"
