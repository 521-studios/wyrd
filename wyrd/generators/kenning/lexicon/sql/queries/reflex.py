"""Queries against the ``reflex`` and ``reflex_etymon`` tables.

The reflex layer maps modern surface fragments to the etymons they
descend from. A surface fragment can descend from multiple etymons
(ON býr / OE byrh both surface as -bury in some toponym endings), so
``reflex_etymon`` is a many-to-many join.

The two queries cover the seed-import flow used by
``seed_from_meanings``. The reflex layer is read-only after seed —
no scholarly mining touches it.
"""

from __future__ import annotations

# Insert a reflex row, or return the existing row id if
# (surface_form, position) already maps. ``DO UPDATE SET
# position = position`` is a no-op-update trick that lets RETURNING
# fire on the conflict path as well as the insert path — without it,
# RETURNING on an INSERT-OR-IGNORE only returns rows the insert
# actually created. Collapsing the original SELECT-then-INSERT pair
# into one round-trip also kills the TOCTOU window where two parallel
# seed runs could both pass the SELECT and both insert the same row.
UPSERT_REFLEX = """
    INSERT INTO reflex (surface_form, position)
    VALUES (?, ?)
    ON CONFLICT(surface_form, position) DO UPDATE SET position = position
    RETURNING id
"""

# Composite-PK insert; duplicates are silently ignored. Multiple
# etymons mapping to the same (surface_form, position) is the whole
# point of the m2m table.
LINK_REFLEX_ETYMON_OR_IGNORE = (
    "INSERT OR IGNORE INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)"
)

# wyrd-ned5 + wyrd-br5o: dump the reflex layer for the ``_reflexes.jsonl``
# L2 round-trip. One row per (reflex × linked etymon); the dumper groups
# by (surface_form, position) and folds the etymon forms into an
# ``etymon_refs`` list. The etymon side is a LEFT JOIN so ORPHAN reflexes
# (no reflex_etymon link — the generation-surface layer seeded from the
# rando meanings) ALSO round-trip, as rows with ``etymon_refs: []``;
# wyrd-ned5's INNER JOIN silently dropped them, so a clean rebuild lost
# ~4k generation surfaces (wyrd-br5o). The merged_into_id filter rides on
# the JOIN (not WHERE) so it skips OCR-cluster losers (D22) WITHOUT turning
# the LEFT JOIN back into an inner one (a WHERE on the nullable side would).
# Sorted for byte-stable output.
SELECT_REFLEXES_FOR_DUMP = """
    SELECT r.surface_form, r.position, e.language, e.canonical_form
    FROM reflex r
    LEFT JOIN reflex_etymon re ON re.reflex_id = r.id
    LEFT JOIN etymon e ON e.id = re.etymon_id AND e.merged_into_id IS NULL
    ORDER BY r.surface_form, r.position, e.language, e.canonical_form
"""

# wyrd-14p: derive reflex.productivity from corpus observations.
#
# For each reflex (id, surface_form, position), count the distinct
# toponyms whose breakdown contains an element pointing at one of the
# reflex's etymons AT a matching position. Position match: the
# element's position is derived from (ordinal, element_count) — first
# = 'pre', last = 'post', middle = 'inner'; single-element toponyms
# count as 'pre' (matches _position_label in empirical_priors).
#
# Returns (reflex_id, distinct_toponym_count) for every reflex that
# has at least one matching toponym. Reflexes with no observed
# toponym usage are left implicit (caller can read them as 0).
SELECT_REFLEX_PRODUCTIVITY_COUNTS = """
WITH element_counts AS (
    -- Compute how many elements each breakdown has, so we can derive
    -- a per-element position from its ordinal. Reused below.
    SELECT toponym_etymology_id, COUNT(*) AS n
    FROM toponym_etymology_element
    GROUP BY toponym_etymology_id
),
element_positions AS (
    SELECT
        tee.toponym_etymology_id,
        tee.etymon_id,
        CASE
            WHEN ec.n = 1            THEN 'pre'
            WHEN tee.ordinal = 0     THEN 'pre'
            WHEN tee.ordinal = ec.n - 1 THEN 'post'
            ELSE 'inner'
        END AS position
    FROM toponym_etymology_element AS tee
    JOIN element_counts AS ec
      ON ec.toponym_etymology_id = tee.toponym_etymology_id
)
SELECT
    re.reflex_id,
    COUNT(DISTINCT te.toponym_id) AS productivity
FROM reflex_etymon AS re
JOIN reflex AS r
  ON r.id = re.reflex_id
JOIN element_positions AS ep
  ON ep.etymon_id = re.etymon_id
 AND ep.position = r.position
JOIN toponym_etymology AS te
  ON te.id = ep.toponym_etymology_id
GROUP BY re.reflex_id
"""

# Bulk-zero before recompute so reflexes that lost all corpus support
# (because etymons were merged / removed) read 0 again rather than
# carrying a stale count. Pair with SELECT_REFLEX_PRODUCTIVITY_COUNTS
# + UPDATE_REFLEX_PRODUCTIVITY in one transaction for atomicity.
RESET_REFLEX_PRODUCTIVITY = "UPDATE reflex SET productivity = 0"

UPDATE_REFLEX_PRODUCTIVITY = "UPDATE reflex SET productivity = ? WHERE id = ?"
