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
