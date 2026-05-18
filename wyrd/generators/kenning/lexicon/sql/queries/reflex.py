"""Queries against the ``reflex`` and ``reflex_etymon`` tables.

The reflex layer maps modern surface fragments to the etymons they
descend from. A surface fragment can descend from multiple etymons
(ON býr / OE byrh both surface as -bury in some toponym endings), so
``reflex_etymon`` is a many-to-many join.

These three queries cover the seed-import flow used by
``seed_from_meanings``. The reflex layer is read-only after seed —
no scholarly mining touches it.
"""

from __future__ import annotations

# Lookup by the (surface_form, position) natural key. Called before
# ``INSERT_REFLEX`` to decide whether to upsert or insert; a future
# refactor could collapse the two into an ON CONFLICT statement
# similar to ``UPSERT_ETYMON``, but the present shape is the
# historical one and any change should ship with a perf comparison
# (this query runs once per modern_usage at seed time).
SELECT_REFLEX_BY_FORM = (
    "SELECT id FROM reflex WHERE surface_form = ? AND position = ?"
)

# Plain insert; the surface form is normalized at the caller, so we
# don't ON CONFLICT here. Callers guard with ``SELECT_REFLEX_BY_FORM``
# first.
INSERT_REFLEX = "INSERT INTO reflex (surface_form, position) VALUES (?, ?)"

# Composite-PK insert; duplicates are silently ignored. Multiple
# etymons mapping to the same (surface_form, position) is the whole
# point of the m2m table.
LINK_REFLEX_ETYMON_OR_IGNORE = (
    "INSERT OR IGNORE INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)"
)
