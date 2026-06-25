"""Single source of truth for the L2 round-trip column lists.

The L2 ``dump`` (DB → JSONL) and ``build`` (JSONL → DB) sides must agree on which
columns make the round trip, or a column written by the dumper but absent from the
inserter is silently dropped on ``rebuild-from-jsonl`` (a reconstruction-fidelity
defect, D21-adjacent). These tuples used to live as independent literals in
``dump.py`` and ``build.py`` — in sync, but one edit away from drifting apart with
no test covering the rarer columns (``transliteration`` / ``original_script`` / …).

Defining them ONCE here, and deriving the inserter's lists from them, makes that
drift impossible by construction: ``build``'s insert columns ARE these (the etymon
list verbatim; the source list is these plus the synthetic ``id``). A new L2 column
is added in one place and both sides pick it up.

Zero-dependency leaf module (imported by both ``dump`` and ``build``) so it can be
the shared base without an import cycle.
"""

from __future__ import annotations

# Etymon L2 state columns — the bare per-etymon scalar fields that round-trip
# through L2 (id, glosses, tags, and relational state are handled separately).
_ETYMON_L2_COLUMNS: tuple[str, ...] = (
    "canonical_form",
    "language",
    "modifier_type",
    "position_pref",
    "notes",
    "pronunciation_ipa",
    "pronunciation_dialect",
    "original_script",
    "transliteration",
)

# Source columns that round-trip through L2 (the row ``id`` is carried separately —
# it's the natural key the inserter supplies, not a dumped payload field).
_SOURCE_COLUMNS: tuple[str, ...] = (
    "author",
    "title",
    "year",
    "region",
    "language_focus",
    "notes",
)

# fantasy_morpheme scalar columns that round-trip through L2 (the ``etymon_id`` FK
# is carried separately by the inserter). dump SELECTs these; build INSERTs them.
_FANTASY_MORPHEME_COLUMNS: tuple[str, ...] = (
    "input_name",
    "input_description",
    "usable",
    "bar_reason",
    "resolution_method",
    "approach_version",
    "confidence",
    "citation",
    "reasoning",
    "unapproved_language",
    "unapproved_form",
    "processed_at",
)
