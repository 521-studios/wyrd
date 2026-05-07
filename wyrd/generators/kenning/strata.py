"""Within-language stratum tagging (wyrd-lr4).

The etymon ``language`` column is too coarse for some palettes —
``language='welsh'`` blends Brittonic substrate, Latin loans,
medieval Welsh, and modern English borrowings into a single bucket.
A 'Welsh-flavored elven' generator pulling indiscriminately from all
of it produces stylistic mush.

The ``stratum`` column on ``etymon`` partitions each language's
etymons into named buckets so callers (eventually a ``--stratum``
filter on the generator, deferred to a follow-up) can target a
specific register.

Strata are language-specific — Welsh's strata don't transfer to
French or Old English. This module keeps the per-language
vocabularies + the heuristic that classifies each etymon by walking
its ``etymon_descent`` parent edges.

Welsh is the only language family covered in the first wyrd-lr4 PR;
French / Old English / Old Norse follow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wyrd.generators.kenning.lexicon import LexiconDB


# --- Welsh strata ---------------------------------------------------------
#
# Five mutually-exclusive buckets. Listed in priority order: when an
# etymon's ``etymon_descent`` parents include languages mapped to
# multiple strata, the first match wins. The order encodes scholarly
# convention: explicit loans displace inherited forms when both
# ancestors are present (a Welsh word that descends from BOTH Latin
# and a Brittonic root is classified as a Latin loan, because the
# Latin path is the load-bearing one for register).

WELSH_STRATA: tuple[str, ...] = (
    "latin-loan",
    "english-loan",
    "brittonic-substrate",
    "medieval-welsh",
    "native-welsh",
)


# Ancestor language → stratum bucket. An etymon with
# ``language='welsh'`` whose ``etymon_descent`` parents include any
# of these gets mapped to the corresponding stratum.
_WELSH_ANCESTOR_TO_STRATUM: dict[str, str] = {
    "latin": "latin-loan",
    "modern-english": "english-loan",
    "middle-english": "english-loan",
    "old-english": "english-loan",
    "english": "english-loan",
    "cel-bry-pro": "brittonic-substrate",
    "proto-celtic": "brittonic-substrate",
    "old-welsh": "brittonic-substrate",
    "middle-welsh": "medieval-welsh",
}


# Self-language → stratum. Used when the etymon's OWN ``language`` is
# a Welsh-family ancestor variety. A middle-welsh etymon is
# medieval-welsh regardless of what its parents look like (it IS the
# ancestor; classifying it by descent would only describe what fed
# INTO the medieval form, not the form's own register).
_WELSH_SELF_LANGUAGE_TO_STRATUM: dict[str, str] = {
    "cel-bry-pro": "brittonic-substrate",
    "old-welsh": "brittonic-substrate",
    "middle-welsh": "medieval-welsh",
}


def _stratum_to_ancestors(
    ancestor_to_stratum: dict[str, str],
) -> dict[str, set[str]]:
    """Invert the ancestor→stratum map so we can ask "which ancestor
    languages would put an etymon in stratum S?"."""
    inverted: dict[str, set[str]] = {}
    for lang, stratum in ancestor_to_stratum.items():
        inverted.setdefault(stratum, set()).add(lang)
    return inverted


def classify_welsh(db: LexiconDB) -> dict[int, str]:
    """Return ``{etymon_id: stratum}`` for the Welsh family.

    Covers etymons whose ``language`` is one of:
    - ``welsh`` — classified by ancestor language via ``etymon_descent``.
    - ``middle-welsh``, ``old-welsh``, ``cel-bry-pro`` — classified by
      self-language (these ARE the ancestors of welsh).

    Pure read; doesn't mutate the DB. The caller decides whether to
    write the proposed assignments to the ``stratum`` column.

    Skips ``merged_into_id IS NOT NULL`` rows — those have been folded
    into a canonical winner by OCR clustering and shouldn't carry an
    independent stratum.
    """
    proposals: dict[int, str] = {}

    for lang, stratum in _WELSH_SELF_LANGUAGE_TO_STRATUM.items():
        rows = db.conn.execute(
            "SELECT id FROM etymon WHERE language = ? AND merged_into_id IS NULL",
            (lang,),
        )
        for r in rows:
            proposals[r["id"]] = stratum

    welsh_rows = db.conn.execute(
        "SELECT id FROM etymon WHERE language = 'welsh' AND merged_into_id IS NULL"
    ).fetchall()
    if not welsh_rows:
        return proposals

    # Pull every parent-language for the welsh etymon set in one pass.
    # Chunked at SQLite's 999-variable limit. The inner SELECT keys on
    # ``child_id`` so the index ``idx_etymon_descent_child`` carries
    # the lookup. ``GROUP_CONCAT`` would also work but the row-iteration
    # form keeps the parent-language set construction obvious.
    welsh_ids = [r["id"] for r in welsh_rows]
    parents_by_child: dict[int, set[str]] = {}
    for i in range(0, len(welsh_ids), 999):
        chunk = welsh_ids[i : i + 999]
        placeholders = ",".join("?" * len(chunk))
        cur = db.conn.execute(
            f"SELECT d.child_id, e.language "  # noqa: S608 — placeholders are ints
            f"FROM etymon_descent d JOIN etymon e ON d.parent_id = e.id "
            f"WHERE d.child_id IN ({placeholders})",
            chunk,
        )
        for row in cur:
            parents_by_child.setdefault(row["child_id"], set()).add(row["language"])

    stratum_ancestors = _stratum_to_ancestors(_WELSH_ANCESTOR_TO_STRATUM)

    for eid in welsh_ids:
        parent_langs = parents_by_child.get(eid, set())
        assigned = "native-welsh"
        # Iterate strata in priority order; first stratum whose
        # ancestor-language set intersects this etymon's parent set wins.
        # ``WELSH_STRATA[-1]`` is the default 'native-welsh' — skip it.
        for stratum_target in WELSH_STRATA[:-1]:
            if parent_langs & stratum_ancestors.get(stratum_target, set()):
                assigned = stratum_target
                break
        proposals[eid] = assigned

    return proposals


__all__ = [
    "WELSH_STRATA",
    "classify_welsh",
]
