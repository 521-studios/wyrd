"""Within-language stratum tagging (wyrd-lr4).

The etymon ``language`` column is too coarse for some palettes —
``language='welsh'`` blends Brittonic substrate, Latin loans,
medieval Welsh, and modern English borrowings into a single bucket.
A 'Welsh-flavored elven' generator pulling indiscriminately from all
of it produces stylistic mush.

The ``stratum`` column on ``etymon`` partitions each language's
etymons into named buckets so callers (the ``--stratum`` filter on
the generator, wyrd-lr4 Phase 3) can target a specific register.

Strata are language-specific — Welsh's strata don't transfer to
French or Old English. This module keeps the per-language
vocabularies + the heuristic that classifies each etymon by walking
its ``etymon_descent`` parent edges. ``_classify_family`` is the
shared engine; each language exposes ``classify_<lang>`` as a thin
wrapper that supplies its constants.

Phase 1 covered Welsh; Phase 4a adds French; Old English / Old
Norse follow.
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

# The default bucket — assigned to a welsh etymon whose
# ``etymon_descent`` parents don't include any language mapped to a
# more specific stratum. Named explicitly so the priority-order loop
# below doesn't have to slice ``WELSH_STRATA[:-1]`` and pin the
# default to a positional invariant.
_WELSH_DEFAULT_STRATUM: str = "native-welsh"

WELSH_STRATA: tuple[str, ...] = (
    "latin-loan",
    "english-loan",
    "brittonic-substrate",
    "medieval-welsh",
    _WELSH_DEFAULT_STRATUM,
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


# --- French strata --------------------------------------------------------
#
# Five mutually-exclusive buckets. Differs from the Welsh layout in
# one important way: French IS Latin-descended, so 'Latin in your
# ancestry' is not a useful loan signal — it'd catch every French
# word. The meaningful strata for French are SUBSTRATE markers
# (frankish, gaulish — pre-Latin influences) and the period self-
# language buckets (medieval-french, gallo-roman). Words without any
# of those markers fall into the native-french default.
#
# The umbrella ticket called for 'Frankish substrate + Gallo-Roman +
# medieval French + early modern + Occitan-influenced south'. We
# fold early-modern into native-french (no clear-cut signal in the
# descent graph distinguishes them) and drop occitan-influenced
# (only ~134 etymons under 'pro' in the live DB; geographically
# narrow, hard to detect via parent-language alone). Phase 4d
# (manual hand-correction) is the catch-all for ambiguous etymons
# the heuristic gets wrong.

_FRENCH_DEFAULT_STRATUM: str = "native-french"

FRENCH_STRATA: tuple[str, ...] = (
    "frankish-substrate",
    "gaulish-substrate",
    "gallo-roman",
    "medieval-french",
    _FRENCH_DEFAULT_STRATUM,
)


# Ancestor language → stratum bucket. An etymon with
# ``language='french'`` whose ``etymon_descent`` parents include any
# of these gets mapped to the corresponding stratum. Latin and
# vulgar-latin are deliberately ABSENT — every French word descends
# from Latin via the standard Romance path, so including them would
# collapse the entire bundle into 'gallo-roman' and erase the
# substrate distinctions.
_FRENCH_ANCESTOR_TO_STRATUM: dict[str, str] = {
    "frk": "frankish-substrate",
    "cel-gau": "gaulish-substrate",
}


# Self-language → stratum. An etymon whose OWN language is a French-
# family ancestor variety classifies by self-language. The grouping
# follows the same convention as Welsh: ancestor varieties (frk,
# cel-gau, vulgar-latin) get mapped to their substrate / loan
# stratum directly, period varieties (old-french, middle-french,
# norman-french) collapse into 'medieval-french'.
_FRENCH_SELF_LANGUAGE_TO_STRATUM: dict[str, str] = {
    "frk": "frankish-substrate",
    "cel-gau": "gaulish-substrate",
    "vulgar-latin": "gallo-roman",
    "old-french": "medieval-french",
    "middle-french": "medieval-french",
    "norman-french": "medieval-french",
}


# --- Old English strata ---------------------------------------------------
#
# Four buckets, one fewer than Welsh / French. The umbrella ticket
# called for 'West Saxon vs Mercian vs Anglian dialect strata; pre-
# Christian vs Christian-influenced vocabulary'. The dialect axis is
# unsupported by the actual data — no ``ang-x-*`` dialect codes
# appear on any etymon row in the live DB (live survey 2026-05-07).
# So the meaningful axis is loan / substrate signals: which post-
# native influence layered onto the standard Germanic descent.
#
# Like French, the standard descent path (gmw-pro / proto-germanic)
# is intentionally ABSENT from the ancestor map — every OE word
# descends from those, so including them would collapse the bundle
# into one bucket and erase the loan distinctions. Only meaningful
# loan / substrate ancestors go in the map.
#
# Christian-influenced vocabulary maps cleanly to ``latin-loan`` —
# Latin entered OE primarily via Christianization (~597 CE onward),
# bringing church / literacy / learned-borrowing layers. Greek loans
# of the same era arrived via Latin / clerical channels and are
# folded into the same bucket. Norse-loan covers the Danelaw period
# (~800-1000 CE). Celtic-substrate is rare (~60 etymons) but
# distinct: pre-Anglo-Saxon Brittonic substrate visible in some
# place names and topographical vocabulary.

_OLD_ENGLISH_DEFAULT_STRATUM: str = "native-old-english"

OLD_ENGLISH_STRATA: tuple[str, ...] = (
    "latin-loan",
    "norse-loan",
    "celtic-substrate",
    _OLD_ENGLISH_DEFAULT_STRATUM,
)


# Ancestor language → stratum bucket. An etymon with
# ``language='old-english'`` whose ``etymon_descent`` parents include
# any of these gets mapped to the corresponding stratum. The standard
# Germanic descent path (gmw-pro / proto-germanic / proto-indo-european)
# is intentionally absent — see the section comment above for the
# rationale.
#
# Latin variants (la-lat / la-med / la-ecc / vulgar-latin) and
# Christian-era Greek loans (which arrived in OE via Latin / clerical
# channels) all fold into ``latin-loan``. The umbrella ticket's
# 'Christian-influenced vocabulary' axis IS this bucket.
#
# Celtic ancestors include the proto / Brittonic codes plus
# old-irish, since some OE words borrowed from Goidelic via Christian
# missionary contact (post-597 CE Irish influence on Northumbrian
# Christianity).
_OLD_ENGLISH_ANCESTOR_TO_STRATUM: dict[str, str] = {
    "latin": "latin-loan",
    "la-lat": "latin-loan",
    "la-med": "latin-loan",
    "la-ecc": "latin-loan",
    "vulgar-latin": "latin-loan",
    "ancient-greek": "latin-loan",
    "old-norse": "norse-loan",
    "proto-celtic": "celtic-substrate",
    "cel-bry-pro": "celtic-substrate",
    "cel-bry": "celtic-substrate",
    "cel": "celtic-substrate",
    "old-irish": "celtic-substrate",
}


# Self-language → stratum. Empty for OE today — the live DB has no
# dialect or sub-period varieties (no ang-x-ws / ang-x-mer / etc.
# rows). If wave-2 mining ever ingests dialect-coded OE corpora,
# this map gets populated with the dialect-stratum mappings; the
# classifier's two-pass shape is ready for it.
_OLD_ENGLISH_SELF_LANGUAGE_TO_STRATUM: dict[str, str] = {}


def _stratum_to_ancestors(
    ancestor_to_stratum: dict[str, str],
) -> dict[str, set[str]]:
    """Invert the ancestor→stratum map so we can ask "which ancestor
    languages would put an etymon in stratum S?"."""
    inverted: dict[str, set[str]] = {}
    for lang, stratum in ancestor_to_stratum.items():
        inverted.setdefault(stratum, set()).add(lang)
    return inverted


def _classify_family(
    db: LexiconDB,
    *,
    modern_lang: str,
    strata_order: tuple[str, ...],
    ancestor_to_stratum: dict[str, str],
    self_lang_to_stratum: dict[str, str],
    default_stratum: str,
) -> dict[int, str]:
    """Generic classifier shared by classify_welsh / classify_french /
    (and Phase 4b/4c follow-ons).

    Two passes:
    1. Self-language pass: every etymon whose ``language`` is a key
       in ``self_lang_to_stratum`` gets assigned the corresponding
       stratum directly. This covers ancestor varieties (proto-
       Brittonic, Frankish, vulgar-Latin, ...) and period varieties
       (middle-welsh, old-french, ...) that ARE the ancestor — their
       own descent walk would describe what fed INTO them, not their
       own register.
    2. Ancestor-walk pass: every etymon whose ``language ==
       modern_lang`` gets its parent-language set built via a single
       chunked join over ``etymon_descent``, then iterates
       ``strata_order`` and picks the first stratum whose
       ancestor-language set intersects the parent set.

    Both passes skip ``merged_into_id IS NOT NULL`` rows — OCR-
    clustering losers that callers already filter out.

    Pure read; doesn't mutate the DB. Caller decides whether to write
    the proposed assignments back to ``etymon.stratum``.
    """
    proposals: dict[int, str] = {}

    for lang, stratum in self_lang_to_stratum.items():
        rows = db.conn.execute(
            "SELECT id FROM etymon WHERE language = ? AND merged_into_id IS NULL",
            (lang,),
        )
        for r in rows:
            proposals[r["id"]] = stratum

    modern_rows = db.conn.execute(
        "SELECT id FROM etymon WHERE language = ? AND merged_into_id IS NULL",
        (modern_lang,),
    ).fetchall()
    if not modern_rows:
        return proposals

    # Pull every parent-language for the modern-variety etymon set
    # in one pass. Chunked at SQLite's 999-variable limit. The inner
    # SELECT keys on ``child_id`` so ``idx_etymon_descent_child``
    # carries the lookup.
    modern_ids = [r["id"] for r in modern_rows]
    parents_by_child: dict[int, set[str]] = {}
    for i in range(0, len(modern_ids), 999):
        chunk = modern_ids[i : i + 999]
        placeholders = ",".join("?" * len(chunk))
        cur = db.conn.execute(
            f"SELECT d.child_id, e.language "  # noqa: S608 — placeholders are ints
            f"FROM etymon_descent d JOIN etymon e ON d.parent_id = e.id "
            f"WHERE d.child_id IN ({placeholders})",
            chunk,
        )
        for row in cur:
            parents_by_child.setdefault(row["child_id"], set()).add(row["language"])

    stratum_ancestors = _stratum_to_ancestors(ancestor_to_stratum)

    for eid in modern_ids:
        parent_langs = parents_by_child.get(eid, set())
        assigned = default_stratum
        # Iterate strata in priority order; first stratum whose
        # ancestor-language set intersects this etymon's parent set
        # wins. The default stratum has no ancestor-language mapping
        # so it naturally falls through.
        for stratum_target in strata_order:
            if stratum_target == default_stratum:
                continue
            if parent_langs & stratum_ancestors.get(stratum_target, set()):
                assigned = stratum_target
                break
        proposals[eid] = assigned

    return proposals


def classify_welsh(db: LexiconDB) -> dict[int, str]:
    """Return ``{etymon_id: stratum}`` for the Welsh family.

    Covers etymons whose ``language`` is one of:
    - ``welsh`` — classified by ancestor language via ``etymon_descent``.
    - ``middle-welsh``, ``old-welsh``, ``cel-bry-pro`` — classified by
      self-language (these ARE the ancestors of welsh).

    Pure read. Skips ``merged_into_id IS NOT NULL`` rows (OCR-cluster
    losers).
    """
    return _classify_family(
        db,
        modern_lang="welsh",
        strata_order=WELSH_STRATA,
        ancestor_to_stratum=_WELSH_ANCESTOR_TO_STRATUM,
        self_lang_to_stratum=_WELSH_SELF_LANGUAGE_TO_STRATUM,
        default_stratum=_WELSH_DEFAULT_STRATUM,
    )


def classify_french(db: LexiconDB) -> dict[int, str]:
    """Return ``{etymon_id: stratum}`` for the French family.

    Covers etymons whose ``language`` is one of:
    - ``french`` — classified by ancestor language via ``etymon_descent``.
    - ``old-french`` / ``middle-french`` / ``norman-french`` —
      classified as ``medieval-french`` via self-language.
    - ``vulgar-latin`` — classified as ``gallo-roman`` (the
      ancestor stage between Latin and Old French).
    - ``frk`` (Frankish) / ``cel-gau`` (Gaulish) — classified by
      self-language as their respective substrate.

    Pure read. Skips ``merged_into_id IS NOT NULL`` rows.

    Note on Latin: Latin is intentionally NOT in the ancestor map.
    Every French word descends from Latin via the standard Romance
    path; treating that as a 'gallo-roman loan' would collapse the
    whole bundle into one stratum and erase the substrate
    distinctions that motivated this classifier in the first place.
    """
    return _classify_family(
        db,
        modern_lang="french",
        strata_order=FRENCH_STRATA,
        ancestor_to_stratum=_FRENCH_ANCESTOR_TO_STRATUM,
        self_lang_to_stratum=_FRENCH_SELF_LANGUAGE_TO_STRATUM,
        default_stratum=_FRENCH_DEFAULT_STRATUM,
    )


def classify_old_english(db: LexiconDB) -> dict[int, str]:
    """Return ``{etymon_id: stratum}`` for the Old English family.

    Covers etymons whose ``language`` is ``old-english``, classified
    by ancestor language via ``etymon_descent``. The Welsh / French
    classifiers also handle ancestor varieties (cel-bry-pro,
    middle-welsh, vulgar-latin, frk, ...) via the self-language map;
    the OE classifier has no equivalent because the live DB has no
    dialect or sub-period varieties (no ang-x-* rows).

    Pure read. Skips ``merged_into_id IS NOT NULL`` rows.

    Note on Germanic descent: gmw-pro / proto-germanic / proto-indo-
    european are intentionally NOT in the ancestor map. Every OE
    word descends from those via the standard Germanic path; treating
    that as a 'germanic-inheritance' loan would collapse the bundle
    into one bucket and erase the distinct loan signals (Latin from
    Christianization, Norse from Danelaw, Celtic from substrate
    contact). Same Latin-absence rationale as classify_french —
    pinned by a regression test.
    """
    return _classify_family(
        db,
        modern_lang="old-english",
        strata_order=OLD_ENGLISH_STRATA,
        ancestor_to_stratum=_OLD_ENGLISH_ANCESTOR_TO_STRATUM,
        self_lang_to_stratum=_OLD_ENGLISH_SELF_LANGUAGE_TO_STRATUM,
        default_stratum=_OLD_ENGLISH_DEFAULT_STRATUM,
    )


__all__ = [
    "FRENCH_STRATA",
    "OLD_ENGLISH_STRATA",
    "WELSH_STRATA",
    "classify_french",
    "classify_old_english",
    "classify_welsh",
]
