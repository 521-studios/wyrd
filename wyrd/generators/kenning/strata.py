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
# Four buckets, one fewer than Welsh / French. Listed in priority
# order: when an etymon's parents include languages mapped to
# multiple strata, the first match wins (so latin-loan displaces
# norse-loan displaces celtic-substrate). The order encodes a
# scholarly-historical convention rooted in how each layer entered
# the language: Latin came through the institutional pipe of
# Christianization (clerical, literate, top-down vocabulary that
# permeated all dialects), Norse came through bottom-up settler
# contact in a defined region (the Danelaw, ~800-1000 CE), and
# Celtic substrate is the residual pre-Anglo-Saxon layer plus
# post-conversion Goidelic influence via Northumbrian Christianity.
# When evidence of multiple layers coexists on a single etymon, the
# institutional / clerical signal carries the highest semantic
# weight, the regional-settler signal next, and the residual /
# substrate signal last.
#
# The umbrella ticket called for 'West Saxon vs Mercian vs Anglian
# dialect strata; pre-Christian vs Christian-influenced vocabulary'.
# The dialect axis is mostly unsupported by the actual data — the
# live DB has 33 dialect-coded etymons in total (31 ``ang-ang``
# (Anglian) + 2 ``ang-wsx`` (West Saxon), live survey 2026-05-07),
# all orphans without descent edges. Phase 4b leaves them in
# language-scope-outside the OE classifier (modern_lang is
# 'old-english' only, self-language map empty), so they don't get
# tagged today. A Phase 4d (manual hand-correction) or wave-2
# dialect-corpus follow-up would extend the self-language map
# and/or modern_lang scope; the classifier's two-pass shape is
# ready for either. So the meaningful axis on Phase 4b is loan /
# substrate signals: which post-native influence layered onto the
# standard Germanic descent. Pre-Christian native vocabulary falls
# into ``native-old-english`` by default.
#
# Like French, the standard descent path (gmw-pro / proto-germanic /
# proto-indo-european) is intentionally ABSENT from the ancestor
# map — every OE word descends from those, so including them would
# collapse the bundle into one bucket and erase the loan
# distinctions. Only meaningful loan / substrate ancestors go in
# the map.
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


# Self-language → stratum. Empty for OE today. The live DB has 33
# dialect-coded rows (31 ``ang-ang`` Anglian + 2 ``ang-wsx`` West
# Saxon, live survey 2026-05-07) but they're all orphans without
# descent edges, the dialect tagging is too sparse to support its
# own bucket, and the per-dialect OE strata the umbrella ticket
# scoped (West Saxon / Mercian / Anglian) would need a wave-2
# dialect-corpus ingest to populate meaningfully. Until then the
# self-language map stays empty and dialect-coded rows stay
# unclassified. The classifier's two-pass shape is ready when the
# data lands.
_OLD_ENGLISH_SELF_LANGUAGE_TO_STRATUM: dict[str, str] = {}


# --- Old Norse strata -----------------------------------------------------
#
# Six buckets: the standard loan/substrate set plus an east-norse
# bucket for the umbrella's Eastern (Old Swedish / Old Danish) vs
# Western (default) dialect axis. Listed in priority order: when an
# etymon's parents include languages mapped to multiple strata, the
# first match wins. Same convention as Welsh / French / OE —
# institutional / clerical signals win, then contact loans, then
# substrate, then dialect axis, then default.
#
# Umbrella ticket axis: 'Old Norse: Eastern (Swedish/Danish) vs
# Western (Norwegian/Icelandic) traditions.' Captured by
# ``east-norse`` (gmq-osw + gmq-oda via self-language) vs the
# ``native-old-norse`` default.
#
# Like French / OE, the standard Germanic descent path
# (proto-germanic / proto-indo-european / gmq-pro / gmw-pro) is
# intentionally ABSENT from the ancestor map. Every Old Norse word
# descends from those, so including them would collapse the bundle
# into one bucket and erase the loan distinctions.
#
# Loan / substrate signal sourcing:
#   * latin-loan: latin (74 parents in live DB) + ancient-greek (20)
#     — Christianization-era learned vocabulary, same convention
#     as OE for Greek-via-Latin folding.
#   * low-german-loan: gml (Middle Low German, 442 parents) — the
#     Hanseatic / merchant register layer, post-1100 CE Northern
#     European trade contact.
#   * english-loan: old-english (164) + osx (Old Saxon, 95) — Anglo-
#     Saxon and Continental Saxon contact via Viking-Age trade and
#     settlement. Mirrors the Welsh classifier's english-loan
#     bucket (which folds modern / middle / old English variants).
#   * gaelic-substrate: old-irish (43) + middle-irish (11) +
#     proto-celtic (10) — Norse-Gaelic contact in the Irish Sea
#     region (Dublin, Mann, Western Isles). The bucket name implies
#     Goidelic (Q-Celtic) specifically; ``proto-celtic`` parents are
#     the common ancestor of both Goidelic and Brittonic
#     (P-Celtic) branches, so a small minority could in principle
#     be Brittonic-flavored. The Norse-Brittonic contact zone
#     (Galloway, Cumbria) is much smaller than the Norse-Gaelic
#     one historically, so 'gaelic' is the load-bearing register
#     name even though the ancestor signal admits both branches.
#
# East-Norse fires only via self-language (gmq-osw / gmq-oda); the
# ancestor walk doesn't apply because old-norse is the PARENT of
# Old Swedish / Old Danish, not vice versa. So an old-norse-language
# row never gets east-norse classification — by design.

_OLD_NORSE_DEFAULT_STRATUM: str = "native-old-norse"

OLD_NORSE_STRATA: tuple[str, ...] = (
    "latin-loan",
    "low-german-loan",
    "english-loan",
    "gaelic-substrate",
    "east-norse",
    _OLD_NORSE_DEFAULT_STRATUM,
)


# Ancestor language → stratum bucket. An etymon with
# ``language='old-norse'`` whose ``etymon_descent`` parents include
# any of these gets mapped to the corresponding stratum. Standard
# Germanic descent (proto-germanic / proto-indo-european / gmq-pro
# / gmw-pro) is intentionally absent — see the section comment above
# for the rationale.
#
# Latin variants (la-lat / la-med / la-ecc / vulgar-latin) and
# Christian-era Greek loans fold into ``latin-loan`` — same channel
# convention as OE.
_OLD_NORSE_ANCESTOR_TO_STRATUM: dict[str, str] = {
    "latin": "latin-loan",
    "la-lat": "latin-loan",
    "la-med": "latin-loan",
    "la-ecc": "latin-loan",
    "vulgar-latin": "latin-loan",
    "ancient-greek": "latin-loan",
    "gml": "low-german-loan",
    "old-english": "english-loan",
    "osx": "english-loan",
    "old-irish": "gaelic-substrate",
    "middle-irish": "gaelic-substrate",
    "proto-celtic": "gaelic-substrate",
}


# Self-language → stratum. East Norse is captured here: an etymon
# whose own language is gmq-osw (Old Swedish) or gmq-oda (Old
# Danish) classifies as ``east-norse`` directly. These ARE the
# Eastern Norse varieties — classifying them by descent would
# describe what fed INTO them (mostly old-norse itself) rather
# than their own register.
#
# gmq-pro (Proto-Norse, 423 rows) is intentionally absent: it's
# the common ancestor of both East and West Norse and would
# nominally fold into the default 'native-old-norse'. A future
# Phase 4d hand-correction could split out a 'proto-norse' bucket
# if the register distinction becomes useful, but Phase 4c keeps
# the design lean.
_OLD_NORSE_SELF_LANGUAGE_TO_STRATUM: dict[str, str] = {
    "gmq-osw": "east-norse",
    "gmq-oda": "east-norse",
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


def classify_old_norse(db: LexiconDB) -> dict[int, str]:
    """Return ``{etymon_id: stratum}`` for the Old Norse family.

    Covers etymons whose ``language`` is one of:
    - ``old-norse`` — classified by ancestor language via ``etymon_descent``.
    - ``gmq-osw`` (Old Swedish) / ``gmq-oda`` (Old Danish) —
      classified as ``east-norse`` via self-language (see
      ``_OLD_NORSE_SELF_LANGUAGE_TO_STRATUM`` for the rationale).

    Pure read. Skips ``merged_into_id IS NOT NULL`` rows.

    Note on Germanic descent: proto-germanic / proto-indo-european /
    gmq-pro / gmw-pro are intentionally NOT in the ancestor map.
    Every Old Norse word descends from those via the standard
    Germanic path; treating that as a 'germanic-inheritance' loan
    would collapse the bundle into one bucket and erase the loan
    distinctions (Latin from Christianization, Low German from
    Hanseatic trade, English from Viking-Age contact, Gaelic from
    Irish-Sea settlement). Same descent-absence rationale as
    classify_french / classify_old_english — pinned by a
    regression test.
    """
    return _classify_family(
        db,
        modern_lang="old-norse",
        strata_order=OLD_NORSE_STRATA,
        ancestor_to_stratum=_OLD_NORSE_ANCESTOR_TO_STRATUM,
        self_lang_to_stratum=_OLD_NORSE_SELF_LANGUAGE_TO_STRATUM,
        default_stratum=_OLD_NORSE_DEFAULT_STRATUM,
    )


# --- Cross-family registry ------------------------------------------------
#
# Union of every stratum name across every classified language
# family. Used by ``lexicon set-stratum`` (wyrd-lr4 Phase 4d) for
# typo protection on hand-corrections, and by future consumers
# (e.g. wyrd-j3gy --stratum filter validation) that need to ask
# 'is this string a known stratum somewhere?'.
#
# Note that several names overlap across families (latin-loan
# appears in Welsh / OE / ON; english-loan in Welsh / ON). The
# value alone is ambiguous about which family it belongs to —
# callers that need family-aware validation must consult the per-
# family STRATA tuples directly.
ALL_STRATA: frozenset[str] = frozenset(
    WELSH_STRATA + FRENCH_STRATA + OLD_ENGLISH_STRATA + OLD_NORSE_STRATA
)


__all__ = [
    "ALL_STRATA",
    "FRENCH_STRATA",
    "OLD_ENGLISH_STRATA",
    "OLD_NORSE_STRATA",
    "WELSH_STRATA",
    "classify_french",
    "classify_old_english",
    "classify_old_norse",
    "classify_welsh",
]
