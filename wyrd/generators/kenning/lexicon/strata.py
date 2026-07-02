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

from typing import TYPE_CHECKING, Any

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
    # Latin loan-source codes. The wiktextract ingester normalizes ``la``→``latin``
    # and ``la-vul``→``vulgar-latin``, but leaves the period-specific variants
    # (``la-lat`` classical, ``la-med`` medieval, ``la-ecc`` ecclesiastical) as
    # stored ``etymon.language`` values — so all of them must be enumerated here to
    # catch a Welsh←Latin loan, exactly as the OE / ON ancestor maps do. (Greek is
    # not included: unlike OE's Christian-era Greek-via-Latin channel, the bundle
    # has no Welsh←ancient-greek descent edges.)
    "latin": "latin-loan",
    "la-lat": "latin-loan",
    "la-med": "latin-loan",
    "la-ecc": "latin-loan",
    "vulgar-latin": "latin-loan",
    # English loan-source codes. Same rule as Latin: the ingester normalizes
    # ``en``→``modern-english`` / ``ang``→``old-english`` / ``enm``→``middle-english``
    # (wiktextract_ingester.py) but leaves the period/dialect/regional variants
    # (``en-ear`` Early Modern English, ``enm-wmi`` a Middle English dialect, ``en-GB``
    # British English) as raw ``etymon.language`` values — so each must be enumerated
    # too, else a Welsh←English loan whose only English-bearing parent uses one of
    # these falls through to native-welsh. (``sco`` Scots is deliberately NOT mapped:
    # it is a distinct language sister to English, not an English variety, so whether
    # a Welsh←Scots borrowing is an "english-loan" is a curation call, not a gap.)
    "modern-english": "english-loan",
    "middle-english": "english-loan",
    "old-english": "english-loan",
    "english": "english-loan",
    "en-ear": "english-loan",
    "enm-wmi": "english-loan",
    "en-GB": "english-loan",
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


# --- Celtic-family strata (wyrd-de77) -------------------------------------
#
# Three NEW families covering the coarse Celtic-family language tags
# the Welsh / French / OE / ON classifiers never touched:
# 'brittonic', 'goidelic', and the generic 'celtic'. 476 of the
# >=2-toponym admit cohort carry one of these tags and were
# unclassified before this phase.
#
# The data drives the design. Of the 476 Celtic admits, only ~60
# (13%) have ANY etymon_descent parent edge; 87% are orphans and
# land in the default native-* bucket. The real parent ancestries
# observed in the live DB are proto-celtic (dominant), old-irish
# (8), middle-irish (2), and cel-bry-pro. The admit cohort carries
# NO Norse / Anglo-Norman / French / English loan ancestries (a
# handful of loan-parented rows exist across the FULL classified
# set — they fall to native-* rather than justifying a bucket), so —
# unlike Welsh / OE / ON — these families ship NO loan buckets.
# proto-germanic parents that show up are mining NOISE (mis-traced
# edges), so they're deliberately NOT mapped, mirroring every
# existing classifier (which never maps proto-germanic).
#
# All three are LEAF language tags: 'brittonic' / 'goidelic' /
# 'celtic' are never themselves the PARENT of another etymon in the
# data, so each has an EMPTY self-language map. The ancestor-walk
# pass over modern_lang rows does all the work; there are no
# ancestor / period varieties to fold via self-language (contrast
# Welsh's middle-welsh / old-welsh, French's old-french, ON's
# gmq-osw). If wave-2 mining ever surfaces a sub-period Celtic
# variety as a parent tag, the two-pass shape is ready to extend.

# Brittonic (P-Celtic branch: Welsh / Cornish / Breton ancestors).
# Two buckets: a proto-celtic substrate layer + the native default.
# cel-bry-pro (Proto-Brythonic) and proto-celtic both route to the
# substrate bucket — they're the deep-time ancestor stages.
_BRITTONIC_DEFAULT_STRATUM: str = "native-brittonic"

BRITTONIC_STRATA: tuple[str, ...] = (
    "proto-celtic-substrate",
    _BRITTONIC_DEFAULT_STRATUM,
)

_BRITTONIC_ANCESTOR_TO_STRATUM: dict[str, str] = {
    "proto-celtic": "proto-celtic-substrate",
    "cel-bry-pro": "proto-celtic-substrate",
}

# Empty: brittonic is a leaf tag, never a parent in the data.
_BRITTONIC_SELF_LANGUAGE_TO_STRATUM: dict[str, str] = {}


# Goidelic (Q-Celtic branch: Irish / Scottish Gaelic / Manx).
# Four buckets in priority order: the proto-celtic substrate first,
# then the two attested Gaelic stages (old-irish, middle-irish),
# then the native default last — matching the Welsh convention of
# substrate-first, period-stages, default-last. scottish-gaelic /
# irish / latin parents are NOT mapped (too few to bucket; they fold
# to the default).
_GOIDELIC_DEFAULT_STRATUM: str = "native-goidelic"

GOIDELIC_STRATA: tuple[str, ...] = (
    "proto-celtic-substrate",
    "old-irish",
    "middle-irish",
    _GOIDELIC_DEFAULT_STRATUM,
)

_GOIDELIC_ANCESTOR_TO_STRATUM: dict[str, str] = {
    "proto-celtic": "proto-celtic-substrate",
    "old-irish": "old-irish",
    "middle-irish": "middle-irish",
}

# Empty: goidelic is a leaf tag, never a parent in the data.
_GOIDELIC_SELF_LANGUAGE_TO_STRATUM: dict[str, str] = {}


# Celtic (the generic / unbranched tag — used when the source didn't
# distinguish P- vs Q-Celtic). Two buckets: proto-celtic substrate +
# native default. Deliberately does NOT map the mixed
# old-irish / old-welsh ancestries some generic-celtic rows carry —
# the generic tag can't honestly claim a branch-specific stage, so
# those fold to the default.
_CELTIC_DEFAULT_STRATUM: str = "native-celtic"

CELTIC_STRATA: tuple[str, ...] = (
    "proto-celtic-substrate",
    _CELTIC_DEFAULT_STRATUM,
)

_CELTIC_ANCESTOR_TO_STRATUM: dict[str, str] = {
    "proto-celtic": "proto-celtic-substrate",
    "cel-bry-pro": "proto-celtic-substrate",
}

# Empty: celtic is a leaf tag, never a parent in the data.
_CELTIC_SELF_LANGUAGE_TO_STRATUM: dict[str, str] = {}


def _stratum_to_ancestors(
    ancestor_to_stratum: dict[str, str],
) -> dict[str, set[str]]:
    """Invert the ancestor→stratum map so we can ask "which ancestor
    languages would put an etymon in stratum S?"."""
    inverted: dict[str, set[str]] = {}
    for lang, stratum in ancestor_to_stratum.items():
        inverted.setdefault(stratum, set()).add(lang)
    return inverted


def _self_language_proposals(db: LexiconDB, self_lang_to_stratum: dict[str, str]) -> dict[int, str]:
    """Pass 1: assign every etymon whose ``language`` is a key in
    ``self_lang_to_stratum`` directly to that stratum.

    These ARE the ancestor/period varieties (proto-Brittonic, Frankish,
    vulgar-Latin, middle-welsh, old-french, ...) — their own descent walk would
    describe what fed INTO them, not their own register, so they're assigned by
    self-language rather than by ancestor walk. Skips ``merged_into_id IS NOT
    NULL`` rows (OCR-clustering losers). ``ORDER BY id`` pins a stable,
    rebuild-independent insertion order: the by_stratum counters in
    ``classify_stratum_all`` and the ``format_enrichment_run`` sections iterate
    this dict, so a physical-row-order drift would be a Phase 3 byte-diff risk.
    """
    proposals: dict[int, str] = {}
    for lang, stratum in self_lang_to_stratum.items():
        rows = db.conn.execute(
            "SELECT id FROM etymon WHERE language = ? AND merged_into_id IS NULL ORDER BY id",
            (lang,),
        )
        for r in rows:
            proposals[r["id"]] = stratum
    return proposals


# wyrd-fljy: the LOAN-class strata across all families (enumerated explicitly,
# NOT by a "-loan" name pattern). A loan stratum means "borrowed a form from the
# mapped ancestor language", so the ancestor walk counts only LOAN edge_types for
# these — inheritance/compound across families is mis-traced mining noise (e.g.
# 501 modern-english→welsh ``inheritance`` edges in the live DB; no genuine
# cross-family inheritance exists), not a real loan. Substrate / native / dialect
# strata are NOT listed: a substrate legitimately rides inheritance, so they keep
# ALL edge types.
LOAN_STRATA: frozenset[str] = frozenset(
    {"latin-loan", "english-loan", "norse-loan", "low-german-loan"}
)

# The ``etymon_descent.edge_type`` values that count as a loan for a LOAN_STRATA
# stratum. Excludes ``inheritance`` and ``compound``.
_LOAN_EDGE_TYPES: frozenset[str] = frozenset({"borrowing", "derivation", "calque"})


def _ancestor_walk_proposals(
    db: LexiconDB,
    *,
    modern_lang: str,
    strata_order: tuple[str, ...],
    ancestor_to_stratum: dict[str, str],
    default_stratum: str,
) -> dict[int, str]:
    """Pass 2: classify every etymon whose ``language == modern_lang`` by its
    ancestor languages.

    Builds each etymon's parent-language set via a single chunked join over
    ``etymon_descent``, then iterates ``strata_order`` and picks the first
    stratum whose ancestor-language set intersects that parent set (the default
    stratum has no ancestor mapping, so it falls through). Skips
    ``merged_into_id IS NOT NULL`` rows; ``ORDER BY id`` pins insertion order
    for the same byte-diff reason as :func:`_self_language_proposals`.
    """
    modern_rows = db.conn.execute(
        "SELECT id FROM etymon WHERE language = ? AND merged_into_id IS NULL ORDER BY id",
        (modern_lang,),
    ).fetchall()
    if not modern_rows:
        return {}

    # Pull every parent-language for the modern-variety etymon set in one pass.
    # Chunked at SQLite's 999-variable limit. The inner SELECT keys on
    # ``child_id`` so ``idx_etymon_descent_child`` carries the lookup.
    modern_ids = [r["id"] for r in modern_rows]
    # Two parent-language sets per child: ALL edges (for substrate/native strata)
    # and LOAN edges only (for LOAN_STRATA — wyrd-fljy). Pull ``edge_type`` in the
    # same pass so the loan set is free.
    parents_by_child: dict[int, set[str]] = {}
    loan_parents_by_child: dict[int, set[str]] = {}
    for i in range(0, len(modern_ids), 999):
        chunk = modern_ids[i : i + 999]
        placeholders = ",".join("?" * len(chunk))
        cur = db.conn.execute(
            f"SELECT d.child_id, e.language, d.edge_type "  # noqa: S608 — placeholders are ints
            f"FROM etymon_descent d JOIN etymon e ON d.parent_id = e.id "
            f"WHERE d.child_id IN ({placeholders})",
            chunk,
        )
        for row in cur:
            parents_by_child.setdefault(row["child_id"], set()).add(row["language"])
            if row["edge_type"] in _LOAN_EDGE_TYPES:
                loan_parents_by_child.setdefault(row["child_id"], set()).add(row["language"])

    stratum_ancestors = _stratum_to_ancestors(ancestor_to_stratum)

    proposals: dict[int, str] = {}
    for eid in modern_ids:
        parent_langs = parents_by_child.get(eid, set())
        loan_parent_langs = loan_parents_by_child.get(eid, set())
        assigned = default_stratum
        # Iterate strata in priority order; first stratum whose ancestor-language
        # set intersects this etymon's parent set wins. The default stratum has
        # no ancestor-language mapping so it naturally falls through.
        for stratum_target in strata_order:
            if stratum_target == default_stratum:
                continue
            # wyrd-fljy: a LOAN-class stratum only counts loan-type parent edges
            # (borrowing/derivation/calque); a substrate/native stratum counts all.
            candidate_parents = loan_parent_langs if stratum_target in LOAN_STRATA else parent_langs
            if candidate_parents & stratum_ancestors.get(stratum_target, set()):
                assigned = stratum_target
                break
        proposals[eid] = assigned
    return proposals


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

    Two passes, merged into one ``{etymon_id: stratum}`` proposal map:

    1. :func:`_self_language_proposals` — the ancestor/period varieties that
       ARE the ancestor, assigned by their own ``language``.
    2. :func:`_ancestor_walk_proposals` — the modern variety, assigned by the
       ancestor languages reached through ``etymon_descent``.

    The self-language pass is applied first, then merged with the ancestor-walk
    pass. This relies on a precondition — ``modern_lang`` must NOT be a key in
    ``self_lang_to_stratum`` — so the two passes select disjoint etymon sets
    (one selects ``language == modern_lang``, the other ``language IN
    self_lang_to_stratum``) and the merge never overwrites, preserving the
    documented self-language-then-modern insertion order. Every current
    ``classify_<lang>`` wrapper satisfies it; it's enforced below because the
    wrappers are an explicit extension point (see the Celtic section). Both
    passes skip ``merged_into_id IS NOT NULL`` rows and emit ``ORDER BY id``
    order (see the helpers).

    Pure read; doesn't mutate the DB. Caller decides whether to write the
    proposed assignments back to ``etymon.stratum``.
    """
    if modern_lang in self_lang_to_stratum:
        raise ValueError(
            f"modern_lang {modern_lang!r} must not also be a self_lang_to_stratum key: "
            "the self-language and ancestor-walk passes would select overlapping etymons, "
            "so pass 2 would silently overwrite pass 1 (and break the proposal order)."
        )
    proposals = _self_language_proposals(db, self_lang_to_stratum)
    proposals.update(
        _ancestor_walk_proposals(
            db,
            modern_lang=modern_lang,
            strata_order=strata_order,
            ancestor_to_stratum=ancestor_to_stratum,
            default_stratum=default_stratum,
        )
    )
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


def classify_brittonic(db: LexiconDB) -> dict[int, str]:
    """Return ``{etymon_id: stratum}`` for the Brittonic family (wyrd-de77).

    Covers etymons whose ``language`` is ``brittonic``, classified by
    ancestor language via ``etymon_descent``. 'brittonic' is a leaf
    tag (never a parent in the data), so the self-language map is
    empty and the ancestor-walk pass does all the work. 87% of
    Brittonic admits are orphans → ``native-brittonic`` default.

    Pure read. Skips ``merged_into_id IS NOT NULL`` rows.

    Note: proto-germanic parents (mining noise) are intentionally NOT
    in the ancestor map — mirrors every other classifier, which never
    maps proto-germanic. No loan buckets: the admit cohort carries no
    Norse / French / English loan ancestries (the few across the full
    classified set fall to native-*).
    """
    return _classify_family(
        db,
        modern_lang="brittonic",
        strata_order=BRITTONIC_STRATA,
        ancestor_to_stratum=_BRITTONIC_ANCESTOR_TO_STRATUM,
        self_lang_to_stratum=_BRITTONIC_SELF_LANGUAGE_TO_STRATUM,
        default_stratum=_BRITTONIC_DEFAULT_STRATUM,
    )


def classify_goidelic(db: LexiconDB) -> dict[int, str]:
    """Return ``{etymon_id: stratum}`` for the Goidelic family (wyrd-de77).

    Covers etymons whose ``language`` is ``goidelic``, classified by
    ancestor language via ``etymon_descent``. 'goidelic' is a leaf
    tag (never a parent), so the self-language map is empty. The
    attested Gaelic stages (old-irish, middle-irish) get their own
    buckets; proto-celtic routes to the substrate bucket; everything
    else (including the rare scottish-gaelic / irish / latin parents,
    too few to bucket) folds to ``native-goidelic``.

    Pure read. Skips ``merged_into_id IS NOT NULL`` rows.

    Note: proto-germanic parents (mining noise) are intentionally NOT
    mapped — same convention as every other classifier.
    """
    return _classify_family(
        db,
        modern_lang="goidelic",
        strata_order=GOIDELIC_STRATA,
        ancestor_to_stratum=_GOIDELIC_ANCESTOR_TO_STRATUM,
        self_lang_to_stratum=_GOIDELIC_SELF_LANGUAGE_TO_STRATUM,
        default_stratum=_GOIDELIC_DEFAULT_STRATUM,
    )


def classify_celtic(db: LexiconDB) -> dict[int, str]:
    """Return ``{etymon_id: stratum}`` for the generic Celtic family (wyrd-de77).

    Covers etymons whose ``language`` is the coarse ``celtic`` tag
    (used when the source didn't distinguish P- vs Q-Celtic),
    classified by ancestor language via ``etymon_descent``. 'celtic'
    is a leaf tag (never a parent), so the self-language map is empty.
    Only the proto-celtic substrate is bucketed; the mixed
    old-irish / old-welsh ancestries some generic-celtic rows carry
    are deliberately NOT mapped (the generic tag can't honestly claim
    a branch-specific stage) and fold to ``native-celtic``.

    Pure read. Skips ``merged_into_id IS NOT NULL`` rows.

    Note: proto-germanic parents (mining noise) are intentionally NOT
    mapped — same convention as every other classifier.
    """
    return _classify_family(
        db,
        modern_lang="celtic",
        strata_order=CELTIC_STRATA,
        ancestor_to_stratum=_CELTIC_ANCESTOR_TO_STRATUM,
        self_lang_to_stratum=_CELTIC_SELF_LANGUAGE_TO_STRATUM,
        default_stratum=_CELTIC_DEFAULT_STRATUM,
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
#
# Typed as ``frozenset`` (not ``set``) so the shared module-level
# registry can't be mutated by a caller, and so it's hashable for
# use as a dict key / Click choice / cache key. Same immutability
# rationale the per-family ``WELSH_STRATA`` / ``FRENCH_STRATA`` /
# etc. tuples carry.
ALL_STRATA: frozenset[str] = frozenset(
    WELSH_STRATA
    + FRENCH_STRATA
    + OLD_ENGLISH_STRATA
    + OLD_NORSE_STRATA
    + BRITTONIC_STRATA
    + GOIDELIC_STRATA
    + CELTIC_STRATA
)


# --- Per-family registry (wyrd-j3gy) --------------------------------------
#
# Maps the per-family STRATA tuples by family name. Keyed by the
# language-family identifier the classifiers use ('welsh' / 'french'
# / 'old-english' / 'old-norse') — same identifiers that appear in
# the ``classify-stratum --language`` Click choices.
STRATA_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "welsh": WELSH_STRATA,
    "french": FRENCH_STRATA,
    "old-english": OLD_ENGLISH_STRATA,
    "old-norse": OLD_NORSE_STRATA,
    "brittonic": BRITTONIC_STRATA,
    "goidelic": GOIDELIC_STRATA,
    "celtic": CELTIC_STRATA,
}


# Maps each lexicon ``etymon.language`` value to its language family
# (the key used in ``STRATA_BY_FAMILY``). Lets ``set-stratum`` pick
# the right per-family STRATA tuple for family-aware validation.
#
# Built programmatically from each classifier's modern_lang + self-
# language map, so adding a new self-language entry to e.g.
# ``_FRENCH_SELF_LANGUAGE_TO_STRATUM`` automatically extends
# LANGUAGE_TO_FAMILY without a separate manual update. The classifier
# constants are the single source of truth.
#
# Languages absent from this map don't yet have a classifier; their
# rows fall back to the broader ``ALL_STRATA`` typo-check only.
def _build_language_to_family() -> dict[str, str]:
    """Source LANGUAGE_TO_FAMILY from the per-classifier maps.

    Each family contributes its modern_lang (the language the
    ancestor walk targets) plus every key in its self-language map.
    Drift-proof: if a classifier adds a new ancestor variety, this
    map picks it up automatically.
    """
    return {
        # Welsh family.
        "welsh": "welsh",
        **dict.fromkeys(_WELSH_SELF_LANGUAGE_TO_STRATUM, "welsh"),
        # French family.
        "french": "french",
        **dict.fromkeys(_FRENCH_SELF_LANGUAGE_TO_STRATUM, "french"),
        # Old English family.
        "old-english": "old-english",
        **dict.fromkeys(_OLD_ENGLISH_SELF_LANGUAGE_TO_STRATUM, "old-english"),
        # Old Norse family.
        "old-norse": "old-norse",
        **dict.fromkeys(_OLD_NORSE_SELF_LANGUAGE_TO_STRATUM, "old-norse"),
        # Celtic-family leaf tags (wyrd-de77). Each has an empty
        # self-language map, so they contribute only their modern_lang.
        "brittonic": "brittonic",
        **dict.fromkeys(_BRITTONIC_SELF_LANGUAGE_TO_STRATUM, "brittonic"),
        "goidelic": "goidelic",
        **dict.fromkeys(_GOIDELIC_SELF_LANGUAGE_TO_STRATUM, "goidelic"),
        "celtic": "celtic",
        **dict.fromkeys(_CELTIC_SELF_LANGUAGE_TO_STRATUM, "celtic"),
    }


LANGUAGE_TO_FAMILY: dict[str, str] = _build_language_to_family()


# Per-culture allowed strata for the runtime --stratum filter
# (wyrd-j3gy). Captures which language families' strata are valid
# choices when generating names for that culture-bundle.
#
# INVARIANT: each culture's set MUST equal the union of STRATA
# tuples for the language families that culture's place-name
# corpus draws from. Adding a family to a culture-bundle (e.g.
# wave-2 mining surfaces a Goidelic stratum classifier and the
# irish bundle starts pulling from it) requires extending the
# culture's set here.
#
# Empty set = 'no per-culture restriction'; falls back to the
# broader ALL_STRATA typo-check in ``_resolve_stratum_param``. Used
# for any culture not enumerated below (none currently — every
# culture with a classified family has an explicit set; irish /
# breton gained theirs in wyrd-de77).
#
# english + scottish span the four core British source families
# because British place-names sample from each (OE base + ON Danelaw
# layer + Norman French superstrate + Brythonic substrate). Welsh
# culture is narrower — primarily Welsh-family forms plus the
# Norman French overlay from post-1283 Welsh place naming.
# wyrd-de77: the Celtic-family classifiers (brittonic / goidelic /
# celtic) now have output, so the irish / breton cultures gain
# non-empty allowed-sets and the welsh / scottish sets widen to
# include their Celtic-family substrate. These culture↔family
# mappings are a CURATION CALL flagged for Devon's review — the
# conservative, data-grounded choices are:
#   * irish    = Goidelic + the generic Celtic tag.
#   * breton   = Brittonic + the generic Celtic tag + French (Breton
#                place-naming carries a heavy French overlay).
#   * welsh    = the existing Welsh + French sets PLUS Brittonic +
#                Celtic (Welsh is the P-Celtic Brittonic branch).
#   * scottish = Goidelic + Brittonic + Celtic (Scotland straddles
#                the Gaelic / Brythonic line), on TOP of the existing
#                OE + ON + French + Welsh British-place-name spread.
_CULTURE_TO_VALID_STRATA: dict[str, frozenset[str]] = {
    "english": frozenset(WELSH_STRATA + FRENCH_STRATA + OLD_ENGLISH_STRATA + OLD_NORSE_STRATA),
    "scottish": frozenset(
        WELSH_STRATA
        + FRENCH_STRATA
        + OLD_ENGLISH_STRATA
        + OLD_NORSE_STRATA
        + GOIDELIC_STRATA
        + BRITTONIC_STRATA
        + CELTIC_STRATA
    ),
    "welsh": frozenset(WELSH_STRATA + FRENCH_STRATA + BRITTONIC_STRATA + CELTIC_STRATA),
    "irish": frozenset(GOIDELIC_STRATA + CELTIC_STRATA),
    "breton": frozenset(BRITTONIC_STRATA + CELTIC_STRATA + FRENCH_STRATA),
}


def valid_strata_for_culture(culture: str) -> frozenset[str]:
    """Return the set of strata valid for a culture-bundle, or an
    empty set if the culture has no per-culture restriction
    configured (caller should fall back to ALL_STRATA typo-check)."""
    return _CULTURE_TO_VALID_STRATA.get(culture, frozenset())


def family_for_language(language: str) -> str | None:
    """Return the language-family identifier for a lexicon language
    code, or None if the language has no classifier yet (caller
    should fall back to ALL_STRATA typo-check). Used by
    ``set-stratum`` for family-aware validation."""
    return LANGUAGE_TO_FAMILY.get(language)


# SQLite's default `SQLITE_MAX_VARIABLE_NUMBER` is 999; chunk at 500 to leave
# headroom for any additional placeholders a caller might splice in.
_NULL_STRATUM_LOOKUP_CHUNK = 500


def _fetch_null_stratum_ids(db, etymon_ids) -> set[int]:
    """Return the subset of ``etymon_ids`` whose ``stratum`` column is NULL.

    Used by the dry-run path of :func:`classify_stratum_all` to model the
    apply-true gate without writing. Chunked to stay under SQLite's
    parameterized-IN limit.
    """
    ids = list(etymon_ids)
    found: set[int] = set()
    for start in range(0, len(ids), _NULL_STRATUM_LOOKUP_CHUNK):
        chunk = ids[start : start + _NULL_STRATUM_LOOKUP_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        found.update(
            row["id"]
            for row in db.conn.execute(
                f"SELECT id FROM etymon WHERE stratum IS NULL AND id IN ({placeholders})",
                chunk,
            )
        )
    return found


def classify_stratum_all(db: LexiconDB, *, apply: bool = True) -> dict[str, Any]:
    """Uniform L3 wrapper for the ``classify-stratum`` pass — runs
    the per-language classifiers (welsh, french, old-english,
    old-norse, brittonic, goidelic, celtic) and persists the stratum
    assignments.

    Wyrd-hidb Phase 2 plumbs this into ``run_full_enrichment``.

    Returns per-language counts: how many etymons gained a stratum,
    how many were skipped (already had one), and which strata fired
    (sample counts per stratum label). Stratum overwrite is gated by
    ``stratum IS NULL`` — re-running is idempotent. Operators who
    need to reclassify a populated row use the standalone
    ``lexicon classify-stratum --force`` path.

    Dry-run (``apply=False``) runs the classifiers but skips the
    UPDATE — useful for previewing how many rows would be tagged
    before committing.
    """
    counts_by_language: dict[str, dict[str, int]] = {}
    for language, classifier in (
        ("welsh", classify_welsh),
        ("french", classify_french),
        ("old-english", classify_old_english),
        ("old-norse", classify_old_norse),
        ("brittonic", classify_brittonic),
        ("goidelic", classify_goidelic),
        ("celtic", classify_celtic),
    ):
        proposals = classifier(db)
        if not proposals:
            counts_by_language[language] = {"proposed": 0, "written": 0, "skipped": 0}
            continue
        written = 0
        skipped = 0
        # Stratum distribution counter for the per-language report
        by_stratum: dict[str, int] = {}
        if apply:
            # Per-row UPDATE — simpler than the CLI's CASE-batched path
            # because the enrichment chain commits at the end of the
            # whole run rather than per-stratum-pass. The CASE-batched
            # optimization matters at 100k+ rows / standalone CLI runs;
            # within run_full_enrichment we're already on one
            # transaction and only paying the syscall cost.
            for etymon_id, stratum in proposals.items():
                cur = db.conn.execute(
                    "UPDATE etymon SET stratum = ? WHERE id = ? AND stratum IS NULL",
                    (stratum, etymon_id),
                )
                if cur.rowcount:
                    written += 1
                    by_stratum[stratum] = by_stratum.get(stratum, 0) + 1
                else:
                    skipped += 1
        else:
            # Dry-run: apply the same stratum-IS-NULL gate as the
            # apply-true branch so written/skipped/by_stratum predict the
            # real write counts. Without this, the dry-run over-reports
            # by the count of etymons already classified.
            null_stratum_ids = _fetch_null_stratum_ids(db, proposals.keys())
            for etymon_id, stratum in proposals.items():
                if etymon_id in null_stratum_ids:
                    written += 1
                    by_stratum[stratum] = by_stratum.get(stratum, 0) + 1
                else:
                    skipped += 1
        counts_by_language[language] = {
            "proposed": len(proposals),
            "written": written,
            "skipped": skipped,
            "by_stratum": by_stratum,
        }
    if apply:
        db.commit()
    return {"applied": int(apply), "languages": counts_by_language}


__all__ = [
    "ALL_STRATA",
    "BRITTONIC_STRATA",
    "CELTIC_STRATA",
    "FRENCH_STRATA",
    "GOIDELIC_STRATA",
    "LANGUAGE_TO_FAMILY",
    "OLD_ENGLISH_STRATA",
    "OLD_NORSE_STRATA",
    "STRATA_BY_FAMILY",
    "WELSH_STRATA",
    "classify_brittonic",
    "classify_celtic",
    "classify_french",
    "classify_goidelic",
    "classify_old_english",
    "classify_old_norse",
    "classify_stratum_all",
    "classify_welsh",
    "family_for_language",
    "valid_strata_for_culture",
]
