"""Wiktextract → etymon_descent ingester (wyrd-4rt / wyrd-hun).

Operator-facing documentation lives in `INGESTION.md` under "Wiktionary
ingestion (wyrd-4rt)" — kaikki.org dump URLs, recommended slices, the
template→edge_type mapping, anti-patterns, and the cluster-cognates
follow-up. This module docstring captures only the load-bearing
implementation invariants:

Etymology section: walked via `etymology_templates`. Template kind
mapping defined by `_UPWARD_TEMPLATE_TO_EDGE`; `_SKIPPED_TEMPLATE_NAMES`
lists explicitly non-edge-producing templates (cognate, mention, link,
qualifier); anything else increments `unsupported_templates` so
operators can extend the maps.

Descendants section: a NESTED TREE of dicts with `lang_code` + `word`
fields directly (NOT flat-with-depth + templates as v1 misread).
`_walk_descendants` recurses into each node's `descendants` array.
Edge_type defaults to 'inheritance' and is overridden via the `tags`
array per `_DESCENDANT_TAG_TO_EDGE` (e.g. `tags=['calque']`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any

from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.morpheme_surface import normalize_morpheme_surface

# wyrd-hun: kaikki.org's wiktextract dump uses Wiktionary lang codes
# (ISO 639 or Wiktionary-private extensions). Wyrd's etymon.language
# column uses dash-prefixed strings ('old-english', 'proto-germanic').
# Map the codes most likely to surface in place-name etymology chains.
# Unmapped codes pass through as-is (with a counter), so the data is
# still loadable and can be canonicalized later — but the canonical
# wyrd languages get the right name immediately.
_WIKTIONARY_LANG_CODE_MAP: dict[str, str] = {
    # Germanic family — the bulk of British Isles place-name ancestry
    "en": "modern-english",
    "ang": "old-english",
    "enm": "middle-english",
    "non": "old-norse",
    "is": "icelandic",
    "fo": "faroese",
    "no": "norwegian",
    "nb": "norwegian-bokmal",
    "nn": "norwegian-nynorsk",
    "sv": "swedish",
    "da": "danish",
    "de": "german",
    "goh": "old-high-german",
    "gmh": "middle-high-german",
    "nl": "dutch",
    "gem-pro": "proto-germanic",
    "gem-x-pro": "proto-germanic",
    # Celtic family — Welsh/Irish/Gaelic/Breton place-names
    "cy": "welsh",
    "owl": "old-welsh",
    "wlm": "middle-welsh",
    "ga": "irish",
    "sga": "old-irish",
    "mga": "middle-irish",
    "gd": "scottish-gaelic",
    "gv": "manx",
    "br": "breton",
    "obt": "old-breton",
    "xbm": "middle-breton",
    "kw": "cornish",
    "cel-pro": "proto-celtic",
    "cel-x-pro": "proto-celtic",
    # Romance — French/Norman-French/Latin substrate
    "la": "latin",
    "fr": "french",
    "fro": "old-french",
    "frm": "middle-french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ro": "romanian",
    "VL.": "vulgar-latin",
    "la-vul": "vulgar-latin",
    # Greek
    "el": "modern-greek",
    "grc": "ancient-greek",
    "grk-pro": "proto-greek",
    # Proto-Indo-European
    "ine-pro": "proto-indo-european",
}

# Etymology-section templates that produce a SINGLE UPWARD edge (this
# entry descends from the parent the template names). Args follow:
#   args[1] = this entry's lang_code (redundant with entry.lang_code)
#   args[2] = parent's lang_code
#   args[3] = parent's word
_UPWARD_TEMPLATE_TO_EDGE: dict[str, str] = {
    "inh": "inheritance",
    "inherited": "inheritance",
    "inh+": "inheritance",
    # wyrd-wse: lite version surfaced in OE slice — same arg shape as inh.
    "inh-lite": "inheritance",
    "bor": "borrowing",
    "borrowed": "borrowing",
    "bor+": "borrowing",
    # wyrd-wse: learned + unattested borrowing variants. Same shape as bor.
    "lbor": "borrowing",
    "ubor": "borrowing",
    "der": "derivation",
    "derived": "derivation",
    "der+": "derivation",
    # wyrd-wse: lite version surfaced in OE slice. Same arg shape as der.
    "der-lite/lang": "derivation",
    # wyrd-wse: 'unknown derivation' / 'user-friendly derived' — verified
    # cross-language 3-arg shape against real OE entries:
    #   {{uder|ang|la|Tigris}} → ang Tigris ← la Tigris
    "uder": "derivation",
    "cal": "calque",
    "calque": "calque",
    "clq": "calque",
    # wyrd-wse: semantic loan is structurally a calque (form-meaning shape
    # carried across language boundary). Same args[2]/args[3] shape.
    "semantic loan": "calque",
    "sl": "calque",
}

# wyrd-prv: PIE-root templates have args
#   args[1] = this entry's lang_code
#   args[2] = ancestor lang_code (typically 'ine-pro')
#   args[3..N] = root word(s) — sometimes multiple parallel roots are cited
# Each root word becomes its own inheritance edge to this entry.
_ROOT_TEMPLATE_NAMES: frozenset[str] = frozenset({"root"})

# wyrd-wse: same-language single-parent derivation templates. These have
# a different arg shape from inh/bor/der/cal — the parent word lives at
# args[2] and the parent lang IS this_lang (args[1]). The derivational
# relationship stays within one language:
#
#   {{clipping|ang|elpend}}      → elpend → elp (OE clipping)
#   {{back-formation|ang|sċeadu}} → sċeadu → scead (OE back-formation)
#   {{bf|non|trǫf}}              → trǫf → traf (ON back-formation)
#
# All map to 'derivation' edges (D27 doesn't have separate kinds for
# these morphological relationships, and they all express
# derivational lineage within a single language).
_SAME_LANG_DERIVATION_TEMPLATE_NAMES: frozenset[str] = frozenset(
    {
        "clipping",
        "bf",
        "back-formation",
        "back-form",
        "contraction",
        "deverbal",
        "nom",
        # Defensive-add: full-name variants of already-verified abbreviations.
        # They don't appear in the four ingested slices (OE/ON/PG/PC) but
        # almost certainly share their abbreviation's arg shape, and the
        # dispatcher returns [] when args are insufficient.
        "nominalization",
        "syncope",
        "alternative form of",
        "abbreviation",
        "reduplication",
        # Verified from real wiktextract entries 2026-05-03:
        #   {{contr|ang|æghwæþer}} → ang ægþer (contraction abbreviation)
        #   {{sync|ang|isern}}     → ang iren  (syncope)
        #   {{apocopic form|ang|æþele}} → ang æþel
        # metathesis sometimes lacks args[2] (just a category marker),
        # but the dispatcher returns [] defensively when args[2] is
        # missing — including it costs nothing and admits the cases
        # where args[2] IS present.
        "contr",
        "sync",
        "apocopic form",
        "metathesis",
        # Verified from real OE entries 2026-05-03:
        #   {{past participle of|ang|sāmwyrċan}} → ang samworht ← sāmwyrċan
        #   {{alt form|ang|ġefulwian}}           → ang gefullian ← ġefulwian
        #   {{abbrev|ang|onġemang}}              → ang amang ← onġemang
        # Same single-parent same-language shape as clipping/contr/etc.
        "past participle of",
        "alt form",
        "abbrev",
        # vrd = vṛddhi gerundive abbreviation. 62 occurrences in PG/ON
        # slices; verified args[2] holds the parent word
        # ({{vrd|gem-pro|*drepaną}} → gem-pro hældræpr ← *drepaną).
        # NOTE: the long-form 'vrddhi' template stores the parent in
        # the 'alt' keyword arg instead of args[2], so it stays in the
        # skip list — handling that variant would need the dispatcher
        # to inspect named kwargs.
        "vrd",
    }
)

# wyrd-prv: compound / affix templates have args
#   args[1] = this entry's lang_code (all parts are in THIS language)
#   args[2..N] = constituent parts
# Each constituent becomes its own compound edge to this entry. The
# 'compound' edge_type lives on D27's CHECK constraint but is NOT in
# _COGNATE_BRIDGING_EDGES — these don't bridge synsets (compositional
# derivations cross lexical-semantic boundaries that cognate clustering
# shouldn't unify).
_COMPOUND_TEMPLATE_NAMES: frozenset[str] = frozenset(
    {
        "compound",
        "compound+",
        "com",
        "com+",
        "prefix",
        "pre",
        "suffix",
        "suf",
        "af",
        "affix",
        # confix is a circumfix derivation (matched prefix+suffix wrapping
        # a stem). Same arg shape as compound; treating each constituent
        # as a 'compound' edge is the right call.
        "confix",
        # wyrd-wse: blend is a portmanteau of two source words; arg shape
        # matches compound (args[1]=this_lang, args[2..]=parts). Treating
        # each part as a compound edge captures the same shape.
        "blend",
        # Univerbation = compound formed from a free word sequence.
        # The OE-slice samples I checked omit args[2..] (use it as a
        # bare category marker), but documentation of {{univerbation}}
        # says args[2..] carry the constituent parts. Compound handler
        # safely returns [] when args[2..] are missing.
        "univerbation",
    }
)

# Templates we explicitly skip because they're either peer relations
# (cognate, mention) or non-etymological (formatting, qualifiers, or
# un-extractable structure like dercat / etymon's nested syntax).
_SKIPPED_TEMPLATE_NAMES: frozenset[str] = frozenset(
    {
        "cog",
        "cognate",
        "m",
        "mention",
        "m+",
        "m-g",
        "l",
        "link",
        "qualifier",
        "qual",
        "gloss",
        "glossary",
        "gl",
        "noncog",
        "ncog",
        "etyl",
        "lb",
        "unk",
        "unc",
        "unknown",
        # wyrd-prv: dercat names a chain of ancestor LANGUAGES without a
        # specific parent word — no edge can be extracted. The category
        # is useful elsewhere (e.g. for the etymology_text rendering)
        # but not for descent edges.
        "dercat",
        # wyrd-prv: etymon uses an inline nested syntax we don't parse
        # (e.g. "ar-<ety:inh<gmw-pro:*uʀ->>") — skip rather than try.
        "etymon",
        # wyrd-prv: surf is a "surface analysis" caveat — no edge.
        "surf",
        # Pure formatting / meta templates discovered during the live
        # OE + Proto-Celtic ingest. None carry an etymon link; they're
        # citation, layout, or category-tag wrappers.
        "yesno",
        "etymid",
        "nonlemma",
        "col-top",
        "lit",
        "langname",
        "word",
        "lg",
        "q",
        "nl",
        "ref",
        "sup",
        "smallsup",
        "ety",
        "PIE word",
        "cat",
        "or else",
        "onomatopoeic",
        "sound-symbolic",
        "desc",
        # str* templates are layout-only (column / index helpers used
        # in big descendants tables) and don't name a parent word.
        "str left",
        "str_index-lite",
        "str index-lite/logic",
        "str len-lite",
        "str len-lite/core",
        "str sub-lite",
        "str sub-lite/2",
        # doublet / dbt name a peer reflex from the same root via a
        # different path — like {{cog}}, peer not chain. No bridging.
        "doublet",
        "dbt",
        "piecewise doublet",
        # Other meta-only templates surfaced by full-slice ingest.
        "wp",
        "normalized",
        "surface analysis",
        ",",
        # wyrd-wse: more skip-only kinds discovered in PG/ON live ingest.
        # Long-name variants of already-skipped templates:
        "nonlemmas",  # plural of nonlemma
        "mention-gloss",  # long name of m+
        "noncognate",  # long name of noncog
        "onomatopoeia",  # long name of onomatopoeic
        "onom",  # abbrev of onomatopoeic
        "m-lite",  # lite of m
        # No-arg category-only markers (no parent etymon to extract):
        "pre-Germanic",
        "vrddhi",  # marker only — args[1] is this_lang; no parent_word
        # Small formatting / unknown-context templates:
        "g",
        "?",
        "s",
        "IPAfont",
        "uncertain",
        "anchor",
        "number box",
        "sno",
        "qinfl",
        "coin",
        "lang",
        "ISSN",
        "tea",
        # pw dbt = proto-word doublet; peer relation, like doublet.
        "pw dbt",
        # wyrd-wse: long-tail single-occurrence kinds across the OE/ON/PG/PC
        # slices. Mostly formatting (smallcaps, angbr, PIE root box,
        # etydate) plus 'alter' and 'displaced', which have arg shapes
        # that don't fit the same-lang or cross-lang single-parent
        # patterns (alter takes two parallel parents at args[2,3];
        # displaced's args[1] is the displacing language, not this_lang).
        "senseno",
        "C.E.",
        "smallcaps",
        "angbr",
        "PIE root box",
        "non-gloss",
        "displaced",  # args[1] is displacing-language, not this_lang — different shape
        "alter",  # 3-arg with two alt forms — different shape from same-lang derivation
        "etydate",
    }
)


def _canonical_language(lang_code: str) -> str:
    """Map a wiktextract lang_code to wyrd's dash-prefixed language
    string, falling back to the raw code if unmapped. Unmapped codes
    show up in counts['unmapped_lang_codes'] so the operator can extend
    the map."""
    return _WIKTIONARY_LANG_CODE_MAP.get(lang_code, lang_code)


def _extract_template_args(tmpl: dict[str, Any]) -> dict[str, str]:
    """Return the template's `args` dict if present and a dict, else
    an empty dict. Defensive against malformed wiktextract output."""
    args = tmpl.get("args")
    if isinstance(args, dict):
        return {str(k): str(v) for k, v in args.items() if v is not None}
    return {}


def _upward_edges_from_template(
    tmpl: dict[str, Any],
) -> list[tuple[str, str, str]]:
    """Dispatch a wiktextract etymology template to the right arg-shape
    handler and return the (parent_lang_code, parent_word, edge_type)
    tuples it produces. Empty list when the template isn't edge-
    producing or its args don't supply enough information.

    The template-set membership tells us the arg shape; each helper
    encapsulates one shape:

      * `_cross_lang_single_parent_edges` — inh/bor/der/cal/sl/lbor/...:
        args[2]=parent_lang, args[3]=parent_word. One edge.
      * `_root_template_edges` — {{root}}: args[2]=ancestor_lang,
        args[3..N]=root_word(s). One edge per root.
      * `_compound_template_edges` — compound/affix/blend/...:
        args[1]=this_lang, args[2..N]=constituent parts. One edge each.
      * `_same_lang_derivation_edges` — clipping/bf/back-formation/...:
        args[1]=this_lang (also parent_lang), args[2]=parent_word. One.
    """
    name = tmpl.get("name", "")
    args = _extract_template_args(tmpl)
    if name in _UPWARD_TEMPLATE_TO_EDGE:
        return _cross_lang_single_parent_edges(name, args)
    if name in _ROOT_TEMPLATE_NAMES:
        return _root_template_edges(args)
    if name in _COMPOUND_TEMPLATE_NAMES:
        return _compound_template_edges(args)
    if name in _SAME_LANG_DERIVATION_TEMPLATE_NAMES:
        return _same_lang_derivation_edges(args)
    return []


def _cross_lang_single_parent_edges(name: str, args: dict[str, str]) -> list[tuple[str, str, str]]:
    """{{inh|en|ang|tūn}} → [(ang, tūn, inheritance)]. Same shape covers
    bor/der/cal and their +/lite variants and semantic loan / sl."""
    parent_lang_code = args.get("2")
    parent_word = args.get("3")
    if parent_lang_code and parent_word:
        return [(parent_lang_code, parent_word, _UPWARD_TEMPLATE_TO_EDGE[name])]
    return []


def _root_template_edges(args: dict[str, str]) -> list[tuple[str, str, str]]:
    """{{root|en|ine-pro|*r1|*r2}} → one inheritance edge per root word.
    Wiktionary editors sometimes cite parallel PIE roots when the chain
    is contested. Walks args[3], args[4], ... while consecutive
    positional args are present, so unusually-long root lists don't
    silently truncate."""
    ancestor_lang = args.get("2")
    if not ancestor_lang:
        return []
    edges: list[tuple[str, str, str]] = []
    i = 3
    while (word := args.get(str(i))) is not None:
        if word:
            edges.append((ancestor_lang, word, "inheritance"))
        i += 1
    return edges


def _compound_template_edges(args: dict[str, str]) -> list[tuple[str, str, str]]:
    """{{compound|ang|Sċott|land}} → one 'compound' edge per constituent.
    Walks args[2], args[3], ... while consecutive positional args are
    present, so multi-part compounds don't silently truncate at a
    fixed bound."""
    this_lang = args.get("1")
    if not this_lang:
        return []
    edges: list[tuple[str, str, str]] = []
    i = 2
    while (part := args.get(str(i))) is not None:
        if part:
            edges.append((this_lang, part, "compound"))
        i += 1
    return edges


def _same_lang_derivation_edges(
    args: dict[str, str],
) -> list[tuple[str, str, str]]:
    """{{clipping|ang|elpend}} → [(ang, elpend, derivation)]. Parent lang
    is the SAME as this_lang (args[1]) — back-formation, contraction,
    syncope, etc. all stay within one language."""
    this_lang = args.get("1")
    parent_word = args.get("2")
    if this_lang and parent_word:
        return [(this_lang, parent_word, "derivation")]
    return []


_KNOWN_TEMPLATE_NAMES: frozenset[str] = frozenset(
    set(_UPWARD_TEMPLATE_TO_EDGE)
    | _ROOT_TEMPLATE_NAMES
    | _COMPOUND_TEMPLATE_NAMES
    | _SAME_LANG_DERIVATION_TEMPLATE_NAMES
)


# wyrd-c9t (2026-05-03): real wiktextract descendants are a NESTED TREE,
# not a flat list with depth markers. Each entry has lang_code + word
# directly (no `templates` field, no `depth` field), with an optional
# `descendants` array of children for recursion. Borrow/derivation/
# calque flagging happens via the `tags` array, not via {{desc|...|bor=1}}
# template flags (which don't exist in real data — that was a v1
# misreading).
#
# Tags that override the default 'inheritance' edge_type. Real OE slice
# uses 'calque' (99 occurrences). 'borrowed' / 'derived' aren't observed
# but mapped defensively in case other slices use them.
_DESCENDANT_TAG_TO_EDGE: dict[str, str] = {
    "calque": "calque",
    "borrowed": "borrowing",
    "borrowing": "borrowing",
    "derived": "derivation",
    "derivation": "derivation",
}


def _descendant_edge_type(node: dict[str, Any]) -> str:
    """Resolve the edge_type for a descendants-tree node from its tag arrays.
    Defaults to 'inheritance' (the implied relationship of any Descendants section
    entry). The first matching tag wins; multiple matches are theoretically
    possible but not observed in real data.

    Reads BOTH ``tags`` and ``raw_tags``: wiktextract puts normalized labels in
    ``tags`` but the ``borrowed`` / ``derived`` markers land in ``raw_tags`` (the
    un-normalized array). Looking only at ``tags`` silently defaulted every
    borrowed descendant to ``inheritance`` — etymologically impossible across
    language families (a Welsh←English descendant is a loan, never inherited) — and
    left the ``borrowed`` / ``borrowing`` map entries as dead code that never fired
    on real data (16k+ such nodes in the English slice alone)."""
    for tag in (node.get("tags") or []) + (node.get("raw_tags") or []):
        mapped = _DESCENDANT_TAG_TO_EDGE.get(tag)
        if mapped is not None:
            return mapped
    return "inheritance"


def ingest_wiktextract_stream(
    db: LexiconDB,
    stream: IO[str],
    *,
    apply: bool = False,
    limit: int | None = None,
    since_line: int = 0,
) -> dict[str, int]:
    """Read wiktextract JSONL from `stream`, line by line, and write
    etymons + etymon_descent edges to the lexicon (wyrd-hun).

    Args:
      stream: any line-iterable text source (an open file, sys.stdin,
        StringIO for tests).
      apply: when False (default) parse + count without writing; when
        True, upsert etymons and insert descent edges.
      limit: stop after processing N entries (for partial / smoke runs).
      since_line: skip the first N lines (for resumability across
        multi-hour ingest sessions).

    Returns counts:
      - lines_read: total lines pulled from the stream
      - entries_parsed: lines that were valid JSON dicts with a 'word'
      - entries_skipped_malformed: lines that failed JSON parse or
        lacked a 'word'
      - upward_edges: count of inh/bor/der/cal edges emitted
      - downward_edges: count of desc edges emitted
      - skipped_templates: count of templates we recognized as
        non-edge-producing (cognate, mention, link, qualifier, etc.)
      - unsupported_templates: count of unfamiliar template names
        (operator may want to extend the maps)
      - applied: whether writes happened
    """
    counts = {
        "lines_read": 0,
        "entries_parsed": 0,
        "entries_skipped_malformed": 0,
        "upward_edges": 0,
        "downward_edges": 0,
        "skipped_templates": 0,
        "unsupported_templates": 0,
        # wyrd-aicu.8: edges dropped because a ref de-dashed to empty junk —
        # counted (not silently skipped) so an operator can tell 0 from N.
        "upward_edges_skipped_junk_parent": 0,
        "descendant_junk_word": 0,
        # wyrd-fqil: per-form variant emission stats
        "forms_emitted": 0,
        "forms_skipped_metadata": 0,
        "forms_skipped_unemittable": 0,
        # wyrd-ha9q Phase 2a: pronunciation + multi-script capture stats
        "pronunciation_captured": 0,
        "original_script_captured": 0,
        "transliteration_captured": 0,
        # wyrd-vsvi: tag-extraction stats from sense categories.
        # tags_added is total tag-writes; entries_with_tags is the
        # number of etymons that received at least one tag (denominator
        # for tag-coverage rate calcs).
        "tags_added": 0,
        "entries_with_tags": 0,
        "applied": int(apply),
    }
    if apply:
        db.upsert_source(id="wiktionary", title="Wiktionary")

    parsed_count = 0
    for line in stream:
        counts["lines_read"] += 1
        if counts["lines_read"] <= since_line:
            continue
        if limit is not None and parsed_count >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            counts["entries_skipped_malformed"] += 1
            continue
        # Drop an entry whose head word is junk after de-dash (a bare '-'
        # headword strips to empty — wyrd-aicu.8) here at the ingest boundary,
        # so the head upsert below never hands junk to the de-dash write choke.
        if (
            not isinstance(entry, dict)
            or normalize_morpheme_surface(entry.get("word")) is None
            or not entry.get("lang_code")
        ):
            counts["entries_skipped_malformed"] += 1
            continue
        parsed_count += 1
        counts["entries_parsed"] += 1
        _process_entry(db, entry, apply=apply, counts=counts)

    if apply:
        db.commit()
    return counts


# Sentinel id used during dry-run so _process_entry / _walk_descendants
# can still recurse and count edges without hitting the DB. Not a real
# row id; never reaches an INSERT because _emit_descent_edge no-ops on
# apply=False.
_DRY_RUN_PLACEHOLDER_ID = -1

# Single source-id literal so the SQL doesn't drift between callers and
# bumping to a new attribution scheme is one constant change.
_WIKTIONARY_SOURCE_ID = "wiktionary"


# wyrd-fqil: tag categorization for the `forms` array. Each form's
# tags list mixes inflectional features (plural/dative/genitive/etc.),
# alternative-spelling markers, romanization markers, canonical-lemma
# markers, and pure metadata (table-tags, inflection-template, class).
# We categorize into a small enum for fast queries; the original tag
# list is still preserved in etymon_variant.tags JSON for fine-grained
# filtering.
_FORM_TAG_METADATA: frozenset[str] = frozenset(
    {
        "table-tags",
        "inflection-template",
        "class",
        "no-table-tags",
        "no-equivalent",
    }
)

# Inflectional tags — having ANY of these on a form classifies it as
# `inflection`. Drawn from the top-30 wiktextract tags survey
# (2026-05-05): nominal/verbal inflection markers across our approved
# language families.
_FORM_TAG_INFLECTION: frozenset[str] = frozenset(
    {
        "plural",
        "singular",
        "dual",
        "nominative",
        "genitive",
        "dative",
        "accusative",
        "vocative",
        "ablative",
        "instrumental",
        "locative",
        "first-person",
        "second-person",
        "third-person",
        "active",
        "middle",
        "passive",
        "indicative",
        "subjunctive",
        "optative",
        "imperative",
        "infinitive",
        "present",
        "past",
        "future",
        "perfect",
        "imperfect",
        "aorist",
        "pluperfect",
        "participle",
        "gerund",
        "supine",
        "verbal",
        "masculine",
        "feminine",
        "neuter",
        "definite",
        "indefinite",
        "weak",
        "strong",
        "comparative",
        "superlative",
        "positive",
        "construct",
    }
)


def _classify_form_variant(tags: list[str]) -> str | None:
    """Return the variant_class label for a form's tag list, or None
    if the row is pure metadata (no actual word form to record).

    Decision precedence:
    - "alternative" tag → 'alternative' (Chaucer-era spellings, etc.)
    - "romanization" tag → 'romanization'
    - "canonical" tag → 'canonical' (Wiktionary's preferred lemma form)
    - any inflectional tag → 'inflection'
    - all tags are metadata → None (skip the row)
    - otherwise → 'other'
    """
    tag_set = set(tags or [])
    # Pure metadata: no actual form variant worth recording
    if tag_set and tag_set.issubset(_FORM_TAG_METADATA):
        return None
    if "alternative" in tag_set:
        return "alternative"
    if "romanization" in tag_set:
        return "romanization"
    if "canonical" in tag_set:
        return "canonical"
    if tag_set & _FORM_TAG_INFLECTION:
        return "inflection"
    return "other"


# wyrd-ha9q Phase 2a: pronunciation + multi-script extractors.
# Documented inline so the wiktextract sample shapes that drove each
# heuristic stay in code rather than just in a PR description.

# `head_templates[0].args` keys that carry vocalized / native-script
# forms. Empirical inspection across the wave-2 slices (2026-05-07):
#
#   - `wv`   — Hebrew vocalized (with niqqud). 10,826 hits / 17k Hebrew
#              entries. The cleanest signal: a single form, vocalized
#              version of the unvocalized lemma. Always preferred
#              when present.
#   - `head` — Arabic vocalized (with harakat) when not equal to the
#              lemma word. 2,548 hits / 3k Egyptian entries (hieroglyphic
#              markup, `<hiero>...</hiero>`). Messier: sometimes a
#              compound display string with `/` separators or affix
#              markers (`ـ` or hyphens), so we filter via
#              `_is_clean_native_form` before accepting.
#
# Order matters: the first key that yields a clean value wins. `wv`
# first because it's the unambiguous canonical convention; `head` is
# the broader fallback for languages whose templates don't carry `wv`.
_HEAD_TEMPLATE_NATIVE_SCRIPT_KEYS: tuple[str, ...] = ("wv", "head")


# Substrings / characters that disqualify a `head` arg as a clean
# native-script form. These would all confuse downstream renderers
# (or, in the case of `/`, indicate the editor packed multiple alternate
# forms into one string for display). Keeps the extractor conservative:
# better to leave original_script NULL than to surface "و / و" as if it
# were a single form.
_NATIVE_FORM_REJECT_TOKENS: tuple[str, ...] = (
    " / ",  # `'و / و'` (multi-form list)
    ", ",  # `'a, b'` (alt forms separated by commas)
    "*",  # reconstructed form marker (proto-language)
)

# `head_templates[0].args` keys that carry academic Latin-script
# transliterations (with diacritics). `tr` is the dominant one across
# Hebrew/Arabic/Sanskrit/Persian wiktextract entries. Some templates
# also use `head` for a pre-formatted transliteration; we don't grab
# that — `tr` is consistent enough that we'd rather have NULL than
# pick up a stray formatting string.
_HEAD_TEMPLATE_TRANSLIT_KEYS: tuple[str, ...] = ("tr",)


def _extract_pronunciation(sounds: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Pick the most authoritative IPA + first dialect tag from a
    wiktextract `sounds` array.

    Strategy: prefer entries with no dialect tag (those represent the
    canonical / undialectal pronunciation when wiktextract has one),
    otherwise fall back to the first entry that carries an IPA string.
    Many wave-2 entries only have dialected IPA (Biblical-Hebrew /
    Tiberian-Hebrew / Yemenite-Hebrew / Modern-Standard-Arabic), so the
    fallback is the common path; the dialect tag is captured alongside
    so the SPA panel can render '(Modern Standard Arabic)' etc.

    Returns (ipa, dialect) — either may be None when the input is empty
    or has no `ipa` key. Non-dict sound entries are silently skipped
    (mirrors `_walk_descendants`'s defensive style for malformed
    wiktextract input).
    """
    untagged_ipa: str | None = None
    fallback_ipa: str | None = None
    fallback_dialect: str | None = None
    for sound in sounds:
        if not isinstance(sound, dict):
            continue
        ipa = sound.get("ipa")
        if not ipa:
            continue
        tags = sound.get("tags") or []
        if not tags and untagged_ipa is None:
            untagged_ipa = ipa
        if fallback_ipa is None:
            fallback_ipa = ipa
            # First tag wins; wiktextract usually puts the most-specific
            # dialect first ('Biblical-Hebrew' before any sub-tags).
            fallback_dialect = tags[0] if tags else None
    if untagged_ipa is not None:
        return untagged_ipa, None
    return fallback_ipa, fallback_dialect


def _is_clean_native_form(value: str, lemma_word: str) -> bool:
    """Filter for `head`-style native-script values. Returns False when
    the value is one of:

    1. Empty or whitespace-only.
    2. Equal to the lemma word — no extra info to surface, NULL is
       the right "use canonical_form" signal.
    3. An affix-position marker — leading or trailing tatweel `ـ`
       (Arabic) or hyphen, signalling a partial form, not a complete
       native-script word.
    4. Carries any token from `_NATIVE_FORM_REJECT_TOKENS`:
       ` / ` (multi-form display string),
       `, ` (alt forms separated by commas),
       `*` (reconstructed form marker — proto-language; the rejection
       triggers anywhere in the string, not just at position 0, since
       reconstructed forms have no native-script render anyway).

    The Hebrew `wv` key wouldn't need this (it's always a single
    vocalized lemma) but the Arabic / Egyptian `head` arg routinely
    packs noisy display content.
    """
    if not value:
        return False
    if value == lemma_word:
        return False
    if value.startswith(("ـ", "-")) or value.endswith(("ـ", "-")):
        return False
    return not any(token in value for token in _NATIVE_FORM_REJECT_TOKENS)


def _extract_head_template_renderings(
    head_templates: list[dict[str, Any]], lemma_word: str
) -> tuple[str | None, str | None]:
    """Pull (original_script, transliteration) from `head_templates[0].args`.

    Wiktextract puts language-specific formatting metadata on the head
    template (e.g. `{{he-noun|wv=כֶּלֶב|tr=kɛ́lɛḇ, kélev|...}}`). The
    first head template is the one that wins; later templates are
    typically alternative POS headers.

    For original_script we walk `_HEAD_TEMPLATE_NATIVE_SCRIPT_KEYS` in
    order: `wv` (Hebrew vocalized — clean) first, `head` (Arabic
    vocalized / Egyptian hieroglyphic markup — sometimes noisy) as
    fallback. Each candidate is run through `_is_clean_native_form` so
    multi-form display strings and affix markers don't leak into
    storage.

    Returns (original_script, transliteration) — either may be None.
    """
    if not head_templates:
        return None, None
    first = head_templates[0]
    if not isinstance(first, dict):
        return None, None
    args = first.get("args")
    if not isinstance(args, dict):
        return None, None
    original: str | None = None
    for key in _HEAD_TEMPLATE_NATIVE_SCRIPT_KEYS:
        candidate = args.get(key)
        if candidate is None:
            continue
        cleaned = str(candidate).strip()
        if _is_clean_native_form(cleaned, lemma_word):
            original = cleaned
            break
    translit = next(
        (
            str(args[k]).strip()
            for k in _HEAD_TEMPLATE_TRANSLIT_KEYS
            if args.get(k) and str(args[k]).strip()
        ),
        None,
    )
    return original, translit


def _is_emittable_form(form_str: str, lemma_word: str) -> bool:
    """Filter out form rows we can't or shouldn't store as variants."""
    if not form_str:
        return False
    # Wiktextract sometimes emits inflection-template name strings
    # (e.g. "ang-decl-noun-a-n") in the forms array — those have
    # 'inflection-template' tags and would be filtered by classify(),
    # but defend against bare forms with template-marker shapes.
    if form_str.startswith("#") or form_str.startswith("-"):
        return False
    # Skip lemma self-reference — already in the etymon table as the
    # canonical_form, no need to duplicate as a variant.
    return form_str != lemma_word


def _emit_form_variants(
    db: LexiconDB,
    etymon_id: int,
    forms: list[dict[str, Any]],
    lemma_word: str,
    *,
    apply: bool,
    counts: dict[str, int],
) -> None:
    """Insert one etymon_variant row per form in `forms`. Counts are
    mutated in place. INSERT OR IGNORE on the (etymon_id, form,
    variant_class) UNIQUE so re-ingesting the same source is a no-op."""
    for form in forms:
        form_str = (form.get("form") or "").strip()
        tags = form.get("tags") or []
        if not _is_emittable_form(form_str, lemma_word):
            counts["forms_skipped_unemittable"] += 1
            continue
        variant_class = _classify_form_variant(tags)
        if variant_class is None:
            counts["forms_skipped_metadata"] += 1
            continue
        counts["forms_emitted"] += 1
        if not apply:
            continue
        db.conn.execute(
            "INSERT OR IGNORE INTO etymon_variant "
            "(etymon_id, form, variant_class, tags, source_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                etymon_id,
                form_str,
                variant_class,
                json.dumps(tags) if tags else None,
                _WIKTIONARY_SOURCE_ID,
            ),
        )


def _emit_descent_edge(
    db: LexiconDB,
    parent_id: int,
    child_id: int,
    edge_type: str,
    *,
    apply: bool,
) -> None:
    """Insert one etymon_descent row, deduped on the D27 UNIQUE
    constraint. Centralises the SQL so _process_entry and
    _walk_descendants don't drift on column names, source attribution,
    or the INSERT OR IGNORE shape."""
    if not apply:
        return
    db.conn.execute(
        "INSERT OR IGNORE INTO etymon_descent "
        "(parent_id, child_id, edge_type, source_id) VALUES (?, ?, ?, ?)",
        (parent_id, child_id, edge_type, _WIKTIONARY_SOURCE_ID),
    )


# wyrd-vsvi: coarse Wiktionary-category → kenning-tag mapping.
# Lives here (not in wiktextract_corpus_miner) because the full
# ingester needs it; the miner re-imports from here. Lookup is
# substring-based so "Christianity" and "en:Christianity" both hit
# the religious tag. Not exhaustive — the goal is to surface the
# most common semantic classes for place-name use; uncovered
# categories just produce no tags (still fine for matching).
_CATEGORY_PATTERNS: list[tuple[str, list[str]]] = [
    ("christianity", ["religious"]),
    ("religion", ["religious"]),
    ("places of worship", ["religious", "architecture"]),
    ("color", ["descriptive", "color"]),
    ("colors", ["descriptive", "color"]),
    ("topograph", ["topography"]),
    ("landform", ["topography", "geology"]),
    ("building", ["architecture"]),
    ("architecture", ["architecture"]),
    ("settlement", ["architecture", "social"]),
    ("plant", ["plant"]),
    ("tree", ["plant", "tree"]),
    ("animal", ["animal"]),
    ("mammal", ["animal", "mammal"]),
    ("bird", ["animal", "bird"]),
    ("water", ["water"]),
    ("river", ["water", "topography"]),
    ("body of water", ["water"]),
    ("agriculture", ["agriculture"]),
    ("farm", ["agriculture", "architecture"]),
    ("metal", ["material"]),
    ("stone", ["material", "geology"]),
    ("size", ["descriptive", "size"]),
    ("direction", ["descriptive", "direction"]),
]


def _map_categories_to_tags(categories: Iterable[str]) -> list[str]:
    """Translate wiktextract category names to kenning tag strings.

    Idempotent dedupe + sort so the resulting tag list is stable across
    re-runs (categories arrive in document order which can shuffle).
    Accepts any iterable so callers can pass a dedup'd set directly
    when they have one already.
    """
    out: set[str] = set()
    for cat in categories:
        cat_l = cat.lower()
        for pattern, tags in _CATEGORY_PATTERNS:
            if pattern in cat_l:
                out.update(tags)
    return sorted(out)


def _extract_entry_tags(entry: dict[str, Any]) -> list[str]:
    """wyrd-vsvi: union sense categories across an entry and map them to
    kenning tag strings. Mirrors the corpus-miner path's tag-extraction
    so both ingesters produce comparable etymon_tag coverage. The
    corpus miner picks ONE sense (the canonical one) and tags from its
    categories; the full ingester unions across ALL senses since an
    etymon can legitimately have multiple senses with distinct semantic
    classes (e.g. 'mill' = grinding-building AND a unit of measure →
    architecture + agriculture + measurement tags).

    Categories are deduplicated at the input layer (via set) before
    pattern-matching, which saves redundant substring scans when an
    entry has many senses sharing categories (common — wiktextract
    duplicates topical categories across senses). Returns sorted-
    dedupe list of tag strings; ``[]`` for entries with no
    categorizable senses (the common case for etymology-template stub
    rows that don't have full sense entries).
    """
    all_categories: set[str] = set()
    for sense in entry.get("senses") or []:
        for cat in sense.get("categories") or []:
            # The current wiktextract format emits each category as a
            # dict with name + kind + parents; older / custom configs
            # sometimes emit raw strings. Handle both for forward-
            # compat against wiktextract version drift.
            if isinstance(cat, dict):
                name = cat.get("name")
            elif isinstance(cat, str):
                name = cat
            else:
                name = None
            if name:
                all_categories.add(name)
    return _map_categories_to_tags(all_categories)


def _upsert_head_and_record(
    db: LexiconDB,
    entry: dict[str, Any],
    *,
    apply: bool,
    counts: dict[str, int],
) -> int:
    """Upsert the entry's head etymon and record its per-entry metadata.

    Pulls pronunciation + multi-script renderings (wyrd-ha9q Phase 2a)
    and sense tags (wyrd-vsvi), upserts the head etymon with them (the
    upsert COALESCEs against existing rows so a pre-column wave-1 etymon
    gets backfilled), and — on ``apply`` — records the
    pronunciation/original-script/transliteration capture counts and
    attaches the sense tags. Returns the head etymon id (or the dry-run
    placeholder). Mutates ``counts``.
    """
    this_word = entry["word"]
    this_lang = _canonical_language(entry["lang_code"])
    pron_ipa, pron_dialect = _extract_pronunciation(entry.get("sounds") or [])
    orig_script, translit = _extract_head_template_renderings(
        entry.get("head_templates") or [], this_word
    )
    tag_list = _extract_entry_tags(entry)
    this_id = (
        db.upsert_etymon(
            this_word,
            this_lang,
            pronunciation_ipa=pron_ipa,
            pronunciation_dialect=pron_dialect,
            original_script=orig_script,
            transliteration=translit,
        )
        if apply
        else _DRY_RUN_PLACEHOLDER_ID
    )
    if apply:
        if pron_ipa:
            counts["pronunciation_captured"] = counts.get("pronunciation_captured", 0) + 1
        if orig_script:
            counts["original_script_captured"] = counts.get("original_script_captured", 0) + 1
        if translit:
            counts["transliteration_captured"] = counts.get("transliteration_captured", 0) + 1
        if tag_list:
            for tag in tag_list:
                db.add_tag(this_id, tag)
            counts["tags_added"] = counts.get("tags_added", 0) + len(tag_list)
            counts["entries_with_tags"] = counts.get("entries_with_tags", 0) + 1
    return this_id


def _emit_upward_edges(
    db: LexiconDB,
    entry: dict[str, Any],
    this_id: int,
    counts: dict[str, int],
    *,
    apply: bool,
) -> None:
    """Emit UPWARD descent edges from the entry's etymology_templates.

    Each template may yield one or more edges from this entry to its
    named parent(s) — single-parent templates (inh/bor/der/cal) yield
    one, multi-parent (compound/affix, parallel roots) yield N. Templates
    that yield no edges bump ``skipped_templates`` (known-but-ignored) or
    ``unsupported_templates`` (unknown). Mutates ``counts``.
    """
    for tmpl in entry.get("etymology_templates") or []:
        name = tmpl.get("name", "")
        edges = _upward_edges_from_template(tmpl)
        if not edges:
            if name in _SKIPPED_TEMPLATE_NAMES:
                counts["skipped_templates"] += 1
            elif name not in _KNOWN_TEMPLATE_NAMES:
                counts["unsupported_templates"] += 1
            continue
        for parent_lang_code, parent_word, edge_type in edges:
            # A junk parent ref (strips to empty after de-dash) has no morpheme
            # to anchor the edge to — skip it rather than crash the upsert choke.
            # Count it (like every other skip in this function) so the drop is
            # visible to an operator rather than silent (wyrd-aicu.8).
            if normalize_morpheme_surface(parent_word) is None:
                counts["upward_edges_skipped_junk_parent"] += 1
                continue
            parent_lang = _canonical_language(parent_lang_code)
            parent_id = (
                db.upsert_etymon(parent_word, parent_lang) if apply else _DRY_RUN_PLACEHOLDER_ID
            )
            _emit_descent_edge(db, parent_id, this_id, edge_type, apply=apply)
            counts["upward_edges"] += 1


def _process_entry(
    db: LexiconDB,
    entry: dict[str, Any],
    *,
    apply: bool,
    counts: dict[str, int],
) -> None:
    """Walk one wiktextract entry: upsert the head etymon, then emit
    upward edges from etymology_templates and downward edges from the
    descendants tree. Counts are mutated in place."""
    this_id = _upsert_head_and_record(db, entry, apply=apply, counts=counts)
    _emit_upward_edges(db, entry, this_id, counts, apply=apply)

    # wyrd-fqil: emit etymon_variant rows for each form Wiktionary
    # records — alternative spellings, inflections, romanizations.
    # Lets surface-form lookups (the wyrd-ami pre-filter, et al.)
    # match historical / inflected forms that aren't separate lemmas.
    forms = entry.get("forms") or []
    if forms:
        _emit_form_variants(db, this_id, forms, entry["word"], apply=apply, counts=counts)

    # Descendants section — a NESTED TREE. Each node has lang_code +
    # word directly, with optional `descendants` for sub-trees.
    descendants = entry.get("descendants") or []
    if descendants:
        _walk_descendants(db, this_id, descendants, apply=apply, counts=counts)


def _walk_descendants(
    db: LexiconDB,
    parent_id: int,
    descendants: list[dict[str, Any]],
    *,
    apply: bool,
    counts: dict[str, int],
) -> None:
    """Walk one level of the wiktextract descendants tree, emitting one
    etymon_descent edge per node, then recursing into each node's own
    `descendants` array.

    Real wiktextract entries have the shape
        {"lang_code": "...", "word": "...", "tags": [...], "descendants": [...]}
    where `descendants` (when present) is the next layer of the tree.
    Some nodes lack `word` (lang-only header rows for sub-trees that
    didn't get specific forms attested) — those are skipped without
    breaking the chain into their children, since there's no etymon
    to anchor sub-descendants to.

    Edge_type per node defaults to 'inheritance' (the implied
    relationship of any Descendants section entry); 'calque' /
    'borrowing' / etc. on the `tags` array override per
    `_DESCENDANT_TAG_TO_EDGE`.
    """
    for node in descendants:
        if not isinstance(node, dict):
            continue
        child_lang_code = node.get("lang_code")
        child_word = node.get("word")
        normalized_child = normalize_morpheme_surface(child_word)
        if not child_lang_code or normalized_child is None:
            # A present-but-junk word (de-dashes to empty) is distinct from a
            # benign lang-only header row (no word at all) — count it so the
            # drop is visible (wyrd-aicu.8), but treat both the same way below.
            # ``child_word is not None`` (not truthiness) so an empty-string word
            # counts as present-but-junk, not as an absent header.
            if child_word is not None and normalized_child is None:
                counts["descendant_junk_word"] += 1
            # Lang-only header row (or that junk word). Don't emit an edge, but
            # DO recurse — sub-descendants of such a row attach to the same
            # parent as the header would have (the most recent anchored
            # ancestor up the recursion stack, which is `parent_id`).
            sub = node.get("descendants") or []
            if sub:
                _walk_descendants(db, parent_id, sub, apply=apply, counts=counts)
            continue
        child_lang = _canonical_language(child_lang_code)
        child_id = db.upsert_etymon(child_word, child_lang) if apply else _DRY_RUN_PLACEHOLDER_ID
        edge_type = _descendant_edge_type(node)
        _emit_descent_edge(db, parent_id, child_id, edge_type, apply=apply)
        counts["downward_edges"] += 1
        # Recurse into this node's own descendants — they hang off
        # this node's child_id, not the head.
        sub = node.get("descendants") or []
        if sub:
            _walk_descendants(db, child_id, sub, apply=apply, counts=counts)


def ingest_wiktextract_path(
    db: LexiconDB,
    path: Path,
    *,
    apply: bool = False,
    limit: int | None = None,
    since_line: int = 0,
) -> dict[str, int]:
    """Convenience wrapper: open ``path`` (gzipped, zstd, or plain) and
    feed it to ``ingest_wiktextract_stream``. Wiktextract dumps from
    kaikki.org are typically ``.jsonl.gz``; the wyrd-0vj3 bulk-source
    pipeline stores them as ``.jsonl.zst`` in ~/.wyrd/sources/.
    """
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as f:
            return ingest_wiktextract_stream(db, f, apply=apply, limit=limit, since_line=since_line)
    if path.suffix == ".zst":
        # wyrd-0vj3: bulk-source slices live in ~/.wyrd/sources/ as
        # zstd-compressed jsonl. Delegate to the shared opener so
        # both ingest paths and any future jsonl reader pick up
        # transparent decompression.
        from wyrd.generators.kenning.bulk_sources import open_jsonl

        with open_jsonl(path) as f:
            return ingest_wiktextract_stream(db, f, apply=apply, limit=limit, since_line=since_line)
    with path.open("r", encoding="utf-8") as f:
        return ingest_wiktextract_stream(db, f, apply=apply, limit=limit, since_line=since_line)
