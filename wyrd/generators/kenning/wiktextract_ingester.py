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
from pathlib import Path
from typing import IO, Any

from wyrd.generators.kenning.lexicon import LexiconDB

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
        "vrd",
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
        # slices. Mostly formatting (smallcaps, angbr, PIE root box, etydate)
        # or specialised derivational hints with un-confirmed arg shapes
        # (past participle of, alt form, alter, displaced). Skipping to
        # silence the audit noise; if the larger language slices surface
        # them at scale, file a follow-up to extend
        # _SAME_LANG_DERIVATION_TEMPLATE_NAMES.
        "senseno",
        "C.E.",
        "smallcaps",
        "angbr",
        "PIE root box",
        "abbrev",
        "non-gloss",
        "displaced",
        "past participle of",
        "alt form",
        "alter",
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
    """Resolve the edge_type for a descendants-tree node from its `tags`
    array. Defaults to 'inheritance' (the implied relationship of any
    Descendants section entry). The first matching tag wins; multiple
    matches are theoretically possible but not observed in real data."""
    for tag in node.get("tags") or []:
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
        if not isinstance(entry, dict) or not entry.get("word") or not entry.get("lang_code"):
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
    this_word = entry["word"]
    this_lang = _canonical_language(entry["lang_code"])
    this_id = db.upsert_etymon(this_word, this_lang) if apply else _DRY_RUN_PLACEHOLDER_ID

    # Etymology templates — each may produce one or more UPWARD edges
    # from this entry to its named parent(s). Single-parent templates
    # (inh/bor/der/cal) yield exactly one edge. Multi-parent templates
    # (compound/affix, root with parallel roots) yield N edges.
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
            parent_lang = _canonical_language(parent_lang_code)
            parent_id = (
                db.upsert_etymon(parent_word, parent_lang) if apply else _DRY_RUN_PLACEHOLDER_ID
            )
            _emit_descent_edge(db, parent_id, this_id, edge_type, apply=apply)
            counts["upward_edges"] += 1

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
        if not child_lang_code or not child_word:
            # Lang-only header row. Don't emit an edge, but DO recurse —
            # sub-descendants of such a row attach to the same parent
            # as the header would have (the most recent anchored
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
    """Convenience wrapper: open `path` (gzipped or plain) and feed
    it to `ingest_wiktextract_stream`. Wiktextract dumps from
    kaikki.org are typically `.jsonl.gz`; we sniff the suffix."""
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as f:
            return ingest_wiktextract_stream(db, f, apply=apply, limit=limit, since_line=since_line)
    with path.open("r", encoding="utf-8") as f:
        return ingest_wiktextract_stream(db, f, apply=apply, limit=limit, since_line=since_line)
