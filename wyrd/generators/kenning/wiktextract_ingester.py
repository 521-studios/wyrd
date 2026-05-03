"""Wiktextract → etymon_descent ingester (wyrd-4rt / wyrd-hun).

Pure-Python parser of wiktextract JSONL output (see kaikki.org). One
input record per line; each record describes a single Wiktionary entry
(word, language, part-of-speech, sections). We extract the Etymology
section's chain templates and the Descendants section's tree, emit
etymon rows for every form mentioned, and insert etymon_descent edges
with the appropriate edge_type per D27.

This is NOT LLM mining — there's no Tier 1/2/3 model dispatch and no
form-in-body validation, because wiktextract has already parsed the
Wiktionary markup into structured JSON. The "extraction" was done by
human Wiktionary editors. Our job is to map their template kinds to
our edge_type taxonomy and load the graph.

Template kind mapping (D27 edge_type column):
  {{inh|<this>|<parent_lang>|<parent_word>}}  → 'inheritance' (UP edge)
  {{bor|<this>|<parent_lang>|<parent_word>}}  → 'borrowing'   (UP edge)
  {{der|<this>|<parent_lang>|<parent_word>}}  → 'derivation'  (UP edge)
  {{cal|<this>|<parent_lang>|<parent_word>}}  → 'calque'      (UP edge)
  {{desc|<child_lang>|<child_word>}}           → 'inheritance' (DOWN edge)
                                                  (descendants default to
                                                  inheritance unless
                                                  flagged explicitly)
  {{cog|<lang>|<word>}}                        → skipped (peer relation,
                                                  no parent direction)
  {{m|<lang>|<word>}}                          → skipped (mention only)
  other                                        → skipped (counter incremented)

Descendants nesting: wiktextract flattens the Descendants tree into a
list of dicts with a `depth` field. depth=1 entries are direct children
of the parent entry; depth=N+1 entries are children of the most recent
preceding entry at depth=N. We track this with a parent_stack indexed
by depth.
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

# Etymology-section templates that produce an UPWARD edge (this entry
# descends from the parent the template names).
_UPWARD_TEMPLATE_TO_EDGE: dict[str, str] = {
    "inh": "inheritance",
    "inherited": "inheritance",
    "bor": "borrowing",
    "borrowed": "borrowing",
    "der": "derivation",
    "derived": "derivation",
    "cal": "calque",
    "calque": "calque",
}

# Descendants-section templates that produce a DOWNWARD edge.
_DOWNWARD_TEMPLATE_NAMES: frozenset[str] = frozenset({"desc", "descendant"})

# Templates we explicitly skip because they're either peer relations
# (cognate, mention) or non-etymological (formatting, qualifiers).
_SKIPPED_TEMPLATE_NAMES: frozenset[str] = frozenset(
    {
        "cog",
        "cognate",
        "m",
        "mention",
        "l",
        "link",
        "qualifier",
        "qual",
        "gloss",
        "noncog",
        "etyl",
        "lb",
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


def _upward_edge_from_template(
    tmpl: dict[str, Any],
) -> tuple[str, str, str] | None:
    """Pull (parent_lang_code, parent_word, edge_type) out of a wiktextract
    etymology template. Returns None if the template isn't one of our
    upward-edge kinds or its args are incomplete.

    Convention: {{inh|<this_lang>|<parent_lang>|<parent_word>}} —
    args["1"] is THIS entry's lang_code (which we already know from
    entry.lang_code), args["2"] is parent lang, args["3"] is parent word.
    """
    name = tmpl.get("name", "")
    edge_type = _UPWARD_TEMPLATE_TO_EDGE.get(name)
    if edge_type is None:
        return None
    args = _extract_template_args(tmpl)
    parent_lang_code = args.get("2")
    parent_word = args.get("3")
    if not parent_lang_code or not parent_word:
        return None
    return parent_lang_code, parent_word, edge_type


def _downward_edge_from_template(
    tmpl: dict[str, Any],
) -> tuple[str, str] | None:
    """Pull (child_lang_code, child_word) out of a wiktextract descendants
    template. Returns None if the template isn't a desc kind or args
    are incomplete.

    Convention: {{desc|<child_lang>|<child_word>}}.
    """
    name = tmpl.get("name", "")
    if name not in _DOWNWARD_TEMPLATE_NAMES:
        return None
    args = _extract_template_args(tmpl)
    child_lang_code = args.get("1")
    child_word = args.get("2")
    if not child_lang_code or not child_word:
        return None
    return child_lang_code, child_word


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
      - depth_jumps_recovered: count of descendants entries whose depth
        skipped one or more levels (treated as direct children of the
        current frontier)
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
        "depth_jumps_recovered": 0,
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

    # Etymology templates — each is an UPWARD edge from this entry to
    # the named parent.
    for tmpl in entry.get("etymology_templates") or []:
        name = tmpl.get("name", "")
        edge = _upward_edge_from_template(tmpl)
        if edge is None:
            if name in _SKIPPED_TEMPLATE_NAMES:
                counts["skipped_templates"] += 1
            elif name not in _UPWARD_TEMPLATE_TO_EDGE:
                counts["unsupported_templates"] += 1
            continue
        parent_lang_code, parent_word, edge_type = edge
        parent_lang = _canonical_language(parent_lang_code)
        if apply:
            parent_id = db.upsert_etymon(parent_word, parent_lang)
            db.conn.execute(
                "INSERT OR IGNORE INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) VALUES (?, ?, ?, ?)",
                (parent_id, this_id, edge_type, "wiktionary"),
            )
        counts["upward_edges"] += 1

    # Descendants section — flat list with depth markers. parent_stack
    # tracks the most recent entry at each depth; the parent of a
    # depth=N entry is the most recent depth=(N-1) entry.
    descendants = entry.get("descendants") or []
    if descendants:
        _walk_descendants(db, this_id, descendants, apply=apply, counts=counts)


# Sentinel id used during dry-run so _walk_descendants can still exercise
# the parent_stack logic without hitting the DB. Not a real row id.
_DRY_RUN_PLACEHOLDER_ID = -1


def _walk_descendants(
    db: LexiconDB,
    head_id: int,
    descendants: list[dict[str, Any]],
    *,
    apply: bool,
    counts: dict[str, int],
) -> None:
    """Walk a flat descendants list with depth markers and emit one
    etymon_descent edge per {{desc}} template. Index 0 of parent_stack
    is the head entry (the etymon whose Descendants section we're in);
    deeper levels grow as we encounter them.

    Defensive against malformed depth jumps — if a depth=N+2 entry
    appears with no preceding depth=N+1, we recover by attaching to
    the deepest available parent and incrementing depth_jumps_recovered.
    """
    parent_stack: list[int] = [head_id]
    for desc in descendants:
        if not isinstance(desc, dict):
            continue
        depth = desc.get("depth", 1)
        if not isinstance(depth, int) or depth < 1:
            depth = 1
        # Trim or extend the stack so parent_stack[depth-1] is the parent.
        if depth <= len(parent_stack):
            parent_stack = parent_stack[:depth]
        else:
            counts["depth_jumps_recovered"] += 1
            # Pad with the deepest known parent so the chain doesn't snap.
            while len(parent_stack) < depth:
                parent_stack.append(parent_stack[-1])
        parent_id = parent_stack[-1]

        # Each desc entry can carry multiple templates (e.g. one for the
        # word, one for a gloss). Emit edges only for desc-kind templates.
        last_child_id: int | None = None
        for tmpl in desc.get("templates") or []:
            name = tmpl.get("name", "")
            edge = _downward_edge_from_template(tmpl)
            if edge is None:
                if name in _SKIPPED_TEMPLATE_NAMES:
                    counts["skipped_templates"] += 1
                elif name not in _DOWNWARD_TEMPLATE_NAMES:
                    counts["unsupported_templates"] += 1
                continue
            child_lang_code, child_word = edge
            child_lang = _canonical_language(child_lang_code)
            if apply:
                child_id = db.upsert_etymon(child_word, child_lang)
                db.conn.execute(
                    "INSERT OR IGNORE INTO etymon_descent "
                    "(parent_id, child_id, edge_type, source_id) VALUES (?, ?, ?, ?)",
                    (parent_id, child_id, "inheritance", "wiktionary"),
                )
                last_child_id = child_id
            else:
                last_child_id = _DRY_RUN_PLACEHOLDER_ID
            counts["downward_edges"] += 1

        # The last child becomes the parent for any deeper entries that
        # follow at depth+1. Push it onto the stack at index = depth.
        if last_child_id is not None:
            parent_stack.append(last_child_id)


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
