"""Etymonline → etymon_descent ingester (wyrd-zqkp).

Takes the parsed Sense list from etymonline_parser.parse_text and
writes:

- One source row, id='etymonline', so descent edges + glosses can be
  attributed back to Etymonline.
- One etymon row per chain link's (language, word) pair (idempotent
  via the existing etymon UNIQUE on canonical_form+language).
- One etymon_descent edge per consecutive chain pair, source_id=
  'etymonline'. The leaf edge (chain[0] → headword) is added when the
  headword is already in the etymon corpus.
- Optionally: one etymon_gloss row per link with a gloss.

Why edges-only and not citation rows: descent edges already carry a
source_id, which is sufficient citation. The etymon_citation table is
for extraction-evidence tied to specific source pages (mining runs);
Etymonline's per-entry pages don't fit that shape — the edge IS the
citation.

Bulk-mode usage from the CLI:

    wyrd kenning lexicon ingest-etymonline path/to/dir/

Where `dir/` contains one .txt file per word (output of `rodney text
"section.prose-lg"`). The CLI iterates and ingests each.
"""

from __future__ import annotations

from wyrd.generators.kenning.etymonline_parser import (
    Sense,
    parse_text,
)
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.morpheme_surface import normalize_morpheme_surface

# Single source-id literal — bumping to a new attribution scheme is
# one constant change.
ETYMONLINE_SOURCE_ID = "etymonline"

# Default headword language. Etymonline is monolingual English, so the
# leaf of every chain is an English word. Most Etymonline entries
# correspond to modern-english lemmas in our wiktextract-derived
# corpus; older Anglo-Saxon entries occasionally surface but are rare
# enough to not warrant a per-entry override.
_HEADWORD_LANGUAGE = "modern-english"


def ensure_source(db: LexiconDB) -> None:
    """Create the 'etymonline' source row if it's not already there."""
    db.upsert_source(
        id=ETYMONLINE_SOURCE_ID,
        title="Etymonline (Online Etymology Dictionary)",
        author="Douglas Harper",
        year=None,
    )


def _confidence_label(confidence: str) -> str:
    """Map ChainLink.confidence to etymon_descent.confidence values
    (CHECK constraint accepts 'high', 'medium', 'low')."""
    return confidence if confidence in ("high", "medium", "low") else "low"


def _emit_descent_edge(
    db: LexiconDB,
    parent_id: int,
    child_id: int,
    edge_type: str,
    confidence: str,
) -> bool:
    """Insert one etymon_descent edge with source='etymonline'.
    Returns True if this is a new row (vs a UNIQUE-conflict skip)."""
    cur = db.conn.execute(
        """INSERT OR IGNORE INTO etymon_descent
           (parent_id, child_id, edge_type, source_id, confidence)
           VALUES (?, ?, ?, ?, ?)""",
        (parent_id, child_id, edge_type, ETYMONLINE_SOURCE_ID, confidence),
    )
    return cur.rowcount > 0


def _lookup_etymon_id(db: LexiconDB, canonical_form: str, language: str) -> int | None:
    """Find an existing etymon row by (canonical_form, language).
    Used for the leaf-edge case where we want to wire chain[0] →
    headword without creating the headword row ourselves (it should
    already exist from wiktextract ingest)."""
    row = db.conn.execute(
        "SELECT id FROM etymon WHERE canonical_form = ? AND language = ?",
        (canonical_form, language),
    ).fetchone()
    return row[0] if row else None


def _upsert_chain(
    db: LexiconDB,
    sense: Sense,
    counts: dict[str, int],
) -> list[int | None]:
    """Upsert each chain link's etymon row, attach glosses, and return
    the resulting etymon ids in chain order. Mutates `counts` for
    `etymons_added_or_existing` and `glosses_added`.

    A junk link that de-dashes to empty (wyrd-aicu.8) would make the upsert
    choke raise, so it is dropped — but its slot is kept as ``None`` so the
    returned list stays INDEX-ALIGNED with ``sense.chain``. The edge emitters
    (`_emit_adjacent_edges` / `_emit_leaf_edge`) pair entries with chain indices
    positionally, so a bare ``continue`` here would mis-pair every edge past the
    drop (or IndexError); the ``None`` placeholder lets them skip edges that
    touch the dropped link instead."""
    link_ids: list[int | None] = []
    for link in sense.chain:
        if normalize_morpheme_surface(link.word) is None:
            link_ids.append(None)
            counts["chain_links_skipped_junk"] += 1
            continue
        eid = db.upsert_etymon(link.word, link.language)
        link_ids.append(eid)
        if link.gloss:
            db.add_gloss(eid, link.gloss)
            counts["glosses_added"] += 1
        counts["etymons_added_or_existing"] += 1
    return link_ids


def _emit_adjacent_edges(
    db: LexiconDB,
    sense: Sense,
    link_ids: list[int | None],
    counts: dict[str, int],
) -> None:
    """For each consecutive (parent, child) pair in the chain, emit
    the etymon_descent edge — chain[i+1] is parent of chain[i] since
    Etymonline lists in descent (leaf-most-first) order."""
    for i in range(len(sense.chain) - 1):
        parent_id = link_ids[i + 1]
        child_id = link_ids[i]
        # Either endpoint may be a dropped junk link (None, wyrd-aicu.8) — no
        # edge to a non-existent etymon.
        if parent_id is None or child_id is None:
            continue
        # The edge label comes from the PARENT link (chain[i+1]): each chain
        # link is introduced by the hedge that describes its descent down to the
        # PREVIOUS element (parse_chain, "X from Y" → ChainLink(Y) carrying the
        # "from" edge_type/confidence for the Y→X edge). So the hedge for the
        # chain[i+1]→chain[i] edge lives on chain[i+1], not chain[i]. Matches
        # _emit_leaf_edge, which correctly reads chain[0] for the
        # chain[0]→headword edge; using chain[i] here mislabeled every adjacent
        # edge with the next-shallower link's hedge (off-by-one).
        edge_type = sense.chain[i + 1].edge_type
        confidence = _confidence_label(sense.chain[i + 1].confidence)
        if _emit_descent_edge(db, parent_id, child_id, edge_type, confidence):
            counts["edges_added"] += 1
        else:
            counts["edges_skipped_dupe"] += 1


def _emit_leaf_edge(
    db: LexiconDB,
    sense: Sense,
    link_ids: list[int | None],
    counts: dict[str, int],
) -> None:
    """Wire chain[0] → headword if the headword exists in the corpus
    as a modern-english etymon. Skipped (with counter) otherwise."""
    # chain[0] (the headword's immediate parent) may be a dropped junk link
    # (None, wyrd-aicu.8) — no etymon to anchor the leaf edge to. The
    # ``not link_ids`` guard is belt-and-suspenders for an empty chain (the
    # caller already short-circuits on it) so this helper is self-safe.
    if not link_ids or link_ids[0] is None:
        return
    # De-dash the headword for the lookup (wyrd-aicu.8, D45): the corpus stores
    # bare surfaces, so a dashed headword (e.g. '-ism') must be normalized or the
    # leaf edge is silently dropped. Normalize into a local — don't mutate Sense.
    # A junk headword normalizes to None — skip the always-failing lookup.
    head_form = normalize_morpheme_surface(sense.headword)
    head_id = (
        _lookup_etymon_id(db, head_form, _HEADWORD_LANGUAGE) if head_form is not None else None
    )
    if head_id is None:
        counts["leaf_edge_skipped_no_headword"] += 1
        return
    edge_type = sense.chain[0].edge_type
    confidence = _confidence_label(sense.chain[0].confidence)
    if _emit_descent_edge(db, link_ids[0], head_id, edge_type, confidence):
        counts["edges_added"] += 1
    else:
        counts["edges_skipped_dupe"] += 1


def ingest_sense(
    db: LexiconDB,
    sense: Sense,
    *,
    apply: bool,
) -> dict[str, int]:
    """Write one parsed Sense's chain into the etymon graph. Counts
    are returned, mutating only the DB.

    Etymonline lists chain in descent order: harpy ← OFr harpie ←
    Latin harpyia ← Greek Harpyia. chain[0] is the immediate parent
    of the headword; chain[-1] is the deepest ancestor. Three phases:
    upsert chain etymons, emit adjacent chain-pair edges, emit the
    leaf edge to the headword.
    """
    counts = {
        "chain_links": len(sense.chain),
        "etymons_added_or_existing": 0,
        "chain_links_skipped_junk": 0,
        "edges_added": 0,
        "edges_skipped_dupe": 0,
        "leaf_edge_skipped_no_headword": 0,
        "glosses_added": 0,
    }
    if not sense.chain:
        return counts
    if not apply:
        # Dry-run: just count the chain links so callers can preview.
        counts["etymons_added_or_existing"] = len(sense.chain)
        return counts
    link_ids = _upsert_chain(db, sense, counts)
    _emit_adjacent_edges(db, sense, link_ids, counts)
    _emit_leaf_edge(db, sense, link_ids, counts)
    return counts


def ingest_text(
    db: LexiconDB,
    text: str,
    *,
    apply: bool,
) -> dict[str, int]:
    """Parse one Etymonline prose dump (possibly multi-sense) and
    ingest each Sense. Returns aggregated counts plus a sense_count
    field for visibility."""
    senses = parse_text(text)
    if apply:
        ensure_source(db)
    totals: dict[str, int] = {
        "senses_parsed": len(senses),
        "senses_with_chain": 0,
        "senses_without_chain": 0,
        "chain_links": 0,
        "etymons_added_or_existing": 0,
        "chain_links_skipped_junk": 0,
        "edges_added": 0,
        "edges_skipped_dupe": 0,
        "leaf_edge_skipped_no_headword": 0,
        "glosses_added": 0,
    }
    for sense in senses:
        if not sense.chain:
            totals["senses_without_chain"] += 1
            continue
        totals["senses_with_chain"] += 1
        sense_counts = ingest_sense(db, sense, apply=apply)
        for k, v in sense_counts.items():
            if k in totals:
                totals[k] += v
    if apply:
        db.commit()
    return totals
