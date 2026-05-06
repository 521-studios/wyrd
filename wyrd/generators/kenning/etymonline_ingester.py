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

from typing import Any

from wyrd.generators.kenning.etymonline_parser import (
    Sense,
    parse_text,
)
from wyrd.generators.kenning.lexicon import LexiconDB

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


def ingest_sense(
    db: LexiconDB,
    sense: Sense,
    *,
    apply: bool,
) -> dict[str, int]:
    """Write one parsed Sense's chain into the etymon graph. Counts
    are returned, mutating only the DB.

    Strategy:
    - Etymonline lists chain in descent order: harpy ← OFr harpie
      ← Latin harpyia ← Greek Harpyia. So chain[0] is the immediate
      parent of the headword; chain[-1] is the deepest ancestor.
    - For each chain link, upsert (language, word) → etymon_id.
    - Edge chain[i+1] → chain[i] for adjacent pairs (going leaf-ward).
    - Leaf edge chain[0] → headword (if headword resolves to an
      existing etymon).
    """
    counts = {
        "chain_links": len(sense.chain),
        "etymons_added_or_existing": 0,
        "edges_added": 0,
        "edges_skipped_dupe": 0,
        "leaf_edge_skipped_no_headword": 0,
        "glosses_added": 0,
    }
    if not sense.chain:
        return counts

    # Upsert each chain etymon and remember its id (in chain order:
    # link[0] = closest to leaf, link[-1] = deepest ancestor).
    link_ids: list[int] = []
    for link in sense.chain:
        if apply:
            eid = db.upsert_etymon(link.word, link.language)
            link_ids.append(eid)
            if link.gloss:
                db.add_gloss(eid, link.gloss)
                counts["glosses_added"] += 1
        else:
            link_ids.append(-1)  # placeholder — dry-run can't track
        counts["etymons_added_or_existing"] += 1

    if not apply:
        return counts

    # Adjacent edges: chain[i+1] is parent of chain[i].
    for i in range(len(sense.chain) - 1):
        parent_id = link_ids[i + 1]
        child_id = link_ids[i]
        # The edge type comes from the CHILD's hedge — that's the link
        # describing how the child relates to its parent.
        edge_type = sense.chain[i].edge_type
        confidence = _confidence_label(sense.chain[i].confidence)
        if _emit_descent_edge(db, parent_id, child_id, edge_type, confidence):
            counts["edges_added"] += 1
        else:
            counts["edges_skipped_dupe"] += 1

    # Leaf edge: chain[0] → headword (if headword exists in the corpus).
    head_id = _lookup_etymon_id(db, sense.headword, _HEADWORD_LANGUAGE)
    if head_id is None:
        counts["leaf_edge_skipped_no_headword"] += 1
    else:
        edge_type = sense.chain[0].edge_type
        confidence = _confidence_label(sense.chain[0].confidence)
        if _emit_descent_edge(db, link_ids[0], head_id, edge_type, confidence):
            counts["edges_added"] += 1
        else:
            counts["edges_skipped_dupe"] += 1

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


def _normalize_count_keys(d: dict[str, Any]) -> dict[str, int]:
    """Helper for tests / callers — return only int-valued count keys."""
    return {k: int(v) for k, v in d.items() if isinstance(v, int)}
