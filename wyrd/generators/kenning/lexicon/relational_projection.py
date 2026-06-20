"""Project Family-B ``descends-from`` assertions into ``etymon_descent`` (wyrd-zrce.1, D50.6).

The relational complement to ``project_canonical`` (which projects Family-A identity
into the ``canonical_*`` graph). Two reasons it is a SEPARATE stage:

* ``project_canonical`` is the TERMINAL enrichment pass (it needs the final identity
  graph); this must run EARLY — before ``cluster_cognates`` (step 5 of
  ``run_full_enrichment``) — so mined cognate descents feed the ``cognate_id`` rollup
  in the same run.
* identity vs relational are orthogonal axes (D28/D50.3).

Reads the live ``descends-from`` assertions from L2 (effective = retraction-resolved;
affirmed; not refuted; confidence ≥ gate) and materializes them as ``etymon_descent``
rows under a dedicated ``source_id``. Idempotent + deterministic (D36.9): clears its
OWN source's rows then re-inserts, so a retracted/refuted edge disappears and the
result is a pure function of the L2 streams. Scoped to its ``source_id`` — never
touches Wiktionary-ingested edges.

Gate defaults to ``"medium"``, not ``"high"``: these are RELATIONAL edges
(never-collapse), so the identity leave-separate bar doesn't apply. ``medium`` =
gloss-corroborated cognate matches project; ``"low"`` (surface-only, homograph risk)
needs an explicit lower gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wyrd.generators.kenning.canonicalization import (
    Assertion,
    effective_assertions,
    load_assertions,
    resolve_confidence_gate,
    score_confidence,
)
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.etymon_refs import resolve_etymon_ref

# Attribution for the edges this projection materializes (kept distinct from
# ``wiktionary`` so a clear/rebuild only ever touches mined cognate-descent edges).
DESCENT_SOURCE_ID = "cognate-descent-uplift"
_DESCENT_SOURCE_TITLE = "Cognate-descent uplift (wyrd-zrce.1, D50 Family B)"

# An edge as the (parent_ref, child_ref, edge_type) string triple an assertion
# describes — descends-from(subject=child, object=parent). The refs are stable
# natural keys ("language:canonical_form", wyrd-c6wu/D50.1); resolution to the
# current build's etymon ids is confined to the INSERT boundary.
_EdgeKey = tuple[str, str, str | None]


def _edge_key(a: Assertion) -> _EdgeKey:
    return (a.object.ref, a.subject.ref, a.qualifiers.get("edge_type"))


@dataclass
class DescentProjectionResult:
    """What the projection did (or would do, in dry-run)."""

    applied: bool = False
    inserted: int = 0  # distinct edges materialized
    cleared: int = 0  # this source's pre-existing rows removed before rebuild
    gated_out: int = 0  # below-gate descends-from assertions skipped
    refuted: int = 0  # affirmed edges dropped by a matching refute (D50.4)
    unresolved: int = 0  # endpoint natural-key ref not in this build (wyrd-c6wu, D24)


def project_descent_assertions(
    db: LexiconDB,
    *,
    mining_dir: Path,
    apply: bool = False,
    confidence_gate: str | float = "medium",
    source_id: str = DESCENT_SOURCE_ID,
) -> DescentProjectionResult:
    """Materialize live ``descends-from`` assertions into ``etymon_descent``.

    ``apply=False`` reports what would change without writing. ``descends-from`` means
    subject (child) descends from object (parent), so the row is
    ``(parent_id=object, child_id=subject, edge_type)``.
    """
    gate = resolve_confidence_gate(confidence_gate)
    result = DescentProjectionResult(applied=apply)
    live = [
        a
        for a in effective_assertions(list(load_assertions(mining_dir)))
        if a.predicate == "descends-from"
    ]

    # A refute asserts NOT-descends for an edge; drop a matching affirm (D50.4).
    refuted = {_edge_key(a) for a in live if a.polarity == "refute" and a.object is not None}

    edges: set[tuple[int, int, str]] = set()
    for a in live:
        if a.polarity != "affirm" or a.object is None:
            continue
        key = _edge_key(a)
        if key in refuted:
            result.refuted += 1
            continue
        if score_confidence(a.confidence) < gate:
            result.gated_out += 1
            continue
        parent_ref, child_ref, edge_type = key
        # wyrd-c6wu: refs are stable natural keys; resolve to THIS build's etymon
        # ids here (the projection/INSERT boundary) — never int(ref), since row-ids
        # aren't stable across rebuilds. An endpoint that isn't in this build (a
        # stale/cross-corpus ref) is skipped + counted (D24 observability), not a
        # crash — the edge simply doesn't materialize.
        parent_id = resolve_etymon_ref(db.conn, parent_ref)
        child_id = resolve_etymon_ref(db.conn, child_ref)
        if parent_id is None or child_id is None:
            result.unresolved += 1
            continue
        edges.add((parent_id, child_id, edge_type))

    result.inserted = len(edges)
    if apply:
        row = db.conn.execute(
            "SELECT count(*) FROM etymon_descent WHERE source_id = ?", (source_id,)
        ).fetchone()
        result.cleared = row[0]
        db.conn.execute("DELETE FROM etymon_descent WHERE source_id = ?", (source_id,))
        db.conn.execute(
            "INSERT OR IGNORE INTO source (id, title) VALUES (?, ?)",
            (source_id, _DESCENT_SOURCE_TITLE),
        )
        db.conn.executemany(
            "INSERT OR IGNORE INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, ?, ?)",
            sorted((p, c, et, source_id) for p, c, et in edges),
        )
        db.commit()
    return result
