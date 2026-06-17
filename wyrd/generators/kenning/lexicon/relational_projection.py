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

from wyrd.generators.kenning.canonicalization import effective_assertions, load_assertions
from wyrd.generators.kenning.lexicon.db import LexiconDB

# Attribution for the edges this projection materializes (kept distinct from
# ``wiktionary`` so a clear/rebuild only ever touches mined cognate-descent edges).
DESCENT_SOURCE_ID = "cognate-descent-uplift"
_DESCENT_SOURCE_TITLE = "Cognate-descent uplift (wyrd-zrce.1, D50 Family B)"

# Mirrors canonicalization_projection._LABEL_SCORE / _resolve_gate — relational has a
# different default gate, but the same label→score scale (kept tiny + local rather
# than importing a private symbol across modules).
_LABEL_SCORE = {"low": 0.3, "medium": 0.6, "high": 0.9}


def _score(confidence: str | float) -> float:
    if isinstance(confidence, str):
        return _LABEL_SCORE.get(confidence, 0.0)
    return float(confidence)


def _resolve_gate(confidence_gate: str | float) -> float:
    """Validate + score the gate; reject a typo'd label / out-of-range float loudly
    (a silent collapse to 0.0 would admit every low-confidence edge)."""
    if isinstance(confidence_gate, str):
        if confidence_gate in _LABEL_SCORE:
            return _LABEL_SCORE[confidence_gate]
        try:
            confidence_gate = float(confidence_gate)
        except ValueError:
            raise ValueError(
                f"unknown confidence_gate {confidence_gate!r}; "
                f"expected one of {sorted(_LABEL_SCORE)} or a float in [0, 1]"
            ) from None
    gate = float(confidence_gate)
    if not 0.0 <= gate <= 1.0:
        raise ValueError(f"confidence_gate float must be in [0, 1], got {gate}")
    return gate


@dataclass
class DescentProjectionResult:
    """What the projection did (or would do, in dry-run)."""

    applied: bool = False
    inserted: int = 0  # distinct edges materialized
    cleared: int = 0  # this source's pre-existing rows removed before rebuild
    gated_out: int = 0  # below-gate descends-from assertions skipped
    refuted: int = 0  # affirmed edges dropped by a matching refute (D50.4)


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
    gate = _resolve_gate(confidence_gate)
    result = DescentProjectionResult(applied=apply)
    live = [
        a
        for a in effective_assertions(list(load_assertions(mining_dir)))
        if a.predicate == "descends-from"
    ]

    # A refute asserts NOT-descends for an edge; drop a matching affirm (D50.4).
    refuted = {
        (a.object.ref, a.subject.ref, a.qualifiers.get("edge_type"))
        for a in live
        if a.polarity == "refute" and a.object is not None
    }

    edges: set[tuple[int, int, str]] = set()
    for a in live:
        if a.polarity != "affirm" or a.object is None:
            continue
        if (a.object.ref, a.subject.ref, a.qualifiers.get("edge_type")) in refuted:
            result.refuted += 1
            continue
        if _score(a.confidence) < gate:
            result.gated_out += 1
            continue
        edges.add((int(a.object.ref), int(a.subject.ref), a.qualifiers["edge_type"]))

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
