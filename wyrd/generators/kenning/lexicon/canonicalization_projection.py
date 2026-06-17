"""Project the L2 canonicalization assertions into the L3 collapse graph (wyrd-u6fn.3, D50.6).

The cleaned collapse graph is a REBUILDABLE L3 projection of the append-only L2
assertion streams (``data/mining/canonicalization/``). This pass reads the live
assertion set (``effective_assertions`` — retractions already resolved) and
materializes the synthetic-node identity layer (D50.1): it mints the
``canonical_morpheme`` / ``canonical_place`` / ``canonical_sense`` hubs, binds each
observation row to its hub (``etymon.canonical_morpheme_id`` etc.), reconciles
minted hubs via ``merge-canonical`` (``merged_into`` self-FK, D22 smallest-id-wins
+ chain-flatten), and lays down per-stratum ``canonical_label`` rows.

ADDITIVE, NOT a cutover (the u6fn.3/u6fn.4 split): this pass only POPULATES the
``canonical_*`` tables + ``canonical_*_id`` columns. It does NOT touch any of the
existing ``merged_into_id`` / ``cognate_id`` / ``lemma_id`` readers (era_reflex,
the bundle export, the ``*_canonical`` views). Migrating those readers onto the
canonical FK join — and authoring legacy-import binds so the projection reproduces
today's clustering — is wyrd-u6fn.4. Until then the canonical graph is built
alongside the legacy columns, regression-free by construction.

Determinism (D36.9): assertions are processed sorted by their stable id, so the
same L2 streams yield a byte-identical L3 graph. Idempotent + reversible: an
``apply`` run clears the canonical graph (the ``clear-enrichment --stage=canonical``
logic) before rebuilding, so re-running reflects the current assertion set exactly
and a retracted/refuted bind cleanly disappears.

Scope (deferred, by design):
  - ``same-sense`` binds (``etymon_gloss``, composite PK) — no ref convention is
    established yet (no author emits them); flagged-skipped (observable, D24), not
    guessed. Turned on when the sense-binding work defines the ref.
  - Family-C flags (``canonical-decomposition`` / ``decomposition-spurious``),
    ``composed-of`` (wyrd-h5u1), and the region edges (``contains`` / ``succeeds``
    / ``located-in``, which need new tables) — separate tickets, no authored data.
  - The relational rollup VIEWS that replace the ``COALESCE`` chains — wyrd-u6fn.4
    (the reader cutover).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from wyrd.generators.kenning.canonicalization import (
    Assertion,
    effective_assertions,
    load_assertions,
)
from wyrd.generators.kenning.lexicon.db import LexiconDB

# Leave-separate confidence gate (D46 asymmetry): a missed merge is harmless
# duplication, a wrong merge is corruption — so binds apply only at/above the gate.
# Labels map to a score; a bare float in [0,1] compares directly. Default 'high';
# env-tunable later per the D43 threshold precedent.
_LABEL_SCORE = {"low": 0.3, "medium": 0.6, "high": 0.9}

# Observation subject type -> (table, canonical FK column, minted-node type). Only
# the kinds with an established integer-id ref convention (Newton fixture) + an L3
# target column. 'same-place' spans toponym (default) + toponym_etymology
# (override) — the override-beats-default precedence is a READ-time rule (u6fn.4),
# so the projection just fills both columns from their own binds.
_SUBJECT_TARGET: dict[str, tuple[str, str, str]] = {
    "etymon": ("etymon", "canonical_morpheme_id", "canonical_morpheme"),
    "toponym": ("toponym", "canonical_place_id", "canonical_place"),
    "toponym_etymology": ("toponym_etymology", "canonical_place_id", "canonical_place"),
}

# Minted-node type -> its table (for mint + merge-canonical).
_NODE_TABLES = ("canonical_morpheme", "canonical_place", "canonical_sense")


def _score(confidence: str | float) -> float:
    """A comparable confidence score: labels via the table, floats as-is."""
    if isinstance(confidence, str):
        return _LABEL_SCORE.get(confidence, 0.0)
    return float(confidence)


@dataclass
class ProjectionResult:
    """What the projection did (or would do, in dry-run). Counts + the observable
    list of conflicts/skips (D24) so nothing is resolved silently."""

    applied: bool = False
    minted: dict[str, int] = field(default_factory=dict)  # node type -> count
    bound: dict[str, int] = field(default_factory=dict)  # subject type -> count
    merged: int = 0
    labels: int = 0
    # An observation bound above-gate to >1 distinct canonical (post-merge): left
    # UNBOUND + flagged (D46 conflict rule). (subject_type, ref, [node ids]).
    conflicts: list[tuple[str, str, list[str]]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # bind.kind values seen but not yet projectable (e.g. same-sense): count by kind.
    skipped_unsupported: dict[str, int] = field(default_factory=dict)


def clear_canonical_graph(db: LexiconDB) -> None:
    """Drop the projected canonical graph: NULL the four binding columns, then the
    labels, then the synthetic nodes. The ``clear-enrichment --stage=canonical``
    body, shared so the projection is a clean rebuild (idempotent)."""
    db.conn.execute(
        "UPDATE etymon SET canonical_morpheme_id = NULL WHERE canonical_morpheme_id IS NOT NULL"
    )
    db.conn.execute(
        "UPDATE etymon_gloss SET canonical_sense_id = NULL WHERE canonical_sense_id IS NOT NULL"
    )
    db.conn.execute(
        "UPDATE toponym SET canonical_place_id = NULL WHERE canonical_place_id IS NOT NULL"
    )
    db.conn.execute(
        "UPDATE toponym_etymology SET canonical_place_id = NULL WHERE canonical_place_id IS NOT NULL"
    )
    db.conn.execute("DELETE FROM canonical_label")
    for table in _NODE_TABLES:
        db.conn.execute(f"DELETE FROM {table}")  # noqa: S608 — table from a fixed tuple, not input


def _collect_minted(by_pred: dict[str, list[Assertion]]) -> dict[str, set[str]]:
    """Affirmed canonical-node declarations, by node type."""
    minted: dict[str, set[str]] = {t: set() for t in _NODE_TABLES}
    for a in by_pred.get("mint-canonical", ()):
        if a.polarity == "affirm" and a.subject.type in minted:
            minted[a.subject.type].add(a.subject.ref)
    return minted


def _merge_roots(
    merges: list[Assertion], minted: dict[str, set[str]], result: ProjectionResult
) -> dict[str, str]:
    """Union the ``merge-canonical`` pairs and return node id -> root id
    (lexicographic-min per component, D22 smallest-id-wins). Chain-flatten is
    automatic: every node in a component points at the single root."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:  # path-compress
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        lo, hi = (ra, rb) if ra < rb else (rb, ra)  # smallest id is the root
        parent[hi] = lo

    all_minted = {nid for ids in minted.values() for nid in ids}
    for a in merges:
        s, o = a.subject.ref, a.object.ref
        if s not in all_minted or o not in all_minted:
            result.warnings.append(f"merge-canonical references unminted node ({s} / {o}); skipped")
            continue
        parent.setdefault(s, s)
        parent.setdefault(o, o)
        union(s, o)
    return {node: find(node) for node in parent if find(node) != node}


def _refuted_binds(binds: list[Assertion]) -> set[tuple]:
    """(subject, object, kind) tuples carrying a refute — these binds don't apply."""
    return {
        (a.subject.type, a.subject.ref, a.object.ref, a.qualifiers.get("kind"))
        for a in binds
        if a.polarity == "refute" and a.object is not None
    }


def _group_binds(
    binds: list[Assertion], gate: float, result: ProjectionResult
) -> dict[tuple[str, str], list[Assertion]]:
    """Affirmed, non-refuted, above-gate binds grouped by observation subject.
    Records unsupported kinds (e.g. same-sense) as observable skips."""
    refuted = _refuted_binds(binds)
    grouped: dict[tuple[str, str], list[Assertion]] = defaultdict(list)
    for a in binds:
        if a.polarity != "affirm" or a.object is None:
            continue
        kind = a.qualifiers.get("kind")
        if a.subject.type not in _SUBJECT_TARGET:
            key = kind or a.subject.type
            result.skipped_unsupported[key] = result.skipped_unsupported.get(key, 0) + 1
            continue
        if (a.subject.type, a.subject.ref, a.object.ref, kind) in refuted:
            continue
        if _score(a.confidence) < gate:
            continue
        grouped[(a.subject.type, a.subject.ref)].append(a)
    return grouped


def _resolve_binds(
    binds: list[Assertion],
    minted: dict[str, set[str]],
    root_of: dict[str, str],
    gate: float,
    result: ProjectionResult,
) -> list[tuple[str, str, str, str]]:
    """Resolve each observation's binds to a single hub, flagging conflicts (D46).
    Returns (table, column, node_id, obs_ref) writes."""
    writes: list[tuple[str, str, str, str]] = []
    bound: dict[str, int] = defaultdict(int)
    for (subj_type, subj_ref), group in _group_binds(binds, gate, result).items():
        table, column, node_type = _SUBJECT_TARGET[subj_type]
        valid = [b for b in group if b.object.ref in minted.get(node_type, set())]
        for b in group:
            if b.object.ref not in minted.get(node_type, set()):
                result.warnings.append(
                    f"bind {subj_type}:{subj_ref} -> unminted {b.object.ref}; skipped"
                )
        if not valid:
            continue
        roots = {root_of.get(b.object.ref, b.object.ref) for b in valid}
        if len(roots) > 1:
            # Bound above-gate to >1 distinct identity: leave UNBOUND + flag (D46).
            result.conflicts.append((subj_type, subj_ref, sorted(roots)))
            continue
        # Deterministic pick among equivalent binds: highest score, tie lowest id.
        winner = min(valid, key=lambda b: (-_score(b.confidence), b.id))
        writes.append((table, column, winner.object.ref, subj_ref))
        bound[subj_type] += 1
    result.bound = dict(bound)
    return writes


def _pick_labels(by_pred: dict[str, list[Assertion]]) -> dict[tuple[str, str, str], Assertion]:
    """Highest-confidence canonical-label per (node, stratum); tie -> lowest id."""
    pick: dict[tuple[str, str, str], Assertion] = {}
    for a in by_pred.get("canonical-label", ()):
        if a.polarity != "affirm":
            continue
        key = (a.subject.type, a.subject.ref, a.qualifiers.get("stratum", ""))
        cur = pick.get(key)
        if (
            cur is None
            or _score(a.confidence) > _score(cur.confidence)
            or (_score(a.confidence) == _score(cur.confidence) and a.id < cur.id)
        ):
            pick[key] = a
    return pick


def _node_type_of(node_id: str, minted: dict[str, set[str]]) -> str | None:
    for node_type, ids in minted.items():
        if node_id in ids:
            return node_type
    return None


def _write_projection(
    db: LexiconDB,
    minted: dict[str, set[str]],
    root_of: dict[str, str],
    bind_writes: list[tuple[str, str, str, str]],
    label_pick: dict[tuple[str, str, str], Assertion],
) -> None:
    """Clear + rebuild the canonical graph from the resolved projection."""
    clear_canonical_graph(db)
    for node_type, ids in minted.items():
        for nid in sorted(ids):
            db.conn.execute(
                f"INSERT INTO {node_type} (id, merged_into) VALUES (?, NULL)",  # noqa: S608 — fixed tuple
                (nid,),
            )
    for node, root in root_of.items():
        node_type = _node_type_of(node, minted)
        if node_type:
            db.conn.execute(
                f"UPDATE {node_type} SET merged_into = ? WHERE id = ?",  # noqa: S608 — fixed tuple
                (root, node),
            )
    for table, column, node_id, obs_ref in bind_writes:
        db.conn.execute(
            f"UPDATE {table} SET {column} = ? WHERE id = ?",  # noqa: S608 — table/column from a fixed map
            (node_id, int(obs_ref)),
        )
    for (canonical_type, canonical_id, stratum), a in label_pick.items():
        db.conn.execute(
            "INSERT INTO canonical_label "
            "(canonical_type, canonical_id, stratum, value, source_assertion) "
            "VALUES (?, ?, ?, ?, ?)",
            (canonical_type, canonical_id, stratum, a.qualifiers.get("value", ""), a.id),
        )
    db.commit()


def project_canonical(
    db: LexiconDB, *, mining_dir, apply: bool = False, confidence_gate: str | float = "high"
) -> ProjectionResult:
    """Project the L2 canonicalization assertions into the L3 collapse graph.

    ``apply=False`` (default) reports what would change without writing. ``apply``
    clears the existing canonical graph then rebuilds it from the live assertion
    set, so the result is a deterministic, idempotent function of the L2 streams.
    """
    gate = _score(confidence_gate)
    assertions = sorted(effective_assertions(list(load_assertions(mining_dir))), key=lambda a: a.id)
    by_pred: dict[str, list[Assertion]] = defaultdict(list)
    for a in assertions:
        by_pred[a.predicate].append(a)

    result = ProjectionResult(applied=apply)
    minted = _collect_minted(by_pred)
    result.minted = {t: len(ids) for t, ids in minted.items() if ids}
    root_of = _merge_roots(
        [a for a in by_pred.get("merge-canonical", ()) if a.polarity == "affirm"], minted, result
    )
    result.merged = len(root_of)
    bind_writes = _resolve_binds(by_pred.get("bind", []), minted, root_of, gate, result)
    label_pick = _pick_labels(by_pred)
    result.labels = len(label_pick)

    if apply:
        _write_projection(db, minted, root_of, bind_writes, label_pick)
    return result
