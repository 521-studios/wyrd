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

Determinism (D36.9): assertions are processed sorted by their stable content-hash
id, so the same L2 streams yield a byte-identical L3 graph. (The id is a 64-bit
content hash; a collision between two distinct assertions is astronomically
unlikely — were it ever to happen, the sort would fall back to append order for
that pair.) Idempotent + reversible: an ``apply`` run clears the canonical graph
(the ``clear-enrichment --stage=canonical`` logic) before rebuilding, inside one
transaction (rolled back on any error), so re-running reflects the current
assertion set exactly and a retracted/refuted bind cleanly disappears.

Robustness: every observation/node a bind or label references is validated before
the destructive clear — non-integer observation refs, binds/labels to unminted
nodes, and cross-type merges are skipped with an observable warning (D24), never
fatal. The leave-separate confidence gate is validated at entry.

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
from pathlib import Path
from typing import NamedTuple

from wyrd.generators.kenning.canonicalization import (
    Assertion,
    NodeRef,
    effective_assertions,
    load_assertions,
    mint_canonical_id,
)
from wyrd.generators.kenning.lexicon.bundle._export import _build_family_rollup
from wyrd.generators.kenning.lexicon.db import LexiconDB

# Legacy deterministic clustering (merged_into_id OCR variants + lemma_id
# inflections) is folded into in-memory assertions stamped with this method, so
# the canonical graph reproduces today's identity clustering WITHOUT a committed
# snapshot (it re-derives free on rebuild from the columns the deterministic
# passes already produce). cognate_id is NOT folded — it is relational (D50.2),
# re-derived from etymon_descent by the rollup (the reader-cutover ticket).
METHOD_LEGACY = "legacy-import"

# Leave-separate confidence gate (D46 asymmetry): a missed merge is harmless
# duplication, a wrong merge is corruption — so binds apply only at/above the gate.
# Labels map to a score; a bare float in [0,1] compares directly. Default 'high';
# tunable per environment.
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

# The identity of a bind for affirm/refute matching: (subject type, subject ref,
# object ref, kind). Built in one place so the refute set and the affirm lookup
# can't drift out of shape.
_BindKey = tuple[str, str, str, str | None]


class BindWrite(NamedTuple):
    """One resolved bind to apply: set ``<table>.<column> = node_id`` for the
    observation row ``obs_id``. (Named so a field transposition is a type error.)"""

    table: str
    column: str
    node_id: str
    obs_id: int


class Conflict(NamedTuple):
    """An observation bound above-gate to >1 distinct canonical (post-merge),
    left UNBOUND + reported (D46 conflict rule, D24 observability)."""

    subject_type: str
    ref: str
    nodes: list[str]


def _bind_key(a: Assertion) -> _BindKey:
    return (a.subject.type, a.subject.ref, a.object.ref, a.qualifiers.get("kind"))


def _score(confidence: str | float) -> float:
    """A comparable confidence score: labels via the table, floats as-is."""
    if isinstance(confidence, str):
        return _LABEL_SCORE.get(confidence, 0.0)
    return float(confidence)


def _resolve_gate(confidence_gate: str | float) -> float:
    """Validate + score the gate. A typo'd label or out-of-range float would
    otherwise silently collapse the leave-separate gate, so reject it loudly."""
    if isinstance(confidence_gate, str):
        if confidence_gate in _LABEL_SCORE:
            return _LABEL_SCORE[confidence_gate]
        # A CLI option arrives as a string, so a float gate like "0.8" comes
        # through as text — parse it before rejecting as an unknown label.
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
class ProjectionResult:
    """What the projection did (or would do, in dry-run). Counts + the observable
    lists of conflicts/warnings/skips (D24) so nothing is resolved silently."""

    applied: bool = False
    minted: dict[str, int] = field(default_factory=dict)  # node type -> count
    bound: dict[str, int] = field(default_factory=dict)  # subject type -> count
    merged: int = 0
    labels: int = 0
    conflicts: list[Conflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # bind.kind values seen but not yet projectable (e.g. same-sense): count by kind.
    skipped_unsupported: dict[str, int] = field(default_factory=dict)


def _node_types(minted: dict[str, set[str]]) -> dict[str, str]:
    """node id -> its minted node type (canonical_morpheme/place/sense)."""
    return {nid: node_type for node_type, ids in minted.items() for nid in ids}


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
    automatic. A merge to an unminted node, or one that crosses node types (a
    morpheme hub into a place hub), is skipped with a warning."""
    node_type = _node_types(minted)
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

    for a in merges:
        s, o = a.subject.ref, a.object.ref
        st, ot = node_type.get(s), node_type.get(o)
        if st is None or ot is None:
            result.warnings.append(f"merge-canonical references unminted node ({s} / {o}); skipped")
            continue
        if st != ot:
            result.warnings.append(
                f"merge-canonical crosses node types ({st}:{s} / {ot}:{o}); skipped"
            )
            continue
        parent.setdefault(s, s)
        parent.setdefault(o, o)
        union(s, o)
    return {node: find(node) for node in parent if find(node) != node}


def _group_binds(
    binds: list[Assertion], gate: float, result: ProjectionResult
) -> dict[tuple[str, str], list[Assertion]]:
    """Affirmed, non-refuted, above-gate binds grouped by observation subject.
    Records unsupported kinds (e.g. same-sense) and non-integer observation refs as
    observable skips — validated HERE, before the destructive write, so a bad ref
    can never abort the projection mid-rebuild."""
    refuted = {_bind_key(a) for a in binds if a.polarity == "refute" and a.object is not None}
    grouped: dict[tuple[str, str], list[Assertion]] = defaultdict(list)
    for a in binds:
        if a.polarity != "affirm" or a.object is None:
            continue
        kind = a.qualifiers.get("kind")
        if a.subject.type not in _SUBJECT_TARGET:
            key = kind or a.subject.type
            result.skipped_unsupported[key] = result.skipped_unsupported.get(key, 0) + 1
            continue
        if _bind_key(a) in refuted:
            continue
        if _score(a.confidence) < gate:
            continue
        # isascii() guards the gap between str.isdigit() and int(): Unicode
        # numerics like '²' / Devanagari digits are isdigit()==True but raise in
        # int(), which would defeat this skip-don't-crash guard.
        if not (a.subject.ref.isascii() and a.subject.ref.isdigit()):
            result.warnings.append(
                f"bind {a.subject.type}:{a.subject.ref} has a non-integer observation ref; skipped"
            )
            continue
        grouped[(a.subject.type, a.subject.ref)].append(a)
    return grouped


def _resolve_binds(
    binds: list[Assertion],
    minted: dict[str, set[str]],
    root_of: dict[str, str],
    gate: float,
    result: ProjectionResult,
) -> list[BindWrite]:
    """Resolve each observation's binds to a single hub, flagging conflicts (D46)."""
    writes: list[BindWrite] = []
    bound: dict[str, int] = defaultdict(int)
    for (subj_type, subj_ref), group in _group_binds(binds, gate, result).items():
        table, column, node_type = _SUBJECT_TARGET[subj_type]
        minted_nodes = minted.get(node_type, set())
        valid: list[Assertion] = []
        for b in group:
            if b.object.ref in minted_nodes:
                valid.append(b)
            else:
                result.warnings.append(
                    f"bind {subj_type}:{subj_ref} -> unminted {b.object.ref}; skipped"
                )
        if not valid:
            continue
        roots = {root_of.get(b.object.ref, b.object.ref) for b in valid}
        if len(roots) > 1:
            # Bound above-gate to >1 distinct identity: leave UNBOUND + flag (D46).
            result.conflicts.append(Conflict(subj_type, subj_ref, sorted(roots)))
            continue
        # Deterministic pick among equivalent binds: highest score, tie lowest id.
        winner = min(valid, key=lambda b: (-_score(b.confidence), b.id))
        writes.append(BindWrite(table, column, winner.object.ref, int(subj_ref)))
        bound[subj_type] += 1
    result.bound = dict(bound)
    return writes


def _pick_labels(
    by_pred: dict[str, list[Assertion]], minted: dict[str, set[str]], result: ProjectionResult
) -> dict[tuple[str, str, str], Assertion]:
    """Highest-confidence canonical-label per (node, stratum); tie -> lowest id.
    A label for an unminted node is skipped with a warning (it has no hub to sit
    on, and ``canonical_label`` carries no FK to enforce it)."""
    pick: dict[tuple[str, str, str], Assertion] = {}
    for a in by_pred.get("canonical-label", ()):
        if a.polarity != "affirm":
            continue
        if a.subject.ref not in minted.get(a.subject.type, set()):
            result.warnings.append(
                f"canonical-label for unminted {a.subject.type}:{a.subject.ref}; skipped"
            )
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


def _write_projection(
    db: LexiconDB,
    minted: dict[str, set[str]],
    root_of: dict[str, str],
    bind_writes: list[BindWrite],
    label_pick: dict[tuple[str, str, str], Assertion],
) -> None:
    """Clear + rebuild the canonical graph in one transaction (rolled back on any
    error, so a failure never leaves a half-projected graph)."""
    node_type = _node_types(minted)
    try:
        clear_canonical_graph(db)
        for nt, ids in minted.items():
            for nid in sorted(ids):
                db.conn.execute(
                    f"INSERT INTO {nt} (id, merged_into) VALUES (?, NULL)",  # noqa: S608 — fixed tuple
                    (nid,),
                )
        for node, root in root_of.items():
            nt = node_type.get(node)
            if nt:
                db.conn.execute(
                    f"UPDATE {nt} SET merged_into = ? WHERE id = ?",  # noqa: S608 — fixed tuple
                    (root, node),
                )
        for w in bind_writes:
            db.conn.execute(
                f"UPDATE {w.table} SET {w.column} = ? WHERE id = ?",  # noqa: S608 — from a fixed map
                (w.node_id, w.obs_id),
            )
        for (canonical_type, canonical_id, stratum), a in label_pick.items():
            db.conn.execute(
                "INSERT INTO canonical_label "
                "(canonical_type, canonical_id, stratum, value, source_assertion) "
                "VALUES (?, ?, ?, ?, ?)",
                (canonical_type, canonical_id, stratum, a.qualifiers.get("value", ""), a.id),
            )
        db.commit()
    except Exception:
        db.conn.rollback()
        raise


def _legacy_identity_assertions(db: LexiconDB) -> list[Assertion]:
    """Fold today's deterministic IDENTITY clustering (merged_into_id OCR variants
    + lemma_id inflections) into in-memory canonical assertions, so the projection
    reproduces it with no committed snapshot — it re-derives free on rebuild from
    the columns the deterministic passes already produce.

    Reuses the same two-step rollup (``_build_family_rollup``) the 1a miner used, so
    legacy node ids content-key on the family root's ``(language, canonical_form)``
    and compose cleanly with authored binds (same node id => same hub, not a
    conflict). Only multi-member families produce assertions; a singleton is its
    own identity (``canonical_morpheme_id`` stays NULL, exactly like
    ``merged_into_id IS NULL`` today). Deterministic: ids are content hashes.

    cognate_id is NOT folded here — it is relational (D50.2: a cognate cluster is
    the closure of descent edges, not one identity), re-derived from
    ``etymon_descent`` by the rollup view in the reader-cutover ticket."""
    members_by_root, _root_of = _build_family_rollup(db)
    multi = {r: members for r, members in members_by_root.items() if len(members) > 1}
    if not multi:
        return []
    needed = set(multi) | {m for members in multi.values() for m in members}
    # id -> (language, canonical_form, is_ocr_variant). The OCR flag picks the
    # bind kind: an OCR variant (merged_into_id set) is the SAME morpheme; a member
    # in the family by lemma_id only is an inflection-of.
    info: dict[int, tuple[str, str, bool]] = {}
    for r in db.conn.execute("SELECT id, language, canonical_form, merged_into_id FROM etymon"):
        if r["id"] in needed:
            info[r["id"]] = (
                r["language"] or "",
                r["canonical_form"] or "",
                r["merged_into_id"] is not None,
            )

    out: list[Assertion] = []
    for root, members in multi.items():
        lang, form, _ = info.get(root, ("", "", False))
        if not form:
            continue  # no stable surface to content-key the node on
        node = NodeRef("canonical_morpheme", mint_canonical_id("canonical_morpheme", lang, form))
        out.append(
            Assertion(
                predicate="mint-canonical",
                subject=node,
                confidence="high",
                method=METHOD_LEGACY,
                rationale=f"legacy identity rollup of '{form}' (merged_into_id/lemma_id)",
            ).with_id()
        )
        for m in members:
            kind = (
                "same-morpheme"
                if (m == root or info.get(m, ("", "", False))[2])
                else "inflection-of"
            )
            out.append(
                Assertion(
                    predicate="bind",
                    subject=NodeRef("etymon", str(m)),
                    object=node,
                    qualifiers={"kind": kind},
                    confidence="high",
                    method=METHOD_LEGACY,
                    rationale=f"legacy {kind} import (rollup root '{form}')",
                ).with_id()
            )
    return out


def assess_identity_fidelity(db: LexiconDB) -> dict:
    """Compare the projected ``canonical_morpheme_id`` partition against the legacy
    ``COALESCE(merged_into_id, lemma_id, id)`` rollup. Each legacy multi-member
    family must map to exactly one canonical group (rolled through ``merged_into``)
    — the canonical partition may JOIN legacy families via logged merges, but must
    never SPLIT one. Returns the gate counts + a sample of any violations."""
    members_by_root, _root_of = _build_family_rollup(db)
    legacy_families = {r: ms for r, ms in members_by_root.items() if len(ms) > 1}

    merged_into = dict(db.conn.execute("SELECT id, merged_into FROM canonical_morpheme").fetchall())

    def canonical_root(node_id: str | None) -> str | None:
        seen: set[str] = set()
        while node_id is not None and merged_into.get(node_id) and node_id not in seen:
            seen.add(node_id)
            node_id = merged_into[node_id]
        return node_id

    canon: dict[int, str | None] = {}
    for r in db.conn.execute(
        "SELECT id, canonical_morpheme_id FROM etymon WHERE canonical_morpheme_id IS NOT NULL"
    ):
        canon[r["id"]] = canonical_root(r["canonical_morpheme_id"])

    faithful = 0
    violations: list[tuple[int, list[int], list[str]]] = []
    for root, members in legacy_families.items():
        roots = {canon.get(m) for m in members}
        if len(roots) == 1 and None not in roots:
            faithful += 1
        elif len(violations) < 10:
            violations.append((root, sorted(members)[:5], sorted(str(x) for x in roots)[:5]))
        elif None in roots or len(roots) > 1:
            violations.append((-1, [], []))  # counted, not sampled
    return {
        "legacy_families": len(legacy_families),
        "faithful": faithful,
        "violations": len(legacy_families) - faithful,
        "sample_violations": [v for v in violations if v[0] != -1],
    }


def project_canonical(
    db: LexiconDB,
    *,
    mining_dir: Path,
    apply: bool = False,
    confidence_gate: str | float = "high",
    fold_legacy: bool = True,
) -> ProjectionResult:
    """Project the L2 canonicalization assertions into the L3 collapse graph.

    ``apply=False`` (default) reports what would change without writing. ``apply``
    clears the existing canonical graph then rebuilds it from the live assertion
    set, so the result is a deterministic, idempotent function of the L2 streams.

    ``fold_legacy`` (default True) additionally folds today's deterministic
    identity clustering (``merged_into_id``/``lemma_id``) into in-memory assertions
    so the canonical graph reproduces it — see ``_legacy_identity_assertions``.
    """
    gate = _resolve_gate(confidence_gate)
    assertions = list(effective_assertions(list(load_assertions(mining_dir))))
    if fold_legacy:
        assertions += _legacy_identity_assertions(db)
    assertions.sort(key=lambda a: a.id)
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
    label_pick = _pick_labels(by_pred, minted, result)
    result.labels = len(label_pick)

    if apply:
        _write_projection(db, minted, root_of, bind_writes, label_pick)
    return result
