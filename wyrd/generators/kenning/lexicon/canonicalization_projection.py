"""Project the L2 canonicalization assertions into the L3 collapse graph (wyrd-u6fn.3, D50.6).

The cleaned collapse graph is a REBUILDABLE L3 projection of the append-only L2
assertion streams (``data/mining/canonicalization/``). This pass reads the live
assertion set (``effective_assertions`` — retractions already resolved) and
materializes the synthetic-node identity layer (D50.1): it mints the
``canonical_morpheme`` / ``canonical_place`` / ``canonical_sense`` hubs, binds each
observation row to its hub (``etymon.canonical_morpheme_id`` etc.), reconciles
minted hubs via ``merge-canonical`` (``merged_into`` self-FK, D22 smallest-id-wins
+ chain-flatten), and lays down per-stratum ``canonical_label`` rows.

ADDITIVE, NOT a reader-cutover: this pass POPULATES the ``canonical_*`` tables +
``canonical_*_id`` columns — including folding today's deterministic IDENTITY
clustering (``merged_into_id``/``lemma_id``; see ``_legacy_identity_assertions``,
wyrd-u6fn.4) so the graph reproduces it. It does NOT yet change the existing
``merged_into_id`` / ``cognate_id`` / ``lemma_id`` READERS (era_reflex, the bundle
export, the ``*_canonical`` views); migrating those onto the canonical FK join is
the separate reader-cutover ticket (wyrd-b2mf). Until then the canonical graph is
built alongside the legacy columns, regression-free by construction.

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
  - The relational rollup VIEWS that replace the ``COALESCE`` chains + the cognate
    rollup that re-derives ``cognate_id`` from ``etymon_descent`` — wyrd-b2mf
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
    resolve_confidence_gate,
    score_confidence,
)
from wyrd.generators.kenning.lexicon.bundle._export import _build_family_rollup
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.etymon_refs import etymon_ref, resolve_etymon_ref

# Legacy deterministic clustering (merged_into_id OCR variants + lemma_id
# inflections) is folded into in-memory assertions stamped with this method, so
# the canonical graph reproduces today's identity clustering WITHOUT a committed
# snapshot (it re-derives free on rebuild from the columns the deterministic
# passes already produce). cognate_id is NOT folded — it is relational (D50.2),
# re-derived from etymon_descent by the rollup (the reader-cutover ticket).
METHOD_LEGACY = "legacy-import"

# Leave-separate confidence gate (D46 asymmetry): a missed merge is harmless
# duplication, a wrong merge is corruption — so binds apply only at/above the gate.

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


class _EtymonInfo(NamedTuple):
    """The per-etymon facts the legacy fold needs: surface for the node key, plus
    the two flags that pick the bind kind. ``is_ocr_variant`` (merged_into_id set)
    and ``is_inflection`` (lemma_id points at ANOTHER etymon) — a member that is
    neither (e.g. joined via a curated reflex-link inheritance edge, a cross-era
    SAME morpheme per wyrd-rogd.9) is same-morpheme, not an inflection."""

    language: str
    canonical_form: str
    is_ocr_variant: bool
    is_inflection: bool


class FidelitySample(NamedTuple):
    """One legacy family whose members did NOT all land on one canonical group."""

    legacy_root: int
    members: list[int]
    canonical_roots: list[str]


class FidelityReport(NamedTuple):
    """assess_identity_fidelity result: every legacy multi-member family must map
    to exactly one canonical group (canonical may JOIN families, never SPLIT)."""

    legacy_families: int
    faithful: int
    violations: int
    sample_violations: list[FidelitySample]


def _bind_key(a: Assertion) -> _BindKey:
    return (a.subject.type, a.subject.ref, a.object.ref, a.qualifiers.get("kind"))


# Identity and relational projections score + gate confidence via the SAME shared
# helpers (canonicalization.assertions) so the two can never drift apart silently.
_score = score_confidence
_resolve_gate = resolve_confidence_gate


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
        # The subject ref shape is type-dependent (etymon = "language:canonical_form"
        # natural key, wyrd-c6wu; toponym/te = id for now) — validation/resolution is
        # deferred to the type-aware _resolve_bind_subject at the write boundary.
        grouped[(a.subject.type, a.subject.ref)].append(a)
    return grouped


def _resolve_bind_subject(conn, subj_type: str, subj_ref: str) -> int | None:
    """Resolve a bind subject ref to its observation row id, or None (skip + flag).

    Etymon refs are the stable natural key ``"language:canonical_form"`` (wyrd-c6wu /
    D50.1). ``toponym`` / ``toponym_etymology`` are still id-keyed — no natural-key
    emitter binds them yet (toponym is future place-canon; te is blocked on the
    dedup task), so the consumer keeps accepting an integer ref for those types.
    ``isascii()`` guards the ``str.isdigit()``/``int()`` gap (Unicode numerics).
    """
    if subj_type == "etymon":
        return resolve_etymon_ref(conn, subj_ref)
    if subj_ref.isascii() and subj_ref.isdigit():
        return int(subj_ref)
    return None


def _resolve_binds(
    binds: list[Assertion],
    minted: dict[str, set[str]],
    root_of: dict[str, str],
    gate: float,
    result: ProjectionResult,
    conn,
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
        subj_id = _resolve_bind_subject(conn, subj_type, subj_ref)
        if subj_id is None:
            result.warnings.append(
                f"bind {subj_type}:{subj_ref} did not resolve to an observation in this build; skipped"
            )
            continue
        writes.append(BindWrite(table, column, winner.object.ref, subj_id))
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


def _legacy_identity_assertions(db: LexiconDB, result: ProjectionResult) -> list[Assertion]:
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
    needed = sorted(set(multi) | {m for members in multi.values() for m in members})
    # Fetch only the family members by PK in chunks (SQLite param-limit-safe),
    # not a full 2.4M-row scan — this pass runs on every rebuild, and the rest of
    # the fold already pays one full scan inside _build_family_rollup.
    info: dict[int, _EtymonInfo] = {}
    for i in range(0, len(needed), 900):
        chunk = needed[i : i + 900]
        qmarks = ",".join("?" * len(chunk))
        for r in db.conn.execute(
            f"SELECT id, language, canonical_form, merged_into_id, lemma_id "  # noqa: S608 — ? placeholders
            f"FROM etymon WHERE id IN ({qmarks})",
            chunk,
        ):
            info[r["id"]] = _EtymonInfo(
                r["language"] or "",
                r["canonical_form"] or "",
                r["merged_into_id"] is not None,
                r["lemma_id"] is not None and r["lemma_id"] != r["id"],
            )

    out: list[Assertion] = []
    for root in sorted(multi):  # sorted -> deterministic warning order (D24 diagnostics)
        members = multi[root]
        ri = info.get(root)
        if ri is None or not ri.canonical_form:
            # No stable surface to content-key the node on (canonical_form is
            # NOT NULL in schema, so this is a pathological empty-string row).
            # Skip — but observably (D24), since it leaves the family unbound.
            result.warnings.append(
                f"legacy family root {root} has no canonical_form; {len(members)} members left unbound"
            )
            continue
        node = NodeRef(
            "canonical_morpheme",
            mint_canonical_id("canonical_morpheme", ri.language, ri.canonical_form),
        )
        out.append(
            Assertion(
                predicate="mint-canonical",
                subject=node,
                confidence="high",
                method=METHOD_LEGACY,
                rationale=f"legacy identity rollup of '{ri.canonical_form}' (merged_into_id/lemma_id)",
            ).with_id()
        )
        for m in members:
            # inflection-of ONLY for a genuine inflection (lemma_id -> another
            # etymon, and not also an OCR variant). The root, OCR variants, and
            # reflex-link members (cross-era SAME morpheme, wyrd-rogd.9) are all
            # same-morpheme.
            mi = info.get(m, _EtymonInfo("", "", False, False))
            kind = (
                "inflection-of" if (mi.is_inflection and not mi.is_ocr_variant) else "same-morpheme"
            )
            out.append(
                Assertion(
                    predicate="bind",
                    # wyrd-c6wu: stable natural-key ref (mi carries language + form).
                    subject=NodeRef("etymon", etymon_ref(mi.language, mi.canonical_form)),
                    object=node,
                    qualifiers={"kind": kind},
                    confidence="high",
                    method=METHOD_LEGACY,
                    rationale=f"legacy {kind} import (rollup root '{ri.canonical_form}')",
                ).with_id()
            )
    return out


def assess_identity_fidelity(db: LexiconDB) -> FidelityReport:
    """Compare the projected ``canonical_morpheme_id`` partition against the legacy
    ``COALESCE(merged_into_id, lemma_id, id)`` rollup. Each legacy multi-member
    family must map to exactly one canonical group (rolled through ``merged_into``)
    — the canonical partition may JOIN legacy families via logged merges, but must
    never SPLIT one. Returns the gate counts + up to 10 sampled violations."""
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
    samples: list[FidelitySample] = []
    for root in sorted(legacy_families):  # sorted -> deterministic sampled violations
        members = legacy_families[root]
        roots = {canon.get(m) for m in members}
        if len(roots) == 1 and None not in roots:
            faithful += 1  # all members on one canonical group => faithful
        elif len(samples) < 10:  # a split (or an unbound member) — sample the first 10
            samples.append(
                FidelitySample(root, sorted(members)[:5], sorted(str(x) for x in roots)[:5])
            )
    return FidelityReport(len(legacy_families), faithful, len(legacy_families) - faithful, samples)


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
    result = ProjectionResult(applied=apply)
    assertions = list(effective_assertions(list(load_assertions(mining_dir))))
    if fold_legacy:
        assertions += _legacy_identity_assertions(db, result)
    assertions.sort(key=lambda a: a.id)
    by_pred: dict[str, list[Assertion]] = defaultdict(list)
    for a in assertions:
        by_pred[a.predicate].append(a)

    minted = _collect_minted(by_pred)
    result.minted = {t: len(ids) for t, ids in minted.items() if ids}
    root_of = _merge_roots(
        [a for a in by_pred.get("merge-canonical", ()) if a.polarity == "affirm"], minted, result
    )
    result.merged = len(root_of)
    bind_writes = _resolve_binds(by_pred.get("bind", []), minted, root_of, gate, result, db.conn)
    label_pick = _pick_labels(by_pred, minted, result)
    result.labels = len(label_pick)

    if apply:
        _write_projection(db, minted, root_of, bind_writes, label_pick)
    return result
