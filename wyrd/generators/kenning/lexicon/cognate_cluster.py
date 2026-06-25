"""Cognate-cluster assignment from etymon_descent (D27 / wyrd-81n).

Walk the inheritance + borrowing edges of ``etymon_descent`` from
every root etymon and assign ``cognate_id`` to every reachable
descendant. Each cluster collapses to the deepest reachable root —
so OE ``tūn`` / ON ``tún`` / mod-English ``town`` all share one
``cognate_id`` pointing at Proto-Germanic ``*tūnaz``.

Cross-language by design; distinct from ``merged_into_id``
(within-language OCR cluster) and ``lemma_id`` (within-language
inflection family). The ``etymon_era_reflexes`` 4-tier picker reads
this column to find cluster mates at a target era.

Idempotent + reversible via
``clear_enrichment(stage="cognates")``. Stamps
``etymon.cognate_method`` so a future cluster-cognates-v2 can
selectively rebuild only its predecessor's work.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.db import LexiconDB

_COGNATE_BRIDGING_EDGES = ("inheritance", "borrowing")

# v2: languages whose etymons must NOT bridge a cognate cluster. Proto-Indo-
# European reconstructions sit above the family layer, so inheritance FROM a PIE
# root fans out across every IE branch — Germanic ``down`` AND Greek ``-thymia``
# AND Sanskrit ``dhūmra`` all "inherit" from PIE ``*dʰewh₂-``. Rooting a cluster
# there produced 1000+-member mega-clusters of unrelated words (143 of 155
# clusters >200 members were PIE-rooted), so the era-grid rendered garbage modern
# reflexes. Dropping every edge that TOUCHES a PIE node removes PIE from the
# cluster graph, so each IE family (Proto-Germanic and below) clusters on its own
# — exactly the scope British-toponym cognates need. Family-level protos
# (Proto-Germanic etc.) still bridge, so OE/ON/etc. cognates stay together.
_NON_BRIDGING_LANGUAGES = ("proto-indo-european",)
_CLUSTER_COGNATES_METHOD = "cluster-cognates-v2"


def _assign_cognate_clusters(
    canonical_edges: list[tuple[int, int]],
) -> tuple[dict[int, int], list[int], set[int]]:
    """Assign every reachable node to its cluster root by BFS over the canonical
    bridging edges. Pure graph computation — no DB.

    Returns ``(assignments, roots, cycle_orphans)``:
      - ``assignments``: ``{etymon_id: root_id}`` — every node reachable from a
        root, including the roots themselves (``root_id == etymon_id``).
      - ``roots``: canonical etymons that appear as a parent but never a child,
        sorted ascending so the smallest root id wins a contested node (the BFS
        skips already-assigned nodes, and roots are walked in ascending order).
      - ``cycle_orphans``: nodes that participate in bridging edges but never
        reached a root — a pure cycle has no anchor.
    """
    # Build child-of-parent index for the BFS. Parent → set of children.
    children_by_parent: dict[int, set[int]] = {}
    bridging_participants: set[int] = set()
    parent_set: set[int] = set()
    child_set: set[int] = set()
    for parent_id, child_id in canonical_edges:
        children_by_parent.setdefault(parent_id, set()).add(child_id)
        bridging_participants.add(parent_id)
        bridging_participants.add(child_id)
        parent_set.add(parent_id)
        child_set.add(child_id)

    # Roots: canonical etymons that appear as parent but never as
    # child in the canonical edge set.
    roots = sorted(parent_set - child_set)

    assignments: dict[int, int] = {}
    for root_id in roots:
        if root_id in assignments:
            # Already claimed by an earlier (smaller-id) root via cross-edges.
            continue
        assignments[root_id] = root_id
        frontier: list[int] = [root_id]
        while frontier:
            next_frontier: list[int] = []
            for node_id in frontier:
                for child_id in children_by_parent.get(node_id, ()):
                    if child_id not in assignments:
                        assignments[child_id] = root_id
                        next_frontier.append(child_id)
            frontier = next_frontier

    # Cycle-orphans: canonical etymons that participate in bridging
    # edges but never reached a root. A pure cycle has no anchor.
    cycle_orphans = bridging_participants - set(assignments.keys())
    return assignments, roots, cycle_orphans


def cluster_cognates(db: LexiconDB, *, apply: bool = False) -> dict:
    """Walk etymon_descent inheritance + borrowing edges from each root
    and assign cognate_id to every reachable descendant (D27 / wyrd-81n).

    A "root" is an etymon that participates in inheritance/borrowing
    descent edges as a parent but never as a child — i.e. the most-
    ancestral known form in its cognate chain (typically a Proto-* form).
    Every descendant reachable via inheritance or borrowing edges from
    that root gets cognate_id = root.id, so cross-language cognates
    cluster behind a single canonical pointer.

    Edge type semantics (D27):
      inheritance — direct lineage. Bridges cognate cluster.
      borrowing   — borrowed across language lines. Bridges cognate cluster (a
                    borrowed word IS part of the borrowing language's
                    cognate set in practice).
      cognate     — peer relation, NOT a chain. Does not bridge — would
                    over-unify, since Wiktionary's cognate sections
                    sometimes cross probable-but-unproven boundaries.
      derivation, calque, compound, unknown — context-specific; treat
                    as non-bridging for v1, refine if mining surfaces
                    a clear case.

    Determinism: when an etymon is reachable from multiple roots, the
    smallest root id wins. Iteration order is sorted by root id so the
    assignment is bit-stable across runs. NOTE this multi-root case is
    NOT rare — ~164k live nodes are reachable from >=2 distinct roots,
    often across unrelated IE families via borrowing chains (e.g.
    ``majem`` reaches Proto-Semitic, Greek, and French roots). Smallest-id
    is therefore a deterministic but ARBITRARY tiebreak, not a closest/
    deepest-ancestor pick; whether that mis-clusters at this volume — and
    whether to switch to a nearest-root tiebreak — is open in wyrd-asi7.

    Operates in CANONICAL space (per D22 / wyrd-223): edges'
    parent_id and child_id are resolved through merged_into_id before
    use, so a descent edge that points at an OCR-merge tombstone is
    treated as if it pointed at the canonical winner. cognate_id is
    written on canonical etymons only; tombstones stay NULL (and are
    rolled up at query time via merged_into_id chain). This bridges
    cross-source canonical-form mismatches like wiktextract's
    `tun` vs the place-name dictionaries' `tūn`.

    With apply=False (default) reports candidate counts without writing.
    With apply=True writes cognate_id + cognate_method='cluster-cognates-v2'
    only on rows whose current (cognate_id, cognate_method) doesn't already
    match the target — re-runs against unchanged data become real no-ops
    instead of redundant UPDATEs. Reverse with
    `clear-enrichment --stage=cognates --apply`.

    Returns a dict of:
      - roots: number of canonical root etymons walked
      - candidates: total canonical etymons that received (or would
        receive) a cognate_id assignment
      - applied: whether writes happened
      - rows_written: count of UPDATE statements that actually changed
        a row (always 0 in dry-run; ≤ candidates when applied)
      - tombstones_cleared: count of tombstoned etymons (merged_into_id
        set) whose stale cognate_id/cognate_method was NULLed to enforce
        the tombstones-stay-NULL invariant (wyrd-hn03). In dry-run this is
        the would-clear count.
      - cycle_orphans: count of canonical etymons that participate in
        bridging edges but couldn't be assigned because they sit in a
        cycle with no external root. Healthy data should report 0 here;
        non-zero means the descent graph has at least one closed loop
        with no anchor — worth flagging in operator output.
    """
    # f-string interpolation of `placeholders` is safe here:
    # _COGNATE_BRIDGING_EDGES is a module-level tuple of code-controlled
    # strings, never user input, so no SQL injection risk. Edge values
    # themselves are bound via parameters.
    placeholders = ", ".join(["?"] * len(_COGNATE_BRIDGING_EDGES))
    nb_placeholders = ", ".join(["?"] * len(_NON_BRIDGING_LANGUAGES))

    # Build the canonical edge set: every descent edge with both
    # endpoints resolved through merged_into_id. Returns
    # (parent_canon_id, child_canon_id) tuples. Self-loops introduced
    # by the resolution (parent and child both merge into the same
    # canonical) are filtered out — they'd waste a BFS node and offer
    # zero clustering signal.
    #
    # v2: drop every edge that TOUCHES a non-bridging-language etymon (PIE) so a
    # cluster can never root at / traverse through the cross-family PIE layer.
    # The language is checked on the ORIGINAL endpoints — a PIE reconstruction is
    # never itself an OCR-merge target, so it equals the canonical's language.
    canonical_edges = [
        (row["parent_canon"], row["child_canon"])
        for row in db.conn.execute(
            f"""
            SELECT
              COALESCE(p.merged_into_id, d.parent_id) AS parent_canon,
              COALESCE(c.merged_into_id, d.child_id) AS child_canon
            FROM etymon_descent d
            JOIN etymon p ON p.id = d.parent_id
            JOIN etymon c ON c.id = d.child_id
            WHERE d.edge_type IN ({placeholders})
              AND p.language NOT IN ({nb_placeholders})
              AND c.language NOT IN ({nb_placeholders})
            """,
            (*_COGNATE_BRIDGING_EDGES, *_NON_BRIDGING_LANGUAGES, *_NON_BRIDGING_LANGUAGES),
        ).fetchall()
        if row["parent_canon"] != row["child_canon"]
    ]

    assignments, roots, cycle_orphans = _assign_cognate_clusters(canonical_edges)

    # wyrd-hn03: enforce the documented "tombstones stay NULL" invariant. The
    # BFS only ever assigns CANONICAL ids (every edge endpoint is resolved
    # through merged_into_id first), so a tombstone is never (re)assigned here.
    # The stale state arises from INCREMENTAL re-enrichment: an etymon clustered
    # while it was canonical on one run, then folded (tombstoned) by a later
    # collapse — and because this pass writes canonical rows only and never
    # cleared a now-tombstone's old pointer, the pre-fold cognate_id persisted
    # across runs (a `clear-enrichment --stage=cognates` between runs WOULD wipe
    # it, but the incremental path doesn't clear). Those stale pointers are NOT
    # cosmetic: the bundle export's per-family era-reflex union
    # (_fetch_family_era_reflexes) calls etymon_era_reflexes for every family
    # member, and a folded member's stale cognate_id makes it return its OLD
    # cluster's reflexes — leaking the wrong forms onto the surviving lemma's
    # grid (the verb do/done leaking onto the toponym -don after old-english:don
    # folded into dūn). A tombstone rolls up via merged_into_id at query time,
    # so its own cognate_id is dead weight; NULL it so reads can't resurrect the
    # pre-fold cluster.
    rows_written = 0
    tombstones_cleared = 0
    if apply:
        for etymon_id, cognate_id in assignments.items():
            cur = db.conn.execute(
                "UPDATE etymon SET cognate_id = ?, cognate_method = ? "
                "WHERE id = ? "
                "  AND (cognate_id IS NOT ? OR cognate_method IS NOT ?)",
                (
                    cognate_id,
                    _CLUSTER_COGNATES_METHOD,
                    etymon_id,
                    cognate_id,
                    _CLUSTER_COGNATES_METHOD,
                ),
            )
            rows_written += cur.rowcount
        cur = db.conn.execute(
            "UPDATE etymon SET cognate_id = NULL, cognate_method = NULL "
            "WHERE merged_into_id IS NOT NULL "
            "  AND (cognate_id IS NOT NULL OR cognate_method IS NOT NULL)"
        )
        # max(0, ...) not `or 0`: a DB-API rowcount of -1 (undetermined) is
        # truthy and would otherwise survive. UPDATE rowcount is reliable here,
        # but be defensive.
        tombstones_cleared = max(0, cur.rowcount)
        db.commit()
    else:
        # dry-run reports the would-clear count (apply gets it from rowcount,
        # so the COUNT(*) — a scan over the etymon table — is dodged on the
        # hot path).
        tombstones_cleared = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon "
            "WHERE merged_into_id IS NOT NULL "
            "  AND (cognate_id IS NOT NULL OR cognate_method IS NOT NULL)"
        ).fetchone()["n"]

    return {
        "roots": len(roots),
        "candidates": len(assignments),
        "applied": apply,
        "rows_written": rows_written,
        "tombstones_cleared": tombstones_cleared,
        "cycle_orphans": len(cycle_orphans),
    }
