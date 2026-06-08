"""LLM descent-edge audit — split over-merged cognate clusters (wyrd-rd69).

Phase 2 of the era-grid over-merge cleanup (sibling of the merge-audit wyrd-6hbv
and the reflex-audit wyrd-862b). ``cluster_cognates`` builds cognate clusters by
transitive closure over ``etymon_descent``; bad descent edges pull
etymologically-distinct words into one cluster, so a morpheme's era-grid
surfaces unrelated modern forms:

* PROTO-FORM FAN-OUT (dominant) — an un-glossed proto conductor with over-broad
  ``inheritance`` children (``itc-pro:*wolwumen`` -> {vallis, vallum, valgus,
  vulgus, vulva, volume}) merges ``old-french:val`` -> ``vale`` with ``vulva``.
* SPURIOUS DIRECT EDGES — ``old-english:hām`` -> ``am``/``om``.

There is no trustworthy deterministic screen — over-merged words ARE
orthographically similar (val/vulva/vulgus) and most cluster members are
glossless, so neither surface nor gloss similarity separates them on its own.
The deterministic pass therefore only SCREENS candidate bridge edges (clusters
whose glossed members split into >=2 gloss-token sense-groups); the LLM is the
arbiter, exactly like ``merge_audit``.

A candidate is a descent EDGE that bridges sense-groups within an incoherent
cluster: either both endpoints glossed in different groups, or an un-glossed
conductor linking a glossed member of a NON-dominant group (judged against the
cluster's dominant sense). The repair is an edge-**detach** in
``_collapses.jsonl`` (``detach_parents``; applied by ``apply_collapses`` BEFORE
the fold pass, so ``cluster_cognates`` re-derives clean clusters). Every verdict
is logged to ``_descent_audit.jsonl`` for idempotent re-runs; the LLM is NEVER
re-run at rebuild — its verdicts round-trip through L2.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass

from wyrd.generators.kenning.lexicon.collapse_detect import is_form_of_pointer
from wyrd.generators.kenning.lexicon.collapse_merge import CONFIDENCE_RANK
from wyrd.generators.kenning.lexicon.merge_audit import _sense_groups
from wyrd.generators.kenning.lexicon.variant_fold_detect import _gloss_tokens

LLM_DESCENT_AUDIT_DETACH_METHOD = "llm-descent-audit-detach-v1"

# Clusters smaller than this never go to the LLM — an over-merge needs a real
# bridge between distinct senses, which a 2-member cluster cannot exhibit.
_MIN_CLUSTER_SIZE = 3


@dataclass(frozen=True)
class DescentAuditCandidate:
    """One descent edge ``parent -> child`` (claimed same word-family) for the
    LLM to judge: are they genuinely cognate (KEEP the edge), or unrelated words
    wrongly linked via an over-broad proto-form / spurious edge (DETACH)?
    ``cluster_dominant_glosses`` gives the cluster's main sense as context when
    an endpoint is an un-glossed conductor."""

    parent_ref: str  # "<language>:<canonical_form>"
    child_ref: str
    edge_type: str
    parent_form: str
    child_form: str
    parent_lang: str
    child_lang: str
    parent_glosses: tuple[str, ...]
    child_glosses: tuple[str, ...]
    cluster_dominant_glosses: tuple[str, ...]

    @property
    def edge_key(self) -> str:
        """Stable identity for the ledger: the directed edge."""
        return f"{self.parent_ref}->{self.child_ref}"


@dataclass(frozen=True)
class DescentAuditVerdict:
    related: bool  # parent & child are the same word-family — KEEP the edge
    confidence: str  # "high" | "medium" | "low"
    reason: str


def _load_clustered_corpus(conn: sqlite3.Connection) -> dict[int, dict]:
    """Every clustered etymon (``cognate_id`` set) + its non-pointer gloss
    tokens. ``by_id[id] = {lang, form, ref, cognate_id, glosses, tokens}``."""
    by_id: dict[int, dict] = {}
    for r in conn.execute(
        "SELECT id, language, canonical_form, cognate_id FROM etymon WHERE cognate_id IS NOT NULL"
    ):
        by_id[r["id"]] = {
            "lang": r["language"],
            "form": r["canonical_form"],
            "ref": f"{r['language']}:{r['canonical_form']}",
            "cognate_id": r["cognate_id"],
            "glosses": set(),
            "tokens": set(),
        }
    # JOIN-filter to clustered etymons only — avoids scanning the whole
    # etymon_gloss table (which dwarfs the clustered subset).
    for r in conn.execute(
        "SELECT eg.etymon_id AS etymon_id, eg.gloss AS gloss FROM etymon_gloss eg "
        "JOIN etymon e ON e.id = eg.etymon_id WHERE e.cognate_id IS NOT NULL"
    ):
        rec = by_id.get(r["etymon_id"])
        if rec is None or not r["gloss"] or is_form_of_pointer(r["gloss"]):
            continue
        rec["glosses"].add(r["gloss"])
        rec["tokens"].update(_gloss_tokens(r["gloss"]))
    return by_id


def _load_cluster_edges(conn: sqlite3.Connection) -> dict[int, list[tuple[int, int, str]]]:
    """All within-cluster descent edges (both endpoints share a ``cognate_id``),
    grouped by cluster, in ONE query — Phase 2b screens nearly every cluster, so
    a per-cluster query would be N roundtrips."""
    by_cluster: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()  # dedup (parent,child): a pair can have >1 edge_type
    for r in conn.execute(
        "SELECT d.parent_id AS p, d.child_id AS c, d.edge_type AS et, e1.cognate_id AS cog "
        "FROM etymon_descent d "
        "JOIN etymon e1 ON e1.id = d.parent_id "
        "JOIN etymon e2 ON e2.id = d.child_id "
        "WHERE e1.cognate_id IS NOT NULL AND e1.cognate_id = e2.cognate_id"
    ):
        key = (r["p"], r["c"])
        if key in seen:  # one tuple per directed pair — keeps out-degree honest
            continue
        seen.add(key)
        by_cluster[r["cog"]].append((r["p"], r["c"], r["et"]))
    return by_cluster


# Min within-cluster out-degree for a proto node to count as a fan-out conductor.
_MIN_PROTO_FANOUT = 3


def _is_proto(lang: str, form: str) -> bool:
    """A reconstructed form (``*``-prefixed) or a proto-language — the over-merge
    hub nodes. Wiktextract names proto languages two ways, both covered:
    ``proto-germanic``/``proto-celtic`` (``proto-`` prefix) and ``itc-pro``/
    ``gmw-pro``/``ine-pro`` (``-pro`` suffix)."""
    return form.startswith("*") or lang.startswith("proto-") or lang.endswith("-pro")


def _candidate(parent: dict, child: dict, edge_type: str, dominant_glosses: tuple[str, ...]):
    return DescentAuditCandidate(
        parent_ref=parent["ref"],
        child_ref=child["ref"],
        edge_type=edge_type,
        parent_form=parent["form"],
        child_form=child["form"],
        parent_lang=parent["lang"],
        child_lang=child["lang"],
        parent_glosses=tuple(sorted(parent["glosses"]))[:4],
        child_glosses=tuple(sorted(child["glosses"]))[:4],
        cluster_dominant_glosses=dominant_glosses,
    )


def _emit_phase2a(
    out: dict[str, DescentAuditCandidate],
    edges: list[tuple[int, int, str]],
    gidx: dict[int, int],
    unique_dom: int | None,
    by_id: dict[int, dict],
    dom_glosses: tuple[str, ...],
) -> None:
    """Phase 2a — bridge edges in an INCOHERENT cluster (>=2 sense-groups): two
    glossed members in different groups, or an un-glossed conductor -> a glossed
    member not in the dominant group (``unique_dom`` is None on a tie => every
    conductor->glossed edge is screened)."""
    for p, c, etype in edges:
        gp, gc = gidx.get(p), gidx.get(c)
        bridge = (
            (gp is not None and gc is not None and gp != gc)
            or (gp is None and gc is not None and gc != unique_dom)
            or (gc is None and gp is not None and gp != unique_dom)
        )
        if bridge:
            cand = _candidate(by_id[p], by_id[c], etype, dom_glosses)
            out.setdefault(cand.edge_key, cand)


def _emit_phase2b(
    out: dict[str, DescentAuditCandidate],
    edges: list[tuple[int, int, str]],
    gidx: dict[int, int],
    unique_dom: int | None,
    by_id: dict[int, dict],
    dom_glosses: tuple[str, ...],
) -> None:
    """Phase 2b — glossless-proto-conductor fan-out (wyrd-2wml). In a cluster with
    a clear dominant glossed sense, a proto/reconstruction conductor (out-degree
    >= _MIN_PROTO_FANOUT) often fans out to GLOSSLESS off-sense descendants the
    gloss-token screen can't see (``itc-pro:*wolwumen`` -> glossless ``vulva`` in
    the ``val`` 'valley' cluster). Screen each conductor edge whose child is NOT
    in the dominant sense (glossless, or glossed but sense-disjoint); the LLM
    judges the child form against the dominant sense.

    "Not in the dominant sense" is tested via sense-group membership
    (``gidx.get(c) != unique_dom``), consistent with 2a's union-find: a glossless
    child has no group (eligible); a glossed child sharing the dominant sense was
    already union-found INTO the dominant group (so it's skipped) — no looser
    token-overlap test needed."""
    outdeg: dict[int, int] = defaultdict(int)
    for p, _c, _et in edges:
        outdeg[p] += 1
    for p, c, etype in edges:
        pr = by_id[p]
        if not _is_proto(pr["lang"], pr["form"]) or outdeg[p] < _MIN_PROTO_FANOUT:
            continue
        if gidx.get(c) == unique_dom:
            continue  # edge into the dominant sense-group — presumed correct
        cand = _candidate(pr, by_id[c], etype, dom_glosses)
        out.setdefault(cand.edge_key, cand)


def detect_descent_audit_candidates(
    conn: sqlite3.Connection, *, scope: str = "both", limit_clusters: int | None = None
) -> list[DescentAuditCandidate]:
    """Descent-edge candidates over the cognate clusters, deduped by directed edge.

    ``scope``:
      * ``2a`` — bridge edges in INCOHERENT clusters (glossed members split into
        >=2 gloss-token sense-groups). High precision; both endpoints' senses known.
      * ``2b`` — glossless-proto-conductor fan-out: in a cluster with a clear
        dominant glossed sense, a proto conductor's edges to off-dominant /
        glossless children (the ``val`` -> ``vulva`` class the gloss screen misses).
      * ``both`` (default) — the union.
    """
    by_id = _load_clustered_corpus(conn)
    clusters: dict[int, list[int]] = defaultdict(list)
    for eid, rec in by_id.items():
        clusters[rec["cognate_id"]].append(eid)
    edges_by_cluster = _load_cluster_edges(conn)

    out: dict[str, DescentAuditCandidate] = {}
    processed = 0
    for cog, ids in sorted(clusters.items()):
        if len(ids) < _MIN_CLUSTER_SIZE:
            continue
        glossed = [(i, by_id[i]["tokens"]) for i in ids if by_id[i]["tokens"]]
        if not glossed:
            continue
        groups = _sense_groups(glossed)
        gidx: dict[int, int] = {}
        for gi, g in enumerate(groups):
            for i in g:
                gidx[i] = gi
        max_size = max(len(g) for g in groups)
        top = [gi for gi in range(len(groups)) if len(groups[gi]) == max_size]
        # A UNIQUE largest group is the dominant sense; a tie => no clear dominant.
        unique_dom = top[0] if len(top) == 1 else None
        dom_glosses: tuple[str, ...] = tuple(
            sorted(
                {
                    gl
                    for gi in (top if unique_dom is None else [unique_dom])
                    for i in groups[gi]
                    for gl in by_id[i]["glosses"]
                }
            )
        )[:4]
        edges = edges_by_cluster.get(cog, [])
        run_2a = scope in ("2a", "both") and len(groups) >= 2
        # 2b needs a clear dominant glossed sense (unique largest, >=2 members).
        run_2b = scope in ("2b", "both") and unique_dom is not None and max_size >= 2
        if not (run_2a or run_2b):
            continue
        if limit_clusters is not None and processed >= limit_clusters:
            break
        processed += 1
        if run_2a:
            _emit_phase2a(out, edges, gidx, unique_dom, by_id, dom_glosses)
        if run_2b:
            _emit_phase2b(out, edges, gidx, unique_dom, by_id, dom_glosses)
    return sorted(out.values(), key=lambda c: (c.parent_ref, c.child_ref))


_DESCENT_JUDGE_SYSTEM = (
    "You are a historical-linguistics editor auditing a place-name etymology "
    "lexicon for WRONGLY LINKED cognates. Two etymons are joined by a descent "
    "edge that claims they belong to the SAME word-family (one descends from / is "
    "cognate with the other). Decide whether they are GENUINELY the same root and "
    "word-family — sharing form AND core meaning across the relevant languages/eras "
    "(edge CORRECT, KEEP) — or DISTINCT words wrongly linked, typically via an "
    "over-broad proto-form that fans out to unrelated descendants, or a spurious "
    "edge (edge WRONG, DETACH). Examples to DETACH: Latin 'vallis' (valley) vs "
    "'vulva' or 'vulgus' (crowd) — they trace to a deep proto-root but are NOT the "
    "same word; Old English 'hām' (homestead) vs a bare 'am'. Examples to KEEP: "
    "English 'home' and German 'Heim' (both <Proto-Germanic *haimaz); Latin "
    "'vallis' and Old French 'val'/'vale' (valley). When one side is an un-glossed "
    "proto-form or reconstruction, judge the CHILD against the cluster's dominant "
    "sense given. Judge by shared etymological identity + core meaning, NOT surface "
    "look-alikeness. Reply ONLY with a JSON object: "
    '{"same_family": true|false, "confidence": "high"|"medium"|"low", '
    '"reason": "<one short clause>"}.'
)


def build_descent_judge_prompt(c: DescentAuditCandidate) -> tuple[str, str]:
    """Return (system, user) prompts for judging one descent-edge candidate."""
    pg = "; ".join(c.parent_glosses) or "(none — reconstruction/affix)"
    cg = "; ".join(c.child_glosses) or "(none — reconstruction/affix)"
    dom = "; ".join(c.cluster_dominant_glosses) or "(unknown)"
    user = (
        f'PARENT ({c.parent_lang}) "{c.parent_form}" — gloss(es): {pg}\n'
        f'CHILD  ({c.child_lang}) "{c.child_form}" — gloss(es): {cg}\n'
        f"Edge type: {c.edge_type}. Cluster dominant sense: {dom}\n\n"
        f'Are "{c.parent_form}" and "{c.child_form}" the SAME word-family (edge '
        f"correct, keep), or distinct words wrongly linked (detach)?"
    )
    return _DESCENT_JUDGE_SYSTEM, user


def parse_descent_verdict(raw: dict) -> DescentAuditVerdict | None:
    """Coerce an LLM JSON response into a verdict, or None if unusable (caller
    skips without recording, so a re-run retries)."""
    if not isinstance(raw, dict):
        return None
    val = raw.get("same_family")
    if isinstance(val, str):
        val = val.strip().lower() in {"true", "yes", "y", "1"}
    if not isinstance(val, bool):
        return None
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in CONFIDENCE_RANK:
        conf = "low"
    reason = str(raw.get("reason", "")).strip()[:200]
    return DescentAuditVerdict(related=val, confidence=conf, reason=reason)


def audit_log_row(c: DescentAuditCandidate, v: DescentAuditVerdict) -> dict:
    """The ``_descent_audit.jsonl`` row — the RAW judgment for one edge, recorded
    once and threshold-INDEPENDENT, so detaches can be (re-)derived at any
    ``min_confidence`` later without re-running the LLM."""
    return {
        "_type": "descent_audit",
        "edge_key": c.edge_key,
        "parent_ref": c.parent_ref,
        "child_ref": c.child_ref,
        "edge_type": c.edge_type,
        "same_family": v.related,
        "confidence": v.confidence,
        "reason": v.reason,
    }


def verdict_from_log(row: dict) -> DescentAuditVerdict | None:
    """Reconstruct a verdict from a recorded ``audit_log_row``, or None if the
    row predates the raw-verdict schema (must be re-judged)."""
    if not isinstance(row.get("same_family"), bool):
        return None
    conf = row.get("confidence")
    if conf not in CONFIDENCE_RANK:
        conf = "low"
    return DescentAuditVerdict(
        related=row["same_family"], confidence=conf, reason=str(row.get("reason", ""))
    )


def detach_row(
    parent_ref: str, child_ref: str, v: DescentAuditVerdict, min_confidence: str
) -> dict | None:
    """The ``_collapses.jsonl`` edge-detach row for a verdict, or None when the
    verdict is KEEP or below ``min_confidence``. Removes the ``parent -> child``
    edge: ``ref`` is the child (the "from"), ``detach_parents`` names the parent.
    Derivable from a logged verdict alone, so a later run can re-emit at a lower
    threshold without re-judging."""
    if v.related or CONFIDENCE_RANK[v.confidence] < CONFIDENCE_RANK[min_confidence]:
        return None
    return {
        "_type": "collapse",
        "ref": child_ref,
        "detach_parents": [parent_ref],
        "method": LLM_DESCENT_AUDIT_DETACH_METHOD,
        "confidence": v.confidence,
        "reason": v.reason,
    }
