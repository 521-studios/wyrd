"""LLM cluster modern-reflex sense-audit (wyrd-1wv2) — the working era-grid
over-merge fix.

The era-grid shows a morpheme's modern reflexes = its cognate cluster's
modern-english members (``etymon_era_reflexes`` Tier-1). Over-merged clusters
mix senses, so implausible modern reflexes surface (``vulva`` on the ``val``
'valley' grid). The descent edge-audit (rd69/2wml) could NOT fix this: edge-detach
can't split the multi-parent deep-proto web, and its verdicts judged the wrong
layer (deep etymon edges) so they don't transfer to the modern reflexes shown.

This audits the MODERN LEAVES directly: judge each modern-english cluster member
against the cluster's dominant glossed sense, and detach the implausible leaf's
bridging in-edges. A modern reflex is a LEAF (one/few bridging parents), so
orphaning it needs no deep-cluster split. The detach round-trips through
``_collapses.jsonl``: at rebuild ``apply_collapses`` deletes the edges and the
fresh ``cluster_cognates`` drops the orphan; on a live deploy you clear cognates
+ re-cluster (the incremental path leaves a stale cognate_id — wyrd-hn03).

Pre-screen: a modern member that shares the dominant sense's gloss tokens is
auto-kept; only glossless / sense-disjoint members go to the LLM. The LLM is the
arbiter (there is no reliable deterministic plausibility test — over-merged words
look alike and most are glossless).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass

from wyrd.generators.kenning.lexicon.cognate_cluster import _NON_BRIDGING_LANGUAGES
from wyrd.generators.kenning.lexicon.collapse_merge import CONFIDENCE_RANK
from wyrd.generators.kenning.lexicon.descent_audit import _load_clustered_corpus
from wyrd.generators.kenning.lexicon.merge_audit import _sense_groups

LLM_CLUSTER_REFLEX_DETACH_METHOD = "llm-cluster-reflex-detach-v1"

_MODERN = "modern-english"
# Only screen clusters with >=2 modern members — a lone modern reflex has no
# sibling to be "noise" relative to and is rarely the visible-pollution case.
_MIN_MODERN_MEMBERS = 2
# Edge types that put a node IN the cognate cluster (mirror cluster_cognates).
_BRIDGING_EDGES = ("inheritance", "borrowing")


@dataclass(frozen=True)
class ClusterReflexCandidate:
    """One modern-english reflex shown on a cluster's era-grid that does NOT
    obviously match the cluster's dominant sense — for the LLM to judge: a
    genuine reflex of that sense (KEEP), or an unrelated word merged into the
    cluster (DETACH the leaf's bridging in-edges)."""

    reflex_ref: str  # "modern-english:<form>"
    reflex_form: str
    dominant_glosses: tuple[str, ...]
    bridging_parents: tuple[str, ...]  # refs of in-cluster bridging parents to detach


@dataclass(frozen=True)
class ClusterReflexVerdict:
    plausible: bool  # R is a genuine reflex of the dominant sense — KEEP
    confidence: str
    reason: str


def _bridging_parents(conn: sqlite3.Connection, child_id: int, by_id: dict[int, dict]) -> list[str]:
    """Refs of ``child_id``'s in-cluster bridging parents — the edges that hold
    the reflex in the cluster and must be detached to orphan it. Must match
    ``cluster_cognates`` exactly: bridging edge types, same cognate cluster, and
    parent NOT in ``_NON_BRIDGING_LANGUAGES`` (only PIE — family-level protos
    like proto-germanic / itc-pro DO bridge, so their edges must be cut too)."""
    cog = by_id[child_id]["cognate_id"]
    ph = ", ".join("?" * len(_BRIDGING_EDGES))
    parents: list[str] = []
    for r in conn.execute(
        f"SELECT parent_id FROM etymon_descent WHERE child_id=? AND edge_type IN ({ph})",
        (child_id, *_BRIDGING_EDGES),
    ):
        pr = by_id.get(r["parent_id"])
        if pr is None or pr["cognate_id"] != cog or pr["lang"] in _NON_BRIDGING_LANGUAGES:
            continue
        parents.append(pr["ref"])
    return parents


def detect_cluster_reflex_candidates(
    conn: sqlite3.Connection, *, limit_clusters: int | None = None, prescreen: bool = True
) -> list[ClusterReflexCandidate]:
    """Modern reflexes to audit across clusters with a clear dominant glossed
    sense + >=2 modern members.

    With ``prescreen=True`` (default, cheap) a modern member is suspect only when
    it is NOT in the dominant sense-group and shares none of its gloss tokens —
    on-sense members are auto-kept and never reach the LLM. With
    ``prescreen=False`` (broad) EVERY modern member is judged: the gloss-token
    pre-screen wrongly auto-keeps noise that ``_sense_groups`` over-grouped into
    the dominant via one incidental shared token (``mons pubis`` ~ 'mount'), so
    broad mode catches it at the cost of (cheaply) re-judging the on-sense members
    too (the judge keeps them)."""
    by_id = _load_clustered_corpus(conn)
    clusters: dict[int, list[int]] = defaultdict(list)
    for eid, rec in by_id.items():
        clusters[rec["cognate_id"]].append(eid)

    out: list[ClusterReflexCandidate] = []
    processed = 0
    for _cog, ids in sorted(clusters.items()):
        moderns = [i for i in ids if by_id[i]["lang"] == _MODERN]
        if len(moderns) < _MIN_MODERN_MEMBERS:
            continue
        glossed = [(i, by_id[i]["tokens"]) for i in ids if by_id[i]["tokens"]]
        if len(glossed) < 2:
            continue
        groups = _sense_groups(glossed)
        sizes = sorted((len(g) for g in groups), reverse=True)
        # need a CLEAR dominant glossed sense (unique largest, >=2 members)
        if sizes[0] < 2 or (len(sizes) > 1 and sizes[0] == sizes[1]):
            continue
        if limit_clusters is not None and processed >= limit_clusters:
            break
        processed += 1
        dom = max(range(len(groups)), key=lambda gi: len(groups[gi]))
        domset = set(groups[dom])
        domtok = {t for i in domset for t in by_id[i]["tokens"]}
        dom_glosses = tuple(sorted({gl for i in domset for gl in by_id[i]["glosses"]}))[:4]
        for m in moderns:
            if prescreen:
                if m in domset:
                    continue  # a glossed modern member OF the dominant sense
                if by_id[m]["tokens"] and (by_id[m]["tokens"] & domtok):
                    continue  # shares the dominant sense — auto-keep
            parents = _bridging_parents(conn, m, by_id)
            if not parents:
                continue  # already orphaned / no in-cluster bridging edge to cut
            out.append(
                ClusterReflexCandidate(
                    reflex_ref=by_id[m]["ref"],
                    reflex_form=by_id[m]["form"],
                    dominant_glosses=dom_glosses,
                    bridging_parents=tuple(sorted(set(parents))),
                )
            )
    return sorted(out, key=lambda c: c.reflex_ref)


_JUDGE_SYSTEM = (
    "You are a historical-linguistics editor auditing a place-name etymology "
    "lexicon. A MODERN ENGLISH word has been grouped into a cognate cluster whose "
    "core/dominant meaning is given. Decide whether the modern word is GENUINELY a "
    "modern reflex/descendant of THAT word-family and meaning (KEEP), or an "
    "UNRELATED word that was wrongly merged into the cluster via over-broad "
    "deep-etymology links (DETACH). Example DETACH: cluster sense 'valley' "
    "(Latin vallis / Old French val) but the modern word is 'vulva' or 'valgus' — "
    "these trace to a deep proto-root but are NOT modern reflexes of 'valley'. "
    "Example KEEP: cluster sense 'valley', modern word 'vale' or 'dale'. Judge by "
    "shared etymological identity AND core meaning, NOT surface look-alikeness. "
    'Reply ONLY with a JSON object: {"reflex_of_sense": true|false, '
    '"confidence": "high"|"medium"|"low", "reason": "<one short clause>"}.'
)


def build_judge_prompt(c: ClusterReflexCandidate) -> tuple[str, str]:
    dom = "; ".join(c.dominant_glosses) or "(unknown)"
    user = (
        f"Cluster dominant sense: {dom}\n"
        f'Modern English word: "{c.reflex_form}"\n\n'
        f'Is "{c.reflex_form}" a genuine modern reflex of that sense (keep), or an '
        f"unrelated word wrongly merged in (detach)?"
    )
    return _JUDGE_SYSTEM, user


def parse_verdict(raw: dict) -> ClusterReflexVerdict | None:
    if not isinstance(raw, dict):
        return None
    val = raw.get("reflex_of_sense")
    if isinstance(val, str):
        val = val.strip().lower() in {"true", "yes", "y", "1"}
    if not isinstance(val, bool):
        return None
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in CONFIDENCE_RANK:
        conf = "low"
    reason = str(raw.get("reason", "")).strip()[:200]
    return ClusterReflexVerdict(plausible=val, confidence=conf, reason=reason)


def audit_log_row(c: ClusterReflexCandidate, v: ClusterReflexVerdict) -> dict:
    """Raw, threshold-independent verdict row for ``_cluster_reflex_audit.jsonl``."""
    return {
        "_type": "cluster_reflex_audit",
        "reflex_ref": c.reflex_ref,
        "dominant_glosses": list(c.dominant_glosses),
        "bridging_parents": list(c.bridging_parents),
        "reflex_of_sense": v.plausible,
        "confidence": v.confidence,
        "reason": v.reason,
    }


def verdict_from_log(row: dict) -> ClusterReflexVerdict | None:
    if not isinstance(row.get("reflex_of_sense"), bool):
        return None
    conf = row.get("confidence")
    if conf not in CONFIDENCE_RANK:
        conf = "low"
    return ClusterReflexVerdict(
        plausible=row["reflex_of_sense"], confidence=conf, reason=str(row.get("reason", ""))
    )


def detach_row(row: dict, min_confidence: str) -> dict | None:
    """The ``_collapses.jsonl`` edge-detach for an IMPLAUSIBLE reflex at/above
    ``min_confidence``: orphan the leaf by detaching all its in-cluster bridging
    parents. None when the reflex is kept or below threshold."""
    v = verdict_from_log(row)
    if v is None or v.plausible or CONFIDENCE_RANK[v.confidence] < CONFIDENCE_RANK[min_confidence]:
        return None
    parents = row.get("bridging_parents") or []
    if not parents:
        return None
    return {
        "_type": "collapse",
        "ref": row["reflex_ref"],
        "detach_parents": sorted(parents),
        "method": LLM_CLUSTER_REFLEX_DETACH_METHOD,
        "confidence": v.confidence,
        "reason": v.reason,
    }
