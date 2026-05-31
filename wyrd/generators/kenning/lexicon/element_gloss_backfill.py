"""wyrd-u9k6: grounded gloss-backfill for unglossed generation surfaces.

~46% of the English generation pool is unglossed (≈2,861/6,142 usages). These
are orphan-reflex *surfaces* (``-ick``, ``-sley``, ``Bourne``) emitted into the
runtime ``meaning_db`` with empty glosses, so the glossed-only default pool
(wyrd-glos) excludes them. Most are REAL place-name elements; the gap is that
the modern surface isn't linked to a glossed etymon.

THE CREDIBLE PATH (validated 2026-05-31)
========================================
Don't ask an LLM to author a gloss for a bare surface — it hallucinates
(qwen mapped ``-swell`` → "village"; the real element is OE *wella* = spring).
Instead GROUND the gloss in real data:

1. For each unglossed surface, find the real place-names it occurs in
   (string match on position: suffix ``-ick`` → names ending "ick").
2. Read those names' ACTUAL mined etymologies (``toponym_etymology_element``)
   and consensus-vote the position-appropriate element's etymon (the LAST
   element for a suffix, the FIRST for a prefix).
3. Inherit that etymon's AUTHORITATIVE glosses (``etymon_gloss``, ≈24.6k
   already-mined glosses). The vote share gives confidence; junk fragments
   (``st``, ``ns``) get a low share against ubiquitous *tūn* and are rejected.

Deterministic (≈4s over all 2,861), free, and credible: the gloss is the
lexicon's existing scholarly data, not LLM invention. Spot-checked correct
across hard cases — ``gor-``→celtic *gort*, ``-huish``→OE *hiwisc*,
``-erby``→ON *bȳ*, ``Chur-``→OE *cirice*.

Ingestion: link the orphan reflex to the consensus etymon(s) via
``reflex_etymon``; the surface then inherits the gloss at export (reflexes
WITH an etymon link are never unglossed). The mined rows persist to
``data/mining/_element_glosses.jsonl`` (the rebuildable L2 record); a future
enrichment step replays them into ``reflex_etymon`` so the backfill survives a
full rebuild (same pattern as ``_collapses.jsonl``).

REMAINING (the long tail, for follow-up)
----------------------------------------
* ~1,756 "weak" surfaces have a sub-threshold consensus (compound endings like
  ``-ingham`` = -inga-+hām where neither component dominates, or genuinely
  ambiguous). Candidate-grounded LLM adjudication (give the model the consensus
  candidates + context, let it PICK — not invent) is the credible next lever.
* ~682 "no-match" surfaces have no toponym-etymology support — mostly junk
  fragments correctly excluded (overlaps wyrd-lp3n degenerate morphemes).
* 15 standalone surfaces (``Bourne``) aren't in the pre/post/inner ``reflex``
  table and need a separate linkage path.
"""
from __future__ import annotations

import json
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from typing import Any

METHOD_VERSION = "grounded-consensus-v1"
# Position marker (bundle usage) → reflex.position enum.
_POSITION_TO_REFLEX = {"prefix": "pre", "suffix": "post", "infix": "inner"}


def _deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s or "") if not unicodedata.combining(c))


def _normform(s: str) -> str:
    """Diacritic- + dash- + case-insensitive key (tūn / tun / -tun- → 'tun')."""
    return _deaccent(s).strip("-").strip().lower()


def _position(usage: str) -> str:
    pre, post = usage.endswith("-"), usage.startswith("-")
    if pre and post:
        return "infix"
    if post:
        return "suffix"
    if pre:
        return "prefix"
    return "standalone"


def _unglossed_surfaces(census: sqlite3.Connection, culture: str) -> list[tuple[str, int]]:
    """Proportion usages with no glossed meaning entry (orphan reflexes)."""
    return census.execute(
        "SELECT p.usage_key, p.weight FROM proportions_usage p WHERE p.culture=? "
        "AND NOT EXISTS (SELECT 1 FROM meaning m WHERE m.usage_key=p.usage_key "
        "AND EXISTS (SELECT 1 FROM json_each(m.data,'$.entries') je "
        "WHERE json_array_length(json_extract(je.value,'$.meaning'))>0)) "
        "ORDER BY p.weight DESC",
        (culture,),
    ).fetchall()


def _load_grounding(live: sqlite3.Connection) -> dict[str, Any]:
    """Build the in-memory indexes the consensus needs from the lexicon DB."""
    eid_gloss: dict[int, set[str]] = defaultdict(set)
    for eid, g in live.execute(
        "SELECT etymon_id, gloss FROM etymon_gloss WHERE gloss IS NOT NULL AND gloss!=''"
    ):
        eid_gloss[eid].add(g.strip())

    # (language, normform) → union of glosses across all matching glossed etymons.
    key_glosses: dict[tuple[str, str], set[str]] = defaultdict(set)
    for eid, lang, form in live.execute(
        "SELECT e.id, e.language, e.canonical_form FROM etymon e "
        "WHERE EXISTS (SELECT 1 FROM etymon_gloss g WHERE g.etymon_id=e.id)"
    ):
        key_glosses[(lang, _normform(form))] |= eid_gloss.get(eid, set())

    # toponym(lower) → {etymology_id → ordered [(ordinal, lang, form)]}
    topo_etys: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for name, teid, ordi, lang, form in live.execute(
        "SELECT t.modern_name, te.id, tee.ordinal, e.language, e.canonical_form "
        "FROM toponym t JOIN toponym_etymology te ON te.toponym_id=t.id "
        "JOIN toponym_etymology_element tee ON tee.toponym_etymology_id=te.id "
        "JOIN etymon e ON e.id=tee.etymon_id"
    ):
        topo_etys[name.lower()][teid].append((ordi, lang, form))
    for nm in topo_etys:
        for teid in topo_etys[nm]:
            topo_etys[nm][teid].sort()
    return {"key_glosses": key_glosses, "topo_etys": topo_etys, "names": list(topo_etys.keys())}


def _matching_names(usage: str, names: list[str]) -> list[str]:
    core = usage.strip("-").strip().lower()
    if len(core) < 2:
        return []
    pos = _position(usage)
    if pos == "suffix":
        return [n for n in names if n.endswith(core)]
    if pos == "prefix":
        return [n for n in names if n.startswith(core)]
    return [n for n in names if core in n]


def _confidence(top_support: int, top_share: float) -> str:
    if top_support >= 8 and top_share >= 0.45:
        return "high"
    if top_support >= 3 and top_share >= 0.25:
        return "medium"
    return "low"


def derive_element_glosses(census: sqlite3.Connection, live: sqlite3.Connection,
                           culture: str = "english") -> list[dict[str, Any]]:
    """Grounded-consensus gloss derivation — one row per unglossed surface."""
    g = _load_grounding(live)
    key_glosses, topo_etys, names = g["key_glosses"], g["topo_etys"], g["names"]
    rows: list[dict[str, Any]] = []
    for usage, weight in _unglossed_surfaces(census, culture):
        pos = _position(usage)
        mn = _matching_names(usage, names)
        votes: Counter = Counter()
        for nm in mn:
            for elems in topo_etys[nm].values():
                if pos == "suffix":
                    picks = [elems[-1]] if elems else []
                elif pos == "prefix":
                    picks = [elems[0]] if elems else []
                else:
                    picks = elems
                for _ordi, lang, form in picks:
                    key = (lang, _normform(form))
                    if key in key_glosses:  # only glossed etymons vote
                        votes[key] += 1
        row: dict[str, Any] = {"surface": usage, "weight": weight, "position": pos,
                               "n_toponyms": len(mn), "method": METHOD_VERSION}
        if not votes:
            row["is_element"], row["candidates"] = False, []
        else:
            total = sum(votes.values())
            cands = [{"language": lang, "form": form, "support": sup,
                      "share": round(sup / total, 2),
                      "glosses": sorted(key_glosses[(lang, form)])[:10]}
                     for (lang, form), sup in votes.most_common(3)]
            row["candidates"] = cands
            row["confidence"] = _confidence(cands[0]["support"], cands[0]["share"])
            row["is_element"] = row["confidence"] != "low"
        rows.append(row)
    return rows


def ingest_element_glosses(live: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, int]:
    """Link each confident surface's orphan reflex to its consensus etymon(s).

    The reflex then inherits the etymon's authoritative gloss at export.
    Additive + idempotent (``INSERT OR IGNORE``). Returns a summary counter.
    """
    ety: dict[tuple[str, str], list[int]] = defaultdict(list)
    for eid, lang, form in live.execute(
        "SELECT e.id, e.language, e.canonical_form FROM etymon e "
        "WHERE EXISTS (SELECT 1 FROM etymon_gloss g WHERE g.etymon_id=e.id)"
    ):
        ety[(lang, _normform(form))].append(eid)
    refl = {(sf, pos): rid for rid, sf, pos in
            live.execute("SELECT id, surface_form, position FROM reflex")}

    summary = Counter()
    for r in rows:
        if not r.get("is_element") or not r.get("candidates"):
            continue
        pos = _POSITION_TO_REFLEX.get(r["position"])
        rid = refl.get((r["surface"], pos)) if pos else None
        if rid is None:
            summary["no_reflex"] += 1
            continue
        cand = r["candidates"][0]
        eids = ety.get((cand["language"], _normform(cand["form"])), [])
        if not eids:
            summary["no_etymon"] += 1
            continue
        for eid in eids:
            live.execute(
                "INSERT OR IGNORE INTO reflex_etymon(reflex_id, etymon_id) VALUES(?,?)",
                (rid, eid),
            )
            summary["links"] += 1
        summary["linked_surfaces"] += 1
    live.commit()
    return dict(summary)


def write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
