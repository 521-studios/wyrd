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

MEASURED (2026-05-31): linking the 394 confident surfaces and re-exporting
dropped English unglossed generation usages 2,861 → 2,488 (46.6% → 40.6%) —
373 surfaces newly glossed end-to-end. (21 of 394 didn't flow — mostly a
Celtic reflex→gloss edge, e.g. ``gor-``; follow-up.)

Ingestion: link the orphan reflex to the consensus etymon(s) via
``reflex_etymon``; the surface then inherits the gloss at export (reflexes
WITH an etymon link are never unglossed). The mined rows persist to
``data/mining/_element_glosses.jsonl`` (the rebuildable L2 record); a future
enrichment step replays them into ``reflex_etymon`` so the backfill survives a
full rebuild (same pattern as ``_collapses.jsonl``).

WEAK-TAIL ADJUDICATION (added 2026-05-31)
-----------------------------------------
The ~1,758 "weak" surfaces have sub-threshold consensus but REAL candidate
etymons (from the toponyms' actual etymologies). ``build_adjudication_user`` +
``accept_adjudication`` have a fast LLM (qwen2.5:7b) PICK among those grounded
candidates — never invent — and accept high/medium-confidence picks (a vowel
guard drops the rare consonant-fragment, ``nt``→tun, that slips through). Yield:
776 accepted (44%), 691 linked → ``data/mining/_element_gloss_adjudications.jsonl``.

REMAINING (for follow-up)
-------------------------
* ~682 "no-match" surfaces have no toponym-etymology support — mostly junk
  fragments correctly excluded (overlaps wyrd-lp3n degenerate morphemes).
* Standalone surfaces (``Bourne``) + 77 weak surfaces aren't in the
  pre/post/inner ``reflex`` table and need a separate linkage path.
* PRODUCTIONIZE: replay both jsonls into ``reflex_etymon`` in the enrichment
  chain (rebuildable, like ``_collapses.jsonl``) + a CLI command.
"""

from __future__ import annotations

import json
import re
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


def derive_element_glosses(
    census: sqlite3.Connection, live: sqlite3.Connection, culture: str = "english"
) -> list[dict[str, Any]]:
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
        row: dict[str, Any] = {
            "surface": usage,
            "weight": weight,
            "position": pos,
            "n_toponyms": len(mn),
            "method": METHOD_VERSION,
        }
        if not votes:
            row["is_element"], row["candidates"] = False, []
        else:
            total = sum(votes.values())
            cands = [
                {
                    "language": lang,
                    "form": form,
                    "support": sup,
                    "share": round(sup / total, 2),
                    "glosses": sorted(key_glosses[(lang, form)])[:10],
                }
                for (lang, form), sup in votes.most_common(3)
            ]
            row["candidates"] = cands
            row["confidence"] = _confidence(cands[0]["support"], cands[0]["share"])
            row["is_element"] = row["confidence"] != "low"
        rows.append(row)
    return rows


def _link_indexes(live: sqlite3.Connection):
    """(glossed-etymon index, reflex index) used by both ingest paths."""
    ety: dict[tuple[str, str], list[int]] = defaultdict(list)
    for eid, lang, form in live.execute(
        "SELECT e.id, e.language, e.canonical_form FROM etymon e "
        "WHERE EXISTS (SELECT 1 FROM etymon_gloss g WHERE g.etymon_id=e.id)"
    ):
        ety[(lang, _normform(form))].append(eid)
    refl = {
        (sf, pos): rid
        for rid, sf, pos in live.execute("SELECT id, surface_form, position FROM reflex")
    }
    return ety, refl


def _link_surface(live, ety, refl, summary, surface, position, candidate) -> None:
    """Link one surface's orphan reflex to the candidate etymon's gloss(es)."""
    pos = _POSITION_TO_REFLEX.get(position)
    rid = refl.get((surface, pos)) if pos else None
    if rid is None:
        summary["no_reflex"] += 1
        return
    eids = ety.get((candidate["language"], _normform(candidate["form"])), [])
    if not eids:
        summary["no_etymon"] += 1
        return
    for eid in eids:
        live.execute(
            "INSERT OR IGNORE INTO reflex_etymon(reflex_id, etymon_id) VALUES(?,?)",
            (rid, eid),
        )
        summary["links"] += 1
    summary["linked_surfaces"] += 1


def ingest_element_glosses(live: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, int]:
    """Link each CONFIDENT consensus surface's orphan reflex to its etymon(s).

    The reflex then inherits the etymon's authoritative gloss at export.
    Additive + idempotent (``INSERT OR IGNORE``). Returns a summary counter.
    """
    ety, refl = _link_indexes(live)
    summary = Counter()
    for r in rows:
        if r.get("is_element") and r.get("candidates"):
            _link_surface(live, ety, refl, summary, r["surface"], r["position"], r["candidates"][0])
    live.commit()
    return dict(summary)


def _has_vowel(surface: str) -> bool:
    """Real place-name elements have a vowel; a vowel-less core ('nt', 'sf') is
    a consonant-fragment that occasionally slips past the adjudicator."""
    return bool(re.search(r"[aeiouyà-öø-ÿ]", surface.strip("-").strip().lower()))


def ingest_adjudications(
    live: sqlite3.Connection, adj_rows: list[dict[str, Any]]
) -> dict[str, int]:
    """Link the ACCEPTED weak-tail adjudications (LLM-picked grounded candidate).

    Each accepted row carries the chosen ``candidate`` (a real consensus etymon
    the model selected over its siblings) — grounded, never invented. A vowel
    guard drops consonant-fragment surfaces the adjudicator wrongly accepted.
    Same linkage + idempotency as the consensus path.
    """
    ety, refl = _link_indexes(live)
    summary = Counter()
    for r in adj_rows:
        if not (r.get("accepted") and r.get("candidate")):
            continue
        if not _has_vowel(r["surface"]):
            summary["dropped_consonant_fragment"] += 1
            continue
        _link_surface(live, ety, refl, summary, r["surface"], r["position"], r["candidate"])
    live.commit()
    return dict(summary)


# --- Weak-tail adjudication prompt (the LLM PICKS among grounded candidates) ---
ADJUDICATION_SYS = (
    "You are an expert in English place-name etymology (English Place-Name Society "
    "tradition). You are given a recurring morpheme surface from real place-names, "
    "example names, and a SHORT LIST of CANDIDATE source elements derived from those "
    "names' actual scholarly etymologies. Choose which candidate is the true element "
    "behind this surface (consider position and the examples). Reply strict JSON: "
    '{"choice": <1-based index, or 0 if none fit / it is just a fragment>, '
    '"confidence":"high|medium|low", "why":"<=12 words"}.'
)


def build_adjudication_user(
    surface: str, position: str, context: list[str], candidates: list[dict[str, Any]]
) -> str:
    """The user prompt presenting the grounded candidates for the model to pick."""
    lines = [
        f"[{i + 1}] {c['language']} {c['form']} = "
        f"{', '.join(c['glosses'][:2]) or '(no gloss)'} "
        f"(seen in {c['support']}, {int(c['share'] * 100)}%)"
        for i, c in enumerate(candidates)
    ]
    return (
        f"Surface: {surface!r}  (position: {position})\n"
        f"Example place-names: {', '.join(context) or 'none'}\n"
        "Candidates:\n" + "\n".join(lines)
    )


def accept_adjudication(row: dict[str, Any], response: dict[str, Any]) -> dict[str, Any] | None:
    """Turn an LLM adjudication response into an accepted candidate, or None.

    Accepts a high/medium-confidence pick of a real candidate; rejects choice 0
    (none/junk) and low confidence.
    """
    try:
        choice = int(response.get("choice") or 0)
    except (TypeError, ValueError):
        return None
    conf = (response.get("confidence") or "").lower()
    cands = row.get("candidates") or []
    if 1 <= choice <= len(cands) and conf in ("high", "medium"):
        return cands[choice - 1]
    return None


def write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
