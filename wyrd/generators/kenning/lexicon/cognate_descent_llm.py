"""LLM cognate-ancestor proposal for the residue (wyrd-zrce.2, D50 1b.2).

zrce.1 placed the unclustered admitted breakdown morphemes that EXACTLY fold-match a
clustered etymon. The residue (~3,071: no clustered fold-match) is diverse + well-
glossed (Phase 0 probe), and 85% have a clustered *stem-mate* (shared fold-prefix) —
a weak/noisy signal. This module lets an LLM adjudicate that noisy signal: is a
residue morpheme a genuine cognate of an existing cluster, or a spurious prefix
collision (leave-separate)?

Two passes (D50.4 leave-separate, adversarial):

1. **propose** — given the residue morpheme (form/lang/gloss) + candidate clusters
   (each a bridge member sharing the fold-prefix, with its gloss), the LLM picks the
   ONE cluster it descends from, or NONE. Default skeptical.
2. **refute** — an INDEPENDENT call tries to REFUTE the proposed cognate link. Only a
   confirmed proposal yields a ``descends-from`` edge at a gate-clearing confidence;
   refuted/none stays out (no edge). The proposal confidence is floored by the verify
   confidence, so a shaky link can't sneak past.

Scope: links residue → EXISTING clusters only (the LLM picks WHICH; we point
``descends-from`` at that cluster's root, reusing zrce.1's ``DescentEdge`` +
projection). Residue-with-residue NEW clusters are wyrd-zrce.3.

Durable outputs: the RAW verdict log (``_cognate_descent_audit.jsonl``, threshold-
independent, for idempotent re-runs — the LLM is NEVER re-run at rebuild) and, for
confirmed verdicts at/above the gate, ``descends-from`` assertions in the
canonicalization L2 stream (replayed + clustered by zrce.1's wired projection).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from wyrd.generators.kenning.lexicon.cognate_descent_mining import DescentEdge
from wyrd.generators.kenning.lexicon.collapse_merge import CONFIDENCE_RANK
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.genitive_priors import fold_surface

METHOD = "cognate-descent-llm-v1"

# Shared fold-prefix length that seeds the candidate set (a weak signal the LLM prunes).
PREFIX_LEN = 4
# Cap candidate clusters per morpheme so the prompt stays bounded + cheap.
MAX_CANDIDATES = 5


@dataclass(frozen=True)
class ClusterCand:
    """A candidate cluster for a residue morpheme: the bridge member that shares the
    fold-prefix (what the LLM judges against) + the cluster root we'd link to."""

    cluster_root: int  # cognate_id (most-ancestral etymon, D27) — the descends-from object
    bridge_form: str
    bridge_lang: str
    bridge_glosses: tuple[str, ...]


@dataclass(frozen=True)
class ResidueItem:
    """A residue morpheme + its candidate clusters, sorted best-first."""

    etymon_id: int
    form: str
    language: str
    glosses: tuple[str, ...]
    candidates: tuple[ClusterCand, ...]

    @property
    def ref(self) -> str:
        return str(self.etymon_id)


@dataclass(frozen=True)
class Proposal:
    """Pass-1 result: the chosen candidate index (or None) + confidence + reason."""

    chosen: int | None  # index into ResidueItem.candidates, or None = leave-separate
    confidence: str
    reason: str


@dataclass(frozen=True)
class Verdict:
    """Pass-2 result: did the independent refute pass CONFIRM the proposed link?"""

    confirmed: bool
    confidence: str
    reason: str


def _shared_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def build_candidates(
    db: LexiconDB, *, prefix_len: int = PREFIX_LEN, max_candidates: int = MAX_CANDIDATES
) -> list[ResidueItem]:
    """The residue cohort (unclustered breakdown morphemes WITH a gloss, no clustered
    fold-match) each paired with up to ``max_candidates`` candidate clusters reached
    via a shared fold-prefix. Glossless residue is skipped — the LLM has no semantic
    signal to adjudicate it, so it leaves-separate by construction."""
    breakdown = {
        r["etymon_id"]
        for r in db.conn.execute("SELECT DISTINCT etymon_id FROM toponym_etymology_element")
    }
    clustered_folds, prefix_index = _clustered_prefix_index(db, prefix_len)

    # Residue rows + the bridge etymon ids we'll need glosses for.
    residue_rows: list[tuple[int, str, str, str]] = []  # (eid, form, lang, fold)
    bridge_ids: set[int] = set()
    for r in db.conn.execute(
        "SELECT id, language, canonical_form FROM etymon "
        "WHERE cognate_id IS NULL AND merged_into_id IS NULL"
    ):
        if r["id"] not in breakdown:
            continue
        f = fold_surface(r["canonical_form"])
        if not f or f in clustered_folds or len(f) < prefix_len:
            continue
        residue_rows.append((r["id"], r["canonical_form"], r["language"] or "", f))
        for _root, _form, _lang, eid in prefix_index.get(f[:prefix_len], ()):
            bridge_ids.add(eid)

    glosses = _glosses_for(db, {eid for eid, *_ in residue_rows} | bridge_ids)
    empty: tuple[str, ...] = ()

    items: list[ResidueItem] = []
    for eid, form, lang, fold in residue_rows:
        gl = glosses.get(eid, empty)
        if not gl:  # no semantic signal → leave-separate, don't spend an LLM call
            continue
        bridges = prefix_index.get(fold[:prefix_len], ())
        cands = _select_candidates(fold, bridges, glosses, max_candidates)
        if cands:
            items.append(
                ResidueItem(etymon_id=eid, form=form, language=lang, glosses=gl, candidates=cands)
            )
    items.sort(key=lambda it: it.etymon_id)
    return items


def _clustered_prefix_index(
    db: LexiconDB, prefix_len: int
) -> tuple[set[str], dict[str, list[tuple[int, str, str, int]]]]:
    """The clustered folds (to exclude from the residue) + a fold-prefix → bridge
    members index ``(cluster_root, form, lang, etymon_id)``."""
    clustered_folds: set[str] = set()
    prefix_index: dict[str, list[tuple[int, str, str, int]]] = defaultdict(list)
    for r in db.conn.execute(
        "SELECT id, cognate_id, language, canonical_form FROM etymon WHERE cognate_id IS NOT NULL"
    ):
        f = fold_surface(r["canonical_form"])
        if not f:
            continue
        clustered_folds.add(f)
        if len(f) >= prefix_len:
            prefix_index[f[:prefix_len]].append(
                (r["cognate_id"], r["canonical_form"], r["language"] or "", r["id"])
            )
    return clustered_folds, prefix_index


def _select_candidates(
    fold: str,
    bridges: Sequence[tuple[int, str, str, int]],
    glosses: dict[int, tuple[str, ...]],
    max_candidates: int,
) -> tuple[ClusterCand, ...]:
    """The best bridge per candidate cluster (longest shared fold wins), the clusters
    sorted best-first and capped to ``max_candidates``. Fully deterministic — ties on
    (shared-fold, glossed) break on the smaller ``bridge_form`` so the chosen
    representative doesn't drift with DB row order (the SELECT has no ORDER BY)."""
    # root -> (shared_fold, has_gloss, bridge_form, bridge_lang, bridge_etymon_id)
    best: dict[int, tuple[int, int, str, str, int]] = {}
    for root, bform, blang, beid in bridges:
        cand = (
            _shared_prefix_len(fold, fold_surface(bform)),
            1 if glosses.get(beid) else 0,
            bform,
            blang,
            beid,
        )
        cur = best.get(root)
        # higher (shared_fold, has_gloss) wins; tie → smaller bridge_form.
        if cur is None or cand[:2] > cur[:2] or (cand[:2] == cur[:2] and cand[2] < cur[2]):
            best[root] = cand
    cands = sorted(
        (
            ClusterCand(
                cluster_root=root,
                bridge_form=bform,
                bridge_lang=blang,
                bridge_glosses=glosses.get(beid, ()),
            )
            for root, (_spl, _hg, bform, blang, beid) in best.items()
        ),
        key=lambda c: (
            -_shared_prefix_len(fold, fold_surface(c.bridge_form)),
            c.bridge_form,
            c.cluster_root,
        ),
    )[:max_candidates]
    return tuple(cands)


def _glosses_for(db: LexiconDB, ids: set[int]) -> dict[int, tuple[str, ...]]:
    acc: dict[int, list[str]] = defaultdict(list)
    for r in db.conn.execute("SELECT etymon_id, gloss FROM etymon_gloss"):
        if r["etymon_id"] in ids and r["gloss"]:
            acc[r["etymon_id"]].append(r["gloss"].strip())
    return {eid: tuple(g) for eid, g in acc.items()}


# --- Pass 1: propose -------------------------------------------------------

_PROPOSE_SYSTEM = (
    "You are a historical-linguistics editor placing a place-name morpheme into a "
    "cognate word-family. You are given ONE morpheme and a numbered list of candidate "
    "word-families (each shown by a member form + gloss). Pick the SINGLE candidate "
    "the morpheme genuinely DESCENDS FROM or is COGNATE WITH — same root AND core "
    "meaning across the relevant languages/eras — or NONE if it matches none. Be "
    "STRICT: a mere look-alike prefix is NOT cognacy; default to NONE when unsure "
    "(a missed link is harmless, a wrong link corrupts the family). Reply ONLY with a "
    'JSON object: {"choice": <1-based index or 0 for none>, '
    '"confidence": "high"|"medium"|"low", "reason": "<one short clause>"}.'
)


def build_propose_prompt(item: ResidueItem) -> tuple[str, str]:
    gl = "; ".join(item.glosses) or "(no gloss)"
    lines = [f'MORPHEME ({item.language}) "{item.form}" — gloss(es): {gl}', "", "CANDIDATES:"]
    for i, c in enumerate(item.candidates, start=1):
        cg = "; ".join(c.bridge_glosses) or "(no gloss)"
        lines.append(f'  {i}. ({c.bridge_lang}) "{c.bridge_form}" — {cg}')
    lines.append("")
    lines.append("Which candidate is it the same word-family as? Reply 0 for NONE.")
    return _PROPOSE_SYSTEM, "\n".join(lines)


def parse_proposal(raw: dict, n_candidates: int) -> Proposal | None:
    """Coerce the propose response; None if unusable (caller retries without recording)."""
    if not isinstance(raw, dict):
        return None
    choice = raw.get("choice")
    try:
        idx = int(choice)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx > n_candidates:
        return None
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in CONFIDENCE_RANK:
        conf = "low"
    reason = str(raw.get("reason", "")).strip()[:200]
    return Proposal(chosen=(idx - 1 if idx else None), confidence=conf, reason=reason)


# --- Pass 2: adversarial refute --------------------------------------------

_REFUTE_SYSTEM = (
    "You are a skeptical historical-linguistics referee. A proposal claims two "
    "place-name morphemes belong to the SAME cognate word-family (one descends from / "
    "is cognate with the other). Your job is to REFUTE it if you can: are they "
    "actually DISTINCT words that merely look alike (different roots, different core "
    "meaning, coincidental shared letters)? Confirm the link ONLY if it genuinely "
    "holds. Reply ONLY with a JSON object: "
    '{"confirmed": true|false, "confidence": "high"|"medium"|"low", '
    '"reason": "<one short clause>"}.'
)


def build_refute_prompt(item: ResidueItem, cand: ClusterCand) -> tuple[str, str]:
    mg = "; ".join(item.glosses) or "(no gloss)"
    cg = "; ".join(cand.bridge_glosses) or "(no gloss)"
    user = (
        f"PROPOSED COGNATES:\n"
        f'  A: ({item.language}) "{item.form}" — {mg}\n'
        f'  B: ({cand.bridge_lang}) "{cand.bridge_form}" — {cg}\n\n'
        f"Are A and B genuinely the same cognate word-family, or distinct look-alikes? "
        f"Refute the link if it does not hold."
    )
    return _REFUTE_SYSTEM, user


def parse_verdict(raw: dict) -> Verdict | None:
    if not isinstance(raw, dict):
        return None
    val = raw.get("confirmed")
    if isinstance(val, str):
        val = val.strip().lower() in {"true", "yes", "y", "1"}
    if not isinstance(val, bool):
        return None
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in CONFIDENCE_RANK:
        conf = "low"
    reason = str(raw.get("reason", "")).strip()[:200]
    return Verdict(confirmed=val, confidence=conf, reason=reason)


# --- verdict log (idempotency; threshold-independent) ----------------------


def audit_log_row(item: ResidueItem, prop: Proposal, verdict: Verdict | None) -> dict:
    """One raw judgment for a residue morpheme — recorded once, threshold-independent,
    so emissions re-derive at any min_confidence without re-calling the LLM."""
    chosen = item.candidates[prop.chosen] if prop.chosen is not None else None
    return {
        "_type": "cognate_descent_audit",
        "morpheme_ref": item.ref,
        "morpheme_form": item.form,
        "cluster_root": chosen.cluster_root if chosen else None,
        "bridge_form": chosen.bridge_form if chosen else None,
        "proposal_confidence": prop.confidence if chosen else None,
        "proposal_reason": prop.reason,
        "confirmed": bool(verdict and verdict.confirmed),
        "verify_confidence": verdict.confidence if verdict else None,
        "verify_reason": verdict.reason if verdict else "",
    }


def judge_item(item: ResidueItem, judge: Callable[[str, str], dict]) -> dict | None:
    """Run the two-pass judgment for one morpheme via ``judge(system, user) -> dict``
    (one LLM call). Returns the audit-log row, or ``None`` if a pass gave an unusable
    response (the caller skips-without-recording so a re-run retries). A NONE proposal
    short-circuits (no verify call) and records a leave-separate row."""
    prop = parse_proposal(judge(*build_propose_prompt(item)), len(item.candidates))
    if prop is None:
        return None
    if prop.chosen is None:
        return audit_log_row(item, prop, None)  # leave-separate; no verification needed
    cand = item.candidates[prop.chosen]
    verdict = parse_verdict(judge(*build_refute_prompt(item, cand)))
    if verdict is None:
        return None
    return audit_log_row(item, prop, verdict)


def edge_from_log(row: dict, *, min_confidence: str) -> DescentEdge | None:
    """Re-derive a ``DescentEdge`` from a logged judgment at ``min_confidence`` — only
    a CONFIRMED link whose effective confidence (the min of proposal + verify, so a
    shaky verify floors a confident proposal) clears the bar. None otherwise."""
    if not row.get("confirmed") or row.get("cluster_root") is None:
        return None
    pconf = row.get("proposal_confidence")
    vconf = row.get("verify_confidence")
    if pconf not in CONFIDENCE_RANK or vconf not in CONFIDENCE_RANK:
        return None
    effective = pconf if CONFIDENCE_RANK[pconf] <= CONFIDENCE_RANK[vconf] else vconf
    # Cap at medium: an LLM cognate inference is never high-confidence IDENTITY — and
    # DescentEdge.confidence is medium|low. A high/high judgment lands at medium.
    if CONFIDENCE_RANK[effective] > CONFIDENCE_RANK["medium"]:
        effective = "medium"
    if CONFIDENCE_RANK[effective] < CONFIDENCE_RANK[min_confidence]:
        return None
    return DescentEdge(
        child_etymon=int(row["morpheme_ref"]),
        cluster_root=int(row["cluster_root"]),
        confidence=effective,
        rationale=(
            f"LLM cognate-descent (propose+verify): '{row.get('morpheme_form')}' "
            f"-> cluster {row['cluster_root']} via '{row.get('bridge_form')}'"
        ),
    )
