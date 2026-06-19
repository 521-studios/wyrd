"""Spurious-breakdown finder (wyrd-u6fn.6) — catches mining hallucinations where a
toponym_etymology row's element morphemes belong to a *different* word than the
toponym it's filed under.

Concrete failure: toponym 'Newton' (attested Newtona/Neuton) carries elements
``ford + botl`` (toponym_etymology 780). The breakdown passed D3 form-in-body
validation (``extractors/llm.validate_response``) because ``ford``/``botl`` appear
elsewhere in the source paragraph (neighbour contamination), but they are not a
decomposition of *this* name. A sibling case: 'Newtown' tagged ``pisu + holm`` —
the breakdown of the neighbour entry 'Peaseholmes'.

Why this needs an LLM, not a surface heuristic: a literal-overlap check cannot
separate a contaminated breakdown (``ford`` vs ``Newton`` — unrelated) from a
*legit* Old-English decomposition where sound-change erased the overlap
(``ceacce`` → Checkendon, ``bucca`` → Boughton). Calibration flagged ~33% of all
rows on pure overlap, dominated by real decompositions. So a cheap deterministic
PRE-SCREEN narrows to low-overlap suspects, and an LLM JUDGE makes the
linguistic-plausibility call per row — the same division the sibling cleanup
passes (audit-merges / audit-descent / audit-toponym-reflexes) use.

Output: reversible ``decomposition-spurious`` assertions (D50, Family C) authored
to ``data/mining/canonicalization/_assert_decomposition_spurious.jsonl``. Like the
same-morpheme bind miner, these are authored + validated but DORMANT until the
canonicalization projection wires Family-C suppression (a separate ticket); they
never delete the observed breakdown — a ``retract`` record reverses one.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from typing import Literal

from wyrd.generators.kenning.canonicalization.assertions import Assertion, NodeRef

METHOD = "spurious-breakdown-finder-v1"
DEFAULT_SOURCE = "spurious-breakdown-audit"

# Pre-screen: a breakdown sharing at least this long a common run with the
# toponym name (after folding) is plausibly a real decomposition (sound-change
# tolerant) and is NOT sent to the judge. Tuned for recall — the LLM does precision.
_MIN_OVERLAP = 3


def _fold(s: str | None) -> str:
    """Lowercase, strip diacritics/macrons/ligatures, keep ascii letters only."""
    if not s:
        return ""
    # þ/ð → th BEFORE the ascii-strip — they're non-ascii and would otherwise be
    # dropped by encode('ascii','ignore') (folding 'þorn' to 'orn', not 'thorn').
    s = s.lower().replace("þ", "th").replace("ð", "th")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s)


def _lcs_len(a: str, b: str) -> int:
    """Length of the longest common contiguous substring of ``a`` and ``b``."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


@dataclass(frozen=True)
class BreakdownCandidate:
    """One toponym_etymology row whose breakdown shares little surface with its
    toponym name — a suspect for the LLM to judge."""

    te_id: int
    modern_name: str
    historical_form: str | None
    notes: str
    element_refs: tuple[str, ...]  # "<lang>:<canonical_form>" in ordinal order

    @property
    def element_forms(self) -> tuple[str, ...]:
        return tuple(r.split(":", 1)[-1] for r in self.element_refs)


@dataclass(frozen=True)
class BreakdownVerdict:
    spurious: bool  # True → the breakdown belongs to a different word (DETACH)
    confidence: Literal["high", "medium", "low"]
    reason: str


def detect_candidates(
    conn: sqlite3.Connection, *, min_overlap: int = _MIN_OVERLAP
) -> list[BreakdownCandidate]:
    """Low-overlap suspects: toponym_etymology rows whose element surfaces AND
    historical_form share no ``min_overlap``-char run with the (folded) toponym
    name. Pre-screen only — recall-biased; the LLM judge decides spuriousness."""
    conn.row_factory = sqlite3.Row
    by_te: dict[int, dict] = {}
    rows = conn.execute(
        """
        SELECT te.id AS te_id, t.modern_name AS nm, te.historical_form AS hf,
               te.notes AS notes, tee.ordinal AS ordinal,
               e.language AS lang, e.canonical_form AS form
        FROM toponym_etymology te
        JOIN toponym t ON t.id = te.toponym_id
        JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id
        JOIN etymon e ON e.id = tee.etymon_id
        ORDER BY te.id, tee.ordinal
        """
    )
    for r in rows:
        d = by_te.setdefault(
            r["te_id"],
            {"nm": r["nm"], "hf": r["hf"], "notes": r["notes"] or "", "els": []},
        )
        d["els"].append((r["ordinal"], f"{r['lang']}:{r['form']}"))

    out: list[BreakdownCandidate] = []
    for te_id, d in by_te.items():
        name = _fold(d["nm"])
        if len(name) < min_overlap or not d["els"]:
            continue
        refs = tuple(ref for _ord, ref in sorted(d["els"]))
        concat = _fold("".join(ref.split(":", 1)[-1] for ref in refs))
        hf = _fold(d["hf"])
        if _lcs_len(name, concat) < min_overlap and (not hf or _lcs_len(name, hf) < min_overlap):
            out.append(
                BreakdownCandidate(
                    te_id=te_id,
                    modern_name=d["nm"],
                    historical_form=d["hf"],
                    notes=d["notes"],
                    element_refs=refs,
                )
            )
    return sorted(out, key=lambda c: c.te_id)


_JUDGE_SYSTEM = (
    "You audit element-breakdowns in a historical ENGLISH/British PLACE-NAME etymology "
    "lexicon. Each breakdown should be the etymological decomposition of ONE specific "
    "toponym. Mining sometimes mis-attributes a breakdown that actually belongs to a "
    "DIFFERENT, neighbouring entry in the same source paragraph (contamination). Given a "
    "toponym, its source note (which lists its real attested historical spellings), and a "
    "proposed breakdown, decide: is the breakdown a plausible decomposition of THIS "
    "toponym (allowing for sound-change / spelling evolution, e.g. OE 'ceacce' -> "
    "Checkendon, 'bucca' -> Boughton, 'tun' -> -ton), or does it clearly belong to a "
    "different word (e.g. 'ford+botl' for Newton, 'pisu+holm' for Newtown)? Judge "
    "etymological plausibility, not exact spelling. Reply ONLY with a JSON object: "
    '{"spurious": true|false, "confidence": "high"|"medium"|"low", "reason": "<one short clause>"}. '
    "spurious=true means the breakdown does NOT belong to this toponym."
)


def build_judge_prompt(c: BreakdownCandidate) -> tuple[str, str]:
    note = (c.notes or "").strip()
    if len(note) > 400:
        note = note[:400] + "…"
    user = (
        f'Toponym: "{c.modern_name}"\n'
        f"Source note (real attested forms are here): {note or '(none)'}\n"
        f"Proposed breakdown: {' + '.join(c.element_forms)}\n"
        f"Is this breakdown a plausible decomposition of this toponym, or does it belong "
        f"to a different word?"
    )
    return _JUDGE_SYSTEM, user


def parse_verdict(raw: dict) -> BreakdownVerdict | None:
    if not isinstance(raw, dict):
        return None
    val = raw.get("spurious")
    if isinstance(val, str):
        val = val.strip().lower() in {"true", "yes", "y", "1"}
    if not isinstance(val, bool):
        return None
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in {"high", "medium", "low"}:
        conf = "low"
    return BreakdownVerdict(
        spurious=val, confidence=conf, reason=str(raw.get("reason", "")).strip()[:200]
    )


def spurious_assertion(
    c: BreakdownCandidate, v: BreakdownVerdict, *, source: str = DEFAULT_SOURCE, actor: str = ""
) -> Assertion:
    """A reversible ``decomposition-spurious`` assertion flagging this breakdown row
    (D50, Family C). Subject is the toponym_etymology observation; never deleted."""
    return Assertion(
        predicate="decomposition-spurious",
        subject=NodeRef("toponym_etymology", str(c.te_id)),
        qualifiers={"reason": v.reason},
        confidence=v.confidence,
        method=METHOD,
        source=source,
        actor=actor,
        rationale=(
            f"breakdown '{' + '.join(c.element_forms)}' inconsistent with toponym "
            f"'{c.modern_name}' (attested forms in source note); likely neighbour "
            f"contamination that passed D3 form-in-body"
        ),
    ).with_id()
