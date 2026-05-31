"""LLM same-meaning merge for collapse candidates — wyrd-6xzs (epic P3).

The deterministic detector (P2b) folds variant-doublings that SHARE an exact
gloss with their lemma. The much larger remainder — doublings whose glosses
don't string-match (``burh``'s "alternative form of burg" vs ``burg``'s
"fort"; genuine spelling variants whose senses are worded differently) —
needs a judgment: do the two etymons actually mean the same thing? That
can't be decided by string equality (the census showed generic glosses like
"hill" are shared by 38 unrelated words), so it goes to an LLM that reads the
two gloss sets and returns a verdict + confidence.

Every verdict is persisted to the ``_collapses.jsonl`` ledger so the
non-deterministic LLM pass never re-runs on a rebuild (the consolidation
epic's reconstructibility rule — re-judging would make rebuilds
non-reproducible): a confident "same meaning" becomes a real collapse row;
everything
else becomes a no-op ``into: ""`` row that records the judgment (and lets a
re-run skip the already-judged pair). The applier replays the positives for
free.

This module is pure logic — DB detection + prompt construction + verdict
parsing. The CLI (``merge-collapses``) owns the Ollama client + the I/O.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
LLM_MERGE_METHOD = "llm-meaning-match-v1"
LLM_REJECT_METHOD = "llm-rejected-v1"


@dataclass(frozen=True)
class MergeCandidate:
    """A variant-doubling pair whose meaning the LLM must judge."""

    from_ref: str
    into_ref: str
    from_form: str
    into_form: str
    from_glosses: tuple[str, ...]
    into_glosses: tuple[str, ...]
    variant_class: str


@dataclass(frozen=True)
class Verdict:
    same_meaning: bool
    confidence: str  # "high" | "medium" | "low"
    reason: str


def detect_merge_candidates(conn: sqlite3.Connection) -> list[MergeCandidate]:
    """Reflex-linked variant-doublings (both etymons glossed) that do NOT
    already share an exact gloss — i.e. exactly the cases the deterministic
    detector leaves behind. One candidate per ``from`` (first parent in
    ``(from_ref, into_ref)`` order). The shared-gloss filter runs in Python
    (the correlated SQL form is prohibitively slow over the variant table)."""
    pairs = conn.execute(
        """
        SELECT DISTINCT
            e.id   AS from_id,
            e.language || ':' || e.canonical_form AS from_ref,
            e.canonical_form AS from_form,
            parent.id AS into_id,
            parent.language || ':' || parent.canonical_form AS into_ref,
            parent.canonical_form AS into_form,
            v.variant_class AS variant_class
        FROM reflex_etymon re
        JOIN etymon e ON e.id = re.etymon_id AND e.merged_into_id IS NULL
        JOIN etymon_variant v ON v.form = e.canonical_form AND v.etymon_id != e.id
        JOIN etymon parent
          ON parent.id = v.etymon_id
         AND parent.language = e.language
         AND parent.id != e.id
         AND parent.merged_into_id IS NULL
        ORDER BY from_ref, into_ref
        """
    ).fetchall()

    # Bulk-load glosses for every etymon involved (one query, not N).
    ids = {r["from_id"] for r in pairs} | {r["into_id"] for r in pairs}
    glosses: dict[int, set[str]] = {i: set() for i in ids}
    norm: dict[int, set[str]] = {i: set() for i in ids}
    if ids:
        placeholders = ",".join("?" * len(ids))
        for row in conn.execute(
            f"SELECT etymon_id, gloss FROM etymon_gloss WHERE etymon_id IN ({placeholders})",
            tuple(ids),
        ):
            glosses[row["etymon_id"]].add(row["gloss"])
            norm[row["etymon_id"]].add(row["gloss"].strip().lower())

    out: list[MergeCandidate] = []
    seen: set[str] = set()
    for r in pairs:
        if r["from_ref"] in seen:
            continue
        fg, ig = glosses[r["from_id"]], glosses[r["into_id"]]
        # both must be glossed; skip if they already share a gloss (that's
        # the deterministic detector's job) or if either is unglossed.
        if not fg or not ig or (norm[r["from_id"]] & norm[r["into_id"]]):
            continue
        seen.add(r["from_ref"])
        out.append(
            MergeCandidate(
                from_ref=r["from_ref"],
                into_ref=r["into_ref"],
                from_form=r["from_form"],
                into_form=r["into_form"],
                from_glosses=tuple(sorted(fg)),
                into_glosses=tuple(sorted(ig)),
                variant_class=r["variant_class"] or "alternative",
            )
        )
    return out


_JUDGE_SYSTEM = (
    "You are a historical-linguistics editor consolidating a place-name "
    "etymology lexicon. You are given two head-words in the same language and "
    "their dictionary glosses. One is recorded as a spelling/inflectional "
    "VARIANT of the other. Decide whether they are genuinely the SAME lexical "
    "item (the same word, merely spelled or inflected differently) — in which "
    "case they should be merged — or DIFFERENT words that happen to be cross-"
    "referenced (homographs, or an over-eager variant link). Judge by meaning: "
    "if the glosses describe the same concept, they are the same item even if "
    "worded differently ('fort' vs 'a stronghold'). If the glosses describe "
    "unrelated concepts, they are different. Reply ONLY with a JSON object: "
    '{"same_meaning": true|false, "confidence": "high"|"medium"|"low", '
    '"reason": "<one short clause>"}.'
)


def build_judge_prompt(c: MergeCandidate) -> tuple[str, str]:
    """Return (system, user) prompts for judging a candidate."""
    lang = c.from_ref.split(":", 1)[0]
    user = (
        f"Language: {lang}\n"
        f'Variant form "{c.from_form}" — glosses: {"; ".join(c.from_glosses)}\n'
        f'Lemma form "{c.into_form}" — glosses: {"; ".join(c.into_glosses)}\n\n'
        f'Is "{c.from_form}" the same lexical item as "{c.into_form}"?'
    )
    return _JUDGE_SYSTEM, user


def parse_verdict(raw: dict) -> Verdict | None:
    """Coerce an LLM JSON response into a :class:`Verdict`, or None if it's
    unusable (so the caller can skip rather than crash the run)."""
    if not isinstance(raw, dict):
        return None
    sm = raw.get("same_meaning")
    if isinstance(sm, str):
        sm = sm.strip().lower() in {"true", "yes", "y", "1"}
    if not isinstance(sm, bool):
        return None
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in CONFIDENCE_RANK:
        conf = "low"  # unrecognized confidence → treat conservatively
    reason = str(raw.get("reason", "")).strip()[:200]
    return Verdict(same_meaning=sm, confidence=conf, reason=reason)


def verdict_to_row(c: MergeCandidate, v: Verdict, min_confidence: str) -> dict[str, str]:
    """Build the ``_collapses.jsonl`` row for a verdict. A confident
    same-meaning verdict becomes a real collapse; anything else is a no-op
    ``into: ""`` row that still records the judgment + lets re-runs skip it."""
    fold = v.same_meaning and CONFIDENCE_RANK[v.confidence] >= CONFIDENCE_RANK[min_confidence]
    return {
        "_type": "collapse",
        "ref": c.from_ref,
        "into": c.into_ref if fold else "",
        "variant_class": c.variant_class,
        "method": LLM_MERGE_METHOD if fold else LLM_REJECT_METHOD,
        "confidence": v.confidence,
        "reason": v.reason,
    }
