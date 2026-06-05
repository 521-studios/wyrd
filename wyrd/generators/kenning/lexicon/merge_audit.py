"""LLM merge-decision audit (wyrd-6hbv).

The era-grid gloss-blobs the staging QA flagged (``-ton`` showing the weight
unit ``ton``; ``-ing`` carrying ~45 glosses spanning patronymic + hill +
water-meadow) come from **bad merge decisions**: distinct homograph morphemes
tombstoned (``merged_into_id``) or lemma-linked (``lemma_id``) into one anchor,
so ``_fetch_member_glosses`` unions their disjoint senses onto the survivor.

This module audits every merge decision with an LLM and reverts the wrong ones.
There is no trustworthy deterministic screen — "same morpheme vs wrongly-merged
homograph" is inherently semantic (gloss-token disjointness alone over-splits
coherent clusters: ``wer``/``wher`` "where", ``arth``/``art`` "bear"). So the
deterministic pass only *screens* candidates (recall-oriented); the LLM is the
arbiter, exactly like ``meaning_merge_detect``.

Two candidate sources (union, deduped by member ref):

1. **Disjoint-sense anchors** — a canonical anchor whose merged family
   (self + ``merged_into_id`` children + ``lemma_id`` children) splits into ≥2
   gloss-token sense-groups; the members in a non-dominant group are intruders.
   Catches the OCR-cluster (``-ton``) and link-lemmas (``ing`` into ``-ing``)
   conflations that never went through ``_collapses.jsonl``.
2. **Collapse folds** — every fold currently live in ``_collapses.jsonl`` (full
   ledger re-audit / hygiene).

**Provenance-routed reverts** (the enrichment phase order is
``cluster-ocr → link-lemmas → apply-curation → … → apply-collapses``):

* ``collapse-fold`` → revert in ``_collapses.jsonl`` (``into: ""`` last-write-
  wins). A curation clear would be re-overwritten by ``apply_collapses``.
* ``ocr-cluster`` → revert in ``_curation.jsonl`` (``merged_into_ref: null``),
  which runs after ``cluster_ocr`` re-sets it on rebuild.
* ``lemma-link`` → revert in ``_curation.jsonl`` (``lemma_ref: null``).

Every verdict (keep + revert) is logged to ``_merge_audit.jsonl`` for idempotent
re-runs and an audit trail; only reverts touch the applied ledgers. The LLM is
NEVER re-run at rebuild — its verdicts round-trip through L2.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass

from wyrd.generators.kenning.lexicon.collapse_detect import is_form_of_pointer
from wyrd.generators.kenning.lexicon.collapse_merge import CONFIDENCE_RANK
from wyrd.generators.kenning.lexicon.variant_fold_detect import _gloss_tokens

LLM_MERGE_AUDIT_REVERT_METHOD = "llm-merge-audit-revert-v1"

# Provenance of a merge decision — determines which ledger reverts it.
PROV_COLLAPSE = "collapse-fold"
PROV_OCR = "ocr-cluster"
PROV_LEMMA = "lemma-link"

# merge_audit log verdict tags.
AUDIT_KEEP = "keep"
AUDIT_REVERT = "revert"


@dataclass(frozen=True)
class MergeAuditCandidate:
    """One merge decision for the LLM to judge: was ``member`` correctly merged
    into ``anchor`` (same morpheme — keep), or is it a distinct morpheme wrongly
    conflated (revert)? ``provenance`` routes the revert to the right ledger."""

    anchor_ref: str  # the survivor, "<language>:<canonical_form>"
    member_ref: str  # the merged-away / linked member that may be a homograph
    provenance: str  # PROV_COLLAPSE | PROV_OCR | PROV_LEMMA
    anchor_form: str
    member_form: str
    anchor_glosses: tuple[str, ...]
    member_glosses: tuple[str, ...]


@dataclass(frozen=True)
class MergeAuditVerdict:
    correct: bool  # the merge is correct (same morpheme) — keep it
    confidence: str  # "high" | "medium" | "low"
    reason: str


def _is_affix(form: str) -> bool:
    """True for a dashed affix surface (``-ton`` / ``Ham-`` / ``-ing-``) — a
    different morphological role from a bare word, used to flag affix↔bare-word
    OCR over-merges where the affix has no gloss of its own."""
    return form.startswith("-") or form.endswith("-")


def _sense_groups(glossed: list[tuple[int, set[str]]]) -> list[list[int]]:
    """Partition glossed members into sense-groups by shared gloss content token
    (union-find). ``glossed`` is ``[(etymon_id, {token, …}), …]`` with non-empty
    token sets. Two members land in one group iff they share ≥1 content token."""
    parent = {eid: eid for eid, _ in glossed}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_for_token: dict[str, int] = {}
    for eid, toks in glossed:
        for t in toks:
            if t in first_for_token:
                parent[find(eid)] = find(first_for_token[t])
            else:
                first_for_token[t] = eid
    groups: dict[int, list[int]] = defaultdict(list)
    for eid, _ in glossed:
        groups[find(eid)].append(eid)
    return list(groups.values())


def _load_corpus(
    conn: sqlite3.Connection,
) -> tuple[dict[int, dict], dict[str, int]]:
    """Load every etymon row + its non-pointer gloss tokens. Returns
    ``(by_id, id_for_ref)`` where ``by_id[id] = {lang, form, ref, merged_into_id,
    lemma_id, glosses:set[str], tokens:set[str]}``."""
    by_id: dict[int, dict] = {}
    for r in conn.execute(
        "SELECT id, language, canonical_form, merged_into_id, lemma_id FROM etymon"
    ):
        by_id[r["id"]] = {
            "lang": r["language"],
            "form": r["canonical_form"],
            "ref": f"{r['language']}:{r['canonical_form']}",
            "merged_into_id": r["merged_into_id"],
            "lemma_id": r["lemma_id"],
            "glosses": set(),
            "tokens": set(),
        }
    for r in conn.execute("SELECT etymon_id, gloss FROM etymon_gloss"):
        rec = by_id.get(r["etymon_id"])
        if rec is None or not r["gloss"] or is_form_of_pointer(r["gloss"]):
            continue
        rec["glosses"].add(r["gloss"])
        rec["tokens"].update(_gloss_tokens(r["gloss"]))
    # last-write-wins ref→id (a (lang, form) pair is corpus-unique among
    # canonical rows; tombstones share the form but the ref points at the row we
    # resolve for reverts — prefer the canonical one if present).
    id_for_ref: dict[str, int] = {}
    for eid, rec in by_id.items():
        if rec["ref"] not in id_for_ref or rec["merged_into_id"] is None:
            id_for_ref[rec["ref"]] = eid
    return by_id, id_for_ref


def _provenance(member: dict, collapse_folds: set[str]) -> str:
    """Classify how ``member`` got merged. A collapse-fold (in the live
    ``_collapses.jsonl`` net state) reverts in that ledger; an OCR tombstone
    (``merged_into_id`` not from a collapse) and a lemma link revert via
    curation."""
    if member["ref"] in collapse_folds:
        return PROV_COLLAPSE
    if member["merged_into_id"] is not None:
        return PROV_OCR
    return PROV_LEMMA


def _candidate(anchor: dict, member: dict, provenance: str) -> MergeAuditCandidate:
    return MergeAuditCandidate(
        anchor_ref=anchor["ref"],
        member_ref=member["ref"],
        provenance=provenance,
        anchor_form=anchor["form"],
        member_form=member["form"],
        anchor_glosses=tuple(sorted(anchor["glosses"]))[:4],
        member_glosses=tuple(sorted(member["glosses"]))[:4],
    )


def _anchor_intruders(
    anchor_id: int, child_ids: list[int], by_id: dict[int, dict], collapse_folds: set[str]
) -> list[MergeAuditCandidate]:
    """Intruder candidates for one anchor family, or [] if its glossed members
    don't split into ≥2 disjoint sense-groups."""
    glossed = [(i, by_id[i]["tokens"]) for i in (anchor_id, *child_ids) if by_id[i]["tokens"]]
    if len(glossed) < 2:
        return []
    groups = _sense_groups(glossed)
    if len(groups) < 2:
        return []
    # Dominant = the group holding the anchor; else the largest.
    dominant = next((g for g in groups if anchor_id in g), None) or max(groups, key=len)
    anchor = by_id[anchor_id]
    return [
        _candidate(anchor, by_id[mid], _provenance(by_id[mid], collapse_folds))
        for g in groups
        if g is not dominant
        for mid in g
        if mid != anchor_id
    ]


def _anchor_disjoint_candidates(
    by_id: dict[int, dict], collapse_folds: set[str]
) -> list[MergeAuditCandidate]:
    """Source 1 — a canonical anchor whose merged family (self + ``merged_into_id``
    / ``lemma_id`` children) splits into ≥2 gloss-token sense-groups; members in a
    non-dominant group are intruders (the ``-ing`` water-meadow class)."""
    family: dict[int, list[int]] = defaultdict(list)
    for eid, rec in by_id.items():
        parent = rec["merged_into_id"] or rec["lemma_id"]
        if parent is not None and parent in by_id:
            family[parent].append(eid)
    cands: list[MergeAuditCandidate] = []
    for anchor_id, child_ids in family.items():
        if by_id[anchor_id]["merged_into_id"] is not None:
            continue  # only canonical survivors anchor a family
        cands.extend(_anchor_intruders(anchor_id, child_ids, by_id, collapse_folds))
    return cands


def _affix_candidates(
    by_id: dict[int, dict], collapse_folds: set[str]
) -> list[MergeAuditCandidate]:
    """Source 2 — a glossless dashed affix merged into a GLOSSED bare (non-affix)
    anchor: the affix inherits the bare word's unrelated sense (the ``-ton`` →
    ``ton`` weight-unit class, which the member-disjoint screen misses because the
    affix carries no gloss). Benign affix↔root pairs (``-land``→``land``) survive
    the LLM judge."""
    cands: list[MergeAuditCandidate] = []
    for member in by_id.values():
        anchor_id = member["merged_into_id"] or member["lemma_id"]
        if anchor_id is None or anchor_id not in by_id:
            continue
        anchor = by_id[anchor_id]
        if (
            _is_affix(member["form"])
            and not member["glosses"]
            and not _is_affix(anchor["form"])
            and anchor["glosses"]
            and anchor["merged_into_id"] is None
        ):
            cands.append(_candidate(anchor, member, _provenance(member, collapse_folds)))
    return cands


def _collapse_fold_candidates(
    by_id: dict[int, dict], id_for_ref: dict[str, int], collapse_state: dict[str, dict]
) -> list[MergeAuditCandidate]:
    """Source 3 — every fold live in ``_collapses.jsonl`` (full ledger re-audit),
    regardless of gloss disjointness."""
    cands: list[MergeAuditCandidate] = []
    for ref, row in collapse_state.items():
        into_ref = (row.get("into") or "").strip()
        if not into_ref:
            continue
        mid, aid = id_for_ref.get(ref), id_for_ref.get(into_ref)
        if mid is None or aid is None:
            continue  # ref resolved away (already reverted / typo)
        cands.append(_candidate(by_id[aid], by_id[mid], PROV_COLLAPSE))
    return cands


def detect_merge_audit_candidates(
    conn: sqlite3.Connection,
    collapse_state: dict[str, dict],
    *,
    scope: str = "both",
) -> list[MergeAuditCandidate]:
    """Gather merge-audit candidates from all sources, deduped by member ref.

    ``collapse_state`` is the net ``_collapses.jsonl`` state (``collect_collapses``
    output: ref → row, last-write-wins). ``scope`` ∈ {"anchors", "affixes",
    "collapses", "both"}. Collapse-fold provenance wins on dedup (it IS a live
    fold), so collapse candidates are layered first.
    """
    by_id, id_for_ref = _load_corpus(conn)
    # Live collapse folds = refs whose net state still folds into something.
    collapse_folds = {ref for ref, row in collapse_state.items() if (row.get("into") or "").strip()}

    sources: list[MergeAuditCandidate] = []
    if scope in ("collapses", "both"):
        sources += _collapse_fold_candidates(by_id, id_for_ref, collapse_state)
    if scope in ("anchors", "both"):
        sources += _anchor_disjoint_candidates(by_id, collapse_folds)
    if scope in ("affixes", "both"):
        sources += _affix_candidates(by_id, collapse_folds)

    out: dict[str, MergeAuditCandidate] = {}  # member_ref → candidate (first wins)
    for cand in sources:
        out.setdefault(cand.member_ref, cand)
    return sorted(out.values(), key=lambda c: (c.anchor_ref, c.member_ref))


_AUDIT_JUDGE_SYSTEM = (
    "You are a historical-linguistics editor auditing a place-name etymology "
    "lexicon for WRONGLY MERGED homographs. You are given an ANCHOR head-word and "
    "a MEMBER head-word that was merged INTO it (treated as the same word-family). "
    "Decide whether the MEMBER is genuinely the SAME morpheme as the ANCHOR (the "
    "same word — an inflection, spelling/dialect variant, or the same sense — so "
    "the merge is CORRECT and should be KEPT), or a DISTINCT morpheme that merely "
    "looks/sounds alike or shares a generic gloss word (a HOMOGRAPH wrongly "
    "merged — so it must be REVERTED). Classic wrong merges: the toponym suffix "
    "'-ton' (Old English tūn, 'farm/enclosure') merged into 'ton' (the weight "
    "unit); a patronymic suffix merged with a topographic 'hill'/'meadow' word. "
    "Classic correct merges: 'Johan'/'Johannes' into 'John'; '-ingas' (nominative "
    "plural) into the suffix '-ing'; 'blak' into 'black'. A dashed AFFIX folded "
    "into its own bare root word is CORRECT — '-land' into 'land', 'up-' into "
    "'up', 'mill-' into 'mill' are the same morpheme; revert an affix only when "
    "the bare anchor is a genuinely different word ('-ton' the place-name suffix "
    "vs 'ton' the weight unit; '-ere' the agent suffix vs 'ere' meaning ear). "
    "Judge by shared "
    "etymological identity and core meaning, NOT surface look-alikeness. Reply "
    "ONLY with a JSON object: "
    '{"same_morpheme": true|false, "confidence": "high"|"medium"|"low", '
    '"reason": "<one short clause>"}.'
)


def build_audit_judge_prompt(c: MergeAuditCandidate) -> tuple[str, str]:
    """Return (system, user) prompts for judging a merge-audit candidate."""
    lang = c.anchor_ref.split(":", 1)[0]
    user = (
        f"Language: {lang}\n"
        f'ANCHOR (survivor) "{c.anchor_form}" — gloss(es): '
        f"{'; '.join(c.anchor_glosses) or '(none)'}\n"
        f'MEMBER (merged into the anchor) "{c.member_form}" — gloss(es): '
        f"{'; '.join(c.member_glosses) or '(none)'}\n\n"
        f'Is "{c.member_form}" the SAME morpheme as "{c.anchor_form}" (merge '
        f"correct, keep), or a distinct homograph wrongly merged (revert)?"
    )
    return _AUDIT_JUDGE_SYSTEM, user


def parse_audit_verdict(raw: dict) -> MergeAuditVerdict | None:
    """Coerce an LLM JSON response into a :class:`MergeAuditVerdict`, or None if
    unusable (caller skips rather than recording, so a re-run retries)."""
    if not isinstance(raw, dict):
        return None
    val = raw.get("same_morpheme")
    if isinstance(val, str):
        val = val.strip().lower() in {"true", "yes", "y", "1"}
    if not isinstance(val, bool):
        return None
    conf = str(raw.get("confidence", "")).strip().lower()
    if conf not in CONFIDENCE_RANK:
        conf = "low"
    reason = str(raw.get("reason", "")).strip()[:200]
    return MergeAuditVerdict(correct=val, confidence=conf, reason=reason)


@dataclass(frozen=True)
class MergeAuditRows:
    """The ledger writes a verdict produces. ``audit_log`` always set (the
    idempotency + audit trail row for ``_merge_audit.jsonl``); ``collapse`` /
    ``curation`` set only for an applied REVERT, routed by provenance."""

    audit_log: dict
    collapse: dict | None = None
    curation: dict | None = None


def audit_verdict_to_rows(
    c: MergeAuditCandidate, v: MergeAuditVerdict, min_confidence: str
) -> MergeAuditRows:
    """Build the ledger rows for a verdict. A confident wrong-merge verdict emits
    a REVERT to the provenance-appropriate ledger; a kept (or low-confidence)
    verdict emits only the audit-log marker so re-runs skip it."""
    revert = (not v.correct) and CONFIDENCE_RANK[v.confidence] >= CONFIDENCE_RANK[min_confidence]
    audit_log = {
        "_type": "merge_audit",
        "member_ref": c.member_ref,
        "anchor_ref": c.anchor_ref,
        "provenance": c.provenance,
        "verdict": AUDIT_REVERT if revert else AUDIT_KEEP,
        "confidence": v.confidence,
        "reason": v.reason,
    }
    if not revert:
        return MergeAuditRows(audit_log=audit_log)
    if c.provenance == PROV_COLLAPSE:
        return MergeAuditRows(
            audit_log=audit_log,
            collapse={
                "_type": "collapse",
                "ref": c.member_ref,
                "into": "",
                "method": LLM_MERGE_AUDIT_REVERT_METHOD,
                "confidence": v.confidence,
                "reason": v.reason,
            },
        )
    curation = {"_type": "etymon_curation", "ref": c.member_ref, "reason": v.reason}
    if c.provenance == PROV_OCR:
        curation["merged_into_ref"] = None
    else:  # PROV_LEMMA
        curation["lemma_ref"] = None
    return MergeAuditRows(audit_log=audit_log, curation=curation)
