"""Gloss-sense clustering (wyrd-u6fn.10) — compress a morpheme's gloss WALL into
its distinct SENSES, each carrying one short canonical label (the "compressed
gloss": the readable 'what does this toponym mean' token).

The shape differs from the duplicate-canonical finder (wyrd-u6fn.5, pairwise
"are these two etymons the same morpheme?"). Here the unit is ONE morpheme's own
gloss wall — ``tūn`` carries ~20 glosses ('enclosure', 'farmstead', 'farm,
settlement', 'manor', 'town', …) that are all ONE sense — and the task is to
PARTITION that wall into senses and name each. Most morphemes are MONOSEMOUS (one
sense, one label, e.g. 'town'); genuine polysemy ('wall' AND 'frog') yields a
small ARRAY, one short label per distinct sense.

Modeled as D50 IDENTITY (not a new predicate, per the projection wiring): each
sense is a synthetic ``canonical_sense`` hub; every gloss row in that sense
``bind``s to it (``kind="same-sense"``); the per-sense ``canonical-label`` is the
short compressed gloss. A gloss row's ref is the composite
``"<etymon_ref>\\x1f<gloss>"`` (:func:`gloss_ref`).

Bias is the MIRROR of the morpheme finder's. There, leave-SEPARATE is the safe
default (a wrong merge corrupts). Here, leave-MERGED (one sense) is the safe
default: over-splitting a monosemous wall into spurious "shades" is the failure
the user's tūn→'town' example calls out, so a split is only kept when a sense
genuinely differs (unrelated meaning), defaulting to ONE sense when unsure.

Two authoring paths share this module's assertion shape + validation:
  - the loop agent (Opus) partitions a slice itself and writes candidate rows
    (``method="opus-gloss-sense-v1"``); ``validate_gloss_sense_candidates`` gates
    them and ``sense_assertions`` authors them.
  - an Ollama batch pass (``cluster_gloss_wall``) does the same headless, for A/B
    and bulk fill (``method="gloss-sense-cluster-v1"``).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Literal

from wyrd.generators.kenning.canonicalization.assertions import (
    Assertion,
    NodeRef,
    mint_canonical_id,
)
from wyrd.generators.kenning.lexicon.collapse_merge import CONFIDENCE_RANK
from wyrd.generators.kenning.lexicon.etymon_refs import (
    gloss_ref,
    resolve_etymon_ref,
    split_gloss_ref,
)

METHOD_OLLAMA = "gloss-sense-cluster-v1"
METHOD_AGENT = "opus-gloss-sense-v1"
DEFAULT_SOURCE = "gloss-sense-audit"
_NODE = "canonical_sense"

# A compressed label is a SHORT token (1-3 words). Longer than this and it's a
# paraphrase, not a compression — rejected by validation so the wall doesn't just
# get copied through.
_MAX_LABEL_WORDS = 3
_MAX_LABEL_CHARS = 40


@dataclass(frozen=True)
class Sense:
    """One sense of a morpheme: a short label + the gloss-wall rows it covers."""

    label: str
    glosses: tuple[str, ...]


@dataclass(frozen=True)
class GlossSenseCandidate:
    """A proposed partition of one morpheme's gloss wall into senses, ready to
    validate + author. ``ref`` is the etymon natural key ``"<language>:<form>"``."""

    ref: str
    language: str
    canonical_form: str
    senses: tuple[Sense, ...]
    confidence: Literal["high", "medium", "low"]
    method: str
    model: str
    rationale: str = ""

    @property
    def all_glosses(self) -> list[str]:
        return [g for s in self.senses for g in s.glosses]


# --- parse a candidate row (agent- or Ollama-authored) -----------------------


def _norm_conf(raw: object) -> Literal["high", "medium", "low"]:
    conf = str(raw or "").strip().lower()
    return conf if conf in CONFIDENCE_RANK else "low"  # type: ignore[return-value]


def parse_candidate(row: dict) -> GlossSenseCandidate | None:
    """Parse one ``_type="gloss_sense"`` candidate row, or None if malformed.

    Shape::

        {"_type":"gloss_sense","ref":"old-english:tūn","language":"old-english",
         "canonical_form":"tūn",
         "senses":[{"label":"town","glosses":["enclosure","farmstead","town"]}],
         "confidence":"high","method":"opus-gloss-sense-v1","model":"opus",
         "rationale":"monosemous farm/settlement wall"}
    """
    if not isinstance(row, dict) or row.get("_type") != "gloss_sense":
        return None
    ref = row.get("ref")
    senses_raw = row.get("senses")
    if not isinstance(ref, str) or not isinstance(senses_raw, list) or not senses_raw:
        return None
    senses: list[Sense] = []
    for s in senses_raw:
        if not isinstance(s, dict):
            return None
        label = str(s.get("label") or "").strip()
        glosses = s.get("glosses")
        if not label or not isinstance(glosses, list) or not glosses:
            return None
        senses.append(Sense(label=label, glosses=tuple(str(g) for g in glosses)))
    return GlossSenseCandidate(
        ref=ref,
        language=str(row.get("language") or ref.split(":", 1)[0]),
        canonical_form=str(row.get("canonical_form") or ref.split(":", 1)[-1]),
        senses=tuple(senses),
        confidence=_norm_conf(row.get("confidence")),
        method=str(row.get("method") or METHOD_AGENT),
        model=str(row.get("model") or ""),
        rationale=str(row.get("rationale") or "").strip()[:300],
    )


# --- validation (the grounding guard) ----------------------------------------


def _label_ok(label: str) -> bool:
    return bool(label) and len(label) <= _MAX_LABEL_CHARS and len(label.split()) <= _MAX_LABEL_WORDS


def wall_glosses(conn: sqlite3.Connection, etymon_id: int) -> set[str]:
    """The morpheme's actual ``etymon_gloss`` rows — the grounding haystack."""
    return {
        r[0]
        for r in conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (etymon_id,))
    }


def validate_gloss_sense_candidates(
    conn: sqlite3.Connection,
    committed_refs: set[str],
    candidates: list[GlossSenseCandidate],
) -> list[str]:
    """Return a list of error strings (empty == OK). The gate, before any write:

    - ``ref`` resolves to a real etymon;
    - EVERY gloss named in a sense is a REAL row of that morpheme's wall (the
      grounding guard — the partition can't invent or alter gloss text);
    - the partition is COMPLETE (every wall gloss assigned) and a clean PARTITION
      (no gloss in two senses);
    - each label is a short compression (not a copied paraphrase);
    - not already sense-bound (committed) and not duplicated in-batch.
    """
    errors: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        eid = resolve_etymon_ref(conn, c.ref)
        if eid is None:
            errors.append(f"{c.ref}: resolves to no etymon")
            continue
        if c.ref in committed_refs:
            errors.append(f"{c.ref}: already sense-bound in the committed ledger")
            continue
        if c.ref in seen:
            errors.append(f"{c.ref}: duplicated in this batch")
            continue
        seen.add(c.ref)
        wall = wall_glosses(conn, eid)
        if not wall:
            errors.append(f"{c.ref}: morpheme has no gloss wall to compress")
            continue
        assigned = c.all_glosses
        assigned_set = set(assigned)
        invented = assigned_set - wall
        if invented:
            errors.append(f"{c.ref}: glosses not in the wall (invented): {sorted(invented)}")
        if len(assigned) != len(assigned_set):
            errors.append(f"{c.ref}: a gloss appears in more than one sense (not a partition)")
        missing = wall - assigned_set
        if missing:
            errors.append(f"{c.ref}: wall glosses left unassigned: {sorted(missing)}")
        bad_labels = [s.label for s in c.senses if not _label_ok(s.label)]
        if bad_labels:
            errors.append(f"{c.ref}: labels not short compressions: {bad_labels}")
    return errors


# --- authoring (mint canonical_sense + bind glosses + label) -----------------


def sense_assertions(
    c: GlossSenseCandidate, *, source: str = DEFAULT_SOURCE, actor: str = ""
) -> list[Assertion]:
    """Author one morpheme's senses: per sense, mint a ``canonical_sense`` hub,
    ``bind`` each of its gloss rows there (``same-sense``), and lay a
    ``canonical-label`` (the short compressed gloss). No ``merge-canonical`` — each
    sense is its own hub; cross-MORPHEME sense identity is the separate
    ``meaning_synset`` axis (D28), not this."""
    out: list[Assertion] = []
    for idx, sense in enumerate(c.senses):
        # Hub id is stable per (morpheme, label) so a re-run with the same label
        # reuses the hub; idx disambiguates the rare same-label collision.
        hub = mint_canonical_id(_NODE, c.language, c.canonical_form, sense.label, str(idx))
        rationale = (
            f"sense '{sense.label}' of '{c.canonical_form}' ({c.language}); {c.rationale}".strip(
                "; "
            )
        )
        out.append(
            Assertion(
                predicate="mint-canonical",
                subject=NodeRef(_NODE, hub),
                confidence="high",
                method=c.method,
                source=source,
                actor=actor,
                rationale=f"sense hub for '{c.canonical_form}' = '{sense.label}'",
            ).with_id()
        )
        for gloss in sense.glosses:
            out.append(
                Assertion(
                    predicate="bind",
                    subject=NodeRef("etymon_gloss", gloss_ref(c.language, c.canonical_form, gloss)),
                    object=NodeRef(_NODE, hub),
                    qualifiers={"kind": "same-sense"},
                    confidence=c.confidence,
                    method=c.method,
                    source=source,
                    actor=actor,
                    rationale=rationale,
                ).with_id()
            )
        out.append(
            Assertion(
                predicate="canonical-label",
                subject=NodeRef(_NODE, hub),
                qualifiers={"stratum": "", "value": sense.label},
                confidence=c.confidence,
                method=c.method,
                source=source,
                actor=actor,
                rationale=rationale,
            ).with_id()
        )
    return out


# --- remainder: which morphemes are already sense-bound (committed) ----------


def committed_sense_refs(assertions) -> set[str]:
    """Etymon refs that already carry an effective ``same-sense`` bind — the
    "done" set ``sense-next-slice`` excludes (remainder from the COMMITTED ledger,
    not the live DB; the wyrd-1rw4 invariant). ``assertions`` is an iterable of
    effective :class:`Assertion` (retractions already resolved)."""
    done: set[str] = set()
    for a in assertions:
        if (
            a.predicate == "bind"
            and a.subject.type == "etymon_gloss"
            and a.qualifiers.get("kind") == "same-sense"
        ):
            parts = split_gloss_ref(a.subject.ref)
            if parts is not None:
                done.add(parts[0])
    return done


# --- Ollama clustering (the headless / A/B path) -----------------------------

# Bias toward MONOSEMY: the wall is ONE sense unless meanings are genuinely
# unrelated. The propose pass partitions; the challenge pass (only when >1 sense)
# argues for collapse, and a split survives only un-refuted.
_PARTITION_SYSTEM = (
    "You compress a historical place-name etymology lexicon. A single morpheme is "
    "listed with ALL the glosses scholars recorded for it. Most morphemes have ONE "
    "sense expressed many ways (tūn: 'enclosure','farmstead','farm, settlement',"
    "'manor','town' are ALL one sense -> label 'town'). Only split into multiple "
    "senses when the meanings are genuinely UNRELATED (a true homograph: 'wall' AND "
    "'frog'), never for shades/near-synonyms of one idea. For EACH sense give a "
    "SHORT label: 1-2 lowercase words, the single clearest word for that meaning. "
    "EVERY listed gloss must be assigned to exactly one sense. When unsure, use ONE "
    "sense.\n"
    'Reply ONLY with JSON: {"senses":[{"label":"<short>","glosses":["<verbatim '
    'gloss>", ...]}, ...], "confidence":"high"|"medium"|"low", "reason":"<one clause>"}.'
)

_COLLAPSE_SYSTEM = (
    "You are a skeptical lexicographer. A proposal split ONE morpheme's glosses "
    "into MULTIPLE senses. Most morphemes are monosemous, so REFUTE the split "
    "unless the senses are genuinely UNRELATED meanings (a true homograph), not "
    "mere shades or near-synonyms of one idea.\n"
    'Reply ONLY with JSON: {"over_split": true|false, "confidence":"high"|"medium"'
    '|"low", "reason":"<one clause>"}.'
)


@dataclass
class GlossWall:
    ref: str
    language: str
    canonical_form: str
    glosses: tuple[str, ...]


@dataclass
class ClusterResult:
    candidate: GlossSenseCandidate | None
    raw_propose: dict = field(default_factory=dict)
    raw_challenge: dict = field(default_factory=dict)
    note: str = ""


def build_partition_prompt(wall: GlossWall) -> tuple[str, str]:
    listed = "\n".join(f"  - {g}" for g in wall.glosses)
    user = (
        f'Morpheme: "{wall.canonical_form}" (language: {wall.language})\n'
        f"Glosses:\n{listed}\n\nPartition these glosses into senses."
    )
    return _PARTITION_SYSTEM, user


def build_collapse_prompt(wall: GlossWall, senses: tuple[Sense, ...]) -> tuple[str, str]:
    shown = "\n".join(f"  sense '{s.label}': {', '.join(s.glosses)}" for s in senses)
    user = (
        f'Morpheme: "{wall.canonical_form}" (language: {wall.language})\n'
        f"Proposed senses:\n{shown}\n\nIs this an over-split of one sense?"
    )
    return _COLLAPSE_SYSTEM, user


def _parse_partition(raw: dict, wall: GlossWall) -> tuple[Sense, ...]:
    senses_raw = raw.get("senses") if isinstance(raw, dict) else None
    if not isinstance(senses_raw, list):
        return ()
    senses: list[Sense] = []
    for s in senses_raw:
        if not isinstance(s, dict):
            continue
        label = str(s.get("label") or "").strip()
        glosses = s.get("glosses")
        if label and isinstance(glosses, list) and glosses:
            senses.append(Sense(label=label, glosses=tuple(str(g) for g in glosses)))
    return tuple(senses)


def _as_bool(val: object) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes", "y", "1"}
    return None


def cluster_gloss_wall(
    client, wall: GlossWall, *, model: str, method: str = METHOD_OLLAMA
) -> ClusterResult:
    """Headless two-pass partition: propose senses, then (if >1) challenge the
    split; collapse to one sense if the skeptic refutes. ``client`` is any
    ``chat_json(system, user, schema)`` LLM client (Ollama / Anthropic)."""
    sys_p, usr_p = build_partition_prompt(wall)
    try:
        raw_p = client.chat_json(sys_p, usr_p, {})
    except Exception as exc:  # noqa: BLE001 — surfaced as a note, never fatal in a batch
        return ClusterResult(None, note=f"propose failed: {exc}")
    senses = _parse_partition(raw_p, wall)
    if not senses:
        return ClusterResult(None, raw_propose=raw_p, note="unparseable partition")
    conf = _norm_conf(raw_p.get("confidence"))
    reason = str(raw_p.get("reason") or "").strip()[:300]
    raw_c: dict = {}
    if len(senses) > 1:
        sys_c, usr_c = build_collapse_prompt(wall, senses)
        try:
            raw_c = client.chat_json(sys_c, usr_c, {})
        except Exception as exc:  # noqa: BLE001
            return ClusterResult(None, raw_propose=raw_p, note=f"challenge failed: {exc}")
        if _as_bool(raw_c.get("over_split")):
            # Collapse to one sense: keep the morpheme but as a single sense whose
            # label is the first (clearest) proposed label, covering the whole wall.
            senses = (Sense(label=senses[0].label, glosses=wall.glosses),)
            conf = "low"
            reason = f"collapsed over-split: {raw_c.get('reason') or ''}".strip()
    cand = GlossSenseCandidate(
        ref=wall.ref,
        language=wall.language,
        canonical_form=wall.canonical_form,
        senses=senses,
        confidence=conf,
        method=method,
        model=model,
        rationale=reason,
    )
    return ClusterResult(cand, raw_propose=raw_p, raw_challenge=raw_c)
