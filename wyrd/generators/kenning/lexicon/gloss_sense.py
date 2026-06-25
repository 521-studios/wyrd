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

Authoring path: the loop agent (Opus) partitions a slice itself and writes
``_type="gloss_sense"`` candidate rows (``method="opus-gloss-sense-v1"``);
``validate_gloss_sense_candidates`` gates them (the grounding guard) and
``sense_assertions`` authors them. (A headless Ollama batch path lands with the
A/B harness — wyrd-u6fn.10 follow-up — wired to its own CLI verb + a test.)
"""

from __future__ import annotations

import sqlite3
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from wyrd.generators.kenning.canonicalization.assertions import (
    Assertion,
    NodeRef,
    mint_canonical_id,
)
from wyrd.generators.kenning.lexicon.collapse_merge import CONFIDENCE_RANK
from wyrd.generators.kenning.lexicon.etymon_refs import (
    etymon_ref,
    gloss_ref,
    resolve_etymon_ref,
    split_gloss_ref,
)
from wyrd.generators.kenning.lexicon.morpheme_surface import normalize_morpheme_surface

METHOD_AGENT = "opus-gloss-sense-v1"
DEFAULT_SOURCE = "gloss-sense-audit"
_NODE = "canonical_sense"

# A compressed label is a SHORT token (1-3 words). Longer than this and it's a
# paraphrase, not a compression — rejected by validation so the wall doesn't just
# get copied through.
_MAX_LABEL_WORDS = 3
_MAX_LABEL_CHARS = 40


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _identity_ref(ref: str) -> str:
    """The ref reduced to the etymon it RESOLVES to: NFC + de-dash the form
    (``normalize_morpheme_surface``, D45 — an affix-position dash is never identity),
    mirroring ``resolve_etymon_ref`` and ``canonicalization_projection._identity_group_ref``.

    The dedup guards MUST key on this. ``resolve_etymon_ref`` de-dashes, and the
    etymon is stored bare, so ``old-english:tun`` and ``old-english:-tun`` resolve to
    one row — but a raw-ref dedup treats them as two morphemes, authoring two
    ``canonical_sense`` hubs. The projection then de-dashes the gloss's binds, sees 2
    distinct roots, conflicts (D46), and leaves the gloss UNBOUND. Keying on the
    de-dashed identity ref closes that hole."""
    nfc = _nfc(ref)
    if ":" in nfc:
        lang, form = nfc.split(":", 1)
        return f"{lang}:{normalize_morpheme_surface(form) or form}"
    return nfc


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
    # Strip first: a whitespace-only label ("   ") is truthy and splits to zero
    # words, so without the strip it would slip through the word-count guard and
    # author an empty canonical-label. parse_candidate already strips on the agent
    # path, but validate is the independent gate (hand-built candidates).
    stripped = label.strip()
    return (
        bool(stripped)
        and len(stripped) <= _MAX_LABEL_CHARS
        and len(stripped.split()) <= _MAX_LABEL_WORDS
    )


def wall_glosses(conn: sqlite3.Connection, etymon_id: int) -> set[str]:
    """The morpheme's actual ``etymon_gloss`` rows — the grounding haystack."""
    return {
        r[0]
        for r in conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (etymon_id,))
    }


def _grounding_errors(conn: sqlite3.Connection, eid: int, c: GlossSenseCandidate) -> list[str]:
    """Wall-grounding + partition + label/degeneracy checks for one resolved
    candidate (the per-candidate half of the gate; split out to keep the caller's
    complexity in check)."""
    wall = wall_glosses(conn, eid)
    if not wall:
        return [f"{c.ref}: morpheme has no gloss wall to compress"]
    errs: list[str] = []
    assigned = c.all_glosses
    assigned_set = set(assigned)
    invented = assigned_set - wall
    if invented:
        errs.append(f"{c.ref}: glosses not in the wall (invented): {sorted(invented)}")
    if len(assigned) != len(assigned_set):
        errs.append(f"{c.ref}: a gloss appears in more than one sense (not a partition)")
    missing = wall - assigned_set
    if missing:
        errs.append(f"{c.ref}: wall glosses left unassigned: {sorted(missing)}")
    bad_labels = [s.label for s in c.senses if not _label_ok(s.label)]
    if bad_labels:
        errs.append(f"{c.ref}: labels not short compressions: {bad_labels}")
    # A sense covering zero gloss rows mints a canonical_sense hub + label bound to
    # nothing. parse_candidate rejects empty glosses per sense; guard it here too
    # for hand-built candidates (wall-completeness above passes as long as the
    # OTHER senses cover the wall).
    empty_senses = [s.label for s in c.senses if not s.glosses]
    if empty_senses:
        errs.append(f"{c.ref}: sense(s) cover no glosses: {empty_senses}")
    return errs


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
        # The gate grounds against the etymon resolved from c.ref, but
        # sense_assertions authors the same-sense binds from c.language /
        # c.canonical_form (via gloss_ref). If those disagree with c.ref the binds
        # resolve to a different/absent gloss row at projection and the minted hub
        # is bound to nothing — reject the inconsistency. (parse_candidate derives
        # both from the ref so they always agree on the agent path; validate is the
        # independent gate for hand-built candidates.)
        if _nfc(etymon_ref(c.language, c.canonical_form)) != _nfc(c.ref):
            errors.append(
                f"{c.ref}: ref disagrees with language/canonical_form "
                f"({c.language}:{c.canonical_form})"
            )
            continue
        # Reduce the ref to its resolved-etymon identity for the dedup checks:
        # committed_refs and the slice/parked sets are all keyed the same way, so a
        # non-NFC (decomposed-diacritic) OR affix-dashed candidate ref — the OE/Welsh
        # forms this campaign targets — must be folded identically or it bypasses the
        # already-bound guard and re-mints a second canonical_sense hub for the same
        # morpheme (old-english:tun vs old-english:-tun both resolve to one row).
        ref_key = _identity_ref(c.ref)
        if ref_key in committed_refs:
            errors.append(f"{c.ref}: already sense-bound in the committed ledger")
            continue
        if ref_key in seen:
            errors.append(f"{c.ref}: duplicated in this batch")
            continue
        seen.add(ref_key)
        errors.extend(_grounding_errors(conn, eid, c))
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
        # Hub id is stable per (morpheme, label, position): idx is part of the
        # hashed identity, so re-running with the same label at the same idx reuses
        # the hub (re-author safe for the common monosemous idx=0 case), but a
        # polysemous re-run that emits the labels in a different order lands a label
        # at a different idx and mints a different hub.
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


def committed_sense_refs(assertions: Iterable[Assertion]) -> set[str]:
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
                # De-dash to the resolved-etymon identity so the "done" set matches
                # the validate-side dedup key (else a committed bare ref wouldn't
                # block a later dashed variant of the same morpheme).
                done.add(_identity_ref(parts[0]))
    return done
