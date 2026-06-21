"""Rail for the scholar-morpheme enrichment campaign (wyrd-eni4.1).

The campaign raises decomposition COVERAGE by giving the ~8,939 etymons the
scholarly toponym breakdowns reference a reflex the matcher can emit/recognize
(only 14.5% had one at baseline). This module is the harness the 30-minute
enrichment loop rides — two primitives plus a progress read-out:

* :func:`next_slice` — the next ``n`` scholar etymons still missing a reflex,
  **impact-ordered** (descending by the number of distinct toponyms the etymon
  appears in, so the highest-coverage morphemes go first), each with the
  grounding evidence the loop needs to author a reflex (the toponyms it appears
  in + their ``surface_in_modern`` spans + the etymon's glosses + what it
  already has).

* :func:`validate_candidates` — the gate run on freshly-authored reflex records
  BEFORE they are appended/committed. The load-bearing check is the **grounding
  guard**: a reflex ``surface_form`` must actually appear (folded) in an
  attested toponym form of a referenced etymon. That is what stops a week of
  unattended runs from inventing plausible-but-spurious reflexes (the
  precision-wrecking ``-ston``→stone class the grader exists to catch).

Two invariants this module enforces by construction:

1. **Remainder from the committed ledger, never the live DB.** "Which etymons
   still need a reflex" is computed from the etymon refs present in the
   committed ``_reflexes.jsonl`` — NOT from ``reflex_etymon`` in the working
   ``lexicon.db``, which carries locally-applied-but-uncommitted reflexes and
   would yield a wrong, non-reproducible remainder (wyrd-1rw4 reproducibility
   trap).

2. **The safe scoped path.** Reflexes are authored as ``_reflexes.jsonl`` rows
   (``surface_form`` / ``position`` / ``etymon_refs``) — the scoped, reviewable
   ledger that shipped at support>=3 — NEVER via ``dump_all_sources`` (blocked
   by the wyrd-tg0f durability bug that drops 7,327 attestation rows).

The reflex row shape + ``language:canonical_form`` ref grammar match the replay
contract in :func:`wyrd.generators.kenning.jsonl.build._insert_reflex_rows`
exactly, so an authored row round-trips through a clean rebuild.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.lexicon.etymon_refs import etymon_ref, resolve_etymon_ref
from wyrd.generators.kenning.lexicon.genitive_priors import fold_surface

# The reflex replay accepts exactly these positions (build._insert_reflex_rows
# upserts (surface_form, position); the matcher reads them by position class).
VALID_POSITIONS = frozenset({"pre", "post", "inner"})

# Cap the toponym evidence per etymon — enough to ground a reflex without
# dumping hundreds of rows the loop can't read.
_EVIDENCE_TOPONYM_LIMIT = 25


@dataclass(frozen=True)
class EtymonEvidence:
    """Everything the loop needs to author a grounded reflex for one etymon."""

    etymon_id: int
    ref: str  # "language:canonical_form" — copy verbatim into etymon_refs
    language: str
    canonical_form: str
    impact: int  # distinct toponyms the scholar breakdowns attribute to it
    glosses: list[str] = field(default_factory=list)
    toponyms: list[dict[str, Any]] = field(
        default_factory=list
    )  # modern_name/historical_form/surface_in_modern
    existing_reflexes: list[dict[str, str]] = field(default_factory=list)
    existing_variants: list[str] = field(default_factory=list)
    existing_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "etymon_id": self.etymon_id,
            "ref": self.ref,
            "language": self.language,
            "canonical_form": self.canonical_form,
            "impact": self.impact,
            "glosses": self.glosses,
            "toponyms": self.toponyms,
            "existing_reflexes": self.existing_reflexes,
            "existing_variants": self.existing_variants,
            "existing_tags": self.existing_tags,
        }


# ---------------------------------------------------------------------------
# Committed-ledger remainder (invariant 1).
# ---------------------------------------------------------------------------


def committed_reflex_refs(reflexes_path: Path) -> set[str]:
    """The set of ``etymon_refs`` present in the committed ``_reflexes.jsonl``.

    An etymon "already has a reflex" (for remainder purposes) iff its
    ``language:canonical_form`` ref appears here. Read from the LEDGER, not the
    DB — see invariant 1 in the module docstring.
    """
    refs: set[str] = set()
    if not reflexes_path.exists():
        return refs
    with reflexes_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_type") != "reflex":
                continue
            for ref in row.get("etymon_refs", []) or []:
                refs.add(ref)
    return refs


def committed_reflex_keys(reflexes_path: Path) -> set[tuple[str, str, str]]:
    """``(surface_form, position, ref)`` triples already in the ledger — the
    dedup key so a re-authored reflex is rejected as a duplicate."""
    keys: set[tuple[str, str, str]] = set()
    if not reflexes_path.exists():
        return keys
    with reflexes_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_type") != "reflex":
                continue
            sf = row.get("surface_form")
            pos = row.get("position")
            for ref in row.get("etymon_refs", []) or []:
                keys.add((sf, pos, ref))
    return keys


# ---------------------------------------------------------------------------
# Scholar etymon ranking + evidence.
# ---------------------------------------------------------------------------


# Scope filter: a morpheme target must be a real, single-token, attributed
# element. We exclude:
#   * language = 'unknown' — explicit non-attributions ('Place-name', 'Obscure
#     element' placeholders; ~8 etymons, all junk), and
#   * multi-word canonical_forms (a space) — phrases/names, not morphemes
#     ('Bartholomew County', 'Obscure element'), which have no single reflex.
# Everything else (incl. dashed connective forms like '-ingtūn') stays in scope;
# the grounding guard handles per-reflex correctness.
_IMPACT_SQL = """
    SELECT tee.etymon_id AS eid,
           e.language     AS language,
           e.canonical_form AS canonical_form,
           COUNT(DISTINCT te.toponym_id) AS impact
    FROM toponym_etymology_element tee
    JOIN toponym_etymology te ON te.id = tee.toponym_etymology_id
    JOIN etymon e ON e.id = tee.etymon_id
    WHERE tee.etymon_id IS NOT NULL
      AND e.language <> 'unknown'
      AND e.canonical_form NOT LIKE '% %'
    GROUP BY tee.etymon_id, e.language, e.canonical_form
    ORDER BY impact DESC, e.language, e.canonical_form
"""


def scholar_impact_ranking(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every scholar-referenced etymon with its impact (distinct toponym count),
    most impactful first. Deterministic tie-break on (language, canonical_form)."""
    return list(conn.execute(_IMPACT_SQL))


def etymon_evidence(conn: sqlite3.Connection, eid: int, *, impact: int) -> EtymonEvidence:
    """Pull the grounding evidence for one etymon (called only for a slice)."""
    erow = conn.execute(
        "SELECT language, canonical_form FROM etymon WHERE id = ?", (eid,)
    ).fetchone()
    language = erow["language"]
    canonical_form = erow["canonical_form"]

    glosses = [
        r["gloss"]
        for r in conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (eid,))
    ]
    toponyms = [
        {
            "modern_name": r["modern_name"],
            "historical_form": r["historical_form"],
            "surface_in_modern": r["surface_in_modern"],
        }
        for r in conn.execute(
            """
            SELECT DISTINCT t.modern_name AS modern_name,
                   te.historical_form AS historical_form,
                   tee.surface_in_modern AS surface_in_modern
            FROM toponym_etymology_element tee
            JOIN toponym_etymology te ON te.id = tee.toponym_etymology_id
            JOIN toponym t ON t.id = te.toponym_id
            WHERE tee.etymon_id = ?
            LIMIT ?
            """,
            (eid, _EVIDENCE_TOPONYM_LIMIT),
        )
    ]
    existing_reflexes = [
        {"surface_form": r["surface_form"], "position": r["position"]}
        for r in conn.execute(
            """
            SELECT rf.surface_form AS surface_form, rf.position AS position
            FROM reflex_etymon re JOIN reflex rf ON rf.id = re.reflex_id
            WHERE re.etymon_id = ?
            """,
            (eid,),
        )
    ]
    existing_variants = [
        r["form"]
        for r in conn.execute("SELECT form FROM etymon_variant WHERE etymon_id = ?", (eid,))
    ]
    existing_tags = [
        r["tag"] for r in conn.execute("SELECT tag FROM etymon_tag WHERE etymon_id = ?", (eid,))
    ]
    return EtymonEvidence(
        etymon_id=eid,
        ref=etymon_ref(language, canonical_form),
        language=language,
        canonical_form=canonical_form,
        impact=impact,
        glosses=glosses,
        toponyms=toponyms,
        existing_reflexes=existing_reflexes,
        existing_variants=existing_variants,
        existing_tags=existing_tags,
    )


def parked_refs(parked_path: Path | None) -> set[str]:
    """Refs the loop has deliberately PARKED — etymons with no confident grounded
    surface (foreign lemmas, gloss-heads). Each parked-ledger line is
    ``{"ref": "<language:canonical_form>", "reason": "..."}``. Parked refs are
    excluded from future slices so the loop always makes forward progress instead
    of re-emitting the same un-authorable top-impact etymon forever.
    """
    refs: set[str] = set()
    if parked_path is None or not parked_path.exists():
        return refs
    with parked_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ref = json.loads(line).get("ref")
            if ref:
                refs.add(ref)
    return refs


def next_slice(
    conn: sqlite3.Connection,
    reflexes_path: Path,
    *,
    n: int = 20,
    parked_path: Path | None = None,
) -> list[EtymonEvidence]:
    """The next ``n`` impact-ordered scholar etymons still missing a committed
    reflex AND not parked, each with grounding evidence. Empty when the phase is
    exhausted (every remaining etymon is either done or parked)."""
    excluded = committed_reflex_refs(reflexes_path) | parked_refs(parked_path)
    out: list[EtymonEvidence] = []
    for row in scholar_impact_ranking(conn):
        ref = etymon_ref(row["language"], row["canonical_form"])
        if ref in excluded:
            continue
        out.append(etymon_evidence(conn, row["eid"], impact=row["impact"]))
        if len(out) >= n:
            break
    return out


def reflex_progress(
    conn: sqlite3.Connection,
    reflexes_path: Path,
    *,
    parked_path: Path | None = None,
) -> tuple[int, int, int]:
    """``(done, parked, total)`` over the scholar etymon set — ``done`` carry a
    committed reflex, ``parked`` were skipped as un-groundable, and
    ``total - done - parked`` remain to attempt."""
    ranking = scholar_impact_ranking(conn)
    total = len(ranking)
    done_refs = committed_reflex_refs(reflexes_path)
    parked = parked_refs(parked_path)
    done = 0
    parked_count = 0
    for r in ranking:
        ref = etymon_ref(r["language"], r["canonical_form"])
        if ref in done_refs:
            done += 1
        elif ref in parked:
            parked_count += 1
    return done, parked_count, total


# ---------------------------------------------------------------------------
# Validation gate (the grounding guard).
# ---------------------------------------------------------------------------


def _attested_folds(conn: sqlite3.Connection, eid: int) -> set[str]:
    """Folded attested toponym forms (modern_name / historical_form /
    surface_in_modern) for an etymon — the haystack the grounding guard
    requires a reflex surface to appear in."""
    folds: set[str] = set()
    for r in conn.execute(
        """
        SELECT t.modern_name AS modern_name,
               te.historical_form AS historical_form,
               tee.surface_in_modern AS surface_in_modern
        FROM toponym_etymology_element tee
        JOIN toponym_etymology te ON te.id = tee.toponym_etymology_id
        JOIN toponym t ON t.id = te.toponym_id
        WHERE tee.etymon_id = ?
        """,
        (eid,),
    ):
        for val in (r["modern_name"], r["historical_form"], r["surface_in_modern"]):
            folded = fold_surface(val)
            if folded:
                folds.add(folded)
    return folds


def _shape_error(rec: dict[str, Any], tag: str) -> str | None:
    """The first structural problem with a candidate row, or None if well-formed.

    Well-formed = ``_type == "reflex"``, ``surface_form`` a non-empty string
    that folds to >=1 letter, ``position`` in {pre, post, inner}, and
    ``etymon_refs`` a non-empty list."""
    if rec.get("_type") != "reflex":
        return f"{tag}: _type must be 'reflex' (got {rec.get('_type')!r})"
    sf = rec.get("surface_form")
    if not isinstance(sf, str) or not sf.strip():
        return f"{tag}: surface_form must be a non-empty string"
    if rec.get("position") not in VALID_POSITIONS:
        return f"{tag}: position {rec.get('position')!r} not in {sorted(VALID_POSITIONS)}"
    refs = rec.get("etymon_refs") or []
    if not isinstance(refs, list) or not refs:
        return f"{tag}: etymon_refs must be a non-empty list"
    if not fold_surface(sf):
        return f"{tag}: surface_form {sf!r} folds to empty (no letters)"
    return None


def _validate_one(
    conn: sqlite3.Connection,
    rec: dict[str, Any],
    tag: str,
    *,
    scholar_eids: set[int],
    scholar_only: bool,
    committed: set[tuple[str, str, str]],
    seen_in_batch: set[tuple[str, str, str]],
) -> list[str]:
    """Validate one candidate reflex row; return its errors (empty = passed)."""
    shape = _shape_error(rec, tag)
    if shape is not None:
        return [shape]

    sf = rec["surface_form"]
    pos = rec["position"]
    refs = rec["etymon_refs"]
    folded_surface = fold_surface(sf)

    errors: list[str] = []
    resolved_eids: list[int] = []
    for ref in refs:
        eid = resolve_etymon_ref(conn, ref)
        if eid is None:
            errors.append(f"{tag}: etymon_ref {ref!r} resolves to no etymon")
        else:
            resolved_eids.append(eid)
    if not resolved_eids:
        return errors

    if scholar_only and not any(eid in scholar_eids for eid in resolved_eids):
        errors.append(
            f"{tag}: no ref is a scholar-referenced etymon (out of campaign scope): {refs}"
        )
    if not any(
        folded_surface in hay for eid in resolved_eids for hay in _attested_folds(conn, eid)
    ):
        errors.append(
            f"{tag}: grounding guard FAILED — surface {sf!r} (folded {folded_surface!r}) "
            f"appears in no attested toponym form of {refs}"
        )
    for ref in refs:
        key = (sf, pos, ref)
        if key in committed or key in seen_in_batch:
            errors.append(f"{tag}: duplicate reflex (committed or in-batch): {key}")
        seen_in_batch.add(key)
    return errors


def validate_candidates(
    conn: sqlite3.Connection,
    reflexes_path: Path,
    candidates: Sequence[dict[str, Any]],
    *,
    scholar_only: bool = True,
) -> list[str]:
    """Validate authored reflex candidate rows. Returns a list of human-readable
    errors — empty means every record passed and is safe to append.

    Checks per record (see :func:`_validate_one`):
      * shape: ``_type == "reflex"``, non-empty ``surface_form``,
        ``position`` in {pre, post, inner}, non-empty ``etymon_refs``;
      * every ref resolves to a real etymon via the natural key (an unresolved
        ref would silently orphan at replay — build skips it — so the reflex
        would link to nothing);
      * campaign scope (``scholar_only``): at least one ref is a
        scholar-referenced etymon;
      * GROUNDING GUARD: the folded ``surface_form`` is a substring of >=1
        folded attested toponym form of a referenced etymon;
      * dedup: ``(surface_form, position, ref)`` not already committed, and not
        duplicated within the candidate batch.
    """
    scholar_eids = {r["eid"] for r in scholar_impact_ranking(conn)} if scholar_only else set()
    committed = committed_reflex_keys(reflexes_path)
    seen_in_batch: set[tuple[str, str, str]] = set()
    errors: list[str] = []
    for i, rec in enumerate(candidates):
        errors.extend(
            _validate_one(
                conn,
                rec,
                f"candidate[{i}]",
                scholar_eids=scholar_eids,
                scholar_only=scholar_only,
                committed=committed,
                seen_in_batch=seen_in_batch,
            )
        )
    return errors
