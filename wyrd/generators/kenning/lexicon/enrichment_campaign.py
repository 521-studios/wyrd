"""Rail for the scholar-morpheme enrichment campaign (wyrd-eni4.1).

The campaign raises decomposition COVERAGE by giving the etymons the scholarly
toponym breakdowns reference a reflex the matcher can emit/recognize. This
module is the harness the enrichment loop rides — the reflex/tag/IPA/gloss
selectors and validators, band-status, collapse validation, and identity-reflex
emission. The reflex rail's two anchors:

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
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.lexicon import tag_mining
from wyrd.generators.kenning.lexicon.etymon_refs import etymon_ref, resolve_etymon_ref
from wyrd.generators.kenning.lexicon.genitive_priors import fold_surface

_TAG_VOCAB_SET = frozenset(tag_mining.TAG_VOCAB)
# An IPA value must be slash- or bracket-delimited (/.../ or [...]) and non-empty
# inside — the same convention the existing _pronunciation.jsonl rows use.
_IPA_RE = re.compile(r"^[/\[].+[/\]]$")


def _nfc(ref: str) -> str:
    """NFC-normalize a ref for comparison. ``etymon.canonical_form`` is stored
    inconsistently — some rows NFC (precomposed ``ō`` U+014D), some NFD (``o`` +
    combining macron U+0304) — so the raw ``language:canonical_form`` ref a slice
    emits and a ref typed by hand into ``park`` can be byte-different for the same
    morpheme. Comparing under NFC makes remainder/dedup/park exclusion robust to
    that, so a parked diacritic etymon actually drops out (else it re-emits every
    slice — the stall park exists to prevent)."""
    return unicodedata.normalize("NFC", ref)


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
                refs.add(_nfc(ref))
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
                keys.add((sf, pos, _nfc(ref)))
    return keys


# SQL: every scope-clean scholar etymon with its attested toponym surface strings
# (modern_name | historical_form | surface_in_modern), one row per etymon.
# ``{gloss_filter}`` is the gloss gate (see identity_reflex_rows).
_IDENTITY_SQL = """
  SELECT e.id AS eid, e.language AS lang, e.canonical_form AS cf,
    GROUP_CONCAT(
      COALESCE(t.modern_name,'')||'|'||COALESCE(te.historical_form,'')||'|'||COALESCE(tee.surface_in_modern,''),
      '~') AS forms
  FROM etymon e
  JOIN toponym_etymology_element tee ON tee.etymon_id = e.id
  JOIN toponym_etymology te ON te.id = tee.toponym_etymology_id
  JOIN toponym t ON t.id = te.toponym_id
  WHERE e.language <> 'unknown' AND e.canonical_form NOT LIKE '% %'
  {gloss_filter}
  GROUP BY e.id
"""


def identity_reflex_rows(
    conn: sqlite3.Connection, reflexes_path: Path, *, min_len: int = 2, require_gloss: bool = True
) -> list[dict[str, Any]]:
    """One reflex per scholar morpheme whose **own folded canonical form** appears
    in one of its attested toponym strings — the *identity reflex* (the morpheme
    recognized by its base spelling, e.g. ``stān`` → ``stan`` so ``Stanton``
    recovers it). Grounded by construction (the surface IS in a toponym), deduped
    against the committed ledger, position derived from the matching span. Skips
    surfaces shorter than ``min_len`` folded chars (avoids 1-char over-matchers).

    ``require_gloss`` (default True) is the **legitimacy gate**: blanket-asserting
    that a morpheme is matchable-by-its-own-name is only safe for morphemes we've
    confirmed are real, and a gloss is that confirmation. Unglossed morphemes are
    excluded — they get per-morpheme judgment in the normal loop instead (worn
    reflex / gloss / park), so we don't inject low-confidence matchables (the
    CAN-IT false-positive risk). The bulk/retroactive complement to per-toponym
    worn-form authoring: every *understood* morpheme is a reflex of itself."""
    gloss_filter = (
        "AND EXISTS (SELECT 1 FROM etymon_gloss g WHERE g.etymon_id = e.id)"
        if require_gloss
        else ""
    )
    existing = committed_reflex_keys(reflexes_path)
    out: list[dict[str, Any]] = []
    for r in conn.execute(_IDENTITY_SQL.format(gloss_filter=gloss_filter)):
        fc = fold_surface(r["cf"] or "")
        if len(fc) < min_len:
            continue
        position = None
        for occ in (r["forms"] or "").split("~"):
            for fld in occ.split("|"):
                ff = fold_surface(fld)
                if fc and fc in ff:
                    position = (
                        "pre" if ff.startswith(fc) else "post" if ff.endswith(fc) else "inner"
                    )
                    break
            if position:
                break
        if position is None:
            continue
        ref = etymon_ref(r["lang"], r["cf"])
        if (fc, position, _nfc(ref)) in existing:
            continue
        out.append(
            {"_type": "reflex", "surface_form": fc, "position": position, "etymon_refs": [ref]}
        )
    return out


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


def _etymon_reflex_folds(conn: sqlite3.Connection) -> dict[int, set[str]]:
    """etymon_id → its folded reflex surfaces (every reflex position). The
    surface inventory the matcher can emit/recognize for that morpheme."""
    folds: dict[int, set[str]] = {}
    for row in conn.execute(
        "SELECT re.etymon_id AS eid, r.surface_form AS sf "
        "FROM reflex r JOIN reflex_etymon re ON re.reflex_id = r.id"
    ):
        f = fold_surface(row["sf"] or "")
        if f:
            folds.setdefault(row["eid"], set()).add(f)
    return folds


def variant_gap_census(conn: sqlite3.Connection) -> dict[str, int]:
    """Classify every scholar (toponym, morpheme) pair by reflex-matchability —
    the wyrd-eni4.3.1 diagnostic that sizes the variant-gap lever:

    * ``matched``     — the morpheme has a reflex surface that substring-matches
      (folded) THIS toponym's modern name;
    * ``variant_gap`` — the morpheme HAS reflex surface(s), but none match this
      toponym's spelling (the 'barthos' class: covered, yet blocks CAN-IT here);
    * ``reflex_less`` — the morpheme has no reflex surface at all.

    Returns counts keyed ``matched`` / ``variant_gap`` / ``reflex_less`` /
    ``total``. Pure folded-substring test — no matcher/trie, so it's a cheap
    deterministic census, not a grade."""
    folds = _etymon_reflex_folds(conn)
    counts = {"matched": 0, "variant_gap": 0, "reflex_less": 0, "total": 0}
    for row in conn.execute(
        "SELECT t.modern_name AS name, tee.etymon_id AS eid "
        "FROM toponym_etymology te "
        "JOIN toponym t ON t.id = te.toponym_id "
        "JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id"
    ):
        counts["total"] += 1
        surfaces = folds.get(row["eid"])
        if not surfaces:
            counts["reflex_less"] += 1
            continue
        folded_name = fold_surface(row["name"] or "")
        if any(s in folded_name for s in surfaces):
            counts["matched"] += 1
        else:
            counts["variant_gap"] += 1
    return counts


@dataclass(frozen=True)
class VariantGapTask:
    """A scholar morpheme that HAS reflex surface(s), none of which substring-match
    some toponym(s) it appears in — the wyrd-eni4.3.1 lever. The loop authors a NEW
    grounded reflex surface (the worn span in such a toponym) for it.

    ``flips_can_it`` is True when the morpheme is the SOLE unmatched morpheme in at
    least one toponym, so a single authored surface flips that toponym to CAN-IT.
    ``gap_toponyms`` is how many of its toponyms it's variant-gap in (frequency)."""

    ref: str
    language: str
    canonical_form: str
    glosses: list[str] = field(default_factory=list)
    existing_surfaces: list[str] = field(default_factory=list)
    flips_can_it: bool = False
    gap_toponyms: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "language": self.language,
            "canonical_form": self.canonical_form,
            "glosses": self.glosses,
            "existing_surfaces": self.existing_surfaces,
            "flips_can_it": self.flips_can_it,
            "gap_toponyms": self.gap_toponyms,
            "evidence": self.evidence,
        }


def _committed_reflex_surfaces_by_ref(reflexes_path: Path | None) -> dict[str, set[str]]:
    """NFC-ref → folded reflex surfaces committed in the JSONL ledger but not yet
    in the DB. The loop has no rebuild between fires, so a surface authored this
    campaign lives only here; folding it in stops the selector re-emitting the same
    morpheme every fire."""
    out: dict[str, set[str]] = {}
    if reflexes_path is None or not Path(reflexes_path).exists():
        return out
    with Path(reflexes_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not (isinstance(row, dict) and row.get("_type") == "reflex"):
                continue
            f = fold_surface(row.get("surface_form") or "")
            if not f:
                continue
            for ref in row.get("etymon_refs") or []:
                out.setdefault(_nfc(ref), set()).add(f)
    return out


def _residual_span(folded_name: str, covered: list[str]) -> str:
    """Best-effort worn-span hint: ``folded_name`` with each covered surface's
    first occurrence blanked, returning the longest remaining contiguous letter
    run — where the unmatched morpheme most likely lives. A hint for the author,
    not a guarantee."""
    s = folded_name
    for c in sorted(covered, key=len, reverse=True):
        i = s.find(c)
        if i >= 0:
            s = s[:i] + " " + s[i + len(c) :]
    frags = [f for f in s.split() if f]
    return max(frags, key=len) if frags else ""


def _collect_variant_gaps(
    conn: sqlite3.Connection,
    surfaces_for: Callable[[int], set[str]],
    allowed: set[int],
    max_evidence: int,
) -> dict[int, dict[str, Any]]:
    """Per scope-clean morpheme, accumulate its variant-gap evidence across the
    toponyms it appears in: a flip flag (it's the SOLE unmatched morpheme there),
    a gap count, and up to ``max_evidence`` sample toponyms. A morpheme is
    variant-gap in a toponym when it has reflex surface(s) but none substring-match
    the folded name."""
    topo_eids: dict[str, set[int]] = {}
    for row in conn.execute(
        "SELECT t.modern_name AS name, tee.etymon_id AS eid "
        "FROM toponym_etymology te JOIN toponym t ON t.id = te.toponym_id "
        "JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id"
    ):
        topo_eids.setdefault(row["name"], set()).add(row["eid"])

    agg: dict[int, dict[str, Any]] = {}
    for name, eids in topo_eids.items():
        folded_name = fold_surface(name or "")
        if not folded_name:
            continue
        covered: list[str] = []
        gap_eids: list[int] = []
        n_unmatched = 0
        for eid in eids:
            surfs = surfaces_for(eid)
            matched = [s for s in surfs if s in folded_name]
            if matched:
                covered.extend(matched)
                continue
            n_unmatched += 1
            if surfs and eid in allowed:
                gap_eids.append(eid)
        for eid in gap_eids:
            a = agg.setdefault(eid, {"flips": False, "count": 0, "evidence": []})
            a["flips"] = a["flips"] or n_unmatched == 1
            a["count"] += 1
            if len(a["evidence"]) < max_evidence:
                a["evidence"].append(
                    {
                        "modern_name": name,
                        "covered_spans": sorted(set(covered)),
                        "residual_span": _residual_span(folded_name, covered),
                        "is_flip": n_unmatched == 1,
                    }
                )
    return agg


def variant_gap_next_slice(
    conn: sqlite3.Connection,
    reflexes_path: Path | None,
    *,
    n: int = 20,
    impact: int | None = None,
    parked_path: Path | None = None,
    max_evidence: int = 5,
) -> list[VariantGapTask]:
    """The next ``n`` scope-clean scholar morphemes that are VARIANT-GAP in one or
    more toponyms (have a reflex, none matching that spelling), prioritized
    flip-CAN-IT-first then by gap-toponym frequency. Per-etymon reflex surfaces are
    the DB reflexes UNIONED with the committed ``_reflexes.jsonl`` ledger. Parked
    refs (``parked_path``, the shared reflex park ledger) are excluded so an
    un-authorable morpheme doesn't re-surface every fire. Each task carries evidence
    toponyms (modern name, covered spans, a residual-span hint) for the author to
    ground a new reflex surface against."""
    db_folds = _etymon_reflex_folds(conn)
    ledger = _committed_reflex_surfaces_by_ref(reflexes_path)
    parked = parked_refs(parked_path)

    meta: dict[int, sqlite3.Row] = {}
    eid_ref: dict[int, str] = {}
    for r in scholar_impact_ranking(conn):
        meta[r["eid"]] = r
        eid_ref[r["eid"]] = etymon_ref(r["language"], r["canonical_form"])

    def surfaces_for(eid: int) -> set[str]:
        s = set(db_folds.get(eid, set()))
        ref = eid_ref.get(eid)
        if ref is not None:
            s |= ledger.get(_nfc(ref), set())
        return s

    allowed = {eid for eid in meta if _nfc(eid_ref[eid]) not in parked}
    agg = _collect_variant_gaps(conn, surfaces_for, allowed, max_evidence)

    tasks: list[VariantGapTask] = []
    for eid, a in agg.items():
        if impact is not None and meta[eid]["impact"] != impact:
            continue
        a["evidence"].sort(key=lambda e: (not e["is_flip"], e["modern_name"]))
        glosses = [
            g["gloss"]
            for g in conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (eid,))
        ]
        tasks.append(
            VariantGapTask(
                ref=eid_ref[eid],
                language=meta[eid]["language"],
                canonical_form=meta[eid]["canonical_form"],
                glosses=glosses,
                existing_surfaces=sorted(surfaces_for(eid)),
                flips_can_it=a["flips"],
                gap_toponyms=a["count"],
                evidence=a["evidence"],
            )
        )
    tasks.sort(key=lambda t: (not t.flips_can_it, -t.gap_toponyms, t.ref))
    return tasks[:n]


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
                refs.add(_nfc(ref))
    return refs


def next_slice(
    conn: sqlite3.Connection,
    reflexes_path: Path,
    *,
    n: int = 20,
    parked_path: Path | None = None,
    impact: int | None = None,
) -> list[EtymonEvidence]:
    """The next ``n`` impact-ordered scholar etymons still missing a committed
    reflex AND not parked, each with grounding evidence. Empty when the phase is
    exhausted (every remaining etymon is either done or parked). ``impact``
    restricts to a single impact band (the band-by-band holistic loop, wyrd-eni4.3)."""
    excluded = committed_reflex_refs(reflexes_path) | parked_refs(parked_path)
    out: list[EtymonEvidence] = []
    for row in scholar_impact_ranking(conn):
        if impact is not None and row["impact"] != impact:
            continue
        ref = etymon_ref(row["language"], row["canonical_form"])
        if _nfc(ref) in excluded:
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
        ref = _nfc(etymon_ref(r["language"], r["canonical_form"]))
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
    if sf.strip().startswith("-") or sf.strip().endswith("-"):
        return (
            f"{tag}: surface_form {sf!r} must be a bare surface — an affix-position "
            f"dash ('-ton' / 'ton-') is never stored identity (D45); the slot lives "
            f"in the separate 'position' field"
        )
    if rec.get("position") not in VALID_POSITIONS:
        return f"{tag}: position {rec.get('position')!r} not in {sorted(VALID_POSITIONS)}"
    refs = rec.get("etymon_refs") or []
    if not isinstance(refs, list) or not refs:
        return f"{tag}: etymon_refs must be a non-empty list"
    if not fold_surface(sf):
        return f"{tag}: surface_form {sf!r} folds to empty (no letters)"
    return None


def _dedup_errors(
    sf: str,
    pos: str,
    refs: list[str],
    tag: str,
    committed: set[tuple[str, str, str]],
    seen_in_batch: set[tuple[str, str, str]],
) -> list[str]:
    """Per-ref dedup keys for an otherwise-valid row, recording each in
    ``seen_in_batch``. Only valid rows reach here — one that failed an earlier
    check won't be written, so it must not pollute the batch set."""
    errors: list[str] = []
    for ref in refs:
        key = (sf, pos, _nfc(ref))
        if key in committed or key in seen_in_batch:
            errors.append(f"{tag}: duplicate reflex (committed or in-batch): {key}")
        seen_in_batch.add(key)
    return errors


def _validate_one(
    conn: sqlite3.Connection,
    rec: dict[str, Any],
    tag: str,
    *,
    scholar_eids: set[int],
    scholar_only: bool,
    committed: set[tuple[str, str, str]],
    seen_in_batch: set[tuple[str, str, str]],
    folds_cache: dict[int, set[str]],
) -> list[str]:
    """Validate one candidate reflex row; return its errors (empty = passed).

    ``folds_cache`` memoises :func:`_attested_folds` per etymon id across the
    batch, so the grounding guard doesn't re-query the same etymon once per
    candidate that references it."""
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
            if eid not in folds_cache:
                folds_cache[eid] = _attested_folds(conn, eid)
    if not resolved_eids:
        return errors

    if scholar_only and not any(eid in scholar_eids for eid in resolved_eids):
        errors.append(
            f"{tag}: no ref is a scholar-referenced etymon (out of campaign scope): {refs}"
        )
    if not any(folded_surface in hay for eid in resolved_eids for hay in folds_cache[eid]):
        errors.append(
            f"{tag}: grounding guard FAILED — surface {sf!r} (folded {folded_surface!r}) "
            f"appears in no attested toponym form of {refs}"
        )
    # Only dedup an otherwise-valid row: one that already failed grounding/scope
    # won't be written, so it must neither pollute seen_in_batch nor draw a
    # second, spurious "duplicate reflex" error on top of its real one.
    if not errors:
        errors.extend(_dedup_errors(sf, pos, refs, tag, committed, seen_in_batch))
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
    if not candidates:
        return []
    scholar_eids = {r["eid"] for r in scholar_impact_ranking(conn)} if scholar_only else set()
    committed = committed_reflex_keys(reflexes_path)
    seen_in_batch: set[tuple[str, str, str]] = set()
    folds_cache: dict[int, set[str]] = {}
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
                folds_cache=folds_cache,
            )
        )
    return errors


# ---------------------------------------------------------------------------
# Tag uplift (phase 2 — the accepted-morpheme quality track, wyrd-eni4.2.3).
# ---------------------------------------------------------------------------
#
# The controlled vocabulary is reused wholesale from tag_mining.TAG_VOCAB. The
# cohort, however, is computed HERE rather than via tag_mining.select_targets:
# that function runs a correlated admit-count subquery for every glossed etymon
# in the 2.4M-row table (a one-time paid-mine cost it can absorb, but minutes per
# call — unusable in a 30-min loop). Instead we scope to the admit ranking (the
# same fast GROUP BY over the ~43k breakdown rows that _IMPACT_SQL already does)
# and add cheap indexed gloss/tag EXISTS over the ~3k-row cohort. The loop's only
# real work is classifying each gloss into the vocab and recording the decision
# (incl. the 'none' outcome) to data/mining/_tags.jsonl — the same ledger
# mine-tags-llm writes, replayed for free by the apply-tag-additions pass.

# admit cohort (>=2 distinct toponyms) that is glossed but untagged, impact-first.
_TAG_COHORT_SQL = """
    WITH admit AS (
        SELECT tee.etymon_id AS eid,
               e.language AS language,
               e.canonical_form AS canonical_form,
               COUNT(DISTINCT te.toponym_id) AS impact
        FROM toponym_etymology_element tee
        JOIN toponym_etymology te ON te.id = tee.toponym_etymology_id
        JOIN etymon e ON e.id = tee.etymon_id
        WHERE tee.etymon_id IS NOT NULL
          AND e.language <> 'unknown'
          AND e.canonical_form NOT LIKE '% %'
          -- wyrd-tze2: exclude merged/tombstoned etymons (the #804 filter, ported
          -- from tag_mining.select_targets). A bare-surface stub folded into its
          -- canonical (wick->wīc) must not be re-tagged on a later campaign slice;
          -- tagging a tombstone re-creates the stub-out-ranks-canonical ranking bug.
          AND e.merged_into_id IS NULL
        GROUP BY tee.etymon_id, e.language, e.canonical_form
        HAVING impact >= 2
    )
    SELECT a.eid, a.language, a.canonical_form, a.impact,
           (SELECT GROUP_CONCAT(g.gloss, ' || ')
            FROM etymon_gloss g WHERE g.etymon_id = a.eid) AS glosses
    FROM admit a
    WHERE EXISTS (SELECT 1 FROM etymon_gloss g WHERE g.etymon_id = a.eid)
      AND NOT EXISTS (SELECT 1 FROM etymon_tag t WHERE t.etymon_id = a.eid)
    ORDER BY a.impact DESC, a.language, a.canonical_form
"""


def tag_cohort(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Admit-cohort (>=2-toponym) etymons that are glossed but untagged in the
    DB, impact-first. The DB-tag filter is equivalent to the committed-ledger
    remainder on a clean rebuild (DB tags == replayed _tags.jsonl)."""
    return list(conn.execute(_TAG_COHORT_SQL))


def tag_next_slice(
    conn: sqlite3.Connection,
    db_path: Path,
    tags_path: Path,
    *,
    n: int = 20,
    impact: int | None = None,
) -> list[dict[str, Any]]:
    """The next ``n`` admit-cohort etymons that have a gloss but no tag (and no
    decision yet in the committed ``_tags.jsonl``), impact-ordered, each with its
    glosses for classification. Empty when the cohort is exhausted. ``impact``
    restricts to a single band. ``db_path`` is accepted for signature parity with
    the reflex rail; the cohort comes from ``conn``."""
    del db_path  # cohort is computed from conn, not a fresh path-opened connection
    done = {_nfc(r) for r in tag_mining.existing_refs(str(tags_path))}
    targets: list[dict[str, Any]] = []
    for row in tag_cohort(conn):
        if impact is not None and row["impact"] != impact:
            continue
        ref = etymon_ref(row["language"], row["canonical_form"])
        if _nfc(ref) in done:
            continue
        targets.append(
            {
                "ref": ref,
                "language": row["language"],
                "canonical_form": row["canonical_form"],
                "glosses": row["glosses"],
                "impact": row["impact"],
                "vocab": list(tag_mining.TAG_VOCAB),
            }
        )
        if len(targets) >= n:
            break
    return targets


def validate_tag_candidates(
    conn: sqlite3.Connection,
    tags_path: Path,
    candidates: Sequence[dict[str, Any]],
) -> list[str]:
    """Validate authored tag-decision rows. Checks per record: ``_type ==
    "tags"``; ``ref`` resolves to a real etymon; every tag is in the controlled
    vocabulary (an empty list is the valid 'none' outcome); no duplicate decision
    for a ref already committed or seen in this batch."""
    done = {_nfc(r) for r in tag_mining.existing_refs(str(tags_path))}
    seen: set[str] = set()
    errors: list[str] = []
    for i, rec in enumerate(candidates):
        tag = f"candidate[{i}]"
        if rec.get("_type") != "tags":
            errors.append(f"{tag}: _type must be 'tags' (got {rec.get('_type')!r})")
            continue
        ref = rec.get("ref")
        tags = rec.get("tags")
        if not isinstance(ref, str) or resolve_etymon_ref(conn, ref) is None:
            errors.append(f"{tag}: ref {ref!r} resolves to no etymon")
            continue
        if not isinstance(tags, list):
            errors.append(f"{tag}: tags must be a list (use [] for the 'none' outcome)")
            continue
        bad = [t for t in tags if t not in _TAG_VOCAB_SET]
        if bad:
            errors.append(f"{tag}: tags not in controlled vocabulary: {bad}")
        key = _nfc(ref)
        if key in done or key in seen:
            errors.append(f"{tag}: duplicate tag decision for {ref}")
        seen.add(key)
    return errors


def tag_progress(conn: sqlite3.Connection, db_path: Path, tags_path: Path) -> int:
    """Count admit-cohort etymons still awaiting a tag decision (cohort from the
    DB, remainder from the committed ``_tags.jsonl`` — equivalent on a clean
    rebuild where DB tags == the committed ledger)."""
    del db_path  # cohort is computed from conn
    done = {_nfc(r) for r in tag_mining.existing_refs(str(tags_path))}
    return sum(
        1
        for row in tag_cohort(conn)
        if _nfc(etymon_ref(row["language"], row["canonical_form"])) not in done
    )


# ---------------------------------------------------------------------------
# IPA + gloss rails + band capability (the band-by-band holistic loop, wyrd-eni4.3).
#
# The campaign target cohort is the scholar-referenced, scope-clean morphemes
# with impact >= 2 (the same set the snapshot reports on). For each impact band,
# the loop brings every morpheme to full capability across the four loop-owned
# dimensions — reflex, tag, IPA, gloss — before descending. lemma/cognate/variant
# stay human-gated. Each dimension authors to its existing L2 ledger:
#   reflex -> _reflexes.jsonl   tag -> _tags.jsonl
#   IPA    -> _pronunciation.jsonl ({_type:pronunciation, ref, ipa, ...})
#   gloss  -> _curation.jsonl ({_type:etymon_gloss_add, ref, additions:[{gloss}]})
# ---------------------------------------------------------------------------


def ledger_refs(path: Path | None, row_type: str) -> set[str]:
    """NFC refs present in a jsonl ledger for rows of ``row_type`` (e.g.
    ``pronunciation`` in _pronunciation.jsonl, ``etymon_gloss_add`` in
    _curation.jsonl). The committed remainder source for IPA/gloss dimensions."""
    refs: set[str] = set()
    if path is None or not path.exists():
        return refs
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Fail loud on a corrupt line — same as committed_reflex_refs/_keys
            # and parked_refs. Silently skipping a bad line would drop its ref
            # from the "done" set and make the IPA/gloss loop re-emit that etymon
            # every fire with no diagnostic.
            row = json.loads(line)
            if isinstance(row, dict) and row.get("_type") == row_type and row.get("ref"):
                refs.add(_nfc(row["ref"]))
    return refs


def _band_cohort(conn: sqlite3.Connection, impact: int | None) -> list[sqlite3.Row]:
    """Scholar scope-clean etymons with impact >= 2 (optionally a single band),
    impact-first. The campaign target set."""
    return [
        r
        for r in scholar_impact_ranking(conn)
        if r["impact"] >= 2 and (impact is None or r["impact"] == impact)
    ]


def ipa_next_slice(
    conn: sqlite3.Connection,
    ipa_path: Path,
    *,
    n: int = 20,
    impact: int | None = None,
) -> list[dict[str, Any]]:
    """Next ``n`` target morphemes missing pronunciation_ipa (DB) and with no
    committed decision in _pronunciation.jsonl, with glosses as authoring evidence."""
    done = ledger_refs(ipa_path, "pronunciation")
    out: list[dict[str, Any]] = []
    for row in _band_cohort(conn, impact):
        ref = etymon_ref(row["language"], row["canonical_form"])
        if _nfc(ref) in done:
            continue
        has_ipa = conn.execute(
            "SELECT 1 FROM etymon WHERE id=? AND pronunciation_ipa IS NOT NULL "
            "AND pronunciation_ipa<>''",
            (row["eid"],),
        ).fetchone()
        if has_ipa:
            continue
        glosses = [
            g["gloss"]
            for g in conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id=?", (row["eid"],))
        ]
        out.append(
            {
                "ref": ref,
                "language": row["language"],
                "form": row["canonical_form"],
                "impact": row["impact"],
                "glosses": glosses,
            }
        )
        if len(out) >= n:
            break
    return out


def validate_ipa_candidates(
    conn: sqlite3.Connection, ipa_path: Path, candidates: Sequence[dict[str, Any]]
) -> list[str]:
    """Validate authored pronunciation rows: ``_type == "pronunciation"``, ``ref``
    resolves, ``ipa`` is slash/bracket-delimited, no duplicate committed/in-batch."""
    done = ledger_refs(ipa_path, "pronunciation")
    seen: set[str] = set()
    errors: list[str] = []
    for i, rec in enumerate(candidates):
        tag = f"candidate[{i}]"
        if rec.get("_type") != "pronunciation":
            errors.append(f"{tag}: _type must be 'pronunciation'")
            continue
        ref = rec.get("ref")
        ipa = rec.get("ipa")
        if not isinstance(ref, str) or resolve_etymon_ref(conn, ref) is None:
            errors.append(f"{tag}: ref {ref!r} resolves to no etymon")
            continue
        if not isinstance(ipa, str) or not _IPA_RE.match(ipa.strip()):
            errors.append(f"{tag}: ipa {ipa!r} must be /.../ or [...] delimited")
        key = _nfc(ref)
        if key in done or key in seen:
            errors.append(f"{tag}: duplicate pronunciation decision for {ref}")
        seen.add(key)
    return errors


def gloss_next_slice(
    conn: sqlite3.Connection,
    gloss_path: Path,
    *,
    n: int = 20,
    impact: int | None = None,
) -> list[dict[str, Any]]:
    """Next ``n`` target morphemes with NO gloss (DB) and no committed
    ``etymon_gloss_add`` decision, with their toponyms as grounding evidence
    (a gloss must be defensible from how the morpheme is used)."""
    done = ledger_refs(gloss_path, "etymon_gloss_add")
    out: list[dict[str, Any]] = []
    for row in _band_cohort(conn, impact):
        ref = etymon_ref(row["language"], row["canonical_form"])
        if _nfc(ref) in done:
            continue
        has_gloss = conn.execute(
            "SELECT 1 FROM etymon_gloss WHERE etymon_id=?", (row["eid"],)
        ).fetchone()
        if has_gloss:
            continue
        toponyms = [
            tr["modern_name"]
            for tr in conn.execute(
                "SELECT DISTINCT t.modern_name FROM toponym_etymology_element tee "
                "JOIN toponym_etymology te ON te.id=tee.toponym_etymology_id "
                "JOIN toponym t ON t.id=te.toponym_id WHERE tee.etymon_id=? LIMIT 12",
                (row["eid"],),
            )
        ]
        out.append(
            {
                "ref": ref,
                "language": row["language"],
                "form": row["canonical_form"],
                "impact": row["impact"],
                "toponyms": toponyms,
            }
        )
        if len(out) >= n:
            break
    return out


def validate_gloss_candidates(
    conn: sqlite3.Connection, gloss_path: Path, candidates: Sequence[dict[str, Any]]
) -> list[str]:
    """Validate authored ``etymon_gloss_add`` rows: ``ref`` resolves, ``additions``
    is a non-empty list of ``{gloss}`` with non-empty gloss text, no dup ref."""
    done = ledger_refs(gloss_path, "etymon_gloss_add")
    seen: set[str] = set()
    errors: list[str] = []
    for i, rec in enumerate(candidates):
        tag = f"candidate[{i}]"
        if rec.get("_type") != "etymon_gloss_add":
            errors.append(f"{tag}: _type must be 'etymon_gloss_add'")
            continue
        ref = rec.get("ref")
        adds = rec.get("additions")
        if not isinstance(ref, str) or resolve_etymon_ref(conn, ref) is None:
            errors.append(f"{tag}: ref {ref!r} resolves to no etymon")
            continue
        if not isinstance(adds, list) or not adds:
            errors.append(f"{tag}: additions must be a non-empty list")
        elif any(not isinstance(a, dict) or not str(a.get("gloss", "")).strip() for a in adds):
            errors.append(f"{tag}: each addition needs a non-empty 'gloss'")
        key = _nfc(ref)
        if key in done or key in seen:
            errors.append(f"{tag}: duplicate gloss decision for {ref}")
        seen.add(key)
    return errors


def bands_status(
    conn: sqlite3.Connection,
    *,
    reflexes_path: Path,
    parked_path: Path | None,
    tags_path: Path,
    ipa_path: Path,
    gloss_path: Path,
) -> list[dict[str, Any]]:
    """Per impact band (descending), the count of target morphemes still missing
    each loop-owned dimension. A morpheme "has" a dimension if it's in the DB OR
    its committed ledger. Only bands with remaining work are returned — the first
    row is the current top band for the band-by-band loop."""
    reflex_done = committed_reflex_refs(reflexes_path) | parked_refs(parked_path)
    tag_led = {_nfc(r) for r in tag_mining.existing_refs(str(tags_path))}
    ipa_led = ledger_refs(ipa_path, "pronunciation")
    gloss_led = ledger_refs(gloss_path, "etymon_gloss_add")
    agg: dict[int, dict[str, int]] = {}
    for row in _band_cohort(conn, None):
        ref = _nfc(etymon_ref(row["language"], row["canonical_form"]))
        eid = row["eid"]
        band = agg.setdefault(
            row["impact"], {"reflex": 0, "tag": 0, "ipa": 0, "gloss": 0, "total": 0}
        )
        band["total"] += 1
        if ref not in reflex_done:
            band["reflex"] += 1
        tagged = conn.execute("SELECT 1 FROM etymon_tag WHERE etymon_id=?", (eid,)).fetchone()
        if not tagged and ref not in tag_led:
            band["tag"] += 1
        has_ipa = conn.execute(
            "SELECT 1 FROM etymon WHERE id=? AND pronunciation_ipa IS NOT NULL "
            "AND pronunciation_ipa<>''",
            (eid,),
        ).fetchone()
        if not has_ipa and ref not in ipa_led:
            band["ipa"] += 1
        has_gloss = conn.execute("SELECT 1 FROM etymon_gloss WHERE etymon_id=?", (eid,)).fetchone()
        if not has_gloss and ref not in gloss_led:
            band["gloss"] += 1
    out = []
    for impact in sorted(agg, reverse=True):
        b = agg[impact]
        if b["reflex"] or b["tag"] or b["ipa"] or b["gloss"]:
            out.append({"impact": impact, **b})
    return out


# ---------------------------------------------------------------------------
# Garbled-form repair via collapse (wyrd-eni4.3.2).
#
# A subset of scholar etymons carry a CORRUPTED canonical_form (OCR/mining junk:
# '6j^', 'stdn' for stān, HTML fragments, '& cetera'). These are almost always
# DUPLICATES of a correct etymon that already exists, so the fix is a MERGE, not
# a rename: a ``collapse`` curation row folds the garbled ``ref`` into its correct
# ``into`` twin (replayed by enrichment.apply_collapses — reproducible, no raw DB
# writes). This is a STRUCTURAL identity op (over-merge is the wyrd-740t footgun),
# so the loop is allowed ONLY high-confidence merges and they MUST pass this gate.
# ---------------------------------------------------------------------------


def validate_collapse_candidates(
    conn: sqlite3.Connection, curation_path: Path, candidates: Sequence[dict[str, Any]]
) -> list[str]:
    """Gate freshly-authored ``collapse`` rows. Each must: be ``_type ==
    "collapse"``; have ``ref`` (the garbled etymon) and ``into`` (the correct
    twin) BOTH resolve to existing, DISTINCT etymons; NOT carry a ``#`` (those are
    intentional sense-disambiguation suffixes, never corruption — must not be
    merged); and not duplicate a committed/in-batch collapse. The grounding-style
    safety net for autonomous merges."""
    done = ledger_refs(curation_path, "collapse")
    seen: set[str] = set()
    errors: list[str] = []
    for i, rec in enumerate(candidates):
        tag = f"candidate[{i}]"
        if rec.get("_type") != "collapse":
            errors.append(f"{tag}: _type must be 'collapse'")
            continue
        ref = rec.get("ref")
        into = rec.get("into")
        if not isinstance(ref, str) or resolve_etymon_ref(conn, ref) is None:
            errors.append(f"{tag}: ref {ref!r} resolves to no etymon")
            continue
        if not isinstance(into, str) or not into or resolve_etymon_ref(conn, into) is None:
            errors.append(
                f"{tag}: into {into!r} resolves to no etymon (need an existing correct twin)"
            )
        if isinstance(into, str) and _nfc(ref) == _nfc(into):
            errors.append(f"{tag}: ref and into are the same etymon — no-op merge")
        if "#" in ref or "#" in (into or ""):
            errors.append(
                f"{tag}: '#' present — that's an intentional sense-disambiguation "
                f"suffix (e.g. bol#hill), NOT corruption; do NOT collapse it"
            )
        key = _nfc(ref)
        if key in done or key in seen:
            errors.append(f"{tag}: duplicate collapse decision for {ref}")
        seen.add(key)
    return errors
