"""Rail for the compressed-gloss loop campaign (wyrd-u6fn.10.1).

Mirrors the eni4 enrichment-campaign rail shape — impact-ordered slices, a
remainder computed from the COMMITTED ledger (never the live DB; the wyrd-1rw4
trap), a validation gate, a park sidecar — but for gloss-sense work. Self-
contained (the eni4 rail lives on a different branch), so it carries its own
impact ranking + cohort query.

Phase ladder (the loop walks it top-down, never idling):
  - Phase 1: generation-eligible (impact >= 2) morphemes that HAVE a gloss wall,
    impact-ordered. Compress the wall into senses.
  - Phase 2: impact >= 2 morphemes with NO gloss — scholar-grounded glossing
    first (a separate verb), then they fall into phase 1.
  - Phase 3 (overflow): drop the impact floor to the full inventory.

"impact" = the count of DISTINCT toponyms a morpheme is attributed to in the
scholarly breakdown corpus — the same generation-eligibility signal the eni4
rail uses (admit cohort = impact >= 2).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from wyrd.generators.kenning.canonicalization import effective_assertions, load_assertions
from wyrd.generators.kenning.lexicon.etymon_refs import etymon_ref
from wyrd.generators.kenning.lexicon.gloss_sense import committed_sense_refs

# Impact ranking: morphemes attributed in the scholarly breakdowns, by distinct
# toponym count. Excludes language='unknown' and multi-word canonical forms (the
# eni4 rail's scope filter). HAVING gates the impact floor; the has-gloss filter
# is applied in Python so phase 2/3 can flip it without a second query.
_IMPACT_SQL = """
SELECT tee.etymon_id AS eid, e.language AS language, e.canonical_form AS canonical_form,
       COUNT(DISTINCT te.toponym_id) AS impact,
       EXISTS (SELECT 1 FROM etymon_gloss g WHERE g.etymon_id = e.id) AS has_gloss
FROM toponym_etymology_element tee
JOIN toponym_etymology te ON te.id = tee.toponym_etymology_id
JOIN etymon e ON e.id = tee.etymon_id
WHERE tee.etymon_id IS NOT NULL
  AND e.language <> 'unknown'
  AND e.canonical_form NOT LIKE '% %'
GROUP BY tee.etymon_id, e.language, e.canonical_form
HAVING COUNT(DISTINCT te.toponym_id) >= ?
ORDER BY impact DESC, e.language, e.canonical_form
"""

# Phase-3 overflow: every glossed etymon, not just breakdown-attributed ones,
# ordered by gloss-wall size (the biggest walls benefit most from compression)
# then a stable key. impact is 0 here (not breakdown-attributed by this query).
_INVENTORY_SQL = """
SELECT e.id AS eid, e.language AS language, e.canonical_form AS canonical_form,
       0 AS impact, COUNT(g.gloss) AS wall
FROM etymon e
JOIN etymon_gloss g ON g.etymon_id = e.id
WHERE e.language <> 'unknown' AND e.canonical_form NOT LIKE '% %'
GROUP BY e.id, e.language, e.canonical_form
ORDER BY wall DESC, e.language, e.canonical_form
"""


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


@dataclass(frozen=True)
class GlossEvidence:
    """One morpheme's compression work-unit: its ref, impact, and gloss wall."""

    ref: str
    language: str
    canonical_form: str
    impact: int
    glosses: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "ref": self.ref,
            "language": self.language,
            "canonical_form": self.canonical_form,
            "impact": self.impact,
            "glosses": list(self.glosses),
        }


def parked_refs(parked_path: str | Path) -> set[str]:
    """Refs parked (fragment / no-consensus / ambiguous) — excluded from slices.

    The park file is an operator-facing append log (``cmd_park`` appends; manual
    triage edits it), so a truncated or hand-mangled line is plausible. A bad line
    is skipped with a stderr warning rather than aborting EVERY ``next-slice`` /
    ``status`` (both call this) — the loop rail would otherwise stop dead on one
    corrupt row, with no indication which."""
    path = Path(parked_path)
    if not path.is_file():
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"{path}:{line_no}: skipping malformed park line", file=sys.stderr)
                continue
            # Valid JSON that isn't an object (a bare list/number/string) has no
            # .get — guard it too, so a hand-mangled line can't crash the rail.
            if not isinstance(data, dict):
                print(f"{path}:{line_no}: skipping malformed park line", file=sys.stderr)
                continue
            ref = data.get("ref")
            if ref:
                out.add(_nfc(ref))
    return out


def committed_done(base_dir: str | Path) -> set[str]:
    """Morpheme refs already sense-bound in the committed canonicalization ledger
    (remainder source — wyrd-1rw4: the ledger, never the live DB).

    NFC-normalized to match both sides of the exclusion check: ``next_slice`` and
    ``sense_progress`` compare against ``_nfc(etymon_ref(...))`` and ``parked_refs``
    is ``_nfc``'d too, but ``committed_sense_refs`` carries the raw authored form.
    Without this, a non-NFC stored ``canonical_form`` (decomposed combining marks —
    the diacritic-heavy Old English / Welsh morphemes this campaign targets) would
    never match its NFC slice ref, so an already-sense-bound morpheme would reappear
    in every slice and be undercounted as remaining."""
    return {
        _nfc(r) for r in committed_sense_refs(effective_assertions(list(load_assertions(base_dir))))
    }


def _wall(conn: sqlite3.Connection, eid: int) -> tuple[str, ...]:
    # ORDER BY gloss for a deterministic wall: _wall's output is emitted verbatim
    # on the next-slice stdout the loop agent reads/diffs, and SQLite row order
    # without ORDER BY is unspecified (can shift across rebuild/VACUUM).
    return tuple(
        r[0]
        for r in conn.execute(
            "SELECT gloss FROM etymon_gloss WHERE etymon_id = ? ORDER BY gloss", (eid,)
        )
    )


def next_slice(
    conn: sqlite3.Connection,
    base_dir: str | Path,
    *,
    n: int = 20,
    min_impact: int = 2,
    require_gloss: bool = True,
    parked_path: str | Path,
    full_inventory: bool = False,
) -> list[GlossEvidence]:
    """The next ``n`` impact-ordered morphemes still needing sense compression.

    Phase 1 (default): impact >= ``min_impact``, has a gloss wall, not done, not
    parked. Phase 3: ``full_inventory=True`` walks every glossed etymon by wall
    size. ``require_gloss=False`` surfaces the phase-2 (unglossed) cohort.
    """
    excluded = committed_done(base_dir) | parked_refs(parked_path)
    if full_inventory:
        rows = conn.execute(_INVENTORY_SQL)
    else:
        rows = conn.execute(_IMPACT_SQL, (min_impact,))
    out: list[GlossEvidence] = []
    for r in rows:
        ref = _nfc(etymon_ref(r["language"], r["canonical_form"]))
        if ref in excluded:
            continue
        has_gloss = bool(r["wall"]) if full_inventory else bool(r["has_gloss"])
        if require_gloss and not has_gloss:
            continue
        if not require_gloss and has_gloss:
            continue  # phase-2 wants the UNGLOSSED ones only
        glosses = _wall(conn, r["eid"]) if has_gloss else ()
        out.append(
            GlossEvidence(
                ref=ref,
                language=r["language"],
                canonical_form=r["canonical_form"],
                impact=r["impact"],
                glosses=glosses,
            )
        )
        if len(out) >= n:
            break
    return out


def sense_progress(
    conn: sqlite3.Connection,
    base_dir: str | Path,
    parked_path: str | Path,
    *,
    min_impact: int = 2,
) -> tuple[int, int, int]:
    """``(done, parked, total)`` over the phase-1 cohort (impact >= ``min_impact``,
    glossed) — the cheap per-fire scoreboard."""
    done_refs = committed_done(base_dir)
    parked = parked_refs(parked_path)
    total = 0
    done = 0
    parked_in_cohort = 0
    for r in conn.execute(_IMPACT_SQL, (min_impact,)):
        if not r["has_gloss"]:
            continue
        ref = _nfc(etymon_ref(r["language"], r["canonical_form"]))
        total += 1
        if ref in done_refs:
            done += 1
        elif ref in parked:
            parked_in_cohort += 1
    return done, parked_in_cohort, total
