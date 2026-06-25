"""Cognate-cluster era-reflex picker (wyrd-skm Phase 3.0b).

Given an etymon (e.g. OE ``ceaster``) and a target era cell, walk
the cognate cluster (D27/D28 — etymons sharing the same root via
inheritance/borrowing edges) and return cluster mates whose
language tag is the canonical pick for that era cell.

Example: ``ceaster`` (OE, cluster_id=3617) at era=me → ME variants
(Chestre, Chester, Cestre, Chestir, ...) tagged middle-english in
the same cluster. The wyrd-rni / wyrd-381 / KenningRewind demos
consume this primitive to render compound names at user-chosen eras.

Four-tier coverage ladder. Tiers 1 + 2 are MUTUALLY EXCLUSIVE
alternates — Tier 1 fires when the etymon has ``cognate_id``,
otherwise Tier 2 fires; only one runs. Tiers 3 + 4 are SEQUENTIAL
FALLBACKS that each run only if every prior tier returned empty.

1. Cognate cluster — cluster_cognates output, the high-quality path.
   Picked when ``cognate_id`` is set.
2. Direct descent — for etymons WITHOUT cognate_id but with
   etymon_descent edges; walks immediate descendants. Picked when
   ``cognate_id`` is NULL.
3. Period-form projection — for etymons whose toponym_attestation
   rows yield period forms via the wyrd-unuo Phase 3.3 projector.
   Requires ``target_family_cell``.
4. Phonology-rule fallback — generate the form via
   ``phonology_rules.rule_form`` when no data path produces one.

Coverage: ~24% of OE toponym etymons have cognate_id (cluster_cognates
output); the remaining tiers cover most of the rest. A known coverage
floor that improves as the descent graph + period-form rows grow.
"""

from __future__ import annotations

from dataclasses import dataclass

from wyrd.generators.kenning.era.cells import canonical_language_for_cell, era_year_range
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.registers.phonology_rules import rule_form as phonology_rule_form


@dataclass
class EraReflex:
    """A single cluster-mate of an etymon at a target era.

    ``etymon_id`` is the cluster mate's row id; ``form`` is its
    ``canonical_form``; ``language`` is its ``etymon.language`` tag.
    Multiple reflexes may surface for the same era when scholarly
    sources record multiple period-specific spellings (the four
    Middle-English variants of OE ``ceaster``: Chestre / Chester /
    Cestre / Chestir / etc.).

    ``source`` distinguishes how the reflex was derived. Consumers
    that surface era progressions (KenningRewind, KenningEraMap) can
    mark phonology-rule-derived forms differently so the user knows
    the form is an inferred derivation rather than an attested
    cluster mate. Values: ``"cluster"`` (Tier 1, cognate cluster),
    ``"descent"`` (Tier 2, descent edge), ``"period-form"`` (Tier 3,
    projected from toponym_attestation), ``"phonology-rule:v1"``
    (Tier 4, derived via phonology_rules.apply_rules). Default
    ``"cluster"`` so existing callers stay unchanged.
    """

    etymon_id: int
    form: str
    language: str
    source: str = "cluster"


def etymon_era_reflexes(
    db: LexiconDB,
    etymon_id: int,
    *,
    target_language: str | None = None,
    target_family_cell: tuple[str, str] | None = None,
) -> list[EraReflex]:
    """Return cluster-mates of ``etymon_id`` matching a target era.

    Two callable shapes:

    * ``target_language='middle-english'`` — direct language pick;
      every cluster mate with that exact language tag is returned.
    * ``target_family_cell=('english', 'me')`` — resolves to the
      canonical language tag via
      ``era.canonical_language_for_cell``, then proceeds as the
      direct-language path.

    Returns ``[]`` when the target cell has no canonical language
    tag (``Norse/modern`` etc.) or when neither lookup path produces
    a match. Output is sorted by ``form`` for deterministic output
    across PYTHONHASHSEED — callers that depend on first-row
    stability get the alphabetically-first reflex.

    **Four lookup tiers** in order of precedence — each is its own
    helper (``_tier1_cluster_reflexes`` etc.) that this dispatcher
    composes. Tiers 1 + 2 are ALTERNATES (cluster wins when
    ``cognate_id`` is set; otherwise descent fires). Tiers 3 + 4
    are sequential FALLBACKS (each runs only if every prior tier
    returned empty).

    1. **Cognate cluster** (D27/D28) — cluster mates of the target
       language. ~24% of OE toponym etymons. High-quality path.
    2. **Direct descent edges** — immediate inheritance / borrowing
       children of the target language. ~4% of OE toponym etymons.
    3. **Period-form projection** (wyrd-unuo Phase 3.3) — projected
       forms from ``etymon_period_form`` whose ``date_year`` falls
       in the target cell's year range. Requires
       ``target_family_cell``. Closes the ~72% coverage gap for
       isolated OE etymons.
    4. **Phonology rule** (wyrd-98cs) — derives the target-era form
       via ``phonology_rules.rule_form`` (which chains apply_rules
       under the hood) for forward / inverse cell walks.
       Lower precision (no mining evidence); reflexes carry
       ``source='phonology-rule:v1'`` so consumers can mark inferred
       forms differently from attested ones.

    Reflexes are filtered to ``merged_into_id IS NULL`` so OCR-
    cluster losers (D22) don't surface; the merge target is the
    canonical reflex for that surface.
    """
    # Defensive guard: exactly ONE target must be provided. An
    # empty-string ``target_language`` would silently slip the
    # ``is None`` check and resolve to zero rows (the SQL ``language
    # = ''`` predicate is technically valid); catch that explicitly.
    # Passing both targets at once is also a caller bug — silently
    # ignoring one would mask a bad merge.
    if (target_language is None) == (target_family_cell is None):
        raise ValueError("must pass exactly one of target_language or target_family_cell")
    if target_language is not None and not target_language:
        raise ValueError("target_language must not be an empty string")
    if target_language is None:
        assert target_family_cell is not None  # narrowed by the guard above
        family, cell = target_family_cell
        target_language = canonical_language_for_cell(family, cell)
        if target_language is None:
            return []

    # Fetch every etymon column we'll need across all four tiers in
    # one round-trip — Tier 4 also reads canonical_form + language, so
    # a second SELECT later would be wasteful.
    row = db.conn.execute(
        "SELECT cognate_id, canonical_form, language FROM etymon WHERE id = ?",
        (etymon_id,),
    ).fetchone()
    if row is None:
        return []

    # Tier 1 + Tier 2 are alternates — cognate cluster wins when the
    # etymon has cognate_id; otherwise descent edges. The "first
    # non-empty wins" pattern doesn't apply here; we pick exactly one.
    if row["cognate_id"] is not None:
        results = _tier1_cluster_reflexes(db, row["cognate_id"], target_language)
    else:
        results = _tier2_descent_reflexes(db, etymon_id, target_language)

    # Tier 3 + Tier 4 are sequential fallbacks: each runs only if the
    # prior tier produced nothing. Both surface lower-confidence data
    # so we'd rather return cluster/descent reflexes alone when they
    # exist.
    if not results and target_family_cell is not None:
        results = _tier3_period_form_reflexes(db, etymon_id, target_language, target_family_cell)
    if not results:
        results = _tier4_phonology_reflexes(
            etymon_id, row["canonical_form"], row["language"], target_language
        )

    return results


def _tier1_cluster_reflexes(
    db: LexiconDB, cognate_id: int, target_language: str
) -> list[EraReflex]:
    """Tier 1: cluster-mates query. Selects every etymon sharing
    ``cognate_id`` whose language matches the target. Highest-quality
    reflex source — backed by the D27/D28 cluster_cognates output."""
    cur = db.conn.execute(
        "SELECT id, canonical_form, language FROM etymon "
        "WHERE cognate_id = ? AND language = ? AND merged_into_id IS NULL "
        "ORDER BY canonical_form",
        (cognate_id, target_language),
    )
    return [
        EraReflex(
            etymon_id=r["id"],
            form=r["canonical_form"],
            language=r["language"],
            source="cluster",
        )
        for r in cur
    ]


def _tier2_descent_reflexes(db: LexiconDB, etymon_id: int, target_language: str) -> list[EraReflex]:
    """Tier 2: direct descent fallback. Only walks immediate children
    via inheritance / borrowing edges (peer 'cognate' edges excluded
    — too loose for v1 era-rendering). Deeper traversal would re-
    implement cluster_cognates."""
    cur = db.conn.execute(
        "SELECT DISTINCT child.id, child.canonical_form, child.language "
        "FROM etymon_descent ed "
        "JOIN etymon child ON ed.child_id = child.id "
        "WHERE ed.parent_id = ? "
        "  AND ed.edge_type IN ('inheritance', 'borrowing') "
        "  AND child.language = ? "
        "  AND child.merged_into_id IS NULL "
        "ORDER BY child.canonical_form",
        (etymon_id, target_language),
    )
    return [
        EraReflex(
            etymon_id=r["id"],
            form=r["canonical_form"],
            language=r["language"],
            source="descent",
        )
        for r in cur
    ]


def _tier3_period_form_reflexes(
    db: LexiconDB,
    etymon_id: int,
    target_language: str,
    target_family_cell: tuple[str, str],
) -> list[EraReflex]:
    """Tier 3: period-form projection (wyrd-unuo Phase 3.3). Queries
    ``etymon_period_form`` for projected period forms whose
    ``date_year`` falls in the cell's year range. Returns empty when
    the cell has no registered range. Joins against ``etymon`` to
    filter ``merged_into_id IS NULL`` — OCR-cluster losers' projected
    period forms don't surface; the merge winner is the canonical
    voice for that surface."""
    family, cell = target_family_cell
    try:
        start, end = era_year_range(family, cell)
    except KeyError:
        return []
    # Build year-range filter; treat None bounds as "open on that
    # side" (no constraint).
    clauses = ["pf.etymon_id = ?", "e.merged_into_id IS NULL"]
    params: list[int] = [etymon_id]
    if start is not None:
        clauses.append("pf.date_year >= ?")
        params.append(start)
    if end is not None:
        clauses.append("pf.date_year < ?")
        params.append(end)
    cur = db.conn.execute(
        f"""
        SELECT DISTINCT pf.form
        FROM etymon_period_form pf
        JOIN etymon e ON e.id = pf.etymon_id
        WHERE {" AND ".join(clauses)}
        ORDER BY pf.form
        """,
        params,
    )
    return [
        EraReflex(
            etymon_id=etymon_id,
            form=r["form"],
            language=target_language,
            source="period-form",
        )
        for r in cur
    ]


def _tier4_phonology_reflexes(
    etymon_id: int,
    canonical_form: str,
    source_language: str,
    target_language: str,
) -> list[EraReflex]:
    """Tier 4: phonology-rule fallback (wyrd-98cs). Walks the
    registered sound-change cells (registers/phonology_rules.py: OE→ME,
    ME→EModE, EModE→ModE, OW→ModW) forward or inverse to derive a
    target-era form from ``canonical_form``. Lower precision than
    Tier 1-3 (no mining evidence behind it), so the resulting
    EraReflex carries ``source='phonology-rule:v1'`` for callers
    that want to mark inferred forms differently. Returns empty when
    no rule fires (the form passes through unchanged)."""
    phon_form = phonology_rule_form(canonical_form, source_language, target_language)
    if phon_form is None:
        return []
    return [
        EraReflex(
            etymon_id=etymon_id,
            form=phon_form,
            language=target_language,
            source="phonology-rule:v1",
        )
    ]
