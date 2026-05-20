"""Top-level bundle build orchestration.

``export_meanings`` is the public driver: walks the lexicon,
selects promoted-root etymons, hydrates per-family data via
``_family._gather_family``, groups families into subjects via
``_subject._group_families_into_subjects``, and finally appends
orphan-reflex subjects via ``_emit._orphan_reflex_subjects``.

``RECOMMENDED_LANG_THRESHOLDS`` is the curated minimum-witness
threshold per language tag — looser for thinly-attested languages
(OE) and tighter for densely-attested ones. Exposed via the
package re-export shim for ``language_quality``'s own
quality-tier gating.

``_select_promoted_root_ids`` is the witness-count + tag-quality
gate that decides which etymons become subjects; the per-language
``min_witnesses`` thresholds + ``lang_thresholds`` override let
``export-meanings`` callers tune the bundle's recall/precision
trade-off without touching the SQL.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from wyrd.generators.kenning.lexicon.bundle._emit import _orphan_reflex_subjects
from wyrd.generators.kenning.lexicon.bundle._family import _gather_family
from wyrd.generators.kenning.lexicon.bundle._subject import (
    _group_families_into_subjects,
)
from wyrd.generators.kenning.lexicon.db import LexiconDB

# Per-language witness thresholds calibrated against corpus availability and
# spot-checked quality at w=2 (analysis 2026-05-02). Languages absent from
# this map fall back to the global ``min_witnesses``. Rationale per language:
#   old-english : 3 — well-mined (32 sources); strict gate keeps Tier-1
#                     prose-extraction noise out (~20% noise at w=2).
#   celtic      : 2 — Qwen weak on Celtic per D13; corpus structurally
#                     thinner per yield. Quality at w=2 ~90% clean.
#   old-norse   : 2 — only one ON-focused dictionary in the corpus; w=3
#                     filters out 93% of ON purely on corpus thinness.
#   modern-english,
#   norman-french,
#   latin,
#   biblical    : 2 — small populations at w≥2 (≤20 each); spot-check 100%
#                     clean. The cost of admitting them is negligible and
#                     the gain (modern English placename elements like mill,
#                     stone, head; NF castle, monte; Latin ecclesia,
#                     ceaster) is meaningful for generation breadth.
#   germanic, greek: nothing reaches w≥2 in the current corpus, so the
#                     threshold is moot — they ride the rando-port path only.
RECOMMENDED_LANG_THRESHOLDS: dict[str, int] = {
    "old-english": 3,
    "celtic": 2,
    "old-norse": 2,
    "modern-english": 2,
    "norman-french": 2,
    "latin": 2,
    "biblical": 2,
}


def export_meanings(
    db: LexiconDB,
    *,
    min_witnesses: int = 3,
    lang_thresholds: dict[str, int] | None = None,
    include_rando: bool = True,
    include_wiktionary_empirical: bool = True,
    include_wave2_enriched: bool = True,
) -> list[dict[str, Any]]:
    """Walk the lexicon and emit a meanings.json structure.

    Promotion rule (D4): a family root is included if any of:
    (a) any etymon in the family is cited by 'rando-port' AND ``include_rando``
        is true (legacy seed kept until corroborated), OR
    (b) the family's witness count (``etymon_consensus.witnesses``) is at
        least the threshold for that family's language, OR
    (c) any etymon in the family is cited by 'wiktionary-empirical' AND
        ``include_wiktionary_empirical`` is true (empirical class — wyrd-4hx7
        corpus-mining headwords matched against unaccounted-fragment misses;
        treated like rando-port: bypass the scholar-witness gate), OR
    (d) any etymon in the family has a non-NULL ``english_shaped`` value AND
        ``include_wave2_enriched`` is true (wyrd-z3cp — wave-2 non-Latin
        etymons enriched by ``derive_english_shaped`` per wyrd-vsrn Phase 2c.
        These rows come from the wiktextract bulk path WITHOUT a
        ``wiktionary-empirical`` citation, so neither (b) nor (c) admits
        them — without (d) the bundle's user-visible multi-script rendering
        siblings (``<lang>_english_shaped``, ``<lang>_transliteration``)
        have nowhere to land).

    The threshold per language is taken from ``lang_thresholds`` (defaults to
    ``RECOMMENDED_LANG_THRESHOLDS``); languages absent from the map use
    ``min_witnesses``. Pass ``lang_thresholds={}`` to apply a uniform
    ``min_witnesses`` threshold across all languages.

    Family roots are computed by the same two-step rollup the
    ``etymon_consensus`` view uses (``merged_into_id`` then ``lemma_id``), so
    OCR-cluster losers and inflected variants don't get their own subjects —
    their canonical_forms ride along in the lemma's language form list (D8,
    D18, D22).

    Subjects are reconstructed by grouping family roots that share the same
    (modifier_type, glosses, tags) signature — the natural inverse of how
    seed_from_meanings shaped the original rando-port subjects. The
    reconstruction is lossy where the original meanings.json had two distinct
    subjects with identical signatures, but the runtime ``load_meanings``
    keys by ``modern_usage`` regardless of subject boundary.
    """
    if lang_thresholds is None:
        lang_thresholds = RECOMMENDED_LANG_THRESHOLDS
    families = _collect_families(
        db,
        min_witnesses=min_witnesses,
        lang_thresholds=lang_thresholds,
        include_rando=include_rando,
        include_wiktionary_empirical=include_wiktionary_empirical,
        include_wave2_enriched=include_wave2_enriched,
    )
    subjects = _group_families_into_subjects(families)
    if include_rando:
        subjects.extend(_orphan_reflex_subjects(db))
    return subjects


def _build_witness_filter(
    lang_thresholds: dict[str, int],
    min_witnesses: int,
) -> tuple[str, list[Any]]:
    """Build a SQL WHERE fragment + bind params that gate etymon_consensus
    rows by per-language thresholds with ``min_witnesses`` as the fallback.

    Returns ``(sql_fragment, params)`` where the fragment is parenthesized
    and references ``language`` / ``witnesses`` columns.
    """
    if not lang_thresholds:
        return "(witnesses >= ?)", [min_witnesses]
    clauses: list[str] = []
    params: list[Any] = []
    sorted_langs = sorted(lang_thresholds)
    for lang in sorted_langs:
        clauses.append("(language = ? AND witnesses >= ?)")
        params.extend([lang, lang_thresholds[lang]])
    placeholders = ",".join("?" * len(sorted_langs))
    clauses.append(f"(language NOT IN ({placeholders}) AND witnesses >= ?)")
    params.extend(sorted_langs)
    params.append(min_witnesses)
    return f"({' OR '.join(clauses)})", params


def _collect_families(
    db: LexiconDB,
    *,
    min_witnesses: int,
    lang_thresholds: dict[str, int],
    include_rando: bool,
    include_wiktionary_empirical: bool = True,
    include_wave2_enriched: bool = True,
) -> list[dict[str, Any]]:
    """Build per-family-root data: forms-by-language, glosses, tags, reflexes.

    Four admission paths feed ``promoted``:
      * scholar-witness threshold (``etymon_consensus``)
      * rando-port seed admit (legacy Wikipedia-derived bundle)
      * wiktionary-empirical admit (wyrd-4hx7 corpus-mined gap-fills)
      * wave-2 enrichment admit (wyrd-z3cp — etymons with non-NULL
        english_shaped; covers the wyrd-vsrn Phase 2c non-Latin source
        languages that have no citations)

    Each empirical-class admit is gated by its own boolean flag so callers
    can A/B by toggling any branch (e.g. ``--no-include-rando``,
    ``--no-include-wiktionary-empirical``, ``--no-include-wave2-enriched``)
    without disturbing the others.
    """
    members_by_root, root_of = _build_family_rollup(db)
    root_ids = _select_promoted_root_ids(
        db,
        lang_thresholds=lang_thresholds,
        min_witnesses=min_witnesses,
        include_rando=include_rando,
        include_wiktionary_empirical=include_wiktionary_empirical,
        include_wave2_enriched=include_wave2_enriched,
        root_of=root_of,
    )
    return _iterate_families_with_progress(db, root_ids, members_by_root)


def _build_family_rollup(
    db: LexiconDB,
) -> tuple[dict[int, list[int]], Callable[[int], int]]:
    """Compute the etymon → root_id rollup in PYTHON rather than via
    a SQL CREATE TEMP TABLE. The relational form needs
    ``LEFT JOIN etymon target ON target.id = COALESCE(e.merged_into_id,
    e.lemma_id)`` — a function-on-columns join predicate that no index
    can satisfy. On a 731K-row etymon table that's a many-minute scan,
    whether materialised once into a temp table or recomputed per
    query. A flat ``SELECT id, merged_into_id, lemma_id FROM etymon``
    plus a Python dict produces the same rollup in seconds:
    731K rows * ~1µs/row = ~1s vs ~minutes for the SQL JOIN.

    The rollup follows the consensus view's two-step rule (D22 +
    D8 flatten):
      1) target = merged_into_id OR lemma_id OR self
      2) root   = lemma_id of target OR target itself
    so OCR-cluster losers and inflected children both surface their
    ultimate lemma as root_id.

    Returns ``(members_by_root, root_of_callable)``.
    """
    rollup_rows = db.conn.execute("SELECT id, merged_into_id, lemma_id FROM etymon").fetchall()
    lemma_by_id = {r["id"]: r["lemma_id"] for r in rollup_rows}
    target_by_id = {r["id"]: (r["merged_into_id"] or r["lemma_id"] or r["id"]) for r in rollup_rows}

    def root_of(eid: int) -> int:
        target = target_by_id.get(eid, eid)
        return lemma_by_id.get(target) or target

    members_by_root: dict[int, list[int]] = {}
    for eid in target_by_id:
        members_by_root.setdefault(root_of(eid), []).append(eid)
    return members_by_root, root_of


def _select_promoted_root_ids(
    db: LexiconDB,
    *,
    lang_thresholds: dict[str, int],
    min_witnesses: int,
    include_rando: bool,
    include_wiktionary_empirical: bool,
    include_wave2_enriched: bool,
    root_of: Callable[[int], int],
) -> list[int]:
    """Promoted root_ids come from four sources:

    * consensus witness threshold per language (the etymon_consensus view
      already keys on lemma_id, no rollup needed),
    * any etymon cited by 'rando-port' (legacy seed),
    * any etymon cited by 'wiktionary-empirical' (wyrd-4hx7),
    * any etymon with non-NULL english_shaped (wyrd-z3cp — wave-2
      enrichment class; covers wyrd-vsrn Phase 2c non-Latin source langs).

    For the three empirical-class branches we SELECT the source etymon_ids
    flat and roll them up via ``root_of`` — much faster than the JOIN-
    based CTE.
    """
    witness_sql, witness_params = _build_witness_filter(lang_thresholds, min_witnesses)
    promoted: set[int] = set()
    for row in db.conn.execute(
        f"SELECT lemma_id AS root_id FROM etymon_consensus WHERE {witness_sql}",
        witness_params,
    ):
        promoted.add(row["root_id"])
    if include_rando:
        for row in db.conn.execute(
            "SELECT etymon_id FROM etymon_citation WHERE source_id = 'rando-port'"
        ):
            promoted.add(root_of(row["etymon_id"]))
    if include_wiktionary_empirical:
        for row in db.conn.execute(
            "SELECT etymon_id FROM etymon_citation WHERE source_id = 'wiktionary-empirical'"
        ):
            promoted.add(root_of(row["etymon_id"]))
    if include_wave2_enriched:
        # wyrd-z3cp: wave-2 non-Latin etymons (he, ar, fa, sa, akk, egy, arc)
        # come from the wiktextract bulk path WITHOUT a wiktionary-empirical
        # citation row, so neither the consensus nor the citation branches
        # admit them. english_shaped IS NOT NULL is the per-row marker that
        # derive_english_shaped (wyrd-vsrn Phase 2c) produced a usable
        # rendering — admitting these as their own promotion class lets the
        # <lang>_english_shaped and <lang>_transliteration sibling fields
        # actually appear in the bundle.
        for row in db.conn.execute("SELECT id FROM etymon WHERE english_shaped IS NOT NULL"):
            promoted.add(root_of(row["id"]))
    return sorted(promoted)


def _iterate_families_with_progress(
    db: LexiconDB,
    root_ids: list[int],
    members_by_root: dict[int, list[int]],
) -> list[dict[str, Any]]:
    """Walk promoted root_ids, gathering each family's data and
    emitting stderr progress every ~2% of total. ``WYRD_EXPORT_QUIET=1``
    silences the progress lines.

    Without progress reporting the user has no way to estimate how
    far along a multi-minute re-export is. Step chosen to keep
    stderr output to ~50 lines on the typical 5-10K-root corpus
    while still emitting first-rate-of-change signal within the
    first minute.
    """
    import os
    import sys
    import time

    quiet = os.environ.get("WYRD_EXPORT_QUIET") == "1"
    n_total = len(root_ids)
    progress_every = max(1, n_total // 50) if n_total else 1
    started = time.monotonic()
    families: list[dict[str, Any]] = []
    for i, root_id in enumerate(root_ids, 1):
        member_ids = members_by_root.get(root_id, [root_id])
        family = _gather_family(db, root_id, member_ids)
        if family is not None and family["forms_by_lang"]:
            families.append(family)
        if not quiet and (i % progress_every == 0 or i == n_total):
            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_total - i) / rate if rate > 0 else 0
            print(
                f"  collect_families {i}/{n_total} "
                f"({100 * i / n_total:.1f}%) "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s "
                f"({rate:.0f} roots/s)",
                file=sys.stderr,
                flush=True,
            )
    return families
