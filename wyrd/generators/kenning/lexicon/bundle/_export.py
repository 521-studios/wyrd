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
# spot-checked quality at w=2 (analysis 2026-05-02; OE relaxed to 2 in
# wyrd-fssn 2026-05-28). Languages absent from this map fall back to the
# global ``min_witnesses`` (also 2 since wyrd-fssn).
#
# Policy (wyrd-fssn, 2026-05-28): uniform ≥2 for net-new mining lemmas
# (path 1). The earlier OE=3 was a precision filter against Tier-1 prose-
# extraction noise — defensible at the time, but the cost was ~360 OE
# lemmas at exactly 2 witnesses (per the 2026-05-28 audit) that the
# dashboard's K-row audit confirms are mostly genuine cites. With the
# rando-port admit path admitting every rando-cited family unconditionally
# by default (path 2), tightening OE alone never offered the precision
# benefit it claimed — most marginal-OE-quality lemmas reach the bundle
# via the rando-port path anyway. Lifting OE to uniform ≥2 closes the
# asymmetry. The orthogonal lever for tightening bundle provenance is
# ``rando_min_corroborators`` (path 2), filed in the same ticket.
#
# Per-language reads (post wyrd-fssn):
#   old-english     : 2 — was 3 pre-fssn; see policy comment above.
#   celtic          : 2 — Qwen weak on Celtic per D13; w=2 ~90% clean.
#   old-norse       : 2 — only one ON-focused dictionary; w=3 over-filters.
#   modern-english,
#   norman-french,
#   latin,
#   biblical        : 2 — small w≥2 populations (≤20 each), spot-check 100%
#                         clean. Meaningful gain (mill, stone, head, castle,
#                         monte, ecclesia, ceaster).
#   germanic, greek : nothing reaches w≥2 in the current corpus, so the
#                     threshold is moot — they ride the rando-port path only.
RECOMMENDED_LANG_THRESHOLDS: dict[str, int] = {
    "old-english": 2,
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
    min_witnesses: int = 2,
    lang_thresholds: dict[str, int] | None = None,
    include_rando: bool = True,
    rando_min_corroborators: int = 0,
    include_wiktionary_empirical: bool = True,
    include_wave2_enriched: bool = True,
) -> list[dict[str, Any]]:
    """Walk the lexicon and emit a meanings.json structure.

    Promotion rule (D4, refined by wyrd-fssn 2026-05-28): a family root is
    included if any of:
    (a) any etymon in the family is cited by 'rando-port' AND
        ``include_rando`` is true AND the family has at least
        ``rando_min_corroborators`` distinct non-rando citation sources
        (legacy seed), OR
    (b) the family's witness count (``etymon_consensus.witnesses``) is at
        least the threshold for that family's language (uniform ≥2 per
        wyrd-fssn; per-language map preserved for future tuning), OR
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

    ``rando_min_corroborators`` defaults to 0 — admit every rando-cited
    family, matching the pre-wyrd-fssn semantic. The knob is a
    forward-flippable lever: once continued mining gives the bulk of
    rando lemmas at least one corroborating citation, the operator
    raises it to 1 to retire the pure-rando tail without touching
    the code path.

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
        rando_min_corroborators=rando_min_corroborators,
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
    rando_min_corroborators: int = 0,
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
        rando_min_corroborators=rando_min_corroborators,
        include_wiktionary_empirical=include_wiktionary_empirical,
        include_wave2_enriched=include_wave2_enriched,
        root_of=root_of,
        members_by_root=members_by_root,
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

    # wyrd-rogd.9: a later-era reflex and its earlier-era ancestor are ONE
    # morpheme across eras (OE ``biscop`` → modern ``bishop``), but the
    # merged_into/lemma rollup keeps them in separate families. Follow the
    # CURATED reflex-link inheritance edges up to the oldest ancestor so the
    # reflex joins its ancestor's family and inherits its era_reflexes (the
    # empty-grid fix).
    #
    # ONLY the ``collapse``-sourced edges (the wyrd-rogd.15 reflex-link ledger:
    # LLM-judged, same-family, adjacent-era, form-similar, one best ancestor per
    # child). We deliberately do NOT follow the raw Wiktionary ``inheritance``
    # edges: that graph is a deep, densely-connected etymological tree (PIE root
    # → … → modern) whose transitive closure collapses ~440K families and pulls
    # >1000 unrelated forms into single "families" — semantically wrong for
    # place-name morphemes and destructive to generation. Derivation/compound
    # edges are derived words, not the morpheme, so they are excluded too
    # (wyrd-rogd.11). A child with several reflex-link parents keeps the lowest
    # parent_id for determinism.
    #
    # Lazy import: enrichment imports the lexicon package, so a module-level
    # import here would close a lexicon → bundle → enrichment → lexicon cycle.
    from wyrd.generators.kenning.enrichment import COLLAPSE_VARIANT_SOURCE_ID

    inheritance_parent: dict[int, int] = {}
    for r in db.conn.execute(
        "SELECT parent_id, child_id FROM etymon_descent "
        "WHERE edge_type = 'inheritance' AND source_id = ?",
        (COLLAPSE_VARIANT_SOURCE_ID,),
    ):
        child, parent = r["child_id"], r["parent_id"]
        if child not in inheritance_parent or parent < inheritance_parent[child]:
            inheritance_parent[child] = parent

    def _lemma_root(eid: int) -> int:
        target = target_by_id.get(eid, eid)
        return lemma_by_id.get(target) or target

    def root_of(eid: int) -> int:
        root = _lemma_root(eid)
        seen = {root}  # cycle guard (the descent graph can be noisy)
        while root in inheritance_parent:
            parent_root = _lemma_root(inheritance_parent[root])
            if parent_root in seen:
                break
            seen.add(parent_root)
            root = parent_root
        return root

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
    rando_min_corroborators: int = 0,
    include_wiktionary_empirical: bool,
    include_wave2_enriched: bool,
    root_of: Callable[[int], int],
    members_by_root: dict[int, list[int]] | None = None,
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
        rando_root_ids: set[int] = set()
        for row in db.conn.execute(
            "SELECT etymon_id FROM etymon_citation WHERE source_id = 'rando-port'"
        ):
            rando_root_ids.add(root_of(row["etymon_id"]))
        if rando_min_corroborators <= 0 or members_by_root is None:
            # wyrd-fssn: default path = current behavior; admit every
            # rando-cited family regardless of corroboration. Also the
            # back-compat path for callers that don't supply
            # ``members_by_root`` — corroboration counting needs the
            # family rollup, so without it we can't gate per-family.
            promoted.update(rando_root_ids)
        else:
            # wyrd-fssn: per-family corroborator gate. Count distinct
            # non-rando-port citation sources across every family member,
            # admit only when count >= rando_min_corroborators. Roll up
            # at the family level (not per-etymon) so an inflection
            # child's citation corroborates the lemma head, matching
            # how ``etymon_consensus`` already rolls.
            corroborated = _families_with_corroborators(
                db, rando_root_ids, members_by_root, rando_min_corroborators
            )
            promoted.update(corroborated)
    if include_wiktionary_empirical:
        for row in db.conn.execute(
            "SELECT etymon_id FROM etymon_citation WHERE source_id = 'wiktionary-empirical'"
        ):
            promoted.add(root_of(row["etymon_id"]))
    if include_wave2_enriched:
        # wyrd-z3cp: wave-2 non-Latin etymons (every code in
        # english_shaping.PHASE2A_NON_LATIN_LANGS — canonical wave-2 plus
        # precursor / postcursor stack codes) come from the wiktextract
        # bulk path WITHOUT a wiktionary-empirical citation row, so
        # neither the consensus nor the citation branches admit them.
        # ``english_shaped IS NOT NULL`` is the per-row marker that
        # derive_english_shaped (wyrd-vsrn Phase 2c) produced a usable
        # rendering — admitting any row with that marker (no language
        # gate; derive_english_shaped already short-circuited on
        # Latin-script source langs at ingest time, so non-NULL implies
        # the row was a wave-2 admit) lets the <lang>_english_shaped
        # and <lang>_transliteration sibling fields actually appear in
        # the bundle.
        for row in db.conn.execute("SELECT id FROM etymon WHERE english_shaped IS NOT NULL"):
            promoted.add(root_of(row["id"]))
    return sorted(promoted)


def _families_with_corroborators(
    db: LexiconDB,
    rando_root_ids: set[int],
    members_by_root: dict[int, list[int]],
    min_corroborators: int,
) -> set[int]:
    """wyrd-fssn: subset of ``rando_root_ids`` whose family has at least
    ``min_corroborators`` distinct non-rando-port citation sources
    (any family member counted, mirroring the ``etymon_consensus``
    rollup so an inflection child's citation corroborates the lemma
    head and an OCR-merge loser's citation corroborates the winner).

    The per-family fold runs in Python — same shape as
    ``_build_family_rollup``. Citations are pulled in chunked
    ``etymon_id IN (...)`` batches to keep the query selective on
    just the rando-family member rows; bulk-SELECTing every non-
    rando citation in the corpus would scan ~80k rows the rollup
    doesn't need.
    """
    member_to_root: dict[int, int] = {}
    for root_id in rando_root_ids:
        for member_id in members_by_root.get(root_id, [root_id]):
            member_to_root[member_id] = root_id
    if not member_to_root:
        return set()
    # Pre-populate every rando root with an empty source set so the
    # docstring contract holds for ``min_corroborators == 0`` (every
    # set has ≥ 0 elements → trivially admit all). Without this, a
    # rando root that turns out to have zero non-rando citations gets
    # dropped from ``sources_by_root`` and excluded from the result —
    # contract-violating even though the sole production caller
    # currently short-circuits before reaching the 0 case.
    sources_by_root: dict[int, set[str]] = {root_id: set() for root_id in rando_root_ids}
    member_ids = list(member_to_root)
    # SQLite's compile-time parameter cap is 999 by default; chunk so
    # we never approach it regardless of corpus size.
    chunk_size = 900
    for start in range(0, len(member_ids), chunk_size):
        chunk = member_ids[start : start + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        for row in db.conn.execute(
            f"SELECT etymon_id, source_id FROM etymon_citation "
            f"WHERE source_id != 'rando-port' AND etymon_id IN ({placeholders})",
            chunk,
        ):
            root_id = member_to_root[row["etymon_id"]]
            sources_by_root[root_id].add(row["source_id"])
    return {
        root_id for root_id, sources in sources_by_root.items() if len(sources) >= min_corroborators
    }


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
