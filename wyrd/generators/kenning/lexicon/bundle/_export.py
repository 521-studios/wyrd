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

``select_promoted_root_ids`` is the witness-count + tag-quality
gate that decides which etymons become subjects; the per-language
``min_witnesses`` thresholds + ``lang_thresholds`` override let
``export-meanings`` callers tune the bundle's recall/precision
trade-off without touching the SQL.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Literal

from wyrd.generators.kenning.lexicon.bundle._emit import _orphan_reflex_subjects
from wyrd.generators.kenning.lexicon.bundle._family import _bulk_attested_years, _gather_family
from wyrd.generators.kenning.lexicon.bundle._subject import (
    _group_families_into_subjects,
)
from wyrd.generators.kenning.lexicon.db import LexiconDB

# A canonical family key: a canonical-hub id (TEXT) for a bound etymon, or a
# ``("legacy", <legacy root etymon id>)`` tag for an unbound one (a legacy
# singleton). The tag keeps the hub-string and legacy-int namespaces disjoint.
FamilyKey = str | tuple[Literal["legacy"], int]

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
    include_toponym_breakdown: bool = False,
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
        have nowhere to land), OR
    (e) any etymon in the family appears as an element in a scholarly
        ``toponym_etymology`` breakdown AND ``include_toponym_breakdown`` is
        true (wyrd-oth3 — a scholar decomposing a real place into this morpheme
        is a place-name-specific evidence channel, distinct from and arguably
        stronger than dictionary witnesses; admits the specifiers/heads scholars
        used that the witness gate misses, so the matcher stops junk-tiling
        ``Aldermaston`` → ``Al+Der+Ma+Ston``). Bypasses the witness gate like
        the other empirical-class paths.

        DEFAULT OFF (unlike its siblings): the grader-validated re-emit
        (wyrd-aicu.9, 2026-06-16) showed the UNFILTERED admit is net-negative on
        scholar agreement — coverage +3.4pts but cluster recall/precision/head
        all DOWN, because the breakdown corpus carries COMPOSITE forms
        (``-ington``, ``-erton``, ``-halton``) and the Occam (fewer-morphemes)
        tiebreaker then prefers the coarse ``Ald+ington`` over the finer correct
        ``Ald+ing+ton``. Gated off until: occurrence-threshold + composite-form
        pruning (wyrd-myv4), the matcher granularity tiebreaker (wyrd-h5u1), and
        the uplift clustering (wyrd-7hbp) make it a graded net win.

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
        include_toponym_breakdown=include_toponym_breakdown,
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
    include_toponym_breakdown: bool = False,
) -> list[dict[str, Any]]:
    """Build per-family-root data: forms-by-language, glosses, tags, reflexes.

    Five admission paths feed ``promoted``:
      * scholar-witness threshold (``etymon_consensus``)
      * rando-port seed admit (legacy Wikipedia-derived bundle)
      * wiktionary-empirical admit (wyrd-4hx7 corpus-mined gap-fills)
      * wave-2 enrichment admit (wyrd-z3cp — etymons with non-NULL
        english_shaped; covers the wyrd-vsrn Phase 2c non-Latin source
        languages that have no citations)
      * toponym-breakdown admit (wyrd-oth3 — any etymon used as an element in
        a scholarly ``toponym_etymology`` breakdown; the place-name evidence
        channel that recovers the specifiers/heads the witness gate misses)

    Each empirical-class admit is gated by its own boolean flag so callers
    can A/B by toggling any branch (e.g. ``--no-include-rando``,
    ``--no-include-wiktionary-empirical``, ``--no-include-wave2-enriched``,
    ``--no-include-toponym-breakdown``) without disturbing the others.
    """
    # wyrd-b2mf: the export reads the AUTHORITATIVE canonical identity layer, not
    # the legacy COALESCE(merged_into_id, lemma_id, id) columns — byte-equivalent
    # today (the u6fn.4 fidelity gate), and reflects 1a's binds once applied. (The
    # fold, the 1a miner, and the fidelity verifier keep using build_family_rollup
    # — the fold BUILDS canonical_morpheme_id from it, so it can't read it back.)
    members_by_root, root_of = _build_canonical_rollup(db)
    root_ids = select_promoted_root_ids(
        db,
        lang_thresholds=lang_thresholds,
        min_witnesses=min_witnesses,
        include_rando=include_rando,
        rando_min_corroborators=rando_min_corroborators,
        include_wiktionary_empirical=include_wiktionary_empirical,
        include_wave2_enriched=include_wave2_enriched,
        include_toponym_breakdown=include_toponym_breakdown,
        root_of=root_of,
        members_by_root=members_by_root,
    )
    return _iterate_families_with_progress(db, root_ids, members_by_root)


def build_family_rollup(
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
    # (wyrd-rogd.11). A child with several reflex-link parents keeps the one
    # with the smallest ``(language, canonical_form)`` — a rebuild-stable
    # content key, NOT the autoincrement parent_id (which is unstable across
    # rebuilds); the detector emits one best ancestor per child so this normally
    # never arbitrates.
    #
    # Lazy import: enrichment imports the lexicon package, so a module-level
    # import here would close a lexicon → bundle → enrichment → lexicon cycle.
    from wyrd.generators.kenning.enrichment import COLLAPSE_VARIANT_SOURCE_ID

    def _lemma_root(eid: int) -> int:
        target = target_by_id.get(eid, eid)
        return lemma_by_id.get(target) or target

    # Collect the candidate parent-roots per child-ROOT. Both sides are passed
    # through _lemma_root so the inheritance hop composes with the merged_into/
    # lemma rollup (a reflex that itself has a lemma_id still finds its edge —
    # root_of lands on the child's lemma-root first, which is what we key on).
    edge_rows = db.conn.execute(
        "SELECT parent_id, child_id FROM etymon_descent "
        "WHERE edge_type = 'inheritance' AND source_id = ?",
        (COLLAPSE_VARIANT_SOURCE_ID,),
    ).fetchall()
    candidates: dict[int, set[int]] = {}
    for r in edge_rows:
        child_root, parent_root = _lemma_root(r["child_id"]), _lemma_root(r["parent_id"])
        if child_root != parent_root:  # already one family via merge/lemma
            candidates.setdefault(child_root, set()).add(parent_root)

    # Content keys for the deterministic tie-break (rebuild-stable).
    parent_roots = list({p for ps in candidates.values() for p in ps})
    form_by_id: dict[int, tuple[str, str]] = {}
    for i in range(0, len(parent_roots), 900):  # chunk: SQLite caps host params at 999
        chunk = parent_roots[i : i + 900]
        qmarks = ",".join("?" * len(chunk))
        for r in db.conn.execute(
            f"SELECT id, language, canonical_form FROM etymon WHERE id IN ({qmarks})",
            tuple(chunk),
        ):
            form_by_id[r["id"]] = (r["language"], r["canonical_form"])

    inheritance_parent: dict[int, int] = {
        child_root: min(prs, key=lambda p: (form_by_id.get(p, ("", "")), p))
        for child_root, prs in candidates.items()
    }

    def root_of(eid: int) -> int:
        root = _lemma_root(eid)
        seen = {root}  # cycle guard (the descent graph can be noisy)
        while root in inheritance_parent:
            parent_root = inheritance_parent[root]
            if parent_root in seen:
                break
            seen.add(parent_root)
            root = parent_root
        return root

    members_by_root: dict[int, list[int]] = {}
    for eid in target_by_id:
        members_by_root.setdefault(root_of(eid), []).append(eid)
    return members_by_root, root_of


def _canonical_hub_root(merged_into: dict[str, str], node_id: str) -> str:
    """Walk ``canonical_morpheme.merged_into`` from ``node_id`` to the surviving
    hub id, cycle-guarded (the merge graph can be noisy). A node with no / empty
    ``merged_into`` target is its own hub."""
    seen: set[str] = set()
    while merged_into.get(node_id) and node_id not in seen:
        seen.add(node_id)
        node_id = merged_into[node_id]
    return node_id


def _build_canonical_rollup(
    db: LexiconDB,
) -> tuple[dict[int, list[int]], Callable[[int], int]]:
    """The identity rollup read from the AUTHORITATIVE ``canonical_morpheme``
    layer — the cutover target (wyrd-b2mf), the canonical analog of
    ``build_family_rollup``. Same contract: ``(members_by_root, root_of)`` with
    an etymon-id representative.

    Partition: an etymon's family is its ``canonical_morpheme_id`` rolled through
    ``canonical_morpheme.merged_into``; an UNBOUND etymon (``canonical_morpheme_id``
    NULL — a singleton, exactly like ``merged_into_id IS NULL`` legacy, since the
    fold binds every multi-member family) is its own family. The REPRESENTATIVE is
    reused from the legacy rollup (``build_family_rollup``) so the subtle
    lemma/inheritance-hop root logic isn't re-implemented: a family's root is the
    smallest-``(language, canonical_form, id)`` among the legacy roots of its
    members (one legacy root per family today → identical to legacy; once 1a's
    binds merge families, the deterministic min picks the survivor).

    Equivalent to ``build_family_rollup`` TODAY (``canonical_morpheme_id``
    reproduces the legacy rollup — the u6fn.4 fidelity gate, 0 violations); the
    point of switching is that once 1a's dormant binds apply, this reflects the
    new merges and the legacy rollup does not. When the canonical graph is
    UNPOPULATED (the projection hasn't run) it falls back to the legacy rollup
    with a warning — legacy is full clustering, so the fallback never
    under-clusters; production always projects (rebuild --with-enrichment).
    """
    legacy_members, legacy_root_of = build_family_rollup(db)

    merged_into = dict(db.conn.execute("SELECT id, merged_into FROM canonical_morpheme").fetchall())

    bound = db.conn.execute(
        "SELECT id, canonical_morpheme_id FROM etymon WHERE canonical_morpheme_id IS NOT NULL"
    ).fetchall()
    if not bound:
        # The canonical graph isn't populated (the projection hasn't run). Fall back
        # to the legacy rollup — which is FULL clustering (byte-identical to canonical
        # today), so this never under-clusters; it only fails to reflect any not-yet-
        # applied 1a binds, of which an unprojected DB has none. Production always runs
        # the projection (the rebuild --with-enrichment terminal pass), so the fallback
        # is a dev/test convenience — surfaced via a warning (not silent), never a raise
        # that would make the export unusable on any unprojected DB (wyrd-b2mf).
        #
        # KNOWN DEBT (wyrd-yopl): this soft contract — export correctness gated on an
        # out-of-band projection step, enforced only by a warning — is gross but correct.
        # Long-term: have the build guarantee/assert the projection ran so this branch
        # is unreachable in real builds.
        if any(len(m) > 1 for m in legacy_members.values()):
            warnings.warn(
                "canonical_morpheme_id is unpopulated — falling back to the legacy "
                "merged_into_id/lemma_id rollup. Run `project-canonical --apply` (or "
                "`rebuild-from-jsonl --with-enrichment`) to export off the authoritative "
                "canonical layer (wyrd-b2mf).",
                stacklevel=2,
            )
        return legacy_members, legacy_root_of
    hub_of: dict[int, str] = {
        r["id"]: _canonical_hub_root(merged_into, r["canonical_morpheme_id"]) for r in bound
    }

    # Group by canonical family, tracking which LEGACY roots land in each. Iterate
    # legacy_members (already keyed by legacy root from build_family_rollup) so we
    # never re-walk legacy_root_of per etymon. ``eid in hub_of`` (not truthiness):
    # a bound etymon's hub id is non-empty, but membership is the precise test.
    members_by_family: dict[FamilyKey, list[int]] = {}
    family_legacy_roots: dict[FamilyKey, set[int]] = {}
    for legacy_root, members in legacy_members.items():
        for eid in members:
            # .get (membership, not truthiness): a bound etymon keeps its hub even
            # if the hub id were empty — never silently falls through to legacy.
            key: FamilyKey = hub_of.get(eid, ("legacy", legacy_root))
            members_by_family.setdefault(key, []).append(eid)
            family_legacy_roots.setdefault(key, set()).add(legacy_root)

    # Representative per family: the smallest-content-key legacy root (deterministic
    # survivor when 1a binds merge several legacy families onto one hub).
    all_roots = {r for roots in family_legacy_roots.values() for r in roots}
    form_key = _content_keys(db, all_roots)

    members_by_root: dict[int, list[int]] = {}
    rep_of_etymon: dict[int, int] = {}
    for key, roots in family_legacy_roots.items():
        rep = min(roots, key=lambda r: (form_key.get(r, ("", "")), r))
        members = members_by_family[key]
        # .extend, not assign: if a legacy family were split across hubs (members
        # bound to different canonical families), both halves share the same legacy
        # root as rep — a plain assign would drop one half. Union under the rep
        # instead (degrades a split to the legacy grouping; no member is ever lost).
        members_by_root.setdefault(rep, []).extend(members)
        for eid in members:
            rep_of_etymon[eid] = rep

    def root_of(eid: int) -> int:
        return rep_of_etymon[eid]

    return members_by_root, root_of


def _content_keys(db: LexiconDB, ids: set[int]) -> dict[int, tuple[str, str]]:
    """``(language, canonical_form)`` per etymon id, fetched in param-safe chunks."""
    out: dict[int, tuple[str, str]] = {}
    id_list = list(ids)
    for i in range(0, len(id_list), 900):
        chunk = id_list[i : i + 900]
        qmarks = ",".join("?" * len(chunk))
        for r in db.conn.execute(
            f"SELECT id, language, canonical_form FROM etymon WHERE id IN ({qmarks})",  # noqa: S608 — ? placeholders
            tuple(chunk),
        ):
            out[r["id"]] = (r["language"] or "", r["canonical_form"] or "")
    return out


def select_promoted_root_ids(
    db: LexiconDB,
    *,
    lang_thresholds: dict[str, int],
    min_witnesses: int,
    include_rando: bool,
    rando_min_corroborators: int = 0,
    include_wiktionary_empirical: bool,
    include_wave2_enriched: bool,
    include_toponym_breakdown: bool = False,
    root_of: Callable[[int], int],
    members_by_root: dict[int, list[int]] | None = None,
) -> list[int]:
    """Promoted root_ids come from five sources:

    * consensus witness threshold per language (the etymon_consensus view
      keys on lemma_id; rolled up via ``root_of`` so an inheritance reflex
      promotes its ancestor's root — wyrd-rogd.9),
    * any etymon cited by 'rando-port' (legacy seed),
    * any etymon cited by 'wiktionary-empirical' (wyrd-4hx7),
    * any etymon with non-NULL english_shaped (wyrd-z3cp — wave-2
      enrichment class; covers wyrd-vsrn Phase 2c non-Latin source langs),
    * any etymon used as an element in a scholarly toponym_etymology
      breakdown (wyrd-oth3 — the place-name evidence channel).

    For the four empirical-class branches we SELECT the source etymon_ids
    flat and roll them up via ``root_of`` — much faster than the JOIN-
    based CTE.
    """
    witness_sql, witness_params = _build_witness_filter(lang_thresholds, min_witnesses)
    promoted: set[int] = set()
    for row in db.conn.execute(
        f"SELECT lemma_id AS root_id FROM etymon_consensus WHERE {witness_sql}",
        witness_params,
    ):
        # wyrd-b2mf: ``etymon_consensus`` stays the LEGACY lemma-rollup witness
        # view (it emits an etymon-id root + per-family witness count; a canonical
        # hub is a string, so the view can't emit one without breaking this
        # ``root_of(int)`` consumer). It is NOT rewritten — instead the
        # legacy-keyed root_id is rolled through the now-CANONICAL ``root_of`` here,
        # so a family is promoted at its canonical root. A canonical family
        # promotes iff any of its legacy sub-families meets the witness gate
        # (post-1a; identical to legacy today — the byte-identical bundle proof).
        #
        # wyrd-rogd.9: root_of also folds an inheritance reflex (``bishop``) onto
        # its ancestor's root (``biscop``), so the reflex isn't promoted standalone
        # AND rolled into the ancestor's family → emitted twice.
        promoted.add(root_of(row["root_id"]))
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
        _promote_flat_admit(
            db,
            "SELECT etymon_id FROM etymon_citation WHERE source_id = 'wiktionary-empirical'",
            root_of,
            promoted,
        )
    if include_wave2_enriched:
        # wyrd-z3cp: wave-2 non-Latin etymons (PHASE2A_NON_LATIN_LANGS) come from
        # the wiktextract bulk path WITHOUT a wiktionary-empirical citation, so
        # neither consensus nor the citation branches admit them. ``english_shaped
        # IS NOT NULL`` is the per-row marker derive_english_shaped (wyrd-vsrn
        # Phase 2c) set (it short-circuits on Latin-script source langs at ingest,
        # so non-NULL implies a wave-2 admit) — needed so the <lang>_english_shaped
        # / _transliteration sibling fields land in the bundle.
        _promote_flat_admit(
            db, "SELECT id FROM etymon WHERE english_shaped IS NOT NULL", root_of, promoted
        )
    if include_toponym_breakdown:
        # wyrd-oth3: any etymon a scholar used as an element in a toponym_etymology
        # breakdown — direct place-name evidence that recovers the specifiers/heads
        # (alder, barlow, ...) the dictionary-witness gate misses, the absence of
        # which made the matcher junk-tile -ston names. Bypasses the witness gate
        # like the other empirical-class admits. Admits on the breakdown signal
        # alone — occurrence-threshold tiering + junk-tail pruning is wyrd-myv4.
        _promote_flat_admit(
            db, "SELECT DISTINCT etymon_id FROM toponym_etymology_element", root_of, promoted
        )
    return sorted(promoted)


def _promote_flat_admit(
    db: LexiconDB,
    sql: str,
    root_of: Callable[[int], int],
    promoted: set[int],
) -> None:
    """Roll up the first column of every ``sql`` row to its family root via
    ``root_of`` and add it to ``promoted``. Shared by the empirical-class admit
    branches (wiktionary-empirical / wave-2 / toponym-breakdown) — each is a flat
    SELECT-then-rollup, much faster than a JOIN-based CTE."""
    for row in db.conn.execute(sql):
        promoted.add(root_of(row[0]))


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
    ``build_family_rollup``. Citations are pulled in chunked
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
    # wyrd-4zyb: bulk-prefetch attested years for EVERY member up front (one
    # indexed-temp-table join) instead of re-running the per-family UNION/MIN
    # query (_fetch_member_attested_years, called by _gather_family) for each of
    # ~67K roots — that per-root query was ~96% of this loop's wall time
    # (~13min → ~tens of seconds).
    all_member_ids = {m for rid in root_ids for m in members_by_root.get(rid, [rid])}
    attested_years_by_member = _bulk_attested_years(db, all_member_ids)
    families: list[dict[str, Any]] = []
    for i, root_id in enumerate(root_ids, 1):
        member_ids = members_by_root.get(root_id, [root_id])
        family = _gather_family(db, root_id, member_ids, attested_years_by_member)
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
