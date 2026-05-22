"""SQL data loaders for the bundle build (leaf module — no internal deps).

``_gather_family`` is the central entry point: given a promoted-root
etymon id + the list of member ids in its cognate cluster, run every
``_fetch_member_*`` SQL loader once, hydrate a single per-family
context dict, and return it for downstream subject/word assembly.

The era-reflex picker (``_fetch_root_era_reflexes``) uses the same
4-tier ladder that ``era_reflex.etymon_era_reflexes`` exposes for
runtime callers — the bundle pre-computes the picks at build time so
the runtime doesn't need to walk the descent graph.

Source-quality preferences are encoded in
``_ERA_REFLEX_SOURCE_PRIORITY`` (cluster > descent > period-form >
phonology-rule) and consulted by ``_better_era_reflex_source`` for
deterministic cross-family merging.
"""

from __future__ import annotations

import re
from typing import Any

from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.era_reflex import etymon_era_reflexes


def _gather_family(db: LexiconDB, root_id: int, member_ids: list[int]) -> dict[str, Any] | None:
    """Collect forms / glosses / tags / reflexes for one family root.

    Includes the root itself plus all etymons that roll up to it
    (inflected children, OCR-cluster losers, and combinations thereof).
    ``member_ids`` is the precomputed family-membership list (root +
    children) — caller is responsible for the rollup, computed once in
    ``_collect_families`` to avoid the per-root SQL JOIN that scaled
    badly on the 731K-row etymon table.
    Pure orchestrator — each per-aspect aggregation lives in a focused
    helper below.
    """
    root_row = db.conn.execute(
        "SELECT canonical_form, language, modifier_type, position_pref FROM etymon WHERE id = ?",
        (root_id,),
    ).fetchone()
    if root_row is None:
        return None

    if not member_ids:
        return None
    placeholders = ",".join("?" * len(member_ids))
    member_rows = db.conn.execute(
        f"""
        SELECT id, canonical_form, language, lemma_id, merged_into_id, inflection,
               english_shaped, original_script, transliteration,
               pronunciation_ipa, pronunciation_dialect, stratum,
               phonological_vector
        FROM etymon
        WHERE id IN ({placeholders})
        ORDER BY language, canonical_form
        """,
        member_ids,
    ).fetchall()
    if not member_rows:
        return None

    member_ids = [r["id"] for r in member_rows]
    member_form_by_id = {r["id"]: (r["language"], r["canonical_form"]) for r in member_rows}
    member_inflection_by_id: dict[int, str | None] = {r["id"]: r["inflection"] for r in member_rows}
    # wyrd-vsrn Phase 2c + wyrd-qhs0 Phase 2d: per-member wyrd-ha9q
    # rendering data. NULL when the row's source language is Latin-
    # script or wyrd-ha9q's derive / wiktextract ingest produced no
    # value. Keyed by member_id so absorb-helpers can look up cheaply.
    member_english_shaped_by_id: dict[int, str | None] = {
        r["id"]: r["english_shaped"] for r in member_rows
    }
    member_original_script_by_id: dict[int, str | None] = {
        r["id"]: r["original_script"] for r in member_rows
    }
    member_transliteration_by_id: dict[int, str | None] = {
        r["id"]: r["transliteration"] for r in member_rows
    }
    # pronunciation is paired (IPA + optional dialect tag); keyed by
    # member_id with a tuple so the absorb-helper can drop both
    # together when either side is NULL (matches the upsert semantics
    # that updated them atomically).
    member_pronunciation_by_id: dict[int, tuple[str, str | None] | None] = {
        r["id"]: (r["pronunciation_ipa"], r["pronunciation_dialect"])
        if r["pronunciation_ipa"]
        else None
        for r in member_rows
    }
    # wyrd-lr4 Phase 2: per-member within-language stratum tag
    # (Welsh-only in Phase 1; French / OE / ON follow). NULL for
    # languages without a Phase 1 classifier and for legacy DBs that
    # pre-date the column.
    member_stratum_by_id: dict[int, str | None] = {r["id"]: r["stratum"] for r in member_rows}
    # wyrd-ecjp.10a: per-member phonological_vector JSON blob (kq7w.1
    # corpus enrichment populates etymon.phonological_vector). NULL for
    # rows the enrichment pass hasn't reached + legacy DBs that
    # pre-date the column. The bundle emit walks these per descendant
    # in _absorb_member_phonological_vector and stamps them as
    # <lang>_phonological_vector siblings per the D26 pattern.
    member_phonological_vector_by_id: dict[int, str | None] = {
        r["id"]: r["phonological_vector"] for r in member_rows
    }
    canonical_forms_lower = {f.lower() for _lang, f in member_form_by_id.values()}

    reflex_links = _fetch_member_reflex_links(db, member_ids)

    return {
        "root_id": root_id,
        "root_canonical_form": root_row["canonical_form"],
        "root_language": root_row["language"],
        "modifier_type": root_row["modifier_type"],
        "position_pref": root_row["position_pref"],
        "forms_by_lang": _build_forms_by_lang(root_row, member_rows),
        "member_form_by_id": member_form_by_id,
        "member_descendants": _compute_member_descendants(member_rows),
        "member_variants": _fetch_member_variants(db, member_ids, canonical_forms_lower),
        "member_inflection_by_id": member_inflection_by_id,
        "member_english_shaped_by_id": member_english_shaped_by_id,
        "member_original_script_by_id": member_original_script_by_id,
        "member_transliteration_by_id": member_transliteration_by_id,
        "member_pronunciation_by_id": member_pronunciation_by_id,
        "member_stratum_by_id": member_stratum_by_id,
        "member_phonological_vector_by_id": member_phonological_vector_by_id,
        "member_citations": _fetch_member_citations(db, member_ids),
        "member_attested_years": _fetch_member_attested_years(db, member_ids),
        "glosses": _fetch_member_glosses(db, member_ids),
        # wyrd-c4wd: dropped the cluster-mate tag union (was wyrd-i1s1).
        # Pre-fix the bundle unioned member tags (lemma + inflections +
        # OCR losers) with EVERY cognate-cluster mate's tags so semantic
        # signal from non-promoted cluster mates would 'ride along' on
        # the promoted root. In practice this manufactured the tag-noise
        # the user reported (2026-05-22): '-y' (Area/Market/Ward) ended
        # up tagged 'animal' because some cluster-mate was animal-tagged;
        # 'Dar-' (Deer/Hind/Stag) ended up tagged 'architecture' /
        # 'food'. The cluster-mate signal turned out to be mostly noise
        # against the rando-port bundle's unverified per-entry tags.
        # Member-only tags are the faithful inventory; future wyrd-c4wd
        # work may validate them against the meanings themselves
        # (semantic-similarity audit) but the cluster-mate drop alone
        # eliminates the bulk of visible mismatches.
        "tags": sorted(set(_fetch_member_tags(db, member_ids))),
        "reflexes": _fetch_member_reflexes(db, member_ids, reflex_links),
        # wyrd-obpw Phase 3.3: era reflexes for the family root,
        # keyed by target language tag. SPA-side rewinder reads this
        # at runtime from the bundle (Lambda has no lexicon DB).
        "era_reflexes": _fetch_root_era_reflexes(db, root_id, root_row["language"]),
    }


def _fetch_root_era_reflexes(
    db: LexiconDB, root_id: int, root_language: str
) -> dict[str, list[dict[str, str]]]:
    """wyrd-obpw Phase 3.3 + wyrd-jbcu source-aware schema: per-root
    era reflexes for bundle export.

    Returns a dict mapping target language tag → sorted list of
    ``{"form": str, "source": str}`` dicts. The SPA-side rewinder
    consumes this at runtime; ``source`` distinguishes attestation-
    backed reflexes ('cluster' / 'descent' / 'period-form') from
    phonology-rule-derived ones ('phonology-rule:v1') so consumers
    can render inferred forms differently if they want to.

    For each language tag in ``CANONICAL_LANGUAGE_FOR_CELL`` of the
    root's family, calls ``etymon_era_reflexes`` and collects the
    forms. When the same form arrives via multiple tiers (cluster
    mate AND a phonology rule that landed on the same surface), the
    higher-quality source wins per ``_ERA_REFLEX_SOURCE_PRIORITY``.

    Empty languages are omitted (no entry in the returned dict).
    Returns ``{}`` when:

    * the root's language has no era family (proto-languages,
      untracked classical languages),
    * no cluster mates / descent edges / period-form rows / phonology
      rule walks match any target language.

    Computed at bundle-build time only — the runtime caller doesn't
    have DB access and reads from the bundle's ``era_reflexes`` field.
    """
    from wyrd.generators.kenning.era.cells import (
        CANONICAL_LANGUAGE_FOR_CELL,
        language_family,
    )

    family = language_family(root_language)
    if family is None:
        return {}
    target_languages: set[str] = {
        lang for (fam, _cell), lang in CANONICAL_LANGUAGE_FOR_CELL.items() if fam == family
    }
    out: dict[str, list[dict[str, str]]] = {}
    for target_language in sorted(target_languages):
        reflexes = etymon_era_reflexes(db, root_id, target_language=target_language)
        if not reflexes:
            continue
        # Dedupe by form, keeping the highest-quality source on
        # collision. Same form might surface via cluster (high) and
        # phonology-rule (low) — prefer the cluster.
        best: dict[str, str] = {}
        for r in reflexes:
            existing = best.get(r.form)
            if existing is None or _better_era_reflex_source(r.source, existing):
                best[r.form] = r.source
        out[target_language] = [{"form": form, "source": best[form]} for form in sorted(best)]
    return out


# Lower number = higher quality. Used by _fetch_root_era_reflexes when
# the same form surfaces via multiple tiers, and by _emit_era_reflexes
# when multiple linked families contribute the same form. Unknown
# sources fall through to default priority so a future tier doesn't
# silently downgrade everything.
_ERA_REFLEX_SOURCE_PRIORITY: dict[str, int] = {
    "cluster": 0,
    "descent": 1,
    "period-form": 2,
    "phonology-rule:v1": 3,
}
_ERA_REFLEX_SOURCE_DEFAULT_PRIORITY: int = 2


def _better_era_reflex_source(candidate: str, current: str) -> bool:
    """True iff ``candidate`` outranks ``current`` per the era-reflex
    source priority. Used to resolve same-form collisions in both
    ``_fetch_root_era_reflexes`` (per-tier dedupe) and
    ``_emit_era_reflexes`` (cross-family merge). Unknown sources fall
    through to default priority on both sides — the comparison stays
    well-defined and a new tier doesn't silently win or lose against
    everything."""
    return _ERA_REFLEX_SOURCE_PRIORITY.get(
        candidate, _ERA_REFLEX_SOURCE_DEFAULT_PRIORITY
    ) < _ERA_REFLEX_SOURCE_PRIORITY.get(current, _ERA_REFLEX_SOURCE_DEFAULT_PRIORITY)


def _build_forms_by_lang(root_row: Any, member_rows: list[Any]) -> dict[str, list[str]]:
    """Group canonical forms by language, root-first.

    Root form first per language so the lemma's own canonical_form leads
    the list (predictable order for snapshot tests).
    """
    forms_by_lang: dict[str, list[str]] = {}
    forms_by_lang.setdefault(root_row["language"], []).append(root_row["canonical_form"])
    for r in member_rows:
        bucket = forms_by_lang.setdefault(r["language"], [])
        if r["canonical_form"] not in bucket:
            bucket.append(r["canonical_form"])
    return forms_by_lang


_GLOSS_SPLIT_RE = re.compile(r"\s*[,;]\s*|\s+or\s+", re.IGNORECASE)


def _fetch_member_glosses(db: LexiconDB, member_ids: list[int]) -> list[str]:
    """Distinct sorted glosses across the family's members.

    Drops glosses that are concatenations of already-present sibling
    glosses (mining noise where the LLM emitted a comma/semicolon list as
    a single gloss when the source bullet split it). Concretely, if
    'lake', 'pond' are present and 'lake, pond' also got extracted, drop
    the latter — it adds no information and clutters the explainer.
    Splitter recognizes ',', ';', and ' or '; case-insensitive on the
    'or' connector. Single-token glosses are kept unconditionally.
    """
    placeholders = ",".join("?" * len(member_ids))
    raw = [
        row["gloss"]
        for row in db.conn.execute(
            f"SELECT DISTINCT gloss FROM etymon_gloss "
            f"WHERE etymon_id IN ({placeholders}) ORDER BY gloss",
            member_ids,
        )
    ]
    return _filter_concatenation_glosses(raw)


def _filter_concatenation_glosses(glosses: list[str]) -> list[str]:
    """Drop glosses whose tokens are entirely covered by other singleton
    glosses already in the list. Pure (no DB), so the caller's order
    is preserved for the survivors."""
    singleton_set = {g.strip().lower() for g in glosses if not _GLOSS_SPLIT_RE.search(g)}
    out: list[str] = []
    for g in glosses:
        if g.strip().lower() in singleton_set and not _GLOSS_SPLIT_RE.search(g):
            out.append(g)
            continue
        tokens = [t.strip().lower() for t in _GLOSS_SPLIT_RE.split(g) if t.strip()]
        if len(tokens) > 1 and all(t in singleton_set for t in tokens):
            continue
        out.append(g)
    return out


def _fetch_member_tags(db: LexiconDB, member_ids: list[int]) -> list[str]:
    """Distinct sorted tags across the family's members."""
    placeholders = ",".join("?" * len(member_ids))
    return [
        row["tag"]
        for row in db.conn.execute(
            f"SELECT DISTINCT tag FROM etymon_tag WHERE etymon_id IN ({placeholders}) ORDER BY tag",
            member_ids,
        )
    ]


def _fetch_cluster_mate_tags(db: LexiconDB, root_id: int) -> list[str]:
    """wyrd-i1s1: tags from cognate-cluster mates of ``root_id``.

    The family rollup (``_build_family_rollup``) traverses
    ``merged_into_id`` and ``lemma_id`` chains only — cluster mates
    sharing a ``cognate_id`` are SEPARATE family roots. Their tags
    don't surface in the rolled-up family's tag list.

    But cluster mates of a promoted OE root (typically ME / ModE
    cognates that didn't pass the per-language witness threshold on
    their own) carry valuable semantic-tag signal that the bundle
    consumer ought to see attached to the bundle subject the OE root
    grounds. This helper pulls those tags so they can be merged into
    ``family["tags"]`` alongside ``_fetch_member_tags``'s output.

    Excludes the root itself (its tags ride in via ``_fetch_member_tags``)
    and OCR-merge tombstones. Returns an empty list when the root has
    no ``cognate_id`` (no cluster) — most non-promoted etymons.
    """
    return [
        row["tag"]
        for row in db.conn.execute(
            """
            SELECT DISTINCT t.tag
            FROM etymon mate
            JOIN etymon_tag t ON t.etymon_id = mate.id
            WHERE mate.cognate_id = (
                SELECT cognate_id FROM etymon WHERE id = ?
              )
              AND mate.cognate_id IS NOT NULL
              AND mate.id != ?
              AND mate.merged_into_id IS NULL
            ORDER BY t.tag
            """,
            (root_id, root_id),
        )
    ]


def _fetch_member_reflex_links(db: LexiconDB, member_ids: list[int]) -> dict[int, list[int]]:
    """Per-reflex linked etymon ids (within this family).

    Lets us narrow the exported language array per word at group-build
    time — without this, a subject grouping a Celtic family and an OE
    family would emit each reflex's word with both languages even when
    the reflex only links to one of them.
    """
    placeholders = ",".join("?" * len(member_ids))
    reflex_links: dict[int, list[int]] = {}
    for row in db.conn.execute(
        f"SELECT re.reflex_id, re.etymon_id "
        f"FROM reflex_etymon re "
        f"WHERE re.etymon_id IN ({placeholders}) "
        f"ORDER BY re.reflex_id, re.etymon_id",
        member_ids,
    ):
        reflex_links.setdefault(row["reflex_id"], []).append(row["etymon_id"])
    return reflex_links


def _fetch_member_reflexes(
    db: LexiconDB, member_ids: list[int], reflex_links: dict[int, list[int]]
) -> list[dict[str, Any]]:
    """Reflex rows (modern surface forms) linked to any family member."""
    placeholders = ",".join("?" * len(member_ids))
    return [
        {
            "id": row["id"],
            "surface_form": row["surface_form"],
            "position": row["position"],
            "linked_member_ids": reflex_links.get(row["id"], []),
        }
        for row in db.conn.execute(
            f"SELECT DISTINCT r.id, r.surface_form, r.position "
            f"FROM reflex r "
            f"JOIN reflex_etymon re ON re.reflex_id = r.id "
            f"WHERE re.etymon_id IN ({placeholders}) "
            f"ORDER BY r.position, r.surface_form",
            member_ids,
        )
    ]


_LOW_CONFIDENCE_METHODS = frozenset({"fuzzy-search-v1", "llm-disambiguated-v1"})


def _fetch_member_variants(
    db: LexiconDB, member_ids: list[int], canonical_forms_lower: set[str]
) -> dict[int, list[tuple[str, int]]]:
    """Spelling variant pool per member_id (D18).

    Pulls from etymon_text_match.matched_form: real attested 19th-c.
    spellings (denu/dene/denū/dená) and post-disambiguator winners. These
    are the surface-form randomization targets for archaic-feel
    generation. Dedupes against canonical_forms_lower so the pool only
    contains FORMS NEW TO THE GENERATOR — emitting "denu" when "denu" is
    also the canonical_form would be redundant.

    Drops "low-confidence singletons": variants whose total match_count
    is 1 AND every contributing row was produced by fuzzy-search or
    llm-disambiguator (vs reverse-search, which scans for canonical-form
    occurrences and is high-precision). A 1-edit-distance fuzzy hit with
    a single attestation in the corpus is the dominant OCR-noise class
    (saearp, drnim, etc.); requiring either ≥2 attestations or any
    high-confidence method removes them without throwing away legitimate
    archaic spellings.
    """
    placeholders = ",".join("?" * len(member_ids))
    member_variants: dict[int, list[tuple[str, int]]] = {}
    for row in db.conn.execute(
        f"SELECT etymon_id, matched_form, "
        f"  SUM(match_count) AS total_count, "
        f"  GROUP_CONCAT(DISTINCT method) AS methods "
        f"FROM etymon_text_match "
        f"WHERE etymon_id IN ({placeholders}) "
        f"GROUP BY etymon_id, LOWER(matched_form) "
        f"ORDER BY etymon_id, total_count DESC, matched_form",
        member_ids,
    ):
        if row["matched_form"].lower() in canonical_forms_lower:
            continue
        methods = set((row["methods"] or "").split(","))
        if row["total_count"] <= 1 and methods.issubset(_LOW_CONFIDENCE_METHODS):
            continue
        member_variants.setdefault(row["etymon_id"], []).append(
            (row["matched_form"], row["total_count"])
        )
    return member_variants


_NON_SCHOLAR_SOURCES = frozenset({"rando-port"})


def _fetch_member_citations(db: LexiconDB, member_ids: list[int]) -> dict[int, list[str]]:
    """Distinct sorted scholarly source_ids per member_id from
    etymon_citation. The runtime explainer surfaces this so a GM holding
    a generated name can see which scholars attest each morpheme.
    Sorted alphabetically for deterministic bundle output.

    Filters out non-scholarly seeds (rando-port — the Wikipedia-derived
    legacy bootstrap that ships every legacy etymon but isn't a real
    citation a GM would recognize). Members with no scholarly citations
    drop out of the result entirely so the emitter omits the
    `<lang>_citations` field on rando-only words.
    """
    placeholders = ",".join("?" * len(member_ids))
    member_citations: dict[int, list[str]] = {}
    for row in db.conn.execute(
        f"SELECT etymon_id, source_id "
        f"FROM etymon_citation "
        f"WHERE etymon_id IN ({placeholders}) "
        f"GROUP BY etymon_id, source_id "
        f"ORDER BY etymon_id, source_id",
        member_ids,
    ):
        if row["source_id"] in _NON_SCHOLAR_SOURCES:
            continue
        member_citations.setdefault(row["etymon_id"], []).append(row["source_id"])
    return member_citations


def _fetch_member_attested_years(db: LexiconDB, member_ids: list[int]) -> dict[int, int]:
    """Earliest attested year per member_id, drawn from BOTH row sources
    that carry year evidence (D5-1):

    * ``etymon_text_match.attested_year`` (PR #47 / wyrd-3ux) —
      reverse-search rows where the etymon's canonical form was found
      in a source body, paired with a form-attached year citation.
    * ``toponym_etymology.attested_year`` (PR #53 / wyrd-bag) — joined
      via toponym_etymology_element so a year cited against a toponym
      lands on each of its breakdown's element etymons.

    Returns ``{member_id: earliest_year}``; members with no attested
    year on either side are absent (caller emits no ``_attested_years``
    sibling for them — D5-2 generator interprets None as 'no era
    filter applies'). Sorted output is incidental — the dict is built
    by iterating SQL results, then the consumer re-sorts when emitting.
    """
    # Use a CTE to bind member_ids exactly once. A naive UNION ALL with
    # two `IN (?,?,?)` branches would require duplicating the bind list,
    # which is fragile if a third year-source is added in future. The
    # CTE makes the contract explicit: 'these are the etymons we care
    # about; gather years from any source that mentions them'.
    targets_values = ",".join("(?)" for _ in member_ids)
    member_years: dict[int, int] = {}
    cur = db.conn.execute(
        f"""
        WITH targets(etymon_id) AS (VALUES {targets_values})
        SELECT etymon_id, MIN(year) AS earliest_year FROM (
            SELECT etm.etymon_id, etm.attested_year AS year
            FROM etymon_text_match etm
            JOIN targets t ON t.etymon_id = etm.etymon_id
            WHERE etm.attested_year IS NOT NULL
            UNION ALL
            SELECT tee.etymon_id, te.attested_year AS year
            FROM toponym_etymology_element tee
            JOIN targets t ON t.etymon_id = tee.etymon_id
            JOIN toponym_etymology te ON te.id = tee.toponym_etymology_id
            WHERE te.attested_year IS NOT NULL
        )
        GROUP BY etymon_id
        """,
        member_ids,
    )
    for row in cur:
        member_years[row["etymon_id"]] = row["earliest_year"]
    return member_years


def _compute_member_descendants(member_rows: list[Any]) -> dict[int, list[int]]:
    """Transitive descendants per member_id (DB-free DFS).

    Lets a reflex linked to a lemma pick up the lemma's inflected children
    + OCR-cluster losers rather than just the directly-linked etymon. The
    graph is shallow (D22 flatten-at-merge-time keeps chains at depth ≤
    2), so a single inversion pass suffices.
    """
    children: dict[int, list[int]] = {}
    for r in member_rows:
        parent_id = r["lemma_id"] or r["merged_into_id"]
        if parent_id is not None:
            children.setdefault(parent_id, []).append(r["id"])
    member_descendants: dict[int, list[int]] = {}
    for r in member_rows:
        member_id = r["id"]
        descendants = [member_id]
        stack = list(children.get(member_id, []))
        while stack:
            d = stack.pop()
            if d in descendants:
                continue
            descendants.append(d)
            stack.extend(children.get(d, []))
        member_descendants[member_id] = descendants
    return member_descendants
