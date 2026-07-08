"""SQL data loaders for the bundle build (leaf module — no internal deps).

``_gather_family`` is the central entry point: given a promoted-root
etymon id + the list of member ids in its cognate cluster, run every
``_fetch_member_*`` SQL loader once, hydrate a single per-family
context dict, and return it for downstream subject/word assembly.

The era-reflex picker (``_fetch_family_era_reflexes``) uses the same
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
from collections.abc import Iterable
from typing import Any

from wyrd.generators.kenning.lexicon.collapse_detect import is_form_of_pointer
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.era_reflex import EraReflex, etymon_era_reflexes


def _gather_family(
    db: LexiconDB,
    root_id: int,
    member_ids: list[int],
    attested_years_by_member: dict[int, int] | None = None,
) -> dict[str, Any] | None:
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

    era_reflexes = _fetch_family_era_reflexes(db, member_ids, root_row["language"])
    forms_by_lang = _build_forms_by_lang(root_row, member_rows)
    # wyrd-nxhh: pre-fix the modern surfaces only landed in
    # ``era_reflexes`` (consumed by KenningRewind for era progression).
    # The main kenning generator reads forms_by_lang via per-language
    # source siblings (modern_english / middle_english / etc.), so
    # chester / -cester / -caster never reached it even though ceaster
    # was promoted and its cluster carried them. Merge here makes the
    # cluster surfaces sampleable; era_reflexes stays emitted unchanged.
    _merge_era_reflexes_into_forms_by_lang(forms_by_lang, era_reflexes)

    return {
        "root_id": root_id,
        "root_canonical_form": root_row["canonical_form"],
        "root_language": root_row["language"],
        "modifier_type": root_row["modifier_type"],
        "position_pref": root_row["position_pref"],
        "forms_by_lang": forms_by_lang,
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
        # wyrd-4zyb: when the caller bulk-prefetched attested years for ALL
        # members (the _iterate_families_with_progress path), slice that map
        # instead of re-running the per-family UNION/MIN query — that per-root
        # query was ~96% of collect_families' wall time. None → the legacy
        # per-root fallback (direct _gather_family callers, e.g. unit tests); a
        # provided map (even empty) is authoritative — sliced directly, no query.
        "member_attested_years": (
            {
                mid: attested_years_by_member[mid]
                for mid in member_ids
                if mid in attested_years_by_member
            }
            if attested_years_by_member is not None
            else _fetch_member_attested_years(db, member_ids)
        ),
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
        "era_reflexes": era_reflexes,
    }


# Tiers 1-3 of ``etymon_era_reflexes`` are attested (cluster mate,
# descent edge, period-form projection); Tier 4 (``phonology-rule:v1``)
# is rule-derived with no mining evidence. The bundle's
# ``era_reflexes`` field preserves ``source`` so KenningRewind can
# render Tier-4 forms differently. The main generator has no such
# differentiation — it samples forms_by_lang uniformly — so we exclude
# Tier 4 from the merge to preserve the "attested only" contract the
# generator's source array had pre-wyrd-nxhh.
_ATTESTED_ERA_REFLEX_SOURCES: frozenset[str] = frozenset({"cluster", "descent", "period-form"})


_MODERN_STAGE_LANGUAGES: frozenset[str] | None = None


def _modern_stage_languages() -> frozenset[str]:
    """The canonical language tag of every family's MODERN cell
    (``modern`` / ``early-modern``) — e.g. ``modern-english``, ``welsh``,
    ``irish``, ``french``. The wyrd-rogd.11 derived-name filter applies only
    to the modern stage, where the cluster proliferates with derived proper
    nouns; historical stages carry the period common forms. Built lazily +
    memoised (matches the lazy CANONICAL_LANGUAGE_FOR_CELL import below)."""
    global _MODERN_STAGE_LANGUAGES
    if _MODERN_STAGE_LANGUAGES is None:
        from wyrd.generators.kenning.era.cells import CANONICAL_LANGUAGE_FOR_CELL

        _MODERN_STAGE_LANGUAGES = frozenset(
            lang
            for (_fam, cell), lang in CANONICAL_LANGUAGE_FOR_CELL.items()
            if cell in ("modern", "early-modern") and lang
        )
    return _MODERN_STAGE_LANGUAGES


# wyrd-pzg5: closed-class function words (articles, determiners, demonstratives,
# pronouns, conjunctions, interjections). These ride into the modern stage via
# deep cluster over-merge (a morpheme's grid showing 'the' / 'they' / 'thy') but
# are NEVER a place-name element's modern reflex. Matched on the EXACT lowercased
# form only (no substring), so a legit element that merely contains these letters
# is untouched. Deliberately omits anything that is also a place-name element —
# notably 'by' (the Norse -by settlement element) and bare prepositions
# (in/on/at/up) whose element-hood is ambiguous — erring toward keeping.
_MODERN_REFLEX_STOPWORDS: frozenset[str] = frozenset(
    {
        # articles / determiners / demonstratives ('a'/'an' note: 'a' is already
        # dropped by the single-char rule, so only multi-char entries are listed)
        "the",
        "an",
        "this",
        "that",
        "these",
        "those",
        # pronouns (personal / possessive / interrogative / relative);
        # single-char 'i' is covered by the single-char rule, so it's omitted here
        "me",
        "my",
        "mine",
        "we",
        "us",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "thy",
        "thee",
        "thou",
        "thine",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "who",
        "whom",
        "whose",
        "which",
        "what",
        # conjunctions
        "and",
        "or",
        "but",
        "nor",
        "not",
        "than",
        "then",
        # interjections / fillers
        "um",
        "uh",
        "oh",
        "ah",
        "eh",
    }
)


def _is_derived_name_pollution(form: str) -> bool:
    """wyrd-rogd.11 + wyrd-pzg5: True when a MODERN-stage reflex is NOT the
    morpheme's own common-noun continuation — a derived proper noun (place or
    personal name), a capitalized cross-orthography foreign cognate, a
    segmentation fragment, or a closed-class function word.

    The cluster/descent tiers dump an etymon's whole modern-attested cluster
    into the modern stage — derived toponyms (``West Ham``, ``Coldingham``,
    ``Westbourne``), anthroponyms (``Wesley``, ``Derek``, ``Hank``), and
    mistagged foreign cognates (``Düne``, ``Zaun``, ``Ouest``) — none of which
    is the morpheme's modern English reflex (``ton``, ``hill``, ``-bury``).
    ``edge_type`` is unreliable (OE ``tūn`` → ``Pendleton`` is tagged
    'inheritance'), so we gate on the FORM:

    - internal whitespace (a multi-word toponym, e.g. ``mons pubis``);
    - an initial uppercase letter (a proper noun or a capitalized foreign
      cognate like ``Düne`` whose orthography betrays a non-English form);
    - a single-character form (``s``, ``k``) — a segmentation/OCR fragment;
      the shortest genuine elements (``ea``, ``ey``) are two characters, and
      the generator already always drops single-character fragments;
    - a form with NO alphabetic character (``123``, ``??``) — a numeric/
      punctuation artifact, never a reflex;
    - a closed-class function word (``the``, ``they``, ``thy`` — see
      ``_MODERN_REFLEX_STOPWORDS``).

    A genuine reflex is a lowercase common form. Erring toward an EMPTY modern
    stage is intended per the ticket."""
    f = form.strip()
    if not f or any(c.isspace() for c in f):  # empty, or multi-word (any whitespace)
        return True
    bare = f.strip("-")  # ignore affix hyphens for the length / stopword checks
    if len(bare) <= 1:  # single-char segmentation/OCR fragment
        return True
    if bare.lower() in _MODERN_REFLEX_STOPWORDS:
        return True
    first_alpha = next((c for c in f if c.isalpha()), "")
    if not first_alpha:  # digits / punctuation only — not a real reflex
        return True
    return first_alpha.isupper()


def _merge_era_reflexes_into_forms_by_lang(
    forms_by_lang: dict[str, list[str]],
    era_reflexes: dict[str, list[dict[str, str]]],
) -> None:
    """wyrd-nxhh: in-place merge of attested era_reflex surface forms
    into ``forms_by_lang`` so cognate-cluster / direct-descent surfaces
    join the per-language sibling arrays the main kenning generator
    reads.

    Each ``era_reflexes`` entry is a target-language → list of
    ``{form, source}`` dicts from ``_fetch_family_era_reflexes``. We
    filter to ``source in _ATTESTED_ERA_REFLEX_SOURCES`` (Tiers 1-3 —
    excluding Tier-4 phonology-rule forms), then union the surviving
    forms into ``forms_by_lang[target_language]`` preserving insertion
    order and dropping exact duplicates.

    A target language whose era_reflex entries are all Tier-4 produces
    no bucket — we don't want to manufacture an empty ``modern_english:
    []`` sibling in the bundle when nothing attested survives the filter.

    No-op for empty ``era_reflexes`` (proto-languages or roots whose
    cluster has no target-language mates), which is the common path
    for non-English-family families today.
    """
    for target_language, entries in era_reflexes.items():
        attested = [
            entry["form"] for entry in entries if entry["source"] in _ATTESTED_ERA_REFLEX_SOURCES
        ]
        if not attested:
            continue
        bucket = forms_by_lang.setdefault(target_language, [])
        existing = set(bucket)
        for form in attested:
            if form not in existing:
                bucket.append(form)
                existing.add(form)


# European-feeder languages of British/European place-names: English + Germanic +
# Romance (incl. every Latin period/variety + Anglo-Norman) + Celtic + Hellenic +
# PIE, with their proto/old/middle/dialect codes. A cognate-cluster reflex whose
# descent parent is OUTSIDE this set is a cross-language Descendants-section
# re-entry (wiktextract noise), not a genuine reflex of a British place-name
# morpheme — see _foreign_reentry_ids. Generous on purpose: a MISSING feeder code
# would wrongly drop a legit reflex, so err toward inclusion.
_FEEDER_LANGUAGES: frozenset[str] = frozenset(
    {
        # English
        "old-english",
        "middle-english",
        "modern-english",
        "english",
        "enm",
        "ang",
        "scots",
        "sco",
        # Germanic
        "proto-germanic",
        "gem-pro",
        "gem-frk",
        "gmw-pro",
        "gmq-pro",
        "gmw-jdt",
        "gmq-osw",
        "gmq-oda",
        "old-norse",
        "non",
        "icelandic",
        "is",
        "faroese",
        "fo",
        "danish",
        "da",
        "swedish",
        "sv",
        "norwegian",
        "norwegian-bokmal",
        "norwegian-nynorsk",
        "no",
        "nb",
        "nn",
        "gothic",
        "got",
        "dutch",
        "nl",
        "dum",
        "middle-dutch",
        "odt",
        "old-dutch",
        "german",
        "de",
        "old-high-german",
        "goh",
        "middle-high-german",
        "gmh",
        "old-saxon",
        "osx",
        "old-frisian",
        "ofs",
        "frisian",
        "west-frisian",
        "fy",
        "saterland-frisian",
        "stq",
        "low-german",
        "nds",
        "middle-low-german",
        "gml",
        "luxembourgish",
        "lb",
        "limburgish",
        "li",
        "afrikaans",
        "af",
        "gsw",
        "bar",
        # Romance + Latin
        "latin",
        "la",
        "la-lat",
        "la-med",
        "la-vul",
        "la-ecc",
        "lat",
        "vulgar-latin",
        "itc-pro",
        "osp",
        "french",
        "fr",
        "old-french",
        "fro",
        "middle-french",
        "frm",
        "norman",
        "nrf",
        "xno",
        "italian",
        "it",
        "spanish",
        "es",
        "portuguese",
        "pt",
        "galician",
        "gl",
        "occitan",
        "oc",
        "pro",
        "catalan",
        "ca",
        "romanian",
        "ro",
        "sardinian",
        "sc",
        "roa-pro",
        "roa-oil",
        "roa-opt",
        "fur",
        "lld",
        # Celtic
        "proto-celtic",
        "cel-pro",
        "cel-bry-pro",
        "cel-gae-pro",
        "cel",
        "welsh",
        "cy",
        "old-welsh",
        "owl",
        "middle-welsh",
        "wlm",
        "cornish",
        "kw",
        "cnx",
        "breton",
        "br",
        "xbm",
        "obt",
        "irish",
        "ga",
        "old-irish",
        "sga",
        "middle-irish",
        "mga",
        "primitive-irish",
        "pgl",
        "scottish-gaelic",
        "gd",
        "manx",
        "gv",
        "gaulish",
        "xtg",
        "cel-gau",
        "pictish",
        "xpi",
        "brythonic",
        "goidelic",
        # Hellenic
        "ancient-greek",
        "grc",
        "greek",
        "el",
        "modern-greek",
        "gkm",
        "grc-koi",
        "koine",
        # Proto-Indo-European (the shared feeder root)
        "proto-indo-european",
        "ine-pro",
    }
)


def _collect_best_by_target(
    db: LexiconDB, target_languages: set[str], sorted_members: list[int]
) -> dict[str, dict[str, EraReflex]]:
    """Per target-language, the best reflex per surface form unioned across all
    family members (before the foreign-reentry filter). For each target, dedupe
    by form across members keeping the highest-quality source on collision
    (``_better_era_reflex_source``; ties keep the lowest-id member since
    ``sorted_members`` is sorted). The modern stage additionally strips derived
    proper nouns / mistagged foreign cognates (wyrd-rogd.11
    ``_is_derived_name_pollution``) so the cleaned set feeds BOTH the era-grid
    and the forms_by_lang merge. Targets with no reflex are omitted."""
    modern_languages = _modern_stage_languages()
    best_by_target: dict[str, dict[str, EraReflex]] = {}
    for target_language in sorted(target_languages):
        is_modern = target_language in modern_languages
        best: dict[str, EraReflex] = {}
        for member_id in sorted_members:
            reflexes = etymon_era_reflexes(db, member_id, target_language=target_language)
            if is_modern:
                reflexes = [r for r in reflexes if not _is_derived_name_pollution(r.form)]
            for r in reflexes:
                existing = best.get(r.form)
                if existing is None or _better_era_reflex_source(r.source, existing.source):
                    best[r.form] = r
        if best:
            best_by_target[target_language] = best
    return best_by_target


def _foreign_reentry_ids(db: LexiconDB, etymon_ids: set[int]) -> set[int]:
    """The subset of ``etymon_ids`` that are FOREIGN RE-ENTRIES: an era-reflex
    etymon that descends (inheritance / borrowing / derivation edge) from at
    least one language OUTSIDE ``_FEEDER_LANGUAGES``. These are cross-language
    descendants wiktextract's Descendants section pulled into the cognate cluster
    — modern-english ``kaona`` (← Hawaiian), ``argaman`` (← Hebrew), ``bungoo``
    (← Dyirbal) — not genuine modern reflexes of a British place-name morpheme,
    so they pollute the era-grid's modern cell (wyrd-6hbv Phase 2).

    Flagged when ANY parent is non-feeder (so a junk edge that also attaches a
    spurious feeder parent — e.g. ``bank`` → ``bungoo`` alongside Dyirbal — is
    still caught). A reflex with no descent parents, or whose every parent is a
    feeder language, is kept. One batched query over the family's reflex ids."""
    if not etymon_ids:
        return set()
    placeholders = ",".join("?" * len(etymon_ids))
    foreign: set[int] = set()
    for row in db.conn.execute(
        f"SELECT ed.child_id AS cid, p.language AS plang "
        f"FROM etymon_descent ed JOIN etymon p ON p.id = ed.parent_id "
        f"WHERE ed.child_id IN ({placeholders}) "
        f"  AND ed.edge_type IN ('inheritance', 'borrowing', 'derivation')",
        list(etymon_ids),
    ):
        if (row["plang"] or "").lower() not in _FEEDER_LANGUAGES:
            foreign.add(row["cid"])
    return foreign


def _fetch_family_era_reflexes(
    db: LexiconDB, member_ids: list[int], root_language: str
) -> dict[str, list[dict[str, str]]]:
    """wyrd-obpw Phase 3.3 + wyrd-jbcu source-aware schema: per-FAMILY
    era reflexes for bundle export.

    Returns a dict mapping target language tag → sorted list of
    ``{"form": str, "source": str, "gloss"?: str}`` dicts. The SPA-side
    rewinder consumes this at runtime; ``source`` distinguishes attestation-
    backed reflexes ('cluster' / 'descent' / 'period-form') from
    phonology-rule-derived ones ('phonology-rule:v1') so consumers
    can render inferred forms differently if they want to. ``gloss``
    (wyrd-rogd.1, sparse) is the reflex etymon's own representative meaning,
    for the SPA era-grid's drift display.

    For each language tag in ``CANONICAL_LANGUAGE_FOR_CELL`` of the
    family, calls ``etymon_era_reflexes`` for EVERY family member and
    UNIONS the results (wyrd-rogd.16). The family is the unified morpheme,
    so its grid must aggregate every member's cluster — not just the root's.
    rogd.9 folds a later-era reflex into its earlier-era ancestor's family;
    computing the grid from the ancestor-root's cluster ALONE orphaned the
    reflex's own (often richer) cluster — e.g. modern 'green' (39 cluster
    reflexes) folded into OE 'green' (1) collapsed the grid 39→1. Unioning
    keeps the morpheme's full cross-era reflex set. When the same form
    arrives via multiple members / tiers, the higher-quality source wins
    (``_ERA_REFLEX_SOURCE_PRIORITY``); ties keep the lowest-id member's
    (``member_ids`` is iterated sorted) for deterministic glosses.

    Empty languages are omitted (no entry in the returned dict).
    Returns ``{}`` when the family's language has no era family
    (proto-languages, untracked classical languages) or no member yields a
    reflex for any target. Computed at bundle-build time only.
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
    best_by_target = _collect_best_by_target(db, target_languages, sorted(member_ids))
    # wyrd-6hbv Phase 2: drop foreign re-entries (cross-language Descendants-section
    # noise — kaona/argaman/bungoo) from EVERY stage, in one batched descent query
    # over all collected reflexes, so the era-grid's modern cell carries only
    # genuine feeder-descended reflexes (and they don't leak into forms_by_lang
    # via _merge_era_reflexes_into_forms_by_lang either).
    foreign = _foreign_reentry_ids(
        db, {r.etymon_id for best in best_by_target.values() for r in best.values()}
    )
    out: dict[str, list[dict[str, str]]] = {}
    for target_language, best in best_by_target.items():
        kept = {form: r for form, r in best.items() if r.etymon_id not in foreign}
        if not kept:
            continue
        glosses = _fetch_reflex_glosses(db, [r.etymon_id for r in kept.values()])
        entries = [_reflex_entry(kept[form], glosses) for form in sorted(kept)]
        out[target_language] = _case_fold_reflex_entries(entries)
    return out


def _case_fold_reflex_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """D55: case is a RENDER concern, never storage. A reflex form differing only
    by case is a duplicate that worsens both the UI and the statistics, so fold
    every reflex surface to lowercase (the render re-applies position case, as it
    already does for the case-folded meaning identity, D53) and collapse
    case-variants of the same surface. Runs AFTER the derived-name pollution
    filter (``_is_derived_name_pollution``, which reads case to drop proper-noun
    re-entries) so those drops still fire. ``str.lower`` keeps macrons/diacritics
    (D45).

    Collapse semantics — glosses are SPARSE (``_reflex_entry`` omits them when the
    reflex etymon has none), so the collapse must NOT hinge on both sides carrying
    the same gloss:

    * Group by the folded surface, then by gloss within it. Two entries with the
      same folded surface but DIFFERENT non-empty glosses are homographs (D54 —
      ``ton``=town vs ``ton``=mass) and are kept apart.
    * A GLOSSLESS variant is an unlabeled instance of the same reflex, not a
      distinct sense. With exactly one distinct gloss present it MERGES into that
      gloss — otherwise the reported bug reappears: ``West``+gloss and
      ``west``+no-gloss would take different ``(folded, gloss)`` keys and both
      survive. With zero or ≥2 distinct glosses it can't be attributed, so one
      glossless entry is kept alongside any homographs.
    * Within any collapse the survivor is the HIGHEST-quality source
      (``_better_era_reflex_source``: cluster > descent > period-form >
      phonology-rule), NOT the first-seen — ``entries`` arrives sorted by the
      original CASE-SENSITIVE form, so a capitalized artifact ("West") sorts
      first and first-seen would let it discard the lowercase member's better
      source. Equal sources keep the first-seen (deterministic — input is
      sorted).

    Re-sorts by the folded surface."""
    by_folded: dict[str, list[dict[str, str]]] = {}
    for e in entries:
        by_folded.setdefault((e.get("form") or "").lower(), []).append(e)

    out: list[dict[str, str]] = []
    for folded, group in by_folded.items():
        out.extend(_collapse_folded_group(folded, group))
    return sorted(out, key=lambda e: e["form"])


def _collapse_folded_group(folded: str, group: list[dict[str, str]]) -> list[dict[str, str]]:
    """Collapse one folded-surface group into its distinct senses, per the gloss
    / glossless / homograph rules in :func:`_case_fold_reflex_entries`. Each
    surviving entry's ``form`` is set to ``folded`` (the lowercased surface)."""
    glossed: dict[str, dict[str, str]] = {}  # gloss -> best-source entry
    glossless: dict[str, str] | None = None
    for e in group:
        gloss = e.get("gloss")
        if gloss:
            incumbent = glossed.get(gloss)
            if incumbent is None or _better_era_reflex_source(
                e.get("source", ""), incumbent.get("source", "")
            ):
                glossed[gloss] = e
        elif glossless is None or _better_era_reflex_source(
            e.get("source", ""), glossless.get("source", "")
        ):
            glossless = e
    if glossless is not None and len(glossed) == 1:
        # A single sense + unlabeled variants → the same reflex; fold the
        # glossless into that gloss (keep its source only if it outranks).
        gloss, incumbent = next(iter(glossed.items()))
        if _better_era_reflex_source(glossless.get("source", ""), incumbent.get("source", "")):
            glossed[gloss] = {**glossless, "gloss": gloss}
        glossless = None
    out = [{**e, "form": folded} for e in glossed.values()]
    if glossless is not None:
        out.append({**glossless, "form": folded})
    return out


def _reflex_entry(reflex: EraReflex, glosses: dict[int, str]) -> dict[str, str]:
    """The bundle entry for one reflex: ``{form, source}`` plus a ``gloss``
    when the reflex etymon has one (sparse — omitted otherwise so glossless
    bundles don't carry empty strings). wyrd-rogd.1."""
    entry = {"form": reflex.form, "source": reflex.source}
    gloss = glosses.get(reflex.etymon_id)
    if gloss:
        entry["gloss"] = gloss
    return entry


def _fetch_reflex_glosses(db: LexiconDB, etymon_ids: list[int]) -> dict[int, str]:
    """A single representative gloss per reflex etymon_id (wyrd-rogd.1).

    Drives the SPA era-grid's drift display: a swapped era variant is a
    cognate cluster-mate whose meaning can have drifted from the morpheme's
    own (saint→seinte is the same word; worth→hirdels is a cognate that no
    longer means 'worth'), so the cell carries its own gloss to surface that
    shift. Picks the alphabetically-first non-pointer gloss for determinism;
    absent when the etymon carries only ``is_form_of_pointer`` cross-references
    or no gloss at all."""
    # Dedupe ids (distinct forms in `best` can share an etymon_id) so the IN
    # clause carries no redundant placeholders; dict.fromkeys keeps it
    # deterministic.
    ids = list(dict.fromkeys(etymon_ids))
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    by_id: dict[int, list[str]] = {}
    for row in db.conn.execute(
        f"SELECT etymon_id, gloss FROM etymon_gloss "
        f"WHERE etymon_id IN ({placeholders}) ORDER BY gloss",
        ids,
    ):
        if is_form_of_pointer(row["gloss"]):
            continue
        by_id.setdefault(row["etymon_id"], []).append(row["gloss"])
    return {eid: glosses[0] for eid, glosses in by_id.items() if glosses}


# Lower number = higher quality. Used by _fetch_family_era_reflexes when
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
    ``_fetch_family_era_reflexes`` (per-tier dedupe) and
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
    # wyrd-6r09: drop form-of pointer glosses ("h-prothesized form of ea",
    # "alternative form of burg"). They are cross-references, not meanings.
    # When a collapse folds a form-of etymon into its lemma, the lemma's
    # real glosses carry the meaning across the family; the folded member's
    # pointer gloss must not leak into the bundle as a displayed sense.
    raw = [g for g in raw if not is_form_of_pointer(g)]
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


def _attested_years_sql(members_table: str) -> str:
    """The ``{etymon_id: earliest_attested_year}`` subquery shared by
    :func:`_bulk_attested_years` (members in a temp table) and
    :func:`_fetch_member_attested_years` (members in a CTE). Single-sources the
    set of year-bearing tables — ``etymon_text_match.attested_year`` +
    ``toponym_etymology.attested_year`` (via ``toponym_etymology_element``) — so
    adding a THIRD year source touches ONE place, not two (the drift the
    per-family docstring warned about).

    ``members_table`` is a trusted internal identifier (the temp table or CTE
    name the caller binds the member-id set into) — NEVER user input, so the
    f-string interpolation carries no injection risk. Members with no attested
    year on either source are absent from the result (both branches filter
    ``attested_year IS NOT NULL``)."""
    # Defence-in-depth: the table name is interpolated (a bind param can't name a
    # table), so enforce the trusted-identifier contract at runtime — a future
    # caller passing anything but a bare identifier fails loud here rather than
    # composing injectable SQL.
    if not members_table.isidentifier():
        raise ValueError(f"members_table must be a bare identifier, got {members_table!r}")
    return f"""
        SELECT etymon_id, MIN(year) AS earliest_year FROM (
            SELECT etm.etymon_id, etm.attested_year AS year
            FROM etymon_text_match etm
            JOIN {members_table} t ON t.etymon_id = etm.etymon_id
            WHERE etm.attested_year IS NOT NULL
            UNION ALL
            SELECT tee.etymon_id, te.attested_year AS year
            FROM toponym_etymology_element tee
            JOIN {members_table} t ON t.etymon_id = tee.etymon_id
            JOIN toponym_etymology te ON te.id = tee.toponym_etymology_id
            WHERE te.attested_year IS NOT NULL
        )
        GROUP BY etymon_id
    """


def _bulk_attested_years(db: LexiconDB, member_ids: Iterable[int]) -> dict[int, int]:
    """wyrd-4zyb: earliest attested year per member for ALL family members in
    ONE pass — the bulk counterpart of :func:`_fetch_member_attested_years`.

    The per-family version re-ran its UNION/MIN-over-two-tables CTE query once
    per promoted root (~67K times) and was ~96% of ``collect_families`` wall
    time. This materializes the whole member-id set into an INDEXED temp table
    and joins against it in a single SQL pass. Same rows, same
    ``{member_id: earliest_year}`` shape
    (members with no attested year on either source are absent); per-member, so
    a family just slices the ids it owns. The temp table is connection-local
    and dropped on the way out.
    """
    ids = list(member_ids)
    out: dict[int, int] = {}
    if not ids:
        return out
    conn = db.conn
    conn.execute("DROP TABLE IF EXISTS _bulk_attested_members")
    conn.execute("CREATE TEMP TABLE _bulk_attested_members (etymon_id INTEGER PRIMARY KEY)")
    cur = None
    try:
        # Commit the INSERT so no write transaction dangles on the shared
        # (StaticPool) connection after we return (Gemini, PR #613).
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO _bulk_attested_members(etymon_id) VALUES (?)",
                ((i,) for i in ids),
            )
        cur = conn.execute(_attested_years_sql("_bulk_attested_members"))
        for row in cur:
            out[row["etymon_id"]] = row["earliest_year"]
    finally:
        # Close the cursor before DROP — a still-open cursor holds a read lock,
        # so DROP would raise "database table is locked" and mask any in-flight
        # exception from the loop above.
        if cur is not None:
            cur.close()
        conn.execute("DROP TABLE IF EXISTS _bulk_attested_members")
    return out


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
        {_attested_years_sql("targets")}
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
