"""Subject + word assembly for the bundle build.

Groups per-family rows from ``_gather_family`` into subjects (one
subject = one canonical-form meaning), then assembles per-word
language buckets by absorbing each member's evidence into per-
language ``_BucketAccumulator`` instances.

A "subject" maps to one meanings.json top-level entry; a "word"
maps to one canonical reflex within that subject. The
``_partition_families_by_reflex`` step demotes a family to a single
synthesized word when its members don't carry distinct era reflexes;
otherwise each reflex anchor produces its own word.

The ``_absorb_member_*`` series is intentionally per-attribute (one
function per JSON field — variants / inflections / attested-years /
english-shaped / original-script / transliteration / pronunciation /
stratum / citations) so a new bundle field plumbs through as a single
absorb + a single ``_emit_*_list`` formatter addition.
"""

from __future__ import annotations

import functools
import json
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from wyrd.generators.kenning.era.cells import family_stage_order, language_family
from wyrd.generators.kenning.lexicon.bundle._emit import (
    _LANG_CODE_TO_JSON_FIELD,
    _BucketAccumulator,
    _emit_attested_years_list,
    _emit_english_shaped_list,
    _emit_inflection_list,
    _emit_original_script_list,
    _emit_phonological_vector_list,
    _emit_pronunciation_list,
    _emit_stratum_list,
    _emit_transliteration_list,
    _emit_variant_list,
    _synthesize_modern_usage,
)
from wyrd.generators.kenning.lexicon.bundle._family import (
    _better_era_reflex_source,
    _case_fold_reflex_entries,
)
from wyrd.generators.kenning.lexicon.pronunciation_backfill import surface_ipa


def _fill_surface_pronunciation(bucket: Any, json_field: str) -> None:
    """wyrd-vm8t Loop 4: deterministic G2P fallback for surface forms
    (variants/reflexes) that are not etymon canonical_forms and so carry no
    etymon IPA. Table languages only; gaps only (never overrides etymon IPA)."""
    for form in bucket.forms:
        if form not in bucket.pronunciation:
            ipa = surface_ipa(form, json_field)
            if ipa:
                bucket.pronunciation[form] = {"ipa": ipa, "dialect": None}


def _group_families_into_subjects(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group families by (modifier_type, glosses, tags) into meanings.json subjects.

    Each group becomes one subject. Reflexes linked to any family in the
    group become "words"; families without any reflex contribute a synthesized
    word using the family's canonical_form (for newly-mined etymons that
    haven't been wired into a modern surface form yet).
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for fam in families:
        key = (
            fam["modifier_type"] or "",
            tuple(sorted(fam["glosses"])),
            tuple(sorted(fam["tags"])),
        )
        groups.setdefault(key, []).append(fam)

    subjects: list[dict[str, Any]] = []
    for (mod_type, glosses_tuple, tags_tuple), fams in groups.items():
        words = _build_words_for_group(fams)
        if not words:
            continue
        subject: dict[str, Any] = {
            "meaning": list(glosses_tuple),
            "modifier_tags": list(tags_tuple),
            "modifier_type": mod_type or None,
            "words": words,
        }
        subjects.append(subject)

    # Fully-discriminating sort key: every field that varies across subjects
    # must be in the tuple. Otherwise ties fall back to dict insertion order,
    # which traces back to AUTOINCREMENT root_ids and re-shuffles on rebuild.
    subjects.sort(
        key=lambda s: (
            s.get("modifier_type") or "",
            tuple(s["meaning"]),
            tuple(s["modifier_tags"]),
            tuple(w["modern_usage"] for w in s["words"]),
        )
    )
    return subjects


def _build_words_for_group(fams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assemble the `words` list for a subject grouping multiple families.

    Sort families deterministically, partition into reflex-linked vs.
    reflex-less, then dispatch each group to a focused word-builder.
    """
    # Sort families by canonical_form/language so the export output is
    # deterministic — root_ids are AUTOINCREMENT and therefore unstable
    # across DB rebuilds, which would otherwise churn the diff in version
    # control even when no semantic content changed.
    fams = sorted(fams, key=lambda f: (f["root_canonical_form"], f["root_language"]))

    reflex_to_links, reflex_meta, families_without_reflex = _partition_families_by_reflex(fams)

    words: list[dict[str, Any]] = []
    for reflex_id in sorted(
        reflex_to_links,
        key=lambda rid: (reflex_meta[rid]["position"], reflex_meta[rid]["surface_form"]),
    ):
        words.append(_word_for_reflex(reflex_meta[reflex_id], reflex_to_links[reflex_id]))

    for fam in families_without_reflex:
        words.append(_synthesize_word_for_family(fam))

    return words


def _partition_families_by_reflex(
    fams: list[dict[str, Any]],
) -> tuple[
    dict[int, list[tuple[dict[str, Any], list[int]]]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Split families into reflex-linked and reflex-less groups.

    Returns ``(reflex_to_links, reflex_meta, families_without_reflex)``.
    ``reflex_to_links[reflex_id]`` is a list of ``(family, linked_member_ids)``
    tuples so each reflex's word entry only includes language forms from
    the etymons it's actually linked to, not from every family in the
    subject. Per the original meanings.json shape (e.g. "Alder tree"),
    reflexes are language-specific: '-farne' carries celtic_mix only,
    'Alder-' carries old_english only. seed_from_meanings preserves this
    in reflex_etymon, and the export must too.
    """
    reflex_to_links: dict[int, list[tuple[dict[str, Any], list[int]]]] = {}
    reflex_meta: dict[int, dict[str, Any]] = {}
    families_without_reflex: list[dict[str, Any]] = []
    for fam in fams:
        if not fam["reflexes"]:
            families_without_reflex.append(fam)
            continue
        for r in fam["reflexes"]:
            reflex_meta[r["id"]] = r
            reflex_to_links.setdefault(r["id"], []).append((fam, r["linked_member_ids"]))
    return reflex_to_links, reflex_meta, families_without_reflex


@dataclass
class _WordLanguageAccumulators:
    """Per-language accumulators populated during family-walk emission.

    Bundle of the ``forms_by_lang`` array plus nine per-language
    sibling dicts (``variants`` / ``inflections`` / ``citations`` /
    ``attested_years`` / ``english_shaped`` / ``original_script`` /
    ``transliteration`` / ``pronunciation`` / ``stratum``) that
    ``_word_for_reflex`` and ``_synthesize_word_for_family``
    independently maintain in lockstep (same keys, populated by the
    same absorb_* helpers, drained into ``_emit_word_languages``
    together). Holding them in one object keeps the call signature
    down to one positional arg per consumer and makes 'add a new
    per-language sibling field' a one-line edit (D26 pattern) rather
    than a 6-touch-site refactor.

    Originally consolidated 5 separate locals (wyrd-k55) when only
    forms + variants + inflections + citations + attested_years
    existed; later waves grew the sibling set to 9 (wyrd-vsrn Phase
    2c added english_shaped; wyrd-qhs0 Phase 2d added
    original_script / transliteration / pronunciation; wyrd-lr4
    Phase 2 added stratum). The one-line-add-a-field property held
    across each wave.
    """

    forms_by_lang: dict[str, list[str]] = field(default_factory=dict)
    variants: dict[str, dict[str, int]] = field(default_factory=dict)
    inflections: dict[str, dict[str, str]] = field(default_factory=dict)
    citations: dict[str, set[str]] = field(default_factory=dict)
    attested_years: dict[str, dict[str, int]] = field(default_factory=dict)
    # wyrd-vsrn Phase 2c: per-language english_shaped pool, keyed by
    # (lang, canonical_form) → english_shaped. Sparse — only non-Latin-
    # source-lang rows whose wyrd-ha9q derive_english_shaped produced a
    # non-None value land here. Empty dict for Latin-script langs +
    # rows that lacked sufficient transliteration / IPA input.
    english_shaped: dict[str, dict[str, str]] = field(default_factory=dict)
    # wyrd-qhs0 Phase 2d: the other three wyrd-ha9q rendering columns,
    # all per-(lang, canonical_form). Together with english_shaped these
    # are the four renderings the SPA's etymological-provenance panel
    # surfaces (D31 four-rendering rule).
    original_script: dict[str, dict[str, str]] = field(default_factory=dict)
    transliteration: dict[str, dict[str, str]] = field(default_factory=dict)
    # Pronunciation pairs IPA + dialect tag; the bucket value is a dict
    # {"ipa": str, "dialect": str | None}. NULL pronunciation_dialect
    # surfaces as a None value for "dialect".
    pronunciation: dict[str, dict[str, dict[str, str | None]]] = field(default_factory=dict)
    # wyrd-lr4 Phase 2: per-(lang, canonical_form) within-language
    # stratum tag. Sparse — only languages with a Phase 1 classifier
    # populate this (Welsh-family today). Other languages stay empty.
    stratum: dict[str, dict[str, str]] = field(default_factory=dict)
    # wyrd-ecjp.10a: per-(lang, canonical_form) phonological_vector dict
    # (kq7w.1 corpus enrichment populates etymon.phonological_vector,
    # which the bundle deserializes back into the runtime D36.1
    # PhonologicalVector). Sparse — only forms whose etymons have been
    # phon-enriched populate this. Value type is dict[str, float]
    # (already JSON-parsed) so emit-time can drop directly into the
    # bundle without re-serialization.
    phonological_vector: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)


def _morpheme_id_for_family(fam: dict[str, Any]) -> str | None:
    """wyrd-rogd.10: the owning-morpheme id = a CONTENT-derived key
    ``"{root_language}:{root_canonical_form}"``. NOT the autoincrement
    ``root_id`` — that shifts across a rebuild and would break the export's
    byte-identity-across-rebuild reconstructibility guarantee (and the export
    already sorts by this tuple precisely because root_id is unstable). The
    content key is reproduced verbatim by ``rebuild-from-jsonl``.

    Returns ``None`` (rather than a malformed ``"None:..."`` key) when the
    family lacks a usable root identity. Real families from ``_gather_family``
    always carry both fields; this is fail-soft belt-and-suspenders."""
    language = fam.get("root_language")
    canonical = fam.get("root_canonical_form")
    if not language or not canonical:
        return None
    return f"{language}:{canonical}"


def _word_morpheme_id(fams: list[dict[str, Any]]) -> str | None:
    """The morpheme id for a word. A reflex can link to several families; pick
    the primary deterministically by the same ``(root_canonical_form,
    root_language)`` key the export sorts on, so the choice is reproducible.
    Families lacking a usable root identity are skipped (so ``min`` can't raise
    on a missing key); ``None`` when none qualify."""
    candidates = [f for f in fams if f.get("root_canonical_form") and f.get("root_language")]
    if not candidates:
        return None
    chosen = min(candidates, key=lambda f: (f["root_canonical_form"], f["root_language"]))
    return _morpheme_id_for_family(chosen)


def _word_for_reflex(
    meta: dict[str, Any], link_pairs: list[tuple[dict[str, Any], list[int]]]
) -> dict[str, Any]:
    """Assemble one word entry for a reflex linked to one or more families.

    Walks each linked etymon's descendants so the reflex picks up the
    lemma's inflected children (D8) and OCR-cluster losers (D22) —
    not just the seeded etymon itself.
    """
    accs = _WordLanguageAccumulators()
    for fam, linked_ids in link_pairs:
        for member_id in linked_ids:
            for descendant_id in fam["member_descendants"][member_id]:
                lang, form = fam["member_form_by_id"][descendant_id]
                bucket = accs.forms_by_lang.setdefault(lang, [])
                if form not in bucket:
                    bucket.append(form)
                _absorb_member_variants(accs, fam, descendant_id, lang)
                _absorb_member_inflection(accs, fam, descendant_id, lang, form)
                _absorb_member_citations(accs, fam, descendant_id, lang)
                _absorb_member_attested_years(accs, fam, descendant_id, lang, form)
                _absorb_member_english_shaped(accs, fam, descendant_id, lang, form)
                _absorb_member_original_script(accs, fam, descendant_id, lang, form)
                _absorb_member_transliteration(accs, fam, descendant_id, lang, form)
                _absorb_member_pronunciation(accs, fam, descendant_id, lang, form)
                _absorb_member_stratum(accs, fam, descendant_id, lang, form)
                _absorb_member_phonological_vector(accs, fam, descendant_id, lang, form)
    word: dict[str, Any] = {"modern_usage": meta["surface_form"]}
    morpheme_id = _word_morpheme_id([fam for fam, _ in link_pairs])
    if morpheme_id is not None:
        word["morpheme_id"] = morpheme_id
    _emit_word_languages(word, accs)
    _emit_era_reflexes(word, link_pairs)
    return word


def _surface_fold(s: str) -> str:
    """Fold a surface the way the SPA's ``accentFold`` does (defined in
    ``spa-next/src/lib/accents.js``, used by era.js's ``cellForSurface``): strip
    the reconstructed-form ``*`` marker + leading/trailing dashes, drop combining
    diacritics, lowercase. Used to dedupe a self-seeded generated surface against
    an era cell that already echoes it (``-bȳ`` / ``-by`` / ``By`` all fold to
    ``by``; ``*mos`` / ``Mos-`` both fold to ``mos``), so we don't emit two
    fold-equal cells that would light up at once."""
    decomposed = unicodedata.normalize("NFD", (s or "").replace("*", "").strip("-"))
    # wyrd-nndd: drop category-Mn combining marks (matching proportions._grid_match_key
    # and the SPA accentFold's \p{Mn} test) rather than filtering on the canonical
    # combining class (the old `combining(c) != 0`). The two criteria diverge on two
    # rare non-Latin classes: Mn marks with ccc==0 (Indic anusvara/vowel-signs —
    # Mn-fold drops them, ccc-fold kept) and non-Mn marks with ccc>0 (a few Mc
    # viramas/tone marks — Mn-fold keeps them, ccc-fold dropped). NEITHER reaches
    # this fold, which only sees romanized-Latin generated surfaces + era-cell forms
    # where every accent NFD-decomposes to a ccc>0 Mn mark (both criteria agree) —
    # byte-identical over all 26,696 live reflex surfaces; drift gate = 0.
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn").lower()


# Cached: pure function of static era.cells data over the closed, finite set of
# language tags (not an external/unbounded key space), called once per well-formed
# word during the bundle build — caching collapses the repeated family_stage_order
# scan to one per distinct language. Unbounded cache is safe given the bounded domain.
@functools.cache
def _own_family_modern_stage(language: str) -> str | None:
    """The canonical language tag of ``language``'s era family's MODERN (newest)
    stage — e.g. ``old-english`` → ``modern-english``, ``old-french`` →
    ``french``, ``irish`` → ``irish``. ``None`` when the language has no era
    family (proto / untracked classical), which the runtime grid drops anyway."""
    family = language_family(language)
    if family is None:
        return None
    stages = family_stage_order(family)
    return stages[-1] if stages else None


def _seed_generated_surface(
    merged: dict[str, dict[str, dict]], own_lang: str, modern_usage: str
) -> None:
    """wyrd-mook: echo the word's GENERATED SURFACE (``modern_usage``) into its
    family's MODERN stage of ``merged`` (in place), source ``"self"``.

    The canonical self-seed in :func:`_emit_era_reflexes` echoes the etymon's own
    form, but the connective surface the generator actually emits often differs
    from it (``-ning-`` / ``Cras-`` vs canonical ``old-english:-ing`` /
    ``old-english:cærse``); without this the grid shows the morpheme's reflexes
    but nothing folds to the surface the user sees, which the SPA reads as "X not
    in its own reflexes". The modern stage is the right home — ``modern_usage``
    IS the modern surface — and lands the initial highlight in the Modern column
    ("as generated").

    No-ops (early return) when: the surface is empty / dash-only; ANY era cell
    already folds to it (so ``surface == canonical``, the common case, adds no
    redundant cell); or the own-language has no era family (proto / untracked
    classical — gridless anyway). Seeded VERBATIM (dashes intact, like the
    canonical seed)."""
    surface = modern_usage.strip()
    if not surface.strip("-"):
        return
    surface_fold = _surface_fold(surface)
    if any(
        _surface_fold(cell_form) == surface_fold for stage in merged.values() for cell_form in stage
    ):
        return
    modern_lang = _own_family_modern_stage(own_lang)
    if modern_lang is None:
        return
    merged.setdefault(modern_lang, {})[surface] = {"form": surface, "source": "self"}


def _emit_era_reflexes(
    word: dict[str, Any],
    link_pairs: list[tuple[dict[str, Any], list[int]]],
) -> None:
    """wyrd-obpw Phase 3.3 + wyrd-jbcu source-aware schema: stamp the
    family root's era_reflexes onto the word dict. Each linked family
    contributes its root's per-target-language reflex list; multiple
    linked families merge per target language with same-form
    collisions resolved by source quality (higher-quality source wins).

    Bundle field: ``era_reflexes`` is ``{target_language: [{form,
    source, gloss?}, ...]}`` per word (``gloss`` sparse, wyrd-rogd.1).

    Every well-formed word also SELF-SEEDS its own ``morpheme_id`` form in its
    own era column (``source="self"``) so a barren morpheme — no cluster, no
    descent reflexes — still grids (shows at least itself). ``era_reflexes`` is
    therefore absent only when the word has no / malformed ``morpheme_id``, a
    dash-only form, or an own-language with no tracked era family.

    wyrd-mook: it ALSO self-seeds the word's GENERATED SURFACE (``modern_usage``)
    into its family's MODERN stage via :func:`_seed_generated_surface`, so the
    connective surface the user sees (``-ning-`` / ``Cras-``, often ≠ the etymon
    canonical) is echoed somewhere the SPA's surface-fold can match — see that
    helper for the placement + skip rules.
    """
    merged: dict[str, dict[str, dict]] = {}
    for fam, _linked_ids in link_pairs:
        for target_language, entries in fam.get("era_reflexes", {}).items():
            bucket = merged.setdefault(target_language, {})
            for entry in entries:
                form = entry["form"]
                existing = bucket.get(form)
                # Keep the whole entry (form + source + optional gloss);
                # higher-quality source wins on a same-form collision.
                if existing is None or _better_era_reflex_source(
                    entry["source"], existing["source"]
                ):
                    bucket[form] = entry
    # Self-seed the morpheme's OWN form in its OWN era column. A barren morpheme
    # (no cognate cluster, no descent reflexes) would otherwise emit no
    # era_reflexes at all and render a completely empty col-3 grid — showing in
    # zero columns though it plainly exists and was just used to compose a name.
    # The morpheme_id is "<language>:<canonical_form>"; ensure that (language,
    # form) is present, tagged source="self" so a FUTURE coverage pass can
    # exclude it from real-reflex counts (wyrd-32t1) — no consumer distinguishes
    # it yet, so the bundle era-reflex coverage metric now counts self-seeds.
    # NOTE: when own_lang has no era family (proto / untracked classical) the
    # runtime grid drops the stage, so those stay gridless — acceptable, they're
    # outside the British-Isles cultures.
    # Seed the form VERBATIM (dashes intact): real reflexes store affixes with
    # their dashes ('-tun', '-chester'), so a dash-stripped key would NOT collide
    # under setdefault and would add a duplicate own-era cell.
    own_lang, sep, own_form = (word.get("morpheme_id") or "").partition(":")
    own_form = own_form.strip()
    if sep and own_lang and own_form.strip("-"):  # well-formed id, content beyond dashes
        merged.setdefault(own_lang, {}).setdefault(own_form, {"form": own_form, "source": "self"})
        _seed_generated_surface(merged, own_lang, word.get("modern_usage") or "")
    if merged:
        # D55: case is render-only. This is the choke point for the emitted
        # ``era_reflexes`` — it folds BOTH the cross-family merge above AND the
        # self-seeds below (``own_form`` = the etymon canonical_form, and the
        # generated modern_usage), which are seeded VERBATIM and so still carry
        # case (canonical_form is un-folded at rest, wyrd-n2j6). The per-family
        # fold in ``_fetch_family_era_reflexes`` covers ``forms_by_lang``
        # sampling; this covers the bundle's ``word.era_reflexes``.
        word["era_reflexes"] = {
            target_language: _case_fold_reflex_entries([forms[form] for form in sorted(forms)])
            for target_language, forms in sorted(merged.items())
        }


def _synthesize_word_for_family(fam: dict[str, Any]) -> dict[str, Any]:
    """Assemble a synthesized word for a family that has no linked reflex.

    Uses the family's canonical_forms en bloc (no per-reflex narrowing
    applies). The whole family's variants and inflections fold in by
    matching language.
    """
    word: dict[str, Any] = {"modern_usage": _synthesize_modern_usage(fam)}
    morpheme_id = _morpheme_id_for_family(fam)  # wyrd-rogd.10 (None-safe)
    if morpheme_id is not None:
        word["morpheme_id"] = morpheme_id
    accs = _WordLanguageAccumulators(
        forms_by_lang={lang: list(fam["forms_by_lang"][lang]) for lang in fam["forms_by_lang"]},
    )
    for member_id, (member_lang, member_form) in fam["member_form_by_id"].items():
        _absorb_member_variants(accs, fam, member_id, member_lang)
        _absorb_member_inflection(accs, fam, member_id, member_lang, member_form)
        _absorb_member_citations(accs, fam, member_id, member_lang)
        _absorb_member_attested_years(accs, fam, member_id, member_lang, member_form)
        _absorb_member_english_shaped(accs, fam, member_id, member_lang, member_form)
        _absorb_member_original_script(accs, fam, member_id, member_lang, member_form)
        _absorb_member_transliteration(accs, fam, member_id, member_lang, member_form)
        _absorb_member_pronunciation(accs, fam, member_id, member_lang, member_form)
        _absorb_member_stratum(accs, fam, member_id, member_lang, member_form)
        _absorb_member_phonological_vector(accs, fam, member_id, member_lang, member_form)
    _emit_word_languages(word, accs)
    # Synthesized word case: link_pairs structure isn't used here, so
    # build a single-element link_pairs from the family directly.
    _emit_era_reflexes(word, [(fam, list(fam["member_form_by_id"].keys()))])
    return word


def _absorb_member_variants(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
) -> None:
    """Aggregate D18 spelling variants for one (member, language) into the
    per-language pool, summing weights on collision. Caller guarantees
    `lang` matches the member's language so callers don't accidentally
    cross-pollinate across languages."""
    for variant_form, weight in fam.get("member_variants", {}).get(member_id, []):
        lang_variants = accs.variants.setdefault(lang, {})
        lang_variants[variant_form] = lang_variants.get(variant_form, 0) + weight


def _absorb_member_inflection(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """Record a member's D8 inflection label (if any) keyed by its surface
    form. Lemmas have inflection=None and are skipped — only inflected
    children carry a grammatical-case label worth surfacing."""
    inflection = fam.get("member_inflection_by_id", {}).get(member_id)
    if inflection:
        accs.inflections.setdefault(lang, {})[form] = inflection


def _absorb_member_attested_years(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """Record a member's earliest attested year (D5-1, wyrd-bag) keyed
    by its surface form. Members with no attested year are skipped —
    the runtime generator interprets the absence as 'no era constraint
    applies' (treat the form as always-includable under any --era)."""
    year = fam.get("member_attested_years", {}).get(member_id)
    if year is not None:
        accs.attested_years.setdefault(lang, {})[form] = year


def _absorb_member_english_shaped(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """wyrd-vsrn Phase 2c: record a member's english_shaped rendering
    keyed by its canonical_form. Skipped when the column is NULL or
    empty: the runtime treats the absence as 'use canonical_form for
    display' for that member. NULL is the documented case (Latin-script
    source langs OR rows that lacked transliteration / IPA inputs);
    empty-string would be an unexpected DB shape but the predicate
    covers both for safety."""
    shaped = fam.get("member_english_shaped_by_id", {}).get(member_id)
    if shaped:
        accs.english_shaped.setdefault(lang, {})[form] = shaped


def _absorb_member_original_script(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """wyrd-qhs0 Phase 2d: record a member's vocalized native-script
    form (Hebrew niqqud, Arabic harakat, Egyptian hieroglyphic markup)
    keyed by canonical_form. NULL is the common case for Latin-script
    rows; skip without absorbing."""
    original = fam.get("member_original_script_by_id", {}).get(member_id)
    if original:
        accs.original_script.setdefault(lang, {})[form] = original


def _absorb_member_transliteration(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """wyrd-qhs0 Phase 2d: record a member's academic Latin-script
    transliteration (with diacritics — ʿifrīt, rakṣasa, kɛ́lɛḇ) keyed
    by canonical_form."""
    translit = fam.get("member_transliteration_by_id", {}).get(member_id)
    if translit:
        accs.transliteration.setdefault(lang, {})[form] = translit


def _absorb_member_pronunciation(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """wyrd-qhs0 Phase 2d: record a member's IPA + dialect pair keyed
    by canonical_form. The pair updates atomically (matches the upsert
    semantics in lexicon.py.upsert_etymon's CASE expression on dialect
    — wyrd-ha9q Phase 2a fix for IPA/dialect decoupling)."""
    pron = fam.get("member_pronunciation_by_id", {}).get(member_id)
    if pron:
        ipa, dialect = pron
        accs.pronunciation.setdefault(lang, {})[form] = {"ipa": ipa, "dialect": dialect}


def _absorb_member_stratum(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """wyrd-lr4 Phase 2: record a member's within-language stratum tag
    keyed by canonical_form. Skipped when stratum is NULL (the common
    case for languages without a Phase 1 classifier — only Welsh-family
    etymons are populated today). The runtime treats absence as 'no
    stratum filter applies' for the consumer wired up in Phase 3."""
    stratum = fam.get("member_stratum_by_id", {}).get(member_id)
    if stratum:
        accs.stratum.setdefault(lang, {})[form] = stratum


def _absorb_member_phonological_vector(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
    form: str,
) -> None:
    """wyrd-ecjp.10a: record a member's phonological_vector keyed by
    (lang, canonical_form). The DB blob is a JSON string emitted by
    ``vector_to_json``; we parse it here so emit-time can drop the
    dict directly into the bundle without re-serialization (the
    bundle JSON serializer handles nested dicts).

    Skipped when phonological_vector is NULL (kq7w.1 enrichment
    hasn't reached this row) or when JSON parse fails (malformed
    blob from an interrupted enrichment run). Per-row exception
    isolation matches the rest of the absorb-helper pattern: one
    bad row shouldn't abort a whole-bundle export. The runtime's
    loader is also defensive (`_parse_phon_vector` returns None on
    parse failure), so a malformed blob that slips through still
    degrades gracefully to "no vector for this form."
    """
    blob = fam.get("member_phonological_vector_by_id", {}).get(member_id)
    if not blob:
        return
    try:
        vector = json.loads(blob)
    except (ValueError, TypeError):
        return
    if isinstance(vector, dict):
        accs.phonological_vector.setdefault(lang, {})[form] = vector


def _absorb_member_citations(
    accs: _WordLanguageAccumulators,
    fam: dict[str, Any],
    member_id: int,
    lang: str,
) -> None:
    """Aggregate scholarly source_ids for one member into the per-language
    citation set (wyrd-9kh.1). Set semantics dedupe across the descendant
    walk; emit-time sorts deterministically. Caller guarantees `lang`
    matches the member's language."""
    citations = fam.get("member_citations", {}).get(member_id, [])
    if citations:
        accs.citations.setdefault(lang, set()).update(citations)


def _emit_word_languages(word: dict[str, Any], accs: _WordLanguageAccumulators) -> None:
    """Stamp per-language form arrays + the ten sibling metadata
    families onto the word dict: ``<lang>_variants``,
    ``<lang>_inflections``, ``<lang>_citations``,
    ``<lang>_attested_years``, ``<lang>_english_shaped``,
    ``<lang>_original_script``, ``<lang>_transliteration``,
    ``<lang>_pronunciation`` (wyrd-qhs0 Phase 2d),
    ``<lang>_stratum`` (wyrd-lr4 Phase 2), and
    ``<lang>_phonological_vector`` (wyrd-ecjp.10a). Per D26, the
    metadata fields are sibling keys so legacy loaders that ignore
    unknown fields keep working.

    Multiple lexicon codes can route to the SAME bundle bucket via
    `_LANG_CODE_TO_JSON_FIELD` — e.g. welsh + old-welsh + middle-welsh
    all land in `celtic_mix`, and wyrd-vsrn's wave-2 stack collapses
    he + hbo + sem-pro + sem-wes-pro + afa-pro into `hebrew`. We MUST
    union (not overwrite) per bundle bucket: if the inner loop rewrote
    `word[json_field]` on each iteration, only the last-sorted lexicon
    code's forms would survive.

    Aggregation is done per bucket via `_BucketAccumulator` so all
    ten sibling families plus the forms array union under the same
    bucket key in one pass; the final emit walks buckets in stable
    order so output is deterministic regardless of source-lang sort
    order.
    """
    _emit_buckets_to_word(word, _aggregate_language_buckets(accs))


# The eight sibling families merged source-lang → bucket by a plain set/dict
# ``.update()`` (forms / variants / attested_years need bespoke merge logic and
# stay explicit in _accumulate_lang_into_bucket).
_BUCKET_UNION_FAMILIES = (
    "inflections",
    "citations",
    "english_shaped",
    "original_script",
    "transliteration",
    "pronunciation",
    "stratum",
    "phonological_vector",
)

# (_BucketAccumulator attr == json-key suffix, emit fn), in the stable output
# order the word dict's keys must follow.
_EMIT_FAMILIES: tuple[tuple[str, Any], ...] = (
    ("variants", _emit_variant_list),
    ("inflections", _emit_inflection_list),
    ("citations", sorted),
    ("attested_years", _emit_attested_years_list),
    ("english_shaped", _emit_english_shaped_list),
    ("original_script", _emit_original_script_list),
    ("transliteration", _emit_transliteration_list),
    ("pronunciation", _emit_pronunciation_list),
    ("stratum", _emit_stratum_list),
    ("phonological_vector", _emit_phonological_vector_list),
)


def _accumulate_lang_into_bucket(
    bucket: _BucketAccumulator, accs: _WordLanguageAccumulators, lang: str
) -> None:
    """Union one source language's forms + sibling metadata into its bundle
    bucket. Forms dedup (preserving first-seen order); variants sum their
    weights; attested_years keep the earliest year on collision (the
    rest-of-pipeline ascending-year convention); the remaining eight families
    are plain set/dict ``.update()``s driven off ``_BUCKET_UNION_FAMILIES``."""
    for form in accs.forms_by_lang[lang]:
        if form not in bucket.forms_set:
            bucket.forms.append(form)
            bucket.forms_set.add(form)
    for form, weight in accs.variants.get(lang, {}).items():
        bucket.variants[form] = bucket.variants.get(form, 0) + weight
    for form, year in accs.attested_years.get(lang, {}).items():
        existing = bucket.attested_years.get(form)
        if existing is None or year < existing:
            bucket.attested_years[form] = year
    for family in _BUCKET_UNION_FAMILIES:
        src = getattr(accs, family).get(lang)
        if src:
            getattr(bucket, family).update(src)


def _aggregate_language_buckets(
    accs: _WordLanguageAccumulators,
) -> dict[str, _BucketAccumulator]:
    """Route each source language's accumulators into its bundle bucket,
    unioning the multiple lexicon codes that map to the same bucket (e.g.
    welsh + old-welsh → ``celtic_mix``). Walks langs in sorted order so the
    union is deterministic regardless of source-lang discovery order."""
    buckets: dict[str, _BucketAccumulator] = {}
    for lang in sorted(accs.forms_by_lang):
        json_field = _LANG_CODE_TO_JSON_FIELD.get(lang)
        if not json_field:
            continue
        bucket = buckets.setdefault(json_field, _BucketAccumulator())
        _accumulate_lang_into_bucket(bucket, accs, lang)
    return buckets


def _emit_buckets_to_word(word: dict[str, Any], buckets: dict[str, _BucketAccumulator]) -> None:
    """Write each bucket's forms + every non-empty sibling family onto the word
    dict under ``<json_field>`` / ``<json_field>_<family>`` keys, in stable
    bucket then ``_EMIT_FAMILIES`` order. ``_fill_surface_pronunciation`` runs
    first (it only fills gaps in ``bucket.pronunciation``) so a surface form
    lacking etymon IPA still contributes before the pronunciation family emits."""
    for json_field in sorted(buckets):
        bucket = buckets[json_field]
        word[json_field] = bucket.forms
        _fill_surface_pronunciation(bucket, json_field)
        for family, emit_fn in _EMIT_FAMILIES:
            value = getattr(bucket, family)
            if value:
                word[f"{json_field}_{family}"] = emit_fn(value)
