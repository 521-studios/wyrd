"""Output formatters for the bundle build (leaf module — no internal deps).

Per-language buckets + serializer for the meanings.json output. The
``_LANG_CODE_TO_JSON_FIELD`` rollup map collapses every Celtic-family
language tag (welsh / old-welsh / middle-welsh / irish / old-irish /
middle-irish / scottish-gaelic / breton / proto-celtic / …) into a
single ``celtic_mix`` bundle bucket; wyrd-vsrn wave-2 stacks (he +
hbo + sem-pro → ``hebrew``; sa + iir-pro + inc-pro → ``sanskrit``)
collapse similarly. The fixed nine-bucket structure on the bundle
side stays runtime-stable; the lexicon keeps finer-grained languages
on the etymon row for future per-language queries.

The ``_BucketAccumulator`` collects per-bucket evidence as a member
is absorbed; the ``_emit_*_list`` formatters then turn the
accumulated dicts into the bundle's stable list-of-dicts shape.

``_orphan_reflex_subjects`` produces a synthesized subject per
``reflex`` row that has no ``reflex_etymon`` link — the rando-port
legacy entries where the original ``word`` carried only a
``modern_usage`` string and no per-language form list. The runtime
still needs the usage registered, so the bundle includes them as
standalone single-word subjects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wyrd.generators.kenning.lexicon.db import LexiconDB

_LANG_CODE_TO_JSON_FIELD = {
    "old-english": "old_english",
    "old-norse": "old_scandinavian",
    "norman-french": "old_french",
    "celtic": "celtic_mix",
    "latin": "latin",
    "germanic": "germanic",
    "greek": "greek",
    "modern-english": "modern_english",
    "biblical": "biblical",
    # wyrd-4hx7: Wiktionary-canonical language codes (per
    # wiktextract_ingester._canonical_language) routed into the same
    # 9 bundle fields the historical scholarly corpus used. The
    # bundle field structure stays at 9 buckets; the lexicon keeps
    # finer-grained languages on the etymon row for future per-language
    # queries. Adding new bundle field names later (e.g. "welsh",
    # "irish") would require runtime + culture-mapping work; routing
    # into existing buckets is the no-runtime-change path.
    "welsh": "celtic_mix",
    "old-welsh": "celtic_mix",
    "middle-welsh": "celtic_mix",
    "irish": "celtic_mix",
    "old-irish": "celtic_mix",
    "middle-irish": "celtic_mix",
    "scottish-gaelic": "celtic_mix",
    "manx": "celtic_mix",
    "cornish": "celtic_mix",
    "breton": "celtic_mix",
    "old-breton": "celtic_mix",
    "middle-breton": "celtic_mix",
    "proto-celtic": "celtic_mix",
    "proto-brythonic": "celtic_mix",
    "old-french": "old_french",
    "anglo-norman": "old_french",
    "middle-french": "old_french",
    "middle-english": "modern_english",
    "scots": "modern_english",
    "icelandic": "old_scandinavian",
    "faroese": "old_scandinavian",
    "old-high-german": "germanic",
    "middle-high-german": "germanic",
    "gothic": "germanic",
    "old-saxon": "germanic",
    "old-dutch": "germanic",
    "old-frisian": "germanic",
    "proto-germanic": "germanic",
    "proto-west-germanic": "germanic",
    "ancient-greek": "greek",
    "proto-greek": "greek",
    # wyrd-vsrn Phase 2c: wave-2 non-Latin source-language buckets.
    # Each canonical wave-2 language gets its own bundle field; the
    # precursor / postcursor stack codes (per extractors.fantasy's
    # _PRECURSOR_POSTCURSOR_STACK) all funnel into the canonical
    # bucket for their family — same pattern as celtic_mix bundles
    # welsh / old-welsh / middle-welsh / etc.
    "he": "hebrew",
    "hbo": "hebrew",
    "ar": "arabic",
    "fa": "persian",
    "peo": "persian",
    "fa-cls": "persian",
    "xpr": "persian",
    "pal": "persian",
    "ira-pro": "persian",
    "sa": "sanskrit",
    "iir-pro": "sanskrit",
    "inc-pro": "sanskrit",
    "pra": "sanskrit",
    "pi": "sanskrit",
    "akk": "akkadian",
    "sux": "akkadian",  # Sumerian — Mesopotamian substrate, group with akkadian
    "egy": "egyptian",
    "cop": "egyptian",  # Coptic — late-stage descendant of Ancient Egyptian
    "arc": "aramaic",
    "syc": "aramaic",  # Classical Syriac — late descendant of Aramaic
    "sem-pro": "hebrew",  # Proto-Semitic — group with Hebrew (closest canonical)
    "sem-wes-pro": "hebrew",
    "afa-pro": "hebrew",  # Proto-Afroasiatic — same bucket
    "axm": "armenian",  # Old / Classical Armenian — own bucket
}


@dataclass
class _BucketAccumulator:
    """Per-bundle-bucket aggregation state used by _emit_word_languages
    when multiple lexicon codes (e.g. welsh + old-welsh; he + hbo +
    sem-pro) collapse into one bundle field. Fields mirror the
    per-language pools on `_WordLanguageAccumulators` but keyed by the
    BUNDLE bucket so cross-language values union cleanly."""

    forms: list[str] = field(default_factory=list)
    forms_set: set[str] = field(default_factory=set)
    variants: dict[str, int] = field(default_factory=dict)
    inflections: dict[str, str] = field(default_factory=dict)
    citations: set[str] = field(default_factory=set)
    attested_years: dict[str, int] = field(default_factory=dict)
    english_shaped: dict[str, str] = field(default_factory=dict)
    # wyrd-qhs0 Phase 2d: the other three wyrd-ha9q rendering columns.
    original_script: dict[str, str] = field(default_factory=dict)
    transliteration: dict[str, str] = field(default_factory=dict)
    pronunciation: dict[str, dict[str, str | None]] = field(default_factory=dict)
    # wyrd-lr4 Phase 2: per-canonical_form within-language stratum tag.
    stratum: dict[str, str] = field(default_factory=dict)
    # wyrd-ecjp.10a: per-canonical_form phonological_vector dict
    # (kq7w.1 corpus enrichment). Value type is dict[str, float] —
    # already JSON-parsed at absorb time so the bundle emit drops it
    # in without re-serialization. Sparse: only forms reached by the
    # phon-vector enrichment pass populate this.
    phonological_vector: dict[str, dict[str, float]] = field(default_factory=dict)


def _emit_original_script_list(scripts: dict[str, str]) -> list[dict[str, str]]:
    """wyrd-qhs0 Phase 2d: serialize {canonical_form: original_script}
    into the meanings.json original_script entry shape. Each dict has
    ``"form"`` (the canonical_form lookup key) and ``"original_script"``
    (the vocalized native-script form from wiktextract head_templates'
    `wv` / `head` arg). Sorted by form for deterministic output."""
    return [
        {"form": form, "original_script": original_form}
        for form, original_form in sorted(scripts.items())
    ]


def _emit_transliteration_list(translits: dict[str, str]) -> list[dict[str, str]]:
    """wyrd-qhs0 Phase 2d: serialize {canonical_form: transliteration}
    into the meanings.json transliteration entry shape. Each dict has
    ``"form"`` (canonical_form) and ``"transliteration"`` (academic
    Latin-script form with diacritics, from wiktextract head_templates
    `tr`). Sorted by form for determinism."""
    return [
        {"form": form, "transliteration": translit_form}
        for form, translit_form in sorted(translits.items())
    ]


def _emit_pronunciation_list(
    pronunciations: dict[str, dict[str, str | None]],
) -> list[dict[str, Any]]:
    """wyrd-qhs0 Phase 2d: serialize {canonical_form: {ipa, dialect}}
    into the meanings.json pronunciation entry shape. Each dict has
    ``"form"`` (canonical_form), ``"ipa"`` (the IPA string), and
    ``"dialect"`` (the first tag on the chosen sound entry, or None
    when the IPA was untagged-canonical). Sorted by form."""
    return [
        {"form": form, "ipa": data["ipa"], "dialect": data.get("dialect")}
        for form, data in sorted(pronunciations.items())
    ]


def _emit_phonological_vector_list(
    vectors: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """wyrd-ecjp.10a: serialize {canonical_form: phon_vector_dict} into
    the meanings.json <lang>_phonological_vector entry shape. Each
    entry is {"form": canonical_form, "phonological_vector": {<dim>:
    <float>, ...}}. Sorted by form for deterministic output.

    The inner phon vector dict goes through unchanged — it's already
    in the runtime D36.1 shape (cluster_density, final_fortition,
    etc.) since absorb-time parsed the etymon column's JSON blob.
    The runtime's _parse_phon_vector accepts this shape directly.
    """
    return [{"form": form, "phonological_vector": vectors[form]} for form in sorted(vectors)]


def _emit_stratum_list(strata: dict[str, str]) -> list[dict[str, str]]:
    """wyrd-lr4 Phase 2: serialize {canonical_form: stratum} into the
    meanings.json stratum entry shape. Each dict has ``"form"`` (the
    canonical_form lookup key, same as the language form array) and
    ``"stratum"`` (the language-specific register tag — see the Phase
    1 classifier for the current vocabulary; Welsh ships with five:
    latin-loan / english-loan / brittonic-substrate / medieval-welsh /
    native-welsh, with French / OE / ON adding their own as Phase 4
    lands). Sorted by form for determinism."""
    return [{"form": form, "stratum": stratum_tag} for form, stratum_tag in sorted(strata.items())]


def _emit_english_shaped_list(shaped: dict[str, str]) -> list[dict[str, str]]:
    """wyrd-vsrn Phase 2c: serialize {canonical_form: english_shaped} into
    the meanings.json english_shaped entry shape. Sorted by form so the
    output is deterministic regardless of aggregation order.

    Each output dict has two keys:
        ``"form"``           — the canonical_form string (the same key the
                                language form array uses, so the runtime
                                can match a chosen form to its shaping).
        ``"english_shaped"`` — the Latin-script rendering produced by
                                wyrd-ha9q's ``derive_english_shaped`` and
                                stored on ``etymon.english_shaped``.
    """
    return [
        {"form": form, "english_shaped": shaped_form}
        for form, shaped_form in sorted(shaped.items())
    ]


def _emit_attested_years_list(years: dict[str, int]) -> list[dict[str, Any]]:
    """Serialize ``{form: year}`` into the meanings.json attested-year
    entry shape. Sorted by ``(year, form)`` so output is deterministic
    and the chronologically-earliest entries lead — useful when the
    runtime --era filter walks the list looking for the first match."""
    return [
        {"form": form, "year": year}
        for form, year in sorted(years.items(), key=lambda kv: (kv[1], kv[0]))
    ]


def _emit_variant_list(variants: dict[str, int]) -> list[dict[str, Any]]:
    """Serialize {form: weight} into the meanings.json variant entry shape.
    Output is sorted by descending weight (most-attested first), with the
    form string as a stable secondary key — matters because two variants
    can share a weight after aggregation across the family."""
    return [
        {"form": form, "weight": weight}
        for form, weight in sorted(variants.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _emit_inflection_list(inflections: dict[str, str]) -> list[dict[str, Any]]:
    """Serialize {form: inflection_label} into the meanings.json inflection
    entry shape. Sorted by form so the output is deterministic regardless of
    aggregation order."""
    return [{"form": form, "inflection": label} for form, label in sorted(inflections.items())]


def _synthesize_modern_usage(family: dict[str, Any]) -> str:
    """Derive a modern_usage for a family that has no linked reflex.

    Uses ``position_pref`` to choose pre/post/inner dash markers; defaults
    to no-dash (the runtime treats undecorated usage as a post-suffix
    via Meaning._set_location).
    """
    form = family["root_canonical_form"]
    position = family.get("position_pref")
    if position == "pre":
        return f"{form}-"
    if position == "inner":
        return f"-{form}-"
    if position == "post":
        return f"-{form}"
    return form


def _orphan_reflex_subjects(db: LexiconDB) -> list[dict[str, Any]]:
    """Emit one subject per reflex that has no linked etymon.

    These come from the rando-port seed where a `word` entry had only
    `modern_usage` (e.g., 'Adam-' as a saint's name) or had a language slot
    with an empty form list (e.g., {'celtic_mix': [], 'modern_usage': 'Bre-'}).
    The runtime needs them in `meaning_db` because per-culture proportions
    JSONs reference them by usage. Emit each as its own subject with empty
    glosses/tags so the runtime can register the usage; promotion to a richer
    subject can land later via the manual seed-source-aware revisit (D7).
    """
    rows = db.conn.execute(
        """
        SELECT r.surface_form
        FROM reflex r
        WHERE NOT EXISTS (
            SELECT 1 FROM reflex_etymon re WHERE re.reflex_id = r.id
        )
        ORDER BY r.position, r.surface_form
        """
    ).fetchall()
    return [
        {
            "meaning": [],
            "modifier_tags": [],
            "modifier_type": None,
            "words": [{"modern_usage": row["surface_form"]}],
        }
        for row in rows
    ]
