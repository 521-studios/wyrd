"""Kenning — town-name generator. Compounds Old English / Norse / Celtic morphemes."""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from importlib import resources
from typing import Any

from wyrd.generators.kenning.era.cells import (
    canonical_language_for_cell,
    era_cell_for_input,
    era_cells_for_family,
    era_year_range,
    family_stage_order,
    resolve_era_input,
)

# Back-compat re-exports for the wyrd-ru5d extractors/ subpackage. Older call
# sites (especially in cli/lexicon/) import these modules as if they were
# top-level kenning submodules — ``from wyrd.generators.kenning import
# gemini_extractor``. After the move, the canonical home is
# ``wyrd.generators.kenning.extractors.<short_name>`` (gemini, anthropic, llm,
# fantasy, pfsrd2_monsters, toponym_mentions). Each old top-level alias is
# preserved here so existing call sites keep working unchanged. New code should
# import from ``wyrd.generators.kenning.extractors`` directly.
from wyrd.generators.kenning.extractors import anthropic as anthropic_extractor  # noqa: F401
from wyrd.generators.kenning.extractors import fantasy as fantasy_pipeline  # noqa: F401
from wyrd.generators.kenning.extractors import gemini as gemini_extractor  # noqa: F401
from wyrd.generators.kenning.extractors import llm as llm_extractor  # noqa: F401
from wyrd.generators.kenning.extractors import (
    pfsrd2_monsters as pfsrd2_monster_extractor,  # noqa: F401
)
from wyrd.generators.kenning.extractors import (
    toponym_mentions as toponym_mention_extractor,  # noqa: F401
)
from wyrd.generators.kenning.lexicon.strata import (  # noqa: F401  (STRATA constants are back-compat re-exports for external callers)
    ALL_STRATA,
    BRITTONIC_STRATA,
    CELTIC_STRATA,
    FRENCH_STRATA,
    GOIDELIC_STRATA,
    OLD_ENGLISH_STRATA,
    OLD_NORSE_STRATA,
    WELSH_STRATA,
    valid_strata_for_culture,
)
from wyrd.generators.kenning.runtime.connective import is_connective
from wyrd.generators.kenning.runtime.decomposition import (
    _decomposition_payload,
    _signature_for_payload,
)
from wyrd.generators.kenning.runtime.meaning import (
    Meaning,
    load_canonical_decompositions,
    load_fantasy_morphemes,
    load_joiners,
    load_meanings,
)
from wyrd.generators.kenning.runtime.proportions import (
    _clear_surface_index_cache,
    load_proportions,
)
from wyrd.generators.kenning.runtime.word import Word
from wyrd.registry import GenerationResult, register

_LEGEND = [
    {"code": "EN", "name": "Old English"},
    {"code": "SC", "name": "Old Scandinavian"},
    {"code": "FR", "name": "Old French"},
    {"code": "CL", "name": "Celtic"},
    {"code": "LA", "name": "Latin"},
    {"code": "GE", "name": "Germanic"},
    {"code": "GR", "name": "Greek"},
]

# Hard ceiling on decomposition results. A short word can produce many
# overlapping morpheme matches, and the SPA renders one item per result; cap
# so a pathological input doesn't dump a hundred near-duplicates.
_MAX_DECOMPOSITIONS = 25

_ROOT_CODES = [
    ("old_english", "EN"),
    ("old_scandinavian", "SC"),
    ("old_french", "FR"),
    ("celtic_mix", "CL"),
    ("latin", "LA"),
    ("germanic", "GE"),
    ("greek", "GR"),
]

# wyrd-hitl round 3 (Gemini MED): module-level stratum priority map
# for _rank_siblings. Hoisted out of _rank_siblings (where it was
# being reconstructed per call) so it's built once at module load.
# Smaller index = more canonical (OE first); used as -rank in the
# sort key so OE sorts FIRST.
_STRATUM_PRIORITY = {src: i for i, (src, _) in enumerate(_ROOT_CODES)}

CULTURES = ["english", "scottish", "welsh", "irish", "breton"]

# D5-2 / wyrd-lyp: which era family the bare-label form of ``--era`` resolves
# against per culture. A request like ``--era me`` against an English culture
# means Middle English; the same request against an Irish culture would
# (correctly) fail since 'me' is an English-family label. Callers can always
# disambiguate with the ``family/label`` form (e.g. ``english/me``).
#
# scottish → english: the Scottish proportions bundle today is
# overwhelmingly Old/Middle English-derived (1752 OE-tagged words vs ~0
# Goidelic-tagged) — Scottish town names mix English and Gaelic morphemes,
# but the empirical inventory skews English. The English era cells line
# up with the bulk of the morphemes; users who want goidelic ranges can
# use the ``goidelic/<label>`` form. Revisit once mining surfaces a
# meaningful Scottish-Gaelic morpheme bucket.
#
# This map is distinct from era.LANGUAGE_TO_FAMILY: that one keys per
# etymon-language code (e.g. 'old-english' → 'english'); this one keys
# per culture-bundle name (e.g. 'english' culture → 'english' family).
# Same destination, different source axes.
_CULTURE_TO_ERA_FAMILY: dict[str, str] = {
    "english": "english",
    "scottish": "english",
    "welsh": "brythonic",
    "irish": "goidelic",
    "breton": "brythonic",
}


def _required_name(params: dict[str, Any]) -> str:
    """The required free-text ``name`` param, normalized to a stripped string.

    A non-string ``name`` (number, list, object) is treated as absent so a
    malformed request raises ``ValueError`` ("name is required") — which the API
    dispatcher maps to a 400 bad_params — rather than ``AttributeError`` from
    calling ``.strip()`` on a non-string (which escapes uncaught as a 500). The
    old ``(params.get("name") or "").strip()`` crashed on a non-string. Mirrors
    ``app._stripped_str``; shared by the name-taking generators (explain, render,
    creature, rewind)."""
    raw = params.get("name")
    text = raw.strip() if isinstance(raw, str) else ""
    if not text:
        raise ValueError("name is required")
    return text


def _era_options_by_culture() -> dict[str, list[str]]:
    """Per-culture list of era STAGE labels for the SPA's dependent select.

    wyrd-rogd.2: the dropdown offers the COMPRESSED per-family stage set
    (``family_stage_order``: English = old-english / middle-english /
    modern-english), the same axis the col-3 grid uses — NOT the raw cells
    (oe-early / oe-late / …). The oe-early-vs-oe-late inventory distinction
    stays in the backend year ranges but isn't surfaced here; a picked stage
    resolves to the UNION year range of its cells (``resolve_era_input`` /
    ``stage_year_range``). Empty string is prepended as the 'no era filter'
    option. Re-evaluated at schema-render time, so era.cells edits take effect
    after a manifest refresh. CLI/API still accept raw cells + bare years.
    """
    return {
        culture: ["", *family_stage_order(family)]
        for culture, family in _CULTURE_TO_ERA_FAMILY.items()
    }


def _stratum_options_by_culture() -> dict[str, list[str]]:
    """Per-culture list of valid stratum tags for the SPA's dependent
    select (wyrd-j3gy). Mirrors ``_era_options_by_culture``: empty
    string prepended as the 'no stratum filter' option, the rest
    sourced from ``valid_strata_for_culture``.

    Cultures with no per-culture restriction (irish / breton today)
    surface as just ``[""]`` — the SPA dropdown shows only the
    'no filter' option, which is honest: there's no classified
    stratum data for those cultures yet, so any --stratum value
    would be a no-op.

    Re-evaluated at schema-render time, so additions to the
    per-culture map propagate to the SPA on the next manifest
    refresh."""
    return {culture: ["", *sorted(valid_strata_for_culture(culture))] for culture in CULTURES}


# wyrd-yan: 'fiction' marks etymons whose etymology is constructed (post-hoc
# applied to bestiary / NPC / homebrew content) rather than drawn from the
# scholarly historical record. Realistic-mode generation excludes these by
# default; the GM opts in via `include_fiction=True` (CLI: --include-fiction).
# The bundle today carries no fiction-tagged morphemes — the gate is in
# place for upcoming wyrd-0ab / wyrd-kjc constructed-etymology pipelines.
_FICTION_TAG = "fiction"

# Tags that are filtering primitives, not meaningful selections to expose
# in the SPA tag dropdown. 'fiction' is a metadata marker, opted into via
# `include_fiction` rather than picked from the dropdown. 'manorial' and
# 'norman' are wyrd-8s71 classification tags on the synthesized
# manorial-affix Meanings — picking them in the tag-filter dropdown
# would restrict generation to manorial families, which is not the
# user-facing 'tag a name with this theme' contract. The user-facing
# manorial knob is the dedicated ``manorial_affix`` probability field.
_INTERNAL_TAGS = {
    "male name",
    "female name",
    "saint",
    "manorial",
    "norman",
    "personal-name",
    _FICTION_TAG,
}

# wyrd-kq7w.3: the legacy MOODS dict (formerly registers/moods.py) was
# replaced by the catalog at wyrd/generators/kenning/data/register_effects.yaml.
# Mood specs resolve through ``parse_mood_spec`` in ``registers.effects``;
# the catalog is the single source of truth for mood-name resolution +
# graduation, consumed directly by the vector adapter.


def _data_path(filename: str):
    return resources.files("wyrd.generators.kenning.data").joinpath(filename)


@lru_cache(maxsize=1)
def _load_norman_manorial_families() -> tuple[str, ...]:
    """Curated tuple of Anglo-Norman manorial-family surnames used as
    optional post-name affixes (wyrd-obu).

    The corpus encodes the post-Conquest political-history layering
    English place-naming captures densely: an English/ON base name
    plus the Norman family that held the manor (Stoke Mandeville,
    Ashby de la Zouch, Stanton Lacy). The list is hand-curated from
    Domesday-and-after subsidy rolls, focusing on surnames that
    actually attach to English toponyms in the historical record.
    """
    with _data_path("norman_manorial_families.json").open() as f:
        return tuple(json.load(f))


@lru_cache(maxsize=1)
def _load_saint_personal_names() -> tuple[dict[str, str], ...]:
    """Curated tuple of saint / personal-name entries used to flesh out
    the empty-gloss morphemes the L3 mining pipeline leaves bare
    (wyrd-z5s8).

    Each entry has ``name`` (canonical form), ``language_field``
    (bundle slot — old_english for OE saints / Anglo-Saxon names,
    old_french for Norman-introduced saints + later French
    dedications), and ``gloss`` (a one-line provenance string the
    explainer surfaces in place of an empty meaning array).

    Personal names show up in two ways in English toponymy: as saint
    dedications (St Botolph's → Boston; St Giles → Gileston) and as
    Anglo-Saxon personal names embedded via the ``-ing`` patronymic
    (Beorma's → Birmingham). The current list focuses on saints, where
    the modern surface remains transparent (Giles, Botolph, Edmund).
    Anglo-Saxon personal names embedded in -ingaham toponyms are a
    future enrichment.
    """
    with _data_path("saint_personal_names.json").open(encoding="utf-8") as f:
        return tuple(json.load(f))


def _saint_personal_name_subjects() -> list[dict[str, Any]]:
    """Synthesize meaning-db subjects for the curated saint /
    personal-name list (wyrd-z5s8). Mirrors ``_norman_manorial_subjects``:
    each entry becomes a Meaning the matcher can resolve when the
    explainer is fed a toponym carrying the saint's name. Pre-fix the
    morpheme either landed in the bundle with empty gloss or didn't
    surface at all, so the explainer fell through to whatever
    shorter / lower-precision match the trie found ('Gileston' →
    'Gil + e + ston' picking Welsh ``gil`` 'splendid' over the actual
    Saint Giles).

    The synthesized subject carries ``personal-name``, ``saint``, and
    ``religious`` tags so the explainer / SPA can surface the
    provenance, plus ``modifier_type='Religious'`` to fit the existing
    taxonomy (matches the ``church`` / ``abbey`` / ``cross`` /
    ``minster`` cohort).
    """
    saints = _load_saint_personal_names()
    subjects: list[dict[str, Any]] = []
    for entry in saints:
        token = entry["name"]
        # wyrd-z5s8: emit BOTH pre-position (``Giles-``) and post-
        # position (``Giles``) variants so saints match at both ends
        # of toponym compounds. Pre handles the clean compound-prefix
        # case (Gileston = Giles + ton, no intervening genitive 's');
        # post handles the dedication-suffix case (Bury St Edmunds,
        # St Giles). Genitive-'s compounds like Petersfield decompose
        # as Peter- + 's' + -field — the 's' is unaccounted today and
        # tracked under wyrd-frpd (joiner mining). Inner (``-Giles-``)
        # is omitted: saint names embedded INSIDE a word are
        # vanishingly rare in English toponymy.
        for surface_usage in (f"{token}-", token):
            subjects.append(
                {
                    "meaning": [entry["gloss"]],
                    "modifier_tags": ["personal-name", "saint", "religious"],
                    "modifier_type": "Religious",
                    "words": [
                        {
                            "modern_usage": surface_usage,
                            entry["language_field"]: [token.lower()],
                        }
                    ],
                }
            )
    return subjects


@lru_cache(maxsize=1)
def _load_curated_senses() -> tuple[dict[str, Any], ...]:
    """Curated common-noun place-name senses the L3 mining pipeline
    missed or couldn't promote (wyrd-v4rc).

    Distinct from the saint / manorial sidecars (which carry personal
    names): this holds ordinary toponymic vocabulary whose etymon
    exists in the lexicon DB but has no glossed, citable home — so it
    never reaches the bundle through the normal promotion paths. The
    seed case is extractive ``mine`` (Old French ``mine`` ← Gaulish,
    cognate cluster *mēy(H)nis): the DB carries the whole Celtic /
    Romance cluster but every member is gloss-less, and the only
    glossed modern-english ``mine`` etymon is the unrelated possessive
    pronoun (PIE *men-). Curation can't mint a new etymon + citation,
    so the sense is injected here as a runtime convenience subject.

    Each entry: ``name`` (surface form), ``language_field`` (bundle
    slot), ``gloss``, ``tags`` (user-facing theme tags — these are NOT
    internal-cohort tags, so the sense shows up as a normal morpheme),
    and ``modifier_type``.

    ``language_field`` should name the sense's actual historical
    stratum (``old_french`` for extractive mine, not ``modern_english``):
    ``_rank_siblings`` drops modern-english-only homographs when a
    historical etymon shares the surface, so a ``modern_english`` slot
    would be silently filtered out of the explainer whenever an older
    etymon (e.g. OE ``mine`` 'a slope') exists at the same surface.
    Slotting by true origin keeps the curated sense visible AND ranks
    it correctly by stratum next to its homographs.
    """
    with _data_path("curated_senses.json").open(encoding="utf-8") as f:
        return tuple(json.load(f))


def _curated_sense_subjects() -> list[dict[str, Any]]:
    """Synthesize meaning-db subjects for the curated common-noun
    senses (wyrd-v4rc). Mirrors ``_saint_personal_name_subjects`` —
    each entry becomes a Meaning the matcher resolves when a toponym
    carries the surface.

    Emits BOTH pre-position (``mine-``) and post-position (``mine``)
    variants so the sense matches at either end of a compound
    (``Minehead`` = mine + head; a trailing ``-mine``). Inner is
    omitted — a common noun embedded strictly inside a longer word is
    rare and would over-match.
    """
    senses = _load_curated_senses()
    subjects: list[dict[str, Any]] = []
    for entry in senses:
        token = entry["name"]
        for surface_usage in (f"{token}-", token):
            subjects.append(
                {
                    "meaning": [entry["gloss"]],
                    "modifier_tags": list(entry.get("tags") or []),
                    "modifier_type": entry["modifier_type"],
                    "words": [
                        {
                            "modern_usage": surface_usage,
                            entry["language_field"]: [token.lower()],
                        }
                    ],
                }
            )
    return subjects


def _norman_manorial_subjects() -> list[dict[str, Any]]:
    """Synthesize meaning-db subjects for the Norman manorial families
    (wyrd-8s71). Generation appends the family name as a post-base
    affix; the explainer needs to recognize the family when fed the
    same name back, so each family becomes a Meaning entry the matcher
    can resolve.

    Multi-word families ("La Zouche", "De Vere") produce a Meaning
    keyed on the surname token — the matcher splits on whitespace, so
    the particle ("La", "De") is left unaccounted by design. That
    keeps the registered keys distinctive — registering "La" and "De"
    as standalone manorial particles would also match in non-manorial
    contexts and pollute decompositions.

    The Meaning's gloss includes the full family string so the
    explainer surfaces "La Zouche" even when only "Zouche" matched.
    """
    families = _load_norman_manorial_families()
    subjects: list[dict[str, Any]] = []
    for family in families:
        # Surname-only token for matching: the last whitespace-split
        # word of the family string. Single-word families pass
        # through unchanged ("Mandeville" → "Mandeville").
        token = family.split()[-1]
        subjects.append(
            {
                "meaning": [f"Norman manorial family: {family}"],
                "modifier_tags": ["manorial", "norman"],
                "modifier_type": "Manorial",
                "words": [{"modern_usage": token, "old_french": [token.lower()]}],
            }
        )
    return subjects


# wyrd-d90t: kenning runtime reads its bundle from the L4 SQLite DB
# (resolved by runtime/runtime_db.get_runtime_db — env-var override
# → S3 ETag cache → S3 download → bundled seed-runtime.db fallback).
# meanings.json + per-culture proportions JSONs are gone — see D38 in
# DECISIONS.md for the architecture.
@lru_cache(maxsize=1)
def _runtime_db_bundle_dict() -> dict[str, Any]:
    """Read the L4 runtime DB and shape it into the dict the
    ``load_meanings`` / ``load_canonical_decompositions`` /
    ``load_fantasy_morphemes`` / ``load_joiners`` loaders consume.
    Process-cached so all four loaders share one SQLite walk + JSON-
    parse pass; the cache invalidates via the coupled clear (search
    ``_coupled_cache_clear``) so tests stay sane.

    Callers MUST treat the returned dict as read-only — the
    sidecar-folding path in :func:`_load_meanings` builds a fresh
    ``{"subjects": [...]}`` dict instead of mutating the cached one.
    """
    from wyrd.generators.kenning.runtime.runtime_db import get_runtime_db
    from wyrd.generators.kenning.runtime.runtime_db_adapter import (
        bundle_dict_from_runtime_db,
    )

    return bundle_dict_from_runtime_db(get_runtime_db())


@lru_cache(maxsize=1)
def _load_meanings():
    """Load the bundled meanings, extending with anglicized-form sidecars.

    The runtime meaning_db unions the main ``meanings.json`` (the
    mining-pipeline-emitted bundle) with hand-curated sidecars that
    cover morphemes the pipeline can't reach via Wiktionary headwords.

    wyrd-1cjg: ``irish_anglicizations.json`` carries anglicized Irish
    place-name elements (Bally-, Cloon-, Kil-, Knock-, etc.) keyed to
    the same gloss + language slot as their native Irish forms. Native
    forms land under ``celtic_mix`` per the lexicon's
    ``_LANG_CODE_TO_JSON_FIELD`` rollup (Irish + Old Irish + Middle
    Irish + Scottish Gaelic + Welsh + Breton all collapse there) so
    the sidecar uses the same key, ensuring the explainer's
    ``_ROOT_CODES`` lookup picks up the morphemes as Celtic ("CL")
    rather than rendering them as ``(?)``. The mining pipeline indexes
    Wiktionary by native headword (cluain, baile, achadh) so anglicized
    forms never surface — but ~27% of the bundled Irish corpus uses
    anglicized prefixes, so omitting them guts Irish coverage. Sidecar
    approach keeps the data reviewable + lets a future Wiktionary
    modern-English-slice mining pass supersede it without bundle
    re-emit.

    wyrd-8s71: Norman manorial families (synthesized at load time
    from ``norman_manorial_families.json``) become Meaning entries
    too, so KenningExplain can decompose the manorial-affix names
    Kenning generates with manorial_affix>0. The synthesized
    subjects use ``old_french`` for the language slot — Norman
    families are Anglo-Norman in origin (Old French dialect of
    11th-12th-c Normandy) so the linguistic bucket is correct, AND
    ``old_french`` is in ``_ROOT_CODES`` so the explainer renders
    the morpheme as "FR" rather than "(?)".
    """
    data = _runtime_db_bundle_dict()
    with _data_path("irish_anglicizations.json").open() as f:
        sidecar = json.load(f)
    manorial = _norman_manorial_subjects()
    saints = _saint_personal_name_subjects()
    curated_senses = _curated_sense_subjects()
    # Build a fresh subjects list with the sidecars folded in —
    # load_meanings only reads ``subjects`` from the dict, and passing
    # a fresh dict avoids mutating the cached _runtime_db_bundle_dict()
    # return value (which feeds the sibling loaders downstream). The
    # four sidecars (irish_anglicizations + Norman manorial families
    # + saint personal names + curated common-noun senses) stay as
    # JSON because they're runtime-only convenience tables that never
    # went through the L3 mining pipeline.
    subjects = list(data.get("subjects") or [])
    subjects.extend(sidecar)
    subjects.extend(manorial)
    subjects.extend(saints)
    subjects.extend(curated_senses)
    return load_meanings({"subjects": subjects})


def _decomposition_matcher(meaning_db: dict):
    """wyrd-04oi: the ``MorphemeTrie`` for the decomposition path (explain /
    era-map / rewind).

    Loads the pickled matcher published beside the runtime DB when its corpus
    stamp matches the ``meaning_db`` we just loaded; otherwise builds the trie
    (the prior behavior). Fail-safe — a missing / stale / corrupt / wrong-
    Python pickle simply falls back to a rebuild, so this can only save the
    cold-container trie build, never change a decomposition. Pass the result to
    ``Name.find_meaning(..., trie=...)``.
    """
    from wyrd.generators.kenning.runtime.matcher_cache import load_matcher, matcher_stamp
    from wyrd.generators.kenning.runtime.name import _trie_for
    from wyrd.generators.kenning.runtime.runtime_db import runtime_matcher_path

    path = runtime_matcher_path()
    if path is not None:
        cached = load_matcher(path, stamp=matcher_stamp(meaning_db))
        if cached is not None:
            return cached
    return _trie_for(meaning_db)


@lru_cache(maxsize=1)
def _load_joiners() -> dict[str, list[tuple[str, int]]]:
    """Load the bundle's joiner pool. Returns
    ``{lang_field: [(form, weight), ...]}``; the L4 schema doesn't
    carry joiners today, so this currently resolves to an empty dict."""
    return load_joiners(_runtime_db_bundle_dict())


@lru_cache(maxsize=1)
def _load_canonical_decompositions() -> dict[str, dict[str, str]]:
    """Load the bundle's per-toponym canonical decomposition map
    (wyrd-h8k1). Returns ``{modern_name: {"signature", "source"}}``."""
    return load_canonical_decompositions(_runtime_db_bundle_dict())


@lru_cache(maxsize=1)
def _load_fantasy_morphemes() -> dict[str, dict]:
    """wyrd-vz7f: load the bundle's fantasy-creature etymology map.
    Returns ``{lowercase_input_name: {input_name, etymon_id, language,
    canonical_form, english_shaped, glosses, citation, era_reflexes}}``."""
    return load_fantasy_morphemes(_runtime_db_bundle_dict())


@lru_cache(maxsize=1)
def _load_empirical_priors():
    """Load the bundled empirical priors from the L4 runtime DB
    (``empirical_priors`` singleton row, schema v2). Returns an empty
    ``EmpiricalPriors`` when the row is absent (L3 hadn't run
    ``mine-empirical-baselines`` at emit time) — vector-scoring path
    degrades to phon + sem + pos in that case.

    Pre-cutover this read ``priors.json`` from the bundled data dir;
    d90t folded the priors payload into the L4 so vector mode works
    out of the box without a separate sidecar."""
    from wyrd.generators.kenning.lexicon.empirical_priors import (
        load_empirical_priors_from_payload,
    )
    from wyrd.generators.kenning.runtime.runtime_db import get_runtime_db
    from wyrd.generators.kenning.runtime.runtime_db_adapter import (
        empirical_priors_payload_from_runtime_db,
    )
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors

    payload = empirical_priors_payload_from_runtime_db(get_runtime_db())
    if payload is None:
        return EmpiricalPriors()
    return load_empirical_priors_from_payload(payload)


@lru_cache(maxsize=1)
def _load_genitive_prior() -> dict[tuple[str, str], float]:
    """wyrd-aicu.9: load the bundled genitive-split prob-map from the L4 runtime
    DB (``genitive_split_prior`` singleton row — an additive/optional table, no
    schema bump). Returns the precomputed ``{(long_form, short_form):
    split_probability}`` float map for the decomposition matcher's
    ``_prefer_genitive_credible`` tiebreaker.

    STAGED, not yet consumed at runtime. The connective fix currently flows via
    EXPORT-time proportions training (``_compute_proportions_inline`` passes the
    map into ``find_meaning``), so the shipped bundle's proportions already
    reflect it. This runtime loader is for a FUTURE consumer — wiring the prior
    into the LIVE generation/explain decompose path (wyrd-aicu.9 follow-on);
    until then nothing calls it (kept tested + bundled so the consumer is a
    one-line change).

    Returns ``{}`` when the row/table is absent (a wipe-without-remine L4, a
    pre-aicu.9 seed lacking the additive table, or one predating
    ``mine-genitive-priors --apply``): graceful degradation, tiebreak silent.
    Load-once (lru_cache) so the request path never re-reads the DB; cleared
    alongside the other bundle caches by ``_coupled_cache_clear``."""
    from wyrd.generators.kenning.runtime.runtime_db import get_runtime_db
    from wyrd.generators.kenning.runtime.runtime_db_adapter import (
        genitive_split_map_from_runtime_db,
    )

    return genitive_split_map_from_runtime_db(get_runtime_db()) or {}


@lru_cache(maxsize=1)
def _load_packs():
    """wyrd-ecjp.10b: load all bundled scenario packs from the
    ``packs/`` subdir of the data root. Returns
    ``{pack_name: PackBundle}`` (empty when packs/ is absent —
    bundles that ship no packs work unchanged).

    Each pack lives in ``packs/<pack_name>/`` with manifest.json +
    meanings.json (operator chose per-pack dirs for max flexibility).
    The bundle loader is the data-layer half of pack support; the
    runtime CLI wiring (--pack flag) is deferred to wyrd-ecjp.11.
    """
    from wyrd.generators.kenning.lexicon.pack_loader import load_packs_from_traversable

    return load_packs_from_traversable(_data_path("packs"))


@lru_cache(maxsize=len(CULTURES))
def _load_culture(culture: str):
    if culture not in CULTURES:
        raise ValueError(f"unknown culture: {culture}; expected one of {CULTURES}")
    meaning_db, tag_db = _load_meanings()
    from wyrd.generators.kenning.runtime.runtime_db import get_runtime_db
    from wyrd.generators.kenning.runtime.runtime_db_adapter import (
        proportions_dict_for_culture,
    )

    proportions = proportions_dict_for_culture(get_runtime_db(), culture)
    # wyrd-hzqs: thread the culture NAME so the per-language structure allowlist
    # (is_structure_enabled) can key on it.
    return load_proportions(proportions, meaning_db, tag_db, culture=culture), tag_db


# wyrd-h3ls: ``_load_culture`` independently caches a ``NameGenerator``
# wrapped over the ``meaning_db`` returned by ``_load_meanings``. So
# calling only ``_load_meanings.cache_clear()`` (the obvious test-side
# invalidation) leaves the per-culture generator stale, holding a
# reference to the old ``meaning_db``. Couple the clears so the
# obvious call also drops the per-culture cache — a future test author
# who mutates the bundle and clears ``_load_meanings`` doesn't need to
# know about the second cache.
_original_load_meanings_cache_clear = _load_meanings.cache_clear


def _coupled_cache_clear() -> None:
    """Clear all per-bundle caches: ``_load_meanings``,
    ``_load_culture``, ``_load_joiners``,
    ``_load_canonical_decompositions``, ``_load_fantasy_morphemes``,
    and the shared ``_runtime_db_bundle_dict``.

    These caches form an aggregate over one runtime-DB read:
    ``_load_culture`` holds a ``NameGenerator`` parameterised on
    the ``meaning_db`` from ``_load_meanings``; the other loaders
    rehydrate from the same ``_runtime_db_bundle_dict`` snapshot.
    Invalidating one without the others yields a stale view next
    time. Replaces ``_load_meanings.cache_clear`` so the standard
    test-side pattern (``_load_meanings.cache_clear()``) clears all.

    The reverse direction (``_load_culture.cache_clear()`` alone) is
    intentionally uncoupled — a caller who only wants to drop the
    per-culture generator shouldn't pay to re-parse the bundle.
    """
    _original_load_meanings_cache_clear()
    _load_culture.cache_clear()
    _load_joiners.cache_clear()
    _load_canonical_decompositions.cache_clear()
    _load_fantasy_morphemes.cache_clear()
    _load_empirical_priors.cache_clear()
    _load_genitive_prior.cache_clear()
    _load_packs.cache_clear()
    _runtime_db_bundle_dict.cache_clear()
    # wyrd-ah53: per-culture tag options derive from the per-culture pools, so
    # they go stale with the bundle too.
    _tags_options_by_culture.cache_clear()
    # wyrd-eyjk/D40: the module-level bare-surface index is keyed by
    # id(meaning_db); after a bundle reload the old meaning_db is freed and
    # CPython can reuse its id() for a NEW (different-bundle) meaning_db, which
    # would then read the OLD bundle's stale index. Drop it here so it rebuilds
    # against the fresh bundle in lockstep with _load_meanings.
    _clear_surface_index_cache()


# mypy flags reassigning a bound method on the lru_cache wrapper as
# method-assign; intentional here — the patch is the whole point.
_load_meanings.cache_clear = _coupled_cache_clear  # type: ignore[method-assign]


def available_tags() -> list[str]:
    """User-visible tags from the meaning DB (excludes internal filtering tags)."""
    _, tag_db = _load_meanings()
    return sorted(t for t in tag_db if t not in _INTERNAL_TAGS)


@lru_cache(maxsize=1)
def _tags_options_by_culture() -> dict[str, list[str]]:
    """Per-culture user-visible tags for the SPA's dependent select (wyrd-ah53):
    the tags carried by >=1 morpheme in that culture's eligible pool.

    The tag vocabulary is culture-agnostic (``available_tags`` is the union over
    every culture), but a tag absent from a culture — e.g. 'monster', a
    fantasy/pfsrd2 tag with zero English morphemes — can never be satisfied: a
    ``--tag`` reserves ONE slot drawn from a pool that HAS the tag (D47), so
    offering it for that culture would only ever raise 'no eligible name' (bare) or
    silently no-op (combined with a satisfiable tag). The SPA filters the tag
    composer to the chosen culture's set via ``x-options-by-culture`` — the same
    dependent-select mechanism as era / stratum.

    Derived from the generation pool: ``build_non_position_eligible`` with only the
    culture attestation (no era / stratum / tag / position / qualifier / pack
    filtering). So this is a conservative SUPERSET of the strictly-placeable set —
    a tag is offered ONLY IF some morpheme in the culture's pool carries it, but a
    tag whose carriers happen to have zero per-position frequency in every slot of
    every struct could still be offered yet 'no eligible name' at roll time. The
    direction that matters is guaranteed: this never OMITS a placeable tag (it only
    ever over-offers, which the runtime degrades safely), so it can't hide a tag a
    real roll could use. Cached (cleared with the bundle via
    ``_coupled_cache_clear``), so tag additions propagate on the next manifest
    refresh."""
    from wyrd.generators.kenning.runtime.vector_name_select import (
        build_non_position_eligible,
    )
    from wyrd.generators.kenning.vectors.schemas import EligibilityGate

    out: dict[str, list[str]] = {}
    for culture in CULTURES:
        name_gen, _ = _load_culture(culture)
        pool = build_non_position_eligible(
            name_gen.meaning_db,
            gate=EligibilityGate(culture=culture),
            exclude_tags=frozenset(),
            pack_meaning_dbs=None,
            packs=(),
            culture_attested_usages=name_gen.culture_attested_usages,
            culture_attested_meanings=name_gen.culture_attested_meanings,
        )
        out[culture] = sorted({t for m in pool for t in m.tags if t not in _INTERNAL_TAGS})
    return out


def _coerce_bool(value: Any) -> bool:
    """Coerce a request-side value to bool, treating common false-tokens as
    False rather than truthy.

    The SPA renders boolean params via a text input today (no checkbox
    branch), so a default of False ships across the wire as the literal
    string ``"false"``. Plain ``bool("false")`` is True, which would
    silently invert the gate. This coercion handles the JSON-bool path
    (passed through unchanged), the SPA string path, and the empty-form
    path uniformly.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Coerce a generation-knob param to a float, defaulting a blank value.

    ``None`` / ``""`` (a missing or empty field) → ``default``. A numeric string
    coerces (the GET query-string path). A NON-coercible value — a list, dict, or
    non-numeric string — raises ``ValueError`` (which the dispatcher maps to a 400
    bad_params) rather than letting a bare ``float()`` raise ``TypeError`` on a
    list/dict, which escapes uncaught as a 500. Replaces the
    ``float(params.get(k, 0.0) or 0.0)`` idiom, byte-for-byte for every scalar/
    string/blank value — only a list/dict knob value changes (500 → 400)."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{value!r} is not a number") from e


_PRESENT_DAY_ERA = "present-day"


def _translate_present_day_era(era: Any, culture: str) -> Any:
    """wyrd-kqyf: ``present-day`` is the culture-AGNOSTIC era token (the value
    ``WYRD_DEFAULT_ERA`` is set to, terraform ``feature_flag_defaults``). Era
    stages are per-culture (``modern-english`` / ``welsh`` / ``irish``), so no
    literal stage can be a global default — a welsh request defaulting to
    ``modern-english`` would 4xx. The token translates to the request
    culture's PRESENT-DAY stage (the last entry of its family's
    chronologically-ordered ``family_stage_order``) and then resolves exactly
    like an explicit stage pick: the stage's union year range as the filter +
    the force-modern render (``_resolve_era_render_language`` returns None for
    contemporary stages). Any other value passes through untouched."""
    if era == _PRESENT_DAY_ERA:
        family = _CULTURE_TO_ERA_FAMILY.get(culture, "english")
        stages = family_stage_order(family)
        if not stages:
            # Unreachable for today's families (every ERA_CELLS family has
            # cells) — the guard turns a future stage-less family edit into
            # the callers' clean ValueError contract instead of IndexError.
            raise ValueError(f"no era stages defined for family {family!r}")
        return stages[-1]
    return era


def _resolve_era_param(era: Any, culture: str) -> tuple[int | None, int | None] | None:
    """Resolve the request-side ``era`` value to a half-open year range,
    or None when no era filter applies.

    Treats both None and ``""`` (the SPA's empty form-field) as 'no
    filter' explicitly, matching the rest of this generator's input
    handling (each knob has its own ``or 0.0`` / ``_coerce_bool`` /
    ``or []`` normalization). Wraps the underlying
    ``resolve_era_input`` errors in a single ValueError naming the bad
    input + the resolved era family — so a ``--era victorian`` typo
    surfaces as a clean 4xx through the API rather than a raw KeyError
    propagating from inside the resolver.
    """
    if era is None or era == "":
        return None
    era_family = _CULTURE_TO_ERA_FAMILY.get(culture, "english")
    try:
        era = _translate_present_day_era(era, culture)
        return resolve_era_input(era, default_family=era_family)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"invalid 'era' value {era!r} for culture {culture!r} "
            f"(era family {era_family!r}): {exc}. Pass a year (e.g. "
            f"1086), a cell label defined in the culture's era family, "
            f"or an explicit 'family/label' pair."
        ) from None


def _contemporary_language_for_family(family: str) -> str | None:
    """The canonical etymon-language of a family's present-day cell — the
    open-ended (``end is None``) cell, e.g. 'modern-english' for english,
    'welsh' for brythonic. None when the family has no open-future cell (a dead
    language like latin, whose cells are all historical).

    Used to suppress era-rendering of the contemporary period (wyrd-6c8x): a
    morpheme's canonical surface is ALREADY in this language, so re-rendering it
    via the cognate-based era-reflex picker would distort it rather than
    period-ize it. Mirrors kenning_rewind's wyrd-8qbi rule ('modern == the
    original') but derived from the cell structure so it generalizes past
    english (welsh/irish/french moderns are bare tags, not 'modern-*')."""
    try:
        cells = era_cells_for_family(family)
    except KeyError:
        return None
    for cell in cells:
        _, end = era_year_range(family, cell)
        if end is None:
            return canonical_language_for_cell(family, cell)
    return None


def _resolve_era_render_language(era: Any, culture: str) -> str | None:
    """wyrd-6c8x (feature A): the canonical etymon-language a requested era
    should RENDER morphemes in, or None for 'render the modern canonical form'.

    Parallel to :func:`_resolve_era_param` — that returns the half-open year
    range used to FILTER the morpheme inventory; this returns the target
    language used to render each picked morpheme in its era-appropriate
    attested form (via ``Meaning.era_reflex_for``), so ``era=oe-early`` yields
    period-LOOKING names, not just a period-eligible inventory.

    Returns None (no era render — render the modern canonical form) when:
    * no era is set (None / ``""``) — the bare-modern path,
    * the era cell has no canonical language (``canonical_language_for_cell``
      → None — eras with no single anchor language, e.g. Norse post-classical),
    * the era cell's language IS the family's contemporary language (the
      present-day / early-modern cells, which map to 'modern-english' etc.) —
      the canonical surface is already that form, so era-rendering would pull
      cognate cluster-mates and distort it (mirrors rewind's wyrd-8qbi), or
    * the era value is malformed — the loud ValueError is left to
      :func:`_resolve_era_param`, which runs alongside this on the same input;
      here we degrade to no-render rather than raise twice.
    """
    if era is None or era == "":
        return None
    era_family = _CULTURE_TO_ERA_FAMILY.get(culture, "english")
    try:
        era = _translate_present_day_era(era, culture)
        family, cell = era_cell_for_input(era, default_family=era_family)
    except (KeyError, ValueError):
        return None
    lang = canonical_language_for_cell(family, cell)
    if lang is None or lang == _contemporary_language_for_family(family):
        # wyrd-3vju.2: the contemporary stage (modern-english) stays FORCE-modern
        # — render the clean modern usage, NOT era_reflex_for("modern-english").
        # Empirically the modern era reflexes carry archaic/wrong forms (New- →
        # "Niwan" OE, gold → "gowth"), so era-rendering modern distorts 42/60
        # surfaces (the wyrd-8qbi concern; id-first does NOT save it). The highlight
        # for this force-modern path is corrected in _set_active_form_id (the
        # present-day usage anchor, wyrd-3vju.3), so modern is a real,
        # correctly-highlighted choice WITHOUT the distortion.
        return None
    return lang


def _resolve_stratum_param(stratum: Any, culture: str) -> str | None:
    """Resolve the request-side ``stratum`` value to a stratum tag,
    or None when no stratum filter applies (wyrd-j3gy).

    Treats both None and ``""`` (the SPA's empty form-field) as 'no
    filter' explicitly, matching ``_resolve_era_param``'s shape.

    Validation has two layers:
      1. Per-culture restriction (when configured) — for english /
         scottish / welsh, the stratum must be in the culture's
         allowed-set. Catches both typos AND culturally-incoherent
         values (e.g. ``--culture welsh --stratum east-norse``,
         where east-norse isn't in any classified Welsh-bundle
         language family).
      2. ALL_STRATA fallback — for cultures without a per-culture
         restriction (irish / breton today), validate against the
         broader cross-family registry. Catches typos but not
         family mismatches; tightens once the missing classifiers
         (Goidelic / Brythonic-Brythonic) ship.

    Bad input → ValueError naming the culture and listing valid
    strata so the SPA / CLI surface a clean 4xx. Same error-wrapping
    pattern as ``_resolve_era_param``.
    """
    if stratum is None or stratum == "":
        return None
    # Empty frozenset = no per-culture restriction (irish/breton, plus
    # any culture not in the dict at all); fall through to the
    # ALL_STRATA typo-check. Truthiness on a frozenset is what selects
    # between the two branches.
    valid = valid_strata_for_culture(culture)
    if valid:
        if stratum not in valid:
            raise ValueError(
                f"invalid 'stratum' value {stratum!r} for culture {culture!r}: "
                f"valid options are {sorted(valid)}."
            )
    elif stratum not in ALL_STRATA:
        raise ValueError(
            f"unknown 'stratum' value {stratum!r}: not in any classifier's "
            f"vocabulary (culture {culture!r} has no per-culture restriction "
            f"configured yet, so this falls back to a typo-check). Known: "
            f"{sorted(ALL_STRATA)}."
        )
    return stratum


def _roots(meaning: Meaning) -> list[str]:
    return [code for src, code in _ROOT_CODES if src in meaning.sources]


def _all_roots(meanings: list[Meaning]) -> list[str]:
    """Union of root codes across every Meaning sharing one usage. Order
    follows _ROOT_CODES so output is stable."""
    seen: set[str] = set()
    for m in meanings:
        seen.update(_roots(m))
    return [code for _, code in _ROOT_CODES if code in seen]


def _all_senses(meanings: list[Meaning]) -> list[str]:
    """Distinct sense strings across every Meaning sharing one usage,
    preserving first-seen order."""
    return list(dict.fromkeys(sense for m in meanings for sense in m.meanings))


# wyrd-o53o + wyrd-aizb: classify gloss strings as 'derivative' vs
# 'semantic'. Derivative glosses point at another lemma rather than
# carrying meaning ('alternative form of homes', 'singular imperative
# of singan', 'plural of bord', 'soft mutation of llys', 'genitive of
# Albain'). 22.6% of bundle glosses match these patterns; surfacing
# them as primary meanings in the SPA's morpheme inspector is
# noise — 'half the lemmas only tell me derivations' (user, 2026-
# 05-22 QA). Suppress them in display when semantic glosses exist
# alongside; surface them in a separate field for the SPA to render
# secondarily if it wants.
#
# Patterns chosen to match common morphological-pointer templates
# WITHOUT matching genuine semantic glosses that happen to use 'of'
# (e.g. 'wife of X', 'son of X', 'a stream of running water' — none
# of those start with grammatical terms like 'plural', 'inflection',
# 'mutation', etc.).
# Grammatical vocabulary for the derivative-gloss matcher. Split into
# two tiers:
#
#   PRIMARY — must LEAD the derivative chain. These are the words that
#     unambiguously signal a morphological pointer when they're the
#     opening grammatical token (alternative/archaic/plural/imperative/
#     etc.).
#
#   SECONDARY — may appear AFTER a primary, but cannot lead alone. These
#     are nouns describing the KIND of derivation (form, forms,
#     spelling, tense, mutation, etc.). They appear in compound
#     templates like 'alternative form of', 'soft mutation of',
#     'present tense of' but are too generic to lead — e.g. 'Form of
#     an enclosure' is a SEMANTIC gloss, not a derivative pointer.
#
# Hyphenated tokens (third-person) and slash-pairs (masculine/neuter)
# are both supported in the regex below.
_GRAMMATICAL_PRIMARY = (
    # cases
    "nominative",
    "accusative",
    "dative",
    "genitive",
    "vocative",
    "locative",
    "ablative",
    "instrumental",
    # mood / voice
    "imperative",
    "subjunctive",
    "indicative",
    "optative",
    "infinitive",
    "conditional",
    "active",
    "passive",
    # tense / aspect (verb categories, not the noun 'tense')
    "past",
    "present",
    "future",
    "perfect",
    "pluperfect",
    "imperfect",
    "preterite",
    "aorist",
    # Celtic verb forms
    "conjunct",
    "absolute",
    "deuterotonic",
    "prototonic",
    # person (both hyphenated 'first-person' and bare 'first' —
    # the bare forms cover slashed compounds like 'first/third-person'
    # that appear in the corpus alongside the hyphenated singletons)
    "first-person",
    "second-person",
    "third-person",
    "first",
    "second",
    "third",
    # number / gender / definiteness
    "singular",
    "plural",
    "dual",
    "masculine",
    "feminine",
    "neuter",
    "common",
    "definite",
    "indefinite",
    # derivational signals
    "alternative",
    "archaic",
    "obsolete",
    "diminutive",
    "augmentative",
    "comparative",
    "superlative",
    "equative",
    # Welsh mutation classes (must lead 'X mutation of' chains)
    "soft",
    "hard",
    "nasal",
    "aspirate",
    # verb-form classifiers usable standalone ('inflection of X',
    # 'plural of X' — the noun forms double as primaries because
    # they're context-specific enough)
    "inflection",
    "conjugation",
    "declension",
    "participle",
    "gerund",
    "verbal",
    "conjugated",
    "inflected",
    "declined",
)
_GRAMMATICAL_SECONDARY = (
    # nouns describing the derivation type — only valid AFTER a primary.
    # 'person' here covers compound forms like 'first/third-person of X'
    # in case the corpus surfaces 'first person of X' as space-separated
    # rather than hyphenated (the hyphenated 'first-person' stays in
    # primary).
    "form",
    "forms",
    "spelling",
    "tense",
    "mutation",
    "person",
)


def _alt(words):
    """Build a regex alternation of escaped words, optionally
    slash-paired (e.g. 'masculine/neuter'). Sorts longest-first so a
    shorter prefix can't silently shadow a longer match in Python's
    left-to-right alternation (e.g. 'first' would otherwise match
    before 'first-person' was tried). wyrd-o53o round 2 (Gemini MED)."""
    inner = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return rf"(?:{inner})(?:/(?:{inner}))*"


_PRIMARY = _alt(_GRAMMATICAL_PRIMARY)
_PRIMARY_OR_SECONDARY = _alt(_GRAMMATICAL_PRIMARY + _GRAMMATICAL_SECONDARY)
# wyrd-o53o round 2 (Gemini MED): [\s-]+ between tokens so glosses like
# 'present-tense of X' or 'first-person-singular of X' match. The final
# ' of ' anchor stays \s+ (the 'of' MUST be space-separated) so we
# don't get false positives like 'past-mistakes-of-the-king'.
_DERIVATIVE_PATTERN = re.compile(
    # wyrd-o53o round 7 (Gemini MED): 'an' added to the optional
    # article prefix. Several primaries start with vowels
    # (alternative, archaic, obsolete, inflected, imperative,
    # absolute, aorist) so 'an alternative form of X' would
    # otherwise miss.
    r"^\s*(?:a\s+|an\s+|the\s+)?"
    rf"(?:{_PRIMARY})(?:[\s-]+(?:{_PRIMARY_OR_SECONDARY}))*\s+of\b",
    re.IGNORECASE,
)


def _is_derivative_gloss(gloss: str) -> bool:
    """Return True if the gloss is a morphological pointer at another
    lemma (e.g. 'alternative form of X', 'plural of X', 'soft mutation
    of X') rather than a semantic definition."""
    return bool(_DERIVATIVE_PATTERN.match(gloss))


def _partition_senses(senses: list[str]) -> tuple[list[str], list[str]]:
    """Split a sense list into (semantic, derivative), preserving order
    within each partition. Use _split_senses_for_display() when you want
    the user-facing 'semantic or fall-back-to-derivative' policy."""
    semantic: list[str] = []
    derivative: list[str] = []
    for s in senses:
        if _is_derivative_gloss(s):
            derivative.append(s)
        else:
            semantic.append(s)
    return semantic, derivative


def _split_senses_for_display(senses: list[str]) -> tuple[list[str], list[str]]:
    """Return (primary_meanings, derivative_meanings) for the SPA
    inspector's two-section layout. When semantic glosses exist,
    primary = semantic, derivative = the morphological pointers (the
    SPA may show these dimmed/secondary). When ONLY derivatives exist
    (single-sibling derivative-only entries like '-sing- → "singular
    imperative of singan"'), primary = derivatives so the card isn't
    blank — better to show a hint at the target lemma than nothing.
    The derivative list returns [] in that case to avoid duplicating
    the primary list."""
    semantic, derivative = _partition_senses(senses)
    if semantic:
        return semantic, derivative
    return derivative, []


_GLOSS_ARTICLE_RE = re.compile(r"^(?:an?|the)\s+", re.IGNORECASE)


def _gloss_dedup_key(gloss: str) -> str:
    """wyrd-0y3k: normalize a gloss for dedup — ASCII-fold + lowercase (reuse
    _normalize_for_similarity), drop a leading article, strip surrounding
    punctuation/space. So 'A hill.', 'Hill', 'a hill', 'hill' all collapse to
    'hill'."""
    k = _normalize_for_similarity(gloss)
    k = _GLOSS_ARTICLE_RE.sub("", k)
    return k.strip(" .,;:").strip()


def _meaning_groups(meanings: list[Meaning]) -> list[list[str]]:
    """wyrd-0y3k: per-sibling sense groups for the SPA inspector. Each ranked
    sibling Meaning is a distinct etymon/sense, so its SEMANTIC glosses form a
    coherent cluster ('valley/dale/dene' vs 'farm/town/estate' vs 'people').
    Grouping them — instead of unioning every sibling's glosses into one flat
    ~50-item list (the 'denu-' mess) — lets the card show the senses visually
    separated. Within each group, glosses are deduped article/case/punct-
    insensitively (keeping the shortest representative). Pure-derivative
    siblings (no semantic gloss) are dropped. Order follows _rank_siblings."""
    groups: list[list[str]] = []
    seen_global: set[str] = set()  # a gloss shows once, in its FIRST group
    for m in meanings:
        semantic, _ = _partition_senses(m.meanings)
        rep: dict[str, str] = {}
        order: list[str] = []
        for g in semantic:
            k = _gloss_dedup_key(g)
            if not k or k in seen_global:
                continue
            if k not in rep:
                rep[k] = g
                order.append(k)
            elif len(g) < len(rep[k]):
                rep[k] = g  # prefer the cleanest (shortest) form, e.g. 'Hill'
        if order:
            seen_global.update(order)
            groups.append([rep[k] for k in order])
    return groups


@lru_cache(maxsize=4096)
def _normalize_for_similarity(s: str) -> str:
    """Lowercase + strip dashes + strip whitespace + strip ASCII-fold
    diacritics ('ē' → 'e', 'ā' → 'a'). The bundle's OE lemmas use
    macrons (hȳll, brād) and the modern_usage forms don't, so we
    fold before comparing surface similarity.

    wyrd-ubbc round 4 (Gemini MED): @lru_cache because the lexicon
    is static and the same forms get normalized many times per
    decomposition (same usages reappear, same source lemmas appear
    in multiple buckets). 4096 caps memory in pathological corpora;
    typical bundle has ~10K distinct forms so the cache mostly
    fills once and stays warm."""
    if not s:
        return ""
    cleaned = s.strip().strip("-").lower()
    # NFD splits combined chars; then drop the combining marks.
    return "".join(c for c in unicodedata.normalize("NFD", cleaned) if not unicodedata.combining(c))


def _max_form_similarity(matcher: SequenceMatcher, usage_norm: str, m: Meaning) -> float:
    """Highest difflib ratio between the normalized usage (already
    pinned as sequence a on `matcher`) and any source-language lemma
    in this Meaning.

    wyrd-ubbc: catches folk-etymology cluster pollution like the
    '-hill' bucket containing OE 'hyll' (highly similar to 'hill'),
    OE 'holt' (less similar), and OE 'cyln' (kiln; less similar).
    Pre-fix all three tied on meaning count; surface match now
    breaks the tie so the morpheme card shows 'Hill' instead of
    'Forest/Grove' or 'Kiln'.

    Caller responsibilities (wyrd-ubbc round 5 Gemini MED): construct
    the SequenceMatcher ONCE per ranking with
    `SequenceMatcher(autojunk=False, a=usage_norm)` and pass it in;
    pass the matching `usage_norm` so the exact-match shortcut can
    string-compare without re-fetching from the matcher. This
    avoids re-preprocessing usage_norm for each sibling in the
    bucket."""
    if not usage_norm:
        return 0.0
    best = 0.0
    for forms in m.sources.values():
        for f in forms:
            if not isinstance(f, str):
                continue
            f_norm = _normalize_for_similarity(f)
            if f_norm == usage_norm:
                return 1.0
            matcher.set_seq2(f_norm)
            score = matcher.ratio()
            if score > best:
                best = score
    return best


def _is_pure_proper_noun(m: Meaning) -> bool:
    """True when ``m`` is name/saint-tagged and carries NO common-noun tag —
    i.e. a personal name / saint with no place-element sense (``Bourne``,
    ``Andrew``), as opposed to a place element merely co-tagged with a name
    (``stān`` 'stone' + 'Stan'). Delegates to ``Meaning.is_pure_proper_noun``
    (wyrd-eyjk/D40) so the predicate has a single definition."""
    return m.is_pure_proper_noun()


def _rank_siblings(siblings: list[Meaning]) -> list[Meaning]:
    """Filter + rank sibling Meanings under one usage bucket so the
    highest-quality etymon's senses + tags surface first in the SPA's
    morpheme inspector.

    Three passes:

    1. FILTER (wyrd-ywm9): when ANY sibling has a historical-stratum
       etymon (i.e. _roots() returns non-empty — one of OE / ON / OF /
       Celtic / Latin / Germanic / Greek), drop the siblings with
       empty _roots(). Those empty-root siblings are modern_english-
       only homographs that pollute British-place-name etymology
       (the 'Green- → step/rung from obsolete gree' case). When NO
       sibling has historical roots (a truly modern morpheme),
       fall back to the unfiltered list rather than empty.

    2. RANK (wyrd-emlb): sort the survivors by signal density. Older
       stratum wins first (_ROOT_CODES order: OE > ON > OF > Celtic >
       …), then richer meaning + tag counts. So 'Green-' picks the
       OE 'grēne (Green / Village green, 9 tags, attested 1275)' over
       the Celtic 'grianán (Bright / Shining / Sunny, 1 tag)'.

    3. SURFACE TIEBREAKER (wyrd-ubbc): within a stratum, prefer
       etymons whose source lemma surface-matches the usage best
       (continuous similarity via difflib.SequenceMatcher.ratio()
       after NFD-stripping diacritics). Catches folk-etymology
       cluster pollution like the '-hill' bucket where OE 'hyll'
       scores 1.00 (exact after macron-strip), 'holt' 0.50, 'roth'
       /' cyln' 0.25 — hyll wins the within-stratum sort even
       though all four tie on meaning count. Without this, 'holt'
       or 'cyln' could win and the morpheme card would show
       'Forest' or 'Kiln' for a -hill suffix.

    The downstream consumers (_all_senses, _all_roots) preserve
    first-seen order, so the BEST Meaning's data appears first in
    the union — the rest of the senses still ride along, just lower
    in the list.

    Always returns a NEW list; safe to call on empty input
    (returns []) or single-element input (returns a copy).
    """
    if len(siblings) <= 1:
        return list(siblings)

    # Pass 1: filter out modern_english-only siblings when historical
    # etymons are present.
    historical = [m for m in siblings if _roots(m)]
    filtered = historical if historical else list(siblings)

    # wyrd-ubbc round 4 (Gemini MED): early-return when filtering
    # leaves ≤1 candidate. Skips the similarity computation + sort
    # for the common case of a single-historical-etymon bucket.
    if len(filtered) <= 1:
        return filtered

    # Pass 2 + 3: combined sort key. All siblings share the same
    # usage (.usage attribute), so the normalized usage is computed
    # once and the SequenceMatcher is constructed once
    # (wyrd-ubbc round 5 Gemini MED) — was per-sibling before.
    # filtered has >= 1 element at this point (single-element early
    # return above + non-empty-fallback in pass 1) so we can index
    # directly without guarding (wyrd-ubbc round 6 Gemini MED).
    usage_norm = _normalize_for_similarity(filtered[0].usage)
    matcher = SequenceMatcher(autojunk=False, a=usage_norm) if usage_norm else None

    def _signal(m: Meaning) -> tuple:
        stratum_rank = min(
            (_STRATUM_PRIORITY[s] for s in m.sources if s in _STRATUM_PRIORITY),
            default=len(_ROOT_CODES),
        )
        # wyrd-5z5j: a common-noun place-name sense outranks a PURE
        # proper-name / saint sense within the same stratum. Defect-A
        # (name-tagged derive_positions) introduced personal-name etymons
        # whose modern surface spelling can EXACTLY match a suffix usage —
        # e.g. the male name 'Bourne' (tags == {male name}, surface 'bourne')
        # beating OE 'burna' on the surface tiebreaker for '-bourne'. We
        # demote ONLY pure proper nouns: a morpheme that is genuinely a
        # place element AND carries an incidental name tag (OE 'stān'
        # 'stone' is also co-tagged 'male name' via the 'Stan' merge) keeps
        # common-noun rank because it still carries geology/topography/etc.
        # tags. Sorted ABOVE surface_similarity so canonicity-of-sense wins
        # over surface match; a no-op when every sibling is a pure name
        # (a genuine name morpheme like 'Saint-') — they all tie and fall
        # through to the existing key.
        is_common_sense = 0 if _is_pure_proper_noun(m) else 1
        # wyrd-ubbc: surface similarity is sorted AFTER stratum
        # (canonicity wins over surface similarity — an OE etymon
        # still beats a Celtic etymon at the same surface match)
        # but BEFORE meaning/tag counts (so 'hyll' beats 'holt'
        # inside OE for the -hill bucket).
        surface_similarity = _max_form_similarity(matcher, usage_norm, m) if matcher else 0.0
        # wyrd-aicu.7: a tagged etymon outranks an UNTAGGED thin stub, sorted ABOVE
        # surface_similarity. The aicu.1 bare-surface merge pooled mining stubs (a
        # literal modern-spelling form → surface_similarity 1.0, but 0 tags + few
        # glosses) with the canonical scholarly etymon (macron/thorn-spelled →
        # <1.0 match, but richly tagged) — and the 1.0 surface match pre-empted
        # richness, so the stub ranked top (wick→ the 'wick' stub over wīc;
        # ley→ the 'ley' stub over lēagum). A coarse tags-present boolean cleanly
        # separates real scholarly etymons from stubs WITHOUT disturbing wyrd-ubbc:
        # the -hill pollution candidates (hyll/holt/cyln) are ALL tagged, so they
        # tie here and fall through to surface_similarity-then-count as before.
        # Full-corpus (6361 multi-sibling pools): 56 top-flips, 46 improvements,
        # 0 pinned-canonical regressions; the ONLY predicate that also fixes ley
        # (gloss/citation-count predicates wrongly promote tūn's 35-gloss pool).
        has_rich_signal = 1 if m.tags else 0
        # wyrd-24s6 (D41): a stable, content-derived final tiebreaker. Without it,
        # siblings tying on every signal above retain meaning_db LOAD order, which
        # varies across SQLite / Python builds — so `siblings[0]` (the canonical
        # etymon) could differ between environments. That non-determinism was
        # latent until D41's native render started routing diversification
        # re-picks through the canonical's tags/languages, surfacing as a
        # cross-environment generation drift (the parity test caught it). Keying
        # on the morpheme identity makes the ranking independent of load order.
        return (
            -stratum_rank,
            is_common_sense,
            has_rich_signal,
            surface_similarity,
            len(m.meanings),
            len(m.tags),
            _sibling_identity(m),
        )

    return sorted(filtered, key=_signal, reverse=True)


def _sibling_identity(m: Meaning) -> str:
    """wyrd-24s6: a stable, content-derived identity string for deterministic
    sibling tie-breaking in :func:`_rank_siblings`. Prefers the ``morpheme_id``
    (the source-language canonical, e.g. ``old-english:hyll``); falls back to a
    sorted ``(lang, forms)`` signature of the etymon's sources, then to the
    repr of a non-dict ``sources`` (synthetic test Meanings). Never depends on
    meaning_db load order, so the ranking is reproducible across environments."""
    mid = getattr(m, "morpheme_id", None)
    if mid:
        return str(mid)
    sources = getattr(m, "sources", None)
    if isinstance(sources, dict):
        return repr(sorted((lang, tuple(forms)) for lang, forms in sources.items()))
    return repr(sources)


# --- compose-time joiner insertion (wyrd-q0g6 Phase 1.5) -----------------


def _shared_lang_fields_with_joiners(
    left: list[Meaning],
    right: list[Meaning],
    joiners: dict[str, list[tuple[str, int]]],
) -> set[str]:
    """Return lang_fields BOTH morpheme groups carry AND that have a
    populated joiner pool. Empty when the pair has no shared family
    or when no joiner pool exists for the shared language(s)."""
    left_langs = {lang for m in left for lang in m.sources}
    right_langs = {lang for m in right for lang in m.sources}
    shared = left_langs & right_langs
    return {lang for lang in shared if joiners.get(lang)}


def _weighted_joiner_choice(
    pool: list[tuple[str, int]],
    rng,
) -> str:
    """Weighted draw from a joiner pool ``[(form, weight), ...]``.
    Falls back to uniform when total weight is zero. Raises
    ``ValueError`` on an empty pool — every call site must filter
    empty pools out via ``_shared_lang_fields_with_joiners``, but the
    helper guards explicitly in case a future caller bypasses the
    filter."""
    if not pool:
        raise ValueError("cannot draw from an empty joiner pool")
    total = sum(max(0, w) for _, w in pool)
    if total <= 0:
        return rng.choice([form for form, _ in pool])
    threshold = rng.uniform(0, total)
    running = 0
    for form, weight in pool:
        if weight <= 0:
            continue
        running += weight
        if threshold < running:
            return form
    return pool[-1][0]


def _word_elements(new_name, wi: int, word) -> list[tuple[str, list[Meaning], str]]:
    """Build the ``(native_surface, meanings, modern_surface)`` tuples for one
    word's picked morphemes, skipping empty (``None``) slots.

    The native surface prefers the rendered (era / native / substituted) form;
    the modern surface mirrors :meth:`NewName.modern_name` — the diversified
    cross-language synonym's own usage when one overrode the slot, else the
    modern bucket key. Both have dash markers stripped (D45). Pure given
    ``new_name`` — no RNG — so the joiner draw order in
    :func:`_apply_joiner_insertion` is unaffected.
    """
    elements: list[tuple[str, list[Meaning], str]] = []
    for ei, e in enumerate(word):
        if e is None:
            continue
        if new_name.rendered is not None and new_name.rendered[wi][ei] is not None:
            surface = new_name.rendered[wi][ei]
        else:
            surface = e.replace("-", "")
        override = new_name._lang_override[wi][ei] if new_name._lang_override else None
        modern_surface = (override.usage if override is not None else e).replace("-", "")
        meanings = new_name.meaning_db[e]
        elements.append((surface, meanings, modern_surface))
    return elements


def _apply_joiner_insertion(
    new_name,
    joiners: dict[str, list[tuple[str, int]]],
    rng,
    density: float,
) -> tuple[str, str, str, list[dict[str, Any]]]:
    """Walk a NewName's per-word morpheme structure and insert joiners
    between adjacent picked morphemes whose lang_fields share a
    populated joiner pool.

    Returns ``(surface_str, modern_str, explanation, components)`` rebuilt
    to incorporate the inserted joiners. wyrd-24s6 (D41): both the native
    ``surface_str`` and the modern ``modern_str`` are built in this single
    walk so the SAME joiner (drawn once per gap) lands in BOTH renderings at
    the SAME position — otherwise the modern companion would lose the joiners
    the native canonical gained. Each inserted joiner appends a component dict
    with ``location='joiner'`` so the API envelope surfaces the breakdown to
    clients (the matcher-side ``Joiner`` sentinel from PR #130 is what
    KenningExplain uses for round-trip decomposition; this component is purely
    the structured-output annotation).

    Within-word only — no cross-word insertion (whitespace is the
    natural separator).
    """
    word_surfaces: list[str] = []
    modern_word_surfaces: list[str] = []
    inserted: list[tuple[str, str]] = []  # (joiner_surface, lang_field)
    for wi, word in enumerate(new_name.name):
        elements = _word_elements(new_name, wi, word)
        word_parts: list[str] = []
        modern_parts: list[str] = []
        for ei in range(len(elements)):
            word_parts.append(elements[ei][0])
            modern_parts.append(elements[ei][2])
            if ei < len(elements) - 1:
                shared = _shared_lang_fields_with_joiners(
                    elements[ei][1], elements[ei + 1][1], joiners
                )
                if shared and rng.random() < density:
                    # Deterministic lang pick — sorted set, then
                    # weighted draw within the lang's pool.
                    lang = rng.choice(sorted(shared))
                    joiner_surface = _weighted_joiner_choice(joiners[lang], rng)
                    word_parts.append(joiner_surface)
                    modern_parts.append(joiner_surface)
                    inserted.append((joiner_surface, lang))
        word_surfaces.append("".join(word_parts))
        modern_word_surfaces.append("".join(modern_parts))

    surface_str = " ".join(word_surfaces).strip()
    modern_str = " ".join(modern_word_surfaces).strip()
    base_explanation = new_name.description()
    base_components = new_name.components()
    if not inserted:
        return surface_str, modern_str, base_explanation, list(base_components)

    new_components = list(base_components)
    for surface, lang in inserted:
        new_components.append(
            {
                "usage": surface,
                "location": "joiner",
                "meanings": [f"phonological joiner ({lang})"],
                "tags": ["joiner"],
                "roots": [],
                "citations": [],
            }
        )
    joiner_descs = [f"+joiner: {surface} ({lang})" for surface, lang in inserted]
    new_explanation = f"{base_explanation} {' '.join(joiner_descs)}"
    return surface_str, modern_str, new_explanation, new_components


def _build_explanation_part(chunk, meaning_db: dict[str, list[Meaning]]) -> str:
    if isinstance(chunk, Meaning):
        # wyrd-ywm9 + wyrd-emlb: rank siblings so the historical
        # etymon's senses + roots appear first (was: insertion order).
        siblings = _rank_siblings(meaning_db.get(chunk.usage, [chunk]))
        fragment = chunk.usage.replace("-", "")
        roots = _all_roots(siblings)
        roots_str = "/".join(roots) if roots else "?"
        # wyrd-o53o + wyrd-aizb: prefer semantic senses; only fall
        # back to derivative pointers ('alternative form of X',
        # 'plural of X', 'soft mutation of X') when no semantic gloss
        # exists for any sibling. Pre-fix, derivative glosses
        # dominated the text explanation; post-fix they only appear
        # when there's no semantic alternative.
        primary, _ = _split_senses_for_display(_all_senses(siblings))
        senses = " / ".join(primary)
        return f'{fragment} "{senses}" ({roots_str})'
    return f"[{chunk}]"


def _build_component_part(chunk, meaning_db: dict[str, list[Meaning]]) -> dict[str, Any]:
    if isinstance(chunk, Meaning):
        # wyrd-ywm9 + wyrd-emlb: rank siblings so the historical
        # etymon dominates the morpheme-card output. Drops modern_
        # english-only homographs when an older etymon exists, then
        # orders by signal density (OE > ON > OF > Celtic > …).
        siblings = _rank_siblings(meaning_db.get(chunk.usage, [chunk]))
        tags = list(dict.fromkeys(tag for m in siblings for tag in m.tags))
        # wyrd-o53o + wyrd-aizb: split semantic vs derivative glosses.
        # 'meanings' carries the primary set (semantic when any exist;
        # otherwise the derivatives so the card isn't blank). The
        # new 'derivative_meanings' field carries the suppressed
        # morphological pointers — the SPA may show them dimmed /
        # secondary, or omit them. Empty list when no derivatives
        # were suppressed (so the SPA can render the absence cleanly).
        primary, derivative = _split_senses_for_display(_all_senses(siblings))
        return {
            "type": "matched",
            "fragment": chunk.usage.replace("-", ""),
            "usage": chunk.usage,
            "location": chunk.location,
            "meanings": primary,
            "derivative_meanings": derivative,
            "tags": tags,
            "roots": _all_roots(siblings),
        }
    return {"type": "unaccounted", "fragment": chunk}


def _decomposition_signature(words: tuple[Word, ...]) -> tuple:
    """Structural fingerprint: usages and unaccounted strings, in order. Used
    to dedupe decompositions that differ only by which Meaning instance was
    selected for a usage with multiple senses."""
    sig: list[tuple[str, str]] = []
    for word in words:
        for chunk in word.word:
            if isinstance(chunk, Meaning):
                sig.append(("M", chunk.usage))
            elif is_connective(chunk):
                # wyrd-aicu.9 (D-1 defensive): a connective (genitive ``-s-``
                # glue) is position-neutral + non-content and contributes nothing
                # to the structural signature — skip it explicitly so the
                # signature stays consistent with ``_decomposition_payload``.
                # Under D-1 no connective reaches this path; defensive only.
                continue
            elif isinstance(chunk, str) and chunk:
                sig.append(("S", chunk))
        sig.append(("|", ""))
    return tuple(sig)


def _build_decomposition_result(
    name_str: str,
    words,
    meaning_db: dict[str, list[Meaning]],
    *,
    canonical: bool = False,
    canonical_source: str | None = None,
) -> GenerationResult:
    explanation_parts: list[str] = []
    components: list[dict[str, Any]] = []
    for word in words:
        word_parts: list[dict[str, Any]] = []
        for chunk in word.word:
            if isinstance(chunk, str) and not chunk:
                continue
            explanation_parts.append(_build_explanation_part(chunk, meaning_db))
            word_parts.append(_build_component_part(chunk, meaning_db))
        components.append({"word": str(word), "parts": word_parts})
    return GenerationResult(
        result=name_str,
        explanation=" + ".join(explanation_parts) or "no morphemes recognized",
        components=components,
        canonical=canonical,
        canonical_source=canonical_source,
    )


def _canonical_signature_for_words(words) -> str:
    """Compute the SHA-1-of-JSON-payload signature used by
    runtime/decomposition.py over a flat slot list across the words of a name.

    The canonical map projected into the bundle (wyrd-h8k1) keys on
    this signature; KenningExplain re-derives it per candidate so the
    matching reading can be marked canonical and front-loaded.

    Slot order mirrors ``decomposition._cross_product_decompositions``
    (concat each Word.word list in word-iteration order, no filtering)
    so populator + runtime payloads byte-match for the same parse.
    Filtering empty-string slots here would diverge from the populator
    and silently miss canonicals on edge-case Word.word lists.
    """
    flat: list = []
    for word in words:
        flat.extend(word.word)
    return _signature_for_payload(_decomposition_payload(flat))


def _bundle_era_form(meaning: Meaning, target_language: str | None) -> str:
    """Pick a target-language reflex from the meaning's bundle data.

    Preference: case-insensitive match for the morpheme's modern
    canonical (matches the CLI rewinder's tier-1 picker rule), else
    the alphabetically-first reflex, else the modern canonical
    fallback.
    """
    canonical = meaning.usage.replace("-", "")
    if target_language is None:
        return canonical
    reflexes = meaning.era_reflex_for(target_language)
    if not reflexes:
        return canonical
    canonical_lower = canonical.lower()
    matching = [r for r in reflexes if r.lower() == canonical_lower]
    return matching[0] if matching else reflexes[0]


def _format_creature_explanation(entry: dict[str, Any]) -> str:
    """Render a creature scorecard as a human-readable explanation
    line. Pulled out so tests can pin the exact format without
    re-instantiating the Generator."""
    parts: list[str] = []
    head = f"{entry['input_name']} ← {entry['language']} {entry['canonical_form']}"
    if entry.get("english_shaped"):
        head += f" ({entry['english_shaped']})"
    parts.append(head)
    if entry.get("glosses"):
        glosses = entry["glosses"][:3]
        parts.append("glosses: " + " / ".join(glosses))
    if entry.get("citation"):
        parts.append(f"source: {entry['citation']}")
    explanation = ". ".join(parts) + "."
    era_lines: list[str] = []
    for lang, refs in entry.get("era_reflexes", {}).items():
        forms = ", ".join(form for form, _source in refs)
        era_lines.append(f"{lang}: {forms}")
    if era_lines:
        explanation += " Era reflexes — " + "; ".join(era_lines) + "."
    return explanation


# wyrd-o9qi: Generator subclasses live in the generators/ subpackage. Import
# them HERE (at the end of this module) so all the underscore-prefixed
# helpers referenced by class method bodies are already defined in the
# module namespace by the time generators.* is imported — its modules
# `from wyrd.generators.kenning import _foo` and would otherwise hit
# partial-init AttributeError on names defined later in this file.
from wyrd.generators.kenning.generators import (  # noqa: E402
    Kenning,
    KenningCreature,
    KenningEraMap,
    KenningExplain,
    KenningRegenerateMorpheme,
    KenningRender,
    KenningRewind,
)


def _kenning_runtime_version(self) -> dict[str, str]:
    """Bundle-identity stamp for the API envelope + defect reports
    (wyrd-dsl5). Every kenning generator reads the same L4 bundle, so they
    all report the same version — bound onto each class here rather than
    duplicated as a method body in seven generator modules. Matches this
    module's existing pattern of binding shared behavior post-import (see
    ``_load_meanings.cache_clear`` above)."""
    from wyrd.generators.kenning.runtime.runtime_db import bundle_version

    return bundle_version()


for _kenning_cls in (
    Kenning,
    KenningExplain,
    KenningRewind,
    KenningRender,
    KenningEraMap,
    KenningCreature,
    KenningRegenerateMorpheme,
):
    _kenning_cls.runtime_version = _kenning_runtime_version  # type: ignore[method-assign]

register(Kenning())
register(KenningExplain())
register(KenningRewind())
register(KenningRender())
register(KenningEraMap())
register(KenningCreature())
# No standalone CLI subcommand — single-morpheme regeneration is the SPA's
# direct-manipulation ⟳ button (wyrd-y0lx); a batch/offline CLI shape has no
# meaningful use (the input is an in-flight SPA pipeline state).
register(KenningRegenerateMorpheme())
