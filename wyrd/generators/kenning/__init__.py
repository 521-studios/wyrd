"""Kenning — town-name generator. Compounds Old English / Norse / Celtic morphemes."""

from __future__ import annotations

import itertools
import json
from functools import lru_cache
from importlib import resources
from typing import Any

from wyrd.generators.kenning.decomposition import (
    _decomposition_payload,
    _signature_for_payload,
)
from wyrd.generators.kenning.era import era_cells_for_family, resolve_era_input
from wyrd.generators.kenning.meaning import (
    Meaning,
    load_canonical_decompositions,
    load_fantasy_morphemes,
    load_joiners,
    load_meanings,
)
from wyrd.generators.kenning.name import Name
from wyrd.generators.kenning.proportions import load_proportions
from wyrd.generators.kenning.strata import (
    ALL_STRATA,
    FRENCH_STRATA,
    OLD_ENGLISH_STRATA,
    OLD_NORSE_STRATA,
    WELSH_STRATA,
    valid_strata_for_culture,
)
from wyrd.generators.kenning.word import Word
from wyrd.registry import GenerationResult, Generator, register
from wyrd.seed import rng_for

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


def _era_options_by_culture() -> dict[str, list[str]]:
    """Per-culture list of era cell labels for the SPA's dependent select.

    Empty string is prepended as the 'no era filter' option. The label
    set is derived from era_cells_for_family so adding a new cell to
    era.py automatically surfaces in the dropdown without touching this
    file. Re-evaluated at schema-render time, so era.ERA_CELLS edits
    take effect after a manifest refresh.
    """
    return {
        culture: ["", *era_cells_for_family(family)]
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
    _FICTION_TAG,
}

# D6 mood presets. A "mood" bundles one or more effects (semantic tag union,
# phonological harshness skew, future axes) under a single GM-facing label so
# 'I want grim names' is one decision rather than separate dials. Each entry
# may carry a "tags" tuple (semantic-tag union) and/or a "harshness" float
# (D6 phonological skew default when the mood is requested with no value).
#
# CLI surface: `--mood grim`, `--mood harsh`, `--mood harsh:0.5` (colon-suffix
# overrides the recipe default for graduated moods like harshness). Multiple
# `--mood` flags compose by tag-union and by max-harshness.
#
# The 'grim' tag set substitutes the original D6 spec names ('grim',
# 'mortuary', 'monstrous', 'battle', 'wilderness') — none of those exist in
# the bundle yet; the closest extant tags fill in (death=18 subjects,
# military=27, monster=8, undead=9, magic=4). Adding the spec-named tags
# later folds in without breaking callers.
MOODS: dict[str, dict[str, Any]] = {
    "grim": {"tags": ("death", "military", "monster", "undead", "magic")},
    "harsh": {"harshness": 1.0},
    # Picked from the 2026-05-02 bundle audit (≥5 subjects per tag, distinct
    # semantic identity, minimal overlap with existing moods). 'noble' was
    # considered but no 'royalty' tag exists yet — defer until mining
    # surfaces it. 'ominous' was too thin (magic=4 below threshold).
    "pastoral": {"tags": ("plant", "animal", "water", "agriculture", "tree", "bird")},
    "devotional": {"tags": ("saint", "religious")},
    "mortuary": {"tags": ("death", "undead")},
}


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
    with _data_path("meanings.json").open() as f:
        data = json.load(f)
    with _data_path("irish_anglicizations.json").open() as f:
        sidecar = json.load(f)
    manorial = _norman_manorial_subjects()
    # The bundle may be list-shape (legacy) or dict-shape
    # ``{"subjects": [...], "joiners": ..., "canonical_decompositions": ...}``
    # (wyrd-q0g6 / wyrd-h8k1). Sidecars extend the subjects list either way;
    # dict-shape callers preserve the bundle keys for downstream loaders
    # (joiners, canonical_decompositions) to read.
    if isinstance(data, dict):
        subjects = list(data.get("subjects") or [])
        subjects.extend(sidecar)
        subjects.extend(manorial)
        data["subjects"] = subjects
    else:
        data.extend(sidecar)
        data.extend(manorial)
    return load_meanings(data)


@lru_cache(maxsize=1)
def _load_joiners() -> dict[str, list[tuple[str, int]]]:
    """Load the bundle's joiner pool. Returns
    ``{lang_field: [(form, weight), ...]}``; empty for legacy
    list-shape bundles."""
    with _data_path("meanings.json").open() as f:
        data = json.load(f)
    return load_joiners(data)


@lru_cache(maxsize=1)
def _load_canonical_decompositions() -> dict[str, dict[str, str]]:
    """Load the bundle's per-toponym canonical decomposition map
    (wyrd-h8k1). Returns ``{modern_name: {"signature", "source"}}``;
    empty for legacy list-shape bundles AND for dict-shape bundles
    that don't carry a ``canonical_decompositions`` field."""
    with _data_path("meanings.json").open() as f:
        data = json.load(f)
    return load_canonical_decompositions(data)


@lru_cache(maxsize=1)
def _load_fantasy_morphemes() -> dict[str, dict]:
    """wyrd-vz7f: load the bundle's fantasy-creature etymology map.
    Returns ``{lowercase_input_name: {input_name, etymon_id, language,
    canonical_form, english_shaped, glosses, citation, era_reflexes}}``.
    Empty for bundles that pre-date wyrd-vz7f or for DBs whose
    ``fantasy_morpheme`` table is empty."""
    with _data_path("meanings.json").open() as f:
        data = json.load(f)
    return load_fantasy_morphemes(data)


@lru_cache(maxsize=len(CULTURES))
def _load_culture(culture: str):
    if culture not in CULTURES:
        raise ValueError(f"unknown culture: {culture}; expected one of {CULTURES}")
    meaning_db, tag_db = _load_meanings()
    with _data_path(f"{culture}_proportions.json").open() as f:
        proportions = json.load(f)
    return load_proportions(proportions, meaning_db, tag_db), tag_db


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
    ``_load_culture``, ``_load_joiners``, ``_load_canonical_decompositions``.

    These caches form an aggregate over a single ``meanings.json``
    read: ``_load_culture`` holds a ``NameGenerator`` parameterised on
    the ``meaning_db`` from ``_load_meanings``; ``_load_joiners`` and
    ``_load_canonical_decompositions`` read the same bundle file.
    Invalidating one without the others yields a stale view next
    time. Replaces ``_load_meanings.cache_clear`` so the standard
    test-side pattern (``_load_meanings.cache_clear()``) clears all.

    The reverse direction (``_load_culture.cache_clear()`` alone) is
    intentionally uncoupled — a caller who only wants to drop the
    per-culture generator shouldn't pay to re-parse meanings.json.
    """
    _original_load_meanings_cache_clear()
    _load_culture.cache_clear()
    _load_joiners.cache_clear()
    _load_canonical_decompositions.cache_clear()
    _load_fantasy_morphemes.cache_clear()


# mypy flags reassigning a bound method on the lru_cache wrapper as
# method-assign; intentional here — the patch is the whole point.
_load_meanings.cache_clear = _coupled_cache_clear  # type: ignore[method-assign]


def available_tags() -> list[str]:
    """User-visible tags from the meaning DB (excludes internal filtering tags)."""
    _, tag_db = _load_meanings()
    return sorted(t for t in tag_db if t not in _INTERNAL_TAGS)


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
        return resolve_era_input(era, default_family=era_family)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"invalid 'era' value {era!r} for culture {culture!r} "
            f"(era family {era_family!r}): {exc}. Pass a year (e.g. "
            f"1086), a cell label defined in the culture's era family, "
            f"or an explicit 'family/label' pair."
        ) from None


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


def _apply_mood(spec: str, tags: list[str], harshness: float) -> tuple[list[str], float]:
    """Resolve one mood spec ('grim' or 'harsh:0.5') into tag and harshness
    contributions, returning the updated tuple. Multiple moods compose by
    tag-union and max-harshness — repeated mood specs are idempotent on
    tags and only ratchet harshness up.
    """
    if ":" in spec:
        name, value = spec.split(":", 1)
    else:
        name, value = spec, None
    if name not in MOODS:
        raise ValueError(f"unknown mood {name!r}; expected one of {sorted(MOODS)}")
    recipe = MOODS[name]
    new_tags = list(tags)
    for t in recipe.get("tags", ()):
        if t not in new_tags:
            new_tags.append(t)
    if "harshness" in recipe:
        v = float(value) if value is not None else recipe["harshness"]
        harshness = max(harshness, v)
    return new_tags, harshness


class Kenning(Generator):
    name = "kenning"
    display_name = "Kenning — Town Names"
    description = (
        "Generates British Isles–style town names by composing Old English, Old Norse, "
        "Old French, and Celtic morphemes. Pick a culture; optionally filter morphemes "
        "by tag (e.g. 'tree', 'water', 'religion')."
    )
    details = (
        "<p>"
        "Town names from the British Isles aren't arbitrary — they're stitched from "
        "old <strong>morphemes</strong>, the small meaning-bearing fragments inside "
        "a word. Place-name scholars call the names themselves <strong>toponyms</strong>; "
        'the morphemes are their building blocks. <em>Ashton</em> is "ash" + "-ton" '
        '(Old English for an enclosed settlement). <em>Bridgwater</em> is "bridge" + '
        '"water". The vocabulary is bounded; the combinations are nearly endless.'
        "</p>"
        "<p>"
        "Kenning learned the patterns by analyzing roughly "
        "<strong>66,000 real British Isles place names</strong> "
        "(English, Scottish, Welsh, Irish) against a corpus of about "
        "2,900 morphemes. For each culture it knows which morphemes show up, "
        "in which structures (prefix + root, two words, saint's name, etc.), "
        "and how often. Each rolled name is a fresh sample from those statistics."
        "</p>"
    )
    legend = _LEGEND

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "culture": {
                    "type": "string",
                    "enum": CULTURES,
                    "default": "english",
                    "description": "Linguistic culture to draw morphemes and structures from.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "enum": available_tags()},
                    "default": [],
                    "description": (
                        "Optional tag filters. Each tag biases the name toward morphemes "
                        "with that meaning category."
                    ),
                },
                "count": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                    "description": "How many names to generate (1–10).",
                },
                "seed": {
                    "type": "integer",
                    "description": "Optional 64-bit seed for reproducible output.",
                },
                "spelling_variety": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Per-morpheme probability of substituting an attested archaic "
                        "spelling variant for the canonical reflex (D18). 0 keeps the "
                        "modern surface form; higher values mix in 19th-century "
                        "scholarly spellings (e.g. 'Brycg' for 'Bridg-') for archaic "
                        "feel. Variant pool is empty for most morphemes today, so the "
                        "knob has limited reach until more mining lands."
                    ),
                },
                "novelty": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Mixture between empirical-frequency sampling and a uniform "
                        "marginal (D17). 0 keeps today's bit-stable behavior, 1 makes "
                        "every in-bucket morpheme equally likely — plausible-but-"
                        "unattested combinations become possible without abandoning "
                        "the corpus. Intermediate values softly blend."
                    ),
                },
                "inflection_density": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Per-morpheme probability of substituting an inflected form "
                        "(genitive, dative, plural) for the lemma (D8). 0 always uses "
                        "the unmarked headword; higher values surface morphological "
                        "variety like 'Cotum-' instead of 'Cot-'. Inflection wins over "
                        "spelling_variety when both knobs would fire on the same "
                        "morpheme."
                    ),
                },
                "mood": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": (
                        "D6 stylistic-mood presets (repeatable). Each entry is one of "
                        f"{sorted(MOODS)!r}, optionally with a colon-suffix value "
                        "(e.g. 'harsh:0.5' for graduated phonological skew). 'grim' "
                        "applies a menacing semantic-tag union; 'harsh' biases sampling "
                        "toward stop-final / cluster-heavy morphemes. Multiple moods "
                        "compose: tags union, harshness takes the max."
                    ),
                },
                "include_fiction": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "wyrd-yan: when True, allow morphemes tagged 'fiction' "
                        "(constructed etymologies for bestiary / NPC / homebrew "
                        "content) to appear in generated names. Default False keeps "
                        "realistic-mode generation drawing only from scholarly-attested "
                        "morphemes. The bundle today carries no fiction-tagged data — "
                        "the gate is in place for upcoming constructed-etymology "
                        "pipelines (wyrd-0ab, wyrd-kjc)."
                    ),
                },
                "harshness": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "D6 phonological-harshness skew (0..1). Power-user knob; for "
                        "GM-facing usage prefer 'mood: [harsh]' or 'mood: [\"harsh:0.5\"]'. "
                        "0 leaves sampling unchanged; 1 drops soft morphemes and gives "
                        "stop-final / cluster-heavy ones 2x weight. The mood resolution "
                        "uses max(harshness, mood-derived) so explicit harshness takes "
                        "effect when it exceeds the mood preset."
                    ),
                },
                "era": {
                    "type": "string",
                    "default": "",
                    # wyrd-awo: dependent-select metadata read by the SPA.
                    # Each culture surfaces only the cell labels defined in
                    # its era family — picking 'oe-late' while culture is
                    # 'irish' would 4xx at runtime, so the dropdown
                    # filters to the family's labels to prevent it.
                    # CLI/API still accept bare-year and 'family/label'
                    # shapes; this property only constrains the SPA UX.
                    "x-options-by-culture": _era_options_by_culture(),
                    "description": (
                        "D5-2 era filter (wyrd-lyp). Restricts the morpheme inventory "
                        "to forms attested in a particular period. The SPA renders this "
                        "as a dropdown filtered to the chosen culture's era family. "
                        "CLI/API also accept a bare year (e.g. '1086' → the cell "
                        "containing 1086 in the culture's era family) or an explicit "
                        "'family/label' pair (e.g. 'english/oe-late') to disambiguate "
                        "when a label is shared across families. Morphemes with no "
                        "attested-year evidence pass through unconditionally — only "
                        "~32% of bundle morphemes carry year data today, so the filter "
                        "narrows the pool rather than gutting it."
                    ),
                },
                "cohesion": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "wyrd-mj2 tag co-occurrence bias (0..1). 0 leaves each slot "
                        "sampling independently from its marginal (today's behavior). "
                        "Higher values bias each slot's pick toward usages whose tags "
                        "co-occur with previously-picked slots' tags in the empirical "
                        "corpus — so 'topography + plant' and 'water + plant' (both "
                        "common) are preferred over 'religion + plant' (rare). Composes "
                        "orthogonally with novelty: cohesion pulls toward attested "
                        "tag-class pairings, novelty blends toward the uniform marginal."
                    ),
                },
                "stratum": {
                    "type": "string",
                    "default": "",
                    # wyrd-j3gy: dependent-select metadata read by the
                    # SPA. Each culture surfaces only the stratum tags
                    # valid for that culture's bundled language families
                    # (e.g. picking 'east-norse' against a Welsh culture
                    # would 4xx at runtime, so the dropdown filters to
                    # the culture's allowed set). CLI/API still accept
                    # any string the per-culture validator accepts;
                    # this property only constrains the SPA UX. Same
                    # shape + load-bearing semantics as the era
                    # property's x-options-by-culture (wyrd-awo).
                    "x-options-by-culture": _stratum_options_by_culture(),
                    "description": (
                        "wyrd-lr4 Phase 3 within-language stratum filter. Restricts the "
                        "morpheme inventory to forms classified into a specific register "
                        f"bucket — for Welsh: {', '.join(repr(s) for s in WELSH_STRATA)}; "
                        f"for French: {', '.join(repr(s) for s in FRENCH_STRATA)}; for "
                        f"Old English: {', '.join(repr(s) for s in OLD_ENGLISH_STRATA)}; "
                        f"for Old Norse: {', '.join(repr(s) for s in OLD_NORSE_STRATA)}. "
                        "The SPA renders this as a dropdown filtered to the chosen "
                        "culture's allowed strata (wyrd-j3gy). CLI/API also accept any "
                        "stratum tag valid for the culture's bundled language families. "
                        "Rejects culturally-incoherent strata (e.g. east-norse on welsh) "
                        "at request time, not just typos. Morphemes with no stratum data "
                        "pass through (Welsh / French / Old English / Old Norse families "
                        "are classified today). Composes with --era via intersection. "
                        "Empty disables the filter — bit-stable behavior."
                    ),
                },
                "manorial_affix": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "wyrd-obu Norman manorial-affix layering (0..1). "
                        "Probability that a generated name gets an Anglo-Norman "
                        "family surname appended (Stoke Mandeville, Ashby de la "
                        "Zouch, Stanton Lacy). Encodes the post-Conquest political "
                        "history layered onto English place-naming. At 0 (default) "
                        "no affix is attached; at 0.5 about half of generated "
                        "names get one; at 1 every name gets one. Only applies "
                        "to the english culture today — Domesday-and-after "
                        "manorial layering is an English place-naming pattern. "
                        "Affix corpus is a curated set of 39 attested Norman "
                        "families (Domesday + post-Conquest subsidy rolls)."
                    ),
                },
                "joiner_density": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Probability (0..1) of inserting a phonological "
                        "joiner between adjacent morphemes that share a "
                        "language family. At 0 (default) no joiners are "
                        "inserted. Bundle currently ships no populated "
                        "joiner pool, so this is a no-op until a future "
                        "data update."
                    ),
                },
            },
            "required": [],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        culture = params.get("culture", "english")
        raw_tags = params.get("tags", []) or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        tags = list(raw_tags)
        spelling_variety = float(params.get("spelling_variety", 0.0) or 0.0)
        novelty = float(params.get("novelty", 0.0) or 0.0)
        inflection_density = float(params.get("inflection_density", 0.0) or 0.0)
        harshness = float(params.get("harshness", 0.0) or 0.0)
        cohesion = float(params.get("cohesion", 0.0) or 0.0)
        manorial_affix = float(params.get("manorial_affix", 0.0) or 0.0)
        joiner_density = float(params.get("joiner_density", 0.0) or 0.0)
        include_fiction = _coerce_bool(params.get("include_fiction", False))

        moods = params.get("mood", []) or []
        if isinstance(moods, str):
            moods = [moods]
        for spec in moods:
            tags, harshness = _apply_mood(spec, tags, harshness)
        tags = tuple(tags)
        exclude_tags: tuple[str, ...] = () if include_fiction else (_FICTION_TAG,)

        era_range = _resolve_era_param(params.get("era"), culture)
        # wyrd-j3gy: _resolve_stratum_param validates against the
        # per-culture allowed-set (with ALL_STRATA fallback for
        # cultures without classifiers yet). A typo'd --stratum
        # surfaces as a clean ValueError rather than silently
        # no-opping.
        stratum = _resolve_stratum_param(params.get("stratum"), culture)

        name_gen, _ = _load_culture(culture)
        rng = rng_for(seed)
        new_name = name_gen.select(
            rng,
            *tags,
            spelling_variety=spelling_variety,
            novelty=novelty,
            inflection_density=inflection_density,
            harshness=harshness,
            exclude_tags=exclude_tags,
            era_range=era_range,
            stratum=stratum,
            cohesion=cohesion,
        )
        result_str = str(new_name)
        explanation = new_name.description()
        components = new_name.components()
        # wyrd-q0g6 Phase 1.5: compose-time joiner insertion. Gated on
        # density>0 + non-empty pool so legacy callers stay bit-stable.
        if joiner_density > 0:
            joiners = _load_joiners()
            if joiners:
                result_str, explanation, components = _apply_joiner_insertion(
                    new_name, joiners, rng, joiner_density
                )
        # wyrd-obu: optional Norman manorial-family affix appended after
        # the morpheme-compounded base name. English-culture only (the
        # post-Conquest manorial-layering pattern is an English place-
        # naming convention; pasting Norman affixes onto Welsh / Irish
        # / Breton bases would be cosmetically jarring and historically
        # wrong). Probability-gated so a region can have a few
        # manorialized names mixed with non-affixed neighbors, which is
        # how the historical pattern actually surfaces.
        if manorial_affix > 0 and culture == "english" and rng.random() < manorial_affix:
            family = rng.choice(_load_norman_manorial_families())
            result_str = f"{result_str} {family}"
            explanation = f"{explanation} + manorial: {family} (Norman family)"
            components.append(
                {
                    "usage": family,
                    "location": "manorial-affix",
                    "meanings": [f"Norman manorial family: {family}"],
                    "tags": ["manorial", "norman"],
                    "roots": ["FR"],
                    "citations": [],
                }
            )
        return GenerationResult(
            result=result_str,
            explanation=explanation,
            components=components,
        )


register(Kenning())


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


def _apply_joiner_insertion(
    new_name,
    joiners: dict[str, list[tuple[str, int]]],
    rng,
    density: float,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Walk a NewName's per-word morpheme structure and insert joiners
    between adjacent picked morphemes whose lang_fields share a
    populated joiner pool.

    Returns ``(surface_str, explanation, components)`` rebuilt to
    incorporate the inserted joiners. Each inserted joiner appends a
    component dict with ``location='joiner'`` so the API envelope
    surfaces the breakdown to clients (the matcher-side ``Joiner``
    sentinel from PR #130 is what KenningExplain uses for round-trip
    decomposition; this component is purely the structured-output
    annotation).

    Within-word only — no cross-word insertion (whitespace is the
    natural separator).
    """
    word_surfaces: list[str] = []
    inserted: list[tuple[str, str]] = []  # (joiner_surface, lang_field)
    for wi, word in enumerate(new_name.name):
        elements: list[tuple[str, list[Meaning]]] = []
        for ei, e in enumerate(word):
            if e is None:
                continue
            if new_name.rendered is not None and new_name.rendered[wi][ei] is not None:
                surface = new_name.rendered[wi][ei]
            else:
                surface = e.replace("-", "")
            meanings = new_name.meaning_db[e]
            elements.append((surface, meanings))
        word_parts: list[str] = []
        for ei in range(len(elements)):
            word_parts.append(elements[ei][0])
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
                    inserted.append((joiner_surface, lang))
        word_surfaces.append("".join(word_parts))

    surface_str = " ".join(word_surfaces).strip()
    base_explanation = new_name.description()
    base_components = new_name.components()
    if not inserted:
        return surface_str, base_explanation, list(base_components)

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
    return surface_str, new_explanation, new_components


def _build_explanation_part(chunk, meaning_db: dict[str, list[Meaning]]) -> str:
    if isinstance(chunk, Meaning):
        siblings = meaning_db.get(chunk.usage, [chunk])
        fragment = chunk.usage.replace("-", "")
        roots = _all_roots(siblings)
        roots_str = "/".join(roots) if roots else "?"
        senses = " / ".join(_all_senses(siblings))
        return f'{fragment} "{senses}" ({roots_str})'
    return f"[{chunk}]"


def _build_component_part(chunk, meaning_db: dict[str, list[Meaning]]) -> dict[str, Any]:
    if isinstance(chunk, Meaning):
        siblings = meaning_db.get(chunk.usage, [chunk])
        tags = list(dict.fromkeys(tag for m in siblings for tag in m.tags))
        return {
            "type": "matched",
            "fragment": chunk.usage.replace("-", ""),
            "usage": chunk.usage,
            "location": chunk.location,
            "meanings": _all_senses(siblings),
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
    decomposition.py over a flat slot list across the words of a name.

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


class KenningExplain(Generator):
    # Co-located in the `kenning` package rather than `wyrd/generators/kenning_explain/`
    # because the explainer shares the meanings DB and decomposition machinery with
    # the main Kenning generator. The SPA path / API name `kenning-explain` does not
    # match a package directory, so `wyrd/cli.py:_mount_generator_clis()` cannot
    # locate a matching `cli.py` for it. This is intentional: the `explain`
    # subcommand on `wyrd kenning ...` is the CLI surface; the silent miss in the
    # mounter is the documented trade-off for sharing data with `Kenning`.
    name = "kenning-explain"
    display_name = "Kenning — Explain a Name"
    description = (
        "Decompose a real or invented British Isles place name into the morphemes "
        "Kenning recognizes. Returns every matching reading; unrecognized fragments "
        "are flagged."
    )
    details = (
        "<p>"
        "Paste any town name from the British Isles — or invent your own — and "
        "Kenning will break it apart into the <strong>morphemes</strong> it "
        "recognizes: small meaning-bearing fragments like <em>aber-</em>, "
        "<em>-ton</em>, <em>-combe</em>. Multiple readings? You'll see them all."
        "</p>"
        "<p>"
        "Pieces flagged as <strong>unrecognized</strong> are gaps — fragments "
        "that real names use but our morpheme corpus doesn't yet know. They're "
        "a research target for expanding the dataset."
        "</p>"
    )
    legend = _LEGEND
    multi_result = True

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Town name to decompose, e.g. 'Bridgwater'.",
                },
            },
            "required": ["name"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        # multi_result generators are dispatched through generate_all; this
        # exists only to satisfy the abstract method.
        results = self.generate_all(params, seed)
        return results[0]

    def generate_all(self, params: dict[str, Any], seed: int) -> list[GenerationResult]:
        text = (params.get("name") or "").strip()
        if not text:
            raise ValueError("name is required")
        meaning_db, _ = _load_meanings()
        name_obj = Name(text)
        # reduce=False keeps every alternative decomposition instead of
        # collapsing to the "best" one.
        name_obj.find_meaning(meaning_db, reduce=False)
        per_word = [name_obj.words[word] for word in text.split()]
        if not per_word:
            return [GenerationResult(result=text, explanation="no morphemes recognized")]

        # wyrd-h8k1: when the bundle carries a canonical signature for
        # this toponym, the matching reading floats to the top of the
        # candidates list and gets marked ``canonical=True`` so the SPA
        # can render it distinctly. Bundle map is empty for legacy
        # list-shape bundles + dict-shape bundles missing the
        # ``canonical_decompositions`` field — this codepath is
        # transparent to legacy data.
        canonical_map = _load_canonical_decompositions()
        canonical_entry = canonical_map.get(text)
        canonical_signature = canonical_entry["signature"] if canonical_entry else None
        canonical_source = canonical_entry["source"] if canonical_entry else None

        # Dedupe by structural signature: when one usage has multiple Meanings
        # (different senses, e.g. -y as both "district" and "island"), the raw
        # cartesian product produces N copies of the same structural break with
        # only the Meaning identity differing. We collapse those into one
        # result and combine senses inside _build_explanation_part.
        seen: set[tuple] = set()
        candidates: list[tuple[int, int, int, tuple]] = []
        sig_to_words: dict[tuple, Any] = {}
        sig_is_canonical: dict[tuple, bool] = {}
        for words in itertools.product(*per_word):
            sig = _decomposition_signature(words)
            if sig in seen:
                continue
            seen.add(sig)
            sig_to_words[sig] = words
            unaccounted = sum(1 for w in words for c in w.word if isinstance(c, str) and c)
            total = sum(1 for w in words for c in w.word if not (isinstance(c, str) and not c))
            is_canonical = (
                canonical_signature is not None
                and _canonical_signature_for_words(words) == canonical_signature
            )
            sig_is_canonical[sig] = is_canonical
            # Canonical sorts first by carrying a leading-zero rank;
            # within-rank order falls back to the heuristic
            # (lowest unaccounted, then min-complexity).
            rank = 0 if is_canonical else 1
            candidates.append((rank, unaccounted, total, sig))

        # Best readings first: canonical (when present), then fewer
        # unaccounted fragments, then simpler.
        candidates.sort()
        return [
            _build_decomposition_result(
                text,
                sig_to_words[sig],
                meaning_db,
                canonical=sig_is_canonical[sig],
                canonical_source=canonical_source if sig_is_canonical[sig] else None,
            )
            for _, _, _, sig in candidates[:_MAX_DECOMPOSITIONS]
        ]


register(KenningExplain())


class KenningRewind(Generator):
    """wyrd-obpw Phase 3.3 — bundle-driven time-rewind explainer.

    Mirrors the CLI rewinder (``wyrd/generators/kenning/rewind.py``)
    but reads era data from the bundle (``Meaning.era_reflex_for``)
    instead of the lexicon DB. The Lambda has no DB access, so this
    Generator class is the SPA-renderable surface for the rewinder
    feature.

    Input schema: ``name`` (string, required) and an optional ``era``
    cell label (defaults to a 3-stop English ladder when absent).

    The decomposition path reuses the existing ``Name`` + meaning_db
    matcher; for each decomposed Meaning the generator picks an
    anchor source language (OE preferred, ON / Celtic / modern as
    fallbacks) and reads the cluster reflexes from
    ``meaning.era_reflex_for(target_language)``. Same anchor-resolver
    + tier preference rules as the CLI rewinder, just with a bundle-
    only data source.

    Out of scope (deferred):
    - Per-toponym attestation lookup (would need projecting the
      toponym_attestation table into the bundle too — substantial
      future work).
    - Picker tier-2 source-form preference (the bundle's per-target
      reflex list is alphabetical; the CLI rewinder's tier-2 rule
      doesn't translate cleanly without anchor-form metadata in the
      bundle).
    """

    name = "kenning-rewind"
    display_name = "Kenning — Time-Rewind a Name"
    description = (
        "Render a British place name as it might have looked at multiple "
        "historical eras: Old English (pre-1100), Middle English (1100-1500), "
        "and modern. Each morpheme of the input renders against its "
        "etymological cluster's surface forms attested at that period."
    )
    legend = _LEGEND
    multi_result = True

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Modern (or invented) British place name.",
                },
            },
            "required": ["name"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        results = self.generate_all(params, seed)
        return results[0]

    def generate_all(self, params: dict[str, Any], seed: int) -> list[GenerationResult]:
        from wyrd.generators.kenning.era import canonical_language_for_cell

        text = (params.get("name") or "").strip()
        if not text:
            raise ValueError("name is required")
        meaning_db, _ = _load_meanings()
        name_obj = Name(text)
        name_obj.find_meaning(meaning_db, reduce=True)
        era_stops = (
            ("english", "oe-late"),
            ("english", "me"),
            ("english", "modern"),
        )
        # Per-era rendered output. Each era-stop becomes one
        # GenerationResult so the SPA can render them as a sequence.
        outputs: list[GenerationResult] = []
        for family, cell in era_stops:
            target_language = canonical_language_for_cell(family, cell)
            rendered_morphemes: list[str] = []
            morpheme_components: list[dict[str, Any]] = []
            unaccounted: list[str] = []
            for word_str in text.split():
                candidates = name_obj.words.get(word_str, [])
                if not candidates:
                    unaccounted.append(word_str)
                    continue
                for chunk in candidates[0].word:
                    if isinstance(chunk, Meaning):
                        form = _bundle_era_form(chunk, target_language)
                        rendered_morphemes.append(form)
                        # wyrd-17t: surface SAMPA-lite respelling next to
                        # the rendered form when the target language has
                        # a respeller (OE / Welsh / ON / Latin / Greek /
                        # Norman-French). Modern-English passes through
                        # with respelling=None since users can already
                        # sound those out.
                        respelling = (
                            chunk.respelling_for(form, target_language) if target_language else None
                        )
                        morpheme_components.append(
                            {
                                "form": form,
                                "respelling": respelling,
                                "language": target_language,
                            }
                        )
                    elif isinstance(chunk, str) and chunk:
                        unaccounted.append(chunk)
            rendered = "-".join(m.strip("-") for m in rendered_morphemes if m)
            outputs.append(
                GenerationResult(
                    result=rendered or text,
                    explanation=f"{family}/{cell}: {rendered or text}",
                    components=[
                        {
                            "era": cell,
                            "family": family,
                            "rendered": rendered or text,
                            "morphemes": morpheme_components,
                            "unaccounted": unaccounted,
                        }
                    ],
                )
            )
        return outputs


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


register(KenningRewind())


class KenningRender(Generator):
    """wyrd-y10 — render an English (or English-rendered) name in
    an alternate phonemic script.

    A phonemic script written for English IS English, just visually
    disguised. Killer GM-handout demo: produce signage / inscriptions
    that look foreign but are decodable for committed players.

    Initial target: Shavian (~48 glyphs, plane-1 codepoints
    U+10450-U+1047F). Future scripts (Tengwar / Cirth / Elder Futhark
    / Ogham) drop in via additional ``transliterate`` dispatch arms
    in ``wyrd.generators.kenning.scripts``.
    """

    name = "kenning-render"
    display_name = "Kenning — Render in an Alternate Script"
    description = (
        "Render an English (or English-rendered) name in an alternate "
        "phonemic script — Shavian today, Tengwar / Cirth / Elder "
        "Futhark on follow-up. Atmospheric for tabletop handouts; the "
        "result is still English, just visually disguised."
    )
    legend = _LEGEND
    multi_result = False

    def input_schema(self) -> dict[str, Any]:
        from wyrd.generators.kenning.scripts import SUPPORTED_SCRIPTS

        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "English-rendered name to transliterate.",
                },
                "script": {
                    "type": "string",
                    "enum": list(SUPPORTED_SCRIPTS),
                    "default": "shavian",
                    "description": "Target script. Currently only Shavian.",
                },
            },
            "required": ["name"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        from wyrd.generators.kenning.scripts import transliterate

        text = (params.get("name") or "").strip()
        if not text:
            raise ValueError("name is required")
        script = params.get("script") or "shavian"
        rendered = transliterate(text, script)
        return GenerationResult(
            result=rendered,
            explanation=f"{text} → {rendered} ({script})",
            components=[
                {
                    "input": text,
                    "rendered": rendered,
                    "script": script,
                }
            ],
        )


register(KenningRender())


class KenningEraMap(Generator):
    """wyrd-381 — bulk-generate N names AND render each at multiple
    historical strata in one shot.

    The Domesday-vs-modern map demo: roll a region of N invented
    toponyms, then surface the SAME underlying names rendered at
    e.g. {oe-late, me, modern}. Same Kenning roll, three eras of
    paper. The killer GM-handout pattern is the "ancient map +
    modern map" pair — both are period-consistent because they
    share the same generated morpheme stack.

    Implementation: composes the existing ``Kenning`` (name
    generator) with ``KenningRewind`` (era-stop renderer) — each
    result carries one generated name plus its era-stop table as
    components, so the SPA / CLI can render the table directly.
    """

    name = "kenning-era-map"
    display_name = "Kenning — Stratified Era Map"
    description = (
        "Bulk-generate N invented toponyms AND render each at "
        "multiple historical strata in one shot. Pairs naturally "
        "with a Domesday-vs-modern handout: same map, three eras "
        "of paper. Each result is one generated name with its "
        "era-stop table attached."
    )
    legend = _LEGEND
    multi_result = True

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "culture": {
                    "type": "string",
                    "enum": CULTURES,
                    "default": "english",
                    "description": (
                        "Linguistic culture. Forwarded to the underlying Kenning generator."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "enum": available_tags()},
                    "default": [],
                    "description": "Optional tag filters (forwarded to Kenning).",
                },
                "count": {
                    "type": "integer",
                    "default": 6,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "How many names to generate (region size).",
                },
            },
            "required": ["culture"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        # multi_result generators surface one-result-per-name; the
        # single-result fallback returns the first.
        results = self.generate_all(params, seed)
        if not results:
            raise ValueError("kenning-era-map produced no results")
        return results[0]

    def generate_all(self, params: dict[str, Any], seed: int) -> list[GenerationResult]:
        culture = params.get("culture") or "english"
        tags = params.get("tags") or []
        count = int(params.get("count") or 6)

        # 1. Generate N names with the existing Kenning generator.
        # Kenning.generate returns ONE name per call (the framework
        # normally loops via the dispatcher); since we're inside a
        # multi_result=True generator we drive the loop directly,
        # offsetting the seed per roll so successive names are
        # distinct.
        kenning = Kenning()
        kenning_params = {"culture": culture, "tags": tags}
        names: list[str] = []
        seen: set[str] = set()
        for i in range(count):
            roll = kenning.generate(kenning_params, seed + i)
            name = (roll.result or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        if not names:
            return []

        # 2. For each name, run KenningRewind to capture the
        # era-stop table. Each generated-name → one
        # GenerationResult whose components hold the era table.
        rewind = KenningRewind()
        outputs: list[GenerationResult] = []
        for name in names:
            try:
                era_results = rewind.generate_all({"name": name}, seed)
            except Exception:
                # A name that doesn't decompose cleanly still
                # belongs in the map — fall back to a one-cell
                # 'modern' result so the table stays uniform.
                era_results = []
            era_cells: list[dict[str, Any]] = []
            for er in era_results:
                # KenningRewind components are a single dict per
                # era stop; flatten into the era-map's table.
                if er.components:
                    era_cells.append(er.components[0])
            if not era_cells:
                era_cells = [
                    {
                        "era": "modern",
                        "family": "english",
                        "rendered": name,
                        "morphemes": [],
                        "unaccounted": [name],
                    }
                ]
            modern_form = next(
                (c["rendered"] for c in era_cells if c.get("era") == "modern"),
                name,
            )
            outputs.append(
                GenerationResult(
                    result=modern_form,
                    explanation=" → ".join(
                        f"{c.get('era', '?')}: {c.get('rendered', '?')}" for c in era_cells
                    ),
                    components=[
                        {
                            "name": name,
                            "era_cells": era_cells,
                        }
                    ],
                )
            )
        return outputs


register(KenningEraMap())


class KenningCreature(Generator):
    """wyrd-vz7f: surface fantasy-creature etymology from the wyrd-ami
    pipeline data. Input is a creature name (Harpy, Daemon, Dwarf,
    Drake, ...); output is the etymology line + descent context + era
    reflexes when available.

    The wyrd-ami pipeline (D30) mines fantasy_morpheme rows linking
    creature input_names to real etymons in the lexicon; this is the
    runtime tap. ``_load_fantasy_morphemes`` reads the bundle's
    ``fantasy_morphemes`` field (populated by
    ``lexicon.collect_fantasy_morphemes`` at export time). 234 usable
    creatures shipped at wyrd-vz7f time; the catalog grows as the
    wyrd-ami pipeline ingests more pfsrd2-monsters input.
    """

    name = "kenning-creature"
    display_name = "Kenning — Creature etymology"
    description = (
        "Look up the historical etymology of a fantasy / mythological "
        "creature name. Returns the linked attested ancestor (language, "
        "canonical form, glosses) and any era reflexes mined for that "
        "etymon's cluster."
    )
    details = (
        "<p>"
        "Type a creature name like <em>Harpy</em>, <em>Daemon</em>, or "
        "<em>Drake</em> and Kenning surfaces the historical etymon the "
        "wyrd-ami research pipeline linked it to: the source language, "
        "the attested ancestral form, and any descendant forms across "
        "later eras."
        "</p>"
        "<p>"
        "Unknown names (modern coinages, non-corpus mythologies, or "
        "names the LLM-research pipeline couldn't resolve) return a "
        "polite 'no etymology found' result rather than an error."
        "</p>"
    )
    legend = _LEGEND
    multi_result = False

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Creature name (e.g. 'Harpy', 'Daemon').",
                },
            },
            "required": ["name"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        text = (params.get("name") or "").strip()
        if not text:
            raise ValueError("name is required")
        creatures = _load_fantasy_morphemes()
        entry = creatures.get(text.lower())
        if entry is None:
            return GenerationResult(
                result=text,
                explanation=(
                    f"No etymology found for {text!r} in the wyrd-ami "
                    "fantasy-morpheme corpus. The pipeline may classify "
                    "it as a modern coinage, an outside-language-family "
                    "name, or simply not yet mined."
                ),
            )
        return GenerationResult(
            result=entry["input_name"],
            explanation=_format_creature_explanation(entry),
        )


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


register(KenningCreature())
