"""Kenning — town-name generator. Compounds Old English / Norse / Celtic morphemes."""

from __future__ import annotations

import itertools
import json
from functools import lru_cache
from importlib import resources
from typing import Any

from wyrd.generators.kenning.era import era_cells_for_family, resolve_era_input
from wyrd.generators.kenning.meaning import Meaning, load_meanings
from wyrd.generators.kenning.name import Name
from wyrd.generators.kenning.proportions import load_proportions
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
        data.extend(json.load(f))
    data.extend(_norman_manorial_subjects())
    return load_meanings(data)


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
    """Clear both ``_load_meanings`` and ``_load_culture`` caches.

    The two caches form an aggregate: ``_load_culture`` holds a
    ``NameGenerator`` parameterised on the ``meaning_db`` from
    ``_load_meanings``, so invalidating one without the other yields
    a stale ``NameGenerator`` the next time a culture is loaded.
    Replaces ``_load_meanings.cache_clear`` so the standard test-side
    pattern (``_load_meanings.cache_clear()``) clears both.

    The reverse direction (``_load_culture.cache_clear()`` alone) is
    intentionally uncoupled — a caller who only wants to drop the
    per-culture generator (e.g. proportions-only refresh) shouldn't
    pay to re-parse meanings.json. The asymmetry is by design.
    """
    _original_load_meanings_cache_clear()
    _load_culture.cache_clear()


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
                    "description": (
                        "wyrd-lr4 Phase 3 within-language stratum filter. Restricts the "
                        "morpheme inventory to forms classified into a specific register "
                        "bucket — for Welsh: 'native-welsh', 'brittonic-substrate', "
                        "'medieval-welsh', 'latin-loan', 'english-loan'. Morphemes with "
                        "no stratum data pass through (Phase 1 has classified Welsh-family "
                        "etymons only; Latin / OE / French / etc. all admit until Phase 4 "
                        "lands). Composes with --era via intersection. Empty disables "
                        "the filter — bit-stable behavior."
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
        include_fiction = _coerce_bool(params.get("include_fiction", False))

        moods = params.get("mood", []) or []
        if isinstance(moods, str):
            moods = [moods]
        for spec in moods:
            tags, harshness = _apply_mood(spec, tags, harshness)
        tags = tuple(tags)
        exclude_tags: tuple[str, ...] = () if include_fiction else (_FICTION_TAG,)

        era_range = _resolve_era_param(params.get("era"), culture)
        stratum = params.get("stratum") or None

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
    name_str: str, words, meaning_db: dict[str, list[Meaning]]
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
    )


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

        # Dedupe by structural signature: when one usage has multiple Meanings
        # (different senses, e.g. -y as both "district" and "island"), the raw
        # cartesian product produces N copies of the same structural break with
        # only the Meaning identity differing. We collapse those into one
        # result and combine senses inside _build_explanation_part.
        seen: set[tuple] = set()
        candidates: list[tuple[int, int, tuple]] = []
        sig_to_words: dict[tuple, Any] = {}
        for words in itertools.product(*per_word):
            sig = _decomposition_signature(words)
            if sig in seen:
                continue
            seen.add(sig)
            sig_to_words[sig] = words
            unaccounted = sum(1 for w in words for c in w.word if isinstance(c, str) and c)
            total = sum(1 for w in words for c in w.word if not (isinstance(c, str) and not c))
            candidates.append((unaccounted, total, sig))

        # Best readings first: fewer unaccounted fragments, then simpler.
        candidates.sort()
        return [
            _build_decomposition_result(text, sig_to_words[sig], meaning_db)
            for _, _, sig in candidates[:_MAX_DECOMPOSITIONS]
        ]


register(KenningExplain())
