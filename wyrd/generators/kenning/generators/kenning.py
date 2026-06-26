"""The main `kenning` Generator — composes morphemes into town names."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import random

    from wyrd.generators.kenning.runtime.proportions import NameGenerator, NewName
    from wyrd.generators.kenning.vectors.schemas import EmpiricalPriors, RequestVector

from wyrd.generators.kenning import (
    _FICTION_TAG,
    _LEGEND,
    CULTURES,
    _apply_joiner_insertion,
    _coerce_bool,
    _coerce_float,
    _era_options_by_culture,
    _load_culture,
    _load_joiners,
    _load_norman_manorial_families,
    _resolve_era_param,
    _resolve_era_render_language,
    _resolve_stratum_param,
    _stratum_options_by_culture,
    _tags_options_by_culture,
    available_tags,
)
from wyrd.generators.kenning.era.cells import ERA_CELLS, family_stage_order
from wyrd.generators.kenning.errors import EmptyEligiblePool
from wyrd.generators.kenning.lexicon.strata import (
    FRENCH_STRATA,
    OLD_ENGLISH_STRATA,
    OLD_NORSE_STRATA,
    WELSH_STRATA,
)
from wyrd.generators.kenning.registers.effects import available_register_effects
from wyrd.registry import GenerationResult, Generator
from wyrd.seed import rng_for

_logger = logging.getLogger(__name__)

# wyrd-rogd.12 / wyrd-rogd.2: the canonical per-family era-stage axis (oldest →
# newest), the SINGLE source of truth for both the col-3 reflex grid's fixed
# columns and the Configure-column generation era control. Exposed on the
# manifest so the SPA lays populated stages onto a stable axis (empty slots
# render a muted '—') instead of carrying its own drifting copy.
_ERA_STAGES: dict[str, list[str]] = {family: family_stage_order(family) for family in ERA_CELLS}


class Kenning(Generator):
    name = "kenning"
    era_stages = _ERA_STAGES  # wyrd-rogd.12: per-family fixed era axis (manifest)
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
                    # The enum is the culture-AGNOSTIC union (CLI/API accept any
                    # real tag); x-options-by-culture (wyrd-ah53) narrows the SPA
                    # composer to tags the chosen culture can actually satisfy —
                    # same dependent-select mechanism as era / stratum, since a
                    # --tag reserves one slot from a pool that HAS the tag (D47),
                    # so a culture-empty tag would only 'no eligible name'.
                    "items": {"type": "string", "enum": available_tags()},
                    "x-options-by-culture": _tags_options_by_culture(),
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
                "novelty": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    # wyrd-0k9o: SPA renders this as a slider (a [0,1] proportion).
                    # `x-ui-*` matches the repo's x-prefixed SPA-extension convention
                    # (x-options-by-culture / x-pick-from); Field.svelte reads it.
                    "x-ui-widget": "slider",
                    # wyrd-tngt: 0 weights by attested frequency (typical); 1 is
                    # uniform over the ELIGIBLE pool (hard filters still apply), not
                    # "random" — the description must not overclaim.
                    "description": (
                        "How adventurous the name is (0–1): 0 favors historically "
                        "common morphemes; 1 makes every eligible morpheme equally "
                        "likely."
                    ),
                },
                "mood": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    # wyrd-vslw: SPA composer reads x-pick-from to populate
                    # the catalog pane (no hardcoded mood list in the
                    # frontend). x-allow-custom signals that operators may
                    # also submit values outside the picker (the 'harsh:0.5'
                    # graduated-strength syntax); phase 2 of the composer
                    # epic surfaces a weight slider that produces those.
                    "x-pick-from": available_register_effects(),
                    "x-allow-custom": True,
                    "description": (
                        "D6 stylistic-mood presets (repeatable). Each entry is a "
                        "register-effect name from the bundled catalog "
                        "(wyrd/generators/kenning/data/register_effects.yaml — see "
                        f"{available_register_effects()!r}), optionally with a "
                        "colon-suffix multiplier (e.g. 'harsh:0.5' for graduated "
                        "strength). 'grim' applies a menacing semantic-tag union; "
                        "'harsh' biases sampling toward stop-final / cluster-heavy "
                        "morphemes. Multiple moods compose: register effects sum "
                        "component-wise."
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
                "include_unglossed": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "wyrd-glos: when True, allow morphemes that carry no gloss "
                        "(no recorded meaning) to appear in generated names — their "
                        "etymology line shows '(unglossed)'. Default False restricts "
                        "generation to morphemes the system can explain (the rando-era "
                        "behavior), because the etymology line is the load-bearing "
                        "feature of a kenning. Enabling this trades explainability for "
                        "a larger morpheme inventory (~46% of the English pool is "
                        "currently unglossed — mostly real place-name elements awaiting "
                        "gloss backfill). Single-character unglossed fragments are "
                        "ALWAYS dropped, regardless of this flag."
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
                    # wyrd-awo + wyrd-rogd.2: dependent-select metadata read by
                    # the SPA. Each culture surfaces its family's compressed era
                    # STAGES (old-english / middle-english / modern-english) —
                    # the same axis as the col-3 grid, not the raw cells. A
                    # picked stage resolves to the UNION year range of its cells.
                    # CLI/API still accept raw cells, bare years, and
                    # 'family/label' shapes; this only constrains the SPA UX.
                    "x-options-by-culture": _era_options_by_culture(),
                    # wyrd-rogd.2: options are language tags, so the SPA labels
                    # them via languageLabel (old-english → "Old English"),
                    # matching the col-3 grid's stage headers.
                    "x-option-language": True,
                    # wyrd-3vju.2: the empty default ('') is the native-per-morpheme
                    # mix — each part rendered in its OWN source-era form — so the
                    # SPA labels it "Mixed Era" (a real, chosen mode), not "(no
                    # filter)". The present-day stage (modern-english) is the
                    # force-modern choice: clean modern spellings throughout.
                    "x-empty-option-label": "Mixed Era",
                    "description": (
                        "Period of the name (D44 render + D46 record gate). The "
                        "default, 'Mixed Era', renders each morpheme in its OWN native "
                        "source-era form (a historical mix) and draws from the full "
                        "morpheme pool, as does the present-day stage. A HISTORICAL "
                        "period draws only from morphemes already IN THE RECORD by the "
                        "period's end — a modern coinage can't appear on a Domesday "
                        "map — while everything as old or older stays available "
                        "forever (pools accrete; nothing expires). A historical "
                        "period also renders each morpheme in its "
                        "era-appropriate attested form (e.g. 'old-english' → Old "
                        "English 'Tūn', 'Sūþ'; 'middle-english' → 'Toun') instead of "
                        "the modern spelling. The present-day stage (modern-english) renders "
                        "the modern canonical spelling throughout. The SPA renders this as a "
                        "dropdown of the chosen culture's compressed era STAGES; a stage "
                        "resolves to the union year range of its cells. The culture-"
                        "agnostic token 'present-day' resolves to the chosen culture's "
                        "present-day stage (modern-english / welsh / irish) — it exists "
                        "so WYRD_DEFAULT_ERA can set a modern default across every "
                        "culture (wyrd-kqyf). CLI/API also "
                        "accept a raw cell, a bare year (e.g. '1086' → the cell "
                        "containing 1086 in the culture's era family), or an explicit "
                        "'family/label' pair (e.g. 'english/oe-late'). Morphemes with "
                        "no era reflex fall back to the modern spelling (~10% of "
                        "generated morphemes), so a period name may mix in some modern "
                        "forms."
                    ),
                },
                "cohesion": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    # wyrd-0k9o: render as a [0,1] slider like novelty (Field.svelte
                    # gates the slider widget on this key). wyrd-e2b4 added it once
                    # the knob actually biased.
                    "x-ui-widget": "slider",
                    # wyrd-e2b4: user-facing copy — no internal jargon, parallel to
                    # the novelty blurb above.
                    "description": (
                        "How naturally the name's parts fit together (0–1): 0 picks "
                        "each part independently; higher values favor meaning "
                        "combinations that are common in real place names (water + "
                        "hill) over odd pairings."
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
                        "are classified today). "
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
                "priors_path": {
                    "type": "string",
                    "description": (
                        "wyrd-ecjp.5 PR C: filesystem path to a JSON "
                        "empirical-priors sidecar (emitted by 'lexicon "
                        "dump-empirical-priors'). Supplies the vector "
                        "scoring path's baseline axis. When absent, the "
                        "baseline axis contributes 0 and the score falls "
                        "back to phon + sem + pos."
                    ),
                },
                "scoring_weights": {
                    "type": "object",
                    "description": (
                        "wyrd-ecjp.9: per-axis ScoringWeights overrides "
                        "(D36.2). Keys: 'phon_w', 'sem_w', 'pos_w', "
                        "'base_w', each a float scalar on its axis. "
                        "Defaults to 1.0 across all four axes "
                        "(ScoringWeights()) — the canonical balanced "
                        "composition. Used by the operator-facing CLI flags "
                        "--baseline-weight / --phonological-weight / "
                        "--semantic-weight / --position-weight."
                    ),
                    "properties": {
                        "phon_w": {"type": "number"},
                        "sem_w": {"type": "number"},
                        "pos_w": {"type": "number"},
                        "base_w": {"type": "number"},
                    },
                },
                "packs": {
                    "type": "array",
                    "description": (
                        "wyrd-ecjp.11: scenario-pack overlay declarations "
                        "(D36.4). Each entry is an object with keys "
                        "'pack_name' (must exist in the bundled pack "
                        "catalog), optional 'weight_override' (float in "
                        "[0, 10]; falls back to the pack manifest's "
                        "default_weight when null/absent), and optional "
                        "'allowed_pack_tags' (list of tags narrowing the "
                        "pack's lemma subset per PackOverlay."
                        "allowed_pack_tags). Used by the operator-"
                        "facing CLI flags --pack / --pack-tag-filter; "
                        "unknown pack_name surfaces as ValueError at "
                        "dispatch time."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "pack_name": {"type": "string"},
                            "weight_override": {"type": ["number", "null"]},
                            "allowed_pack_tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["pack_name"],
                    },
                },
            },
            "required": [],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        culture = params.get("culture", "english")
        raw_tags = params.get("tags", []) or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        spelling_variety = _coerce_float(params.get("spelling_variety"))
        inflection_density = _coerce_float(params.get("inflection_density"))
        novelty = _coerce_float(params.get("novelty"))
        harshness = _coerce_float(params.get("harshness"))
        cohesion = _coerce_float(params.get("cohesion"))
        manorial_affix = _coerce_float(params.get("manorial_affix"))
        joiner_density = _coerce_float(params.get("joiner_density"))
        include_fiction = _coerce_bool(params.get("include_fiction", False))
        # wyrd-glos: glossing is a project pillar — the etymology line is
        # the load-bearing feature — so the generator only draws morphemes
        # it can explain unless the operator opts in. Default False
        # (glossed-only, the rando-era behavior). Single-char unglossed
        # fragments are dropped regardless of this flag.
        include_unglossed = _coerce_bool(params.get("include_unglossed", False))
        priors_path = params.get("priors_path")
        # wyrd-ecjp.9: per-axis ScoringWeights (vector mode only). The
        # CLI flag layer composes a dict {phon_w, sem_w, pos_w, base_w}
        # which we'll instantiate into a ScoringWeights dataclass at
        # the dispatch boundary. Absent / None → use ScoringWeights()
        # defaults (1.0 across all four axes).
        scoring_weights_raw = params.get("scoring_weights")
        # wyrd-ecjp.11: pack overlays (vector mode only). The CLI flag
        # layer composes a list of {pack_name, weight_override,
        # allowed_pack_tags} dicts; the dispatch helper resolves each
        # entry against the bundled pack catalog to construct
        # PackOverlay + pack_meaning_dbs.
        packs_raw = params.get("packs") or []

        moods = params.get("mood", []) or []
        if isinstance(moods, str):
            moods = [moods]
        # The vector adapter expands mood specs + base harshness through the
        # register-effect catalog itself, so pass the raw specs + base values
        # straight through (no tag/harshness pre-mutation here).
        original_moods = tuple(moods)
        original_harshness = harshness
        exclude_tags: tuple[str, ...] = () if include_fiction else (_FICTION_TAG,)

        # D46 (refining D44): the resolved era range's END is the record-entry
        # cutoff — a bounded era draws only from morphemes already "in the
        # record by then"; pools accrete and nothing expires. None (no era, or
        # an open-ended era like the deployed present-day default) gates
        # nothing. The call doubles as request validation (typo'd era → 4xx).
        era_range = _resolve_era_param(params.get("era"), culture)
        era_record_cutoff = era_range[1] if era_range else None
        # wyrd-3tvd: a bounded HISTORICAL era also leans the priors-baseline
        # toward that period's naming FASHION (D36.7 per-era frequency cells) via
        # era_midpoint — not just the D46 inventory cutoff above. An era with no
        # upper bound (the deployed present-day default, (1500, None)) → 0, the
        # D44 wildcard, so default generation is unchanged. (See
        # _request_era_midpoint for the keyed-on-upper-bound rule.)
        era_midpoint = _request_era_midpoint(era_range)
        # wyrd-6c8x (feature A): the target language to RENDER morphemes in for
        # the requested era — under D44 this is era's ONLY effect. None =
        # render the modern canonical form (no era set, or a cell with no
        # canonical language). Threaded into the render step.
        era_render_language = _resolve_era_render_language(params.get("era"), culture)
        # wyrd-j3gy: _resolve_stratum_param validates against the
        # per-culture allowed-set (with ALL_STRATA fallback for
        # cultures without classifiers yet). A typo'd --stratum
        # surfaces as a clean ValueError rather than silently
        # no-opping.
        stratum = _resolve_stratum_param(params.get("stratum"), culture)

        # Phase timing: load_culture is lru_cached after first miss, so
        # this is ~0ms warm and the per-culture bundle parse time on
        # cold first-touch. Sampling is the per-name work the operator
        # actually pays for on every call. Emit as DEBUG so prod stays
        # quiet at default log levels; staging flips LOG_LEVEL=DEBUG to
        # see the breakdown.
        _trace = _logger.isEnabledFor(logging.DEBUG)
        t_load = time.perf_counter()
        name_gen, _ = _load_culture(culture)
        t_load_ms = (time.perf_counter() - t_load) * 1000

        rng = rng_for(seed)
        t_sample = time.perf_counter()
        new_name = _generate_via_vector(
            name_gen,
            rng,
            culture=culture,
            # The requested tags; the adapter expands mood specs (original_moods)
            # + harshness through the register-effect catalog on its own.
            tags=list(raw_tags),
            mood=original_moods,
            harshness=original_harshness,
            era_record_cutoff=era_record_cutoff,
            era_midpoint=era_midpoint,
            stratum=stratum,
            cohesion=cohesion,
            exclude_tags=exclude_tags,
            priors_path=priors_path,
            scoring_weights_raw=scoring_weights_raw,
            packs_raw=packs_raw,
            include_unglossed=include_unglossed,
            spelling_variety=spelling_variety,
            inflection_density=inflection_density,
            novelty=novelty,
            era_render_language=era_render_language,
            # wyrd-24s6 (D41): True for ANY requested era (incl. "modern-english"),
            # so the renderer can tell era="" (→ native default) from an explicit
            # era that resolves to no render-language (→ modern).
            era_requested=bool(params.get("era")),
            # wyrd-dag4: a REQUEST-scoped cache for the vector eligibility pool.
            # It lives in ``params`` — the same dict the app's count loop reuses
            # across all N names — so the pool is built once per request and
            # reused across the count loop, then discarded with ``params`` when
            # the request ends. NOT on the shared per-culture name_gen, so it
            # never leaks across requests (the wyrd-i7uy rule). Single-result /
            # test callers pass no count loop → a fresh {} each call (no reuse,
            # prior behavior).
            pool_cache=params.setdefault("_vector_pool_cache", {}),
        )
        if new_name is None:
            # Vector path filtered everything (empty register + empty
            # priors, or every meaning gated out). Loud failure with an
            # operator-readable diagnostic. EmptyEligiblePool (not a bare
            # ValueError) so the web layer maps a valid-but-unsatisfiable
            # request to a 422 instead of an opaque 500 (wyrd-hh2m).
            raise EmptyEligiblePool(
                "vector scoring produced no eligible name — check that the "
                "bundle carries phonological_vector data (kq7w.1), "
                "priors_path is set (or the register carries non-trivial "
                "weights), and the gate predicates (culture / era / "
                "stratum) match available meanings."
            )
        t_sample_ms = (time.perf_counter() - t_sample) * 1000
        t_render = time.perf_counter()
        result_str = str(new_name)
        # wyrd-24s6 (D41): the MODERN companion rendering, surfaced beside the
        # canonical native result_str.
        result_modern = new_name.modern_name()
        explanation = new_name.description()
        components = new_name.components()
        # wyrd-q0g6 Phase 1.5: compose-time joiner insertion. Gated on
        # density>0 + non-empty pool so legacy callers stay bit-stable.
        if joiner_density > 0:
            joiners = _load_joiners()
            if joiners:
                # wyrd-24s6 (D41): joiner insertion rebuilds BOTH the native
                # result_str and the modern result_modern in one walk, so the
                # same joiners land in both renderings at the same positions.
                result_str, result_modern, explanation, components = _apply_joiner_insertion(
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
            result_modern = f"{result_modern} {family}"  # wyrd-24s6: same affix on both
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
        # wyrd-cp2d: per-word morpheme breakdown for continuity flow.
        # Captured AFTER joiner insertion / manorial-affix so the dump
        # mirrors the rendered output's morpheme stack. The dict
        # includes per-language source lemmas so 'rewind --from-json'
        # can skip re-decomposition and use the actual picks the
        # generator committed to (avoiding the 'Cranwith → corn /
        # Cornlandian' situation where trie re-decomposition picks
        # a different etymological cluster).
        morphemes_by_word = new_name.to_dict()["words"]
        if _trace:
            t_render_ms = (time.perf_counter() - t_render) * 1000
            _logger.debug(
                "kenning.generate culture=%s "
                "load_ms=%.1f sample_ms=%.1f render_ms=%.1f "
                "result=%r words=%d",
                culture,
                t_load_ms,
                t_sample_ms,
                t_render_ms,
                result_str,
                len(morphemes_by_word),
            )
        return GenerationResult(
            result=result_str,
            result_modern=result_modern,
            explanation=explanation,
            components=components,
            morphemes_by_word=morphemes_by_word,
        )


@lru_cache(maxsize=8)
def _load_priors_sidecar_cached(priors_path: str):
    """Process-wide cached load of an operator-supplied priors
    sidecar. Keyed by the absolute path string. lru_cache(maxsize=8)
    caps total memory under a heavy-experimentation workflow that
    rotates sidecars; oldest unused entry gets evicted on overflow.

    This memoizes a disk read keyed by the file path — the sidecar's
    content is immutable for a given path, so it's the "load bundled/file
    data once" case, not request-derived state (wyrd-i7uy): re-reading +
    re-parsing the sidecar JSON on every request that names it would be
    pure waste.
    """
    from pathlib import Path

    from wyrd.generators.kenning.lexicon.empirical_priors import (
        load_empirical_priors_from_json,
    )

    return load_empirical_priors_from_json(Path(priors_path))


def _resolve_vector_inputs(
    *,
    culture: str,
    tags: list[str],
    mood: tuple[str, ...],
    harshness: float,
    era_record_cutoff: int | None,
    stratum: str | None,
    priors_path: str | None,
    scoring_weights_raw: dict[str, float] | None = None,
    packs_raw: list[dict[str, Any]] | None = None,
) -> tuple[RequestVector, EmpiricalPriors, dict[str, dict]]:
    """Translate per-call knobs into the vector path's shared inputs:
    ``(request, priors, pack_meaning_dbs)``.

    Era reaches the gate only as ``era_record_cutoff`` — the D46
    record-entry rule ("in the record by then"); it still never drives
    the baseline-axis midpoint, and its render effect stays downstream
    (era_render_language, D44).

    Extracted from :func:`_generate_via_vector` (wyrd-y0lx) so the
    single-slot regeneration endpoint (``kenning-regenerate-morpheme``)
    builds its RequestVector / priors / pack_meaning_dbs through the exact
    same resolution as a full generate, instead of duplicating it and
    letting the two copies drift apart."""

    from wyrd.generators.kenning.runtime.vector_kenning_adapter import build_request_vector
    from wyrd.generators.kenning.vectors.schemas import PackOverlay, ScoringWeights

    # wyrd-ecjp.9: build a ScoringWeights from the CLI-supplied dict
    # if present, else fall back to defaults (1.0 across all axes).
    # The dict shape is {phon_w, sem_w, pos_w, base_w} — keys mirror
    # the dataclass field names directly so this stays a thin pass-
    # through with no key remapping. Unknown keys are silently dropped
    # rather than raising TypeError: the CLI controls its own keys
    # tightly, but SPA / Lambda callers may send extra params from
    # forward-compat client versions; ignoring unknown keys keeps the
    # generator entry point resilient. The set of known keys is the
    # ScoringWeights dataclass's own field names (single source of
    # truth — no hard-coded list to drift).
    if scoring_weights_raw:
        known = set(ScoringWeights.__dataclass_fields__)
        filtered = {k: v for k, v in scoring_weights_raw.items() if k in known}
        weights = ScoringWeights(**filtered)
    else:
        weights = ScoringWeights()

    # wyrd-ecjp.11: resolve declared packs against the bundled pack
    # catalog. Each parsed pack-spec dict references a pack_name; the
    # bundle's PackBundle supplies template_donor + template_recipient
    # + default_weight (used when the operator's weight_override is
    # None). Pack-tag filters land on PackOverlay.allowed_pack_tags
    # (ecjp.8's pre-score gate). The CLI layer already validated that
    # every declared pack exists in _load_packs() — but we re-resolve
    # here because non-CLI callers (Lambda, SPA, tests) bypass that
    # validation and could supply an unknown pack_name. Unknown pack
    # names raise KeyError here rather than failing silently.
    pack_overlays: tuple[PackOverlay, ...] = ()
    pack_meaning_dbs: dict[str, dict] = {}
    if packs_raw:
        from wyrd.generators.kenning import _load_packs

        loaded = _load_packs()
        overlays: list[PackOverlay] = []
        for spec in packs_raw:
            pack_name = spec["pack_name"]
            if pack_name not in loaded:
                available = ", ".join(sorted(loaded.keys())) or "(no packs bundled)"
                raise ValueError(f"unknown pack {pack_name!r}; available: {available}")
            bundle = loaded[pack_name]
            weight = spec.get("weight_override")
            if weight is None:
                weight = bundle.manifest.default_weight
            overlays.append(
                PackOverlay(
                    pack_name=pack_name,
                    template_donor=bundle.manifest.template_donor,
                    template_recipient=bundle.manifest.template_recipient,
                    weight=float(weight),
                    allowed_pack_tags=frozenset(spec.get("allowed_pack_tags") or ()),
                )
            )
            pack_meaning_dbs[pack_name] = bundle.meaning_db
        pack_overlays = tuple(overlays)

    request = build_request_vector(
        culture=culture,
        tags=tags,
        harshness=harshness,
        mood=mood,
        era_record_cutoff=era_record_cutoff,
        stratum=stratum,
        weights=weights,
        packs=pack_overlays,
    )

    if priors_path:
        # Cache the sidecar's parsed content by ``priors_path`` so repeated
        # requests naming the same sidecar don't re-read + re-parse the JSON
        # from disk each time (immutable file data keyed by path — fine under
        # the no-request-state rule, wyrd-i7uy).
        priors = _load_priors_sidecar_cached(str(priors_path))
    else:
        # wyrd-ecjp.10b: fall back to the bundled priors.json sidecar
        # (if shipped) before degrading to empty. Operator who passes
        # --priors-path explicitly still overrides the bundle (path-
        # arg wins). When neither path arg nor bundled sidecar exists,
        # priors is empty + baseline axis contributes 0 (legacy
        # behavior preserved).
        from wyrd.generators.kenning import _load_empirical_priors

        priors = _load_empirical_priors()

    return request, priors, pack_meaning_dbs


def _request_era_midpoint(era_range: tuple[int | None, int | None] | None) -> int:
    """The baseline-axis ``era_midpoint`` for the requested era (wyrd-3tvd): so
    generation at a bounded historical period also leans toward that period's
    naming FASHION (the D36.7 per-era frequency cells), not just its inventory
    floor.

    Keyed on the era's UPPER bound:
      * no era at all → ``0``;
      * no upper bound (``end is None``) → ``0`` — the era runs to the present, so
        there is no bygone period to lean toward. This is the deployed default
        ``present-day`` (resolves to ``(1500, None)``), so default generation is
        unchanged by design (NOT merely because no 1500 cell happens to exist);
      * bounded historical (both bounds) → ``(start + end) // 2``;
      * open START, bounded recent end (e.g. early-OE, ``(None, 800)``) → the end
        bound, the only known anchor.

    ``0`` is the D44 wildcard sentinel = no era-fashion lean. (Era cells are all
    CE/AD, so a genuine bounded era never midpoints to year 0 — no collision with
    the sentinel.) This is the REQUEST convention, deliberately distinct from
    ``empirical_priors._era_midpoint`` (which uses an open-bound OFFSET so every
    era CELL gets a deterministic identity year) — here an open upper bound means
    'wildcard', so it must NOT borrow that offset.
    """
    if era_range is None:
        return 0
    start, end = era_range
    if end is None:  # open-ended toward the present (present-day) → wildcard
        return 0
    if start is None:  # open start, bounded recent end → that bound
        return end
    return (start + end) // 2


def _generate_via_vector(
    name_gen: NameGenerator,
    rng: random.Random,
    *,
    culture: str,
    tags: list[str],
    mood: tuple[str, ...],
    harshness: float,
    era_record_cutoff: int | None,
    era_midpoint: int,
    stratum: str | None,
    cohesion: float,
    exclude_tags: tuple[str, ...],
    priors_path: str | None,
    scoring_weights_raw: dict[str, float] | None = None,
    packs_raw: list[dict[str, Any]] | None = None,
    include_unglossed: bool = False,
    spelling_variety: float = 0.0,
    inflection_density: float = 0.0,
    novelty: float = 0.0,
    era_render_language: str | None = None,
    era_requested: bool = False,
    pool_cache: dict | None = None,
) -> NewName | None:
    """Dispatch helper for vector scoring (the only scoring path).

    Translates the per-call knobs into a RequestVector via the
    adapter (:func:`_resolve_vector_inputs`), loads priors from disk
    if provided, and calls NameGenerator.select_via_vector.

    Returns a NewName or None when the vector path's gate / scoring
    filtered every candidate.
    """
    request, priors, pack_meaning_dbs = _resolve_vector_inputs(
        culture=culture,
        tags=tags,
        mood=mood,
        harshness=harshness,
        era_record_cutoff=era_record_cutoff,
        stratum=stratum,
        priors_path=priors_path,
        scoring_weights_raw=scoring_weights_raw,
        packs_raw=packs_raw,
    )

    return name_gen.select_via_vector(
        rng,
        request=request,
        priors=priors,
        era_midpoint=era_midpoint,
        cohesion=cohesion,
        exclude_tags=exclude_tags,
        pack_meaning_dbs=pack_meaning_dbs or None,
        include_unglossed=include_unglossed,
        spelling_variety=spelling_variety,
        inflection_density=inflection_density,
        novelty=novelty,
        era_render_language=era_render_language,
        era_requested=era_requested,
        pool_cache=pool_cache,
    )
