"""The `kenning-regenerate-morpheme` Generator — re-roll ONE morpheme of an
already-generated name in the context of the others (wyrd-y0lx)."""

from __future__ import annotations

import logging
from typing import Any

from wyrd.generators.kenning import (
    _FICTION_TAG,
    _LEGEND,
    CULTURES,
    _coerce_bool,
    _coerce_float,
    _coerce_str_list,
    _load_culture,
    _resolve_era_param,
    _resolve_era_render_language,
    _resolve_stratum_param,
)
from wyrd.generators.kenning.errors import EmptyEligiblePool
from wyrd.generators.kenning.runtime.word import (
    WORD_POSITION_KEY_PREFIX,
    word_position_for,
)
from wyrd.registry import GenerationResult, Generator
from wyrd.seed import rng_for

_logger = logging.getLogger(__name__)


class NoEligibleReplacementError(EmptyEligiblePool):
    """No candidate can fill the slot: the pool is empty after the context
    gates + in-use exclusions, or every surviving candidate's weight is
    zero/negative so the weighted draw can't pick. A valid request with no
    fulfillable answer, distinct from the malformed-input ValueErrors
    raised by ``_validate_target``.

    Subclasses ``EmptyEligiblePool`` (wyrd-hh2m): this is the per-slot flavor
    of the same valid-but-unsatisfiable empty-pool case, so the dispatcher
    maps it to a 422 — consistent with a fresh roll's empty pool, rather than
    the old 400. (It was previously a ``ValueError`` → 400; unified so two
    sibling generators don't return different statuses for the same concept.)
    Callers / tests can still catch this specifically."""


class KenningRegenerateMorpheme(Generator):
    """wyrd-y0lx — single-slot regeneration for the SPA's per-morpheme
    "regenerate" button.

    The caller supplies an already-generated name's ``words`` breakdown
    (the API envelope's ``morphemes_by_word``, same shape kenning-rewind
    accepts per wyrd-y9aa), the ``(word_index, morpheme_index)`` of the
    slot to re-roll, and the generation knobs the original roll used
    (culture / tags / mood / era / stratum / cohesion / novelty / …).
    The endpoint re-runs the vector path's gate → score → sample for
    JUST that slot while holding every other slot fixed:

    - **Same hard gates as a fresh roll** (culture / era / stratum) —
      EXCEPT ``--tag``, which (wyrd-c6o1.4) is no longer a per-lemma gate:
      like a fresh roll it reserves a single slot, so a re-roll only forces
      a tagged replacement when this slot is the name's sole tag-carrier
      (see Tag-context preservation below).
    - **Position derived from the slot's index** (D40: sole→bare,
      first→pre, last→post, interior→inner — never from stored dashes),
      with the slot's qualifier (name / saint) inferred from the
      morpheme being replaced so a name-slot re-rolls to another name.
    - **Cohesion context from the OTHER slots' tags** when cohesion>0,
      mirroring the prior-tags accumulation of a full generate.
    - **Mood-context preservation**: when the request carries mood tags
      and the slot being replaced is the only mood-carrier, the re-roll
      is restricted to mood-tagged candidates (the wyrd-4rp8 reserved-
      slot rule, re-derived from the held context) — falling back to
      the unrestricted pool when no mood-tagged candidate exists.
    - **Tag-context preservation** (wyrd-c6o1.4): the same rule for
      ``--tag`` — when the slot being replaced is the name's only
      tag-carrier, the re-roll is restricted to tag-carrying candidates
      so the >=1-tagged guarantee survives; otherwise the slot samples
      freely (another slot already satisfies the tag).
    - **No duplicates, ever** (operator decision, 2026-06-11): the
      candidate pool excludes the replaced morpheme and every morpheme
      already in use elsewhere in the name (both by modern-usage fold
      and by native-form fold), so each click visibly changes the name
      and can never create a "Hill Hill". Repeat-diversification is
      suppressed (the exclusions already guarantee the invariant, and
      diversification must not mutate the held slots).

    The response is one GenerationResult whose ``morphemes_by_word``
    carries the FULL name with the regenerated slot replaced — the
    target slot's entry is the authoritative payload (the SPA splices
    exactly that one morpheme dict into its pipeline state and
    re-renders the name client-side, composing with any prior
    transform steps); held slots pass through with their supplied
    surfaces, so the response's full ``result`` strings are a
    best-effort convenience, not the composition contract.
    """

    name = "kenning-regenerate-morpheme"
    display_name = "Kenning — Regenerate One Morpheme"
    description = (
        "Re-roll a single morpheme of a generated town name, keeping every "
        "other morpheme fixed. Honors the original generation context "
        "(culture, tags, mood, era, stratum, cohesion) and never re-picks a "
        "morpheme already in use in the name."
    )
    legend = _LEGEND
    multi_result = True  # exactly one result; the count param doesn't apply

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The generated name being edited (display only).",
                },
                "words": {
                    "type": "array",
                    "description": (
                        "The name's per-word morpheme breakdown — the API "
                        "envelope's morphemes_by_word field (same pre-picked "
                        "shape kenning-rewind accepts, wyrd-y9aa)."
                    ),
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"usage": {"type": "string"}},
                            "required": ["usage"],
                        },
                    },
                },
                "word_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Index of the word containing the slot to re-roll.",
                },
                "morpheme_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Index of the morpheme within that word.",
                },
                "seed": {
                    "type": "integer",
                    "description": (
                        "64-bit seed for a reproducible re-roll. The SPA bakes a "
                        "fresh seed into the transform step at click time so "
                        "pipeline re-runs and restored workspaces replay the same "
                        "pick deterministically."
                    ),
                },
                # Generation-context knobs, mirroring the kenning generator's
                # schema (only the subset that affects the eligible pool /
                # scoring / rendering of a single slot; count is meaningless
                # here and joiner/manorial post-processing never applies to a
                # single-slot re-roll).
                "culture": {"type": "string", "enum": CULTURES, "default": "english"},
                "tags": {"type": "array", "items": {"type": "string"}, "default": []},
                "mood": {"type": "array", "items": {"type": "string"}, "default": []},
                "harshness": {"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0},
                "era": {"type": "string", "default": ""},
                "stratum": {"type": "string", "default": ""},
                "cohesion": {"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0},
                "novelty": {"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 1.0},
                "spelling_variety": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "inflection_density": {
                    "type": "number",
                    "default": 0.0,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "include_fiction": {"type": "boolean", "default": False},
                "include_unglossed": {"type": "boolean", "default": False},
                "priors_path": {"type": "string"},
                "scoring_weights": {"type": "object"},
                "packs": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["words", "word_index", "morpheme_index"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        return self.generate_all(params, seed)[0]

    def generate_all(self, params: dict[str, Any], seed: int) -> list[GenerationResult]:
        # Lazy imports to dodge the circular dep: wyrd.generators.kenning.
        # __init__ imports this module (via generators/__init__.py) at the END
        # of its body, so module-level imports that trace back through that
        # __init__ — or through runtime modules it is mid-loading — would hit
        # partial-init AttributeErrors. Same pattern + reason as
        # kenning_rewind.py's per-call-site lazy imports.
        from wyrd.generators.kenning import _rank_siblings
        from wyrd.generators.kenning.generators.kenning import _resolve_vector_inputs
        from wyrd.generators.kenning.runtime.proportions import (
            NewName,
            _native_form_for_morpheme_id,
            _resolve_surface,
        )
        from wyrd.generators.kenning.runtime.vector_name_select import _weighted_choice
        from wyrd.generators.kenning.runtime.word import _position_form

        usages, wi, mi = _validate_target(params)
        knobs = _parse_knobs(params)
        name_gen, _ = _load_culture(knobs["culture"])
        meaning_db = name_gen.meaning_db

        # Resolve every slot's canonical display sibling (the same
        # _rank_siblings pick to_dict anchors on) — held slots contribute
        # their tags (cohesion / mood context) + their folds (exclusions).
        ranked_first: list[list[Any]] = [
            [next(iter(_rank_siblings(_resolve_surface(meaning_db, u))), None) for u in word]
            for word in usages
        ]
        old_meaning = ranked_first[wi][mi]
        position = _derived_position(len(usages[wi]), mi)
        qualifier = _slot_qualifier(old_meaning, usages[wi][mi])
        bucket_key = _bucket_key(position, qualifier, single=len(usages[wi]) == 1)
        # wyrd-c6o1.2 / wyrd-rogd.13 (D43): a BARE word in a multi-word name draws
        # from the per-WORD-position bare pool — its bucket key carries wp-pre /
        # wp-inner / wp-post derived from the word's index. Initial generation adds
        # this in _flatten_struct_slots; the re-roll must too, or a re-rolled bare
        # word samples the un-positioned pool and can land at a word-position it's
        # never attested in ('Parva Wigston' shapes). Single-word names → None → no
        # extension (no word-sequence evidence), matching the build-side tally.
        word_position = word_position_for(wi, len(usages))
        if position == "bare" and word_position is not None:
            bucket_key = (*bucket_key, f"{WORD_POSITION_KEY_PREFIX}{word_position}")

        request, priors, pack_meaning_dbs = _resolve_vector_inputs(
            culture=knobs["culture"],
            tags=knobs["tags"],
            mood=knobs["mood"],
            harshness=knobs["harshness"],
            era_record_cutoff=knobs["era_record_cutoff"],
            stratum=knobs["stratum"],
            priors_path=knobs["priors_path"],
            scoring_weights_raw=knobs["scoring_weights_raw"],
            packs_raw=knobs["packs_raw"],
        )
        non_position_eligible, slot_base_scores = name_gen._build_vector_pools(
            request,
            frozenset(knobs["exclude_tags"]),
            pack_meaning_dbs or None,
            knobs["include_unglossed"],
        )

        prior_tags = frozenset(
            t
            for w, word in enumerate(ranked_first)
            for e, m in enumerate(word)
            if m is not None and (w, e) != (wi, mi)
            for t in m.tags
        )
        # wyrd-c6o1.4: a --tag reserves ONE slot per name. On a re-roll, restrict
        # the candidate pool to tagged morphemes ONLY when this slot is the name's
        # SOLE tag-carrier (no held slot satisfies the tag) — so re-rolling it
        # can't drop the name's >=1-tagged guarantee. Restricting at the pool's
        # membership level (vs a post-hoc filter) keeps it cohesion-drop-safe,
        # matching the generate path. A free slot (tag satisfied elsewhere) keeps
        # require_tags empty and samples freely.
        required_tags = request.gate.required_tags
        require_tags = (
            required_tags if required_tags and required_tags.isdisjoint(prior_tags) else frozenset()
        )
        weighted = _weighted_pool_with_fallback(
            name_gen,
            non_position_eligible,
            position=position,
            qualifier=qualifier,
            bucket_key=bucket_key,
            request=request,
            priors=priors,
            knobs=knobs,
            prior_tags=prior_tags,
            slot_base_scores=slot_base_scores,
            require_tags=require_tags,
        )

        # Operator decision (wyrd-y0lx): exclude the replaced morpheme and
        # every morpheme already in use elsewhere in the name. Fold by
        # dash-stripped lowercase (the _diversify_repeats convention), on
        # both the candidate's modern usage and its native form so a
        # native-rendered collision ("tūn" vs a held "tūn") is excluded too.
        used_folds = _used_folds(usages, ranked_first)
        weighted = [
            (m, w)
            for m, w in weighted
            if not _collides(m, used_folds, _native_form_for_morpheme_id)
        ]
        if not weighted:
            raise NoEligibleReplacementError(
                "no eligible replacement morpheme — every candidate for this "
                "slot is either already in use in the name or filtered by the "
                "generation context (culture / tags / era / stratum)"
            )

        # The hard --tag restriction is applied at the pool's membership level
        # above (require_tags); the mood is a soft post-hoc re-weight on top, so on
        # a sole-carrier-of-both slot the hard tag is already enforced and the mood
        # re-weights within the tagged pool.
        weighted = _apply_mood_restriction(weighted, request.mood_tags, prior_tags)

        rng = rng_for(seed)
        pick = _weighted_choice(rng, weighted)
        if pick is None:
            raise NoEligibleReplacementError(
                "no eligible replacement morpheme (all candidate scores were zero)"
            )

        # Rebuild the full name with the target slot replaced. Held slots
        # keep their supplied surfaces verbatim (rendered=None → __str__
        # falls back to the supplied usage), so prior client-side transforms
        # aren't re-derived server-side; only the target slot gets the full
        # render treatment (era / native / D8 / D18).
        new_words = [list(word) for word in usages]
        new_words[wi][mi] = _position_form(pick, position)
        picked_ids: list[list[str | None]] = [[None] * len(word) for word in usages]
        picked_ids[wi][mi] = getattr(pick, "morpheme_id", None)
        struct = tuple(
            tuple(
                _bucket_key(
                    _derived_position(len(word), e),
                    _slot_qualifier(ranked_first[w][e], word[e]),
                    single=len(word) == 1,
                )
                for e in range(len(word))
            )
            for w, word in enumerate(usages)
        )
        new_name = NewName(struct, meaning_db, new_words, picked_ids=picked_ids)
        # Exclusions above already guarantee no duplicate; diversification
        # must not re-pick the HELD slots (the user's fixed context).
        new_name._diversified = True
        name_gen._apply_render(
            rng,
            new_name,
            knobs["spelling_variety"],
            knobs["inflection_density"],
            knobs["era_render_language"],
            knobs["era_requested"],
        )
        _strip_held_renders(new_name, wi, mi)

        return [
            GenerationResult(
                result=str(new_name),
                result_modern=new_name.modern_name(),
                explanation=new_name.description(),
                components=new_name.components(),
                morphemes_by_word=new_name.to_dict()["words"],
            )
        ]


def _validate_target(params: dict[str, Any]) -> tuple[list[list[str]], int, int]:
    """Validate ``words`` / ``word_index`` / ``morpheme_index`` and return the
    per-word usage strings plus the target indices. Raises ValueError (→ 400
    bad_params) on any malformed input."""
    words = params.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError("words is required (the morphemes_by_word breakdown)")
    usages: list[list[str]] = []
    for word in words:
        if not isinstance(word, list) or not word:
            raise ValueError("words must be a non-empty list of non-empty morpheme lists")
        word_usages = []
        for morph in word:
            usage = (morph.get("usage") or "").strip() if isinstance(morph, dict) else ""
            if not usage:
                raise ValueError("every morpheme needs a non-empty usage")
            word_usages.append(usage)
        usages.append(word_usages)
    try:
        wi = int(params.get("word_index"))
        mi = int(params.get("morpheme_index"))
    except (TypeError, ValueError) as e:
        raise ValueError("word_index and morpheme_index must be integers") from e
    if not (0 <= wi < len(usages)) or not (0 <= mi < len(usages[wi])):
        raise ValueError(f"target slot ({wi}, {mi}) is out of range for the supplied words")
    return usages, wi, mi


def _parse_knobs(params: dict[str, Any]) -> dict[str, Any]:
    """Coerce the generation-context knobs, mirroring Kenning.generate's
    parsing so the regeneration pool resolves under the same context as the
    original roll."""
    culture = params.get("culture", "english")
    raw_tags = _coerce_str_list(params.get("tags"), "tags")
    moods = _coerce_str_list(params.get("mood"), "mood")
    include_fiction = _coerce_bool(params.get("include_fiction", False))
    # Lazy import (the wyrd.generators.kenning.__init__ cycle; same reason as the
    # function-local _resolve_vector_inputs import in generate_all above).
    from wyrd.generators.kenning.generators.kenning import _request_era_midpoint

    # D46: the resolved era range's END is the record-entry cutoff (a
    # bounded era replaces only with morphemes "in the record by then");
    # the call doubles as request validation (typo'd era → 4xx).
    era_range = _resolve_era_param(params.get("era"), culture)
    return {
        "culture": culture,
        "tags": list(raw_tags),
        "mood": tuple(moods),
        "harshness": _coerce_float(params.get("harshness")),
        "cohesion": _coerce_float(params.get("cohesion")),
        "novelty": _coerce_float(params.get("novelty")),
        "spelling_variety": _coerce_float(params.get("spelling_variety")),
        "inflection_density": _coerce_float(params.get("inflection_density")),
        "include_unglossed": _coerce_bool(params.get("include_unglossed", False)),
        "exclude_tags": () if include_fiction else (_FICTION_TAG,),
        "era_record_cutoff": era_range[1] if era_range else None,
        # wyrd-951i: same priors-baseline era cell the main path leans toward
        # (wyrd-3tvd), so a single-slot re-roll scores against the requested era's
        # naming fashion instead of the wildcard-0 cell.
        "era_midpoint": _request_era_midpoint(era_range),
        "era_render_language": _resolve_era_render_language(params.get("era"), culture),
        "era_requested": bool(params.get("era")),
        "stratum": _resolve_stratum_param(params.get("stratum"), culture),
        "priors_path": params.get("priors_path"),
        "scoring_weights_raw": params.get("scoring_weights"),
        "packs_raw": params.get("packs") or [],
    }


def _weighted_pool_with_fallback(
    name_gen: Any,
    non_position_eligible: list,
    *,
    position: str,
    qualifier: str | None,
    bucket_key: tuple[str, ...],
    request: Any,
    priors: Any,
    knobs: dict[str, Any],
    prior_tags: frozenset[str],
    slot_base_scores: dict[tuple, list[tuple[Any, float]]],
    require_tags: frozenset[str] = frozenset(),
) -> list[tuple[Any, float]]:
    """The weighted candidate pool for the target slot, with the bucket-key
    fallback: when the reconstructed bucket key doesn't exist in this
    culture's frequency tables (the supplied breakdown can carry shapes the
    proportions never recorded — e.g. a post-transform structure), degrade
    to the unweighted score-only pool rather than failing. The degrade is
    logged at WARNING so an operator can see the frequency layer was
    bypassed (a `dispatch ok` line alone would hide it).

    ``require_tags`` (wyrd-c6o1.4) restricts the pool to tagged morphemes at the
    membership level — set by the caller for a sole-tag-carrier re-roll so the
    name's >=1-tagged guarantee survives the re-roll even under cohesion (the same
    cohesion-drop-safe restriction the generate path uses)."""
    # Lazy import — same wyrd.generators.kenning.__init__ cycle as
    # generate_all's imports (see the comment there).
    from wyrd.generators.kenning.runtime.vector_name_select import _slot_weighted_pool

    weighted = _slot_weighted_pool(
        non_position_eligible,
        slot_position=position,
        slot_qualifier=qualifier,
        slot_bucket_key=bucket_key,
        request=request,
        priors=priors,
        cohesion=knobs["cohesion"],
        cohesion_table=name_gen.cohesion_table or None,
        usage_frequency_by_bucket=name_gen.usage_frequency_by_bucket,
        novelty=knobs["novelty"],
        prior_tags=prior_tags,
        slot_base_scores=slot_base_scores,
        require_tags=require_tags,
        era_midpoint=knobs["era_midpoint"],
    )
    if weighted:
        return weighted
    _logger.warning(
        "kenning_regenerate bucket_key_miss position=%s qualifier=%s bucket_key=%r "
        "— falling back to the unweighted score-only pool",
        position,
        qualifier,
        bucket_key,
    )
    return _slot_weighted_pool(
        non_position_eligible,
        slot_position=position,
        slot_qualifier=qualifier,
        slot_bucket_key=None,
        request=request,
        priors=priors,
        cohesion=knobs["cohesion"],
        cohesion_table=name_gen.cohesion_table or None,
        usage_frequency_by_bucket=None,
        novelty=knobs["novelty"],
        prior_tags=prior_tags,
        slot_base_scores=None,
        require_tags=require_tags,
        era_midpoint=knobs["era_midpoint"],
    )


def _apply_mood_restriction(
    weighted: list[tuple[Any, float]],
    mood_tags: dict[str, float],
    prior_tags: frozenset[str],
) -> list[tuple[Any, float]]:
    """wyrd-4rp8 context rule: when the request carries mood tags and the
    slot being replaced is the only mood-carrier among the held slots
    (no prior tag intersects the mood set), keep the theme — restrict to
    mood-tagged candidates, re-weighted by mood fit. Graceful: an empty
    restriction (no candidate carries any mood tag) falls back to the
    full pool unchanged."""
    # Lazy import — same wyrd.generators.kenning.__init__ cycle as
    # generate_all's imports (see the comment there).
    from wyrd.generators.kenning.runtime.vector_name_select import _mood_morpheme_weight

    mood_tag_set = frozenset(mood_tags)
    if not mood_tag_set or not mood_tag_set.isdisjoint(prior_tags):
        return weighted
    mood_restricted = [
        (m, w * _mood_morpheme_weight(m, mood_tags))
        for m, w in weighted
        if not mood_tag_set.isdisjoint(m.tags)
    ]
    return mood_restricted or weighted


def _derived_position(word_len: int, index: int) -> str:
    """D40: position is DERIVED from where the slot lands in its word —
    sole piece → bare, first → pre, last → post, interior → inner."""
    if word_len == 1:
        return "bare"
    if index == 0:
        return "pre"
    if index == word_len - 1:
        return "post"
    return "inner"


def _slot_qualifier(meaning: Any | None, usage: str) -> str | None:
    """Infer the slot's qualifier flag from the morpheme being held /
    replaced, mirroring ``Meaning.key``'s bucket-assignment if/elif: a
    name-tagged morpheme routes through "name" even when saint-tagged; the
    literal Saint- morpheme is "saint". An unresolvable usage falls back to
    the literal-saint string check (the one half that needs no Meaning)."""
    if meaning is not None and meaning.is_name():
        return "name"
    if usage.replace("-", "").lower() == "saint":
        return "saint"
    return None


def _bucket_key(position: str, qualifier: str | None, *, single: bool) -> tuple[str, ...]:
    """Reconstruct the proportions bucket key for a slot, matching
    ``word_to_key``'s element shape: ``[location, "name"?, "saint"?,
    "single"?]`` (the single flag rides on single-morpheme words)."""
    key = [position]
    if qualifier == "name":
        key.append("name")
    elif qualifier == "saint":
        key.append("saint")
    if single:
        key.append("single")
    return tuple(key)


def _used_folds(usages: list[list[str]], ranked_first: list[list[Any]]) -> set[str]:
    """Every fold already in use in the name: the supplied display surfaces
    (which may be post-transform era forms) plus each slot's resolved
    modern bucket usage."""
    folds: set[str] = set()
    for word, word_meanings in zip(usages, ranked_first, strict=True):
        for usage, meaning in zip(word, word_meanings, strict=True):
            folds.add(usage.replace("-", "").lower())
            if meaning is not None:
                folds.add(meaning.usage.replace("-", "").lower())
    folds.discard("")
    return folds


def _collides(meaning: Any, used_folds: set[str], native_form_for: Any) -> bool:
    """A candidate collides when its modern usage fold OR its native-form
    fold is already in use — covering both the modern companion and the
    D41 native rendering.

    ``native_form_for`` is ``proportions._native_form_for_morpheme_id``,
    injected by the caller (which lazy-imports it once per request) so this
    per-candidate filter predicate doesn't re-run the import machinery for
    every pool entry."""
    if meaning.usage.replace("-", "").lower() in used_folds:
        return True
    mid = getattr(meaning, "morpheme_id", None)
    if mid:
        native = native_form_for(mid)
        if native and native.replace("-", "").lower() in used_folds:
            return True
    return False


def _null_non_target(grid: list[list[Any]], wi: int, mi: int) -> None:
    """Set every entry of a NewName-parallel 2D grid to None except the
    target slot ``(wi, mi)``."""
    for w, word in enumerate(grid):
        for e in range(len(word)):
            if (w, e) != (wi, mi):
                word[e] = None


def _strip_held_renders(new_name: Any, wi: int, mi: int) -> None:
    """Null out the render substitutions on every HELD slot so the response
    preserves the caller's supplied surfaces verbatim (rendered=None →
    ``__str__`` / to_dict fall back to the usage we were given). Only the
    regenerated slot keeps its freshly-computed render."""
    if new_name.rendered is not None:
        _null_non_target(new_name.rendered, wi, mi)
    if new_name.inflection_labels is not None:
        _null_non_target(new_name.inflection_labels, wi, mi)
