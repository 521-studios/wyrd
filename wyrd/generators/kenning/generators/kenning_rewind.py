"""The `kenning-rewind` Generator — render a name at multiple era stops (D33)."""

from __future__ import annotations

from typing import Any

from wyrd.generators.kenning import (
    _LEGEND,
    _bundle_era_form,
    _decomposition_matcher,
    _load_meanings,
    _required_name,
)
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.name import Name
from wyrd.registry import GenerationResult, Generator


class KenningRewind(Generator):
    """wyrd-obpw Phase 3.3 — bundle-driven time-rewind explainer.

    Mirrors the CLI rewinder (``wyrd/generators/kenning/era/rewind.py``)
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
                # wyrd-y9aa: pre-picked morpheme list (same shape as
                # 'kenning generate --json' output's per-result 'words'
                # / API envelope's morphemes_by_word). When provided,
                # skip the trie-based string decomposition + use the
                # supplied picks directly — fixes the 'render then age
                # this name' SPA flow where re-decomposition can land
                # on a different etymological cluster than the
                # generator picked.
                "words": {
                    "type": "array",
                    "description": (
                        "Optional: pre-picked morpheme breakdown from a "
                        "prior 'kenning generate' (the morphemes_by_word "
                        "API field). When provided, the rewinder skips "
                        "trie re-decomposition and uses these picks."
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
            },
            "required": ["name"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        results = self.generate_all(params, seed)
        return results[0]

    def generate_all(self, params: dict[str, Any], seed: int) -> list[GenerationResult]:
        from wyrd.generators.kenning.era.cells import canonical_language_for_cell
        from wyrd.generators.kenning.era.rewind import render_form_particle_pairs

        text = _required_name(params)
        supplied_words = params.get("words")
        meaning_db, _ = _load_meanings()
        # wyrd-y9aa: branch on whether the caller supplied pre-picked
        # morphemes. When yes, build the per-word Meaning lists from
        # meaning_db lookup (no trie); when no, fall back to the
        # existing string-decompose path. supplied_missing carries
        # usages that the bundle has dropped since the JSON was
        # captured — pre-seeded into each era's unaccounted so the
        # 'lacked era data' surface is the same for both paths.
        supplied_missing: list[str] = []
        if supplied_words:
            per_word_meanings, supplied_missing = _meanings_from_supplied_words(
                supplied_words, meaning_db
            )
            name_obj = None
        else:
            name_obj = Name(text)
            # wyrd-04oi: reuse the pickled matcher published beside the runtime
            # DB (cold-start: skip the trie rebuild); fail-safe to a build.
            name_obj.find_meaning(meaning_db, reduce=True, trie=_decomposition_matcher(meaning_db))
            per_word_meanings = None
        era_stops = (
            ("english", "oe-late"),
            ("english", "me"),
            ("english", "modern"),
        )
        # Per-era rendered output. Each era-stop becomes one
        # GenerationResult so the SPA can render them as a sequence.
        #
        # wyrd-2pio: reuse the shared rewinder renderer from
        # era.rewind so the SPA-facing path (and downstream era-map)
        # inherits the same smart-join + title-case + modern round-
        # trip + free-particle treatment that the CLI rewinder uses.
        # Pre-fix this module had its own hyphenated-join (the
        # pre-wyrd-085k shape) plus no title-case, so era-map output
        # diverged from the CLI rewind output for the same input.
        outputs: list[GenerationResult] = []
        for family, cell in era_stops:
            target_language = canonical_language_for_cell(family, cell)
            morpheme_components: list[dict[str, Any]] = []
            # wyrd-y9aa: pre-seed unaccounted with supplied-path
            # bundle-dropped usages so the 'lacked era data' surface
            # mirrors the trie path's bookkeeping.
            unaccounted: list[str] = list(supplied_missing)
            # Per-word morpheme grouping so render_form_particle_pairs
            # gets called once per input word; outputs join across
            # words with a space (preserves operator's input shape
            # per wyrd-t2bh).
            words_pairs: list[list[tuple[str, bool]]] = []
            if per_word_meanings is not None:
                # wyrd-y9aa: pre-picked morphemes path.
                for word_meanings in per_word_meanings:
                    word_pairs = _meanings_to_word_pairs(
                        word_meanings,
                        target_language,
                        morpheme_components,
                    )
                    if word_pairs:
                        words_pairs.append(word_pairs)
            else:
                for word_str in text.split():
                    candidates = name_obj.words.get(word_str, [])
                    if not candidates:
                        unaccounted.append(word_str)
                        continue
                    word_pairs = _decompose_word_at_era(
                        candidates[0],
                        target_language,
                        morpheme_components,
                        unaccounted,
                    )
                    if word_pairs:
                        words_pairs.append(word_pairs)
            # wyrd-t2bh: smart-join at OE/ME (historical scribal
            # pattern); simple concat at modern (round-trip operator
            # input shape).
            smart_join = target_language != "modern-english"
            word_renders = [
                render_form_particle_pairs(word, smart_join=smart_join) for word in words_pairs
            ]
            rendered = " ".join(r for r in word_renders if r)
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


def _decompose_word_at_era(
    word: Any,
    target_language: str | None,
    morpheme_components: list[dict[str, Any]],
    unaccounted: list[str],
) -> list[tuple[str, bool]]:
    """wyrd-2pio: per-word morpheme walk for KenningRewind's trie
    path. For each chunk: if Meaning, delegate per-morpheme work to
    ``_apply_meaning`` (the shared 8qbi / 2pio / 085k / 17t pipeline);
    if leftover str, push to unaccounted. wyrd-y9aa round 2: extracted
    the per-Meaning body to eliminate the duplicate with
    ``_meanings_to_word_pairs``."""
    word_pairs: list[tuple[str, bool]] = []
    for chunk in word.word:
        if isinstance(chunk, Meaning):
            word_pairs.append(_apply_meaning(chunk, target_language, morpheme_components))
        elif isinstance(chunk, str) and chunk:
            unaccounted.append(chunk)
    return word_pairs


def _meanings_from_supplied_words(
    supplied_words: list[Any],
    meaning_db: dict[str, list[Meaning]],
) -> tuple[list[list[Meaning]], list[str]]:
    """wyrd-y9aa: resolve the per-word morpheme picks from a 'kenning
    generate --json' / API morphemes_by_word payload into Meaning
    objects via meaning_db lookup. Returns (per_word_meanings,
    missing_usages) — missing_usages carries usage strings the bundle
    has dropped since the JSON was captured (the supplied list named
    them but meaning_db doesn't know them anymore), surfaced to the
    caller for unaccounted bookkeeping. First Meaning wins for
    ambiguous usages, mirroring NewName.to_dict.
    """
    # wyrd-7cvv: dash/case-insensitive fallback index. morphemes_by_word
    # carries the POSITIONAL surface ("by" at a word end, "m-" as a prefix)
    # while meaning_db is keyed by the canonical form ("-by", "m"). An exact
    # lookup misses those and the morpheme gets dropped — which made rewind
    # silently shed real morphemes and render only a fragment ("Mhead By" →
    # "Hēæd"). Build a normalized-key index (first key per dash-stripped,
    # lowercased form wins, mirroring first-Meaning-wins) so a positional
    # usage still resolves. Built lazily on the first miss.
    #
    # First-wins is benign: colliding norms in the bundle are same-morpheme
    # positional variants of ONE Meaning ("-a"/"A-", "-by"/"By-"), and this
    # only fires on the exact-lookup MISS path (an exact key always wins
    # above). A future bundle introducing a genuinely different morpheme with
    # a colliding norm could mis-resolve here — acceptable given the data.
    norm_index: dict[str, list[Meaning]] | None = None

    # wyrd-om67: rank a usage's sibling Meanings the SAME way NewName.to_dict
    # does (drop modern_english homographs, prefer older strata) and take the
    # canonical, so the rewind anchors on the etymon the SPA actually DISPLAYS
    # for this morpheme — not meaning_db's raw first sibling, which can be a
    # different-language homograph (e.g. "-ton" picking a Celtic etymon → OE
    # "Wertūn" came out as "Werettan"). Lazy import to dodge the circular dep
    # with wyrd.generators.kenning.__init__ at module load.
    from wyrd.generators.kenning import _rank_siblings

    def _norm(s: str) -> str:
        return s.strip("-").lower()

    per_word: list[list[Meaning]] = []
    missing: list[str] = []
    for word_morphs in supplied_words or []:
        word_meanings: list[Meaning] = []
        for morph in word_morphs or []:
            usage = morph.get("usage") if isinstance(morph, dict) else None
            if not usage:
                continue
            candidates = meaning_db.get(usage, [])
            if not candidates:
                if norm_index is None:
                    norm_index = {}
                    for key, ms in meaning_db.items():
                        norm_index.setdefault(_norm(key), ms)
                candidates = norm_index.get(_norm(usage), [])
            if candidates:
                # NOTE: to_dict resolves the candidate set via _resolve_surface
                # (which unions all dash-variant keys for a bare surface) while
                # this path uses exact get() + the norm_index fallback. For the
                # typical one-dash-variant-per-morpheme bundle these match; a
                # morpheme stored under several distinct dash-variants could
                # diverge — a future pass could unify both on _resolve_surface.
                ranked = _rank_siblings(candidates)
                word_meanings.append(ranked[0] if ranked else candidates[0])
            else:
                missing.append(usage)
        per_word.append(word_meanings)
    return per_word, missing


def _apply_meaning(
    meaning: Meaning,
    target_language: str | None,
    morpheme_components: list[dict[str, Any]],
) -> tuple[str, bool]:
    """wyrd-y9aa: per-morpheme render pipeline shared by both
    KenningRewind paths (trie-decomposed + supplied-words).

    For ONE Meaning: pick era-form (or canonical at modern,
    wyrd-8qbi), strip positional hyphens (wyrd-2pio), classify as
    free particle (wyrd-085k), compute respelling (wyrd-17t),
    append to morpheme_components (side effect). Returns the
    (form, is_particle) tuple the renderer needs.

    Pulled out of _decompose_word_at_era + _meanings_to_word_pairs
    so the two iteration shapes stay thin wrappers; pre-extraction
    the per-Meaning body was duplicated between the two helpers.
    """
    from wyrd.generators.kenning.era.rewind import _is_free_particle

    form = _bundle_era_form(meaning, target_language)
    if target_language == "modern-english":
        form = meaning.usage.replace("-", "")
    form = form.strip("-")
    respelling = meaning.respelling_for(form, target_language) if target_language else None
    morpheme_components.append(
        {
            "form": form,
            "respelling": respelling,
            "language": target_language,
            # wyrd-7cvv: the morpheme's ORIGINAL modern usage, so a consumer
            # (the SPA rewind transform) can align each rewound form back to
            # the input morpheme it came from — and omit input morphemes the
            # rewind dropped (no era reflex) rather than mismatch the rewound
            # name against the full original morpheme set.
            "canonical": meaning.usage,
        }
    )
    return form, _is_free_particle(meaning)


def _meanings_to_word_pairs(
    word_meanings: list[Meaning],
    target_language: str | None,
    morpheme_components: list[dict[str, Any]],
) -> list[tuple[str, bool]]:
    """wyrd-y9aa: per-word iteration shape for the pre-picked-input
    path. Thin wrapper over ``_apply_meaning`` — no chunk-walking or
    str fallback since the upstream JSON shape only ever contains
    Meaning-shaped entries."""
    return [
        _apply_meaning(meaning, target_language, morpheme_components) for meaning in word_meanings
    ]
