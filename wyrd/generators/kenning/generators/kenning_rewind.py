"""The `kenning-rewind` Generator — render a name at multiple era stops (D33)."""

from __future__ import annotations

from typing import Any

from wyrd.generators.kenning import (
    _LEGEND,
    _bundle_era_form,
    _load_meanings,
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
            },
            "required": ["name"],
        }

    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult:
        results = self.generate_all(params, seed)
        return results[0]

    def generate_all(self, params: dict[str, Any], seed: int) -> list[GenerationResult]:
        from wyrd.generators.kenning.era.cells import canonical_language_for_cell
        from wyrd.generators.kenning.era.rewind import (
            _is_free_particle,
            render_form_particle_pairs,
        )

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
            unaccounted: list[str] = []
            # Per-word morpheme grouping so render_form_particle_pairs
            # gets called once per input word; outputs join across
            # words with a space (preserves operator's input shape
            # per wyrd-t2bh).
            words_pairs: list[list[tuple[str, bool]]] = []
            for word_str in text.split():
                candidates = name_obj.words.get(word_str, [])
                if not candidates:
                    unaccounted.append(word_str)
                    continue
                word_pairs: list[tuple[str, bool]] = []
                for chunk in candidates[0].word:
                    if isinstance(chunk, Meaning):
                        form = _bundle_era_form(chunk, target_language)
                        # wyrd-8qbi: at modern, prefer the morpheme's
                        # canonical (modern_usage with dashes stripped)
                        # over _bundle_era_form's cluster pick — the
                        # operator already knows the modern form.
                        if target_language == "modern-english":
                            form = chunk.usage.replace("-", "")
                        # wyrd-2pio: strip leading/trailing positional
                        # hyphens (the per-morpheme positional marker
                        # '-ham' style) so they don't leak into the
                        # joined output as '-healdteoruell'-style
                        # artifacts. Matches era.rewind._pick_form's
                        # strip behavior.
                        form = form.strip("-")
                        word_pairs.append((form, _is_free_particle(chunk)))
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
