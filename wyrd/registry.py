"""Generator plugin contract and registry."""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    """A single generator output. Wrapped by envelope.envelope() for the API."""

    result: str
    explanation: str = ""
    # wyrd-24s6 (D38): the MODERN rendering of the same name, beside the
    # canonical NATIVE ``result``. Kenning surfaces both so the SPA can show
    # native (primary) + modern (secondary). None for generators that don't
    # distinguish the two (the value then collapses to ``result`` downstream).
    result_modern: str | None = None
    components: list[dict[str, Any]] = field(default_factory=list)
    # wyrd-h8k1: KenningExplain marks the canonical decomposition reading
    # so the SPA can render it distinctly. ``canonical_source`` carries
    # the rule that picked it (``'scholar'``, ``'unique-zero-unaccounted'``,
    # or ``'scholar-disagreement'``). Both default to falsy for generators
    # that don't surface canonical info.
    canonical: bool = False
    canonical_source: str | None = None
    # wyrd-cp2d: per-word morpheme breakdown preserving the bucket-key
    # picks the generator committed to (with per-language source-lemma
    # data). Optional; populated by name-shaped generators (Kenning)
    # whose output downstream commands (rewind, era-map) might want
    # to re-process without re-running the trie decomposition.
    # ``None`` for generators that don't surface morpheme structure.
    # Per-word grouping preserves the operator's space-shape so
    # multi-word names round-trip cleanly through the json continuity
    # flow.
    morphemes_by_word: list[list[dict[str, Any]]] | None = None


class Generator(ABC):
    """Base class for a wyrd generator plugin.

    Each generator owns its own input shape but returns a uniform GenerationResult.
    The Flask app discovers subclasses via wyrd.generators auto-import and lists
    them on /api/manifest. The dispatcher calls generate() with a coerced params
    dict and a resolved seed.
    """

    name: str
    display_name: str
    description: str

    # Optional long-form explanation for the SPA's pre-roll "About" panel. May
    # contain a small subset of HTML (paragraphs, em, strong). None means the
    # SPA shows no panel for this generator.
    details: str | None = None

    # Optional legend of source/category codes the generator embeds in its
    # results, surfaced under the results list. Each entry is
    # {"code": "EN", "name": "Old English"}. None means no legend.
    legend: list[dict[str, str]] | None = None

    # When True, the dispatcher uses generate_all() and ignores the count
    # param. Use this when the result count is determined by the input itself
    # (e.g. an explainer that returns every decomposition that matches), not
    # by a "roll N times" knob.
    multi_result: bool = False

    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the request body. SPA renders forms from this."""

    @abstractmethod
    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult: ...

    def generate_all(self, params: dict[str, Any], seed: int) -> list[GenerationResult]:
        """Return a variable-length result set in one call. Override when
        multi_result is True. The default wraps a single generate() so
        single-result generators don't need to implement both."""
        return [self.generate(params, seed)]

    def runtime_version(self) -> dict[str, str] | None:
        """Identify the data/bundle version this generator's output was
        produced against, for defect-report reproducibility (wyrd-dsl5).

        Returned verbatim on the API envelope as ``bundle_version`` and
        echoed back when a user flags a name defective, so a triager can
        tell whether a reported name is still reproducible against the
        current data. ``None`` (the default) means the generator has no
        versioned data layer — the envelope omits the key."""
        return None


_registry: dict[str, Generator] = {}


def register(generator: Generator) -> None:
    if generator.name in _registry:
        raise ValueError(f"Generator already registered: {generator.name}")
    _registry[generator.name] = generator


def get(name: str) -> Generator | None:
    return _registry.get(name)


def all_generators() -> list[Generator]:
    return sorted(_registry.values(), key=lambda g: g.name)


def discover() -> None:
    """Import every subpackage of wyrd.generators so plugins self-register."""
    # Deferred import: wyrd.generators subpackages import register() from this
    # module, so importing them at module load would create a circular import.
    import wyrd.generators as gens

    for module_info in pkgutil.iter_modules(gens.__path__):
        importlib.import_module(f"wyrd.generators.{module_info.name}")
