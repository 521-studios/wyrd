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
    components: list[dict[str, Any]] = field(default_factory=list)


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

    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the request body. SPA renders forms from this."""

    @abstractmethod
    def generate(self, params: dict[str, Any], seed: int) -> GenerationResult: ...


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
