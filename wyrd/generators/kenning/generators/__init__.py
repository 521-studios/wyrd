"""The kenning Generator subclasses (wyrd-o9qi).

One module per registered ``Generator`` subclass. Each module imports
the helpers it needs back from ``wyrd.generators.kenning`` (the
back-compat shim). The parent ``kenning/__init__.py`` re-exports
every Generator subclass at the end of its body — after all
underscore-prefixed helpers are defined — so the partial-init cycle
resolves cleanly.

Adding a new Generator: create ``generators/<name>.py`` with the
class definition + its needed helpers imported from kenning, then
add ``from wyrd.generators.kenning.generators.<name> import <Class>``
to ``kenning/__init__.py``'s re-export block.
"""

from wyrd.generators.kenning.generators.kenning import Kenning
from wyrd.generators.kenning.generators.kenning_creature import KenningCreature
from wyrd.generators.kenning.generators.kenning_era_map import KenningEraMap
from wyrd.generators.kenning.generators.kenning_explain import KenningExplain
from wyrd.generators.kenning.generators.kenning_regenerate import (
    KenningRegenerateMorpheme,
)
from wyrd.generators.kenning.generators.kenning_render import KenningRender
from wyrd.generators.kenning.generators.kenning_rewind import KenningRewind

__all__ = [
    "Kenning",
    "KenningCreature",
    "KenningEraMap",
    "KenningExplain",
    "KenningRegenerateMorpheme",
    "KenningRender",
    "KenningRewind",
]
