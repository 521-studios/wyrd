"""``wyrd kenning lexicon browse`` — read-only navigation sub-group (wyrd-7oma).

Grep-friendly markdown views of toponyms, etymons, decompositions, and
sources. Use BEFORE running a curation pass (lemma normalize, dead-rando
prune, language retag) to verify which rows you're about to edit.

The browse subcommand modules live alongside this ``__init__`` (each
exposes an ``add_to(browse_group)`` hook). The parent
``cli/lexicon/__init__.py`` calls this package's ``add_to(lexicon)`` to
mount the browse group on the @lexicon click group.
"""

from __future__ import annotations

import click

from wyrd.generators.kenning.cli.lexicon.browse import (
    decomposition as _decomposition_module,
)
from wyrd.generators.kenning.cli.lexicon.browse import etymon as _etymon_module
from wyrd.generators.kenning.cli.lexicon.browse import source as _source_module
from wyrd.generators.kenning.cli.lexicon.browse import toponym as _toponym_module


@click.group("browse")
def lexicon_browse() -> None:
    """Read-only navigation of the lexicon DB (wyrd-7oma).

    Grep-friendly markdown views of toponyms, etymons, decompositions,
    and sources. Use BEFORE running a curation pass (lemma normalize,
    dead-rando prune, language retag) to verify which rows you're
    about to edit.
    """


# Register the four browse subcommands on the @lexicon_browse group.
_etymon_module.add_to(lexicon_browse)
_toponym_module.add_to(lexicon_browse)
_decomposition_module.add_to(lexicon_browse)
_source_module.add_to(lexicon_browse)


def add_to(parent: click.Group) -> None:
    """Register the ``browse`` sub-group on the parent @lexicon group."""
    parent.add_command(lexicon_browse)
