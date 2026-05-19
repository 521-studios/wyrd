"""``wyrd kenning lexicon synsets`` — meaning_synset commands sub-group (wyrd-7tz Phase 1).

Manage the meaning_synset / etymon_meaning_synset tables — fine-grained
semantic equivalence classes used by upcoming generator transforms
(calque, anglicize, drift-toward-X). Distinct from the cognate-cluster
etymon.cognate_id populated by cluster-cognates.

The 5 synsets subcommand modules live alongside this ``__init__``
(each exposes ``add_to(synsets_group)``). The parent
``cli/lexicon/__init__.py`` calls this package's ``add_to(lexicon)``
to mount the synsets group on the @lexicon click group.
"""

from __future__ import annotations

import click

from wyrd.generators.kenning.cli.lexicon.synsets import assign as _assign_module
from wyrd.generators.kenning.cli.lexicon.synsets import candidates as _candidates_module
from wyrd.generators.kenning.cli.lexicon.synsets import list_ as _list_module
from wyrd.generators.kenning.cli.lexicon.synsets import seed as _seed_module
from wyrd.generators.kenning.cli.lexicon.synsets import show as _show_module


@click.group("synsets")
def lexicon_synsets() -> None:
    """Meaning-synset commands (wyrd-7tz Phase 1).

    Manage the meaning_synset / etymon_meaning_synset tables — fine-
    grained semantic equivalence classes used by upcoming generator
    transforms (calque, anglicize, drift-toward-X). Distinct from the
    cognate-cluster etymon.cognate_id populated by cluster-cognates.
    """


_assign_module.add_to(lexicon_synsets)
_candidates_module.add_to(lexicon_synsets)
_list_module.add_to(lexicon_synsets)
_seed_module.add_to(lexicon_synsets)
_show_module.add_to(lexicon_synsets)


def add_to(parent: click.Group) -> None:
    """Register the ``synsets`` sub-group on the parent @lexicon group."""
    parent.add_command(lexicon_synsets)
