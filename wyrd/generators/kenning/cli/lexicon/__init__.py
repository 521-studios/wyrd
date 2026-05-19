"""The ``wyrd kenning lexicon`` click sub-group.

This package holds the ``lexicon``-scoped CLI subcommands as part of
the wyrd-g143 split. Slice 1b sets up the group definition + the
``add_to`` registration hook; subsequent slices (2-5) move the
75 individual ``@lexicon.command`` subcommands (plus two nested
sub-groups, ``browse`` and ``synsets``) out of the cli back-compat
shim and into per-subcommand modules under this subpackage.

Note: this package's name collides at the directory level with the
sibling ``wyrd.generators.kenning.lexicon`` DB-layer package — they
are distinct absolute import paths (``wyrd.generators.kenning.cli.lexicon``
vs ``wyrd.generators.kenning.lexicon``) and Python's absolute-import
resolution keeps them apart. This package holds CLI subcommands; the
DB-layer package holds the SQLite + enrichment surface.
"""

from __future__ import annotations

import click


@click.group("lexicon")
def lexicon() -> None:
    """Manage the authoring lexicon DB (etymology data store)."""


def add_to(parent: click.Group) -> None:
    """Register the ``lexicon`` sub-group on a parent click group.

    Called once by ``wyrd.generators.kenning.cli.__init__`` while the
    top-level ``@cli`` group is being constructed. Per-subcommand
    modules inside this package decorate against the shared
    ``lexicon`` group; this function wires the group into the parent
    so ``wyrd kenning lexicon ...`` resolves.
    """
    parent.add_command(lexicon, name="lexicon")
