"""CLI commands for Kenning. Mounted under ``wyrd kenning <subcommand>``.

This module is the final back-compat shim from the wyrd-g143 split.
The top-level ``@cli`` click group lives here; every other ``@cli`` or
``@lexicon`` subcommand lives in its own module under
``wyrd/generators/kenning/cli/`` (top-level) or
``wyrd/generators/kenning/cli/lexicon/`` (lexicon group). Each
per-command module exposes an ``add_to(parent: click.Group)`` hook
this file calls to wire the command onto the @cli group.

Adding a new top-level subcommand: create ``cli/<name>.py`` exposing
``add_to(parent)``, then add a single-line module import + an
``<name>_module.add_to(cli)`` call below. The @lexicon sub-group
follows the same pattern at ``cli/lexicon/``.

The few private helpers re-exported below exist purely for tests that
grab CLI internals directly (e.g. ``from wyrd.generators.kenning.cli
import _select_parser_and_run``). The list is narrow on purpose;
extending it means coordinating with the tests that reach in.
"""

from __future__ import annotations

import click

from wyrd.generators.kenning.cli import (
    add_meaning as _add_meaning_module,
)
from wyrd.generators.kenning.cli import (
    creature as _creature_module,
)
from wyrd.generators.kenning.cli import (
    dump_structures as _dump_structures_module,
)
from wyrd.generators.kenning.cli import (
    era_map as _era_map_module,
)
from wyrd.generators.kenning.cli import (
    explain as _explain_module,
)
from wyrd.generators.kenning.cli import (
    generate as _generate_module,
)
from wyrd.generators.kenning.cli import (
    rebuild_proportions as _rebuild_proportions_module,
)
from wyrd.generators.kenning.cli import (
    rewind as _rewind_module,
)
from wyrd.generators.kenning.cli import (
    unaccounted as _unaccounted_module,
)
from wyrd.generators.kenning.cli import (
    validate_meanings as _validate_meanings_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    add_to as _register_lexicon_group,
)
from wyrd.generators.kenning.cli.lexicon import (
    lexicon,  # noqa: F401  (re-exported for tests that grab the lexicon group)
)

# Back-compat re-exports of private helpers tests grab directly.
# ``tests/test_kenning_strata`` reaches in for _build_case_update;
# ``tests/test_kenning_cooccurrence`` reaches in for the rebuild-
# proportions tag-cooccurrence helpers; ``tests/test_kenning_dictionary_parser_numbered_list``
# and ``tests/test_kenning_decomposition`` reach in for the utils
# helpers; ``tests/test_kenning_lexicon_mine_concurrency`` reaches in
# for _mine_entries; ``tests/test_kenning_review_candidates`` reaches
# in for _select_review_candidates.
from wyrd.generators.kenning.cli.lexicon.classify_stratum import (  # noqa: F401
    _build_case_update,
)
from wyrd.generators.kenning.cli.lexicon.mine_skeat import (  # noqa: F401
    _mine_entries,
)
from wyrd.generators.kenning.cli.lexicon.review import (  # noqa: F401
    PAID_CONCURRENCY_WARN_THRESHOLD,
    _select_review_candidates,
    _warn_if_paid_concurrency_too_high,
)
from wyrd.generators.kenning.cli.rebuild_proportions import (  # noqa: F401
    _ordered_tag_pairs,
    _proportions_from,
)
from wyrd.generators.kenning.cli.utils import (  # noqa: F401  (_decompose_corpus)
    _DEFAULT_LEXICON_PATH,
    _decompose_corpus,
    _load_meanings_data,
    _select_parser_and_run,
)


@click.group()
def cli() -> None:
    """Kenning — town name generation and authoring tools."""


# Register the extracted top-level subcommands on the @cli group.
_add_meaning_module.add_to(cli)
_creature_module.add_to(cli)
_dump_structures_module.add_to(cli)
_era_map_module.add_to(cli)
_explain_module.add_to(cli)
_generate_module.add_to(cli)
_rebuild_proportions_module.add_to(cli)
_rewind_module.add_to(cli)
_unaccounted_module.add_to(cli)
_validate_meanings_module.add_to(cli)

# Register the @lexicon sub-group + all its subcommands. Every
# @lexicon.command body lives in a per-command module under
# cli/lexicon/, registered by cli/lexicon/__init__.py.
_register_lexicon_group(cli)
