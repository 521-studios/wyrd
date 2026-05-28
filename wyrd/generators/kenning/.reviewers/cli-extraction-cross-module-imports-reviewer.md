# cli-extraction-cross-module-imports-reviewer

When `cli.py` is being split into per-subcommand modules under a
`cli/` subpackage (the wyrd-g143 pattern), per-command modules must
NOT import from the `cli/__init__.py` back-compat shim. Imports
must go either (a) to a sibling per-command module
(`from wyrd.generators.kenning.cli.lexicon.review import _build_llm_client`),
(b) to a shared helper module like `cli/utils.py`
(`from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH`),
or (c) to a non-cli sibling package
(`from wyrd.generators.kenning.lexicon import LexiconDB`).

The back-compat shim re-exports test-direct private helpers; it
must not become a load-bearing dependency for the production
graph. Importing from it from a per-command module creates an
import-time partial-init problem (the shim is in the middle of
importing the per-command module when the per-command module
asks the shim to resolve a name) and silently couples the
production graph to a surface that exists for test back-compat
only.

**FLAG when a file under `wyrd/generators/kenning/cli/`
(except cli/__init__.py itself) contains:**

* `from wyrd.generators.kenning.cli import X` (any name, including
  the `cli` click group object). Per-command modules should not
  reach into the back-compat shim.

**Acceptable patterns** (don't flag):

* `from wyrd.generators.kenning.cli.utils import _X` — utils.py is
  the explicit shared-helper module.
* `from wyrd.generators.kenning.cli.<sibling> import _X` — sibling
  per-command module re-using a co-located helper.
* `from wyrd.generators.kenning.cli.lexicon import lexicon` from
  inside `cli/lexicon/<sub>.py` — the @lexicon group is the parent
  the sub-command's `add_to` registers against, NOT the back-compat
  shim. This is a legitimate parent-package import.
* `from wyrd.generators.kenning.cli.lexicon.<sibling> import _X` —
  same shape; sibling lexicon command sharing a helper.

**Review approach:**

1. For each new file under `cli/` in the PR diff, grep for
   `from wyrd.generators.kenning.cli ` (note the trailing space
   to exclude `cli.utils` / `cli.lexicon`).
2. Any hit is a flag — propose the right alternative
   (cli.utils for genuinely shared; sibling module for co-located).

