# cli-extraction-placement-reviewer

When extracting from a monolithic CLI file, helper functions and
module-level constants need a deliberate placement decision: the
auto-range tooling tends to put everything between command N and
command N+1 into command N's module, but the right home depends on
the consumer set.

**Placement rules:**

1. **Single-consumer helper or constant**: co-locate with the
   command that uses it, in the same per-command module. Example:
   `_RANDO_SOURCE` is only used by `lexicon build` → moved to
   `cli/lexicon/build.py`. NOT to `cli/utils.py` (which would
   leak a one-off into a shared surface) and NOT left in
   `cli/__init__.py` (which would block the shim from shrinking).

2. **Multi-consumer helper SHARED across cli/ subpackages**:
   move to `cli/utils.py`. Example: `_readonly_lexicon` (used by
   browse + era-timeline + era-coverage + enrichment-status); the
   wyrd-g143 slice 2 move from inline-in-browse-block to
   cli/utils.py was the right call because consumers fan out
   across multiple subcommand families.

3. **Multi-consumer helper LOCAL to one family**: co-locate with
   the natural-home command in that family; the other consumers
   import from the sibling. Example: `_build_extractor_client` is
   used by mine-toponym-mentions{,-tiered,-staged} + review;
   defining it in mine_toponym_mentions.py and importing from
   siblings is cleaner than promoting to cli/utils.py (where it
   would be a single-family concern leaking into the cli-wide
   shared surface).

4. **Nested click sub-groups (`@lexicon.group("X")`)**: must live
   in their own subpackage at `cli/lexicon/X/`, mirroring the
   parent's structure (`__init__.py` for the group def + add_to
   hook; one per-command module per sub-command). Example: the
   wyrd-g143 browse + synsets groups both became subpackages. A
   NEW `@click.group` inside a per-command module body is a
   structural error — it would collide with the parent's add_to
   contract and break the "every subcommand has its own module"
   invariant.

**FLAG when a CLI extraction PR contains:**

* A helper/constant in `cli/utils.py` (or any other shared module)
  whose only consumer is a single per-command module in the same
  slice's diff (rule 1 violation).
* A helper duplicated across multiple per-command modules in the
  same family that could be co-located with one of them and
  imported from siblings (rule 3 violation; usually shows up as
  identical-body helpers in two files).
* A `@click.group(...)` decorator on a function inside a
  per-command module body — should be promoted to a subpackage
  (rule 4 violation).

**Review approach:**

1. For each new module-level helper / constant in the PR, grep
   `wyrd-*/` for its consumers. Apply rules 1–3.
2. For each new `@click.group(`, confirm it's at the `__init__.py`
   of a subpackage, not buried in a per-command module body.

