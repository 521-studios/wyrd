# Kenning-specific agent reviewers

These reviewers compose with the universal `AGENT-REVIEWERS.md` at
repo root — they don't replace it. Scope: files under
`wyrd/generators/kenning/` and `tests/test_kenning_*`.

## lexicon-package-structure-reviewer

Enforce that lexicon-DB concerns live under
`wyrd/generators/kenning/lexicon/`. Per wyrd-67fv, the lexicon was
split out of a 7,272-line `lexicon.py` into a package. Every file
that touches the lexicon DB belongs in the package; the rest of
`wyrd/generators/kenning/` stays runtime-only (`__init__.py`,
`meaning.py`, `name.py`, `proportions.py`, `trie_matcher.py`, etc.).

**FLAG when a file outside `lexicon/` contains:**

1. **A SQL `CREATE TABLE`, `CREATE INDEX`, or `CREATE VIEW`.** These
   are schema concerns and belong in an Alembic migration under
   `lexicon/sql/migrations/versions/`.

2. **An `import sqlite3` + opening `wyrd.generators.kenning` data
   from disk.** That code is a candidate lexicon caller and should
   either use `LexiconDB` (which already encapsulates the connection
   + pragmas) or move into the package next to its peer enrichment
   modules.

3. **A new module that does "scan the lexicon DB and write back to
   it"** (enrichment / ingest / audit / report pass). New modules of
   this shape belong inside `lexicon/` — at minimum under
   `lexicon/<descriptive_name>.py`, and ideally inside the
   subpackage that wyrd-67fv follow-ups will create
   (`lexicon/enrichment/`, `lexicon/ingest/`, etc.). Existing
   modules outside the package (`disambiguator.py`, `strata.py`,
   `english_shaping.py`, `wiktextract_ingester.py`, etc.) are
   pre-split files that will move in follow-up tickets — DON'T
   re-flag those; only flag NEW files of the same shape.

**Acceptable** (don't flag):

* Runtime-layer files (`__init__.py`, `meaning.py`, `name.py`,
  `proportions.py`, `era.py`, `respelling.py`, `scripts.py`,
  `trie_matcher.py`, `decomposition.py`, `rewind.py`,
  `phonology.py`, `vectors/schemas.py`) — these read the bundled
  `meanings.json` and have no DB access.
* `cli.py` — CLI wiring lives here per project convention; will be
  split separately under wyrd-g143.
* Pre-existing files in `wyrd/generators/kenning/` that touch the
  lexicon DB (above list). Their move is a known follow-up.

**Review approach:**
1. Grep changed files for `CREATE TABLE` / `CREATE INDEX` /
   `CREATE VIEW`.
2. For each new `.py` file in `wyrd/generators/kenning/` (not under
   `lexicon/`), open it and check whether it does lexicon-side I/O.
3. If a NEW file is misplaced: **block the PR.** Request the file
   move before merge. Placement is a structural decision; landing
   new code in the wrong place and ticketing the cleanup recreates
   the exact debt wyrd-67fv was filed to repay. There is no
   "defer with a ticket" path for new files.
4. If pre-existing out-of-place code is touched (the
   disambiguator / strata / english_shaping / ingester family),
   that's not the same — those moves are explicit follow-up
   tickets. Don't flag those re-edits.

## alembic-migration-discipline-reviewer

Schema changes must go through Alembic. Per wyrd-67fv, the layered
migrations under `lexicon/sql/migrations/versions/` (currently
0001_sources … 0008_views) are the source of truth for the lexicon
schema. The committed `data/lexicon.sql` file is regenerated from
those migrations and is documentation, not authoritative.

**FLAG when:**

1. **`data/lexicon.sql` is edited directly** — the file is a
   regenerated artifact. Schema changes belong in a new alembic
   migration under `lexicon/sql/migrations/versions/`; regenerate
   `data/lexicon.sql` as the last step.

2. **A new `_create_*_table` or `_migrate_*` helper function lands
   in `lexicon/__init__.py`.** The historical helpers (~1,300
   lines) exist because the schema evolved before Alembic was
   adopted. New schema changes go in Alembic migrations, not in
   another `_migrate_*` helper. Adding to the legacy chain locks in
   the old pattern and grows the file the wyrd-67fv split is
   shrinking.

3. **A new migration is added but the SA Core MetaData in
   `lexicon/sql/tables.py` isn't updated.** The MetaData is the
   target of `alembic revision --autogenerate` and should mirror
   what the migrations produce on a fresh DB. Drift between them
   silently breaks future autogenerate runs.

4. **A migration uses `op.create_table()` with constraint or
   default formatting that doesn't survive a round-trip through
   `sqlite_master.sql`.** SQLite stores DDL verbatim, so the
   side-by-side equality check (init_schema-vs-alembic) the
   wyrd-67fv design relies on can be broken by SA's auto-quoting.
   Prefer `op.execute("""CREATE TABLE ...""")` with the SQL written
   to match the existing `data/lexicon.sql` shape when capturing
   the current schema. Use `op.create_table()` etc. for genuinely
   new tables added after the wyrd-67fv baseline.

**Acceptable** (don't flag):

* Edits to `data/lexicon.sql` that are the regenerated output of a
  new migration (the migration is in the same PR).
* Removing `_migrate_*` helpers — that's the deprecation path
  we're on.

**Review approach:**
1. Check that any change to `data/lexicon.sql` has a corresponding
   new file under `lexicon/sql/migrations/versions/`.
2. Check that any new `_create_*_table` / `_migrate_*` function
   isn't a duplicate of an Alembic migration that should have
   landed instead.
3. Confirm that the next migration's `revision` ID is
   monotonically ordered and its `down_revision` chains to the
   previous head.

## importlib-resources-reviewer

Package data files (JSON sidecars, SQL schema, fixture text) must
be loaded via `importlib.resources`, not `Path(__file__).parent`.
The `__file__.parent` pattern silently broke when `lexicon.py` was
renamed to `lexicon/__init__.py` because the parent directory
shifted by one level (caught in wyrd-67fv, fix at
`lexicon/__init__.py:_load_norman_manorial_family_tokens`). The
importlib.resources pattern is robust to package moves and works
identically in dev (editable install) and Lambda (frozen package).

**FLAG when a file under `wyrd/generators/kenning/` contains:**

* `Path(__file__).parent / "data"` (or any `__file__.parent` /
  `Path(__file__).parents[N]` to navigate to a sibling data file).

**Acceptable pattern:**

```python
from importlib import resources

data = resources.files("wyrd.generators.kenning.data").joinpath(
    "norman_manorial_families.json"
)
families = json.loads(data.read_text())
```

**Acceptable** (don't flag):

* `Path(__file__).parent` used for write paths (e.g. test fixtures
  writing to a tmp dir relative to the test file). Resources are
  read-only by definition.
* `__file__` references in `tests/` (tests have a stable layout
  and aren't packaged).

**Review approach:**
1. Grep changed files under `wyrd/generators/kenning/` for
   `Path(__file__).parent` and `__file__.parents`.
2. For each hit, check whether the path resolves to a bundled
   data file. If so, recommend the `importlib.resources` form.

This rule is kenning-specific only because the package's layered
structure makes path drift more likely than in flat packages. The
underlying principle (use importlib.resources for package data) is
universal; if other generators grow data sidecars, promote this
reviewer to the repo-root `AGENT-REVIEWERS.md`.

## docstring-grep-verify-reviewer

Module and function docstrings that name specific tables, columns,
function names, regex patterns, or output shapes must be
grep-verified against the code before merge. The wyrd-67fv slice D
review loop spent 5 fix rounds chasing 7 docstring inaccuracies
(round 2 introduced a NEW inaccuracy while fixing round 1's
fantasy_export claim — fixes can be wrong too) because NEW module
docstrings were written without checking each claim against the
function body. Examples included:

* `ingest.py` claimed "Idempotent per `(source_id, page)`" — code
  never references `page` and there's no UNIQUE constraint that
  would make it idempotent.
* `decomposition_export.py` claimed "walks every toponym +
  toponym_etymology pairing" — SQL actually joins
  `toponym_decomposition` and never touches `toponym_etymology`.
* `fantasy_export.py` claimed source was "etymon tagged
  fantasy-suitable" with output shape `{morpheme, language, gloss,
  source_book, source_page}` — actual source is `fantasy_morpheme`
  rows where `usable=1`, with a completely different output shape.
* `bundle/_emit.py` claimed `_LANG_CODE_TO_JSON_FIELD` rolls
  `welsh + old-welsh + middle-welsh → welsh` — actually they all
  collapse into `celtic_mix` (a single shared bucket).
* `bundle/_subject.py` `_emit_word_languages` and
  `_WordLanguageAccumulators` docstrings listed 5 sibling field
  families — the actual code emits 9 (wyrd-vsrn / wyrd-qhs0 /
  wyrd-lr4 grew the set without updating prose).

**FLAG when a NEW or MODIFIED docstring contains:**

1. A SQL table or column name that doesn't appear in the function's
   SQL string literal.
2. A function name referenced as called/used that doesn't appear in
   the function body (grep for the bare name).
3. A regex count (e.g. "Three regex patterns", "Five precision
   gates") that doesn't match `len(re.findall(r'^_[A-Z_]+_RE\b', ...))`
   or the equivalent obvious count.
4. An output-shape claim (`{key1, key2, ...}` or
   `dict[K, V]`) where the keys don't match the actual `SELECT`
   columns or dict construction in the function body.
5. A field count or list ("5 dicts", "the four renderings", "the
   nine sibling families") that doesn't match the actual dataclass
   field count or list construction.

**Acceptable** (don't flag):

* Stable conceptual descriptions that don't name specific symbols
  ("this module owns the per-language phonological-bridge pass").
* Examples that use placeholder names (`<lang>_variants`).
* Forward-looking docs that explicitly say "Phase 2 will add ...".

**Review approach:**

1. For each NEW module docstring (file added in the PR), open the
   first function below it and grep-verify every named symbol /
   table / regex count / output key.
2. For MODIFIED module docstrings, compare the diff's new lines
   against the function body line-by-line.
3. For dataclass-bundle docstrings (`_BucketAccumulator`,
   `_WordLanguageAccumulators`, etc.), count the actual
   `field(default_factory=...)` declarations and compare to any
   count words in the prose.
4. If a count or shape claim is wrong: flag it. These are
   factual errors that mislead future maintainers (and they
   accumulate — slice D's `_WordLanguageAccumulators` doc was
   wrong because three separate wyrd-ticket waves grew the set
   without anyone re-reading the class doc).

## dataclass-extraction-decorator-reviewer

When a `@dataclass` class is extracted from one file into another
via line-range copy, the `@dataclass` decorator line above the
`class X:` line is often missed if the extraction range starts at
`class`. The wyrd-67fv slice C caught this on `EraReflex` (lost the
decorator during extraction, all era-reflex tests failed with
"`EraReflex() takes no arguments`"); slice D caught it on
`_BucketAccumulator` (bundle build failed with
"`argument of type 'Field' is not iterable`" because the dataclass
`field(default_factory=set)` became a literal `Field` object instead
of an empty set).

**FLAG when a PR adds a NEW file containing a class that uses
`field(default_factory=...)` or `field(default=...)` without a
`@dataclass` (or `@dataclass(...)`) decorator on the line above.**

These two patterns are the load-bearing telltale (the imports are
required for the example to run):

```python
from dataclasses import dataclass, field

class X:
    forms: list[str] = field(default_factory=list)  # ← needs @dataclass!
    citations: set[str] = field(default_factory=set)
```

```python
from dataclasses import dataclass, field

class Y:
    name: str = field(default="")  # ← needs @dataclass!
```

**Acceptable** (don't flag):

* Plain `@dataclass`-decorated classes with `field(...)` defaults
  (the decorator is what makes `field()` resolve to a Field
  descriptor at class-construction time).
* Classes that use `field` only as a typing annotation
  (`x: dataclasses.Field[int]`) without calling it.
* `attrs.field(...)` or `pydantic.Field(...)` — those have their
  own decorator chains (`@attrs.define`, model inheritance) that
  don't fit this rule's signature.

**Review approach:**

1. Grep new files for `field(default` — every hit must be inside
   a `@dataclass`-decorated class.
2. If a `field(...)` call is in a class without the decorator: flag
   it. The bug surfaces at instance-construction time, not at
   import time, so type-check tools and quick smoke imports won't
   catch it.

This rule fires almost exclusively on extraction PRs (you don't
normally write `field(default_factory=...)` outside a dataclass on
purpose). Promote to repo-root `AGENT-REVIEWERS.md` if other
generator packages start growing extraction PRs.

## pre-push-extraction-sweep (practice, not a reviewer)

For PRs that extract >500 lines from one file into multiple new
modules, run a comprehensive sweep agent BEFORE pushing. The
wyrd-67fv slice D introduced this pattern and caught 5 things up
front (a dead-code drag-in, a stranded constant block, a docstring
inaccuracy, a missing production re-export that would have hit
`ImportError` at collection time, a broken test monkey-patch).

Slices A and B each landed in 1 review round; slice C needed 4
rounds (the "rule/transform" terminology drift kept recurring
across 6 bridges files because no up-front sweep). Slice D, with
the sweep, still needed 5 docstring-fix rounds — the sweep didn't
catch every claim — but the rounds were shorter and avoided the
production-import break the sweep DID catch. The lesson isn't
"sweep replaces review rounds"; it's "sweep catches the cheap
class of problems (missing re-exports, broken monkey-patches,
orphan comments) before they cost a round, leaving the review
loop for the harder docstring-accuracy class".

**When to use:**

* Extraction PRs that touch >500 LOC across >3 new files.
* Subpackage creation (the new `subpackage/__init__.py` is a public
  surface that needs auditing before any external caller sees it).
* PRs that move underscore-prefixed helpers callers depend on.

**Don't bother:**

* Bug-fix PRs, single-file refactors, or anything that fits in a
  ~200-line diff.

**Reusable sweep prompt template:**

```
Pre-push sweep for [refactor description] in [worktree path].

CONTEXT: [what was extracted, why]
NEW FILES: [list each new file + 1-line purpose]
PACKAGE __init__: [path to the back-compat re-export shim, e.g.
                   wyrd/generators/kenning/lexicon/__init__.py]
PRE-EXISTING LESSON: [most recent slice's caught patterns, so
                     the sweep doesn't waste time re-discovering them]

YOUR TASK — comprehensive sweep, no fixes:

1. ORPHAN COMMENTS: scan each new file for comment lines that
   start mid-sentence (lowercase first word, no preceding
   continuation). Extraction artifacts.
2. DEAD CODE / DRAG-INS: identify any function or constant in the
   new files that has NO call site anywhere in the repo (grep
   wyrd/ tests/ for the name).
3. DATACLASS DECORATORS: confirm @dataclass on every class that
   uses field(default_factory=...) or field(default=...).
4. DOCSTRING ACCURACY: for each NEW module docstring, verify every
   factual claim against the code body (named tables/functions/
   output shape match the SQL/dict construction).
5. CROSS-MODULE IMPORTS: verify deferred imports and shared
   helpers resolve through the back-compat re-export.
6. RE-EXPORT-SURFACE: re-exports through the PACKAGE __init__ path
   above must have at least one external caller. Grep production
   code AND tests for each re-exported name; a missing re-export
   is a worse failure than a dead one (the production caller
   ImportErrors at test-collection time, not at import-time).

REPORT: numbered list of concrete findings with file:line refs.
"No issues found" if nothing of substance.
```

This is documented as a practice (not a reviewer) because it runs
once per PR, before review starts, rather than as part of the
review loop's per-round agents.

### Extension from wyrd-g143 (cli.py 10k → 110-line shim, 5 slices)

The cli.py refactor surfaced two new failure modes the slice-D
sweep template above didn't cover. Add these checks to step 1 +
step 2:

7. **AUTO-RANGE HELPER OVERFLOW**: when extraction is driven by
   "decorator → next anchor" line-range detection, the trailing
   `def _helper` definitions between two commands get pulled into
   the PRECEDING command's range — but they may belong to the
   FOLLOWING command (or be shared). Scan each new module's last
   ~50 lines for `def _` defs and decide if they belong with that
   command, with the next one, or in a shared utils.
   Concrete examples from wyrd-g143:
   * slice 3 diff-bundle range auto-grabbed `_append_remove_event`
     (actually a prune-toponym / prune-etymon helper). Range
     hand-overridden to stop at the real end of `lexicon_diff_bundle`.
   * slice 4 fuzzy-search range auto-grabbed
     `_classify_dry_run_row_counts` + `_disambiguate_dispatch`
     (actually disambiguate-fuzzy helpers). Moved post-hoc.

8. **LAMBDA-BIND IN NEW TEST FIXTURES**: when an extraction
   requires updating a test's monkeypatch site, do not introduce
   `_stub = lambda x: ...` style assignments. ruff E731 rejects
   them and CI will fail. Use `def _stub(x): return ...` instead.
   wyrd-g143 slice 5 burned a CI cycle on this — 5 sites in
   `tests/test_kenning_lexicon.py` had the lambda-bind shape after
   the monkeypatch retargeting; the round-1 fix was mechanical.

## cli-extraction-cross-module-imports-reviewer

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

## cli-extraction-test-monkeypatch-reviewer

When a CLI helper moves from cli/__init__.py into a per-command
module via the wyrd-g143 extraction pattern, tests that did
`monkeypatch.setattr(cli_mod, "_helper", stub)` no longer
intercept the consumer's call. The consumer (per-command module)
imported the helper into its OWN module namespace at import time;
patching the shim sets `cli_mod._helper = stub` but leaves the
consumer's local-bound reference pointing at the original
function. The test runs against the un-monkey-patched original
without warning, and any assertions on the stub's captured state
fail with bewildering `None`s and `0`s instead of an
`AttributeError`.

wyrd-g143 slice 5 burned a full CI cycle on this: 9 monkeypatch
sites across 3 test files needed retargeting. Same pattern is
likely to recur on every cli-extraction slice that moves a
test-monkeypatched helper.

**FLAG when a PR that extracts code from cli/__init__.py into a
per-command module under cli/ leaves UNCHANGED test files that
contain:**

* `monkeypatch.setattr(cli_mod, "<helper_name>", ...)` where
  `<helper_name>` is one of the helpers moved in this slice.
* `monkeypatch.setattr("wyrd.generators.kenning.cli.<helper>", ...)`
  (string-form variant — same problem).

**Acceptable patterns** (don't flag):

* Tests updated to patch the CONSUMER module:
  `monkeypatch.setattr(_mine_llm_mod, "_select_parser_and_run", stub)`.
* Tests updated to patch the SOURCE module path:
  `monkeypatch.setattr("wyrd.generators.kenning.cli.utils._helper", stub)`
  — works because cli.utils is the import source. (Per-consumer
  patching is still preferred when the helper is consumed in only
  a handful of modules; source-module patching is broader but
  harder to reason about when N consumers vary.)
* Tests that patch the back-compat shim for a helper that is NOT
  in this slice's diff (the shim re-export is still the source for
  that helper).

**Review approach:**

1. List the helpers extracted in this slice's diff (anything that
   moved from `cli/__init__.py` into `cli/<name>.py` or
   `cli/lexicon/<name>.py`).
2. For each, grep `tests/` for
   `monkeypatch.setattr(cli_mod, "<helper>"` and
   `monkeypatch.setattr("wyrd.generators.kenning.cli.<helper>"`.
3. Any hit is a stale patch — the test will run against the
   original at the next pytest pass. Flag with the right
   replacement.

## cli-extraction-placement-reviewer

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
