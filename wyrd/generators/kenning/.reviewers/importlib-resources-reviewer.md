# importlib-resources-reviewer

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

