# dataclass-extraction-decorator-reviewer

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

