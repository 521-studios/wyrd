# cli-extraction-test-monkeypatch-reviewer

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

