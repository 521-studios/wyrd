# fixture-data-reviewer

Review test changes for **isolation from live operator data**. Tests must run against fixtures, not the operator's live lexicon DB or live bundle data, for two reasons:

1. **CI doesn't have the live data.** `~/.wyrd/lexicon.db` doesn't exist on CI runners. A test that depends on it works locally and fails on CI — false-green that surfaces only at deploy time.
2. **The live DB is huge (2.4M+ etymon rows, 5GB+ wiktextract slices in `~/.wyrd/sources/`).** A test path that walks it is minutes-long instead of milliseconds. Slow tests discourage running pytest before commit, which discourages catching regressions early.

**Default severity: P2** — these typically surface as CI failures after the fact, not as merge-blocking findings, but the fix is always same-cycle.

**Rules to enforce:**

1. **No live-DB dependency.** Tests must NOT assume `~/.wyrd/lexicon.db` exists. The conftest autouse `_isolate_default_lexicon_db` fixture redirects `_DEFAULT_LEXICON_PATH` to `tmp_path/test.db` — fine. Tests that need a populated DB build one inside the test (`sqlite3.connect(":memory:")` + `executescript(SCHEMA_SQL)` + a few INSERT rows, OR open `tmp_path/test.db` and populate).

2. **No live-bundle dependency.** Tests that exercise generator behaviour use the `swap_bundle` / `bundle_swapper` fixtures from `tests/conftest.py` rather than the runtime `meanings.json`. Broad-stroke smoke tests against the live bundle are explicitly named (`test_find_meaning_runs_full_bundled_corpus_without_crashing`, `test_sidecar_lifts_irish_corpus_perfect_rate`) and are the exception, not the rule.

3. **No live-bulk-manifest dependency.** Tests must NOT trigger the L1 wiktextract ingest against the operator's `~/.wyrd/sources/`. The conftest autouse `_isolate_bulk_manifest` fixture handles this for `rebuild-from-jsonl` tests; tests that exercise bulk-source paths directly should monkeypatch `bulk_sources.MANIFEST_PATH` to a tmp manifest with synthetic slices.

4. **Fixture data must stay representative.** When the production data shape changes (schema migration, new column, manifest schema_version bump, new row type), the fixture data has to evolve in lockstep. A test that pins a 2-row fixture against an obsolete shape passes locally but is no longer testing what its name claims. Watch for:
   - Schema changes in `data/seed/lexicon.sql` not reflected in test fixtures that `executescript` partial schemas
   - Manifest `schema_version` bumps in `bulk_sources.py` not reflected in test-fixture manifests
   - New required columns / fields on JSONL row types that fixtures still elide
   - New L3 derivations or enrichment columns that fixtures don't seed

5. **CLI tests must override `--db` (or rely on the autouse default redirect).** A test that invokes `lexicon <command>` without explicit `--db` defaults to whatever `_DEFAULT_LEXICON_PATH` resolves to — locally the live DB, on CI nonexistent. The conftest autouse fixture redirects this to `tmp_path/test.db`, but tests should still prefer explicit `--db <tmp>` for clarity.

**Review approach:**

1. Walk the diff for test files (any `tests/test_*.py` change).
2. Grep new tests for `~/.wyrd/`, `_DEFAULT_LEXICON_PATH`, `meanings.json`, `MANIFEST_PATH`, `data/mining/_bulk_manifest.json` — any reference is a flag-worthy review point. Confirm the test either:
   * passes an explicit `tmp_path` override, OR
   * uses an autouse fixture from `tests/conftest.py`, OR
   * is one of the documented live-data smoke tests (then ask whether it should be).
3. For any non-test source diff that changes a data shape (schema, manifest, row type), grep `tests/` for fixtures of the same shape and flag whether the fixtures need updates.
4. Note any test that takes >1s in the `--durations` output as a possible live-data leak.

**Acceptable:**

- A test deliberately opting out via `@pytest.mark.no_lexicon_isolation` (with comment explaining why fixture data won't work).
- Creating a beads ticket to track a fixture update if the production shape changed and updating fixtures is out of scope.

**Do NOT allow:**

- "Local run works, CI will figure it out" — CI doesn't have the local data.
- "It's only a smoke test" — smoke tests against the live bundle are documented exceptions; new ones need explicit justification.
- Skipping fixture updates after a schema/manifest change with "we'll catch it later" — the fixture stops being representative the moment the production shape diverges.

