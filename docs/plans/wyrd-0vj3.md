# wyrd-0vj3 — Bulk-data storage strategy for wiktextract slices

Implementation plan for the bulk-source S3 + local-cache hybrid the
operator selected for wyrd-0vj3 (see ticket for option enumeration).

## Goal

Make `~/.wyrd/sources/` re-fillable from a single command, backed by
S3. The pipeline that consumes `sources/*.jsonl` (wiktextract
slices) keeps working unchanged.

## Non-goals

- No changes to runtime data shape, schema, or ingest semantics.
- No production bucket — staging-only (this is a dev workflow asset).
- No multi-environment promotion logic (staging is source-of-truth for
  bulk data).

## Naming + regions

| | Value |
|---|---|
| AWS profile | `521-Staging-Admin` |
| Region | `us-east-2` (matches staging in `infra/terraform/environments/staging/backend.tf`) |
| Bucket | `521studios-staging-wyrd-lexicon-bulk` (matches `pfsrd2-data` naming) |
| S3 layout | `s3://.../wiktextract/v1/<slice>.jsonl.zst` |
| Manifest | `data/mining/_bulk_manifest.json` (in git) |
| Local cache | `~/.wyrd/sources/` (gitignored, mirrors the bucket key prefix) |
| Config | `~/.wyrd/config.toml` (gitignored; env-var overridable) |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ git repo                                                │
│  data/mining/_bulk_manifest.json  (small; sha256 + key) │
│  wyrd/...                          (ingest CLI changes) │
└──────────────────┬──────────────────────────────────────┘
                   │ lexicon fetch-bulk-sources
                   ▼
            ┌──────────────┐         ┌─────────────────────┐
            │ S3 bucket    │ ←─────  │ ~/.wyrd/sources/    │
            │ wiktextract/ │         │ wiktextract_*.jsonl │
            │   v1/*.zst   │  pull   │ (gitignored)        │
            └──────────────┘         └─────────────────────┘
                                              │
                                              ▼
                                     lexicon ingest-wiktionary
                                     (reads .jsonl OR .jsonl.zst)
```

## Components

### 1. Terraform module — `infra/terraform/modules/wyrd-lexicon-bulk/`

Mirror of `pfsrd2-data/main.tf`. Same shape:

- `aws_s3_bucket` named `521studios-${var.environment}-wyrd-lexicon-bulk`
  with `lifecycle { prevent_destroy = true }`.
- `aws_s3_bucket_versioning` enabled (recover from accidental delete).
- `aws_s3_bucket_public_access_block` — fully blocked.
- `aws_s3_bucket_server_side_encryption_configuration` — AES256.
- `aws_iam_policy` `wyrd-lexicon-bulk-rw` — read/write to bucket.
- `aws_s3_bucket_lifecycle_configuration` — noncurrent versions
  expire after 90 days (keeps versioning useful without bloat).

Outputs: `bucket_name`, `bucket_arn`, `rw_iam_policy_arn`.

### 2. Staging-only environment wiring

`infra/terraform/environments/staging/main.tf`:

```hcl
module "wyrd_lexicon_bulk" {
  source      = "../../modules/wyrd-lexicon-bulk"
  environment = var.environment
}
```

**NOT in production.** wyrd is dev-only data. Document in the
module README that production wiring would be a separate decision.

### 3. Manifest format — `data/mining/_bulk_manifest.json`

```json
{
  "version": 1,
  "bucket": "521studios-staging-wyrd-lexicon-bulk",
  "region": "us-east-2",
  "s3_prefix": "wiktextract/v1",
  "compression": "zstd",
  "slices": [
    {
      "name": "wiktextract_old_english",
      "s3_key": "wiktextract/v1/old_english.jsonl.zst",
      "sha256": "abc123...",
      "raw_size_bytes": 116883612,
      "compressed_size_bytes": 8650000
    }
  ]
}
```

Manifest in git, data in S3. The bucket name + prefix here is the
default; can be overridden in `~/.wyrd/config.toml` for forks /
non-521 contributors.

### 4. Local config — `~/.wyrd/config.toml`

```toml
[bulk_storage]
bucket = "521studios-staging-wyrd-lexicon-bulk"
region = "us-east-2"
profile = "521-Staging-Admin"
local_cache_dir = "~/.wyrd/sources"
```

Env-var overrides (preserved order: env > config.toml > manifest
default):

- `WYRD_BULK_BUCKET`
- `WYRD_BULK_REGION`
- `WYRD_AWS_PROFILE`
- `WYRD_SOURCES_DIR`

Bootstrap behavior: on first run of any bulk command, if
`~/.wyrd/config.toml` is missing, write a defaults file derived
from the manifest.

### 5. Python deps — `pyproject.toml`

Add (function-level import in CLI commands so no startup cost):

- `boto3`
- `zstandard`

### 6. New module — `wyrd/generators/kenning/bulk_sources.py`

```python
def load_manifest(repo_root: Path) -> Manifest: ...
def load_config(home: Path = Path.home()) -> Config: ...
def expected_slice_path(config: Config, slice_name: str) -> Path: ...
def fetch_missing_slices(manifest: Manifest, config: Config) -> FetchResult: ...
def upload_slices(slices: list[Path], manifest: Manifest, config: Config) -> UploadResult: ...
def validate_sha256(path: Path, expected: str) -> bool: ...
```

Pure functions where possible. boto3 client lives in one place.
Decompression on read happens transparently in the ingest path
(see component 8).

### 7. CLI commands — additions to `cli.py`

```text
lexicon fetch-bulk-sources [--slice NAME] [--force] [--dry-run]
  Reads data/mining/_bulk_manifest.json, downloads any missing or
  checksum-mismatched slices from S3 to ~/.wyrd/sources/.
  --slice NAME: just one slice (repeatable)
  --force: re-download even if sha256 matches
  --dry-run: report what would happen

lexicon push-bulk-sources [--slice NAME]
  Uploads ~/.wyrd/sources/*.jsonl to S3 as zstd-compressed,
  regenerates data/mining/_bulk_manifest.json with new sha256s
  and sizes. Operator commits the manifest change to git.

lexicon verify-bulk-sources
  Checks each manifest slice against the local cache (sha256 +
  presence). Useful as a gate before rebuild-from-jsonl
  --with-enrichment.
```

### 8. `ingest-wiktionary` change — transparent zstd read

```python
def _open_jsonl(path: Path):
    if path.suffix == ".zst":
        import zstandard
        dctx = zstandard.ZstdDecompressor()
        return io.TextIOWrapper(
            dctx.stream_reader(path.open("rb")),
            encoding="utf-8",
        )
    return path.open(encoding="utf-8")
```

Backwards compatible: existing `.jsonl` files still work.

### 9. `.gitignore` additions

```
# wyrd-0vj3 bulk-sources transient state
data/mining/_bulk_manifest.json.lock
```

(`~/.wyrd/sources/` and `~/.wyrd/config.toml` are already outside
the repo, no change needed.)

### 10. Docs — `wyrd/generators/kenning/L2_L3_BOUNDARY.md`

Add a **Bulk sources** section explaining: manifest in git, data in
S3, fetch script bootstraps a fresh checkout, contributor config in
`~/.wyrd/config.toml`.

## Implementation phasing

Recommend **two PRs**:

### PR A — infra + ingest plumbing (no behavior change yet)

- Terraform module + staging composition
- `bulk_sources.py` with all the pure-function helpers
- `lexicon fetch-bulk-sources` / `push-bulk-sources` /
  `verify-bulk-sources` CLIs
- `ingest-wiktionary` reads `.jsonl.zst` transparently
- Add boto3 + zstandard to pyproject
- One-time upload: operator runs `push-bulk-sources` on current
  `sources/` to seed the bucket; commits the resulting manifest
- Tests: pure functions covered with synthetic manifests; S3 via
  `moto` mock; ingest with a fixture `.jsonl.zst`

### PR B — wire into rebuild-from-jsonl (this is wyrd-hidb)

- Add `--fetch-bulk` flag to `rebuild-from-jsonl`
- Or fail loudly with a hint if bulk sources are missing
- Wire all L3 derivations into `--with-enrichment` canonical order
- Round-trip integration test

PR A is mostly mechanical / infra. PR B closes the loop. Keeping
them separate makes review easier and lets the bucket + upload
settle before depending on it.

## Open questions

1. **IAM granularity** — read-only for contributors, read-write for
   ops? Or single rw policy? For dev-only this is probably fine as
   single rw since the AWS profile is the dev's own admin role.
2. **Versioning vs snapshot** — per-slice sha256 in the manifest
   covers content versioning; bump the manifest `version` field
   only on schema changes to the manifest itself.
3. **CI integration** — should CI run `verify-bulk-sources` and
   fail on mismatch? Or skip (bulk sources aren't needed for unit
   tests)? Recommend: skip in unit-test CI; gate only the
   rebuild/deploy workflow on it.
4. **Migration of existing slices** — `sources/` has 4.6 GB
   locally right now. The one-time upload is the seed; future
   operators (incl. fresh checkout) `fetch-bulk-sources`. Worth a
   single-line note in PR A's commit message: "after merging,
   operator runs `push-bulk-sources` once to seed."

## Risks

- **boto3 cold-start** — slow first import. Mitigated by
  function-level lazy import in CLI commands.
- **zstd wheel availability** — `zstandard` is widely available and
  well-maintained, but does require a wheel install. Document in
  pyproject.
- **Manifest drift** — if a contributor edits `~/.wyrd/sources/`
  but doesn't push, the manifest in git is ahead/behind reality.
  `verify-bulk-sources` exits non-zero; document the workflow.
- **S3 cost** — 4.6 GB raw → ~340-460 MB compressed at rest. At
  staging rates this is pennies/month. Egress is the concern, but
  fetches are rare (once per fresh checkout).
- **Bucket destruction via terraform** — covered by
  `lifecycle { prevent_destroy = true }` on the bucket resource.

## Test plan

- `bulk_sources.py` pure functions: 100% coverage with synthetic
  manifests + tmp_path
- S3 round-trip: `moto` (boto3 mock library) for offline tests
- ingest-wiktionary: existing tests + new fixture `.jsonl.zst`
- Manifest validation: dataclass parsing or jsonschema
- CLI smoke: `--dry-run` paths verified; real S3 paths gated by env
  var

## Effort estimate

- **PR A**: 1-2 days. ~600 LoC + 200 LoC tests. Lower if `moto`
  works cleanly.
- **PR B (wyrd-hidb)**: 1-2 days; mostly wiring + L3 derivation
  ordering investigation.
