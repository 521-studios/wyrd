"""Bulk-source S3 + local cache helpers — wyrd-0vj3.

Pure-function helpers for the bulk-source sync workflow. The
runtime pipeline (``lexicon ingest-wiktionary``) reads ``.jsonl``
or ``.jsonl.zst`` files from the local cache at
``~/.wyrd/sources/``. This module's job is keeping that cache in
sync with the canonical S3 bucket.

Design (see ``docs/plans/wyrd-0vj3.md``):

* **Manifest** at ``data/mining/_bulk_manifest.json`` in git —
  small, lists each slice with its S3 key and sha256.
* **Bucket** at ``s3://521studios-staging-wyrd-lexicon-bulk/`` —
  authoritative. Versioning enabled (so re-uploads don't lose
  history); manifest sha256 is for integrity-check only.
* **Local cache** at ``~/.wyrd/sources/`` — gitignored, mirrors
  the bucket's key prefix.
* **Config** at ``~/.wyrd/config.toml`` — bucket/region/profile
  overrides. Env vars trump file, file trumps manifest defaults.

The boto3 / zstandard imports are deferred to the functions that
actually need them so CLI startup stays fast for non-bulk paths.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

MANIFEST_PATH = Path("data/mining/_bulk_manifest.json")
DEFAULT_CONFIG_PATH = Path.home() / ".wyrd" / "config.toml"
DEFAULT_LOCAL_CACHE_DIR = Path.home() / ".wyrd" / "sources"

# Env-var override names. Env > config.toml > manifest default.
ENV_BUCKET = "WYRD_BULK_BUCKET"
ENV_REGION = "WYRD_BULK_REGION"
ENV_PROFILE = "WYRD_AWS_PROFILE"
ENV_SOURCES_DIR = "WYRD_SOURCES_DIR"

# Number of bytes per sha256 hashing chunk. 1 MiB is a good balance
# between syscall overhead and memory pressure for ~100MB-3GB files.
_HASH_CHUNK_SIZE = 1024 * 1024

# zstd compression level for raw → .jsonl.zst conversion. Level 19
# was the wyrd-0vj3-plan-sampled sweet spot for jsonl: ~7.4% ratio,
# encode cost still tractable for the once-per-mine workflow. Decode
# cost is constant-time regardless of level.
_ZSTD_LEVEL = 19


class ManifestError(ValueError):
    """The manifest file is missing, malformed, or schema-incompatible."""


@dataclass(frozen=True)
class Slice:
    """One bulk source slice tracked in the manifest."""

    name: str
    s3_key: str
    sha256: str
    raw_size_bytes: int
    compressed_size_bytes: int


@dataclass(frozen=True)
class Manifest:
    """Parsed ``_bulk_manifest.json``."""

    schema_version: int
    bucket: str
    region: str
    s3_prefix: str
    compression: str
    slices: tuple[Slice, ...]


@dataclass(frozen=True)
class Config:
    """Resolved bulk-storage config — manifest defaults overlaid
    with ``~/.wyrd/config.toml`` and env-var overrides."""

    bucket: str
    region: str
    profile: str | None
    local_cache_dir: Path


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path | None = None) -> Manifest:
    """Read ``_bulk_manifest.json``. Raises :class:`ManifestError`
    on missing file / parse failure / unknown schema_version.

    ``manifest_path=None`` reads the module-level :data:`MANIFEST_PATH`
    at call time (not bind time), so a test conftest's monkeypatch
    of MANIFEST_PATH takes effect on every fresh call.
    """
    if manifest_path is None:
        manifest_path = MANIFEST_PATH
    if not manifest_path.exists():
        raise ManifestError(f"manifest not found at {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestError(f"manifest at {manifest_path} is not valid JSON: {e}") from e

    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise ManifestError(
            f"manifest schema_version={schema_version!r} unsupported (this code reads v1 only)"
        )
    try:
        slices = tuple(
            Slice(
                name=s["name"],
                s3_key=s["s3_key"],
                sha256=s["sha256"],
                raw_size_bytes=s["raw_size_bytes"],
                compressed_size_bytes=s["compressed_size_bytes"],
            )
            for s in raw.get("slices", [])
        )
    except (KeyError, TypeError) as e:
        raise ManifestError(f"slice in manifest missing required field: {e}") from e

    return Manifest(
        schema_version=schema_version,
        bucket=raw["bucket"],
        region=raw["region"],
        s3_prefix=raw["s3_prefix"],
        compression=raw["compression"],
        slices=slices,
    )


def manifest_to_json(manifest: Manifest) -> str:
    """Serialize a manifest back to the on-disk shape. 2-space
    indent for git-readability."""
    return (
        json.dumps(
            {
                "schema_version": manifest.schema_version,
                "bucket": manifest.bucket,
                "region": manifest.region,
                "s3_prefix": manifest.s3_prefix,
                "compression": manifest.compression,
                "slices": [
                    {
                        "name": s.name,
                        "s3_key": s.s3_key,
                        "sha256": s.sha256,
                        "raw_size_bytes": s.raw_size_bytes,
                        "compressed_size_bytes": s.compressed_size_bytes,
                    }
                    for s in manifest.slices
                ],
            },
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def load_config(
    manifest: Manifest,
    config_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    """Resolve effective config in precedence: env > config.toml > manifest.

    The default-manifest values are the 521 Studios staging bucket;
    forks / non-521 contributors override via config.toml or env vars.

    ``config_path=None`` reads :data:`DEFAULT_CONFIG_PATH` at call
    time (not bind time), so tests can monkeypatch the module
    constant to redirect from ``~/.wyrd/config.toml``.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    env = env if env is not None else dict(os.environ)

    bucket = manifest.bucket
    region = manifest.region
    profile: str | None = None
    local_cache_dir = DEFAULT_LOCAL_CACHE_DIR

    if config_path.exists():
        try:
            cfg_raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ManifestError(f"{config_path}: not valid TOML: {e}") from e
        section = cfg_raw.get("bulk_storage") or {}
        bucket = section.get("bucket", bucket)
        region = section.get("region", region)
        profile = section.get("profile", profile)
        if "local_cache_dir" in section:
            local_cache_dir = Path(section["local_cache_dir"]).expanduser()

    if ENV_BUCKET in env:
        bucket = env[ENV_BUCKET]
    if ENV_REGION in env:
        region = env[ENV_REGION]
    if ENV_PROFILE in env:
        profile = env[ENV_PROFILE]
    if ENV_SOURCES_DIR in env:
        local_cache_dir = Path(env[ENV_SOURCES_DIR]).expanduser()

    return Config(
        bucket=bucket,
        region=region,
        profile=profile,
        local_cache_dir=local_cache_dir,
    )


# ---------------------------------------------------------------------------
# Local cache + integrity
# ---------------------------------------------------------------------------


def expected_slice_path(config: Config, slice_: Slice) -> Path:
    """Where the cached zstd-compressed slice lives on disk. Mirrors
    the bucket's key relative to the cache root."""
    return config.local_cache_dir / Path(slice_.s3_key).name


def hash_file_sha256(path: Path) -> str:
    """Compute sha256 of a file in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class SliceStatus:
    """Per-slice verification result against the manifest."""

    slice_name: str
    cache_path: Path
    present: bool
    sha256_matches: bool

    @property
    def needs_fetch(self) -> bool:
        return not self.present or not self.sha256_matches


def verify_local_cache(manifest: Manifest, config: Config) -> list[SliceStatus]:
    """Check each manifest slice against the local cache. Returns
    one :class:`SliceStatus` per slice (in manifest order)."""
    statuses: list[SliceStatus] = []
    for slice_ in manifest.slices:
        path = expected_slice_path(config, slice_)
        present = path.exists()
        sha_ok = False
        if present:
            sha_ok = hash_file_sha256(path) == slice_.sha256
        statuses.append(
            SliceStatus(
                slice_name=slice_.name,
                cache_path=path,
                present=present,
                sha256_matches=sha_ok,
            )
        )
    return statuses


# ---------------------------------------------------------------------------
# S3 fetch / push
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    fetched: list[str]
    skipped: list[str]
    failed: list[tuple[str, str]]


def _boto3_client(config: Config) -> Any:
    """Return a boto3 S3 client honoring the resolved config's
    profile + region. Import deferred so non-bulk paths pay no
    startup cost."""
    import boto3

    session = boto3.Session(
        profile_name=config.profile,
        region_name=config.region,
    )
    return session.client("s3")


def fetch_missing_slices(
    manifest: Manifest,
    config: Config,
    *,
    slice_names: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    s3_client: Any | None = None,
) -> FetchResult:
    """Download slices whose local cache is missing or whose sha256
    doesn't match the manifest. Pass ``slice_names`` to restrict to
    a subset (None means all); ``force=True`` re-downloads even on
    sha match. The ``s3_client`` arg lets tests inject a mock."""
    wanted = set(slice_names) if slice_names is not None else None
    statuses = verify_local_cache(manifest, config)

    fetched: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    client = None

    # verify_local_cache builds statuses parallel to manifest.slices in
    # order, so indexing-by-position is structurally safe — no need
    # for zip strict-mode acrobatics.
    for idx, slice_ in enumerate(manifest.slices):
        status = statuses[idx]
        if wanted is not None and slice_.name not in wanted:
            continue
        if not force and not status.needs_fetch:
            skipped.append(slice_.name)
            continue
        if dry_run:
            fetched.append(slice_.name)
            continue
        if client is None:
            client = s3_client or _boto3_client(config)
        try:
            status.cache_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(config.bucket, slice_.s3_key, str(status.cache_path))
            actual = hash_file_sha256(status.cache_path)
            if actual != slice_.sha256:
                # Remove the bad bytes so verify-bulk-sources doesn't
                # report a stale present-but-mismatched state and a
                # follow-up fetch starts from a clean slot.
                with contextlib.suppress(OSError):
                    status.cache_path.unlink()
                failed.append(
                    (
                        slice_.name,
                        f"sha256 mismatch after download "
                        f"(expected {slice_.sha256[:16]}…, got {actual[:16]}…)",
                    )
                )
                continue
            fetched.append(slice_.name)
        except Exception as e:  # noqa: BLE001 — surface as failure
            failed.append((slice_.name, str(e)))

    return FetchResult(fetched=fetched, skipped=skipped, failed=failed)


@dataclass
class UploadResult:
    uploaded: list[str]
    skipped: list[str]
    failed: list[tuple[str, str]]
    new_manifest: Manifest


def _resolve_local_form(slice_: Slice, config: Config) -> tuple[Path | None, int, str | None]:
    """Decide which local file (if any) to upload for ``slice_``.

    Returns ``(local_path, raw_size_bytes, fail_reason)``:

    * ``(<.zst path>, raw_size, None)`` — pre-compressed cache hit;
      ``raw_size`` is the .jsonl size if available, else the manifest's
      ``raw_size_bytes`` (best available estimate).
    * ``(<.zst path>, raw_size, None)`` — raw .jsonl present, was
      successfully compressed to the .zst path as a side effect.
    * ``(None, 0, "compression failed: ...")`` — raw .jsonl present but
      compression raised. ``fail_reason`` carries the error message.
    * ``(None, 0, None)`` — neither form present locally; not a failure,
      just nothing to upload.
    """
    compressed_path = expected_slice_path(config, slice_)
    raw_path = compressed_path.with_suffix("")  # strip .zst -> .jsonl

    if compressed_path.exists():
        raw_size = raw_path.stat().st_size if raw_path.exists() else slice_.raw_size_bytes
        return compressed_path, raw_size, None
    if raw_path.exists():
        try:
            _compress_raw_to_zst(raw_path, compressed_path)
        except Exception as e:  # noqa: BLE001 — surface to caller
            return None, 0, f"compression failed: {e}"
        return compressed_path, raw_path.stat().st_size, None
    return None, 0, None


def upload_slices(
    manifest: Manifest,
    config: Config,
    *,
    slice_names: list[str] | None = None,
    s3_client: Any | None = None,
) -> UploadResult:
    """Upload local-cache slices to S3 (compressing on the fly if
    raw .jsonl) and emit an updated manifest. Returns the new
    manifest so the caller can write it to disk + commit it.

    For each slice in the manifest:
    * If a ``<name>.jsonl.zst`` exists locally, upload it as-is.
    * Else if a ``<name>.jsonl`` exists, zstd-compress it to the
      cache path and upload.
    * Else: skip (slice unavailable locally — operator hasn't
      mined it yet).
    """
    wanted = set(slice_names) if slice_names is not None else None
    uploaded: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    new_slices: list[Slice] = []
    client = None

    for slice_ in manifest.slices:
        if wanted is not None and slice_.name not in wanted:
            new_slices.append(slice_)
            continue

        local, raw_size, fail_reason = _resolve_local_form(slice_, config)
        if fail_reason is not None:
            failed.append((slice_.name, fail_reason))
            new_slices.append(slice_)
            continue
        if local is None:
            skipped.append(slice_.name)
            new_slices.append(slice_)
            continue

        if client is None:
            client = s3_client or _boto3_client(config)
        try:
            client.upload_file(str(local), config.bucket, slice_.s3_key)
            actual_sha = hash_file_sha256(local)
            new_slices.append(
                Slice(
                    name=slice_.name,
                    s3_key=slice_.s3_key,
                    sha256=actual_sha,
                    raw_size_bytes=raw_size,
                    compressed_size_bytes=local.stat().st_size,
                )
            )
            uploaded.append(slice_.name)
        except Exception as e:  # noqa: BLE001
            failed.append((slice_.name, str(e)))
            new_slices.append(slice_)

    new_manifest = Manifest(
        schema_version=manifest.schema_version,
        bucket=manifest.bucket,
        region=manifest.region,
        s3_prefix=manifest.s3_prefix,
        compression=manifest.compression,
        slices=tuple(new_slices),
    )
    return UploadResult(
        uploaded=uploaded,
        skipped=skipped,
        failed=failed,
        new_manifest=new_manifest,
    )


def _compress_raw_to_zst(raw_path: Path, compressed_path: Path) -> None:
    """Stream-compress a raw .jsonl to .jsonl.zst. Level 19 (~7.4%
    ratio for jsonl per the wyrd-0vj3 plan's sampling). Import
    deferred to keep CLI startup fast."""
    import zstandard

    cctx = zstandard.ZstdCompressor(level=_ZSTD_LEVEL)
    with raw_path.open("rb") as src, compressed_path.open("wb") as dst:
        cctx.copy_stream(src, dst)


def open_jsonl(path: Path) -> TextIO:
    """Open a ``.jsonl`` or ``.jsonl.zst`` file for text reading.

    Used by ``ingest-wiktionary`` so a .jsonl.zst slice from the
    bulk cache decompresses transparently. zstandard import is
    deferred to first .zst open.
    """
    import io

    if path.suffix == ".zst":
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        # stream_reader gives a binary file-like; wrap for utf-8.
        return io.TextIOWrapper(
            dctx.stream_reader(path.open("rb")),
            encoding="utf-8",
        )
    return path.open(encoding="utf-8")


@dataclass
class IngestAllResult:
    """Outcome of :func:`ingest_all_slices`. ``per_slice`` is the
    raw counts dict from each ``ingest_wiktextract_path`` call,
    keyed by slice name. ``totals`` sums the scalar fields across
    slices for a one-line summary."""

    per_slice: dict[str, dict[str, int]]
    fetched: list[str]
    failed: list[tuple[str, str]]

    @property
    def totals(self) -> dict[str, int]:
        sums: dict[str, int] = {}
        for slice_counts in self.per_slice.values():
            for k, v in slice_counts.items():
                if isinstance(v, int):
                    sums[k] = sums.get(k, 0) + v
        return sums


def ingest_all_slices(
    db: Any,  # LexiconDB — forward-ref to avoid the lexicon import cost
    *,
    apply: bool = True,
    fetch: bool = False,
    manifest_path: Path | None = None,
    config_path: Path | None = None,
    s3_client: Any | None = None,
) -> IngestAllResult:
    """Walk the bulk-source manifest and ingest each slice into the
    lexicon DB (wyrd-hidb Phase 1).

    For each slice in :func:`load_manifest`:

    1. If the local cache file is missing or sha-mismatched:
       * with ``fetch=True``: download from S3 first (via
         :func:`fetch_missing_slices`).
       * with ``fetch=False``: append to ``failed`` with a hint to
         run ``lexicon fetch-bulk-sources`` and skip the slice.
    2. Delegate to :func:`ingest_wiktextract_path` for the actual
       wiktextract ingest — same code path as
       ``lexicon ingest-wiktionary`` against an individual slice.

    The ``ingest_wiktextract_path`` import is deferred inside the
    function body to avoid a top-of-module circular reference: the
    ingester transitively imports ``bulk_sources.open_jsonl`` for
    .zst handling.
    """
    from wyrd.generators.kenning.lexicon.wiktextract_ingester import (
        ingest_wiktextract_path,
    )

    manifest = load_manifest(manifest_path)
    config = load_config(manifest, config_path=config_path)

    fetched: list[str] = []
    failed: list[tuple[str, str]] = []

    if fetch:
        # Single batched fetch up front rather than per-slice — boto3
        # client reuse + parallel S3 ops are cheap inside the helper.
        fetch_result = fetch_missing_slices(manifest, config, s3_client=s3_client)
        fetched = list(fetch_result.fetched)
        failed.extend(fetch_result.failed)

    per_slice: dict[str, dict[str, int]] = {}
    for slice_ in manifest.slices:
        cache_path = expected_slice_path(config, slice_)
        if not cache_path.exists():
            failed.append(
                (
                    slice_.name,
                    f"local cache miss at {cache_path}; "
                    f"run `lexicon fetch-bulk-sources` first "
                    f"(or pass --fetch-bulk to rebuild-from-jsonl).",
                )
            )
            continue
        try:
            counts = ingest_wiktextract_path(db, cache_path, apply=apply)
            per_slice[slice_.name] = counts
        except Exception as e:  # noqa: BLE001 — surface as failure
            failed.append((slice_.name, str(e)))

    return IngestAllResult(per_slice=per_slice, fetched=fetched, failed=failed)
