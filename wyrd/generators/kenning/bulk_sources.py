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

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    def slice_by_name(self, name: str) -> Slice | None:
        for s in self.slices:
            if s.name == name:
                return s
        return None


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


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> Manifest:
    """Read ``_bulk_manifest.json``. Raises :class:`ManifestError`
    on missing file / parse failure / unknown schema_version."""
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
    config_path: Path = DEFAULT_CONFIG_PATH,
    env: dict[str, str] | None = None,
) -> Config:
    """Resolve effective config in precedence: env > config.toml > manifest.

    The default-manifest values are the 521 Studios staging bucket;
    forks / non-521 contributors override via config.toml or env vars.
    """
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


def write_default_config(manifest: Manifest, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Write a default ``config.toml`` derived from the manifest.
    Called on first bulk-command run when no config file exists."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        f"# wyrd-0vj3 bulk-source storage config.\n"
        f"# Auto-generated from data/mining/_bulk_manifest.json defaults.\n"
        f"# Override via env vars: {ENV_BUCKET}, {ENV_REGION}, "
        f"{ENV_PROFILE}, {ENV_SOURCES_DIR}.\n\n"
        f"[bulk_storage]\n"
        f'bucket = "{manifest.bucket}"\n'
        f'region = "{manifest.region}"\n'
        f'profile = "521-Staging-Admin"\n'
        f'local_cache_dir = "{DEFAULT_LOCAL_CACHE_DIR}"\n'
    )
    config_path.write_text(contents, encoding="utf-8")


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

    for slice_, status in zip(manifest.slices, statuses, strict=True):
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
        compressed_path = expected_slice_path(config, slice_)
        raw_path = compressed_path.with_suffix("")  # strip .zst -> .jsonl

        if compressed_path.exists():
            local = compressed_path
            raw_size = raw_path.stat().st_size if raw_path.exists() else slice_.raw_size_bytes
        elif raw_path.exists():
            try:
                _compress_raw_to_zst(raw_path, compressed_path)
            except Exception as e:  # noqa: BLE001
                failed.append((slice_.name, f"compression failed: {e}"))
                new_slices.append(slice_)
                continue
            local = compressed_path
            raw_size = raw_path.stat().st_size
        else:
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

    cctx = zstandard.ZstdCompressor(level=19)
    with raw_path.open("rb") as src, compressed_path.open("wb") as dst:
        cctx.copy_stream(src, dst)


def open_jsonl(path: Path) -> Any:
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
