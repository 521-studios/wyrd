"""Tests for the bulk-source S3 sync helpers — wyrd-0vj3."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyrd.generators.kenning.bulk_sources import (
    DEFAULT_LOCAL_CACHE_DIR,
    ENV_BUCKET,
    ENV_PROFILE,
    ENV_REGION,
    ENV_SOURCES_DIR,
    Config,
    Manifest,
    ManifestError,
    Slice,
    expected_slice_path,
    fetch_missing_slices,
    hash_file_sha256,
    load_config,
    load_manifest,
    manifest_to_json,
    open_jsonl,
    upload_slices,
    verify_local_cache,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_manifest(path: Path, slices: list[dict] | None = None) -> None:
    payload = {
        "schema_version": 1,
        "bucket": "test-bucket",
        "region": "us-east-2",
        "s3_prefix": "wiktextract/v1",
        "compression": "zstd",
        "slices": slices or [],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_slice(name: str = "wiktextract_xx", sha: str = "x" * 64) -> Slice:
    return Slice(
        name=name,
        s3_key=f"wiktextract/v1/{name.split('_')[-1]}.jsonl.zst",
        sha256=sha,
        raw_size_bytes=1000,
        compressed_size_bytes=100,
    )


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    p = tmp_path / "_bulk_manifest.json"
    _write_manifest(
        p,
        slices=[
            {
                "name": "wiktextract_xx",
                "s3_key": "wiktextract/v1/xx.jsonl.zst",
                "sha256": "a" * 64,
                "raw_size_bytes": 1234,
                "compressed_size_bytes": 200,
            }
        ],
    )
    return p


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


def test_load_manifest_happy_path(manifest_path: Path):
    """Well-formed manifest parses to a Manifest with typed slices."""
    m = load_manifest(manifest_path)
    assert m.schema_version == 1
    assert m.bucket == "test-bucket"
    assert m.region == "us-east-2"
    assert len(m.slices) == 1
    assert m.slices[0].name == "wiktextract_xx"
    assert m.slices[0].sha256 == "a" * 64


def test_load_manifest_missing_file_raises(tmp_path: Path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nope.json")


def test_load_manifest_invalid_json_raises(tmp_path: Path):
    """Surface JSON errors at the boundary — operator gets a clear message."""
    p = tmp_path / "bad.json"
    p.write_text("{ not json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(p)


def test_load_manifest_unsupported_schema_raises(tmp_path: Path):
    """schema_version is the forward-compat lever; reject unknowns
    rather than silently parsing potentially-wrong shapes."""
    p = tmp_path / "future.json"
    p.write_text(json.dumps({"schema_version": 999, "slices": []}))
    with pytest.raises(ManifestError, match="schema_version=999"):
        load_manifest(p)


def test_load_manifest_missing_slice_field_raises(tmp_path: Path):
    """A slice without sha256 / s3_key / etc is unusable — error loudly."""
    p = tmp_path / "partial.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bucket": "test-bucket",
                "region": "us-east-2",
                "s3_prefix": "wiktextract/v1",
                "compression": "zstd",
                "slices": [{"name": "x"}],
            }
        )
    )
    with pytest.raises(ManifestError, match="missing required field"):
        load_manifest(p)


def test_manifest_to_json_round_trips(manifest_path: Path, tmp_path: Path):
    """Read → reserialize → re-read gives the same Manifest. Stable
    so commit diffs only show real changes."""
    m1 = load_manifest(manifest_path)
    text = manifest_to_json(m1)
    p2 = tmp_path / "rt.json"
    p2.write_text(text)
    m2 = load_manifest(p2)
    assert m1 == m2


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def _basic_manifest() -> Manifest:
    return Manifest(
        schema_version=1,
        bucket="default-bucket",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(),
    )


def test_load_config_manifest_defaults_when_no_overrides(tmp_path: Path):
    """No config file + no env → use manifest's bucket/region.
    Profile defaults to None (boto3 then picks up default chain)."""
    m = _basic_manifest()
    cfg = load_config(m, config_path=tmp_path / "missing.toml", env={})
    assert cfg.bucket == "default-bucket"
    assert cfg.region == "us-east-2"
    assert cfg.profile is None
    assert cfg.local_cache_dir == DEFAULT_LOCAL_CACHE_DIR


def test_load_config_toml_overrides_manifest(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "[bulk_storage]\n"
        'bucket = "my-bucket"\n'
        'region = "us-west-2"\n'
        'profile = "my-profile"\n'
        f'local_cache_dir = "{tmp_path / "sources"}"\n'
    )
    cfg = load_config(_basic_manifest(), config_path=cfg_path, env={})
    assert cfg.bucket == "my-bucket"
    assert cfg.region == "us-west-2"
    assert cfg.profile == "my-profile"
    assert cfg.local_cache_dir == tmp_path / "sources"


def test_load_config_env_overrides_toml(tmp_path: Path):
    """Env > config.toml > manifest — confirm each rung."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[bulk_storage]\nbucket = "toml-bucket"\nregion = "us-west-2"\nprofile = "toml-profile"\n'
    )
    env = {
        ENV_BUCKET: "env-bucket",
        ENV_REGION: "eu-west-1",
        ENV_PROFILE: "env-profile",
        ENV_SOURCES_DIR: str(tmp_path / "env-sources"),
    }
    cfg = load_config(_basic_manifest(), config_path=cfg_path, env=env)
    assert cfg.bucket == "env-bucket"
    assert cfg.region == "eu-west-1"
    assert cfg.profile == "env-profile"
    assert cfg.local_cache_dir == tmp_path / "env-sources"


def test_load_config_malformed_toml_raises(tmp_path: Path):
    p = tmp_path / "bad.toml"
    p.write_text("not = toml = [")
    with pytest.raises(ManifestError, match="not valid TOML"):
        load_config(_basic_manifest(), config_path=p, env={})


# ---------------------------------------------------------------------------
# hash + expected_slice_path
# ---------------------------------------------------------------------------


def test_hash_file_sha256_deterministic(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    h = hash_file_sha256(p)
    # sha256 of b"hello world"
    assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_expected_slice_path_in_local_cache(tmp_path: Path):
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=tmp_path)
    s = _make_slice()
    assert expected_slice_path(cfg, s) == tmp_path / "xx.jsonl.zst"


# ---------------------------------------------------------------------------
# verify_local_cache
# ---------------------------------------------------------------------------


def test_verify_local_cache_all_missing(tmp_path: Path):
    """Empty cache → every slice flagged needs_fetch."""
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(_make_slice(),),
    )
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=tmp_path)
    statuses = verify_local_cache(m, cfg)
    assert len(statuses) == 1
    assert not statuses[0].present
    assert statuses[0].needs_fetch


def test_verify_local_cache_sha_match(tmp_path: Path):
    """Present file with matching sha256 → not needs_fetch."""
    payload = b"slice contents"
    sha = hash_file_sha256(_tmp_write(tmp_path / "xx.jsonl.zst", payload))
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(_make_slice(sha=sha),),
    )
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=tmp_path)
    statuses = verify_local_cache(m, cfg)
    assert statuses[0].present
    assert statuses[0].sha256_matches
    assert not statuses[0].needs_fetch


def test_verify_local_cache_sha_mismatch(tmp_path: Path):
    """Present file with wrong sha → flagged needs_fetch."""
    _tmp_write(tmp_path / "xx.jsonl.zst", b"contents")
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(_make_slice(sha="0" * 64),),
    )
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=tmp_path)
    statuses = verify_local_cache(m, cfg)
    assert statuses[0].present
    assert not statuses[0].sha256_matches
    assert statuses[0].needs_fetch


def _tmp_write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------------------
# fetch_missing_slices (mock S3 client)
# ---------------------------------------------------------------------------


class _FakeS3Client:
    """Tiny stand-in: ``download_file`` writes the bytes from
    ``self.objects[key]`` to ``dest``; ``upload_file`` records the
    upload. Side-stepping moto entirely for these tests keeps them
    fast and dependency-light."""

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects: dict[str, bytes] = dict(objects or {})
        self.uploads: list[tuple[str, str, str]] = []

    def download_file(self, bucket: str, key: str, dest: str) -> None:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.objects[key])

    def upload_file(self, src: str, bucket: str, key: str) -> None:
        self.objects[key] = Path(src).read_bytes()
        self.uploads.append((bucket, key, src))


def test_fetch_missing_slices_downloads_when_absent(tmp_path: Path):
    """Slice absent → fetched. sha256 from manifest gets validated
    against downloaded bytes."""
    payload = b"compressed contents"
    sha = hash_file_sha256(_tmp_write(tmp_path / "_seed.bin", payload))
    slice_ = Slice(
        name="wiktextract_xx",
        s3_key="wiktextract/v1/xx.jsonl.zst",
        sha256=sha,
        raw_size_bytes=1,
        compressed_size_bytes=len(payload),
    )
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(slice_,),
    )
    cache = tmp_path / "cache"
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=cache)
    client = _FakeS3Client({slice_.s3_key: payload})

    result = fetch_missing_slices(m, cfg, s3_client=client)
    assert result.fetched == ["wiktextract_xx"]
    assert result.skipped == []
    assert result.failed == []
    assert (cache / "xx.jsonl.zst").exists()


def test_fetch_missing_slices_sha_mismatch_after_download_fails(tmp_path: Path):
    """Downloaded bytes whose sha doesn't match the manifest → the
    slice lands in failed[], NOT in fetched[]. This protects
    against silent corruption on the bucket side."""
    payload = b"real contents"
    slice_ = Slice(
        name="wiktextract_xx",
        s3_key="wiktextract/v1/xx.jsonl.zst",
        sha256="0" * 64,  # deliberately wrong
        raw_size_bytes=1,
        compressed_size_bytes=len(payload),
    )
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(slice_,),
    )
    cache = tmp_path / "cache"
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=cache)
    client = _FakeS3Client({slice_.s3_key: payload})

    result = fetch_missing_slices(m, cfg, s3_client=client)
    assert result.fetched == []
    assert len(result.failed) == 1
    assert "sha256 mismatch" in result.failed[0][1]
    # The corrupt bytes are unlinked so verify-bulk-sources doesn't
    # report a stale present-but-mismatched state on the next run.
    assert not (tmp_path / "cache" / "xx.jsonl.zst").exists()


class _RaisingS3Client:
    """Stand-in that raises a generic exception on ``download_file``
    — exercises the ``except Exception`` boto3-error path."""

    def download_file(self, bucket: str, key: str, dest: str) -> None:
        raise RuntimeError(f"simulated S3 failure on {key}")


def test_fetch_missing_slices_generic_boto3_exception_routes_to_failed(tmp_path: Path):
    """A boto3 download failure (network, NoSuchKey, AccessDenied,
    anything) lands the slice in failed[] with the error message,
    not in fetched[]. Keeps the bare-except path covered."""
    slice_ = _make_slice()
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(slice_,),
    )
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=tmp_path)
    result = fetch_missing_slices(m, cfg, s3_client=_RaisingS3Client())
    assert result.fetched == []
    assert len(result.failed) == 1
    assert "simulated S3 failure" in result.failed[0][1]


def test_fetch_missing_slices_skips_when_sha_matches(tmp_path: Path):
    """Local file already matches manifest → skip, don't re-download."""
    payload = b"matching contents"
    sha = hash_file_sha256(_tmp_write(tmp_path / "_seed.bin", payload))
    cache = tmp_path / "cache"
    _tmp_write(cache / "xx.jsonl.zst", payload)
    slice_ = Slice(
        name="wiktextract_xx",
        s3_key="wiktextract/v1/xx.jsonl.zst",
        sha256=sha,
        raw_size_bytes=1,
        compressed_size_bytes=len(payload),
    )
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(slice_,),
    )
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=cache)
    client = _FakeS3Client()

    result = fetch_missing_slices(m, cfg, s3_client=client)
    assert result.fetched == []
    assert result.skipped == ["wiktextract_xx"]


def test_fetch_missing_slices_force_redownloads_even_on_match(tmp_path: Path):
    payload = b"matching"
    sha = hash_file_sha256(_tmp_write(tmp_path / "_seed.bin", payload))
    cache = tmp_path / "cache"
    _tmp_write(cache / "xx.jsonl.zst", payload)
    slice_ = Slice(
        name="wiktextract_xx",
        s3_key="wiktextract/v1/xx.jsonl.zst",
        sha256=sha,
        raw_size_bytes=1,
        compressed_size_bytes=len(payload),
    )
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(slice_,),
    )
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=cache)
    client = _FakeS3Client({slice_.s3_key: payload})

    result = fetch_missing_slices(m, cfg, force=True, s3_client=client)
    assert result.fetched == ["wiktextract_xx"]


def test_fetch_missing_slices_dry_run_skips_io(tmp_path: Path):
    """--dry-run reports candidates without downloading."""
    slice_ = _make_slice()
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(slice_,),
    )
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=tmp_path)
    client = _FakeS3Client({slice_.s3_key: b"would-be-downloaded"})

    result = fetch_missing_slices(m, cfg, dry_run=True, s3_client=client)
    assert result.fetched == ["wiktextract_xx"]
    assert not (tmp_path / "xx.jsonl.zst").exists()


def test_fetch_missing_slices_slice_filter(tmp_path: Path):
    """slice_names=[...] restricts to a subset."""
    payload = b"contents"
    sha = hash_file_sha256(_tmp_write(tmp_path / "_seed.bin", payload))
    s1 = Slice("wiktextract_a", "wiktextract/v1/a.jsonl.zst", sha, 1, len(payload))
    s2 = Slice("wiktextract_b", "wiktextract/v1/b.jsonl.zst", sha, 1, len(payload))
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(s1, s2),
    )
    cache = tmp_path / "cache"
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=cache)
    client = _FakeS3Client({s1.s3_key: payload, s2.s3_key: payload})

    result = fetch_missing_slices(m, cfg, slice_names=["wiktextract_a"], s3_client=client)
    assert result.fetched == ["wiktextract_a"]
    assert (cache / "a.jsonl.zst").exists()
    assert not (cache / "b.jsonl.zst").exists()


# ---------------------------------------------------------------------------
# upload_slices
# ---------------------------------------------------------------------------


def test_upload_slices_uploads_existing_zst(tmp_path: Path):
    """Pre-compressed .jsonl.zst in cache → upload as-is, manifest
    sha256 updated to actual."""
    payload = b"compressed bytes"
    cache = tmp_path / "cache"
    _tmp_write(cache / "xx.jsonl.zst", payload)
    actual_sha = hash_file_sha256(cache / "xx.jsonl.zst")
    slice_ = Slice(
        name="wiktextract_xx",
        s3_key="wiktextract/v1/xx.jsonl.zst",
        sha256="0" * 64,  # stale; gets refreshed
        raw_size_bytes=999,
        compressed_size_bytes=999,
    )
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(slice_,),
    )
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=cache)
    client = _FakeS3Client()

    result = upload_slices(m, cfg, s3_client=client)
    assert result.uploaded == ["wiktextract_xx"]
    assert client.uploads == [("b", "wiktextract/v1/xx.jsonl.zst", str(cache / "xx.jsonl.zst"))]
    # New manifest reflects actual sha + compressed size
    new = result.new_manifest.slices[0]
    assert new.sha256 == actual_sha
    assert new.compressed_size_bytes == len(payload)


def test_upload_slices_skips_when_neither_local_form_present(tmp_path: Path):
    """No .jsonl.zst AND no .jsonl in cache → skipped (operator
    hasn't mined this slice locally yet)."""
    slice_ = _make_slice()
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(slice_,),
    )
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=tmp_path)
    client = _FakeS3Client()

    result = upload_slices(m, cfg, s3_client=client)
    assert result.skipped == ["wiktextract_xx"]
    assert result.uploaded == []
    # Skipped slices keep their existing manifest entry unchanged.
    assert result.new_manifest.slices[0] == slice_


def test_upload_slices_compresses_raw_jsonl_before_upload(tmp_path: Path):
    """Cache has only .jsonl (no .zst) → compress on the fly, upload
    .jsonl.zst. The .zst lands in the cache as a side effect."""
    cache = tmp_path / "cache"
    raw = cache / "xx.jsonl"
    _tmp_write(
        raw,
        b'{"row": 1}\n{"row": 2}\n' * 200,  # compressible repeated text
    )
    slice_ = Slice(
        name="wiktextract_xx",
        s3_key="wiktextract/v1/xx.jsonl.zst",
        sha256="0" * 64,
        raw_size_bytes=999,
        compressed_size_bytes=999,
    )
    m = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(slice_,),
    )
    cfg = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=cache)
    client = _FakeS3Client()

    result = upload_slices(m, cfg, s3_client=client)
    assert result.uploaded == ["wiktextract_xx"]
    assert (cache / "xx.jsonl.zst").exists()
    # Compressed should be smaller than raw for this repetitive payload.
    assert (cache / "xx.jsonl.zst").stat().st_size < raw.stat().st_size


# ---------------------------------------------------------------------------
# open_jsonl transparent decompression
# ---------------------------------------------------------------------------


def test_open_jsonl_plain_text(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n')
    with open_jsonl(p) as fh:
        lines = list(fh)
    assert lines == ['{"a": 1}\n', '{"b": 2}\n']


def test_open_jsonl_zstd(tmp_path: Path):
    """``.jsonl.zst`` decompresses transparently — ingest path can
    read either form."""
    import zstandard

    raw = '{"a": 1}\n{"b": 2}\n'
    p = tmp_path / "x.jsonl.zst"
    with p.open("wb") as fh:
        cctx = zstandard.ZstdCompressor()
        fh.write(cctx.compress(raw.encode("utf-8")))
    with open_jsonl(p) as fh:
        out = fh.read()
    assert out == raw


# ---------------------------------------------------------------------------
# CLI smoke tests
#
# The bulk-source CLI commands wrap library helpers that are
# thoroughly unit-tested above; these tests only exercise the CLI
# wiring (Click option parsing, manifest path resolution, exit
# codes, push-side manifest write-back). Each test patches
# load_manifest / load_config / fetch / upload / verify on the
# bulk_sources module so no real S3 or filesystem chatter happens.
# ---------------------------------------------------------------------------


def _patched_cli_test(monkeypatch, **overrides):
    """Wire monkeypatch overrides onto wyrd.generators.kenning.bulk_sources.
    Returns a CliRunner ready to invoke `cli`."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import bulk_sources as bs

    for name, value in overrides.items():
        monkeypatch.setattr(bs, name, value)
    return CliRunner()


def test_cli_fetch_bulk_sources_dry_run_exits_zero(tmp_path: Path, monkeypatch):
    """fetch-bulk-sources --dry-run: helper returns clean FetchResult,
    CLI prints 'dry-run' marker and exits 0."""
    from wyrd.generators.kenning.bulk_sources import FetchResult

    manifest = _basic_manifest()
    config = Config(bucket="b", region="us-east-2", profile=None, local_cache_dir=tmp_path)
    calls: dict[str, object] = {}

    def fake_fetch(m, c, *, slice_names=None, force=False, dry_run=False, s3_client=None):
        calls.update(manifest_=m, config_=c, slice_names=slice_names, force=force, dry_run=dry_run)
        return FetchResult(fetched=["wiktextract_xx"], skipped=[], failed=[])

    runner = _patched_cli_test(
        monkeypatch,
        load_manifest=lambda *a, **kw: manifest,
        load_config=lambda *a, **kw: config,
        fetch_missing_slices=fake_fetch,
    )
    from wyrd.generators.kenning.cli import cli as cli_root

    result = runner.invoke(cli_root, ["lexicon", "fetch-bulk-sources", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert calls["dry_run"] is True


def test_cli_fetch_bulk_sources_failure_exits_one(tmp_path: Path, monkeypatch):
    """A populated FetchResult.failed list → exit 1 + the reason
    prints in the output. Lets CI / wrapper scripts detect the
    fetch-broke condition without parsing markdown."""
    from wyrd.generators.kenning.bulk_sources import FetchResult

    runner = _patched_cli_test(
        monkeypatch,
        load_manifest=lambda *a, **kw: _basic_manifest(),
        load_config=lambda *a, **kw: Config(
            bucket="b", region="us-east-2", profile=None, local_cache_dir=tmp_path
        ),
        fetch_missing_slices=lambda *a, **kw: FetchResult(
            fetched=[], skipped=[], failed=[("slice_a", "AccessDenied")]
        ),
    )
    from wyrd.generators.kenning.cli import cli as cli_root

    result = runner.invoke(cli_root, ["lexicon", "fetch-bulk-sources"])
    assert result.exit_code == 1
    assert "AccessDenied" in result.output


def test_cli_push_bulk_sources_writes_new_manifest(tmp_path: Path, monkeypatch):
    """push-bulk-sources success path: the upload helper's
    new_manifest is written to the --manifest path on disk."""
    from wyrd.generators.kenning.bulk_sources import Slice, UploadResult

    manifest_path = tmp_path / "out_manifest.json"
    starting_manifest = _basic_manifest()
    updated = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(Slice("wiktextract_xx", "wiktextract/v1/xx.jsonl.zst", "a" * 64, 100, 50),),
    )

    runner = _patched_cli_test(
        monkeypatch,
        load_manifest=lambda *a, **kw: starting_manifest,
        load_config=lambda *a, **kw: Config(
            bucket="b", region="us-east-2", profile=None, local_cache_dir=tmp_path
        ),
        upload_slices=lambda *a, **kw: UploadResult(
            uploaded=["wiktextract_xx"], skipped=[], failed=[], new_manifest=updated
        ),
    )
    from wyrd.generators.kenning.cli import cli as cli_root

    result = runner.invoke(
        cli_root, ["lexicon", "push-bulk-sources", "--manifest", str(manifest_path)]
    )
    assert result.exit_code == 0, result.output
    assert manifest_path.exists()
    written = json.loads(manifest_path.read_text())
    assert written["slices"][0]["name"] == "wiktextract_xx"


def test_cli_verify_bulk_sources_missing_exits_one(tmp_path: Path, monkeypatch):
    """verify-bulk-sources reports mismatches + missing slices and
    exits 1 so a wrapper script can detect drift."""
    from wyrd.generators.kenning.bulk_sources import SliceStatus

    runner = _patched_cli_test(
        monkeypatch,
        load_manifest=lambda *a, **kw: _basic_manifest(),
        load_config=lambda *a, **kw: Config(
            bucket="b", region="us-east-2", profile=None, local_cache_dir=tmp_path
        ),
        verify_local_cache=lambda *a, **kw: [
            SliceStatus(
                slice_name="wiktextract_xx",
                cache_path=tmp_path / "xx.jsonl.zst",
                present=False,
                sha256_matches=False,
            )
        ],
    )
    from wyrd.generators.kenning.cli import cli as cli_root

    result = runner.invoke(cli_root, ["lexicon", "verify-bulk-sources"])
    assert result.exit_code == 1
    assert "MISSING" in result.output


def test_cli_verify_bulk_sources_clean_exits_zero(tmp_path: Path, monkeypatch):
    """All slices present + sha-matched → exit 0."""
    from wyrd.generators.kenning.bulk_sources import SliceStatus

    runner = _patched_cli_test(
        monkeypatch,
        load_manifest=lambda *a, **kw: _basic_manifest(),
        load_config=lambda *a, **kw: Config(
            bucket="b", region="us-east-2", profile=None, local_cache_dir=tmp_path
        ),
        verify_local_cache=lambda *a, **kw: [
            SliceStatus(
                slice_name="wiktextract_xx",
                cache_path=tmp_path / "xx.jsonl.zst",
                present=True,
                sha256_matches=True,
            )
        ],
    )
    from wyrd.generators.kenning.cli import cli as cli_root

    result = runner.invoke(cli_root, ["lexicon", "verify-bulk-sources"])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# ingest_all_slices — wyrd-hidb Phase 1
# ---------------------------------------------------------------------------


def test_ingest_all_slices_calls_ingester_per_slice(tmp_path: Path, monkeypatch):
    """Each manifest slice with a present cache file triggers exactly
    one ``ingest_wiktextract_path`` call. The IngestAllResult.per_slice
    map keys by slice name; totals sum scalars across slices."""
    from wyrd.generators.kenning.bulk_sources import (
        IngestAllResult,
        Slice,
        ingest_all_slices,
    )

    cache = tmp_path / "cache"
    cache.mkdir()
    # Two slices, both present locally
    (cache / "a.jsonl.zst").write_bytes(b"slice-a")
    (cache / "b.jsonl.zst").write_bytes(b"slice-b")

    manifest = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(
            Slice("slice_a", "wiktextract/v1/a.jsonl.zst", "a" * 64, 10, 7),
            Slice("slice_b", "wiktextract/v1/b.jsonl.zst", "b" * 64, 10, 7),
        ),
    )

    monkeypatch.setattr(
        "wyrd.generators.kenning.bulk_sources.load_manifest",
        lambda *a, **kw: manifest,
    )
    monkeypatch.setattr(
        "wyrd.generators.kenning.bulk_sources.load_config",
        lambda *a, **kw: Config(
            bucket="b", region="us-east-2", profile=None, local_cache_dir=cache
        ),
    )

    calls: list[Path] = []

    def fake_ingest(db, path, *, apply=False, limit=None, since_line=0):
        calls.append(path)
        return {"lines_read": 100, "entries_parsed": 50}

    monkeypatch.setattr(
        "wyrd.generators.kenning.lexicon.wiktextract_ingester.ingest_wiktextract_path",
        fake_ingest,
    )

    result: IngestAllResult = ingest_all_slices(db=None, apply=True)
    assert len(calls) == 2
    assert {p.name for p in calls} == {"a.jsonl.zst", "b.jsonl.zst"}
    assert set(result.per_slice.keys()) == {"slice_a", "slice_b"}
    assert result.totals == {"lines_read": 200, "entries_parsed": 100}
    assert result.failed == []


def test_ingest_all_slices_missing_cache_without_fetch_reports_failure(tmp_path: Path, monkeypatch):
    """Slice absent locally + fetch=False → land in failed[] with a
    pointer to fetch-bulk-sources. Operator gets a clear next step
    instead of a stack trace."""
    from wyrd.generators.kenning.bulk_sources import Slice, ingest_all_slices

    cache = tmp_path / "cache"
    cache.mkdir()  # empty

    manifest = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(Slice("slice_a", "wiktextract/v1/a.jsonl.zst", "a" * 64, 10, 7),),
    )

    monkeypatch.setattr(
        "wyrd.generators.kenning.bulk_sources.load_manifest",
        lambda *a, **kw: manifest,
    )
    monkeypatch.setattr(
        "wyrd.generators.kenning.bulk_sources.load_config",
        lambda *a, **kw: Config(
            bucket="b", region="us-east-2", profile=None, local_cache_dir=cache
        ),
    )

    result = ingest_all_slices(db=None, apply=True, fetch=False)
    assert result.per_slice == {}
    assert len(result.failed) == 1
    assert "local cache miss" in result.failed[0][1]
    assert "fetch-bulk-sources" in result.failed[0][1]


def test_ingest_all_slices_with_fetch_downloads_first(tmp_path: Path, monkeypatch):
    """fetch=True → call fetch_missing_slices before iterating.
    Mocked S3 client materializes the cache file; ingester then runs."""
    from wyrd.generators.kenning.bulk_sources import Slice, ingest_all_slices

    payload = b"compressed-slice"
    sha = hash_file_sha256(_tmp_write(tmp_path / "_seed.bin", payload))
    cache = tmp_path / "cache"
    cache.mkdir()

    slice_ = Slice("slice_a", "wiktextract/v1/a.jsonl.zst", sha, 10, len(payload))
    manifest = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(slice_,),
    )

    monkeypatch.setattr(
        "wyrd.generators.kenning.bulk_sources.load_manifest",
        lambda *a, **kw: manifest,
    )
    monkeypatch.setattr(
        "wyrd.generators.kenning.bulk_sources.load_config",
        lambda *a, **kw: Config(
            bucket="b", region="us-east-2", profile=None, local_cache_dir=cache
        ),
    )

    fake = _FakeS3Client({slice_.s3_key: payload})
    monkeypatch.setattr(
        "wyrd.generators.kenning.lexicon.wiktextract_ingester.ingest_wiktextract_path",
        lambda db, path, **kw: {"lines_read": 1},
    )

    result = ingest_all_slices(db=None, apply=True, fetch=True, s3_client=fake)
    assert result.fetched == ["slice_a"]
    assert (cache / "a.jsonl.zst").exists()
    assert result.per_slice == {"slice_a": {"lines_read": 1}}


def test_ingest_all_slices_ingester_exception_routes_to_failed(tmp_path: Path, monkeypatch):
    """Ingester raising mid-slice → slice lands in failed[]; later
    slices still run. Defensive isolation so one bad slice doesn't
    abort the whole rebuild."""
    from wyrd.generators.kenning.bulk_sources import Slice, ingest_all_slices

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.jsonl.zst").write_bytes(b"a")
    (cache / "b.jsonl.zst").write_bytes(b"b")

    manifest = Manifest(
        schema_version=1,
        bucket="b",
        region="us-east-2",
        s3_prefix="wiktextract/v1",
        compression="zstd",
        slices=(
            Slice("bad", "wiktextract/v1/a.jsonl.zst", "a" * 64, 1, 1),
            Slice("good", "wiktextract/v1/b.jsonl.zst", "b" * 64, 1, 1),
        ),
    )

    monkeypatch.setattr(
        "wyrd.generators.kenning.bulk_sources.load_manifest",
        lambda *a, **kw: manifest,
    )
    monkeypatch.setattr(
        "wyrd.generators.kenning.bulk_sources.load_config",
        lambda *a, **kw: Config(
            bucket="b", region="us-east-2", profile=None, local_cache_dir=cache
        ),
    )

    def fake_ingest(db, path, *, apply=False, limit=None, since_line=0):
        if path.name == "a.jsonl.zst":
            raise RuntimeError("simulated parser crash")
        return {"lines_read": 5}

    monkeypatch.setattr(
        "wyrd.generators.kenning.lexicon.wiktextract_ingester.ingest_wiktextract_path",
        fake_ingest,
    )

    result = ingest_all_slices(db=None, apply=True)
    assert result.per_slice == {"good": {"lines_read": 5}}
    assert len(result.failed) == 1
    assert result.failed[0][0] == "bad"
    assert "simulated parser crash" in result.failed[0][1]
