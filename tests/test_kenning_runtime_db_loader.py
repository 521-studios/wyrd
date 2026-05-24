"""Tests for ``runtime/runtime_db.py`` loader (wyrd-d90t PR 4).

Covers the four-step resolution order (env var → /tmp cache → S3
download → bundled seed), lazy + cached connection behavior, and the
read-only sqlite3 contract.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from wyrd.generators.kenning.runtime import runtime_db as mod
from wyrd.generators.kenning.runtime.runtime_db import (
    ENV_AWS_PROFILE,
    ENV_BUCKET,
    ENV_LOCAL_PATH,
    _read_cached_etag,
    get_runtime_db,
    reset_runtime_db_cache,
)

TEST_BUCKET = "kenning-runtime-test"


@pytest.fixture(autouse=True)
def _clear_module_cache():
    """Every test starts with a fresh module-cached connection — the
    process-wide cache would otherwise leak across tests and produce
    confusing 'why did this test see the previous test's DB' bugs."""
    reset_runtime_db_cache()
    yield
    reset_runtime_db_cache()


@pytest.fixture
def _clear_env(monkeypatch):
    """Strip every WYRD_RUNTIME_DB* env var the loader cares about so
    tests start from a known state. Individual tests opt into specific
    overrides on top of this baseline."""
    for name in (ENV_LOCAL_PATH, ENV_BUCKET, ENV_AWS_PROFILE):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def _tiny_db(tmp_path: Path) -> Path:
    """A minimal valid sqlite3 DB the loader can open read-only."""
    path = tmp_path / "fixture.db"
    conn = sqlite3.connect(str(path))
    conn.executescript("CREATE TABLE meaning(k TEXT); INSERT INTO meaning VALUES('hi');")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def fake_s3(monkeypatch):
    """moto S3 fake + fake AWS creds so boto3's credential resolution
    doesn't hit operator SSO."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "moto-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "moto-test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=TEST_BUCKET)
        yield s3


# ---------- env var resolution ----------


def test_env_var_path_wins(_clear_env, _tiny_db, monkeypatch) -> None:
    """WYRD_RUNTIME_DB is the highest-priority resolution path —
    completely bypasses S3 + the bundled seed."""
    monkeypatch.setenv(ENV_LOCAL_PATH, str(_tiny_db))
    conn = get_runtime_db()
    row = conn.execute("SELECT k FROM meaning").fetchone()
    assert row[0] == "hi"


def test_env_var_missing_file_raises(_clear_env, tmp_path, monkeypatch) -> None:
    """Operator typo on WYRD_RUNTIME_DB should fail loudly at the
    loader, not produce a confusing 'unable to open database' deep in
    sqlite."""
    monkeypatch.setenv(ENV_LOCAL_PATH, str(tmp_path / "missing.db"))
    with pytest.raises(FileNotFoundError, match=ENV_LOCAL_PATH):
        get_runtime_db()


# ---------- bundled seed fallback ----------


def test_falls_back_to_bundled_seed(_clear_env) -> None:
    """With no env var + no bucket, the loader resolves to the
    committed seed-runtime.db. Validates the offline-dev path that
    the test suite + local dev rely on."""
    conn = get_runtime_db()
    # The bundled seed has the L4 schema; meaning table exists.
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "meaning" in tables


# ---------- read-only contract ----------


def test_connection_is_read_only(_clear_env, _tiny_db, monkeypatch) -> None:
    """The connection MUST reject writes — the L4 DB is treated as
    immutable for the container's lifetime."""
    monkeypatch.setenv(ENV_LOCAL_PATH, str(_tiny_db))
    conn = get_runtime_db()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("INSERT INTO meaning VALUES('write')")


# ---------- caching ----------


def test_returns_same_connection_across_calls(_clear_env, _tiny_db, monkeypatch) -> None:
    """Lazy + idempotent contract: two calls return the same handle.
    Warm-Lambda reuse depends on this."""
    monkeypatch.setenv(ENV_LOCAL_PATH, str(_tiny_db))
    a = get_runtime_db()
    b = get_runtime_db()
    assert a is b


def test_reset_cache_returns_fresh_connection(_clear_env, _tiny_db, monkeypatch) -> None:
    """reset_runtime_db_cache drops the cached handle so the next
    call re-opens. Tests + a future hot-reload hook both depend on
    this."""
    monkeypatch.setenv(ENV_LOCAL_PATH, str(_tiny_db))
    a = get_runtime_db()
    reset_runtime_db_cache()
    b = get_runtime_db()
    assert a is not b


# ---------- S3 download ----------


def test_downloads_from_s3_when_bucket_set(_clear_env, fake_s3, _tiny_db, monkeypatch) -> None:
    """With WYRD_RUNTIME_DB_BUCKET set + no env-var local path, the
    loader fetches current.json and downloads the referenced
    versioned key to /tmp."""
    # Seed S3 with a versioned key + pointer.
    version_key = "v/2026-05-24T20-30Z.db"
    fake_s3.put_object(Bucket=TEST_BUCKET, Key=version_key, Body=_tiny_db.read_bytes())
    pointer = json.dumps({"key": version_key, "etag": "abc123"})
    fake_s3.put_object(Bucket=TEST_BUCKET, Key="current.json", Body=pointer)

    monkeypatch.setenv(ENV_BUCKET, TEST_BUCKET)
    # Direct /tmp paths into the test's tmp_path so concurrent test
    # runs don't collide (and don't leave artifacts on the host).
    cache_dir = Path(_tiny_db).parent
    monkeypatch.setattr(mod, "_RUNTIME_DB_TMP_PATH", str(cache_dir / "wyrd-cache.db"))
    monkeypatch.setattr(mod, "_RUNTIME_DB_ETAG_PATH", str(cache_dir / "wyrd-cache.etag"))

    conn = get_runtime_db()
    row = conn.execute("SELECT k FROM meaning").fetchone()
    assert row[0] == "hi"
    # Cache file is what got opened.
    assert (cache_dir / "wyrd-cache.db").is_file()
    assert (cache_dir / "wyrd-cache.etag").read_text() == "abc123"


def test_etag_match_skips_download(_clear_env, fake_s3, _tiny_db, monkeypatch) -> None:
    """When the cached ETag matches the pointer's etag, no download
    happens — the loader returns the existing /tmp file. Saves the
    full GET on every cold start where the container's /tmp survived."""
    version_key = "v/2026-05-24T20-30Z.db"
    fake_s3.put_object(Bucket=TEST_BUCKET, Key=version_key, Body=b"stale-from-s3")
    pointer = json.dumps({"key": version_key, "etag": "cached-etag"})
    fake_s3.put_object(Bucket=TEST_BUCKET, Key="current.json", Body=pointer)

    cache_dir = Path(_tiny_db).parent
    cache_db = cache_dir / "wyrd-cache.db"
    cache_etag = cache_dir / "wyrd-cache.etag"
    # Pre-populate the cache with the tiny db + matching etag.
    cache_db.write_bytes(_tiny_db.read_bytes())
    cache_etag.write_text("cached-etag")

    monkeypatch.setenv(ENV_BUCKET, TEST_BUCKET)
    monkeypatch.setattr(mod, "_RUNTIME_DB_TMP_PATH", str(cache_db))
    monkeypatch.setattr(mod, "_RUNTIME_DB_ETAG_PATH", str(cache_etag))

    conn = get_runtime_db()
    # If we accidentally re-downloaded, the cache would now hold
    # "stale-from-s3" which isn't a valid sqlite file — connection
    # would have raised. Since this passes, the etag short-circuit
    # worked.
    row = conn.execute("SELECT k FROM meaning").fetchone()
    assert row[0] == "hi"


def test_etag_mismatch_redownloads(_clear_env, fake_s3, _tiny_db, monkeypatch) -> None:
    """When the cached ETag differs from the pointer's etag, redownload."""
    version_key = "v/2026-05-24T21-00Z.db"
    fake_s3.put_object(Bucket=TEST_BUCKET, Key=version_key, Body=_tiny_db.read_bytes())
    pointer = json.dumps({"key": version_key, "etag": "fresh-etag"})
    fake_s3.put_object(Bucket=TEST_BUCKET, Key="current.json", Body=pointer)

    cache_dir = Path(_tiny_db).parent
    cache_db = cache_dir / "wyrd-cache.db"
    cache_etag = cache_dir / "wyrd-cache.etag"
    # Pre-populate with a stale cache (different etag).
    cache_db.write_bytes(b"stale bytes that aren't a valid sqlite db")
    cache_etag.write_text("stale-etag")

    monkeypatch.setenv(ENV_BUCKET, TEST_BUCKET)
    monkeypatch.setattr(mod, "_RUNTIME_DB_TMP_PATH", str(cache_db))
    monkeypatch.setattr(mod, "_RUNTIME_DB_ETAG_PATH", str(cache_etag))

    conn = get_runtime_db()
    # Successful query proves the stale bytes got replaced.
    row = conn.execute("SELECT k FROM meaning").fetchone()
    assert row[0] == "hi"
    assert cache_etag.read_text() == "fresh-etag"


def test_s3_failure_falls_back_to_bundled_seed(_clear_env, monkeypatch) -> None:
    """If WYRD_RUNTIME_DB_BUCKET is set but the S3 call fails (no
    creds / wrong bucket / network), the loader should fall back to
    the bundled seed rather than hard-crashing the runtime."""
    # No moto context = real boto3 trying to hit S3 = will fail since
    # AWS creds env vars aren't valid for real AWS. The fallback
    # should kick in.
    monkeypatch.setenv(ENV_BUCKET, "nonexistent-bucket-3e5q")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "bogus")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "bogus")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")

    conn = get_runtime_db()
    # Fell back to seed-runtime.db, which has the L4 schema.
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "meaning" in tables


# ---------- _read_cached_etag helper ----------


def test_read_cached_etag_returns_none_when_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "_RUNTIME_DB_ETAG_PATH", str(tmp_path / "nope.etag"))
    assert _read_cached_etag() is None


def test_read_cached_etag_returns_stripped_value(tmp_path, monkeypatch) -> None:
    etag_file = tmp_path / "etag"
    etag_file.write_text("  abc123\n")
    monkeypatch.setattr(mod, "_RUNTIME_DB_ETAG_PATH", str(etag_file))
    assert _read_cached_etag() == "abc123"


def test_read_cached_etag_handles_empty_file(tmp_path, monkeypatch) -> None:
    etag_file = tmp_path / "etag"
    etag_file.write_text("")
    monkeypatch.setattr(mod, "_RUNTIME_DB_ETAG_PATH", str(etag_file))
    assert _read_cached_etag() is None
