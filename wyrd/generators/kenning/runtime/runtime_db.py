"""Runtime SQLite loader for the L4 kenning DB (wyrd-d90t PR 4).

Resolves a sqlite3 connection to the L4 DB on first call, then hands
out the same connection for the lifetime of the process (warm-Lambda
reuse). Resolution order:

1. **``WYRD_RUNTIME_DB`` env var** — absolute local path. Local dev /
   tests bypass S3 entirely by pointing at any ``.db`` file.
2. **/tmp/wyrd-runtime.db with cached ETag** — if ``WYRD_RUNTIME_DB_BUCKET``
   is set, HEAD the current.json pointer (~10ms) and skip the download
   when the cached ETag still matches. Cold-start fast path on warm
   container reuse where /tmp survives.
3. **Download from S3** — fetch ``current.json`` to learn the
   versioned key, download that key to /tmp, record the new ETag.
4. **Bundled seed fallback** — when no env var is set AND no S3
   bucket is configured (or the download fails), fall back to the
   committed ``wyrd/generators/kenning/data/seed-runtime.db``. Lets
   tests + offline dev produce names without AWS.

The connection is opened with the ``file:...?mode=ro&immutable=1``
URI so SQLite knows it can skip locking + WAL setup — both are pure
overhead on a read-only file the runtime treats as immutable for the
container's lifetime. ``check_same_thread=False`` lets gunicorn-style
worker pools share the connection without re-opening per request
(CPython's sqlite3 module serializes concurrent calls on a per-
Connection lock; the access pattern here is read-only so workers
contend on the lock but don't corrupt state).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from importlib import resources
from pathlib import Path

_logger = logging.getLogger(__name__)

# Cache locations on /tmp — chosen to match Lambda's writable
# ephemeral storage. The container's lifetime determines how long the
# cache survives; the ETag check on cold start verifies it's still
# current before reusing.
_RUNTIME_DB_TMP_PATH = "/tmp/wyrd-runtime.db"
_RUNTIME_DB_ETAG_PATH = "/tmp/wyrd-runtime.etag"

# Environment variable surface. WYRD_RUNTIME_DB is the operator
# override; WYRD_RUNTIME_DB_BUCKET names the S3 bucket to fetch from
# (production sets it via terraform — see infra/). When neither is set
# the loader falls back to the bundled seed.
ENV_LOCAL_PATH = "WYRD_RUNTIME_DB"
ENV_BUCKET = "WYRD_RUNTIME_DB_BUCKET"
ENV_AWS_PROFILE = "WYRD_RUNTIME_DB_AWS_PROFILE"  # optional; usually unset in Lambda

# Pointer key matches what bin/publish-runtime-db.sh writes.
_CURRENT_POINTER_KEY = "current.json"


# Process-wide cached connection + lock. The lock guards the first-call
# resolution so concurrent threads (gunicorn workers, async handlers)
# don't all race to download.
_conn: sqlite3.Connection | None = None
_conn_lock = threading.Lock()


def get_runtime_db() -> sqlite3.Connection:
    """Return the process-wide read-only SQLite connection to the L4 DB.

    Lazy + idempotent: first call resolves the DB path (env var → /tmp
    cache → S3 download → bundled seed), opens a read-only sqlite3
    connection, and caches it on the module. Every subsequent call —
    same process or warm-Lambda reuse — returns the cached handle.

    Read-only at the SQLite level via ``file:...?mode=ro&immutable=1``
    URI; the L4 DB is treated as a build-time artifact for the
    container's lifetime, and re-emit + re-upload (not in-place edit)
    is the only update path.
    """
    global _conn
    if _conn is not None:
        return _conn
    with _conn_lock:
        if _conn is not None:
            return _conn
        path = _resolve_db_path()
        _conn = _open_readonly(path)
    return _conn


def reset_runtime_db_cache() -> None:
    """Drop the cached connection so the next ``get_runtime_db`` call
    re-resolves the path + reopens. For tests and operator-driven
    reload (a stat-the-pointer hook in a future PR could rotate
    connections without restarting the container)."""
    global _conn
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except sqlite3.Error:
                _logger.exception("ignoring close error on cached runtime DB")
        _conn = None


def _resolve_db_path() -> Path:
    """Pick the L4 DB file the connection should open against.

    The four-step resolution order is documented at module level; this
    function is the executable copy of that contract. Each branch
    logs at INFO so Lambda Cloudwatch surfaces which path served the
    cold start.
    """
    env_path = os.environ.get(ENV_LOCAL_PATH)
    if env_path:
        path = Path(env_path)
        if not path.is_file():
            raise FileNotFoundError(f"{ENV_LOCAL_PATH}={env_path!r} but the file doesn't exist")
        _logger.info("runtime DB resolved via %s: %s", ENV_LOCAL_PATH, path)
        return path

    bucket = os.environ.get(ENV_BUCKET)
    if bucket:
        downloaded = _download_with_etag_cache(bucket)
        if downloaded is not None:
            return downloaded
        # S3 path was configured but the download failed; fall through
        # to the bundled seed so the runtime still serves rather than
        # 500ing — the failure is logged so the operator notices.
        _logger.warning("runtime DB S3 download failed; falling back to bundled seed")

    seed_path = _bundled_seed_path()
    _logger.info("runtime DB resolved via bundled seed: %s", seed_path)
    return seed_path


def _download_with_etag_cache(bucket: str) -> Path | None:
    """Fetch the L4 DB from S3 with HEAD-then-GET ETag short-circuit.

    Returns the path to the cached file on success, or None on any
    failure (caller falls back to the bundled seed).

    Flow:
      1. GET current.json — learn the versioned-key + expected etag.
      2. If /tmp/wyrd-runtime.db exists AND /tmp/wyrd-runtime.etag
         matches the pointer's etag, skip the download.
      3. Otherwise, GET the versioned key to /tmp, write the etag.

    Lazy-imports boto3 so the loader's import cost stays minimal when
    the deployment doesn't need S3 (tests, bundled-seed-only builds).
    """
    try:
        import json

        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        _logger.warning("boto3 unavailable; S3 runtime DB download skipped")
        return None

    profile = os.environ.get(ENV_AWS_PROFILE)
    try:
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        s3 = session.client("s3")

        pointer_obj = s3.get_object(Bucket=bucket, Key=_CURRENT_POINTER_KEY)
        # Close the StreamingBody via context manager so the underlying
        # HTTP connection returns to the pool. TypeError covers the case
        # where the JSON parses to a non-dict (null / list / scalar) and
        # the subsequent ["key"] subscript raises rather than KeyError.
        with pointer_obj["Body"] as stream:
            pointer = json.load(stream)
        version_key = pointer["key"]
        expected_etag = pointer["etag"]
    except (ClientError, BotoCoreError, KeyError, ValueError, TypeError):
        _logger.exception(
            "failed to fetch s3://%s/%s — falling back to bundled seed",
            bucket,
            _CURRENT_POINTER_KEY,
        )
        return None

    cached_path = Path(_RUNTIME_DB_TMP_PATH)
    cached_etag = _read_cached_etag()
    if cached_path.is_file() and cached_etag == expected_etag:
        _logger.info(
            "runtime DB cache hit (%s, etag=%s); skipping download",
            cached_path,
            expected_etag,
        )
        return cached_path

    try:
        s3.download_file(bucket, version_key, str(cached_path))
    except (ClientError, BotoCoreError):
        _logger.exception(
            "failed to download s3://%s/%s — falling back to bundled seed",
            bucket,
            version_key,
        )
        return None

    # Etag-cache write is best-effort: a failure here only means the
    # NEXT cold start won't get the short-circuit (it'll redownload).
    # Falling through with the downloaded DB intact is still a valid
    # serving state — log + continue rather than crash the load.
    try:
        Path(_RUNTIME_DB_ETAG_PATH).write_text(expected_etag)
    except OSError:
        _logger.exception(
            "failed to write etag cache at %s; next cold start will re-download",
            _RUNTIME_DB_ETAG_PATH,
        )
    _logger.info(
        "runtime DB downloaded from s3://%s/%s to %s",
        bucket,
        version_key,
        cached_path,
    )
    return cached_path


def _read_cached_etag() -> str | None:
    """Return the etag of the currently-cached /tmp DB, or None if no
    cache exists. Safe against partial-write states — a half-written
    etag file just produces a cache miss + redownload."""
    try:
        return Path(_RUNTIME_DB_ETAG_PATH).read_text().strip() or None
    except (OSError, UnicodeDecodeError):
        return None


def _bundled_seed_path() -> Path:
    """Return the on-disk path of the committed seed-runtime.db.

    Falls back to the bundled artifact. Raises FileNotFoundError when
    the seed is missing — the runtime can't do anything without a
    DB to read.
    """
    seed = resources.files("wyrd.generators.kenning.data").joinpath("seed-runtime.db")
    # resources.files returns a Traversable; on a normal filesystem
    # install the str() round-trip is fine. Zip-loader installs (where
    # the package is shipped as a .zip on sys.path) would need
    # resources.as_file() to materialize the binary on disk. Lambda's
    # default deployment shape is unzipped, but Lambda Layers + future
    # zipapp packaging would hit this — filed as a follow-up; see the
    # wyrd-d90t epic for the open ticket.
    path = Path(str(seed))
    if not path.is_file():
        raise FileNotFoundError(
            f"no runtime DB available: seed at {path} is missing AND no "
            f"{ENV_LOCAL_PATH} / {ENV_BUCKET} env var is set"
        )
    return path


def _open_readonly(path: Path) -> sqlite3.Connection:
    """Open ``path`` as a read-only sqlite3 connection.

    URI flags:
      * ``mode=ro`` — error on any write attempt; lets SQLite skip
        WAL setup and acquire a shared (rather than exclusive) lock.
      * ``immutable=1`` — promises the file won't change on disk for
        the connection's lifetime. SQLite can drop additional locking
        + skip per-page checksums.

    ``check_same_thread=False`` lets WSGI worker pools share the
    connection without per-request reopen. Safe here because the
    access is read-only and CPython's sqlite3 module wraps each
    Connection in a per-instance lock that serializes concurrent
    calls — workers may contend on that lock but won't corrupt state.
    """
    # Path.as_uri() handles spaces, '?', '#', and Windows backslashes
    # correctly; the URI-query flags get appended after.
    uri = f"{path.absolute().as_uri()}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)
