"""Runtime SQLite loader for the L4 kenning DB (wyrd-d90t PR 4).

Resolves a sqlite3 connection to the L4 DB on first call, then hands
out the same connection for the lifetime of the process (warm-Lambda
reuse). Resolution order:

1. **``WYRD_RUNTIME_DB`` env var** — absolute local path. Local dev /
   tests bypass S3 entirely by pointing at any ``.db`` file.
2. **/tmp/wyrd-runtime-<etag>.db cache hit** — if ``WYRD_RUNTIME_DB_BUCKET``
   is set, fetch the current.json pointer and short-circuit when an
   etag-named cache file from a prior cold start already exists in
   /tmp (warm-container reuse where /tmp survives).
3. **Download from S3** — fetch ``current.json`` to learn the
   versioned key + etag, download that key to a unique .tmp file via
   ``tempfile.mkstemp``, atomic-rename into ``wyrd-runtime-<etag>.db``.
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

import contextlib
import logging
import os
import sqlite3
import tempfile
import threading
from importlib import resources
from pathlib import Path

_logger = logging.getLogger(__name__)

# Must match ``lexicon.runtime_db_export.SCHEMA_VERSION``. Bumped in
# lockstep when the L4 table shape changes incompatibly. The schema
# check on open (see ``_verify_schema_version``) raises a clear
# operator-facing error when the cached / downloaded DB's stamped
# version doesn't match — way more discoverable than letting a
# missing-table OperationalError surface inside a sampling query.
_EXPECTED_SCHEMA_VERSION = "4"  # wyrd-aicu.9 (D-4): genitive_split_prior singleton blob


class RuntimeDBVersionMismatch(RuntimeError):
    """Raised when the resolved L4 DB's ``bundle_metadata.schema_version``
    doesn't match :data:`_EXPECTED_SCHEMA_VERSION`. Operator action:
    re-run ``wyrd kenning lexicon export-runtime-db`` + re-publish to
    S3 / replace the bundled seed so the consumer is on the same
    schema as the producer."""


# Cache locations on /tmp — chosen to match Lambda's writable
# ephemeral storage. The container's lifetime determines how long the
# cache survives. Cache files are named ``wyrd-runtime-<etag>.db`` so
# that:
#   * Existence of the file IS the etag-match check (no side-file
#     to read first).
#   * A redownload triggered by a pointer-flip writes a DIFFERENT
#     filename, so any existing reader's ``immutable=1`` connection
#     keeps pointing at unchanged bytes — no multi-process race in
#     WSGI / gunicorn deployments where two workers might overlap
#     between a redownload and a still-open connection.
# Trade-off: old etag files accumulate in /tmp. On Lambda /tmp is
# ephemeral so it's free GC; on long-lived gunicorn the operator
# would want a periodic cleanup (out of scope for this loader).
_RUNTIME_DB_TMP_DIR = "/tmp"
_RUNTIME_DB_FILENAME_TEMPLATE = "wyrd-runtime-{etag}.db"

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

# wyrd-04oi: the DB file path the cached connection opened against, recorded so
# the decomposition matcher cache can find its ``<db>.matcher.pkl`` sidecar
# beside whichever DB served this container (env override / /tmp etag-cache /
# downloaded / bundled seed) — same file, same version.
_resolved_db_path: Path | None = None


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
        global _resolved_db_path
        _resolved_db_path = path
        # Assign to a local + verify BEFORE promoting to the module
        # global. Promoting first would leak the unverified handle on
        # ``RuntimeDBVersionMismatch``: the exception bubbles, but the
        # cached ``_conn`` survives + every subsequent call short-
        # circuits at the top-of-function cache-hit check, silently
        # bypassing the schema gate for the rest of the container's
        # lifetime.
        conn = _open_readonly(path)
        try:
            _verify_schema_version(conn, path)
        except Exception:
            try:
                conn.close()
            except sqlite3.Error:
                _logger.exception("ignoring close error on rejected runtime DB")
            raise
        _conn = conn
    return _conn


def runtime_matcher_path() -> Path | None:
    """wyrd-04oi: the decomposition matcher pickle beside the live runtime DB,
    or None when there's no sidecar to load.

    Resolved off the DB path the container actually opened (recorded by
    :func:`get_runtime_db`), so the matcher version always matches the DB
    version — they ship + cache as siblings (``<db>`` / ``<db>.matcher.pkl``).
    Returns None when no DB has been opened yet, or no sidecar exists beside it
    (cold-load then rebuilds the trie from the corpus)."""
    from .matcher_cache import matcher_sidecar_path

    if _resolved_db_path is None:
        return None
    sidecar = matcher_sidecar_path(_resolved_db_path)
    return sidecar if sidecar.exists() else None


def _verify_schema_version(conn: sqlite3.Connection, path: Path) -> None:
    """Read ``bundle_metadata.schema_version`` and raise
    :class:`RuntimeDBVersionMismatch` if it doesn't match the runtime's
    expected version. Pre-v2 DBs (no ``bundle_metadata`` table at all)
    also raise — there's no path forward for a v1 DB once the runtime
    has moved on.

    Called once per fresh connection (on the cache miss branch of
    ``get_runtime_db``). Warm calls reuse the verified connection."""
    try:
        row = conn.execute(
            "SELECT value FROM bundle_metadata WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.OperationalError as exc:
        raise RuntimeDBVersionMismatch(
            f"L4 runtime DB at {path} has no bundle_metadata table "
            f"(pre-v1 schema or corrupt file). Re-emit + re-publish."
        ) from exc
    if row is None or row[0] != _EXPECTED_SCHEMA_VERSION:
        actual = row[0] if row is not None else "(no schema_version row)"
        raise RuntimeDBVersionMismatch(
            f"L4 runtime DB at {path} stamped schema_version={actual!r}; "
            f"runtime expects {_EXPECTED_SCHEMA_VERSION!r}. Re-run "
            f"``wyrd kenning lexicon export-runtime-db`` against the L3 "
            f"and re-publish to S3 / replace the bundled seed."
        )


_bundle_version_cache: dict[str, str] | None = None


def bundle_version() -> dict[str, str]:
    """Return the L4 bundle's identity stamp from ``bundle_metadata``
    (wyrd-dsl5): ``schema_version`` / ``built_at`` / ``emitter_version`` /
    ``source_lexicon_db``.

    Surfaced on the API envelope (``bundle_version``) and stored on defect
    reports so a triager can tell which data slice a flagged name came from
    — the same seed + params can yield a different name after a re-emit, so
    the report pins the bundle it was reproducible against.

    Process-cached under ``_conn_lock`` (double-checked, mirroring
    ``get_runtime_db``): the bundle is immutable for the connection's
    lifetime (re-emit + re-upload is the only update path, which restarts
    the process). Reset via :func:`reset_runtime_db_cache`. Returns ``{}``
    (and caches it) when the DB predates the ``bundle_metadata`` table, so a
    callable that reads this never re-queries a missing table per request."""
    global _bundle_version_cache
    if _bundle_version_cache is not None:
        return _bundle_version_cache
    # Resolve the connection BEFORE taking _conn_lock: get_runtime_db()
    # acquires _conn_lock itself, and threading.Lock is non-reentrant — calling
    # it while holding the lock would self-deadlock on a cold connection.
    conn = get_runtime_db()
    with _conn_lock:
        if _bundle_version_cache is not None:
            return _bundle_version_cache
        try:
            rows = conn.execute("SELECT key, value FROM bundle_metadata").fetchall()
            _bundle_version_cache = {str(k): str(v) for k, v in rows}
        except sqlite3.OperationalError:
            # Pre-bundle_metadata DB. Cache empty so we don't re-query +
            # re-raise on every roll; the envelope treats {} as "no version".
            _logger.debug("bundle_metadata table absent; no version stamp")
            _bundle_version_cache = {}
    return _bundle_version_cache


def reset_runtime_db_cache() -> None:  # noqa: V103 — test/ops cache-reset helper
    """Drop the cached connection so the next ``get_runtime_db`` call
    re-resolves the path + reopens. For tests and operator-driven
    reload (a stat-the-pointer hook in a future PR could rotate
    connections without restarting the container)."""
    global _conn, _bundle_version_cache, _resolved_db_path
    global _seed_exit_stack, _seed_path
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except sqlite3.Error:
                _logger.exception("ignoring close error on cached runtime DB")
        _conn = None
        _bundle_version_cache = None
        _resolved_db_path = None
        # wyrd-5kw0: release any materialized seed (deletes the extracted temp
        # under zip-loader) AFTER the connection is closed, so the next resolve
        # re-materializes. No-op when the seed was never used.
        if _seed_exit_stack is not None:
            _seed_exit_stack.close()
            _seed_exit_stack = None
            _seed_path = None


def reset_runtime_db_connection() -> None:
    """Drop ONLY the cached sqlite connection — keep the bundle_version cache.

    For SnapStart restore (wyrd-g1wp): the snapshot captures the open ``_conn``
    object, but Lambda wipes ``/tmp`` on restore so the file backing it is gone
    and the handle is stale. Null it (after a best-effort close) so any later
    ``get_runtime_db`` re-resolves cleanly — WITHOUT clearing
    ``_bundle_version_cache`` (plain Python data already captured in the
    snapshot). That keeps the post-restore request path served from the warm
    parsed caches (the ``_load_meanings`` / ``_load_culture`` lru_caches + the
    version cache) so a restored container never re-downloads the DB. Distinct
    from :func:`reset_runtime_db_cache`, which also drops the version cache and
    would force a re-query → re-download on the first post-restore request."""
    global _conn
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except sqlite3.Error:
                # WARNING (not DEBUG) so a close failing on the SnapStart restore
                # path is visible in default-INFO Lambda logs. (reset_runtime_db_cache
                # / get_runtime_db log the equivalent at exception/ERROR level; this
                # restore-path close is unexpected-but-recoverable, so WARNING.)
                _logger.warning(
                    "ignoring close error on connection reset (snapstart restore)",
                    exc_info=True,
                )
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
    """Fetch the L4 DB from S3 with etag-named cache short-circuit.

    Returns the path to the cached file on success, or None on any
    failure (caller falls back to the bundled seed).

    Flow:
      1. GET current.json — learn the versioned-key + expected etag.
      2. If the etag-named file ``wyrd-runtime-<etag>.db`` already
         exists in /tmp, skip the download (the filename IS the
         cache-hit check now that the etag's encoded in it).
      3. Otherwise, download the versioned key to a unique mkstemp
         .tmp file, then atomic-rename into the etag-named path.

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
        # Explicit type check — current.json with numeric / null values
        # in 'key' or 'etag' would otherwise raise TypeError later in
        # download_file or the cached-path computation, outside this
        # except block.
        if not isinstance(version_key, str) or not isinstance(expected_etag, str):
            raise TypeError(
                f"current.json values must be strings; got key={version_key!r} "
                f"etag={expected_etag!r}"
            )
        # Path-traversal guard: expected_etag is interpolated into the
        # /tmp cache filename. An adversarial S3 (operator with write
        # access to the bucket) could supply etag='../../etc/passwd'
        # and trick us into writing outside /tmp. Defense in depth even
        # though the bucket is terraform-controlled.
        if (
            "/" in expected_etag
            or "\\" in expected_etag
            or ".." in expected_etag
            or "\x00" in expected_etag
        ):
            raise ValueError(
                f"current.json etag {expected_etag!r} contains path-traversal "
                "characters; refusing to use as filename component"
            )
    except (ClientError, BotoCoreError, KeyError, ValueError, TypeError, OSError):
        # OSError covers stream-read failures that can surface from
        # the StreamingBody.read inside json.load — a half-read
        # response from S3 shouldn't crash the cold start.
        _logger.exception(
            "failed to fetch s3://%s/%s — falling back to bundled seed",
            bucket,
            _CURRENT_POINTER_KEY,
        )
        return None

    # Cache path includes the etag so concurrent workers reading old
    # versions don't see their immutable=1 file overwritten by a
    # parallel pointer-flip.
    cached_path = Path(_RUNTIME_DB_TMP_DIR) / _RUNTIME_DB_FILENAME_TEMPLATE.format(
        etag=expected_etag
    )
    if cached_path.is_file():
        _logger.info(
            "runtime DB cache hit (%s); skipping download",
            cached_path,
        )
        _download_matcher_sidecar(s3, bucket, version_key, cached_path)
        return cached_path

    # Download to a unique .tmp file then atomic-rename into place so
    # a partial write never produces a half-downloaded file at the
    # canonical name. mkstemp gives us a name guaranteed unique across
    # threads + processes (better than os.getpid(), which would still
    # collide if two threads of one process raced); Path.replace
    # atomic-renames on the same filesystem. Whoever loses the
    # replace race ends up with byte-equal output since both
    # downloaded the same etag's content.
    fd, tmp_download_str = tempfile.mkstemp(
        dir=cached_path.parent, prefix=cached_path.name + ".", suffix=".tmp"
    )
    os.close(fd)
    tmp_download = Path(tmp_download_str)
    try:
        s3.download_file(bucket, version_key, str(tmp_download))
        tmp_download.replace(cached_path)
    except (ClientError, BotoCoreError, OSError):
        _logger.exception(
            "failed to download s3://%s/%s — falling back to bundled seed",
            bucket,
            version_key,
        )
        # Clean up the unique .tmp so it doesn't leak under repeated
        # failures (each retry creates a new mkstemp file).
        tmp_download.unlink(missing_ok=True)
        return None

    _logger.info(
        "runtime DB downloaded from s3://%s/%s to %s",
        bucket,
        version_key,
        cached_path,
    )
    _download_matcher_sidecar(s3, bucket, version_key, cached_path)
    return cached_path


def _download_matcher_sidecar(s3, bucket: str, version_key: str, cached_db_path: Path) -> None:
    """wyrd-04oi: best-effort fetch of the decomposition matcher pickle that
    sits beside the DB in S3 (``<version_key>.matcher.pkl``) into the local
    sidecar (``<cached_db_path>.matcher.pkl``).

    Silent on absence/failure — a DB published before this feature (or a
    transient S3 error) simply has no pickle, and the cold-load rebuilds the
    trie. No-op when the local sidecar already exists (warm /tmp reuse)."""
    from botocore.exceptions import BotoCoreError, ClientError

    from .matcher_cache import matcher_sidecar_path

    local = matcher_sidecar_path(cached_db_path)
    if local.is_file():
        return
    key = f"{version_key}.matcher.pkl"
    fd, tmp_str = tempfile.mkstemp(dir=str(local.parent), prefix=local.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_str)
    try:
        s3.download_file(bucket, key, str(tmp))
        tmp.replace(local)
        _logger.info("matcher pickle downloaded from s3://%s/%s to %s", bucket, key, local)
    except (ClientError, BotoCoreError, OSError):
        # Absent (DB predates the sidecar) or transient — fine; cold-load
        # rebuilds. INFO not exception: a missing pickle is an expected,
        # non-error state.
        _logger.info("no matcher pickle at s3://%s/%s; cold-load will rebuild", bucket, key)
        tmp.unlink(missing_ok=True)


# wyrd-5kw0: the seed materialization is held open for the process. On a normal
# (unzipped) install ``as_file`` hands back the real on-disk path and the
# context cleanup is a no-op; under zip-loader packaging (Lambda Layers as zip,
# zipapp) it EXTRACTS the binary to a temp file that must outlive the cached
# read-only connection (the seed is read for the container's lifetime). The
# ExitStack keeps that temp alive until the process exits, the cache is reset,
# or the seed is re-materialized (which rotates the stack — see
# _bundled_seed_path — so a vanished temp can't accumulate replacements).
_seed_exit_stack: contextlib.ExitStack | None = None
_seed_path: Path | None = None


def _bundled_seed_path() -> Path:
    """Return the on-disk path of the committed seed-runtime.db.

    Falls back to the bundled artifact. Raises FileNotFoundError when the seed
    is missing — the runtime can't do anything without a DB to read.

    wyrd-5kw0: routes through ``importlib.resources.as_file`` so it works under
    zip-loader packaging (Lambda Layers as zip, future zipapp) — not just normal
    filesystem installs — materializing the binary on disk before it's opened.
    Memoized: materialized once and reused (re-entering ``as_file`` per cold
    resolve would re-extract under zip-loader).
    """
    global _seed_exit_stack, _seed_path
    if _seed_path is not None and _seed_path.is_file():
        return _seed_path
    seed = resources.files("wyrd.generators.kenning.data").joinpath("seed-runtime.db")
    # Re-materializing (memo missed — first call, OR a prior temp vanished, e.g.
    # a SnapStart restore wiped /tmp while these globals survived the snapshot).
    # ROTATE the ExitStack rather than appending: pushing a fresh as_file
    # context onto a stale stack would pile up extracted temps across restore
    # cycles. Closing the old stack also drops any now-dead temp. Safe because
    # the memo only misses once the prior temp is gone (an open connection keeps
    # _seed_path.is_file() true → early return above), and the SnapStart /
    # cache-reset paths close _conn before we get here.
    if _seed_exit_stack is not None:
        _seed_exit_stack.close()
    _seed_exit_stack = contextlib.ExitStack()
    path = _seed_exit_stack.enter_context(resources.as_file(seed))
    if not path.is_file():
        raise FileNotFoundError(
            f"no runtime DB available: seed at {path} is missing AND no "
            f"{ENV_LOCAL_PATH} / {ENV_BUCKET} env var is set"
        )
    _seed_path = path
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
