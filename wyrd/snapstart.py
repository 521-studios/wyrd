"""Lambda SnapStart support (wyrd-g1wp).

SnapStart snapshots the initialized container at deploy and restores it on each
cold start, so heavy INIT work (downloading + parsing the runtime DB, building
every culture's NameGenerator) is paid ONCE at snapshot time and replayed in
~sub-second on every cold restore — instead of re-run per cold container on the
first request (the ~2-4s lazy load that, pre-trim, was ~10s and timed out).

Two halves, both invoked from ``create_app`` at init:
  * :func:`preload_runtime` — warm the kenning runtime caches so the parsed
    Python objects (meaning_db, per-culture generators, cohesion table, version
    stamp) live in the snapshot.
  * :func:`register_snapstart_hooks` — register an after-restore hook that drops
    the stale sqlite connection captured in the snapshot (its ``/tmp`` file is
    gone on restore) WITHOUT touching the parsed caches, so a restored container
    serves from memory and never re-downloads the DB.

Both are no-ops off SnapStart: preload only runs when
``AWS_LAMBDA_INITIALIZATION_TYPE == "snap-start"`` (Lambda sets this during
snapshotting), and the after-restore hook is registered through the
``snapshot_restore_py`` runtime library and simply never fires off-SnapStart.
"""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)


def preload_runtime() -> None:
    """Warm the kenning runtime caches at init so SnapStart captures the parsed
    bundle + per-culture generators in the snapshot.

    Only runs when SnapStart is actually active for this init — Lambda sets
    ``AWS_LAMBDA_INITIALIZATION_TYPE=snap-start`` during snapshotting. On a
    normal (on-demand) init, and locally / in tests, this is a no-op so the
    lazy per-request load is unchanged: preloading without SnapStart would just
    move the cost from first-request to init, not remove it. Best-effort: a
    preload failure logs and falls back to lazy loading rather than breaking
    startup."""
    if os.environ.get("AWS_LAMBDA_INITIALIZATION_TYPE") != "snap-start":
        return
    try:
        from wyrd.generators.kenning import (
            CULTURES,
            _load_canonical_decompositions,
            _load_culture,
            _load_empirical_priors,
            _load_fantasy_morphemes,
            _load_joiners,
            _tags_options_by_culture,
            available_tags,
        )
        from wyrd.generators.kenning.runtime import runtime_db

        runtime_db.get_runtime_db()  # resolve + download + open the connection
        available_tags()  # _load_meanings → _runtime_db_bundle_dict: parse the bundle
        # Warm EVERY lru-cache the default generate() path touches so the snapshot
        # is fully warm and the first post-restore request needs no DB read. These
        # read the now-warm bundle dict; _load_empirical_priors is the critical one
        # — it calls get_runtime_db() DIRECTLY, so without warming it the first
        # restored request would re-resolve (re-download the DB), defeating SnapStart.
        _load_joiners()
        _load_canonical_decompositions()
        _load_fantasy_morphemes()
        _load_empirical_priors()
        for culture in CULTURES:
            _load_culture(culture)  # proportions + NameGenerator per culture
        # wyrd-ah53: the manifest's per-culture tag options iterate every culture's
        # eligible pool — warm it so the first post-restore /api/manifest is cheap.
        _tags_options_by_culture()
        runtime_db.bundle_version()  # cache the per-request version stamp
        _logger.info("snapstart preload complete: bundle + %d cultures warm", len(CULTURES))
    except Exception:
        _logger.warning("snapstart preload failed; falling back to lazy load", exc_info=True)


_hooks_registered = False


def register_snapstart_hooks() -> None:
    """Register the after-restore hook that invalidates the snapshot's stale
    sqlite connection (wyrd-g1wp). No-op when ``snapshot_restore_py`` isn't
    importable (off-SnapStart) and idempotent — ``create_app`` may be called more
    than once (tests), but the hook registry de-dups via the module flag so the
    callback isn't appended repeatedly."""
    global _hooks_registered
    if _hooks_registered:
        return
    try:
        from snapshot_restore_py import register_after_restore
    except ImportError:
        return
    _hooks_registered = True

    @register_after_restore
    def _drop_stale_runtime_db_connection() -> None:
        # The snapshot captured an open connection to
        # /tmp/wyrd-runtime-<etag>.db; /tmp is wiped on restore so the handle is
        # dead. Drop the connection ONLY (keep the parsed lru-caches + version
        # cache) so generation serves from memory and a restored container never
        # re-downloads the DB.
        from wyrd.generators.kenning.runtime import runtime_db

        runtime_db.reset_runtime_db_connection()
        _logger.info("snapstart restore: dropped stale runtime DB connection")
