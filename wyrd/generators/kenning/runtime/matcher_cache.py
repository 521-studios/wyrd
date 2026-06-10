"""wyrd-04oi: pickle cache for the decomposition ``MorphemeTrie``.

The decomposition path (kenning-explain / era-map / rewind) builds a
``MorphemeTrie`` from the full ~90k-morpheme corpus. The corpus is a build-time
artifact, so rebuilding the trie on every cold container is wasted work: build
it once at publish time, pickle it beside the runtime DB, and load the pickle
on cold start instead of rebuilding (``build_morpheme_trie``).

**Fail-safe.** :func:`load_matcher` returns ``None`` — never raises — whenever
the pickle can't be trusted: a missing/corrupt file, a Python-version mismatch
(pickle is interpreter-shape sensitive), a format-version mismatch, or a
corpus-stamp mismatch (the trie was built from a different corpus than the one
now loaded). ``None`` just means "rebuild from the meaning_db" — the prior
behavior — so the cache can only speed things up, never change a result.

**Trust.** The pickle is produced by our own publish step and fetched from the
same S3 object the runtime DB comes from (same trust boundary as the DB
itself); it is never loaded from operator-supplied input.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

from .trie_matcher import MorphemeTrie

_logger = logging.getLogger(__name__)

# Bump when the pickle envelope OR the MorphemeTrie / _TrieNode / Meaning class
# shape changes in a way that makes previously-written pickles unsafe to load.
_MATCHER_PICKLE_FORMAT = 1

# Pickle is sensitive to the interpreter minor version (bytecode/class layout),
# so the stamp pins major.minor; a pickle built on a different one is rebuilt.
_PY = f"{sys.version_info.major}.{sys.version_info.minor}"


def matcher_stamp(meaning_db: dict[str, list[Any]]) -> str:
    """A cheap, deterministic content stamp for ``meaning_db``.

    Computed identically at publish (build) time and at cold-load time, so a
    pickle built from a *different* corpus is rejected. It hashes the usage set
    + per-usage sense counts — the staleness vectors a corpus re-emit moves —
    plus the corpus size, format version, and Python version. O(N log N) over
    the keys, far cheaper than building the trie (which processes every
    Meaning char-by-char), so validating the stamp on load is still a net win.

    Granularity note: a change to a Meaning's *internal* data (same usage key,
    same sense count, different gloss/source) is not directly captured — but
    such a change comes from a corpus re-emit that moves many keys/counts, so
    the hash differs in practice; and the fail-safe is a stale gloss, never a
    crash. This matches the ticket's "validate against the bundle version"
    intent at finer granularity.
    """
    h = hashlib.sha256()
    for k in sorted(meaning_db):
        h.update(k.encode())
        h.update(f":{len(meaning_db[k])}\n".encode())
    return f"matcher-v{_MATCHER_PICKLE_FORMAT}|n={len(meaning_db)}|h={h.hexdigest()[:16]}|py{_PY}"


def matcher_sidecar_path(db_path: str | os.PathLike[str]) -> Path:
    """The matcher pickle path beside a runtime DB file: ``<db>.matcher.pkl``.

    A plain suffix-append (not ``with_suffix``) so a ``…/foo.db`` DB maps to
    ``…/foo.db.matcher.pkl`` — keeping the DB's own name intact lets the S3
    key + the /tmp etag-cache name derive trivially from the DB's."""
    return Path(f"{os.fspath(db_path)}.matcher.pkl")


def dump_matcher(trie: MorphemeTrie, path: str | os.PathLike[str], *, stamp: str) -> None:
    """Pickle ``trie`` to ``path`` (atomic temp-write + rename) inside the
    envelope ``{fmt, py, stamp, trie}`` that :func:`load_matcher` validates."""
    path = Path(path)
    payload = {"fmt": _MATCHER_PICKLE_FORMAT, "py": _PY, "stamp": stamp, "trie": trie}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".matcher.tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def load_matcher(path: str | os.PathLike[str], *, stamp: str) -> MorphemeTrie | None:
    """Load a pickled ``MorphemeTrie`` from ``path``, or ``None`` to mean
    "rebuild". Returns ``None`` (never raises) on a missing file, an unpickling
    error / corrupt file, a format-version mismatch, a Python-version mismatch,
    or a stamp mismatch (different corpus)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception:
        _logger.warning("matcher pickle %s failed to unpickle; rebuilding", path, exc_info=True)
        return None
    if not isinstance(payload, dict) or payload.get("fmt") != _MATCHER_PICKLE_FORMAT:
        return None
    if payload.get("py") != _PY:
        _logger.info(
            "matcher pickle %s built on py%s != py%s; rebuilding", path, payload.get("py"), _PY
        )
        return None
    if payload.get("stamp") != stamp:
        _logger.info("matcher pickle %s stamp mismatch (corpus changed); rebuilding", path)
        return None
    trie = payload.get("trie")
    if not isinstance(trie, MorphemeTrie):
        return None
    return trie
