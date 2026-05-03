"""Resolve user-data file locations for the kenning authoring layer.

The lexicon DB is a build artifact tied to a single user, not a specific
repo checkout. Storing it under the user's home keeps it cwd-/repo-/
worktree-independent — one DB per user, survives ``rm -rf`` of any
checkout, fresh clones inherit the existing corpus.

Resolution order for the lexicon DB path:

1. ``WYRD_LEXICON_DB`` env var (explicit override; useful for tests
   and operational ad-hoc runs against an alternate corpus).
2. ``~/.wyrd/lexicon.db`` (default user-data location).

On the first call where (2) doesn't exist but a legacy in-repo DB does
exist at ``<repo-root>/wyrd/generators/kenning/data/lexicon.db``, the
file is moved to (2) and a stderr notice is emitted. This carries
long-time users' mining work across the wyrd-366 upgrade without
requiring a manual ``mv``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_WYRD_DATA_DIR = Path.home() / ".wyrd"
_LEXICON_DB_NAME = "lexicon.db"
_ENV_VAR = "WYRD_LEXICON_DB"

# Sub-path inside a git repo where the legacy DB used to live
# (pre-wyrd-366). Asked of git for the main worktree's root.
_LEGACY_RELATIVE_PATH = Path("wyrd/generators/kenning/data/lexicon.db")


def default_lexicon_path() -> Path:
    """Return the user-data lexicon DB path.

    Used as Click's ``default=`` callable on every ``--db`` option, so
    each CLI invocation re-resolves the path (and triggers migration
    once on first post-upgrade call).
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser()

    new_path = _WYRD_DATA_DIR / _LEXICON_DB_NAME
    if new_path.exists():
        return new_path

    legacy = _find_legacy_lexicon_db()
    if legacy is not None:
        _migrate_legacy_db(legacy, new_path)
    return new_path


def _find_legacy_lexicon_db() -> Path | None:
    """Use git to locate the legacy DB. Checks the current worktree
    first, then the main worktree (since linked worktrees don't carry
    a copy of gitignored build artifacts). Returns None if we're not
    in a git repo or the legacy DB isn't present in either location.
    """
    current_root = _git_path("--show-toplevel")
    if current_root is not None:
        candidate = current_root / _LEGACY_RELATIVE_PATH
        if candidate.is_file():
            return candidate

    # Fall back to the MAIN worktree. ``--git-common-dir`` returns the
    # shared .git/ for the repo (same across all linked worktrees);
    # its parent is the main worktree's root.
    common_git_dir = _git_path("--path-format=absolute", "--git-common-dir")
    if common_git_dir is not None:
        main_worktree = common_git_dir.parent
        candidate = main_worktree / _LEGACY_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    return None


def _git_path(*args: str) -> Path | None:
    """Run ``git rev-parse <args>`` and return the result as a Path,
    or None on any failure (not in a git repo, git not installed,
    timeout). Bounded to 2 seconds so a hung git can't stall the CLI.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    line = result.stdout.strip()
    return Path(line) if line else None


def _migrate_legacy_db(legacy: Path, new_path: Path) -> None:
    """Move ``legacy`` to ``new_path`` with a stderr notice. Creates
    the parent dir if needed. Idempotent in practice: if new_path
    already exists, ``default_lexicon_path`` skips this branch."""
    _WYRD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy), str(new_path))
    print(
        f"[wyrd] migrated lexicon DB: {legacy} → {new_path} (wyrd-366)",
        file=sys.stderr,
    )
