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

_WYRD_DATA_DIR_NAME = ".wyrd"
_LEXICON_DB_NAME = "lexicon.db"
_ENV_VAR = "WYRD_LEXICON_DB"

# Pretty form of the default location for ``show_default=`` strings on
# Click options. The actual path is resolved at command-invocation time
# via ``default_lexicon_path()``; using a static string here keeps
# ``--help`` from triggering the callable (which would otherwise run
# the legacy-DB migration as a side effect of viewing help).
LEXICON_DB_DEFAULT_DISPLAY = "$WYRD_LEXICON_DB or ~/.wyrd/lexicon.db"

# Sub-path inside a git repo where the legacy DB used to live
# (pre-wyrd-366). Asked of git for the main worktree's root.
_LEGACY_RELATIVE_PATH = Path("wyrd/generators/kenning/data/lexicon.db")


def _wyrd_data_dir() -> Path:
    """Lazy-resolve the user-data directory at call time. Avoids
    crashing CLI import in environments where ``Path.home()`` can't
    resolve (no $HOME, missing entry in /etc/passwd, etc.) — the
    failure surfaces at the actual command invocation instead, with
    a clearer traceback."""
    return Path.home() / _WYRD_DATA_DIR_NAME


def default_lexicon_path() -> Path:
    """Return the user-data lexicon DB path.

    Used as Click's ``default=`` callable on every ``--db`` option, so
    each CLI invocation re-resolves the path (and triggers migration
    once on first post-upgrade call). Click options paired with this
    callable should pass ``show_default=LEXICON_DB_DEFAULT_DISPLAY``
    instead of ``show_default=True`` — otherwise ``--help`` ends up
    invoking the callable (which prints the migration notice as a
    side effect of viewing help).
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser()

    new_path = _wyrd_data_dir() / _LEXICON_DB_NAME
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
    if current_root is None:
        # Not in a git repo at all → no second lookup will succeed
        # either; skip the wasted subprocess spawn.
        return None
    candidate = current_root / _LEGACY_RELATIVE_PATH
    if candidate.is_file():
        return candidate

    # Fall back to the MAIN worktree. ``--git-common-dir`` returns the
    # shared .git/ for the repo (same across all linked worktrees);
    # its parent is the main worktree's root. NOTE: don't pass
    # ``--path-format=absolute`` — that flag was added in Git 2.31
    # (2021). Older Git (e.g. Ubuntu 20.04 LTS ships 2.25) errors
    # out with exit 128. We resolve the relative path manually instead.
    common_git_dir = _git_path("--git-common-dir")
    if common_git_dir is not None:
        main_worktree = common_git_dir.parent
        candidate = main_worktree / _LEGACY_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    return None


def _git_path(*args: str) -> Path | None:
    """Run ``git rev-parse <args>`` and return the result as a resolved
    absolute Path, or None on any failure (not in a git repo, git not
    installed, timeout). Bounded to 2 seconds so a hung git can't
    stall the CLI. ``.resolve()`` handles older Git versions that
    return relative paths (e.g. ``--git-common-dir`` in a main worktree
    yields ``.git`` rather than the absolute path)."""
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
    return Path(line).resolve() if line else None


def _migrate_legacy_db(legacy: Path, new_path: Path) -> None:
    """Move ``legacy`` to ``new_path`` with a stderr notice. Creates
    the parent dir if needed. Idempotent in practice: if new_path
    already exists, ``default_lexicon_path`` skips this branch."""
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy), str(new_path))
    print(
        f"[wyrd] migrated lexicon DB: {legacy} → {new_path} (wyrd-366)",
        file=sys.stderr,
    )
