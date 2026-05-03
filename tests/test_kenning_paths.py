"""Tests for wyrd.generators.kenning.paths — default DB resolution +
one-time legacy migration (wyrd-366)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from wyrd.generators.kenning import paths


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the resolver's "~/.wyrd" target into a tmpdir so tests
    can't touch the real user-home DB. Returns the tmp data dir."""
    fake_home = tmp_path / "home"
    fake_data_dir = fake_home / ".wyrd"
    monkeypatch.setattr(paths, "_WYRD_DATA_DIR", fake_data_dir)
    # Clear any env override leaking from the host shell.
    monkeypatch.delenv(paths._ENV_VAR, raising=False)
    return fake_data_dir


def test_default_lexicon_path_returns_home_default(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No env override, no legacy DB → returns the configured home
    location verbatim. The path needn't exist; existence is the
    caller's concern."""
    # Make _find_legacy_lexicon_db return None (no repo, or no DB in repo)
    monkeypatch.setattr(paths, "_find_legacy_lexicon_db", lambda: None)
    result = paths.default_lexicon_path()
    assert result == isolated_home / "lexicon.db"


def test_env_override_wins_over_default(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``WYRD_LEXICON_DB`` short-circuits everything else — no legacy
    detection, no migration, no home default. Useful for ops + tests."""
    override = tmp_path / "custom" / "lex.db"
    monkeypatch.setenv(paths._ENV_VAR, str(override))
    # If migration accidentally fires we'd see a different result.
    monkeypatch.setattr(paths, "_find_legacy_lexicon_db", lambda: None)
    result = paths.default_lexicon_path()
    assert result == override


def test_env_override_expands_user_tilde(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env override is meant for humans — accept ``~`` so users
    can set ``WYRD_LEXICON_DB=~/scratch/lex.db`` without resolving by
    hand."""
    monkeypatch.setenv(paths._ENV_VAR, "~/scratch/lex.db")
    monkeypatch.setattr(paths, "_find_legacy_lexicon_db", lambda: None)
    result = paths.default_lexicon_path()
    # Path.home() differs across CI hosts; confirm via expanduser equiv.
    assert result == Path("~/scratch/lex.db").expanduser()


def test_legacy_db_is_migrated_to_home_on_first_call(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When ~/.wyrd/lexicon.db doesn't exist yet but a legacy in-repo
    DB does, the file is moved (mv, not copy) and the new path is
    returned. Pin this behavior so a future regression to copy + leave
    or to wrong-target-path surfaces immediately."""
    legacy = tmp_path / "repo" / "wyrd" / "generators" / "kenning" / "data" / "lexicon.db"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"sentinel-lexicon-data")

    monkeypatch.setattr(paths, "_find_legacy_lexicon_db", lambda: legacy)

    result = paths.default_lexicon_path()
    new_path = isolated_home / "lexicon.db"

    assert result == new_path
    assert new_path.is_file(), "new home path should exist after migration"
    assert new_path.read_bytes() == b"sentinel-lexicon-data"
    assert not legacy.exists(), "legacy DB should be moved (not copied)"


def test_no_migration_when_home_db_already_exists(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Once ~/.wyrd/lexicon.db exists, default_lexicon_path is a pure
    lookup — it never re-migrates. Otherwise a stale legacy DB sitting
    in some other clone would clobber the canonical user-home DB."""
    isolated_home.mkdir(parents=True)
    home_db = isolated_home / "lexicon.db"
    home_db.write_bytes(b"current-canonical")

    legacy = tmp_path / "repo" / "wyrd" / "generators" / "kenning" / "data" / "lexicon.db"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"would-clobber")

    # Spy on the legacy-finder to confirm we don't even ask.
    finder_calls = []

    def spy() -> Path | None:
        finder_calls.append(True)
        return legacy

    monkeypatch.setattr(paths, "_find_legacy_lexicon_db", spy)

    result = paths.default_lexicon_path()

    assert result == home_db
    assert home_db.read_bytes() == b"current-canonical"
    assert legacy.read_bytes() == b"would-clobber"  # untouched
    assert finder_calls == [], "shouldn't query legacy when home DB exists"


def test_no_migration_when_no_legacy_db_present(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh user, no legacy DB anywhere → default_lexicon_path returns
    the home path without creating any files. The dir creation happens
    later, on first DB write."""
    monkeypatch.setattr(paths, "_find_legacy_lexicon_db", lambda: None)
    result = paths.default_lexicon_path()
    assert result == isolated_home / "lexicon.db"
    assert not isolated_home.exists(), "shouldn't pre-create the dir"


def test_git_path_returns_none_outside_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Defensive: `_git_path` swallows non-zero returns (cwd not in a
    git repo) and returns None instead of raising. Otherwise a user
    running `wyrd kenning lexicon path` from /tmp would crash."""

    # Patch subprocess.run to simulate "fatal: not a git repository"
    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(paths.subprocess, "run", fake_run)
    assert paths._git_path("--show-toplevel") is None


def test_git_path_returns_none_when_git_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If git isn't on PATH at all (FileNotFoundError), don't crash."""

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("git")

    monkeypatch.setattr(paths.subprocess, "run", fake_run)
    assert paths._git_path("--show-toplevel") is None


def test_find_legacy_lexicon_db_checks_main_worktree_when_current_lacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A linked git worktree's filesystem doesn't carry build artifacts
    like `data/lexicon.db` (the main worktree does). The finder must
    fall back to ``--git-common-dir`` to locate the main worktree's
    legacy DB. Pin this so a regression that only checks the current
    worktree silently misses the migration on first-call from a
    feature-branch worktree."""
    main_worktree = tmp_path / "main"
    legacy = main_worktree / "wyrd" / "generators" / "kenning" / "data" / "lexicon.db"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"main-worktree-db")

    # Simulate: current worktree (linked) reports its own root with NO legacy
    # DB; --git-common-dir points back at the main worktree's .git/.
    def fake_git_path(*args: str) -> Path | None:
        if "--show-toplevel" in args:
            return tmp_path / "linked"  # different from main_worktree
        if "--git-common-dir" in args:
            return main_worktree / ".git"
        return None

    monkeypatch.setattr(paths, "_git_path", fake_git_path)
    result = paths._find_legacy_lexicon_db()
    assert result == legacy
