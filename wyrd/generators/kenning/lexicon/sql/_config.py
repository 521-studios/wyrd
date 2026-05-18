"""Programmatic Alembic configuration for the lexicon DB.

We ship Alembic embedded in the ``wyrd.generators.kenning.lexicon.sql``
package rather than as a standalone ``alembic.ini`` + ``migrations/``
tree at repo root. Operators don't run ``alembic upgrade head`` from a
shell; ``init_schema(path)`` calls ``upgrade_head(path)`` from the
existing CLI entry points, and tests build fresh tmp-path DBs via the
same helper. Keeping the config programmatic avoids an alembic.ini at
repo root that would be the wrong scope (one config for one schema in
one subpackage).
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

from alembic.config import Config


def _migrations_dir() -> Path:
    """Return the absolute path to the ``migrations/`` directory inside
    this package. Used as Alembic's ``script_location``."""
    pkg = importlib.resources.files("wyrd.generators.kenning.lexicon.sql.migrations")
    return Path(str(pkg))


def alembic_config(db_path: Path | str) -> Config:
    """Build an in-memory Alembic ``Config`` pointed at ``db_path``.

    Sets ``script_location`` to the embedded ``migrations/`` directory
    and ``sqlalchemy.url`` to a ``sqlite:///<db_path>`` URL. No
    alembic.ini on disk is consulted — every setting is explicit here.
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(_migrations_dir()))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{Path(db_path).resolve()}")
    return cfg


def upgrade_head(db_path: Path | str) -> None:
    """Run ``alembic upgrade head`` against the SQLite DB at ``db_path``.

    Creates the schema from scratch when the DB has no
    ``alembic_version`` row, or rolls forward to the latest revision
    otherwise. Idempotent — re-running on an already-up-to-date DB
    is a no-op.
    """
    # Deferred import for ``alembic.command`` specifically — it pulls
    # in alembic.runtime.migration + util + script chains the
    # alembic.config import above does NOT load. Importing
    # ``lexicon.sql`` (which transitively loads ``alembic.config``
    # via this module) shouldn't cost the migration-runtime chain
    # too; only the init_schema / CLI paths that actually run
    # migrations need it.
    from alembic.command import upgrade

    upgrade(alembic_config(db_path), "head")
