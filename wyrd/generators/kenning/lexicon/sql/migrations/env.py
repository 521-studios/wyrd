"""Alembic environment for the lexicon DB.

Builds a SQLite engine from the ``sqlalchemy.url`` set by
``lexicon.sql._config.alembic_config`` and runs the layered migrations
against it. Offline mode (``alembic upgrade --sql``) is also supported
so operators can inspect the DDL before applying.

``target_metadata`` is the SA Core MetaData from ``lexicon.sql.tables``
so ``alembic revision --autogenerate`` can diff a live DB against the
canonical schema.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from wyrd.generators.kenning.lexicon.sql.tables import metadata

config = context.config
target_metadata = metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a DB."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the configured DB and apply migrations.

    Uses ``engine.begin()`` (not ``engine.connect()``) so SA 2.0's
    "commit-as-you-go" mode auto-commits when the block exits.
    Without this, alembic's ``begin_transaction`` rolls back the
    ``alembic_version`` stamp at function return — DDL persists
    (SQLite auto-commits CREATE TABLE) but the version table stays
    empty, so a subsequent ``upgrade`` re-runs every migration and
    fails on the duplicate CREATE.
    """
    url = config.get_main_option("sqlalchemy.url")
    engine = create_engine(url, future=True)
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        # Match LexiconDB's per-connection pragmas — init_schema
        # enables WAL on the file before alembic runs, so
        # synchronous=NORMAL is the recommended pairing (crash
        # safety via WAL replay without per-transaction fsync).
        # Without this every fresh-DB test pays per-transaction
        # fsync during alembic's migration writes.
        connection.exec_driver_sql("PRAGMA synchronous = NORMAL")
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
