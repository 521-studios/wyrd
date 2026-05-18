"""Alembic environment for the lexicon DB.

Configured to run online migrations against a SQLite engine built from
the ``sqlalchemy.url`` set by ``lexicon.sql._config.alembic_config``.
Offline migration (``alembic upgrade --sql``) is also supported for
operators who want to inspect the generated DDL before applying it.

``target_metadata`` is the SA Core MetaData from ``lexicon.sql.tables``
so a future ``alembic revision --autogenerate`` can diff the live DB
against the canonical schema.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from wyrd.generators.kenning.lexicon.sql.tables import metadata

config = context.config
target_metadata = metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a DB."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the configured DB and apply migrations in a transaction."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        # SQLite needs foreign_keys=ON enabled per-connection.
        connection.exec_driver_sql("PRAGMA foreign_keys = ON")
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
