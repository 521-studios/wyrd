"""Lexicon SQL layer: table metadata, queries, and Alembic migrations.

Layout:
  tables.py        — SQLAlchemy Core MetaData mirroring the lexicon schema.
                     Source of truth for tooling that needs introspection
                     (Alembic autogenerate, query helpers, future ORM).
  migrations/      — Alembic environment + layered versions/ scripts.
                     ``versions/0001_sources.py`` is the bibliographic
                     baseline; subsequent migrations add the etymon /
                     citation / toponym / view layers.
  _config.py       — programmatic Alembic Config so callers can run
                     ``upgrade_head(path)`` without an alembic.ini file
                     on disk.

The Lambda runtime never imports this package — it reads the bundled
``meanings.json`` and never touches the lexicon DB. Both SQLAlchemy and
Alembic are authoring-side dependencies.
"""

from wyrd.generators.kenning.lexicon.sql._config import alembic_config, upgrade_head
from wyrd.generators.kenning.lexicon.sql.tables import metadata

__all__ = ["alembic_config", "metadata", "upgrade_head"]
