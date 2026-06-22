"""Lexicon DB connection wrapper.

``LexiconDB`` is the authoring-side entry point every lexicon caller
opens. After wyrd-67fv the class is hybrid (per operator decision):

* **SQLAlchemy Engine as the primary handle.** ``self.engine`` is the
  surface new code uses for query-builder calls (SA Core expressions
  against ``lexicon.sql.tables.metadata``) and for any future
  ``alembic revision --autogenerate`` consumers.
* **sqlite3.Connection shim for back-compat.** ``self.conn`` is the
  long-lived sqlite3 connection underneath the engine's pool, exposed
  so the ~1,000 existing call sites that do ``db.conn.execute(...)``
  across the kenning package keep working unchanged. The two views
  share one underlying DBAPI connection (via ``StaticPool``), so
  writes through either are immediately visible to the other.

The wyrd-67fv split moved every SQL string out of this file into
``lexicon.sql.queries.*`` (one file per table). The methods on the
class are now thin parameter-marshallers around named query
constants — open the relevant queries module to see what each method
actually writes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL, Engine
from sqlalchemy.pool import StaticPool

from wyrd.generators.kenning.lexicon.morpheme_surface import normalize_morpheme_surface
from wyrd.generators.kenning.lexicon.sql.queries import (
    INSERT_CITATION_IF_ABSENT,
    INSERT_GLOSS_OR_IGNORE,
    INSERT_TAG_OR_IGNORE,
    LINK_REFLEX_ETYMON_OR_IGNORE,
    STATS_COUNT_TEMPLATE,
    STATS_TABLES,
    UPSERT_ETYMON,
    UPSERT_REFLEX,
    UPSERT_SOURCE,
    citation_params,
)


def _apply_persistent_pragmas(conn: sqlite3.Connection) -> None:
    """Set journal_mode=WAL on a writable connection. journal_mode is
    file-persistent (stored in the SQLite header), so this only needs
    to run once at init_schema (fresh DB) or migrate_schema (legacy DB)
    — every subsequent open inherits the mode automatically.

    WAL gives concurrent readers + one writer without blocking —
    critical for the multi-session workflow where one Claude is
    mining (writer) while another queries corpus state (reader).

    synchronous=NORMAL is per-connection (not file-persistent), so it
    rides on the SA engine's connect-event listener (see
    ``_register_per_connection_pragmas``) and runs on every checkout.
    """
    conn.execute("PRAGMA journal_mode = WAL")


def _register_per_connection_pragmas(engine: Engine) -> None:
    """Apply per-connection pragmas to every DBAPI connection the SA
    engine's pool hands out.

    ``foreign_keys`` and ``synchronous`` are per-connection (not
    file-persistent like ``journal_mode``), so they have to be set on
    every checkout from the pool. The ``connect`` event fires once
    per new DBAPI connection — with ``StaticPool`` that's exactly
    once per LexiconDB instance, but the event-listener shape keeps
    the contract correct if a future caller swaps in a different
    pool class.

    Also sets ``row_factory = sqlite3.Row`` so existing
    ``db.conn.execute(...).fetchone()['column']`` patterns keep
    working under the shim.
    """

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn: sqlite3.Connection, _record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        # synchronous=NORMAL is the recommended pairing under WAL —
        # preserves crash safety via the WAL replay path without the
        # per-transaction fsync of synchronous=FULL.
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.close()
        dbapi_conn.row_factory = sqlite3.Row


# Re-export sqlite3's base error so cli/lexicon callers can catch DB
# failures without importing sqlite3 directly (keeps them off the
# package-boundary skip-list, wyrd-mewu). LexiconError IS sqlite3.Error.
LexiconError = sqlite3.Error


class LexiconDB:
    """SQLAlchemy-backed wrapper around the lexicon SQLite DB.

    Holds two coherent views of the same file:

    * ``self.engine`` — ``sqlalchemy.Engine`` for query-builder /
      autogenerate-driven code paths.
    * ``self.conn`` — long-lived ``sqlite3.Connection`` from the
      engine's pool, exposed for back-compat with existing
      ``db.conn.execute(...)`` callers.

    Both share one underlying DBAPI connection via ``StaticPool``,
    so transaction state is consistent across the two surfaces.
    The two handles are exposed as read-only properties; reassigning
    ``db.engine`` or ``db.conn`` would break that coherence
    invariant, so the underlying fields are name-mangled and the
    setters are intentionally absent.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        # ``check_same_thread=False`` lets SA's pool hand the
        # connection across threads, matching the behaviour of the
        # historical sqlite3.connect() call this class used to do.
        # The lexicon writers are single-threaded today, but the
        # default would break tests that share a LexiconDB across
        # threading.Thread instances.
        #
        # ``URL.create`` (rather than f"sqlite:///{path}") handles
        # paths with special characters (``#``, ``?``, ``:``) without
        # silently truncating into URL fragment/query/auth fields.
        # Wyrd's lexicon-DB lives at ~/.wyrd/lexicon.db in production,
        # but tmp-path tests on CI can land under names containing
        # any of those.
        self.__engine = create_engine(
            URL.create("sqlite", database=str(self.path)),
            future=True,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        _register_per_connection_pragmas(self.__engine)
        # Long-lived raw DBAPI proxy from the engine pool. SA names
        # this object ``_ConnectionFairy``; we hold it for the
        # LexiconDB instance's lifetime so the underlying sqlite3
        # connection isn't recycled out from under existing callers.
        # ``self.conn`` (below) is the raw sqlite3 connection itself.
        self.__raw_proxy = self.__engine.raw_connection()
        # ``driver_connection`` is typed Optional in SA. With
        # ``StaticPool`` on a fresh engine the proxy always wraps a
        # live sqlite3 connection, but the explicit None guard makes
        # the contract readable and raises eagerly if a future change
        # to the pool ever puts us in a state where it could fail.
        dbapi_conn = self.__raw_proxy.driver_connection
        if dbapi_conn is None:
            raise RuntimeError(
                "engine.raw_connection().driver_connection is None — "
                "engine pool failed to hand out a DBAPI connection"
            )
        self.__conn: sqlite3.Connection = dbapi_conn
        self.__closed = False

    @property
    def engine(self) -> Engine:
        """SA Engine over the shared DBAPI connection. Read-only."""
        return self.__engine

    @property
    def conn(self) -> sqlite3.Connection:
        """Raw sqlite3 connection from the engine pool. Read-only."""
        return self.__conn

    def __enter__(self) -> LexiconDB:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release the raw proxy back to the pool, then dispose the
        engine so the underlying sqlite3 connection is closed.

        Idempotent: a second call is a no-op. Lets callers safely
        invoke ``close()`` explicitly inside a ``with`` block when
        they need to release the file lock before the block exits,
        without the ``__exit__`` double-close raising.
        """
        if self.__closed:
            return
        self.__closed = True
        self.__raw_proxy.close()
        self.__engine.dispose()

    def commit(self) -> None:
        self.conn.commit()

    # --- writers ---------------------------------------------------------

    def upsert_source(
        self,
        *,
        id: str,
        title: str,
        author: str | None = None,
        year: int | None = None,
        region: str | None = None,
        language_focus: str | None = None,
        notes: str | None = None,
    ) -> str:
        self.conn.execute(
            UPSERT_SOURCE,
            (id, author, title, year, region, language_focus, notes),
        )
        return id

    def upsert_etymon(
        self,
        canonical_form: str,
        language: str,
        *,
        modifier_type: str | None = None,
        position_pref: str | None = None,
        notes: str | None = None,
        pronunciation_ipa: str | None = None,
        pronunciation_dialect: str | None = None,
        original_script: str | None = None,
        transliteration: str | None = None,
    ) -> int:
        """Insert or update an etymon row keyed by (canonical_form, language).

        See ``lexicon.sql.queries.etymon.UPSERT_ETYMON`` for the
        CASE-on-conflict contract that keeps pronunciation_ipa +
        pronunciation_dialect updating atomically.

        ``canonical_form`` is de-dashed to its bare morpheme surface here
        (wyrd-aicu.8, D45) so the boundary affix-position dash never enters the
        store. This is the de-dash point for the UPSERT path — most etymon
        writers (seed / etymonline / ingest / modern-reflex / wiktextract) funnel
        through here; the two writers that bypass the upsert with their own raw
        ``INSERT INTO etymon`` — ``jsonl.build._insert_etymon`` and the
        ``enrichment`` split-child — de-dash at their own INSERT. Doing it on
        write (vs a one-time UPDATE) is what makes it durable: the bulk of the
        dashed etymons are re-injected by the L1 wiktextract slices on every
        rebuild, so a write-time strip re-applies where a one-shot UPDATE would
        be undone.

        The de-dash is a pure idempotent transform; the strip-to-empty *junk*
        drop is a record-level decision that belongs at the ingest boundary —
        every caller (the wiktextract bulk paths and the curated seed /
        etymonline / ingest / modern-reflex writers) pre-filters a junk form and
        drops the whole record before reaching here — so a form that normalizes
        to nothing is a contract violation and raises rather than silently
        minting an empty-keyed row.
        """
        normalized = normalize_morpheme_surface(canonical_form)
        if normalized is None:
            raise ValueError(
                "upsert_etymon requires a non-empty morpheme surface; "
                f"{canonical_form!r} is empty after de-dash (drop junk at the "
                "ingest boundary, not here)"
            )
        canonical_form = normalized
        cur = self.conn.execute(
            UPSERT_ETYMON,
            (
                canonical_form,
                language,
                modifier_type,
                position_pref,
                notes,
                pronunciation_ipa,
                pronunciation_dialect,
                original_script,
                transliteration,
            ),
        )
        return cur.fetchone()[0]

    def add_gloss(self, etymon_id: int, gloss: str) -> None:
        self.conn.execute(INSERT_GLOSS_OR_IGNORE, (etymon_id, gloss))

    def add_tag(self, etymon_id: int, tag: str) -> None:
        self.conn.execute(INSERT_TAG_OR_IGNORE, (etymon_id, tag))

    def add_citation(
        self,
        etymon_id: int,
        source_id: str,
        *,
        page: str | None = None,
        short_quote: str | None = None,
        context_snippet: str | None = None,
    ) -> None:
        """Insert an etymon_citation row.

        ``context_snippet`` (wyrd-9kh.3) is the surrounding
        scholarly-prose context window for in-app citation display.
        Populated by siblings (.4 reverse-search snippet capture, .5
        page-number parser); leave None for callers that haven't been
        updated to capture it yet.

        Dedupe semantics — including the wyrd-2pd page-NULL +
        page-non-NULL split — are documented on
        ``lexicon.sql.queries.etymon_citation.INSERT_CITATION_IF_ABSENT``.
        """
        self.conn.execute(
            INSERT_CITATION_IF_ABSENT,
            citation_params(
                etymon_id,
                source_id,
                page=page,
                short_quote=short_quote,
                context_snippet=context_snippet,
            ),
        )

    def upsert_reflex(self, surface_form: str, position: str) -> int:
        # ON CONFLICT … RETURNING in one round-trip — no TOCTOU
        # window, and no stdlib-typed-as-int-but-Optional lastrowid
        # to type-cast around. See UPSERT_REFLEX for the no-op
        # DO UPDATE pattern that lets RETURNING fire on conflict.
        cur = self.conn.execute(UPSERT_REFLEX, (surface_form, position))
        return cur.fetchone()[0]

    def link_reflex_etymon(self, reflex_id: int, etymon_id: int) -> None:
        self.conn.execute(LINK_REFLEX_ETYMON_OR_IGNORE, (reflex_id, etymon_id))

    # --- readers / stats -------------------------------------------------

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for table in STATS_TABLES:
            # Safe interpolation: ``STATS_TABLES`` is a frozen tuple
            # of hardcoded table names; no user input ever reaches
            # this string format.
            cur = self.conn.execute(STATS_COUNT_TEMPLATE.format(table=table))
            out[table] = cur.fetchone()["n"]
        return out
