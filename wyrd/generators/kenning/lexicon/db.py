"""Lexicon DB connection wrapper.

``LexiconDB`` is the thin sqlite3 facade every authoring-side caller
opens to read or write the lexicon. It is intentionally light:

* No ORM, no query DSL. Methods are small SQL one-liners (or short
  multi-statement UPSERTs) so reviewers can see exactly what is being
  written without having to map a model class to a table.
* Inline SQL for now. The forward direction (per wyrd-67fv design) is
  to pull the larger query strings out into the ``lexicon.sql``
  subpackage and reference them by name, so reviewers can grep one
  place for "every CREATE / UPDATE that touches etymon_citation". The
  current class is the seed surface that future query extractions will
  reference back to.

``_apply_persistent_pragmas`` lives here because journal_mode=WAL is a
file-persistent pragma that gets set once on the writable connection
``init_schema`` opens before alembic runs, AND every subsequent open
of ``LexiconDB`` inherits the mode automatically. Keeping the pragma
helper next to the class that consumes it makes the WAL/sync coupling
visible.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def _apply_persistent_pragmas(conn: sqlite3.Connection) -> None:
    """Set journal_mode=WAL on a writable connection. journal_mode is
    file-persistent (stored in the SQLite header), so this only needs
    to run once at init_schema (fresh DB) or migrate_schema (legacy DB)
    — every subsequent open inherits the mode automatically.

    WAL gives concurrent readers + one writer without blocking —
    critical for the multi-session workflow where one Claude is
    mining (writer) while another queries corpus state (reader).

    synchronous=NORMAL is per-connection (not file-persistent), so it
    lives in LexiconDB.__init__ and runs on every open.
    """
    conn.execute("PRAGMA journal_mode = WAL")


class LexiconDB:
    """Thin convenience wrapper over sqlite3 for lexicon authoring tasks."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        # synchronous is per-connection (NOT file-persistent like
        # journal_mode), so it has to be set on every open. NORMAL is
        # the recommended pairing under WAL — preserves crash safety
        # via the WAL replay path without the per-transaction fsync of
        # synchronous=FULL. Harmless on a read-only connection (the
        # setting just declares how synchronous writes WOULD be).
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.row_factory = sqlite3.Row

    def __enter__(self) -> LexiconDB:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

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
            """
            INSERT INTO source (id, author, title, year, region, language_focus, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                author = excluded.author,
                title = excluded.title,
                year = excluded.year,
                region = excluded.region,
                language_focus = excluded.language_focus,
                notes = excluded.notes
            """,
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
        # ON CONFLICT lets us do this in a single statement: insert if new,
        # otherwise fill in any missing fields without overwriting existing
        # non-null values. RETURNING id avoids a follow-up SELECT.
        #
        # The wyrd-ha9q pronunciation / script kwargs follow the same
        # COALESCE-on-conflict pattern as the original modifier_type /
        # position_pref / notes triple. EXCEPTION: pronunciation_ipa and
        # pronunciation_dialect are conceptually paired (the dialect tag
        # describes a specific IPA), so they update atomically: if the
        # existing row has IPA already, we keep BOTH the existing IPA and
        # dialect (even if the existing dialect is NULL); if the existing
        # row has no IPA, we take both fields from the incoming row. The
        # CASE on dialect prevents a dialect-only-non-NULL re-ingest from
        # leaving the dialect tag describing an IPA we never stored.
        cur = self.conn.execute(
            """
            INSERT INTO etymon (
                canonical_form, language, modifier_type, position_pref, notes,
                pronunciation_ipa, pronunciation_dialect,
                original_script, transliteration
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_form, language) DO UPDATE SET
                modifier_type         = COALESCE(etymon.modifier_type, excluded.modifier_type),
                position_pref         = COALESCE(etymon.position_pref, excluded.position_pref),
                notes                 = COALESCE(etymon.notes, excluded.notes),
                pronunciation_ipa     = COALESCE(etymon.pronunciation_ipa, excluded.pronunciation_ipa),
                pronunciation_dialect = CASE
                                            WHEN etymon.pronunciation_ipa IS NULL
                                            THEN excluded.pronunciation_dialect
                                            ELSE etymon.pronunciation_dialect
                                        END,
                original_script       = COALESCE(etymon.original_script, excluded.original_script),
                transliteration       = COALESCE(etymon.transliteration, excluded.transliteration)
            RETURNING id
            """,
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
        self.conn.execute(
            "INSERT OR IGNORE INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
            (etymon_id, gloss),
        )

    def add_tag(self, etymon_id: int, tag: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO etymon_tag (etymon_id, tag) VALUES (?, ?)",
            (etymon_id, tag),
        )

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

        ``context_snippet`` (wyrd-9kh.3) is the surrounding scholarly-prose
        context window for in-app citation display. Populated by siblings
        (.4 reverse-search snippet capture, .5 page-number parser); leave
        None for callers that haven't been updated to capture it yet.

        Dedupe (wyrd-2pd): the unique index treats (etymon, source, NULL)
        and (etymon, source, '15') as distinct because COALESCE(page, '')
        differs. So after backfill_citation_pages writes page='15' on an
        existing row, a re-mine that calls add_citation(page=None) would
        slip past INSERT OR IGNORE and split the same evidence into two
        rows. The INSERT ... WHERE NOT EXISTS form below collapses the
        check and write into a single atomic statement so two parallel
        writers can't both pass a pre-check and both insert. Matches the
        intended one-row-per-evidence-pair semantic the unique index
        meant to enforce.
        """
        self.conn.execute(
            """
            INSERT INTO etymon_citation
                (etymon_id, source_id, page, short_quote, context_snippet)
            SELECT ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM etymon_citation
                WHERE etymon_id = ? AND source_id = ?
            )
            """,
            (etymon_id, source_id, page, short_quote, context_snippet, etymon_id, source_id),
        )

    def upsert_reflex(self, surface_form: str, position: str) -> int:
        cur = self.conn.execute(
            "SELECT id FROM reflex WHERE surface_form = ? AND position = ?",
            (surface_form, position),
        )
        row = cur.fetchone()
        if row is not None:
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO reflex (surface_form, position) VALUES (?, ?)",
            (surface_form, position),
        )
        return cur.lastrowid

    def link_reflex_etymon(self, reflex_id: int, etymon_id: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)",
            (reflex_id, etymon_id),
        )

    # --- readers / stats -------------------------------------------------

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for table in (
            "source",
            "etymon",
            "etymon_gloss",
            "etymon_tag",
            "etymon_citation",
            "reflex",
            "reflex_etymon",
            "toponym",
            "toponym_attestation",
            "toponym_etymology",
            "toponym_etymology_element",
        ):
            cur = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}")  # noqa: S608
            out[table] = cur.fetchone()["n"]
        return out
