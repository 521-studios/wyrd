"""Shared helpers for the synsets subcommands."""

from __future__ import annotations

from wyrd.generators.kenning.lexicon import LexiconDB


def _resolve_etymon_ident(db: LexiconDB, ident: str, language: str | None) -> int:
    """Resolve an etymon-identifying CLI arg to an etymon.id.

    Without ``--language``, ident must be a numeric id. With
    ``--language``, ident is treated as a canonical_form and looked up
    via (form, language). Raises ValueError with a friendly message on
    every failure mode (non-numeric without --language; missing form;
    ambiguous form). The caller surfaces the message on stderr +
    exits non-zero, matching the rest of this CLI's error pattern.
    """
    if language is None:
        try:
            return int(ident)
        except ValueError:
            raise ValueError(
                f"etymon ident {ident!r} is not numeric; pass --language to "
                f"look up by canonical_form, or use an integer etymon.id"
            ) from None
    row = db.conn.execute(
        "SELECT id FROM etymon WHERE canonical_form = ? AND language = ?",
        (ident, language),
    ).fetchone()
    if row is None:
        raise ValueError(f"no etymon with canonical_form={ident!r} language={language!r}")
    return row["id"]
