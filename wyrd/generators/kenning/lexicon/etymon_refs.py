"""Stable natural-key refs for etymon observations in L2 canonicalization assertions.

wyrd-c6wu / D50.1: a canonicalization assertion's reference to an etymon
observation is its **natural key** ``"<language>:<canonical_form>"`` — NEVER the
``etymon.id`` autoincrement row-id. Row-ids are reassigned on every
``rebuild-from-jsonl`` (etymon has no game_id), so an id-keyed assertion would
orphan/mis-resolve after any corpus change. ``(canonical_form, language)`` is a
UNIQUE-enforced natural key on ``etymon`` (one row per pair), so the ref resolves
back to exactly one observation. This mirrors the ``_collapses.jsonl`` /
``_reflexes.jsonl`` convention (``ref: "old-english:beacnes"``).

Emitters convert ``id -> ref`` at WRITE time (so the committed JSONL carries the
stable key) via :func:`etymon_refs_for`; the L2->L3 projection converts
``ref -> id`` at READ time via :func:`resolve_etymon_ref`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

_SEP = ":"


def etymon_ref(language: str | None, canonical_form: str) -> str:
    """The stable ``"<language>:<canonical_form>"`` ref for an etymon observation.

    ``etymon.language`` is ``TEXT NOT NULL`` (and the write paths canonicalize it
    to a non-empty value), so the stored language is never NULL; ``language or ''``
    only defends a ``None`` passed by a caller, coalescing it to ``""`` to match
    the resolver. ``canonical_form`` may itself contain ``:`` — only the FIRST
    separator splits language from form, so the form is preserved.
    """
    return f"{language or ''}{_SEP}{canonical_form}"


def etymon_refs_for(conn: sqlite3.Connection, ids: Iterable[int]) -> dict[int, str]:
    """Map ``{etymon_id: "language:canonical_form"}`` for exactly the given ids.

    Only the ids the caller emits about are looked up (not the whole 2.4M-row
    table). A missing id is omitted from the result — the caller treats that as
    an unresolved emit (it should never happen for a freshly-mined id, but is
    surfaced rather than crashing on a bad str()).
    """
    want = {int(i) for i in ids}
    if not want:
        return {}
    out: dict[int, str] = {}
    # Chunk the IN-list so a large emit set stays under SQLite's variable cap.
    ids_list = list(want)
    for start in range(0, len(ids_list), 900):
        chunk = ids_list[start : start + 900]
        placeholders = ",".join("?" * len(chunk))
        for row in conn.execute(
            f"SELECT id, language, canonical_form FROM etymon WHERE id IN ({placeholders})",  # noqa: S608 — placeholders only
            chunk,
        ):
            out[row["id"]] = etymon_ref(row["language"], row["canonical_form"])
    return out


def resolve_etymon_ref(conn: sqlite3.Connection, ref: str) -> int | None:
    """Resolve ``"<language>:<canonical_form>"`` -> ``etymon.id`` (None if unknown).

    The inverse of :func:`etymon_ref`: splits on the FIRST ``:`` and matches the
    UNIQUE ``(canonical_form, language)`` key. Used at the L2->L3 projection
    boundary in place of the old ``int(ref)``.
    """
    if _SEP not in ref:
        return None
    language, canonical_form = ref.split(_SEP, 1)
    row = conn.execute(
        "SELECT id FROM etymon WHERE language = ? AND canonical_form = ?",
        (language, canonical_form),
    ).fetchone()
    return row["id"] if row else None
