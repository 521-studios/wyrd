"""Adapter: L4 runtime DB → bundle-dict shape consumed by ``load_meanings``.

The L4 runtime DB (wyrd-d90t PR 1 + 4) stores per-usage rows whose
``data`` blob is the JSON shape ``{entries: [{meaning, modifier_tags,
modifier_type, word}]}`` that ``runtime_db_export._write_meanings``
emits. This module rehydrates those rows into the bundle-dict shape
``{subjects: [...], canonical_decompositions: ..., fantasy_morphemes:
..., joiners: ...}`` that the existing JSON-bundle loaders consume —
so the runtime can flip from the JSON bundle to the runtime DB
without touching ``load_meanings`` / ``load_canonical_decompositions``
/ ``load_fantasy_morphemes`` / ``load_joiners`` themselves.

The adapter synthesizes one pseudo-subject per L4 entry rather than
re-grouping entries back into multi-word subjects: ``load_meanings``
iterates per-word and the subject-level grouping doesn't affect the
final ``Meaning`` objects, so the simpler shape produces an
identical ``meaning_db``.

(Note: irish_anglicizations + Norman manorial families are still
loaded from JSON bundle sidecars by the caller — they're not in the
L4 runtime DB today. Folding them into the L4 emit is a follow-up;
filed under the d90t epic.)
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def bundle_dict_from_runtime_db(conn: sqlite3.Connection) -> dict[str, Any]:
    """Read every L4 table the bundle loaders care about and return a
    dict shaped like ``meanings.json``.

    Output keys (matching ``lexicon export-meanings`` output):
      * ``subjects`` — list of pseudo-subjects (one per L4 entry).
      * ``canonical_decompositions`` — dict[toponym_name → payload].
      * ``fantasy_morphemes`` — dict[input_name → payload].

    The ``joiners`` key is omitted (the L4 schema doesn't carry it
    today — it lived on a separate sidecar in the JSON-bundle world
    and never made it into the L4 emit).
    """
    return {
        "subjects": _read_subjects(conn),
        "canonical_decompositions": _read_canonical_decompositions(conn),
        "fantasy_morphemes": _read_fantasy_morphemes(conn),
    }


def _read_subjects(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Walk every meaning row + each row's ``data.entries[*]`` and
    synthesize a one-word pseudo-subject per entry.

    ``load_meanings`` iterates per-word and assigns subject-level
    metadata (meaning / modifier_tags / modifier_type) to each
    resulting Meaning — collapsing multi-entry rows into one-word
    subjects produces identical Meaning objects since the per-word
    iteration is what matters.

    Cursor-iterated rather than ``fetchall()`` so the meaning table's
    blobs (largest column in the L4) don't all land in memory at once
    on production-scale bundles.
    """
    subjects: list[dict[str, Any]] = []
    cursor = conn.execute("SELECT data FROM meaning ORDER BY usage_key")
    for (blob,) in cursor:
        payload = json.loads(blob)
        for entry in payload.get("entries") or []:
            word = entry.get("word")
            if not word:
                continue
            subjects.append(
                {
                    "meaning": entry.get("meaning") or [],
                    "modifier_tags": entry.get("modifier_tags") or [],
                    "modifier_type": entry.get("modifier_type"),
                    "words": [word],
                }
            )
    return subjects


def _read_canonical_decompositions(conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    """Project ``canonical_decomposition`` rows back into the
    ``{toponym_name: {signature, source}}`` map ``load_canonical_decompositions``
    expects. Cursor-iterated for the same memory reason as
    ``_read_subjects``."""
    cursor = conn.execute(
        "SELECT toponym_name, data FROM canonical_decomposition ORDER BY toponym_name"
    )
    return {name: json.loads(blob) for name, blob in cursor}


def _read_fantasy_morphemes(conn: sqlite3.Connection) -> dict[str, dict]:
    """Project ``fantasy_morpheme`` rows back into the
    ``{lowercase_input_name: payload}`` map ``load_fantasy_morphemes``
    expects. Keys come back lowercased per the L4 COLLATE NOCASE PK
    contract (the emitter already lowercases on write)."""
    cursor = conn.execute(
        "SELECT usage_key, data FROM fantasy_morpheme ORDER BY usage_key"
    )
    return {key: json.loads(blob) for key, blob in cursor}
