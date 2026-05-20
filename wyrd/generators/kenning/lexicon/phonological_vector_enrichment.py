"""Phonological-vector enrichment pass for the lexicon (wyrd-kq7w.1).

Walks every etymon row, computes the 14-dim
:class:`PhonologicalVector` from its ``canonical_form`` +
``pronunciation_ipa`` columns, and writes the result as a JSON blob
into the ``etymon.phonological_vector`` column.

The pass is:
  * **Idempotent** — same input always produces the same output.
    Re-running on an unchanged DB writes byte-identical blobs.
  * **Incremental by default** — only processes rows with
    ``phonological_vector IS NULL`` unless ``force=True``.
  * **Read-side cheap** — the vector JSON is computed once and cached
    on the row; the scoring runtime decodes once per (lemma, request)
    via :func:`vectors.scoring.compute_phon_score`.

Wired into the canonical L3 enrichment chain in
:func:`wyrd.generators.kenning.enrichment.run_full_enrichment` after
``derive_english_shaped_all`` (both passes consume
``pronunciation_ipa``; placing the vector pass second keeps the IPA
column's value stable through the chain).
"""

from __future__ import annotations

import sqlite3
from typing import Any

from wyrd.generators.kenning.registers.phonological_vector_compute import (
    compute_phonological_vector,
    vector_to_json,
)

from .db import LexiconDB

PHON_VECTOR_METHOD_VERSION = "compute-phon-vector-v1"


def tag_phonological_vectors_all(
    db: LexiconDB,
    *,
    apply: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Compute + persist the PhonologicalVector for every etymon.

    Args:
        db: lexicon DB handle.
        apply: when False (default), counts candidates without writing.
        force: when True, recomputes vectors for every etymon (including
            rows that already have a non-null ``phonological_vector``).
            Use after a classifier-rule change or dimension addition
            to refresh the corpus. Default False = incremental
            (NULL-only).

    Returns:
        Dict with keys:
            ``method_version`` — stamp written alongside the blob for
            audit trail.
            ``candidates`` — count of rows that would be / were
            (re)processed.
            ``written`` — count of rows that actually got a new blob
            (only meaningful when ``apply=True``).
            ``sample`` — up to 5 (etymon_id, canonical_form,
            phonological_vector_json) tuples for spot-check / debug.
    """
    conn: sqlite3.Connection = db.conn

    # Select candidates. Force-mode walks every row; incremental mode
    # filters to rows with NULL phonological_vector.
    where_clause = "" if force else " WHERE phonological_vector IS NULL"
    rows = list(
        conn.execute("SELECT id, canonical_form, pronunciation_ipa FROM etymon" + where_clause)
    )

    candidates = len(rows)
    written = 0
    sample: list[tuple[int, str, str]] = []

    if apply:
        for row in rows:
            etymon_id = row["id"]
            canonical_form = row["canonical_form"] or ""
            ipa = row["pronunciation_ipa"]
            vector = compute_phonological_vector(canonical_form, ipa)
            blob = vector_to_json(vector)
            conn.execute(
                "UPDATE etymon SET phonological_vector = ? WHERE id = ?",
                (blob, etymon_id),
            )
            written += 1
            if len(sample) < 5:
                sample.append((etymon_id, canonical_form, blob))
        conn.commit()
    else:
        # Dry-run: still computes a few vectors so the sample is
        # populated for operator spot-check. Don't materialize the
        # full set — same result, more time.
        for row in rows[:5]:
            etymon_id = row["id"]
            canonical_form = row["canonical_form"] or ""
            ipa = row["pronunciation_ipa"]
            vector = compute_phonological_vector(canonical_form, ipa)
            sample.append((etymon_id, canonical_form, vector_to_json(vector)))

    return {
        "method_version": PHON_VECTOR_METHOD_VERSION,
        "candidates": candidates,
        "written": written,
        "sample": sample,
        "force": force,
    }


def phonological_vector_coverage(conn: sqlite3.Connection) -> dict[str, Any]:
    """Coverage breakdown for the ``lexicon enrichment-status`` report.

    Returns:
        ``total`` — total etymon rows.
        ``tagged`` — rows with ``phonological_vector IS NOT NULL``.
        ``ratio`` — tagged / total (0.0 when total == 0).
        ``by_language`` — sorted list of (language, total, tagged)
            tuples for languages with at least one tagged row.
            Surfaces the per-language coverage so the dashboard can
            flag languages that mining missed.
    """
    row = conn.execute(
        "SELECT  COUNT(*) AS total,  COUNT(phonological_vector) AS tagged FROM etymon"
    ).fetchone()
    total = row["total"]
    tagged = row["tagged"]
    by_language: list[tuple[str, int, int]] = []
    if total:
        lang_rows = conn.execute(
            "SELECT language, "
            " COUNT(*) AS total, "
            " COUNT(phonological_vector) AS tagged "
            "FROM etymon "
            "GROUP BY language "
            "HAVING tagged > 0 "
            "ORDER BY language"
        )
        by_language = [(r["language"], r["total"], r["tagged"]) for r in lang_rows]
    return {
        "total": total,
        "tagged": tagged,
        "ratio": (tagged / total) if total else 0.0,
        "by_language": by_language,
    }
