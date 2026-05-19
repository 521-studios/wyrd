"""Cross-language exact-form lookup bridging.

Walk every etymon and, if its canonical_form matches the
canonical_form of an etymon in a DIFFERENT specified language, link
them via lemma_id. Useful for trivial cross-language matches like NF
``cot`` → OE ``cot`` that don't need phonological transformation.

``bridge_celtic_forms`` extends this for Anglicized Celtic-substrate
toponyms; ``bridge_phonological_oe`` / ``bridge_phonological_on``
extend it for phonological transformations.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.db import LexiconDB


def bridge_generic_language(
    db: LexiconDB,
    *,
    generic_lang: str,
    candidate_langs: tuple[str, ...],
    apply: bool = False,
) -> dict:
    """Bridge a generic-language etymon (e.g. 'celtic') to the matching
    specific-language entry (e.g. 'irish' / 'welsh' / 'old-irish') by
    setting merged_into_id (D22 OCR-cluster style — non-destructive).

    Place-name dictionaries often write generic language tags like
    'celtic' for morphemes whose specific Celtic-family origin isn't
    pinned (or doesn't matter to the place-name analysis). Wiktextract
    entries are language-specific. This pass bridges the lookup
    mismatch by canonicalizing each generic-language etymon onto a
    specific-language match with the same canonical_form.

    For each generic-language etymon (canonical, not already merged):
      - Look up specific-language etymons matching `LOWER(canonical_form)`
        across `candidate_langs`.
      - If matches exist, pick the highest-priority one per
        `candidate_langs` order (typically Proto-* > Old-* > Middle-* >
        modern). Within a language, pick the smallest etymon id for
        determinism.
      - Set the generic-language etymon's `merged_into_id` to the picked
        canonical winner.

    The cluster-cognates pass is already redirect-aware, so any descent
    edges that previously pointed at the specific-language etymon now
    also cluster the merged generic etymon via the merged_into_id rollup
    chain.

    With apply=False (default) reports candidate counts without writing.
    Returns:
      - generic_etymons: total canonical generic-language etymons examined
      - bridged: count that found a specific-language match (would
        merge / did merge)
      - unmatched: count with no specific-language candidate (will
        remain as standalone canonical entries)
      - rows_written: actual UPDATE row count when apply=True (always 0
        in dry-run)
    """
    if not candidate_langs:
        raise ValueError("candidate_langs must be non-empty")

    # Build a (lower(canonical_form), language) → smallest_id lookup over
    # canonical specific-language etymons. One pass over the candidate
    # set keeps the bridging O(N) on the generic-language list.
    placeholders = ", ".join(["?"] * len(candidate_langs))
    candidate_index: dict[tuple[str, str], int] = {}
    for row in db.conn.execute(
        f"""
        SELECT id, canonical_form, language
        FROM etymon
        WHERE merged_into_id IS NULL
          AND language IN ({placeholders})
        ORDER BY id
        """,
        candidate_langs,
    ).fetchall():
        key = (row["canonical_form"].lower(), row["language"])
        # First-seen wins (smallest id) per (form, lang) — within a
        # language the lower id is the older entry, deterministic.
        if key not in candidate_index:
            candidate_index[key] = row["id"]

    # Walk generic-language canonical etymons; pick the first matching
    # candidate language per the priority order in `candidate_langs`.
    generic_rows = db.conn.execute(
        "SELECT id, canonical_form FROM etymon "
        "WHERE language = ? AND merged_into_id IS NULL ORDER BY id",
        (generic_lang,),
    ).fetchall()

    bridges: list[tuple[int, int]] = []  # (generic_id, target_id)
    for gen_row in generic_rows:
        form_lower = gen_row["canonical_form"].lower()
        for cand_lang in candidate_langs:
            target_id = candidate_index.get((form_lower, cand_lang))
            if target_id is not None:
                bridges.append((gen_row["id"], target_id))
                break

    rows_written = 0
    if apply and bridges:
        # Mirror cluster_ocr_variants's chain-flatten (D22): set
        # merged_into_id on the generic row AND re-route any
        # pre-existing redirect that pointed AT the generic row.
        # Without the OR-clause a 2-deep chain X → generic → target
        # would form, and the single-level COALESCE rollup in
        # etymon_consensus / etymon_canonical would split witnesses
        # across two GROUP BY buckets.
        cur = db.conn.executemany(
            "UPDATE etymon SET merged_into_id = ? "
            "WHERE (id = ? OR merged_into_id = ?) "
            "  AND merged_into_id IS NOT ?",
            [(tid, gid, gid, tid) for gid, tid in bridges],
        )
        rows_written = cur.rowcount
        # Re-parent any inflected children the generic row was acting
        # as a lemma for. Mining evidence stays on the original etymon.
        db.conn.executemany(
            "UPDATE etymon SET lemma_id = ? WHERE lemma_id = ?",
            [(tid, gid) for gid, tid in bridges],
        )
        db.commit()

    return {
        "generic_etymons": len(generic_rows),
        "bridged": len(bridges),
        "unmatched": len(generic_rows) - len(bridges),
        "rows_written": rows_written,
        "applied": apply,
    }
