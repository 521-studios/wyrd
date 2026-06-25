"""Shared lookup-table driven phonological bridge (OE / ON).

Both OE and ON phonological bridges run the same algorithm against
different lookup tables: walk every canonical etymon in the target
language, look the lowercased canonical_form up in the table (a flat
``dict[str, str]`` mapping modernized / Anglicized spelling to its
scholarly orthographic form), and link any matched etymon to the
lookup target via ``merged_into_id`` (D22 non-destructive shape — the
loser stays addressable through the merge chain). Underscore-prefixed
because it's package-private to ``bridges.oe`` / ``bridges.on``;
callers that want a different language supply their own table.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.db import LexiconDB


def _compute_phonological_bridges(
    examined_rows: list, table: dict[str, str], target_index: dict[str, int]
) -> tuple[list[tuple[int, int]], int]:
    """Match each examined (canonical) row to its bridge target.

    For each row, look its lower-cased ``canonical_form`` up in the bridge
    ``table`` to get the Wiktionary form, resolve that form to a target etymon id
    via ``target_index``, and collect the ``(source_id, target_id)`` redirect.
    Returns ``(bridges, missing_target)`` where ``missing_target`` counts rows
    whose bridged Wiktionary form had no matching etymon. Rows absent from the
    table, and rows that would self-bridge (target is the row itself), are
    skipped. Pure — no DB.
    """
    bridges: list[tuple[int, int]] = []
    missing_target = 0
    for row in examined_rows:
        form = row["canonical_form"].lower()
        wiktionary_form = table.get(form)
        if wiktionary_form is None:
            continue
        target_id = target_index.get(wiktionary_form.lower())
        if target_id is None:
            missing_target += 1
            continue
        if target_id == row["id"]:
            continue
        bridges.append((row["id"], target_id))
    return bridges, missing_target


def _bridge_same_language_phonological(
    db: LexiconDB, *, language: str, table: dict[str, str], apply: bool
) -> dict:
    """Shared engine for same-language phonological bridges.

    OE and ON place-name etymologies both suffer from the same structural
    mismatch against Wiktionary: scholar place-name dictionaries write
    modernized / Anglicized spellings (ton, lea, by, holm) while
    Wiktionary uses scholarly orthography with macrons, æ, þ/ð, etc.
    (tūn, lēah, býr, hólmr). Each language gets its own hand-curated
    bridge table; this function applies one against the canonical rows
    of `language`, walking merged_into_id chains so a bridge value that
    names a tombstone still resolves to the live canonical.

    Returns the same shape as the public bridge_phonological_* wrappers.
    """
    all_rows = db.conn.execute(
        "SELECT id, canonical_form, merged_into_id FROM etymon WHERE language = ? ORDER BY id",
        (language,),
    ).fetchall()
    chain: dict[int, int | None] = {r["id"]: r["merged_into_id"] for r in all_rows}

    def _resolve_canonical(start_id: int) -> int:
        cid = start_id
        visited: set[int] = set()
        while (next_id := chain.get(cid)) is not None and cid not in visited:
            visited.add(cid)
            cid = next_id
        return cid

    target_index: dict[str, int] = {}
    for row in all_rows:
        key = row["canonical_form"].lower()
        if key not in target_index:
            target_index[key] = _resolve_canonical(row["id"])

    examined_rows = [r for r in all_rows if r["merged_into_id"] is None]
    bridges, missing_target = _compute_phonological_bridges(examined_rows, table, target_index)

    rows_written = 0
    if apply and bridges:
        # Chain-flatten + lemma-reparent in batch (mirrors
        # cluster_ocr_variants's D22 pattern). The OR-clause re-routes
        # any pre-existing redirect that pointed AT this place-name
        # etymon onto the canonical target, preventing a 2-deep chain
        # that would split witnesses in the single-level COALESCE
        # rollup used by etymon_consensus / etymon_canonical.
        cur = db.conn.executemany(
            "UPDATE etymon SET merged_into_id = ? "
            "WHERE (id = ? OR merged_into_id = ?) "
            "  AND merged_into_id IS NOT ?",
            [(tid, sid, sid, tid) for sid, tid in bridges],
        )
        rows_written = cur.rowcount
        db.conn.executemany(
            "UPDATE etymon SET lemma_id = ? WHERE lemma_id = ?",
            [(tid, sid) for sid, tid in bridges],
        )
        db.commit()

    return {
        "examined": len(examined_rows),
        "bridged": len(bridges),
        "unmatched": len(examined_rows) - len(bridges) - missing_target,
        "missing_target": missing_target,
        "rows_written": rows_written,
        "applied": apply,
    }
