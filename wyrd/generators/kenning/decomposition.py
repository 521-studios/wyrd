"""Decomposition multiplicity (wyrd-08m Phase 1).

The matcher (``Name.find_meaning(reduce=False)``) returns every plausible
morpheme breakdown for a toponym. With ``reduce=True`` it picks one via a
fixed heuristic, but a single heuristic pick conflates two failure modes:

- **False gaps**: fragment X is unaccounted under decomposition A, while
  decomposition B accounts for everything. If A wins, X looks like a
  corpus gap that authoring effort would target.
- **False hits**: morpheme M is counted as 'matched' because it's a
  substring under decomposition A, but the scholarly decomposition B
  doesn't use M at all. M's apparent corpus frequency is inflated.

This module persists every decomposition the matcher produces (one row
per (toponym, signature)) and runs a rule-driven canonical picker that
favors scholarly attribution where available, then unique-zero-
unaccounted decompositions, leaving multi-zero ties for a Phase 2
tiebreaker.

Phase 1 ships:
- Schema (``toponym_decomposition``).
- ``compute_decompositions`` populator: reads toponym names, runs the
  matcher with ``reduce=False``, persists every decomposition.
- ``pick_canonical_decomposition`` rules (a) and (b).
- CLI: ``wyrd kenning lexicon decompose``.

Phase 2 will wire the canonical pick into ``rebuild-proportions``,
``unaccounted``, and the explainer; that's where the false-gap / false-
hit failure modes get resolved at the consumer side.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from wyrd.generators.kenning.lexicon import _LANG_CODE_TO_JSON_FIELD
from wyrd.generators.kenning.meaning import Meaning
from wyrd.generators.kenning.name import Name

if TYPE_CHECKING:
    from wyrd.generators.kenning.lexicon import LexiconDB


# --- decomposition signature ----------------------------------------------
#
# A decomposition is a list of ``Meaning | str`` items in slot order. The
# signature must be stable: the same shape produces the same hash across
# processes / Python versions. Items hash on:
#   - Meaning -> ("morpheme", usage)  — usage includes the dash markers
#     ("Bridg-", "-ford", "-ham-") that pin location.
#   - str     -> ("unaccounted", chars)
#
# The list ordering is the slot ordering — two decompositions with the
# same morphemes in different positions are different shapes (`Strat +
# ford` vs `ford + Strat` are different even when the multiset is equal).


def _decomposition_payload(decomposition: list) -> list:
    """Serialize a decomposition (list[Meaning|str]) into a stable JSON-
    safe payload. Each slot becomes a 2-list ``[kind, value]``:
    ``["morpheme", usage]`` for Meaning, ``["unaccounted", chars]`` for
    string."""
    payload = []
    for slot in decomposition:
        if isinstance(slot, Meaning):
            payload.append(["morpheme", slot.usage])
        else:
            payload.append(["unaccounted", str(slot)])
    return payload


def _signature_for_payload(payload: list) -> str:
    """SHA-1 hex digest of the canonical-JSON serialization of the payload."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _unaccounted_fragments_from(payload: list) -> list[str]:
    """Pull out just the unaccounted strings from a payload, preserving
    order. Used to populate ``unaccounted_fragments`` for fast querying
    without re-parsing ``morpheme_ids``."""
    return [value for kind, value in payload if kind == "unaccounted"]


def _morpheme_count_from(payload: list) -> int:
    return sum(1 for kind, _ in payload if kind == "morpheme")


# --- populator ------------------------------------------------------------


def compute_decompositions(
    db: LexiconDB,
    toponym_id: int,
    modern_name: str,
    word_db: dict,
) -> list[int]:
    """Run the matcher (reduce=False) on ``modern_name`` and persist every
    decomposition into ``toponym_decomposition``.

    Idempotent on (toponym_id, decomposition_signature) via INSERT OR
    IGNORE. A re-emit that unlocks new morphemes appends new rows;
    previously-seen rows keep their canonical assignment until
    ``pick_canonical_decomposition`` re-runs.

    Returns the ids of the rows touched, in matcher emit order.
    """
    name = Name(modern_name)
    name.find_meaning(word_db, reduce=False)
    # The matcher splits on whitespace and decomposes per word; a multi-
    # word toponym ("Saint Mary Bourne") has multiple words each with
    # multiple decompositions. We treat the cross-product of per-word
    # decompositions as the toponym's decompositions — each combination
    # is one full breakdown.
    decompositions = _cross_product_decompositions(name)
    touched_ids: list[int] = []
    for decomp in decompositions:
        payload = _decomposition_payload(decomp)
        signature = _signature_for_payload(payload)
        morpheme_ids = json.dumps(
            payload, ensure_ascii=False, sort_keys=False, separators=(",", ":")
        )
        unaccounted = _unaccounted_fragments_from(payload)
        unaccounted_json = json.dumps(
            unaccounted, ensure_ascii=False, sort_keys=False, separators=(",", ":")
        )
        morpheme_count = _morpheme_count_from(payload)
        cur = db.conn.execute(
            """
            INSERT OR IGNORE INTO toponym_decomposition (
              toponym_id, decomposition_signature, morpheme_ids,
              unaccounted_fragments, unaccounted_count, morpheme_count
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                toponym_id,
                signature,
                morpheme_ids,
                unaccounted_json,
                len(unaccounted),
                morpheme_count,
            ),
        )
        if cur.lastrowid:
            touched_ids.append(cur.lastrowid)
        else:
            row = db.conn.execute(
                "SELECT id FROM toponym_decomposition "
                "WHERE toponym_id = ? AND decomposition_signature = ?",
                (toponym_id, signature),
            ).fetchone()
            if row is not None:
                touched_ids.append(row["id"])
    return touched_ids


def _cross_product_decompositions(name: Name) -> list[list]:
    """Build the cross product of per-word decompositions for a name.

    A single-word toponym just returns the matcher's per-word list with
    each entry promoted to a list of ``Meaning|str`` (Word.word). A
    multi-word toponym ('Saint Mary Bourne') returns every combination
    of (word_1_decomp, word_2_decomp, word_3_decomp), with whitespace
    elided — slot ordering tracks the toponym's left-to-right reading.
    """
    if not name.words:
        return []
    per_word_lists: list[list[list]] = []
    for word_str in name.name.split(" "):
        word_objs = name.words.get(word_str, [])
        # Each Word has .word which is a list[Meaning|str]. Empty word
        # entries are surface-only fallbacks (no match found); the
        # matcher guarantees at least one entry per word.
        per_word: list[list] = [list(w.word) for w in word_objs] if word_objs else [[word_str]]
        per_word_lists.append(per_word)
    # Cross-product: start with [[]], extend with each word's options.
    combos: list[list] = [[]]
    for per_word in per_word_lists:
        new_combos = []
        for prefix in combos:
            for option in per_word:
                new_combos.append(prefix + option)
        combos = new_combos
    return combos


# --- canonical picker -----------------------------------------------------


def _meaning_form_keys(meaning: Meaning) -> set[tuple[str, str]]:
    """All (lang_field, canonical_form) pairs this Meaning carries.

    The set is the candidate identity for a slot in scholar-matching:
    a scholar's etymon (translated to lang_field) matches the slot if
    its (lang_field, canonical_form) appears in this set.
    """
    keys: set[tuple[str, str]] = set()
    for lang_field, forms in meaning.sources.items():
        if not isinstance(forms, list):
            continue
        for form in forms:
            if isinstance(form, str):
                keys.add((lang_field, form))
    return keys


def _decomposition_slots(decomposition: list) -> list[set[tuple[str, str]] | str]:
    """For each slot in order: a set of candidate (lang_field, form) pairs
    (Meaning slot), or the raw unaccounted string."""
    slots: list[set[tuple[str, str]] | str] = []
    for elem in decomposition:
        if isinstance(elem, Meaning):
            slots.append(_meaning_form_keys(elem))
        else:
            slots.append(str(elem))
    return slots


def _dedupe_scholar_seqs(
    seqs: list[list[tuple[str, str]]],
) -> list[list[tuple[str, str]]]:
    """Remove content-duplicate scholar sequences while preserving the
    first occurrence's order. Two identical sequences from different
    toponym_etymology rows mean two scholars said the same thing —
    that's corroboration, not disagreement."""
    seen: set[tuple[tuple[str, str], ...]] = set()
    distinct: list[list[tuple[str, str]]] = []
    for seq in seqs:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            distinct.append(seq)
    return distinct


def _scholar_form_sequences(db: LexiconDB, toponym_id: int) -> list[list[tuple[str, str]]]:
    """Fetch each scholar's ordered (lang_field, canonical_form) sequence
    for a toponym.

    One sequence per ``toponym_etymology`` row (one scholar's claim).
    Each sequence is the etymon list ordered by ``ordinal``, with each
    etymon's ``language`` translated to its bundle ``lang_field`` via
    ``_LANG_CODE_TO_JSON_FIELD``. Etymons whose language doesn't map to
    a bundle field are dropped from the sequence — they can't possibly
    match a Meaning's sources.
    """
    rows = db.conn.execute(
        """
        SELECT te.id AS etymology_id,
               tee.ordinal,
               e.language,
               e.canonical_form
          FROM toponym_etymology te
          JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id
          JOIN etymon e ON e.id = tee.etymon_id
         WHERE te.toponym_id = ?
         ORDER BY te.id, tee.ordinal
        """,
        (toponym_id,),
    ).fetchall()
    by_etymology: dict[int, list[tuple[str, str]]] = {}
    for row in rows:
        lang_field = _LANG_CODE_TO_JSON_FIELD.get(row["language"])
        if not lang_field:
            continue
        canonical = row["canonical_form"]
        if not canonical:
            continue
        by_etymology.setdefault(row["etymology_id"], []).append((lang_field, canonical))
    return [seq for seq in by_etymology.values() if seq]


def _scholar_seq_matches_decomposition(
    scholar_seq: list[tuple[str, str]],
    slots: list[set[tuple[str, str]] | str],
) -> bool:
    """A scholar's ordered (lang_field, form) sequence matches the
    decomposition iff:

    - the decomposition has zero unaccounted slots (a scholarly
      breakdown can't have unaccounted fragments — every piece is
      identified);
    - the morpheme-slot count equals the scholar element count; AND
    - each scholar slot fits in the corresponding morpheme slot's
      candidate set, in order.

    Strict ordering rather than multiset matching: scholars publish
    breakdowns in left-to-right reading order ('Strat + ford', not
    'ford + Strat'), and the matcher preserves slot order.
    """
    if any(isinstance(slot, str) for slot in slots):
        return False
    morpheme_slots = [slot for slot in slots if isinstance(slot, set)]
    if len(morpheme_slots) != len(scholar_seq):
        return False
    return all(
        scholar_form in slot for slot, scholar_form in zip(morpheme_slots, scholar_seq, strict=True)
    )


def pick_canonical_decomposition(
    db: LexiconDB,
    toponym_id: int,
    word_db: dict,
) -> dict[str, int | str | None]:
    """Apply Phase 1 canonical-picker rules (a) and (b) to a toponym's
    stored decompositions.

    Rule (a) — scholar match: if a stored decomposition's slot list
    aligns with any ``toponym_etymology`` row's ordered etymon list,
    that decomposition is canonical with source=``'scholar'``.
    Multiple matching scholar rows (i.e. scholars proposed different
    breakdowns AND each has a matching stored decomposition) result in
    every matching stored decomposition being marked canonical with
    source=``'scholar-disagreement'`` — the schema preserves the
    disagreement rather than picking a single 'truth'.

    Rule (b) — unique-zero-unaccounted: if rule (a) didn't fire AND
    exactly one stored decomposition has zero unaccounted fragments,
    that one is canonical with source=``'unique-zero-unaccounted'``.

    Rule (c) — multi-zero tiebreaker — is Phase 2, NOT applied here.
    Toponyms that fall through both rules end with no canonical pick;
    Phase 2 will fill the gap.

    Returns a result dict ``{rule, canonical_count, decomposition_count}``
    so callers (CLI, tests) can summarize.
    """
    # Reset any prior canonical pick for this toponym; the rule chain
    # below fully redetermines canonical state. Idempotent across
    # re-runs because it's a clean reset + reapply, not a partial
    # update.
    db.conn.execute(
        """
        UPDATE toponym_decomposition
           SET is_canonical = 0, canonical_source = NULL
         WHERE toponym_id = ?
        """,
        (toponym_id,),
    )

    rows = db.conn.execute(
        """
        SELECT id, morpheme_ids, unaccounted_count
          FROM toponym_decomposition
         WHERE toponym_id = ?
         ORDER BY id
        """,
        (toponym_id,),
    ).fetchall()
    if not rows:
        return {"rule": None, "canonical_count": 0, "decomposition_count": 0}

    decomposition_count = len(rows)

    # Rebuild slot-candidate sets per stored row from morpheme_ids JSON.
    # Note: stored rows hold stable ``usage`` strings for matched slots,
    # not Meaning objects. We rehydrate slot-candidate sets by looking
    # up each ``usage`` against the live word_db. A usage absent from
    # word_db (e.g. after a bundle re-emit removed it) yields an empty
    # candidate set — a scholar-form lookup against empty fails, so the
    # row safely passes scholar matching without false positives.
    slots_per_row: list[list[set[tuple[str, str]] | str]] = []
    for row in rows:
        payload = json.loads(row["morpheme_ids"])
        slots: list[set[tuple[str, str]] | str] = []
        for kind, value in payload:
            if kind == "morpheme":
                slots.append(_candidates_for_usage(word_db, value))
            else:
                slots.append(value)
        slots_per_row.append(slots)

    scholar_seqs = _scholar_form_sequences(db, toponym_id)
    # De-dupe scholar sequences by content. Two scholars who happened
    # to publish IDENTICAL etymon lists are not in disagreement; they
    # corroborate. Without this dedup, two duplicate scholar_etymology
    # rows would both match the same stored decomposition and inflate
    # the distinct-scholar count to >=2, mis-tagging as
    # 'scholar-disagreement'.
    distinct_scholar_seqs = _dedupe_scholar_seqs(scholar_seqs)

    # Rule (a): collect every (row_idx, scholar_idx) pair that matches
    # against the deduplicated scholar list. Disagreement iff >=2
    # DISTINCT scholar breakdowns each matched.
    matching_pairs: list[tuple[int, int]] = []
    matched_scholar_indices: set[int] = set()
    for r_idx, slots in enumerate(slots_per_row):
        for s_idx, seq in enumerate(distinct_scholar_seqs):
            if _scholar_seq_matches_decomposition(seq, slots):
                matching_pairs.append((r_idx, s_idx))
                matched_scholar_indices.add(s_idx)

    if matching_pairs:
        is_disagreement = len(matched_scholar_indices) >= 2
        source = "scholar-disagreement" if is_disagreement else "scholar"
        matching_row_ids = sorted({rows[r_idx]["id"] for r_idx, _ in matching_pairs})
        _set_canonical(db, matching_row_ids, source)
        return {
            "rule": source,
            "canonical_count": len(matching_row_ids),
            "decomposition_count": decomposition_count,
        }

    # Rule (b): unique zero-unaccounted.
    zero_rows = [row for row in rows if row["unaccounted_count"] == 0]
    if len(zero_rows) == 1:
        _set_canonical(db, [zero_rows[0]["id"]], "unique-zero-unaccounted")
        return {
            "rule": "unique-zero-unaccounted",
            "canonical_count": 1,
            "decomposition_count": decomposition_count,
        }

    # Phase 1 stops here. Multi-zero / no-zero cases get a Phase 2
    # tiebreaker.
    return {
        "rule": None,
        "canonical_count": 0,
        "decomposition_count": decomposition_count,
    }


def _candidates_for_usage(word_db: dict, usage: str) -> set[tuple[str, str]]:
    """Union of (lang_field, canonical_form) candidates across every
    Meaning sharing a ``modern_usage`` in the bundle.

    A usage may map to multiple Meanings — same surface, different
    senses ('-y' as 'island' OR 'district') — each carrying its own
    sources. The union is the right candidate set: a scholar form
    matches the slot if ANY co-located Meaning carries it.
    """
    candidates: set[tuple[str, str]] = set()
    for meaning in word_db.get(usage, []):
        candidates |= _meaning_form_keys(meaning)
    return candidates


def _set_canonical(db: LexiconDB, row_ids: list[int], source: str) -> None:
    """Mark ``row_ids`` as canonical with the given source. Caller is
    expected to have cleared the prior canonical state before calling
    (avoids partial-state on non-idempotent re-runs)."""
    if not row_ids:
        return
    placeholders = ",".join("?" * len(row_ids))
    db.conn.execute(
        f"""
        UPDATE toponym_decomposition
           SET is_canonical = 1, canonical_source = ?
         WHERE id IN ({placeholders})
        """,
        (source, *row_ids),
    )


# --- top-level orchestration ----------------------------------------------


def populate_and_pick(
    db: LexiconDB,
    toponym_id: int,
    modern_name: str,
    word_db: dict,
) -> dict[str, int | str | None]:
    """Convenience: compute decompositions for one toponym, then run the
    canonical picker.

    Useful as the per-toponym body of the CLI's batch loop. Returns the
    picker's summary dict so the caller can roll up rule-firing counts.
    """
    compute_decompositions(db, toponym_id, modern_name, word_db)
    return pick_canonical_decomposition(db, toponym_id, word_db)
