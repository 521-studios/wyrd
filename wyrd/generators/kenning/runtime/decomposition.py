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
unaccounted decompositions, leaving multi-zero ties for a future
tiebreaker rule (c).

Phase 1 ships:
- Schema (``toponym_decomposition``).
- ``compute_decompositions`` populator: reads toponym names, runs the
  matcher with ``reduce=False``, persists every decomposition.
- ``pick_canonical_decomposition`` rules (a) and (b).
- CLI: ``wyrd kenning lexicon decompose``.

Phase 3 (this module's ``decompose_with_canonical`` helper) wires the
canonical pick into ``rebuild-proportions`` and ``unaccounted`` at the
consumer side. KenningExplain (Lambda / SPA path) has no DB access and
remains heuristic-only until a future bundle-side projection lands.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from importlib import resources
from typing import TYPE_CHECKING

from wyrd.generators.kenning.lexicon import _LANG_CODE_TO_JSON_FIELD

from .meaning import Meaning, load_meanings
from .name import Name

if TYPE_CHECKING:
    from wyrd.generators.kenning.lexicon import LexiconDB


_logger = logging.getLogger(__name__)


# Cap on how many cross-product decomposition combos a single toponym
# can produce. Without a cap, a multi-word toponym whose words each
# admit many parses could explode the candidate list (worst case:
# K^N where K is the per-word option count, N is the word count). The
# cap protects against bulk-`--apply` runs against the live DB
# producing pathologically-large rows for a few outlier names.
#
# 256 is generous for real toponyms (most have 1-2 words with 1-5
# decompositions each); when a single name overflows the cap, only
# the first 256 combos persist and a warning surfaces on the logger.
_MAX_CROSS_PRODUCT_COMBOS: int = 256


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

    Single-word toponyms return the matcher's per-word list as-is.
    Multi-word toponyms ('Saint Mary Bourne') return every combination
    of (word_1_decomp, word_2_decomp, word_3_decomp), with whitespace
    elided.

    Capped at ``_MAX_CROSS_PRODUCT_COMBOS``. When the projected combo
    count would exceed the cap, per-word option lists are truncated
    BEFORE multiplying so every emitted combo remains full-length.
    Mid-multiplication truncation would emit prefix-only combos
    missing later words' contributions — those would persist as
    'complete' decompositions with bogus zero-unaccounted scores.

    Truncation is deterministic (first K options per word kept, where
    K is sized so the product stays under the cap) and warns via the
    module logger. Re-runs against the same input produce identical
    row sets.
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

    per_word_lists = _cap_per_word_options(per_word_lists, name.name)

    combos: list[list] = [[]]
    for per_word in per_word_lists:
        combos = [prefix + option for prefix in combos for option in per_word]
    return combos


def _cap_per_word_options(
    per_word_lists: list[list[list]],
    toponym_name: str,
) -> list[list[list]]:
    """Cap each per-word option list so the cross-product stays under
    ``_MAX_CROSS_PRODUCT_COMBOS``. Truncates the longest per-word
    lists first (greedy) until the projected product fits.

    All emitted combos remain full-length: each word still contributes
    one option per combo, just drawn from a smaller pool. Deterministic
    (keeps the first K options of each list, no sampling).
    """
    if not per_word_lists:
        return per_word_lists
    projected = 1
    for per_word in per_word_lists:
        projected *= max(1, len(per_word))
    if projected <= _MAX_CROSS_PRODUCT_COMBOS:
        return per_word_lists

    # Trim the longest list down by one option at a time until the
    # projected product fits. Stable: order of equal-length lists
    # respects input order so re-runs produce identical truncations.
    capped = [list(pw) for pw in per_word_lists]
    while projected > _MAX_CROSS_PRODUCT_COMBOS:
        # Find the longest list with >1 option (can't trim singleton
        # lists without dropping a word entirely).
        idx = max(
            range(len(capped)),
            key=lambda i: len(capped[i]) if len(capped[i]) > 1 else -1,
        )
        if len(capped[idx]) <= 1:
            break  # everything is singleton; can't trim further
        # Drop the last option from this list. Re-project.
        capped[idx].pop()
        projected = 1
        for per_word in capped:
            projected *= max(1, len(per_word))

    _logger.warning(
        "Cross-product cap (%d) reached for toponym %r — capped per-word "
        "options (was %s, now %s) so all combos stay full-length.",
        _MAX_CROSS_PRODUCT_COMBOS,
        toponym_name,
        [len(pw) for pw in per_word_lists],
        [len(pw) for pw in capped],
    )
    return capped


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
    a bundle field — OR whose ``canonical_form`` is empty — are dropped
    from the sequence and a debug log surfaces the drop, since the
    silently-shortened sequence can later fail length-equality checks
    against the matcher's slot count.
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
            _logger.debug(
                "Dropping etymon (toponym=%d, etymology=%d): unmapped language %r.",
                toponym_id,
                row["etymology_id"],
                row["language"],
            )
            continue
        canonical = row["canonical_form"]
        if not canonical:
            _logger.debug(
                "Dropping etymon (toponym=%d, etymology=%d): empty canonical_form.",
                toponym_id,
                row["etymology_id"],
            )
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
    """Apply canonical-picker rules to a toponym's stored decompositions.

    Rule (a) scholar match: if a stored decomposition's slot list aligns
    with any ``toponym_etymology`` row's ordered etymon list, mark it
    canonical with source ``'scholar'``. Multiple matching scholar
    breakdowns → every matching stored row tagged
    ``'scholar-disagreement'``.

    Rule (b) unique-zero-unaccounted: if rule (a) didn't fire AND
    exactly one stored row has zero unaccounted fragments, that one
    wins ``'unique-zero-unaccounted'``.

    Rule (c) multi-zero tiebreaker (Phase 4a): when 2+ stored rows
    have zero unaccounted, prefers the smallest ``morpheme_count``
    (Occam) with lex-order on ``morpheme_ids`` as the deterministic
    final tiebreak. Tagged ``'tiebreaker'`` so consumers can audit
    these picks. Witness-count + rare-morpheme refinements (from the
    parent ticket's rule-c sub-criteria) are deferred to Phase 4b.

    Toponyms with no zero-unaccounted decomposition (every breakdown
    has at least one leftover) fall through with no canonical pick.

    Returns ``{rule, canonical_count, decomposition_count}``.
    """
    _reset_canonical(db, toponym_id)
    rows = _load_stored_decompositions(db, toponym_id)
    if not rows:
        return {"rule": None, "canonical_count": 0, "decomposition_count": 0}

    decomposition_count = len(rows)
    slots_per_row = _build_slot_candidates_per_row(rows, word_db)

    scholar_match = _try_rule_scholar(db, toponym_id, rows, slots_per_row)
    if scholar_match is not None:
        source, matching_row_ids = scholar_match
        _set_canonical(db, matching_row_ids, source)
        return {
            "rule": source,
            "canonical_count": len(matching_row_ids),
            "decomposition_count": decomposition_count,
        }

    zero_row_id = _try_rule_unique_zero(rows)
    if zero_row_id is not None:
        _set_canonical(db, [zero_row_id], "unique-zero-unaccounted")
        return {
            "rule": "unique-zero-unaccounted",
            "canonical_count": 1,
            "decomposition_count": decomposition_count,
        }

    tiebreak_row_id = _try_rule_tiebreaker(rows)
    if tiebreak_row_id is not None:
        _set_canonical(db, [tiebreak_row_id], "tiebreaker")
        return {
            "rule": "tiebreaker",
            "canonical_count": 1,
            "decomposition_count": decomposition_count,
        }

    return {
        "rule": None,
        "canonical_count": 0,
        "decomposition_count": decomposition_count,
    }


def _reset_canonical(db: LexiconDB, toponym_id: int) -> None:
    """Clear prior canonical state for a toponym so the rule chain
    fully redetermines canonical assignments. Idempotent across
    re-runs (clean reset + reapply, not partial update)."""
    db.conn.execute(
        """
        UPDATE toponym_decomposition
           SET is_canonical = 0, canonical_source = NULL
         WHERE toponym_id = ?
        """,
        (toponym_id,),
    )


def _load_stored_decompositions(db: LexiconDB, toponym_id: int) -> list:
    """Fetch (id, morpheme_ids JSON, unaccounted_count, morpheme_count)
    rows for a toponym, ordered by id (insert order). morpheme_count
    is read because the Phase 4a tiebreaker rule (c) needs it for
    Occam-ordering among multi-zero-unaccounted candidates."""
    return db.conn.execute(
        """
        SELECT id, morpheme_ids, unaccounted_count, morpheme_count
          FROM toponym_decomposition
         WHERE toponym_id = ?
         ORDER BY id
        """,
        (toponym_id,),
    ).fetchall()


def _build_slot_candidates_per_row(
    rows: list,
    word_db: dict,
) -> list[list[set[tuple[str, str]] | str]]:
    """Rehydrate the per-row slot-candidate lists from stored
    morpheme_ids JSON.

    Stored rows hold stable ``usage`` strings for matched slots, not
    Meaning objects. Each row's slots are rebuilt by looking up each
    usage's (lang_field, canonical_form) candidate set against the
    live word_db. A usage absent from word_db (e.g. after a bundle
    re-emit removed it) yields an empty candidate set; the scholar
    matcher then can't false-positive on it.
    """
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
    return slots_per_row


def _try_rule_scholar(
    db: LexiconDB,
    toponym_id: int,
    rows: list,
    slots_per_row: list[list[set[tuple[str, str]] | str]],
) -> tuple[str, list[int]] | None:
    """Rule (a): scholar match. Returns ``(source, matching_row_ids)``
    if any scholar breakdown matched a stored decomposition; ``None``
    otherwise.

    ``source`` is ``'scholar-disagreement'`` if >=2 distinct scholar
    breakdowns each matched; ``'scholar'`` if exactly one breakdown
    matched (possibly multiple stored rows, but they share the same
    breakdown).
    """
    distinct_scholar_seqs = _dedupe_scholar_seqs(_scholar_form_sequences(db, toponym_id))
    matching_pairs: list[tuple[int, int]] = []
    matched_scholar_indices: set[int] = set()
    for r_idx, slots in enumerate(slots_per_row):
        for s_idx, seq in enumerate(distinct_scholar_seqs):
            if _scholar_seq_matches_decomposition(seq, slots):
                matching_pairs.append((r_idx, s_idx))
                matched_scholar_indices.add(s_idx)
    if not matching_pairs:
        return None
    is_disagreement = len(matched_scholar_indices) >= 2
    source = "scholar-disagreement" if is_disagreement else "scholar"
    matching_row_ids = sorted({rows[r_idx]["id"] for r_idx, _ in matching_pairs})
    return source, matching_row_ids


def _try_rule_unique_zero(rows: list) -> int | None:
    """Rule (b): unique zero-unaccounted. Returns the row id of the
    sole zero-unaccounted decomposition, or ``None`` if zero or
    multiple stored rows have unaccounted_count==0."""
    zero_rows = [row for row in rows if row["unaccounted_count"] == 0]
    if len(zero_rows) != 1:
        return None
    return zero_rows[0]["id"]


def _try_rule_tiebreaker(rows: list) -> int | None:
    """Rule (c) Phase 4a: multi-zero tiebreaker via Occam (fewest
    morphemes) + lex-order final fallback.

    Fires when two-or-more stored rows have ``unaccounted_count == 0``
    and rule (a) [scholar match] didn't cover them. Among the zero-
    unaccounted candidates, prefers:

    1. The row with the smallest ``morpheme_count`` (Occam: fewer
       morphemes = simpler analysis, less likely to be over-segmented).
    2. If still tied, the lex-smallest ``morpheme_ids`` JSON string
       (deterministic — re-runs against the same DB always pick the
       same row).

    Witness-count and rare-morpheme penalties from the parent ticket
    are deferred to Phase 4b (require an etymon→toponym_decomposition
    mapping that doesn't exist yet). Phase 4a covers the common case:
    most multi-zero ties differ in morpheme count.

    Returns the chosen row id, or ``None`` if there are fewer than
    two zero-unaccounted candidates (rule (b) already handled those).
    """
    zero_rows = [row for row in rows if row["unaccounted_count"] == 0]
    if len(zero_rows) < 2:
        return None
    # Sort by (morpheme_count ASC, morpheme_ids ASC) — both tiebreakers
    # in one pass. Lex-order on the JSON serialization is deterministic
    # across Python versions because morpheme_ids is built with
    # ``sort_keys=False, separators=(",", ":")`` at populate time.
    sorted_rows = sorted(zero_rows, key=lambda r: (r["morpheme_count"], r["morpheme_ids"]))
    return sorted_rows[0]["id"]


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


def decompose_all(
    db: LexiconDB,
    *,
    apply: bool = True,
    word_db: dict | None = None,
) -> dict[str, int]:
    """Uniform L3 wrapper for the ``decompose`` pass — runs
    :func:`populate_and_pick` against every toponym in ``db`` and
    rolls up the rule-firing counts.

    Wyrd-hidb Phase 2 plumbs this into ``run_full_enrichment``.
    Dry-run (``apply=False``) writes per-toponym decompositions into the
    open transaction but skips the final ``db.commit()`` — the
    transaction's eventual implicit rollback (CLI exit) or the
    orchestrator's outer transaction handling is what unwinds them.
    Decompose_all does NOT call ``rollback()`` itself: that would wipe
    uncommitted writes from earlier passes in the enrichment chain. The
    other L3 wrappers all use the "if apply: write+commit; else: no-op"
    pattern; decompose_all is the outlier because
    compute_decompositions/pick_canonical_decomposition write
    unconditionally and would be too invasive to split.

    ``word_db`` is the matcher's word dictionary (loaded from the
    runtime ``meanings.json`` bundle). Passing it explicitly keeps
    this function pure relative to bundle resolution — tests inject
    a synthetic word_db; the CLI / enrichment chain load the live
    bundle at the call site.

    Empty-DB fast path: when the toponym table has no rows the
    bundle load (~34 MB of JSON) is skipped entirely. Speeds up
    schema-only rebuild smokes from ~1 s to ~10 ms.
    """
    rule_counts: dict[str, int] = {
        "scholar": 0,
        "scholar-disagreement": 0,
        "unique-zero-unaccounted": 0,
        "tiebreaker": 0,
        "no-canonical": 0,
    }
    decompositions = 0
    # Cheap pre-check before the bundle load on schema-only DBs.
    has_rows = db.conn.execute("SELECT 1 FROM toponym LIMIT 1").fetchone() is not None
    if not has_rows:
        return {
            "applied": int(apply),
            "toponyms_scanned": 0,
            "decompositions": 0,
            **rule_counts,
        }

    if word_db is None:
        bundle_text = (
            resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
        )
        word_db, _ = load_meanings(json.loads(bundle_text))

    # Iterate the cursor directly instead of fetchall() so the toponym
    # table never has to fit in memory in one shot. populate_and_pick
    # writes to toponym_decomposition (not toponym), so the outer cursor
    # is not invalidated by per-row INSERTs.
    toponyms_scanned = 0
    cursor = db.conn.execute("SELECT id, modern_name FROM toponym ORDER BY id")
    for row in cursor:
        summary = populate_and_pick(db, row["id"], row["modern_name"], word_db)
        decompositions += int(summary["decomposition_count"] or 0)
        rule_key = summary["rule"] or "no-canonical"
        rule_counts[rule_key] = rule_counts.get(rule_key, 0) + 1
        toponyms_scanned += 1

    if apply:
        db.commit()

    return {
        "applied": int(apply),
        "toponyms_scanned": toponyms_scanned,
        "decompositions": decompositions,
        **rule_counts,
    }


# --- Phase 3 consumer integration -----------------------------------------
#
# Phase 1 stored every plausible decomposition + tagged a canonical pick;
# Phase 2 refactored the picker. Phase 3 wires the pick into the matcher
# consumers (``rebuild-proportions`` and ``unaccounted``) so their
# proportion / gap counts honour the canonical breakdown instead of the
# matcher's per-name reduce=True heuristic.
#
# The integration is OPT-IN at the CLI layer: callers pass an optional
# LexiconDB; consumers narrow each matched ``Name`` to the canonical
# cross-product cell when one exists, and fall back to the heuristic
# pick otherwise.


def lookup_canonical_signature(
    db: LexiconDB,
    modern_name: str,
    region: str | None = None,
) -> tuple[str, str] | None:
    """Find the canonical decomposition for a toponym by name.

    Returns ``(decomposition_signature, canonical_source)`` for the
    canonical row, or ``None`` if no toponym row exists OR no
    canonical was picked (Phase 1 multi-zero ties, or coverage gaps
    waiting on rule (c)).

    Toponym lookup keys on ``modern_name``. When ``region`` is supplied
    (place_names JSON files carry country/region grouping — wyrd-8m87)
    the matcher prefers an exact-region row, then falls back to a
    NULL-region row for the same name (legacy ingestion compat).
    Other-region rows are deliberately excluded — silently picking
    'Stratford in Bedfordshire' when the corpus passed
    'Stratford in Yorkshire' would attribute Bedfordshire's
    scholarship to Yorkshire's toponym, biasing proportion + gap
    counts. Returning ``None`` on cross-region miss is safer; the
    caller falls back to the heuristic.

    When ``region`` is ``None`` the lookup matches any toponym row
    sharing the name (lowest-id wins for determinism) — used by
    consumers that don't carry region context yet.

    Multiple canonical rows under one toponym (``scholar-disagreement``
    source — every matching scholar breakdown is flagged canonical)
    collapse to the lowest-id canonical row, keeping consumer counts
    stable across re-runs.
    """
    if region is not None:
        # Prefer exact-region match; fall back to a NULL-region row for
        # the same name (legacy ingestion compat). ORDER prefers exact
        # region over NULL via ``region IS NULL`` ascending — non-NULL
        # rows sort first.
        cur = db.conn.execute(
            """
            SELECT id FROM toponym
             WHERE modern_name = ?
               AND (region = ? OR region IS NULL)
             ORDER BY (region IS NULL) ASC, id ASC
             LIMIT 1
            """,
            (modern_name, region),
        )
    else:
        cur = db.conn.execute(
            "SELECT id FROM toponym WHERE modern_name = ? ORDER BY id LIMIT 1",
            (modern_name,),
        )
    row = cur.fetchone()
    if row is None:
        return None
    toponym_id = row["id"]
    cur = db.conn.execute(
        """
        SELECT decomposition_signature, canonical_source
          FROM toponym_decomposition
         WHERE toponym_id = ? AND is_canonical = 1
         ORDER BY id LIMIT 1
        """,
        (toponym_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return row["decomposition_signature"], row["canonical_source"]


def apply_canonical_to_name(name: Name, canonical_signature: str) -> bool:
    """Narrow ``name.words`` to the cross-product cell whose payload
    signature matches ``canonical_signature``.

    Caller must invoke ``name.find_meaning(word_db, reduce=False)``
    first so all alternates are populated. Iterates the cross-product
    of per-word options, computes each cell's payload signature, and
    on first match replaces every ``name.words[word]`` list with a
    singleton holding the chosen Word.

    Per-word option lists are capped via ``_cap_per_word_options`` so
    the search space matches the populator's. Without this symmetry,
    ``compute_decompositions`` would emit signatures over a capped
    cross-product while ``apply`` iterates the uncapped one — fine
    when the canonical is in the kept first-K options of every word
    (the common case), but a determinism crack when truncation order
    shifts between populate and apply runs (e.g. trie cache rebuild
    on a fresh process). Capping here keeps both sides on the same
    K-prefix per word.

    Returns ``True`` on hit, ``False`` on miss. ``name.words`` is
    untouched on miss so the caller can fall back to the heuristic.
    A miss happens when the bundle's word_db has shifted since the
    canonical was picked — re-running ``decompose --apply`` repairs
    the canonical.
    """
    word_keys = name.name.split(" ")
    per_word_options = [name.words.get(w, []) for w in word_keys]
    if not all(per_word_options):
        return False
    capped = _cap_per_word_options(per_word_options, name.name)
    for combo_idx in itertools.product(*[range(len(opts)) for opts in capped]):
        flat: list = []
        for w_idx, opt_idx in enumerate(combo_idx):
            flat.extend(capped[w_idx][opt_idx].word)
        sig = _signature_for_payload(_decomposition_payload(flat))
        if sig == canonical_signature:
            for w_idx, opt_idx in enumerate(combo_idx):
                name.words[word_keys[w_idx]] = [capped[w_idx][opt_idx]]
            return True
    return False


def decompose_with_canonical(
    name_str: str,
    word_db: dict,
    db: LexiconDB | None,
    region: str | None = None,
) -> tuple[Name, str | None]:
    """Decompose ``name_str`` honouring the canonical pick when one exists.

    When ``db`` is None, falls through to ``find_meaning(reduce=True)``
    — the legacy heuristic path. With a db: looks up the canonical
    signature for the toponym, runs ``find_meaning(reduce=False)``,
    then narrows to the canonical cross-product cell. Falls back to
    the heuristic when (a) no toponym row exists, (b) no canonical
    was picked, or (c) the canonical signature doesn't match any
    cross-product cell of the current word_db.

    Returns ``(name, source)`` where ``source`` is the canonical
    source string (``'scholar'`` / ``'scholar-disagreement'`` /
    ``'unique-zero-unaccounted'``) when canonical was applied, or
    ``None`` when the heuristic was used.
    """
    name = Name(name_str)
    if db is not None:
        canonical = lookup_canonical_signature(db, name_str, region)
        if canonical is not None:
            signature, source = canonical
            name.find_meaning(word_db, reduce=False)
            if apply_canonical_to_name(name, signature):
                return name, source
            # Canonical recorded but signature doesn't match any cell —
            # word_db drifted. The reduce=False alternates are already
            # populated; ``Name.reduce()`` filters them down to the
            # heuristic pick (lowest unaccounted, then min-complexity)
            # without a second matcher pass.
            _logger.debug(
                "Canonical signature %s for %r missed cross-product; falling back to heuristic.",
                signature,
                name_str,
            )
            name.reduce()
            return name, None
    name.find_meaning(word_db, reduce=True)
    return name, None
