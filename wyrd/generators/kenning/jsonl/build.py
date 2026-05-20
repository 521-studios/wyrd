"""Per-source JSONL → fresh SQLite rebuild — wyrd-f295.

Reads ``data/mining/*.jsonl`` (the L2 source-of-truth) and inserts the
replayed state into a freshly-initialized SQLite lexicon. The companion
to :mod:`.dump` — together they close the round-trip:

    dump → wipe → rebuild → dump-again → semantically identical L2

Build contract
==============

Each JSONL file must contain exactly **one** ``source`` canonical-state
row whose ``ref`` is the source ID for every source-attributed row in
the file (citations, descent edges, mining-run audits, etymology
elements). The filename is hint-only; the source ID comes from the
``source`` row inside the file.

Build is a three-pass operation against a connection whose schema is
already initialized:

1. **Pass 1** — replay every JSONL file, collect:
   - one ``source`` row per file
   - merged etymon dict (union glosses/tags across files; scalar
     last-write-wins by file order)
   - merged toponym dict (same merge rules)

2. **Pass 2** — insert ``source``, ``etymon``, ``toponym`` rows and
   build ``ref → integer-id`` lookup maps for FK resolution.

3. **Pass 3** — insert source-attributed list-type rows
   (citations, descent edges, mining-run audits, etymology element
   parent + child rows). Each row's ``source_id`` comes from the file
   it lived in; FK refs to etymons / toponyms resolve through the
   Pass-2 maps.

Out of scope for this v0
========================

- Re-running wiktextract bulk ingest from ``sources/wiktextract_*.jsonl``
  (the L1 raw layer). Today the operator wraps ``lexicon rebuild`` with
  an explicit ``lexicon mine-wiktextract-corpus`` step. Phase 3 tickets
  convert that ingester to read from L1 + emit JSONL events directly.
- Re-running L3 derived enrichment passes (decompose, link-lemmas,
  normalize-ocr, etc.). Same operator-wrapping pattern.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .log import ReplayState, replay_file

_logger = logging.getLogger(__name__)


class BuildError(ValueError):
    """Raised when a JSONL file fails the build contract or a ref
    cannot be resolved during insert."""


# Columns we insert when upserting the source row. NULL-omitted columns
# stay NULL in the DB.
_SOURCE_INSERT_COLUMNS: tuple[str, ...] = (
    "id",
    "author",
    "title",
    "year",
    "region",
    "language_focus",
    "notes",
)

# L2 etymon-row columns the build pipeline writes on INSERT. Allowlist
# scope: only scalar L2-authored fields appear here. L3 enrichment
# columns (lemma_id, merged_into_id, cognate_id, stratum, english_shaped,
# phonological_vector, etc.) are deliberately absent — they're populated
# by the post-build enrichment chain, not by L2 replay. Keeping them out
# of this tuple is the load-bearing seam that prevents L2 replay from
# clobbering L3-derived state on rebuild.
_ETYMON_INSERT_COLUMNS: tuple[str, ...] = (
    "canonical_form",
    "language",
    "modifier_type",
    "position_pref",
    "notes",
    "pronunciation_ipa",
    "pronunciation_dialect",
    "original_script",
    "transliteration",
)


def _merge_etymon(target: dict[str, Any], payload: dict[str, Any]) -> None:
    """Merge etymon payload from one file into running state.

    Scalar fields: last-write-wins (later file overrides earlier).
    Array fields (glosses, tags): set-union preserving insertion order.
    """
    for key, value in payload.items():
        if key in ("glosses", "tags"):
            existing = list(target.get(key, []))
            seen = set(existing)
            for item in value:
                if item not in seen:
                    existing.append(item)
                    seen.add(item)
            target[key] = existing
        else:
            target[key] = value


def _merge_toponym(target: dict[str, Any], payload: dict[str, Any]) -> None:
    """Merge toponym payload from one file into running state.

    Toponyms only carry scalars (modern_name, country, region) — last-
    write-wins keeps it simple.
    """
    target.update(payload)


def _parse_etymon_ref(ref: str) -> tuple[str, str]:
    """Split ``"<language>:<canonical_form>"``. The canonical_form may
    contain ``:`` (rare but possible in original_script forms); we split
    on the first colon only.

    ``language`` is required (the natural key needs a language). An empty
    ``canonical_form`` is accepted — the live DB has junk rows with empty
    canonical_form that we preserve through the round-trip; cleanup
    happens in a separate prune pass."""
    if ":" not in ref:
        raise BuildError(f"etymon ref must be 'lang:form', got {ref!r}")
    lang, form = ref.split(":", 1)
    if not lang:
        raise BuildError(f"etymon ref has empty language: {ref!r}")
    return lang, form


def _upsert_source(conn: sqlite3.Connection, source_id: str, payload: dict[str, Any]) -> None:
    """Insert a source row. Conflicts are rare (each file = one
    source) but we guard with INSERT OR REPLACE for idempotency."""
    cols = ["id"] + [c for c in _SOURCE_INSERT_COLUMNS if c != "id" and c in payload]
    vals = [source_id] + [payload[c] for c in cols[1:]]
    placeholders = ", ".join("?" * len(cols))
    conn.execute(
        f"INSERT OR REPLACE INTO source ({', '.join(cols)}) VALUES ({placeholders})",
        tuple(vals),
    )


def _merge_etymon_conflict(
    conn: sqlite3.Connection,
    payload: dict[str, Any],
    cols: list[str],
) -> int:
    """Look up the existing etymon row that triggered an INSERT OR
    IGNORE conflict, UPDATE non-key scalar columns from the L2 payload,
    and return its id.

    INSERT OR IGNORE swallows constraint failures of every kind (UNIQUE,
    NOT NULL, CHECK, FK), so rowcount=0 doesn't strictly prove the cause
    was a UNIQUE conflict. But here it does in practice: canonical_form
    and language were validated non-null in the caller; etymon has no
    CHECK constraints; AUTOINCREMENT id makes PK conflict impossible
    without an explicit id (we never supply one). The only constraint
    left that can trigger rowcount=0 is the (canonical_form, language)
    UNIQUE — so the SELECT below is guaranteed to find the row. The
    None check is belt-and-braces: a future schema change that
    introduced a new constraint would fail loudly here rather than
    TypeError on the missing row.

    Fields the L2 payload doesn't touch (omitted from ``cols``) are
    left at whatever the prior write left them — typically L1's value,
    matching the "L2 explicitly overrides; L1 wins when L2 is silent"
    model.
    """
    row = conn.execute(
        "SELECT id FROM etymon WHERE canonical_form = ? AND language = ?",
        (payload["canonical_form"], payload["language"]),
    ).fetchone()
    if row is None:
        raise BuildError(
            "etymon INSERT OR IGNORE was rejected but the "
            f"(canonical_form, language) row could not be located: "
            f"{payload['canonical_form']!r} / {payload['language']!r}"
        )
    # row[0] works whether the connection has sqlite3.Row row_factory
    # set or returns plain tuples — sidesteps a hasattr check.
    eid = row[0]
    update_cols = [c for c in cols if c not in ("canonical_form", "language")]
    if update_cols:
        set_clause = ", ".join(f"{c} = ?" for c in update_cols)
        conn.execute(
            f"UPDATE etymon SET {set_clause} WHERE id = ?",
            tuple(payload[c] for c in update_cols) + (eid,),
        )
    return eid


def _insert_etymon(conn: sqlite3.Connection, payload: dict[str, Any]) -> int:
    """Insert (or merge into) an etymon row. Returns the row id.

    L2 replay walks files that may reference etymons the L1 wiktextract
    bulk ingest already wrote — e.g. wyrd-4hx7 wiktionary-empirical
    citations name canonical forms shipped by the L1 slices. On
    conflict against the ``(canonical_form, language)`` UNIQUE index,
    :func:`_merge_etymon_conflict` handles the SELECT-then-UPDATE merge.
    Glosses + tags use INSERT OR IGNORE so the merge is a set-union
    (matches ``_merge_etymon``'s in-memory semantics across multiple L2
    files; here we just extend it to handle the L1-already-wrote-this-
    row case)."""
    cols = [c for c in _ETYMON_INSERT_COLUMNS if c in payload]
    if "canonical_form" not in cols or "language" not in cols:
        raise BuildError(f"etymon row missing canonical_form or language: {payload}")
    vals = [payload[c] for c in cols]
    placeholders = ", ".join("?" * len(cols))
    cur = conn.execute(
        f"INSERT OR IGNORE INTO etymon ({', '.join(cols)}) VALUES ({placeholders})",
        tuple(vals),
    )
    if cur.rowcount > 0:
        eid = cur.lastrowid
    else:
        eid = _merge_etymon_conflict(conn, payload, cols)
    for gloss in payload.get("glosses", []):
        conn.execute(
            "INSERT OR IGNORE INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
            (eid, gloss),
        )
    for tag in payload.get("tags", []):
        conn.execute(
            "INSERT OR IGNORE INTO etymon_tag (etymon_id, tag) VALUES (?, ?)",
            (eid, tag),
        )
    return eid


def _insert_toponym(conn: sqlite3.Connection, payload: dict[str, Any]) -> int:
    """Plain INSERT (no OR IGNORE / no merge), unlike :func:`_insert_etymon`.
    Currently safe: the toponym table has no UNIQUE constraint (multiple
    villages share modern_name + region; uniqueness is enforced via the
    COALESCE-padded ``idx_toponym_unique`` index at the bridge layer,
    not as a table-level UNIQUE). If a future schema change adds a
    plain UNIQUE here, this function will need the same conflict-merge
    path _insert_etymon uses."""
    cur = conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
        (
            payload["modern_name"],
            payload.get("country"),
            payload.get("region"),
        ),
    )
    return cur.lastrowid


def _insert_citation(
    conn: sqlite3.Connection,
    etymon_id: int,
    source_id: str,
    row: dict[str, Any],
) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO etymon_citation
           (etymon_id, source_id, page, short_quote, context_snippet)
           VALUES (?, ?, ?, ?, ?)""",
        (
            etymon_id,
            source_id,
            row.get("page"),
            row.get("short_quote"),
            row.get("context_snippet"),
        ),
    )


def _insert_descent(
    conn: sqlite3.Connection,
    parent_id: int,
    child_id: int,
    source_id: str,
    row: dict[str, Any],
) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO etymon_descent
           (parent_id, child_id, edge_type, source_id, confidence, notes)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            parent_id,
            child_id,
            row["edge_type"],
            source_id,
            row.get("confidence"),
            row.get("notes"),
        ),
    )


def _insert_mining_run(conn: sqlite3.Connection, source_id: str, row: dict[str, Any]) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO mining_run
           (source_id, provider, model, mode, started_at, completed_at,
            parsed_count, accepted, declined, rejected, by_failure, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            source_id,
            row["provider"],
            row["model"],
            row["mode"],
            row.get("started_at"),
            row.get("completed_at"),
            row.get("parsed_count", 0),
            row.get("accepted", 0),
            row.get("declined", 0),
            row.get("rejected", 0),
            row.get("by_failure"),
            row.get("notes"),
        ),
    )


def _insert_attestation(
    conn: sqlite3.Connection,
    toponym_id: int,
    row: dict[str, Any],
) -> None:
    """Insert one ``toponym_attestation`` row (wyrd-3ypp).

    The schema has no ``source_id`` FK — the only source attribution
    is the free-text ``source_doc`` column. Build callers stamp a
    source-identifying prefix (e.g. ``"OS Open Names {NAMES_URI}"``)
    into ``source_doc`` so dump-time bucketing can route the rows
    back to a per-source JSONL file later.
    """
    conn.execute(
        """INSERT INTO toponym_attestation
           (toponym_id, form, date_year, source_doc)
           VALUES (?, ?, ?, ?)""",
        (
            toponym_id,
            row["form"],
            row.get("date_year"),
            row.get("source_doc"),
        ),
    )


def _insert_etymology_element(
    conn: sqlite3.Connection,
    toponym_id: int,
    source_id: str,
    row: dict[str, Any],
    etymon_id_by_ref: dict[str, int],
) -> None:
    """Insert one ``toponym_etymology`` row + its ``toponym_etymology_element``
    children. The dumped row carries the element list inline."""
    cur = conn.execute(
        """INSERT INTO toponym_etymology
           (toponym_id, source_id, page, historical_form, confidence, notes, attested_year)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            toponym_id,
            source_id,
            row.get("page"),
            row.get("historical_form"),
            row.get("confidence"),
            row.get("notes"),
            row.get("attested_year"),
        ),
    )
    etymology_id = cur.lastrowid
    for el in row.get("elements", []):
        ref = el.get("etymon_ref")
        eid = etymon_id_by_ref.get(ref)
        if eid is None:
            raise BuildError(
                f"etymology_element references unknown etymon ref {ref!r} "
                f"(toponym_etymology source={source_id})"
            )
        conn.execute(
            """INSERT INTO toponym_etymology_element
               (toponym_etymology_id, ordinal, etymon_id, inflection, surface_in_modern)
               VALUES (?, ?, ?, ?, ?)""",
            (
                etymology_id,
                el["ordinal"],
                eid,
                el.get("inflection"),
                el.get("surface_in_modern"),
            ),
        )


# ---------------------------------------------------------------------------
# Pass-3 per-row-type helpers
#
# Each helper consumes one fact-row list from a single source's replay
# state, mutates the shared ``counts`` dict in place, and handles its
# own orphan-skip + dedup conventions. Symmetric signature shape across
# helpers keeps the build pipeline easy to extend with new row types.
#
# Orphan-skip rationale (wyrd-lene contract): unresolved refs count
# into ``*_orphans`` + sampled ``*_orphan_refs`` and skip the insert
# rather than raising. Operators get telemetry so unexpected orphans
# (typos, stale refs) are still visible — they just don't abort the
# build. This makes ``_op: remove`` events on toponyms / etymons safe
# to apply without manually cleaning every referencing fact-row first.
# ---------------------------------------------------------------------------


def _insert_citation_rows(
    conn: sqlite3.Connection,
    source_id: str,
    rows: list[dict[str, Any]],
    etymon_id_by_ref: dict[str, int],
    counts: dict[str, Any],
) -> None:
    """Insert ``citation`` rows for one source. Unknown ``etymon_ref``
    → counted in ``citation_orphans`` and skipped (wyrd-lene)."""
    for row in rows:
        eref = row["etymon_ref"]
        eid = etymon_id_by_ref.get(eref)
        if eid is None:
            counts["citation_orphans"] += 1
            if len(counts["citation_orphan_refs"]) < ORPHAN_SAMPLE_LIMIT:
                counts["citation_orphan_refs"].append(eref)
            continue
        _insert_citation(conn, eid, source_id, row)
        counts["citation"] += 1


def _insert_descent_rows(
    conn: sqlite3.Connection,
    source_id: str,
    rows: list[dict[str, Any]],
    etymon_id_by_ref: dict[str, int],
    counts: dict[str, Any],
) -> None:
    """Insert ``etymon_descent`` rows for one source. Either endpoint
    unresolved → counted in ``etymon_descent_orphans`` + skipped."""
    for row in rows:
        pid = etymon_id_by_ref.get(row["parent_ref"])
        cid = etymon_id_by_ref.get(row["child_ref"])
        if pid is None or cid is None:
            counts["etymon_descent_orphans"] += 1
            if len(counts["etymon_descent_orphan_refs"]) < ORPHAN_SAMPLE_LIMIT:
                counts["etymon_descent_orphan_refs"].append(
                    f"{row['parent_ref']} → {row['child_ref']}"
                )
            continue
        _insert_descent(conn, pid, cid, source_id, row)
        counts["etymon_descent"] += 1


def _insert_mining_run_rows(
    conn: sqlite3.Connection,
    source_id: str,
    rows: list[dict[str, Any]],
    counts: dict[str, Any],
) -> None:
    """Insert ``mining_run`` audit rows for one source. No orphan path
    (mining_run has no FK refs — every row is self-contained)."""
    for row in rows:
        _insert_mining_run(conn, source_id, row)
        counts["mining_run"] += 1


def _insert_etymology_element_rows(
    conn: sqlite3.Connection,
    source_id: str,
    rows: list[dict[str, Any]],
    toponym_id_by_ref: dict[str, int],
    etymon_id_by_ref: dict[str, int],
    counts: dict[str, Any],
) -> None:
    """Insert ``etymology_element`` rows for one source. Two skip
    conditions:

    - Unknown ``toponym_ref`` → ``etymology_element_orphans`` (wyrd-lene)
    - Byte-identical content fingerprint already seen in this source →
      ``etymology_element_dupes_skipped`` (wyrd-tzf2). Dedup is per-
      source: same content across sources is scholar agreement.

    Element-list etymon_refs are still validated inside
    :func:`_insert_etymology_element` and raise ``BuildError`` on
    missing — coarse orphan check at the toponym level only.
    """
    seen_fingerprints: set[tuple] = set()
    for row in rows:
        tref = row["toponym_ref"]
        tid = toponym_id_by_ref.get(tref)
        if tid is None:
            counts["etymology_element_orphans"] += 1
            if len(counts["etymology_element_orphan_refs"]) < ORPHAN_SAMPLE_LIMIT:
                counts["etymology_element_orphan_refs"].append(tref)
            continue
        fingerprint = _etymology_element_fingerprint(row)
        if fingerprint in seen_fingerprints:
            counts["etymology_element_dupes_skipped"] += 1
            continue
        seen_fingerprints.add(fingerprint)
        _insert_etymology_element(conn, tid, source_id, row, etymon_id_by_ref)
        counts["etymology_element"] += 1


def _insert_attestation_rows(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    toponym_id_by_ref: dict[str, int],
    counts: dict[str, Any],
) -> None:
    """Insert ``toponym_attestation`` rows for one source (wyrd-3ypp).
    Unknown ``toponym_ref`` → counted in ``attestation_orphans`` +
    skipped, same orphan pattern as the other fact-row helpers.

    No ``source_id`` parameter — the schema has no source FK; row's
    ``source_doc`` carries the (free-text) attribution. The build's
    per-file dispatch loop already scopes per source, but the source
    information lives in the row itself.
    """
    for row in rows:
        tref = row["toponym_ref"]
        tid = toponym_id_by_ref.get(tref)
        if tid is None:
            counts["attestation_orphans"] += 1
            if len(counts["attestation_orphan_refs"]) < ORPHAN_SAMPLE_LIMIT:
                counts["attestation_orphan_refs"].append(tref)
            continue
        _insert_attestation(conn, tid, row)
        counts["attestation"] += 1


# wyrd-2thc: fantasy_morpheme scalar columns inserted as-is from the
# JSONL row. ``etymon_id`` is resolved separately from ``etymon_ref``
# below; the schema's autoincrement ``id`` is allocated by SQLite.
_FANTASY_MORPHEME_INSERT_COLUMNS: tuple[str, ...] = (
    "input_name",
    "input_description",
    "usable",
    "bar_reason",
    "resolution_method",
    "approach_version",
    "confidence",
    "citation",
    "reasoning",
    "unapproved_language",
    "unapproved_form",
    "processed_at",
)


# Columns the schema declares NOT NULL on ``fantasy_morpheme``. The
# insert helper validates these explicitly before issuing the
# ``INSERT OR IGNORE`` — see ``_insert_fantasy_morpheme``.
_FANTASY_MORPHEME_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"input_name", "usable", "resolution_method", "approach_version", "processed_at"}
)


def _insert_fantasy_morpheme(
    conn: sqlite3.Connection,
    etymon_id: int | None,
    row: dict[str, Any],
) -> bool:
    """Insert one ``fantasy_morpheme`` row (wyrd-2thc).

    Returns ``True`` if a new row was inserted, ``False`` if INSERT OR
    IGNORE skipped a duplicate ``(input_name, approach_version)``
    UNIQUE conflict — idempotent re-replay against a populated DB.

    ``etymon_id`` may be None — barred rows (``usable=0``) often have
    no resolved etymon. NOT NULL columns (``input_name``, ``usable``,
    ``resolution_method``, ``approach_version``, ``processed_at``) are
    pre-validated here because ``INSERT OR IGNORE`` would otherwise
    swallow the IntegrityError and the caller would count a phantom
    insert. Missing required fields raise ``BuildError`` with the
    field name so operators get an actionable diagnostic instead of a
    silent count discrepancy.
    """
    missing = [c for c in _FANTASY_MORPHEME_REQUIRED_COLUMNS if row.get(c) is None]
    if missing:
        raise BuildError(
            f"fantasy_morpheme row missing required NOT NULL field(s) {sorted(missing)}: "
            f"input_name={row.get('input_name')!r} "
            f"approach_version={row.get('approach_version')!r}"
        )
    columns = ("etymon_id", *_FANTASY_MORPHEME_INSERT_COLUMNS)
    placeholders = ", ".join("?" * len(columns))
    values = (etymon_id, *(row.get(c) for c in _FANTASY_MORPHEME_INSERT_COLUMNS))
    cur = conn.execute(
        f"INSERT OR IGNORE INTO fantasy_morpheme ({', '.join(columns)}) "  # noqa: S608
        f"VALUES ({placeholders})",
        values,
    )
    return cur.rowcount == 1


def _insert_fantasy_morpheme_rows(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    etymon_id_by_ref: dict[str, int],
    counts: dict[str, Any],
) -> None:
    """Insert ``fantasy_morpheme`` rows (wyrd-2thc). Orphan = a row
    whose ``etymon_ref`` doesn't resolve in ``etymon_id_by_ref``.
    Orphaned rows are skipped (counted in
    ``fantasy_morpheme_orphans``) rather than inserted with NULL
    etymon_id, because the dump emits ``etymon_ref`` only when the
    DB had a real etymon — a missing ref on rebuild means the cited
    etymon was deliberately removed via a kernel ``remove`` event,
    and silently nulling the FK would lose the operator's intent.

    Rows that legitimately have no ``etymon_ref`` field (barred
    morphemes whose etymon_id was NULL at dump time) insert with
    etymon_id=NULL and are counted as normal ``fantasy_morpheme``
    rows — not orphans.
    """
    for row in rows:
        eref = row.get("etymon_ref")
        if eref is None:
            # Originally NULL etymon_id — barred row, valid state.
            if _insert_fantasy_morpheme(conn, None, row):
                counts["fantasy_morpheme"] += 1
            continue
        eid = etymon_id_by_ref.get(eref)
        if eid is None:
            counts["fantasy_morpheme_orphans"] += 1
            if len(counts["fantasy_morpheme_orphan_refs"]) < ORPHAN_SAMPLE_LIMIT:
                counts["fantasy_morpheme_orphan_refs"].append(eref)
            continue
        if _insert_fantasy_morpheme(conn, eid, row):
            counts["fantasy_morpheme"] += 1


# wyrd-11zh: Briggs personal-name insert columns. The ``id`` column
# is omitted — SQLite allocates it via AUTOINCREMENT, and
# ``_insert_personal_name`` recovers the value via the post-insert
# lookup against the UNIQUE(headform, source_doc) index. All other
# columns map 1:1 from the JSONL payload.
_PERSONAL_NAME_INSERT_COLUMNS: tuple[str, ...] = (
    "headform",
    "normalized_form",
    "language_hints",
    "is_feminine",
    "pase_count",
    "has_dlv",
    "ascharter_refs",
    "source_doc",
    "raw_entry",
)


def _insert_personal_name(conn: sqlite3.Connection, payload: dict[str, Any]) -> int | None:
    """INSERT OR IGNORE one ``personal_name`` row (wyrd-11zh). Returns
    the row id on success — either freshly-allocated (cur.lastrowid) or
    looked up via the UNIQUE(headform, source_doc) index on conflict.

    Returns None if the payload is malformed (missing headform or
    source_doc); the caller's orphan counter handles the gap.
    """
    headform = payload.get("headform")
    source_doc = payload.get("source_doc")
    if not headform or not source_doc:
        return None
    cols = [c for c in _PERSONAL_NAME_INSERT_COLUMNS if c in payload]
    vals = [payload[c] for c in cols]
    placeholders = ", ".join("?" * len(cols))
    cur = conn.execute(
        f"INSERT OR IGNORE INTO personal_name ({', '.join(cols)}) VALUES ({placeholders})",
        tuple(vals),
    )
    if cur.rowcount > 0:
        return int(cur.lastrowid)
    # INSERT OR IGNORE hit the UNIQUE(headform, source_doc); look up.
    row = conn.execute(
        "SELECT id FROM personal_name WHERE headform = ? AND source_doc = ?",
        (headform, source_doc),
    ).fetchone()
    return int(row[0]) if row is not None else None


def _insert_personal_name_toponym_attestation(
    conn: sqlite3.Connection,
    personal_name_id: int,
    row: dict[str, Any],
) -> None:
    """INSERT OR IGNORE one ``personal_name_toponym_attestation`` row
    (wyrd-11zh). The COALESCE-padded UNIQUE index absorbs duplicates so
    re-running rebuild on an already-populated DB is a no-op."""
    conn.execute(
        """INSERT OR IGNORE INTO personal_name_toponym_attestation
           (personal_name_id, toponym_form, attested_variant, county_code,
            county_canonical, date_qualifier, is_uncertain, is_serious_doubt,
            source_doc, raw_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            personal_name_id,
            row["toponym_form"],
            row.get("attested_variant"),
            row["county_code"],
            row.get("county_canonical", ""),
            row.get("date_qualifier"),
            1 if row.get("is_uncertain") else 0,
            1 if row.get("is_serious_doubt") else 0,
            row.get("source_doc", ""),
            row.get("raw_text"),
        ),
    )


def _insert_personal_name_rows(
    conn: sqlite3.Connection,
    rows: dict[str, dict[str, Any]],
    counts: dict[str, Any],
) -> dict[str, int]:
    """Insert every ``personal_name`` row from the replayed keyed-state.
    Returns headform → id lookup map for the attestation pass."""
    pn_id_by_headform: dict[str, int] = {}
    for headform, payload in rows.items():
        # The ref IS the headform; carry it into the payload so the
        # insert helper has the natural key without depending on the
        # caller to set it.
        merged = dict(payload)
        merged.setdefault("headform", headform)
        pn_id = _insert_personal_name(conn, merged)
        if pn_id is None:
            counts["personal_name_orphans"] += 1
            if len(counts["personal_name_orphan_refs"]) < ORPHAN_SAMPLE_LIMIT:
                counts["personal_name_orphan_refs"].append(headform)
            continue
        pn_id_by_headform[headform] = pn_id
        counts["personal_name"] += 1
    return pn_id_by_headform


def _insert_personal_name_attestation_rows(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    pn_id_by_headform: dict[str, int],
    counts: dict[str, Any],
) -> None:
    """Insert ``personal_name_toponym_attestation`` rows for one file
    (wyrd-11zh). Unknown ``personal_name_ref`` → counted in
    ``personal_name_attestation_orphans`` + skipped, same orphan
    pattern as the other fact-row helpers."""
    for row in rows:
        pn_ref = row.get("personal_name_ref")
        if pn_ref is None:
            counts["personal_name_attestation_orphans"] += 1
            if len(counts["personal_name_attestation_orphan_refs"]) < ORPHAN_SAMPLE_LIMIT:
                counts["personal_name_attestation_orphan_refs"].append("(missing ref)")
            continue
        pn_id = pn_id_by_headform.get(pn_ref)
        if pn_id is None:
            counts["personal_name_attestation_orphans"] += 1
            if len(counts["personal_name_attestation_orphan_refs"]) < ORPHAN_SAMPLE_LIMIT:
                counts["personal_name_attestation_orphan_refs"].append(pn_ref)
            continue
        _insert_personal_name_toponym_attestation(conn, pn_id, row)
        counts["personal_name_toponym_attestation"] += 1


def _extract_source_id(path: Path, state: ReplayState) -> str:
    """Validate the contract: exactly one ``source`` row per file.
    Returns its ref (= the source_id)."""
    sources = state.keyed["source"]
    if not sources:
        raise BuildError(f"{path}: no source row found; every L2 file must contain one")
    if len(sources) > 1:
        raise BuildError(
            f"{path}: found {len(sources)} source rows; each L2 file must contain exactly one"
        )
    return next(iter(sources))


def build_from_jsonl(
    conn: sqlite3.Connection,
    jsonl_paths: Iterable[Path],
) -> dict[str, int]:
    """Insert L2 JSONL rows into ``conn`` (schema already initialized).

    Returns a counter dict for telemetry:
    ``{table: rows_inserted}``.
    """
    paths_sorted: list[Path] = sorted(Path(p) for p in jsonl_paths)
    counts: dict[str, Any] = {
        "source": 0,
        "etymon": 0,
        "toponym": 0,
        "citation": 0,
        "etymon_descent": 0,
        "mining_run": 0,
        "etymology_element": 0,
        "etymology_element_dupes_skipped": 0,
        # Orphan counts (wyrd-lene): when a fact-row references an
        # entity that was removed via a kernel ``remove`` event, the
        # row is skipped rather than raising. Operators get telemetry
        # so unexpected orphans are still visible — the lists capture
        # the specific orphan refs (capped) so operators can tell
        # "expected post-remove orphans" from "unexpected typos in
        # mined data" without a separate diff pass. Lists are capped
        # at ORPHAN_SAMPLE_LIMIT entries each to keep the result dict
        # bounded on large prunes.
        "citation_orphans": 0,
        "citation_orphan_refs": [],
        "etymon_descent_orphans": 0,
        "etymon_descent_orphan_refs": [],
        "etymology_element_orphans": 0,
        "etymology_element_orphan_refs": [],
        "attestation": 0,
        "attestation_orphans": 0,
        "attestation_orphan_refs": [],
        # wyrd-2thc: fantasy_morpheme round-trip. Orphan = a row whose
        # etymon_ref doesn't resolve in the rebuilt etymon table (the
        # cited etymon was removed via a kernel ``remove`` event, or
        # the JSONL claims an etymon that was never dumped). Rows with
        # no ``etymon_ref`` field at all (barred morphemes with NULL
        # etymon_id) are NOT orphans — they insert with etymon_id=NULL.
        "fantasy_morpheme": 0,
        "fantasy_morpheme_orphans": 0,
        "fantasy_morpheme_orphan_refs": [],
        # wyrd-11zh: Briggs personal-names ingest round-trip.
        # ``personal_name_orphans`` counts dropped PN rows (missing
        # headform / source_doc); ``personal_name_attestation_orphans``
        # counts attestation rows whose ``personal_name_ref`` doesn't
        # resolve in the rebuilt PN table.
        "personal_name": 0,
        "personal_name_orphans": 0,
        "personal_name_orphan_refs": [],
        "personal_name_toponym_attestation": 0,
        "personal_name_attestation_orphans": 0,
        "personal_name_attestation_orphan_refs": [],
    }

    # ----- Pass 1: replay every file, accumulate merged state.
    file_states: list[tuple[Path, str, ReplayState]] = []
    merged_etymons: dict[str, dict[str, Any]] = {}
    merged_toponyms: dict[str, dict[str, Any]] = {}

    for path in paths_sorted:
        state = replay_file(path)
        source_id = _extract_source_id(path, state)
        file_states.append((path, source_id, state))
        for ref, payload in state.keyed["etymon"].items():
            _merge_etymon(merged_etymons.setdefault(ref, {}), payload)
        for ref, payload in state.keyed["toponym"].items():
            _merge_toponym(merged_toponyms.setdefault(ref, {}), payload)

    # ----- Pass 2: insert sources, etymons, toponyms; build FK maps.
    for _path, source_id, state in file_states:
        _upsert_source(conn, source_id, state.keyed["source"][source_id])
        counts["source"] += 1

    etymon_id_by_ref: dict[str, int] = {}
    for ref, payload in merged_etymons.items():
        _parse_etymon_ref(ref)  # validate shape; result unused
        eid = _insert_etymon(conn, payload)
        etymon_id_by_ref[ref] = eid
        counts["etymon"] += 1

    toponym_id_by_ref: dict[str, int] = {}
    for ref, payload in merged_toponyms.items():
        tid = _insert_toponym(conn, payload)
        toponym_id_by_ref[ref] = tid
        counts["toponym"] += 1

    # ----- Pass 3: source-attributed list rows.
    #
    # Per-row-type helpers carry the orphan-skip + dedup semantics; this
    # loop is the per-source dispatch. personal_name rows are inserted
    # per-file BEFORE the attestation rows so the FK lookup map is
    # populated in scope.
    for _path, source_id, state in file_states:
        _insert_citation_rows(conn, source_id, state.lists["citation"], etymon_id_by_ref, counts)
        _insert_descent_rows(
            conn, source_id, state.lists["etymon_descent"], etymon_id_by_ref, counts
        )
        _insert_mining_run_rows(conn, source_id, state.lists["mining_run"], counts)
        _insert_etymology_element_rows(
            conn,
            source_id,
            state.lists["etymology_element"],
            toponym_id_by_ref,
            etymon_id_by_ref,
            counts,
        )
        _insert_attestation_rows(conn, state.lists["attestation"], toponym_id_by_ref, counts)
        _insert_fantasy_morpheme_rows(
            conn, state.lists["fantasy_morpheme"], etymon_id_by_ref, counts
        )
        # wyrd-11zh: Briggs personal-names. Per-file scope (FK lookup
        # map is per-source since (headform, source_doc) is the
        # natural key); attestation rows reference PNs by headform.
        pn_id_by_headform = _insert_personal_name_rows(
            conn, state.keyed.get("personal_name", {}), counts
        )
        _insert_personal_name_attestation_rows(
            conn,
            state.lists.get("personal_name_toponym_attestation", []),
            pn_id_by_headform,
            counts,
        )

    conn.commit()
    return counts


# Cap on how many orphan refs the build retains in its result dict.
# Counts (citation_orphans, etc.) are unbounded; the ref-list samples
# stop at this limit to keep the dict size predictable on bulk prunes.
ORPHAN_SAMPLE_LIMIT = 50


def _etymology_element_fingerprint(row: dict[str, Any]) -> tuple:
    """Stable content-fingerprint for an ``etymology_element`` row
    (wyrd-tzf2). Two rows with the same fingerprint are content-
    duplicates that an idempotent INSERT should skip.

    Includes every field that influences the inserted
    ``toponym_etymology`` + ``toponym_etymology_element`` rows:
    toponym ref, ordered element list (ordinal + etymon_ref +
    inflection + surface_in_modern per element), historical_form,
    confidence, notes, attested_year, page.

    Lists become tuples so the result is hashable. Element ordering
    is preserved as-is (ordinal IS the order; sorting would change
    semantics for any badly-formed input).
    """
    elements_key = tuple(
        (
            el.get("ordinal"),
            el.get("etymon_ref"),
            el.get("inflection"),
            el.get("surface_in_modern"),
        )
        # `or []` handles an explicit ``"elements": null`` in JSONL —
        # ``get("elements", [])`` would still return None in that case.
        for el in (row.get("elements") or [])
    )
    return (
        row.get("toponym_ref"),
        elements_key,
        row.get("historical_form"),
        row.get("confidence"),
        row.get("notes"),
        row.get("attested_year"),
        row.get("page"),
    )


def jsonl_paths_in(directory: str | Path) -> list[Path]:
    """All ``*.jsonl`` files in ``directory``, sorted ASCII. Used as the
    default input set for :func:`build_from_jsonl`."""
    p = Path(directory)
    return sorted(p.glob("*.jsonl"))


# Tables the diff-rebuild check compares (wyrd-f4nl). Scoped to the L2
# rows the build pass inserts; L3-derived tables (reflex,
# toponym_decomposition, etymon_meaning_synset, etc.) are expected to
# be empty in a fresh rebuild without enrichment passes, so comparing
# them would flag everything as a delta and drown out real regressions.
DIFF_REBUILD_TABLES: tuple[str, ...] = (
    "source",
    "etymon",
    "etymon_gloss",
    "etymon_tag",
    "etymon_citation",
    "etymon_descent",
    "mining_run",
    "toponym",
    "toponym_etymology",
    "toponym_etymology_element",
    # wyrd-11zh: Briggs PN ingest round-trip.
    "personal_name",
    "personal_name_toponym_attestation",
)


def table_counts(
    conn: sqlite3.Connection, table_names: Iterable[str] = DIFF_REBUILD_TABLES
) -> dict[str, int]:
    """Return ``{table: row_count}`` for each named table. Missing
    tables are reported as ``-1`` (distinct from ``0`` = present but
    empty) so a schema-mismatch case shows up clearly."""
    counts: dict[str, int] = {}
    for table in table_names:
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = row[0]
        except sqlite3.OperationalError as exc:
            # Logged so a "database is locked" or schema-mismatch case
            # in CI surfaces a diagnostic instead of silently flipping
            # has_any_delta True. counts[table] = -1 is still the wire
            # signal to downstream consumers.
            _logger.warning("table_counts: cannot read %s: %s", table, exc)
            counts[table] = -1
    return counts


def diff_table_counts(before: dict[str, int], after: dict[str, int]) -> list[dict[str, Any]]:
    """Side-by-side count comparison. Returns one row per table in the
    union of both sides, sorted by table name. Each row is
    ``{table, before, after, delta}``.

    Delta semantics:
    - Both sides ≥ 0: ``after - before``. Zero means clean round-trip;
      positive means rebuild has MORE rows (suspicious); negative
      means rebuild has FEWER (typical regression).
    - Either side is ``-1`` (table missing/inaccessible per
      :func:`table_counts`): delta is ``None``. Arithmetic mixing
      missing-sentinel and real counts would be misleading; the
      formatter renders ``None`` as an explicit marker.
    """
    tables = sorted(set(before) | set(after))
    rows: list[dict[str, Any]] = []
    for t in tables:
        b = before.get(t, -1)
        a = after.get(t, -1)
        delta: int | None
        if b < 0 or a < 0:
            delta = None
        else:
            delta = a - b
        rows.append({"table": t, "before": b, "after": a, "delta": delta})
    return rows


def format_diff_rebuild(rows: list[dict[str, Any]]) -> str:
    """Render :func:`diff_table_counts` output as a fixed-width
    markdown-friendly table. Number columns sized for 9 digits so
    wiktionary-scale rows (~2M+) fit without wrapping. ``∆`` column
    is signed for easy grep (``grep '∆.*-' ...``); ``None`` delta
    (one side missing) renders as ``—`` so it can't be mistaken for
    zero; ``-1`` count renders as ``(missing)`` for the same reason."""
    lines = [
        "| Table                           |    Before |     After |         ∆ |",
        "|---------------------------------|-----------|-----------|-----------|",
    ]
    for r in rows:
        before_str = "(missing)" if r["before"] < 0 else f"{r['before']:>9}"
        after_str = "(missing)" if r["after"] < 0 else f"{r['after']:>9}"
        if r["delta"] is None:
            delta_str = "        —"
        elif r["delta"] == 0:
            delta_str = f"{0:>9}"
        else:
            # ``+`` flag forces sign on positives; negatives carry
            # their own. Keeps sign + magnitude contiguous (e.g.
            # ``-2367667`` not ``- 2367667``) so grep '∆.*-' works.
            delta_str = f"{r['delta']:>+9d}"
        lines.append(f"| {r['table']:<31} | {before_str:>9} | {after_str:>9} | {delta_str:>9} |")
    return "\n".join(lines)


def has_any_delta(rows: list[dict[str, Any]]) -> bool:
    """True when at least one table's row count changed across the
    rebuild. Drives the CLI exit code so CI can gate regressions.
    ``None`` delta (table missing on one side) counts as a delta too —
    schema mismatch shouldn't pass silently."""
    return any(r["delta"] != 0 for r in rows)


def collect_curation_overrides(jsonl_paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Scan ``jsonl_paths`` for ``etymon_curation`` rows and return the
    merged curation state (wyrd-2jhs).

    Each row is keyed on its etymon ref (``"<language>:<canonical_form>"``);
    cross-file patches accumulate via the kernel's replay semantics — the
    latest event wins for each scalar field. ``lemma_ref`` /
    ``merged_into_ref`` / ``inflection`` set when present; explicit
    ``null`` clears the override (revert to auto-clustering output).

    Curation rows can live in any file — in practice they cluster in
    ``data/mining/_curation.jsonl``, but the function doesn't require
    that convention. Files containing only mining facts contribute
    nothing to the curation state.

    Build-time integration: this runs SEPARATELY from
    :func:`build_from_jsonl`. The L2-fact insert (citations, etymologies,
    descent) shouldn't be entangled with the L3-override application;
    curation gets applied by ``enrichment.apply_curation_overrides``
    AFTER the auto-clustering passes have written their initial output.
    """
    from .log import replay_file

    merged: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(p) for p in jsonl_paths):
        state = replay_file(path)
        for ref, payload in state.keyed["etymon_curation"].items():
            target = merged.setdefault(ref, {})
            # Scalar fields: last-write-wins. Explicit None preserved
            # (means "clear this override") — distinguished from "field
            # not present" by direct key check.
            target.update(payload)
    return merged
