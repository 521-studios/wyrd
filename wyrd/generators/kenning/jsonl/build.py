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

from ..lexicon.morpheme_surface import normalize_morpheme_surface
from ._l2_columns import _ETYMON_L2_COLUMNS, _SOURCE_COLUMNS
from .log import ReplayState, replay_file

_logger = logging.getLogger(__name__)


class BuildError(ValueError):
    """Raised when a JSONL file fails the build contract or a ref
    cannot be resolved during insert."""


# Columns we insert when upserting the source row, derived from the single-sourced
# L2 schema (_l2_columns) + the synthetic row ``id`` the inserter supplies. NULL-
# omitted columns stay NULL in the DB. Deriving (rather than re-listing) makes a
# dump/insert column drift — and the silent round-trip data loss it causes —
# impossible by construction.
_SOURCE_INSERT_COLUMNS: tuple[str, ...] = ("id", *_SOURCE_COLUMNS)

# L2 etymon-row columns the build pipeline writes on INSERT — exactly the dumped L2
# columns (_ETYMON_L2_COLUMNS), so dump and build can't drift apart. Allowlist
# scope: only scalar L2-authored fields appear here. L3 enrichment columns
# (lemma_id, merged_into_id, cognate_id, stratum, english_shaped,
# phonological_vector, etc.) are deliberately absent from the L2 list — they're
# populated by the post-build enrichment chain, not by L2 replay. Keeping them out
# is the load-bearing seam that prevents L2 replay from clobbering L3-derived state
# on rebuild.
_ETYMON_INSERT_COLUMNS: tuple[str, ...] = _ETYMON_L2_COLUMNS


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
    happens in a separate prune pass.

    NB the form is returned VERBATIM (not de-dashed). The de-dash of
    ``etymon.canonical_form`` (wyrd-aicu.8, D45) lives in :func:`_insert_etymon`,
    which writes the bare surface to the row; ref *resolution* keys
    ``etymon_id_by_ref`` on the raw ref string on both sides (the etymon's own
    ref and the fact rows that point at it), so it matches raw-to-raw and needs
    no de-dash here. Dash-duplicate etymon rows collapse onto one row via the
    ``UNIQUE(canonical_form, language)`` conflict in ``_insert_etymon`` once both
    de-dash to the same bare form."""
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
    # De-dash to the bare morpheme surface (wyrd-aicu.8, D45) on the payload so
    # both the INSERT below and the _merge_etymon_conflict SELECT (which reads
    # the same payload) key on the bare form. This is the load-bearing build-side
    # de-dash: two dash-duplicate L2 rows (celtic:-ach and celtic:ach) now both
    # write canonical_form='ach', so the second INSERT hits the
    # UNIQUE(canonical_form, language) conflict and _merge_etymon_conflict folds
    # them onto one row. An empty-after-de-dash junk form is preserved as the
    # existing empty-form sentinel (the prune pass owns its removal).
    payload["canonical_form"] = normalize_morpheme_surface(payload["canonical_form"]) or ""
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
    """INSERT, or REUSE the existing row on conflict. Uniqueness is
    enforced by the COALESCE-padded ``idx_toponym_unique`` index on
    ``(modern_name, COALESCE(country,''), COALESCE(region,''))``. An
    additive replay whose toponym already exists (e.g. KEPN ingesting
    Bedford/Bedfordshire when a scholarly source already mined it) must
    reuse the existing row's id so the colliding place merges its
    decomposition onto the existing toponym instead of aborting the
    whole build with an IntegrityError. Mirrors :func:`_insert_etymon`'s
    INSERT-OR-IGNORE-then-lookup contract (wyrd-bo01.1)."""
    # Validate up front (mirrors _insert_etymon's guard): modern_name is
    # NOT NULL in the schema, and INSERT OR IGNORE below would otherwise
    # SWALLOW a NOT NULL violation — leaving rowcount=0, a lookup that
    # matches nothing, and a stale ``lastrowid`` from the prior insert.
    modern_name = payload.get("modern_name")
    if not modern_name:
        raise BuildError(f"toponym row missing modern_name: {payload}")
    country = payload.get("country")
    region = payload.get("region")
    cur = conn.execute(
        "INSERT OR IGNORE INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
        (modern_name, country, region),
    )
    if cur.rowcount > 0:
        return int(cur.lastrowid)
    # INSERT OR IGNORE hit idx_toponym_unique; reuse the existing row.
    # COALESCE padding matches the index so NULL country/region collide.
    row = conn.execute(
        "SELECT id FROM toponym "
        "WHERE modern_name = ? "
        "AND COALESCE(country, '') = COALESCE(?, '') "
        "AND COALESCE(region, '') = COALESCE(?, '')",
        (modern_name, country, region),
    ).fetchone()
    if row is None:
        # With modern_name validated, the only rowcount==0 cause is the
        # genuine unique conflict — which the lookup MUST find. None here
        # means schema/logic drift; fail loud like _merge_etymon_conflict.
        raise BuildError(
            "toponym INSERT OR IGNORE was rejected but no conflicting row "
            f"could be located: modern_name={modern_name!r} "
            f"country={country!r} region={region!r}"
        )
    return int(row[0])


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
            # surface_in_modern is NOT restored from L2 — it is L3-derived by
            # enrichment.derive_surface_in_modern (wyrd-ujyo) on every rebuild,
            # so it's left NULL here (like lemma_id / merged_into_id).
            """INSERT INTO toponym_etymology_element
               (toponym_etymology_id, ordinal, etymon_id, inflection, confidence)
               VALUES (?, ?, ?, ?, ?)""",
            (
                etymology_id,
                el["ordinal"],
                eid,
                el.get("inflection"),
                # wyrd-2n1: per-element confidence flows through the JSONL
                # alongside inflection. Old JSONL files (pre-2n1) won't carry
                # the field; .get() returns None which the schema accepts as
                # "no rating".
                el.get("confidence"),
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
    known_source_ids: set[str],
    counts: dict[str, Any],
) -> None:
    """Insert ``citation`` rows. Default attribution is the file's
    ``source_id``; a row may override it with an explicit ``source_id``
    naming a ``cited_source`` the file registered (wyrd-2b50 — lets one
    file record attestations from sources it cites on behalf of, without
    a separate JSONL per source). Unknown ``etymon_ref`` → counted in
    ``citation_orphans`` and skipped (wyrd-lene). An override ``source_id``
    that no file registered → counted in ``citation_source_orphans`` and
    skipped, rather than silently swallowed by the FK + INSERT OR IGNORE
    (wyrd-2b50)."""
    for row in rows:
        eref = row["etymon_ref"]
        eid = etymon_id_by_ref.get(eref)
        if eid is None:
            counts["citation_orphans"] += 1
            if len(counts["citation_orphan_refs"]) < ORPHAN_SAMPLE_LIMIT:
                counts["citation_orphan_refs"].append(eref)
            continue
        # Resolve attribution: an explicit per-row source_id (wyrd-2b50)
        # else the file's primary source. ``.get(k, default)`` keys on
        # key-ABSENCE, not falsiness (unlike ``or``), so a present-but-
        # empty source_id is kept verbatim and then surfaces as an orphan
        # below rather than silently falling back to the file source.
        cite_source = row.get("source_id", source_id)
        if cite_source not in known_source_ids:
            counts["citation_source_orphans"] += 1
            if len(counts["citation_source_orphan_refs"]) < ORPHAN_SAMPLE_LIMIT:
                counts["citation_source_orphan_refs"].append(f"{eref} → {cite_source!r}")
            continue
        _insert_citation(conn, eid, cite_source, row)
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


def _insert_reflex_rows(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    etymon_id_by_ref: dict[str, int],
    counts: dict[str, Any],
) -> None:
    """Replay ``reflex`` rows (wyrd-ned5): upsert the (surface_form,
    position) reflex, then link it to every etymon named in
    ``etymon_refs`` that resolves in ``etymon_id_by_ref``.

    Reflexes are the modern_usage→etymon layer (e.g. -ton → OE tūn).
    The SEED reflexes (rando-port modern_usage) are created by
    ``seed_from_meanings`` and are NOT carried anywhere else in L2, so a
    full rebuild dropped them — the bug this round-trip fixes. (The
    place-name-derived reflexes that ``derive_positions`` writes during
    ``mine-wiktextract-corpus`` ARE re-derivable from the corpus, but
    round-tripping the whole layer here is simpler and idempotent.)
    ``productivity`` is intentionally NOT carried in L2 — it's recomputed
    from corpus observations by ``_recompute_reflex_productivity_after_build``
    at the rebuild tail, matching the existing wyrd-14p contract.

    Orphan = an ``etymon_ref`` that doesn't resolve (the linked etymon
    was removed via a kernel ``remove`` event, or — common on an
    L2-only rebuild — is a bulk/wiktextract etymon not present without
    ``--fetch-bulk``). Orphaned refs are skipped + counted; a reflex
    whose refs ALL orphan still upserts the reflex row (an empty link
    set is a valid intermediate state — a later mining pass may add the
    etymon).
    """
    from ..lexicon.sql.queries.reflex import (  # lazy: keep build import-time light (cold-start)
        LINK_REFLEX_ETYMON_OR_IGNORE,
        UPSERT_REFLEX,
    )

    for row in rows:
        surface_form = row.get("surface_form")
        position = row.get("position")
        if not surface_form or not position:
            raise BuildError(f"reflex row missing surface_form/position: {row!r}")
        reflex_id = conn.execute(UPSERT_REFLEX, (surface_form, position)).fetchone()[0]
        counts["reflex"] += 1
        for eref in row.get("etymon_refs", []) or []:
            eid = etymon_id_by_ref.get(eref)
            if eid is None:
                counts["reflex_link_orphans"] += 1
                if len(counts["reflex_link_orphan_refs"]) < ORPHAN_SAMPLE_LIMIT:
                    counts["reflex_link_orphan_refs"].append(eref)
                continue
            conn.execute(LINK_REFLEX_ETYMON_OR_IGNORE, (reflex_id, eid))


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
        # wyrd-2b50: a citation row whose (explicit or file-default)
        # source_id names a source no file registered. etymon_citation
        # .source_id FKs source(id), so such a row would be silently
        # swallowed by INSERT OR IGNORE; count + skip it visibly instead.
        "citation_source_orphans": 0,
        "citation_source_orphan_refs": [],
        # wyrd-ned5: seed reflex layer round-trip. ``reflex`` counts
        # every upserted (surface_form, position) reflex row — including
        # all-orphan reflexes that ended up with zero links.
        # ``reflex_link_orphans`` counts the individual etymon_refs that
        # didn't resolve in the rebuilt etymon table (linked etymon
        # removed via a kernel ``remove`` event, or — on an L2-only
        # rebuild — a bulk/wiktextract etymon absent without --fetch-bulk).
        "reflex": 0,
        "reflex_link_orphans": 0,
        "reflex_link_orphan_refs": [],
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
    known_source_ids: set[str] = set()
    for _path, source_id, state in file_states:
        _upsert_source(conn, source_id, state.keyed["source"][source_id])
        counts["source"] += 1
        known_source_ids.add(source_id)
        # wyrd-2b50: register secondary sources this file cites on behalf
        # of (e.g. Briggs indexes PASE/DLV/Anglo-Saxon-charter attestations
        # we have never ingested directly). etymon_citation.source_id is an
        # FK to source(id), so each must exist before pass 3 inserts the
        # citations that target it. _upsert_source uses INSERT OR REPLACE,
        # so re-declaring the same cited_source across files is idempotent.
        for cited_id, cited_payload in state.keyed.get("cited_source", {}).items():
            _upsert_source(conn, cited_id, cited_payload)
            counts["source"] += 1
            known_source_ids.add(cited_id)

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
    # loop is the per-source dispatch.
    for _path, source_id, state in file_states:
        _insert_citation_rows(
            conn, source_id, state.lists["citation"], etymon_id_by_ref, known_source_ids, counts
        )
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
        # wyrd-ned5: replay the seed reflex layer (from _reflexes.jsonl).
        # Runs after etymons are inserted so etymon_refs resolve.
        _insert_reflex_rows(conn, state.lists["reflex"], etymon_id_by_ref, counts)

    conn.commit()

    # wyrd-2n1: backfill per-element confidence from parent for any
    # rebuilt elements that the JSONL didn't carry the field on (pre-2n1
    # JSONL is the common case — the parent row's confidence is present
    # but the element-level field hasn't been mined yet). Same SQL
    # shape as migration 0013's backfill so a rebuilt DB and a
    # migrated-legacy DB land at the same state. Idempotent — the
    # WHERE-confidence-IS-NULL guard makes this a no-op for post-2n1
    # JSONL whose elements already carry the field.
    _backfill_element_confidence_from_parent(conn)

    # wyrd-14p: chain the productivity recompute at the rebuild tail so
    # `rm db && rebuild-from-jsonl` produces a complete DB. The recompute
    # is deterministic + cheap (one CTE-driven SELECT + one UPDATE per
    # reflex with corpus support), and JSONL doesn't carry productivity
    # values — the column is derivable from toponym_etymology_element
    # joins alone. Without this chain the rebuild leaves every
    # reflex.productivity at 0 and the runtime weighting collapses to
    # flat-uniform until an operator remembers to run
    # `lexicon recompute-reflex-productivity` separately.
    _recompute_reflex_productivity_after_build(conn)

    return counts


def _backfill_element_confidence_from_parent(conn: sqlite3.Connection) -> None:
    """wyrd-2n1 post-build hook: copy parent toponym_etymology.confidence
    onto every NULL-confidence element. Same SQL the migration's
    upgrade() runs on first init so a rebuilt-from-JSONL DB matches
    the migrated-legacy DB. No-op when the column or the parent are
    absent (minimal-schema test fixtures + pre-init schemas)."""
    has_column = any(
        row[1] == "confidence"
        for row in conn.execute("PRAGMA table_info(toponym_etymology_element)")
    )
    if not has_column:
        return
    with conn:
        conn.execute(
            """
            UPDATE toponym_etymology_element
               SET confidence = (
                    SELECT te.confidence
                      FROM toponym_etymology AS te
                     WHERE te.id = toponym_etymology_element.toponym_etymology_id
               )
             WHERE confidence IS NULL
            """
        )


def _recompute_reflex_productivity_after_build(conn: sqlite3.Connection) -> None:
    """Post-build hook: derive reflex.productivity from the rebuilt
    corpus. Imported lazily so the build.py module doesn't pull the
    helper at import time (the build.py + helper sit on opposite
    sides of the SQL-queries / DB-handle boundary; importing eagerly
    would inflate the lambda cold-start budget for cases that don't
    rebuild).

    No-ops when the reflex table is absent from the connection's
    schema — test fixtures use a minimal L2-only schema slice
    (``_build_fixture_db``) that omits the L3 reflex layer, and the
    post-build hook shouldn't error on those. Production callers
    always run against the full ``init_schema``-built DB which
    includes reflex.
    """
    has_reflex = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'reflex'"
    ).fetchone()
    if not has_reflex:
        return

    from wyrd.generators.kenning.lexicon.sql.queries.reflex import (
        RESET_REFLEX_PRODUCTIVITY,
        SELECT_REFLEX_PRODUCTIVITY_COUNTS,
        UPDATE_REFLEX_PRODUCTIVITY,
    )

    with conn:
        conn.execute(RESET_REFLEX_PRODUCTIVITY)
        for reflex_id, productivity in conn.execute(SELECT_REFLEX_PRODUCTIVITY_COUNTS):
            conn.execute(UPDATE_REFLEX_PRODUCTIVITY, (productivity, reflex_id))


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
    inflection per element), historical_form, confidence, notes,
    attested_year, page. (surface_in_modern is L3-derived, not an L2
    element field — wyrd-ujyo — so it doesn't participate.)

    Lists become tuples so the result is hashable. Element ordering
    is preserved as-is (ordinal IS the order; sorting would change
    semantics for any badly-formed input).
    """
    elements_key = tuple(
        (
            el.get("ordinal"),
            el.get("etymon_ref"),
            el.get("inflection"),
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


# Synthetic ledgers EXCLUDED from the generic build_from_jsonl replay (wyrd-5qg7).
# Each contains at least one row that doesn't conform to the replay schema (a
# bespoke ``_type`` not in ``log.ALL_TYPES``, e.g. ``pronunciation`` /
# ``modern_reflex`` / the ``*_reflex/descent/cluster`` audit verdicts; or no
# ``_type`` at all, e.g. ``_reflex_audit`` / ``_element_gloss*`` — though several
# ALSO carry a conforming ``source`` row). Sweeping them into the generic replay
# raised ReplayError, which silently broke from-scratch rebuild-from-jsonl. They
# fall into two groups, neither of which needs generic replay:
#   1. Audit verdict logs (``_reflex_audit`` / ``_descent_audit`` /
#      ``_cluster_reflex_audit`` / ``_toponym_reflex_audit``) — idempotency/audit
#      only; NOT read at rebuild at all. Their applied detaches round-trip through
#      ``_collapses.jsonl`` / ``_reflexes.jsonl`` (which ARE replayed). Their own
#      ``source`` row is referenced by nothing post-rebuild.
#   2. Read-directly ledgers (``_element_glosses`` / ``_element_gloss_adjudications``
#      → element-gloss enrichment; ``_pronunciation`` → collect_pronunciation;
#      ``_modern_reflexes`` → import-modern-reflexes) — re-applied by their own pass,
#      which upserts its own source.
# test_no_unhandled_nonconforming_ledgers guards that any future non-conforming
# ledger is listed here. (``_merge_audit`` / ``_tags`` are NOT here — they conform
# and replay inert.)
REPLAY_EXCLUDED_LEDGERS: frozenset[str] = frozenset(
    {
        "_reflex_audit.jsonl",
        "_descent_audit.jsonl",
        "_cognate_descent_audit.jsonl",
        "_cluster_reflex_audit.jsonl",
        "_toponym_reflex_audit.jsonl",
        "_element_glosses.jsonl",
        "_element_gloss_adjudications.jsonl",
        "_pronunciation.jsonl",
        "_modern_reflexes.jsonl",
        # park-list sidecars (rows are {ref, reason}); consumed by the campaign
        # selectors (next-slice/status), not the generic replay schema.
        "_sense_parked.jsonl",
        "_reflex_parked.jsonl",
    }
)


def jsonl_paths_in(directory: str | Path) -> list[Path]:
    """All ``*.jsonl`` files in ``directory`` EXCEPT the replay-excluded ledgers
    (:data:`REPLAY_EXCLUDED_LEDGERS`), sorted ASCII. Used as the default input set
    for :func:`build_from_jsonl`. The excluded ledgers don't conform to the generic
    replay schema; each is an audit log or is consumed by its own enrichment/import
    pass (wyrd-5qg7)."""
    p = Path(directory)
    return sorted(f for f in p.glob("*.jsonl") if f.name not in REPLAY_EXCLUDED_LEDGERS)


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
    # wyrd-jr37: attestation round-trips since PR #204; needed in the
    # monitored set so a regression that silently drops attestations
    # produces a non-zero diff instead of going invisible.
    "toponym_attestation",
    "toponym_etymology",
    "toponym_etymology_element",
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


def collect_gloss_suppressions(jsonl_paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Scan ``jsonl_paths`` for ``etymon_gloss_suppression`` rows and
    return the merged ``{etymon_ref: payload}`` state (wyrd-kutx).

    The payload's ``suppressions`` field is the operator's current list
    of glosses to drop from that etymon. Last-write-wins per ref —
    re-emit the full list to revise.
    """
    from .log import replay_file

    merged: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(p) for p in jsonl_paths):
        state = replay_file(path)
        for ref, payload in state.keyed["etymon_gloss_suppression"].items():
            merged[ref] = dict(payload)
    return merged


def collect_etymon_splits(jsonl_paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Scan ``jsonl_paths`` for ``etymon_split`` rows and return the
    merged ``{etymon_ref: payload}`` state (wyrd-kutx).

    The payload's ``into`` array names the child etymons. Last-write-wins
    per ref — re-emit the full ``into`` array to revise. Pass
    ``into: []`` to revert a split.
    """
    from .log import replay_file

    merged: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(p) for p in jsonl_paths):
        state = replay_file(path)
        for ref, payload in state.keyed["etymon_split"].items():
            merged[ref] = dict(payload)
    return merged


def collect_collapses(jsonl_paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Scan ``jsonl_paths`` for ``collapse`` rows and return the merged
    ``{from_ref: payload}`` state (wyrd-y651, the consolidation epic).

    A ``collapse`` row folds the ``from`` etymon (``ref``) into the
    ``into`` lemma — both as ``"<language>:<canonical_form>"``. Replayed
    by :func:`enrichment.apply_collapses` in the curation slot of
    ``run_full_enrichment``. Last-write-wins per ref; pass ``into: ""``
    (or drop ``into``) to revert a recorded collapse.

    This is the single ledger for ALL consolidations — the deterministic
    detector (P2) and the LLM merge pass (P3) both append here, differing
    only in the ``method`` field.

    Cycles are resolved HERE, at the single load point, so the ledger is
    always safe to apply regardless of what wrote it. The LLM merge can
    judge a pair same-meaning in BOTH directions (``fau`` ≈ ``fou`` AND
    ``fou`` ≈ ``fau``) — a ``merged_into_id`` cycle that would never settle.
    :func:`_resolve_collapse_cycles` drops the cycle-closers (to no-ops)
    and flattens chains to their terminal survivor.
    """
    from .log import replay_file

    merged: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(p) for p in jsonl_paths):
        state = replay_file(path)
        for ref, payload in state.keyed["collapse"].items():
            merged[ref] = dict(payload)
    return _resolve_collapse_cycles(merged)


def _uf_find(parent: dict[str, str], x: str) -> str:
    """Union-find root of ``x`` with path compression, mutating ``parent``.

    Unseen nodes are initialized as their own root (``setdefault``)."""
    parent.setdefault(x, x)
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _collapse_survivor(kept_into: dict[str, str], ref: str) -> str:
    """Walk ``ref`` along the kept ``into`` edges to its terminal survivor.

    ``kept_into`` is acyclic by construction (cycle-closers are dropped
    before this runs); the ``seen`` guard is a belt-and-suspenders stop."""
    cur, seen = ref, set()
    while cur in kept_into and cur not in seen:
        seen.add(cur)
        cur = kept_into[cur]
    return cur


def _resolve_collapse_cycles(
    state: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Make the collapse graph acyclic + flat. Positive edges (non-empty
    ``into``) are unioned in deterministic ``ref`` order; an edge that would
    close a cycle is demoted to a no-op (``into: ""``), and every surviving
    edge is re-pointed at its cluster's terminal survivor so a chain
    ``a -> b -> c`` becomes ``a -> c`` / ``b -> c`` (apply-order independent).
    No-op rows (rejections, reverts) pass through untouched."""
    parent: dict[str, str] = {}
    kept_into: dict[str, str] = {}
    for ref in sorted(state):
        into = state[ref].get("into")
        if not into:
            continue
        ra, rb = _uf_find(parent, ref), _uf_find(parent, into)
        if ra == rb:
            continue  # cycle-closer — demoted to a no-op below
        parent[ra] = rb
        kept_into[ref] = into

    out: dict[str, dict[str, Any]] = {}
    for ref, payload in state.items():
        if not payload.get("into"):
            out[ref] = payload  # already a no-op
        elif ref in kept_into:
            out[ref] = {**payload, "into": _collapse_survivor(kept_into, ref)}  # flatten
        else:
            out[ref] = {**payload, "into": ""}  # cycle-closer → no-op
    return out


def collect_gloss_additions(jsonl_paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Scan ``jsonl_paths`` for ``etymon_gloss_add`` rows and return
    the merged ``{etymon_ref: payload}`` state (wyrd-wz82).

    Mirrors :func:`collect_gloss_suppressions` shape — the payload's
    ``additions`` array carries ``{gloss, reason?}`` entries the
    operator wants ADDED to the etymon (inverse of suppression).
    Last-write-wins per ref — re-emit the full additions list to
    revise.
    """
    from .log import replay_file

    merged: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(p) for p in jsonl_paths):
        state = replay_file(path)
        for ref, payload in state.keyed["etymon_gloss_add"].items():
            merged[ref] = dict(payload)
    return merged
