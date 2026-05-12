"""Per-source JSONL → fresh SQLite rebuild — wyrd-f295.

Reads ``data/mining/*.jsonl`` (the L2 source-of-truth) and inserts the
replayed state into a freshly-initialized SQLite lexicon. The companion
to :mod:`jsonl_dump` — together they close the round-trip:

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
- ``toponym_attestation`` rows (the dumper currently skips them too;
  ``source_doc`` is free-text and needs a parser to bucket per source).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .jsonl_log import ReplayState, replay_file
from .jsonl_dump import etymon_ref as _etymon_ref


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


def _insert_etymon(conn: sqlite3.Connection, payload: dict[str, Any]) -> int:
    """Insert an etymon row. Returns the new row id. Assumes
    (canonical_form, language) is unique — caller is responsible for
    pre-merging across files so this is the first insert for the pair."""
    cols = [c for c in _ETYMON_INSERT_COLUMNS if c in payload]
    if "canonical_form" not in cols or "language" not in cols:
        raise BuildError(f"etymon row missing canonical_form or language: {payload}")
    vals = [payload[c] for c in cols]
    placeholders = ", ".join("?" * len(cols))
    cur = conn.execute(
        f"INSERT INTO etymon ({', '.join(cols)}) VALUES ({placeholders})", tuple(vals)
    )
    eid = cur.lastrowid
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
    counts: dict[str, int] = {
        "source": 0,
        "etymon": 0,
        "toponym": 0,
        "citation": 0,
        "etymon_descent": 0,
        "mining_run": 0,
        "etymology_element": 0,
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
    for path, source_id, state in file_states:
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
    for path, source_id, state in file_states:
        for row in state.lists["citation"]:
            eref = row["etymon_ref"]
            eid = etymon_id_by_ref.get(eref)
            if eid is None:
                raise BuildError(f"{path}: citation references unknown etymon {eref!r}")
            _insert_citation(conn, eid, source_id, row)
            counts["citation"] += 1

        for row in state.lists["etymon_descent"]:
            pid = etymon_id_by_ref.get(row["parent_ref"])
            cid = etymon_id_by_ref.get(row["child_ref"])
            if pid is None or cid is None:
                raise BuildError(
                    f"{path}: etymon_descent references unknown etymon "
                    f"(parent={row['parent_ref']!r}, child={row['child_ref']!r})"
                )
            _insert_descent(conn, pid, cid, source_id, row)
            counts["etymon_descent"] += 1

        for row in state.lists["mining_run"]:
            _insert_mining_run(conn, source_id, row)
            counts["mining_run"] += 1

        for row in state.lists["etymology_element"]:
            tref = row["toponym_ref"]
            tid = toponym_id_by_ref.get(tref)
            if tid is None:
                raise BuildError(f"{path}: etymology_element references unknown toponym {tref!r}")
            _insert_etymology_element(conn, tid, source_id, row, etymon_id_by_ref)
            counts["etymology_element"] += 1

    conn.commit()
    return counts


def jsonl_paths_in(directory: str | Path) -> list[Path]:
    """All ``*.jsonl`` files in ``directory``, sorted ASCII. Used as the
    default input set for :func:`build_from_jsonl`."""
    p = Path(directory)
    return sorted(p.glob("*.jsonl"))
