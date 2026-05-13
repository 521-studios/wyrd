"""SQLite → per-source JSONL dumper — wyrd-f295.

Reads the L3 SQLite lexicon (the build artifact) and emits L2 JSONL files
at ``data/mining/<source_id>.jsonl`` — one file per ``source`` row in the
DB. Each file is a list of canonical-state rows (no ``_op``); the
:mod:`jsonl_log` kernel's :func:`replay` is what gives semantic meaning
to the row sequence.

What goes IN a source file (L2-attributable facts)
==================================================

For each ``source`` row, the dump writes:

1. One ``source`` canonical-state row (the bibliographic record).
2. One ``etymon`` canonical-state row for every etymon CITED by this
   source via ``etymon_citation``. Carries L2 fields only:
   canonical_form, language, modifier_type, position_pref, notes,
   pronunciation_*, original_script, transliteration, glosses, tags.
   L3-derived columns (lemma_id, merged_into_id, cognate_id, stratum,
   english_shaped) are dropped — they get rebuilt by post-replay
   enrichment passes.
3. One ``citation`` list row per ``etymon_citation`` row.
4. One ``etymon_descent`` list row per descent edge attributed to
   this source.
5. One ``mining_run`` list row per audit-log entry.
6. One ``toponym`` canonical-state row for each toponym this source
   has a ``toponym_etymology`` row about.
7. One ``etymology_element`` list row per ``toponym_etymology`` row,
   carrying the full ordered element list inline (the elements live
   in a child table but are tightly coupled to their parent etymology
   row).

Cross-source refs
=================

Etymon refs are ``"<language>:<canonical_form>"``. Toponym refs are
``"<modern_name>@<region>"`` (region falls back to ``"-"`` when null).
Multiple sources may emit the same etymon/toponym ref — the build pass
merges them.

Deferred from v0
================

- ``toponym_attestation``: ``source_doc`` is free-text, no FK. A
  later pass parses ``source_doc`` and emits attestations per source.
- ``etymon_text_match`` and ``etymon_variant``: classified as L3 for
  v0; revisit when text-match has scholar-only filter and variant
  ingest is itself JSONL-driven.
- Uncited bulk etymons (wiktextract import): handled separately by
  L1 re-ingest path, not this dump.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .jsonl_log import write_jsonl

# Etymon columns that are L2-attributable facts. Order is significant
# only for diff stability — the kernel doesn't care about field order.
_ETYMON_L2_COLUMNS: tuple[str, ...] = (
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

_SOURCE_COLUMNS: tuple[str, ...] = (
    "author",
    "title",
    "year",
    "region",
    "language_focus",
    "notes",
)

_MINING_RUN_COLUMNS: tuple[str, ...] = (
    "provider",
    "model",
    "mode",
    "started_at",
    "completed_at",
    "parsed_count",
    "accepted",
    "declined",
    "rejected",
    "by_failure",
    "notes",
)


def etymon_ref(language: str, canonical_form: str) -> str:
    """Build the cross-file ref for an etymon. Stable on
    ``(language, canonical_form)`` — the etymon table's natural key."""
    return f"{language}:{canonical_form}"


def toponym_ref(modern_name: str, region: str | None) -> str:
    """Build the cross-file ref for a toponym. ``region`` is the
    disambiguator (Acton@Cheshire vs Acton@Suffolk); when null we use
    ``-`` so the ref is still parseable."""
    return f"{modern_name}@{region or '-'}"


def _drop_nulls(d: dict[str, Any]) -> dict[str, Any]:
    """Strip ``None`` values for cleaner JSONL diffs."""
    return {k: v for k, v in d.items() if v is not None}


def _dump_source_row(conn: sqlite3.Connection, source_id: str) -> dict[str, Any]:
    src = conn.execute("SELECT * FROM source WHERE id=?", (source_id,)).fetchone()
    if src is None:
        raise ValueError(f"no source row for id={source_id!r}")
    row: dict[str, Any] = {"_type": "source", "ref": source_id}
    row.update(_drop_nulls({col: src[col] for col in _SOURCE_COLUMNS}))
    return row


def _dump_cited_etymons(conn: sqlite3.Connection, source_id: str) -> Iterable[dict[str, Any]]:
    """One ``etymon`` canonical-state row per etymon REFERENCED by this
    source — whether via citation, descent edge, or etymology element.

    Each source's JSONL file is self-contained: every etymon ref it
    uses (in any list-type row) gets a corresponding canonical-state
    etymon row in the same file. Build's cross-file merge unifies
    duplicates across sources."""
    etymons = conn.execute(
        """
        SELECT DISTINCT e.id AS etymon_id, e.canonical_form, e.language,
               e.modifier_type, e.position_pref, e.notes,
               e.pronunciation_ipa, e.pronunciation_dialect,
               e.original_script, e.transliteration
          FROM etymon e
         WHERE e.id IN (
                SELECT etymon_id FROM etymon_citation WHERE source_id = ?
                UNION
                SELECT parent_id FROM etymon_descent WHERE source_id = ?
                UNION
                SELECT child_id FROM etymon_descent WHERE source_id = ?
                UNION
                SELECT el.etymon_id
                  FROM toponym_etymology_element el
                  JOIN toponym_etymology te ON te.id = el.toponym_etymology_id
                 WHERE te.source_id = ?
         )
         ORDER BY e.language, e.canonical_form
        """,
        (source_id, source_id, source_id, source_id),
    ).fetchall()
    for e in etymons:
        ref = etymon_ref(e["language"], e["canonical_form"])
        glosses = [
            g["gloss"]
            for g in conn.execute(
                "SELECT gloss FROM etymon_gloss WHERE etymon_id=? ORDER BY gloss",
                (e["etymon_id"],),
            )
        ]
        tags = [
            t["tag"]
            for t in conn.execute(
                "SELECT tag FROM etymon_tag WHERE etymon_id=? ORDER BY tag",
                (e["etymon_id"],),
            )
        ]
        row: dict[str, Any] = {"_type": "etymon", "ref": ref}
        row.update(_drop_nulls({col: e[col] for col in _ETYMON_L2_COLUMNS}))
        if glosses:
            row["glosses"] = glosses
        if tags:
            row["tags"] = tags
        yield row


def _dump_citations(conn: sqlite3.Connection, source_id: str) -> Iterable[dict[str, Any]]:
    citations = conn.execute(
        """
        SELECT e.language, e.canonical_form,
               c.page, c.short_quote, c.context_snippet
          FROM etymon_citation c
          JOIN etymon e ON e.id = c.etymon_id
         WHERE c.source_id = ?
         ORDER BY c.id
        """,
        (source_id,),
    ).fetchall()
    for c in citations:
        row: dict[str, Any] = {
            "_type": "citation",
            "etymon_ref": etymon_ref(c["language"], c["canonical_form"]),
        }
        row.update(
            _drop_nulls(
                {
                    "page": c["page"],
                    "short_quote": c["short_quote"],
                    "context_snippet": c["context_snippet"],
                }
            )
        )
        yield row


def _dump_descent_edges(conn: sqlite3.Connection, source_id: str) -> Iterable[dict[str, Any]]:
    edges = conn.execute(
        """
        SELECT pe.language AS p_lang, pe.canonical_form AS p_form,
               ce.language AS c_lang, ce.canonical_form AS c_form,
               d.edge_type, d.confidence, d.notes
          FROM etymon_descent d
          JOIN etymon pe ON pe.id = d.parent_id
          JOIN etymon ce ON ce.id = d.child_id
         WHERE d.source_id = ?
         ORDER BY d.id
        """,
        (source_id,),
    ).fetchall()
    for d in edges:
        row: dict[str, Any] = {
            "_type": "etymon_descent",
            "parent_ref": etymon_ref(d["p_lang"], d["p_form"]),
            "child_ref": etymon_ref(d["c_lang"], d["c_form"]),
            "edge_type": d["edge_type"],
        }
        row.update(_drop_nulls({"confidence": d["confidence"], "notes": d["notes"]}))
        yield row


def _dump_mining_runs(conn: sqlite3.Connection, source_id: str) -> Iterable[dict[str, Any]]:
    runs = conn.execute(
        """
        SELECT provider, model, mode, started_at, completed_at,
               parsed_count, accepted, declined, rejected, by_failure, notes
          FROM mining_run
         WHERE source_id = ?
         ORDER BY id
        """,
        (source_id,),
    ).fetchall()
    for m in runs:
        row: dict[str, Any] = {"_type": "mining_run"}
        row.update(_drop_nulls({col: m[col] for col in _MINING_RUN_COLUMNS}))
        yield row


def _dump_toponyms_and_etymologies(
    conn: sqlite3.Connection, source_id: str
) -> Iterable[dict[str, Any]]:
    """Toponym canonical-state rows + etymology_element list rows.

    Emits each touched toponym once (sorted by ref) then every
    ``toponym_etymology`` from this source as one ``etymology_element``
    row carrying its ordered element list inline.
    """
    topo_rows = conn.execute(
        """
        SELECT DISTINCT t.id AS toponym_id, t.modern_name, t.country, t.region
          FROM toponym t
          JOIN toponym_etymology te ON te.toponym_id = t.id
         WHERE te.source_id = ?
         ORDER BY t.modern_name, t.region, t.country
        """,
        (source_id,),
    ).fetchall()
    for t in topo_rows:
        ref = toponym_ref(t["modern_name"], t["region"])
        row: dict[str, Any] = {
            "_type": "toponym",
            "ref": ref,
            "modern_name": t["modern_name"],
        }
        row.update(_drop_nulls({"country": t["country"], "region": t["region"]}))
        yield row

    etys = conn.execute(
        """
        SELECT te.id AS etymology_id, t.modern_name, t.region,
               te.page, te.historical_form, te.confidence, te.notes, te.attested_year
          FROM toponym_etymology te
          JOIN toponym t ON t.id = te.toponym_id
         WHERE te.source_id = ?
         ORDER BY t.modern_name, te.id
        """,
        (source_id,),
    ).fetchall()
    for te in etys:
        elements: list[dict[str, Any]] = []
        for el in conn.execute(
            """
            SELECT el.ordinal, el.inflection, el.surface_in_modern,
                   e.language, e.canonical_form
              FROM toponym_etymology_element el
              JOIN etymon e ON e.id = el.etymon_id
             WHERE el.toponym_etymology_id = ?
             ORDER BY el.ordinal
            """,
            (te["etymology_id"],),
        ):
            elem: dict[str, Any] = {
                "ordinal": el["ordinal"],
                "etymon_ref": etymon_ref(el["language"], el["canonical_form"]),
            }
            elem.update(
                _drop_nulls(
                    {
                        "inflection": el["inflection"],
                        "surface_in_modern": el["surface_in_modern"],
                    }
                )
            )
            elements.append(elem)
        row: dict[str, Any] = {
            "_type": "etymology_element",
            "toponym_ref": toponym_ref(te["modern_name"], te["region"]),
            "elements": elements,
        }
        row.update(
            _drop_nulls(
                {
                    "page": te["page"],
                    "historical_form": te["historical_form"],
                    "confidence": te["confidence"],
                    "notes": te["notes"],
                    "attested_year": te["attested_year"],
                }
            )
        )
        yield row


def dump_source_to_rows(conn: sqlite3.Connection, source_id: str) -> list[dict[str, Any]]:
    """Return the full canonical-state row list for one source. Useful
    for tests + for stream-then-write workflows."""
    rows: list[dict[str, Any]] = []
    rows.append(_dump_source_row(conn, source_id))
    rows.extend(_dump_cited_etymons(conn, source_id))
    rows.extend(_dump_citations(conn, source_id))
    rows.extend(_dump_descent_edges(conn, source_id))
    rows.extend(_dump_mining_runs(conn, source_id))
    rows.extend(_dump_toponyms_and_etymologies(conn, source_id))
    return rows


# Sources skipped by ``dump-jsonl`` by default, for two reasons:
#
# 1. Bulk wiktextract-derived sources whose contribution is re-derivable
#    from L1 raw inputs (``sources/wiktextract_*.jsonl``). Their dump
#    would produce huge files that don't belong in git — the build
#    pipeline re-runs the wiktextract ingester to recreate their rows.
# 2. The synthetic ``manual-curation`` source (wyrd-2jhs / wyrd-tzf2)
#    whose JSONL file is operator-maintained at
#    ``data/mining/_curation.jsonl``. Dumping it would create a competing
#    ``manual-curation.jsonl`` and overwrite-by-accident the hand-
#    authored curation events that don't survive as DB rows.
#
# Operators can override this default (``--include-bulk`` CLI flag, or
# pass ``exclude=()`` to :func:`dump_all_sources`) for one-off diffing
# or archival.
DEFAULT_BULK_EXCLUDED_SOURCES: frozenset[str] = frozenset(
    {
        "wiktionary",
        "wiktionary-empirical",
        "wiktionary-forms",
        "manual-curation",
        # wyrd-3ypp: OS Open Names data product. ~200K populated-place
        # rows regenerated on demand from the L1 CSV in sources/;
        # dump-jsonl would produce a competing 30+MB file otherwise.
        # CAVEAT: before removing from this list, implement
        # attestation dump-side (build_from_jsonl's "Out of scope"
        # section). Without it, dump-jsonl would emit toponym rows
        # but silently drop the per-row attestations that carry the
        # OS-URI provenance — the rebuild would resurrect a partial
        # source.
        "os_open_names",
    }
)


def list_source_ids(
    conn: sqlite3.Connection,
    *,
    exclude: Iterable[str] = (),
) -> list[str]:
    """All ``source.id`` values minus ``exclude``, sorted ASCII.
    Drives the all-sources dump."""
    excluded = set(exclude)
    return [
        r["id"]
        for r in conn.execute("SELECT id FROM source ORDER BY id")
        if r["id"] not in excluded
    ]


def dump_source_to_file(
    conn: sqlite3.Connection,
    source_id: str,
    out_dir: str | Path,
) -> tuple[Path, int]:
    """Write one source's JSONL to ``<out_dir>/<source_id>.jsonl``.
    Returns ``(path, row_count)``."""
    rows = dump_source_to_rows(conn, source_id)
    path = Path(out_dir) / f"{source_id}.jsonl"
    n = write_jsonl(path, rows)
    return path, n


def dump_all_sources(
    conn: sqlite3.Connection,
    out_dir: str | Path,
    *,
    exclude: Iterable[str] = DEFAULT_BULK_EXCLUDED_SOURCES,
) -> dict[str, int]:
    """Write per-source JSONL files for every ``source`` row not in
    ``exclude``. Returns ``{source_id: row_count}`` for telemetry.

    Default ``exclude`` skips bulk wiktextract-derived sources whose
    contribution is re-derivable from L1 raw inputs — see
    :data:`DEFAULT_BULK_EXCLUDED_SOURCES`. Pass ``exclude=()`` to dump
    everything.
    """
    counts: dict[str, int] = {}
    for sid in list_source_ids(conn, exclude=exclude):
        _, n = dump_source_to_file(conn, sid, out_dir)
        counts[sid] = n
    return counts
