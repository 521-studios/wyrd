"""SQLite → per-source JSONL dumper — wyrd-f295.

Reads the L3 SQLite lexicon (the build artifact) and emits L2 JSONL files
at ``data/mining/<source_id>.jsonl`` — one file per ``source`` row in the
DB. Each file is a list of canonical-state rows (no ``_op``); the
:mod:`.log` kernel's :func:`replay` is what gives semantic meaning
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

Attestation dump (wyrd-6gpy)
============================

8. One ``toponym`` canonical-state row for each toponym attested-only
   by this source — i.e. has a ``toponym_attestation`` row whose
   ``source_doc`` routes to this source but no ``toponym_etymology``
   row. Needed so Domesday-shaped toponyms (no scholarly etymology,
   only a Phillimore-format attestation) round-trip through dump →
   rebuild.
9. One ``attestation`` list row per ``toponym_attestation`` row
   whose ``source_doc`` routes to this source via the resolver in
   :func:`_source_for_attestation_doc`. The build side already
   accepts these (``_insert_attestation_rows`` per wyrd-3ypp); this
   closes the documented one-way ingester→build→DB flow.

Deferred from v0
================

- ``etymon_text_match`` and ``etymon_variant``: classified as L3 for
  v0; revisit when text-match has scholar-only filter and variant
  ingest is itself JSONL-driven.
Fantasy morpheme dump (wyrd-2thc)
=================================

The ``fantasy_morpheme`` table has no source attribution column —
rows are written by ``lexicon mine-fantasy-name`` and link to an
existing ``etymon`` row by FK. Per-source dumping doesn't apply.
Instead, :func:`dump_fantasy_morphemes_to_file` writes a single
synthetic file at ``data/mining/_fantasy_morphemes.jsonl`` with a
``fantasy-mining`` synthetic source declaration + one
``fantasy_morpheme`` list row per DB row (etymon FK projected to
``etymon_ref``). Mirrors the ``_curation.jsonl`` pattern of a
leading-underscore synthetic-source file with a non-per-source
contract.
- Uncited bulk etymons (wiktextract import): handled separately by
  L1 re-ingest path, not this dump.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .log import write_jsonl

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


# Free-text source_doc prefixes that an ingester stamps onto
# ``toponym_attestation.source_doc`` so dump-time can route the row
# back to the right source.id. The build-side contract is documented
# in build.py:_insert_attestation. Add entries here when a new
# ingester writes free-text source_doc that isn't an exact source.id
# match. The most prominent example today is Open Domesday Hull
# (wyrd-el93), whose ingester stamps "Phillimore <citation>" into
# every attestation's source_doc.
#
# INVARIANT: no prefix here may be (or be a prefix of) any actual
# source.id in the DB. _attestation_source_doc_filter generates SQL
# that OR's the exact source_id match with the LIKE-prefix matches;
# if a prefix collided with a source.id, that source's attestations
# would be double-routed (emitted to both source dumps). The
# resolver in :func:`_source_for_attestation_doc` checks exact match
# first, so the Python oracle hides the ambiguity — but the SQL
# filter doesn't, and divergence would silently double-emit. Add
# only ingester-specific brand names that no source.id will ever
# legitimately match.
_ATTESTATION_DOC_PREFIX_TO_SOURCE: tuple[tuple[str, str], ...] = (
    ("Phillimore", "open_domesday_hull"),
)


def _source_for_attestation_doc(source_doc: str | None, known_source_ids: set[str]) -> str | None:
    """Resolve a ``toponym_attestation.source_doc`` value to the
    ``source.id`` that should own its dump entry.

    Resolution order:
    1. ``None`` → ``None`` (orphan; not dumped).
    2. Exact match against ``known_source_ids`` (direct source
       attribution — many ingesters write ``source_doc = "<source_id>"``).
    3. Prefix match against
       ``_ATTESTATION_DOC_PREFIX_TO_SOURCE`` (free-text source_doc
       ingesters like Open Domesday's Phillimore citations).
    4. No match → ``None`` (orphan; will be silently skipped by the
       per-source dump — operator workflow is responsible for
       extending the prefix map when new ingesters land).

    NOTE: this is the parity oracle for
    :func:`_attestation_source_doc_filter`. Production dump code uses
    the SQL filter directly (O(M) per source). The Python resolver is
    kept as the human-readable definition of bucketing semantics and
    is exercised by ``test_attestation_source_doc_filter_resolver_parity``
    to ensure the SQL filter selects the same set of rows for every
    sample ``source_doc``. Do not delete as dead code.
    """
    if source_doc is None:
        return None
    if source_doc in known_source_ids:
        return source_doc
    for prefix, source_id in _ATTESTATION_DOC_PREFIX_TO_SOURCE:
        if source_doc.startswith(prefix):
            return source_id
    return None


def _attestation_source_doc_filter(source_id: str) -> tuple[str, list[Any]]:
    """Build a SQL WHERE-clause fragment + bind params selecting
    ``toponym_attestation`` rows whose ``source_doc`` routes to
    ``source_id`` per :func:`_source_for_attestation_doc`.

    Pushes the bucketing into SQL so per-source dumps don't do an
    O(N×M) scan when ``dump_all_sources`` iterates N sources over
    M attestations (the live DB has 21K attestations × 50+ sources,
    so the Python-side filter was a 1M-row hot loop)."""
    prefixes = [p for p, sid in _ATTESTATION_DOC_PREFIX_TO_SOURCE if sid == source_id]
    clauses = ["ta.source_doc = ?"]
    params: list[Any] = [source_id]
    for prefix in prefixes:
        # GLOB is case-sensitive by default; LIKE is ASCII-case-INsensitive
        # in SQLite (and COLLATE BINARY doesn't override that — LIKE's
        # case rule is separate from the comparison collation). Use GLOB
        # so the SQL filter matches the Python resolver's
        # ``str.startswith()`` semantics — a lowercase ``phillimore 1L1``
        # is orphaned by both rather than routed by only one.
        #
        # The ``NOT IN (SELECT id FROM source)`` guard enforces the
        # Python oracle's exact-match-wins priority directly in SQL.
        # Without it, a future entry that violates the
        # _ATTESTATION_DOC_PREFIX_TO_SOURCE invariant — say
        # ``("Mawer", "mawer_volumes")`` when a source.id ``Mawer``
        # exists — would silently double-route the row matching
        # ``source_doc = 'Mawer'`` (it satisfies both the exact-match
        # clause of one source and the prefix clause of another). The
        # subquery is a one-shot SELECT against the ``source`` table
        # (< 100 rows in production) and SQLite caches it.
        clauses.append("(ta.source_doc GLOB ? AND ta.source_doc NOT IN (SELECT id FROM source))")
        # Registered prefixes are hand-written constants — no glob
        # metacharacters in them today, but if a future prefix needs
        # `[`/`?`/`*` the caller must escape per SQLite GLOB rules.
        params.append(prefix + "*")
    return "(" + " OR ".join(clauses) + ")", params


def _dump_attestation_only_toponyms(
    conn: sqlite3.Connection,
    source_id: str,
) -> Iterable[dict[str, Any]]:
    """Yield ``toponym`` canonical-state rows for toponyms that have
    at least one ``toponym_attestation`` routing to ``source_id`` AND
    have no ``toponym_etymology`` row for this source.

    Without this path, attestation-only toponyms (e.g. the ~17K
    Domesday settlements that carry Phillimore attestations but no
    scholarly etymology) never appear in dump output — the rebuild's
    attestation pass would orphan-skip every one of them. Matches
    ``_dump_toponyms_and_etymologies``'s emit shape so the build's
    :func:`_merge_toponym` dedupes any cross-emit overlap silently.
    """
    where_clause, params = _attestation_source_doc_filter(source_id)
    rows = conn.execute(
        f"""
        SELECT DISTINCT t.id, t.modern_name, t.country, t.region
          FROM toponym t
          JOIN toponym_attestation ta ON ta.toponym_id = t.id
         WHERE t.id NOT IN (
             SELECT toponym_id FROM toponym_etymology WHERE source_id = ?
         )
           AND {where_clause}
         ORDER BY t.modern_name, t.region, t.country, t.id
        """,  # noqa: S608 — where_clause built from registered constants
        [source_id, *params],
    ).fetchall()
    for r in rows:
        row: dict[str, Any] = {
            "_type": "toponym",
            "ref": toponym_ref(r["modern_name"], r["region"]),
            "modern_name": r["modern_name"],
        }
        row.update(_drop_nulls({"country": r["country"], "region": r["region"]}))
        yield row


def _dump_attestations_for_source(
    conn: sqlite3.Connection,
    source_id: str,
) -> Iterable[dict[str, Any]]:
    """Yield ``attestation`` rows for every ``toponym_attestation``
    whose ``source_doc`` routes to ``source_id``. Build side already
    accepts the ``attestation`` _type per
    :func:`.build._insert_attestation_rows` (wyrd-3ypp); this
    closes the documented one-way gap (ingester → JSONL → build but
    not back out of the DB)."""
    where_clause, params = _attestation_source_doc_filter(source_id)
    rows = conn.execute(
        f"""
        SELECT t.modern_name, t.region, ta.form, ta.date_year, ta.source_doc
          FROM toponym_attestation ta
          JOIN toponym t ON t.id = ta.toponym_id
         WHERE {where_clause}
         ORDER BY t.modern_name, t.region, ta.date_year, ta.id
        """,  # noqa: S608 — where_clause built from registered constants
        params,
    ).fetchall()
    for r in rows:
        row: dict[str, Any] = {
            "_type": "attestation",
            "toponym_ref": toponym_ref(r["modern_name"], r["region"]),
            "form": r["form"],
        }
        row.update(_drop_nulls({"date_year": r["date_year"], "source_doc": r["source_doc"]}))
        yield row


def _personal_name_tables_exist(conn: sqlite3.Connection) -> bool:
    """Briggs PN tables are added by migration 0010; tests + legacy
    snapshots that build a partial schema may not have them. Guard
    the dump helpers so they no-op rather than raising OperationalError
    when the tables are absent."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='personal_name'"
    ).fetchone()
    return row is not None


def _dump_personal_names_for_source(
    conn: sqlite3.Connection, source_id: str
) -> Iterable[dict[str, Any]]:
    """One ``personal_name`` canonical-state row per PN headform whose
    ``source_doc`` matches this source (wyrd-11zh). The ``ref`` is the
    headform — sufficient for the build pass's FK resolution map."""
    if not _personal_name_tables_exist(conn):
        return
    pns = conn.execute(
        """SELECT headform, normalized_form, language_hints, is_feminine,
                  pase_count, has_dlv, ascharter_refs, source_doc, raw_entry
             FROM personal_name
            WHERE source_doc = ?
            ORDER BY headform""",
        (source_id,),
    ).fetchall()
    for pn in pns:
        row: dict[str, Any] = {"_type": "personal_name", "ref": pn["headform"]}
        row.update(
            _drop_nulls(
                {
                    "headform": pn["headform"],
                    "normalized_form": pn["normalized_form"],
                    "language_hints": pn["language_hints"],
                    "is_feminine": pn["is_feminine"],
                    "pase_count": pn["pase_count"],
                    "has_dlv": pn["has_dlv"],
                    "ascharter_refs": pn["ascharter_refs"],
                    "source_doc": pn["source_doc"],
                    "raw_entry": pn["raw_entry"],
                }
            )
        )
        yield row


def _dump_personal_name_attestations_for_source(
    conn: sqlite3.Connection, source_id: str
) -> Iterable[dict[str, Any]]:
    """One ``personal_name_toponym_attestation`` row per (PN, toponym,
    county) occurrence in this source (wyrd-11zh). Carries
    ``personal_name_ref`` (= the PN's headform) so the build pass can
    resolve it against the per-file PN insert map."""
    if not _personal_name_tables_exist(conn):
        return
    rows = conn.execute(
        """SELECT pn.headform AS personal_name_ref,
                  a.toponym_form, a.attested_variant, a.county_code,
                  a.county_canonical, a.date_qualifier,
                  a.is_uncertain, a.is_serious_doubt, a.source_doc, a.raw_text
             FROM personal_name_toponym_attestation a
             JOIN personal_name pn ON pn.id = a.personal_name_id
            WHERE a.source_doc = ?
            ORDER BY pn.headform, a.toponym_form, a.county_code""",
        (source_id,),
    ).fetchall()
    for r in rows:
        yield {
            "_type": "personal_name_toponym_attestation",
            **_drop_nulls(
                {
                    "personal_name_ref": r["personal_name_ref"],
                    "toponym_form": r["toponym_form"],
                    "attested_variant": r["attested_variant"],
                    "county_code": r["county_code"],
                    "county_canonical": r["county_canonical"],
                    "date_qualifier": r["date_qualifier"],
                    "is_uncertain": r["is_uncertain"],
                    "is_serious_doubt": r["is_serious_doubt"],
                    "source_doc": r["source_doc"],
                    "raw_text": r["raw_text"],
                }
            ),
        }


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
    rows.extend(_dump_attestation_only_toponyms(conn, source_id))
    rows.extend(_dump_attestations_for_source(conn, source_id))
    # wyrd-11zh: Briggs PN ingest round-trip. Only emits rows when
    # personal_name rows actually exist for this source — non-PN
    # sources see zero overhead.
    rows.extend(_dump_personal_names_for_source(conn, source_id))
    rows.extend(_dump_personal_name_attestations_for_source(conn, source_id))
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
        # wyrd-2thc: synthetic source declared inline by
        # :func:`dump_fantasy_morphemes_to_file` at
        # ``data/mining/_fantasy_morphemes.jsonl``. The live DB
        # doesn't carry it in the ``source`` table, but a post-rebuild
        # DB does (the synthetic source row from the JSONL is inserted
        # by build_from_jsonl). Excluding here prevents ``dump_all_sources``
        # from emitting a competing per-source ``fantasy-mining.jsonl``
        # on a rebuilt DB.
        "fantasy-mining",
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
        # wyrd-3atv: Hundred Rolls (1279-80) ingester output.
        # Operator-transcribed CSV → JSONL via lexicon
        # ingest-hundred-rolls. Same CAVEAT as os_open_names: every
        # attestation row carries source_doc provenance (hundred +
        # county) that the current dump-jsonl path doesn't preserve,
        # so a dump→rebuild cycle would silently drop the per-Roll
        # provenance. Re-include this source in dumps only after
        # attestation dump-side is in scope of build_from_jsonl.
        "rotuli_hundredorum",
        # wyrd-myh1: Speed 1611 / Hearth Tax operator-CSV ingesters.
        # Same CAVEAT as Hundred Rolls / OS Open Names — each
        # attestation carries per-row source_doc provenance (parish,
        # year_specific) that the current dump-jsonl attestation path
        # would drop. Re-include only after attestation dump-side is
        # in scope.
        "speed_1611",
        "hearth_tax_1660s",
        # wyrd-j2bv: Retired rando-port etymons tombstone. Lives at
        # data/mining/retired/rando-retired.jsonl — outside the
        # non-recursive *.jsonl glob, so it's NOT loaded by the build
        # pipeline. Excluded from dump-jsonl too in case a future
        # rebuild somehow imports the source ID (would write to the
        # wrong path). If you ever need to re-examine retired
        # etymons, read the tombstone file directly rather than
        # round-tripping through dump-jsonl.
        "rando-retired",
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


# --- fantasy_morpheme dump (wyrd-2thc) ---------------------------------

# Synthetic source declared at the head of ``_fantasy_morphemes.jsonl``.
# The live DB has no ``fantasy-mining`` row in ``source`` — fantasy
# morphemes are written by ``lexicon mine-fantasy-name`` without source
# attribution. This synthetic row exists only in the JSONL output (and
# is inserted into the rebuilt DB by ``build_from_jsonl``). Same
# asymmetry as ``manual-curation`` (declared only in
# ``_curation.jsonl``); accepted because the rebuild's downstream
# bundle export (``collect_fantasy_morphemes``) reads from the table,
# not from ``source``, so the synthetic source has no runtime effect.
_FANTASY_MINING_SOURCE_ID = "fantasy-mining"
_FANTASY_MINING_FILENAME = "_fantasy_morphemes.jsonl"
_FANTASY_MINING_SOURCE_ROW = {
    "_type": "source",
    "ref": _FANTASY_MINING_SOURCE_ID,
    "title": "Fantasy morpheme mining (wyrd-vz7f)",
    "notes": (
        "Synthetic source for fantasy_morpheme L2 round-trip (wyrd-2thc). "
        "Rows are written by `lexicon mine-fantasy-name` against the "
        "DB's fantasy_morpheme table — etymon FK resolved on dump via "
        "etymon_ref. This source declaration is JSONL-only; the live "
        "DB has no fantasy-mining row in source."
    ),
}

# Fantasy morpheme L2 columns dumped to ``_fantasy_morphemes.jsonl``.
# ``id`` is omitted (autoincrement; redundant under the
# (input_name, approach_version) UNIQUE on rebuild). ``etymon_id`` is
# resolved to ``etymon_ref`` for cross-rebuild stability — etymon ids
# aren't stable across rebuilds but (language, canonical_form) is.
_FANTASY_MORPHEME_SCALAR_COLUMNS: tuple[str, ...] = (
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


def _dump_fantasy_morpheme_rows(conn: sqlite3.Connection) -> Iterable[dict[str, Any]]:
    """Yield one ``fantasy_morpheme`` list row per DB row, ordered by
    ``input_name`` for diff stability. Joins ``etymon`` to project
    ``etymon_id`` → ``etymon_ref``; rows with NULL etymon_id (barred
    morphemes that never resolved a corpus etymon) emit no
    ``etymon_ref`` field — replaying skips the JOIN on build."""
    rows = conn.execute(
        f"""
        SELECT {", ".join("fm." + c for c in _FANTASY_MORPHEME_SCALAR_COLUMNS)},
               e.language AS _etymon_language,
               e.canonical_form AS _etymon_canonical_form
          FROM fantasy_morpheme fm
          LEFT JOIN etymon e ON e.id = fm.etymon_id
         ORDER BY fm.input_name, fm.approach_version
        """  # noqa: S608 — column names are hand-written module constants
    ).fetchall()
    for row in rows:
        out: dict[str, Any] = {"_type": "fantasy_morpheme"}
        for col in _FANTASY_MORPHEME_SCALAR_COLUMNS:
            v = row[col]
            if v is not None:
                out[col] = v
        if row["_etymon_language"] is not None:
            out["etymon_ref"] = etymon_ref(row["_etymon_language"], row["_etymon_canonical_form"])
        yield out


def dump_fantasy_morphemes_to_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Yield the rows that ``_fantasy_morphemes.jsonl`` carries: the
    synthetic ``fantasy-mining`` source declaration followed by one
    ``fantasy_morpheme`` row per DB row.

    Returns an empty list if the ``fantasy_morpheme`` table does not
    exist (older DBs predate wyrd-ami) or carries zero rows. Skipping
    the file entirely keeps the source-row contract from creating an
    empty-but-meaningful artifact on DBs that never ran fantasy
    mining."""
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fantasy_morpheme'"
    ).fetchone()
    if not table_exists:
        return []
    morphemes = list(_dump_fantasy_morpheme_rows(conn))
    if not morphemes:
        return []
    return [_FANTASY_MINING_SOURCE_ROW, *morphemes]


def dump_fantasy_morphemes_to_file(
    conn: sqlite3.Connection,
    out_dir: str | Path,
) -> tuple[Path, int]:
    """Write ``<out_dir>/_fantasy_morphemes.jsonl``. Returns ``(path,
    row_count)``; row_count is 0 (file not written) when the DB has
    no fantasy_morpheme rows."""
    rows = dump_fantasy_morphemes_to_rows(conn)
    path = Path(out_dir) / _FANTASY_MINING_FILENAME
    if not rows:
        return path, 0
    n = write_jsonl(path, rows)
    return path, n
