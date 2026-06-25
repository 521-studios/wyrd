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
``fantasy-mining`` synthetic source declaration, then one ``etymon``
canonical-state row per referenced etymon (wyrd-ruvk — these are
frequently UNCITED, so no per-source dump carries them; without them
a rebuild can't resolve the morphemes' ``etymon_ref`` and drops them
as orphans), then one ``fantasy_morpheme`` list row per DB row (etymon
FK projected to ``etymon_ref``). Mirrors the ``_curation.jsonl``
pattern of a leading-underscore synthetic-source file with a
non-per-source contract.
- Uncited bulk etymons (wiktextract import): handled separately by
  L1 re-ingest path, not this dump.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ._l2_columns import (
    _ETYMON_L2_COLUMNS,
    _FANTASY_MORPHEME_COLUMNS,
    _SOURCE_COLUMNS,
)
from .log import write_jsonl

_logger = logging.getLogger(__name__)


class MergeChainError(RuntimeError):
    """The lexicon has an un-flattened multi-hop ``merged_into_id`` chain (or
    cycle): a tombstone whose ``merged_into_id`` points at another tombstone.
    The dump's follow-to-winner is single-hop, so dumping such a DB would emit a
    mid-chain tombstone that resurrects as a live unmerged etymon on rebuild
    (D22). Re-run enrichment (``flatten_merge_chains``) to flatten before dumping
    (wyrd-lpxq)."""


def assert_merge_chains_flat(conn: sqlite3.Connection) -> None:
    """Fail loud (read-only) if any ``etymon.merged_into_id`` chain is longer
    than one hop — a tombstone pointing at another tombstone (multi-hop chain or
    cycle). The dump follows winners single-hop and assumes ``merged_into_id``
    names a TERMINAL winner; enrichment's ``flatten_merge_chains`` establishes
    that on every rebuild, but the standalone read-only ``dump-jsonl`` CLI can't
    enrich, so it verifies the invariant here rather than silently producing a
    non-reconstructible dump (wyrd-lpxq)."""
    row = conn.execute(
        "SELECT loser.id AS loser, mid.id AS mid "
        "FROM etymon loser JOIN etymon mid ON mid.id = loser.merged_into_id "
        "WHERE mid.merged_into_id IS NOT NULL LIMIT 1"
    ).fetchone()
    if row is not None:
        raise MergeChainError(
            f"etymon {row['loser']} merges into tombstone {row['mid']} "
            "(un-flattened merge chain); run enrichment to flatten merged_into_id "
            "to terminal winners before dumping — wyrd-lpxq."
        )


# (The etymon + source L2 column lists are single-sourced in _l2_columns, imported
# above; build.py derives its insert columns from them.) Mining-run columns are
# dump-only — the rebuilder writes mining_run inline, so this list stays local.
# Order is significant only for diff stability — the kernel doesn't care.
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


# Columns _etymon_state_row needs from a fetched etymon row: the id (for the
# gloss/tag lookups) + every L2 column it projects. Derived from
# _ETYMON_L2_COLUMNS so a new L2 column can't drift out of the SELECT and
# KeyError at projection time (the natural key — canonical_form, language —
# rides along inside _ETYMON_L2_COLUMNS).
_ETYMON_STATE_SELECT_COLUMNS = ("e.id AS etymon_id", *(f"e.{col}" for col in _ETYMON_L2_COLUMNS))


def _etymon_state_row(conn: sqlite3.Connection, e: sqlite3.Row) -> dict[str, Any]:
    """Build one ``etymon`` canonical-state row (ref + L2 columns + glosses +
    tags) from a fetched etymon row that includes ``etymon_id`` and the
    columns in :data:`_ETYMON_STATE_SELECT_COLUMNS`."""
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
    row: dict[str, Any] = {"_type": "etymon", "ref": etymon_ref(e["language"], e["canonical_form"])}
    row.update(_drop_nulls({col: e[col] for col in _ETYMON_L2_COLUMNS}))
    if glosses:
        row["glosses"] = glosses
    if tags:
        row["tags"] = tags
    return row


def _dump_cited_etymons(conn: sqlite3.Connection, source_id: str) -> Iterable[dict[str, Any]]:
    """One ``etymon`` canonical-state row per etymon REFERENCED by this
    source — whether via citation, descent edge, or etymology element.

    Each source's JSONL file is self-contained: every etymon ref it
    uses (in any list-type row) gets a corresponding canonical-state
    etymon row in the same file. Build's cross-file merge unifies
    duplicates across sources.

    Resolves each referenced etymon through ``merged_into_id`` to its
    surviving winner (``COALESCE(ref.merged_into_id, ref.id)``) — matching
    ``_dump_fantasy_etymons`` (wyrd-ruvk) — so a reference to an OCR-cluster
    loser dumps the winner, never the tombstone. ``build_from_jsonl`` does
    not carry ``merged_into_id`` (absent from ``_ETYMON_INSERT_COLUMNS``),
    so emitting a loser would resurrect it as a live unmerged etymon on
    rebuild (D22: no merged etymon resurfaces).

    The single-hop COALESCE assumes ``merged_into_id`` names a TERMINAL
    (unmerged) winner. That holds because (a) the OCR/collapse/bridge writers
    resolve merge targets to live etymons and self-flatten, and (b) the curated
    path's multi-hop chains (``loser → mid → winner``) are flattened by
    ``enrichment.flatten_merge_chains`` on every enrichment. The read-only
    ``dump-jsonl`` CLI can't enrich, so it VERIFIES this fail-loud via
    ``assert_merge_chains_flat`` rather than silently emitting a mid-tombstone
    (wyrd-lpxq). Latent-bug origin: a cited etymon becoming a merge loser
    (wyrd-q6ro)."""
    etymons = conn.execute(
        f"""
        SELECT DISTINCT {", ".join(_ETYMON_STATE_SELECT_COLUMNS)}
          FROM etymon e
         WHERE e.id IN (
                SELECT COALESCE(ref.merged_into_id, ref.id)
                  FROM etymon ref
                 WHERE ref.id IN (
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
         )
         ORDER BY e.language, e.canonical_form
        """,  # noqa: S608 — interpolated names are module constants
        (source_id, source_id, source_id, source_id),
    ).fetchall()
    for e in etymons:
        yield _etymon_state_row(conn, e)


def _dump_citations(conn: sqlite3.Connection, source_id: str) -> Iterable[dict[str, Any]]:
    # Resolve the cited etymon through merged_into_id to its surviving winner
    # so the emitted etymon_ref matches the (winner) row _dump_cited_etymons
    # emits. OCR clustering keeps the citation attached to the loser (D22,
    # non-destructive), so without this follow-to-winner a re-dump after a
    # merge would emit a citation referencing the loser — which no etymon row
    # carries post-fix, orphaning the witness on rebuild (D21 evidence loss).
    # Single-hop COALESCE — relies on the terminal-winner invariant (enrichment
    # flattens; dump-jsonl asserts it fail-loud); see _dump_cited_etymons /
    # wyrd-lpxq. wyrd-q6ro.
    citations = conn.execute(
        """
        SELECT w.language, w.canonical_form,
               c.page, c.short_quote, c.context_snippet
          FROM etymon_citation c
          JOIN etymon ref ON ref.id = c.etymon_id
          JOIN etymon w ON w.id = COALESCE(ref.merged_into_id, ref.id)
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
    # Resolve both endpoints through merged_into_id to their winners (same
    # follow-to-winner as _dump_cited_etymons / _dump_citations, wyrd-q6ro):
    # an endpoint that is an OCR-cluster loser would otherwise emit a ref no
    # etymon row carries post-fix, orphaning the edge on rebuild. When BOTH
    # endpoints collapse to the SAME winner (an intra-cluster edge between two
    # OCR variants of one morpheme), the edge degenerates to a self-loop and
    # is skipped — it carries no real descent signal, and build's
    # _insert_descent has no self-edge guard. Single-hop COALESCE relies on the
    # terminal-winner invariant (see _dump_cited_etymons / wyrd-lpxq); a chain
    # whose endpoints flatten to one winner still self-loops correctly here.
    edges = conn.execute(
        """
        SELECT pe.language AS p_lang, pe.canonical_form AS p_form,
               ce.language AS c_lang, ce.canonical_form AS c_form,
               d.edge_type, d.confidence, d.notes
          FROM etymon_descent d
          JOIN etymon pref ON pref.id = d.parent_id
          JOIN etymon pe ON pe.id = COALESCE(pref.merged_into_id, pref.id)
          JOIN etymon cref ON cref.id = d.child_id
          JOIN etymon ce ON ce.id = COALESCE(cref.merged_into_id, cref.id)
         WHERE d.source_id = ?
           AND COALESCE(pref.merged_into_id, pref.id)
               != COALESCE(cref.merged_into_id, cref.id)
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
            # Resolve the element etymon through merged_into_id to its winner
            # (follow-to-winner, wyrd-q6ro): a loser element id would emit a
            # ref no etymon row carries post-fix, orphaning the element on
            # rebuild. Single-hop COALESCE — relies on the terminal-winner
            # invariant (see _dump_cited_etymons / wyrd-lpxq).
            """
            SELECT el.ordinal, el.inflection, el.confidence,
                   e.language, e.canonical_form
              FROM toponym_etymology_element el
              JOIN etymon ref ON ref.id = el.etymon_id
              JOIN etymon e ON e.id = COALESCE(ref.merged_into_id, ref.id)
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
                        # surface_in_modern is L3-derived (wyrd-ujyo:
                        # derive_surface_in_modern re-derives it on every
                        # enrichment), so it is NOT dumped — it would otherwise
                        # go stale in L2 like lemma_id / merged_into_id.
                        # wyrd-2n1: per-element confidence; round-trips
                        # alongside inflection. NULL elements drop the key
                        # entirely, matching the surrounding _drop_nulls pattern.
                        "confidence": el["confidence"],
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


def _source_for_attestation_doc(source_doc: str | None, known_source_ids: set[str]) -> str | None:  # noqa: V103 — test parity oracle, do not delete
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
    return rows


# Sources skipped by ``dump-jsonl`` by default, for two reasons:
#
# 1. Bulk wiktextract-derived sources whose contribution is re-derivable
#    from L1 raw inputs (``sources/wiktextract_*.jsonl``). Their dump
#    would produce huge files that don't belong in git — the build
#    pipeline re-runs the wiktextract ingester to recreate their rows.
#    NOTE: ``wiktionary-empirical`` is deliberately NOT here (wyrd-x33t) —
#    its re-mine is non-convergent, so it round-trips via its own
#    per-source ``wiktionary-empirical.jsonl`` like any L2 source.
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
        # wyrd-x33t: wiktionary-empirical was excluded as an L3-only re-mine
        # (mine-wiktextract-corpus). But that mine is NON-CONVERGENT — a clean
        # rebuild reproduces only ~1,870-2,676 of the backup's accumulated
        # ~3,682 citations, which shifts the global promotion landscape so OE
        # place-name elements (worð/worþ) stop clearing the gate (worth gate +
        # breton realism regress, wyrd-ruvk). So it now round-trips through L2
        # like reflexes (wyrd-ned5/br5o) + fantasy etymons (ruvk): dump_all_sources
        # emits data/mining/wiktionary-empirical.jsonl (etymons + citations) and
        # build_from_jsonl replays it. Its reflexes already ride _reflexes.jsonl.
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
        # wyrd-2b50: the Briggs personal-names ecosystem. The Briggs
        # JSONL is an ingester-owned committed artifact regenerated by
        # ``emit_briggs_jsonl`` (it carries cited_source rows + the
        # secondary citations dump-jsonl has no concept of), so a bulk
        # dump must not re-emit it — that would strip the cited_source
        # registration and the PASE/DLV/charter citations, and clobber
        # the hand-correct committed file. ``pase`` / ``dlv`` /
        # ``anglo_saxon_charters`` are secondary sources the Briggs file
        # cites ON BEHALF OF; they live in the post-rebuild ``source``
        # table but are owned by the Briggs file, so dumping them
        # standalone would promote them to competing primary-source
        # files. Same ingester-owns-the-artifact rule as fantasy-mining.
        "briggs_2024_personal_names_index",
        "pase",
        "dlv",
        "anglo_saxon_charters",
        # wyrd-bo01.2: gazetteer-converter-owned committed artifacts. Each
        # {culture}_place_names.jsonl is regenerated by `convert-place-names`
        # from the bundled gazetteer JSON and carries BARE toponym rows with
        # no etymology/attestation source-link — so a per-source dump would
        # emit ONLY the source row (zero toponyms, since the toponym table has
        # no source column) and clobber the committed file, dropping every
        # gazetteer toponym on the next rebuild. Same converter-owns-the-
        # artifact rule as os_open_names / fantasy-mining.
        "english_place_names",
        "scottish_place_names",
        "welsh_place_names",
        "irish_place_names",
        "breton_place_names",
        # wyrd-ned5: synthetic source owning the seed reflex layer at
        # data/mining/_reflexes.jsonl. The live DB has no "reflex-seed"
        # row in source; a rebuilt DB does (inserted from the JSONL).
        # Excluded so dump_all_sources doesn't emit a competing
        # per-source reflex-seed.jsonl — the reflex layer is dumped via
        # the dedicated dump_reflexes_to_file path instead.
        "reflex-seed",
        # wyrd-vewk: synthetic source owning the curated modern-reflex layer
        # at data/mining/_modern_reflexes.jsonl. Re-inserted into `source` by
        # `import-modern-reflexes` on rebuild; excluded so dump_all_sources
        # doesn't emit a competing per-source modern-reflex-curation.jsonl
        # (which would carry the etymons/edges WITHOUT the enrichment-derived
        # cognate_id and drift from the hand-curated file). Same
        # importer-owns-the-artifact rule as reflex-seed / fantasy-mining.
        "modern-reflex-curation",
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


def count_orphan_attestations(conn: sqlite3.Connection) -> int:
    """Count ``toponym_attestation`` rows whose ``source_doc`` routes
    nowhere — neither matches any ``source.id`` exactly nor starts with
    any registered prefix in :data:`_ATTESTATION_DOC_PREFIX_TO_SOURCE`.

    These rows are silently dropped by :func:`dump_all_sources` (no
    per-source file would carry them). wyrd-2t28: surface the count so
    a new ingester or a typo introducing un-routable source_docs shows
    up in operator output instead of disappearing after rebuild."""
    prefix_globs = [p + "*" for p, _sid in _ATTESTATION_DOC_PREFIX_TO_SOURCE]
    clauses = ["ta.source_doc IS NOT NULL", "ta.source_doc NOT IN (SELECT id FROM source)"]
    params: list[Any] = []
    if prefix_globs:
        # Exclude rows whose source_doc matches a registered prefix —
        # those are routed to a real source, not orphans.
        prefix_check = " AND ".join(["ta.source_doc NOT GLOB ?"] * len(prefix_globs))
        clauses.append(prefix_check)
        params.extend(prefix_globs)
    where = " AND ".join(clauses)
    row = conn.execute(
        f"SELECT COUNT(*) FROM toponym_attestation ta WHERE {where}",  # noqa: S608
        params,
    ).fetchone()
    return int(row[0])


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

    Emits a stderr warning via :mod:`logging` when one or more
    ``toponym_attestation`` rows would be silently dropped because
    their ``source_doc`` routes nowhere (wyrd-2t28). Programmatic
    callers can read the count back via :func:`count_orphan_attestations`.
    """
    counts: dict[str, int] = {}
    for sid in list_source_ids(conn, exclude=exclude):
        _, n = dump_source_to_file(conn, sid, out_dir)
        counts[sid] = n
    orphan_count = count_orphan_attestations(conn)
    if orphan_count > 0:
        # Build a SQL that mirrors count_orphan_attestations exactly —
        # without the NOT GLOB excludes, the suggested SELECT returns
        # every prefix-routed row alongside the true orphans, drowning
        # the actual signal.
        prefix_excludes = " ".join(
            f"AND source_doc NOT GLOB '{p}*'" for p, _sid in _ATTESTATION_DOC_PREFIX_TO_SOURCE
        )
        _logger.warning(
            "dump_all_sources: %d toponym_attestation row(s) have "
            "source_doc that routes nowhere — neither matches a source.id "
            "exactly nor starts with a registered prefix in "
            "_ATTESTATION_DOC_PREFIX_TO_SOURCE. These rows will not appear "
            "in any per-source JSONL and will be lost on rebuild. Inspect "
            "with: SELECT DISTINCT source_doc FROM toponym_attestation "
            "WHERE source_doc IS NOT NULL "
            "AND source_doc NOT IN (SELECT id FROM source) %s.",
            orphan_count,
            prefix_excludes,
        )
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
# The scalar surface dumped per fantasy_morpheme — single-sourced in _l2_columns so
# it can't drift from build's inserter (see _FANTASY_MORPHEME_INSERT_COLUMNS).
_FANTASY_MORPHEME_SCALAR_COLUMNS: tuple[str, ...] = _FANTASY_MORPHEME_COLUMNS


def _dump_fantasy_morpheme_rows(conn: sqlite3.Connection) -> Iterable[dict[str, Any]]:
    """Yield one ``fantasy_morpheme`` list row per DB row, ordered by
    ``input_name`` for diff stability. Joins ``etymon`` to project
    ``etymon_id`` → ``etymon_ref``; rows with NULL etymon_id (barred
    morphemes that never resolved a corpus etymon) emit no
    ``etymon_ref`` field — replaying skips the JOIN on build.

    wyrd-ruvk: when ``etymon_id`` points to an OCR-cluster *loser*
    (``merged_into_id NOT NULL`` — e.g. a diacritic variant ``latin:aeon``
    merged into ``latin:æon``), the ref follows ``merged_into_id`` to the
    surviving winner. This keeps the ref off a tombstone (so
    :func:`_dump_fantasy_etymons` never emits a merged etymon to resurrect)
    while preserving the link. One COALESCE resolves the canonical etymon —
    relies on the terminal-winner invariant (enrichment flattens curated chains;
    dump-jsonl asserts it fail-loud); see _dump_cited_etymons / wyrd-lpxq."""
    rows = conn.execute(
        f"""
        SELECT {", ".join("fm." + c for c in _FANTASY_MORPHEME_SCALAR_COLUMNS)},
               w.language AS _etymon_language,
               w.canonical_form AS _etymon_canonical_form
          FROM fantasy_morpheme fm
          LEFT JOIN etymon e ON e.id = fm.etymon_id
          LEFT JOIN etymon w ON w.id = COALESCE(e.merged_into_id, e.id)
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


def _dump_fantasy_etymons(conn: sqlite3.Connection) -> Iterable[dict[str, Any]]:
    """Emit canonical-state ``etymon`` rows for every etymon a
    ``fantasy_morpheme`` references (``fm.etymon_id``).

    These etymons frequently carry ZERO citations — ``mine-fantasy-name``
    resolves a fantasy fragment to a morpheme and creates the etymon without a
    source attribution — so NO per-source dump emits them. Without this, a
    rebuild can't resolve the ``fantasy_morpheme.etymon_ref`` and drops the
    morpheme as an orphan (wyrd-ruvk; sibling of the wyrd-ned5/br5o reflex
    round-trip).

    Resolves each referenced etymon through ``merged_into_id`` to its surviving
    winner — matching the ref ``_dump_fantasy_morpheme_rows`` emits — so a
    fantasy_morpheme pointing at an OCR-cluster loser dumps the winner, never
    the tombstone (D22: no merged etymon resurfaces on rebuild). The winners are
    themselves unmerged. Build's cross-file merge unifies any etymon also cited
    by a real source."""
    etymons = conn.execute(
        f"""
        SELECT DISTINCT {", ".join(_ETYMON_STATE_SELECT_COLUMNS)}
          FROM etymon e
         WHERE e.id IN (
                SELECT COALESCE(ref.merged_into_id, ref.id)
                  FROM fantasy_morpheme fm
                  JOIN etymon ref ON ref.id = fm.etymon_id
                 WHERE fm.etymon_id IS NOT NULL
         )
         ORDER BY e.language, e.canonical_form
        """,  # noqa: S608 — interpolated names are module constants
    ).fetchall()
    for e in etymons:
        yield _etymon_state_row(conn, e)


def dump_fantasy_morphemes_to_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Yield the rows that ``_fantasy_morphemes.jsonl`` carries: the synthetic
    ``fantasy-mining`` source declaration, the canonical-state ``etymon`` rows
    the morphemes reference (wyrd-ruvk — these are otherwise uncited and carried
    by no source), then one ``fantasy_morpheme`` row per DB row.

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
    etymons = list(_dump_fantasy_etymons(conn))
    return [_FANTASY_MINING_SOURCE_ROW, *etymons, *morphemes]


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


# wyrd-ned5: seed reflex layer round-trip. The reflex / reflex_etymon
# tables map modern_usage surface fragments (-ton, -ham) to the etymons
# they descend from. The SEED reflexes (rando-port modern_usage) are
# created by seed_from_meanings and are NOT carried anywhere else in L2,
# so a full rebuild-from-jsonl drops them — the bug wyrd-ned5 fixes. (The
# place-name-derived reflexes that derive_positions writes during
# mine-wiktextract-corpus ARE re-derivable from the corpus, but dumping
# the whole layer is simpler and idempotent.) Dumping to a synthetic-
# source file (same pattern as _fantasy_morphemes.jsonl) lets the rebuild
# restore them.
_REFLEX_SEED_SOURCE_ID = "reflex-seed"
_REFLEX_SEED_FILENAME = "_reflexes.jsonl"
_REFLEX_SEED_SOURCE_ROW = {
    "_type": "source",
    "ref": _REFLEX_SEED_SOURCE_ID,
    "title": "Seed reflex layer (wyrd-ned5)",
    "notes": (
        "Synthetic source for the reflex / reflex_etymon L2 round-trip "
        "(wyrd-ned5). Rows map a modern_usage surface fragment to the "
        "etymons it descends from; etymon FKs are resolved on dump via "
        "etymon_ref and re-linked on rebuild. This source declaration is "
        "JSONL-only; the live DB has no reflex-seed row in source. "
        "productivity is NOT carried — it's recomputed from corpus "
        "observations at the rebuild tail (wyrd-14p)."
    ),
}


def _dump_reflex_rows(conn: sqlite3.Connection) -> Iterable[dict[str, Any]]:
    """Yield one ``reflex`` list row per (surface_form, position),
    folding every surviving linked etymon into a sorted ``etymon_refs``
    list (an orphan reflex with no surviving link emits ``etymon_refs: []``).

    Ordered by (surface_form, position) for diff-stable output. OCR-
    cluster loser etymons (merged_into_id NOT NULL) are excluded by the
    query so a tombstone doesn't resurface as a reflex target. ORPHAN
    reflexes (no surviving etymon link — the LEFT JOIN yields NULL etymon
    columns) emit with ``etymon_refs: []`` so the generation-surface layer
    round-trips too (wyrd-br5o)."""
    from ..lexicon.sql.queries.reflex import (  # lazy: keep dump import-time light
        SELECT_REFLEXES_FOR_DUMP,
    )

    current: dict[str, Any] | None = None
    key: tuple[str, str] | None = None
    for row in conn.execute(SELECT_REFLEXES_FOR_DUMP).fetchall():
        rkey = (row["surface_form"], row["position"])
        if rkey != key:
            if current is not None:
                yield current
            key = rkey
            current = {
                "_type": "reflex",
                "surface_form": row["surface_form"],
                "position": row["position"],
                "etymon_refs": [],
            }
        # An orphan reflex (no link, or its only link was a merged loser) yields a
        # NULL etymon row from the LEFT JOIN — keep the reflex, append no ref.
        if row["language"] is not None and row["canonical_form"] is not None:
            current["etymon_refs"].append(etymon_ref(row["language"], row["canonical_form"]))
    if current is not None:
        yield current


def dump_reflexes_to_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """The rows ``_reflexes.jsonl`` carries: the synthetic ``reflex-seed``
    source declaration followed by one ``reflex`` row per
    (surface_form, position).

    Returns ``[]`` (file not written) when the ``reflex`` table is absent
    (older DBs) or carries zero reflexes — so a DB that never
    seeded reflexes doesn't get an empty-but-meaningful artifact."""
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='reflex'"
    ).fetchone()
    if not table_exists:
        return []
    reflexes = list(_dump_reflex_rows(conn))
    if not reflexes:
        return []
    return [_REFLEX_SEED_SOURCE_ROW, *reflexes]


def dump_reflexes_to_file(
    conn: sqlite3.Connection,
    out_dir: str | Path,
) -> tuple[Path, int]:
    """Write ``<out_dir>/_reflexes.jsonl``. Returns ``(path, row_count)``;
    row_count is 0 (file not written) when the DB has no reflexes (linked or orphan)."""
    rows = dump_reflexes_to_rows(conn)
    path = Path(out_dir) / _REFLEX_SEED_FILENAME
    if not rows:
        return path, 0
    n = write_jsonl(path, rows)
    return path, n
