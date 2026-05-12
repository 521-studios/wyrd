"""L3 enrichment orchestrator — wyrd-ilam (Phase 2a).

After ``lexicon rebuild-from-jsonl`` produces a fresh SQLite from the
L2 JSONL source-of-truth, the derived columns on ``etymon``
(``lemma_id``, ``inflection``, ``lemma_method``, ``merged_into_id``,
``cognate_id``, ``stratum``, ``english_shaped``) are NULL — the dump
explicitly excludes them per the L2/L3 boundary contract. Operators
re-derive them by running the enrichment passes documented in
``L2_L3_BOUNDARY.md``.

This module ships the first migration in that pattern: an orchestrator
:func:`run_ocr_lemma_enrichment` that runs ``normalize-ocr`` (OCR
variant clustering) followed by ``link-lemmas`` (inflected → lemma
linkage) in the canonical order. Order matters: link-lemmas needs
canonical etymons as targets, so OCR-cluster has to tombstone the
losers first.

Future PRs extend this to cluster-cognates, classify-stratum, etc.;
each pass keeps its existing standalone CLI command for operators who
want fine-grained control.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .lexicon import LexiconDB, cluster_ocr_variants, link_lemmas

# Method-version provenance currently lives in code (the writer-side
# UPDATE statements stamp these strings). Surfacing them here in one
# place makes :func:`enrichment_status` reports + L2_L3_BOUNDARY.md
# stay in sync.
OCR_METHOD_VERSION = "cluster-ocr-v1"
LEMMA_METHOD_VERSION = "link-lemmas-v1"


def run_ocr_lemma_enrichment(db: LexiconDB, *, apply: bool = False) -> dict[str, Any]:
    """Run OCR clustering then lemma linkage in canonical order.

    Returns a flat dict combining both passes' result counts plus an
    explicit ``order`` field so callers can verify nothing got swapped.

    Dry-run (``apply=False``) walks both passes and reports candidates
    without writing. Apply mode commits each pass's writes via
    ``LexiconDB`` before moving to the next — so a partial run leaves
    the DB in a coherent intermediate state (OCR-merged but not yet
    lemma-linked is meaningful).
    """
    ocr_result = cluster_ocr_variants(db, apply=apply)
    lemma_result = link_lemmas(db, apply=apply)
    return {
        "order": ["normalize-ocr", "link-lemmas"],
        "applied": apply,
        "ocr": {
            "method_version": OCR_METHOD_VERSION,
            "groups": ocr_result["groups"],
            "etymons_merged": ocr_result["etymons_merged"],
            "sample_groups": ocr_result.get("sample_groups", []),
        },
        "lemmas": {
            "method_version": LEMMA_METHOD_VERSION,
            "candidates": lemma_result["candidates"],
            "sample": lemma_result.get("sample", []),
        },
    }


def enrichment_status(conn: sqlite3.Connection) -> dict[str, Any]:
    """Report which L3 enrichment columns are populated on ``etymon``.

    Per-column coverage: how many rows have the column set vs total
    etymon rows. Pulls method-version distributions for the two
    columns that carry one (``lemma_method``, ``cognate_method``).

    All counts come from a single aggregated SELECT — one table scan
    on a 2M+ row table instead of six. ``COUNT(col)`` in SQLite
    counts only non-NULL values, so each ``populated`` figure is one
    column in the same aggregate. The GROUP BY method-distribution
    queries are separate (one per method-bearing column) but use
    hardcoded column names, never operator input — no SQL injection
    surface.

    Read-only against the DB. Temporarily swaps ``conn.row_factory``
    to ``sqlite3.Row`` so column-name access works regardless of the
    caller's factory, then restores the original via try/finally —
    the connection's state is preserved across the call.
    """
    original_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        counts = conn.execute(
            """SELECT COUNT(*)              AS total,
                      COUNT(lemma_id)       AS lemma_id_pop,
                      COUNT(merged_into_id) AS merged_into_id_pop,
                      COUNT(cognate_id)     AS cognate_id_pop,
                      COUNT(stratum)        AS stratum_pop,
                      COUNT(english_shaped) AS english_shaped_pop
                 FROM etymon"""
        ).fetchone()

        lemma_methods = [
            {"method": r["lemma_method"], "count": r["count"]}
            for r in conn.execute(
                "SELECT lemma_method, COUNT(*) AS count FROM etymon "
                "WHERE lemma_method IS NOT NULL "
                "GROUP BY lemma_method ORDER BY count DESC"
            )
        ]
        cognate_methods = [
            {"method": r["cognate_method"], "count": r["count"]}
            for r in conn.execute(
                "SELECT cognate_method, COUNT(*) AS count FROM etymon "
                "WHERE cognate_method IS NOT NULL "
                "GROUP BY cognate_method ORDER BY count DESC"
            )
        ]
    finally:
        conn.row_factory = original_factory

    return {
        "total_etymons": counts["total"],
        "columns": {
            "lemma_id": {
                "populated": counts["lemma_id_pop"],
                "methods": lemma_methods,
            },
            "merged_into_id": {
                "populated": counts["merged_into_id_pop"],
                # cluster_ocr_variants doesn't write a per-row method
                # version (the 'cluster-ocr-v1' string is implicit in
                # the writer). When that changes the methods list grows.
                "methods": [],
            },
            "cognate_id": {
                "populated": counts["cognate_id_pop"],
                "methods": cognate_methods,
            },
            "stratum": {
                "populated": counts["stratum_pop"],
                "methods": [],
            },
            "english_shaped": {
                "populated": counts["english_shaped_pop"],
                "methods": [],
            },
        },
    }


def format_enrichment_status(status: dict[str, Any]) -> str:
    """Render :func:`enrichment_status` as a grep-friendly markdown
    report."""
    total = status["total_etymons"]
    lines: list[str] = [
        "## L3 enrichment status",
        "",
        f"- Total etymons: {total}",
        "",
        "### Column coverage",
    ]
    for column, info in status["columns"].items():
        pct = (info["populated"] / total * 100.0) if total else 0.0
        lines.append(f"- `{column}`: {info['populated']:>8} / {total} ({pct:5.1f}%)")
        for m in info.get("methods", []):
            lines.append(f"  - method=`{m['method']}`: {m['count']}")
    return "\n".join(lines)


def format_enrichment_run(result: dict[str, Any]) -> str:
    """Render :func:`run_ocr_lemma_enrichment` output as markdown."""
    verb_ocr = "merged" if result["applied"] else "mergeable"
    verb_lemmas = "linked" if result["applied"] else "linkable"
    lines: list[str] = [
        "## L3 enrichment: OCR + lemmas",
        "",
        f"- Order: {' → '.join(result['order'])}",
        f"- Applied: {result['applied']}",
        "",
        "### OCR clustering (`" + result["ocr"]["method_version"] + "`)",
        f"- Variant groups: {result['ocr']['groups']}",
        f"- Etymons {verb_ocr}: {result['ocr']['etymons_merged']}",
        "",
        "### Lemma linkage (`" + result["lemmas"]["method_version"] + "`)",
        f"- Inflected etymons {verb_lemmas}: {result['lemmas']['candidates']}",
    ]
    return "\n".join(lines)
