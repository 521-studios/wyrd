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
    etymon rows. Pulls method-version distributions where they're
    stored (lemma_method, cognate_method).

    Read-only — does not write or modify state.
    """
    total = conn.execute("SELECT COUNT(*) AS n FROM etymon").fetchone()["n"]

    def _populated(column: str) -> int:
        return conn.execute(
            f"SELECT COUNT(*) AS n FROM etymon WHERE {column} IS NOT NULL"
        ).fetchone()["n"]

    def _method_distribution(column: str) -> list[dict[str, Any]]:
        return [
            {"method": r[column], "count": r["count"]}
            for r in conn.execute(
                f"SELECT {column}, COUNT(*) AS count FROM etymon "
                f"WHERE {column} IS NOT NULL GROUP BY {column} ORDER BY count DESC"
            )
        ]

    return {
        "total_etymons": total,
        "columns": {
            "lemma_id": {
                "populated": _populated("lemma_id"),
                "methods": _method_distribution("lemma_method"),
            },
            "merged_into_id": {
                "populated": _populated("merged_into_id"),
                # cluster_ocr_variants doesn't write a per-row method
                # version (the 'cluster-ocr-v1' string is implicit in
                # the writer). When that changes the methods list grows.
                "methods": [],
            },
            "cognate_id": {
                "populated": _populated("cognate_id"),
                "methods": _method_distribution("cognate_method"),
            },
            "stratum": {
                "populated": _populated("stratum"),
                "methods": [],
            },
            "english_shaped": {
                "populated": _populated("english_shaped"),
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
