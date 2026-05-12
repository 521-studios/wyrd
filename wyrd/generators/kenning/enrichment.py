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

# Stamped on etymon rows whose lemma_id was set by a curation override
# (wyrd-2jhs) rather than the auto-clustering link_lemmas pass. Keeps
# the L3 provenance distinguishable: a future audit can find every
# operator-curated row by querying ``lemma_method='manual-curation-v1'``.
CURATION_METHOD_VERSION = "manual-curation-v1"


def _resolve_etymon_id(conn: sqlite3.Connection, ref: str) -> int | None:
    """Resolve ``"<language>:<canonical_form>"`` → etymon.id. Returns
    None when the ref doesn't match any row."""
    if ":" not in ref:
        return None
    lang, form = ref.split(":", 1)
    row = conn.execute(
        "SELECT id FROM etymon WHERE language = ? AND canonical_form = ?",
        (lang, form),
    ).fetchone()
    return row["id"] if row else None


def _build_ref_resolver(conn: sqlite3.Connection):
    """Memoized ref → etymon_id resolver. Multiple curation events
    often target the same lemma (e.g. dozens of inflected forms
    pointing at a single canonical lemma), so caching saves the
    redundant SELECT round-trips."""
    cache: dict[str, int | None] = {}

    def resolve(ref: str) -> int | None:
        if ref not in cache:
            cache[ref] = _resolve_etymon_id(conn, ref)
        return cache[ref]

    return resolve


def apply_curation_overrides(
    db: LexiconDB,
    curation_state: dict[str, dict[str, Any]],
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Apply operator curation overrides to L3 enrichment columns (wyrd-2jhs).

    ``curation_state`` is the merged ``{etymon_ref: payload}`` dict
    from :func:`jsonl_build.collect_curation_overrides`. For each
    curated etymon:

    - ``lemma_ref`` set → ``lemma_id`` pointed at that etymon, plus
      ``lemma_method='manual-curation-v1'``. Also clears
      ``merged_into_id`` (the etymon stays canonical; its inflection
      target is the curated lemma, not a tombstone).
    - ``inflection`` set → written alongside ``lemma_id``. Required
      when ``lemma_ref`` is set; otherwise the inflection column would
      lie about what's there.
    - ``merged_into_ref`` set → ``merged_into_id`` pointed at that
      etymon (OCR-cluster tombstone). Clears ``lemma_id`` so the
      tombstone doesn't claim a lemma role.
    - Explicit ``null`` on any of those refs clears the corresponding
      column.

    Run AFTER the auto-clustering passes (normalize-ocr + link-lemmas)
    so curation overrides their output. Bad curation events
    (unresolved refs, self-references) are logged in the counts dict
    and skipped — never raise, so a stale ref doesn't block the
    rebuild.

    Validation always runs (resolves refs, counts unresolved /
    self-references); DB writes only happen when ``apply=True``.
    Dry-run mode is useful for operators previewing a curation batch
    before committing.

    Self-reference guard: an etymon can't be its own lemma_id or
    merged_into_id target. Such curation events would create cycles
    (e.g. ``find_canonical(x)`` resolving infinitely). Counted under
    ``self_reference_lemma`` / ``self_reference_merge`` and skipped.

    Returns a counts dict for telemetry.
    """
    counts = {
        "curations": len(curation_state),
        "lemma_id_set": 0,
        "lemma_id_cleared": 0,
        "merged_into_set": 0,
        "merged_into_cleared": 0,
        "unresolved_etymon": 0,
        "unresolved_lemma_ref": 0,
        "unresolved_merge_ref": 0,
        "self_reference_lemma": 0,
        "self_reference_merge": 0,
        "applied": apply,
    }
    resolve = _build_ref_resolver(db.conn)

    for etymon_ref, payload in curation_state.items():
        etymon_id = resolve(etymon_ref)
        if etymon_id is None:
            counts["unresolved_etymon"] += 1
            continue

        # lemma_ref handling: present key (even with None value) means
        # "operator made a decision about lemma_id"; absent key means
        # "no opinion, leave whatever auto-clustering chose alone".
        # Both ref fields process independently within a payload — a
        # failed lemma_ref resolution shouldn't skip merged_into_ref
        # in the same event.
        if "lemma_ref" in payload:
            lemma_ref = payload["lemma_ref"]
            inflection = payload.get("inflection")
            if lemma_ref is None:
                counts["lemma_id_cleared"] += 1
                if apply:
                    db.conn.execute(
                        "UPDATE etymon SET lemma_id = NULL, inflection = NULL, "
                        "lemma_method = NULL WHERE id = ?",
                        (etymon_id,),
                    )
            else:
                lemma_id = resolve(lemma_ref)
                if lemma_id is None:
                    counts["unresolved_lemma_ref"] += 1
                elif lemma_id == etymon_id:
                    counts["self_reference_lemma"] += 1
                else:
                    counts["lemma_id_set"] += 1
                    if apply:
                        # Curation winners stay canonical even if auto-
                        # clustering would have tombstoned them — clear
                        # merged_into_id when lemma_id is operator-set.
                        db.conn.execute(
                            "UPDATE etymon SET lemma_id = ?, inflection = ?, "
                            "lemma_method = ?, merged_into_id = NULL WHERE id = ?",
                            (lemma_id, inflection, CURATION_METHOD_VERSION, etymon_id),
                        )

        if "merged_into_ref" in payload:
            merge_ref = payload["merged_into_ref"]
            if merge_ref is None:
                counts["merged_into_cleared"] += 1
                if apply:
                    db.conn.execute(
                        "UPDATE etymon SET merged_into_id = NULL WHERE id = ?",
                        (etymon_id,),
                    )
            else:
                merge_id = resolve(merge_ref)
                if merge_id is None:
                    counts["unresolved_merge_ref"] += 1
                elif merge_id == etymon_id:
                    counts["self_reference_merge"] += 1
                else:
                    counts["merged_into_set"] += 1
                    if apply:
                        # Tombstone target: clear lemma_id (a tombstone
                        # shouldn't double as a lemma parent).
                        db.conn.execute(
                            "UPDATE etymon SET merged_into_id = ?, "
                            "lemma_id = NULL, inflection = NULL, "
                            "lemma_method = NULL WHERE id = ?",
                            (merge_id, etymon_id),
                        )

    if apply:
        db.commit()
    return counts


def format_curation_run(counts: dict[str, Any]) -> str:
    """Render :func:`apply_curation_overrides` output as markdown."""
    applied = counts.get("applied", True)
    verb_set = "set" if applied else "would set"
    verb_clear = "cleared" if applied else "would clear"
    lines: list[str] = [
        "## L3 enrichment: curation overrides",
        "",
        f"- Method version: `{CURATION_METHOD_VERSION}`",
        f"- Curations processed: {counts['curations']}",
        f"- Lemma overrides {verb_set}: {counts['lemma_id_set']}",
        f"- Lemma overrides {verb_clear}: {counts['lemma_id_cleared']}",
        f"- Merge overrides {verb_set}: {counts['merged_into_set']}",
        f"- Merge overrides {verb_clear}: {counts['merged_into_cleared']}",
    ]
    unresolved = (
        counts["unresolved_etymon"]
        + counts["unresolved_lemma_ref"]
        + counts["unresolved_merge_ref"]
    )
    if unresolved:
        lines.append("")
        lines.append("### Unresolved refs (skipped)")
        if counts["unresolved_etymon"]:
            lines.append(f"- Curated etymon not found: {counts['unresolved_etymon']}")
        if counts["unresolved_lemma_ref"]:
            lines.append(f"- Lemma target not found: {counts['unresolved_lemma_ref']}")
        if counts["unresolved_merge_ref"]:
            lines.append(f"- Merge target not found: {counts['unresolved_merge_ref']}")
    self_refs = counts.get("self_reference_lemma", 0) + counts.get("self_reference_merge", 0)
    if self_refs:
        lines.append("")
        lines.append("### Self-references (skipped)")
        if counts.get("self_reference_lemma"):
            lines.append(f"- Etymon would be its own lemma: {counts['self_reference_lemma']}")
        if counts.get("self_reference_merge"):
            lines.append(f"- Etymon would tombstone to itself: {counts['self_reference_merge']}")
    return "\n".join(lines)


def run_ocr_lemma_enrichment(
    db: LexiconDB,
    *,
    apply: bool = False,
    curation_state: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run OCR clustering, lemma linkage, and (optional) curation
    overrides in canonical order.

    Order:
    1. ``normalize-ocr`` tombstones OCR-variant duplicates.
    2. ``link-lemmas`` links inflected forms to canonicals (the
       previous step's tombstones aren't candidate targets).
    3. ``apply_curation_overrides`` (when ``curation_state`` is passed)
       overrides specific lemma/merge assignments where auto-clustering
       got it wrong.

    Dry-run (``apply=False``) walks the enrichment passes and reports
    candidates without writing. Curation overrides, when
    ``curation_state`` is passed, ARE validated on dry-run — refs
    resolve, unresolved/self-reference cases get counted — but no DB
    writes occur. Lets operators preview a curation batch before
    committing.

    When ``curation_state`` is None, the returned dict carries
    ``curation: None`` (the apply-curation step didn't run). When it's
    a dict, the returned ``curation`` value is the counts dict from
    :func:`apply_curation_overrides` regardless of ``apply``.
    """
    ocr_result = cluster_ocr_variants(db, apply=apply)
    lemma_result = link_lemmas(db, apply=apply)
    curation_counts: dict[str, Any] | None = None
    if curation_state is not None:
        # Always run validation + counting; only writes are gated by apply.
        # Lets `lexicon enrich` (dry-run) preview curation issues without
        # committing — surfaces unresolved refs / self-references early.
        curation_counts = apply_curation_overrides(db, curation_state, apply=apply)

    return {
        "order": (
            ["normalize-ocr", "link-lemmas", "apply-curation"]
            if curation_counts is not None
            else ["normalize-ocr", "link-lemmas"]
        ),
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
        "curation": curation_counts,
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
        "## L3 enrichment: OCR + lemmas" + (" + curation" if result.get("curation") else ""),
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
    if result.get("curation"):
        lines.append("")
        lines.append(format_curation_run(result["curation"]))
    return "\n".join(lines)
