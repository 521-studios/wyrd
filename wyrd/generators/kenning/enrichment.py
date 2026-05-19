"""L3 enrichment orchestrator — wyrd-ilam → wyrd-hidb.

After ``lexicon rebuild-from-jsonl`` produces a fresh SQLite from the
L2 JSONL source-of-truth, the derived columns on ``etymon``
(``lemma_id``, ``inflection``, ``lemma_method``, ``merged_into_id``,
``cognate_id``, ``stratum``, ``english_shaped``) plus the derived
tables (``toponym_decomposition``, ``etymon_period_form``) are empty
— the dump explicitly excludes them per the L2/L3 boundary contract.
Operators re-derive them by running the enrichment passes documented
in ``L2_L3_BOUNDARY.md``.

:func:`run_full_enrichment` is the canonical orchestrator. It walks
the L3 chain in dependency order:

    cluster-ocr → link-lemmas → apply-curation → decompose →
    cluster-cognates → classify-stratum → derive-english-shaped →
    project-period-forms

Each pass keeps its existing standalone CLI command for operators
who want fine-grained control; the wrappers
(:func:`decomposition.decompose_all`,
:func:`english_shaping.derive_english_shaped_all`,
:func:`strata.classify_stratum_all`) are the uniform
``(db, *, apply)`` shape this module's orchestrator can call directly.

``skip_l3_derivations=True`` stops after the OCR+lemma+curation
prefix — useful for fast curation iteration + for the focused
synthetic-DB enrichment / curation tests that don't seed the
toponym / descent rows the downstream passes scan.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .decomposition import decompose_all
from .lexicon import (
    LexiconDB,
    cluster_cognates,
    cluster_ocr_variants,
    link_lemmas,
    project_period_forms,
)
from .lexicon.english_shaping import derive_english_shaped_all
from .lexicon.strata import classify_stratum_all

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
    from :func:`jsonl.build.collect_curation_overrides`. For each
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
            if lemma_ref is None:
                counts["lemma_id_cleared"] += 1
                if apply:
                    # Stamp manual-curation-v1 even on clear so audit
                    # queries (WHERE lemma_method = 'manual-curation-v1')
                    # find the operator decision. lemma_id=NULL with a
                    # method version means "operator decided no lemma".
                    db.conn.execute(
                        "UPDATE etymon SET lemma_id = NULL, inflection = NULL, "
                        "lemma_method = ? WHERE id = ?",
                        (CURATION_METHOD_VERSION, etymon_id),
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
                        # "Absent inflection key = no opinion": only touch
                        # the column when the operator explicitly provided
                        # it. Otherwise an auto-detected inflection from
                        # link-lemmas survives the curation. Dynamic SET
                        # list keeps the semantic clean.
                        set_clauses = [
                            "lemma_id = ?",
                            "lemma_method = ?",
                            "merged_into_id = NULL",
                        ]
                        values: list[Any] = [lemma_id, CURATION_METHOD_VERSION]
                        if "inflection" in payload:
                            set_clauses.append("inflection = ?")
                            values.append(payload["inflection"])
                        values.append(etymon_id)
                        db.conn.execute(
                            f"UPDATE etymon SET {', '.join(set_clauses)} WHERE id = ?",
                            tuple(values),
                        )

        if "merged_into_ref" in payload:
            merge_ref = payload["merged_into_ref"]
            if merge_ref is None:
                counts["merged_into_cleared"] += 1
                if apply:
                    # Stamp method on clear so the operator decision is
                    # findable via the same WHERE lemma_method query
                    # used for lemma curations.
                    db.conn.execute(
                        "UPDATE etymon SET merged_into_id = NULL, lemma_method = ? WHERE id = ?",
                        (CURATION_METHOD_VERSION, etymon_id),
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
                        # shouldn't double as a lemma parent). Stamp the
                        # method so curation-tombstones are auditable.
                        db.conn.execute(
                            "UPDATE etymon SET merged_into_id = ?, "
                            "lemma_id = NULL, inflection = NULL, "
                            "lemma_method = ? WHERE id = ?",
                            (merge_id, CURATION_METHOD_VERSION, etymon_id),
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


def run_full_enrichment(
    db: LexiconDB,
    *,
    apply: bool = False,
    curation_state: dict[str, dict[str, Any]] | None = None,
    skip_l3_derivations: bool = False,
) -> dict[str, Any]:
    """Run the canonical L3 enrichment chain (wyrd-hidb Phase 2).

    Order (with dependency rationale per the wyrd-hidb plan):

    1. ``cluster_ocr_variants`` — tombstone OCR-variant duplicates so
       later passes don't waste work on dead rows.
    2. ``link_lemmas`` — inflected → canonical-lemma linkage. Must
       follow OCR (lemma candidates can't be tombstoned variants).
    3. ``apply_curation_overrides`` (when ``curation_state`` passed)
       — overlay operator decisions on lemma / merge assignments.
    4. ``decompose_all`` — matcher-derived toponym decompositions.
       Matches the toponym's modern_name against the live
       ``meanings.json`` bundle, so it doesn't strictly need OCR /
       lemma settled — but running it after the auto-clustering +
       curation passes means the etymon ids the canonical breakdowns
       reference correspond to the post-cluster canonical rows, not
       tombstoned ones.
    5. ``cluster_cognates`` — walks the ``etymon_descent`` graph
       from each Proto-* root and assigns ``cognate_id`` to every
       reachable descendant. The OCR pass populates
       ``merged_into_id`` (tombstone pointer), which cluster_cognates
       respects when picking the canonical etymon for each cluster;
       so OCR must precede this step.
    6. ``classify_stratum_all`` — per-language stratum buckets. Reads
       ``etymon.language`` + the ``etymon_descent`` parent edges
       populated by L2 replay; idempotent under the
       ``stratum IS NULL`` gate.
    7. ``derive_english_shaped_all`` — non-Latin-script
       transliteration. Reads ``etymon.canonical_form`` +
       ``transliteration`` + ``pronunciation_ipa``; independent of the
       earlier passes' column writes, but placed after them so a
       round-trip rebuild has stable canonical_form values by the
       time english_shaped derivation runs.
    8. ``project_period_forms`` — wyrd-unuo Phase 3.3: Tier-3
       period-form fallback. Reads ``toponym_attestation`` rows + the
       cognate_id assignments from step 5 (cluster mates contribute
       suffix candidates), so cluster_cognates must precede this
       step.

    Dry-run (``apply=False``) walks every pass and reports
    candidates without writing. Curation overrides ARE validated on
    dry-run; the per-pass dry-run behavior is delegated to each
    helper.

    ``skip_l3_derivations=True`` stops the chain after the
    OCR+lemma+curation prefix — useful for tests that don't care
    about the derived columns + for fast iteration on curation
    work. Defaults to False (Phase 2 default is "run everything").
    """
    ocr_result = cluster_ocr_variants(db, apply=apply)
    lemma_result = link_lemmas(db, apply=apply)
    curation_counts: dict[str, Any] | None = None
    if curation_state is not None:
        # Always run validation + counting; only writes are gated by apply.
        # Lets `lexicon enrich` (dry-run) preview curation issues without
        # committing — surfaces unresolved refs / self-references early.
        curation_counts = apply_curation_overrides(db, curation_state, apply=apply)

    order: list[str] = ["normalize-ocr", "link-lemmas"]
    if curation_counts is not None:
        order.append("apply-curation")

    decompose_result: dict[str, Any] | None = None
    cognate_result: dict[str, Any] | None = None
    stratum_result: dict[str, Any] | None = None
    english_shaped_result: dict[str, Any] | None = None
    period_form_result: dict[str, Any] | None = None

    if not skip_l3_derivations:
        decompose_result = decompose_all(db, apply=apply)
        order.append("decompose")
        cognate_result = cluster_cognates(db, apply=apply)
        order.append("cluster-cognates")
        stratum_result = classify_stratum_all(db, apply=apply)
        order.append("classify-stratum")
        english_shaped_result = derive_english_shaped_all(db, apply=apply)
        order.append("derive-english-shaped")
        period_form_result = project_period_forms(db, apply=apply)
        order.append("project-period-forms")

    return {
        "order": order,
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
        "decompose": decompose_result,
        "cognates": cognate_result,
        "stratum": stratum_result,
        "english_shaped": english_shaped_result,
        "period_forms": period_form_result,
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
    """Render :func:`run_full_enrichment` output as markdown."""
    verb_ocr = "merged" if result["applied"] else "mergeable"
    verb_lemmas = "linked" if result["applied"] else "linkable"
    lines: list[str] = [
        "## L3 enrichment chain",
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

    if result.get("decompose") is not None:
        d = result["decompose"]
        lines.extend(
            [
                "",
                "### Decomposition",
                f"- Toponyms scanned: {d['toponyms_scanned']}",
                f"- Decompositions written: {d['decompositions']}",
                f"- Rule firings: scholar={d.get('scholar', 0)}, "
                f"scholar-disagreement={d.get('scholar-disagreement', 0)}, "
                f"unique-zero={d.get('unique-zero-unaccounted', 0)}, "
                f"tiebreaker={d.get('tiebreaker', 0)}, "
                f"no-canonical={d.get('no-canonical', 0)}",
            ]
        )
    if result.get("cognates") is not None:
        c = result["cognates"]
        lines.extend(
            [
                "",
                "### Cognate clustering",
                f"- Roots walked: {c.get('roots', 0)}",
                f"- Cognate_id assignments: {c.get('candidates', 0)}",
            ]
        )
    if result.get("stratum") is not None:
        s = result["stratum"]
        lines.append("")
        lines.append("### Stratum classification")
        for lang, counts in s["languages"].items():
            lines.append(
                f"- {lang}: proposed={counts['proposed']}, "
                f"written={counts.get('written', 0)}, "
                f"skipped={counts.get('skipped', 0)}"
            )
            # Stratum distribution: same shape as the rule-firings line
            # in the decomposition section. Sorted for deterministic
            # report output (the underlying dict is built by
            # classify_stratum_all in proposal-iteration order).
            by_stratum = counts.get("by_stratum") or {}
            if by_stratum:
                breakdown = ", ".join(f"{stratum}={n}" for stratum, n in sorted(by_stratum.items()))
                lines.append(f"  - by stratum: {breakdown}")
    if result.get("english_shaped") is not None:
        e = result["english_shaped"]
        lines.extend(
            [
                "",
                "### English-shaped derivation",
                f"- Candidates: {e['candidates']}",
                f"- Written: {e['written']}",
                f"- Skipped (no input): {e['skipped_no_input']}",
            ]
        )
    if result.get("period_forms") is not None:
        p = result["period_forms"]
        lines.extend(
            [
                "",
                "### Period-form projection",
                f"- Rows scanned: {p.get('rows_scanned', 0)}",
                f"- Projected: {p.get('rows_projected', 0)}",
                f"- Candidates: {p.get('candidates', 0)}",
                f"- Rows written: {p.get('rows_written', 0)}",
            ]
        )
    return "\n".join(lines)
