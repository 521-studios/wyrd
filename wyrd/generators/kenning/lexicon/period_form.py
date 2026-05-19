"""Per-etymon period-form projection (wyrd-unuo Phase 3.3).

For each binary toponym breakdown, find the longest suffix of the
attested historical form that matches a known reflex of the second
morpheme's etymon, then assign the corresponding period form to the
FIRST morpheme. Writes ``etymon_period_form`` rows so the
``etymon_era_reflexes`` Tier 3 picker can return them.

Example: toponym 'Westminster' attested as 'Westmynstre' in 1086 with
breakdown OE ``west`` + OE ``mynster``. The suffix-matcher finds
'mynstre' as a reflex of ``mynster``, leaves 'West' as the projected
period form for ``west`` at 1086.

Binary breakdowns only (ternary alignment isn't reliable without
phonetic distance); ≥2-char projected segments; skips breakdowns whose
components point at OCR-cluster losers (``merged_into_id IS NOT NULL``).

Idempotent + reversible via
``clear_enrichment(stage="period-forms")``. ~85-90% precision on a
40-row stratified spot-check.
"""

from __future__ import annotations

import unicodedata

from wyrd.generators.kenning.lexicon.db import LexiconDB


def _strip_diacritics(text: str) -> str:
    """NFKD-decompose ``text`` and drop combining marks. Used for
    suffix matching where 'tūn' (with macron) should match against
    a historical 'tun' surface."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _suffix_candidates_for_etymon(
    db: LexiconDB,
    etymon_id: int,
    canonical_form: str,
    cognate_id: int | None,
) -> set[str]:
    """Build the set of forms a historical-form suffix may match
    against for THIS etymon: canonical_form + cognate-cluster mates
    (any language; gmw-msc / Middle Scots forms like 'tone' are real
    historical reflexes that survive into the broader cognate set) +
    etymon_variant rows.

    All forms returned lowercased. Both diacritic-bearing and
    diacritic-stripped variants are included so 'tūn' and 'tun' both
    match against historical 'tun'-suffix forms.
    """
    candidates: set[str] = set()

    def _add(form: str) -> None:
        if not form or len(form) < 2:
            return
        candidates.add(form.lower())
        candidates.add(_strip_diacritics(form.lower()))

    _add(canonical_form)
    if cognate_id is not None:
        cur = db.conn.execute(
            "SELECT canonical_form FROM etymon WHERE cognate_id = ? AND merged_into_id IS NULL",
            (cognate_id,),
        )
        for row in cur:
            _add(row["canonical_form"])
    cur = db.conn.execute(
        "SELECT form FROM etymon_variant WHERE etymon_id = ?",
        (etymon_id,),
    )
    for row in cur:
        _add(row["form"])
    return candidates


def _find_longest_suffix_match(attested_form: str, candidates: set[str]) -> str | None:
    """Return the longest suffix of ``attested_form`` (preserving
    casing) that matches any entry in ``candidates`` (case-insensitive
    + diacritic-insensitive). Returns None when no candidate is a
    suffix of the attested form.

    Matches are minimum-length 2 to avoid spurious 1-char hits
    (a trailing 's' / 'e' is too noisy to project as a morpheme
    surface).
    """
    af_lower = attested_form.lower()
    af_stripped = _strip_diacritics(af_lower)
    best_len = 0
    for cand in candidates:
        if len(cand) < 2:
            continue
        if (af_lower.endswith(cand) or af_stripped.endswith(cand)) and len(cand) > best_len:
            best_len = len(cand)
    if best_len == 0:
        return None
    return attested_form[-best_len:]


def project_period_forms(
    db: LexiconDB,
    *,
    apply: bool = False,
    progress_every: int = 200,
) -> dict:
    """Project per-etymon period-keyed surface forms from
    toponym_attestation rows (wyrd-unuo Phase 3.3). For each binary
    toponym breakdown, segment the attested form's suffix against the
    last morpheme's known reflexes; the remaining prefix is the first
    morpheme's projected period form.

    v1 limits (mirroring the module-level comment block above for
    callers that read help() / IDE hovers):

    * **Binary breakdowns only** — ternary+ skipped because middle-
      morpheme alignment isn't reliable without phonetic distance.
    * **Suffix-anchoring only** — last-morpheme matched against
      cluster mates / variants; first-morpheme is whatever's left
      after the suffix split.
    * **Skip merged-into etymons** — toponyms whose breakdown
      points at OCR-cluster losers (merged_into_id IS NOT NULL)
      are skipped so loser etymons don't accumulate stale period-
      form rows that then surface via the era-reflex Tier 3.
    * **Reasonability gates** — both projected segments must be
      ≥2 chars; sub-2-char prefixes / suffixes are noise.

    Idempotent via the unique index ``idx_period_form_unique``;
    re-runs are no-ops on already-projected (etymon, form, year,
    source_doc) tuples.

    Progress lines emit to stderr every ``progress_every`` rows
    (CLAUDE.md mining-progress shape). ``progress_every`` clamps to
    ≥1.

    Returns a dict with row-counts so callers can report.
    """
    import sys
    import time

    progress_every = max(progress_every, 1)
    started = time.monotonic()

    cur = db.conn.execute(
        "SELECT id, toponym_id, form, date_year, source_doc "
        "FROM toponym_attestation "
        "WHERE date_year IS NOT NULL "
        "ORDER BY toponym_id"
    )

    rows_scanned = 0
    rows_projected = 0
    candidate_inserts: list[tuple[int, str, int, str | None, int]] = []
    # Eager-load all binary breakdowns in one query (instead of one
    # per toponym_id during the scan loop). At ~700 unique toponyms
    # the per-row query was ~500ms cumulative; the eager-load is
    # ~50ms total. Scales linearly to 100k+ toponyms without
    # hitting the per-row cliff.
    cached_breakdowns: dict[int, list[list[dict]]] = _preload_binary_breakdowns(db)
    cached_candidates: dict[int, set[str]] = {}

    for ta in cur:
        rows_scanned += 1
        toponym_id = ta["toponym_id"]
        breakdowns = cached_breakdowns.get(toponym_id, [])
        for breakdown in breakdowns:
            first, last = breakdown[0], breakdown[1]
            if last["etymon_id"] not in cached_candidates:
                cached_candidates[last["etymon_id"]] = _suffix_candidates_for_etymon(
                    db,
                    last["etymon_id"],
                    last["canonical_form"],
                    last["cognate_id"],
                )
            candidates = cached_candidates[last["etymon_id"]]
            match = _find_longest_suffix_match(ta["form"], candidates)
            if match is None:
                continue
            last_form = match
            first_form = ta["form"][: len(ta["form"]) - len(match)]
            # Reasonability gates: both projected segments must be
            # ≥2 chars (single-letter prefixes / suffixes are noise).
            if len(first_form) < 2 or len(last_form) < 2:
                continue
            candidate_inserts.append(
                (first["etymon_id"], first_form, ta["date_year"], ta["source_doc"], ta["id"])
            )
            candidate_inserts.append(
                (last["etymon_id"], last_form, ta["date_year"], ta["source_doc"], ta["id"])
            )
            rows_projected += 1
        if rows_scanned % progress_every == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"  [{rows_scanned}]  rows_projected={rows_projected} "
                f"candidates={len(candidate_inserts)} "
                f"({elapsed / rows_scanned:.4f}s/entry)",
                file=sys.stderr,
                flush=True,
            )

    rows_written = 0
    if apply and candidate_inserts:
        result = db.conn.executemany(
            "INSERT OR IGNORE INTO etymon_period_form "
            "(etymon_id, form, date_year, source_doc, attestation_id) "
            "VALUES (?, ?, ?, ?, ?)",
            candidate_inserts,
        )
        rows_written = result.rowcount
        db.commit()

    elapsed = max(time.monotonic() - started, 1e-6)
    print(
        f"  [{rows_scanned}/{rows_scanned}]  rows_projected={rows_projected} "
        f"candidates={len(candidate_inserts)} rows_written={rows_written} "
        f"({elapsed / max(rows_scanned, 1):.4f}s/entry)",
        file=sys.stderr,
        flush=True,
    )

    return {
        "rows_scanned": rows_scanned,
        "rows_projected": rows_projected,
        "candidates": len(candidate_inserts),
        "rows_written": rows_written,
        "applied": apply,
    }


def _preload_binary_breakdowns(db: LexiconDB) -> dict[int, list[list[dict]]]:
    """Eager-load every binary breakdown for toponyms that have at
    least one toponym_attestation row. Returns ``{toponym_id:
    [breakdown, ...]}`` where each breakdown is a list of 2 element
    dicts ordered by ordinal.

    Single SQL query (vs. a per-toponym version called inside the
    scan loop) — avoids the N+1 pattern when project_period_forms
    iterates the attestation table. Filters out breakdowns whose
    components are merged_into-id tombstones.
    """
    cur = db.conn.execute(
        """
        SELECT te.toponym_id, te.id AS te_id, tee.ordinal, tee.etymon_id,
               e.canonical_form, e.language, e.cognate_id, e.merged_into_id
        FROM toponym_etymology te
        JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id
        JOIN etymon e ON tee.etymon_id = e.id
        WHERE te.toponym_id IN (
            SELECT DISTINCT toponym_id FROM toponym_attestation
            WHERE date_year IS NOT NULL
        )
        ORDER BY te.toponym_id, te.id, tee.ordinal
        """
    )
    grouped: dict[tuple[int, int], list[dict]] = {}
    skip_te_keys: set[tuple[int, int]] = set()
    for row in cur:
        key = (row["toponym_id"], row["te_id"])
        if row["merged_into_id"] is not None:
            skip_te_keys.add(key)
            continue
        grouped.setdefault(key, []).append(
            {
                "ordinal": row["ordinal"],
                "etymon_id": row["etymon_id"],
                "canonical_form": row["canonical_form"],
                "language": row["language"],
                "cognate_id": row["cognate_id"],
            }
        )
    out: dict[int, list[list[dict]]] = {}
    for (topo_id, te_id), elements in grouped.items():
        if len(elements) == 2 and (topo_id, te_id) not in skip_te_keys:
            out.setdefault(topo_id, []).append(elements)
    return out
