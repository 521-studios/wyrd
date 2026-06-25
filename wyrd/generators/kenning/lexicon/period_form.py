"""Suffix-anchoring projection over binary toponym breakdowns (wyrd-unuo
Phase 3.3 + wyrd-ujyo).

Two projections share the same machinery — for each binary breakdown, find the
longest suffix of a target string that matches a known reflex of the SECOND
morpheme's etymon, then split the target into (prefix, suffix):

* :func:`project_period_forms` anchors against the **attested historical form**
  (``toponym_attestation.form``) and writes ``etymon_period_form`` rows so the
  ``etymon_era_reflexes`` Tier-3 picker can return them.
* :func:`derive_surface_in_modern` anchors against the **modern toponym name**
  (``toponym.modern_name``) and writes the per-element surface slice back onto
  ``toponym_etymology_element.surface_in_modern`` (e.g. Ardeley = Earda + lēah →
  'Arde' + 'ley', recording that Earda surfaces as 'Arde').

Both: binary breakdowns only (ternary alignment isn't reliable without phonetic
distance); ≥2-char projected segments; skip breakdowns whose components point at
OCR-cluster losers (``merged_into_id IS NOT NULL``).

Example (period forms): toponym 'Westminster' attested as 'Westmynstre' in 1086
with breakdown OE ``west`` + OE ``mynster``. The suffix-matcher finds 'mynstre'
as a reflex of ``mynster``, leaving 'West' as the projected period form for
``west`` at 1086.

Both passes are deterministic, LLM-free, and idempotent. period-forms is
reversible via ``clear_enrichment(stage="period-forms")`` (~85-90% precision on
a 40-row stratified spot-check); derive_surface_in_modern re-derives the same
surface on every run.
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
    *,
    include_reflexes: bool = False,
) -> set[str]:
    """Build the set of forms a suffix may match against for THIS etymon:
    canonical_form + cognate-cluster mates (any language; gmw-msc / Middle Scots
    forms like 'tone' are real historical reflexes that survive into the broader
    cognate set) + etymon_variant rows.

    When ``include_reflexes`` is set (the MODERN-name projection,
    :func:`derive_surface_in_modern`), the etymon's mined modern reflex surfaces
    (``reflex.surface_form`` via ``reflex_etymon``) are added too. Those are the
    forms that actually appear in modern names — OE ``tūn`` surfaces as ``ton`` /
    ``tone`` in Houghton / Boulton — whereas the ``etymon_variant`` rows are OE
    inflections (``tūn`` / ``tūnas`` / ``tūne``) that almost never suffix-match a
    modern name. The historical-form projection (:func:`project_period_forms`)
    leaves reflexes out: medieval attested forms align to the OE variants, and the
    matched span is scoped to the known last morpheme of a known binary breakdown
    + a ≥2-char-segment guard, so this stays string-only matching (wyrd-5b1a).

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
    if include_reflexes:
        for sf in _reflex_suffix_candidates(db, etymon_id, canonical_form):
            _add(sf)
    return candidates


def _reflex_suffix_candidates(db: LexiconDB, etymon_id: int, canonical_form: str) -> set[str]:
    """Return the MINIMAL modern reflex surfaces for ``etymon_id`` (>=2 chars) — the
    ``include_reflexes`` half of :func:`_suffix_candidates_for_etymon`, with COMPOSITE
    surfaces dropped.
    """
    reflex_surfaces: set[str] = set()
    cur = db.conn.execute(
        "SELECT DISTINCT r.surface_form "
        "FROM reflex r JOIN reflex_etymon re ON re.reflex_id = r.id "
        "WHERE re.etymon_id = ?",
        (etymon_id,),
    )
    for row in cur:
        sf = (row["surface_form"] or "").lower()
        if len(sf) >= 2:
            reflex_surfaces.add(sf)
    # Drop COMPOSITE reflex surfaces — ones that merely append the PREVIOUS
    # element's context onto a shorter reflex of the SAME morpheme (OE tūn
    # mines 'ton' but also 'eton'/'aughton'/'oulton', which all end in 'ton').
    # Keeping the longer composite would let the longest-suffix anchor eat a
    # char of the previous element (Stapleton → 'Stapl'+'eton' instead of
    # 'Staple'+'ton').
    #
    # But a longer reflex that ends in a shorter one is NOT always a composite:
    # the shorter one may be a phonetic REDUCTION of the same morpheme (hām →
    # 'ham' base + 'am' h-dropped; lēah → 'ley' + 'ey'). There the extra prefix
    # ('h' of 'ham', 'l' of 'ley') belongs to the morpheme itself, so it
    # prefixes the canonical form — whereas a composite's extra prefix ('e' of
    # 'eton') comes from the preceding element and does NOT. So drop sf only
    # when its extra prefix is NOT a prefix of the canonical form (wyrd-5b1a;
    # Gemini #741: without this guard, base 'ham'/'east'/'bury' etc. get
    # wrongly dropped, mis-splitting Birmingham → 'Birmingh'+'am').
    #
    # All comparands are diacritic-stripped together: norm_canonical is
    # stripped, so a diacritic-bearing reflex (e.g. 'môr', 'cwmbrân') must be
    # too — otherwise an 'ā'-prefixed extra would never match the stripped
    # canonical and a legit base form would be wrongly dropped (Gemini #741 r2).
    norm_canonical = _strip_diacritics(canonical_form.lower())
    stripped = {sf: _strip_diacritics(sf) for sf in reflex_surfaces}
    return {
        sf
        for sf in reflex_surfaces
        # reflex_surfaces is already filtered to len >= 2 above; `and stripped[o]`
        # guards the rare all-combining-marks surface that strips to "".
        if not any(
            stripped[o] != stripped[sf]
            and stripped[o]
            and stripped[sf].endswith(stripped[o])
            and not norm_canonical.startswith(stripped[sf][: -len(stripped[o])])
            for o in reflex_surfaces
        )
    }


def _find_longest_suffix_match(attested_form: str, candidates: set[str]) -> str | None:
    """Return the longest suffix of ``attested_form`` (preserving
    casing) that matches any entry in ``candidates`` (case-insensitive
    + diacritic-insensitive). Returns None when no candidate is a
    suffix of the attested form.

    Matches are minimum-length 2 to avoid spurious 1-char hits
    (a trailing 's' / 'e' is too noisy to project as a morpheme
    surface).
    """
    cands = {c for c in candidates if len(c) >= 2}
    if not cands:
        return None
    # Walk RAW suffixes longest-first and return the first whose normalized form is a
    # candidate. Matching and slicing must happen in the SAME string: ``_strip_diacritics``
    # uses NFKD, which is not length-preserving (a ligature ``ﬁ`` → ``fi`` expands 1→2
    # chars), so measuring a candidate's length against the stripped form and slicing
    # that count off the RAW form would shift the morpheme boundary. Returning the raw
    # suffix itself keeps the cut raw-form-aligned and preserves casing/diacritics.
    for i in range(len(attested_form)):
        suffix = attested_form[i:]
        suffix_lower = suffix.lower()
        if suffix_lower in cands or _strip_diacritics(suffix_lower) in cands:
            return suffix
    return None


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


def _preload_binary_breakdowns_for_modern(db: LexiconDB) -> list[tuple[str, list[dict]]]:
    """Every binary breakdown whose toponym has a ``modern_name``, as
    ``(modern_name, [element0, element1])`` ordered by ordinal. Each element dict
    carries its ``toponym_etymology_element`` natural key (``te_id`` +
    ``ordinal``) so the caller can UPDATE its ``surface_in_modern`` (the table
    has no surrogate id), plus the etymon fields the suffix-matcher needs.

    Mirror of :func:`_preload_binary_breakdowns` (single query, drops breakdowns
    whose components are ``merged_into_id`` tombstones) but anchored on the
    toponym's modern name rather than its attestations.
    """
    cur = db.conn.execute(
        """
        SELECT t.id AS toponym_id, t.modern_name, te.id AS te_id,
               tee.ordinal, tee.etymon_id,
               e.canonical_form, e.cognate_id, e.merged_into_id
        FROM toponym t
        JOIN toponym_etymology te ON te.toponym_id = t.id
        JOIN toponym_etymology_element tee ON tee.toponym_etymology_id = te.id
        JOIN etymon e ON tee.etymon_id = e.id
        WHERE t.modern_name IS NOT NULL AND t.modern_name != ''
        ORDER BY t.id, te.id, tee.ordinal
        """
    )
    grouped: dict[tuple[int, int], dict] = {}
    skip_keys: set[tuple[int, int]] = set()
    for row in cur:
        key = (row["toponym_id"], row["te_id"])
        if row["merged_into_id"] is not None:
            skip_keys.add(key)
            continue
        entry = grouped.setdefault(key, {"modern_name": row["modern_name"], "elements": []})
        entry["elements"].append(
            {
                "te_id": row["te_id"],
                "ordinal": row["ordinal"],
                "etymon_id": row["etymon_id"],
                "canonical_form": row["canonical_form"],
                "cognate_id": row["cognate_id"],
            }
        )
    out: list[tuple[str, list[dict]]] = []
    for key, entry in grouped.items():
        if len(entry["elements"]) == 2 and key not in skip_keys:
            out.append((entry["modern_name"], entry["elements"]))
    return out


def derive_surface_in_modern(
    db: LexiconDB,
    *,
    apply: bool = False,
    progress_every: int = 200,
) -> dict:
    """Derive ``toponym_etymology_element.surface_in_modern`` by suffix-anchoring
    each binary breakdown against the toponym's MODERN name (wyrd-ujyo).

    For each binary breakdown, segment the modern name's suffix against the LAST
    morpheme's known reflexes (canonical form + cognate-cluster mates + etymon
    variants + the morpheme's mined ``reflex`` surfaces — wyrd-5b1a, so the modern
    reflex ``ton`` aligns where the OE inflection ``tūn`` cannot); the matched
    suffix is that morpheme's surface, and the remaining prefix is the FIRST
    morpheme's surface. e.g. 'Ardeley' = OE ``earda`` + OE ``lēah`` → first
    morpheme surfaces as 'Arde', last as 'ley'.

    Limits mirror :func:`project_period_forms`: binary breakdowns only,
    last-morpheme suffix-anchoring only, skip ``merged_into_id`` losers, both
    projected segments ≥2 chars. v1 also skips MULTI-WORD modern names
    ('Stratford upon Avon') — the prefix slice would otherwise absorb the leading
    words + whitespace; word-aware alignment is out of scope. Deterministic +
    idempotent — re-runs UPDATE the same surface (the column starts NULL at
    ingest). Dry-run unless ``apply``.

    Progress lines emit to stderr every ``progress_every`` breakdowns scanned
    (CLAUDE.md mining-progress shape; clamps to ≥1). Returns row-counts.
    """
    import sys
    import time

    progress_every = max(progress_every, 1)
    started = time.monotonic()

    breakdowns = _preload_binary_breakdowns_for_modern(db)
    cached_candidates: dict[int, set[str]] = {}
    rows_scanned = 0
    rows_projected = 0
    multiword_skipped = 0
    updates: list[tuple[str, int, int]] = []  # (surface_in_modern, te_id, ordinal)

    def _progress(rows_written: int | None = None) -> None:
        elapsed = max(time.monotonic() - started, 1e-6)
        written = "" if rows_written is None else f" rows_written={rows_written}"
        print(
            f"  [{rows_scanned}/{len(breakdowns)}]  rows_projected={rows_projected} "
            f"elements_updated={len(updates)}{written} "
            f"({elapsed / max(rows_scanned, 1):.4f}s/entry)",
            file=sys.stderr,
            flush=True,
        )

    for modern_name, elements in breakdowns:
        rows_scanned += 1
        # v1: single-token modern names only — a multi-word name's prefix slice
        # would absorb the leading words + whitespace (e.g. 'Stratford upon A').
        if " " in modern_name:
            multiword_skipped += 1
            continue
        first, last = elements[0], elements[1]
        if last["etymon_id"] not in cached_candidates:
            cached_candidates[last["etymon_id"]] = _suffix_candidates_for_etymon(
                db,
                last["etymon_id"],
                last["canonical_form"],
                last["cognate_id"],
                # Modern names end in modern reflex surfaces (ton/tone), not OE
                # inflections (tūn/tūnas) — pull the mined reflexes too (wyrd-5b1a).
                include_reflexes=True,
            )
        match = _find_longest_suffix_match(modern_name, cached_candidates[last["etymon_id"]])
        if match is not None:
            last_surface = match
            first_surface = modern_name[: len(modern_name) - len(match)]
            # Both projected segments ≥2 chars (single-letter slices are noise).
            if len(first_surface) >= 2 and len(last_surface) >= 2:
                updates.append((first_surface, first["te_id"], first["ordinal"]))
                updates.append((last_surface, last["te_id"], last["ordinal"]))
                rows_projected += 1
        if rows_scanned % progress_every == 0:
            _progress()

    rows_written = 0
    if apply and updates:
        result = db.conn.executemany(
            "UPDATE toponym_etymology_element SET surface_in_modern = ? "
            "WHERE toponym_etymology_id = ? AND ordinal = ?",
            updates,
        )
        rows_written = result.rowcount
        db.commit()

    _progress(rows_written=rows_written)

    return {
        "rows_scanned": rows_scanned,
        "rows_projected": rows_projected,
        "multiword_skipped": multiword_skipped,
        "elements_updated": len(updates),
        "rows_written": rows_written,
        "applied": apply,
    }
