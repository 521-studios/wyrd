"""Attested-year lookup against toponym_etymology.notes (D5-1 / wyrd-3ux).

Scans the LLM-extracted notes column for year-citations (3-4 digit
integers in the 100-1700 range, gated by date-context heuristics) and
writes the earliest valid year per toponym to
``toponym_etymology.attested_year``. Lets the D5-3 era runtime filter
narrow the bundle's morpheme inventory to forms attested in a given
period.

Pattern: every Mawer / Skeat / Ekwall entry typically carries dense
date citations ("Tune, 1086 (DB); Tunes, 1242"). The scan picks them
up with high recall but conservatively — date-context filters drop
publication-year noise (1880-1928 in the scholarly source pool),
page / folio numbers, and the like.

Idempotent + reversible via
``clear_enrichment(stage="attested-years")``.
"""

from __future__ import annotations

import re
from pathlib import Path

from wyrd.generators.kenning.lexicon.constants import normalize_ocr_form
from wyrd.generators.kenning.lexicon.db import LexiconDB

_ATTESTED_YEAR_MIN_LOOKUP = 100
_ATTESTED_YEAR_MAX_LOOKUP = 1700


# requiring the year to follow the matched form within a small window of
# punctuation / whitespace:
#
#   1. "c." prefix — explicit "circa" marker (any year in range):
#         Tune c. 950        Tunna, c. 1066        Tune (c. 950)
#
#   2. Parenthesized year — distinctive citation form:
#         Tune (1086)        Tunes (1242)
#
#   3. Bare year ≥ 700 — post-Roman British/OE attestations begin
#      around then, so a 3-4 digit number ≥700 is far more likely to be
#      a real date than a page reference (page numbers in toponym
#      dictionaries cluster in the low hundreds):
#         Tune, 1086 (DB)    Tunes; 1242    Tunna 800
#
# Years <700 only qualify under (1) or (2). This filters the dominant
# false-positive class — common-word reverse-search forms (`with`,
# `wind`, `port`) followed by page references like "form, 134" — while
# admitting bona-fide Roman-era citations when they're explicitly marked
# with "c." or parens.
#
# Built fresh per matched_form because re.escape() makes it form-specific.
def _build_form_year_pattern(form: str) -> re.Pattern[str]:
    return re.compile(
        r"\b" + re.escape(form) + r"\b"
        r"(?:"
        # c./circa prefix — accept any 3-4 digit year (Roman era OK)
        r"[\s,;:.()\[\]]{0,4}c\.?\s*(\d{3,4})\b"
        r"|"
        # parenthesized year — strong citation marker
        r"[\s,;:.]*\((\d{3,4})\b"
        r"|"
        # bare year ≥ 700 — post-Roman / OE attestations. Three-digit
        # range (700-999) plus four-digit range (1000-1700) explicitly
        # listed so a 4-digit 8xxx/9xxx publication ID, ISBN fragment,
        # or page reference can't slip in. The trailing |1700 admits the
        # year-range upper bound that 1[0-6]\d{2} cuts off at 1699.
        r"[\s,;:.]{0,4}(7\d{2}|[89]\d{2}|1[0-6]\d{2}|1700)\b"
        r")"
    )


def _extract_attested_year_from_body(text: str, form: str) -> int | None:
    """Return the EARLIEST plausibly-attested year cited DIRECTLY against
    ``form`` in ``text``, or None if no qualifying citation is found.

    ``text`` is expected lowercased + OCR-normalized (the same shape the
    reverse-search snippets are stored in). ``form`` should match.

    A candidate year qualifies when:
      * The matched_form is followed (within a few chars of intervening
        punctuation / space, and optionally a "c." prefix) by a 3-4
        digit year — see _build_form_year_pattern. This is the dominant
        scholarly toponym citation shape.
      * The year parses to an integer in [_ATTESTED_YEAR_MIN_LOOKUP,
        _ATTESTED_YEAR_MAX_LOOKUP].

    "Earliest" is a deliberate v1 choice: scholarly toponym citations
    often run ascending ("Tune, 1086; Tunes, 1242; Tunne, 1340"), and the
    earliest reflects the earliest known attestation of the form. Future
    work (D5-2) may want all years, not just the earliest.
    """
    if not form:
        return None
    # ``text`` is lowercased + OCR-normalized before reaching this fn
    # (see lookup_attested_years). Lowercase the form so case-mismatched
    # matched_form values still produce hits.
    pattern = _build_form_year_pattern(form.lower())
    earliest: int | None = None
    for m in pattern.finditer(text):
        # Three capture groups in the alternation; exactly one will be
        # populated per match.
        captured = m.group(1) or m.group(2) or m.group(3)
        if captured is None:
            continue
        year = int(captured)
        if year < _ATTESTED_YEAR_MIN_LOOKUP or year > _ATTESTED_YEAR_MAX_LOOKUP:
            continue
        if earliest is None or year < earliest:
            earliest = year
    return earliest


_TOPONYM_NOTE_YEAR_PATTERN = re.compile(r"\b(7\d{2}|[89]\d{2}|1[0-6]\d{2}|1700)\b")

# A digit run preceded by one of these abbreviations is a page or
# volume reference, not a date. Page numbers in long EPNS volumes
# occasionally exceed 700, so filtering on year-range alone leaks
# them through.
#
# Matched as whole words at the END of the preceding slice — `\b`
# prevents false-skips on words that happen to end in "p." (e.g.
# "Bp." for Bishop) by requiring a word boundary before the marker.
# Substring match would also incorrectly fire "p." inside "chap.",
# but that's a no-op (chap is itself a marker); the real FP class
# was words like "Hp." or stray "p" letters at the end of the slice.
_TOPONYM_NOTE_PAGE_MARKER_RE = re.compile(
    r"\b(?:p|pp|vol|vols|ch|chap|no|nr)\.\s*$",
    re.IGNORECASE,
)


def _earliest_year_in_notes(notes: str | None) -> int | None:
    """Find the earliest plausible year (700-1700) in a
    ``toponym_etymology.notes`` value. Skips digit runs preceded by
    page / volume markers (``"p. 755"``, ``"vol. 1244"``) — those are
    bibliographic references, not dates. Returns None when no
    qualifying year appears."""
    if not notes:
        return None
    earliest: int | None = None
    for m in _TOPONYM_NOTE_YEAR_PATTERN.finditer(notes):
        ystart = m.start()
        # Pass the full prefix and rely on the regex's $ anchor to
        # match only when the marker IMMEDIATELY precedes the year.
        # A fixed-size window would miss "p.   1086" with extra spaces;
        # the linear search cost is trivial since notes is already in
        # memory and finditer call frequency is bounded by year-hit
        # count (a few per row in the worst case).
        if _TOPONYM_NOTE_PAGE_MARKER_RE.search(notes, 0, ystart):
            continue
        year = int(m.group(1))
        if earliest is None or year < earliest:
            earliest = year
    return earliest


def _scan_etymon_text_match_for_years(db: LexiconDB, sources_path: Path, *, apply: bool) -> dict:
    """Stream ``etymon_text_match`` rows ordered by source_id; for each
    row, scan the matching source body for a form-attached year
    citation (PR #47 / wyrd-3ux pattern).

    Memory characteristics:
    * ONE source body in memory at a time (the heavy thing — 100s of MB
      for big OCR'd corpora).
    * ONE row's metadata at a time during iteration.
    * candidate_updates accumulates (year, row_id) tuples for every hit
      and flushes via executemany at the end. At ~3% hit rate even a
      million-row text-match table yields ~30k tuples (~1 MB), which
      is fine; chunking the writes would be premature optimisation.
    """
    available_sources = {f.stem: f for f in sources_path.glob("*.txt")}

    cur = db.conn.execute(
        "SELECT id, source_id, matched_form FROM etymon_text_match "
        "WHERE attested_year IS NULL ORDER BY source_id"
    )

    candidate_updates: list[tuple[int, int]] = []  # (year, row_id)
    sources_missing: set[str] = set()
    rows_scanned = 0
    current_source_id: str | None = None
    text: str | None = None
    for row in cur:
        rows_scanned += 1
        if row["source_id"] != current_source_id:
            current_source_id = row["source_id"]
            source_file = available_sources.get(current_source_id)
            if source_file is None:
                sources_missing.add(current_source_id)
                text = None
                continue
            text = source_file.read_text(errors="replace").lower()
            text = normalize_ocr_form(text)
        if text is None:
            continue
        year = _extract_attested_year_from_body(text, row["matched_form"])
        if year is not None:
            candidate_updates.append((year, row["id"]))

    rows_written = 0
    if apply and candidate_updates:
        result = db.conn.executemany(
            "UPDATE etymon_text_match SET attested_year = ? WHERE id = ? AND attested_year IS NULL",
            candidate_updates,
        )
        rows_written = result.rowcount
        db.commit()

    return {
        "rows_scanned": rows_scanned,
        "candidates": len(candidate_updates),
        "rows_written": rows_written,
        "sources_missing": len(sources_missing),
    }


def _scan_toponym_etymology_for_years(db: LexiconDB, *, apply: bool) -> dict:
    """wyrd-bag: scan ``toponym_etymology.notes`` for the earliest
    plausible year. Unlike the ``etymon_text_match`` scan, this doesn't
    need source-body files — the LLM-extracted notes are stored inline
    in the DB and are densely populated with scholarly citations
    (``"Tune, 1086 (DB); Tunes, 1242"``).

    Memory characteristics:
    * Full match list (candidate_updates) is held in memory before the
      executemany write. At ~80% density on the production corpus
      (~5,200 rows × 0.8 = ~4,200 tuples = ~130 KB) this is fine.
    * If a future corpus pushes this past tens of millions of rows,
      switching to chunked writes (yield + flush per N hits) would
      cap peak memory; not worth the complexity at current scale.
    """
    cur = db.conn.execute(
        "SELECT id, notes FROM toponym_etymology WHERE attested_year IS NULL AND notes IS NOT NULL"
    )

    candidate_updates: list[tuple[int, int]] = []
    rows_scanned = 0
    for row in cur:
        rows_scanned += 1
        year = _earliest_year_in_notes(row["notes"])
        if year is not None:
            candidate_updates.append((year, row["id"]))

    rows_written = 0
    if apply and candidate_updates:
        result = db.conn.executemany(
            "UPDATE toponym_etymology SET attested_year = ? WHERE id = ? AND attested_year IS NULL",
            candidate_updates,
        )
        rows_written = result.rowcount
        db.commit()

    return {
        "rows_scanned": rows_scanned,
        "candidates": len(candidate_updates),
        "rows_written": rows_written,
    }


def lookup_attested_years(
    db: LexiconDB,
    sources_dir: Path | str,
    *,
    apply: bool = False,
) -> dict:
    """Populate ``attested_year`` on the two row sources that carry
    date-citation evidence (D5-1):

    * ``etymon_text_match.attested_year`` (PR #47 / wyrd-3ux) — scans
      source bodies via the form-attached pattern. Lower density (~3%)
      because reverse-search rows are mentions, not citations.
    * ``toponym_etymology.attested_year`` (PR #5x / wyrd-bag) — scans
      LLM-extracted notes for the earliest year ≥700. Higher density
      (~80%) because the notes are scholarly date strings.

    LLM-free, idempotent, reversible. Per D21/D22 this is enrichment —
    operates on already-mined data without touching mining evidence.
    Re-runs are no-ops on rows where ``attested_year`` is already set.
    Reverse via ``clear-enrichment --stage=attested-years --apply``.

    Returns a dict with both per-source breakdowns and aggregate keys
    so existing callers (PR #47's CLI output, tests, etc.) continue
    to read ``rows_scanned`` / ``candidates`` / ``rows_written``
    unchanged while new callers can read the per-source detail.
    """
    sources_path = Path(sources_dir)
    if not sources_path.is_dir():
        raise ValueError(f"sources_dir not found: {sources_path}")

    etm = _scan_etymon_text_match_for_years(db, sources_path, apply=apply)
    te = _scan_toponym_etymology_for_years(db, apply=apply)

    return {
        # Aggregate keys (PR #47 back-compat).
        "rows_scanned": etm["rows_scanned"] + te["rows_scanned"],
        "candidates": etm["candidates"] + te["candidates"],
        "rows_written": etm["rows_written"] + te["rows_written"],
        "sources_missing": etm["sources_missing"],
        "applied": apply,
        # Per-source breakdown for richer CLI output + new tests.
        "etymon_text_match": etm,
        "toponym_etymology": te,
    }
