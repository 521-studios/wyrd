"""Page-number resolution + quote-context utilities for citations.

Two scholarly-source conventions are supported for resolving the page
a citation came from:

* **Mawer / alphabetical-headword running headers** — every page top
  shows the first headword + page number (``BACKWORTH 9``,
  ``ABBEY DORE 1``, ``ST PETER'S 47``). Used by Mawer 1920, Bannister
  1916, Ekwall, etc.
* **Skeat § running headers** — ``§ N. NAMES ENDING IN -X. <page>``
  shape used by Skeat 1901+ county dictionaries. Title block is
  all-caps and ends with a trailing integer page.

``detect_running_headers`` picks the better-yielding parser for a
given source. ``backfill_citation_pages`` is the operator entry point
called by ``wyrd kenning lexicon backfill-citation-pages`` and is the
only writer of ``etymon_citation.page`` / ``toponym_etymology.page``.

All of this is post-mining enrichment — runs after extraction lands
rows with NULL page, idempotent against re-runs.
"""

from __future__ import annotations

import re

from wyrd.generators.kenning.lexicon.db import LexiconDB

# wyrd-9kh.5: pattern for Mawer-style alphabetical-headword running headers
# (e.g. 'BACKWORTH 9' on Mawer 1920, 'ABBEY DORE 1' on Bannister 1916,
# 'ST PETER'S 47' for saint-prefixed places). The leftmost word is the
# first headword on the page; the rightmost integer is the page number.
# Single-word forms require 3+ chars to filter OCR noise; multi-word
# forms accept a 2+ char first word so 'ST PETER'S' / 'DR FOO' admit.
_RUNNING_HEADER_RE = re.compile(
    r"^\s*("
    r"[A-Z][A-Z\-']*\s+[A-Z][A-Z\-']+(?:\s+[A-Z][A-Z\-']+)*"  # multi-word: 2+ char first word
    r"|"
    r"[A-Z][A-Z\-']{2,}"  # single-word: 3+ chars total
    r")\s+(\d+)\s*$",
    re.MULTILINE,
)

# wyrd-w5wh: cross-reference lead-words. An all-caps body cross-reference ending
# in a stray OCR integer ('SEE ALSO 12') matches the running-header shape; if its
# first word is one of these it is a body line, not a page header. Compared after
# stripping a trailing '.' ('CF.' → 'CF').
_CROSSREF_LEAD_WORDS = frozenset({"SEE", "CF", "VIDE", "SUB", "V"})

# wyrd-w5wh: largest plausible page jump between consecutive real running headers.
# Each printed page carries one header (pages increase by ~1); this tolerates
# header-less / OCR-dropped pages while rejecting a stray mid-book integer far
# from the running page count. Tune against real Mawer OCR if under-yielding.
_MAX_PAGE_JUMP = 15


# wyrd-8st: pattern for Skeat-style §-section running headers
# (e.g. '§ 2. NAMES ENDING IN -TON. 9' on Skeat 1901 Cambridgeshire).
# Body of every printed page begins with one of these.
#
# Distinguished from non-running TOC + section openers ('§ 2. The suffix
# -ton.' — title case, no trailing page number) by requiring an
# ALL-CAPS title block (≥4 chars including allowed punctuation) AND a
# trailing integer page.
#
# OCR variants tolerated:
# - '§8.' (no space between § and number) and '§ 8.' (variable spacing)
# - 'IX -BRIDGE' (OCR misread of 'IN -BRIDGE')
# - Multi-suffix headers like 'NAMES ENDING IN -BRIDGE, -HITHE.'
_SKEAT_SECTION_HEADER_RE = re.compile(
    r"^\s*§\s*\d+\.\s+"  # § N.
    r"([A-Z\-][A-Z\s,\-\.\']{4,}?)"  # ALL-CAPS title block
    r"\s+(\d+)\s*$",  # trailing page
    re.MULTILINE,
)


def parse_running_header_pages(text: str) -> list[tuple[int, int]]:
    """Extract (offset, page_number) pairs from running headers in OCR'd
    alphabetical-headword books (Mawer / Bannister / Ekwall style).

    Pattern: a line containing one or more all-caps words followed by an
    integer (e.g. 'BACKWORTH 9', 'ABBEY DORE 1'). Returns sorted-by-offset
    pairs. Skeat-style books use a different convention — see
    `parse_skeat_section_header_pages`. Returns an empty list when no
    headers match.

    wyrd-w5wh data-quality guards, because an all-caps BODY line ending in a
    stray OCR integer ('SEE ALSO 12', or an all-caps entry name + a spurious
    page) otherwise spoofs a header and injects a false page boundary:

    * (b) a candidate whose first word is a cross-reference lead-word
      (``SEE`` / ``CF`` / ``VIDE`` / ``SUB`` / ``V``) is a body cross-reference,
      dropped.
    * (a) the load-bearing guard: each real header's page is just past the
      previous accepted one (``prev < page <= prev + _MAX_PAGE_JUMP``); the
      first header seeds the sequence. A mid-book stray integer far from the
      running page count fails the window and is dropped.

    Both guards only *reduce* false headers (an inherent limit of header-pattern
    scraping); they never assign a wrong page where the plain regex was correct.
    """
    out: list[tuple[int, int]] = []
    prev_page: int | None = None
    for m in _RUNNING_HEADER_RE.finditer(text):
        page = int(m.group(2))
        if m.group(1).split()[0].rstrip(".") in _CROSSREF_LEAD_WORDS:
            continue  # (b) cross-reference body line, not a page header
        if prev_page is not None and not (prev_page < page <= prev_page + _MAX_PAGE_JUMP):
            continue  # (a) out of the monotonic page-sequence window
        out.append((m.start(), page))
        prev_page = page
    return out


def parse_skeat_section_header_pages(text: str) -> list[tuple[int, int]]:
    """Extract (offset, page_number) pairs from Skeat-style running
    headers (`§ N. NAMES ENDING IN -X. <page>`).

    Returns the same shape as `parse_running_header_pages` so
    `page_for_offset` works on either parser's output. Returns an empty
    list when no headers match — caller can then try the Mawer parser
    or report `no_headers`.
    """
    return [(m.start(), int(m.group(2))) for m in _SKEAT_SECTION_HEADER_RE.finditer(text)]


def detect_running_headers(text: str) -> tuple[list[tuple[int, int]], str]:
    """Try both header conventions and return (headers, parser_name)
    for whichever produced more matches. Per-source-book dispatch:
    Mawer-style and Skeat-§ are mutually exclusive in practice, so
    'highest yield wins' is unambiguous. Returns (`[]`, `"none"`) when
    neither convention matches.

    Tie-break: when both parsers return the SAME non-zero count
    (theoretically possible with crafted OCR; not seen in the live
    corpus), Mawer wins. Mawer is the older / more permissive
    pattern; defaulting to it on ties keeps page-resolution behavior
    bit-stable for any source already routed through it."""
    mawer = parse_running_header_pages(text)
    skeat = parse_skeat_section_header_pages(text)
    if not mawer and not skeat:
        return [], "none"
    if len(skeat) > len(mawer):
        return skeat, "skeat-§"
    return mawer, "mawer"


def page_for_offset(headers: list[tuple[int, int]], offset: int) -> int | None:
    """Find the page number for a body character at `offset`. Returns the
    page from the closest preceding running header, or None if no header
    precedes the offset (e.g. the entry sits in the front matter before
    the first numbered page). `headers` must be sorted by offset, which
    is the natural output of `parse_running_header_pages`."""
    if not headers:
        return None
    page = None
    for hdr_offset, hdr_page in headers:
        if hdr_offset > offset:
            break
        page = hdr_page
    return page


def _normalize_for_quote_match(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to a single space. Returns
    (normalized_text, norm_to_orig) where norm_to_orig[i] is the
    original-text offset of normalized character i — so a substring
    found in normalized space can be translated back to an original
    offset (which is the coordinate system parse_running_header_pages
    returns)."""
    out: list[str] = []
    norm_to_orig: list[int] = []
    in_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_ws:
                out.append(" ")
                norm_to_orig.append(i)
                in_ws = True
        else:
            out.append(ch)
            norm_to_orig.append(i)
            in_ws = False
    return "".join(out), norm_to_orig


def _quote_body_excerpt(quote: str) -> str:
    """Strip the provider-attribution prefix the mining writer prepends.

    Format produced by `assemble_extraction_result`:
      `extracted_by:<provider>:<model>; <LLM commentary> | <body excerpt>`

    Body excerpt is what the parser pulled from the source and is the
    only part that exists in source_text. Returns the substring after
    the FIRST ` | ` separator; if absent (legacy citations or
    commentary-only rows), returns the whole quote so the caller can
    still try a literal match.

    First-` | ` (not last) because the prefix `extracted_by:...; <commentary>`
    never contains the separator, while OCR body excerpts can pick up
    spurious ` | ` from table rules / page-number artefacts. A leftmost
    split is unambiguous about where the body actually starts."""
    sep = " | "
    idx = quote.find(sep)
    if idx < 0:
        return quote
    return quote[idx + len(sep) :]


def _resolve_quote_page(
    quote: str | None,
    *,
    norm_text: str,
    norm_to_orig: list[int],
    headers: list[tuple[int, int]],
) -> tuple[int | None, str, bool]:
    """Resolve one quoted excerpt to its source page.

    Pure given the pre-normalized body (``norm_text`` + its ``norm_to_orig``
    offset map) and the header sequence — no DB, no shared counters. Returns
    ``(page, status, ambiguous)``:

      - ``status`` is ``"ok"`` (``page`` set), or one of the skip reasons
        ``"no_quote"`` / ``"quote_not_in_text"`` / ``"before_first_page"``
        (``page`` is None).
      - ``ambiguous`` (wyrd-3yu) is True when the normalized excerpt occurs at
        more than one position, so the leftmost-match heuristic was forced to
        guess. It is independent of ``status`` — a quote can be both ambiguous
        and resolve to a page, or be ambiguous yet land ``before_first_page``;
        the caller tallies it regardless of outcome.
    """
    if not quote:
        return None, "no_quote", False
    body = _quote_body_excerpt(quote).strip()
    if not body:
        return None, "no_quote", False
    norm_quote, _ = _normalize_for_quote_match(body)
    norm_quote = norm_quote.strip()
    if not norm_quote:
        return None, "no_quote", False
    norm_offset = norm_text.find(norm_quote)
    if norm_offset < 0:
        return None, "quote_not_in_text", False
    ambiguous = norm_text.find(norm_quote, norm_offset + 1) >= 0
    orig_offset = norm_to_orig[norm_offset]
    page = page_for_offset(headers, orig_offset)
    if page is None:
        return None, "before_first_page", ambiguous
    return page, "ok", ambiguous


def backfill_citation_pages(
    db: LexiconDB,
    source_id: str,
    source_text: str,
    *,
    apply: bool = False,
) -> dict[str, int]:
    """Backfill etymon_citation.page and toponym_etymology.page from
    running headers in the source body (wyrd-azv, wyrd-8st).

    For each row in ``source_id`` where page IS NULL, strip the provider-
    attribution prefix from the row's quoted excerpt (everything before
    the first ` | ` — see ``_quote_body_excerpt`` for why first-not-last),
    normalize whitespace (the parser collapses OCR
    multi-spaces before sending to the LLM, so the stored quote uses
    single-spacing while the source has the original OCR runs), find
    the normalized excerpt in normalized source_text, translate the
    match offset back to original coordinates, and look up the page
    from the header sequence. `detect_running_headers` chooses between
    Mawer-style and Skeat-§ conventions per source.

    Returns counts:
      - citations_updated, etymologies_updated: rows whose page was
        successfully resolved (or would be, in dry-run).
      - quote_not_in_text: rows whose excerpt did not appear in
        source_text (OCR drift, commentary-only quotes). Summed across
        both tables.
      - no_quote: rows with NULL/empty excerpt — nothing to anchor on.
      - before_first_page: offset preceded the first running header.
      - no_headers: 1 if neither convention produced any headers; else
        0. On no_headers, the function returns immediately without
        touching any rows.
      - ambiguous_match (wyrd-3yu): rows whose normalized excerpt
        appears at multiple positions in the body. Resolution still
        picks the leftmost match (better than NULL — citations cluster
        by alphabet so misattribution is to a nearby page) but the
        counter surfaces how often the heuristic was forced to guess.

    Idempotent: only operates on rows where page IS NULL, so a re-run
    is a no-op for already-resolved rows.
    """

    counts = {
        "citations_updated": 0,
        "etymologies_updated": 0,
        "quote_not_in_text": 0,
        "no_quote": 0,
        "before_first_page": 0,
        "no_headers": 0,
        "ambiguous_match": 0,
    }
    headers, _parser = detect_running_headers(source_text)
    if not headers:
        counts["no_headers"] = 1
        return counts

    norm_text, norm_to_orig = _normalize_for_quote_match(source_text)

    def _resolve_page(quote: str | None) -> tuple[int | None, str]:
        page, status, ambiguous = _resolve_quote_page(
            quote, norm_text=norm_text, norm_to_orig=norm_to_orig, headers=headers
        )
        if ambiguous:
            counts["ambiguous_match"] += 1
        return page, status

    _backfill_table_pages(
        db,
        "etymon_citation",
        "short_quote",
        source_id,
        _resolve_page,
        apply,
        counts,
        "citations_updated",
    )
    _backfill_table_pages(
        db,
        "toponym_etymology",
        "notes",
        source_id,
        _resolve_page,
        apply,
        counts,
        "etymologies_updated",
    )

    if apply:
        db.commit()
    return counts


def _backfill_table_pages(
    db: LexiconDB,
    table: str,
    quote_col: str,
    source_id: str,
    resolve_fn,
    apply: bool,
    counts: dict[str, int],
    updated_key: str,
) -> None:
    """Iterate page-NULL rows of `table` for `source_id`, resolve each
    via `resolve_fn(quote)`, and update `table.page` if apply is set.
    Mutates `counts` in place: bumps `updated_key` on resolution, or the
    status key returned by resolve_fn on skip. Table identifiers are
    code-controlled (callers in this module only), so f-string SQL is
    safe here."""
    rows = db.conn.execute(
        f"SELECT id, {quote_col} AS quote FROM {table} WHERE source_id = ? AND page IS NULL",  # noqa: S608
        (source_id,),
    ).fetchall()
    for row in rows:
        page, status = resolve_fn(row["quote"])
        if status != "ok":
            counts[status] += 1
            continue
        if apply:
            db.conn.execute(
                f"UPDATE {table} SET page = ? WHERE id = ?",  # noqa: S608
                (str(page), row["id"]),
            )
        counts[updated_key] += 1
