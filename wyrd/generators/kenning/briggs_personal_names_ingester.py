"""Parse + ingest Keith Briggs's *Index to Personal Names in English
Place-Names* (EPNS Supplementary Series 2, 3rd revised pdf edn 2024).

wyrd-uzoh. The source is a pdftotext-flattened two-column index
(~13 K lines of body text) keyed on personal-name headforms with
toponym attestations and source citations per entry. Two artifacts
land in the lexicon DB:

1. ``personal_name`` — one row per unique headform (with diacritics
   preserved verbatim, plus a diacritic-stripped ``normalized_form``
   for ASCII-lookup). Carries citation-count metadata (PASE count,
   DLV presence) and language hints.

2. ``personal_name_toponym_attestation`` — one row per (PN, toponym,
   county) occurrence. Each row also captures the attested
   orthographic variant of the PN as it appears in the toponym
   (e.g. for Ēadwulf, the toponym Adlington gives an *Adl-* variant),
   so downstream consumers can read both queries from one table:

   - "Given a toponym, what PN attestations exist?" — index on
     (toponym_form, county_canonical).
   - "Given a PN, what variants did its toponyms produce?" — index
     on (personal_name_id).

The pdftotext output uses a two-column layout. Each physical line
has the left column in chars 0:38 and the right column in chars 38:
(measured from page 19 of the PDF / line 795 of the .txt). Linear
reading order is per-page-left then per-page-right. Page boundaries
are single-number lines (centered with whitespace padding).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- constants

# Identifier this ingest writes to ``source_doc`` rows. Downstream
# supersede/re-ingest logic keys on it.
SOURCE_DOC = "briggs_2024_personal_names_index"

# Column boundary measured from the pdftotext output. The left
# column ends at char 38 (inclusive of the trailing pad space); the
# right column begins at char 38. Verified by inspection of the
# index opening at page 19 (Aalfra DLV. ... Abbud PASE2.).
COLUMN_BOUNDARY = 38

# Line where the index proper begins (first "—A—" marker). Everything
# before this is front-matter (title page, preface, county-code
# coverage list, editorial conventions, bibliography). Indexed
# zero-based when used by enumerate.
INDEX_FIRST_LINE_NUM = 794  # 0-based; line 795 in the file

# County codes → canonical county name. Extracted from §1.2 Coverage
# (pages 1-3 of the PDF). Includes the two non-county sigla used by
# Briggs in the same positional slot:
#  - BdHu : ambiguous Bd/Hu field-name attribution
#  - KW   : Wallenberg's Kent
# Note: Briggs uses ``(Bib, OE)`` etc as LANGUAGE HINTS in the same
# parenthesized position. Those tokens are NOT county codes and are
# filtered out at parse time.
COUNTY_CODE_TO_NAME: dict[str, str] = {
    "Bd": "Bedfordshire",
    "BdHu": "Bedfordshire/Huntingdonshire (ambiguous)",
    "Brk": "Berkshire",
    "Bk": "Buckinghamshire",
    "Ca": "Cambridgeshire",
    "Ch": "Cheshire",
    "Cu": "Cumberland",
    "Db": "Derbyshire",
    "D": "Devon",
    "Do": "Dorset",
    "Du": "County Durham",
    "Ess": "Essex",
    "Gl": "Gloucestershire",
    "Ha": "Hampshire",
    "He": "Herefordshire",
    "Hrt": "Hertfordshire",
    "Hu": "Huntingdonshire",
    "IoW": "Isle of Wight",
    "KW": "Kent (Wallenberg)",
    "La": "Lancashire",
    "Le": "Leicestershire",
    "Li": "Lincolnshire",
    "Mx": "Middlesex",
    "Nf": "Norfolk",
    "Nth": "Northamptonshire",
    "ND": "Northumberland and County Durham",
    "Nt": "Nottinghamshire",
    "O": "Oxfordshire",
    "R": "Rutland",
    "Sa": "Shropshire",
    "So": "Somerset",
    "St": "Staffordshire",
    "Sf": "Suffolk",
    "Sr": "Surrey",
    "Sx": "Sussex",
    "Wa": "Warwickshire",
    "We": "Westmorland",
    "W": "Wiltshire",
    "Wo": "Worcestershire",
    "YE": "Yorkshire (East Riding)",
    "YN": "Yorkshire (North Riding)",
    "YW": "Yorkshire (West Riding)",
}

# Language hints that appear in parentheses in PN headform position.
# Filtered out so they're not mistaken for county codes. Drawn from
# Briggs's §1.3 conventions text.
LANGUAGE_HINTS: set[str] = {
    "OE",
    "ME",
    "EModE",
    "ODan",
    "ON",
    "OF",
    "OFr",
    "OFr.",
    "L",
    "Lat",
    "Welsh",
    "OW",
    "Br",
    "Bib",
    "Celt",
    "Goth",
    "OGmc",
    "OHG",
    "MLG",
    "OSwed",
    "OFris",
    "AN",
    "ANorm",
    "OBret",
    "PrW",
    "OScand",
    "PrGmc",
    "Pr",
    "fem",
    "femin",
    "masc",
}

# Source-citation token regexes. These appear in the bodyless head
# of an entry (Aalfra DLV.) AND interspersed in entries with toponym
# attestations (Abel PASE1 (Bib, OE) Ablishmare 1601 (Le); ...).
RE_PASE = re.compile(r"^PASE(\d+)$")
RE_DLV = re.compile(r"^DLV$")
RE_ASCH = re.compile(r"^ASCh[\d–\-]+(?:\.\d+)?$")

# Headform-start heuristic: a non-indented line beginning with a
# capital letter (possibly preceded by a diacritic-letter or ?, ??).
# After column reconstruction, every entry's first line starts at
# col 0 (left-of-left-col) or col 38 (left-of-right-col); after
# stream reconstruction both become col 0.
RE_HEADFORM_START = re.compile(
    r"^(\?{0,2})"  # optional ?, ??
    r"([A-ZĀĒĪŌŪÆǢÐÞŒŁǷǺ]"  # capital incl. common Old-English / Norse diacritics
    r"[A-Za-zĀāĒēĪīŌōŪūÆæǢǣÐðÞþŒœǷƿǺǻĠġȳȲ̄̆̃̇̈ \-(),fem]*?"
    r"(?:\([a-zA-Z]+\))?)"  # optional (b)a-style insertion
    r"(?:\s|$)"
)

# A toponym attestation looks like ``Toponym (CC)``, ``Toponym CC``
# (no parens), with optional date qualifier and uncertainty marker.
# Date qualifiers: 4-digit year, Hy3, Edw1, 13th, n.d., 1278×84,
# Hy3, 1257, etc.
RE_DATE_QUALIFIER = re.compile(
    r"^(?:n\.d\.|\d+(?:[×-]\d+)?(?:th)?|"
    r"Hy[1-8]|Edw[1-3]|Ric[1-3]|John|Steph|Ric)$"
)


# ---------------------------------------------------------------- data shapes


@dataclass
class PersonalNameRecord:
    """One PN headform parsed from the index."""

    headform: str
    normalized_form: str
    language_hints: list[str] = field(default_factory=list)
    is_feminine: bool = False
    pase_count: int | None = None
    has_dlv: bool = False
    ascharter_refs: list[str] = field(default_factory=list)
    raw_entry: str = ""


@dataclass
class AttestationRecord:
    """One (PN, toponym, county) attestation row."""

    toponym_form: str
    attested_variant: str | None
    county_code: str
    county_canonical: str
    date_qualifier: str | None
    is_uncertain: bool
    is_serious_doubt: bool
    source_citation: str | None
    raw_text: str


@dataclass
class ParsedEntry:
    """One Briggs entry: a PN headform + its toponym attestations."""

    name: PersonalNameRecord
    attestations: list[AttestationRecord]


# ---------------------------------------------------------------- text shaping


def _normalize_form(text: str) -> str:
    """Drop combining marks and lower-case; ASCII-fold remaining
    letters so a lookup on "Eadwulf" finds the Ēadwulf headform."""
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in nfd if not unicodedata.combining(c))
    # Compatibility for letters with no NFD decomposition (æ, þ, etc.)
    mapping = {
        "æ": "ae",
        "Æ": "Ae",
        "œ": "oe",
        "Œ": "Oe",
        "þ": "th",
        "Þ": "Th",
        "ð": "d",
        "Ð": "D",
        "ƿ": "w",
        "Ƿ": "W",
        "ø": "o",
        "Ø": "O",
        "ł": "l",
        "Ł": "L",
    }
    folded = "".join(mapping.get(c, c) for c in stripped)
    return folded.lower().strip()


def _strip_uncertainty(token: str) -> tuple[str, bool, bool]:
    """Peel leading '?' / '??' off a token. Returns (cleaned, is_uncertain,
    is_serious_doubt)."""
    if token.startswith("??"):
        return token[2:], True, True
    if token.startswith("?"):
        return token[1:], True, False
    return token, False, False


# ---------------------------------------------------------------- column reconstruction


def _split_pages(lines: list[str]) -> list[list[str]]:
    """Split the index lines into pages by detecting standalone
    page-number markers. The marker is a line whose content (after
    strip) is a pure decimal integer; in this PDF they're centered
    with leading whitespace padding to ~col 40."""
    pages: list[list[str]] = [[]]
    for line in lines:
        stripped = line.strip()
        if stripped.isdigit() and len(stripped) <= 4:
            # End the current page, start a new one. Drop the
            # page-number line itself.
            if pages[-1]:
                pages.append([])
            continue
        pages[-1].append(line)
    # Drop trailing empty bucket if the file ended right after a page
    # marker.
    if pages and not pages[-1]:
        pages.pop()
    return pages


def _column_reconstruct(page_lines: list[str]) -> str:
    """For one page, emit left-column content top-to-bottom, then
    right-column content top-to-bottom — giving a linear reading
    order from the original two-column layout.

    Pure whitespace lines act as paragraph breaks within a column
    (PN entries don't span across them within one column, but the
    columns themselves are independent vertical strips of text)."""
    left: list[str] = []
    right: list[str] = []
    for line in page_lines:
        if len(line) <= COLUMN_BOUNDARY:
            # Short line: left-column-only content (or blank).
            left.append(line.rstrip())
            right.append("")
        else:
            left.append(line[:COLUMN_BOUNDARY].rstrip())
            right.append(line[COLUMN_BOUNDARY:].rstrip())
    # Trim leading/trailing blanks in each column to avoid spurious
    # blank-line entry boundaries.
    while left and not left[0]:
        left.pop(0)
    while left and not left[-1]:
        left.pop()
    while right and not right[0]:
        right.pop(0)
    while right and not right[-1]:
        right.pop()
    return "\n".join(left + right)


def _reconstruct_index_stream(path: Path) -> Iterator[str]:
    """Yield the index body in linear reading order, page by page.

    Skips everything before line ``INDEX_FIRST_LINE_NUM`` (front
    matter through the coverage / editorial-conventions sections)."""
    raw = path.read_text(encoding="utf-8").splitlines()
    body = raw[INDEX_FIRST_LINE_NUM:]
    # Drop any "The index" header lines and section markers like "—A—"
    # (they're not entries; they appear at the top of each letter group).
    # We keep them in the stream because the parser uses them as
    # alphabet-section delimiters.
    pages = _split_pages(body)
    for page in pages:
        yield _column_reconstruct(page)


# ---------------------------------------------------------------- entry splitting

# Sentence-terminator period: a period followed by whitespace and a
# capital-letter (or ?-prefixed capital) start. This is the most
# reliable signal we have post-bold-stripping for "this entry just
# ended; next entry starts here." The lookbehind/lookahead deliberately
# avoid splitting on intra-body abbreviation periods like ``n.d.``
# (which are followed by lowercase, not capital) and ``pp.61–4`` (which
# are followed by digit).
RE_ENTRY_TERMINATOR = re.compile(r"\.\s+(?=\?{0,2}[A-ZĀĒĪŌŪÆǢÐÞŒŁǷǺ])")

# Section markers like ``—A—`` / ``—B—`` head each alphabet group;
# they are typographic, not entries.
RE_SECTION_HEADER = re.compile(r"\s*—[A-Za-z]—\s*")


def _entry_blocks(stream: str) -> Iterator[str]:
    """Split a reconstructed-stream string into one entry blob per
    yield. The Briggs source ends every entry with ``.`` (sometimes
    inside a county-code paren: ``(O).``); the boundary between
    entries is therefore "period followed by whitespace and the
    next entry's capital-letter headform."

    Column-internal line wraps are NOT entry boundaries — entries
    routinely span multiple lines within one column. Joining the
    page stream into one continuous string and splitting only on
    the period-terminator regex sidesteps the wrap-vs-boundary
    ambiguity that defeated the line-based heuristic.
    """
    # Drop page-header echo and the alphabet-section markers.
    cleaned = stream.replace("The index", " ")
    cleaned = RE_SECTION_HEADER.sub(" ", cleaned)
    # Collapse column-internal line wraps into spaces. Two-space and
    # multi-space runs (table-of-contents columnar gaps) compress to
    # single spaces.
    one_line = " ".join(line.strip() for line in cleaned.split("\n") if line.strip())
    one_line = re.sub(r"\s+", " ", one_line)
    # Split at sentence-terminating periods.
    parts = RE_ENTRY_TERMINATOR.split(one_line)
    for part in parts:
        part = part.strip().rstrip(".").strip()
        if len(part) < 2:
            continue
        yield part


# ---------------------------------------------------------------- entry parsing

# Tokenize an entry into headform + body. The headform is the first
# capitalized token (with optional ? prefix and (b)a-style insertion);
# the body is the rest, which holds citation refs + attestations.
RE_HEADFORM_TOKEN = re.compile(
    r"^(\?{0,2})"
    r"([A-ZĀĒĪŌŪÆǢÐÞŒŁǷǺĀ-ſ]"
    r"[A-Za-zÀ-ɏǣ̄̆̃̇̈Ā-ſʹ()]*)"
    r"(?:\s+|$)"
)


def _expand_bracket_variants(name: str) -> list[str]:
    """Expand ``Ab(b)a`` → ``["Aba", "Abba"]`` and ``Gǣg(a)`` →
    ``["Gǣg", "Gǣga"]``. The bracketed-character form is Briggs's
    convention for compactly listing two near-identical headforms;
    expanding here means downstream lookups against either variant
    find the row."""
    if "(" not in name:
        return [name]
    # Single-bracket case: keep with/without the bracketed content.
    out: list[str] = []
    without = re.sub(r"\([^)]*\)", "", name)
    with_ = re.sub(r"[()]", "", name)
    if without:
        out.append(without)
    if with_ and with_ != without:
        out.append(with_)
    return out or [name]


def _parse_entry(entry_text: str) -> ParsedEntry | None:
    """Parse one entry blob. Returns ``None`` for blobs we couldn't
    recognise as a PN entry (alphabet headers etc. should already be
    filtered by ``_entry_blocks`` but the safety net stays)."""
    m = RE_HEADFORM_TOKEN.match(entry_text)
    if not m:
        return None
    raw_headform = m.group(2)
    body = entry_text[m.end() :].strip()

    # Briggs occasionally appends "fem" right after the headform to
    # mark feminine names ("Abarhilda fem PASE1."). Detect + strip.
    is_feminine = False
    fem_match = re.match(r"^(fem|femin)\b\s*", body)
    if fem_match:
        is_feminine = True
        body = body[fem_match.end() :].strip()

    # Briggs's entry head order is:
    #   Headform [fem]? PASE# [DLV]? [ASCh#...]? [(lang_hints)]? attestations
    # So citation tokens come BEFORE the optional language-hint paren.
    # Pull citations first, then the lang-hint group, then attestations.
    pase_count: int | None = None
    has_dlv = False
    ascharter_refs: list[str] = []
    tokens = body.split()
    consumed = 0
    for tok in tokens:
        cleaned = tok.rstrip(",;.")
        if RE_PASE.match(cleaned):
            pase_count = int(RE_PASE.match(cleaned).group(1))
            consumed += 1
            continue
        if RE_DLV.match(cleaned):
            has_dlv = True
            consumed += 1
            continue
        if RE_ASCH.match(cleaned):
            ascharter_refs.append(cleaned)
            consumed += 1
            continue
        break
    body = " ".join(tokens[consumed:]).strip()

    # Language hints in parens immediately after the citation block:
    # ``(OE)``, ``(Bib, OE)``, ``(ODan, OE)``. Only treat the leading
    # paren group as a lang hint if every token inside is a known
    # language code — otherwise this is the first toponym's (CC) tag.
    language_hints: list[str] = []
    lang_match = re.match(r"^\(([^)]+)\)\s*", body)
    if lang_match:
        tokens = [t.strip() for t in lang_match.group(1).split(",")]
        if all(t in LANGUAGE_HINTS for t in tokens):
            language_hints = tokens
            body = body[lang_match.end() :].strip()
            # The "fem" marker can also live inside the language-hint
            # parens for ME-form names; treat it the same way.
            if "fem" in language_hints:
                is_feminine = True
                language_hints = [h for h in language_hints if h != "fem"]

    name = PersonalNameRecord(
        headform=raw_headform,
        normalized_form=_normalize_form(raw_headform),
        language_hints=language_hints,
        is_feminine=is_feminine,
        pase_count=pase_count,
        has_dlv=has_dlv,
        ascharter_refs=ascharter_refs,
        raw_entry=entry_text,
    )

    attestations = list(_parse_attestations(body))

    return ParsedEntry(name=name, attestations=attestations)


# ---------------------------------------------------------------- attestation parsing

# A toponym attestation candidate is a token (or short multi-word
# token) followed by a county code in parens or trailing position.
# Examples from the text:
#   "?Absol (Ess)"
#   "Abban wylle 996 in Benson (O)"
#   "Abingdon (Bk)"
#   "Aca DLV; ?Acton (ND)"
#   "Abington×2 (Ca)"
#   "Adelwesdenn in Rolvenden (KW)"
#
# Strategy: split body on `;` to get semicolon-separated attestation
# groups (Briggs's convention), then within each group split on
# `,` to get individual toponyms. Each terminal toponym carries the
# trailing (CC) tag; preceding ones in the group inherit it.
#
# Date qualifiers (1278, Hy3, n.d., 13th) appear immediately after
# the toponym form. Multi-word toponyms (Abban wylle, Habels Hous)
# are kept by treating the whole token before the date / paren
# as the toponym_form.

RE_TRAILING_COUNTY = re.compile(r"\s*\(([A-Za-z]+)\)\s*$")
RE_DATE_PREFIX = re.compile(
    r"\s+("
    r"n\.d\.|\d{3,4}(?:[×-]\d{1,4})?(?:th)?|"
    r"Hy[1-8]|Edw[1-3]|Ric[1-3]|John|Steph"
    r")(?:\s+|$)"
)


def _parse_attestations(body: str) -> Iterator[AttestationRecord]:
    """Yield AttestationRecord rows from the body text.

    The grammar (loosely): groups are separated by ``;``; within a
    group, attestations are separated by ``,``; each group ends with
    a county code in parens, which all attestations in the group
    inherit. Date qualifiers ride between the toponym form and the
    county tag.

    This is forgiving — we accept whatever county code we see and
    leave validation to the caller (county_canonical lookup). Forms
    that don't end in a recognisable (CC) marker are skipped (front-
    matter noise, language-only tokens, etc.).
    """
    if not body:
        return
    # Trim final period the source uses to end each entry.
    body = body.rstrip(".")
    for group in body.split(";"):
        group = group.strip()
        if not group:
            continue
        # Pull the trailing county code; if absent, this group is
        # malformed for our purposes and we skip it.
        m = RE_TRAILING_COUNTY.search(group)
        if not m:
            # Some groups end with citation refs only (no toponym),
            # e.g. "Acca PASE10 DLV; ASCh7–8.72; Accott, Acland (D);"
            # The middle "ASCh7–8.72" group is a citation; not an
            # attestation. Drop.
            continue
        county_code = m.group(1)
        if county_code in LANGUAGE_HINTS:
            # False positive: this paren group is a language-hint tail.
            continue
        if county_code not in COUNTY_CODE_TO_NAME:
            continue
        county_canonical = COUNTY_CODE_TO_NAME[county_code]
        head = group[: m.start()].strip().rstrip(",")
        if not head:
            continue
        for raw_item in head.split(","):
            item = raw_item.strip()
            if not item:
                continue
            cleaned, is_unc, is_doubt = _strip_uncertainty(item)
            # Pull a date qualifier embedded in the item.
            date_match = RE_DATE_PREFIX.search(cleaned)
            if date_match:
                date_qualifier = date_match.group(1)
                # Toponym is everything before the date; anything
                # after the date (e.g. "in Asthall") is parenthetical
                # context. We keep the leading toponym only.
                toponym_form = cleaned[: date_match.start()].strip()
            else:
                date_qualifier = None
                toponym_form = cleaned.strip()
            toponym_form = toponym_form.rstrip(",;")
            if not toponym_form:
                continue
            # Some Briggs items embed an attested-orthographic-variant
            # in their toponym form via "X in Y" — e.g.
            # "Adelwesdenn in Rolvenden". We treat "Rolvenden" (post-
            # "in") as the modern toponym and "Adelwesdenn" (pre-"in")
            # as the attested variant of the PN's surface in that
            # location. Date qualifier (if any) is parsed FIRST so the
            # "in" split sees no date-bearing fragments.
            attested_variant: str | None = None
            if " in " in toponym_form:
                left, right = toponym_form.split(" in ", 1)
                attested_variant = left.strip().rstrip(",")
                toponym_form = right.strip()
            yield AttestationRecord(
                toponym_form=toponym_form,
                attested_variant=attested_variant,
                county_code=county_code,
                county_canonical=county_canonical,
                date_qualifier=date_qualifier,
                is_uncertain=is_unc,
                is_serious_doubt=is_doubt,
                source_citation=None,
                raw_text=raw_item.strip(),
            )


# ---------------------------------------------------------------- top-level


def parse_briggs_index(path: Path) -> Iterator[ParsedEntry]:
    """Yield ParsedEntry instances for every PN entry in the index.

    The top-level orchestrator: reconstructs the two-column stream,
    splits into entry blobs, and parses each. Entries with no
    toponym attestations (Aalfra DLV.) are still yielded — they
    carry useful citation metadata.
    """
    for page_stream in _reconstruct_index_stream(path):
        for blob in _entry_blocks(page_stream):
            entry = _parse_entry(blob)
            if entry is None:
                continue
            # Expand bracket-variant headforms into separate entries
            # that share the same attestations.
            variants = _expand_bracket_variants(entry.name.headform)
            if len(variants) == 1:
                yield entry
                continue
            for variant in variants:
                cloned_name = PersonalNameRecord(
                    headform=variant,
                    normalized_form=_normalize_form(variant),
                    language_hints=list(entry.name.language_hints),
                    is_feminine=entry.name.is_feminine,
                    pase_count=entry.name.pase_count,
                    has_dlv=entry.name.has_dlv,
                    ascharter_refs=list(entry.name.ascharter_refs),
                    raw_entry=entry.name.raw_entry,
                )
                yield ParsedEntry(name=cloned_name, attestations=list(entry.attestations))


# ---------------------------------------------------------------- ingestion


@dataclass
class IngestStats:
    """Per-run counters reported on stderr."""

    entries_seen: int = 0
    personal_names_inserted: int = 0
    personal_names_skipped: int = 0  # already present (idempotent)
    attestations_inserted: int = 0
    attestations_skipped: int = 0  # dedup-key collision
    attestations_unknown_county: int = 0


def ingest_briggs_index(
    db,
    txt_path: Path,
    *,
    on_progress=None,
) -> IngestStats:
    """Stream entries from a Briggs .txt and INSERT OR IGNORE them into
    the personal_name and personal_name_toponym_attestation tables.

    Idempotent: re-running on the same DB is a no-op (UNIQUE indexes
    on (headform, source_doc) and the COALESCE-padded attestation
    dedup tuple absorb duplicates). The ``on_progress`` callback is
    invoked every ~250 entries with the current IngestStats so a
    long-running ingest can surface progress per the workspace
    convention.

    ``db`` is a ``LexiconDB`` (it just needs ``.conn`` and
    ``.commit()``). Import-cycle-avoiding duck-type to keep this
    module light.
    """
    import json

    stats = IngestStats()
    conn = db.conn
    for entry in parse_briggs_index(txt_path):
        stats.entries_seen += 1

        # personal_name UPSERT — UNIQUE(headform, source_doc) makes
        # INSERT OR IGNORE the idempotent path. We then SELECT the row
        # to get its id for the attestation foreign keys regardless of
        # whether this run inserted it or a prior run did.
        lang_hints_json = (
            json.dumps(entry.name.language_hints) if entry.name.language_hints else None
        )
        ascharter_json = (
            json.dumps(entry.name.ascharter_refs) if entry.name.ascharter_refs else None
        )
        cur = conn.execute(
            "INSERT OR IGNORE INTO personal_name ("
            "headform, normalized_form, language_hints, is_feminine, "
            "pase_count, has_dlv, ascharter_refs, source_doc, raw_entry"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.name.headform,
                entry.name.normalized_form,
                lang_hints_json,
                1 if entry.name.is_feminine else 0,
                entry.name.pase_count,
                1 if entry.name.has_dlv else 0,
                ascharter_json,
                SOURCE_DOC,
                entry.name.raw_entry,
            ),
        )
        if cur.rowcount > 0:
            stats.personal_names_inserted += 1
        else:
            stats.personal_names_skipped += 1
        pn_row = conn.execute(
            "SELECT id FROM personal_name WHERE headform = ? AND source_doc = ?",
            (entry.name.headform, SOURCE_DOC),
        ).fetchone()
        if pn_row is None:
            # Should not happen post-INSERT OR IGNORE, but guard anyway.
            continue
        pn_id = pn_row["id"] if hasattr(pn_row, "keys") else pn_row[0]

        for att in entry.attestations:
            if att.county_code not in COUNTY_CODE_TO_NAME:
                stats.attestations_unknown_county += 1
                continue
            cur = conn.execute(
                "INSERT OR IGNORE INTO personal_name_toponym_attestation ("
                "personal_name_id, toponym_form, attested_variant, "
                "county_code, county_canonical, date_qualifier, "
                "is_uncertain, is_serious_doubt, source_doc, raw_text"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pn_id,
                    att.toponym_form,
                    att.attested_variant,
                    att.county_code,
                    att.county_canonical,
                    att.date_qualifier,
                    1 if att.is_uncertain else 0,
                    1 if att.is_serious_doubt else 0,
                    SOURCE_DOC,
                    att.raw_text,
                ),
            )
            if cur.rowcount > 0:
                stats.attestations_inserted += 1
            else:
                stats.attestations_skipped += 1

        if on_progress is not None and stats.entries_seen % 250 == 0:
            on_progress(stats)

    db.commit()
    if on_progress is not None:
        on_progress(stats)
    return stats


if __name__ == "__main__":
    # Quick visual sanity check against the staged file.
    import sys

    src = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/devon/521Studios/wyrd-source-staging/briggs_2024_personal_names_index.txt"
    )
    count_entries = 0
    count_attestations = 0
    for entry in parse_briggs_index(src):
        count_entries += 1
        count_attestations += len(entry.attestations)
        if count_entries <= 25:
            atts = ", ".join(f"{a.toponym_form} ({a.county_code})" for a in entry.attestations)
            print(
                f"{entry.name.headform:>16s}  "
                f"lang={entry.name.language_hints}  "
                f"PASE={entry.name.pase_count}  "
                f"DLV={entry.name.has_dlv}  "
                f"fem={entry.name.is_feminine}  "
                f"-> {atts}"
            )
    print(f"\nTOTAL entries={count_entries} attestations={count_attestations}")
