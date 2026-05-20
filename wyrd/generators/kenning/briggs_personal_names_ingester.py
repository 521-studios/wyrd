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

import json
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- constants

# Identifier this ingest writes to ``source_doc`` rows. Downstream
# supersede/re-ingest logic keys on it.
SOURCE_DOC = "briggs_2024_personal_names_index"

# Column boundary measured from the pdftotext output. The left column
# occupies chars 0–37 and the right column begins at char 38. Verified
# by inspection of the index opening at page 19 (Aalfra DLV. ...
# Abbud PASE2.).
COLUMN_BOUNDARY = 38

# Zero-based slice index — everything before this line of the .txt is
# front-matter (title page, preface, county-code coverage list,
# editorial conventions, bibliography). Used in a `raw[N:]` slice in
# ``_reconstruct_index_stream``. Equivalent to line 795 of the .txt
# (1-based), where the first "—A—" section marker appears.
INDEX_FIRST_LINE_NUM = 794

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


# ---------------------------------------------------------------- data shapes


@dataclass
class PersonalNameRecord:
    """One PN headform parsed from the index.

    ``normalized_form`` is auto-derived from ``headform`` in
    ``__post_init__`` so callers can't construct one with a mismatched
    pair. Pass ``normalized_form=""`` (the default) and let the
    post-init compute it.
    """

    headform: str
    normalized_form: str = ""
    language_hints: list[str] = field(default_factory=list)
    is_feminine: bool = False
    pase_count: int | None = None
    has_dlv: bool = False
    ascharter_refs: list[str] = field(default_factory=list)
    raw_entry: str = ""

    def __post_init__(self) -> None:
        # Always re-derive so a caller passing a stale value can't
        # introduce drift between the two fields.
        self.normalized_form = _normalize_form(self.headform)


@dataclass
class AttestationRecord:
    """One (PN, toponym, county) attestation row.

    ``county_canonical`` is exposed as a property derived from
    ``county_code`` against ``COUNTY_CODE_TO_NAME`` so the two fields
    can't drift apart. Unknown county codes return an empty string
    instead of raising — the ingester counts and drops those rows
    via the ``attestations_unknown_county`` counter, which keeps the
    schema-validation seam in one place.

    The ``is_serious_doubt`` flag implies ``is_uncertain`` (``??`` is
    a strict refinement of ``?``); ``__post_init__`` enforces this.
    """

    toponym_form: str
    attested_variant: str | None
    county_code: str
    date_qualifier: str | None
    is_uncertain: bool
    is_serious_doubt: bool
    raw_text: str

    def __post_init__(self) -> None:
        if self.is_serious_doubt and not self.is_uncertain:
            # ?? is a strict refinement of ? — silently promote rather
            # than reject so an upstream parser bug doesn't lose the row.
            self.is_uncertain = True

    @property
    def county_canonical(self) -> str:
        """Canonical county name for ``county_code``; empty string if
        the code isn't in ``COUNTY_CODE_TO_NAME`` (caller should treat
        empty-canonical as an unknown-county skip signal)."""
        return COUNTY_CODE_TO_NAME.get(self.county_code, "")


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
# ended; next entry starts here." The positive lookahead deliberately
# avoids splitting on intra-body abbreviation periods: ``pp.61–4`` is
# followed by digits (not capital) and ``n.d.`` ends in a period
# followed by lowercase ``in`` / ``and`` (not capital).
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
        pase_m = RE_PASE.match(cleaned)
        if pase_m:
            pase_count = int(pase_m.group(1))
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
        # Don't filter unknown county codes here — let the ingester
        # see them, increment its unknown-county counter, and drop the
        # row at insert time. Single seam for the visibility signal.
        head = group[: m.start()].strip().rstrip(",")
        if not head:
            continue
        for raw_item in head.split(","):
            item = raw_item.strip()
            if not item:
                continue
            cleaned, is_unc, is_doubt = _strip_uncertainty(item)
            # Pull a date qualifier embedded in the item. The
            # surrounding shape can be:
            #   "Abington×2"                  — no date, no "in Y"
            #   "Adelwesdenn in Rolvenden"    — no date, "X in Y"
            #   "Abington 1257"               — date, no "in Y"
            #   "Abban wylle 996 in Benson"   — date AND "X date in Y"
            # In all forms with " in ", X is the attested orthographic
            # variant of the PN in that toponym and Y is the modern
            # toponym. The date (if any) sits between X and "in Y".
            date_qualifier: str | None = None
            attested_variant: str | None = None
            date_match = RE_DATE_PREFIX.search(cleaned)
            if date_match:
                date_qualifier = date_match.group(1)
                pre_date = cleaned[: date_match.start()].strip()
                post_date = cleaned[date_match.end() :].strip()
                if post_date.startswith("in "):
                    # "X <date> in Y": X is attested variant, Y is the
                    # modern toponym. Without this branch we'd silently
                    # drop "in Benson" and treat "Abban wylle" as the
                    # toponym.
                    attested_variant = pre_date.rstrip(",")
                    toponym_form = post_date[3:].strip()
                else:
                    toponym_form = pre_date
            else:
                toponym_form = cleaned.strip()
                if " in " in toponym_form:
                    left, right = toponym_form.split(" in ", 1)
                    attested_variant = left.strip().rstrip(",")
                    toponym_form = right.strip()
            toponym_form = toponym_form.rstrip(",;")
            if not toponym_form:
                continue
            yield AttestationRecord(
                toponym_form=toponym_form,
                attested_variant=attested_variant,
                county_code=county_code,
                date_qualifier=date_qualifier,
                is_uncertain=is_unc,
                is_serious_doubt=is_doubt,
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
    pn_rows_emitted: int = 0
    attestation_rows_emitted: int = 0
    attestations_unknown_county: int = 0
    personal_names_inserted: int = 0
    attestations_inserted: int = 0
    jsonl_path: Path | None = None


# Title used in the per-source JSONL header row. The build pipeline
# stores this in the ``source.title`` column on rebuild so dump→wipe→
# rebuild→dump is identity.
_BRIGGS_SOURCE_TITLE = (
    "An Index to Personal Names in English Place-Names "
    "(Keith Briggs, EPNS Supplementary Series 2, 3rd revised pdf edn 2024)"
)


def _emit_source_row(sink) -> None:
    """Write the canonical ``_type: source`` row that anchors every L2
    JSONL file (the build pipeline requires exactly one per file)."""
    row = {
        "_type": "source",
        "ref": SOURCE_DOC,
        "title": _BRIGGS_SOURCE_TITLE,
        "notes": "Generated by `wyrd kenning lexicon ingest-briggs-personal-names`.",
    }
    sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def _emit_personal_name_row(sink, entry: ParsedEntry) -> None:
    """One ``_type: personal_name`` keyed-state row per Briggs entry.
    JSON-encoded list payloads (language_hints, ascharter_refs) match
    the column shapes the DB inserter expects."""
    row: dict[str, object] = {
        "_type": "personal_name",
        "ref": entry.name.headform,
        "headform": entry.name.headform,
        "normalized_form": entry.name.normalized_form,
        "is_feminine": 1 if entry.name.is_feminine else 0,
        "has_dlv": 1 if entry.name.has_dlv else 0,
        "source_doc": SOURCE_DOC,
    }
    if entry.name.language_hints:
        row["language_hints"] = json.dumps(entry.name.language_hints, ensure_ascii=False)
    if entry.name.ascharter_refs:
        row["ascharter_refs"] = json.dumps(entry.name.ascharter_refs, ensure_ascii=False)
    if entry.name.pase_count is not None:
        row["pase_count"] = entry.name.pase_count
    if entry.name.raw_entry:
        row["raw_entry"] = entry.name.raw_entry
    sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def _emit_attestation_row(sink, headform: str, att: AttestationRecord) -> None:
    """One ``_type: personal_name_toponym_attestation`` list row per
    (PN, toponym, county) occurrence. ``personal_name_ref`` is the FK
    the build pipeline resolves to a ``personal_name.id`` via the
    per-file headform → id map."""
    row: dict[str, object] = {
        "_type": "personal_name_toponym_attestation",
        "personal_name_ref": headform,
        "toponym_form": att.toponym_form,
        "county_code": att.county_code,
        "county_canonical": att.county_canonical,
        "is_uncertain": 1 if att.is_uncertain else 0,
        "is_serious_doubt": 1 if att.is_serious_doubt else 0,
        "source_doc": SOURCE_DOC,
    }
    if att.attested_variant:
        row["attested_variant"] = att.attested_variant
    if att.date_qualifier:
        row["date_qualifier"] = att.date_qualifier
    if att.raw_text:
        row["raw_text"] = att.raw_text
    sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def emit_briggs_jsonl(
    txt_path: Path,
    jsonl_path: Path,
    *,
    on_progress=None,
) -> IngestStats:
    """Parse a Briggs .txt and write its events as a self-contained
    L2 JSONL file at ``jsonl_path`` (wyrd-11zh).

    The file is the canonical artifact — once written, the DB can be
    blown away and reconstructed via ``rebuild-from-jsonl`` without
    re-parsing the source PDF. Per-entry flushes mean a SIGTERM
    leaves a still-loadable partial file (every committed entry's
    rows are followed by a flush).

    Does NOT touch the DB. Use :func:`ingest_briggs_index` for the
    parse-then-load CLI path.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    stats = IngestStats(jsonl_path=jsonl_path)
    # The parser can yield the same headform multiple times — bracket-
    # variant expansion (Ab(b)a → Aba+Abba), and the index itself
    # occasionally lists a name in both a diacritic-bearing and an
    # ASCII form that normalize to the same headform when both appear
    # verbatim. To preserve the first-rich-payload-wins semantics the
    # original ingester relied on (which carried, e.g., PASE=58 for
    # "Eadwulf" from its first occurrence rather than PASE=None from a
    # later citation-light duplicate), dedupe headforms during emit.
    # Attestations are emitted for every parser yield — the build pass
    # dedupes them via the COALESCE-padded UNIQUE index.
    seen_headforms: set[str] = set()
    with jsonl_path.open("w", encoding="utf-8") as sink:
        _emit_source_row(sink)
        sink.flush()
        for entry in parse_briggs_index(txt_path):
            stats.entries_seen += 1
            if entry.name.headform not in seen_headforms:
                seen_headforms.add(entry.name.headform)
                _emit_personal_name_row(sink, entry)
                stats.pn_rows_emitted += 1
            for att in entry.attestations:
                if att.county_code not in COUNTY_CODE_TO_NAME:
                    stats.attestations_unknown_county += 1
                    # Still emit the row so the artifact is faithful;
                    # the build pipeline drops unknown-county rows via
                    # the same path so dump→rebuild stays consistent.
                _emit_attestation_row(sink, entry.name.headform, att)
                stats.attestation_rows_emitted += 1
            sink.flush()
            if on_progress is not None and stats.entries_seen % 250 == 0:
                on_progress(stats)
    if on_progress is not None:
        on_progress(stats)
    return stats


def ingest_briggs_index(
    db,
    txt_path: Path,
    *,
    jsonl_path: Path | None = None,
    on_progress=None,
) -> IngestStats:
    """Parse a Briggs .txt, emit JSONL events to ``jsonl_path``, AND
    load them into the lexicon DB via the shared
    :func:`wyrd.generators.kenning.jsonl.build.build_from_jsonl`
    pipeline (wyrd-11zh).

    Architectural contract: the JSONL file is the canonical artifact;
    the DB is a queryable index rebuilt from JSONL. Per the
    db-reconstructibility-reviewer (wyrd-fe1v), the recipe
    ``rm ~/.wyrd/lexicon.db && wyrd kenning lexicon rebuild-from-jsonl``
    must reproduce the Briggs rows from the JSONL alone — no .txt
    re-parse, no PDF re-fetch.

    Idempotent: the build pipeline uses INSERT OR IGNORE with UNIQUE
    indexes on (headform, source_doc) and the COALESCE-padded
    attestation dedup tuple. Re-running on an already-populated DB
    is a no-op.

    ``db`` is a ``LexiconDB`` (needs ``.conn`` and ``.commit()``).
    ``jsonl_path`` defaults to ``data/mining/<SOURCE_DOC>.jsonl``.
    """
    from wyrd.generators.kenning.jsonl.build import build_from_jsonl

    if jsonl_path is None:
        jsonl_path = Path("data/mining") / f"{SOURCE_DOC}.jsonl"
    stats = emit_briggs_jsonl(txt_path, jsonl_path, on_progress=on_progress)
    counts = build_from_jsonl(db.conn, [jsonl_path])
    stats.personal_names_inserted = counts.get("personal_name", 0)
    stats.attestations_inserted = counts.get("personal_name_toponym_attestation", 0)
    return stats


if __name__ == "__main__":
    # Parser-debugging entry point. Use the CLI subcommand
    # ``wyrd kenning lexicon ingest-briggs-personal-names`` for
    # production ingest; this is just a visual smoke check.
    import sys

    if len(sys.argv) < 2:
        sys.exit(
            "Usage: python -m wyrd.generators.kenning.briggs_personal_names_ingester <path_to_briggs.txt>"
        )
    src = Path(sys.argv[1])
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
