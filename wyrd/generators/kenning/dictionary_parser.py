"""Parser for alphabetical place-name dictionaries.

Mawer's *Northumberland and Durham* (1920), Ekwall's *Lancashire* (1922),
Johnston's *Place-Names of Scotland* (1892), and most other non-Skeat
volumes in the corpus follow a flat alphabetical layout rather than Skeat's
suffix-section structure:

    Acklington (Warkworth). 1176 Pipe Eclinton; 1186 Aclinton; ...

    O.E. Æcceling(a)tun = farm of Æccel or of his sons. ...

    Acomb [ækəm] (Bywell St Peter). 1268 Ipm. Akum; ...

    ...

The parser segments by detecting headwords at paragraph starts, then takes
each entry body up to the next headword. The result is a list of
`ParsedEntry` instances (same shape as the Skeat parser produces) so the
existing LLM extractor and ingester work unchanged.

Books are typically front-matter-heavy with phonological intros, source
abbreviation tables, and indexes. We use a "body section" detector to find
the start of alphabetical content (often headed "PART I" or similar) and
stop at the next part marker (Part II = elements, Part III = personal
names) which we don't want to mine as toponyms.
"""

from __future__ import annotations

import re

from wyrd.generators.kenning.skeat_parser import ParsedEntry, _shorten

# --- body-section detection ------------------------------------------------

# Markers that identify the start of the alphabetical place-name section.
# Order matters: try the most specific first.
_BODY_START_PATTERNS = [
    re.compile(r"^\s*PART\s+I\b", re.IGNORECASE),
    re.compile(r"^\s*PLACE-NAMES.*ALPHABETICAL", re.IGNORECASE),
    re.compile(r"^\s*ALPHABETICAL\s+(LIST|INDEX)", re.IGNORECASE),
]

# Markers that end the alphabetical section: "Part II" (elements) or
# "Part III" (personal names) signal we're past the toponym dictionary and
# into supplementary material we don't want as toponyms.
_BODY_END_PATTERNS = [
    re.compile(r"^\s*PART\s+(II|III|IV)\b", re.IGNORECASE),
    re.compile(r"^\s*ELEMENTS\s+FOUND", re.IGNORECASE),
    re.compile(r"^\s*PERSONAL\s+NAMES\s+FOUND", re.IGNORECASE),
    re.compile(r"^\s*INDEX\s*$", re.IGNORECASE),
    re.compile(r"^\s*PHONOLOGY\s*$", re.IGNORECASE),
    re.compile(r"^\s*APPENDIX\b", re.IGNORECASE),
]


def find_body_bounds(lines: list[str]) -> tuple[int, int]:
    """Locate the alphabetical body inside `lines`.

    Returns (start, end) line indices.

    Strategy: find ALL occurrences of body-start markers. The first is
    typically the TOC entry, the second is the actual body header. We take
    the last occurrence as the body start (works for Mawer's "PART I"
    appearing in TOC then again as a body chapter heading).

    If no explicit marker is found we fall back to the first headword-shaped
    line in the latter half of the document.
    """
    start = -1
    starts: list[int] = []
    for i, line in enumerate(lines):
        if any(p.match(line) for p in _BODY_START_PATTERNS):
            starts.append(i)
    if starts:
        # Last match is the body header; first is usually the TOC. Take the
        # last so we skip the TOC entirely.
        start = starts[-1] + 1

    if start < 0:
        # Fall back: first plausible headword in the second half.
        midpoint = len(lines) // 2
        for i in range(midpoint, len(lines)):
            m = _ENTRY_HEADWORD.match(lines[i])
            if m and _is_real_headword(m.group("name")):
                start = i
                break
    if start < 0:
        return 0, len(lines)

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if any(p.match(lines[i]) for p in _BODY_END_PATTERNS):
            end = i
            break
    return start, end


# --- headword detection ----------------------------------------------------

# A headword line starts with a capitalized place name, optionally followed
# by phonetic brackets and/or a parish in parentheses, then a period or comma.
# The place name may be multi-word ("School Aycliffe") and may contain
# hyphens or apostrophes ("Stockton-on-Tees", "St John's").
#
# We DO NOT match lines that look like sentences mid-paragraph; the rule is
# this regex must apply to the full line (or paragraph start), and the
# headword token is followed by a clear delimiter — period + space, or
# parentheses, or square brackets.
_PLACE_NAME = (
    r"[A-Z][A-Za-z'-]+"  # first word (caps initial)
    r"(?:\s+(?:on|in|upon|le|de|of|the|and|St\.?|Saint))*"  # connectors
    r"(?:\s+[A-Z][A-Za-z'-]+){0,4}"  # additional capitalized tokens
)
_ENTRY_HEADWORD = re.compile(
    rf"""
    ^\s*
    (?P<name>{_PLACE_NAME})
    \s*
    (?:\[[^\]]*\]\s*)?    # optional phonetic bracket
    (?:\([^)]*\)\s*)?     # optional parish in parens
    [.,:]                 # period (Mawer/Skeat), comma, or colon (Ekwall)
    """,
    re.VERBOSE,
)


# --- low-quality headword filter -------------------------------------------

# These tokens at the start of a paragraph look like place names but are
# almost always the start of a sentence in Skeat-style prose. We reject
# headwords matching these to keep noise down.
_FALSE_HEADWORDS = {
    "It",
    "This",
    "That",
    "These",
    "There",
    "Here",
    "Now",
    "Then",
    "If",
    "Although",
    "When",
    "While",
    "Where",
    "Why",
    "How",
    "The",
    "A",
    "An",
    "And",
    "But",
    "Or",
    "So",
    "For",
    "As",
    "From",
    "To",
    "In",
    "On",
    "At",
    "By",
    "Of",
    "With",
    "About",
    "Many",
    "Several",
    "Both",
    "All",
    "Some",
    "Most",
    "Few",
    "We",
    "Our",
    "He",
    "She",
    "They",
    "His",
    "Her",
    "Their",
    "I",
    "My",
    "Me",
    # Common etymology-book abbreviations that can appear at line start.
    "OE",
    "ON",
    "AS",
    "Cf",
    "See",
    "Mod",
    "Lat",
    "Goth",
    "OHG",
    "OFr",
    "ME",
    "MFr",
    "MHG",
    "OS",
    "Skr",
    "Welsh",
    "Ir",
    "Gael",
    "DB",
    "BCS",
    "KCD",
    "FA",
    "RB",
    "PR",
    "TN",
    "Ipm",
    "MS",
    "MSS",
    "Pipe",
    "Hist",
    "Ant",
    "Searle",
    "Ekwall",
    "Mawer",
    "Skeat",
    # OCR-mangled abbreviations.
    "Pl",
    "PI",  # plural abbreviation, sometimes OCR'd with capital-I
    # Generic English words that look like headwords but are usually
    # mid-prose continuations or page-artifact fragments in OCR output.
    # These ARE real toponymic elements (Mawer has them in Part II), but
    # we exclude Part II via body bounds, so any occurrence here is noise.
    "Burn",
    "Hill",
    "House",
    "Beck",
    "Law",
    "Field",
    "Hall",
    "Stone",
    "Self-explanatory",
}


# An entry body is real if it contains at least one of these signals.
# Real Mawer/Skeat/Ekwall entries always have a date attestation, an
# OE/ON/etc. marker, or a cross-reference. Sentence fragments don't.
_ENTRY_BODY_SIGNALS = re.compile(
    r"""
    \b(?:
        \d{3,4}                                    # 3-4 digit year (e.g. 1086, 1382)
        | O\.\s*E\.                                # Old English marker
        | A\.\s*S\.                                # Anglo-Saxon marker
        | O\.\s*N\.                                # Old Norse marker
        | A\.\s*F\.                                # Anglo-French marker
        | M\.\s*E\.                                # Middle English marker
        | M\.\s*Lat\.                              # Medieval Latin
        | Domesday | D\.\s*B\.                     # Domesday Book reference
        | Phonology                                # cross-reference to phonology section
        | I\.\s*p\.\s*m\.                          # Inquisitiones post Mortem
        | I\.\s*C\.\s*C\.                          # Inquisitio Comitatus Cantabrigiensis
        | Hatf | Pipe | F\.\s*A\.                  # other common attestation refs
        | Newm | Cl | Ipm
    )\b
    """,
    re.VERBOSE,
)

# Minimum body length safety net. We don't reject every short entry —
# some legitimate Mawer/Skeat entries are brief — but bodies under this
# floor are almost always page-fragment artifacts.
_MIN_ENTRY_BODY_CHARS = 20


def _entry_body_is_real(body: str) -> bool:
    """A body counts as a real entry if it contains at least one
    etymology-signal marker (year, O.E./A.S./O.N. abbreviation, Domesday,
    Phonology cross-ref) AND isn't extreme-short. Filters OCR-broken
    sentence fragments and standalone page artifacts.

    The signal requirement is the load-bearing check; the length floor is
    a last-resort safety net for things like "Burn." with no body content.
    """
    if len(body) < _MIN_ENTRY_BODY_CHARS:
        return False
    return bool(_ENTRY_BODY_SIGNALS.search(body))


# Pattern for detecting a fresh entry start inside an already-flowed paragraph.
# Matches "<Place> (<parish>). <year>" or "<Place>. <year>" — Mawer's signature
# entry opener. Lets us split paragraphs that pack multiple short entries.
_INLINE_ENTRY_BOUNDARY = re.compile(
    rf"""
    (?<=[\s.])              # boundary
    (?P<name>{_PLACE_NAME})
    \s*
    (?:\(\s*[A-Za-z][\w\s.,'-]*\s*\)\s*)?   # parens parish
    \.\s+                  # period + space
    (?:c\.\s*)?            # optional "c." for circa
    \d{{3,4}}              # year
    """,
    re.VERBOSE,
)


def _is_real_headword(name: str) -> bool:
    """Reject obvious false positives.

    Page running titles ("PLACE-NAMES OF SCOTLAND", "DURHAM IN ALPHABETICAL
    ORDER") match the headword regex but are all-caps and multi-word — easy
    to filter.
    """
    if not name:
        return False
    first = name.split()[0]
    if first in _FALSE_HEADWORDS:
        return False
    if len(first) < 2:
        return False
    # All-caps multi-word strings are page running titles, not entries.
    # Real entry headwords are mixed-case (Acklington, Stockton-on-Tees).
    if name.isupper() and " " in name:
        return False
    # Single all-caps tokens like "TAIN" are usually page-header running
    # titles too, since real headwords are capitalized in mixed case
    # ("Tain" not "TAIN").
    return not (name.isupper() and len(name) >= 3)


# --- segmentation ----------------------------------------------------------

# A page-header noise pattern: standalone year markers, page numbers, running
# titles. Stripping these collapses entry continuations.
_PAGE_NOISE = [
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*[A-Z][A-Z\s\-]+\s*\d*\s*$"),  # "ACTON" or "PLACE-NAMES OF DURHAM 5"
    re.compile(r"^\s*\d+\s+[A-Z][A-Z\s\-]+\s*$"),
]


def _strip_noise(lines: list[str]) -> list[str]:
    return [ln for ln in lines if not any(p.match(ln) for p in _PAGE_NOISE)]


def parse_alphabetical_text(text: str, *, max_entries: int | None = None) -> list[ParsedEntry]:
    """Segment an alphabetical-dictionary text into ParsedEntry instances.

    Entries are not etymologically pre-parsed — `elements` is empty,
    `confidence` is "low". Caller is expected to feed `body_text` to a
    semantic extractor (the LLM extractor).
    """
    lines = text.split("\n")
    start, end = find_body_bounds(lines)
    body_lines = _strip_noise(lines[start:end])

    # Group lines into paragraphs (blank-line-delimited).
    paragraphs: list[str] = []
    current: list[str] = []
    for ln in body_lines:
        if ln.strip():
            current.append(ln)
        else:
            if current:
                paragraphs.append(" ".join(s.strip() for s in current))
                current = []
    if current:
        paragraphs.append(" ".join(s.strip() for s in current))

    # Walk paragraphs: each headword paragraph starts a new entry; subsequent
    # paragraphs without a headword are continuation (Mawer often spans an
    # etymology over multiple paragraphs).
    out: list[ParsedEntry] = []
    pending_topo: str | None = None
    pending_body: list[str] = []

    def flush() -> None:
        if pending_topo is None:
            return
        body = " ".join(pending_body).strip()
        body = re.sub(r"\s+", " ", body)
        # Reject paragraph-fragments: bodies too short or lacking any of the
        # etymology-signal markers a real entry always carries (year / OE /
        # AS / Phonology / Domesday). Catches OCR line-break artifacts where
        # mid-sentence fragments look like headword paragraphs.
        if not _entry_body_is_real(body):
            return
        out.append(
            ParsedEntry(
                toponym=pending_topo,
                section_suffix=None,
                historical_form=None,
                elements=[],
                confidence="low",
                source_quote=_shorten(body),
                body_text=body,
            )
        )

    for para in paragraphs:
        flat = re.sub(r"\s+", " ", para).strip()
        # Split paragraphs that pack multiple entries (e.g. Mawer often
        # places three short etymologies in one OCR'd paragraph).
        for chunk in _split_packed_paragraph(flat):
            m = _ENTRY_HEADWORD.match(chunk)
            if m:
                name = m.group("name").strip()
                if _is_real_headword(name):
                    flush()
                    pending_topo = name
                    pending_body = [chunk]
                    if max_entries is not None and len(out) >= max_entries:
                        flush()
                        return out
                    continue
            if pending_topo is not None:
                pending_body.append(chunk)

    flush()
    return out


def _split_packed_paragraph(flat: str) -> list[str]:
    """Split a paragraph at every internal headword-shaped position.

    Mawer's OCR sometimes joins three or four short entries (Langhope,
    Langley, Langton) into one paragraph. We detect each "Place(s). YEAR"
    boundary inside the paragraph and split there. The first chunk keeps
    the paragraph head; subsequent chunks each start at a fresh headword.
    """
    matches = list(_INLINE_ENTRY_BOUNDARY.finditer(flat))
    # Filter to matches that aren't at position 0 (the leading headword
    # is handled by _ENTRY_HEADWORD already) and whose name passes the
    # false-headword filter.
    boundaries = [
        m.start("name")
        for m in matches
        if m.start("name") > 0 and _is_real_headword(m.group("name"))
    ]
    if not boundaries:
        return [flat]
    out: list[str] = []
    last = 0
    for pos in boundaries:
        out.append(flat[last:pos].strip())
        last = pos
    out.append(flat[last:].strip())
    return [c for c in out if c]
