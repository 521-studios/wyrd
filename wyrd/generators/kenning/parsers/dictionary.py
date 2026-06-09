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

from .skeat import ParsedEntry, _shorten

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


def _find_body_end(lines: list[str], from_line: int) -> int:
    """First body-end marker at or after ``from_line`` (else end-of-document).

    Scans from ``from_line`` (not +1) so an end-marker on the very next line —
    common in TOC lines like "PART II. ELEMENTS .... 100" — is recognized and
    collapses the candidate span to zero."""
    for j in range(from_line, len(lines)):
        if any(p.match(lines[j]) for p in _BODY_END_PATTERNS):
            return j
    return len(lines)


def _fallback_body_start(lines: list[str]) -> int:
    """First plausible headword in the second half of the document, or -1 when
    none — the fallback when no body-start marker was found."""
    midpoint = len(lines) // 2
    for i in range(midpoint, len(lines)):
        m = _ENTRY_HEADWORD.match(lines[i])
        if m and _is_real_headword(m.group("name")):
            return i
    return -1


def find_body_bounds(lines: list[str]) -> tuple[int, int]:
    """Locate the alphabetical body inside `lines`.

    Returns (start, end) line indices.

    Strategy:
    1. Find all body-start markers ("PART I", "ALPHABETICAL LIST", etc.).
       For each, find the next body-end marker. Pick the (start, end) span
       that yields the LARGEST body — this beats "always take the last
       Part I", which on books with nested subsection labels (Moore IoM has
       five Part I headings, only the first is the real body) collapses to
       a tiny region.
    2. If no marker is found, fall back to the first headword-shaped line
       in the latter half of the document.
    3. **Tiny-body safety net**: if the resulting span is under ~5% of the
       document, the heuristic almost certainly mis-identified the body
       boundaries. Fall through to scanning the whole doc — the downstream
       `_entry_body_is_real` filter will reject TOC/preface noise. Better
       to scan too much than too little (Joyce 1875, Moore 1890 both
       under-yielded by ~99% before this safety net).
    """

    starts: list[int] = []
    for i, line in enumerate(lines):
        if any(p.match(line) for p in _BODY_START_PATTERNS):
            starts.append(i)

    candidates = [(s + 1, _find_body_end(lines, s + 1)) for s in starts]

    if candidates:
        # Pick the candidate with the largest body span.
        start, end = max(candidates, key=lambda se: se[1] - se[0])
    else:
        start = _fallback_body_start(lines)
        if start < 0:
            return 0, len(lines)
        end = _find_body_end(lines, start + 1)

    # Safety net: tiny bodies on big documents almost always mean the
    # heuristic mis-fired. 5% is conservative — the smallest legitimate
    # dictionary section in the corpus (Mawer 1922's place-name index) is
    # ~30%. Skip the safety net for small docs so unit-test fixtures don't
    # trip it.
    if len(lines) > 1000 and (end - start) < len(lines) // 20:
        return 0, len(lines)
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
#
# Celtic / Gaelic / Manx / Norse markers were added per wyrd-3qq because
# the original English-only set was filtering out legitimate Celtic-corpus
# entries (Joyce, Moore, Johnston Scotland 1892, Gillies Argyll). The
# upstream headword-shape check still gates the regex from firing on every
# prose mention of "Manx" or "Norse"; this just lets the gated paragraphs
# qualify as substantive etymology bodies.
#
# A note on regex anchoring: the closing assertion is `(?!\w)`, not `\b`.
# After a literal `.` character (which most of these dotted markers end with —
# `O.E.`, `M. Ir.`), `\b` only matches when the NEXT character is a word
# character. In typical body text, abbreviations are followed by a space or
# punctuation, so `\b` would silently fail. `(?!\w)` does the right thing
# regardless of what comes after the period. The opening `\b` is safe because
# every alternative starts with either a digit or a letter.
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
        # --- Celtic / Gaelic / Manx / Norse markers (wyrd-3qq) ----------
        # The (?!\w) closing anchor handles trailing periods, so `Gael` matches
        # both "Gael" and "Gael." — no need to enumerate dotted variants.
        | Gael(?:ic)? | Gaedh(?:ealg|elic)?         # Gaelic / Goidelic forms
        | Goidel(?:ic)? | Brython(?:ic)?            # Insular Celtic branch labels
        | (?:Old|Mid\.?|Mod\.?|M\.)\s*Ir(?:ish)?    # Old/Mid/Mod/M. Irish
        | Irish | Ir\.                             # standalone Irish (full word + abbreviation)
        | Manx | Erse                              # Manx (Gaelg) / archaic name for Irish/Gaelic
        | (?:Old\s+)?Norse                         # Norse / Old Norse
        | Welsh | Cymr(?:ic|aeg)?                  # Welsh / Cymric
        | Corn(?:ish)? | Bret(?:on|onique)?        # Cornish / Breton
        | Brit(?:tonic|ish)?                       # Brittonic / British
        | Pictish                                  # Pictish substratum claims
        | gen\.\s*sing\.                           # gen.sing. citations
    )(?!\w)
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

    paragraphs = _group_into_paragraphs(body_lines)

    # Walk paragraphs: each headword paragraph starts a new entry; subsequent
    # paragraphs without a headword are continuation (Mawer often spans an
    # etymology over multiple paragraphs).
    out: list[ParsedEntry] = []
    pending_topo: str | None = None
    pending_body: list[str] = []

    for para in paragraphs:
        flat = re.sub(r"\s+", " ", para).strip()
        # Split paragraphs that pack multiple entries (e.g. Mawer often
        # places three short etymologies in one OCR'd paragraph).
        for chunk in _split_packed_paragraph(flat):
            m = _ENTRY_HEADWORD.match(chunk)
            if m:
                name = m.group("name").strip()
                if _is_real_headword(name):
                    _emit_alpha_entry(out, pending_topo, pending_body)
                    pending_topo = name
                    pending_body = [chunk]
                    if max_entries is not None and len(out) >= max_entries:
                        _emit_alpha_entry(out, pending_topo, pending_body)
                        return out
                    continue
            if pending_topo is not None:
                pending_body.append(chunk)

    _emit_alpha_entry(out, pending_topo, pending_body)
    return out


def _group_into_paragraphs(body_lines: list[str]) -> list[str]:
    """Group body lines into blank-line-delimited paragraphs (each a single
    whitespace-collapsed string)."""
    paragraphs: list[str] = []
    current: list[str] = []
    for ln in body_lines:
        if ln.strip():
            current.append(ln)
        elif current:
            paragraphs.append(" ".join(s.strip() for s in current))
            current = []
    if current:
        paragraphs.append(" ".join(s.strip() for s in current))
    return paragraphs


def _emit_alpha_entry(
    out: list[ParsedEntry], pending_topo: str | None, pending_body: list[str]
) -> None:
    """Append a low-confidence ParsedEntry for the in-progress alphabetical
    entry to ``out``, unless there's no headword or the body fails the
    etymology-signal filter (rejects OCR line-break fragments)."""
    if pending_topo is None:
        return
    body = " ".join(pending_body).strip()
    body = re.sub(r"\s+", " ", body)
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


# --- numbered-list parser (wyrd-5af) ---------------------------------------

# Some sources organize their etymology bodies as a numbered list rather
# than an alphabetical-headword sequence. Longnon's "Les Noms de lieu de
# la France" vol 2 saint-names section is the canonical example: each
# entry begins with a 3- or 4-digit ordinal followed by a period, the
# Latin source name, and the derived French Saint-X form(s):
#
#     1659.  Caradocus ;  Saint-Caradu  (Côtes-du-Nord, Morbihan),
#     Saint-Cadreuc  (Côtes-du-Nord), Saint-Carreuc  (Côtes-du-Nord).
#
#     1660.  Garaunus, martyr (...) :  Saint-Chéron  (Eure, ...).
#
# The alphabetical parser misses these because the headword shape is the
# Latin name (lowercase or capitalized) embedded after the ordinal —
# capital-letter-anchored detection segments by the wrong token. A
# numbered-list-aware parser splits at ordinal boundaries and lets the
# LLM extractor work on each entry's body independently.

# An entry boundary: line begins with one or more digits, a period, then
# whitespace. Optional leading whitespace tolerates OCR'd indentation.
_NUMBERED_ENTRY_START = re.compile(r"^\s*(\d{1,5})\.\s+(?P<rest>\S.*)$")

# The first place-name in an entry body — used to populate `toponym` so
# the existing form-in-body validator (D3) has something concrete to
# match. Matches French saint-prefixed toponyms across every shape that
# shows up in the corpus:
#   Saint-Caradu, Sainte-Catherine     — full canonical
#   St-Pierre, St. Marie-aux-Mines     — abbreviated masculine
#   Ste-Marie, Ste. Foy                — abbreviated feminine
#   Sts-Pierre-et-Paul, Sts. Innocents — plural (rare but real)
# Plus the trailing French diacritic / hyphen / apostrophe characters
# that show up after the prefix. Diacritic class includes capital + lower
# variants of every accented letter that appears in French place names.
_FIRST_TOPONYM = re.compile(
    r"\b(?P<name>(?:Saint|Sainte|Sts|Ste|St)\.?[-\s]"
    r"[A-ZÀÂÄÆÇÈÉÊËÎÏÔŒÙÛÜŸ]"
    r"[A-Za-zÀÂÄÆÇÈÉÊËÎÏÔŒÙÛÜŸàâäæçèéêëîïôœùûüÿ'-]+)"
)


def parse_numbered_list_text(
    text: str,
    *,
    min_entry_number: int = 1,
    max_entry_number: int | None = None,
    max_entries: int | None = None,
) -> list[ParsedEntry]:
    """Segment ``text`` by numbered-list ordinals into ParsedEntry rows.

    Each entry's `toponym` is the first place-name detected in its body
    (Saint-X / Sainte-X for the Longnon-vol-2 saint-names case); the
    full entry body becomes `body_text`. Multi-line entries are joined
    with continuation whitespace until the next ordinal boundary.

    ``min_entry_number`` and ``max_entry_number`` define an INCLUSIVE
    range ``[min, max]``: an ordinal exactly equal to either bound IS
    treated as an entry boundary. Useful when the numbered section
    starts mid-document (Longnon vol 2's saint-names section opens at
    entry 1659 and ends at 2138; earlier text is front-matter, later
    text is index-appendix cross-references). Default ``min=1`` and
    ``max=None`` accept every ordinal.

    Bounds also act as a noise filter against ordinal-shaped lines in
    body prose: a continuation line that happens to start with a year
    like ``1900. ...`` would otherwise flush the in-progress entry
    prematurely. When ``max_entry_number`` is set, out-of-range
    candidates are treated as body continuation rather than entry
    starts.

    Entries whose body has no detectable toponym are still emitted (with
    `toponym=""`) so the caller can decide whether to skip them or pass
    them through; the LLM validator will reject empty-toponym entries
    anyway, so emitting them is a no-op on accept rate but useful for
    parser diagnostics.
    """
    entries: list[ParsedEntry] = []
    current_number: int | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        m = _NUMBERED_ENTRY_START.match(line)
        if m:
            candidate = int(m.group(1))
            # Bounds-aware boundary detection: an "ordinal" outside the
            # caller's [min, max] window is more likely a year citation
            # or page reference appearing at line start (real OCR
            # produces lines like "1900. Some text" inside body prose)
            # than a real entry header. Treat out-of-range matches as
            # continuation text, NOT entry boundaries — otherwise the
            # in-progress entry would be flushed prematurely and the
            # next real ordinal would still be a fresh start, but the
            # body in between gets shredded across two ParsedEntry rows.
            #
            # When neither bound is set (default min=1, max=None) every
            # candidate is in-bounds and we get the historic behavior.
            if not _ordinal_in_bounds(candidate, min_entry_number, max_entry_number):
                if current_number is not None:
                    current_lines.append(line)
                continue
            _emit_numbered_entry(
                entries, current_number, current_lines, min_entry_number, max_entry_number
            )
            if max_entries is not None and len(entries) >= max_entries:
                # Already at the cap — stop accumulating new entries.
                # The cap is checked AFTER the flush so the in-progress
                # entry is committed before we exit.
                current_number = None
                current_lines = []
                break
            current_number = candidate
            current_lines = [m.group("rest")]
        elif current_number is not None:
            current_lines.append(line)
    _emit_numbered_entry(entries, current_number, current_lines, min_entry_number, max_entry_number)

    if max_entries is not None:
        entries = entries[:max_entries]
    return entries


def _ordinal_in_bounds(candidate: int, min_n: int, max_n: int | None) -> bool:
    """Whether a numbered-list ordinal falls in the inclusive [min, max] window."""
    if candidate < min_n:
        return False
    return not (max_n is not None and candidate > max_n)


def _emit_numbered_entry(
    entries: list[ParsedEntry],
    number: int | None,
    lines: list[str],
    min_n: int,
    max_n: int | None,
) -> None:
    """Append a ParsedEntry for the in-progress numbered entry to ``entries``,
    unless it's out of bounds or has an empty body. Toponym is the first
    detected place-name (or "")."""
    if number is None or not _ordinal_in_bounds(number, min_n, max_n):
        return
    body = " ".join(line.strip() for line in lines if line.strip())
    if not body:
        return
    match = _FIRST_TOPONYM.search(body)
    toponym = match.group("name").strip() if match else ""
    entries.append(
        ParsedEntry(
            toponym=toponym,
            section_suffix=None,
            historical_form=None,
            elements=[],
            confidence="low",
            source_quote=_shorten(body),
            body_text=body,
        )
    )
