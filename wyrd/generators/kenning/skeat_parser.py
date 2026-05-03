"""Parser for Skeat-style place-name etymology dictionaries.

Skeat's volumes (Cambridgeshire, Bedfordshire, Berkshire, Suffolk) all share
a recognizable structure:

* The book is divided into sections, each headed "§ N. The suffix -X" or
  similar. Every entry inside a section ends in that suffix.
* Each entry starts with the modern place name as a heading word.
* The prose etymology mentions the historical form, often hyphenated at the
  morpheme boundary, attributed to A.S. (Old English), O.N. (Old Norse),
  Welsh, etc.

This parser is deliberately conservative. It uses structural cues (section
headers, entry headings, the TOC) where they exist; it falls back to regex
extraction over the prose for the historical form and component glosses;
and it emits a confidence tier per entry rather than guessing.

Entries we can't confidently parse are returned with `confidence='low'` and
no elements — caller can persist them as toponym + post-element-only rows
or skip them entirely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- text cleanup ----------------------------------------------------------

# OCR adds running headers like "§ 2. NAMES ENDING IN -TON. 7", page numbers
# alone on a line, and the title repeated as a footer. Strip these so they
# don't appear inside entry bodies.
_LINE_NOISE_PATTERNS = [
    re.compile(r"^\s*§\s*\d+\.\s+NAMES\s+ENDING\s+IN.*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s+THE\s+PLACE-NAMES\s+OF.*$", re.IGNORECASE),
    re.compile(r"^\s*THE\s+PLACE-NAMES\s+OF.*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,4}\s*$"),  # bare page number
]


def clean_ocr_text(text: str) -> str:
    """Remove running headers, page numbers, and footnote markers from OCR
    text. Returns the cleaned text with original line structure preserved
    where possible."""
    out_lines = []
    for line in text.split("\n"):
        if any(p.match(line) for p in _LINE_NOISE_PATTERNS):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


# --- section detection -----------------------------------------------------

# The TOC lists sections like:
#   §  2. The suffix -ton : — Barton, Carlton, ...
# and the body has matching headers like:
#   §  2.    The Suffix -ton.
# We detect section starts in the BODY and use the suffix as the post-element
# hint for every entry inside.

# Body section headers end with "." and have no ":" — distinct from TOC
# entries, which look like "1.  The suffix -ton  : — Barton, Carlton, ...".
# That difference saves us from a deduplication pass.
#
# Skeat varied the format across volumes:
#   Cambridgeshire 1901:  "§  2.    The Suffix -ton."
#   Bedfordshire   1906:  "46.    Wick."
#   Suffolk        1913:  "1.  The suffix -acre."
# The shared invariants: number + period, optionally a "The suffix" preamble,
# then the suffix word(s), then a final period and end-of-line.
_SECTION_HEADER = re.compile(
    r"""
    ^\s*
    (?:§\s*)?              # optional § prefix (Cambridgeshire only)
    \d+\.?\s+              # section number
    (?:The\s+[Ss][a-z]{1,4}ix(?:es)?\s+)?   # optional "The suffix(es)"
    (?P<hint>[^:\n.]+?)    # suffix or topic
    \s*\.\s*$
    """,
    re.VERBOSE,
)


@dataclass
class Section:
    suffix_hint: str  # the raw suffix declaration as printed, e.g. "-ton"
    suffixes: list[str]  # cleaned/normalized: ["-ton"] or ["-wick", "-cote"]
    body: str  # body text, between this header and the next one


def split_sections(text: str) -> list[Section]:
    """Walk the cleaned text and return one Section per body chapter header.

    The regex only matches body-style headers (period-terminated, no trailing
    list), so TOC entries are excluded automatically — no dedup needed.
    """
    lines = text.split("\n")
    headers: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _SECTION_HEADER.match(line)
        if not m:
            continue
        headers.append((i, m.group("hint").strip()))

    sections: list[Section] = []
    for idx, (line_no, hint) in enumerate(headers):
        end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
        body = "\n".join(lines[line_no + 1 : end])
        sections.append(
            Section(
                suffix_hint=hint,
                suffixes=_normalize_suffixes(hint),
                body=body,
            )
        )
    return sections


def _normalize_suffixes(hint: str) -> list[str]:
    """Pull the dashed suffix forms out of header text like "-ton" or
    "-bridge, -hithe, -low, and -well" or "camp, Chester, dike, hale ..."."""
    out: list[str] = []
    for token in re.split(r"[,\s]+(?:and\s+)?", hint):
        token = token.strip().strip(".:")
        if not token:
            continue
        if not token.startswith("-"):
            # bare suffix words from later sections — normalize to "-token"
            # but only if alphabetic and short-ish.
            if token.isalpha() and 2 <= len(token) <= 12:
                token = f"-{token.lower()}"
            else:
                continue
        else:
            token = token.lower()
        out.append(token)
    return out


# --- TOC parsing -----------------------------------------------------------

# A TOC line declares a section and lists its entries:
#   §  2.  The suffix -ton  : — Barton, Carlton, Caxton, Cherry Hinton,
# These often span multiple lines in the OCR; we collect line continuations
# until the next "§" or page-number-only line.

_TOC_HEADER = re.compile(
    r"""
    ^\s*
    (?:§\s*)?              # optional §
    \d+\.?\s+              # section number
    (?:The\s+[Ss][a-z]{1,4}ix(?:es)?\s+)?   # optional "The suffix"
    (?P<hint>.+?)          # the suffix(es)
    \s*:\s*[—–-]?\s*       # ": —" delimiter (TOC-only)
    (?P<rest>.*)$
    """,
    re.VERBOSE,
)


def parse_toc(text: str) -> dict[str, list[str]]:
    """Parse TOC section headers into {suffix_hint: [toponym, ...]}.

    Skeat's TOC entries declare both the section's suffix and the place names
    in it. Continuation lines (text following the header line, no leading "§"
    and not a page number) extend the list until the next header.
    """
    lines = text.split("\n")
    out: dict[str, list[str]] = {}
    i = 0
    while i < len(lines):
        m = _TOC_HEADER.match(lines[i])
        if not m:
            i += 1
            continue
        hint = m.group("hint").strip().rstrip(".")
        listing = m.group("rest").strip()
        i += 1
        # Continuation: pull lines until next "§ N." or a clear stop. TOC
        # listings end with a tab/whitespace + a page number.
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt:
                break
            if nxt.startswith("§"):
                break
            listing += " " + nxt
            i += 1
        # Strip trailing page number and dot leaders.
        listing = re.sub(r"\.{2,}", " ", listing)
        listing = re.sub(r"\s+\d+\s*$", "", listing).strip()
        toponyms = _split_toc_listing(listing)
        if toponyms:
            out[hint] = toponyms
    return out


def _split_toc_listing(listing: str) -> list[str]:
    """Split a TOC listing into normalized place names.

    The listing contains commas, em-dashes, parentheses with alternates, and
    OCR artifacts. We extract anything that looks like a place name token
    (one or more capitalized words, possibly hyphenated).
    """
    # Drop parenthesized alternates like "(Saxon Street)" — they're cross-
    # references, not separate entries.
    cleaned = re.sub(r"\([^)]*\)", "", listing)
    # Em-dashes separate sub-clusters within a multi-suffix section; treat as
    # commas.
    cleaned = re.sub(r"[—–]", ",", cleaned)
    out: list[str] = []
    for chunk in cleaned.split(","):
        name = chunk.strip()
        if not name:
            continue
        # Each name is one or more capitalized words. Reject anything else
        # (e.g. stray "and", page numbers).
        if not re.match(r"^[A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)*$", name):
            continue
        # Re-join hyphenated OCR splits like "Hard wick" → keep as is; Skeat
        # does write some entries that way.
        out.append(name)
    return out


# --- entry segmentation ----------------------------------------------------


def split_entries_by_toc(section_body: str, toc_toponyms: list[str]) -> list[tuple[str, str]]:
    """Find each TOC-declared toponym at a paragraph start in the body.

    For each toponym, we look for a paragraph beginning with that exact phrase
    (case-sensitive, optionally followed by '.', 'is', 'was', etc.). We then
    take the entry body up to the next matching toponym or the end of the
    section.
    """
    paragraphs = re.split(r"\n\s*\n", section_body)
    flat_paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]

    # For each toponym, find paragraphs that start with it.
    out: list[tuple[str, str]] = []
    for toponym in toc_toponyms:
        # Match either "Toponym." / "Toponym is" / "Toponym," / "Toponym was"
        # / "Toponym (" / "Toponym occurs" — anything that looks like an entry
        # opener rather than a mid-sentence reference.
        pattern = re.compile(
            r"^" + re.escape(toponym) + r"\b\s*[.,(]|"
            r"^" + re.escape(toponym) + r"\s+(?:is|was|occurs|or|appears|spelt|seems)\b"
        )
        for flat in flat_paragraphs:
            if pattern.match(flat):
                out.append((toponym, flat))
                break  # take the first paragraph that opens the entry
    return out


# --- etymology extraction --------------------------------------------------

# Patterns for "the A.S. (form)" where the form is the historical reconstruction.
# Skeat sometimes writes "A.S. bere-tun", sometimes "A.S. ceaster-tun", sometimes
# just "A.S. drceg" (and the suffix is implied from the section).
_AS_FORM = re.compile(
    r"""
    \bA\.\s*S\.\s+
    (?P<form> [a-zA-ZæðþāēīōūȳÆÐÞĀĒĪŌŪȲ]+ (?:-[a-zA-ZæðþāēīōūȳÆÐÞĀĒĪŌŪȲ]+)* )
    """,
    re.VERBOSE,
)

_OE_FORM = re.compile(
    r"""
    \b(?:O\.?\s*E\.?|Old\s+English)\s+
    (?P<form> [a-zA-ZæðþāēīōūȳÆÐÞĀĒĪŌŪȲ]+ (?:-[a-zA-ZæðþāēīōūȳÆÐÞĀĒĪŌŪȲ]+)* )
    """,
    re.VERBOSE,
)

_ON_FORM = re.compile(
    r"""
    \b(?:O\.\s*N\.|Old\s+Norse|Icelandic|Scandinavian)\s+
    (?:form\s+)?
    (?P<form> [a-zA-ZæðþāēīōūȳÆÐÞĀĒĪŌŪȲ]+ (?:-[a-zA-ZæðþāēīōūȳÆÐÞĀĒĪŌŪȲ]+)* )
    """,
    re.VERBOSE,
)

# Personal-name pattern. Skeat repeatedly writes "X, a personal name" or
# "X, gen. case of Y, a personal name", or "represents A.S. Xan, genitive of
# X, a personal name". Combined with the section suffix this gives us a clean
# breakdown: prefix = personal name (genitive), suffix = section suffix.
_PERSONAL_NAME = re.compile(
    r"""
    (?:gen(?:itive)?\.?\s+(?:case\s+)?of\s+|represents\s+(?:the\s+)?A\.S\.\s+|A\.S\.\s+)
    (?P<form> [A-Za-zæðþÆÐÞāēīōūȳ]{3,} )
    [,\s]+
    (?:a\s+)? personal\s+name
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Inline gloss after the historical form: "lit. 'X'" or 'lit. "X"'.
_LIT_GLOSS = re.compile(r"\blit\.\s*['\"](?P<gloss>[^'\"]{2,50})['\"]")

# "from X, gloss" — single-element case (no "and Y").
_FROM_SINGLE = re.compile(
    r"""
    \bfrom\s+
    (?:A\.S\.\s+)?
    (?P<form> [a-zA-ZæðþāēīōūȳÆÐÞĀĒĪŌŪȲ]+ )
    \s*,\s*
    (?P<gloss> [^,;.]{2,50}? )
    \s*[.;]
    """,
    re.VERBOSE,
)

# "from X, gloss, and Y" — Skeat's standard component listing.
# Glosses are non-greedy and bounded so we don't gobble across sentences.
_FROM_AND = re.compile(
    r"""
    \bfrom\s+
    (?P<a> [a-zA-ZæðþāēīōūȳÆÐÞĀĒĪŌŪȲ]+ )
    \s*,\s*
    (?P<a_gloss> [^,;.]{1,40}? )
    \s*,?\s*and\s+
    (?P<b> [a-zA-ZæðþāēīōūȳÆÐÞĀĒĪŌŪȲ]+ )
    (?:\s*,\s* (?P<b_gloss> [^,;.]{1,40}? ))?
    \s*[.;]
    """,
    re.VERBOSE,
)


@dataclass
class ParsedElement:
    form: str
    language: str  # canonical language code (old-english, old-norse, ...)
    gloss: str | None = None
    position: str | None = None  # 'pre' | 'post' | 'inner' | None
    inflection: str | None = None


# Budget for the source_quote text on a ParsedEntry — kept on this side of
# the parser/extractor boundary so the regex parser (here) and the LLM
# extractor (which imports ParsedEntry from this module) share one cap.
# Lives here rather than in llm_extractor because the inverse import
# direction would be a cycle. wyrd-9v1.
SOURCE_QUOTE_BUDGET = 500


@dataclass
class ParsedEntry:
    toponym: str
    section_suffix: str | None  # the suffix hint from the section header
    historical_form: str | None  # 'bere-tun' if found
    elements: list[ParsedElement] = field(default_factory=list)
    confidence: str = "low"  # 'high' | 'medium' | 'low'
    # ≤SOURCE_QUOTE_BUDGET char verbatim, for the citation table.
    source_quote: str = ""
    body_text: str = ""  # full untruncated entry body, for re-extraction


def _split_hyphenated(form: str) -> list[str]:
    """Split a hyphenated etymological form like 'bere-tun' into its parts."""
    return [p for p in form.split("-") if p]


def _shorten(text: str, limit: int = SOURCE_QUOTE_BUDGET) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _bare_suffix(suffix_hint: str | None) -> str | None:
    """'-ton' → 'ton'; '-ing-ton' → 'ington'; None → None."""
    if not suffix_hint:
        return None
    return suffix_hint.lstrip("-").replace("-", "")


# Modern English place-name suffix → list of historical (mostly Old English)
# spellings the same morpheme appears under in the etymology corpus. The
# unhyphenated-form splitter consults this so that "dictun" with section
# "-ton" still recovers tun as the post-element. Order matters within each
# list — longer/more-specific variants first, since the splitter takes the
# longest match.
SUFFIX_VARIANTS: dict[str, list[str]] = {
    "ton": ["tun", "ton"],
    "ham": ["ham"],
    "ington": ["ington", "ingatun", "ingtun"],
    "worth": ["worth", "weorð", "wyrth"],
    "bury": ["byrig", "burh", "bury"],
    "borough": ["burh", "borough"],
    "low": ["hlaw", "low"],
    "stead": ["stede", "stead"],
    "wick": ["wic", "wick"],
    "cote": ["cote", "cot"],
    "bridge": ["brycg", "bridge"],
    "hithe": ["hyð", "hithe"],
    "well": ["wella", "well"],
    "ford": ["ford"],
    "dun": ["dun"],
    "don": ["dun", "don"],
    "den": ["denu", "den"],
    "down": ["dun", "down"],
    "ea": ["eg", "ea"],
    "fen": ["fen", "fenn"],
    "field": ["feld", "field"],
    "lea": ["leah", "lea"],
    "ley": ["leah", "ley"],
    "mere": ["mere"],
    "pool": ["pol", "pool"],
}


def _candidate_post_forms(suffix_root: str | None) -> list[str]:
    """Return the historical-form variants we'll try when splitting an
    unhyphenated form. Always includes the modern suffix itself as a
    fallback, longest-first."""
    if not suffix_root:
        return []
    variants = SUFFIX_VARIANTS.get(suffix_root.lower(), [])
    seen: list[str] = []
    for v in variants + [suffix_root]:
        if v.lower() not in {s.lower() for s in seen}:
            seen.append(v)
    seen.sort(key=len, reverse=True)
    return seen


def _split_form_by_suffix(form: str, suffix_root: str) -> tuple[str, str] | None:
    """If `form` ends in `suffix_root` (or any of its historical variants in
    SUFFIX_VARIANTS), return (prefix, matched-tail) split. Otherwise None.

    Skeat's unhyphenated forms ("Drcegtun", "dictun") let us recover the
    boundary using the section suffix as a structural hint, but the section
    suffix is given in modern spelling while the form is in Old English —
    so we try both.
    """
    if not form or not suffix_root:
        return None
    for tail in _candidate_post_forms(suffix_root):
        if len(form) <= len(tail):
            continue
        if form.lower().endswith(tail.lower()):
            prefix = form[: -len(tail)]
            if len(prefix) >= 2:
                return prefix, form[-len(tail) :]
    return None


def parse_entry(toponym: str, body: str, *, suffix_hint: str | None) -> ParsedEntry:
    """Extract whatever structure we can from one entry body.

    Confidence ladder:
      high   - historical form recovered AND split into ≥2 elements with
               positions (either via hyphenation or section-suffix matching)
      medium - personal-name prefix recovered, paired with section suffix
      low    - nothing more than the section suffix (post-element)
    """
    entry = ParsedEntry(
        toponym=toponym,
        section_suffix=suffix_hint,
        historical_form=None,
        source_quote=_shorten(body),
        body_text=body,
    )

    # Try A.S. / O.E. / O.N. forms in priority order.
    historical_form: str | None = None
    historical_lang: str | None = None
    for pattern, lang in (
        (_AS_FORM, "old-english"),
        (_OE_FORM, "old-english"),
        (_ON_FORM, "old-norse"),
    ):
        m = pattern.search(body)
        if m:
            historical_form = m.group("form")
            historical_lang = lang
            break
    entry.historical_form = historical_form

    suffix_root = _bare_suffix(suffix_hint)

    # Pattern A: hyphenated historical form.
    if historical_form and "-" in historical_form and historical_lang:
        parts = _split_hyphenated(historical_form)
        if 2 <= len(parts) <= 4:
            for i, part in enumerate(parts):
                if i == 0:
                    pos = "pre"
                elif i == len(parts) - 1:
                    pos = "post"
                else:
                    pos = "inner"
                entry.elements.append(
                    ParsedElement(form=part.lower(), language=historical_lang, position=pos)
                )
            entry.confidence = "high"

    # Pattern A': unhyphenated historical form that ends with the section's
    # suffix root. We split at that boundary.
    if not entry.elements and historical_form and historical_lang and suffix_root:
        split = _split_form_by_suffix(historical_form, suffix_root)
        if split is not None:
            prefix, post = split
            entry.elements = [
                ParsedElement(form=prefix.lower(), language=historical_lang, position="pre"),
                ParsedElement(form=post.lower(), language=historical_lang, position="post"),
            ]
            entry.historical_form = f"{prefix}-{post}"
            entry.confidence = "high"

    # Pattern B: "from X, gloss, and Y" — fills in glosses or supplies a
    # 2-element breakdown when no historical form was parsed.
    m = _FROM_AND.search(body)
    if m and historical_lang:
        a_form = m.group("a").lower()
        a_gloss = (m.group("a_gloss") or "").strip() or None
        b_form = m.group("b").lower()
        b_gloss = (m.group("b_gloss") or "").strip() or None
        if entry.elements:
            for elem in entry.elements:
                if elem.form == a_form and not elem.gloss:
                    elem.gloss = a_gloss
                elif elem.form == b_form and not elem.gloss:
                    elem.gloss = b_gloss
        else:
            entry.elements = [
                ParsedElement(form=a_form, language=historical_lang, gloss=a_gloss, position="pre"),
                ParsedElement(
                    form=b_form, language=historical_lang, gloss=b_gloss, position="post"
                ),
            ]
            entry.confidence = "medium"

    # Pattern C: personal-name prefix + section suffix. Matches only when no
    # elements have been recovered yet, since hyphenated A.S. forms are
    # higher-priority evidence.
    if not entry.elements and suffix_hint and suffix_root:
        m = _PERSONAL_NAME.search(body)
        if m:
            personal = m.group("form").lower()
            entry.elements = [
                ParsedElement(
                    form=personal,
                    language="old-english",
                    position="pre",
                    inflection="genitive",
                ),
                ParsedElement(
                    form=suffix_root,
                    language="old-english",
                    position="post",
                ),
            ]
            entry.confidence = "medium"

    # Pattern D: "lit. 'X-Y'" gloss applies to the whole compound; attach to
    # the historical form if found.
    m = _LIT_GLOSS.search(body)
    if m and entry.elements and not any(el.gloss for el in entry.elements):
        # Distribute as a single overall gloss on the first element so it
        # isn't lost; per-element glosses come from Pattern B.
        entry.elements[0].gloss = m.group("gloss").strip()

    if not entry.elements:
        entry.confidence = "low"

    return entry


# --- public API ------------------------------------------------------------


def parse_skeat_text(text: str) -> list[ParsedEntry]:
    """Parse a Skeat-format etymology text into structured entries.

    Strategy: parse the TOC to get the authoritative `(suffix → [toponyms])`
    list, then locate each toponym in its corresponding body section.

    For multi-suffix sections like "-bridge, -hithe, -low, and -well", the
    suffix hint is the section's full hint string; we infer the per-toponym
    suffix later by matching the toponym's tail against the section's
    suffix list.
    """
    cleaned = clean_ocr_text(text)
    toc = parse_toc(cleaned)
    # Index body sections by a normalized form of the hint, so the TOC's
    # "-borough" matches the body's "Borough" (Skeat varies these between
    # volumes — Cambridgeshire keeps dashes, Bedfordshire drops them).
    sections_by_hint: dict[str, Section] = {}
    for s in split_sections(cleaned):
        for variant in _hint_variants(s.suffix_hint):
            sections_by_hint.setdefault(variant, s)

    out: list[ParsedEntry] = []
    for hint, toponyms in toc.items():
        section = None
        for variant in _hint_variants(hint):
            section = sections_by_hint.get(variant)
            if section is not None:
                break
        if section is None:
            continue
        for toponym, body in split_entries_by_toc(section.body, toponyms):
            # Pick the per-toponym suffix from the section's normalized list:
            # the longest suffix in section.suffixes that the toponym ends in.
            per_suffix = _suffix_for_toponym(toponym, section.suffixes)
            entry = parse_entry(toponym, body, suffix_hint=per_suffix)
            out.append(entry)
    return out


def _hint_variants(hint: str) -> list[str]:
    """Yield normalization variants of a section hint so TOC↔body matching
    is robust to Skeat's per-volume formatting drift.

    For "-borough" we yield ["-borough", "borough"]; for "Bourne, Burn" we
    yield each token both with and without leading dashes, lowercased.
    """
    out: list[str] = []
    raw = hint.lower().strip().rstrip(".")
    out.append(raw)
    out.append(raw.lstrip("-"))
    # Multi-suffix hints like "-bourne, -burn": match on first token too.
    first = re.split(r"[,\s]+(?:and\s+)?", raw)[0].strip()
    if first:
        out.append(first)
        out.append(first.lstrip("-"))
    seen: list[str] = []
    for v in out:
        if v and v not in seen:
            seen.append(v)
    return seen


def _suffix_for_toponym(toponym: str, suffixes: list[str]) -> str | None:
    """Pick the longest section-declared suffix that matches the toponym's
    tail, ignoring case and the leading dash.
    """
    name_lower = toponym.lower().replace(" ", "")
    candidates = sorted(suffixes, key=len, reverse=True)
    for s in candidates:
        bare = s.lstrip("-")
        if name_lower.endswith(bare):
            return s
    # Fall back to the only suffix if there's exactly one.
    return suffixes[0] if len(suffixes) == 1 else None
