"""Parser for Etymonline per-entry prose (wyrd-zqkp).

Etymonline pages have a clean prose-lg section per word sense
containing a headword line ("harpy(n.)") followed by a first
paragraph carrying the etymological chain:

    harpy(n.)

    winged monster of ancient mythology, late 14c., from Old French
    harpie (14c.), from Latin harpyia, from Greek Harpyia (plural),
    literally "snatchers," which is probably related to harpazein
    "to snatch" (see rapid (adj.)).

This module turns that prose into a structured list of chain links —
(language, word, hedge, gloss, attested_year) tuples — so the
downstream ingester can write etymon rows + descent edges. The
parser is intentionally pure-text: scraping (with `rodney` or any
other tool) saves text to disk, the parser reads it. That keeps
tests offline and the ingest path testable without a network.

Heuristics, not exhaustive grammar:
- Each chain link is introduced by a hedge phrase ("from", "directly
  from", "probably from", "apparently from", "ultimately from",
  "perhaps from", "borrowed from") followed by a known language name
  ("Old French", "Latin", "Greek", "PIE", etc.).
- The word is the first lexical token after the language name (an
  italic-rendered form in the HTML; in the text, just the next word).
- An optional gloss appears in the immediately following `"..."`.
- An optional attested-year parenthetical appears in `(14c.)` or
  `(c. 1400)` form.
- Cognate lists in `(source also of ...)` are ignored — they're
  parallel branches of the same root, not chain links.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Map Etymonline language phrases to wyrd's etymon.language values.
# Verified against the etymon table (PR #78 audit) so the ingest
# anchors to existing rows when present.
LANGUAGE_NAME_MAP: dict[str, str] = {
    "old english": "old-english",
    "middle english": "middle-english",
    "modern english": "modern-english",
    "old norse": "old-norse",
    "icelandic": "icelandic",
    "old french": "old-french",
    "norman french": "old-french",  # Etymonline uses both for the same period
    "anglo-french": "old-french",
    "modern french": "french",
    "french": "french",
    "old high german": "old-high-german",
    "middle high german": "middle-high-german",
    "german": "german",
    "old saxon": "osx",
    "old frisian": "ofs",
    "gothic": "got",
    "dutch": "dutch",
    "old dutch": "old-dutch",
    "middle dutch": "middle-dutch",
    "low german": "nds",
    "swedish": "swedish",
    "danish": "danish",
    "norwegian": "norwegian",
    "latin": "latin",
    "vulgar latin": "vulgar-latin",
    "medieval latin": "medieval-latin",
    "late latin": "late-latin",
    "greek": "ancient-greek",
    "ancient greek": "ancient-greek",
    "modern greek": "modern-greek",
    "byzantine greek": "byzantine-greek",
    "sanskrit": "sanskrit",
    "old irish": "old-irish",
    "middle irish": "middle-irish",
    "irish": "irish",
    "scottish gaelic": "scottish-gaelic",
    "welsh": "welsh",
    "old welsh": "old-welsh",
    "middle welsh": "middle-welsh",
    "breton": "breton",
    "cornish": "cornish",
    "manx": "manx",
    "proto-germanic": "proto-germanic",
    "proto-celtic": "proto-celtic",
    "proto-indo-european": "proto-indo-european",
    "pie": "proto-indo-european",
    "albanian": "albanian",
    "hebrew": "hebrew",
    "arabic": "arabic",
    "spanish": "spanish",
    "italian": "italian",
    "portuguese": "portuguese",
    "scots": "sco",
    "norn": "nrn",
}

# Order matters: longer phrases must be tried before shorter prefixes
# (e.g. "old high german" before "german") so partial matches don't
# clip the language code.
_LANGUAGE_NAMES_BY_LENGTH = sorted(LANGUAGE_NAME_MAP, key=len, reverse=True)

# Hedge phrases that introduce a chain link. Each maps to (edge_type,
# confidence). Edge types match etymon_descent.edge_type CHECK.
_HEDGE_PATTERNS: tuple[tuple[str, str, str], ...] = (
    # Highest-confidence chain links: "from", "directly from".
    # We default to 'inheritance' since a chain "X from Y" expresses
    # lineage; cross-family hops Etymonline phrases as "borrowed from"
    # to call out the borrowing explicitly.
    ("borrowed from", "borrowing", "high"),
    ("directly from", "inheritance", "high"),
    ("ultimately from", "inheritance", "high"),
    ("probably from", "inheritance", "medium"),
    ("apparently from", "inheritance", "medium"),
    ("perhaps from", "inheritance", "low"),
    ("from", "inheritance", "high"),
)

# Regex that picks up an attested-year parenthetical.
# Matches "(14c.)", "(c. 1400)", "(early 13c.)", "(1610s)".
# Returns the raw matched group; downstream callers can normalize.
_ATTESTED_YEAR_RE = re.compile(
    r"\(\s*(?P<year>"
    r"(?:c\.\s*\d{3,4}|"  # c. 1400
    r"(?:early|mid|late)\s+\d{1,2}c\.|"  # early 14c.
    r"\d{1,2}c\.|"  # 14c.
    r"\d{4}s?)"  # 1610s, 1212
    r")\s*\)"
)

# Cognate-list parens we want to skip past wholesale — they're not
# chain links, just parallel branches of the same PIE root.
#
# The body is ``(?:[^()]|\([^()]*\))*`` — non-paren runs PLUS one level of
# NESTED parens — not a flat ``[^)]*``. Real cognate lists carry inner parens
# (an attestation year / gloss, "Greek drakon (genitive drakontos)"); a flat
# matcher stops at that first inner ``)`` and leaks the rest of the list into the
# prose, where a cognate's own hedge ("derived from Latin draco") is then minted
# as a SPURIOUS descent link — a fabricated ancestor from a parallel cognate.
# A truly unbalanced ``(source also of …`` (no close) still doesn't match and is
# left in place, exactly as before.
_COGNATE_LIST_RE = re.compile(r"\(\s*source also of(?:[^()]|\([^()]*\))*\)", re.IGNORECASE)

# Compound parens (Modern French trôler) — also skip past for chain
# extraction; they're alt-form annotations on the previous link. Same
# nested-paren-aware body as _COGNATE_LIST_RE (see its note).
_INTERPOLATION_RE = re.compile(
    r"\(\s*(?:compare|cf\.|see|Modern\s)(?:[^()]|\([^()]*\))*\)", re.IGNORECASE
)


@dataclass(frozen=True)
class ChainLink:
    """One step in an etymological chain extracted from Etymonline prose."""

    language: str  # wyrd-canonical code from LANGUAGE_NAME_MAP
    word: str  # the historical form (lowercase, special chars preserved)
    edge_type: str  # 'inheritance' | 'borrowing' | etc.
    confidence: str  # 'high' | 'medium' | 'low'
    gloss: str | None = None
    attested_year: str | None = None  # raw form like "14c." or "1610s"


@dataclass(frozen=True)
class Sense:
    """One word sense from an Etymonline page (corresponds to one
    section.prose-lg block in the rendered HTML)."""

    headword: str  # 'harpy', 'troll'
    pos: str | None  # 'n.', 'v.', 'n.1', 'adj.', etc.
    summary: str  # the first paragraph of prose
    chain: list[ChainLink]


def split_senses(text: str) -> list[tuple[str, str]]:
    """Split a multi-sense Etymonline prose dump into (headword_line,
    summary) pairs.

    Each sense block looks like:
        headword(pos)
        <blank line>
        first paragraph (the etymological chain)
        <blank line>
        ... possibly more paragraphs ...

    We split on the headword pattern: a line ending in '(...)' that's
    followed by content. The first paragraph after the headword is
    the summary; the rest is preserved-but-ignored mythology /
    historical context.
    """
    senses: list[tuple[str, str]] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    if not blocks:
        return senses

    i = 0
    while i < len(blocks):
        head = blocks[i].strip()
        # Headword line shape: a single line ending in '(<pos>)'.
        if "\n" in head or not re.search(r"\([\w.\d]+\)\s*$", head):
            i += 1
            continue
        # The next non-empty block is the etymological summary.
        if i + 1 >= len(blocks):
            break
        summary = blocks[i + 1].strip()
        senses.append((head, summary))
        i += 2  # skip the headword + summary; loop re-scans for next headword
    return senses


def parse_headword(head_line: str) -> tuple[str, str | None]:
    """Split 'troll(n.1)' into ('troll', 'n.1'); 'harpy(n.)' → ('harpy', 'n.')."""
    m = re.match(r"^(.+?)\(([^)]+)\)\s*$", head_line.strip())
    if not m:
        return head_line.strip(), None
    return m.group(1).strip(), m.group(2).strip()


def _strip_interpolations(text: str) -> str:
    """Remove `(compare ...)`, `(see ...)`, and cognate-list parens
    that aren't chain links. Leaves attested-year parens intact."""
    text = _COGNATE_LIST_RE.sub(" ", text)
    text = _INTERPOLATION_RE.sub(" ", text)
    return text


def parse_chain(summary: str) -> list[ChainLink]:
    """Walk the prose summary left-to-right, emitting one ChainLink
    per matched hedge → language → word triple. Stops at unknown
    language names, sentence-final punctuation, or end of text.

    Conservative-by-default: when the summary's structure doesn't
    match the expected hedge → language → word pattern, we emit no
    chain link rather than guess. The caller can flag the entry for
    manual review.
    """
    cleaned = _strip_interpolations(summary)
    links: list[ChainLink] = []
    cursor = 0
    while cursor < len(cleaned):
        match = _find_next_hedge(cleaned, cursor)
        if match is None:
            break
        link, advance = match
        if link is not None:
            links.append(link)
        cursor = advance
    return links


def _pick_best_hedge(text: str, start: int) -> tuple[int, str, str, str] | None:
    """Find every hedge phrase appearing at or after `start` and pick
    the EARLIEST. Ties break to the LONGER hedge so 'probably from'
    wins over the bare 'from' that overlaps with it.

    Returns (idx, hedge_phrase, edge_type, confidence) or None if no
    hedge appears past `start`."""
    best: tuple[int, int, str, str, str] | None = None  # (pos, -length, hedge, edge, conf)
    for hedge, edge_type, confidence in _HEDGE_PATTERNS:
        idx = _find_hedge_at_word_boundary(text, hedge, start)
        if idx == -1:
            continue
        candidate = (idx, -len(hedge), hedge, edge_type, confidence)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    idx, _neg_len, hedge, edge_type, confidence = best
    return (idx, hedge, edge_type, confidence)


def _find_next_hedge(text: str, start: int) -> tuple[ChainLink | None, int] | None:
    """Find the next hedge phrase at or after `start`, parse the
    language + word that follows, and return (ChainLink, advance_to).

    Returns None when no more hedges fit. Returns (None, advance) when
    a hedge was found but the language/word that followed was
    unparseable — caller advances past it without emitting.
    """
    pick = _pick_best_hedge(text, start)
    if pick is None:
        return None
    idx, hedge, edge_type, confidence = pick
    after_hedge = idx + len(hedge)
    rest = text[after_hedge:].lstrip()
    consumed_ws = len(text[after_hedge:]) - len(rest)
    parsed = _parse_language_and_word(rest)
    if parsed is None:
        return (None, after_hedge + consumed_ws + 1)
    language, word, gloss, year, parsed_len = parsed
    link = ChainLink(
        language=language,
        word=word,
        edge_type=edge_type,
        confidence=confidence,
        gloss=gloss,
        attested_year=year,
    )
    return (link, after_hedge + consumed_ws + parsed_len)


def _find_hedge_at_word_boundary(text: str, hedge: str, start: int) -> int:
    """Find `hedge` in `text[start:]` at a word boundary. Returns the
    index in the original text or -1."""
    pattern = r"\b" + re.escape(hedge) + r"\b"
    m = re.search(pattern, text[start:], re.IGNORECASE)
    if not m:
        return -1
    return start + m.start()


def _match_language_prefix(rest: str) -> tuple[str, int] | None:
    """If `rest` starts with a known language phrase at a word
    boundary, return (wyrd_language_code, n_chars_consumed). Else
    None. Uses the by-length-descending list so 'old high german'
    wins over 'german'."""
    rest_lower = rest.lower()
    for phrase in _LANGUAGE_NAMES_BY_LENGTH:
        if not rest_lower.startswith(phrase):
            continue
        # Word boundary: the char after `phrase` must be space/punct.
        after = rest[len(phrase) : len(phrase) + 1]
        if after == "" or not after.isalpha():
            return (LANGUAGE_NAME_MAP[phrase], len(phrase))
    return None


def _read_word_after_language(rest: str) -> tuple[str, int] | None:
    """After a matched language prefix, read the next whitespace-
    separated word (preserving asterisk-prefix and diacritics) and
    return (word, n_chars_consumed). None if no word follows."""
    word_match = re.match(r"\s+(\*?[\w\-’'.ǣæðþāēīōūǣǣǣáéíóúǫöäüß]+)", rest)
    if not word_match:
        return None
    word = word_match.group(1).strip().rstrip(",.;:")
    return (word, word_match.end())


def _read_optional_gloss(rest: str) -> tuple[str, int] | None:
    """If `rest` immediately starts with a quoted gloss, return
    (gloss_text, n_chars_consumed). Trailing punctuation stripped."""
    m = re.match(r'\s+"([^"]+)"', rest)
    if not m:
        return None
    return (m.group(1).strip().rstrip(",.;: "), m.end())


def _read_optional_attested_year(rest: str) -> str | None:
    """If `rest` (after stripping leading whitespace) starts with an
    attested-year paren like '(14c.)' or '(c. 1400)', return the year
    string. Does NOT advance the cursor — the year stays anchored to
    its link without affecting where the next chain candidate begins."""
    m = _ATTESTED_YEAR_RE.match(rest.lstrip())
    if not m:
        return None
    return m.group("year").strip()


def _parse_language_and_word(
    rest: str,
) -> tuple[str, str, str | None, str | None, int] | None:
    """Read a `<Language> <word> [gloss] [year]` sequence at the start
    of `rest`. Returns (language_code, word, gloss, attested_year,
    n_chars_consumed) or None if no language matches or no word
    follows the language."""
    lang_match = _match_language_prefix(rest)
    if lang_match is None:
        return None
    language_code, after_lang = lang_match
    word_result = _read_word_after_language(rest[after_lang:])
    if word_result is None:
        return None
    word, word_len = word_result
    cursor = after_lang + word_len
    gloss = None
    gloss_result = _read_optional_gloss(rest[cursor:])
    if gloss_result is not None:
        gloss, gloss_len = gloss_result
        cursor += gloss_len
    year = _read_optional_attested_year(rest[cursor:])
    return (language_code, word, gloss, year, cursor)


def parse_sense(headword_line: str, summary: str) -> Sense:
    """Compose a full Sense from headword line + summary prose."""
    headword, pos = parse_headword(headword_line)
    chain = parse_chain(summary)
    return Sense(headword=headword, pos=pos, summary=summary, chain=chain)


def parse_text(text: str) -> list[Sense]:
    """Parse the entire prose dump (possibly multi-sense) into a list
    of Senses. Empty list if no headword pattern is found."""
    return [parse_sense(head, summary) for head, summary in split_senses(text)]
