"""wyrd-y10: render English names in alternate phonemic scripts.

A phonemic script written for English IS English, just visually
disguised. A player who learned Shavian can READ a 'foreign
inscription' as plain English; a player who didn't sees mysterious
glyphs. That's the perfect intersection of 'exotic enough to be
atmospheric' and 'literally readable if you commit'.

Initial target: **Shavian**. Phonemic alphabet, ~48 letters
(U+10450-U+1047F). Maps to English phonemes 1:1.

Output approach: grapheme-aware substitution that handles the
common English digraphs (`ch`, `sh`, `th`, `ng`, `ph`) before
falling through to single-letter approximations. Lossy compared
to a full Read Lex lookup (which would handle silent letters and
vowel disambiguation precisely) but produces a Shavian-flavored
output that's atmospheric and decodable for committed readers.

Future scripts (each is just a glyph table once the phoneme
plumbing is in):
* **Tengwar** — Tolkien's elven; phoneme-based, multiple 'modes'
  per language.
* **Cirth** — Tolkien's runes; letter-mapped, simpler.
* **Elder Futhark** — real historical Germanic runes; fits
  norse / dwarvish coding.
* **Ogham** — real Irish stick-script; fits celtic / elven coding.

Per the wyrd-y10 ticket, the Read Lex (~30K-word English-to-
Shavian dictionary, openly available on GitHub) would replace
the grapheme-substitution heuristic with phoneme-precise glyphs.
That's a future refinement; v1 ships a serviceable approximation
without a third-party dictionary dependency.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shavian phoneme glyphs
# ---------------------------------------------------------------------------
#
# Phoneme → Shavian codepoint. Shavian's 48 glyphs span U+10450
# through U+10477; the assignment here is best-effort against the
# Wikipedia chart. v1's heuristic is intentionally lossy (no
# phoneme-precise vowel disambiguation), so fine-grained accuracy
# isn't critical for atmosphere — the goal is "looks Shavian-ish
# AND a committed reader can decode it." A future refinement using
# Read Lex would replace this whole table with phoneme-perfect
# lookups.
_SHAVIAN: dict[str, str] = {
    # Stops
    "p": "\U00010450",
    "t": "\U00010451",
    "k": "\U00010452",
    "b": "\U0001045a",
    "d": "\U0001045b",
    "g": "\U0001045c",
    # Fricatives
    "f": "\U00010453",
    "v": "\U0001045d",
    "θ": "\U00010454",
    "ð": "\U0001045e",
    "s": "\U00010455",
    "z": "\U0001045f",
    "ʃ": "\U00010456",
    "ʒ": "\U00010460",
    "tʃ": "\U00010457",
    "dʒ": "\U00010461",
    # Approximants / nasals / liquids
    "w": "\U00010462",
    "j": "\U00010458",
    "h": "\U00010463",
    "l": "\U00010464",
    "r": "\U0001046e",
    "m": "\U00010465",
    "n": "\U00010466",
    "ŋ": "\U00010459",
    # Vowels (short)
    "æ": "\U00010468",
    "ɛ": "\U0001046a",
    "ɪ": "\U0001046c",
    "ɒ": "\U00010470",
    "ʌ": "\U00010472",
    "ʊ": "\U00010474",
    "ə": "\U00010469",
    # Vowels (long / diphthong)
    "iː": "\U0001046d",
    "uː": "\U00010475",
    "ɑː": "\U00010471",
    "ɔː": "\U00010473",
    "eɪ": "\U0001046b",
    "aɪ": "\U0001046f",
    "ɔɪ": "\U00010477",
    "aʊ": "\U00010476",
    "oʊ": "\U00010467",
    "ɝ": "\U0001045f",  # placeholder; rare in toponyms
}

# The above table is best-effort against the Shavian chart. The
# v1 grapheme-based transliteration below is intentionally lossy
# (no phoneme-precise vowel disambiguation), so fine-grained
# accuracy isn't critical for atmosphere. A later refinement using
# Read Lex would replace the heuristic mapping with phoneme-perfect
# lookups.


# ---------------------------------------------------------------------------
# English-grapheme → Shavian heuristic
# ---------------------------------------------------------------------------
#
# Order matters: digraphs must be substituted before the single-
# letter passes consume them. Each rule is `(pattern, glyph_key)`
# where `glyph_key` indexes ``_SHAVIAN`` above.

_DIGRAPH_RULES: tuple[tuple[str, str], ...] = (
    # Voiceless
    ("ch", "tʃ"),
    ("sh", "ʃ"),
    ("th", "θ"),
    ("ph", "f"),
    ("ng", "ŋ"),
    # Common vowel digraphs (lossy: no contextual disambiguation)
    ("oo", "uː"),
    ("ee", "iː"),
    ("ea", "iː"),
    ("ou", "aʊ"),
    ("ow", "aʊ"),
    ("ai", "eɪ"),
    ("ay", "eɪ"),
    ("oi", "ɔɪ"),
    ("oy", "ɔɪ"),
    ("oa", "oʊ"),
    ("ie", "aɪ"),
)

_SINGLE_RULES: dict[str, str] = {
    "p": "p",
    "b": "b",
    "t": "t",
    "d": "d",
    "k": "k",
    "g": "g",
    "c": "k",
    "q": "k",
    "f": "f",
    "v": "v",
    "s": "s",
    "z": "z",
    "m": "m",
    "n": "n",
    "l": "l",
    "r": "r",
    "h": "h",
    "w": "w",
    "y": "j",
    "j": "dʒ",
    "x": "k",  # crude approximation; 'x' is /ks/ but folds to /k/ in v1
    "a": "æ",
    "e": "ɛ",
    "i": "ɪ",
    "o": "ɒ",
    "u": "ʌ",
}


def _to_shavian(text: str) -> str:
    """Heuristic English-grapheme → Shavian-glyph mapping.

    Substitution proceeds digraph-first then per-character; each
    matched span is replaced with the corresponding glyph from
    ``_SHAVIAN``. Non-letter characters (hyphens, spaces, digits)
    pass through verbatim so a compound name's structure stays
    legible.

    Lossy: no silent-letter handling, no contextual vowel
    disambiguation. The Read Lex replacement for this heuristic
    is filed as a future refinement (wyrd-y10 ticket text).
    """
    out: list[str] = []
    i = 0
    s = text.lower()
    while i < len(s):
        c = s[i]
        # Pass non-letter chars verbatim, preserving the original
        # case of the input character to keep hyphens / numbers /
        # spaces intact.
        if not c.isalpha():
            out.append(text[i])
            i += 1
            continue
        # Digraph attempt: peek next char and look up the pair.
        if i + 1 < len(s):
            digraph = s[i : i + 2]
            for pat, key in _DIGRAPH_RULES:
                if digraph == pat:
                    glyph = _SHAVIAN.get(key)
                    if glyph is not None:
                        out.append(glyph)
                        i += 2
                        break
            else:
                # No digraph matched; fall through to single-char.
                key = _SINGLE_RULES.get(c)
                glyph = _SHAVIAN.get(key) if key else None
                if glyph is not None:
                    out.append(glyph)
                else:
                    out.append(text[i])
                i += 1
        else:
            key = _SINGLE_RULES.get(c)
            glyph = _SHAVIAN.get(key) if key else None
            if glyph is not None:
                out.append(glyph)
            else:
                out.append(text[i])
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

# Supported scripts. Adding a new script means adding an entry here
# + a transliteration function. The function takes a string and
# returns the script-rendered string; non-letter characters
# (hyphens, spaces, digits) should pass through.
SUPPORTED_SCRIPTS: tuple[str, ...] = ("shavian",)


def transliterate(text: str, script: str) -> str:
    """Render ``text`` in ``script``.

    Currently supports ``'shavian'``. Future entries (Tengwar /
    Cirth / Elder Futhark / Ogham) drop in as new dispatch arms
    once their glyph tables land.

    Raises ``ValueError`` for unknown scripts so the SPA / CLI can
    surface an unambiguous error rather than silently passing the
    input through.
    """
    if not text:
        return ""
    if script == "shavian":
        return _to_shavian(text)
    raise ValueError(f"unsupported script {script!r}; supported: {sorted(SUPPORTED_SCRIPTS)}")
