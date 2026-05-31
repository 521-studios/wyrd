"""Per-language phonology tables for the kenning generator.

Foundation data for several downstream features:
  - Pronunciation guide (English-respelling output for any generated name)
  - Shavian / Tengwar / Elder Futhark transliteration
  - Register conversion (mapping a name's phonemes between languages)

Each language has:
  - GRAPHEMES: ordered list of (grapheme, ipa, respelling) tuples. Order
    matters — multi-character digraphs (e.g. OE 'cg', Welsh 'll') must be
    tried before single chars to avoid greedy single-char matches.
  - STRESS: a function or rule string saying where stress falls in a word.

The respelling column is a SAMPA-lite English approximation aimed at GMs:
capital letters mark the stressed syllable, hyphens separate syllables,
vowel digraphs use English conventions (OO, EE, AY, OW, AH).

This is deliberately small and focused on PLACE-NAME morphology, not
full-language phonology. We're not building a translator; we're giving a
GM a pronunciation hint they can read at the table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phoneme:
    """One grapheme-to-sound mapping.

    ``coda_ipa`` / ``coda_respelling`` (wyrd-vm8t): when set, the grapheme takes
    these values in CODA position — i.e. when it is NOT followed by a vowel
    (before a consonant, or word-final). Used for OE ``h``, which is [h] in the
    onset (``hām`` → /hɑːm/) but the velar fricative [x] in the coda (``burh`` →
    /burx/, ``niht`` → /nixt/). ``None`` means the grapheme is position-neutral.
    """

    grapheme: str  # The letter or digraph in the orthography.
    ipa: str  # IPA representation (onset / position-neutral).
    respelling: str  # English-friendly approximation.
    coda_ipa: str | None = None
    coda_respelling: str | None = None


# Vowel graphemes across the supported orthographies — drives the onset-vs-coda
# decision for position-sensitive phonemes (a grapheme is in coda position when
# the NEXT character is not one of these, or it is word-final).
_VOWELS: frozenset[str] = frozenset("aeiouyæøǫ" + "āēīōūȳǣáéíóúýǽǿœ")


# --- Old English ----------------------------------------------------------
#
# Sources: Campbell, *Old English Grammar* (1959); Hogg & Fulk, *A Grammar
# of Old English* (1992-2011, but conventions are stable). Place-name
# morphology in particular has predictable phonology.
#
# Notable features for us:
#   - æ = "a as in cat" (we render as A)
#   - ð / þ = both "th" (sometimes voiced /ð/ as in 'this', sometimes
#     voiceless /θ/ as in 'thin'; conventionally both rendered "th")
#   - cg = /dʒ/ "j as in judge" (Brycg → "brij")
#   - sc = /ʃ/ "sh" (scēap "sheep" → "shape")
#   - c before front vowels = /tʃ/ "ch" (cild "child" → "child")
#   - g before front vowels = /j/ "y" (geong → "young")
#   - long vowels (ā, ē, ī, ō, ū) marked with macrons; double the English
#     respelling vowel (ā → "AH", ē → "AY", ī → "EE", ō → "OH", ū → "OO")

OLD_ENGLISH = [
    # Long vowels (try first, before bare-vowel fallback)
    Phoneme("ǣ", "æː", "AY"),
    Phoneme("ā", "ɑː", "AH"),
    Phoneme("ē", "eː", "AY"),
    Phoneme("ī", "iː", "EE"),
    Phoneme("ō", "oː", "OH"),
    Phoneme("ū", "uː", "OO"),
    Phoneme("ȳ", "yː", "EE"),
    # Digraph consonants
    Phoneme("cg", "dʒ", "j"),
    Phoneme("sc", "ʃ", "sh"),
    Phoneme("hl", "l̥", "hl"),
    Phoneme("hr", "r̥", "hr"),
    Phoneme("hn", "n̥", "hn"),
    Phoneme("hw", "ʍ", "hw"),
    # Single-char consonants with palatalization context (handled by rule
    # in apply_phonology, not via fixed mapping — c → ch before front
    # vowel, c → k otherwise)
    Phoneme("þ", "θ", "th"),
    Phoneme("ð", "ð", "th"),
    Phoneme("æ", "æ", "a"),
    # Vowels (short)
    Phoneme("a", "ɑ", "ah"),
    Phoneme("e", "ɛ", "eh"),
    Phoneme("i", "ɪ", "ih"),
    Phoneme("o", "ɔ", "oh"),
    Phoneme("u", "ʊ", "uh"),
    Phoneme("y", "y", "ih"),
    # Single consonants
    Phoneme("c", "k", "k"),  # default; palatalization handled separately
    Phoneme("g", "g", "g"),  # ditto
    # OE h: [h] in onset (hām), velar fricative [x] in coda (burh, niht).
    # The hl/hr/hn/hw clusters above are matched first, so this single-h rule
    # only fires for h before a vowel ([h]) or before a non-cluster consonant /
    # word-end ([x]).
    Phoneme("h", "h", "h", coda_ipa="x", coda_respelling="kh"),
    Phoneme("b", "b", "b"),
    Phoneme("d", "d", "d"),
    Phoneme("f", "f", "f"),
    Phoneme("l", "l", "l"),
    Phoneme("m", "m", "m"),
    Phoneme("n", "n", "n"),
    Phoneme("p", "p", "p"),
    Phoneme("r", "r", "r"),
    Phoneme("s", "s", "s"),
    Phoneme("t", "t", "t"),
    Phoneme("w", "w", "w"),
]


# --- Old Norse ------------------------------------------------------------
#
# Sources: Faarlund, *The Syntax of Old Norse* (2004); place-name conventions
# from the EPNS volumes and Ekwall's *Place-Names of Lancashire*. Standard
# Old Icelandic orthography assumed.
#
# Notable for us:
#   - þ / ð = "th" (same convention as OE)
#   - j = /j/ "y" (Jórvík → "yor-VEEK")
#   - hj / hl / hr / hn = aspirated cluster ("h" sounded)
#   - á = /aː/ "ow" (lots of variance; "ah" or "ow" both used in respelling
#     conventions; we go with "AH" for clarity)
#   - ǫ = /ɔ/ "or" (open o)
#   - long vowels marked with acute accent

OLD_NORSE = [
    # Long vowels with acute accent
    Phoneme("á", "aː", "AH"),
    Phoneme("é", "eː", "AY"),
    Phoneme("í", "iː", "EE"),
    Phoneme("ó", "oː", "OH"),
    Phoneme("ú", "uː", "OO"),
    Phoneme("ý", "yː", "EE"),
    Phoneme("ǽ", "ɛː", "AY"),
    Phoneme("ǿ", "øː", "EU"),
    Phoneme("ø", "ø", "eu"),
    Phoneme("ǫ", "ɔ", "or"),
    # Aspirated consonant clusters
    Phoneme("hj", "ç", "hy"),
    Phoneme("hl", "l̥", "hl"),
    Phoneme("hr", "r̥", "hr"),
    Phoneme("hn", "n̥", "hn"),
    Phoneme("hv", "ʍ", "hv"),
    # Norse-specific consonants
    Phoneme("þ", "θ", "th"),
    Phoneme("ð", "ð", "th"),
    Phoneme("æ", "æ", "a"),
    # Vowels (short)
    Phoneme("a", "a", "ah"),
    Phoneme("e", "ɛ", "eh"),
    Phoneme("i", "ɪ", "ih"),
    Phoneme("o", "ɔ", "oh"),
    Phoneme("u", "ʊ", "uh"),
    Phoneme("y", "y", "ih"),
    # Single consonants
    Phoneme("j", "j", "y"),
    Phoneme("b", "b", "b"),
    Phoneme("d", "d", "d"),
    Phoneme("f", "f", "f"),
    Phoneme("g", "g", "g"),
    Phoneme("h", "h", "h"),
    Phoneme("k", "k", "k"),
    Phoneme("l", "l", "l"),
    Phoneme("m", "m", "m"),
    Phoneme("n", "n", "n"),
    Phoneme("p", "p", "p"),
    Phoneme("r", "r", "r"),
    Phoneme("s", "s", "s"),
    Phoneme("t", "t", "t"),
    Phoneme("v", "v", "v"),
]


# --- Welsh / Brittonic ----------------------------------------------------
#
# Sources: Thomas, *The Welsh Language: Studies in its Pronunciation* (1958);
# practical conventions from BBC Cymru pronunciation guides. We focus on
# place-name elements and ignore the more exotic mutation rules — those
# happen at composition time, not in this table.
#
# Notable:
#   - dd = /ð/ "th as in this"
#   - ll = /ɬ/ — voiceless lateral fricative, no English equivalent;
#     respelled "thl" or "(c)hl" for English speakers
#   - f = /v/ (Welsh f is English V); ff = /f/ (Welsh ff is English F)
#   - w can be vowel /uː/ between consonants (Cwm → "koom")
#   - y = /ə/ usually, /ɨ/ in final syllables
#   - rh = /r̥/ (voiceless r)
#   - Stress: usually penultimate

WELSH = [
    # Digraphs
    Phoneme("dd", "ð", "th"),
    Phoneme("ll", "ɬ", "hl"),  # voiceless lateral; closest English approx
    Phoneme("ff", "f", "f"),
    Phoneme("ph", "f", "f"),
    Phoneme("ch", "x", "kh"),  # back fricative
    Phoneme("rh", "r̥", "hr"),
    Phoneme("th", "θ", "th"),
    Phoneme("ng", "ŋ", "ng"),
    # Consonants
    Phoneme("f", "v", "v"),
    Phoneme("c", "k", "k"),
    Phoneme("b", "b", "b"),
    Phoneme("d", "d", "d"),
    Phoneme("g", "g", "g"),
    Phoneme("h", "h", "h"),
    Phoneme("l", "l", "l"),
    Phoneme("m", "m", "m"),
    Phoneme("n", "n", "n"),
    Phoneme("p", "p", "p"),
    Phoneme("r", "r", "r"),
    Phoneme("s", "s", "s"),
    Phoneme("t", "t", "t"),
    # W as vowel between consonants (e.g. cwm) vs as glide (gwlad)
    Phoneme("w", "u", "oo"),  # default vowel reading; consonant rule special-cases
    # Vowels
    Phoneme("a", "a", "ah"),
    Phoneme("e", "ɛ", "eh"),
    Phoneme("i", "i", "ee"),
    Phoneme("o", "ɔ", "oh"),
    Phoneme("u", "ɨ", "ih"),  # Welsh "u" is unrounded high central, weird to English
    Phoneme("y", "ə", "uh"),  # default; final-syllable /ɨ/ handled separately
    # Long vowels (marked with circumflex)
    Phoneme("â", "aː", "AH"),
    Phoneme("ê", "eː", "AY"),
    Phoneme("î", "iː", "EE"),
    Phoneme("ô", "oː", "OH"),
    Phoneme("û", "uː", "OO"),
    Phoneme("ŵ", "uː", "OO"),
    Phoneme("ŷ", "ɨː", "EE"),
]


# --- registry ------------------------------------------------------------

PHONOLOGY: dict[str, list[Phoneme]] = {
    "old-english": OLD_ENGLISH,
    "old-norse": OLD_NORSE,
    "welsh": WELSH,
}


# --- application engine --------------------------------------------------


def to_respelling(form: str, language: str) -> str:
    """Convert an etymon spelling to an English-friendly respelling.

    Greedy left-to-right longest-match against the language's phoneme list.
    Returns the respelled form with a hyphen between detected syllables
    when we can identify them (currently only via vowel-consonant
    transitions; not perfect).

    For now, doesn't handle stress — that's a separate pass once we have
    syllabification working.
    """
    rules = PHONOLOGY.get(language)
    if rules is None:
        return form  # no rules — pass through unchanged
    out: list[str] = []
    i = 0
    s = form.lower()
    while i < len(s):
        match = None
        for p in rules:
            if s.startswith(p.grapheme, i):
                match = p
                break
        if match is None:
            out.append(s[i])
            i += 1
        else:
            nxt = i + len(match.grapheme)
            coda = match.coda_respelling is not None and (nxt >= len(s) or s[nxt] not in _VOWELS)
            out.append(match.coda_respelling if coda else match.respelling)
            i = nxt
    return "".join(out)


def to_ipa(form: str, language: str) -> str:
    """Convert an etymon spelling to a (rough) IPA representation."""
    rules = PHONOLOGY.get(language)
    if rules is None:
        return form
    out: list[str] = []
    i = 0
    s = form.lower()
    while i < len(s):
        match = None
        for p in rules:
            if s.startswith(p.grapheme, i):
                match = p
                break
        if match is None:
            out.append(s[i])
            i += 1
        else:
            nxt = i + len(match.grapheme)
            coda = match.coda_ipa is not None and (nxt >= len(s) or s[nxt] not in _VOWELS)
            out.append(match.coda_ipa if coda else match.ipa)
            i = nxt
    return "/" + "".join(out) + "/"
