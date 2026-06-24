"""wyrd-03cx: deterministic IPA → reader-friendly pronunciation.

The bundle's morpheme renderings carry IPA per source-language form
(``/xɑːm/``, ``/byrjj/``, ``/ɬɨ̞n/``) which is precise but opaque to
non-linguists. ``english_shaped`` was designed to hold the reader-
friendly version but has zero populated entries for British place-
name morphemes; the ``irish_anglicizations.json`` sidecar covers
only ~10 Irish prefixes.

This module exposes ``anglicize_ipa(ipa)`` — a deterministic mapper
covering the OE / ME / Welsh / Old French / Celtic phonemic inventory
that actually appears in the corpus. Output is a single uppercase
ASCII token suitable for inline rendering next to the source form
(``hām`` /xɑːm/ KHAAM).

Scope: best-effort, not a substitute for hand-curated english_shaped
data when it exists. Returns ``None`` on input that's not IPA-shaped
or that has no recognizable phonemes left after stripping markers.
"""

from __future__ import annotations

# Multi-char IPA digraphs/clusters — checked FIRST so /t͡ʃ/ doesn't
# get rendered as 'TUH' (t + U-tie + ʃ) before the SH-cluster fires.
# Order matters within this table: longer keys before shorter prefixes
# of the same key would shadow the shorter match in dict iteration,
# but we iterate sorted-by-length-desc at runtime so any order works.
_DIGRAPH_MAP: dict[str, str] = {
    "t͡ʃ": "CH",
    "d͡ʒ": "J",
    "tʃ": "CH",
    "dʒ": "J",
    "ǰ": "J",
    "ɬ": "LL",  # Welsh voiceless lateral fricative (Llanelli's ll)
    "ŋ": "NG",
    "θ": "TH",
    "ð": "DH",
    "ʃ": "SH",
    "ʒ": "ZH",
    "ʧ": "CH",
    "ʤ": "J",
}

# Digraph keys longest-first, so a cluster is matched before any shorter key
# that is its substring (``tʃ`` before ``ʃ``). Materialised once at import
# rather than re-sorted on every anglicize_ipa call.
_SORTED_DIGRAPHS: tuple[str, ...] = tuple(sorted(_DIGRAPH_MAP, key=len, reverse=True))

# Single-char IPA → reader letter. wyrd-03cx round 2: vowels are
# single-letter so the length marker (ː) can double them cleanly
# (``/xɑːm/`` → ``KH + A + A (doubled) + M`` → ``KHAAM``). The pre-
# round-2 multi-char vowels (ɑ → "AH", ə → "UH", ɔ → "AW") had the
# length marker silently no-op because ``out[-1][-1]`` was a consonant
# letter — preserving the bug for one edge case isn't worth the
# multi-char distinctiveness. Short/long vowel distinctions collapse
# to the same base letter; that's normal for English respelling.
_PHONEME_MAP: dict[str, str] = {
    # consonants
    "x": "KH",  # OE 'h' before back vowel (hām, niht)
    "ʔ": "",  # glottal stop — drop
    "ɣ": "GH",
    "ç": "KH",  # German-style ich-laut
    "χ": "KH",  # uvular fricative
    "j": "Y",
    "ɹ": "R",
    "ɾ": "R",
    "ʁ": "R",
    # wyrd-03cx round 2: explicit IPA consonants that real-corpus
    # samples hit but the round-1 table dropped silently. ɡ (U+0261)
    # vs ASCII g (U+0067) is the critical one — ɡ is the IPA voiced
    # velar stop and appears in '/ˈinˌɡɑː/'-style transcriptions in
    # meanings.json. Without an entry it dropped to nothing.
    "ɡ": "G",
    "ɲ": "NY",  # Castilian ñ, Italian gn
    "β": "V",  # Spanish 'b' between vowels — closest English bucket
    "ɫ": "L",  # velarized 'l' (English 'dark l')
    "ɟ": "J",  # voiced palatal stop
    "ʝ": "Y",  # voiced palatal fricative
    "ʰ": "",  # aspiration mark — drop (audible 'h' is rare next-letter)
    # vowels
    "ɑ": "A",
    "æ": "A",
    "ɛ": "E",
    "ɪ": "I",
    "ʊ": "U",
    "ʌ": "U",
    "ɔ": "O",
    "ə": "U",
    "ɨ": "I",
    "ʏ": "I",
    "œ": "U",
    "ø": "U",
    "ɒ": "O",
    "y": "I",  # OE /y/ — front-rounded; modern reflex usually /i/
    # base ASCII vowels that survive unchanged in IPA + need uppercasing
    "a": "A",
    "e": "E",
    "i": "I",
    "o": "O",
    "u": "U",
    # base ASCII consonants that survive unchanged
    "b": "B",
    "d": "D",
    "f": "F",
    "g": "G",
    "h": "H",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "p": "P",
    "r": "R",
    "s": "S",
    "t": "T",
    "v": "V",
    "w": "W",
    "z": "Z",
}

# Markers to strip wholesale before mapping. Length (ː ˑ), stress
# (ˈ ˌ), syllable separator (.), tone marks, brackets, slashes, ties
# (͡ ͜). Tildes (nasalization) and combining diacritics in the BMP
# combining range get dropped too — they don't change the broad
# reader-friendly bucket.
_STRIPPED: frozenset[str] = frozenset(
    {
        "/",
        "[",
        "]",
        "(",
        ")",
        " ",
        "\t",
        "\n",
        "ˈ",
        "ˌ",
        ".",
        "‿",
        "·",
        "ː",
        "ˑ",
        "˞",
        "͡",
        "͜",  # tie bars
        "˜",  # nasal
    }
)


# Combining diacritic range (U+0300-U+036F) covers most of the
# combining marks that show up in the corpus IPA without us needing
# to enumerate them: combining grave, acute, breve, ring, etc.
def _is_combining_diacritic(ch: str) -> bool:
    return "̀" <= ch <= "ͯ"


def anglicize_ipa(ipa: str | None) -> str | None:
    """wyrd-03cx: map an IPA transcription to an uppercase reader-
    pronunciation token.

    Strips bracketing (``/.../`` or ``[...]``), stress + length +
    syllable markers, ties, and combining diacritics. Maps the
    remaining IPA phonemes via ``_DIGRAPH_MAP`` (multi-char clusters
    first) then ``_PHONEME_MAP`` (singles). The ``ː`` length marker
    doubles the preceding vowel letter (``/xɑːm/`` → ``KHAAM`` — the
    ``A`` from ``ɑ`` doubles to ``AA``). Vowels intentionally map to
    single letters (not multi-char digraphs) so the doubling rule
    fires reliably; short/long-vowel distinctions collapse to the
    same base letter, which matches dictionary respelling convention.

    Returns ``None`` for empty input or for input whose phonemes are
    all stripped (no recognizable content). Returns the rendered
    token otherwise.

    Examples:
        /xɑːm/   → 'KHAAM'  (KH + A + length-doubled A + M)
        /wyrm/   → 'WIRM'
        /kumb/   → 'KUMB'
        /byrjj/  → 'BIRYY'  (jj is two glides, both map to Y)
        /ɬɨ̞n/   → 'LLIN'
        /æ͜ɑː/  → 'AAA'    (æ→A, ͜ stripped, ɑ→A, ː doubles → AAA)
        /ˈbɛrən/ → 'BERUN'  (stress stripped; ə → U)
    """
    if ipa is None:
        return None
    s = ipa.strip()
    if not s:
        return None

    # Strip combining diacritics first so subsequent char-by-char
    # mapping doesn't trip on a base+diacritic pair.
    s = "".join(ch for ch in s if not _is_combining_diacritic(ch))
    result = "".join(_map_phonemes(_mark_digraphs(s)))
    if not result:
        return None
    # Collapse runs of 3+ same letter to 2. Preserves intentional
    # doubled letters ('LL' from ɬ; 'AA' from length-marked vowel)
    # while clipping pathological triples (a hypothetical /ɑːː/
    # would produce AAA which reads as a typo, collapsed to AA).
    return _collapse_runs(result)


def _mark_digraphs(s: str) -> str:
    """Wrap each multi-char IPA cluster's mapping in ``\\0`` sentinels.

    The sentinels let the single-phoneme pass lift the mapping out verbatim
    instead of re-mapping its letters. Clusters are applied longest-first
    (``_SORTED_DIGRAPHS``) so ``tʃ`` is matched before ``ʃ`` (its suffix, also
    a digraph key)."""
    # Drop any pre-existing NUL so every ``\0`` in the result is a sentinel we
    # added (paired). Otherwise a stray ``\0`` in the input leaves _map_phonemes'
    # ``s.index("\0", i + 1)`` searching for a non-existent partner → ValueError.
    s = s.replace("\0", "")
    for digraph in _SORTED_DIGRAPHS:
        s = s.replace(digraph, f"\0{_DIGRAPH_MAP[digraph]}\0")
    return s


def _map_phonemes(s: str) -> list[str]:
    """Render the digraph-marked string into output letter-tokens.

    The load-bearing detail is the coupling to :func:`_mark_digraphs`: this
    consumes the ``\\0`` sentinels it planted, lifting each wrapped digraph
    mapping out verbatim. Beyond that it doubles a vowel before a ``ː`` length
    marker, drops stripped/unknown chars, and maps singles via ``_PHONEME_MAP``."""
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\0":
            # Sentinel for an already-mapped digraph; lift the run
            # between sentinels into the output verbatim.
            j = s.index("\0", i + 1)
            out.append(s[i + 1 : j])
            i = j + 1
            continue
        if ch in _STRIPPED:
            # Length marker doubles the previous vowel. Other
            # stripped chars vanish.
            if ch == "ː" and out and out[-1] and out[-1][-1] in "AEIOUY":
                out[-1] = out[-1] + out[-1][-1]
            i += 1
            continue
        mapped = _PHONEME_MAP.get(ch)
        if mapped is not None:
            out.append(mapped)
        # Unknown chars (rare diacritics, exotic phonemes) silently
        # drop — better to ship a partial reader-pronunciation than
        # to bail entirely.
        i += 1
    return out


def _collapse_runs(s: str) -> str:
    """Collapse runs of 3+ same letter to 2. The length-marker
    expansion produces a doubled vowel (single-char vowel mappings
    + the length-extend rule); two ː markers in a row (rare,
    pathological) would produce AAA which collapses back to AA."""
    if len(s) < 3:
        return s
    out: list[str] = [s[0], s[1]]
    for ch in s[2:]:
        if ch == out[-1] == out[-2]:
            continue
        out.append(ch)
    return "".join(out)
