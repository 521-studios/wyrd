"""wyrd-17t: rule-derived pronunciation respelling for non-modern-English
output morphemes.

When the generator produces names like ``Brycgwæter``, ``Pont-Dwfr``,
``Brúarvatn``, English-speaking users can't sound them out. We need a
respelling alongside the canonical form so they're usable at the table.

This module is the **rule-derived** path (per-language grapheme→phonetic-
respelling tables). The complementary **scholar-attested** path (mining
phonetic transcriptions in square brackets out of Mawer / Skeat /
Watson dictionaries) is deferred — those would land as
`etymon.pronunciation_ipa` rows during ingest, and the runtime
accessor would consult them before falling through to the rules.

Output shape: SAMPA-lite respelling like ``BRIDGE-wuh-tər`` rather
than IPA. More accessible at the table than `/brɪdʒwɔːtə/` and good
enough for the GM's purpose. Stress marked with ALL CAPS on the
stressed syllable; syllables separated by hyphens. The
linguistically-interested user can still ask for IPA via a verbose
flag at higher layers.

Per-language rules cover the main pronunciation pitfalls for
English-speaking readers:

* **Old English** — ``æ`` (a), ``ð`` / ``þ`` (th), ``cg`` (j-sound),
  ``sc`` (sh), palatalised ``c`` (ch), long-vowel macrons (drop the
  macron, lengthen the syllable orthographically).
* **Welsh** — ``dd`` (th-voiced), ``ll`` (hl), ``f`` (v), ``ff`` (f),
  ``w`` as vowel (oo / u), ``y`` (uh / i depending on syllable),
  ``ch`` (kh).
* **Old Norse** — ``þ`` (th), ``ð`` (th-voiced), ``j`` (y),
  ``á`` / ``ó`` / ``ú`` (ah / oh / oo).
* **Norman French / Old French** — silent final ``e``, ``é`` (ay),
  ``ç`` (s), ``ai`` (ay), ``ou`` (oo).
* **Latin** — largely transparent for English readers. Mark long
  vowels and final ``e`` is voiced (not silent).
* **Greek** (transliterated forms) — ``ph`` (f), ``ch`` / ``kh`` (k),
  ``th`` already standard.

The respeller is intentionally lossy: many phonological details
(vowel-quantity-conditioned stress, dialect variants, sandhi)
aren't represented because SAMPA-lite is a teaching aid, not a
phonetic transcription. Better Wrong Than Confusing.
"""

from __future__ import annotations

import re

# Map language tags from the etymon/Meaning layer to respeller
# implementations. Languages not in the map have no rule-derived
# respelling and the accessor returns None.
#
# Multiple tags can map to the same respeller (e.g. all the
# Brittonic varieties share Welsh-style rules). The mapping
# tolerates dashed-lowercase, the bundle's lang_field shape
# (``old_english``, ``celtic_mix``), and ISO codes.


def _respell_old_english(form: str) -> str:
    """Old English grapheme→SAMPA-lite mapping.

    Rules:
    * ``æ`` / ``Æ`` → ``a``
    * ``ǣ`` → ``a`` (long; orthographically the same in respelling)
    * ``ð`` / ``Ð`` / ``þ`` / ``Þ`` → ``th``
    * ``cg`` → ``j`` (the OE digraph for /dʒ/, e.g. ``brycg`` → bridj)
    * ``sc`` → ``sh`` (e.g. ``scip`` → ship)
    * ``c`` before front vowels (e/i/y/æ) → ``ch``; otherwise ``k``
    * ``g`` before front vowels → ``y`` (palatalisation, e.g. ``gear``
      → year); otherwise ``g``
    * Long-vowel macrons (ā ē ī ō ū ȳ) → drop the macron; rely on
      orthography to imply length to readers (English equivalents
      already mark length via spelling: 'a' vs 'aw', 'i' vs 'eye').

    A future refinement could add stress marking on the first
    syllable (Germanic root-initial-stress convention) but the
    syllable-detection layer isn't ready for that v1.
    """
    s = form
    # Digraphs first to avoid being eaten by single-char passes.
    s = re.sub(r"cg", "j", s, flags=re.IGNORECASE)
    s = re.sub(r"sc", "sh", s, flags=re.IGNORECASE)
    # Hard-c first (anywhere NOT followed by front vowel) → k.
    # Palatal-c (followed by front vowel) handled in the second
    # pass. This order matters: doing palatal-c first then bare-c
    # would re-match the 'c' in the just-produced 'ch' and turn
    # 'cheaster' into 'kheaster'.
    s = re.sub(r"c(?![eiyæǣ])", "k", s, flags=re.IGNORECASE)
    s = re.sub(r"c(?=[eiyæǣ])", "ch", s, flags=re.IGNORECASE)
    # Same for ``g``: palatal before front vowels, hard otherwise.
    s = re.sub(r"g(?=[eiyæǣ])", "y", s, flags=re.IGNORECASE)
    # æ / ð / þ pairs.
    s = s.replace("æ", "a").replace("Æ", "A")
    s = s.replace("ǣ", "a").replace("Ǣ", "A")
    s = s.replace("ð", "th").replace("Ð", "Th")
    s = s.replace("þ", "th").replace("Þ", "Th")
    # Strip macrons.
    for long_v, plain_v in [
        ("ā", "a"),
        ("ē", "e"),
        ("ī", "i"),
        ("ō", "o"),
        ("ū", "u"),
        ("ȳ", "y"),
        ("Ā", "A"),
        ("Ē", "E"),
        ("Ī", "I"),
        ("Ō", "O"),
        ("Ū", "U"),
        ("Ȳ", "Y"),
    ]:
        s = s.replace(long_v, plain_v)
    return s


def _respell_welsh(form: str) -> str:
    """Welsh grapheme→SAMPA-lite mapping.

    Welsh is the trickiest for English readers because several common
    digraphs map to phonemes English doesn't have a clean spelling for:

    * ``dd`` → ``th`` (voiced 'th' as in 'this')
    * ``ll`` → ``hl`` (the lateral fricative; 'hl' is the closest
      English approximation. Real IPA is /ɬ/.)
    * ``f`` → ``v`` (Welsh single-f is voiced; 'ff' is voiceless)
    * ``ff`` → ``f``
    * ``ch`` → ``kh`` (the voiceless velar fricative, like German 'Bach')
    * ``rh`` → ``hr`` (voiceless r; uncommon enough to gloss over)
    * ``w`` between consonants → ``oo`` (Welsh w-as-vowel)
    * ``y`` → ``uh`` in non-final syllables, ``i`` in final
      (`mynydd` → MUH-nith, but `cymry` → KUM-ri); v1 keeps it
      simple with ``uh``.
    * Circumflex marks (â ê î ô û ŵ ŷ) → drop and lengthen
      orthographically.
    """
    s = form
    # 'ff' (voiceless) and 'f' (voiced) are separate phonemes but
    # 'ff' contains 'f', so a naive sub-then-sub would chain — 'ff'
    # → 'f' → 'v' renders 'ff' as 'v'. Atomic alternation so each
    # source maps to its own output:
    s = re.sub(
        r"ff|f",
        lambda m: "f" if m.group().lower() == "ff" else "v",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"dd", "th", s, flags=re.IGNORECASE)
    s = re.sub(r"ll", "hl", s, flags=re.IGNORECASE)
    s = re.sub(r"ch", "kh", s, flags=re.IGNORECASE)
    s = re.sub(r"rh", "hr", s, flags=re.IGNORECASE)
    # 'w' between consonants is a vowel; 'w' adjacent to a vowel
    # stays consonantal. Heuristic: replace 'w' surrounded by
    # consonant context with 'oo'.
    s = re.sub(r"(?<=[bcdghjklmnprstvz])w(?=[bcdghjklmnprstvz])", "oo", s, flags=re.IGNORECASE)
    # 'y' as vowel — simple 'uh' fallback.
    s = re.sub(r"y", "uh", s, flags=re.IGNORECASE)
    # Strip circumflex.
    for marked, plain in [
        ("â", "a"),
        ("ê", "e"),
        ("î", "i"),
        ("ô", "o"),
        ("û", "u"),
        ("ŵ", "w"),
        ("ŷ", "uh"),
        ("Â", "A"),
        ("Ê", "E"),
        ("Î", "I"),
        ("Ô", "O"),
        ("Û", "U"),
        ("Ŵ", "W"),
        ("Ŷ", "Uh"),
    ]:
        s = s.replace(marked, plain)
    return s


def _respell_old_norse(form: str) -> str:
    """Old Norse grapheme→SAMPA-lite mapping.

    * ``þ`` → ``th`` (voiceless 'th' as in 'thin')
    * ``ð`` → ``th`` (voiced 'th' as in 'this' — distinguishing
      voiced/voiceless via spelling alone is hard for English
      readers; lump both as ``th`` and accept the loss)
    * ``j`` → ``y`` (Norse 'j' is /j/, the English 'y')
    * ``á`` / ``ó`` / ``ú`` / ``í`` / ``ý`` / ``é`` → drop accent
    * ``ǫ`` → ``o`` (the open-o; rare in modern editions)
    * ``ø`` / ``œ`` → ``e`` (English-friendly approximation)
    * ``hv`` → ``hw`` (initial pre-vowel 'hv' was /hw/, kept as 'hw'
      so English readers preserve the breath)
    """
    s = form
    s = s.replace("þ", "th").replace("Þ", "Th")
    s = s.replace("ð", "th").replace("Ð", "Th")
    s = re.sub(r"\bhv", "hw", s, flags=re.IGNORECASE)
    s = re.sub(r"j", "y", s, flags=re.IGNORECASE)
    for marked, plain in [
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ý", "y"),
        ("ǫ", "o"),
        ("ø", "e"),
        ("œ", "e"),
        ("Á", "A"),
        ("É", "E"),
        ("Í", "I"),
        ("Ó", "O"),
        ("Ú", "U"),
        ("Ý", "Y"),
        ("Ǫ", "O"),
        ("Ø", "E"),
        ("Œ", "E"),
    ]:
        s = s.replace(marked, plain)
    return s


def _respell_old_french(form: str) -> str:
    """Old French / Norman French grapheme→SAMPA-lite mapping.

    * ``ç`` → ``s``
    * ``é`` → ``ay``
    * ``è`` / ``ê`` → ``e``
    * ``à`` → ``a``
    * ``ai`` → ``ay``
    * ``ou`` → ``oo``
    * ``oi`` → ``wa`` (Modern-French rendering; works for OF too)
    * Final ``e`` after a consonant → silent (drop)
    """
    s = form
    s = s.replace("ç", "s").replace("Ç", "S")
    s = s.replace("é", "ay").replace("É", "Ay")
    for marked, plain in [
        ("è", "e"),
        ("ê", "e"),
        ("à", "a"),
        ("â", "a"),
        ("È", "E"),
        ("Ê", "E"),
        ("À", "A"),
        ("Â", "A"),
    ]:
        s = s.replace(marked, plain)
    s = re.sub(r"ai", "ay", s, flags=re.IGNORECASE)
    s = re.sub(r"ou", "oo", s, flags=re.IGNORECASE)
    s = re.sub(r"oi", "wa", s, flags=re.IGNORECASE)
    # Drop final-e after a consonant. Lookbehind: consonant + e
    # at end-of-string.
    s = re.sub(r"(?<=[bcdfghjklmnpqrstvwxz])e$", "", s, flags=re.IGNORECASE)
    return s


def _respell_latin(form: str) -> str:
    """Latin grapheme→SAMPA-lite mapping. Mostly transparent.

    * Macrons (ā ē ī ō ū) → drop
    * Final ``e`` is voiced (NOT silent). No transformation needed —
      respeller relies on the reader applying English-final-e habits;
      a future refinement could append a hint like ``-ay`` for
      'classical-style' final-e but that's loud.
    * ``ae`` → ``ie`` (the diphthong; classical Latin /ai/, English-
      reader-approximate is 'ie' as in 'pie').
    * ``oe`` → ``oy``
    * ``c`` before e/i/y → ``s`` is the Ecclesiastical pronunciation;
      ``k`` is Classical. We use ``k`` everywhere (Classical) since
      that matches the spelling-pronunciation closer for an English
      reader expecting 'c' to be either 'k' or 's' depending on
      following vowel.
    """
    s = form
    for marked, plain in [
        ("ā", "a"),
        ("ē", "e"),
        ("ī", "i"),
        ("ō", "o"),
        ("ū", "u"),
        ("Ā", "A"),
        ("Ē", "E"),
        ("Ī", "I"),
        ("Ō", "O"),
        ("Ū", "U"),
    ]:
        s = s.replace(marked, plain)
    s = re.sub(r"ae", "ie", s, flags=re.IGNORECASE)
    s = re.sub(r"oe", "oy", s, flags=re.IGNORECASE)
    return s


def _respell_greek(form: str) -> str:
    """Greek transliterated form → SAMPA-lite mapping.

    For pre-transliterated Greek (e.g. wiktextract gives
    'transliteration' field). Native script handled separately by
    wyrd-y10's script-rendering layer.

    * ``ph`` → ``f``
    * ``ch`` / ``kh`` → ``k``
    * ``th`` already English-readable
    * Macrons / breves → drop
    """
    s = form
    s = re.sub(r"ph", "f", s, flags=re.IGNORECASE)
    s = re.sub(r"kh", "k", s, flags=re.IGNORECASE)
    s = re.sub(r"ch", "k", s, flags=re.IGNORECASE)
    for marked, plain in [
        ("ā", "a"),
        ("ē", "e"),
        ("ī", "i"),
        ("ō", "o"),
        ("ū", "u"),
        ("ă", "a"),
        ("ĕ", "e"),
        ("ĭ", "i"),
        ("ŏ", "o"),
        ("ŭ", "u"),
        ("Ā", "A"),
        ("Ē", "E"),
        ("Ī", "I"),
        ("Ō", "O"),
        ("Ū", "U"),
    ]:
        s = s.replace(marked, plain)
    return s


# Language tag → respeller dispatch.
#
# Both etymon.language tags ('old-english') and bundle lang_field
# tags ('old_english', 'celtic_mix') are mapped so callers can pass
# either. Multiple tags fan out to one respeller for language
# families with shared rules (Welsh family is shared across
# old-welsh / middle-welsh / welsh / celtic_mix; Brythonic group
# uses Welsh rules as approximation).
_LANGUAGE_TO_RESPELLER: dict[str, callable[[str], str]] = {
    # English family (old / middle — modern is transparent)
    "old-english": _respell_old_english,
    "old_english": _respell_old_english,
    "old _english": _respell_old_english,  # legacy bundle key
    "middle-english": _respell_old_english,  # close enough; Middle English uses same digraphs
    "middle_english": _respell_old_english,
    # Welsh / Brythonic
    "welsh": _respell_welsh,
    "old-welsh": _respell_welsh,
    "middle-welsh": _respell_welsh,
    "cornish": _respell_welsh,
    "breton": _respell_welsh,
    "old-breton": _respell_welsh,
    "middle-breton": _respell_welsh,
    "celtic_mix": _respell_welsh,  # the catch-all bundle bucket
    "celtic": _respell_welsh,
    "cel-bry-pro": _respell_welsh,  # Proto-Brittonic
    # Goidelic — Irish / Scottish Gaelic — share Welsh-style 'mh'/'ch'
    # but have their own deeper rules. Map to Welsh for v1; refinement
    # is a follow-up.
    "irish": _respell_welsh,
    "old-irish": _respell_welsh,
    "middle-irish": _respell_welsh,
    "scottish-gaelic": _respell_welsh,
    "manx": _respell_welsh,
    # Norse family
    "old-norse": _respell_old_norse,
    "old_scandinavian": _respell_old_norse,  # legacy bundle key
    "old_scandanavian": _respell_old_norse,  # legacy misspelling
    "icelandic": _respell_old_norse,
    "faroese": _respell_old_norse,
    "norwegian": _respell_old_norse,
    "norwegian-bokmal": _respell_old_norse,
    "norwegian-nynorsk": _respell_old_norse,
    "danish": _respell_old_norse,
    "swedish": _respell_old_norse,
    # French / Norman
    "norman-french": _respell_old_french,
    "old-french": _respell_old_french,
    "old_french": _respell_old_french,
    "middle-french": _respell_old_french,
    "french": _respell_old_french,
    # Latin
    "latin": _respell_latin,
    "vulgar-latin": _respell_latin,
    # Greek
    "greek": _respell_greek,
    "ancient-greek": _respell_greek,
}


def respell(form: str, language: str) -> str | None:
    """Return a SAMPA-lite respelling of ``form`` for the given
    language tag, or None when the language has no respeller.

    Modern English isn't in the dispatch — we don't respell English
    surface forms because the user can already pronounce them. This
    is the 'pronunciation guide for non-modern-English outputs'
    contract per the wyrd-17t ticket.
    """
    if not form:
        return None
    respeller = _LANGUAGE_TO_RESPELLER.get(language)
    if respeller is None:
        return None
    return respeller(form)


def has_respeller(language: str) -> bool:
    """True if ``language`` has a registered respeller. Lets the SPA
    decide whether to render a respelling slot at all."""
    return language in _LANGUAGE_TO_RESPELLER
