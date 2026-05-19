"""IPA + transliteration → English-shaped renderings (wyrd-ha9q Phase 2b).

Etymons from non-Latin source languages (Hebrew, Arabic, Sanskrit, Egyptian,
etc.) carry transliteration with diacritics academic readers expect (rakṣasa,
ʿifrīt, gōlem) but town-name generation needs Latin-only renderings English
readers can decode without decoding ʿ ʾ ḍ ḥ ḫ ṣ ṭ ṯ ẓ ʂ ṛ ṇ ṅ ñ macrons etc.
This module derives `english_shaped` from the four data sources Phase 2a
captured, in priority order:

  1. KNOWN_FORM_OVERRIDES — established English forms (rakshasa, jinn, golem,
     ifrit, marid, shaitan, sphinx, ...). These are NOT rule-derived; they
     reflect cultural / literary precedent and override anything mechanical.
  2. transliteration — strip diacritics + apply transliteration-specific
     digraph rules (ṯ→th, ḫ→kh, ǧ→j, š→sh, ʿ→silent, etc.). The dominant
     path; covers ~80%+ of wave-2 rows that have transliteration data.
  3. pronunciation_ipa — last-resort fallback when transliteration is
     absent. Strips IPA-specific markers (slashes, brackets, length marks,
     stress marks) and reduces the segment string to a Latin-letter
     approximation. Coarser than (2); intended for the small wave-2 tail
     where we have IPA but no transliteration.

Returns None when none of the inputs is sufficient OR the source language
is in `_LATIN_SCRIPT_LANGS` (canonical_form is the right display value for
those — old-english `tūn`, latin `pons`, welsh `bryn`, etc. handle their
own register elsewhere). Note: ancient-greek is intentionally NOT in the
Latin-script set — its canonical_form uses the Greek alphabet (ἅρπυια)
and benefits from the same diacritic-strip pipeline (η→e, ω→o, etc.) the
wave-2 languages do.
"""

from __future__ import annotations

import re

# Languages whose canonical_form is already English-readable (or
# transliteration-style). english_shaped should stay NULL for these — town
# name generation already uses canonical_form for them. Includes the
# wave-1 Indo-European stack and wave-2 precursor codes whose attested
# forms are written in Latin script.
_LATIN_SCRIPT_LANGS: frozenset[str] = frozenset(
    {
        "old-english",
        "middle-english",
        "modern-english",
        "old-norse",
        "icelandic",
        "faroese",
        "old-french",
        "middle-french",
        "french",
        "latin",
        "vulgar-latin",
        "old-high-german",
        "middle-high-german",
        "german",
        "dutch",
        "old-dutch",
        "old-saxon",
        "old-frisian",
        "got",
        "osx",
        "ofs",
        "scots",
        "sco",
        "nrn",
        "irish",
        "welsh",
        "scottish-gaelic",
        "old-irish",
        "middle-irish",
        "breton",
        "old-breton",
        "middle-breton",
        "cornish",
        "manx",
        "old-welsh",
        "middle-welsh",
        "proto-germanic",
        "proto-celtic",
        "proto-indo-european",
        "proto-italic",
        "proto-slavic",
        "gmw-pro",
        "iir-pro",
        "ira-pro",
        "inc-pro",
        "sem-pro",
        "sem-wes-pro",
        "afa-pro",
        "celt-pro",
        "ine-pro",
    }
)

# ---------------------------------------------------------------------
# Cultural / literary overrides — case-insensitive match on the
# lowercase transliteration string. These are forms English readers
# recognize as the "standard" spelling regardless of academic
# transliteration convention.
# ---------------------------------------------------------------------

KNOWN_FORM_OVERRIDES: dict[str, str] = {
    # Sanskrit (Hindu / Buddhist creature canon, D&D-canonical names)
    "rakṣasa": "rakshasa",
    "rākṣasa": "rakshasa",
    "rakṣas": "rakshasa",
    "nāga": "naga",
    "garuḍa": "garuda",
    "yakṣa": "yaksha",
    "deva": "deva",
    "asura": "asura",
    "kinnara": "kinnara",
    "apsara": "apsara",
    "apsaras": "apsara",
    "vidyādhara": "vidyadhara",
    "gandharva": "gandharva",
    "piśāca": "pishacha",
    "bhūta": "bhuta",
    "preta": "preta",
    "mātṛkā": "matrika",
    "ḍākinī": "dakini",
    # Arabic — established English creature names
    "ʿifrīt": "ifrit",
    "ifrīt": "ifrit",
    "ʿifrit": "ifrit",
    "marīd": "marid",
    "ǧinn": "jinn",
    "jinn": "jinn",
    "ǧinnī": "jinni",
    # Both 'djinn' and 'djinnī' drop the leading 'd' in modern English
    # ('jinn' / 'jinni' are the prevailing forms; 'djinn' is the older
    # 19th-century spelling). Both override values are 'jinn' / 'jinni'
    # for consistency.
    "djinn": "jinn",
    "djinnī": "jinni",
    "ǧannī": "janni",
    "jānn": "jann",
    "šayṭān": "shaitan",
    "ġūl": "ghoul",
    "ghūl": "ghoul",
    "rūḥ": "ruh",
    "qarīn": "qarin",
    # Hebrew (biblical creatures)
    "gōlem": "golem",
    "golem": "golem",
    "behēmōṯ": "behemoth",
    "behemoth": "behemoth",
    "līwyāṯān": "leviathan",
    "leviathan": "leviathan",
    "lilīṯ": "lilith",
    "lilith": "lilith",
    "kerūḇ": "cherub",
    "śərāp̄": "seraph",
    "śərāp̄īm": "seraphim",
    "nepilīm": "nephilim",
    "nephilim": "nephilim",
    # Persian
    "parī": "peri",
    "dīv": "div",
    "sīmurġ": "simurgh",
    "zuhra": "zuhra",
    "zohak": "zohak",
    "rukh": "roc",
    # Akkadian (Mesopotamian cosmology)
    "tiāmat": "tiamat",
    "lamassu": "lamassu",
    "apsû": "apsu",
    "humbaba": "humbaba",
    "anzû": "anzu",
    # Egyptian — well-known deity / creature names (Faulkner-style
    # transliteration → English convention)
    "anpw": "anpu",
    "wsjr": "osiris",
    "ḥr": "horus",
    "ḫprr": "khepri",
    "tḥwt": "thoth",
    "bjk": "bik",
    "spẖt": "sphinx",
    # Aramaic
    "šēḏ": "shed",
    "shedah": "shedah",
}


# ---------------------------------------------------------------------
# Diacritic stripping rules.
#
# Two passes:
#   1. Multi-char digraphs first (longest-match-first): ṯ→th, ḫ→kh,
#      š→sh, ǧ→j, etc. Has to run before single-char stripping or the
#      base letter would be lost.
#   2. Single-char replacements: emphatics (ḍ→d, ḥ→h), long vowels
#      (ā→a, ī→i), silenced glottals (ʿ ʾ ʔ → ''), combining marks.
# ---------------------------------------------------------------------

# "Digraphs": replacements that must run BEFORE single-char stripping.
# Most produce two-letter output (ṯ→th, ḫ→kh, š→sh) but some collapse
# to one letter (ǧ→j, ʤ→j) — the unifying property is that the source
# character carries semantic content that would be lost if we treated
# it as a base-letter-plus-diacritic and stripped the diacritic alone.
_DIGRAPH_MAP: dict[str, str] = {
    # Semitic transliteration — produce two-letter digraphs
    "ṯ": "th",
    "Ṯ": "Th",
    "ḏ": "dh",
    "Ḏ": "Dh",
    "ḫ": "kh",
    "Ḫ": "Kh",
    "ġ": "gh",
    "Ġ": "Gh",
    "š": "sh",
    "Š": "Sh",
    "ǧ": "j",
    "Ǧ": "J",
    "č": "ch",
    "Č": "Ch",
    # Sanskrit retroflex / palatal sibilants. Default ṣ → 'sh' is
    # Sanskrit-canonical (rakṣasa → rakshasa, yakṣa → yaksha). Arabic
    # ṣ is the same Unicode character but rendered 's' in English
    # (ṣabr → sabr, not shabr); the language-aware override below
    # corrects that during the strip pipeline.
    "ś": "sh",
    "Ś": "Sh",
    "ṣ": "sh",
    "Ṣ": "Sh",
    "ṝ": "ri",
    "Ṝ": "Ri",
    # IPA segments that sometimes appear in transliteration strings
    "ʤ": "j",
    "ʧ": "ch",
    "ʃ": "sh",
    "ʒ": "zh",
    "θ": "th",
    "ð": "dh",
    "ŋ": "ng",
    "χ": "kh",  # IPA voiceless uvular fricative
}

# Single-char diacritic strips — emphatics, long vowels, retroflexes
# without a digraph form.
_SINGLE_CHAR_MAP: dict[str, str] = {
    # Emphatics
    "ḍ": "d",
    "Ḍ": "D",
    "ḥ": "h",
    "Ḥ": "H",
    "ṭ": "t",
    "Ṭ": "T",
    "ẓ": "z",
    "Ẓ": "Z",
    "ḳ": "q",
    "Ḳ": "Q",
    # Sanskrit retroflex / vocalic single-char
    "ṛ": "r",
    "Ṛ": "R",
    "ḷ": "l",
    "Ḷ": "L",
    "ṇ": "n",
    "Ṇ": "N",
    # Sanskrit palatal / velar nasals — single char NOT digraph. The
    # following consonant supplies the velar / palatal quality, so
    # ṅ→ng + g would double-count: liṅga → "linga" (correct), not
    # "lingga". añjali → "anjali" (correct), not "anyjali".
    "ñ": "n",
    "Ñ": "N",
    "ṅ": "n",
    "Ṅ": "N",
    # Long vowels (macron + circumflex variants)
    "ā": "a",
    "Ā": "A",
    "ī": "i",
    "Ī": "I",
    "ū": "u",
    "Ū": "U",
    "ē": "e",
    "Ē": "E",
    "ō": "o",
    "Ō": "O",
    "â": "a",
    "Â": "A",
    "î": "i",
    "Î": "I",
    "û": "u",
    "Û": "U",
    "ê": "e",
    "Ê": "E",
    "ô": "o",
    "Ô": "O",
    # Acute / grave-accented vowels (Hebrew transliteration uses these
    # for stress: káp, bósem, shabát)
    "á": "a",
    "Á": "A",
    "é": "e",
    "É": "E",
    "í": "i",
    "Í": "I",
    "ó": "o",
    "Ó": "O",
    "ú": "u",
    "Ú": "U",
    "à": "a",
    "À": "A",
    "è": "e",
    "È": "E",
    "ì": "i",
    "Ì": "I",
    "ò": "o",
    "Ò": "O",
    "ù": "u",
    "Ù": "U",
    # IPA-style vowels in transliteration
    "ɛ": "e",
    "ɔ": "o",
    "ə": "e",
    "ɐ": "a",
    "ɪ": "i",  # IPA near-close near-front
    "ʊ": "u",  # IPA near-close near-back
    "ɑ": "a",  # IPA open back unrounded
    "æ": "ae",  # IPA + Old English æ — falls through diacritic-stripping
    "Æ": "Ae",
    # Hebrew rafe consonants (bet with rafe — "v" sound)
    "ḇ": "v",
    "Ḇ": "V",
    # Arabic ḥā / Maltese ħ (IPA "barred h" U+0127): voiceless
    # pharyngeal — typically rendered as 'h' in English.
    "ħ": "h",
    "Ħ": "H",
    # IPA voiced pharyngeal / epiglottal — silent in English shaping
    "ʕ": "",
    "ʡ": "",
    # IPA uvular fricative — usually 'r' in English borrowings.
    "ʁ": "r",
}

# Characters silenced (removed without replacement).
_SILENT_CHARS: frozenset[str] = frozenset(
    {
        "ʿ",  # Hebrew ayin / Arabic ʿayn
        "ʾ",  # Hebrew alef / Arabic hamza
        "ʔ",  # IPA glottal stop
        "ˀ",
        "ː",  # IPA length marker
        "˹",
        "˺",
        "ʹ",
        "'",  # ASCII apostrophe sometimes used for hamza
        # Combining diacritics that shouldn't have survived NFD; safety net.
        "́",  # combining acute
        "̀",  # combining grave
        "̄",  # combining macron
        "̇",  # combining dot above
        "̣",  # combining dot below
        "̱",  # combining macron below
        "̃",  # combining tilde
        "̆",  # combining breve
        "̌",  # combining caron
    }
)


# Match brackets, slashes, length marks, stress marks in IPA strings.
_IPA_STRIP_RE = re.compile(r"[\[\]/⟨⟩⌈⌉ˈˌˑ.\\͡]")
# Subscripts that appear in Akkadian-style transliterations (cuneiform
# sign indices: ar-ga-man-nu, GIR₄, etc.) — strip them since they're
# disambiguation tags, not pronunciation.
_SUBSCRIPT_RE = re.compile(r"[₀-₉]")


# Per-language overrides on the digraph map. Same Unicode character
# can have different conventional English renderings depending on which
# language it came from. Currently only Arabic ṣ deviates (sh in
# Sanskrit / Hebrew / Aramaic / Akkadian; s in Arabic).
_LANGUAGE_DIGRAPH_OVERRIDES: dict[str, dict[str, str]] = {
    "ar": {
        "ṣ": "s",  # Arabic ṣād → 's', not 'sh' (e.g. ṣabr → sabr)
        "Ṣ": "S",
    },
}


def _apply_digraphs(s: str, language: str | None = None) -> str:
    """Apply digraph replacements (multi-char output) before single-char.
    Each replacement is independent — order within the dict doesn't
    matter because no digraph key is a prefix of another.

    `language` (optional) selects per-language overrides on top of the
    base map. e.g. Arabic ṣ uses 's' instead of the Sanskrit-default
    'sh'. Pass-through when `language` has no overrides defined.
    """
    overrides = _LANGUAGE_DIGRAPH_OVERRIDES.get(language or "", {})
    for src, dst in _DIGRAPH_MAP.items():
        if src in s:
            s = s.replace(src, overrides.get(src, dst))
    return s


def _apply_single_chars(s: str) -> str:
    """Apply single-char replacements after digraphs. Splits the string
    into characters, replaces each from the single-char map (or
    silences via _SILENT_CHARS), and rejoins."""
    out: list[str] = []
    for ch in s:
        if ch in _SILENT_CHARS:
            continue
        out.append(_SINGLE_CHAR_MAP.get(ch, ch))
    return "".join(out)


def _strip_transliteration(value: str, language: str | None = None) -> str:
    """Run the transliteration string through digraphs → single-chars →
    cleanup. Output is best-effort ASCII; the caller MUST run the result
    through `_looks_english_readable` before storing — characters our
    maps don't cover survive into the output. `language` (optional)
    routes through `_apply_digraphs` for per-language overrides
    (e.g. Arabic ṣ → s instead of the default sh)."""
    s = _apply_digraphs(value, language=language)
    s = _apply_single_chars(s)
    # Collapse the leftover length markers and brackets.
    s = _IPA_STRIP_RE.sub("", s)
    s = _SUBSCRIPT_RE.sub("", s)
    return s.strip()


def _looks_english_readable(s: str) -> bool:
    """Sanity check on the stripped output: is there at least one ASCII
    letter, and is the string free of remaining non-ASCII characters
    that would still confuse English readers? A residual non-ASCII
    char means our maps missed a case — better to return None and let
    the SPA fall back to canonical_form than to surface a half-stripped
    value that mixes Latin with stragglers."""
    if not s:
        return False
    has_letter = any(c.isalpha() and c.isascii() for c in s)
    if not has_letter:
        return False
    # Allow ASCII alphanumerics, spaces, and a small punctuation set
    # (hyphens / apostrophes appear in Hebrew transliterations like
    # 'ba'al zvuv'). Reject anything else.
    return all(c.isascii() and (c.isalnum() or c in " -'") for c in s)


def _ipa_to_english_fallback(ipa: str, language: str | None = None) -> str | None:
    """Last-resort: collapse an IPA string to a Latin-letter
    approximation. Used only when transliteration is missing.

    This is intentionally coarse — IPA → English orthography is a
    deep linguistic problem. We strip the IPA delimiters / length
    marks, run the same digraph + single-char pipeline, and verify
    the result looks readable. Most rows that take this path will
    return something like 'dzhin' for /d͡ʒɪn/, which is good enough
    as a fallback display string until somebody fills in
    transliteration or adds a known-form override. `language` is
    forwarded for per-language overrides (e.g. Arabic ṣ→s)."""
    s = _IPA_STRIP_RE.sub("", ipa)
    s = _strip_transliteration(s, language=language)
    if _looks_english_readable(s):
        return s
    return None


def derive_english_shaped(
    *,
    canonical_form: str,
    language: str,
    transliteration: str | None,
    pronunciation_ipa: str | None,
) -> str | None:
    """Compute the english_shaped value for a single etymon row.

    Priority order:
        1. KNOWN_FORM_OVERRIDES — case-insensitive match on the
           pre-strip transliteration. Cultural / literary precedent.
        2. Diacritic-stripped transliteration when readable.
        3. IPA fallback when transliteration absent.

    Returns None when:
        - the source language is in `_LATIN_SCRIPT_LANGS` (canonical_form
          is already English-readable; no derivation needed),
        - all input fields are NULL / empty,
        - the rule output didn't pass `_looks_english_readable`.
    """
    if language in _LATIN_SCRIPT_LANGS:
        return None

    if transliteration:
        normalized_key = transliteration.strip().lower()
        if normalized_key in KNOWN_FORM_OVERRIDES:
            return KNOWN_FORM_OVERRIDES[normalized_key]
        stripped = _strip_transliteration(transliteration, language=language)
        if _looks_english_readable(stripped):
            return stripped

    if pronunciation_ipa:
        return _ipa_to_english_fallback(pronunciation_ipa, language=language)

    return None


# Languages targeted by the english_shaped derivation. Latin-script
# source languages would yield None anyway (the function short-circuits
# on Latin-script input), so we filter the candidate SELECT to the
# wave-2 non-Latin set to avoid sweeping millions of OE / latin / etc.
# rows that wouldn't yield anything. Mirror of cli.py's existing tuple
# of the same name; the CLI command imports from here in Phase 2.
PHASE2A_NON_LATIN_LANGS: tuple[str, ...] = (
    "he",
    "ar",
    "fa",
    "sa",
    "akk",
    "egy",
    "arc",
    "pal",
    "hbo",
    "peo",
    "fa-cls",
    "xpr",
    "syc",
    "cop",
    "axm",
    "pra",
    "pi",
)


def derive_english_shaped_all(
    db,  # LexiconDB — string-forward-ref via untyped to avoid circular
    *,
    apply: bool = True,
    languages: tuple[str, ...] | None = None,
    reshape: bool = False,
) -> dict[str, int | dict[str, int]]:
    """Uniform L3 wrapper for the ``derive-english-shaped`` pass —
    walks etymon rows whose ``english_shaped`` is NULL (or all rows
    when ``reshape=True``) and UPDATEs the row when
    :func:`derive_english_shaped` produces a non-None result.

    Wyrd-hidb Phase 2 plumbs this into ``run_full_enrichment``.

    ``languages``: restrict to a tuple of source-language codes.
    Defaults to :data:`PHASE2A_NON_LATIN_LANGS`. Passing ``("he",)``
    etc. lets the CLI's ``--language`` filter reuse this wrapper.

    Dry-run (``apply=False``) walks candidates but skips the UPDATE.
    """
    if languages is None:
        languages = PHASE2A_NON_LATIN_LANGS

    placeholders = ",".join("?" * len(languages))
    where = f"language IN ({placeholders})"
    if not reshape:
        where += " AND english_shaped IS NULL"
    rows = db.conn.execute(
        f"""SELECT id, canonical_form, language,
                   transliteration, pronunciation_ipa
              FROM etymon
             WHERE {where}
             ORDER BY language, id""",  # noqa: S608 — fixed template, parameterized
        languages,
    ).fetchall()

    written = 0
    skipped_no_input = 0
    by_language: dict[str, int] = {}
    for row in rows:
        shaped = derive_english_shaped(
            canonical_form=row["canonical_form"],
            language=row["language"],
            transliteration=row["transliteration"],
            pronunciation_ipa=row["pronunciation_ipa"],
        )
        if shaped is None:
            skipped_no_input += 1
            continue
        written += 1
        by_language[row["language"]] = by_language.get(row["language"], 0) + 1
        if apply:
            db.conn.execute(
                "UPDATE etymon SET english_shaped = ? WHERE id = ?",
                (shaped, row["id"]),
            )
    if apply:
        db.commit()

    return {
        "applied": int(apply),
        "candidates": len(rows),
        "written": written,
        "skipped_no_input": skipped_no_input,
        "by_language": by_language,
    }
