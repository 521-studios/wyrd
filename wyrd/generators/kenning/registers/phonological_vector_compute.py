"""Compute a PhonologicalVector from an IPA string (wyrd-kq7w.1).

The vector schema lives at :mod:`wyrd.generators.kenning.vectors.schemas`
(``PhonologicalVector``). This module owns the IPA → vector computation:
parse the IPA string into phonemes, classify each phoneme on the relevant
phonetic axes, and aggregate into the 14 named dimensions.

Pure-python implementation — no external IPA library dependency (panphon /
epitran would inflate the Lambda cold-start budget for a one-shot
enrichment computation). The phoneme inventory is a hand-curated table
derived from the IPA Handbook (1999) + Wells' Pronouncing Dictionary +
Hayes' "Introductory Phonology" (2009) chapter on phonological features.

Calibration anchors (per the kq7w.1 ticket's calibration plan):
  - Whissell's "Dictionary of Affect in Language" phonemic-emotion
    scores: angular/abrupt phonemes (k, t, stop clusters) drive low
    ``soft_consonants`` + high ``cluster_density``; sonorant phonemes
    (l, m, w, ʃ) drive high ``soft_consonants``.
  - Hinton/Nichols/Ohala (1994) "Sound Symbolism": the bouba/kiki
    effect maps to softness vs angularity → roughly soft_consonants
    vs cluster_density.
  - Ohala (1984) "Frequency code": high F2 (front vowels) ↔ small,
    light; low F2 (back vowels) ↔ large, heavy.

Deterministic: same (canonical_form, ipa) tuple always produces the
same vector. Idempotent — re-running the enrichment pass on an
unchanged etymon row yields a byte-identical
``phonological_vector`` JSON column.

Fallback chain when IPA is sparse:
  1. If ``ipa`` is non-empty, parse it (preferred path).
  2. If ``ipa`` is empty but ``canonical_form`` is Latin-script,
     approximate via the orthography (one-letter ≈ one-phoneme,
     vowel-letters → vowels, consonant-letters → consonants).
  3. If ``canonical_form`` is non-Latin script with no IPA, return
     an empty vector (all-zero) so scoring falls back to the
     dimension-mean defaults. The corpus has <2% of etymons in this
     state and they're predominantly newly-mined non-Latin rows
     whose IPA mining is a separate pipeline.
"""

from __future__ import annotations

import json
import unicodedata

from wyrd.generators.kenning.vectors.schemas import PhonologicalVector

# ---- IPA phoneme inventory ------------------------------------------------
#
# The classifier is set-based: each phoneme symbol maps into 0+ feature
# sets. A single phoneme can be in multiple sets (e.g. ``ʃ`` is both
# fricative and sibilant and palatal). The aggregation pass counts
# set memberships and normalizes by the relevant denominator (phoneme
# count, consonant count, syllable count) per the dimensions.

# Multi-character IPA tokens that should parse as ONE phoneme. Matched
# greedily before single-char tokens. Affricates + aspirated stops are
# the main shapes (``tʃ`` not ``t`` + ``ʃ``).
_MULTI_CHAR_IPA: tuple[str, ...] = (
    # affricates (treat as single phoneme — they ARE single phonemes in
    # most languages' inventories, and the stop/continuant ratio reads
    # better when they don't double-count as both)
    "tʃ",
    "dʒ",
    "ts",
    "dz",
    "tɕ",
    "dʑ",
    "ʈʂ",
    "ɖʐ",
    "tθ",
    "dð",
    # aspirated voiceless (Sanskrit + Tibetan + Korean + some Germanic)
    "pʰ",
    "tʰ",
    "kʰ",
    "cʰ",
    "qʰ",
    "ʈʰ",
    "tsʰ",
    "tʃʰ",
    "tɕʰ",
    # palatalized (Russian + Irish + Welsh-y)
    "tʲ",
    "dʲ",
    "sʲ",
    "lʲ",
    "rʲ",
    "nʲ",
    "pʲ",
    "bʲ",
    "fʲ",
    "vʲ",
    # labialized (Welsh kw/gw + some Bantu)
    "kʷ",
    "gʷ",
    "qʷ",
    # pharyngealized (Arabic emphatic)
    "tˤ",
    "dˤ",
    "sˤ",
    "ðˤ",
    "ɫˤ",
    # ejective (Caucasian, some Native American)
    "kʼ",
    "tʼ",
    "pʼ",
    "qʼ",
    "tsʼ",
    # diphthongs we want to count as 1 syllable. Heuristic: vowel +
    # glide / vowel pair. The syllable counter uses this to avoid
    # double-counting diphthong nuclei. Covers both English-style
    # (aɪ/eɪ/aʊ/oʊ — IPA narrow) and the bare-glide forms
    # (ai/ei/au/oi/ou — Hawaiian, Italian, Latin, French) so the
    # syllable count tracks linguistic intuition across transcription
    # conventions.
    "aɪ",
    "eɪ",
    "ɔɪ",
    "aʊ",
    "oʊ",
    "ɪə",
    "ʊə",
    "eə",
    "ai",
    "ei",
    "au",
    "oi",
    "ou",
    "iu",
    "ie",
    "uo",
)

# Stress markers + secondary stress + length marks — strip during parse,
# they don't carry a phonological-feature value. ˈ (primary), ˌ (secondary),
# ː (length), ˑ (half-length), · (syllable break), . (syllable break ASCII).
_STRESS_AND_LENGTH = "ˈˌːˑ·."

# Vowel features. Each entry: (F1_signed, F2_signed) where F1_signed
# is ``vowel_height`` ([-1=close, +1=open]) and F2_signed is
# ``vowel_backness`` ([+1=front, -1=back]). Tracks IPA's vowel chart.
_VOWELS: dict[str, tuple[float, float]] = {
    # close
    "i": (-1.0, +1.0),
    "y": (-1.0, +1.0),
    "ɨ": (-1.0, 0.0),
    "ʉ": (-1.0, 0.0),
    "ɯ": (-1.0, -1.0),
    "u": (-1.0, -1.0),
    # near-close
    "ɪ": (-0.7, +0.7),
    "ʏ": (-0.7, +0.7),
    "ʊ": (-0.7, -0.7),
    # close-mid
    "e": (-0.3, +1.0),
    "ø": (-0.3, +1.0),
    "ɘ": (-0.3, 0.0),
    "ɵ": (-0.3, 0.0),
    "ɤ": (-0.3, -1.0),
    "o": (-0.3, -1.0),
    # mid (schwa)
    "ə": (0.0, 0.0),
    # open-mid
    "ɛ": (+0.3, +1.0),
    "œ": (+0.3, +1.0),
    "ɜ": (+0.3, 0.0),
    "ɞ": (+0.3, 0.0),
    "ʌ": (+0.3, -1.0),
    "ɔ": (+0.3, -1.0),
    # near-open
    "æ": (+0.7, +1.0),
    "ɐ": (+0.7, 0.0),
    # open
    "a": (+1.0, +1.0),
    "ɶ": (+1.0, +1.0),
    "ɑ": (+1.0, -1.0),
    "ɒ": (+1.0, -1.0),
}

# wyrd-mkry: tense vs lax vowel classification. Whissell 2000 found
# tense/lax matters MORE than front/back for valence — but the
# mapping is NON-MONOTONIC: /iː/ is Gentle while /ɪ/ is Harsh, and
# /ɔ/ is Gentle while /uː/ is Harsh. The dimension surfaces the
# tense/lax signal (length marks are stripped upstream so iː → i)
# and lets the register-effect catalog decide direction per-effect
# rather than baking in a global "tense = Gentle" assumption.
#
# Tense set (closed + peripheral monophthongs from the _VOWELS chart):
#   i, u, e, o, y, ø, ɨ, ʉ, ɯ, ɵ, ɤ, ɶ, ɑ (13 entries).
# Lax set (near-close + open-mid + near-open centralized vowels):
#   ɪ, ʊ, ɛ, ɔ, æ, ʌ, ʏ, ɜ, ɞ, ɐ (10 entries).
# Schwa (ə) is intentionally in neither set — contributes 0 to the
# tenseness dimension, matching its corpus-mean neutrality on
# vowel_height / vowel_backness.
_TENSE_VOWELS = set("iueoyøɵɤɨʉɯɶɑ")
_LAX_VOWELS = set("ɪʊɛɔæʌʏɜɞɐ")

# Consonant feature classifications (sets — a phoneme can appear in
# multiple). Derived from the IPA pulmonic consonant chart with
# adjustments for the loan-relationship sources the corpus covers.

# Stops (plosives): full closure + release. Includes ejectives + the
# affricates list above for the stop_vs_continuant ratio. Glottal stop
# ʔ included — it's a stop by manner.
_STOPS = set("pbtdʈɖcɟkgqɢʔ") | {
    "tʃ",
    "dʒ",
    "ts",
    "dz",
    "tɕ",
    "dʑ",
    "ʈʂ",
    "ɖʐ",
    "tθ",
    "dð",
    "pʰ",
    "tʰ",
    "kʰ",
    "cʰ",
    "qʰ",
    "ʈʰ",
    "tsʰ",
    "tʃʰ",
    "tɕʰ",
    "tʲ",
    "dʲ",
    "pʲ",
    "bʲ",
    "kʷ",
    "gʷ",
    "qʷ",
    "tˤ",
    "dˤ",
    "kʼ",
    "tʼ",
    "pʼ",
    "qʼ",
    "tsʼ",
}

# Nasals: m, n, ɲ (palatal), ŋ (velar), ɴ (uvular), ɱ (labiodental),
# ɳ (retroflex). Plus palatalized nasals.
_NASALS = set("mnɲŋɴɱɳ") | {"nʲ"}

# wyrd-119p: laterals (the gentler half — l, ɭ, ʎ, ʟ, ɬ, ɮ + plus
# the palatalized + pharyngealized variants). Whissell 2000 rates the
# lateral / nasal sonorants firmly Gentle.
_LATERALS = set("lɭʎʟɬɮ") | {"lʲ", "ɫˤ"}

# wyrd-119p: rhotics (the harsh half — r, ɾ, ɽ, ɹ, ɻ, ʀ, ʁ + the
# palatalized variant). Whissell 2000 rates /r/ firmly Harsh.
# Kept disjoint from _LATERALS so the new liquid_l_m_n / rhotic_r
# dimensions partition the historical "liquid" class cleanly.
_RHOTICS = set("rɾɽɹɻʀʁ") | {"rʲ"}

# Liquids: laterals + rhotics. Welsh ɬ + ɮ are voiceless lateral
# fricatives; we count them in fricatives + liquids both (they ARE
# liquids by manner, fricatives by phonation). Kept as the union of
# _LATERALS + _RHOTICS so existing consumers (soft_consonants share,
# _CONSONANTS denominator) behave unchanged; the new wyrd-119p
# dimensions read the two halves directly.
_LIQUIDS = _LATERALS | _RHOTICS

# Fricatives: turbulent airflow. Subdivides into sibilants below.
# Includes alveolo-palatal ɕ / ʑ (Mandarin pinyin x/j, Polish ś/ź,
# Japanese しゃ, Serbo-Croatian ć/đ) so they propagate via the union
# into ``_CONSONANTS`` — round-1 reviewer caught the omission.
_FRICATIVES = set("fvθðszʃʒçʝxɣχʁħʕhɦɸβʂʐɕʑɬɮ") | {"sʲ", "fʲ", "vʲ", "sˤ", "ðˤ"}

# Sibilants: high-frequency fricatives. s/z + post-alveolar +
# retroflex sibilants + the sibilant component of affricates.
_SIBILANTS = set("szʃʒʂʐɕʑ") | {
    "tʃ",
    "dʒ",
    "ts",
    "dz",
    "tɕ",
    "dʑ",
    "ʈʂ",
    "ɖʐ",
    "tʃʰ",
    "tsʰ",
    "tɕʰ",
    "tsʼ",
    "sʲ",
    "sˤ",
}

# Approximants + glides — soft consonants by sound-symbolism conventions.
_APPROXIMANTS = set("wjɥɰʋ")

# Palatal phonemes (including palatalized variants from the multi-char
# table). Used for the palatalization dimension.
_PALATALS = set("ɲʎçʝjɕʑcɟ") | {
    "tʃ",
    "dʒ",
    "tɕ",
    "dʑ",
    "tʃʰ",
    "tɕʰ",
    "tʲ",
    "dʲ",
    "sʲ",
    "lʲ",
    "rʲ",
    "nʲ",
    "pʲ",
    "bʲ",
    "fʲ",
    "vʲ",
    "cʰ",
}

# Retroflex phonemes. Hard consonants common in Dravidian, Sanskrit,
# Norwegian, Mandarin pinyin r-, some Australian languages.
_RETROFLEXES = set("ʈɖɳɽɻʂʐɭ") | {"ʈʂ", "ɖʐ", "ʈʰ"}

# Pharyngeal + emphatic phonemes. Arabic ħ/ʕ + pharyngealized series.
_PHARYNGEALS = set("ħʕʢʡχ") | {"tˤ", "dˤ", "sˤ", "ðˤ", "ɫˤ", "qˤ", "q"}

# Aspirated voiceless stops (multi-char tokens from the table above).
_ASPIRATED_VOICELESS = {"pʰ", "tʰ", "kʰ", "cʰ", "qʰ", "ʈʰ", "tsʰ", "tʃʰ", "tɕʰ"}

# Soft consonant superset for the soft_consonants dimension =
# fricatives ∪ liquids ∪ nasals ∪ approximants. The
# sound-symbolism literature (Köhler bouba, Hinton/Nichols/Ohala
# 1994) treats these as the "soft" end of the consonant continuum.
_SOFT_CONSONANTS = _FRICATIVES | _LIQUIDS | _NASALS | _APPROXIMANTS

# All consonants. Used as the denominator for share metrics. The
# affricate tokens are stops by manner but count once each in this
# set (matches the parse output's one-token-per-phoneme convention).
_CONSONANTS = (
    _STOPS | _NASALS | _LIQUIDS | _FRICATIVES | _APPROXIMANTS | _RETROFLEXES | _PHARYNGEALS
)


# ---- IPA parsing ----------------------------------------------------------


def _strip_brackets_and_stress(ipa: str) -> str:
    """Strip surrounding ``/.../`` or ``[...]`` IPA brackets and
    stress / length / boundary marks. Keeps the bare phoneme stream.

    Some sources double-bracket their IPA (``[ˈkʰæt]``) and some don't.
    Most also include stress + length marks the dimensions don't read.
    Strip those uniformly so the per-char classification step starts
    from a clean stream.
    """
    s = ipa.strip()
    # outer brackets — match ``/.../`` or ``[...]`` only, balanced
    if (s.startswith("/") and s.endswith("/")) or (s.startswith("[") and s.endswith("]")):
        s = s[1:-1].strip()
    # strip stress + length + syllable-break marks
    for ch in _STRESS_AND_LENGTH:
        s = s.replace(ch, "")
    # NFC-normalize so combining marks (tilde, diacritics) bind to
    # their base letters rather than appearing as separate codepoints.
    # The classifier expects NFC-form input.
    s = unicodedata.normalize("NFC", s)
    return s


def parse_ipa(ipa: str) -> list[str]:
    """Tokenize an IPA string into a list of phoneme symbols.

    Greedy multi-character match first (affricates, aspirated stops,
    palatalized / pharyngealized / labialized series, common
    diphthongs), then single-char fallback. Unknown characters fall
    through unchanged — the classifier ignores them in dimension
    aggregation, which keeps the parser robust against IPA
    transcription quirks (zero-width joiners, source-specific
    diacritics) without raising.
    """
    s = _strip_brackets_and_stress(ipa)
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        # try multi-char (longest first by sorting the multi-char table)
        matched = False
        for tok in sorted(_MULTI_CHAR_IPA, key=len, reverse=True):
            if s.startswith(tok, i):
                out.append(tok)
                i += len(tok)
                matched = True
                break
        if matched:
            continue
        out.append(s[i])
        i += 1
    return out


def _is_vowel(token: str) -> bool:
    """Vowel-classifier used by the syllable counter + share metrics.

    Single-char in _VOWELS, plus diphthong tokens (which are 1
    syllable with a vowel nucleus). Tone letters / non-vowel
    diacritics fall through as False.
    """
    if token in _VOWELS:
        return True
    # diphthongs — first char is a vowel, second is a glide/vowel
    return len(token) == 2 and token[0] in _VOWELS


def _is_consonant(token: str) -> bool:
    return token in _CONSONANTS


# ---- per-dimension scoring ------------------------------------------------


def _cluster_density(phonemes: list[str]) -> float:
    """CC- consonant-cluster phoneme ratio.

    Per the ticket spec: ``CC- count / phoneme count``. The
    operational reading: count consonant phonemes that are immediately
    preceded OR followed by another consonant phoneme (so the C in CCV
    OR VCC counts; a lone VCV consonant doesn't). Normalize by total
    phoneme count. Range [0, 1].

    Languages with rich onset/coda clusters (Polish, Russian, Welsh)
    read high; vowel-heavy languages (Italian, Hawaiian) read low.
    """
    if not phonemes:
        return 0.0
    n = len(phonemes)
    in_cluster = 0
    for i, tok in enumerate(phonemes):
        if not _is_consonant(tok):
            continue
        prev_is_c = i > 0 and _is_consonant(phonemes[i - 1])
        next_is_c = i < n - 1 and _is_consonant(phonemes[i + 1])
        if prev_is_c or next_is_c:
            in_cluster += 1
    return in_cluster / n


def _final_fortition(phonemes: list[str]) -> float:
    """Rate of word-final stops (fortis = strong/abrupt).

    Single-pass: 1.0 if the last phoneme is a stop, 0.0 otherwise.
    Vector-aggregate across a corpus this reads as "what fraction of
    forms end in a stop". Languages with strong final-stop bias
    (Germanic, Slavic) score high; vowel-final languages (Italian,
    Japanese) score low.
    """
    if not phonemes:
        return 0.0
    return 1.0 if phonemes[-1] in _STOPS else 0.0


def _final_cluster_rate(phonemes: list[str]) -> float:
    """Rate of word-final consonant clusters (e.g. English -lks, -mps).

    1.0 if the form ends in CC+ (two or more consonants), 0.0 otherwise.
    Languages with rich coda clusters (English, German, Polish) read
    high; CV-shape languages (Italian, Japanese, Hawaiian) read 0.
    """
    if len(phonemes) < 2:
        return 0.0
    return 1.0 if (_is_consonant(phonemes[-1]) and _is_consonant(phonemes[-2])) else 0.0


def _vowel_final_bias(phonemes: list[str]) -> float:
    """Rate of vowel-final forms.

    1.0 if the form ends in a vowel, 0.0 if in a consonant. Inverse
    of final-stop measures: Italian / Japanese / Hawaiian read ~1.0;
    Germanic / Slavic read low.
    """
    if not phonemes:
        return 0.0
    return 1.0 if _is_vowel(phonemes[-1]) else 0.0


def _soft_consonants_share(phonemes: list[str]) -> float:
    """Proportion of consonants that are 'soft' (fricative + liquid +
    nasal + approximant).

    The 'soft' superset is the sound-symbolism literature's
    melodic/sonorant cluster (vs the 'hard' stop/obstruent cluster).
    Range [0, 1] of consonant inventory. Returns 0 when there are no
    consonants (avoids divide-by-zero).

    wyrd-119p note: lumps Gentle laterals/nasals with Harsh rhotics;
    use ``_liquid_l_m_n_share`` + ``_rhotic_r_share`` for finer
    Whissell-accurate distinctions. Kept for back-compat — existing
    stored vectors have this dimension populated and the deprecation
    runs through register-effect catalog evolution, not a schema
    drop.
    """
    consonants = [p for p in phonemes if _is_consonant(p)]
    if not consonants:
        return 0.0
    soft = sum(1 for p in consonants if p in _SOFT_CONSONANTS)
    return soft / len(consonants)


def _liquid_l_m_n_share(phonemes: list[str]) -> float:
    """wyrd-119p: share of phonemes that are lateral liquids OR nasals.
    The Gentle half of the historical soft_consonants compound, per
    Whissell 2000 ('l as sweet'). Range [0, 1] of total phoneme count
    (matches the share-in-set shape used by palatalization /
    sibilance / retroflexion etc.)."""
    if not phonemes:
        return 0.0
    feature_set = _LATERALS | _NASALS
    return sum(1 for p in phonemes if p in feature_set) / len(phonemes)


def _rhotic_r_share(phonemes: list[str]) -> float:
    """wyrd-119p: share of phonemes that are rhotics. The Harsh half
    of the historical soft_consonants compound, per Whissell 2000
    ('r as tough'). Range [0, 1] of total phoneme count."""
    if not phonemes:
        return 0.0
    return sum(1 for p in phonemes if p in _RHOTICS) / len(phonemes)


def _vowel_tenseness(phonemes: list[str]) -> float:
    """wyrd-mkry: mean (in _TENSE_VOWELS = +1, in _LAX_VOWELS = -1,
    other = 0) across monophthong vowel phonemes. The result is a
    [-1, +1] tenseness score; mapping that score to a register
    direction (Gentle vs Harsh) is the catalog's job — Whissell 2000
    shows the partition isn't monotonic.

    Filters to ``p in _VOWELS`` (single-char monophthong dict) for
    consistency with ``_vowel_height`` / ``_vowel_backness`` — both
    of those exclude diphthong tokens like 'ie' / 'au' from their
    F1/F2 averages because the multi-char tokens have no single
    formant-position. Same choice here: diphthongs contribute
    nothing to the tense/lax score; only clean monophthongs do.

    Returns 0.0 when no monophthongs are present, when every vowel is
    neutral (schwa-only), or when tense and lax exactly balance —
    matches the corpus-mean default for the [-1, +1] signed-feature
    family. Vowels not in either classification set (e.g. schwa,
    less-common IPA glyphs) contribute 0 individually but still
    count in the denominator so a single schwa doesn't inflate the
    magnitude of a paired tense / lax vowel's signal."""
    monophthongs = [p for p in phonemes if p in _VOWELS]
    if not monophthongs:
        return 0.0
    total = 0.0
    for v in monophthongs:
        if v in _TENSE_VOWELS:
            total += 1.0
        elif v in _LAX_VOWELS:
            total -= 1.0
    return total / len(monophthongs)


def _count_syllables(phonemes: list[str]) -> int:
    """Estimate syllable count = vowel nucleus count.

    Each vowel (or diphthong) token = one syllable nucleus. Adjacent
    vowels in hiatus (rare) count as separate syllables. Returns 0
    when there are no vowels (forms like 'Mladenovac' onset clusters
    don't show up in the corpus but we want a sensible fallback).
    """
    return sum(1 for p in phonemes if _is_vowel(p))


def _polysyllabic_bias(canonical_form: str, phonemes: list[str]) -> float:
    """Syllable count normalized by form length (chars).

    Per the ticket: ``syllable count / form length``. Range [0, 1]
    in practice — most forms have syllable count ≤ form length.
    Languages with long polysyllabic words (Finnish, Hungarian,
    Sanskrit) score high; monosyllabic-bias languages (English
    monosyllables, Mandarin) score low.

    Form length is the orthographic length (canonical_form), not the
    IPA length, since IPA includes diacritics + multi-char tokens that
    don't correspond to orthographic length.
    """
    if not canonical_form:
        return 0.0
    syl = _count_syllables(phonemes)
    return min(1.0, syl / max(1, len(canonical_form)))


def _share_in_set(phonemes: list[str], feature_set: set[str]) -> float:
    """Proportion of phonemes that are members of ``feature_set``.

    The standard rate-metric shape: count phonemes in the set,
    divide by total phoneme count. Range [0, 1]. Used by the
    palatalization / sibilance / retroflexion / pharyngeal /
    aspirated-voiceless dimensions.
    """
    if not phonemes:
        return 0.0
    return sum(1 for p in phonemes if p in feature_set) / len(phonemes)


def _vowel_height(phonemes: list[str]) -> float:
    """Mean F1 across vowel phonemes, signed [-1=close, +1=open].

    The Ohala frequency-code reading: low F1 (close vowels: i, u,
    ɨ) is "small, light, bright"; high F1 (open vowels: a, ɑ) is
    "large, dark, heavy". Languages dominated by close vowels
    (Quechua i/u) read negative; open-vowel-dominated languages
    (Hawaiian a-heavy) read positive.

    Returns 0.0 when no vowels (centred default).
    """
    vowels = [p for p in phonemes if p in _VOWELS]
    if not vowels:
        return 0.0
    return sum(_VOWELS[p][0] for p in vowels) / len(vowels)


def _vowel_backness(phonemes: list[str]) -> float:
    """Mean F2 across vowel phonemes, signed [+1=front, -1=back].

    Front vowels (i, e, ɛ, æ) high F2 → "small, bright, sharp"
    (kiki side of the bouba/kiki effect). Back vowels (u, o, ɔ,
    ɑ) low F2 → "large, dark, round" (bouba side).

    Returns 0.0 when no vowels.
    """
    vowels = [p for p in phonemes if p in _VOWELS]
    if not vowels:
        return 0.0
    return sum(_VOWELS[p][1] for p in vowels) / len(vowels)


def _stop_vs_continuant(phonemes: list[str]) -> float:
    """Signed ratio of stops vs continuants (fricatives + approximants
    + liquids + nasals).

    Positive = stop-dominated (English voiceless stops cluster, Welsh
    /k g/ heavy); negative = continuant-dominated (Polynesian / Italian
    melodic profile). Range [-1, +1] — computed as
    ``(stops - continuants) / (stops + continuants)``. Returns 0.0
    when there are no consonants.
    """
    consonants = [p for p in phonemes if _is_consonant(p)]
    if not consonants:
        return 0.0
    stops = sum(1 for p in consonants if p in _STOPS)
    continuants = sum(
        1
        for p in consonants
        if p in _FRICATIVES or p in _APPROXIMANTS or p in _LIQUIDS or p in _NASALS
    )
    total = stops + continuants
    if total == 0:
        return 0.0
    return (stops - continuants) / total


# ---- orthographic fallback (when IPA is empty) ---------------------------

# Crude letter-class fallback for IPA-less Latin-script forms. Captures
# the broad shape without IPA precision — close vs open vowels, stops
# vs continuants. The fallback's vector is necessarily coarser than
# the IPA path's (no diphthong recognition, no palatalization, no
# retroflexion / pharyngeal / aspiration); those dimensions read 0
# in the fallback path which is what we want — the corpus mean for
# those features IS 0 for IPA-less rows by construction.

_FALLBACK_VOWELS = set("aeiouyAEIOUY")
_FALLBACK_STOPS = set("pbtdkgqPBTDKGQ")
_FALLBACK_SIBILANTS = set("szxSZX")
_FALLBACK_LIQUIDS = set("lrLR")
_FALLBACK_NASALS = set("mnMN")
_FALLBACK_FRICATIVES = set("fvszhSHFVSZH")


def _orthographic_vector(canonical_form: str) -> PhonologicalVector:
    """Compute a PhonologicalVector from canonical_form alone.

    Coarser than the IPA path — only the dimensions that can be
    estimated from Latin-script orthography get populated. Used when
    pronunciation_ipa is empty. The dimensions we cannot reliably
    estimate from orthography (palatalization, retroflexion,
    pharyngeal, aspirated_voiceless) read 0.0, which is the
    corpus-centered default for those features.
    """
    if not canonical_form:
        return PhonologicalVector()

    # Strip non-letters; the remaining letters approximate phonemes
    # one-letter-one-phoneme. Coarse but it's the fallback.
    letters = [c for c in canonical_form if c.isalpha()]
    if not letters:
        return PhonologicalVector()
    n = len(letters)

    vowels = [c for c in letters if c in _FALLBACK_VOWELS]
    consonants = [c for c in letters if c not in _FALLBACK_VOWELS]
    nc = len(consonants)

    # cluster density via consecutive-consonant pairs
    in_cluster = 0
    for i, c in enumerate(letters):
        if c in _FALLBACK_VOWELS:
            continue
        prev_c = i > 0 and letters[i - 1] not in _FALLBACK_VOWELS
        next_c = i < n - 1 and letters[i + 1] not in _FALLBACK_VOWELS
        if prev_c or next_c:
            in_cluster += 1

    last = letters[-1]
    second_last = letters[-2] if n >= 2 else ""

    soft_share = 0.0
    if nc:
        soft = sum(
            1
            for c in consonants
            if c in _FALLBACK_FRICATIVES or c in _FALLBACK_LIQUIDS or c in _FALLBACK_NASALS
        )
        soft_share = soft / nc

    sib_share = sum(1 for c in letters if c in _FALLBACK_SIBILANTS) / n

    stops = sum(1 for c in consonants if c in _FALLBACK_STOPS)
    continuants = sum(
        1
        for c in consonants
        if c in _FALLBACK_FRICATIVES or c in _FALLBACK_LIQUIDS or c in _FALLBACK_NASALS
    )
    stop_cont = 0.0
    if (stops + continuants) > 0:
        stop_cont = (stops - continuants) / (stops + continuants)

    # wyrd-119p: orthographic l/m/n vs r partition. Coarse — the
    # fallback path can't distinguish palatalized lʲ / rʲ etc. — but
    # the IPA path covers that fidelity when pronunciation is present.
    liquid_lmn = sum(1 for c in letters if c in "lmnLMN") / n
    rhotic_share = sum(1 for c in letters if c in "rR") / n

    return PhonologicalVector(
        cluster_density=in_cluster / n,
        final_fortition=1.0 if last in _FALLBACK_STOPS else 0.0,
        final_cluster_rate=1.0
        if (second_last and last not in _FALLBACK_VOWELS and second_last not in _FALLBACK_VOWELS)
        else 0.0,
        vowel_final_bias=1.0 if last in _FALLBACK_VOWELS else 0.0,
        soft_consonants=soft_share,
        polysyllabic_bias=min(1.0, len(vowels) / max(1, len(canonical_form))),
        sibilance=sib_share,
        stop_vs_continuant=stop_cont,
        liquid_l_m_n=liquid_lmn,
        rhotic_r=rhotic_share,
        # The remaining dimensions read 0 in the fallback path — we
        # can't reliably classify palatalization / retroflexion /
        # pharyngeal / aspirated_voiceless / vowel_height /
        # vowel_backness / vowel_tenseness from orthography alone.
        # Latin-script orthography doesn't preserve tense / lax
        # distinctions (English "bit" vs "beet" both contain "i").
    )


# ---- public API -----------------------------------------------------------


def compute_phonological_vector(
    canonical_form: str,
    ipa: str | None,
) -> PhonologicalVector:
    """Compute the PhonologicalVector for an etymon.

    Preferred path: IPA-driven (when ``ipa`` is non-empty). Fallback:
    orthographic approximation from ``canonical_form`` alone.

    Deterministic + idempotent — same input always produces the same
    vector. Safe to re-run the enrichment pass without invalidating
    downstream caches as long as the IPA + canonical_form columns
    haven't changed.
    """
    if not ipa or not ipa.strip():
        return _orthographic_vector(canonical_form)

    phonemes = parse_ipa(ipa)
    if not phonemes:
        return _orthographic_vector(canonical_form)

    return PhonologicalVector(
        cluster_density=_cluster_density(phonemes),
        final_fortition=_final_fortition(phonemes),
        final_cluster_rate=_final_cluster_rate(phonemes),
        vowel_final_bias=_vowel_final_bias(phonemes),
        soft_consonants=_soft_consonants_share(phonemes),
        polysyllabic_bias=_polysyllabic_bias(canonical_form, phonemes),
        palatalization=_share_in_set(phonemes, _PALATALS),
        sibilance=_share_in_set(phonemes, _SIBILANTS),
        retroflexion=_share_in_set(phonemes, _RETROFLEXES),
        pharyngeal=_share_in_set(phonemes, _PHARYNGEALS),
        vowel_height=_vowel_height(phonemes),
        vowel_backness=_vowel_backness(phonemes),
        stop_vs_continuant=_stop_vs_continuant(phonemes),
        aspirated_voiceless=_share_in_set(phonemes, _ASPIRATED_VOICELESS),
        liquid_l_m_n=_liquid_l_m_n_share(phonemes),
        rhotic_r=_rhotic_r_share(phonemes),
        vowel_tenseness=_vowel_tenseness(phonemes),
    )


def vector_to_json(v: PhonologicalVector) -> str:
    """Serialize a PhonologicalVector to the JSON shape stored in
    ``etymon.phonological_vector``.

    Dimensions are rounded to 4 decimal places for storage stability
    (float-formatting drift between platforms shouldn't surface as a
    no-op diff after re-running the enrichment pass).
    """
    payload: dict[str, float | dict[str, float]] = {
        "cluster_density": round(v.cluster_density, 4),
        "final_fortition": round(v.final_fortition, 4),
        "final_cluster_rate": round(v.final_cluster_rate, 4),
        "vowel_final_bias": round(v.vowel_final_bias, 4),
        "soft_consonants": round(v.soft_consonants, 4),
        "polysyllabic_bias": round(v.polysyllabic_bias, 4),
        "palatalization": round(v.palatalization, 4),
        "sibilance": round(v.sibilance, 4),
        "retroflexion": round(v.retroflexion, 4),
        "pharyngeal": round(v.pharyngeal, 4),
        "vowel_height": round(v.vowel_height, 4),
        "vowel_backness": round(v.vowel_backness, 4),
        "stop_vs_continuant": round(v.stop_vs_continuant, 4),
        "aspirated_voiceless": round(v.aspirated_voiceless, 4),
        # wyrd-119p + wyrd-mkry new dimensions. Older stored blobs
        # missing these deserialize with 0.0 defaults — that
        # tolerance is the back-compat seam until the enrichment pass
        # is re-run against the corpus.
        "liquid_l_m_n": round(v.liquid_l_m_n, 4),
        "rhotic_r": round(v.rhotic_r, 4),
        "vowel_tenseness": round(v.vowel_tenseness, 4),
    }
    if v.extras:
        payload["extras"] = {k: round(val, 4) for k, val in v.extras.items()}
    return json.dumps(payload, sort_keys=True)
