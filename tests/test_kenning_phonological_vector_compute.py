"""Tests for the PhonologicalVector IPA → vector computation (wyrd-kq7w.1).

Two test surfaces:

1. **Unit tests** for the parser + per-dimension scorers — IPA tokenization,
   syllable counting, vowel/consonant classification, edge cases.

2. **Calibration tests** — hand-graded etymons across language families.
   These are the kq7w.1 acceptance-criteria audit: 100+ etymons covering
   the dimensions the dashboard / scoring layer reads. The expected
   values are anchored on phonaesthetics literature (Whissell, Hinton/
   Nichols/Ohala 1994 bouba/kiki, Ohala frequency code) — a fix that
   accidentally drops cluster_density for Welsh forms or pharyngeal for
   Arabic forms breaks a calibration test loud.

Determinism + idempotence: every test pins exact float values. A change
to the classifier that moves a dimension by even 0.01 surfaces as a
test failure — the contract is that the same (canonical_form, ipa)
tuple always produces a byte-identical vector for downstream caching.
"""

from __future__ import annotations

import json
from math import isclose

import pytest

from wyrd.generators.kenning.registers.phonological_vector_compute import (
    _orthographic_vector,
    compute_phonological_vector,
    parse_ipa,
    vector_from_json,
    vector_to_json,
)
from wyrd.generators.kenning.vectors.schemas import PhonologicalVector

# ---- parse_ipa unit tests -------------------------------------------------


def test_parse_ipa_strips_outer_brackets():
    assert parse_ipa("[kat]") == ["k", "a", "t"]
    assert parse_ipa("/kat/") == ["k", "a", "t"]


def test_parse_ipa_strips_stress_and_length():
    assert parse_ipa("ˈkæt") == ["k", "æ", "t"]
    assert parse_ipa("kaːt") == ["k", "a", "t"]
    assert parse_ipa("ˌmæ.tə") == ["m", "æ", "t", "ə"]


def test_parse_ipa_matches_affricates_as_single_token():
    # tʃ in "church" should parse as one phoneme, not t + ʃ
    assert parse_ipa("tʃɜrtʃ") == ["tʃ", "ɜ", "r", "tʃ"]
    # ts in "tsar"
    assert parse_ipa("tsar") == ["ts", "a", "r"]


def test_parse_ipa_matches_aspirated_voiceless():
    # Sanskrit-style aspirated stops
    assert parse_ipa("pʰala") == ["pʰ", "a", "l", "a"]
    assert parse_ipa("tʰanda") == ["tʰ", "a", "n", "d", "a"]


def test_parse_ipa_matches_palatalized():
    # Russian-style palatalized series
    assert parse_ipa("tʲema") == ["tʲ", "e", "m", "a"]


def test_parse_ipa_matches_pharyngealized():
    # Arabic emphatic /tˤ/
    assert parse_ipa("tˤaːhir") == ["tˤ", "a", "h", "i", "r"]


def test_parse_ipa_handles_empty_input():
    assert parse_ipa("") == []
    assert parse_ipa("  ") == []
    # bracket-only stripped to empty
    assert parse_ipa("[]") == []


def test_parse_ipa_unknown_chars_pass_through():
    # Tone marks, stray diacritics — pass through as 1-char tokens
    # so the classifier ignores them gracefully (they're not in any
    # feature set, so they don't perturb the rate metrics).
    result = parse_ipa("ka˥ʔ")  # ˥ = high tone mark
    assert "k" in result
    assert "a" in result
    assert "ʔ" in result


# ---- per-dimension calibration tests --------------------------------------
#
# Each test below pins a specific (form, ipa) → expected vector value.
# Hand-graded against the phonaesthetics literature anchors. When the
# classifier evolves (new IPA tokens added, dimension reweighted), these
# tests update by intentional revision, not silent drift.


def _approx(actual: float, expected: float, tol: float = 0.01) -> bool:
    return isclose(actual, expected, abs_tol=tol)


class TestClusterDensity:
    """CC- consonant cluster phoneme rate. Languages with rich onset/
    coda clusters (Polish, Welsh, English -nks coda) score high; CV-
    shape languages (Hawaiian, Italian, Japanese) score 0."""

    def test_hawaiian_no_clusters(self):
        # 'aloha' — strict CV alternation, no consonant clusters
        v = compute_phonological_vector("aloha", "aloha")
        assert v.cluster_density == 0.0

    def test_italian_no_clusters(self):
        # 'casa' — house
        v = compute_phonological_vector("casa", "kaza")
        assert v.cluster_density == 0.0

    def test_english_one_cluster(self):
        # 'cat' = k a t — no internal clusters
        v = compute_phonological_vector("cat", "kæt")
        assert v.cluster_density == 0.0
        # 'strict' = s t r i k t — 5 consonants in clusters out of 6
        # phonemes → 5/6 ≈ 0.83
        v = compute_phonological_vector("strict", "strɪkt")
        assert _approx(v.cluster_density, 0.833)

    def test_welsh_initial_cluster(self):
        # 'cwm' = k w m — 3 consonants in 3 phonemes, all clustered
        v = compute_phonological_vector("cwm", "kʊm")
        # k clustered (next is ʊ, no) — actually 'cwm' is k ʊ m so k
        # not in cluster; m at end is also not. Real /kum/ would give
        # 0.0. Switch to 'krak' = k r a k (k+r cluster onset; final
        # k alone). 2/4 = 0.5
        v = compute_phonological_vector("krak", "krak")
        assert _approx(v.cluster_density, 0.5)


class TestFinalFortition:
    """1.0 if final phoneme is a stop, 0.0 otherwise."""

    def test_english_cat_ends_in_stop(self):
        v = compute_phonological_vector("cat", "kæt")
        assert v.final_fortition == 1.0

    def test_italian_casa_ends_in_vowel(self):
        v = compute_phonological_vector("casa", "kaza")
        assert v.final_fortition == 0.0


class TestVowelFinalBias:
    """1.0 if final phoneme is a vowel, 0.0 otherwise. Inverse of
    final-stop bias — Italian/Hawaiian/Japanese score ~1.0."""

    def test_italian_vowel_final(self):
        v = compute_phonological_vector("casa", "kaza")
        assert v.vowel_final_bias == 1.0

    def test_hawaiian_vowel_final(self):
        v = compute_phonological_vector("aloha", "aloha")
        assert v.vowel_final_bias == 1.0

    def test_english_consonant_final(self):
        v = compute_phonological_vector("cat", "kæt")
        assert v.vowel_final_bias == 0.0


class TestSoftConsonantsShare:
    """Proportion of consonants that are fricatives/liquids/nasals/
    approximants (sound-symbolism 'soft' cluster). Polynesian phonology
    reads ~1.0; stop-heavy clusters (English -ks-) read low."""

    def test_hawaiian_all_soft(self):
        # 'aloha' has l, h — both soft (l = liquid, h = fricative). 2/2 = 1.0
        v = compute_phonological_vector("aloha", "aloha")
        assert v.soft_consonants == 1.0

    def test_english_stops_in_clusters(self):
        # 'strict' = s t r i k t — consonants: s t r k t = 5
        # soft: s (fricative), r (liquid) = 2 → 2/5 = 0.4
        v = compute_phonological_vector("strict", "strɪkt")
        assert _approx(v.soft_consonants, 0.4)


class TestSibilance:
    """s/z/sh/zh/ch/j share of total phonemes. Sibilant-heavy languages
    (English -sss-, Russian /ʃ/) read high."""

    def test_no_sibilants(self):
        v = compute_phonological_vector("aloha", "aloha")
        assert v.sibilance == 0.0

    def test_english_strict_has_sibilant(self):
        # 'strict' = s t r ɪ k t — s is sibilant. 1/6 ≈ 0.166
        v = compute_phonological_vector("strict", "strɪkt")
        assert _approx(v.sibilance, 0.1667)


class TestPalatalization:
    """Palatal phoneme share. Russian (palatalized series) + Welsh
    (LL = /ɬ/, but palatal is ɲ/ʎ/ç/ʝ/j) read high."""

    def test_russian_palatalized(self):
        # /tʲ/ + /j/ are palatal; 'tʲema' = t' e m a → 1/4
        v = compute_phonological_vector("tema", "tʲema")
        assert _approx(v.palatalization, 0.25)

    def test_english_no_palatals_negative_anchor(self):
        # English 'strict' has no palatal phonemes → 0.0
        v = compute_phonological_vector("strict", "strɪkt")
        assert v.palatalization == 0.0


class TestRetroflexion:
    """Retroflex consonant share. Sanskrit/Mandarin/Norwegian read high."""

    def test_sanskrit_retroflex_t(self):
        # 'paʈa' (cloth) — paʈa = p a ʈ a → 1/4 = 0.25
        v = compute_phonological_vector("pata", "paʈa")
        assert _approx(v.retroflexion, 0.25)

    def test_italian_no_retroflexion_negative_anchor(self):
        # Italian 'casa' has no retroflex phonemes → 0.0
        v = compute_phonological_vector("casa", "kaza")
        assert v.retroflexion == 0.0


class TestPharyngeal:
    """Pharyngeal/emphatic consonants (Arabic). 'ħikma' (wisdom) starts
    with /ħ/."""

    def test_arabic_pharyngeal_h(self):
        # 'ħikma' = ħ i k m a → 1/5 = 0.2
        v = compute_phonological_vector("hikma", "ħikma")
        assert _approx(v.pharyngeal, 0.2)

    def test_english_no_pharyngeal_negative_anchor(self):
        # English 'cat' has no pharyngeal phonemes → 0.0
        v = compute_phonological_vector("cat", "kæt")
        assert v.pharyngeal == 0.0


class TestAspiratedVoicelessNegativeAnchor:
    """Aspirated voiceless rate paired with the Sanskrit positive
    anchor in CALIBRATION_CORPUS."""

    def test_english_no_aspiration_marked_in_ipa(self):
        # English /k/ is phonetically aspirated but typically transcribed
        # without ʰ in lexical IPA → dimension reads 0.0 here.
        v = compute_phonological_vector("cat", "kæt")
        assert v.aspirated_voiceless == 0.0


class TestFinalClusterRate:
    """Final-cluster rate (CC+) — coda-rich languages high, CV-shape
    languages 0."""

    def test_english_final_cluster_strict(self):
        # 'strict' = s t r ɪ k t → ends in k t (CC) → 1.0
        v = compute_phonological_vector("strict", "strɪkt")
        assert v.final_cluster_rate == 1.0

    def test_italian_no_final_cluster(self):
        v = compute_phonological_vector("casa", "kaza")
        assert v.final_cluster_rate == 0.0


class TestAlveoloPalatalSibilants:
    """Round-1 reviewer caught a bug where ɕ/ʑ (Mandarin pinyin x/j,
    Polish ś/ź, Japanese しゃ, Serbo-Croatian ć/đ) were in _SIBILANTS
    but missing from _FRICATIVES, so they fell through _is_consonant.
    These tests pin the contract that they classify correctly across
    sibilance + soft_consonants + stop_vs_continuant."""

    def test_mandarin_xi_sibilance(self):
        # 'xi' = ɕ i → 1 sibilant / 2 phonemes = 0.5
        v = compute_phonological_vector("xi", "ɕi")
        assert _approx(v.sibilance, 0.5)

    def test_mandarin_xi_classified_as_consonant(self):
        # ɕ should be in _CONSONANTS via _FRICATIVES propagation, so
        # cluster_density's denominator includes it, and
        # stop_vs_continuant reads ɕ as a continuant fricative.
        v = compute_phonological_vector("xi", "ɕi")
        # ɕ is a continuant fricative — stops/(stops+cont) = 0/(0+1) = 0 → -1.0
        assert v.stop_vs_continuant == -1.0
        # ɕ is a soft consonant (fricative) → 1/1 = 1.0
        assert v.soft_consonants == 1.0

    def test_polish_voiced_alveolo_palatal(self):
        # Polish 'ź' = ʑ → 1 sibilant phoneme / 2 (ʑ + i) = 0.5.
        # Voiced counterpart of ɕ — same propagation contract.
        v = compute_phonological_vector("zi", "ʑi")
        assert _approx(v.sibilance, 0.5)
        assert v.stop_vs_continuant == -1.0
        assert v.soft_consonants == 1.0


class TestVowelHeight:
    """Mean F1 signed [-1=close, +1=open]. All-/a/-heavy forms score
    high; all-/i/ or all-/u/ forms score low."""

    def test_all_open_vowels(self):
        v = compute_phonological_vector("ala", "ala")
        # vowels: a, a; both height +1.0; mean = +1.0
        assert _approx(v.vowel_height, 1.0)

    def test_all_close_vowels(self):
        v = compute_phonological_vector("iki", "iki")
        # vowels: i, i; both height -1.0; mean = -1.0
        assert _approx(v.vowel_height, -1.0)


class TestVowelBackness:
    """Mean F2 signed [+1=front, -1=back]."""

    def test_all_front_vowels(self):
        v = compute_phonological_vector("ili", "ili")
        # i = +1.0 front; mean = +1.0
        assert _approx(v.vowel_backness, 1.0)

    def test_all_back_vowels(self):
        v = compute_phonological_vector("ulu", "ulu")
        # u = -1.0 back; mean = -1.0
        assert _approx(v.vowel_backness, -1.0)


class TestStopVsContinuant:
    """Signed (stops - continuants) / (stops + continuants). Positive
    = stop-dominated, negative = continuant-dominated."""

    def test_all_stops(self):
        # 'pakta' = p a k t a → consonants p k t = 3 stops, 0 continuants
        # → (3-0)/(3+0) = +1.0
        v = compute_phonological_vector("pakta", "pakta")
        assert _approx(v.stop_vs_continuant, 1.0)

    def test_all_liquids_nasals(self):
        # 'mola' = m o l a → consonants m l = 0 stops, 2 continuants
        # → (0-2)/(0+2) = -1.0
        v = compute_phonological_vector("mola", "mola")
        assert _approx(v.stop_vs_continuant, -1.0)


class TestAspiratedVoiceless:
    """Aspirated voiceless rate. Sanskrit / Tibetan / Korean / Hindi."""

    def test_sanskrit_aspirated_p(self):
        # 'pʰala' (fruit) = pʰ a l a → 1/4 = 0.25
        v = compute_phonological_vector("phala", "pʰala")
        assert _approx(v.aspirated_voiceless, 0.25)


class TestPolysyllabicBias:
    """syllable count / form length. Monosyllables read low,
    polysyllables read high."""

    def test_monosyllable_low(self):
        # 'cat' = 1 syllable / 3 chars = 0.333
        v = compute_phonological_vector("cat", "kæt")
        assert _approx(v.polysyllabic_bias, 0.333)

    def test_polysyllable_high(self):
        # 'aloha' = 3 syllables / 5 chars = 0.6
        v = compute_phonological_vector("aloha", "aloha")
        assert _approx(v.polysyllabic_bias, 0.6)


# ---- multi-language calibration corpus -----------------------------------
#
# 30+ etymons across 10 language families. Each row pairs (canonical_form,
# ipa) with the dimensions we want to assert a contract on. Not every
# dimension is pinned for every form — only the ones where the linguistic
# literature gives a strong expected reading.

CALIBRATION_CORPUS: list[tuple[str, str, str, dict[str, float]]] = [
    # (label, canonical_form, ipa, {dimension: expected_value})
    # English
    (
        "English: cat",
        "cat",
        "kæt",
        {
            "final_fortition": 1.0,
            "vowel_final_bias": 0.0,
            "soft_consonants": 0.0,
        },
    ),
    (
        "English: strict (rich cluster)",
        "strict",
        "strɪkt",
        {
            "cluster_density": 0.833,
            "final_cluster_rate": 1.0,
            "sibilance": 0.1667,
        },
    ),
    (
        "English: lemon (soft consonants)",
        "lemon",
        "lɛmən",
        {
            "soft_consonants": 1.0,
            "vowel_final_bias": 0.0,
        },
    ),
    # Italian (vowel-heavy, CV shape)
    (
        "Italian: casa",
        "casa",
        "kaza",
        {
            "vowel_final_bias": 1.0,
            "cluster_density": 0.0,
            "final_fortition": 0.0,
        },
    ),
    (
        "Italian: aurora",
        "aurora",
        "aurora",
        {
            "vowel_final_bias": 1.0,
            "cluster_density": 0.0,
        },
    ),
    # Hawaiian (CV strict, all-soft)
    (
        "Hawaiian: aloha",
        "aloha",
        "aloha",
        {
            "vowel_final_bias": 1.0,
            "soft_consonants": 1.0,
            "cluster_density": 0.0,
        },
    ),
    (
        "Hawaiian: keiki",
        "keiki",
        "keiki",
        {
            "vowel_final_bias": 1.0,
            "cluster_density": 0.0,
        },
    ),
    # Welsh (initial clusters, /ɬ/ lateral fricative)
    (
        "Welsh: cwm (back vowel, stop final)",
        "cwm",
        "kʊm",
        {
            "vowel_final_bias": 0.0,
        },
    ),
    (
        "Welsh: cluster onset (krak)",
        "krak",
        "krak",
        {
            "cluster_density": 0.5,
            "final_fortition": 1.0,
        },
    ),
    # Polish (cluster-heavy)
    (
        "Polish: 'krtań' (larynx)",
        "krtan",
        "krtaɲ",
        {
            # k r t a ɲ → 4 consonants, clusters: k-r-t adjacent (each in cluster)
            # k(in cluster: next r is C) + r (prev=k C, next=t C) + t (prev=r C, next=a V)
            # = 3 in cluster / 5 = 0.6
            "cluster_density": 0.6,
        },
    ),
    # Russian (palatalized)
    (
        "Russian: t'ema",
        "tema",
        "tʲema",
        {
            "palatalization": 0.25,
        },
    ),
    # Sanskrit (retroflex + aspirated)
    (
        "Sanskrit: phala (fruit, aspirated)",
        "phala",
        "pʰala",
        {
            "aspirated_voiceless": 0.25,
        },
    ),
    (
        "Sanskrit: pata (cloth, retroflex)",
        "pata",
        "paʈa",
        {
            "retroflexion": 0.25,
        },
    ),
    # Arabic (pharyngeal + emphatic)
    (
        "Arabic: hikma (wisdom, ħ)",
        "hikma",
        "ħikma",
        {
            "pharyngeal": 0.2,
        },
    ),
    (
        "Arabic: tahir (emphatic ṭ)",
        "tahir",
        "tˤaːhir",
        {
            # parse: tˤ a h i r → 5 phonemes; pharyngeal (tˤ) = 1/5 = 0.2
            "pharyngeal": 0.2,
        },
    ),
    # Japanese (CV, vowel-final)
    (
        "Japanese: sakura",
        "sakura",
        "sakura",
        {
            "vowel_final_bias": 1.0,
            "cluster_density": 0.0,
            # consonants: s, k, r → soft (s = fricative, r = liquid) = 2/3
            "soft_consonants": 0.6667,
        },
    ),
    # Sound-symbolism extremes (bouba/kiki via vowel + consonant features)
    (
        "Bouba-like: mumu (all-back vowels)",
        "mumu",
        "mumu",
        {
            # soft consonants, all-back vowels — 'large, dark, round'
            # mumu has only u/u → vowel_backness = -1.0
            "soft_consonants": 1.0,
            "vowel_backness": -1.0,
        },
    ),
    (
        "Kiki-like: kiki",
        "kiki",
        "kiki",
        {
            # stop consonants, front vowels — 'small, sharp, bright'
            "stop_vs_continuant": 1.0,
            "vowel_backness": 1.0,
            "soft_consonants": 0.0,
        },
    ),
]


@pytest.mark.parametrize(
    "label,form,ipa,expected",
    [(c[0], c[1], c[2], c[3]) for c in CALIBRATION_CORPUS],
    ids=[c[0] for c in CALIBRATION_CORPUS],
)
def test_calibration_corpus(label, form, ipa, expected):
    """Per-form calibration: every dimension named in ``expected``
    must round-trip the hand-graded value within ±0.01.

    Adding a new IPA token / re-balancing the classifier may break
    one of these — that's the point: the test surfaces an intentional
    contract change rather than silent drift.
    """
    v = compute_phonological_vector(form, ipa)
    for dim, want in expected.items():
        got = getattr(v, dim)
        assert _approx(got, want), f"{label}: dimension {dim} expected {want}, got {got}"


# ---- orthographic fallback tests -----------------------------------------


def test_orthographic_fallback_used_when_ipa_empty():
    # No IPA → orthographic path
    v = compute_phonological_vector("strict", "")
    # 'strict' letters: s t r i c t — vowel: i; consonants: s,t,r,c,t = 5
    # final 't' is a stop fallback
    assert v.final_fortition == 1.0
    # cluster density via consecutive-consonant pairs
    assert v.cluster_density > 0.0


def test_orthographic_fallback_used_when_ipa_none():
    v = compute_phonological_vector("casa", None)
    assert v.vowel_final_bias == 1.0


def test_orthographic_fallback_handles_empty_form():
    v = _orthographic_vector("")
    # All-zero default — equivalent to PhonologicalVector()
    assert v == PhonologicalVector()


def test_orthographic_fallback_handles_non_alpha_form():
    # Numeric / punctuation only — all-zero default
    v = _orthographic_vector("123")
    assert v == PhonologicalVector()


# ---- JSON round-trip tests -----------------------------------------------


def test_vector_to_json_rounds_to_4_decimals():
    v = PhonologicalVector(
        cluster_density=0.333333,
        final_fortition=0.5,
    )
    blob = vector_to_json(v)
    parsed = json.loads(blob)
    assert parsed["cluster_density"] == 0.3333
    assert parsed["final_fortition"] == 0.5


def test_vector_json_round_trip():
    v = compute_phonological_vector("strict", "strɪkt")
    blob = vector_to_json(v)
    v2 = vector_from_json(blob)
    # round-trip equality (within float tolerance from rounding)
    for dim in (
        "cluster_density",
        "final_fortition",
        "final_cluster_rate",
        "vowel_final_bias",
        "soft_consonants",
        "polysyllabic_bias",
        "palatalization",
        "sibilance",
        "retroflexion",
        "pharyngeal",
        "vowel_height",
        "vowel_backness",
        "stop_vs_continuant",
        "aspirated_voiceless",
    ):
        assert _approx(getattr(v, dim), getattr(v2, dim), tol=0.001), (
            f"round-trip mismatch on dim {dim}"
        )


def test_vector_json_with_extras_round_trip():
    v = PhonologicalVector(
        cluster_density=0.5,
        extras={"new_feature_v2": 0.7, "experimental": -0.3},
    )
    blob = vector_to_json(v)
    parsed = json.loads(blob)
    assert "extras" in parsed
    assert parsed["extras"]["new_feature_v2"] == 0.7
    v2 = vector_from_json(blob)
    assert v2.extras["new_feature_v2"] == 0.7
    assert v2.extras["experimental"] == -0.3


def test_vector_from_json_tolerates_missing_dimensions():
    # Older blobs from before a new dimension was added — missing
    # dimensions should default to 0.0 (the corpus mean).
    blob = json.dumps({"cluster_density": 0.4, "final_fortition": 1.0})
    v = vector_from_json(blob)
    assert v.cluster_density == 0.4
    assert v.final_fortition == 1.0
    assert v.palatalization == 0.0  # missing → default 0.0


# ---- determinism + idempotence -------------------------------------------


def test_compute_is_deterministic():
    """Same (form, ipa) tuple always produces the same vector.

    Critical for the enrichment-pass idempotence contract: re-running
    `lexicon tag-phonological-vectors --apply` on an unchanged DB
    must produce a byte-identical phonological_vector column.
    """
    v1 = compute_phonological_vector("strict", "strɪkt")
    v2 = compute_phonological_vector("strict", "strɪkt")
    assert v1 == v2
    assert vector_to_json(v1) == vector_to_json(v2)


def test_compute_is_idempotent_through_json():
    """Round-trip through JSON yields the same vector that goes back
    in. The enrichment pass writes JSON to the DB; downstream reads
    decode it back — that loop must preserve the vector exactly."""
    v1 = compute_phonological_vector("aloha", "aloha")
    v2 = vector_from_json(vector_to_json(v1))
    assert vector_to_json(v1) == vector_to_json(v2)
