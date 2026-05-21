"""wyrd-03cx: tests for anglicize_ipa (deterministic IPA → reader
pronunciation mapper) + its integration into _collect_renderings."""

from __future__ import annotations

from wyrd.generators.kenning.runtime.anglicize import anglicize_ipa


def test_anglicize_handles_real_oe_corpus_samples():
    """Common OE morphemes from the bundle should produce readable
    uppercase tokens — no IPA characters surviving in output."""
    assert anglicize_ipa("/xɑːm/") == "KHAHM"
    assert anglicize_ipa("/wyrm/") == "WIRM"  # OE dragon-word
    assert anglicize_ipa("/kumb/") == "KUMB"  # OE valley
    assert anglicize_ipa("/hwiːt/") == "HWIIT"  # length-doubled vowel
    # Verify no IPA chars leak into output.
    for sample in ["/xɑːm/", "/byrjj/", "/hʷiːt/"]:
        out = anglicize_ipa(sample) or ""
        assert all(ch.isupper() or ch.isdigit() for ch in out), f"IPA leak in {sample}: {out}"


def test_anglicize_handles_welsh_celtic_special_consonants():
    """Welsh /ɬ/ (the ll digraph) is the trickiest sound for English
    readers — render as 'LL' so 'Llanelli' is recognizable."""
    assert anglicize_ipa("/ɬɨn/") == "LLIN"
    assert anglicize_ipa("/ɬanelli/") == "LLANELLI"


def test_anglicize_strips_markers_and_diacritics():
    """Length / stress / syllable separators / brackets / combining
    diacritics all drop without breaking the mapping."""
    # ˈ stress marker
    assert anglicize_ipa("/ˈbɛrən/") == "BERUHN"
    # Syllable separator (ɑ → AH, m. drops separator)
    assert anglicize_ipa("/ˈxɑm.mɑs/") == "KHAHMMAHS"
    # Square-bracket transcription
    assert anglicize_ipa("[haːmˠ]") == "HAAM"


def test_anglicize_clusters_render_as_english_orthography():
    """Affricates (t͡ʃ, d͡ʒ) and fricatives (θ, ð, ʃ, ʒ) need their
    English-spelling forms."""
    assert anglicize_ipa("/t͡ʃɛst/") == "CHEST"
    assert anglicize_ipa("/brɪd͡ʒ/") == "BRIJ"
    assert anglicize_ipa("/θɔrn/") == "THAWRN"
    assert anglicize_ipa("/ðæt/") == "DHAT"
    assert anglicize_ipa("/ʃeːp/") == "SHEEP"
    assert anglicize_ipa("/ŋ/") == "NG"


def test_anglicize_returns_none_on_empty_or_unrecognized():
    """Empty input / pure-marker input → None so callers can omit
    the field rather than emit an empty string."""
    assert anglicize_ipa(None) is None
    assert anglicize_ipa("") is None
    assert anglicize_ipa("   ") is None
    assert anglicize_ipa("///") is None  # just brackets


def test_anglicize_collapses_excessive_runs():
    """Length-marker doubling shouldn't produce triple-letter runs.
    E.g., a hypothetical /ɑːː/ shouldn't render as AHAHAH."""
    out = anglicize_ipa("/ɑːː/")
    assert "AHAHAH" not in (out or "")


def test_collect_renderings_derives_reader_pronunciation_from_ipa():
    """Integration: a Meaning with IPA but no english_shaped should
    have reader_pronunciation populated by _collect_renderings via
    anglicize_ipa fallback."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import _collect_renderings

    m = Meaning(
        "-ham",
        tags=[],
        meanings=["enclosed pasture"],
        sources={"old_english": ["hām"]},
    )
    m.pronunciation = {"old_english": {"hām": {"ipa": "/xɑːm/"}}}
    m.original_script = {}
    m.transliteration = {}
    m.english_shaped = {}
    result = _collect_renderings([m])
    slot = result["old_english"]["hām"]
    assert slot["ipa"] == "/xɑːm/"
    assert slot["reader_pronunciation"] == "KHAHM"


def test_collect_renderings_prefers_english_shaped_over_anglicize():
    """When the lexicon HAS a hand-curated english_shaped form,
    don't overwrite it with the auto-generated reader_pronunciation.
    english_shaped wins; reader_pronunciation is skipped for that
    slot."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import _collect_renderings

    m = Meaning(
        "-ham",
        tags=[],
        meanings=["enclosed pasture"],
        sources={"old_english": ["hām"]},
    )
    m.pronunciation = {"old_english": {"hām": {"ipa": "/xɑːm/"}}}
    m.english_shaped = {"old_english": {"hām": "haam"}}  # curated
    m.original_script = {}
    m.transliteration = {}
    result = _collect_renderings([m])
    slot = result["old_english"]["hām"]
    assert slot["english_shaped"] == "haam"
    # Auto-anglicized form should NOT overwrite the curated one.
    assert "reader_pronunciation" not in slot


def test_collect_renderings_omits_reader_pronunciation_when_no_ipa():
    """Sparse contract: slots without IPA don't get a fallback
    reader_pronunciation. The field is derived from IPA — no IPA,
    no derivation."""
    from wyrd.generators.kenning.runtime.meaning import Meaning
    from wyrd.generators.kenning.runtime.proportions import _collect_renderings

    m = Meaning(
        "-ham",
        tags=[],
        meanings=["enclosed pasture"],
        sources={"old_english": ["hām"]},
    )
    m.pronunciation = {}
    m.original_script = {"old_english": {"hām": "hām"}}  # script only
    m.english_shaped = {}
    m.transliteration = {}
    result = _collect_renderings([m])
    slot = result["old_english"]["hām"]
    assert "reader_pronunciation" not in slot
    assert "ipa" not in slot
