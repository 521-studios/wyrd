"""wyrd-03cx: tests for anglicize_ipa (deterministic IPA → reader
pronunciation mapper) + its integration into _collect_renderings."""

from __future__ import annotations

from wyrd.generators.kenning.runtime.anglicize import anglicize_ipa


def test_anglicize_handles_real_oe_corpus_samples():
    """Common OE morphemes from the bundle should produce readable
    uppercase tokens — no IPA characters surviving in output.

    wyrd-03cx round 2: vowel mappings collapsed to single letters
    so the length marker doubles cleanly (KHAAM not KHAHM)."""
    assert anglicize_ipa("/xɑːm/") == "KHAAM"
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
    assert anglicize_ipa("/ˈbɛrən/") == "BERUN"
    # Syllable separator (ɑ → A, m. drops separator)
    assert anglicize_ipa("/ˈxɑm.mɑs/") == "KHAMMAS"
    # Square-bracket transcription
    assert anglicize_ipa("[haːmˠ]") == "HAAM"


def test_anglicize_clusters_render_as_english_orthography():
    """Affricates (t͡ʃ, d͡ʒ) and fricatives (θ, ð, ʃ, ʒ) need their
    English-spelling forms."""
    assert anglicize_ipa("/t͡ʃɛst/") == "CHEST"
    assert anglicize_ipa("/brɪd͡ʒ/") == "BRIJ"
    assert anglicize_ipa("/θɔrn/") == "THORN"
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


def test_anglicize_handles_corpus_specific_consonants_beyond_ascii():
    """wyrd-03cx round 2: IPA ɡ (U+0261, distinct from ASCII g
    U+0067), ɫ (dark l), ɲ (palatal n), ʝ (palatal fricative),
    β (Spanish-style fricative-b), χ (uvular fricative) all appear
    in real meanings.json transcriptions and must NOT silently
    drop. Round 1's table missed all of these — '/ˈinˌɡɑː/'
    would render as 'INAH' instead of 'INGA'."""
    # ɡ is the critical one — voiced velar stop, U+0261, NOT
    # the same codepoint as ASCII 'g' (U+0067).
    assert "G" in (anglicize_ipa("/ˈinˌɡɑː/") or "")
    assert anglicize_ipa("/ɡɑː/") == "GAA"
    # ɫ (dark l, English 'l' in 'wall')
    assert anglicize_ipa("/wɫ/") == "WL"
    # ɲ (Spanish ñ, Italian gn)
    assert anglicize_ipa("/ɲ/") == "NY"
    # ʝ (voiced palatal fricative)
    assert anglicize_ipa("/ʝ/") == "Y"
    # β (Spanish v/b)
    assert anglicize_ipa("/β/") == "V"
    # χ (uvular voiceless fricative)
    assert anglicize_ipa("/χ/") == "KH"
    # ʰ (aspiration mark) drops cleanly
    assert anglicize_ipa("/tʰɑ/") == "TA"


def test_anglicize_collapses_excessive_runs():
    """Length-marker doubling shouldn't produce triple-letter runs.
    E.g., a hypothetical /ɑːː/ shouldn't render as AHAHAH."""
    out = anglicize_ipa("/ɑːː/")
    assert "AHAHAH" not in (out or "")


def test_anglicize_tolerates_pre_existing_nul_in_input():
    """A literal NUL in the input must not crash: the digraph pass uses NUL as
    a paired sentinel, and an unpaired stray NUL previously made the phoneme
    pass' `str.index("\\0", ...)` raise ValueError. A NUL is never a phoneme,
    so it's stripped and the result matches the NUL-free input."""
    assert anglicize_ipa("a\x00b") == anglicize_ipa("ab")
    assert anglicize_ipa("\x00wyrm\x00") == anglicize_ipa("wyrm")
    # A NUL adjacent to a real digraph (which itself uses sentinels) is fine.
    assert anglicize_ipa("\x00tʃ\x00") == anglicize_ipa("tʃ")


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
    # Round 2: vowels mapped to single letters so ː doubles cleanly.
    assert slot["reader_pronunciation"] == "KHAAM"


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


def test_kenning_generate_json_surfaces_reader_pronunciation_end_to_end():
    """wyrd-03cx round 2: end-to-end CLI test pinning that the
    reader_pronunciation field actually flows all the way through
    `kenning generate --json` to operator-visible output. Pre-fix
    only the unit _collect_renderings tests covered the derivation;
    a wiring regression (e.g. components() bypassed) would slip
    past."""
    import json as _json

    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.generate import generate

    runner = CliRunner()
    # Seed 42 lands on names with rich pronunciation data ('-ham',
    # '-bridge') per the wyrd-cp2d demo session output.
    result = runner.invoke(generate, ["english", "-n", "3", "--seed", "42", "--json"])
    assert result.exit_code == 0, result.output
    json_lines = [line for line in result.output.splitlines() if line.strip().startswith("{")]
    assert json_lines

    # At least one morpheme across the batch must carry a
    # reader_pronunciation; without that the integration is broken.
    found_reader_pron = False
    for line in json_lines:
        obj = _json.loads(line)
        for word in obj.get("words", []):
            for morph in word:
                renderings = morph.get("renderings", {})
                for forms in renderings.values():
                    for slot in forms.values():
                        rp = slot.get("reader_pronunciation")
                        if rp:
                            found_reader_pron = True
                            # Reader pronunciations are uppercase ASCII +
                            # never contain IPA artifacts.
                            assert rp.isupper() or rp.isdigit(), rp
                            for ipa_artifact in ["ɑ", "ə", "ː", "ˈ", "ʃ", "θ"]:
                                assert ipa_artifact not in rp, rp
    assert found_reader_pron, "No reader_pronunciation surfaced in --json output"


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
