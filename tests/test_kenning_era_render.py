"""wyrd-6c8x (feature A): era-driven rendering at generation.

Two layers, both exercised here without the runtime DB:

* ``_resolve_era_render_language`` / ``_contemporary_language_for_family`` —
  pure resolution over the static era-cell tables (which era renders in which
  historical language, and which eras are 'contemporary' and so render the
  canonical modern form instead).
* ``NameGenerator._render_era_forms`` — the post-pick render step, exercised
  against a stub ``meaning_gen._surface_index`` with synthetic reflex data so
  the form-selection rules (attested-over-reconstructed, ``*``-stripping,
  no-reflex fallback, case projection) are pinned deterministically.
"""

from __future__ import annotations

from types import SimpleNamespace

from wyrd.generators.kenning import (
    _contemporary_language_for_family,
    _resolve_era_render_language,
)
from wyrd.generators.kenning.runtime.proportions import NameGenerator

# --- resolution layer (static era tables, no DB) ---------------------------


def test_resolve_era_render_language_historical_cells():
    # Past English cells render in their historical language.
    assert _resolve_era_render_language("oe-early", "english") == "old-english"
    assert _resolve_era_render_language("oe-late", "english") == "old-english"
    assert _resolve_era_render_language("me", "english") == "middle-english"


def test_resolve_era_render_language_contemporary_and_empty_are_none():
    # No era → no render (the bare-modern path).
    assert _resolve_era_render_language("", "english") is None
    assert _resolve_era_render_language(None, "english") is None
    # Contemporary cells map to the family's present-day language, whose
    # canonical surface IS the morpheme — era-rendering would distort it, so
    # they resolve to None (render the canonical modern form). early-modern and
    # modern both map to 'modern-english' in the corpus, so BOTH skip.
    assert _resolve_era_render_language("early-modern", "english") is None
    assert _resolve_era_render_language("modern", "english") is None


def test_resolve_era_render_language_malformed_degrades_to_none():
    # The loud ValueError on a bad era is _resolve_era_param's job (it runs on
    # the same input); the render-language resolver degrades to no-render rather
    # than raising twice.
    assert _resolve_era_render_language("victorian", "english") is None
    assert _resolve_era_render_language("1.2.3", "english") is None


def test_contemporary_language_for_family():
    # The open-ended (end=None) cell's canonical language.
    assert _contemporary_language_for_family("english") == "modern-english"
    assert _contemporary_language_for_family("brythonic") == "welsh"
    # A dead language has no open-future cell → no contemporary language, so
    # none of its (all-historical) cells get suppressed.
    assert _contemporary_language_for_family("latin") is None
    assert _contemporary_language_for_family("no-such-family") is None


# --- render layer (synthetic reflexes, no DB) ------------------------------


class _FakeMeaning:
    def __init__(self, reflexes: dict[str, list[str]]) -> None:
        self._reflexes = reflexes

    def era_reflex_for(self, target_language: str) -> list[str]:
        return self._reflexes.get(target_language, [])


def _render(name, index, lang="old-english"):
    """Call _render_era_forms with a stub self exposing only what it reads:
    self.meaning_gen._surface_index()."""
    stub = SimpleNamespace(meaning_gen=SimpleNamespace(_surface_index=lambda: index))
    return NameGenerator._render_era_forms(stub, name, lang)


def test_render_prefers_attested_over_reconstructed():
    # The reflex list is sorted and '*' sorts before letters, so a naive
    # forms[0] would pick the reconstructed form. We must pick 'tun', not '*xa'.
    index = {"ton": [_FakeMeaning({"old-english": ["*xa", "tun"]})]}
    assert _render([["-ton"]], index) == [["tun"]]


def test_render_strips_marker_when_all_reflexes_reconstructed():
    index = {"ton": [_FakeMeaning({"old-english": ["*tun"]})]}
    assert _render([["-ton"]], index) == [["tun"]]


def test_render_falls_back_to_none_when_no_reflex():
    # No reflex for the target language → None, which NewName.__str__ renders as
    # the canonical usage (the ~10% with no era data).
    index = {"ton": [_FakeMeaning({"middle-english": ["toun"]})]}  # asked for OE
    assert _render([["-ton"]], index, lang="old-english") == [[None]]
    # Unknown surface → no meanings → None.
    assert _render([["-zzz"]], {}) == [[None]]


def test_render_uses_first_sense_that_carries_a_reflex():
    # usage maps to multiple senses; the first WITH a reflex wins.
    index = {
        "y": [
            _FakeMeaning({"old-english": []}),
            _FakeMeaning({"old-english": ["ieg"]}),
        ]
    }
    assert _render([["-y"]], index) == [["ieg"]]


def test_render_projects_slot_case_onto_the_era_form():
    # _mimic_case projects the position-form's case: a capitalized pre-slot
    # usage yields a capitalized era form; a lowercase post-slot stays lower.
    index = {"ton": [_FakeMeaning({"old-english": ["tun"]})]}
    assert _render([["Ton-"]], index) == [["Tun"]]
    assert _render([["-ton"]], index) == [["tun"]]


def test_render_passes_through_none_slots():
    # vector path can leave permissive None slots; they stay None.
    index = {"ton": [_FakeMeaning({"old-english": ["tun"]})]}
    assert _render([["-ton", None]], index) == [["tun", None]]


# --- end-to-end (committed dev bundle) -------------------------------------

# Old-English-distinctive characters: a name carrying any of these is being
# rendered against OE reflexes, not the modern canonical surface (which is
# ASCII). The committed seed-runtime.db carries OE reflexes for the common
# morphemes generation draws, so this is stable in CI.
_OE_CHARS = set("þðæāēīōūȳǣ")


def test_era_render_end_to_end_produces_period_forms():
    """Integration: era=oe-early threads through both scoring modes and renders
    Old-English forms — distinctive characters appear, and the output differs
    from the un-era'd baseline. Pins the full resolve → thread → render path
    against the committed bundle (the wired-up feature, not just its pieces)."""
    from wyrd.generators.kenning import Kenning

    gen = Kenning()
    for mode in ("proportions", "vector"):
        oe = [
            gen.generate(
                {"culture": "english", "era": "oe-early", "scoring_mode": mode}, seed=s
            ).result
            for s in range(10)
        ]
        base = [
            gen.generate({"culture": "english", "era": "", "scoring_mode": mode}, seed=s).result
            for s in range(10)
        ]
        assert oe != base, f"{mode}: era=oe-early produced the same output as no era"
        assert any(set(name) & _OE_CHARS for name in oe), (
            f"{mode}: no Old-English forms rendered across {oe}"
        )


def test_era_modern_is_not_distorted_by_reflex_cognates():
    """Regression: the contemporary cell must render the clean canonical modern
    surface (ASCII), NOT cognate cluster-mates pulled from the modern-english
    reflex picker. era=modern carries an era FILTER (so it need not equal the
    no-era output) but must not contain OE-era characters."""
    from wyrd.generators.kenning import Kenning

    gen = Kenning()
    modern = [
        gen.generate(
            {"culture": "english", "era": "modern", "scoring_mode": "vector"}, seed=s
        ).result
        for s in range(10)
    ]
    assert not any(set(name) & _OE_CHARS for name in modern), (
        f"era=modern leaked period forms (should render canonical): {modern}"
    )
