"""wyrd-lftl: per-morpheme family × era reflex grid for the SPA col-3
inspector.

Two layers, both exercised without the runtime DB:

* ``family_stage_order`` (era/cells.py) — the distinct-canonical-language-tag
  column set per family, collapsing era cells that share a tag.
* ``_era_grid`` (runtime/proportions.py) — buckets a morpheme's
  ``era_reflexes`` into family sections of ordered stages, each carrying
  forms + per-form pronunciation + source. Exercised against real ``Meaning``
  objects so the real ``era_reflex_for`` / ``era_reflex_sources_for`` /
  ``respelling_for`` paths run, plus a ``NewName.to_dict`` integration test
  that pins the serializer wiring.
"""

from __future__ import annotations

from wyrd.generators.kenning.era.cells import family_stage_order
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.proportions import NewName, _era_grid

# --- family_stage_order: cell collapse ------------------------------------


def test_family_stage_order_collapses_cells_sharing_a_tag():
    # English has 5 cells but only 3 distinct canonical tags: oe-early +
    # oe-late both → old-english; early-modern + modern both → modern-english.
    assert family_stage_order("english") == [
        "old-english",
        "middle-english",
        "modern-english",
    ]
    # Norse maps only on-classical (later cells have no canonical tag).
    assert family_stage_order("norse") == ["old-norse"]
    assert family_stage_order("brythonic") == ["old-welsh", "middle-welsh", "welsh"]


def test_family_stage_order_unknown_family_is_empty_not_error():
    # Returns [] (not KeyError) so callers treat 'no era axis' uniformly.
    assert family_stage_order("klingon") == []


# --- _era_grid: bucketing, ordering, sources ------------------------------


def _meaning_with_reflexes(reflexes):
    """A bare Meaning carrying only era_reflexes (the field _era_grid reads).
    era_reflexes is stored as-is in the runtime ``{lang: [(form, source)]}``
    shape."""
    return Meaning(
        "-tun",
        tags=[],
        meanings=["farm"],
        sources={"old_english": ["tun"]},
        era_reflexes=reflexes,
    )


def test_era_grid_buckets_by_family_and_orders_stages():
    m = _meaning_with_reflexes(
        {
            # Deliberately out of order in the dict; the grid must order
            # english stages oldest→newest and put english before norse.
            "middle-english": [("toun", "period-form")],
            "old-norse": [("tún", "descent")],
            "old-english": [("tūn", "cluster"), ("tun", "cluster")],
            "modern-english": [("town", "phonology-rule:v1")],
        }
    )
    grid = _era_grid(m, renderings={})

    families = [section["family"] for section in grid]
    assert families == ["english", "norse"]  # _GRID_FAMILY_ORDER

    english = grid[0]["stages"]
    assert [s["language"] for s in english] == [
        "old-english",
        "middle-english",
        "modern-english",
    ]
    # Forms + sources carried through; multiple forms per stage preserved.
    oe_forms = english[0]["forms"]
    assert [f["form"] for f in oe_forms] == ["tūn", "tun"]
    assert english[2]["forms"][0]["source"] == "phonology-rule:v1"


def test_era_grid_empty_when_no_reflexes():
    assert _era_grid(_meaning_with_reflexes({}), renderings={}) == []
    assert _era_grid(None, renderings={}) == []


def test_era_grid_skips_languages_with_no_era_family():
    # A reflex tag with no era family (proto/untracked) contributes nothing
    # rather than crashing or making a bogus section.
    grid = _era_grid(
        _meaning_with_reflexes({"proto-germanic": [("*tūnaz", "descent")]}),
        renderings={},
    )
    assert grid == []


# --- _era_grid: per-cell pronunciation ------------------------------------


def test_era_grid_pronunciation_prefers_rich_rendering_then_respells():
    m = _meaning_with_reflexes({"old-english": [("tūn", "cluster")]})
    # renderings is keyed by the UNDERSCORED language ('old_english'); the rich
    # entry carries IPA, so the cell takes it verbatim.
    renderings = {"old_english": {"tūn": {"ipa": "/tuːn/", "reader_pronunciation": "TOON"}}}
    cell = _era_grid(m, renderings)[0]["stages"][0]["forms"][0]
    assert cell["ipa"] == "/tuːn/"
    assert cell["reader_pronunciation"] == "TOON"


def test_era_grid_respells_when_no_rich_rendering_and_omits_absent_ipa():
    # No renderings → reader comes from the rule respeller; IPA is absent, and
    # the cell omits the key entirely (sparse, no null noise).
    m = _meaning_with_reflexes({"old-english": [("stan", "cluster")]})
    cell = _era_grid(m, renderings={})[0]["stages"][0]["forms"][0]
    assert "ipa" not in cell
    # OE has a respeller, so reader_pronunciation is populated by rule.
    assert cell.get("reader_pronunciation")


# --- to_dict wiring -------------------------------------------------------


def test_to_dict_emits_era_grid_on_the_morpheme():
    m = _meaning_with_reflexes({"old-english": [("tūn", "cluster")]})
    nn = NewName(struct=None, meaning_db={"-tun": [m]}, name=[["-tun"]])
    morpheme = nn.to_dict()["words"][0][0]
    assert "era_grid" in morpheme
    assert morpheme["era_grid"][0]["family"] == "english"
