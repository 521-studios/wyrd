"""Tests for the Irish anglicization sidecar (wyrd-1cjg).

The bundled ``meanings.json`` is mining-pipeline-emitted from
Wiktionary headwords (cluain, baile, achadh, …) but the bundled Irish
place-name corpus uses anglicized forms (Bally-, Cloon-, Kil-, …).
``data/irish_anglicizations.json`` is a curated sidecar carrying the
anglicized-form Meanings keyed to the same gloss + language slot as
their native Irish forms; ``_load_meanings`` unions it into the
runtime ``meaning_db`` so the matcher recognizes both forms.

These tests pin:
- the sidecar's structural fields (the data is hand-curated and a
  schema typo would silently no-op);
- end-to-end matcher coverage on canonical anglicized names that
  should decompose perfectly with the sidecar in place;
- a corpus-level lift floor so a regression that drops the sidecar
  (e.g. a refactor that bypasses the union) would surface.
"""

from __future__ import annotations

import json
from importlib import resources

from wyrd.generators.kenning import _load_meanings
from wyrd.generators.kenning.meaning import load_meanings
from wyrd.generators.kenning.name import Name, load_names


def test_sidecar_data_file_has_expected_anglicized_prefixes() -> None:
    """Every prefix the test suite asserts on must exist in the sidecar.
    Pin so a curation pass that drops a high-impact entry (e.g. Bally-,
    which alone covers ~3000 corpus names) trips the test rather than
    quietly degrading coverage."""
    payload = json.loads(
        resources.files("wyrd.generators.kenning.data")
        .joinpath("irish_anglicizations.json")
        .read_text()
    )
    usages = {w["modern_usage"] for s in payload for w in s["words"]}
    # High-impact prefixes the bundle's Irish corpus relies on.
    for required in ("Bally-", "Cloon-", "Kil-", "Knock-", "Lis-", "Glen-"):
        assert required in usages, f"sidecar missing required prefix {required!r}"


def test_sidecar_subjects_carry_old_irish_language_slot() -> None:
    """Every word entry must include an ``old_irish`` array so the
    runtime emits the morpheme into the celtic_mix family during
    proportion building. Without the language slot the entry would
    be a 'modern_usage with no roots' row, which the matcher still
    parses but the explainer can't cite a source for."""
    payload = json.loads(
        resources.files("wyrd.generators.kenning.data")
        .joinpath("irish_anglicizations.json")
        .read_text()
    )
    for subject in payload:
        for word in subject["words"]:
            assert "old_irish" in word, (
                f"sidecar entry {word['modern_usage']!r} must carry old_irish slot"
            )
            assert isinstance(word["old_irish"], list)
            assert all(isinstance(x, str) for x in word["old_irish"])


def test_load_meanings_unions_sidecar_into_runtime_meaning_db() -> None:
    """End-to-end: ``_load_meanings`` (the runtime cache) must include
    sidecar entries alongside the main bundle. Verifies the union
    happens on the load path the SPA / CLI / generator all hit."""
    _load_meanings.cache_clear()
    word_db, _ = _load_meanings()
    assert "Bally-" in word_db
    assert "Cloon-" in word_db
    # Both senses (pre + inner) for Bally- should be present.
    assert "-bally-" in word_db


def test_anglicized_prefix_decomposes_real_irish_place_names() -> None:
    """Spot-check that canonical anglicized-prefix names decompose
    perfectly with the sidecar in place. These names were unaccounted
    pre-wyrd-1cjg because the matcher had no entry for the anglicized
    prefix (only the native Irish form, e.g. cluain, lios, doire)."""
    _load_meanings.cache_clear()
    word_db, _ = _load_meanings()
    for name in ("Cloondara", "Kildare", "Lismore", "Derrymore", "Tullymore"):
        n = Name(name)
        n.find_meaning(word_db)
        assert n.count_unaccounted() == 0, (
            f"{name} should decompose perfectly with anglicization sidecar; "
            f"got unaccounted={n.count_unaccounted()}"
        )


def test_sidecar_lifts_irish_corpus_perfect_rate() -> None:
    """Corpus-level regression: the sidecar must lift the Irish
    perfect-decomposition rate vs. the main-bundle-only baseline.
    Pre-wyrd-1cjg measured 5816/34041 = 17.09%; with the v1 sidecar
    the rate sits at 17.74% (+223 names absolutely, +0.65pp). Asserts
    >= 17.5% so a regression that drops or breaks the sidecar trips,
    while leaving headroom for future Irish mining churn that might
    shift the absolute number."""
    main_text = (
        resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
    )
    sidecar_text = (
        resources.files("wyrd.generators.kenning.data")
        .joinpath("irish_anglicizations.json")
        .read_text()
    )
    place_names_text = (
        resources.files("wyrd.generators.kenning.data")
        .joinpath("irish_place_names.json")
        .read_text()
    )

    word_db, _ = load_meanings(json.loads(main_text) + json.loads(sidecar_text))
    names = load_names(json.loads(place_names_text))

    perfect = 0
    for n in names:
        nn = Name(n.name)
        nn.find_meaning(word_db)
        if nn.count_unaccounted() == 0:
            perfect += 1
    rate = perfect / len(names)
    assert rate >= 0.175, (
        f"Irish perfect-rate fell to {rate * 100:.2f}% — below the "
        f"sidecar floor of 17.5%; the sidecar may be silently bypassed"
    )
