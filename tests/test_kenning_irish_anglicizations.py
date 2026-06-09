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
import os
from importlib import resources

import pytest

from wyrd.generators.kenning import _load_meanings
from wyrd.generators.kenning.paths import seed_data_path
from wyrd.generators.kenning.runtime.meaning import load_meanings
from wyrd.generators.kenning.runtime.name import Name, load_names


def _require_full_bundle() -> None:
    """The sidecar tests pin Irish-corpus perfect-decomposition rates
    that depend on the FULL bundled meaning_db. The seed-runtime.db
    subset doesn't carry the necessary native-Irish morphemes for the
    sidecar's anglicized prefixes to compose against, so the assertion
    floors don't apply. Skip when no full bundle is configured."""
    if not (os.environ.get("WYRD_RUNTIME_DB") or os.environ.get("WYRD_RUNTIME_DB_BUCKET")):
        pytest.skip(
            "Full-corpus L4 not configured; set WYRD_RUNTIME_DB to run the "
            "Irish anglicization coverage tests."
        )


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


def test_sidecar_subjects_have_required_meaning_and_tags_fields() -> None:
    """meaning.py:load_meanings indexes each subject by ``modifier_tags``
    and ``meaning`` with bare ``[]`` subscripts — a typo like
    ``modifier_tag`` (singular) would raise KeyError at runtime. Pin
    so a curation regression surfaces at test time, not when an
    operator first hits the runtime."""
    payload = json.loads(
        resources.files("wyrd.generators.kenning.data")
        .joinpath("irish_anglicizations.json")
        .read_text()
    )
    for subject in payload:
        assert "meaning" in subject, "every subject must declare meaning"
        assert isinstance(subject["meaning"], list) and subject["meaning"]
        assert "modifier_tags" in subject, "every subject must declare modifier_tags"
        assert isinstance(subject["modifier_tags"], list) and subject["modifier_tags"]
        assert "words" in subject and isinstance(subject["words"], list)


def test_sidecar_subjects_carry_celtic_mix_language_slot() -> None:
    """Every word entry must include a ``celtic_mix`` array so the
    runtime emits the morpheme into Celtic-family rollups during
    proportion building AND the explainer's _ROOT_CODES lookup
    (CL → "Celtic") picks it up. Using ``old_irish`` directly would
    keep load_meanings happy (the matcher just sees a custom source
    key) but the explainer would render the morpheme as ``(?)``
    because ``_ROOT_CODES`` only knows the rollup names. The lexicon
    pipeline's ``_LANG_CODE_TO_JSON_FIELD`` collapses Irish / Old
    Irish / Middle Irish / Scottish Gaelic / Welsh / Breton into
    ``celtic_mix``; the sidecar matches that convention."""
    payload = json.loads(
        resources.files("wyrd.generators.kenning.data")
        .joinpath("irish_anglicizations.json")
        .read_text()
    )
    for subject in payload:
        for word in subject["words"]:
            assert "celtic_mix" in word, (
                f"sidecar entry {word['modern_usage']!r} must carry celtic_mix slot"
            )
            assert isinstance(word["celtic_mix"], list)
            assert all(isinstance(x, str) for x in word["celtic_mix"])


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


def test_anglicized_prefix_decomposes_sidecar_dependent_names() -> None:
    """Spot-check that canonical sidecar-dependent names decompose
    perfectly with the sidecar in place. Each of these names was
    fully unaccounted pre-wyrd-1cjg because no canonical morpheme
    mapped to the anglicized fragment (verified by running the
    matcher against the main bundle alone). With the sidecar in
    place they decompose cleanly via the new prefix entries.

    ``Ballyknock`` and ``Liscloon`` exercise TWO sidecar entries
    each (Bally-+Knock- and Lis-+Cloon-), proving the sidecar
    composes cleanly with itself; ``Cloon`` exercises the bare
    prefix as a standalone toponym.
    """
    _require_full_bundle()
    _load_meanings.cache_clear()
    word_db, _ = _load_meanings()
    for name in ("Cloon", "Cloondara", "Ballyknock", "Liscloon", "Kilballyboy"):
        n = Name(name)
        n.find_meaning(word_db)
        assert n.count_unaccounted() == 0, (
            f"{name} should decompose perfectly with anglicization sidecar; "
            f"got unaccounted={n.count_unaccounted()}"
        )


def test_anglicized_inner_position_decomposes_compound_names() -> None:
    """The sidecar carries inner-position variants (-bally-, -cloon-,
    -knock-, -tully-) so anglicized morphemes can match mid-word, not
    just at the start. ``Cloghballymore`` has Bally as a non-prefix
    component (Clogh + bally + more); a regression that dropped the
    inner-position entries would silently leave that whole class of
    names unaccounted."""
    _require_full_bundle()
    _load_meanings.cache_clear()
    word_db, _ = _load_meanings()
    n = Name("Cloghballymore")
    n.find_meaning(word_db)
    assert n.count_unaccounted() == 0, (
        f"Cloghballymore should decompose via inner -bally- entry; "
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
    _require_full_bundle()
    from wyrd.generators.kenning import _runtime_db_bundle_dict

    main = _runtime_db_bundle_dict()
    sidecar_text = (
        resources.files("wyrd.generators.kenning.data")
        .joinpath("irish_anglicizations.json")
        .read_text()
    )
    place_names_text = seed_data_path("irish_place_names.json").read_text()

    main_subjects = main.get("subjects") or []
    # Sidecar stays list-shape (its own format, separate from the L4 emit).
    word_db, _ = load_meanings(main_subjects + json.loads(sidecar_text))
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
