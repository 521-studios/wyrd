"""wyrd-c6o1.3 regression gate: core OE morphemes stay present in
present-day english generation, attributed to the english family.

The defect this pins: ``-ton`` (OE ``tūn`` — roughly 20% of real British
place names) all but vanished from present-day english output. Two
compounding causes, both fixed under wyrd-c6o1.3:

* The era gate required an attestation year INSIDE the present-day
  window ``(1700, None)``, but the bundle's scholarly attestations are
  inherently medieval (etymology dictionaries cite Domesday-era forms)
  — so every well-documented OE morpheme was filtered out of the
  SPA's default (``era=present-day``, wyrd-kqyf) generation while
  no-data homographs passed through. Open-ended windows now pass every
  morpheme: place names accrete, the present contains all strata.
* The wyrd-pfoo per-Meaning attested-language narrowing was keyed by
  exact usage_key, so dash-variant keys (the bare ``ton``, ``Ton-``) bypassed
  it and the welsh 'wave' homograph absorbed the ``-ton`` picks that
  survived. The map is now bare-surface-keyed (fold-unioned).

Deterministic for a given bundle (seeded RNG over a fixed seed range);
runs against whatever runtime DB the suite resolves (the committed
``seed-runtime.db`` in CI). The floor is deliberately loose — the gate
exists to catch "vanished", not to pin a frequency.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.generators.kenning import Kenning

# Live-bundle smoke test (runtime DB via the committed seed) — the authoring-
# lexicon isolation fixtures are irrelevant here; opt out like the sister
# live-bundle file (tests/test_kenning_present_day_era.py).
pytestmark = pytest.mark.no_lexicon_isolation

N_SEEDS = 150
# Every `-ton` pick in an english request must resolve to one of english's
# real culture-source languages (mirrors CULTURE_LANGUAGES["english"] in the
# proportions builder: the OE/ME/ModE line PLUS Old Norse + Old French —
# english place-names are genuinely multi-source). The Old Norse `tún` cognate
# of OE `tūn` (same "enclosure/farmstead" gloss, same cognate cluster) IS the
# accurate source for a Danelaw -ton name — not a leak. What this guards
# against is a different-GLOSS homograph absorbing the surface: welsh / celtic
# / irish `ton` (wave / bottom / tone), which are celtic — outside english's
# sources — so an allowlist of those sources still excludes them by
# construction. (`modern-english:ton`=mass is a same-family homograph an
# allowlist can't catch, but it doesn't win the -ton pool in practice.)
_ENGLISH_CULTURE_LANGUAGES = {
    "old-english",
    "middle-english",
    "modern-english",
    "old-norse",
    "old-french",
}


def test_present_day_english_keeps_oe_ton_and_excludes_celtic_homograph():
    gen = Kenning()
    # One shared params dict across the loop — the request-scoped vector
    # pool cache (wyrd-dag4) lives in it, mirroring the app's count loop.
    params = {"culture": "english", "era": "present-day"}
    ton_ids: list[str] = []
    for seed in range(N_SEEDS):
        result = gen.generate(params, seed)
        for word in result.morphemes_by_word:
            for morpheme in word:
                surface = (morpheme.get("usage") or "").lower().replace("-", "")
                if surface == "ton":
                    ton_ids.append(morpheme.get("active_form_id") or "")

    assert len(ton_ids) >= 5, (
        f"-ton appeared only {len(ton_ids)} times in {N_SEEDS} present-day "
        "english names — the core OE tūn morpheme has been starved out of "
        "default generation again (era gate or eligibility regression)"
    )
    missing_id = [i for i in ton_ids if not i]
    assert not missing_id, (
        f"{len(missing_id)} `-ton` pick(s) carried no active_form_id — the "
        "era-grid anchor stamping has regressed (every present-day pick should "
        "anchor a grid cell); fix that before reading the family assertion"
    )
    wrong_source = [i for i in ton_ids if i.partition(":")[0] not in _ENGLISH_CULTURE_LANGUAGES]
    assert not wrong_source, (
        f"english `-ton` picks resolved outside english's culture-source "
        f"languages: {wrong_source} — a different-gloss homograph (welsh 'wave' / "
        "celtic / irish `ton`) is absorbing the OE tūn surface. (Old Norse `tún` "
        "is a same-gloss cognate of OE tūn and IS allowed — Danelaw -ton names "
        "descend from it; english is multi-source per CULTURE_LANGUAGES.)"
    )
