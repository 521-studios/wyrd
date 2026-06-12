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
  exact usage_key, so dash-variant keys (``ton`` / ``Ton-``) bypassed
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
# Every `ton` pick in an english request must resolve into the english
# family. An allowlist (rather than a celtic blocklist) can't go stale as
# new cultures/stages land: the welsh 'wave' / irish homograph leak — or
# any future wrong-language attribution — fails this by construction.
_ENGLISH_FAMILY_PREFIXES = {"old-english", "middle-english", "modern-english"}


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
    wrong_family = [i for i in ton_ids if i.partition(":")[0] not in _ENGLISH_FAMILY_PREFIXES]
    assert not wrong_family, (
        f"english `-ton` picks resolved outside the english family: "
        f"{wrong_family} — the wyrd-pfoo surface-keyed narrowing has regressed "
        "(a homograph like welsh 'wave' ton is absorbing the OE tūn surface)"
    )
