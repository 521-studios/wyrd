"""D44 invariant gate: era selects the REFLEX, never the MORPHEME.

The morpheme inventory is time-invariant (town names change over time;
their morphemes don't — and we record the deliberate simplification
that a name's structure is constant over time, DECISIONS.md D44). The
guaranteed contract pinned here: the same ``(culture, params, seed)``
produces the same name SKELETON — the same picked morphemes, hence the
same ``modern_name()`` — at EVERY requested era; only the rendered
surface tracks the period.

Two failure modes this guards against:

* era leaking back into eligibility or scoring (the retired D5-2/D5-3
  inventory filter, or the request-era → priors-midpoint coupling) —
  the skeleton would differ between eras;
* render-aware skeleton mutations (the pre-D44 ``_diversify_repeats``
  detected repeats on the RENDERED surface, so a native-render
  collision could swap a morpheme that the force-modern render kept).

Deterministic for a given bundle (seeded RNG over a fixed seed range);
runs against whatever runtime DB the suite resolves (the committed
``seed-runtime.db`` in CI).
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.generators.kenning import Kenning

# Live-bundle smoke test (runtime DB via the committed seed) — the authoring-
# lexicon isolation fixtures are irrelevant here; opt out like the other
# live-bundle files.
pytestmark = pytest.mark.no_lexicon_isolation

N_SEEDS = 80
# Mixed-era (native per-morpheme), force-modern, and a bounded historical
# stage — the three render regimes; the skeleton must agree across all.
ERAS = ("", "present-day", "oe-late")


def _roll(gen: Kenning, era: str, seed: int):
    params: dict = {"culture": "english"}
    if era:
        params["era"] = era
    return gen.generate(params, seed)


def test_same_seed_same_skeleton_at_every_era():
    gen = Kenning()
    render_diverged = 0
    for seed in range(N_SEEDS):
        results = {era: _roll(gen, era, seed) for era in ERAS}
        moderns = {era: r.result_modern for era, r in results.items()}
        assert len(set(moderns.values())) == 1, (
            f"seed {seed}: the skeleton differs between eras — era is gating "
            f"or weighting the morpheme draw again (D44 regression): {moderns}"
        )
        # The morpheme stacks themselves must agree, not just the rendered
        # modern string (two different stacks could in principle render the
        # same modern surface).
        usages = {
            era: [[m.get("usage") for m in word] for word in r.morphemes_by_word]
            for era, r in results.items()
        }
        assert usages[""] == usages["present-day"] == usages["oe-late"], (
            f"seed {seed}: morpheme stacks differ between eras: {usages}"
        )
        if results["oe-late"].result != results["present-day"].result:
            render_diverged += 1
    # The flip side: era must still DO something — the historical render has
    # to actually differ from force-modern for a healthy share of names
    # (every-name-identical would mean the era render machinery broke).
    assert render_diverged >= N_SEEDS // 4, (
        f"oe-late rendered identically to present-day on "
        f"{N_SEEDS - render_diverged}/{N_SEEDS} seeds — the era reflex "
        "rendering looks broken (era should change the SURFACE)"
    )


def test_skeleton_invariant_holds_with_tags_combined():
    """The invariant must hold per-PARAMS, not just for the bare request —
    an era leak gated behind another axis (here the --tag hard gate) would
    slip past the headline test. Smaller sweep; same contract."""
    gen = Kenning()
    for seed in range(30):
        results = {}
        for era in ERAS:
            params: dict = {"culture": "english", "tags": ["water"]}
            if era:
                params["era"] = era
            results[era] = gen.generate(params, seed)
        moderns = {era: r.result_modern for era, r in results.items()}
        assert len(set(moderns.values())) == 1, (
            f"seed {seed} (tags=water): skeleton differs between eras: {moderns}"
        )
