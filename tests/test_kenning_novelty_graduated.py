"""wyrd-fca0: graduated-novelty smoke guards.

Pins the invariants the graduated-novelty quality pass validated across the
novelty range:

* generation stays well-formed (non-empty, has morphemes) at every stop;
* the knob actually has effect (guards against a silent no-op — cf. the
  wyrd-7f22 default-never-applied class of bug);
* high novelty does NOT trigger a within-name morpheme-repeat blowup (the
  wyrd-57d8 diversification guard, generalized beyond the manorial canary —
  novelty=1's uniform blend makes degenerate repeats more likely).

Deliberately does NOT pin absolute realism values (brittle as the corpus grows,
per D36.9) — corpus-fidelity bands live in test_kenning_realism_absolute.py.
This file only guards the robust, range-wide invariants.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.generators.kenning import Kenning

# Reused for morpheme-level extraction: run_realism_samples (the public entry) hardwires
# vector mode and exposes no novelty knob, so the per-sample helper is the seam.
from wyrd.generators.kenning.runtime.drift_runner import _sample_from_generation_result

# Sample sizes (wyrd-yzul: sized to the statistic each test checks, not a flat
# 120). N covers the well-formed invariant and the divergence floors — both have
# enormous headroom (divergence runs ~93–100% true against a 20% floor, so even
# ~a dozen samples clear it by ~10σ; well-formed is a per-sample non-empty
# invariant), so 50 keeps them robust while ~halving this file's CI cost.
# N_RATE is the within-name dup-rate test, the one genuinely statistical guard
# (true rate ~1% vs a 5% ceiling), so it keeps the larger sample for a non-flaky
# margin.
N = 50
N_RATE = 120
# english + welsh; welsh was the most novelty-sensitive culture in the fca0 sweep,
# so if anything degrades fastest with novelty it shows here.
CULTURES = ["english", "welsh"]


def _samples(k, culture, novelty, n):
    return [
        _sample_from_generation_result(k.generate({"culture": culture, "novelty": novelty}, seed=i))
        for i in range(n)
    ]


@pytest.mark.parametrize("culture", CULTURES)
@pytest.mark.parametrize("novelty", [0.0, 0.5, 1.0])
def test_novelty_stays_well_formed(culture, novelty):
    """Every name is non-empty and carries at least one morpheme at every
    novelty stop — novelty reorders within the eligible pool, it never produces
    empty/undecomposable output."""
    k = Kenning()
    for s in _samples(k, culture, novelty, N):
        assert s.surface, f"{culture} nov={novelty}: empty name"
        assert s.morphemes, f"{culture} nov={novelty}: {s.surface!r} has no morphemes"


@pytest.mark.parametrize("culture", CULTURES)
def test_high_novelty_does_not_blow_up_repeats(culture):
    """At novelty=1 (uniform blend → degenerate same-surface picks more likely),
    diversification keeps within-name morpheme duplication rare. The fca0 sweep
    measured ≤1.1% across all cultures; 5% is a generous, non-flaky ceiling that
    still catches a diversification regression (the wyrd-57d8 failure mode)."""
    k = Kenning()
    samples = _samples(k, culture, 1.0, N_RATE)
    dup = sum(1 for s in samples if len({m.lower() for m in s.morphemes}) != len(s.morphemes))
    rate = dup / len(samples)
    assert rate <= 0.05, f"{culture} nov=1.0: within-name dup rate {rate:.2%} > 5%"


@pytest.mark.parametrize("culture", CULTURES)
def test_novelty_actually_changes_output(culture):
    """Max novelty must diverge from zero novelty on a meaningful share of seeds
    — a guard against the dial being silently disabled (the score→selection
    coupling regressing to a no-op). Asserts only that novelty DOES something,
    not a direction or quality (those are the realism suite's job). The sweep
    showed near-total divergence; 20% is a conservative floor."""
    k = Kenning()
    base = [k.generate({"culture": culture, "novelty": 0.0}, seed=i).result for i in range(N)]
    high = [k.generate({"culture": culture, "novelty": 1.0}, seed=i).result for i in range(N)]
    changed = sum(1 for a, b in zip(base, high, strict=True) if a != b)
    assert changed / N >= 0.2, f"{culture}: novelty barely changed output ({changed}/{N})"


@pytest.mark.parametrize("culture", CULTURES)
def test_mid_novelty_is_distinct_from_both_endpoints(culture):
    """A mid stop must differ from BOTH endpoints — guards the *graduated* nature
    of the dial, not just the two extremes. Without this, a regression collapsing
    novelty to a binary (0.5 silently == 0.0, or snapping to 1.0) passes the
    other tests yet is exactly the silent-no-op class they aim to catch, at the
    midpoint. The sweep showed 0.5 diverges ~93–94% from 0.0; 20% is a
    conservative floor on each side."""
    k = Kenning()
    low = [k.generate({"culture": culture, "novelty": 0.0}, seed=i).result for i in range(N)]
    mid = [k.generate({"culture": culture, "novelty": 0.5}, seed=i).result for i in range(N)]
    high = [k.generate({"culture": culture, "novelty": 1.0}, seed=i).result for i in range(N)]
    vs_low = sum(1 for a, b in zip(mid, low, strict=True) if a != b) / N
    vs_high = sum(1 for a, b in zip(mid, high, strict=True) if a != b) / N
    assert vs_low >= 0.2, f"{culture}: novelty=0.5 barely differs from 0.0 ({vs_low:.0%})"
    assert vs_high >= 0.2, f"{culture}: novelty=0.5 barely differs from 1.0 ({vs_high:.0%})"
